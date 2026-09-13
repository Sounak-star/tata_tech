"""
tests/test_identify.py — the sit-down state machine, with no camera and no model.

Drives Identifier through the sequences that actually happen in a cab:
an enrolled operator sits down, a stranger sits down, the seat changes
mid-shift, and the dashboard overrides a bad match.

Run:  python -m tests.test_identify
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.faceid import EMBED_DIM, FaceEmbedder, l2                 # noqa: E402
from brain.identify import Identifier, IdentifyConfig, State          # noqa: E402
from brain.templates import TemplateStore                             # noqa: E402

RNG = np.random.default_rng(101)


def unit() -> np.ndarray:
    v = RNG.standard_normal(EMBED_DIM).astype(np.float32)
    return l2(v)


def fake_landmarks():
    """478 normalised landmarks laid out as a plausible forward-facing face."""
    pts = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for i, (x, y) in zip([362, 385, 387, 263, 373, 380],
                         [(0.60, 0.42), (0.58, 0.40), (0.62, 0.40),
                          (0.64, 0.42), (0.62, 0.44), (0.58, 0.44)]):
        pts[i] = SimpleNamespace(x=x, y=y)
    for i, (x, y) in zip([33, 160, 158, 133, 153, 144],
                         [(0.36, 0.42), (0.38, 0.40), (0.42, 0.40),
                          (0.40, 0.42), (0.42, 0.44), (0.38, 0.44)]):
        pts[i] = SimpleNamespace(x=x, y=y)
    pts[1] = SimpleNamespace(x=0.50, y=0.55)      # nose tip
    pts[61] = SimpleNamespace(x=0.42, y=0.68)     # mouth corners
    pts[291] = SimpleNamespace(x=0.58, y=0.68)
    return pts


def sharp_frame() -> np.ndarray:
    """Noise passes the blur gate and sits mid-brightness."""
    return RNG.integers(60, 200, (480, 640, 3), dtype=np.uint8)


class StubEmbedder(FaceEmbedder):
    """Returns whatever identity the test says is currently in the seat."""

    def __init__(self):
        super().__init__(session=lambda nchw: np.zeros(EMBED_DIM, np.float32))
        self.current: np.ndarray | None = None
        self.noise = 0.10

    def embed_frame(self, bgr, landmarks):
        if self.current is None:
            return None
        return l2(self.current + self.noise * unit())


def build(cfg: IdentifyConfig | None = None):
    td = tempfile.mkdtemp()
    store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
    people = {oid: unit() for oid in ("ravi", "priya", "arjun")}
    for oid, base in people.items():
        store.enroll(oid, [l2(base + 0.08 * unit()) for _ in range(5)])
    emb = StubEmbedder()
    ident = Identifier(emb, store, cfg or IdentifyConfig(), machine_id="EX-07")
    return ident, emb, people


def run(ident, emb, n: int, has_face: bool = True, ear_cycle: bool = True):
    """Feed n frames; returns the last result and any identity-change events."""
    events, last = [], None
    for i in range(n):
        ear = 0.30 if not ear_cycle else (0.12 if i % 5 == 0 else 0.30)  # blinks
        last = ident.observe(sharp_frame() if has_face else None,
                             fake_landmarks() if has_face else None,
                             has_face=has_face, ear=ear)
        if last.changed:
            events.append(last)
    return last, events


# ── 1. an enrolled operator sits down ────────────────────────────────────
def test_enrolled_operator_locks_on():
    ident, emb, people = build()
    emb.current = people["ravi"]

    last, events = run(ident, emb, 60)
    assert last.state is State.LOCKED, f"expected LOCKED, got {last.state} ({last.reason})"
    assert ident.operator_id == "ravi", f"locked onto {ident.operator_id}"
    assert events and events[0].operator_id == "ravi", "no identity-change event fired"
    print(f"ok  sit-down: locked 'ravi' — {events[0].reason}, votes {events[0].votes}")


# ── 2. a stranger must NOT inherit the last operator's profile ───────────
def test_stranger_falls_back_to_unknown():
    ident, emb, people = build()
    emb.current = people["priya"]
    run(ident, emb, 60)
    assert ident.operator_id == "priya"

    run(ident, emb, 20, has_face=False)              # priya gets out
    assert ident.state is State.IDLE, "seat did not clear"

    emb.current = unit()                             # someone not enrolled sits down
    last, _ = run(ident, emb, 60)

    assert last.state is State.UNKNOWN, f"stranger produced {last.state}"
    assert ident.operator_id is None, \
        "UNKNOWN must yield no operator so the caller applies the strict profile"
    print(f"ok  stranger: UNKNOWN, no operator inherited — {last.reason}")


# ── 3. the margin guard rejects an ambiguous match ───────────────────────
def test_margin_guard_rejects_lookalikes():
    ident, emb, people = build()
    # Sit down as something almost exactly between two enrolled operators:
    # both score similarly, so top1 - top2 collapses.
    emb.current = l2(people["ravi"] + people["arjun"])
    emb.noise = 0.02

    last, _ = run(ident, emb, 60)
    assert last.state is State.UNKNOWN, \
        f"ambiguous face was accepted as {last.operator_id} ({last.reason})"
    assert "margin" in last.reason or "threshold" in last.reason, last.reason
    print(f"ok  margin guard: ambiguous face rejected — {last.reason}")


# ── 4. seat change mid-shift ─────────────────────────────────────────────
def test_seat_change_is_detected():
    ident, emb, people = build()
    emb.current = people["ravi"]
    run(ident, emb, 60)
    assert ident.operator_id == "ravi"

    run(ident, emb, 20, has_face=False)              # ravi gets out
    emb.current = people["arjun"]                    # arjun gets in
    last, events = run(ident, emb, 60)

    assert last.state is State.LOCKED and ident.operator_id == "arjun", \
        f"failed to switch: {ident.operator_id} ({last.reason})"
    assert any(e.operator_id == "arjun" for e in events)
    print("ok  seat change: ravi -> empty -> arjun, profile switch fired once")


def test_lock_is_stable_no_flapping():
    ident, emb, people = build()
    emb.current = people["priya"]
    run(ident, emb, 60)

    _, events = run(ident, emb, 300)                 # a long steady stretch
    assert not events, f"identity flapped {len(events)} times while locked"
    assert ident.operator_id == "priya"
    print("ok  stability: 300 locked frames, zero spurious profile switches")


# ── 5. manual override always wins ───────────────────────────────────────
def test_manual_override_beats_face():
    ident, emb, people = build()
    emb.current = people["ravi"]
    run(ident, emb, 60)
    assert ident.operator_id == "ravi"

    ident.force("priya")                             # supervisor picks from the list
    last, _ = run(ident, emb, 40)
    assert last.state is State.LOCKED and ident.operator_id == "priya", \
        "manual override was overruled by the face — an operator could be locked out"
    print("ok  override: manual selection holds against a contradicting face")


# ── 6. no templates / no model => quiet manual fallback ──────────────────
def test_no_enrolment_degrades_quietly():
    td = tempfile.mkdtemp()
    store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
    ident = Identifier(StubEmbedder(), store, machine_id="EX-07")

    assert not ident.available
    last, _ = run(ident, StubEmbedder(), 30)
    assert last.state is State.IDLE and last.operator_id is None
    print("ok  empty gallery: stays IDLE, no crash, caller uses the manual list")


# ── 7. augmentation records the new camera, anchors intact ───────────────
def test_augmentation_on_new_machine():
    ident, emb, people = build(IdentifyConfig(augment=True))
    emb.current = people["ravi"]
    emb.noise = 0.02                                  # a very confident sighting

    run(ident, emb, 60)
    summary = ident.store.summary("ravi")
    assert "EX-07" in summary["machines"], f"new camera not recorded: {summary['machines']}"
    assert summary["anchors"] == 5, "enrolment anchors changed"
    assert summary["vectors"] == 6, f"expected one augmentation, got {summary['vectors']}"
    print(f"ok  augmentation: EX-07 folded in ({summary['vectors']} vectors, 5 anchors)")


# ── 8. regressions: a disabled or blind camera must not steal the profile ─
def test_manual_pick_survives_the_face_disappearing():
    """Observed on the live dashboard: pick Priya, look away, and the cab
    silently dropped to the guest profile. A manual selection is the
    supervisor's decision — an empty seat is not a reason to discard it."""
    ident, emb, people = build()
    ident.force("priya")

    last, events = run(ident, emb, 60, has_face=False)
    assert ident.operator_id == "priya",         f"manual pick wiped by an empty seat (now {ident.operator_id})"
    assert last.state is State.LOCKED
    assert not events, "an absent face must not fire a profile change"
    print("ok  regression: manual pick survives the camera losing the face")


def test_unavailable_faceid_never_moves_the_operator():
    """With no ArcFace model (or an empty gallery) face ID is off. It must be
    inert — not quietly reset whoever the dashboard selected."""
    td = tempfile.mkdtemp()
    store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
    emb = StubEmbedder()
    emb.available = False                      # as if onnxruntime/model missing
    emb.last_error = "ArcFace model not found"
    ident = Identifier(emb, store, machine_id="EX-07")

    assert not ident.available
    ident.force("ravi")

    _, events = run(ident, emb, 40, has_face=False)
    _, more = run(ident, emb, 40, has_face=True)
    assert ident.operator_id == "ravi", f"identity moved to {ident.operator_id}"
    assert not events and not more, "disabled face ID emitted profile changes"
    print("ok  regression: face ID with no model is inert, never switches profile")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
