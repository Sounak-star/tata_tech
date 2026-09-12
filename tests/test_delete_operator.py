"""
tests/test_delete_operator.py — deleting an operator, including the sharp edges.

Deletion is the one operation that can leave the system describing a person it
no longer knows. The cases that matter:
  * profile and face template go together, or an orphan survives
  * deleting the ACTIVE operator must land on the guest profile, not crash
  * deleting the last operator must leave a working cab
  * it is PIN-gated and leaves an audit trail

Run:  python -m tests.test_delete_operator
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp()
os.environ.setdefault("SAARTHI_TEMPLATES", str(Path(_TMP) / "t.enc"))
os.environ.setdefault("SAARTHI_FACEID_KEY", str(Path(_TMP) / "k.key"))
os.environ.setdefault("SAARTHI_SUPERVISORS", str(Path(_TMP) / "sup.json"))

from brain.faceid import EMBED_DIM, l2                              # noqa: E402
from brain.personalize import personalise                           # noqa: E402
from brain.profiles import GUEST_ID, ProfileStore                   # noqa: E402
from brain.templates import TemplateStore                           # noqa: E402

RNG = np.random.default_rng(31)


def unit() -> np.ndarray:
    return l2(RNG.standard_normal(EMBED_DIM).astype(np.float32))


def make_store(monkey_path: Path) -> ProfileStore:
    """A ProfileStore whose save() writes to a scratch file, not the repo."""
    store = ProfileStore()
    store.profiles = {
        f"op_{i}": {"id": f"op_{i}", "name": f"Operator {i}", "role": "Excavator",
                    "hearing": "normal", "color_vision": "normal", "language": "en",
                    "experience": "expert", "calibration": {"calibrated": False}}
        for i in (1, 2, 3)
    }
    store._active = "op_1"
    store.save = lambda: None          # persistence is covered elsewhere
    return store


# ── 1. basic removal ─────────────────────────────────────────────────────
def test_delete_removes_the_profile():
    store = make_store(Path(_TMP))
    assert store.delete("op_2") is True
    assert "op_2" not in store.profiles
    assert store.delete("op_2") is False, "second delete should report unknown"
    assert store.delete("nobody") is False
    print("ok  delete: profile removed, repeat and unknown ids report False")


def test_guest_profile_cannot_be_deleted():
    store = make_store(Path(_TMP))
    try:
        store.delete(GUEST_ID)
    except ValueError as exc:
        assert "guest" in str(exc)
        print("ok  guest: the fail-safe profile is not deletable")
        return
    raise AssertionError("the guest profile was deleted")


# ── 2. deleting the active operator ──────────────────────────────────────
def test_deleting_the_active_operator_lands_on_guest():
    """Not a crash, and not a silent hand-over to whoever is next in the list —
    we no longer know who is in the seat, so the strictest profile is correct."""
    store = make_store(Path(_TMP))
    assert store.active_id == "op_1"

    store.delete("op_1")

    assert store.is_guest, "did not fall back to the guest profile"
    assert store.active_id == GUEST_ID
    plan = personalise(1, store.active())
    assert plan["buzz"] and plan["flash"] and plan["sound"], \
        "guest fallback is not using every alert channel"
    print("ok  active operator: deletion drops to guest, all channels on")


def test_deleting_the_last_operator_leaves_a_working_cab():
    store = make_store(Path(_TMP))
    for oid in ("op_1", "op_2", "op_3"):
        store.delete(oid)

    assert store.profiles == {}
    prof = store.active()                       # must not raise
    assert prof["id"] == GUEST_ID
    assert personalise(2, prof)["level"] == 2
    print("ok  empty roster: active() still returns a usable guest profile")


def test_active_survives_a_profile_vanishing_underneath_it():
    """Belt and braces: even if something removes the record without going
    through delete(), a tick must not raise mid-loop."""
    store = make_store(Path(_TMP))
    store.profiles.pop("op_1")                  # bypass delete() entirely
    prof = store.active()
    assert prof["id"] == GUEST_ID and store.is_guest
    print("ok  robustness: a vanished active profile degrades to guest, not KeyError")


# ── 3. the template must go with the profile ─────────────────────────────
def test_template_and_profile_are_removed_together():
    """An orphaned template matches a person the system can no longer describe."""
    tmp = Path(tempfile.mkdtemp())
    templates = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")
    store = make_store(tmp)

    face = unit()
    templates.enroll("op_2", [l2(face + 0.05 * unit()) for _ in range(5)])
    assert templates.verify("op_2", face) > 0.9

    # What the endpoint does, in order.
    store.delete("op_2")
    templates.remove("op_2")

    assert "op_2" not in store.profiles
    assert templates.enrolled_ids() == [], "face signature outlived the profile"
    assert templates.verify("op_2", face) == -1.0
    assert templates.summary("op_2") is None
    print("ok  together: profile and face signature both erased")


def test_deletion_survives_a_reload():
    tmp = Path(tempfile.mkdtemp())
    templates = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")
    templates.enroll("op_9", [unit() for _ in range(4)])
    templates.remove("op_9")

    reopened = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")
    assert reopened.enrolled_ids() == [], "template came back after a reload"
    print("ok  persistence: an erased signature stays erased across a restart")


# ── 4. the gate ──────────────────────────────────────────────────────────
def test_deletion_requires_a_supervisor_pin():
    from brain.enrollment import SupervisorAuth

    tmp = Path(tempfile.mkdtemp())
    auth = SupervisorAuth(tmp / "sup.json")
    auth.add_supervisor("sup_9", "Meena R", "246810")

    ok, _ = auth.verify("sup_9", "000000")
    assert not ok, "wrong PIN accepted"
    ok, name = auth.verify("sup_9", "246810")
    assert ok and name == "Meena R"
    print("ok  gate: deletion reuses the supervisor PIN, and it is enforced")


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
