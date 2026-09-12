"""
tests/test_faceid.py — the face-ID maths, with no camera and no ONNX model.

These cover the three claims the design rests on:
  • the cancellable projection is loss-free (cosine survives Q exactly)
  • max-over-set beats the mean on a pose-diverse template — the reason we
    store five vectors instead of averaging them
  • enrolment anchors survive augmentation pressure (no template poisoning)

Run:  python -m tests.test_faceid      (or: pytest tests/test_faceid.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.faceid import EMBED_DIM, FaceEmbedder, l2          # noqa: E402
from brain.templates import MAX_VECTORS, Template, TemplateStore  # noqa: E402


def _store(tmp: Path) -> TemplateStore:
    return TemplateStore(path=tmp / "templates.enc", keyfile=tmp / "faceid.key")


def _unit(rng, n: int = 1) -> np.ndarray:
    v = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    return np.squeeze(v / np.linalg.norm(v, axis=1, keepdims=True))


# ── 1. the projection is orthogonal and loss-free ────────────────────────
def test_projection_preserves_cosine():
    with tempfile.TemporaryDirectory() as td:
        st = _store(Path(td))
        rng = np.random.default_rng(7)

        q = st._Q
        assert np.allclose(q @ q.T, np.eye(EMBED_DIM), atol=1e-4), "Q is not orthogonal"

        for _ in range(50):
            a, b = _unit(rng), _unit(rng)
            before = float(a @ b)
            after = float(st.project(a) @ st.project(b))
            assert abs(before - after) < 1e-4, f"cosine drifted {before} -> {after}"

        # ...and the stored numbers are genuinely not the original embedding.
        a = _unit(rng)
        assert float(st.project(a) @ a) < 0.5, "projection left the vector recognisable"
    print("ok  projection: orthogonal, cosine-preserving, non-invertible without Q")


def test_rotation_revokes_templates():
    with tempfile.TemporaryDirectory() as td:
        st = _store(Path(td))
        rng = np.random.default_rng(11)
        vecs = [_unit(rng) for _ in range(5)]
        st.enroll("ravi", vecs, enrolled_by="sup_1")
        assert st.verify("ravi", vecs[0]) > 0.99

        st.rotate_projection()
        assert st.enrolled_ids() == [], "rotation must invalidate every template"
    print("ok  rotation: revokes the whole gallery (re-enrolment required)")


# ── 2. encryption round-trip, and nothing readable on disk ───────────────
def test_encrypted_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rng = np.random.default_rng(3)
        vecs = [_unit(rng) for _ in range(5)]

        st = _store(tmp)
        st.enroll("priya", vecs, enrolled_by="sup_2", machine_id="EX-07")

        blob = (tmp / "templates.enc").read_bytes()
        assert b"priya" not in blob, "operator id leaked in plaintext"
        assert b"vectors" not in blob

        reopened = _store(tmp)
        assert reopened.enrolled_ids() == ["priya"]
        assert reopened.verify("priya", vecs[2]) > 0.99, "template did not survive reload"
        assert reopened.summary("priya")["enrolled_by"] == "sup_2"
    print("ok  storage: AES-GCM round-trip, no plaintext identifiers on disk")


# ── 3. the headline claim: max-over-set > mean, on a pose-diverse template ─
def test_max_over_set_beats_mean():
    """Enrolment poses are multi-modal; the centroid sits between them.

    Simulated as two pose clusters ~65 degrees apart (straight-on vs turned).
    A probe taken in the second pose should match strongly. The mean of the
    enrolment vectors matches it far less well - which is exactly why the
    store keeps the set.
    """
    with tempfile.TemporaryDirectory() as td:
        st = _store(Path(td))
        rng = np.random.default_rng(42)

        pose_a, pose_b = _unit(rng), _unit(rng)
        pose_b = l2(0.42 * pose_a + 0.91 * pose_b)      # ~65 deg from pose_a

        def jitter(base, k):
            return [l2(base + 0.12 * _unit(rng)) for _ in range(k)]

        enrolment = jitter(pose_a, 3) + jitter(pose_b, 2)
        st.enroll("arjun", enrolment)

        probe = l2(pose_b + 0.12 * _unit(rng))          # operator turns their head

        max_score = st.verify("arjun", probe)
        mean_vec = l2(np.mean([st.project(v) for v in enrolment], axis=0))
        mean_score = float(mean_vec @ st.project(probe))

        print(f"    max-over-set={max_score:.3f}   mean-of-set={mean_score:.3f}")
        assert max_score > mean_score + 0.10, "max-over-set should clearly win"
        assert max_score > 0.90, "the matching pose should score high"
    print("ok  matching: max-over-set beats the centroid on pose-diverse templates")


def test_identify_ranks_and_margins():
    with tempfile.TemporaryDirectory() as td:
        st = _store(Path(td))
        rng = np.random.default_rng(5)

        people = {f"op_{i:03d}": _unit(rng) for i in range(60)}
        for oid, base in people.items():
            st.enroll(oid, [l2(base + 0.10 * _unit(rng)) for _ in range(5)])

        target = "op_017"
        probe = l2(people[target] + 0.10 * _unit(rng))
        ranked = st.identify(probe, top_k=3)

        assert ranked[0][0] == target, f"top-1 was {ranked[0][0]}, expected {target}"
        margin = ranked[0][1] - ranked[1][1]
        print(f"    top1={ranked[0][1]:.3f} ({ranked[0][0]})  "
              f"top2={ranked[1][1]:.3f} ({ranked[1][0]})  margin={margin:.3f}")
        assert margin > 0.20, "a genuine match should clear the runner-up comfortably"

        # An operator who was never enrolled must not win convincingly.
        stranger = _unit(rng)
        s_ranked = st.identify(stranger, top_k=2)
        print(f"    stranger best={s_ranked[0][1]:.3f} — must stay under threshold")
        assert s_ranked[0][1] < 0.35, "a stranger scored like a match"
    print("ok  identify: 1:N over 60 operators ranks correctly with a wide margin")


# ── 4. anchors survive augmentation pressure ─────────────────────────────
def test_anchors_are_never_evicted():
    with tempfile.TemporaryDirectory() as td:
        st = _store(Path(td))
        rng = np.random.default_rng(9)

        base = _unit(rng)
        anchors = [l2(base + 0.08 * _unit(rng)) for _ in range(5)]
        st.enroll("ravi", anchors, machine_id="EX-07")

        for i in range(30):                              # heavy augmentation
            st.augment("ravi", l2(base + 0.15 * _unit(rng)), machine_id=f"EX-{i:02d}")

        tpl = st.templates["ravi"]
        assert tpl.vectors.shape[0] == MAX_VECTORS, "set should be capped"
        assert tpl.anchors == 5

        stored_anchors = tpl.vectors[:5]
        for i, a in enumerate(anchors):
            assert float(stored_anchors[i] @ st.project(a)) > 0.999, \
                f"anchor {i} was evicted or overwritten"
        assert len(tpl.machines) > 1, "augmentation should record new cameras"
    print(f"ok  anchors: {MAX_VECTORS}-vector cap enforced, enrolment rows immutable")


def test_expiry_purges_on_load():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rng = np.random.default_rng(13)
        st = _store(tmp)
        st.enroll("stale", [_unit(rng) for _ in range(3)], retention_days=0)
        st.templates["stale"].expires = "2020-01-01T00:00:00+00:00"
        st.save()

        assert _store(tmp).enrolled_ids() == [], "expired template was not purged"
    print("ok  retention: expired templates are dropped at load")


# ── 5. mirror invariance holds for any embedder ──────────────────────────
def test_embedding_is_mirror_invariant():
    """A browser feed is often flipped; the cab camera is not. e(x)+e(flip x) is
    the same either way round, so enrolment and recognition can disagree about
    mirroring without costing us a match."""
    rng = np.random.default_rng(23)
    proj = rng.standard_normal((EMBED_DIM, 112 * 112 * 3)).astype(np.float32) * 0.01

    def stub(nchw):                       # deliberately flip-sensitive
        return proj @ nchw.reshape(-1)

    emb = FaceEmbedder(session=stub)
    assert emb.available and emb.backend == "injected"

    img = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    import cv2
    a = emb.embed(img)
    b = emb.embed(cv2.flip(img, 1))

    assert a is not None and abs(np.linalg.norm(a) - 1.0) < 1e-5, "embedding not unit-norm"
    assert float(a @ b) > 0.9999, f"mirror invariance broken (cos={float(a @ b):.5f})"
    print("ok  embedder: flip-TTA makes the signature mirror-invariant")


def test_missing_model_degrades_quietly():
    emb = FaceEmbedder(model_path=Path("does-not-exist.onnx"))
    assert not emb.available
    assert emb.embed(np.zeros((112, 112, 3), np.uint8)) is None
    assert emb.last_error
    print("ok  degradation: no model -> unavailable, caller falls back to manual")


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
