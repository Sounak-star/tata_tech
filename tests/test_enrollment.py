"""
tests/test_enrollment.py — the in-cab enrolment session, headless.

Walks a whole enrolment the way the cab UI will: supervisor PIN, profile form,
guided poses with quality and pose gates, the shared fatigue baseline, then
commit into the profile dict and the encrypted template store.

Run:  python -m tests.test_enrollment
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.enrollment import (                                   # noqa: E402
    COLOR_VISION, EXPERIENCE, FRAMES_PER_POSE, MIN_GAP_FRAMES,
    EnrollmentSession, OperatorDetails, Step, SupervisorAuth, build_script,
)
from brain.faceid import EMBED_DIM, FaceEmbedder, l2               # noqa: E402
from brain.personalize import SUPPORTED_LANGUAGES, personalise     # noqa: E402
from brain.pose import metrics_from_landmarks                      # noqa: E402
from brain.templates import TemplateStore                          # noqa: E402

RNG = np.random.default_rng(77)


def unit() -> np.ndarray:
    return l2(RNG.standard_normal(EMBED_DIM).astype(np.float32))


def fake_landmarks(yaw_shift: float = 0.0):
    pts = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    s = yaw_shift
    for i, (x, y) in zip([362, 385, 387, 263, 373, 380],
                         [(0.60, 0.42), (0.58, 0.40), (0.62, 0.40),
                          (0.64, 0.42), (0.62, 0.44), (0.58, 0.44)]):
        pts[i] = SimpleNamespace(x=x + s, y=y)
    for i, (x, y) in zip([33, 160, 158, 133, 153, 144],
                         [(0.36, 0.42), (0.38, 0.40), (0.42, 0.40),
                          (0.40, 0.42), (0.42, 0.44), (0.38, 0.44)]):
        pts[i] = SimpleNamespace(x=x + s, y=y)
    pts[1] = SimpleNamespace(x=0.50 + s, y=0.55)
    pts[61] = SimpleNamespace(x=0.42 + s, y=0.68)
    pts[291] = SimpleNamespace(x=0.58 + s, y=0.68)
    return pts


def sharp_frame() -> np.ndarray:
    return RNG.integers(60, 200, (480, 640, 3), dtype=np.uint8)


def blurry_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


class StubEmbedder(FaceEmbedder):
    def __init__(self, identity: np.ndarray):
        super().__init__(session=lambda nchw: np.zeros(EMBED_DIM, np.float32))
        self.identity = identity
        self.noise = 0.08

    def embed_frame(self, bgr, landmarks):
        return l2(self.identity + self.noise * unit())


def calib_rows(n: int = 12) -> list[dict]:
    """Shaped like live_camera.calibrate() output for an alert operator.

    All 12 model features, because a partial row silently falls back to the
    summary-only baseline — which is exactly the gap that stops the profile
    from replacing the 25 s startup calibration.
    """
    rows = []
    for i in range(n):
        j = i % 3
        rows.append({
            "ear_mean": 0.30 + 0.01 * j, "ear_min": 0.21 + 0.01 * j, "ear_std": 0.03,
            "perclos": 0.05 + 0.005 * j, "blink_rate": 15.0 + j,
            "mar_mean": 0.20, "mar_max": 0.35, "is_yawn": 0.0,
            "pitch_mean": 2.0 + j, "yaw_mean": -1.0 + j, "roll_mean": 0.5,
            "pitch_std": 1.5,
        })
    return rows


def details(**kw) -> OperatorDetails:
    base = dict(id="op_900", name="Test Operator", role="Excavator Operator",
                experience="expert", hearing="impaired", color_vision="normal",
                language="hi")
    base.update(kw)
    return OperatorDetails(**base)


def drive_poses(sess: EnrollmentSession, frames_per_attempt: int = 400) -> None:
    """Feed frames until every pose is captured, obeying each pose's gate."""
    for _ in range(frames_per_attempt):
        pose = sess.current_pose
        if pose is None:
            return
        yaw = 0.0
        pitch = 0.0
        if pose.require_turn:
            # Alternate direction per pose key so the two turns go opposite ways.
            yaw = 25.0 if pose.key == "turn_a" else -25.0
        if pose.require_chin:
            pitch = 18.0
        sess.feed_frame(sharp_frame(), fake_landmarks(), yaw=yaw, pitch=pitch)


# ── 1. supervisor gate ───────────────────────────────────────────────────
def test_supervisor_pin_is_hashed_and_rate_limited():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "supervisors.json"
        auth = SupervisorAuth(path)
        auth.add_supervisor("sup_007", "Meena R", "481920")

        raw = path.read_text(encoding="utf-8")
        assert "481920" not in raw, "PIN stored in plaintext"

        ok, who = auth.verify("sup_007", "481920")
        assert ok and who == "Meena R"

        for _ in range(SupervisorAuth.MAX_FAILS):
            auth.verify("sup_007", "000000")
        ok, msg = auth.verify("sup_007", "481920")
        assert not ok and "locked" in msg, f"no lockout after brute force: {msg}"
    print("ok  supervisor: scrypt-hashed PIN, correct verify, lockout after 5 fails")


# ── 2. the form matches what the personaliser actually consumes ──────────
def test_form_vocabulary_matches_personaliser():
    bad = details(language="fr").validate()
    assert any("language" in e for e in bad), "invalid language accepted"
    assert not details().validate(), "valid form rejected"

    # Every allowed value must survive personalise() without falling to defaults.
    for exp in EXPERIENCE:
        for cv in COLOR_VISION:
            for lang in SUPPORTED_LANGUAGES:
                d = details(experience=exp, color_vision=cv, language=lang)
                assert not d.validate()
                plan = personalise(2, d.to_profile())
                assert plan["language"] == lang
                assert plan["use_icons"] == (cv != "normal")
                assert plan["simplified_ui"] == (exp == "trainee")
    print(f"ok  form: {len(EXPERIENCE)}x{len(COLOR_VISION)}x{len(SUPPORTED_LANGUAGES)} "
          f"combinations all drive personalise() correctly")


def test_notes_explain_accessibility_automatically():
    p = details(hearing="impaired", color_vision="deuteranopia",
                experience="trainee", notes="").to_profile()
    assert "sound" in p["notes"] and "olour-blind" in p["notes"] and "Trainee" in p["notes"]
    print("ok  form: blank notes auto-explain the accessibility consequences")


# ── 3. the pose script is a liveness challenge ───────────────────────────
def test_pose_script_is_randomised_after_centre():
    orders = {tuple(p.key for p in build_script(rng=random.Random(s))) for s in range(30)}
    assert len(orders) > 1, "script order is fixed — no liveness value"
    assert all(o[0] == "center" for o in orders), "centre must lead (strongest anchor)"
    print(f"ok  poses: centre-first, {len(orders)} distinct randomised orders seen")


def test_pose_gates_reject_wrong_pose_and_bad_frames():
    sess = EnrollmentSession(details(), StubEmbedder(unit()), supervisor="sup_007",
                             shuffle=False)

    st = sess.feed_frame(blurry_frame(), fake_landmarks())
    assert st["captured"] == 0 and "blurr" in st["message"], st["message"]

    sess.pose_idx = 1                                    # a turn pose
    st = sess.feed_frame(sharp_frame(), fake_landmarks(), yaw=2.0)
    assert st["captured"] == 0 and "turn" in st["message"], st["message"]

    sess.pose_idx = 0
    st = sess.feed_frame(sharp_frame(), fake_landmarks(), yaw=40.0)
    assert st["captured"] == 0 and "squarely" in st["message"], st["message"]
    print("ok  poses: blur, wrong-pose and off-axis frames are all refused")


def test_consecutive_frames_are_spaced_out():
    sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    sess.feed_frame(sharp_frame(), fake_landmarks())     # first accept
    st = sess.feed_frame(sharp_frame(), fake_landmarks())
    assert st["captured"] == 1 and "hold" in st["message"], \
        "back-to-back frames accepted — the set would be three copies of one moment"
    print(f"ok  poses: captures spaced by {MIN_GAP_FRAMES} frames, no duplicate moments")


# ── 4. a full enrolment, end to end ──────────────────────────────────────
def test_full_enrolment_commits_profile_and_template():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")
        identity = unit()
        sess = EnrollmentSession(details(), StubEmbedder(identity),
                                 supervisor="sup_007", machine_id="EX-07",
                                 rng=random.Random(1))

        drive_poses(sess)
        assert sess.step is Step.BASELINE, f"poses did not complete: {sess.status()}"
        expected = FRAMES_PER_POSE * len(sess.script)
        assert sess.captured_count() == expected if hasattr(sess, "captured_count") \
            else len(sess.vectors) == expected, f"got {len(sess.vectors)} vectors"

        sess.set_baseline(calib_rows())
        assert sess.step is Step.REVIEW
        assert sess.review()["ready"], sess.review()

        profiles: dict = {}
        res = sess.commit(profiles, store)
        assert res["ok"], res
        assert sess.step is Step.COMMITTED

        prof = profiles["op_900"]
        assert prof["hearing"] == "impaired" and prof["language"] == "hi"
        assert prof["calibration"]["calibrated"] is True
        assert prof["calibration"]["ear_baseline"] > 0

        # The full 12-feature baseline must be there — without it, sit-down
        # recognition still has to run the 25 s calibration.
        full = prof["calibration"].get("baseline")
        assert full, "full baseline missing — calibration cannot be skipped"
        assert len(full) == 12, f"expected 12 features, got {sorted(full)}"
        assert all({"mean", "std"} <= set(v) for v in full.values())
        assert prof["enrolled"]["by"] == "sup_007" and prof["enrolled"]["machine"] == "EX-07"

        # the template is queryable and matches the enrolled person
        assert store.verify("op_900", identity) > 0.9
        assert store.verify("op_900", unit()) < 0.4
        print(f"ok  end-to-end: {res['vectors']} vectors + baseline committed, "
              f"template matches ({store.verify('op_900', identity):.2f}) "
              f"and rejects a stranger")


def test_stored_baseline_replaces_live_calibration():
    """The payoff: an enrolled operator's baseline rebuilds the fatigue engine
    directly, so sit-down does not need the 25 s calibration pass."""
    from brain.fatigue import FatigueEngine

    sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    drive_poses(sess)
    sess.set_baseline(calib_rows())
    profiles: dict = {}
    with tempfile.TemporaryDirectory() as td:
        store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
        assert sess.commit(profiles, store)["ok"]

    stored = profiles["op_900"]["calibration"]["baseline"]
    engine = FatigueEngine.from_baseline(stored)
    assert engine is not None, "stored baseline was rejected — calibration cannot be skipped"
    assert engine.backend == "xgboost"

    # A malformed or partial baseline must be refused, not limped along with.
    assert FatigueEngine.from_baseline({k: stored[k] for k in list(stored)[:5]}) is None
    assert FatigueEngine.from_baseline({"ear_mean": "not-a-number"}) is None
    print("ok  baseline: enrolment baseline rebuilds the engine; partial ones refused")


def test_enrolment_never_retains_an_image():
    """The privacy claim, enforced: no ndarray bigger than a vector on the session."""
    sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    drive_poses(sess)
    sess.set_baseline(calib_rows())

    for name, value in vars(sess).items():
        if isinstance(value, np.ndarray):
            assert value.ndim == 1, f"{name} holds an image-shaped array {value.shape}"
        if isinstance(value, list):
            for item in value:
                if isinstance(item, np.ndarray):
                    assert item.ndim == 1 and item.size == EMBED_DIM, \
                        f"{name} holds a non-vector array {item.shape}"
    print("ok  privacy: session holds only 512-d vectors — no frame is retained")


def test_commit_refuses_without_baseline_or_enough_captures():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")

        sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
        drive_poses(sess)
        res = sess.commit({}, store)
        assert not res["ok"] and "baseline" in res["errors"][0], res
        assert store.enrolled_ids() == [], "template written despite failed commit"

        thin = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
        thin.feed_frame(sharp_frame(), fake_landmarks())
        thin.set_baseline(calib_rows())
        res = thin.commit({}, store)
        assert not res["ok"] and "usable face captures" in res["errors"][0], res
    print("ok  commit: refused without a baseline, and with too few captures")


def test_outlier_frames_are_dropped():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = TemplateStore(path=tmp / "t.enc", keyfile=tmp / "k.key")
        emb = StubEmbedder(unit())
        sess = EnrollmentSession(details(), emb, shuffle=False)

        drive_poses(sess)
        sess.vectors.append(unit())                      # a different person walks in
        sess.vector_poses.append("center")
        sess.set_baseline(calib_rows())

        review = sess.review()
        assert review["vectors_dropped"] == 1, review
        assert "inconsistent" in review["warnings"][0]

        res = sess.commit({}, store)
        assert res["ok"] and res["dropped"] == 1
        assert store.summary("op_900")["vectors"] == len(sess.vectors) - 1
    print("ok  outliers: a stray face in the set is detected and dropped at commit")


def test_optional_pose_can_be_skipped():
    sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    sess.pose_idx = len(sess.script) - 1                 # the 'no glasses' step
    assert sess.current_pose.optional
    st = sess.skip_pose()
    assert st["message"] == "skipped" and sess.step is Step.BASELINE

    sess2 = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    st = sess2.skip_pose()                               # centre is not optional
    assert "cannot be skipped" in st["message"]
    print("ok  poses: only the optional glasses step may be skipped")


def test_drowsy_baseline_is_refused():
    """A baseline captured from an already-tired operator makes real fatigue
    look normal later — the dangerous direction. It must be refused outright."""
    sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
    drive_poses(sess)

    drowsy = calib_rows()
    for r in drowsy:
        r["perclos"] = 0.45          # eyes closed nearly half the time
        r["ear_mean"] = 0.14
    st = sess.set_baseline(drowsy)

    assert sess.step is not Step.REVIEW, "drowsy baseline was accepted"
    assert "drowsy" in st["message"], st["message"]
    assert not sess.baseline_summary

    assert sess.set_baseline(calib_rows())["message"] == "baseline captured"
    assert sess.step is Step.REVIEW
    print("ok  baseline: drowsy capture refused, a clean retry is accepted")


def _pose_landmarks(head_y=0.40):
    from brain.pose import L_EAR, L_SHOULDER, NOSE, R_EAR, R_SHOULDER
    pts = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]

    def put(i, x, y):
        pts[i] = SimpleNamespace(x=x, y=y, visibility=0.95)

    put(L_SHOULDER, 0.34, 0.70); put(R_SHOULDER, 0.66, 0.70)
    put(L_EAR, 0.45, head_y);    put(R_EAR, 0.55, head_y)
    put(NOSE, 0.50, head_y + 0.03)
    return pts


def test_posture_baseline_is_captured_in_the_same_sitting():
    """The 25 s fatigue baseline and the seated-posture baseline come from one
    sitting — the operator is already there and already still."""
    with tempfile.TemporaryDirectory() as td:
        store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
        sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
        drive_poses(sess)

        for _ in range(40):                       # what the baseline step feeds in
            sess.add_posture_sample(metrics_from_landmarks(_pose_landmarks()))
        sess.set_baseline(calib_rows())

        assert sess.posture_baseline is not None, "no posture baseline built"
        assert sess.review()["ready"]

        profiles: dict = {}
        assert sess.commit(profiles, store)["ok"]

        posture = profiles["op_900"]["calibration"].get("posture")
        assert posture and posture["samples"] >= 40, posture
        assert posture["head_height"][0] > 0
        print(f"ok  enrolment: posture baseline stored "
              f"({posture['samples']} samples) beside the fatigue baseline")


def test_enrolment_without_pose_still_succeeds_but_warns():
    """A machine with no pose model must still enrol operators — it just cannot
    monitor them with their face covered, and should say so."""
    with tempfile.TemporaryDirectory() as td:
        store = TemplateStore(path=Path(td) / "t.enc", keyfile=Path(td) / "k.key")
        sess = EnrollmentSession(details(), StubEmbedder(unit()), shuffle=False)
        drive_poses(sess)
        sess.set_baseline(calib_rows())           # no posture samples at all

        review = sess.review()
        assert review["ready"], "posture should not be required to enrol"
        assert any("face is covered" in w for w in review["warnings"]), review

        profiles: dict = {}
        assert sess.commit(profiles, store)["ok"]
        assert "posture" not in profiles["op_900"]["calibration"]
    print("ok  no pose model: enrolment still completes, with a clear warning")


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
