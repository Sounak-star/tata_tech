"""
brain/enrollment.py — in-cab operator enrolment (profile + face + baseline).

Why enrolment happens in the cab and not at an HR desk:

  • Same sensor.  A template built from a bright office phone camera and a probe
    taken on a greasy 640x480 cab webcam are separated by a domain gap nobody
    needs to fight.  Enrol on the camera that will do the recognising.
  • Same 30 seconds.  live_camera.calibrate() already sits the operator down and
    records an ALERT fatigue baseline.  That is the same sitting.  One session
    now produces BOTH the face signature and the personal fatigue baseline, so
    recognition at sit-down can skip the 25 s calibration entirely.
  • Same privacy story, but true.  Frames are embedded and dropped inside the
    request. No photograph is created at any point — not written, not buffered,
    not held on the session object.  Only vectors and summary statistics leave
    this module.

The pose script doubles as a liveness challenge: the order is randomised, and a
printed photo cannot turn its head on demand.

    DETAILS ──► POSES (4-5 guided captures) ──► BASELINE (25 s) ──► REVIEW ──► COMMIT
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faceid import FaceEmbedder, Quality, assess_quality, l2
from .paths import DATA
from .personalize import SUPPORTED_LANGUAGES
from .pose import PoseMetrics
from .posture_fatigue import PostureBaseline, build_baseline
from .templates import TemplateStore

SUPERVISORS_JSON = Path(os.environ.get("SAARTHI_SUPERVISORS", str(DATA / "supervisors.json")))

# Bootstrap PIN for a machine that has never been enrolled on. supervisors.json
# is git-ignored (it holds PIN hashes), so a random bootstrap meant every fresh
# clone got a different PIN and nobody on the team could enrol without first
# reading it out of their own console. A shared, documented default fixes that;
# the cold-start banner says loudly that it is a default, and a real deployment
# overrides it with SAARTHI_ENROL_PIN or a later add_supervisor() call.
DEFAULT_ENROL_PIN = "2027"
BOOTSTRAP_PIN = os.environ.get("SAARTHI_ENROL_PIN", DEFAULT_ENROL_PIN)

# Field vocabularies — these are exactly what personalise() branches on, so a
# value outside them would silently degrade to a default delivery plan.
EXPERIENCE = ("trainee", "intermediate", "expert")
HEARING = ("normal", "impaired")
COLOR_VISION = ("normal", "deuteranopia", "protanopia", "tritanopia")

FRAMES_PER_POSE = 3
MIN_GAP_FRAMES = 4          # don't accept three near-identical consecutive frames
TURN_MIN_DEG = 14.0         # a "slight" head turn, comfortable in a seat
CHIN_MIN_DEG = 10.0
MIN_INTRA_COS = 0.35        # outlier gate — see _reject_outliers()
MIN_ACCEPTED_VECTORS = 3

# Same limits live_camera.calibrate() enforces. A baseline captured while the
# operator was already drowsy makes every later reading wrong in the dangerous
# direction (real fatigue looks "normal"), so it is refused, not warned about.
MAX_BASELINE_PERCLOS = 0.20
MIN_BASELINE_EAR = 0.20


class Step(str, Enum):
    DETAILS = "details"
    POSES = "poses"
    BASELINE = "baseline"
    REVIEW = "review"
    COMMITTED = "committed"
    ABORTED = "aborted"


# ── supervisor gate ──────────────────────────────────────────────────────
class SupervisorAuth:
    """PIN gate for Enrolment Mode.

    A PIN is weak authentication and we treat it as such: per-supervisor rather
    than shared, scrypt-hashed (never stored in the clear), rate-limited, and
    every enrolment is stamped with the supervisor id so there is an audit trail
    of who authorised a template. The audit trail is the part that matters.
    """

    MAX_FAILS = 5
    LOCKOUT_S = 900.0

    def __init__(self, path: Path = SUPERVISORS_JSON) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fails: Dict[str, List[float]] = {}
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._warn_if_stale(data)
            return data
        pin = BOOTSTRAP_PIN
        data = {"supervisors": {}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(data)
        self.add_supervisor("sup_001", "Site Supervisor", pin,
                            source="bootstrap")
        print("\n" + "=" * 68, flush=True)
        print(f"  ENROLMENT PIN CREATED — supervisor 'sup_001' PIN: {pin}", flush=True)
        if pin == DEFAULT_ENROL_PIN:
            print("  Shared default for the demo build. Override it with the", flush=True)
            print("  SAARTHI_ENROL_PIN env var before any real deployment.", flush=True)
        else:
            print("  From SAARTHI_ENROL_PIN. Change it with add_supervisor().", flush=True)
        print("=" * 68 + "\n", flush=True)
        return json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _warn_if_stale(data: dict) -> None:
        """Flag a machine still holding a pre-shared-default bootstrap PIN.

        The shared default only seeds a machine that has never run the server,
        so anyone who tested before it landed keeps their old random PIN and
        finds the documented one rejected with no explanation. Deliberately a
        WARNING and not an automatic re-seed: silently rewriting a credential at
        startup would also clobber a PIN somebody set on purpose.
        """
        rec = data.get("supervisors", {}).get("sup_001")
        if rec is None or rec.get("source"):
            return
        print("\n" + "!" * 68, flush=True)
        print("!  This machine has an OLD random enrolment PIN.", flush=True)
        print(f"!  The team default ({DEFAULT_ENROL_PIN}) will NOT work here — "
              f"supervisors.json", flush=True)
        print("!  predates it, and bootstrap only seeds a machine with no file.", flush=True)
        print("!", flush=True)
        print("!  Fix it with:   python -m brain.enrollment --reset-pin", flush=True)
        print("!" * 68 + "\n", flush=True)

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _hash(pin: str, salt: bytes) -> str:
        return hashlib.scrypt(pin.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex()

    def add_supervisor(self, sup_id: str, name: str, pin: str,
                       source: str = "manual") -> None:
        if len(pin) < 4:
            raise ValueError("PIN must be at least 4 digits")
        with self._lock:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            salt = secrets.token_bytes(16)
            data.setdefault("supervisors", {})[sup_id] = {
                "name": name, "salt": salt.hex(), "hash": self._hash(pin, salt),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # Which PIN this record was seeded from. Records written before
                # the shared default existed carry no marker, which is how we
                # spot a machine still holding a random bootstrap PIN.
                "source": source,
            }
            self._write(data)
            self._data = data

    def verify(self, sup_id: str, pin: str) -> Tuple[bool, str]:
        with self._lock:
            now = time.time()
            recent = [t for t in self._fails.get(sup_id, []) if now - t < self.LOCKOUT_S]
            self._fails[sup_id] = recent
            if len(recent) >= self.MAX_FAILS:
                wait = int((self.LOCKOUT_S - (now - recent[0])) / 60) + 1
                return False, f"too many attempts — locked for ~{wait} min"

            rec = self._data.get("supervisors", {}).get(sup_id)
            if rec is None:
                self._fails.setdefault(sup_id, []).append(now)
                return False, "unknown supervisor"

            ok = secrets.compare_digest(
                self._hash(pin, bytes.fromhex(rec["salt"])), rec["hash"])
            if not ok:
                self._fails.setdefault(sup_id, []).append(now)
                left = self.MAX_FAILS - len(self._fails[sup_id])
                return False, f"incorrect PIN ({left} attempt(s) left)"

            self._fails[sup_id] = []
            return True, rec.get("name", sup_id)


# ── the profile form ─────────────────────────────────────────────────────
@dataclass
class OperatorDetails:
    """Everything the personaliser needs, and nothing it does not."""
    id: str
    name: str
    role: str = "Operator"
    experience: str = "trainee"
    hearing: str = "normal"
    color_vision: str = "normal"
    language: str = "en"
    notes: str = ""

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            errs.append("id must be alphanumeric (underscores and dashes allowed)")
        if not self.name.strip():
            errs.append("name is required")
        if self.experience not in EXPERIENCE:
            errs.append(f"experience must be one of {EXPERIENCE}")
        if self.hearing not in HEARING:
            errs.append(f"hearing must be one of {HEARING}")
        if self.color_vision not in COLOR_VISION:
            errs.append(f"color_vision must be one of {COLOR_VISION}")
        if self.language not in SUPPORTED_LANGUAGES:
            errs.append(f"language must be one of {SUPPORTED_LANGUAGES}")
        return errs

    def to_profile(self) -> dict:
        prof = asdict(self)
        # Auto-note the accessibility consequences so the dashboard card explains
        # itself even when the supervisor left notes blank.
        if not prof["notes"]:
            bits = []
            if self.hearing == "impaired":
                bits.append("Hard of hearing — alerts must not rely on sound (buzz + flash).")
            if self.color_vision != "normal":
                bits.append(f"Colour-blind ({self.color_vision}) — blue/white + icons, "
                            "never red-vs-green alone.")
            if self.experience == "trainee":
                bits.append("Trainee — earlier warnings, simpler UI.")
            prof["notes"] = " ".join(bits)
        return prof


# ── the pose script ──────────────────────────────────────────────────────
@dataclass
class Pose:
    key: str
    prompt: str
    require_turn: bool = False
    require_chin: bool = False
    optional: bool = False


POSE_SCRIPT: List[Pose] = [
    Pose("center", "Look straight at the screen"),
    Pose("turn_a", "Turn your head slightly to one side", require_turn=True),
    Pose("turn_b", "Now turn slightly to the other side", require_turn=True),
    Pose("chin_down", "Tilt your chin down a little", require_chin=True),
    Pose("no_glasses", "If you wear glasses, take them off and look ahead",
         optional=True),
]


def build_script(shuffle: bool = True, rng: Optional[random.Random] = None) -> List[Pose]:
    """Centre first (it is the strongest anchor); the rest in random order.

    Randomising is not cosmetic — an unpredictable sequence is what makes
    completing the script evidence of a live human rather than a held-up photo.
    """
    head, tail = POSE_SCRIPT[0], POSE_SCRIPT[1:]
    if shuffle:
        tail = tail[:]
        (rng or random).shuffle(tail)
    return [head] + tail


def sanity_check_baseline(rows: List[dict]) -> Tuple[bool, str]:
    """Refuse a baseline captured from an operator who was not actually alert.

    This mirrors the check in live_camera.calibrate() so it holds for every
    tracker, including the browser-streamed path which does not go through
    calibrate() at all.
    """
    if not rows:
        return False, "no calibration windows captured"

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else 0.0

    perclos, ear = mean("perclos"), mean("ear_mean")
    if perclos > MAX_BASELINE_PERCLOS:
        return False, (f"baseline looks drowsy (PERCLOS {perclos:.2f}) — eyes were "
                       f"mostly closed. Sit up, look ahead and try again.")
    if 0.0 < ear < MIN_BASELINE_EAR:
        return False, (f"baseline looks drowsy (EAR {ear:.2f}) — eyes were mostly "
                       f"closed. Sit up, look ahead and try again.")
    return True, "ok"


# ── the session ──────────────────────────────────────────────────────────
@dataclass
class PoseProgress:
    key: str
    prompt: str
    needed: int
    captured: int = 0
    done: bool = False
    optional: bool = False


class EnrollmentSession:
    """One operator being enrolled. Holds vectors and numbers — never an image."""

    def __init__(self, details: OperatorDetails, embedder: FaceEmbedder, *,
                 supervisor: str = "unknown", machine_id: str = "local",
                 shuffle: bool = True, rng: Optional[random.Random] = None) -> None:
        self.details = details
        self.embedder = embedder
        self.supervisor = supervisor
        self.machine_id = machine_id
        self.started = time.time()

        self.step = Step.POSES
        self.script = build_script(shuffle, rng)
        self.progress = [PoseProgress(p.key, p.prompt, FRAMES_PER_POSE, optional=p.optional)
                         for p in self.script]
        self.pose_idx = 0

        self.vectors: List[np.ndarray] = []
        self.vector_poses: List[str] = []
        self.yaw_signs: Dict[str, float] = {}
        self.baseline: Optional[dict] = None
        self.baseline_summary: Dict[str, float] = {}
        self.baseline_progress = 0        # windows captured so far
        # Posture samples from the same sitting. The operator is already there
        # and already still, so a seated-posture baseline costs nothing extra —
        # and without one the posture fallback has no notion of this person's
        # normal and will refuse to score them.
        self.posture_samples: List[PoseMetrics] = []
        self.posture_baseline: Optional[PostureBaseline] = None
        self.last_quality: Optional[Quality] = None
        self.rejected = 0
        self.error: Optional[str] = None

        self._frames_since_accept = MIN_GAP_FRAMES

    # ── pose capture ────────────────────────────────────────────────────
    @property
    def current_pose(self) -> Optional[Pose]:
        return self.script[self.pose_idx] if self.pose_idx < len(self.script) else None

    def skip_pose(self) -> dict:
        """Only optional poses (e.g. 'no glasses') may be skipped."""
        pose = self.current_pose
        if pose is None or not pose.optional:
            return self.status("that step cannot be skipped")
        self.progress[self.pose_idx].done = True
        self._advance()
        return self.status("skipped")

    def feed_frame(self, bgr: np.ndarray, landmarks, *, yaw: float = 0.0,
                   pitch: float = 0.0) -> dict:
        """Offer one frame to the current pose. Returns UI-shaped progress.

        The frame is read, embedded and dropped. Nothing is retained.
        """
        if self.step is not Step.POSES:
            return self.status()

        pose = self.current_pose
        if pose is None:
            return self.status()

        self._frames_since_accept += 1

        quality = assess_quality(bgr, landmarks, strict=True, yaw=yaw, pitch=pitch)
        self.last_quality = quality
        if not quality.ok:
            return self.status(quality.reasons[0])

        # Pose gates. For the two turns we require magnitude only and then check
        # at commit time that they went opposite ways — that keeps us independent
        # of whichever yaw sign convention the head-pose solver happens to use.
        if pose.require_turn and abs(yaw) < TURN_MIN_DEG:
            return self.status("turn your head a little further")
        if pose.require_chin and abs(pitch) < CHIN_MIN_DEG:
            return self.status("tilt your chin down a little further")
        if not pose.require_turn and not pose.require_chin and abs(yaw) > 20.0:
            return self.status("face the camera squarely")

        if self._frames_since_accept < MIN_GAP_FRAMES:
            return self.status("hold that…")

        vec = self.embedder.embed_frame(bgr, landmarks)
        if vec is None:
            return self.status("could not read the face — adjust your position")

        self.vectors.append(vec)
        self.vector_poses.append(pose.key)
        if pose.require_turn:
            self.yaw_signs[pose.key] = float(np.sign(yaw))
        self._frames_since_accept = 0

        prog = self.progress[self.pose_idx]
        prog.captured += 1
        if prog.captured >= prog.needed:
            prog.done = True
            self._advance()
        return self.status("captured")

    def _advance(self) -> None:
        self.pose_idx += 1
        if self.pose_idx >= len(self.script):
            self.step = Step.BASELINE

    # ── baseline ────────────────────────────────────────────────────────
    def add_posture_sample(self, metrics: PoseMetrics) -> None:
        """Record one posture observation during the baseline capture."""
        if metrics is not None and metrics.present:
            self.posture_samples.append(metrics)

    def set_baseline(self, calibration_rows: List[dict]) -> dict:
        """Take the ALERT baseline from the same sitting.

        `calibration_rows` are exactly what live_camera.calibrate() returns, so
        the existing sanity checks (which refuse a drowsy baseline) still apply
        upstream. We keep the FULL 12-feature baseline, not just the three
        summary numbers, because that is what FatigueMonitor actually consumes —
        storing only the summary is why profiles.json currently cannot skip
        calibration.
        """
        if not calibration_rows:
            self.error = "calibration produced no usable windows"
            return self.status(self.error)

        ok, why = sanity_check_baseline(calibration_rows)
        if not ok:
            self.error = why
            return self.status(why)

        try:
            from fatigue_monitor import calibrate      # via paths.py sys.path wiring
            self.baseline = calibrate(calibration_rows)
        except Exception as exc:                        # noqa: BLE001
            self.baseline = None
            print(f"[Enrol] full baseline unavailable ({exc}); keeping summary only.",
                  flush=True)

        def _mean(key: str) -> float:
            vals = [r[key] for r in calibration_rows if key in r]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        self.baseline_summary = {
            "ear_baseline": _mean("ear_mean"),
            "perclos_baseline": _mean("perclos"),
            "blink_rate_baseline": _mean("blink_rate"),
            "calibrated": True,
        }

        # Posture is optional: a machine with no pose model still enrols fine,
        # it just cannot fall back to posture for this operator later.
        self.posture_baseline = build_baseline(self.posture_samples)
        if self.posture_baseline is None and self.posture_samples:
            print(f"[Enrol] only {len(self.posture_samples)} posture samples — "
                  f"no posture baseline for this operator.", flush=True)
        self.step = Step.REVIEW
        return self.status("baseline captured")

    # ── outlier rejection + commit ──────────────────────────────────────
    def _reject_outliers(self) -> Tuple[List[np.ndarray], List[str]]:
        """Drop vectors that do not look like the same person.

        The gate is deliberately loose. Enrolment poses are *meant* to differ, so
        a tight threshold would throw away the variation we came for; what we are
        catching is the genuinely wrong thing — a second person stepping into
        frame, or a crop that landed on a headrest.
        """
        if len(self.vectors) < 2:
            return list(self.vectors), []
        mat = np.stack(self.vectors)
        sims = mat @ mat.T
        np.fill_diagonal(sims, np.nan)
        med = np.nanmedian(sims, axis=1)
        keep, dropped = [], []
        for i, m in enumerate(med):
            if m >= MIN_INTRA_COS:
                keep.append(self.vectors[i])
            else:
                dropped.append(f"{self.vector_poses[i]} ({m:.2f})")
        return keep, dropped

    def review(self) -> dict:
        keep, dropped = self._reject_outliers()
        warnings: List[str] = []
        if dropped:
            warnings.append(f"{len(dropped)} frame(s) rejected as inconsistent: "
                            + ", ".join(dropped))
        signs = [s for s in self.yaw_signs.values() if s != 0]
        if len(signs) >= 2 and len(set(signs)) < 2:
            warnings.append("both turn poses went the same way — pose variety is reduced")
        if self.baseline is None:
            warnings.append("full fatigue baseline unavailable; summary only")
        if self.posture_baseline is None:
            warnings.append("no posture baseline captured — this operator cannot "
                            "be monitored while their face is covered")
        return {"vectors_kept": len(keep), "vectors_dropped": len(dropped),
                "warnings": warnings,
                "ready": len(keep) >= MIN_ACCEPTED_VECTORS and bool(self.baseline_summary)}

    def commit(self, profiles: dict, store: TemplateStore) -> dict:
        """Write the profile record and the encrypted template. Atomic-ish:
        the template is only stored once the profile validates."""
        errs = self.details.validate()
        if errs:
            self.error = "; ".join(errs)
            return {"ok": False, "errors": errs}

        keep, dropped = self._reject_outliers()
        if len(keep) < MIN_ACCEPTED_VECTORS:
            self.error = (f"only {len(keep)} usable face captures "
                          f"(need {MIN_ACCEPTED_VECTORS}) — re-run enrolment")
            return {"ok": False, "errors": [self.error]}
        if not self.baseline_summary:
            self.error = "no fatigue baseline captured"
            return {"ok": False, "errors": [self.error]}

        profile = self.details.to_profile()
        profile["calibration"] = dict(self.baseline_summary)
        if self.baseline:
            # The full 12-feature baseline — this is what lets recognition skip
            # the 25 s calibration at sit-down.
            profile["calibration"]["baseline"] = {
                k: {"mean": float(v["mean"]), "std": float(v["std"])}
                for k, v in self.baseline.items()
            }
        if self.posture_baseline is not None:
            # Seated-posture normal, for the sunglasses/dust-mask fallback.
            profile["calibration"]["posture"] = self.posture_baseline.as_dict()
        profile["enrolled"] = {
            "by": self.supervisor,
            "machine": self.machine_id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "poses": [p.key for p in self.script],
            "vectors": len(keep),
        }
        profiles[self.details.id] = profile

        store.enroll(self.details.id, keep, enrolled_by=self.supervisor,
                     machine_id=self.machine_id)

        self.step = Step.COMMITTED
        return {"ok": True, "operator": profile, "vectors": len(keep),
                "dropped": len(dropped),
                "template": store.summary(self.details.id)}

    def abort(self) -> dict:
        self.step = Step.ABORTED
        self.vectors.clear()            # vectors die with the session
        return self.status("aborted")

    # ── UI shape ────────────────────────────────────────────────────────
    def status(self, message: str = "") -> dict:
        pose = self.current_pose
        return {
            "step": self.step.value,
            "operator_id": self.details.id,
            "pose": pose.key if pose else None,
            "prompt": pose.prompt if pose else None,
            "optional": pose.optional if pose else False,
            "pose_index": self.pose_idx,
            "pose_count": len(self.script),
            "progress": [asdict(p) for p in self.progress],
            "captured": len(self.vectors),
            "baseline_progress": self.baseline_progress,
            "posture_samples": len(self.posture_samples),
            "posture_ready": self.posture_baseline is not None,
            "message": message,
            "quality": self.last_quality.as_dict() if self.last_quality else None,
            "error": self.error,
            "elapsed": round(time.time() - self.started, 1),
        }


def _cli() -> int:
    """`python -m brain.enrollment --reset-pin [PIN]`

    Resets supervisor sup_001 to the shared default, or to a PIN given on the
    command line. Exists because the shared-default bootstrap cannot reach a
    machine that already has a supervisors.json.
    """
    import sys

    argv = sys.argv[1:]
    if "--reset-pin" not in argv:
        print(__doc__)
        return 0

    idx = argv.index("--reset-pin")
    pin = argv[idx + 1] if len(argv) > idx + 1 and not argv[idx + 1].startswith("-") \
        else BOOTSTRAP_PIN

    auth = SupervisorAuth()
    try:
        auth.add_supervisor("sup_001", "Site Supervisor", pin, source="reset")
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1

    ok, who = auth.verify("sup_001", pin)
    print(f"supervisor sup_001 PIN set to {pin} — verify: "
          f"{'OK (' + who + ')' if ok else 'FAILED'}")
    print(f"stored at {auth.path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
