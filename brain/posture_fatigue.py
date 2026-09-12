"""
brain/posture_fatigue.py — a drowsiness estimate from posture alone.

This is the fallback that keeps the cab protected when sunglasses and a dust
mask make EAR, MAR and PERCLOS impossible.  It is deliberately NOT a model.

Why not a model.  Pipeline A's XGBoost works because it was trained on a
labelled drowsiness dataset of face features.  There is no equivalent labelled
pose dataset here, so a "pose model" would be a hand-tuned heuristic wearing an
ML costume — harder to audit, no more accurate, and impossible to defend when
somebody asks what it was trained on.  What is here instead is an explicit,
explainable score: how far each posture signal has drifted from THIS operator's
own seated baseline, in standard deviations, combined with stated weights.  It
produces the same shape of output as FatigueEngine so the rest of the pipeline
does not care which one is speaking, and it can always say why.

Why it must be treated as weaker evidence.  PERCLOS is predictive — eyelid
closure creeps up in the minutes before a microsleep.  A head drop is coincident
with one, or later.  So posture mode sees the same event later and with less
certainty.  That is bought back two ways, both outside this module: the engine
alerts earlier in posture mode, and it refuses to let a posture score alone
trigger a Level 3 stop.  Here we only promise an honest number and a reason.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .pose import PoseMetrics

# Signals we score, and what each contributes to the final probability.
# Head drop dominates: it is the least ambiguous of the three. Sway is the
# weakest — a cab shakes on its own, so a lot of it is the machine, not the
# operator — and it is weighted accordingly.
WEIGHTS: Dict[str, float] = {
    "head_drop": 0.50,
    "slump": 0.30,
    "sway": 0.20,
}

# Deviation, in baseline standard deviations, at which a signal counts as fully
# expressed. Generous because a seated baseline captured over 25 s underestimates
# the natural variation of a whole shift.
SIGMA_FULL = 3.0
SIGMA_ONSET = 1.0          # below this, a signal contributes nothing

SWAY_WINDOW_SEC = 10.0     # sway needs the longest view: one lean is not sway
SMOOTH_WINDOW = 5          # windows of smoothing on the final score

# A baseline std can be near zero if the operator sat unnaturally still during
# enrolment. Floor it, or a millimetre of real movement reads as 40 sigma.
MIN_STD = {"head_height": 0.02, "shoulder_y": 0.006, "head_forward": 0.02,
           "shoulder_width": 0.004, "sway": 0.004}


@dataclass
class PostureBaseline:
    """This operator's normal seated posture, captured during enrolment."""
    head_height: Tuple[float, float] = (0.0, 0.0)      # (mean, std)
    shoulder_y: Tuple[float, float] = (0.0, 0.0)
    head_forward: Tuple[float, float] = (0.0, 0.0)
    shoulder_width: Tuple[float, float] = (0.0, 0.0)
    sway: Tuple[float, float] = (0.0, 0.0)
    samples: int = 0

    def as_dict(self) -> dict:
        return {"head_height": list(self.head_height),
                "shoulder_y": list(self.shoulder_y),
                "head_forward": list(self.head_forward),
                "shoulder_width": list(self.shoulder_width),
                "sway": list(self.sway),
                "samples": self.samples}

    @staticmethod
    def from_dict(d: dict) -> Optional["PostureBaseline"]:
        if not isinstance(d, dict) or not d.get("samples"):
            return None
        try:
            return PostureBaseline(
                head_height=tuple(d["head_height"]), shoulder_y=tuple(d["shoulder_y"]),
                head_forward=tuple(d["head_forward"]),
                shoulder_width=tuple(d["shoulder_width"]),
                sway=tuple(d.get("sway", (0.0, 0.0))), samples=int(d["samples"]))
        except (KeyError, TypeError, ValueError):
            return None

    def is_usable(self) -> bool:
        return self.samples >= 8 and self.head_height[0] > 0.0


def build_baseline(samples: List[PoseMetrics], *,
                   fps: float = 2.5) -> Optional[PostureBaseline]:
    """Summarise a run of alert, seated posture into a personal baseline.

    Called with the same sitting that produces the fatigue baseline at enrolment —
    the operator is already there, already still, already being measured.
    """
    good = [m for m in samples if m.present]
    if len(good) < 8:
        return None

    def stat(vals: List[float], key: str) -> Tuple[float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(max(arr.std(), MIN_STD.get(key, 1e-3)))

    centres = [m.centre_x for m in good]
    win = max(2, int(SWAY_WINDOW_SEC * fps))
    sway_vals = [float(np.std(centres[max(0, i - win):i + 1]))
                 for i in range(1, len(centres))] or [0.0]

    return PostureBaseline(
        head_height=stat([m.head_height for m in good], "head_height"),
        shoulder_y=stat([m.shoulder_y for m in good], "shoulder_y"),
        head_forward=stat([m.head_forward for m in good], "head_forward"),
        shoulder_width=stat([m.shoulder_width for m in good], "shoulder_width"),
        sway=stat(sway_vals, "sway"),
        samples=len(good),
    )


def _ramp(sigma: float) -> float:
    """Deviation in sigmas -> 0..1, flat until SIGMA_ONSET then linear."""
    if sigma <= SIGMA_ONSET:
        return 0.0
    return float(min(1.0, (sigma - SIGMA_ONSET) / (SIGMA_FULL - SIGMA_ONSET)))


class PostureFatigue:
    """Scores posture against a personal baseline. Explains itself every tick."""

    def __init__(self, baseline: Optional[PostureBaseline] = None, *,
                 fps: float = 2.5) -> None:
        self.baseline = baseline
        self.fps = fps
        self.backend = "posture-baseline"
        self._centres: Deque[float] = deque(maxlen=max(2, int(SWAY_WINDOW_SEC * fps)))
        self._scores: Deque[float] = deque(maxlen=SMOOTH_WINDOW)

    @property
    def ready(self) -> bool:
        return self.baseline is not None and self.baseline.is_usable()

    def reset(self) -> None:
        self._centres.clear()
        self._scores.clear()

    def update(self, m: PoseMetrics) -> dict:
        """Score one posture sample.

        Mirrors FatigueEngine.update()'s output shape so the pipeline can consume
        either without special-casing — plus `source` so nothing downstream can
        mistake a posture estimate for a PERCLOS-backed one.
        """
        if not m.present:
            return self._out(0.0, [], "operator not visible", confident=False)
        if not self.ready:
            # No personal baseline means no notion of "slumped" for this person.
            # Say so rather than inventing a threshold from a population average.
            return self._out(0.0, [], "no posture baseline for this operator",
                             confident=False)

        b = self.baseline
        self._centres.append(m.centre_x)

        # 1. Head drop — chin toward chest. Only a DROP counts: sitting up
        #    straighter than baseline is not drowsiness.
        drop_sigma = max(0.0, (b.head_height[0] - m.head_height) / b.head_height[1])

        # 2. Slump — shoulders lower in frame, and/or leaning forward. Forward
        #    lean also foreshortens the shoulders, so a narrowing shoulder width
        #    corroborates it; take the strongest of the three.
        lower_sigma = max(0.0, (m.shoulder_y - b.shoulder_y[0]) / b.shoulder_y[1])
        fwd_sigma = max(0.0, (m.head_forward - b.head_forward[0]) / b.head_forward[1])
        narrow_sigma = max(0.0, (b.shoulder_width[0] - m.shoulder_width)
                           / b.shoulder_width[1])
        slump_sigma = max(lower_sigma, fwd_sigma, narrow_sigma)

        # 3. Sway — trunk control loosening. Needs a filled window first.
        if len(self._centres) >= self._centres.maxlen:
            sway = float(np.std(self._centres))
            sway_sigma = max(0.0, (sway - b.sway[0]) / b.sway[1])
        else:
            sway_sigma = 0.0

        parts = {"head_drop": _ramp(drop_sigma), "slump": _ramp(slump_sigma),
                 "sway": _ramp(sway_sigma)}
        raw = sum(WEIGHTS[k] * v for k, v in parts.items())

        self._scores.append(raw)
        smooth = float(np.mean(self._scores))

        sigmas = {"head_drop": drop_sigma, "slump": slump_sigma, "sway": sway_sigma}
        reasons = self._reasons(parts, sigmas)
        return self._out(smooth, reasons, "", confident=True, parts=parts,
                         sigmas=sigmas)

    # ── output shaping ──────────────────────────────────────────────────
    @staticmethod
    def _reasons(parts: Dict[str, float], sigmas: Dict[str, float]) -> List[dict]:
        labels = {"head_drop": "head dropped toward chest",
                  "slump": "shoulders slumped / leaning forward",
                  "sway": "swaying side to side"}
        # The score saturates at SIGMA_FULL, so a raw 34 SD says nothing more than
        # "far past the limit" — and reads as a broken gauge. Clamp what is shown.
        def shown(x: float) -> str:
            return "9.9+" if x > 9.9 else f"{x:.1f}"

        out = [{"feature": k, "value": round(min(sigmas[k], 99.9), 2),
                "direction": "up",
                "text": f"{labels[k]} ({shown(sigmas[k])} SD above your normal)"}
               for k, v in parts.items() if v > 0]
        return sorted(out, key=lambda r: -r["value"])

    def _out(self, p: float, reasons: List[dict], note: str, *, confident: bool,
             parts: Optional[dict] = None, sigmas: Optional[dict] = None) -> dict:
        if not confident:
            decision, severity = "NO SIGNAL", note or "posture unavailable"
        elif p >= 0.60:
            decision, severity = "AT-RISK", "!! STRONG"
        elif p >= 0.35:
            decision, severity = "AT-RISK", "! MILD"
        else:
            decision, severity = "ALERT", "OK"
        return {
            "p_at_risk": round(float(p), 4),
            "p_raw": round(float(p), 4),
            "decision": decision,
            "severity": severity,
            "reasons": reasons,
            # Everything downstream can see this came from posture, not eyes.
            "source": "posture",
            "confident": confident,
            "note": note,
            "parts": parts or {},
            "sigmas": {k: round(v, 2) for k, v in (sigmas or {}).items()},
        }
