"""
brain/fatigue.py — Q2: "Is the operator tired/distracted?"

Wraps Pipeline A's trained XGBoost fatigue classifier (FatigueMonitor) for the
live loop, including the 30-second personal calibration and SHAP-driven reasons.

In the full product the 12 features come from MediaPipe (extract_features.py) on
the cabin webcam. When MediaPipe/webcam aren't available (offline demo, weak
laptop), `synth_window()` produces realistic feature windows so the REAL trained
model still drives the decision — only the feature *source* is simulated.

If xgboost/shap can't be imported at all, a transparent heuristic fallback keeps
the loop alive (clearly flagged via `engine.backend`).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

# 12 canonical features (must match feature_names.json).
FEATURES: List[str] = [
    "ear_mean", "ear_min", "ear_std", "perclos", "blink_rate",
    "mar_mean", "mar_max", "is_yawn", "pitch_mean", "yaw_mean",
    "roll_mean", "pitch_std",
]


def synth_window(drowsiness: float, rng: Optional[random.Random] = None) -> Dict[str, float]:
    """Generate one ~1.5s feature window for a given drowsiness level (0..1).

    0.0 = wide awake, 1.0 = micro-sleeping. Used for the offline demo and for
    calibration (call with drowsiness≈0 to get 'alert' baseline rows).
    """
    r = rng or random
    d = max(0.0, min(1.0, drowsiness))
    j = r.uniform  # jitter helper
    return {
        "ear_mean":  0.31 - 0.16 * d + j(-0.01, 0.01),
        "ear_min":   0.24 - 0.18 * d + j(-0.01, 0.01),
        "ear_std":   0.02 + 0.04 * d + j(-0.005, 0.005),
        "perclos":   0.04 + 0.70 * d + j(-0.02, 0.02),
        "blink_rate": 14.0 - 8.0 * d + j(-1.5, 1.5),
        "mar_mean":  0.18 + 0.25 * d + j(-0.02, 0.02),
        "mar_max":   0.30 + 0.45 * d + j(-0.03, 0.03),
        "is_yawn":   1.0 if (d > 0.5 and r.random() < d) else 0.0,
        "pitch_mean": 2.0 + 12.0 * d + j(-2.0, 2.0),   # head nodding forward
        "yaw_mean":   j(-5.0, 5.0),
        "roll_mean":  j(-4.0, 4.0),
        "pitch_std":  1.0 + 5.0 * d + j(-0.5, 0.5),
    }


class FatigueEngine:
    """Live fatigue assessment with personal calibration."""

    def __init__(self, calibration_rows: Optional[List[dict]] = None) -> None:
        self.backend = "fallback"
        self._monitor = None
        rng = random.Random(11)
        # Calibrate on a realistic spread of ALERT-state windows (low drowsiness
        # with natural variation) so per-feature std reflects normal variation —
        # otherwise tiny std inflates every later deviation into "at-risk".
        rows = calibration_rows or [
            synth_window(rng.uniform(0.0, 0.18), rng) for _ in range(60)
        ]
        try:
            from fatigue_monitor import FatigueMonitor, calibrate

            baseline = calibrate(rows)
            self._monitor = FatigueMonitor(baseline)
            self.backend = "xgboost"
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            self._err = str(exc)
            self._baseline = self._simple_baseline(rows)

    # ── public ──────────────────────────────────────────────────────────
    def update(self, features: Dict[str, float]) -> dict:
        if self._monitor is not None:
            res = self._monitor.update(features)
            res["reasons"] = [
                {"feature": f, "value": round(float(v), 3), "direction": dr}
                for (f, v, dr) in res["reasons"]
            ]
            return res
        return self._fallback_update(features)

    def reset(self) -> None:
        if self._monitor is not None:
            self._monitor.reset()

    # ── fallback (no xgboost/shap) ──────────────────────────────────────
    @staticmethod
    def _simple_baseline(rows: List[dict]) -> Dict[str, float]:
        n = len(rows)
        return {
            "perclos": sum(r["perclos"] for r in rows) / n,
            "ear_mean": sum(r["ear_mean"] for r in rows) / n,
        }

    def _fallback_update(self, f: Dict[str, float]) -> dict:
        # Transparent heuristic: PERCLOS and eye-closure dominate drowsiness.
        p = max(
            0.0,
            min(1.0, 1.2 * f.get("perclos", 0.0) + 0.6 * (0.30 - f.get("ear_mean", 0.30))),
        )
        decision = "AT-RISK" if p >= 0.40 else "ALERT"
        severity = "STRONG" if p >= 0.70 else ("MILD" if p >= 0.40 else "OK")
        reasons = [
            {"feature": "perclos", "value": round(f.get("perclos", 0.0), 3), "direction": "high"},
            {"feature": "ear_mean", "value": round(f.get("ear_mean", 0.0), 3), "direction": "low"},
        ]
        return {"decision": decision, "p_at_risk": round(p, 4), "p_raw": round(p, 4),
                "severity": severity, "reasons": reasons}
