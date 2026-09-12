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

import math
import sys
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
        if calibration_rows is None:
            # No camera at all — synthetic is the only option, and it's known.
            rows = [synth_window(rng.uniform(0.0, 0.18), rng) for _ in range(60)]
            print("[FatigueEngine] No camera — using synthetic alert baseline (offline demo).",
                  file=sys.stderr, flush=True)
        elif len(calibration_rows) == 0:
            # calibrate() should have retried until it got real rows.  Reaching
            # here means something bypassed that logic — refuse to silently use
            # synthetic, because results would be unreliable for a live operator.
            raise RuntimeError(
                "[FatigueEngine] Received empty calibration_rows from a live-camera "
                "session.  calibrate() must retry until it captures real face data — "
                "aborting rather than silently using a synthetic baseline."
            )
        else:
            rows = calibration_rows
        try:
            from fatigue_monitor import FatigueMonitor, calibrate

            baseline = calibrate(rows)
            # smooth_window=3 (~4.5 s): fast enough to show recovery when eyes open;
            # minimum safe value — below 3 a single long blink would false-alarm.
            self._monitor = FatigueMonitor(baseline, smooth_window=3)
            self.backend = "xgboost"

            # Diagnostic: print the baseline so we can verify calibration worked.
            _src = "WEBCAM" if calibration_rows else "SYNTHETIC-FALLBACK"
            print(f"[FatigueEngine] Baseline ({_src}, {len(rows)} rows):",
                  file=sys.stderr, flush=True)
            _bl = baseline  # shorthand for sanity-flag checks below
            for feat, stats in _bl.items():
                _flag = ""
                if feat == "blink_rate":
                    if stats["std"] < 0.5:
                        _flag = "  ⚠ FLAG: std near-zero — blink_rate was nearly constant during calib!"
                    if stats["mean"] < 2.0:
                        _flag += "  ⚠ FLAG: mean implausibly low — held still during calibration?"
                if feat == "perclos" and stats["mean"] > 0.20:
                    _flag = f"  ⚠ FLAG: PERCLOS mean={stats['mean']:.3f} — calibration captured DROWSY state, NOT alert!"
                if feat == "ear_mean" and stats["mean"] < 0.20:
                    _flag = f"  ⚠ FLAG: ear_mean mean={stats['mean']:.3f} — eyes looked mostly closed during calibration!"
                print(f"  {feat}: mean={stats['mean']:.4f} std={stats['std']:.4f}{_flag}",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            self._err = str(exc)
            self._baseline = self._simple_baseline(rows)
            print(f"[FatigueEngine] XGBoost unavailable ({exc}); using fallback heuristic.",
                  file=sys.stderr, flush=True)

    # ── personal baseline from enrolment ────────────────────────────────
    @classmethod
    def from_baseline(cls, baseline: Dict[str, dict]) -> Optional["FatigueEngine"]:
        """Build an engine from a baseline captured at enrolment.

        This is what lets face recognition skip the 25-second calibration: the
        operator already sat through it once, in this cab, on this camera, and
        the result was stored on their profile.

        Returns None if the stored baseline is incomplete or malformed — the
        caller must then fall back to a live calibration.  Spending 25 seconds
        is always better than running a live operator against a broken baseline.
        """
        try:
            from fatigue_monitor import FatigueMonitor
        except Exception as exc:  # noqa: BLE001
            print(f"[FatigueEngine] cannot use stored baseline ({exc}).",
                  file=sys.stderr, flush=True)
            return None

        if not isinstance(baseline, dict):
            return None
        missing = [f for f in FEATURES if f not in baseline]
        if missing:
            print(f"[FatigueEngine] stored baseline is missing {missing} — "
                  f"falling back to live calibration.", file=sys.stderr, flush=True)
            return None

        try:
            clean = {f: {"mean": float(baseline[f]["mean"]), "std": float(baseline[f]["std"])}
                     for f in FEATURES}
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[FatigueEngine] stored baseline is malformed ({exc}) — "
                  f"falling back to live calibration.", file=sys.stderr, flush=True)
            return None

        self = cls.__new__(cls)
        self.backend = "xgboost"
        self._monitor = None
        try:
            self._monitor = FatigueMonitor(clean, smooth_window=3)
        except Exception as exc:  # noqa: BLE001
            print(f"[FatigueEngine] FatigueMonitor rejected the stored baseline ({exc}).",
                  file=sys.stderr, flush=True)
            return None

        print(f"[FatigueEngine] Baseline (ENROLMENT, {len(clean)} features) — "
              f"calibration skipped.", file=sys.stderr, flush=True)
        return self

    # ── public ──────────────────────────────────────────────────────────
    def update(self, features: Dict[str, float]) -> dict:
        if self._monitor is not None:
            bl = self._monitor.baseline

            # ── Step-1 instrumentation ───────────────────────────────────────
            # 1. Key raw live features
            br_raw   = features.get("blink_rate", math.nan)
            pc_raw   = features.get("perclos",    math.nan)
            ear_raw  = features.get("ear_mean",   math.nan)
            yawn_raw = features.get("is_yawn",    math.nan)
            eyes_open = ear_raw >= 0.20 and pc_raw < 0.30

            # 2. Normalized values the model sees
            norm_vals = {
                f: (features.get(f, math.nan) - bl[f]["mean"]) / (bl[f]["std"] + 1e-6)
                for f in FEATURES
            }
            _nan_n = sum(1 for v in norm_vals.values() if math.isnan(v))

            # 3. Smoothing buffer info (before this window is appended)
            _buf_len   = self._monitor.window_count
            _buf_limit = self._monitor._buffer.maxlen
            _buf_sec   = _buf_len * 1.5

            print(
                f"  [FATIGUE-DIAG] "
                f"EYES={'OPEN' if eyes_open else 'CLOSED'} "
                f"EAR={ear_raw:.3f} blink_rate={br_raw:.1f} perclos={pc_raw:.3f} is_yawn={yawn_raw:.0f} "
                f"| buf={_buf_len}/{_buf_limit} ({_buf_sec:.1f}s) | NaNs={_nan_n}",
                file=sys.stderr, flush=True,
            )
            # Baseline for the two most diagnostic features
            print(
                f"  [BASELINE]    "
                f"blink_rate mean={bl['blink_rate']['mean']:.2f} std={bl['blink_rate']['std']:.2f}  "
                f"perclos   mean={bl['perclos']['mean']:.3f}  std={bl['perclos']['std']:.3f}",
                file=sys.stderr, flush=True,
            )
            # Key normalized values (what the model actually receives)
            print(
                f"  [NORM]        "
                f"blink_rate={norm_vals.get('blink_rate', float('nan')):.2f}  "
                f"perclos={norm_vals.get('perclos', float('nan')):.2f}  "
                f"ear_mean={norm_vals.get('ear_mean', float('nan')):.2f}  "
                f"is_yawn={norm_vals.get('is_yawn', float('nan')):.2f}",
                file=sys.stderr, flush=True,
            )
            # ─────────────────────────────────────────────────────────────────

            res = self._monitor.update(features)

            # 4. Raw vs smoothed P side-by-side + SHAP top drivers
            print(
                f"  [P]           "
                f"raw={res.get('p_raw', float('nan')):.4f}  "
                f"smooth={res['p_at_risk']:.4f}  "
                f"decision={res['decision']}  severity={res['severity']}",
                file=sys.stderr, flush=True,
            )
            if res.get("reasons"):
                _top = res["reasons"][:3]
                # reasons from FatigueMonitor are TUPLES (feature, value, direction)
                _rstr = "  ".join(f"{feat}={val}({dr})" for (feat, val, dr) in _top)
                print(f"  [SHAP-TOP3]   {_rstr}", file=sys.stderr, flush=True)

            res["reasons"] = [
                {"feature": feat, "value": round(float(val), 3), "direction": dr}
                for (feat, val, dr) in res["reasons"]
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
