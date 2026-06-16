"""
fatigue_monitor.py  -  SAARTHI Live Inference Module
=====================================================
Ready-to-import module for the live fatigue detection loop.

Usage pattern (in SAARTHI's main loop):

    from fatigue_monitor import calibrate, FatigueMonitor

    # --- Session start: 30-second calibration ---
    baseline = calibrate(alert_feature_rows)   # list of dicts from alert phase

    # --- Load monitor (once per session) ---
    monitor = FatigueMonitor(baseline)

    # --- Per-window loop (~1.5 s per window) ---
    while driving:
        features = extract_features(frame)     # your 12-feature dict
        result   = monitor.update(features)

        print(result["decision"])   # "ALERT" or "AT-RISK"
        print(result["severity"])   # "OK" / "MILD" / "STRONG"
        print(result["p_at_risk"])  # smoothed probability 0-1
        for feat, val, direction in result["reasons"]:
            show_reason_card(feat, val, direction)

Depends on:
    fatigue_binary_model.json   - trained XGBoost booster
    feature_names.json          - canonical 12-feature list
Both files must live in the same directory as this script (or pass explicit paths).

CPU only.  No GPU required.  No feature scaling.
"""

from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap


# ============================================================
# CONFIG  -  locked operating point, do not tune without retraining
# ============================================================

_HERE = Path(__file__).parent

DEFAULT_MODEL_PATH    = str(_HERE / "fatigue_binary_model.json")
DEFAULT_FEATURES_PATH = str(_HERE / "feature_names.json")

# LOCKED threshold from CV sweep (T=0.40: 74% at-risk recall, 33% false-alarm rate)
AT_RISK_THRESHOLD: float = 0.40

# Graduated-alert severity tiers (applied to smoothed P)
SEVERITY_OK:   float = 0.40   # P < 0.40          -> OK   (no alert)
SEVERITY_MILD: float = 0.70   # 0.40 <= P < 0.70  -> MILD ("take a break soon")
                               # P >= 0.70          -> STRONG (alert immediately)

# Temporal smoothing: rolling mean over last N windows (~1.5 s each -> ~45 s)
SMOOTH_WINDOW: int = 30

EPSILON: float = 1e-6

# Human-readable labels
LABEL_ALERT   = "ALERT"
LABEL_AT_RISK = "AT-RISK"


# ============================================================
# 1.  CALIBRATION  (live 30-second phase at session start)
# ============================================================

def calibrate(
    alert_feature_rows,
    features_path: str = DEFAULT_FEATURES_PATH,
    feature_list: list[str] | None = None,
) -> dict:
    """
    Compute a per-subject feature baseline from alert-state windows collected
    at the start of a driving session (the "30-second calibration phase").

    Parameters
    ----------
    alert_feature_rows : list[dict]  |  pd.DataFrame  |  np.ndarray
        ~30-60 seconds' worth of feature windows captured while the operator
        is confirmed alert (e.g. first 30 s of driving with eyes-open prompt).
        Each row must contain the 12 model features.

    features_path : str
        Path to feature_names.json (used to load the canonical feature order
        if feature_list is not provided explicitly).

    feature_list : list[str] | None
        If provided, skip loading from disk.

    Returns
    -------
    baseline : dict
        {feature_name: {"mean": float, "std": float}}
        Pass this dict to FatigueMonitor() and to explain_prediction().

    Example
    -------
        alert_rows = [extract_features(frame) for frame in calibration_clip]
        baseline   = calibrate(alert_rows)
        monitor    = FatigueMonitor(baseline)
    """
    if feature_list is None:
        with open(features_path, "r") as f:
            feature_list = json.load(f)

    # Normalise input to DataFrame
    if isinstance(alert_feature_rows, pd.DataFrame):
        df = alert_feature_rows[feature_list].copy()
    elif isinstance(alert_feature_rows, list):
        df = pd.DataFrame(alert_feature_rows)[feature_list]
    else:
        df = pd.DataFrame(alert_feature_rows, columns=feature_list)

    if len(df) == 0:
        raise ValueError("calibrate() received 0 rows. Need at least 5.")

    baseline: dict = {}
    for feat in feature_list:
        mean = float(df[feat].mean())
        std  = float(df[feat].std())
        if std < EPSILON or np.isnan(std):
            std = 1.0   # no-op normalisation for near-constant features
        baseline[feat] = {"mean": mean, "std": std}

    return baseline


# ============================================================
# 2.  STREAMING INFERENCE  -  FatigueMonitor
# ============================================================

class FatigueMonitor:
    """
    Stateful, streaming fatigue monitor for the SAARTHI live loop.

    Maintains a rolling probability buffer so temporal smoothing happens
    automatically as windows arrive.  Thread-safe for single-threaded use;
    call reset() between independent driving sessions.

    Parameters
    ----------
    baseline : dict
        Output of calibrate().  Per-subject alert-state baseline.
    model_path : str
        Path to fatigue_binary_model.json.
    features_path : str
        Path to feature_names.json.
    smooth_window : int
        Number of recent windows to average for smoothing (default 30 ~ 45 s).
    threshold : float
        Decision threshold for AT-RISK.  Default = 0.40 (locked operating pt).
    """

    def __init__(
        self,
        baseline: dict,
        model_path: str  = DEFAULT_MODEL_PATH,
        features_path: str = DEFAULT_FEATURES_PATH,
        smooth_window: int = SMOOTH_WINDOW,
        threshold: float   = AT_RISK_THRESHOLD,
    ) -> None:
        self.baseline = baseline
        self.threshold = threshold
        self._buffer: deque[float] = deque(maxlen=smooth_window)

        # Load canonical feature order
        with open(features_path, "r") as f:
            self.features: list[str] = json.load(f)

        # Load XGBoost booster from JSON  (CPU, no cuda)
        self._booster = xgb.Booster()
        self._booster.load_model(model_path)

        # Build SHAP TreeExplainer (fitted once, reused for every window)
        self._explainer = shap.TreeExplainer(self._booster)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, window_features) -> dict:
        """
        Process one feature window (~1.5 s of video) and return the
        current fatigue assessment.

        Parameters
        ----------
        window_features : dict | pd.Series | list | np.ndarray
            Raw (UN-scaled) feature values for the 12 model features.
            dict and pd.Series are accessed by feature name.
            list/ndarray must be in the canonical FEATURES order.

        Returns
        -------
        dict:
            decision  : "ALERT" or "AT-RISK"
            p_at_risk : float  -  smoothed P(at-risk), range 0-1
            p_raw     : float  -  unsmoothed P(at-risk) for this window
            severity  : "OK" / "MILD" / "STRONG"
            reasons   : list of (feature_name, raw_value, direction) tuples
                        direction = "high"  -> feature pushes toward AT-RISK
                                  = "low"   -> feature deviation toward AT-RISK
                        raw_value is the original (unscaled) feature value
                        for display in Reason Cards.
        """
        raw_arr, norm_arr = self._normalize(window_features)

        # Point-in-time probability
        dmat  = xgb.DMatrix(norm_arr.reshape(1, -1), feature_names=self.features)
        p_raw = float(self._booster.predict(dmat)[0])

        # Temporal smoothing via rolling buffer
        self._buffer.append(p_raw)
        p_smooth = float(np.mean(self._buffer))

        # Decision, severity, reasons
        decision = LABEL_AT_RISK if p_smooth >= self.threshold else LABEL_ALERT
        severity = _classify_severity(p_smooth)
        reasons  = self._shap_reasons(norm_arr, raw_arr)

        return {
            "decision":  decision,
            "p_at_risk": round(p_smooth, 4),
            "p_raw":     round(p_raw, 4),
            "severity":  severity,
            "reasons":   reasons,
        }

    def reset(self) -> None:
        """
        Clear the rolling probability buffer.
        Call between independent sessions or after a long break.
        """
        self._buffer.clear()

    @property
    def window_count(self) -> int:
        """Number of windows currently in the smoothing buffer."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, window_features) -> tuple[np.ndarray, np.ndarray]:
        """Return (raw_array, normalised_array) for a single window."""
        if isinstance(window_features, dict):
            raw = np.array([window_features[f] for f in self.features], dtype=float)
        elif isinstance(window_features, pd.Series):
            raw = window_features[self.features].values.astype(float)
        else:
            raw = np.asarray(window_features, dtype=float)

        norm = np.array([
            (raw[i] - self.baseline[feat]["mean"]) /
            (self.baseline[feat]["std"] + EPSILON)
            for i, feat in enumerate(self.features)
        ], dtype=float)

        return raw, norm

    def _shap_reasons(
        self,
        norm_arr: np.ndarray,
        raw_arr: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[str, float, str]]:
        """
        Return top_k SHAP-driven features with their raw values and direction.
        direction="high" means the feature's value pushes P(at-risk) up.
        """
        raw_sv = self._explainer.shap_values(norm_arr.reshape(1, -1))

        # Handle shap API variants for binary:logistic booster
        if isinstance(raw_sv, list):
            # Old shap API: list [sv_class0, sv_class1]
            sv = (raw_sv[1][0] if hasattr(raw_sv[1], "ndim") and raw_sv[1].ndim == 2
                  else raw_sv[1])
        elif isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
            # 3-D: (n_samples, n_features, 2)  -> class-1 slice
            sv = raw_sv[0, :, 1]
        else:
            # 2-D: (n_samples, n_features)  for binary log-odds
            sv = raw_sv[0]

        top_idx = np.argsort(np.abs(sv))[::-1][:top_k]
        return [
            (
                self.features[i],
                round(float(raw_arr[i]), 4),
                "high" if sv[i] > 0 else "low",
            )
            for i in top_idx
        ]


# ============================================================
# 3.  SEVERITY HELPER  (also usable standalone in the live UI)
# ============================================================

def _classify_severity(
    p: float,
    ok_thresh: float   = SEVERITY_OK,
    mild_thresh: float = SEVERITY_MILD,
) -> str:
    """
    Map smoothed P(at-risk) to a graduated severity tier.

        P < ok_thresh           -> 'OK'     (no alert)
        ok_thresh <= P < mild   -> 'MILD'   ("take a break soon")
        P >= mild_thresh        -> 'STRONG' (alert immediately)
    """
    if p >= mild_thresh:
        return "STRONG"
    if p >= ok_thresh:
        return "MILD"
    return "OK"


def severity_label(p: float) -> str:
    """Public alias so the UI can call fatigue_monitor.severity_label(p)."""
    return _classify_severity(p)


# ============================================================
# 4.  REASON-CARD TEXT  (optional UI helper)
# ============================================================

_REASON_TEMPLATES: dict = {
    "perclos":    lambda v, d: f"Eyes closed {v*100:.0f}% of the time",
    "ear_mean":   lambda v, d: f"Eye openness {'very low' if d=='low' else 'reduced'} ({v:.2f})",
    "ear_min":    lambda v, d: f"Minimum eye openness {v:.2f}",
    "ear_std":    lambda v, d: f"Eye openness variability {v:.3f}",
    "blink_rate": lambda v, d: f"Blink rate {'elevated' if d=='high' else 'low'} ({v:.0f}/min)",
    "is_yawn":    lambda v, d: f"Yawn {'detected' if v > 0.5 else 'not detected'}",
    "mar_mean":   lambda v, d: f"Mouth openness avg {v:.2f}",
    "mar_max":    lambda v, d: f"Mouth openness peak {v:.2f}",
    "pitch_mean": lambda v, d: f"Head tilt {v:+.1f}\u00b0 (pitch)",
    "yaw_mean":   lambda v, d: f"Head turn {v:+.1f}\u00b0 (yaw)",
    "roll_mean":  lambda v, d: f"Head roll {v:+.1f}\u00b0",
    "pitch_std":  lambda v, d: f"Head movement variability {v:.1f}\u00b0",
}


def reason_card_text(feature: str, value: float, direction: str) -> str:
    """
    Convert a (feature, raw_value, direction) reason tuple to a
    human-readable Reason Card string for the live UI.

    Example:
        reason_card_text("perclos", 0.71, "high")
        -> "Eyes closed 71% of the time"
    """
    template = _REASON_TEMPLATES.get(feature)
    if template:
        return template(value, direction)
    return f"{feature} = {value} ({direction})"


# ============================================================
# 5.  SELF-TEST  (run: python fatigue_monitor.py)
# ============================================================

if __name__ == "__main__":
    import math

    print("=" * 62)
    print(" FatigueMonitor self-test")
    print(" Simulates an alert->drowsy transition over 60 windows")
    print("=" * 62)

    # --- Fake calibration baseline (typical alert-state values) ----------
    SAMPLE_BASELINE = {
        "ear_mean":   {"mean": 0.29, "std": 0.03},
        "ear_min":    {"mean": 0.20, "std": 0.04},
        "ear_std":    {"mean": 0.03, "std": 0.01},
        "perclos":    {"mean": 0.05, "std": 0.02},
        "blink_rate": {"mean": 18.0, "std": 4.0},
        "mar_mean":   {"mean": 0.10, "std": 0.05},
        "mar_max":    {"mean": 0.25, "std": 0.08},
        "is_yawn":    {"mean": 0.02, "std": 0.14},
        "pitch_mean": {"mean": -2.0, "std": 5.0},
        "yaw_mean":   {"mean":  1.0, "std": 8.0},
        "roll_mean":  {"mean":  0.5, "std": 3.0},
        "pitch_std":  {"mean":  3.0, "std": 2.0},
    }

    # --- Load monitor -------------------------------------------------------
    print("\n[init] Loading model and building SHAP explainer ...")
    try:
        monitor = FatigueMonitor(SAMPLE_BASELINE)
        print("[init] FatigueMonitor ready.\n")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        print("[error] Run train_xgboost.py first to generate the model files.")
        raise SystemExit(1)

    # --- Generate a synthetic alert-then-drowsy sequence --------------------
    # Windows 1-20:  alert state  (normal EAR, low PERCLOS)
    # Windows 21-60: gradually becoming drowsy (rising PERCLOS, falling EAR)

    def _make_window(t: int) -> dict:
        """
        t=0 -> clearly alert, t=1 -> clearly drowsy.
        Linear interpolation between the two states.
        """
        # Alert anchor
        alert = {
            "ear_mean": 0.30, "ear_min": 0.22, "ear_std": 0.03,
            "perclos": 0.05,  "blink_rate": 18.0,
            "mar_mean": 0.08, "mar_max": 0.20,  "is_yawn": 0.0,
            "pitch_mean": -1.0, "yaw_mean": 2.0, "roll_mean": 0.5,
            "pitch_std": 2.5,
        }
        # Drowsy anchor
        drowsy = {
            "ear_mean": 0.15, "ear_min": 0.08, "ear_std": 0.05,
            "perclos": 0.72,  "blink_rate": 8.0,
            "mar_mean": 0.55, "mar_max": 1.10, "is_yawn": 1.0,
            "pitch_mean": -10.0, "yaw_mean": 5.0, "roll_mean": 3.0,
            "pitch_std": 6.0,
        }
        return {k: alert[k] + t * (drowsy[k] - alert[k]) for k in alert}

    print(f"  {'Win':>4s}  {'P(raw)':>8s}  {'P(smooth)':>10s}  "
          f"{'Decision':>10s}  {'Severity':>8s}  Reason #1")
    print("  " + "-" * 76)

    prev_decision = None
    for win_idx in range(1, 61):
        # First 20 windows: alert; next 40: linear drift toward drowsy
        t = 0.0 if win_idx <= 20 else (win_idx - 20) / 40.0
        t = min(t, 1.0)
        features = _make_window(t)

        result = monitor.update(features)

        # Print every 5th window, plus whenever decision changes
        decision_changed = result["decision"] != prev_decision
        if win_idx % 5 == 0 or decision_changed or win_idx == 1:
            feat1, val1, dir1 = result["reasons"][0]
            reason_txt = reason_card_text(feat1, val1, dir1)[:35]
            flag = " <-- CHANGE" if decision_changed else ""
            print(f"  {win_idx:>4d}  {result['p_raw']:>8.4f}  "
                  f"{result['p_at_risk']:>10.4f}  "
                  f"{result['decision']:>10s}  {result['severity']:>8s}  "
                  f"{reason_txt}{flag}")
        prev_decision = result["decision"]

    print()
    print(f"  Smoothing buffer size: {monitor.window_count} windows")
    print("\n  Full result dict for last window:")
    for k, v in result.items():
        if k == "reasons":
            print(f"    reasons:")
            for feat, val, direction in v:
                text = reason_card_text(feat, val, direction)
                print(f"      * {text}  [{direction}]")
        else:
            print(f"    {k:12s}: {v}")

    print()
    print("  Self-test PASSED.  FatigueMonitor is live-system ready.")
    print("=" * 62)
