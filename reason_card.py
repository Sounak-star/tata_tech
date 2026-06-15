"""
reason_card.py  -  SAARTHI Reason Card Text Module
====================================================
Standalone UI helper that turns SHAP feature attributions into
clean, display-ready English sentences.

DESIGN RULE: The SHAP explainer lives in FatigueMonitor._explainer.
             This module REUSES it — it never creates a second explainer.
             Call init(monitor) once after creating your FatigueMonitor.

Quick-start
-----------
    from fatigue_monitor  import calibrate, FatigueMonitor
    from reason_card      import init, get_alert_reasons

    baseline = calibrate(alert_rows)
    monitor  = FatigueMonitor(baseline)
    init(monitor)                              # wire to existing explainer

    # In the live loop:
    reasons = get_alert_reasons(window_features, baseline)
    # -> ["Eyes closed 68% of the time",
    #     "Eyes barely blinking (9/min)",
    #     "Yawning detected"]

Standalone (no live monitor):
    python reason_card.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reuse constants and types from fatigue_monitor — no reimporting xgb/shap
from fatigue_monitor import (
    FatigueMonitor,
    EPSILON,
)


# ============================================================
# 1.  MODULE STATE  -  single shared reference to the explainer
# ============================================================

_monitor: FatigueMonitor | None = None   # set by init()


def init(monitor: FatigueMonitor | None = None) -> None:
    """
    Wire reason_card to an existing FatigueMonitor's SHAP explainer.

    If monitor is provided: zero-cost — no new explainer is built.
    If monitor is None: a temporary FatigueMonitor is built from disk
    (only use this for standalone testing; the live system should always
    pass its existing monitor).

    Parameters
    ----------
    monitor : FatigueMonitor | None
        The already-initialised FatigueMonitor from your live loop.

    Example
    -------
        monitor = FatigueMonitor(baseline)
        init(monitor)          # done — explainer is now shared
    """
    global _monitor
    if monitor is not None:
        _monitor = monitor
    else:
        # Standalone path: load from disk with a neutral dummy baseline.
        # The baseline is irrelevant for SHAP — it only affects normalization.
        import json
        from fatigue_monitor import DEFAULT_FEATURES_PATH
        with open(DEFAULT_FEATURES_PATH) as f:
            features = json.load(f)
        dummy_baseline = {f: {"mean": 0.0, "std": 1.0} for f in features}
        _monitor = FatigueMonitor(dummy_baseline)


def _require_init() -> FatigueMonitor:
    """Return the shared monitor, auto-initialising from disk if needed."""
    global _monitor
    if _monitor is None:
        init()   # lazy standalone init
    return _monitor  # type: ignore[return-value]


# ============================================================
# 2.  PHRASING MAP  (feature, direction) -> callable(raw_value) -> str
#
#   direction = "high"  means SHAP value > 0, i.e. this feature is
#               PUSHING P(at-risk) UP for this window.
#   direction = "low"   means SHAP value < 0 — feature is suppressing
#               the at-risk signal (less useful for warning cards but
#               included for completeness).
#
#   The raw value inserted into each template is the ORIGINAL unscaled
#   feature value (e.g. 0.71 for PERCLOS, not its z-score).
# ============================================================

_PHRASING: dict[tuple[str, str], object] = {
    # ── Eye-closure metrics ──────────────────────────────────────────────
    ("perclos",  "high"): lambda v: f"Eyes closed {v:.0%} of the time",
    ("perclos",  "low"):  lambda v: f"Eyes staying open ({v:.0%} closure)",

    ("ear_mean", "high"): lambda v: f"Eyes drooping (avg openness {v:.2f})",
    ("ear_mean", "low"):  lambda v: f"Eye openness elevated ({v:.2f})",

    ("ear_min",  "high"): lambda v: "Prolonged eye closure detected",
    ("ear_min",  "low"):  lambda v: f"Eyes not fully closing ({v:.2f} min)",

    ("ear_std",  "high"): lambda v: "Erratic eye movement",
    ("ear_std",  "low"):  lambda v: "Eyes locked open — barely blinking",

    # ── Blink rate ───────────────────────────────────────────────────────
    ("blink_rate", "high"): lambda v: f"Erratic blinking ({v:.0f}/min)",
    ("blink_rate", "low"):  lambda v: f"Eyes barely blinking ({v:.0f}/min)",

    # ── Yawn / mouth metrics ─────────────────────────────────────────────
    ("is_yawn",  "high"): lambda v: "Yawning detected",
    ("is_yawn",  "low"):  lambda v: "No yawning activity",

    ("mar_mean", "high"): lambda v: f"Mouth repeatedly open (avg {v:.2f})",
    ("mar_mean", "low"):  lambda v: f"Mouth closed ({v:.2f} avg)",

    ("mar_max",  "high"): lambda v: "Wide yawn detected",
    ("mar_max",  "low"):  lambda v: f"Moderate mouth opening (peak {v:.2f})",

    # ── Head pose ────────────────────────────────────────────────────────
    # pitch: positive = head forward (nodding), negative = head back
    ("pitch_mean", "high"): lambda v: f"Head nodding forward ({v:+.1f}°)",
    ("pitch_mean", "low"):  lambda v: f"Head tilting back ({v:+.1f}°)",

    # yaw: positive or negative = turned away from road
    ("yaw_mean", "high"):   lambda v: f"Head turned away from road ({v:+.1f}°)",
    ("yaw_mean", "low"):    lambda v: f"Head turned away from road ({v:+.1f}°)",

    # roll: head leaning to one side (microsleep indicator)
    ("roll_mean", "high"):  lambda v: f"Head tilting to the side ({v:+.1f}°)",
    ("roll_mean", "low"):   lambda v: f"Head tilting to the side ({v:+.1f}°)",

    # pitch variability: high = unsteady, low = unnaturally still (microsleep)
    ("pitch_std", "high"):  lambda v: f"Unsteady head movement ({v:.1f}° variation)",
    ("pitch_std", "low"):   lambda v: f"Head unnaturally still ({v:.1f}° variation)",
}


def _phrase(feature: str, raw_value: float, direction: str) -> str:
    """
    Look up a phrasing template and render it with the raw value.
    Falls back gracefully for any (feature, direction) not in the map.
    """
    fn = _PHRASING.get((feature, direction))
    if fn is not None:
        return fn(raw_value)
    # Sensible catch-all  (should never fire for the 12 trained features)
    return f"{feature.replace('_', ' ').title()} is {direction} ({raw_value:.3f})"


# ============================================================
# 3.  SHAP EXTRACTION  (internal, reuses monitor's explainer)
# ============================================================

def _extract_shap_reasons(
    feature_row,
    baseline: dict,
    top_n: int,
) -> list[tuple[str, float, str, str]]:
    """
    Normalise one window, run SHAP via the shared explainer, and return
    the top_n features ranked by |SHAP contribution|.

    Returns list of (feature, raw_value, direction, sentence) tuples.
    raw_value is the ORIGINAL unscaled value (not z-score).
    """
    m = _require_init()

    # Parse raw feature values
    if isinstance(feature_row, dict):
        raw = np.array([feature_row[f] for f in m.features], dtype=float)
    elif isinstance(feature_row, pd.Series):
        raw = feature_row[m.features].values.astype(float)
    else:
        raw = np.asarray(feature_row, dtype=float)

    # Normalise using the caller's per-subject baseline
    norm = np.array([
        (raw[i] - baseline[feat]["mean"]) / (baseline[feat]["std"] + EPSILON)
        for i, feat in enumerate(m.features)
    ], dtype=float)

    # Run SHAP through the shared explainer (no new explainer created)
    raw_sv = m._explainer.shap_values(norm.reshape(1, -1))

    # Handle shap API variants for binary:logistic
    if isinstance(raw_sv, list):
        sv = raw_sv[1][0] if (hasattr(raw_sv[1], "ndim") and
                               raw_sv[1].ndim == 2) else raw_sv[1]
    elif isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
        sv = raw_sv[0, :, 1]
    else:
        sv = raw_sv[0]   # 2-D (n_samples, n_features) for binary log-odds

    # Rank by absolute SHAP contribution
    ranked = np.argsort(np.abs(sv))[::-1][:top_n]

    results = []
    for i in ranked:
        feat      = m.features[i]
        raw_val   = float(raw[i])
        direction = "high" if sv[i] > 0 else "low"
        sentence  = _phrase(feat, raw_val, direction)
        results.append((feat, raw_val, direction, sentence))

    return results


# ============================================================
# 4.  PUBLIC API
# ============================================================

def get_alert_reason(feature_row, baseline: dict) -> str:
    """
    Return ONE display-ready sentence for the single strongest SHAP driver
    pushing this window toward AT-RISK.

    The raw (un-normalized) feature value is embedded directly in the
    sentence so the UI shows real numbers, not z-scores.

    Parameters
    ----------
    feature_row : dict | pd.Series | array-like
        Raw (UN-scaled) feature values for the 12 model features.
    baseline : dict
        Per-subject calibration baseline from calibrate().

    Returns
    -------
    str
        e.g. "Eyes closed 75% of the time"
             "Yawning detected"
             "Head nodding forward (-9.5°)"

    Example
    -------
        reason = get_alert_reason(window_features, baseline)
        ui.show_banner(reason)
    """
    return _extract_shap_reasons(feature_row, baseline, top_n=1)[0][3]


def get_alert_reasons(
    feature_row,
    baseline: dict,
    top_n: int = 3,
) -> list[str]:
    """
    Return the top N display-ready reason sentences, ranked by SHAP importance.
    Use for a richer Reason Card showing multiple contributing factors.

    Parameters
    ----------
    feature_row : dict | pd.Series | array-like
        Raw (UN-scaled) feature values for the 12 model features.
    baseline : dict
        Per-subject calibration baseline from calibrate().
    top_n : int
        Number of reasons to return (default 3).

    Returns
    -------
    list[str]
        e.g. ["Eyes closed 68% of the time",
               "Eyes barely blinking (9/min)",
               "Yawning detected"]

    Example
    -------
        reasons = get_alert_reasons(window_features, baseline, top_n=3)
        for i, text in enumerate(reasons, 1):
            ui.show_reason_card(i, text)
    """
    return [row[3] for row in _extract_shap_reasons(feature_row, baseline, top_n)]


def get_alert_reasons_rich(
    feature_row,
    baseline: dict,
    top_n: int = 3,
) -> list[dict]:
    """
    Same as get_alert_reasons but returns structured dicts instead of strings,
    so the UI can style each card differently (e.g. colour-code by severity).

    Returns list of:
        {
          "feature":   str    (e.g. "perclos")
          "raw_value": float  (e.g. 0.71)
          "direction": str    ("high" or "low")
          "text":      str    (display sentence)
          "weight":    float  (|SHAP| value, for sorting/bar-charts)
        }
    """
    m   = _require_init()
    raw_results = _extract_shap_reasons(feature_row, baseline, top_n)

    # Re-run to get the raw SHAP magnitudes (they aren't stored in the tuple)
    if isinstance(feature_row, dict):
        raw_arr = np.array([feature_row[f] for f in m.features], dtype=float)
    elif isinstance(feature_row, pd.Series):
        raw_arr = feature_row[m.features].values.astype(float)
    else:
        raw_arr = np.asarray(feature_row, dtype=float)

    norm = np.array([
        (raw_arr[i] - baseline[feat]["mean"]) / (baseline[feat]["std"] + EPSILON)
        for i, feat in enumerate(m.features)
    ], dtype=float)

    raw_sv = m._explainer.shap_values(norm.reshape(1, -1))
    if isinstance(raw_sv, list):
        sv = raw_sv[1][0] if (hasattr(raw_sv[1], "ndim") and
                               raw_sv[1].ndim == 2) else raw_sv[1]
    elif isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
        sv = raw_sv[0, :, 1]
    else:
        sv = raw_sv[0]

    feat_to_shap = {m.features[i]: float(abs(sv[i])) for i in range(len(m.features))}

    return [
        {
            "feature":   feat,
            "raw_value": raw_val,
            "direction": direction,
            "text":      sentence,
            "weight":    round(feat_to_shap.get(feat, 0.0), 6),
        }
        for feat, raw_val, direction, sentence in raw_results
    ]


# ============================================================
# 5.  DEMO  (python reason_card.py)
# ============================================================

if __name__ == "__main__":

    print("=" * 62)
    print(" reason_card.py  -  standalone demo")
    print("=" * 62)

    # Calibration baseline (typical alert-state for a sample driver)
    BASELINE = {
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

    # Three sample windows to exercise the phrasing map
    SAMPLES = [
        {
            "label": "Clearly drowsy (high PERCLOS + yawn)",
            "row": {
                "ear_mean": 0.17, "ear_min": 0.10, "ear_std": 0.04,
                "perclos": 0.71,  "blink_rate": 9.0,
                "mar_mean": 0.60, "mar_max": 1.10, "is_yawn": 1.0,
                "pitch_mean": -9.0, "yaw_mean": 3.0,
                "roll_mean": 2.5,  "pitch_std": 5.5,
            },
        },
        {
            "label": "Head nodding + barely blinking",
            "row": {
                "ear_mean": 0.24, "ear_min": 0.18, "ear_std": 0.02,
                "perclos": 0.12,  "blink_rate": 6.0,
                "mar_mean": 0.08, "mar_max": 0.18, "is_yawn": 0.0,
                "pitch_mean": -14.0, "yaw_mean": 1.0,
                "roll_mean": 0.5,   "pitch_std": 7.0,
            },
        },
        {
            "label": "Alert (control — should show low-risk drivers)",
            "row": {
                "ear_mean": 0.30, "ear_min": 0.22, "ear_std": 0.03,
                "perclos": 0.04,  "blink_rate": 19.0,
                "mar_mean": 0.08, "mar_max": 0.20, "is_yawn": 0.0,
                "pitch_mean": -1.5, "yaw_mean": 2.0,
                "roll_mean": 0.5,  "pitch_std": 2.5,
            },
        },
    ]

    # Auto-init from disk (no existing FatigueMonitor in this standalone run)
    print("\n[init] Auto-loading model + SHAP explainer from disk ...")
    try:
        init()
        print("[init] Ready.\n")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        print("[error] Run train_xgboost.py first to generate the model files.")
        raise SystemExit(1)

    for sample in SAMPLES:
        print(f"  Sample: {sample['label']}")
        print(f"  {'-' * 55}")

        # Single reason
        top1 = get_alert_reason(sample["row"], BASELINE)
        print(f"  get_alert_reason()        -> \"{top1}\"")

        # Top 3 reasons
        top3 = get_alert_reasons(sample["row"], BASELINE, top_n=3)
        print(f"  get_alert_reasons(top_n=3):")
        for i, text in enumerate(top3, 1):
            print(f"    {i}. {text}")

        # Rich dict form
        rich = get_alert_reasons_rich(sample["row"], BASELINE, top_n=2)
        print(f"  get_alert_reasons_rich(top_n=2):")
        for card in rich:
            print(f"    [{card['direction']:4s}] {card['text']:<45s}  "
                  f"weight={card['weight']:.4f}")

        print()

    print("  Demo complete — reason_card.py is UI-ready.")
    print("=" * 62)
