"""
brain/risk.py — the transparent live risk score (0–100).

A judge can interrogate every number: the score is a visible weighted formula of
the three live signals, NOT a black box. (This is distinct from the Context Risk
Model, which scores the *situation* from historical OSHA data — see context_risk.)
"""

from __future__ import annotations

# Visible weights — sum to 100 at full danger.
W_FATIGUE = 45.0   # fatigue probability (0..1)
W_ZONE = 35.0      # blind-spot zone (0,1,2 → 0,0.5,1)
W_TILT = 20.0      # tilt toward critical (0..1)

TILT_WARN = 15.0
TILT_CRITICAL = 25.0


def live_risk_score(fatigue_p: float, zone: int, tilt_angle: float) -> dict:
    zone_factor = min(1.0, zone / 2.0)
    tilt_factor = max(0.0, min(1.0, (tilt_angle - TILT_WARN) / (TILT_CRITICAL - TILT_WARN)))

    fatigue_pts = W_FATIGUE * max(0.0, min(1.0, fatigue_p))
    zone_pts = W_ZONE * zone_factor
    tilt_pts = W_TILT * tilt_factor
    score = round(fatigue_pts + zone_pts + tilt_pts, 1)

    return {
        "score": score,
        "breakdown": {
            "fatigue": round(fatigue_pts, 1),
            "zone": round(zone_pts, 1),
            "tilt": round(tilt_pts, 1),
        },
        "formula": f"{W_FATIGUE}*fatigue + {W_ZONE}*zone + {W_TILT}*tilt",
    }
