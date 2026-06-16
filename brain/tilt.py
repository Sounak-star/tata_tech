"""
brain/tilt.py — Q4: "Is the machine itself at risk?" (one clean telemetry rule).

Tilt beyond a safe angle while moving → slope warning; beyond a critical angle
→ Level-3 territory (the hard-rule decision is made in engine.py; this module
just classifies the tilt signal).
"""

from __future__ import annotations

TILT_WARN = 15.0       # degrees: slope caution
TILT_CRITICAL = 25.0   # degrees: tip-over risk
SPEED_MOVING = 0.3     # m/s


def assess_tilt(tilt_angle: float, machine_speed: float) -> dict:
    moving = machine_speed > SPEED_MOVING
    if tilt_angle >= TILT_CRITICAL and moving:
        status = "critical"
    elif tilt_angle >= TILT_WARN and moving:
        status = "warning"
    else:
        status = "ok"
    return {"status": status, "tilt_angle": round(float(tilt_angle), 1), "moving": moving}
