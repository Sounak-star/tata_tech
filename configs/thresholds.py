"""
configs/thresholds.py

Defines baseline thresholds and adaptation parameters for the operator state monitoring.
All values are designed for deterministic fallback policy updates.
"""

# Eye Aspect Ratio (EAR) Threshold Config
# A lower EAR indicates eyes closing (fatigue/drowsiness).
# Increasing this threshold makes the system more sensitive (alerting earlier).
EAR_BASELINE = 0.25
EAR_MIN_LIMIT = 0.20
EAR_MAX_LIMIT = 0.35
EAR_ADAPTATION_STEP = 0.02

# Percentage of Eye Closure (PERCLOS) Threshold Config
# A higher PERCLOS indicates fatigue.
# Lowering this threshold makes the system more sensitive (alerting at lower drowsiness levels).
PERCLOS_BASELINE = 0.15
PERCLOS_MIN_LIMIT = 0.05
PERCLOS_MAX_LIMIT = 0.30
PERCLOS_ADAPTATION_STEP = 0.02

# Proximity Threshold Config (in meters)
# Distance to nearest obstacle.
# Increasing this threshold makes the system alert when obstacles are further away (more sensitive).
PROXIMITY_BASELINE = 10.0  # meters
PROXIMITY_MIN_LIMIT = 5.0
PROXIMITY_MAX_LIMIT = 20.0
PROXIMITY_ADAPTATION_STEP = 2.0
