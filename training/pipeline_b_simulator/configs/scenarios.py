"""
configs/scenarios.py

Defines range configurations for various construction site scenarios.
Each scenario specifies min/max values for generating the OperatorState parameters.
"""

SCENARIOS = {
    "fatigued_operator": {
        "fatigue_score": (0.70, 0.95),
        "attention_score": (0.30, 0.60),
        "hazard_risk": (0.40, 0.75),
        "obstacle_distance": (12.0, 25.0),
        "operator_response_rate": (0.15, 0.40),  # Slow/unresponsive
    },
    "distracted_operator": {
        "fatigue_score": (0.10, 0.35),
        "attention_score": (0.10, 0.40),  # Low attention
        "hazard_risk": (0.50, 0.80),
        "obstacle_distance": (8.0, 20.0),
        "operator_response_rate": (0.30, 0.60),
    },
    "blind_spot_worker": {
        "fatigue_score": (0.10, 0.30),
        "attention_score": (0.75, 0.95),  # Operator is alert
        "hazard_risk": (0.70, 0.95),      # High hazard due to blind spot proximity
        "obstacle_distance": (3.0, 8.0),   # Critical distance
        "operator_response_rate": (0.70, 0.90),
    },
    "poor_visibility": {
        "fatigue_score": (0.20, 0.45),
        "attention_score": (0.60, 0.85),
        "hazard_risk": (0.60, 0.90),      # Elevated risk due to fog/dust
        "obstacle_distance": (10.0, 18.0),
        "operator_response_rate": (0.40, 0.70),  # Response delayed by visibility limits
    },
    "heavy_load_slope_risk": {
        "fatigue_score": (0.10, 0.30),
        "attention_score": (0.80, 1.00),  # High operator alertness
        "hazard_risk": (0.80, 0.98),      # High load & slope hazard
        "obstacle_distance": (15.0, 30.0),
        "operator_response_rate": (0.85, 1.00),  # Very responsive, but operating near physical limits
    }
}
