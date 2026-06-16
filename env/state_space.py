"""
env/state_space.py

Defines the state space representation of the operator and environment.
Uses python dataclass for structured representation and verification.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class OperatorState:
    """
    State space representation for the Edge AI Safety Copilot.
    
    Attributes:
        fatigue_score (float): Drowsiness/fatigue measure (0.0 = fully alert, 1.0 = sleeping/unresponsive).
        attention_score (float): Focus level (0.0 = distracted/gazing away, 1.0 = fully focused on operations).
        hazard_risk (float): Likelihood of an incident based on environment conditions (0.0 = safe, 1.0 = extreme hazard).
        obstacle_distance (float): Proximity to the nearest obstacle/worker in meters (0.0 to 100.0).
        operator_response_rate (float): Probability or speed score of operator reaction (0.0 = ignores warnings, 1.0 = instant correction).
    """
    fatigue_score: float
    attention_score: float
    hazard_risk: float
    obstacle_distance: float
    operator_response_rate: float

    def __post_init__(self) -> None:
        """Enforces limits on state values."""
        self.fatigue_score = max(0.0, min(1.0, float(self.fatigue_score)))
        self.attention_score = max(0.0, min(1.0, float(self.attention_score)))
        self.hazard_risk = max(0.0, min(1.0, float(self.hazard_risk)))
        self.obstacle_distance = max(0.0, float(self.obstacle_distance))
        self.operator_response_rate = max(0.0, min(1.0, float(self.operator_response_rate)))

    def to_dict(self) -> dict[str, float]:
        """Converts state space to a standard dictionary representation."""
        return asdict(self)

    def to_vector(self) -> list[float]:
        """
        Converts the state to a list (vector) for easy feeding to reinforcement learning models.
        Values are normalized between [0, 1] except obstacle distance, which is normalized relative to 50m.
        """
        norm_dist = min(1.0, self.obstacle_distance / 50.0)
        return [
            self.fatigue_score,
            self.attention_score,
            self.hazard_risk,
            norm_dist,
            self.operator_response_rate,
        ]

    def to_numpy(self) -> np.ndarray:
        """Returns the observation vector as a numpy float32 array for RL framework compatibility."""
        return np.array(self.to_vector(), dtype=np.float32)

    def __str__(self) -> str:
        return (
            f"OperatorState(Fatigue: {self.fatigue_score:.2f}, "
            f"Attention: {self.attention_score:.2f}, "
            f"Hazard Risk: {self.hazard_risk:.2f}, "
            f"Obstacle Dist: {self.obstacle_distance:.1f}m, "
            f"Response Rate: {self.operator_response_rate:.2f})"
        )
