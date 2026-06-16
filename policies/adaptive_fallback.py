"""
policies/adaptive_fallback.py

Implements deterministic adaptive threshold fallback logic.
If operator ignores warnings, thresholds adjust to become more sensitive.
If operator responds safely, thresholds gradually decay back to baseline.
"""

from configs.thresholds import (
    EAR_BASELINE, EAR_MIN_LIMIT, EAR_MAX_LIMIT, EAR_ADAPTATION_STEP,
    PERCLOS_BASELINE, PERCLOS_MIN_LIMIT, PERCLOS_MAX_LIMIT, PERCLOS_ADAPTATION_STEP,
    PROXIMITY_BASELINE, PROXIMITY_MIN_LIMIT, PROXIMITY_MAX_LIMIT, PROXIMITY_ADAPTATION_STEP
)
from env.action_space import InterventionAction


class AdaptiveFallbackPolicy:
    """
    Manages operator alertness and safety thresholds.
    Adjusts thresholds based on whether safety intervention actions are ignored or heeded.
    """
    
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets all thresholds to baseline values."""
        self.ear_threshold: float = EAR_BASELINE
        self.perclos_threshold: float = PERCLOS_BASELINE
        self.proximity_threshold: float = PROXIMITY_BASELINE
        self.consecutive_ignored: int = 0

    def adjust_thresholds(self, action: InterventionAction, responded_safely: bool) -> None:
        """
        Adjusts thresholds dynamically based on intervention outcomes.
        
        Args:
            action (InterventionAction): The action taken in the previous step.
            responded_safely (bool): Whether the operator responded appropriately/safely.
        """
        # We only adapt thresholds if an alert was triggered
        if action != InterventionAction.NO_ACTION:
            if not responded_safely:
                # Alert ignored: Increase sensitivity
                self.consecutive_ignored += 1
                
                # Make EAR more sensitive (raise threshold to trigger when eyes are more open)
                self.ear_threshold = min(
                    EAR_MAX_LIMIT, 
                    self.ear_threshold + EAR_ADAPTATION_STEP * self.consecutive_ignored
                )
                
                # Make PERCLOS more sensitive (lower threshold to trigger at lower closure percent)
                self.perclos_threshold = max(
                    PERCLOS_MIN_LIMIT, 
                    self.perclos_threshold - PERCLOS_ADAPTATION_STEP * self.consecutive_ignored
                )
                
                # Make Proximity more sensitive (raise threshold to trigger at larger distance)
                self.proximity_threshold = min(
                    PROXIMITY_MAX_LIMIT, 
                    self.proximity_threshold + PROXIMITY_ADAPTATION_STEP * self.consecutive_ignored
                )
            else:
                # Alert heeded: Gradually return towards baseline, reset ignored count
                self.consecutive_ignored = 0
                
                # Decay EAR towards baseline
                if self.ear_threshold > EAR_BASELINE:
                    self.ear_threshold = max(EAR_BASELINE, self.ear_threshold - EAR_ADAPTATION_STEP)
                elif self.ear_threshold < EAR_BASELINE:
                    self.ear_threshold = min(EAR_BASELINE, self.ear_threshold + EAR_ADAPTATION_STEP)
                    
                # Decay PERCLOS towards baseline
                if self.perclos_threshold < PERCLOS_BASELINE:
                    self.perclos_threshold = min(PERCLOS_BASELINE, self.perclos_threshold + PERCLOS_ADAPTATION_STEP)
                elif self.perclos_threshold > PERCLOS_BASELINE:
                    self.perclos_threshold = max(PERCLOS_BASELINE, self.perclos_threshold - PERCLOS_ADAPTATION_STEP)
                    
                # Decay Proximity towards baseline
                if self.proximity_threshold > PROXIMITY_BASELINE:
                    self.proximity_threshold = max(PROXIMITY_BASELINE, self.proximity_threshold - PROXIMITY_ADAPTATION_STEP)
                elif self.proximity_threshold < PROXIMITY_BASELINE:
                    self.proximity_threshold = min(PROXIMITY_BASELINE, self.proximity_threshold + PROXIMITY_ADAPTATION_STEP)
        else:
            # If no action, slowly decay thresholds back to baseline if they deviated
            self.consecutive_ignored = max(0, self.consecutive_ignored - 1)
            
            # Decay EAR
            if self.ear_threshold > EAR_BASELINE:
                self.ear_threshold = max(EAR_BASELINE, self.ear_threshold - EAR_ADAPTATION_STEP)
            elif self.ear_threshold < EAR_BASELINE:
                self.ear_threshold = min(EAR_BASELINE, self.ear_threshold + EAR_ADAPTATION_STEP)
                
            # Decay PERCLOS
            if self.perclos_threshold < PERCLOS_BASELINE:
                self.perclos_threshold = min(PERCLOS_BASELINE, self.perclos_threshold + PERCLOS_ADAPTATION_STEP)
            elif self.perclos_threshold > PERCLOS_BASELINE:
                self.perclos_threshold = max(PERCLOS_BASELINE, self.perclos_threshold - PERCLOS_ADAPTATION_STEP)
                
            # Decay Proximity
            if self.proximity_threshold > PROXIMITY_BASELINE:
                self.proximity_threshold = max(PROXIMITY_BASELINE, self.proximity_threshold - PROXIMITY_ADAPTATION_STEP)
            elif self.proximity_threshold < PROXIMITY_BASELINE:
                self.proximity_threshold = min(PROXIMITY_BASELINE, self.proximity_threshold + PROXIMITY_ADAPTATION_STEP)

    def get_thresholds(self) -> dict[str, float]:
        """Returns the current active threshold values."""
        return {
            "ear_threshold": self.ear_threshold,
            "perclos_threshold": self.perclos_threshold,
            "proximity_threshold": self.proximity_threshold,
            "consecutive_ignored": self.consecutive_ignored
        }
