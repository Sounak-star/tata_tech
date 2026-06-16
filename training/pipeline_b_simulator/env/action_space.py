"""
env/action_space.py

Defines the action space available to the Safety Copilot.
"""

from enum import IntEnum


class InterventionAction(IntEnum):
    """
    Actions that the Edge AI Safety Copilot can take.
    
    NO_ACTION: Do nothing, keep monitoring.
    LEVEL_1_NUDGE: Mild alert (e.g. haptic feedback, gentle audio chime, HUD indicator).
    LEVEL_2_WARNING: Critical warning (e.g. flashing red lights, loud horn, auto-braking preparation).
    """
    NO_ACTION = 0
    LEVEL_1_NUDGE = 1
    LEVEL_2_WARNING = 2

    @classmethod
    def get_name(cls, action: int) -> str:
        try:
            return cls(action).name
        except ValueError:
            return "UNKNOWN"
