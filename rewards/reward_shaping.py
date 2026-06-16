"""
rewards/reward_shaping.py

Implements reward calculation logic for the reinforcement learning environment.
Rewards reinforce safe operator behaviors and penalize copilot errors.
"""

from env.state_space import OperatorState
from env.action_space import InterventionAction


def calculate_reward(state: OperatorState, action: InterventionAction, outcome: dict) -> float:
    """
    Calculates the reward based on the state, intervention action, and simulation outcome.
    
    Reward Rules:
    - Safe operation = +10
    - Hazard resolved = +10
    - False warning = -5
    - Ignored warning = -20
    - Near miss = -100
    - Emergency event = -1000
    
    Args:
        state (OperatorState): The state before the action's full transition effect.
        action (InterventionAction): The action chosen by the copilot policy.
        outcome (dict): A dictionary describing the outcome of the transition.
                        Expected keys: "collision", "near_miss", "responded_safely",
                                      "hazard_present", "hazard_resolved", "false_warning".
                                      
    Returns:
        float: The calculated reward.
    """
    # 1. Emergency Event (Collision)
    if outcome.get("collision", False):
        return -1000.0

    # 2. Near Miss
    if outcome.get("near_miss", False):
        return -100.0

    # 3. Ignored warning
    # Alert was issued, but operator did not take corrective action / ignored it
    if action != InterventionAction.NO_ACTION and not outcome.get("responded_safely", True):
        return -20.0

    # 4. False warning
    # Alert was issued when no hazard was present and operator was fully alert
    if action != InterventionAction.NO_ACTION and outcome.get("false_warning", False):
        return -5.0

    # 5. Hazard resolved
    # Alert was issued, and the operator responded safely, resolving the hazard
    if action != InterventionAction.NO_ACTION and outcome.get("hazard_resolved", False):
        return 10.0

    # 6. Safe operation
    # No action was taken, and everything is running safely (no hazard was present, or no incident occurred)
    if action == InterventionAction.NO_ACTION and not outcome.get("hazard_present", False):
        return 10.0

    # Fallback default: small step penalty/reward to guide exploration
    return 0.0
