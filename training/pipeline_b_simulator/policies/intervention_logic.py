"""
policies/intervention_logic.py

Implements a rule-based safety intervention logic.
Acts as a deterministic control loop comparing current state parameters 
against adaptive thresholds to select the appropriate safety copilot action.
This acts as a temporary replacement/baseline for a PPO agent.
"""

from env.state_space import OperatorState
from env.action_space import InterventionAction
from policies.adaptive_fallback import AdaptiveFallbackPolicy  # Wait, wait, let's just accept the fallback policy thresholds.


class RuleBasedInterventionLogic:
    """
    Decides safety actions using deterministic rules.
    Compares operator features to the active adaptive thresholds.
    """

    def choose_action(self, state: OperatorState, fallback_policy: AdaptiveFallbackPolicy) -> InterventionAction:
        """
        Selects an InterventionAction based on the operator state and active thresholds.
        
        Args:
            state (OperatorState): The current state.
            fallback_policy (AdaptiveFallbackPolicy): The policy containing the current thresholds.
            
        Returns:
            InterventionAction: The selected safety action.
        """
        # Map operator fatigue score to simulated sensor readings:
        # High fatigue -> Lower eye aspect ratio (EAR)
        # 0.0 fatigue -> 0.35 EAR (fully alert, open eyes)
        # 1.0 fatigue -> 0.15 EAR (closed eyes)
        simulated_ear = 0.35 - 0.20 * state.fatigue_score

        # High fatigue -> Higher percentage of eye closure (PERCLOS)
        # 0.0 fatigue -> 0.0 PERCLOS
        # 1.0 fatigue -> 0.5 PERCLOS
        simulated_perclos = 0.50 * state.fatigue_score

        # Extract active thresholds
        ear_thresh = fallback_policy.ear_threshold
        perclos_thresh = fallback_policy.perclos_threshold
        proximity_thresh = fallback_policy.proximity_threshold

        # Define check flags
        ear_violation = simulated_ear < ear_thresh
        perclos_violation = simulated_perclos > perclos_thresh
        proximity_violation = state.obstacle_distance < proximity_thresh

        # Evaluate critical (Level 2 Warning) conditions:
        # - Extremely close to obstacle (less than 60% of proximity threshold)
        # - Critical fatigue violation (extremely low EAR or extremely high PERCLOS)
        # - Combined high risk with low attention
        is_critical_obstacle = state.obstacle_distance < (proximity_thresh * 0.6)
        is_critical_fatigue = state.fatigue_score > 0.80 or (simulated_ear < 0.22)
        is_critical_distraction = state.attention_score < 0.25 and state.hazard_risk > 0.70

        if ear_violation or perclos_violation or proximity_violation:
            if is_critical_obstacle or is_critical_fatigue or is_critical_distraction:
                return InterventionAction.LEVEL_2_WARNING
            else:
                return InterventionAction.LEVEL_1_NUDGE

        # Secondary rules (moderate/critical checks even if sensor thresholds are not strictly violated)
        if state.hazard_risk > 0.80:
            return InterventionAction.LEVEL_2_WARNING
        elif state.attention_score < 0.50 or state.hazard_risk > 0.50:
            return InterventionAction.LEVEL_1_NUDGE

        # Default: Safe operation
        return InterventionAction.NO_ACTION
