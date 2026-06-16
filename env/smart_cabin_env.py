"""
env/smart_cabin_env.py

Implements a lightweight reinforcement learning environment for SAARTHI.
Simulates state transitions, operator response, hazard developments, and reward collection.
Does not require Gymnasium dependency yet.
"""

import random
from typing import Optional, Tuple

from env.state_space import OperatorState
from env.action_space import InterventionAction
from env.scenario_generator import ScenarioGenerator
from policies.adaptive_fallback import AdaptiveFallbackPolicy
from rewards.reward_shaping import calculate_reward


class SmartCabinEnv:
    """
    Lightweight simulation environment for the Edge AI Safety Copilot.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.scenario_generator = ScenarioGenerator(seed=seed)
        self.fallback_policy = AdaptiveFallbackPolicy()
        
        self.current_state: Optional[OperatorState] = None
        self.steps_taken: int = 0
        self.max_steps: int = 15
        
        # Set seed for random choices in transition logic
        if seed is not None:
            random.seed(seed)

    def reset(self, scenario_type: Optional[str] = None) -> OperatorState:
        """
        Resets the environment.
        
        Args:
            scenario_type (str, optional): The construction hazard scenario.
            
        Returns:
            OperatorState: The initial state of the environment.
        """
        self.current_state = self.scenario_generator.generate_state(scenario_type)
        self.fallback_policy.reset()
        self.steps_taken = 0
        return self.current_state

    def get_state(self) -> OperatorState:
        """Returns the current state."""
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before getting the state.")
        return self.current_state

    def _is_hazard_present(self, state: OperatorState) -> bool:
        """Helper to determine if a physical safety hazard is present."""
        return (
            state.obstacle_distance < self.fallback_policy.proximity_threshold
            or state.hazard_risk > 0.50
            or state.fatigue_score > 0.60
            or state.attention_score < 0.50
        )

    def _is_false_warning(self, state: OperatorState) -> bool:
        """Helper to determine if alert triggered was a false alarm."""
        return (
            state.fatigue_score < 0.30
            and state.attention_score > 0.70
            and state.obstacle_distance > 15.0
            and state.hazard_risk < 0.30
        )

    def step(self, action: InterventionAction) -> Tuple[OperatorState, float, bool, dict]:
        """
        Applies an action to the environment, transitioning it to the next state.
        
        Flow: State -> Action -> Environment Update -> Reward -> Next State
        
        Args:
            action (InterventionAction): Action taken by the copilot.
            
        Returns:
            Tuple[OperatorState, float, bool, dict]:
                - next_state (OperatorState): The new state.
                - reward (float): Reward obtained from the step.
                - done (bool): True if episode finished (collision, hazard resolved, max steps).
                - info (dict): Diagnostic dictionary with outcome details.
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before taking a step.")

        state = self.current_state
        self.steps_taken += 1

        # Determine if a hazard is currently present
        hazard_present = self._is_hazard_present(state)
        false_warning = False
        responded_safely = False
        hazard_resolved = False
        collision = False
        near_miss = False

        # Transition Dynamics
        next_fatigue = state.fatigue_score
        next_attention = state.attention_score
        next_hazard = state.hazard_risk
        next_distance = state.obstacle_distance
        next_response_rate = state.operator_response_rate

        # Evaluate action effects
        if action == InterventionAction.NO_ACTION:
            # If no action is taken, thresholds decay back to baseline.
            self.fallback_policy.adjust_thresholds(action, responded_safely=True)
            
            if hazard_present:
                # Hazard escalates without warnings!
                next_distance = max(0.0, next_distance - random.uniform(1.5, 3.5))
                next_hazard = min(1.0, next_hazard + random.uniform(0.05, 0.15))
                # Operator attention decays if they were already distracted
                if next_attention < 0.6:
                    next_attention = max(0.0, next_attention - 0.05)
            else:
                # Normal drift: state remains stable or slowly drifts
                next_distance = max(0.0, next_distance + random.uniform(-1.0, 1.0))
                next_fatigue = min(1.0, next_fatigue + random.uniform(-0.02, 0.02))

        else:
            # An intervention alert was issued (LEVEL_1_NUDGE or LEVEL_2_WARNING)
            false_warning = self._is_false_warning(state)

            # Determine response probability
            response_probability = state.operator_response_rate
            if action == InterventionAction.LEVEL_2_WARNING:
                # Level 2 warning is louder/more intrusive, higher response chance
                response_probability = min(1.0, response_probability * 1.35)

            # Simulate if operator responds
            if random.random() < response_probability:
                responded_safely = True
                
                # Update thresholds (heeded -> decay thresholds to baseline)
                self.fallback_policy.adjust_thresholds(action, responded_safely=True)

                if hazard_present:
                    hazard_resolved = True
                    # Operator corrects: attention goes up, fatigue down, hazard decreases
                    next_attention = min(1.0, next_attention + random.uniform(0.20, 0.40))
                    next_fatigue = max(0.0, next_fatigue - random.uniform(0.05, 0.15))
                    next_hazard = max(0.0, next_hazard - random.uniform(0.20, 0.40))
                    next_distance = next_distance + random.uniform(3.0, 6.0)  # braking/steering away
                else:
                    # Nudge in a safe state: operator is annoyed but remains alert
                    next_attention = min(1.0, next_attention + 0.1)
            else:
                # Operator ignored the warning
                responded_safely = False
                
                # Make adaptive thresholds more sensitive!
                self.fallback_policy.adjust_thresholds(action, responded_safely=False)

                # Hazard escalates faster since warning was ignored
                next_distance = max(0.0, next_distance - random.uniform(2.5, 5.0))
                next_hazard = min(1.0, next_hazard + random.uniform(0.10, 0.25))
                next_attention = max(0.0, next_attention - 0.10)
                next_fatigue = min(1.0, next_fatigue + 0.05)

        # Check critical boundary conditions
        if next_distance <= 1.5 or next_hazard >= 0.97:
            collision = True
        elif next_distance <= 4.5 or next_hazard >= 0.88:
            near_miss = True

        # Assemble the outcome details for reward calculation
        outcome = {
            "collision": collision,
            "near_miss": near_miss,
            "responded_safely": responded_safely,
            "hazard_present": hazard_present,
            "hazard_resolved": hazard_resolved,
            "false_warning": false_warning
        }

        # Update current state
        self.current_state = OperatorState(
            fatigue_score=next_fatigue,
            attention_score=next_attention,
            hazard_risk=next_hazard,
            obstacle_distance=next_distance,
            operator_response_rate=next_response_rate
        )

        # Calculate step reward
        reward = calculate_reward(state, action, outcome)

        # Define ending conditions
        done = False
        if collision:
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True
        # If the hazard is fully resolved and everything is safe, we can end the episode
        elif not hazard_present and next_distance > 15.0 and next_hazard < 0.25 and next_attention > 0.70:
            done = True

        return self.current_state, reward, done, outcome
