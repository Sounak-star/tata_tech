"""
demo.py

Demonstration script for the SAARTHI reinforcement learning decision framework.
Runs simulations for all construction-site scenarios, evaluates rule-based
interventions, applies adaptive thresholds, and shapes rewards.
"""

import sys
import time
from typing import List

# Ensure current folder is in path for imports
sys.path.append(".")

from env.smart_cabin_env import SmartCabinEnv
from env.action_space import InterventionAction
from policies.intervention_logic import RuleBasedInterventionLogic


def run_scenario_simulation(scenario_name: str, env: SmartCabinEnv, policy: RuleBasedInterventionLogic) -> None:
    """Runs a single scenario simulation and outputs structured telemetry."""
    print("=" * 90)
    print(f" SIMULATING SCENARIO: {scenario_name.upper().replace('_', ' ')} ")
    print("=" * 90)

    # Reset environment and get initial state
    state = env.reset(scenario_name)
    done = False
    step_num = 0
    total_reward = 0.0

    # Print header for step logs
    print(
        f"{'Step':<5} | {'State Telemetry':<75} | {'Action':<15} | {'Reward':<7} | {'Outcome Description'}"
    )
    print("-" * 140)

    while not done:
        step_num += 1
        
        # Get active thresholds before action selection
        thresholds = env.fallback_policy.get_thresholds()
        
        # Choose action based on current state and thresholds
        action = policy.choose_action(state, env.fallback_policy)
        action_name = action.name
        
        # Format active thresholds for visualization
        thresh_str = (
            f" [Thresh: EAR={thresholds['ear_threshold']:.2f}, "
            f"PERCLOS={thresholds['perclos_threshold']:.2f}, "
            f"PROX={thresholds['proximity_threshold']:.1f}m, "
            f"IgnoredCount={thresholds['consecutive_ignored']}]"
        )
        
        # Format current state values
        state_str = (
            f"F:{state.fatigue_score:.2f} A:{state.attention_score:.2f} "
            f"H:{state.hazard_risk:.2f} D:{state.obstacle_distance:.1f}m "
            f"R:{state.operator_response_rate:.2f}"
        )
        
        telemetry = f"{state_str}{thresh_str}"

        # Take a step in the environment
        next_state, reward, done, outcome = env.step(action)
        total_reward += reward

        # Construct outcome message
        outcome_msgs = []
        if outcome["collision"]:
            outcome_msgs.append("[COLLISION] CRITICAL COLLISION (Emergency)")
        elif outcome["near_miss"]:
            outcome_msgs.append("[NEAR MISS] NEAR MISS DETECTED")
            
        if action != InterventionAction.NO_ACTION:
            if outcome["false_warning"]:
                outcome_msgs.append("[FALSE WARN] False Warning Issued")
            elif outcome["responded_safely"]:
                outcome_msgs.append("[SUCCESS] Operator Responded Safely")
            else:
                outcome_msgs.append("[IGNORED] Alert Ignored by Operator")
        else:
            if outcome["hazard_present"]:
                outcome_msgs.append("[HAZARD] Hazard escalating (No Alert)")
            else:
                outcome_msgs.append("[SAFE] Safe operation")

        outcome_desc = ", ".join(outcome_msgs) if outcome_msgs else "[SAFE] Idle/Normal"

        # Print step row
        print(
            f"{step_num:<5} | {telemetry:<75} | {action_name:<15} | {reward:<7.1f} | {outcome_desc}"
        )
        
        # Update state for next iteration
        state = next_state

    print("-" * 140)
    print(f"Scenario Finished in {step_num} steps. Total Episode Reward: {total_reward:.1f}\n")


def main() -> None:
    print("\n" + "#" * 90)
    print(" SAARTHI - EDGE AI SAFETY COPILOT FOUNDATIONAL FRAMEWORK ")
    print("#" * 90 + "\n")

    # Set seed for reproducible results
    env = SmartCabinEnv(seed=42)
    policy = RuleBasedInterventionLogic()

    scenarios = [
        "fatigued_operator",
        "distracted_operator",
        "blind_spot_worker",
        "poor_visibility",
        "heavy_load_slope_risk"
    ]

    for scenario in scenarios:
        run_scenario_simulation(scenario, env, policy)


if __name__ == "__main__":
    main()
