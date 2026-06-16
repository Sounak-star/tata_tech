"""
env/scenario_generator.py

Generates OperatorStates representing realistic construction-site hazards and operator behaviors.
"""

import random
from typing import Optional

from configs.scenarios import SCENARIOS
from env.state_space import OperatorState


class ScenarioGenerator:
    """
    Simulates construction-site hazard environments.
    Generates OperatorState configurations according to specific scenarios.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

    def generate_state(self, scenario_type: Optional[str] = None) -> OperatorState:
        """
        Generates an OperatorState representing a scenario.
        
        Args:
            scenario_type (str, optional): One of the predefined scenario keys from configs/scenarios.
                                          If None, a random scenario is selected.
                                          
        Returns:
            OperatorState: A randomized state conforming to the scenario's boundaries.
        """
        available_scenarios = list(SCENARIOS.keys())
        
        if scenario_type is None:
            scenario_type = random.choice(available_scenarios)
            
        if scenario_type not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario type '{scenario_type}'. Available scenarios: {available_scenarios}"
            )
            
        config = SCENARIOS[scenario_type]
        
        # Generate random state values within configured bounds
        fatigue = random.uniform(*config["fatigue_score"])
        attention = random.uniform(*config["attention_score"])
        hazard = random.uniform(*config["hazard_risk"])
        distance = random.uniform(*config["obstacle_distance"])
        response_rate = random.uniform(*config["operator_response_rate"])
        
        return OperatorState(
            fatigue_score=fatigue,
            attention_score=attention,
            hazard_risk=hazard,
            obstacle_distance=distance,
            operator_response_rate=response_rate
        )
