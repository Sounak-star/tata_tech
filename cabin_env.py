"""
cabin_env.py  —  Pipeline B / Dev 1 (Simulation & Environment Lead)

Custom Gymnasium environment for SAARTHI's intervention-timing agent.

This is the canonical Pipeline-B simulator. It wires together all three devs:
  • Dev 1 (this file + telemetry_provider + hazard_injector): the Gym
    infrastructure, telemetry replay, hazard injection, and the deterministic
    Tier-1 hard-rule engine that owns Level 3.
  • Dev 2 (rewards/reward_shaping.calculate_reward): the reward function, plugged
    in through the `_calculate_reward()` hook exactly as the brief requires.
  • Dev 3 (track 3 OSHA analysis): real hazard frequencies feed the injector.

Design rule from the plan: the agent's action only dictates Level 0–2 soft
interventions. Level 3 is reserved for the environment-enforced hard rules and
can never be produced or suppressed by the agent.

    action_space      = Discrete(3)   # 0 none · 1 nudge · 2 escalated warning
    observation_space = Box([fatigue_probability, outside_zone_status,
                             machine_speed, tilt_angle, is_reversing])
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cabin_env.py requires gymnasium. Install with: pip install gymnasium"
    ) from exc

from telemetry_provider import TelemetryProvider
from hazard_injector import HazardInjector, DEV3_HAZARD_FREQ

# Dev 2's reward function and shared enums/state — plugged into the hook below.
from rewards.reward_shaping import calculate_reward
from env.action_space import InterventionAction
from env.state_space import OperatorState

# ── Tier-1 hard-rule thresholds (deterministic, untouchable) ────────────
TICKS_PER_SECOND = 10
FATIGUE_CRITICAL = 0.85          # "eyes closed" equivalent probability
FATIGUE_HARD_SECONDS = 2.0       # sustained critical fatigue → Level 3
SPEED_MOVING = 0.3               # m/s above which the machine counts as moving
TILT_WARN = 15.0                 # slope warning (soft)
TILT_CRITICAL = 25.0             # tip-over emergency → Level 3

# ── Soft-hazard thresholds (Tier-2 territory the agent acts on) ─────────
FATIGUE_HIGH = 0.70
FATIGUE_MODERATE = 0.55
BASE_RESPONSE_RATE = 0.75        # operator's baseline chance to heed an alert

LEVEL_3 = 3


class SmartCabinEnv(gym.Env):
    """Gymnasium env replaying excavator telemetry with injected hazards."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        telemetry_csv: Optional[str] = None,
        hazard_freq_csv: Optional[str] = None,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self._telemetry_csv = telemetry_csv
        self._hazard_freq_csv = hazard_freq_csv

        self.action_space = spaces.Discrete(3)
        # [fatigue_prob(0-1), zone(0-2), machine_speed(0-20 m/s),
        #  tilt_angle(0-90 deg), is_reversing(0/1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.0, 20.0, 90.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Built lazily in reset() so seeding is honoured.
        self.telemetry: Optional[TelemetryProvider] = None
        self.injector: Optional[HazardInjector] = None

        self._steps = 0
        self._fatigue_critical_ticks = 0
        self._obs = np.zeros(5, dtype=np.float32)

    # ── gym API ─────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.telemetry = TelemetryProvider(self._telemetry_csv)
        self.injector = HazardInjector(
            freq_csv=self._hazard_freq_csv
            if self._hazard_freq_csv is not None
            else DEV3_HAZARD_FREQ,
            episode_len=self.max_steps,
            rng=self.np_random,
        )
        self.telemetry.reset()
        self.injector.reset()

        self._steps = 0
        self._fatigue_critical_ticks = 0
        self._obs = self._build_observation()
        return self._obs.copy(), {"reset": True}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.telemetry is None or self.injector is None:
            raise RuntimeError("Call reset() before step().")
        action = int(action)
        self._steps += 1

        obs = self._obs
        fatigue, zone, speed, tilt, reversing = (
            float(obs[0]), int(round(obs[1])), float(obs[2]),
            float(obs[3]), int(round(obs[4])),
        )
        moving = speed > SPEED_MOVING

        # ── Tier 1: hard rules. Deterministic. Level 3. Agent cannot touch. ──
        emergency, emergency_reason = self._check_hard_rules(
            fatigue, zone, tilt, moving, reversing
        )

        # ── Tier 2: evaluate the agent's soft (Level 0–2) intervention. ──
        outcome = self._evaluate_soft_action(action, fatigue, zone, tilt, moving)
        if emergency:
            # An unmitigated emergency overrides the soft outcome for reward.
            outcome["collision"] = True

        reward = self._calculate_reward(action, outcome)

        # ── Advance the world to the next observation. ──
        next_obs = self._build_observation()
        self._obs = next_obs

        terminated = bool(emergency)
        truncated = self._steps >= self.max_steps

        info = {
            "level": LEVEL_3 if emergency else action,
            "emergency": emergency,
            "emergency_reason": emergency_reason,
            "outcome": outcome,
            "tilt": tilt,
            "zone": zone,
            "fatigue": fatigue,
            "is_reversing": reversing,
        }
        return next_obs.copy(), reward, terminated, truncated, info

    # ── internals ───────────────────────────────────────────────────────
    def _build_observation(self) -> np.ndarray:
        """Pull the next telemetry row and apply injected hazards."""
        row = self.telemetry.step()
        injected = self.injector.step(
            base_fatigue=0.20,  # replay baseline; live system feeds real fatigue
            base_zone=0,
        )
        return np.array(
            [
                injected["fatigue_probability"],
                float(injected["outside_zone_status"]),
                float(row["machine_speed"]),
                float(row["tilt_angle"]),
                float(row["is_reversing"]),
            ],
            dtype=np.float32,
        )

    def _check_hard_rules(
        self, fatigue: float, zone: int, tilt: float, moving: bool, reversing: int
    ) -> Tuple[bool, str]:
        """Tier-1 deterministic emergency checks → (is_emergency, reason)."""
        # Rule 1: sustained critical fatigue while moving.
        if fatigue >= FATIGUE_CRITICAL and moving:
            self._fatigue_critical_ticks += 1
        else:
            self._fatigue_critical_ticks = 0
        if self._fatigue_critical_ticks >= int(FATIGUE_HARD_SECONDS * TICKS_PER_SECOND):
            return True, "fatigue_critical_sustained"

        # Rule 2: person in RED zone while reversing.
        if zone >= 2 and reversing == 1:
            return True, "red_zone_while_reversing"

        # Rule 3: tilt past critical while moving.
        if tilt >= TILT_CRITICAL and moving:
            return True, "tilt_critical"

        return False, ""

    def _evaluate_soft_action(
        self, action: int, fatigue: float, zone: int, tilt: float, moving: bool
    ) -> dict:
        """Model the Level 0–2 interaction → outcome dict for the reward."""
        # What the ideal soft response would be for this situation.
        if zone >= 2 or fatigue >= FATIGUE_HIGH:
            required = 2
        elif zone == 1 or fatigue >= FATIGUE_MODERATE or (tilt >= TILT_WARN and moving):
            required = 1
        else:
            required = 0
        hazard_present = required > 0

        responded_safely = False
        hazard_resolved = False
        false_warning = False
        near_miss = False

        if action == InterventionAction.NO_ACTION:
            if hazard_present:
                # No nudge while a real hazard builds — risk of a near miss.
                responded_safely = False
                if self.np_random.random() < 0.5 * required:
                    near_miss = True
            else:
                responded_safely = True  # correct restraint → safe operation
        else:
            if not hazard_present:
                false_warning = True
                responded_safely = True  # nuisance alert, operator unaffected
            else:
                p = BASE_RESPONSE_RATE * (1.2 if action == 2 else 1.0)
                responded_safely = self.np_random.random() < min(1.0, p)
                if responded_safely:
                    hazard_resolved = True
                elif action < required:  # under-reacted to a serious hazard
                    near_miss = self.np_random.random() < 0.5

        return {
            "collision": False,
            "near_miss": near_miss,
            "responded_safely": responded_safely,
            "hazard_present": hazard_present,
            "hazard_resolved": hazard_resolved,
            "false_warning": false_warning,
        }

    def _calculate_reward(self, action: int, outcome: dict) -> float:
        """Hook for Dev 2's reward function (rewards/reward_shaping.py).

        Dev 2 owns the reward shaping; Dev 1 only supplies the state/action/
        outcome triple. The OperatorState is reconstructed from the current
        observation so Dev 2's signature is honoured exactly.
        """
        fatigue, zone, speed, _tilt, _rev = self._obs
        state = OperatorState(
            fatigue_score=float(fatigue),
            attention_score=float(1.0 - min(1.0, zone / 2.0)),
            hazard_risk=float(min(1.0, zone / 2.0)),
            obstacle_distance=float(50.0 * (1.0 - min(1.0, zone / 2.0))),
            operator_response_rate=BASE_RESPONSE_RATE,
        )
        return calculate_reward(state, InterventionAction(int(action)), outcome)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        f, z, s, t, r = self._obs
        print(
            f"step {self._steps:3d} | fatigue {f:.2f} | zone {int(z)} | "
            f"speed {s:.1f} | tilt {t:.1f} | rev {int(r)}"
        )


if __name__ == "__main__":
    env = SmartCabinEnv(max_steps=200)
    obs, _ = env.reset(seed=42)
    total = 0.0
    emergencies = 0
    for step in range(env.max_steps):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        total += reward
        if info["emergency"]:
            emergencies += 1
            print(f"  LEVEL 3 @ step {step + 1}: {info['emergency_reason']}")
        if terminated or truncated:
            break
    print(f"random-policy reward over the run: {total:.1f}; emergencies: {emergencies}")
