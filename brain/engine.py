"""
brain/engine.py — the Hybrid Decision Engine (Block 3, the heart).

  Tier 1 — hard rules. Deterministic. Level 3. Nothing can override them:
     • critical fatigue sustained > 2 s while the machine is moving
     • a person in the RED zone while the machine is reversing
     • tilt past the critical angle while moving
  Tier 2 — the learned/adaptive policy for Levels 0–2 (timing & strength).

The PPO agent from Pipeline B is the roadmap target; the *shipped* Tier-2 is the
adaptive-threshold fallback (the plan's checkpoint rule), reused directly from
Pipeline B — `RuleBasedInterventionLogic` + `AdaptiveFallbackPolicy`. If an
operator keeps dismissing gentle nudges, the next warning comes earlier/stronger.

The agent (Tier 2) can NEVER produce or suppress a Level 3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import paths

from env.state_space import OperatorState
from env.action_space import InterventionAction
from policies.adaptive_fallback import AdaptiveFallbackPolicy
from policies.intervention_logic import RuleBasedInterventionLogic

PPO_MODEL_PATH = paths.PIPELINE_B / "models" / "ppo_intervention_policy.zip"


def _load_ppo():
    """Load the trained PPO policy, or None if SB3/model unavailable.

    Trained on cabin_env.py's observation space (raw live signals), NOT the
    OperatorState vector — see _build_obs below.
    """
    try:
        from stable_baselines3 import PPO
        return PPO.load(str(PPO_MODEL_PATH), device="cpu")  # edge inference = CPU
    except Exception as exc:  # missing dep, missing file, version skew
        print(f"[engine] PPO unavailable ({exc}); using adaptive fallback.")
        return None

TICKS_PER_SECOND = 6   # matches server.py TICK_HZ so hard-rule fires at true 2 s
FATIGUE_CRITICAL = 0.85
FATIGUE_HARD_SECONDS = 2.0
SPEED_MOVING = 0.3
TILT_CRITICAL = 25.0

# Posture-mode escalation points. Lower than the face-mode equivalents because
# posture sees drowsiness later; see brain/posture_fatigue.py for why.
POSTURE_NUDGE = 0.30
POSTURE_WARN = 0.50

LEVEL_LABELS = {
    0: "All clear",
    1: "Nudge",
    2: "Warning",
    3: "EMERGENCY",
}


class HybridDecisionEngine:
    """Two-tier engine: untouchable hard rules + adaptive soft policy."""

    def __init__(self, ticks_per_second: int = TICKS_PER_SECOND) -> None:
        self.tps = ticks_per_second
        self.fallback = AdaptiveFallbackPolicy()
        self.logic = RuleBasedInterventionLogic()
        self.model = _load_ppo()
        self._fatigue_critical_ticks = 0
        self._last_action = InterventionAction.NO_ACTION

    def reset(self) -> None:
        self.fallback.reset()
        self._fatigue_critical_ticks = 0
        self._last_action = InterventionAction.NO_ACTION

    # ── Tier 1 ──────────────────────────────────────────────────────────
    def _hard_rules(
        self, fatigue_p: float, zone: int, tilt: float,
        machine_speed: float, is_reversing: int, posture_mode: bool = False,
    ) -> tuple[bool, str]:
        moving = machine_speed > SPEED_MOVING

        # In posture mode the fatigue number comes from shoulders and head angle,
        # not from eyelids. That is real evidence but it is weaker and later, and
        # it is not "eyes closed" — so it may not, on its own, stop the machine.
        # The zone and tilt rules below are untouched: they never needed the face.
        if posture_mode:
            self._fatigue_critical_ticks = 0
        elif fatigue_p >= FATIGUE_CRITICAL and moving:
            self._fatigue_critical_ticks += 1
        else:
            self._fatigue_critical_ticks = 0
        if self._fatigue_critical_ticks >= int(FATIGUE_HARD_SECONDS * self.tps):
            return True, "Eyes closed > 2 s while machine moving"

        if zone >= 2 and is_reversing == 1:
            return True, "Person in RED zone while machine reversing"

        if tilt >= TILT_CRITICAL and moving:
            return True, "Tilt past critical angle while moving"

        return False, ""

    # ── decide ──────────────────────────────────────────────────────────
    def decide(
        self,
        fatigue_p: float,
        zone: int,
        tilt: float,
        machine_speed: float,
        is_reversing: int,
        experience: str = "expert",
        posture_mode: bool = False,
    ) -> dict:
        """Return the alert decision for this tick."""
        emergency, reason = self._hard_rules(
            fatigue_p, zone, tilt, machine_speed, is_reversing, posture_mode
        )
        if emergency:
            self._last_action = InterventionAction.NO_ACTION  # hard rule, not policy
            return {
                "level": 3,
                "label": LEVEL_LABELS[3],
                "tier": "hard-rule",
                "reason": reason,
                "thresholds": self.fallback.get_thresholds(),
            }

        # Tier 2 — the trained PPO brain owns Levels 0–2.
        if self.model is not None:
            # Observation = the raw live signals, in the exact order/scale the
            # model saw in cabin_env.py: [fatigue, zone(0-2), speed, tilt, rev].
            obs = np.array(
                [fatigue_p, float(zone), machine_speed, tilt, float(is_reversing)],
                dtype=np.float32,
            )
            raw, _ = self.model.predict(obs, deterministic=True)
            action = InterventionAction(int(raw))
            tier = "ppo-policy"
        else:
            # Fallback: adaptive rule-based policy when the model can't load.
            state = OperatorState(
                fatigue_score=fatigue_p,
                attention_score=float(1.0 - min(1.0, zone / 2.0)),
                hazard_risk=float(min(1.0, zone / 2.0)),
                obstacle_distance=float(40.0 * (1.0 - min(1.0, zone / 2.0)) + 2.0),
                operator_response_rate=0.7,
            )
            action = self.logic.choose_action(state, self.fallback)
            tier = "adaptive-policy"

        # Trainees get earlier warnings: floor a soft nudge whenever any hazard
        # signal is present but the policy chose to stay silent.
        if experience == "trainee" and action == InterventionAction.NO_ACTION:
            if fatigue_p >= 0.35 or zone >= 1 or tilt >= 12.0:
                action = InterventionAction.LEVEL_1_NUDGE

        # Posture mode detects the same event later than PERCLOS would, so buy
        # that back by alerting sooner. Losing the hard rule (above) has to be
        # paid for somewhere, and an earlier warning is the safe side to err on.
        if posture_mode:
            if fatigue_p >= POSTURE_WARN and action < InterventionAction.LEVEL_2_WARNING:
                action = InterventionAction.LEVEL_2_WARNING
            elif fatigue_p >= POSTURE_NUDGE and action == InterventionAction.NO_ACTION:
                action = InterventionAction.LEVEL_1_NUDGE

        self._last_action = action
        return {
            "level": int(action),
            "label": LEVEL_LABELS[int(action)],
            "tier": tier,
            "reason": "",
            "thresholds": self.fallback.get_thresholds(),
        }

    def feedback(self, responded_safely: bool) -> None:
        """Adapt thresholds from the last issued alert's outcome.

        This is what makes the fallback *adaptive*: ignored nudges make the next
        warning fire earlier and stronger; heeded ones decay back to baseline.
        """
        self.fallback.adjust_thresholds(self._last_action, responded_safely)
