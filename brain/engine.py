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

from . import paths  # noqa: F401 — ensures Pipeline B is importable

from env.state_space import OperatorState
from env.action_space import InterventionAction
from policies.adaptive_fallback import AdaptiveFallbackPolicy
from policies.intervention_logic import RuleBasedInterventionLogic

TICKS_PER_SECOND = 10
FATIGUE_CRITICAL = 0.85
FATIGUE_HARD_SECONDS = 2.0
SPEED_MOVING = 0.3
TILT_CRITICAL = 25.0

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
        self._fatigue_critical_ticks = 0
        self._last_action = InterventionAction.NO_ACTION

    def reset(self) -> None:
        self.fallback.reset()
        self._fatigue_critical_ticks = 0
        self._last_action = InterventionAction.NO_ACTION

    # ── Tier 1 ──────────────────────────────────────────────────────────
    def _hard_rules(
        self, fatigue_p: float, zone: int, tilt: float,
        machine_speed: float, is_reversing: int,
    ) -> tuple[bool, str]:
        moving = machine_speed > SPEED_MOVING

        if fatigue_p >= FATIGUE_CRITICAL and moving:
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
    ) -> dict:
        """Return the alert decision for this tick."""
        emergency, reason = self._hard_rules(
            fatigue_p, zone, tilt, machine_speed, is_reversing
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

        # Tier 2 — adaptive soft policy. Translate live signals to OperatorState.
        state = OperatorState(
            fatigue_score=fatigue_p,
            attention_score=float(1.0 - min(1.0, zone / 2.0)),
            hazard_risk=float(min(1.0, zone / 2.0)),
            obstacle_distance=float(40.0 * (1.0 - min(1.0, zone / 2.0)) + 2.0),
            operator_response_rate=0.7,
        )
        action = self.logic.choose_action(state, self.fallback)

        # Trainees get earlier warnings: floor a soft nudge whenever any hazard
        # signal is present but the policy chose to stay silent.
        if experience == "trainee" and action == InterventionAction.NO_ACTION:
            if fatigue_p >= 0.35 or zone >= 1 or tilt >= 12.0:
                action = InterventionAction.LEVEL_1_NUDGE

        self._last_action = action
        return {
            "level": int(action),
            "label": LEVEL_LABELS[int(action)],
            "tier": "adaptive-policy",
            "reason": "",
            "thresholds": self.fallback.get_thresholds(),
        }

    def feedback(self, responded_safely: bool) -> None:
        """Adapt thresholds from the last issued alert's outcome.

        This is what makes the fallback *adaptive*: ignored nudges make the next
        warning fire earlier and stronger; heeded ones decay back to baseline.
        """
        self.fallback.adjust_thresholds(self._last_action, responded_safely)
