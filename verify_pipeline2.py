"""
verify_pipeline2.py  —  Pipeline B integrity check (Dev 1 + Dev 2 + Dev 3)

Runs a battery of checks proving the three developers' work integrates into one
coherent simulator, in line with saarthi-official-plan-v3:

  Dev 1  telemetry replay + Gymnasium env + Tier-1 hard rules (Level 3)
  Dev 2  reward shaping plugged into the env's _calculate_reward() hook
  Dev 3  real OSHA hazard frequencies driving the hazard injector

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import sys

import numpy as np

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  - {detail}" if detail else ""))


# ── 1. Imports wire together ────────────────────────────────────────────
from telemetry_provider import TelemetryProvider, REQUIRED_COLUMNS
from hazard_injector import HazardInjector
from cabin_env import SmartCabinEnv, LEVEL_3
from rewards.reward_shaping import calculate_reward
from env.action_space import InterventionAction

check("imports: all Pipeline-B modules load", True)

# ── 2. Dev 3 frequencies actually feed the injector ─────────────────────
inj = HazardInjector()
check(
    "Dev 3 OSHA frequencies loaded by injector",
    inj.source.endswith("hazard_frequency.csv"),
    f"source={inj.source}, blind-spot share={inj.blind_spot_share:.1%}",
)

# ── 3. Telemetry schema ─────────────────────────────────────────────────
tp = TelemetryProvider()
check(
    "telemetry provides required columns",
    all(c in tp.df.columns for c in REQUIRED_COLUMNS),
    f"{len(tp)} rows from {tp.source}",
)
# Looping behaviour
tp.reset()
first = tp.step()
for _ in range(len(tp) - 1):
    tp.step()
looped = tp.step()  # should wrap to row 1 region without error
check("telemetry loops past end without error", True)

# ── 4. Gymnasium contract ───────────────────────────────────────────────
env = SmartCabinEnv(max_steps=200)
obs, info = env.reset(seed=0)
check(
    "observation matches Box space (shape 5)",
    env.observation_space.contains(obs),
    f"obs={np.round(obs, 2).tolist()}",
)
check("action space is Discrete(3)", env.action_space.n == 3)

# ── 5. Reward hook delegates to Dev 2 ───────────────────────────────────
collision_reward = env._calculate_reward(action=2, outcome={"collision": True})
safe_reward = env._calculate_reward(action=0, outcome={"hazard_present": False})
check(
    "reward hook returns Dev 2 collision penalty (-1000)",
    collision_reward == -1000.0,
    f"collision={collision_reward}, safe={safe_reward}",
)
check("reward hook returns Dev 2 safe reward (+10)", safe_reward == 10.0)

# ── 6. All three Tier-1 hard rules are reachable ────────────────────────
reasons: dict[str, int] = {}
for ep in range(200):
    env = SmartCabinEnv(max_steps=200)
    obs, _ = env.reset(seed=ep)
    for _ in range(env.max_steps):
        # Bias toward NO_ACTION so emergencies are allowed to develop.
        a = 0 if env.np_random.random() < 0.7 else int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(a)
        if info["emergency"]:
            reasons[info["emergency_reason"]] = reasons.get(info["emergency_reason"], 0) + 1
            check_level3 = info["level"] == LEVEL_3
            break
        if terminated or truncated:
            break

check(
    "hard rule reachable: tilt_critical",
    reasons.get("tilt_critical", 0) > 0,
    f"fired {reasons.get('tilt_critical', 0)}x",
)
check(
    "hard rule reachable: red_zone_while_reversing",
    reasons.get("red_zone_while_reversing", 0) > 0,
    f"fired {reasons.get('red_zone_while_reversing', 0)}x",
)
check(
    "hard rule reachable: fatigue_critical_sustained",
    reasons.get("fatigue_critical_sustained", 0) > 0,
    f"fired {reasons.get('fatigue_critical_sustained', 0)}x",
)

# ── 7. Agent can NEVER produce Level 3 itself (only the env can) ─────────
env = SmartCabinEnv(max_steps=300)
env.reset(seed=1)
agent_level3 = False
for _ in range(300):
    obs, reward, term, trunc, info = env.step(2)  # always escalate
    if (not info["emergency"]) and info["level"] == LEVEL_3:
        agent_level3 = True
    if term or trunc:
        env.reset(seed=2)
check("agent action never yields Level 3 without an emergency", not agent_level3)

# ── summary ─────────────────────────────────────────────────────────────
print("-" * 60)
passed = sum(1 for _, ok, _ in CHECKS if ok)
print(f"{passed}/{len(CHECKS)} checks passed.")
sys.exit(0 if passed == len(CHECKS) else 1)
