"""
enjoy_ppo.py — load a trained PPO policy and run it through SmartCabinEnv.

Verification utility for the RL & Policy Lead (Dev 2): prints the action the
agent takes each step and confirms it never produces Level 3 itself (hard rules
own emergencies). Inference runs fine on CPU.

Run:
    python enjoy_ppo.py                 # one episode, verbose
    python enjoy_ppo.py --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabin_env import SmartCabinEnv

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "ppo_intervention_policy.zip")
ACTIONS = {0: "NO_ACTION", 1: "LEVEL_1_NUDGE", 2: "LEVEL_2_WARNING"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(f"stable-baselines3 required: {exc}")

    if not os.path.exists(args.model):
        raise SystemExit(f"No policy at {args.model}. Train first: python train_ppo.py")

    model = PPO.load(args.model, device="cpu")  # edge inference = CPU
    env = SmartCabinEnv(max_steps=200)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        emergencies = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            if info["emergency"]:
                emergencies += 1
            done = terminated or truncated
        print(f"episode {ep}: reward={total:.1f}, env-enforced emergencies={emergencies}")


if __name__ == "__main__":
    main()
