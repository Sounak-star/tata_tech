"""
train_ppo.py — Pipeline B / RL & Policy Lead (Dev 2)

Trains a PPO policy (Stable-Baselines3, PyTorch backend) to optimise early-warning
intervention timing inside the custom Gymnasium SmartCabinEnv.

  • Safety-bounded: PPO only chooses soft Levels 0–2. Tier-1 hard rules (Level 3)
    live in the environment and can never be modified or overridden by the agent.
  • CUDA-accelerated: PyTorch trains the MLP policy on the NVIDIA GPU when present
    (auto-detected), drastically cutting training time; falls back to CPU.
  • TensorBoard logging of reward convergence + policy loss.

Run:
    python train_ppo.py                  # default 200k steps
    python train_ppo.py --steps 500000   # longer run
    tensorboard --logdir runs/ppo_tb     # watch convergence

If RL doesn't converge by the checkpoint, the shipped fallback is the deterministic
adaptive-threshold policy (policies/), so the demo is never at risk.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cabin_env import SmartCabinEnv

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ppo_tb")
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_intervention_policy.zip")


def make_env() -> SmartCabinEnv:
    return SmartCabinEnv(max_steps=200)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO intervention policy.")
    ap.add_argument("--steps", type=int, default=200_000, help="total timesteps")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 + torch (CUDA) required. Install:\n"
            "  pip install -r requirements_dev2.txt\n"
            f"  (import error: {exc})"
        )

    try:
        from gpu_utils import torch_device, describe
        device = args.device or torch_device()
        print(f"[train_ppo] {describe()} | training device: {device}")
    except Exception:
        device = args.device or "cpu"

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    # 4 parallel envs for throughput; VecMonitor records episode rewards for TB.
    venv = VecMonitor(make_vec_env(make_env, n_envs=4))

    model = PPO(
        "MlpPolicy",
        venv,
        device=device,                 # CUDA enforced when available
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=TB_DIR,
    )

    print(f"[train_ppo] training for {args.steps:,} steps ...")
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(MODEL_PATH)
    print(f"[train_ppo] saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
