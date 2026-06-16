"""
training/gpu_utils.py — shared GPU helpers for the offline training scripts.

The dev briefs require CUDA-accelerated training (XGBoost `device='cuda'`, PPO on
a torch CUDA device). These helpers auto-detect a usable GPU and fall back to CPU
so the same scripts run on any teammate's laptop without edits.
"""

from __future__ import annotations


def xgb_device() -> str:
    """Return 'cuda' if an NVIDIA GPU usable by XGBoost is present, else 'cpu'."""
    # Prefer torch's detector (most reliable); fall back to nvidia-smi presence.
    try:
        import torch

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


def torch_device() -> str:
    """Return 'cuda' if torch sees a CUDA GPU, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def describe() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "CPU (no CUDA detected)"
