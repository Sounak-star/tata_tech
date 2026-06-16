"""SAARTHI live system (Pipeline C) — the in-cab edge-AI copilot.

This package is the live product: it fuses the three offline pipelines
(A: fatigue, B: intervention/simulator, context-risk) into a real-time loop
that recognises the operator, watches for fatigue and blind-spot danger,
decides an alert level through the Hybrid Decision Engine, personalises the
delivery, and explains every alert with a Reason Card.
"""

from . import paths  # noqa: F401  (side effect: wire sys.path)

__all__ = ["paths"]
