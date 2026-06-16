"""
hazard_injector.py  —  Pipeline B / Dev 1 (consumes Dev 3 analysis)

Probability-based hazard injector for the SmartCabin simulator.

The *relative mix* of hazards comes from Dev 3's real OSHA Severe-Injury
analysis (`track 3/cleaned_data/hazard_frequency.csv`). Those are
incident-level shares, not per-tick rates, so we convert them into per-step
injection probabilities with a single, transparent `hazard_intensity` knob:
roughly `hazard_intensity` hazard events are expected per episode, and *which*
hazard fires is drawn from the real OSHA distribution.

Two simulator-relevant event families are derived from the OSHA hazard types:
  • blind-spot incursion (a person appears behind / beside the machine)
        ← Struck-by + Back-over + Run-over + Collision
  • fatigue spike (drowsiness surge, as replayed from UTA-RLDD live)
        ← not an OSHA category; uses a configured base share

Tilt / tip-over is NOT injected here — it is replayed straight from telemetry
and handled by the Tier-1 hard rules in cabin_env.py.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Default Dev 3 output location (overall hazard distribution).
DEV3_HAZARD_FREQ = os.path.join("track 3", "cleaned_data", "hazard_frequency.csv")

# OSHA hazard types that mean "a person was in the machine's path/blind spot".
_BLIND_SPOT_HAZARDS = {"Struck-by", "Back-over", "Run-over", "Collision", "Caught-in"}

# Fallback distribution (the real numbers Dev 3 reported, in case the CSV is
# unavailable at runtime). Shares of construction heavy-equipment incidents.
_FALLBACK_FREQ = {
    "Struck-by": 0.4033,
    "Fall": 0.2326,
    "Caught-in": 0.1645,
    "Other": 0.0462,
    "Electrocution": 0.0392,
    "Tip-over": 0.0348,
    "Contact-with": 0.0217,
    "Pinch-Point": 0.0201,
    "Collision": 0.0159,
    "Burns": 0.0098,
    "Run-over": 0.0072,
    "Back-over": 0.0049,
}


class HazardInjector:
    """Injects fatigue spikes and blind-spot incursions at OSHA-derived rates."""

    def __init__(
        self,
        freq_csv: Optional[str] = DEV3_HAZARD_FREQ,
        hazard_intensity: float = 4.0,
        episode_len: int = 200,
        fatigue_event_share: float = 0.45,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            freq_csv: path to Dev 3's hazard_frequency.csv.
            hazard_intensity: expected number of hazard events per episode.
            episode_len: nominal episode length (steps) used to spread events.
            fatigue_event_share: fraction of hazard events that are fatigue
                spikes (the rest are physical/blind-spot incursions).
            rng: numpy Generator for reproducibility.
        """
        self.rng = rng or np.random.default_rng()
        self.freq = self._load_frequencies(freq_csv)
        self.source = freq_csv if (freq_csv and os.path.exists(freq_csv)) else "fallback"

        # Share of physical hazards that are blind-spot (person) incursions.
        blind = sum(self.freq.get(h, 0.0) for h in _BLIND_SPOT_HAZARDS)
        total_physical = sum(self.freq.values())
        self.blind_spot_share = blind / total_physical if total_physical else 0.6

        # Convert "events per episode" into a per-step base probability.
        ep = max(1, episode_len)
        self.p_event_per_step = min(1.0, hazard_intensity / ep)
        self.fatigue_event_share = float(np.clip(fatigue_event_share, 0.0, 1.0))

        self.reset()

    @staticmethod
    def _load_frequencies(path: Optional[str]) -> Dict[str, float]:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            return dict(zip(df["hazard_type"], df["frequency"].astype(float)))
        return dict(_FALLBACK_FREQ)

    def reset(self) -> None:
        """Clear any in-progress injected hazard state."""
        self._injected_fatigue = 0.0   # additive fatigue bump (plateau then decay)
        self._fatigue_hold = 0         # ticks left on the fatigue plateau
        self._injected_zone = 0        # 0 GREEN, 1 AMBER, 2 RED (persists briefly)
        self._zone_ttl = 0             # ticks the incursion stays active

    def step(self, base_fatigue: float, base_zone: int) -> Dict[str, float]:
        """Advance one tick and return the (possibly) hazard-modified signals.

        Args:
            base_fatigue: fatigue probability coming from telemetry/replay.
            base_zone: outside-zone status before injection (usually 0/GREEN).

        Returns:
            dict with keys: fatigue_probability, outside_zone_status,
            injected_fatigue (bool), injected_incursion (bool).
        """
        injected_fatigue = False
        injected_incursion = False

        # Hold the fatigue plateau, then decay once the hold expires.
        if self._fatigue_hold > 0:
            self._fatigue_hold -= 1
        else:
            self._injected_fatigue = max(0.0, self._injected_fatigue - 0.05)

        # Persist an active incursion for a few ticks, then clear.
        if self._zone_ttl > 0:
            self._zone_ttl -= 1
            if self._zone_ttl == 0:
                self._injected_zone = 0

        # Roll for a new hazard event this tick.
        if self.rng.random() < self.p_event_per_step:
            if self.rng.random() < self.fatigue_event_share:
                # Fatigue spike (UTA-RLDD-style surge). A minority are sustained
                # "microsleep" episodes that hold high fatigue long enough to
                # trip the Tier-1 hard rule (>2 s critical while moving).
                if self.rng.random() < 0.35:
                    self._injected_fatigue = float(self.rng.uniform(0.70, 0.80))
                    self._fatigue_hold = int(self.rng.integers(22, 30))
                else:
                    self._injected_fatigue = float(self.rng.uniform(0.35, 0.55))
                    self._fatigue_hold = int(self.rng.integers(3, 7))
                injected_fatigue = True
            else:
                # Physical incursion — AMBER vs RED weighted by blind-spot share.
                if self.rng.random() < self.blind_spot_share:
                    self._injected_zone = 2  # RED: directly in the danger zone
                else:
                    self._injected_zone = 1  # AMBER: caution proximity
                self._zone_ttl = int(self.rng.integers(2, 6))
                injected_incursion = True

        fatigue = float(np.clip(base_fatigue + self._injected_fatigue, 0.0, 1.0))
        zone = int(max(base_zone, self._injected_zone))

        return {
            "fatigue_probability": fatigue,
            "outside_zone_status": zone,
            "injected_fatigue": injected_fatigue,
            "injected_incursion": injected_incursion,
        }


if __name__ == "__main__":
    inj = HazardInjector()
    print(f"Frequency source: {inj.source}")
    print(f"Blind-spot share of physical hazards: {inj.blind_spot_share:.2%}")
    print(f"Per-step event probability: {inj.p_event_per_step:.4f}")
    inj.reset()
    fired = 0
    for _ in range(200):
        out = inj.step(base_fatigue=0.2, base_zone=0)
        if out["injected_fatigue"] or out["injected_incursion"]:
            fired += 1
    print(f"Hazard events in 200-step episode: {fired}")
