"""
brain/demo_source.py — scripted, looping signal source for the live demo.

Replays excavator telemetry (Pipeline B's TelemetryProvider) for machine speed
and tilt, and overlays scripted fatigue / blind-spot / operator-switch beats that
mirror the 5-minute demo in the plan:

    calm → fatigue droop → recover → blind-spot RED while reversing (Level 3)
         → slope/tilt warning → switch to trainee (Priya) → loop

Everything is reproducible and runs with zero hardware. When a real webcam +
MediaPipe + outside camera are present, swap this for the live capture source.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

from . import paths  # noqa: F401
from telemetry_provider import TelemetryProvider


class DemoSource:
    """Yields one signal dict per tick, looping through the demo beats."""

    def __init__(self) -> None:
        self.telemetry = TelemetryProvider()
        self.telemetry.reset()
        self._t = 0
        # (name, duration_ticks). Tuned for ~6–10 ticks/sec playback.
        self._script = [
            ("calm", 30),
            ("fatigue_ramp", 30),
            ("fatigue_peak", 25),
            ("recover", 20),
            ("blind_spot", 30),
            ("tilt", 25),
            ("switch_priya", 1),
            ("trainee_run", 40),
            ("switch_ravi", 1),
        ]
        self._cycle = sum(d for _, d in self._script)

    def _phase(self, t: int) -> tuple[str, int]:
        """Return (phase_name, ticks_into_phase) for cycle position t."""
        pos = t % self._cycle
        for name, dur in self._script:
            if pos < dur:
                return name, pos
            pos -= dur
        return "calm", 0

    def step(self) -> Dict:
        row = self.telemetry.step()
        speed = float(row["machine_speed"])
        tilt = float(row["tilt_angle"])
        reversing = int(row["is_reversing"])

        phase, k = self._phase(self._t)
        self._t += 1

        drowsiness = 0.08
        zone = 0
        responded: Optional[bool] = None
        switch: Optional[str] = None

        if phase == "fatigue_ramp":
            drowsiness = 0.15 + 0.025 * k          # ramps up
            responded = False                      # operator ignores nudges
        elif phase == "fatigue_peak":
            drowsiness = 0.92
            speed = max(speed, 1.0)                # moving → fatigue is dangerous
            responded = k > 15                     # finally responds late
        elif phase == "recover":
            drowsiness = max(0.08, 0.9 - 0.05 * k)
            responded = True
        elif phase == "blind_spot":
            # Worker walks into the RED zone while the machine is reversing.
            zone = 2 if k >= 8 else (1 if k >= 4 else 0)
            reversing = 1 if k >= 6 else reversing
            speed = max(speed, 0.6)
        elif phase == "tilt":
            # Scripted slope climb: force a steep tilt while moving.
            tilt = 28.0 if k >= 10 else (18.0 if k >= 4 else tilt)
            speed = max(speed, 0.8)
        elif phase == "switch_priya":
            switch = "priya"
        elif phase == "trainee_run":
            drowsiness = 0.2 + 0.01 * k            # mild — trainee warned earlier
        elif phase == "switch_ravi":
            switch = "ravi"

        # Outside the dedicated tilt beat, keep telemetry tilt in a calm range so
        # random replay spikes don't fire spurious emergencies mid-demo.
        if phase != "tilt":
            tilt = min(tilt, 12.0)

        return {
            "phase": phase,
            "drowsiness": round(drowsiness, 3),
            "zone": zone,
            "machine_speed": round(speed, 3),
            "tilt_angle": round(tilt, 2),
            "is_reversing": reversing,
            "responded": responded,
            "_switch_operator": switch,
        }

    def __iter__(self) -> Iterator[Dict]:
        while True:
            yield self.step()
