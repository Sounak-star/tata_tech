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

        # Add dynamic terrain slope noise when moving to reflect sensor changes realistically
        if speed > 0.2:
            import math
            wave = 2.5 * math.sin(self._t / 6.0)
            jitter = (self._t % 3 - 1) * 0.2
            tilt = max(0.0, tilt + wave + jitter)

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


class HybridLiveSource:
    """
    Wraps DemoSource for simulated phase, zone, tilt, etc.
    But uses real live features from BackgroundCameraTracker instead of drowsiness.
    Also uses BackgroundBlindspotTracker for zone and reversing if available.
    """
    def __init__(self, demo_source: DemoSource, camera_tracker, blindspot_tracker=None):
        self.demo_source = demo_source
        self.camera_tracker = camera_tracker
        self.blindspot_tracker = blindspot_tracker

    def step(self) -> Dict:
        signal = self.demo_source.step()
        # In live-camera mode the camera is the sole fatigue source.
        # Zero drowsiness so the pipeline's synth fallback (used only while the
        # first camera window is still filling) yields ALERT rather than whatever
        # dramatic value the demo script currently has (e.g. 0.92 in fatigue_peak).
        # The script continues to drive zone / tilt / machine_speed only.
        signal["drowsiness"] = 0.0
        latest = self.camera_tracker.get_latest_features()
        if latest is not None:
            signal["features"] = latest
            
        if self.blindspot_tracker and getattr(self.blindspot_tracker, "is_available", False):
            # The blind-spot tracker is running on the clip.
            # Replace the scripted zone with the real YOLO zone.
            signal["zone"] = self.blindspot_tracker.get_latest_zone()
            
            # If the manual trigger activated reversing, override the demo script.
            if self.blindspot_tracker.get_is_reversing():
                signal["is_reversing"] = 1
                signal["machine_speed"] = max(float(signal.get("machine_speed", 0.0)), 0.6)

        return signal

    def __iter__(self) -> Iterator[Dict]:
        while True:
            yield self.step()
