"""
brain/persons.py — Q3: "Is anyone in danger outside?" (blind-spot zones).

In the full product YOLO11-nano draws boxes around people in the outside feed;
box size maps to honest zones: GREEN safe / AMBER caution / RED danger.
When ultralytics/YOLO isn't installed (offline demo, weak laptop), the zone is
supplied by the simulator's hazard injector and simply passed through.
"""

from __future__ import annotations

from typing import Optional

ZONE_NAMES = {0: "GREEN", 1: "AMBER", 2: "RED"}


class PersonDetector:
    """Maps the outside feed to a blind-spot zone (0/1/2)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.backend = "simulated"
        self._model = None
        try:
            from ultralytics import YOLO  # noqa: F401

            if model_path:
                self._model = YOLO(model_path)
                self.backend = "yolo"
        except Exception:
            self.backend = "simulated"

    def detect(self, frame=None, simulated_zone: int = 0) -> dict:
        """Return the current zone.

        With YOLO + an outside frame, this would detect people and bucket by
        proximity. Offline, `simulated_zone` (from the hazard injector) is used.
        """
        if self._model is not None and frame is not None:
            zone = self._zone_from_yolo(frame)
        else:
            zone = int(simulated_zone)
        zone = max(0, min(2, zone))
        return {"zone": zone, "zone_name": ZONE_NAMES[zone]}

    def _zone_from_yolo(self, frame) -> int:
        results = self._model(frame, classes=[0], verbose=False)  # class 0 = person
        max_zone = 0
        for r in results:
            for box in r.boxes.xywhn:  # normalised w,h
                area = float(box[2] * box[3])
                if area > 0.20:
                    max_zone = max(max_zone, 2)   # large/close → RED
                elif area > 0.05:
                    max_zone = max(max_zone, 1)   # mid → AMBER
        return max_zone
