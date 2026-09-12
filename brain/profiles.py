"""
brain/profiles.py — operator profiles + face recognition (Q1: "Who is it?").

Loads per-operator profile cards (hearing, colour vision, language, experience,
personal calibration) and resolves who is in the seat.

Face ID uses an ArcFace ONNX embedder over templates captured during in-cab
enrolment (see brain/enrollment.py).  When the model or the gallery is missing,
identity is set explicitly by the live loop — kiosk/demo mode — and nothing
downstream notices the difference.

The UNKNOWN case is deliberately not "keep the last operator".  Not knowing who
is driving is a reason to be more careful, so we fall back to GUEST_PROFILE:
conservative thresholds, colour-safe palette, every alert channel on.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Dict, Optional

from .paths import PROFILES_JSON

# The strictest profile we can express. Used whenever the operator is unknown.
GUEST_ID = "__guest__"
GUEST_PROFILE: dict = {
    "id": GUEST_ID,
    "name": "Unidentified Operator",
    "role": "Unverified",
    "hearing": "normal",
    # Colour-safe blue/white + icons is legible to colour-normal viewers too,
    # so it is the correct choice when we cannot know which they are.
    "color_vision": "deuteranopia",
    "language": "en",
    "experience": "trainee",        # earliest warnings, simplest UI
    "all_channels": True,           # buzz + flash + sound, from level 1
    "photo": "",
    "calibration": {"calibrated": False},
    "notes": "Operator not recognised — running on the most conservative profile. "
             "Select the operator manually or run enrolment.",
}


def load_profiles() -> Dict[str, dict]:
    with open(PROFILES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


class ProfileStore:
    """Holds all operator profiles and the currently active operator."""

    def __init__(self) -> None:
        self.profiles = load_profiles()
        self._lock = threading.RLock()
        self._active: Optional[str] = None
        self._is_guest = False
        if self.profiles:
            self._active = next(iter(self.profiles))

    @property
    def active_id(self) -> Optional[str]:
        return GUEST_ID if self._is_guest else self._active

    @property
    def is_guest(self) -> bool:
        return self._is_guest

    def active(self) -> dict:
        if self._is_guest:
            return GUEST_PROFILE
        return self.profiles[self._active]

    def switch(self, operator_id: str) -> dict:
        if operator_id == GUEST_ID:
            return self.switch_to_guest()
        if operator_id not in self.profiles:
            raise KeyError(f"Unknown operator '{operator_id}'")
        self._active = operator_id
        self._is_guest = False
        return self.active()

    def switch_to_guest(self) -> dict:
        """Fail-safe: unidentified operator gets the most conservative profile."""
        self._is_guest = True
        return GUEST_PROFILE

    def list_ids(self) -> list[str]:
        return list(self.profiles.keys())

    def get(self, operator_id: str) -> Optional[dict]:
        if operator_id == GUEST_ID:
            return GUEST_PROFILE
        return self.profiles.get(operator_id)

    # ── persistence (enrolment writes here) ─────────────────────────────
    def upsert(self, profile: dict) -> dict:
        """Add or replace a profile and persist. Returns the stored record."""
        oid = profile.get("id")
        if not oid:
            raise ValueError("profile needs an 'id'")
        if oid == GUEST_ID:
            raise ValueError("the guest profile is not editable")
        with self._lock:
            self.profiles[oid] = profile
            self.save()
            return self.profiles[oid]

    def save(self) -> None:
        """Atomic write — a half-written profiles.json would brick every cab."""
        with self._lock:
            target = PROFILES_JSON
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.profiles, f, indent=2, ensure_ascii=False)
                os.replace(tmp, target)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    def baseline_for(self, operator_id: str) -> Optional[dict]:
        """The stored 12-feature fatigue baseline, if this operator has one.

        Its presence is what lets sit-down recognition skip the 25 s calibration.
        """
        prof = self.get(operator_id) or {}
        baseline = (prof.get("calibration") or {}).get("baseline")
        return baseline if isinstance(baseline, dict) and baseline else None


class FaceID:
    """Q1 — recognise the operator and load their profile.

    Owns the embedder, the encrypted template gallery and the sit-down state
    machine.  `recognise()` keeps its original signature so the pipeline does
    not care whether a real model is present.
    """

    def __init__(self, store: ProfileStore, *, machine_id: str = "local",
                 auto_download: bool = False) -> None:
        self.store = store
        self.machine_id = machine_id
        self.backend = "manual"
        self.identifier = None
        self.templates = None
        self.last_error: Optional[str] = None
        # When off, the camera never changes the operator — pure kiosk mode.
        self.auto_enabled = True

        try:
            from .faceid import FaceEmbedder
            from .identify import Identifier
            from .templates import TemplateStore

            self.embedder = FaceEmbedder(auto_download=auto_download)
            self.templates = TemplateStore()
            self.identifier = Identifier(self.embedder, self.templates,
                                         machine_id=machine_id)
            if self.embedder.available:
                self.backend = f"{self.embedder.backend} ({len(self.templates.enrolled_ids())} enrolled)"
            else:
                self.last_error = self.embedder.last_error
        except Exception as exc:            # noqa: BLE001 — never take the cab down
            self.last_error = str(exc)
            print(f"[FaceID] disabled ({exc}); manual operator selection only.", flush=True)

    @property
    def available(self) -> bool:
        return bool(self.identifier and self.identifier.available)

    def observe(self, frame, landmarks, *, has_face: bool, ear: float = 0.0):
        """Feed one frame to the state machine. Returns an IdentifyResult or None."""
        if self.identifier is None:
            return None
        return self.identifier.observe(frame, landmarks, has_face=has_face, ear=ear)

    def set_manual(self, operator_id: Optional[str]) -> None:
        """Dashboard override — always available, and it always wins.

        This LATCHES: the face cannot override a supervisor's choice. Release it
        with reidentify() when the seat changes, otherwise the cab stays pinned
        to whoever was picked last.
        """
        if self.identifier is not None:
            self.identifier.force(operator_id)

    def reidentify(self) -> dict:
        """Drop the manual latch and any locked identity, and scan again.

        Used when the operator changes: without it, the only way out of a manual
        selection was a server restart.
        """
        if self.identifier is None:
            return {"ok": False, "error": self.last_error or "face ID unavailable"}
        self.identifier.reset()
        return {"ok": True, **self.identifier.status()}

    def set_auto(self, enabled: bool) -> dict:
        """Turn automatic recognition on or off without restarting."""
        self.auto_enabled = bool(enabled)
        if self.identifier is not None and not self.auto_enabled:
            # Freeze on whoever is active rather than drifting to guest.
            self.identifier.manual = self.identifier.operator_id is not None
        return self.status()

    def status(self) -> dict:
        if self.identifier is None:
            return {"state": "unavailable", "backend": self.backend, "available": False,
                    "auto": self.auto_enabled,
                    "reason": self.last_error or "not initialised"}
        return {**self.identifier.status(), "auto": self.auto_enabled}

    def recognise(self, frame=None) -> dict:
        """Return the active operator's profile (unchanged legacy signature)."""
        return self.store.active()
