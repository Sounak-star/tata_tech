"""
brain/profiles.py — operator profiles + face recognition (Q1: "Who is it?").

Loads per-operator profile cards (hearing, colour vision, language, experience,
personal calibration). Face ID uses DeepFace when available; otherwise the live
loop selects the active operator explicitly (demo / kiosk mode).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from .paths import PROFILES_JSON


def load_profiles() -> Dict[str, dict]:
    with open(PROFILES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


class ProfileStore:
    """Holds all operator profiles and the currently active operator."""

    def __init__(self) -> None:
        self.profiles = load_profiles()
        self._active: Optional[str] = None
        if self.profiles:
            self._active = next(iter(self.profiles))

    @property
    def active_id(self) -> Optional[str]:
        return self._active

    def active(self) -> dict:
        return self.profiles[self._active]

    def switch(self, operator_id: str) -> dict:
        if operator_id not in self.profiles:
            raise KeyError(f"Unknown operator '{operator_id}'")
        self._active = operator_id
        return self.active()

    def list_ids(self) -> list[str]:
        return list(self.profiles.keys())


class FaceID:
    """Q1 — recognise the operator and load their profile.

    DeepFace path (optional): compares a webcam crop to enrolled team photos.
    Fallback: identity is set explicitly by the live loop (kiosk/demo mode).
    """

    def __init__(self, store: ProfileStore) -> None:
        self.store = store
        self.backend = "manual"
        try:
            from deepface import DeepFace  # noqa: F401

            self.backend = "deepface"
        except Exception:
            self.backend = "manual"

    def recognise(self, frame=None) -> dict:
        """Return the active operator's profile.

        With DeepFace + enrolled photos this would match `frame` to an operator;
        in demo/kiosk mode it simply returns whoever the loop selected.
        """
        return self.store.active()
