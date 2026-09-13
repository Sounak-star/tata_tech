"""
brain/identify.py — the sit-down recognition state machine.

Recognition is not a per-frame classifier.  A cab is dim, dusty and vibrating,
and a single frame that happens to score 0.41 is not evidence that a particular
human is in the seat.  So identity is resolved by a short vote and then LOCKED:

    IDLE ──face present & stable──► SCANNING
    SCANNING ──~2 s of embeddings──► vote
        accept  → LOCKED(operator)     top1 >= tau AND margin over top2 AND quorum
        reject  → UNKNOWN              conservative profile, never the last driver
    LOCKED ──cheap 1:1 re-verify every 30 s──► LOCKED
    LOCKED ──face absent, or 3 straight re-verify failures──► IDLE (seat changed)

Two guards carry the safety argument:

  • The MARGIN requirement.  In 1:N over a hundred operators, "top-1 cleared the
    threshold" is much weaker than "top-1 cleared the threshold and left the
    runner-up behind".  Without a margin you accept confident-but-wrong matches.

  • The fail-safe DIRECTION.  On UNKNOWN we hand back no operator, so the caller
    applies the strictest profile.  Not knowing who is driving is a reason to be
    more careful, not to inherit the previous operator's expert-level leniency.

The lock also stops identity flapping mid-shift while still catching a genuine
operator swap, because unlocking needs the seat to actually empty.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faceid import FaceEmbedder, assess_quality
from .templates import TemplateStore


class State(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    LOCKED = "locked"
    UNKNOWN = "unknown"


@dataclass
class IdentifyConfig:
    """Operating point.

    THESE NUMBERS ARE STARTING POINTS, NOT TUNED VALUES.  Measure FAR/FRR on the
    real gallery, on the real cab camera, and pick the point where FAR is
    effectively zero — a false accept here means the wrong fatigue baselines on
    a live excavator, which is a safety failure, not a UX annoyance.
    """
    accept_cos: float = 0.40        # ArcFace cosine, 1:N accept threshold
    margin: float = 0.06            # top1 must beat top2 by this
    vote_fraction: float = 0.60     # share of scanned frames that must agree
    scan_embeds: int = 4            # embeddings collected per scan
    scan_timeout_s: float = 3.0
    embed_every: int = 1            # embed 1 frame in N (cost control)
    stable_frames: int = 1          # face present this long before scanning
    absent_frames: int = 12         # face gone this long => seat is empty

    # After an UNKNOWN verdict, stop re-scanning for a while. Without this the
    # dashboard flickers frantically as the system keeps re-trying a face it
    # already knows it can't match.
    unknown_hold_s: float = 1.0
    # and the dashboard flickers UNKNOWN/SCANNING. The seat emptying clears it.
    unknown_cooldown_s: float = 1.0

    reverify_interval_s: float = 30.0
    reverify_cos: float = 0.32      # 1:1 is easier than 1:N — looser is correct
    reverify_fails: int = 3

    # Template augmentation: only from sightings well clear of the accept point.
    augment: bool = True
    augment_cos: float = 0.55
    augment_margin: float = 0.20

    # Passive liveness. Off by default so a mis-tuned blink detector cannot lock
    # an operator out of their machine; turn on once measured on-site.
    require_liveness: bool = False
    liveness_ear_delta: float = 0.06   # EAR range across the scan implying a blink


@dataclass
class IdentifyResult:
    state: State
    operator_id: Optional[str] = None
    score: float = 0.0
    margin: float = 0.0
    votes: str = ""
    liveness: bool = False
    reason: str = ""
    changed: bool = False           # True only on the tick identity changed

    def as_dict(self) -> dict:
        return {"state": self.state.value, "operator_id": self.operator_id,
                "score": round(self.score, 3), "margin": round(self.margin, 3),
                "votes": self.votes, "liveness": self.liveness,
                "reason": self.reason, "changed": self.changed}


class Identifier:
    """Drives IDLE → SCANNING → LOCKED from the frames the tracker already has."""

    def __init__(self, embedder: FaceEmbedder, store: TemplateStore,
                 config: Optional[IdentifyConfig] = None,
                 machine_id: str = "local") -> None:
        self.embedder = embedder
        self.store = store
        self.cfg = config or IdentifyConfig()
        self.machine_id = machine_id

        self.state = State.IDLE
        self.operator_id: Optional[str] = None
        self.last_score = 0.0
        self.last_margin = 0.0
        self.last_reason = ""
        self.manual = False             # identity came from the dashboard, not a face

        self._present = 0
        self._absent = 0
        self._frame = 0
        self._scan_start = 0.0
        self._scan_votes: List[Tuple[str, float, float]] = []
        self._scan_ears: List[float] = []
        self._scan_best_vec: Optional[np.ndarray] = None
        self._scan_best_score = -2.0
        self._last_reverify = 0.0
        self._reverify_fails = 0
        self._unknown_until = 0.0

    # ── public API ──────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.embedder.available and bool(self.store.enrolled_ids())

    def force(self, operator_id: Optional[str]) -> None:
        """Manual override from the dashboard. Always available as a fallback —
        a failed match must never lock a rostered operator out of a machine."""
        self.operator_id = operator_id
        self.manual = operator_id is not None
        self.state = State.LOCKED if operator_id else State.IDLE
        self.last_reason = "selected manually" if operator_id else ""
        self._unknown_until = 0.0
        self._reset_scan()

    def reset(self) -> None:
        self.state = State.IDLE
        self.operator_id = None
        self.manual = False
        self.last_reason = ""
        self._unknown_until = 0.0
        self._reset_scan()

    def status(self) -> dict:
        return {"state": self.state.value, "operator_id": self.operator_id,
                "manual": self.manual, "score": round(self.last_score, 3),
                "margin": round(self.last_margin, 3), "reason": self.last_reason,
                "available": self.available, "backend": self.embedder.backend,
                "enrolled": len(self.store.enrolled_ids()),
                "progress": len(self._scan_votes) if self.state is State.SCANNING else 0,
                "scan_target": self.cfg.scan_embeds}

    def observe(self, bgr: Optional[np.ndarray], landmarks, *,
                has_face: bool, ear: float = 0.0) -> IdentifyResult:
        """Feed one frame. Cheap on most ticks; only embeds 1 frame in N."""
        self._frame += 1

        # Nothing to recognise with: no model, or an empty gallery. Report the
        # state and change nothing — a disabled feature must never move the
        # active operator, least of all off one a supervisor picked by hand.
        if not self.available:
            return IdentifyResult(self.state, self.operator_id,
                                  reason=self.embedder.last_error or "no templates enrolled")

        # A manual pick is the supervisor's decision and outranks the camera,
        # including the camera seeing nobody. Only another manual selection, or
        # a positive identification of a different face, replaces it.
        if self.manual:
            return IdentifyResult(State.LOCKED, self.operator_id, reason="manual")

        if not has_face or landmarks is None or bgr is None:
            return self._on_absent()

        self._absent = 0
        self._present += 1

        if self.state is State.LOCKED:
            return self._maybe_reverify(bgr, landmarks)

        if self.state is State.UNKNOWN and time.time() < self._unknown_until:
            return IdentifyResult(State.UNKNOWN, None, reason=self.last_reason)

        if self.state in (State.IDLE, State.UNKNOWN):
            if self._present < self.cfg.stable_frames:
                return IdentifyResult(self.state, self.operator_id, reason="settling")
            self._begin_scan()

        return self._scan_step(bgr, landmarks, ear)

    # ── internals ───────────────────────────────────────────────────────
    def _reset_scan(self) -> None:
        self._scan_votes = []
        self._scan_ears = []
        self._scan_best_vec = None
        self._scan_best_score = -2.0
        self._present = 0
        self._absent = 0

    def _begin_scan(self) -> None:
        self.state = State.SCANNING
        self._scan_start = time.time()
        self._scan_votes = []
        self._scan_ears = []
        self._scan_best_vec = None
        self._scan_best_score = -2.0
        self.last_reason = "identifying…"

    def _on_absent(self) -> IdentifyResult:
        self._present = 0
        self._absent += 1
        if self._absent >= self.cfg.absent_frames and self.state is not State.IDLE:
            # Seat emptied — the next person to sit down gets a fresh scan.
            was = self.operator_id
            self.state = State.IDLE
            self.operator_id = None
            self._unknown_until = 0.0
            self._reset_scan()
            self.last_reason = "seat empty"
            return IdentifyResult(State.IDLE, None, reason="seat empty", changed=was is not None)
        return IdentifyResult(self.state, self.operator_id, reason="no face")

    def _scan_step(self, bgr, landmarks, ear: float) -> IdentifyResult:
        if ear:
            self._scan_ears.append(float(ear))

        if self._frame % self.cfg.embed_every:
            return IdentifyResult(State.SCANNING, None, reason="identifying…")

        if assess_quality(bgr, landmarks, strict=False).ok:
            vec = self.embedder.embed_frame(bgr, landmarks)
            if vec is not None:
                ranked = self.store.identify(vec, top_k=2)
                if ranked:
                    top_id, top_s = ranked[0]
                    second = ranked[1][1] if len(ranked) > 1 else -1.0
                    self._scan_votes.append((top_id, top_s, top_s - second))
                    if top_s > self._scan_best_score:
                        self._scan_best_score, self._scan_best_vec = top_s, vec

        timed_out = (time.time() - self._scan_start) > self.cfg.scan_timeout_s
        if len(self._scan_votes) >= self.cfg.scan_embeds or timed_out:
            return self._decide(timed_out)

        return IdentifyResult(State.SCANNING, None, reason="identifying…")

    def _liveness_ok(self) -> bool:
        if len(self._scan_ears) < 4:
            return False
        return (max(self._scan_ears) - min(self._scan_ears)) >= self.cfg.liveness_ear_delta

    def _decide(self, timed_out: bool) -> IdentifyResult:
        votes = self._scan_votes
        alive = self._liveness_ok()

        if not votes:
            return self._to_unknown("no usable frames" if timed_out else "no match", alive)

        counts = Counter(v[0] for v in votes)
        winner, n = counts.most_common(1)[0]
        quorum = n / len(votes)
        wins = [v for v in votes if v[0] == winner]
        score = float(np.median([v[1] for v in wins]))
        margin = float(np.median([v[2] for v in wins]))

        self.last_score, self.last_margin = score, margin

        if quorum < self.cfg.vote_fraction:
            return self._to_unknown(f"frames disagreed ({quorum:.0%})", alive)
        if score < self.cfg.accept_cos:
            return self._to_unknown(f"below threshold ({score:.2f})", alive)
        if margin < self.cfg.margin:
            return self._to_unknown(f"too close to call (margin {margin:.2f})", alive)
        if self.cfg.require_liveness and not alive:
            return self._to_unknown("liveness check failed", alive)

        # Confident sighting on a camera this operator has not been seen on:
        # fold it into their template so the fleet gets easier over time.
        if (self.cfg.augment and self._scan_best_vec is not None
                and score >= self.cfg.augment_cos and margin >= self.cfg.augment_margin
                and self.machine_id not in (self.store.summary(winner) or {}).get("machines", [])):
            self.store.augment(winner, self._scan_best_vec, self.machine_id)

        changed = self.operator_id != winner
        self.state = State.LOCKED
        self.operator_id = winner
        self.manual = False
        self._last_reverify = time.time()
        self._reverify_fails = 0
        self.last_reason = f"matched {score:.2f} (margin {margin:.2f})"
        votes_str = f"{n}/{len(votes)}"
        self._reset_scan()
        return IdentifyResult(State.LOCKED, winner, score, margin, votes_str,
                              alive, self.last_reason, changed)

    def _to_unknown(self, reason: str, alive: bool) -> IdentifyResult:
        changed = self.operator_id is not None
        self.state = State.UNKNOWN
        self.operator_id = None          # caller falls back to the strict profile
        self.last_reason = reason
        self._unknown_until = time.time() + self.cfg.unknown_cooldown_s
        self._reset_scan()
        return IdentifyResult(State.UNKNOWN, None, self.last_score, self.last_margin,
                              "", alive, reason, changed)

    def _maybe_reverify(self, bgr, landmarks) -> IdentifyResult:
        now = time.time()
        if now - self._last_reverify < self.cfg.reverify_interval_s:
            return IdentifyResult(State.LOCKED, self.operator_id, self.last_score,
                                  self.last_margin, reason="locked")

        self._last_reverify = now
        vec = self.embedder.embed_frame(bgr, landmarks)
        if vec is None:
            return IdentifyResult(State.LOCKED, self.operator_id, reason="locked")

        score = self.store.verify(self.operator_id, vec)
        if score >= self.cfg.reverify_cos:
            self._reverify_fails = 0
            return IdentifyResult(State.LOCKED, self.operator_id, score,
                                  reason=f"re-verified {score:.2f}")

        self._reverify_fails += 1
        if self._reverify_fails < self.cfg.reverify_fails:
            return IdentifyResult(State.LOCKED, self.operator_id, score,
                                  reason=f"re-verify miss {self._reverify_fails}")

        # Someone else appears to be in the seat — drop the lock and rescan.
        was = self.operator_id
        self.operator_id = None
        self.state = State.IDLE
        self._reset_scan()
        self.last_reason = "operator changed"
        return IdentifyResult(State.IDLE, None, score, reason="operator changed",
                              changed=was is not None)
