"""
brain/templates.py — the encrypted operator signature store.

What is on disk is one AES-GCM blob of 512-d float vectors.  There is no image,
and there never was one: enrolment converts frames to vectors in memory.

Two properties matter more than the encryption itself:

  • Cancellable.  Every vector is stored through a fixed random orthogonal
    matrix Q.  Orthogonal maps preserve inner products, so <Qa, Qb> == <a, b>
    and cosine matching is bit-for-bit unaffected — but the stored numbers are
    not valid ArcFace embeddings any more, so a leaked file cannot be replayed
    against a public face model.  And it is revocable in a way a face is not:
    rotate Q, re-enrol, every previously exfiltrated template is now noise.
    Q lives in the key file, deliberately apart from the data it protects.

  • Anchored.  The first `anchors` rows of a template are the enrolment vectors
    and can never be evicted.  Later rows are augmentations picked up when the
    operator is recognised with high confidence on a different machine (a
    different camera, which is the whole reason we enrol in-cab).  Without the
    anchors, a long chain of near-miss augmentations could walk one operator's
    template towards another's — template poisoning.  The anchors pin it.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faceid import EMBED_DIM, l2
from .paths import DATA

TEMPLATES_ENC = Path(os.environ.get("SAARTHI_TEMPLATES", str(DATA / "templates.enc")))
FACEID_KEYFILE = Path(os.environ.get("SAARTHI_FACEID_KEY", str(DATA / "faceid.key")))

MAX_VECTORS = 12            # anchors + augmentations
DEFAULT_RETENTION_DAYS = 730  # purge templates nobody has re-enrolled in 2 years


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CryptoUnavailable(RuntimeError):
    """Raised rather than silently writing biometric vectors in plaintext."""


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:              # pragma: no cover - env dependent
        raise CryptoUnavailable(
            "the `cryptography` package is required to store face templates. "
            "Install it (pip install cryptography) — templates are never written "
            "in plaintext."
        ) from exc
    return AESGCM


# ── template record ──────────────────────────────────────────────────────
@dataclass
class Template:
    operator_id: str
    vectors: np.ndarray                                  # (N, 512) projected, unit-norm
    anchors: int                                         # immutable leading rows
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    enrolled_by: str = "unknown"                         # supervisor id — audit trail
    machines: List[str] = field(default_factory=list)    # cameras seen on
    expires: str = ""

    def as_json(self) -> dict:
        return {
            "operator_id": self.operator_id,
            # float16 halves the file and is far below the noise floor of a
            # cosine threshold; 12 x 512 x 2 bytes = 12 KB per operator.
            "vectors": base64.b64encode(
                self.vectors.astype(np.float16).tobytes()).decode("ascii"),
            "rows": int(self.vectors.shape[0]),
            "anchors": self.anchors,
            "created": self.created,
            "updated": self.updated,
            "enrolled_by": self.enrolled_by,
            "machines": self.machines,
            "expires": self.expires,
        }

    @staticmethod
    def from_json(d: dict) -> "Template":
        raw = np.frombuffer(base64.b64decode(d["vectors"]), dtype=np.float16)
        return Template(
            operator_id=d["operator_id"],
            vectors=raw.reshape(int(d["rows"]), EMBED_DIM).astype(np.float32),
            anchors=int(d.get("anchors", d["rows"])),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
            enrolled_by=d.get("enrolled_by", "unknown"),
            machines=list(d.get("machines", [])),
            expires=d.get("expires", ""),
        )

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if not self.expires:
            return False
        try:
            return (now or datetime.now(timezone.utc)) > datetime.fromisoformat(self.expires)
        except ValueError:
            return False


# ── the store ────────────────────────────────────────────────────────────
class TemplateStore:
    """Encrypted, projection-protected gallery with max-cosine matching."""

    def __init__(self, path: Path = TEMPLATES_ENC, keyfile: Path = FACEID_KEYFILE) -> None:
        self.path = Path(path)
        self.keyfile = Path(keyfile)
        self._lock = threading.RLock()
        self.templates: Dict[str, Template] = {}

        self._key, self._seed = self._load_or_create_key()
        self._Q = self._orthogonal(self._seed)

        # Flat matrix cache for one-shot matching against the whole gallery.
        self._flat: Optional[np.ndarray] = None
        self._owners: List[str] = []

        self.load()

    # ── key material ────────────────────────────────────────────────────
    def _load_or_create_key(self) -> Tuple[bytes, int]:
        if self.keyfile.exists():
            meta = json.loads(self.keyfile.read_text(encoding="utf-8"))
            return base64.b64decode(meta["key"]), int(meta["proj_seed"])

        _aesgcm()   # fail before creating a key we cannot use
        key = secrets.token_bytes(32)
        seed = secrets.randbits(63)
        self.keyfile.parent.mkdir(parents=True, exist_ok=True)
        self.keyfile.write_text(json.dumps({
            "key": base64.b64encode(key).decode("ascii"),
            "proj_seed": seed,
            "created": _now(),
            "note": "AES-256 key + cancellable-projection seed. In production this "
                    "belongs in the machine TPM / secure element, not on the rootfs.",
        }, indent=2), encoding="utf-8")
        try:
            os.chmod(self.keyfile, 0o600)
        except OSError:
            pass                            # Windows dev boxes — ACLs, not modes
        print(f"[Templates] new key material created at {self.keyfile}", flush=True)
        return key, seed

    @staticmethod
    def _orthogonal(seed: int) -> np.ndarray:
        """Deterministic Haar-ish orthogonal matrix from a seed."""
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.standard_normal((EMBED_DIM, EMBED_DIM)))
        # Sign-fix the columns so QR is unique for a given seed.
        return (q * np.sign(np.diag(r))).astype(np.float32)

    def project(self, vec: np.ndarray) -> np.ndarray:
        return l2(self._Q @ np.asarray(vec, dtype=np.float32))

    def rotate_projection(self) -> None:
        """Revoke every stored template. Requires full re-enrolment — by design."""
        with self._lock:
            meta = json.loads(self.keyfile.read_text(encoding="utf-8"))
            meta["proj_seed"] = secrets.randbits(63)
            meta["rotated"] = _now()
            self.keyfile.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._seed = int(meta["proj_seed"])
            self._Q = self._orthogonal(self._seed)
            self.templates.clear()
            self._invalidate()
            self.save()

    # ── persistence ─────────────────────────────────────────────────────
    def load(self) -> None:
        with self._lock:
            self.templates = {}
            if not self.path.exists():
                self._invalidate()
                return
            try:
                blob = self.path.read_bytes()
                nonce, ct = blob[:12], blob[12:]
                plain = _aesgcm()(self._key).decrypt(nonce, ct, b"saarthi-faceid-v1")
                payload = json.loads(plain.decode("utf-8"))
            except CryptoUnavailable:
                raise
            except Exception as exc:        # noqa: BLE001
                print(f"[Templates] could not read {self.path}: {exc}", flush=True)
                self._invalidate()
                return

            purged = 0
            for row in payload.get("templates", []):
                tpl = Template.from_json(row)
                if tpl.is_expired():
                    purged += 1
                    continue
                self.templates[tpl.operator_id] = tpl
            if purged:
                print(f"[Templates] purged {purged} expired template(s).", flush=True)
            self._invalidate()

    def save(self) -> None:
        with self._lock:
            payload = json.dumps({
                "version": 1,
                "saved": _now(),
                "templates": [t.as_json() for t in self.templates.values()],
            }).encode("utf-8")
            nonce = secrets.token_bytes(12)
            ct = _aesgcm()(self._key).encrypt(nonce, payload, b"saarthi-faceid-v1")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(nonce + ct)
            tmp.replace(self.path)          # atomic — never a half-written gallery

    # ── mutation ────────────────────────────────────────────────────────
    def enroll(self, operator_id: str, raw_vectors: List[np.ndarray], *,
               enrolled_by: str = "unknown", machine_id: str = "local",
               retention_days: int = DEFAULT_RETENTION_DAYS) -> Template:
        """Replace an operator's template with a fresh set of enrolment vectors."""
        if not raw_vectors:
            raise ValueError("enrolment needs at least one accepted vector")
        mat = np.stack([self.project(v) for v in raw_vectors]).astype(np.float32)
        expires = (datetime.now(timezone.utc) + timedelta(days=retention_days)).isoformat(
            timespec="seconds") if retention_days else ""
        tpl = Template(operator_id=operator_id, vectors=mat, anchors=int(mat.shape[0]),
                       enrolled_by=enrolled_by, machines=[machine_id], expires=expires)
        with self._lock:
            self.templates[operator_id] = tpl
            self._invalidate()
            self.save()
        return tpl

    def augment(self, operator_id: str, raw_vector: np.ndarray, machine_id: str) -> bool:
        """Add a high-confidence sighting from a new camera. Anchors are never evicted."""
        with self._lock:
            tpl = self.templates.get(operator_id)
            if tpl is None:
                return False
            vec = self.project(raw_vector)[None, :]
            tpl.vectors = np.vstack([tpl.vectors, vec])
            if tpl.vectors.shape[0] > MAX_VECTORS:
                # Drop the oldest *augmentation*; rows [0:anchors] are immutable.
                keep = np.r_[0:tpl.anchors, tpl.anchors + 1:tpl.vectors.shape[0]]
                tpl.vectors = tpl.vectors[keep]
            if machine_id not in tpl.machines:
                tpl.machines.append(machine_id)
            tpl.updated = _now()
            self._invalidate()
            self.save()
            return True

    def remove(self, operator_id: str) -> bool:
        with self._lock:
            if self.templates.pop(operator_id, None) is None:
                return False
            self._invalidate()
            self.save()
            return True

    # ── matching ────────────────────────────────────────────────────────
    def _invalidate(self) -> None:
        self._flat = None
        self._owners = []

    def _build(self) -> None:
        if self._flat is not None or not self.templates:
            return
        mats, owners = [], []
        for oid, tpl in self.templates.items():
            mats.append(tpl.vectors)
            owners.extend([oid] * tpl.vectors.shape[0])
        self._flat = np.vstack(mats).astype(np.float32)
        self._owners = owners

    def identify(self, raw_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Rank operators by MAX cosine over each template set (never the mean).

        Averaging the enrolment poses would blur exactly the variation enrolment
        went to the trouble of capturing; the best-matching pose is the signal.
        """
        with self._lock:
            self._build()
            if self._flat is None:
                return []
            sims = self._flat @ self.project(raw_vector)
            best: Dict[str, float] = {}
            for owner, s in zip(self._owners, sims):
                f = float(s)
                if f > best.get(owner, -2.0):
                    best[owner] = f
            return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def verify(self, operator_id: str, raw_vector: np.ndarray) -> float:
        """1:1 score against one locked operator — the cheap periodic re-check."""
        with self._lock:
            tpl = self.templates.get(operator_id)
            if tpl is None:
                return -1.0
            return float(np.max(tpl.vectors @ self.project(raw_vector)))

    # ── introspection ───────────────────────────────────────────────────
    def enrolled_ids(self) -> List[str]:
        with self._lock:
            return list(self.templates.keys())

    def summary(self, operator_id: str) -> Optional[dict]:
        with self._lock:
            tpl = self.templates.get(operator_id)
            if tpl is None:
                return None
            return {"operator_id": tpl.operator_id, "vectors": int(tpl.vectors.shape[0]),
                    "anchors": tpl.anchors, "created": tpl.created, "updated": tpl.updated,
                    "enrolled_by": tpl.enrolled_by, "machines": tpl.machines,
                    "expires": tpl.expires}
