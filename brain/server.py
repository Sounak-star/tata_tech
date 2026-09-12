"""
brain/server.py — FastAPI + WebSocket: the brain → screen pipe (Block 5).

Runs the live loop and pushes each frame as JSON to the dashboard and phone over
a WebSocket. Also serves the dashboard (machine-panel UI) and the phone buzz page,
and exposes a small REST API for profiles / operator switching / the event log.

Run:
    python -m brain.server          # then open http://localhost:8000
    # or:  uvicorn brain.server:app --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import threading
import time
from typing import List, Optional

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import paths
from .demo_source import DemoSource, HybridLiveSource
from .enrollment import EnrollmentSession, OperatorDetails, SupervisorAuth
from .pipeline import Brain
from .live_camera import BackgroundCameraTracker, RemoteCameraTracker, HAS_MEDIAPIPE
from .occlusion import OcclusionTracker
from .pose import PoseTracker
from .blindspot import BackgroundBlindspotTracker

TICK_HZ = 6.0  # readable playback rate for the demo (raise toward 10 for "live")
CAMERA_INDEX = 0  # Configurable camera index

app = FastAPI(title="SAARTHI — SmartCabin AI Copilot")

# Remote tracker for browser-streamed webcam (always available if mediapipe installed)
remote_tracker = RemoteCameraTracker() if HAS_MEDIAPIPE else None

if HAS_MEDIAPIPE:
    tracker = BackgroundCameraTracker(camera_index=CAMERA_INDEX)
    blindspot_tracker = BackgroundBlindspotTracker()
    brain = Brain()  # Will be re-initialized after calibration
    source = HybridLiveSource(DemoSource(), tracker, blindspot_tracker)
else:
    tracker = None
    blindspot_tracker = BackgroundBlindspotTracker()
    brain = Brain()
    source = DemoSource()


# Posture fallback. Shares the frames the face pipeline already decodes and
# runs at a low duty cycle — posture changes over seconds, not frames.
pose_tracker = PoseTracker(every_n=2) if HAS_MEDIAPIPE else None

# Is the face merely FITTED, or actually visible? MediaPipe keeps reporting a
# mesh through sunglasses, so "has_face" cannot answer this and the fatigue
# model would be fed geometry nobody can see.
occlusion = OcclusionTracker()

# ── face ID + enrolment state ───────────────────────────────────────────
MACHINE_ID = os.environ.get("SAARTHI_MACHINE_ID", "local")
BASELINE_SECONDS = 25.0
BASELINE_MAX_SECONDS = 45.0   # keep waiting for windows, up to this
MIN_BASELINE_WINDOWS = 5      # below this, mean/std per feature is noise

_auth_inst: Optional[SupervisorAuth] = None
_enroll: Optional[EnrollmentSession] = None
_enroll_lock = threading.Lock()
_tokens: dict = {}                      # token -> (supervisor_id, name, expires)
TOKEN_TTL_S = 900.0

# Identity changes are queued here by the camera thread and applied by _loop(),
# so the brain is never mutated halfway through a tick.
_pending_identity: Optional[tuple] = None
_last_identify: dict = {}

# Which camera is actually feeding us. Two trackers can be alive at once (the
# local OpenCV one and the browser one) and they see DIFFERENT things, so face
# features and posture must come from the same camera or we would be scoring one
# person's posture against another camera's face. Whichever tracker last
# delivered a frame is the operator camera.
_source_seen: dict = {"remote": 0.0, "local": 0.0}
ACTIVE_SOURCE_TIMEOUT = 3.0


def _active_source_name() -> Optional[str]:
    """Which camera is the operator camera right now.

    Deterministic preference, not last-frame-wins: when both trackers are alive
    they each deliver frames constantly, so "whoever spoke last" thrashes between
    them and the fatigue source flips every tick. The browser stream wins while
    it is live because connecting it is an explicit act by the operator; the
    local camera is the fallback for a real cab install.
    """
    now = time.time()
    for name in ("remote", "local"):
        if now - _source_seen[name] <= ACTIVE_SOURCE_TIMEOUT:
            return name
    return None


def _active_tracker():
    name = _active_source_name()
    if name is None:
        return None
    return remote_tracker if name == "remote" else tracker


def auth() -> SupervisorAuth:
    """Built on first use — importing this module must not mint a PIN."""
    global _auth_inst
    if _auth_inst is None:
        _auth_inst = SupervisorAuth()
    return _auth_inst


def _issue_token(sup_id: str, name: str) -> str:
    token = secrets.token_urlsafe(24)
    _tokens[token] = (sup_id, name, time.time() + TOKEN_TTL_S)
    return token


def _check_token(token: str) -> Optional[tuple]:
    rec = _tokens.get(token or "")
    if rec is None:
        return None
    if time.time() > rec[2]:
        _tokens.pop(token, None)
        return None
    return rec


def _finite(x):
    """JSON-safe float: inf/NaN become None."""
    import math
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), 2)


def _face_router(frame, landmarks, *, has_face: bool, ear: float = 0.0,
                 yaw: float = 0.0, pitch: float = 0.0,
                 source: str = "local") -> None:
    """Called by the camera thread for every landmarked frame.

    Enrolment takes priority over recognition — while a supervisor is enrolling
    somebody we must not also be trying to identify them.
    """
    global _pending_identity, _last_identify

    _source_seen[source] = time.time()

    # Only the operator camera drives identity and posture. A second tracker
    # still running in the background must not inject its own view of the world.
    if source != _active_source_name():
        return

    # Pose runs whether or not the face is readable — that is the entire point,
    # and it is also how we tell an occluded operator from an empty seat.
    if pose_tracker is not None and frame is not None:
        pose_tracker.observe(frame)

    with _enroll_lock:
        session = _enroll

    # During the 25 s baseline the operator is sitting still and alert — exactly
    # the posture we want as their reference. Same sitting, no extra step.
    if (session is not None and session.step.value == "baseline"
            and pose_tracker is not None and pose_tracker.available):
        session.add_posture_sample(pose_tracker.latest)

    # Only meaningful when a face is being reported — that is exactly the case
    # this check exists for.
    if has_face and frame is not None and landmarks is not None:
        occlusion.update(frame, landmarks, ear=ear)
    elif not has_face:
        occlusion.reset()

    if session is not None and session.step.value == "poses":
        if has_face and frame is not None:
            session.feed_frame(frame, landmarks, yaw=yaw, pitch=pitch)
        return

    if not brain.faceid.auto_enabled:
        return

    # An occluded operator is not an empty seat. Without this, sunglasses and a
    # dust mask look exactly like the driver getting out, and the identity lock
    # drops to the guest profile while they are still sitting there.
    if (not has_face and pose_tracker is not None and pose_tracker.present
            and brain.faceid.identifier is not None
            and brain.faceid.identifier.operator_id is not None):
        return

    result = brain.faceid.observe(frame, landmarks, has_face=has_face, ear=ear)
    if result is None:
        return
    _last_identify = result.as_dict()
    if result.changed:
        _pending_identity = ((result.operator_id, result.reason)
                             if result.operator_id else (None, result.reason))


class Hub:
    """Tracks connected dashboard/phone clients."""

    def __init__(self) -> None:
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = Hub()


async def _loop() -> None:
    """The heartbeat: pull a signal, run the brain, broadcast the frame."""
    global _pending_identity
    while True:
        signal = source.step()

        # Apply any identity resolved by the camera thread since the last tick.
        pending, _pending_identity = _pending_identity, None
        if pending is not None:
            op_id, why = pending
            try:
                if op_id:
                    brain.switch_operator(op_id, source="face")
                else:
                    brain.switch_to_guest(why)
            except KeyError:
                # Enrolled face with no profile record — safest is the guest profile.
                brain.switch_to_guest(f"no profile for '{op_id}'")

        # Face features come from the camera that is actually delivering frames,
        # and from nowhere else. HybridLiveSource pre-fills signal["features"]
        # from the LOCAL tracker; if the operator is on the browser camera that
        # stale local reading would shadow everything, face_ok would never go
        # false, and the posture fallback could never engage no matter what the
        # operator did in front of the real camera.
        active = _active_tracker()
        if active is not None:
            feats = active.get_latest_features()
            if feats is not None:
                signal["features"] = feats
                signal["drowsiness"] = 0.0   # real features override the demo script
            else:
                # Expired or never seen: say so, rather than letting the other
                # camera answer for this one.
                signal.pop("features", None)

        if pose_tracker is not None and pose_tracker.available:
            signal["pose"] = pose_tracker.latest

        # Eyes covered means EAR/PERCLOS/blink are fiction, even though the
        # landmarker is still producing them. Treat that as no face signal.
        signal["eyes_covered"] = occlusion.eyes_covered
        signal["occlusion_reason"] = occlusion.reason()

        frame = brain.tick(signal)
        frame["phase"] = signal.get("phase")
        # Keep the camera preview overlay in sync with the latest fatigue decision.
        if HAS_MEDIAPIPE:
            tracker.set_fatigue_info(frame["fatigue"])
        if remote_tracker and remote_tracker.is_available:
            remote_tracker.set_fatigue_info(frame["fatigue"])
            # Piggyback the annotated frame onto the WS broadcast so the
            # dashboard can show the face-mesh overlay without MJPEG streaming
            jpeg = remote_tracker.get_latest_annotated_frame()
            if jpeg:
                frame["cam_frame"] = base64.b64encode(jpeg).decode("ascii")
                
        # Inject debugging info for the dashboard
        frame["cam_status"] = {
            "has_mediapipe": HAS_MEDIAPIPE,
            "tracker_exists": remote_tracker is not None,
            "is_available": remote_tracker.is_available if remote_tracker else False,
            "error": getattr(remote_tracker, "last_error", "No tracker") if remote_tracker else "No tracker"
        }
            
        frame["faceid"] = {
            **brain.faceid.status(),
            "last": _last_identify,
            "calibration_source": brain.calibration_source,
            "guest": brain.store.is_guest,
        }
        if pose_tracker is not None:
            frame.setdefault("posture", {})["tracker"] = pose_tracker.status()
        # How stale the face signal is, so the dashboard can show the whole chain
        # (face -> body -> baseline -> mode) instead of only the end result.
        _act = _active_tracker()
        frame.setdefault("posture", {})["face_age"] = _finite(
            _act.features_age() if _act is not None else None)
        frame["posture"]["camera"] = _active_source_name() or "none"
        frame["posture"]["occlusion"] = occlusion.status()

        with _enroll_lock:
            frame["enrolling"] = _enroll is not None and _enroll.step.value in ("poses", "baseline")

        await hub.broadcast(frame)
        await asyncio.sleep(1.0 / TICK_HZ)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    """MJPEG stream of the annotated camera feed for the dashboard preview."""
    # Prefer local camera tracker; fall back to remote (browser) tracker
    active_tracker = None
    if HAS_MEDIAPIPE and tracker and tracker.cap and tracker.cap.isOpened():
        active_tracker = tracker
    elif remote_tracker and remote_tracker.is_available:
        active_tracker = remote_tracker

    if active_tracker is None:
        return JSONResponse({"error": "camera not available"}, status_code=503)

    _trk = active_tracker  # capture for the closure

    async def _gen():
        while True:
            jpeg = _trk.get_latest_annotated_frame()
            if jpeg is not None:
                chunk = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                yield chunk
            await asyncio.sleep(1.0 / 20.0)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/blindspot_feed")
async def blindspot_feed() -> StreamingResponse:
    """MJPEG stream of the annotated blind-spot clip."""
    if not getattr(blindspot_tracker, "is_available", False):
        return JSONResponse({"error": "blind-spot tracker not available"}, status_code=503)

    async def _gen():
        while True:
            jpeg = blindspot_tracker.get_latest_annotated_frame()
            if jpeg is not None:
                chunk = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                yield chunk
            await asyncio.sleep(1.0 / 20.0)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _calibrate_and_start_camera():
    if HAS_MEDIAPIPE:
        print("[Server] Starting calibration background task...")
        loop = asyncio.get_running_loop()
        try:
            # Run the blocking calibration in a separate thread
            calibration_rows = await loop.run_in_executor(None, tracker.calibrate, 25.0)
            
            # Inject the calibrated engine into the live brain
            from .fatigue import FatigueEngine
            brain.fatigue = FatigueEngine(calibration_rows)
            
            tracker.start()
            print("[Server] Calibration complete. Live features now driving fatigue.")
        except Exception as exc:
            print(f"[Server] Webcam calibration bypassed or failed: {exc}. Running on synthetic baseline.")


@app.on_event("startup")
async def _startup() -> None:
    # Face ID reuses the frames these trackers already decode and landmark.
    if HAS_MEDIAPIPE and tracker is not None:
        tracker.set_face_observer(
            lambda *a, **kw: _face_router(*a, source="local", **kw))
    if remote_tracker is not None:
        remote_tracker.set_face_observer(
            lambda *a, **kw: _face_router(*a, source="remote", **kw))
    blindspot_tracker.start()
    asyncio.create_task(_calibrate_and_start_camera())
    asyncio.create_task(_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    blindspot_tracker.stop()
    if HAS_MEDIAPIPE:
        tracker.stop()
    if remote_tracker:
        remote_tracker.stop()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] and remote_tracker:
                # Binary message = JPEG frame from browser webcam
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, remote_tracker.feed_frame, msg["bytes"])
            # Text messages are keepalive pings — ignore
    except WebSocketDisconnect:
        hub.disconnect(ws)


# ── REST API ────────────────────────────────────────────────────────────
@app.get("/api/profiles")
def api_profiles() -> JSONResponse:
    """Every profile, each tagged with whether it has a face template.

    The full roster stays available: face ID failing must never stop a
    supervisor selecting a rostered operator by hand.
    """
    templates = brain.faceid.templates
    enrolled = set(templates.enrolled_ids()) if templates else set()
    return JSONResponse({
        oid: {**prof, "enrolled": oid in enrolled}
        for oid, prof in brain.store.profiles.items()
    })


@app.post("/api/operator/{operator_id}")
def api_switch(operator_id: str) -> JSONResponse:
    try:
        prof = brain.switch_operator(operator_id, source="manual")
        # A manual pick outranks the face — otherwise a bad match could
        # keep yanking an operator off the profile a supervisor just chose.
        brain.faceid.set_manual(operator_id)
        return JSONResponse({"ok": True, "operator": prof})
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown operator"}, status_code=404)

@app.delete("/api/operator/{operator_id}")
def api_delete_operator(operator_id: str, token: str = "") -> JSONResponse:
    """Delete an operator: profile card AND face signature, together.

    Deleting only one of the two leaves the system in a state it cannot explain.
    An orphaned template matches a person there is no profile for; an orphaned
    profile claims an operator the camera can no longer recognise. So this is one
    operation, and it is PIN-gated and logged — erasing someone's biometric data
    is a right they have under the DPDP Act, and a supervisor needs to be able to
    show who exercised it and when.
    """
    rec = _check_token(token)
    if rec is None:
        return JSONResponse({"ok": False, "error": "supervisor PIN required"},
                            status_code=401)

    profile = brain.store.get(operator_id)
    if profile is None:
        return JSONResponse({"ok": False, "error": "unknown operator"},
                            status_code=404)

    was_active = brain.store.active_id == operator_id
    templates = brain.faceid.templates
    had_template = bool(templates and templates.summary(operator_id))

    try:
        removed = brain.store.delete(operator_id)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not removed:
        return JSONResponse({"ok": False, "error": "unknown operator"},
                            status_code=404)
    if had_template:
        templates.remove(operator_id)

    # Whoever was recognised as this person must stop being recognised as them.
    if brain.faceid.identifier is not None:
        brain.faceid.identifier.reset()
    if was_active:
        brain.switch_to_guest(f"operator {operator_id} deleted")

    brain.log.log(rec[0], 1, "admin", {
        "title": "Operator deleted",
        "detail": (f"{profile.get('name', operator_id)} ({operator_id}) removed by "
                   f"{rec[1]}"
                   f"{' — face signature erased' if had_template else ''}"),
        "factors": [], "source": "admin",
    }, 0.0)

    print(f"[Admin] operator {operator_id} deleted by {rec[0]} "
          f"(template={'yes' if had_template else 'none'})", flush=True)
    return JSONResponse({"ok": True, "deleted": operator_id,
                         "template_removed": had_template,
                         "was_active": was_active,
                         "remaining": len(brain.store.profiles)})


@app.post("/api/trigger_blindspot")
def api_trigger_blindspot() -> JSONResponse:
    if blindspot_tracker and getattr(blindspot_tracker, "is_available", False):
        blindspot_tracker.trigger_blindspot()
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False, "error": "Tracker unavailable"}, status_code=503)


@app.get("/api/timeline")
def api_timeline() -> JSONResponse:
    return JSONResponse(brain.log.recent(30))


@app.get("/api/health")
def api_health() -> JSONResponse:
    person_backend = "YOLO11n (live detection)" if getattr(blindspot_tracker, "is_available", False) else brain.persons.backend
    return JSONResponse({
        "ok": True,
        "fatigue_backend": brain.fatigue.backend,
        "person_backend": person_backend,
        "faceid_backend": brain.faceid.backend,
        "pose_backend": pose_tracker.backend if pose_tracker else "unavailable",
        "tick_hz": TICK_HZ,
    })


# ── face ID ─────────────────────────────────────────────────────────────
@app.get("/api/faceid/status")
def api_faceid_status() -> JSONResponse:
    templates = brain.faceid.templates
    return JSONResponse({
        "backend": brain.faceid.backend,
        "available": brain.faceid.available,
        "error": brain.faceid.last_error,
        "machine_id": MACHINE_ID,
        "enrolled": templates.enrolled_ids() if templates else [],
        "state": brain.faceid.status(),
        "last": _last_identify,
        "active_operator": brain.store.active_id,
        "guest": brain.store.is_guest,
        "calibration_source": brain.calibration_source,
    })


@app.get("/api/posture/status")
def api_posture_status() -> JSONResponse:
    return JSONResponse({
        "tracker": pose_tracker.status() if pose_tracker else
                   {"available": False, "error": "mediapipe unavailable"},
        "mode": brain.posture_mode,
        "reason": brain.posture_reason,
        "baseline_ready": brain.posture.ready,
        "occlusion": occlusion.status(),
        "occlusion_reason": occlusion.reason(),
        # features_age() is inf when no face window has ever landed, and inf is
        # not JSON. None means "never seen", which is what the UI wants anyway.
        "camera": _active_source_name() or "none",
        "face_signal_age": _finite(
            _active_tracker().features_age() if _active_tracker() else None),
    })


@app.post("/api/faceid/reidentify")
def api_reidentify() -> JSONResponse:
    """Forget who is in the seat and scan again — used when the operator changes.

    Clicking a name in the directory latches a manual override so a bad match
    cannot yank the profile away. That latch has to be releasable, or swapping
    operators means restarting the machine.
    """
    result = brain.faceid.reidentify()
    if brain.store.is_guest:
        brain.store.switch_to_guest()          # stay conservative until resolved
    return JSONResponse(result)


@app.post("/api/faceid/auto")
def api_faceid_auto(body: dict = Body(...)) -> JSONResponse:
    enabled = bool(body.get("enabled", True))
    status = brain.faceid.set_auto(enabled)
    if enabled:
        brain.faceid.reidentify()              # turning it back on rescans
    print(f"[FaceID] automatic recognition {'enabled' if enabled else 'disabled'}",
          flush=True)
    return JSONResponse({"ok": True, **status})


@app.get("/api/faceid/template/{operator_id}")
def api_template_summary(operator_id: str) -> JSONResponse:
    templates = brain.faceid.templates
    summary = templates.summary(operator_id) if templates else None
    if summary is None:
        return JSONResponse({"ok": False, "error": "not enrolled"}, status_code=404)
    return JSONResponse({"ok": True, "template": summary})


@app.delete("/api/faceid/template/{operator_id}")
def api_delete_template(operator_id: str, token: str = "") -> JSONResponse:
    """Erase one operator's face signature. Their profile card is left alone."""
    if _check_token(token) is None:
        return JSONResponse({"ok": False, "error": "supervisor PIN required"},
                            status_code=401)
    templates = brain.faceid.templates
    if templates is None or not templates.remove(operator_id):
        return JSONResponse({"ok": False, "error": "not enrolled"}, status_code=404)
    return JSONResponse({"ok": True, "removed": operator_id})


# ── enrolment ───────────────────────────────────────────────────────────
@app.post("/api/enroll/auth")
def api_enroll_auth(body: dict = Body(...)) -> JSONResponse:
    ok, msg = auth().verify(str(body.get("supervisor_id", "")), str(body.get("pin", "")))
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=401)
    return JSONResponse({"ok": True, "name": msg,
                         "token": _issue_token(str(body["supervisor_id"]), msg)})


@app.post("/api/enroll/start")
def api_enroll_start(body: dict = Body(...)) -> JSONResponse:
    global _enroll
    rec = _check_token(str(body.get("token", "")))
    if rec is None:
        return JSONResponse({"ok": False, "error": "supervisor PIN required"},
                            status_code=401)
    if not brain.faceid.embedder.available:
        return JSONResponse({"ok": False, "error":
                             brain.faceid.embedder.last_error or "face model unavailable"},
                            status_code=503)

    fields = {k: body.get(k) for k in
              ("id", "name", "role", "experience", "hearing", "color_vision",
               "language", "notes") if body.get(k) is not None}
    try:
        details = OperatorDetails(**fields)
    except TypeError as exc:
        return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=400)

    errors = details.validate()
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)

    with _enroll_lock:
        _enroll = EnrollmentSession(details, brain.faceid.embedder,
                                    supervisor=rec[0], machine_id=MACHINE_ID)
        status = _enroll.status("look at the camera")
    # Recognition must not fight the enrolment for the same face.
    if brain.faceid.identifier is not None:
        brain.faceid.identifier.reset()
    return JSONResponse({"ok": True, **status})


@app.get("/api/enroll/status")
def api_enroll_status() -> JSONResponse:
    with _enroll_lock:
        if _enroll is None:
            return JSONResponse({"active": False})
        return JSONResponse({"active": True, **_enroll.status(), **_enroll.review()})


@app.post("/api/enroll/skip")
def api_enroll_skip(body: dict = Body(...)) -> JSONResponse:
    if _check_token(str(body.get("token", ""))) is None:
        return JSONResponse({"ok": False, "error": "unauthorised"}, status_code=401)
    with _enroll_lock:
        if _enroll is None:
            return JSONResponse({"ok": False, "error": "no session"}, status_code=404)
        return JSONResponse({"ok": True, **_enroll.skip_pose()})


async def _collect_baseline(session: EnrollmentSession) -> None:
    """Gather ~25 s of ALERT feature windows from whichever tracker is live.

    Deliberately reuses the windows the fatigue pipeline already produces rather
    than opening the camera a second time — same frames, same features, and so
    the same numbers the live loop will compare against later.

    We pick the source by which tracker is actually PRODUCING windows, never by
    whether it holds a camera handle. When the operator enrols from the browser,
    OpenCV can still own /dev/video0 from the failed startup calibration while
    every real frame arrives over the WebSocket — polling the handle picks the
    dead one and yields nothing for 25 s.
    """
    sources = [t for t in (tracker, remote_tracker) if t is not None]
    if not sources:
        session.error = ("no camera pipeline is running — start the browser camera "
                         "on the dashboard, or attach a cab camera.")
        return

    chosen = None
    rows: list = []
    seen: set = set()
    started = time.time()

    while True:
        elapsed = time.time() - started
        if elapsed >= BASELINE_MAX_SECONDS:
            break
        if elapsed >= BASELINE_SECONDS and len(rows) >= MIN_BASELINE_WINDOWS:
            break

        for src in sources:
            feats = src.get_latest_features()
            if not feats:
                continue
            # The first few windows compute PERCLOS over a barely-filled 60 s
            # trailing buffer. Those numbers are not this operator's normal.
            if float(feats.get("is_warming_up", 0.0)) > 0.5:
                continue
            if chosen is None:
                chosen = src                    # lock onto the first live producer
                print(f"[Enrol] baseline source: {type(src).__name__}", flush=True)
            if src is not chosen:
                continue
            key = tuple(round(float(v), 6) for v in feats.values())
            if key not in seen:                 # the tracker repeats the last window
                seen.add(key)
                rows.append(dict(feats))
                session.baseline_progress = len(rows)
        await asyncio.sleep(0.25)

    if len(rows) < MIN_BASELINE_WINDOWS:
        session.error = (
            f"only {len(rows)} of {MIN_BASELINE_WINDOWS} baseline windows captured in "
            f"{int(time.time() - started)}s. Keep your face in view of the camera for "
            f"the whole countdown — the preview must show the face mesh."
            if chosen is not None else
            "no feature windows arrived from the camera. Check that the camera preview "
            "is live and your face is detected, then try again."
        )
        print(f"[Enrol] baseline failed: {session.error}", flush=True)
        return

    session.set_baseline(rows)


@app.post("/api/enroll/baseline")
async def api_enroll_baseline(body: dict = Body(...)) -> JSONResponse:
    if _check_token(str(body.get("token", ""))) is None:
        return JSONResponse({"ok": False, "error": "unauthorised"}, status_code=401)
    with _enroll_lock:
        session = _enroll
    if session is None:
        return JSONResponse({"ok": False, "error": "no session"}, status_code=404)
    if session.step.value != "baseline":
        return JSONResponse({"ok": False,
                             "error": f"not at the baseline step "
                                      f"(currently {session.step.value})"},
                            status_code=409)
    asyncio.create_task(_collect_baseline(session))
    return JSONResponse({"ok": True, "seconds": BASELINE_SECONDS,
                         "message": "Sit still, look ahead, stay awake."})


@app.post("/api/enroll/commit")
def api_enroll_commit(body: dict = Body(...)) -> JSONResponse:
    global _enroll
    if _check_token(str(body.get("token", ""))) is None:
        return JSONResponse({"ok": False, "error": "unauthorised"}, status_code=401)
    with _enroll_lock:
        session = _enroll
        if session is None:
            return JSONResponse({"ok": False, "error": "no session"}, status_code=404)

        result = session.commit(brain.store.profiles, brain.faceid.templates)
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        brain.store.save()
        _enroll = None

    # The new operator is the one sitting in the seat — make them active, which
    # also loads the baseline they just recorded.
    brain.switch_operator(session.details.id, source="enrolment")
    if brain.faceid.identifier is not None:
        brain.faceid.identifier.reset()
    return JSONResponse(result)


@app.post("/api/enroll/abort")
def api_enroll_abort(body: dict = Body(...)) -> JSONResponse:
    global _enroll
    if _check_token(str(body.get("token", ""))) is None:
        return JSONResponse({"ok": False, "error": "unauthorised"}, status_code=401)
    with _enroll_lock:
        if _enroll is not None:
            _enroll.abort()
        _enroll = None
    if brain.faceid.identifier is not None:
        brain.faceid.identifier.reset()
    return JSONResponse({"ok": True})


# ── static UIs ──────────────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(paths.DASHBOARD / "index.html"))


@app.get("/enroll")
def enroll_page() -> FileResponse:
    return FileResponse(str(paths.DASHBOARD / "enroll.html"))


@app.get("/phone")
def phone() -> FileResponse:
    return FileResponse(str(paths.PHONE / "index.html"))


app.mount("/dashboard", StaticFiles(directory=str(paths.DASHBOARD)), name="dashboard")
app.mount("/phone-static", StaticFiles(directory=str(paths.PHONE)), name="phone-static")


def main() -> None:
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
