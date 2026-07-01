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
import json
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import paths
from .demo_source import DemoSource, HybridLiveSource
from .pipeline import Brain
from .live_camera import BackgroundCameraTracker, RemoteCameraTracker, HAS_MEDIAPIPE
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
    while True:
        signal = source.step()

        # If remote tracker has features (browser cam), inject them into the signal
        if remote_tracker and remote_tracker.is_available:
            remote_feats = remote_tracker.get_latest_features()
            if remote_feats is not None:
                signal["features"] = remote_feats
                signal["drowsiness"] = 0.0  # real features override demo drowsiness

        frame = brain.tick(signal)
        frame["phase"] = signal.get("phase")
        # Keep the camera preview overlay in sync with the latest fatigue decision.
        if HAS_MEDIAPIPE:
            tracker.set_fatigue_info(frame["fatigue"])
        if remote_tracker and remote_tracker.is_available:
            remote_tracker.set_fatigue_info(frame["fatigue"])
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
    return JSONResponse(brain.store.profiles)


@app.post("/api/operator/{operator_id}")
def api_switch(operator_id: str) -> JSONResponse:
    try:
        prof = brain.switch_operator(operator_id)
        return JSONResponse({"ok": True, "operator": prof})
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown operator"}, status_code=404)

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
        "tick_hz": TICK_HZ,
    })


# ── static UIs ──────────────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(paths.DASHBOARD / "index.html"))


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
