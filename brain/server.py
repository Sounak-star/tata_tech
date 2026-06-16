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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import paths
from .demo_source import DemoSource
from .pipeline import Brain

TICK_HZ = 6.0  # readable playback rate for the demo (raise toward 10 for "live")

app = FastAPI(title="SAARTHI — SmartCabin AI Copilot")

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
        if signal.get("_switch_operator"):
            brain.switch_operator(signal["_switch_operator"])
        frame = brain.tick(signal)
        frame["phase"] = signal.get("phase")
        await hub.broadcast(frame)
        await asyncio.sleep(1.0 / TICK_HZ)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_loop())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive / client pings
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


@app.get("/api/timeline")
def api_timeline() -> JSONResponse:
    return JSONResponse(brain.log.recent(30))


@app.get("/api/health")
def api_health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "fatigue_backend": brain.fatigue.backend,
        "person_backend": brain.persons.backend,
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

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
