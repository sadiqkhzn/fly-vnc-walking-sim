"""FastAPI + WebSocket live spike stream for the Three.js viewer.

Protocol:
  GET  /neurons           → JSON of {body_id, xyz, type, side, region}
  GET  /synapses          → JSON of {pre, post, weight}  (filtered server-side if too large)
  WS   /spikes            → binary frames, 10 ms cadence:
                             uint32 t_ms, uint16 n_active, uint32[n_active] indices
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Fly VNC live stream")


@app.get("/neurons")
async def neurons():
    # Served from a pre-computed JSON in data/ once the connectome is extracted.
    return {"status": "not implemented", "hint": "run scripts/01_fetch_connectome.py first"}


@app.get("/synapses")
async def synapses():
    return {"status": "not implemented"}


@app.websocket("/spikes")
async def spikes(ws: WebSocket):
    await ws.accept()
    try:
        # Placeholder loop: emit a heartbeat every second.
        # Real impl: subscribe to the sim runner's on_spikes callback and forward.
        while True:
            await ws.send_json({"heartbeat": True})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


def serve():
    import uvicorn

    uvicorn.run("src.server.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    serve()
