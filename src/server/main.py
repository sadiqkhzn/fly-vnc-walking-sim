"""FastAPI + WebSocket backend for the live VNC viewer.

Endpoints
  GET  /                   serves viewer/index.html (static mount)
  GET  /neurons            JSON: [{body_id, idx, xyz, role, side}, ...]
  GET  /meta               JSON: dataset, counts, command-neuron indices
  WS   /spikes             binary frames: uint32 t_ms, uint16 n, uint32[n] indices
  POST /drive              {"command": "mdn" | "dnp09" | "mdn+dnp09" | "silence"}

The simulator runs in a background thread with its own LIF loop at 1 ms LIF
tick. It pushes a frame every PUBLISH_EVERY_MS so the WebSocket doesn't
flood the browser.
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.sim.lif import LIFBrain, LIFParams

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "vnc_graph.pt"
NEURONS_PARQUET = REPO_ROOT / "data" / "vnc_neurons__male-cns_v1.0__w3.parquet"
ASSIGN_PARQUET = REPO_ROOT / "data" / "motor_leg_assignment.parquet"
VIEWER_DIR = REPO_ROOT / "viewer"

PUBLISH_EVERY_MS = 20          # publish a spike frame every N LIF ticks (50 Hz)
STEP_INTERVAL_S = 0.001        # target wall interval per LIF tick (will slip if sim slower)
DRIVE_LEVEL = 2.0              # suprathreshold


# ------------------------------------------------------------------------
# Shared sim state
# ------------------------------------------------------------------------

class SimState:
    def __init__(self):
        self.brain: LIFBrain | None = None
        self.N: int = 0
        self.body_ids: list[int] = []
        self.neuron_meta: list[dict[str, Any]] = []  # per-neuron metadata (role, side, xyz)
        self.command_idx: dict[str, list[int]] = {}
        self.motor_idx: set[int] = set()
        self.descending_idx: set[int] = set()
        self.drive_vec: torch.Tensor | None = None
        self.drive_name: str = "silence"
        self.t_ms: int = 0
        self.lock = threading.Lock()
        self.loop_task: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.wall_hz: float = 0.0

    def load(self):
        print("[sim] loading brain + metadata")
        self.brain = LIFBrain.from_bundle(GRAPH_PATH, params=LIFParams())
        bundle = self.brain.bundle
        self.N = bundle["meta"]["n_neurons"]
        self.body_ids = bundle["body_ids"]
        self.command_idx = bundle["command_idx"]
        self.motor_idx = {i for ids in bundle["motor_idx_by_type"].values() for i in ids}
        self.descending_idx = set(bundle["descending_idx"])
        self.drive_vec = torch.zeros(self.N)

        # Build neuron metadata list for /neurons
        neurons = pd.read_parquet(NEURONS_PARQUET)
        assign = pd.read_parquet(ASSIGN_PARQUET) if ASSIGN_PARQUET.exists() else None
        body_to_leg = (
            dict(zip(assign.index.astype(int), assign["leg"])) if assign is not None else {}
        )
        # Reverse index for command neurons
        cmd_of = {}
        for name, ids in self.command_idx.items():
            for idx in ids:
                cmd_of[idx] = name

        meta = []
        for i, bid in enumerate(self.body_ids):
            row = neurons.loc[bid]
            role = (
                "command"
                if i in cmd_of
                else "motor"
                if i in self.motor_idx
                else "descending"
                if i in self.descending_idx
                else "local"
            )
            meta.append({
                "idx": i,
                "body_id": int(bid),
                "role": role,
                "command": cmd_of.get(i),
                "leg": body_to_leg.get(int(bid)),
                "side": str(row["soma_side"]) if pd.notna(row["soma_side"]) else None,
                "x": float(row["x"]) if pd.notna(row["x"]) else None,
                "y": float(row["y"]) if pd.notna(row["y"]) else None,
                "z": float(row["z"]) if pd.notna(row["z"]) else None,
            })
        self.neuron_meta = meta
        print(f"[sim] loaded {self.N:,} neurons   "
              f"with coords: {sum(1 for m in meta if m['x'] is not None):,}   "
              f"commands: {sum(1 for m in meta if m['role']=='command')}   "
              f"motor: {sum(1 for m in meta if m['role']=='motor')}   "
              f"descending: {sum(1 for m in meta if m['role']=='descending')}")

    def set_drive(self, name: str):
        name = name.lower()
        valid = {"mdn", "dnp09", "mdn+dnp09", "silence"}
        if name not in valid:
            raise HTTPException(400, f"drive must be one of {sorted(valid)}")
        with self.lock:
            vec = torch.zeros(self.N)
            if "mdn" in name:
                for idx in self.command_idx.get("MDN", []):
                    vec[idx] = DRIVE_LEVEL
            if "dnp09" in name:
                for idx in self.command_idx.get("DNp09", []):
                    vec[idx] = DRIVE_LEVEL
            self.drive_vec = vec
            self.drive_name = name
        print(f"[sim] drive -> {name}")

    def publish(self, t_ms: int, active_idx: list[int]):
        # Binary frame: uint32 t_ms, uint16 n, uint32[n] indices
        n = len(active_idx)
        buf = bytearray(4 + 2 + 4 * n)
        struct.pack_into("<IH", buf, 0, t_ms, n)
        if n:
            arr = np.asarray(active_idx, dtype=np.uint32)
            buf[6:] = arr.tobytes()
        for q in list(self.subscribers):
            if q.qsize() < 10:
                try:
                    q.put_nowait(bytes(buf))
                except asyncio.QueueFull:
                    pass


state = SimState()


async def sim_loop():
    """Background: step the brain forever. Publishes every PUBLISH_EVERY_MS ms."""
    assert state.brain is not None
    frame_buffer: list[int] = []
    tick_count = 0
    bench_t0 = time.perf_counter()
    bench_ticks = 0

    while True:
        tick_start = time.perf_counter()
        with state.lock:
            drive = state.drive_vec
        spikes = state.brain.step(drive)
        state.t_ms += 1
        tick_count += 1
        bench_ticks += 1

        # Accumulate active-neuron indices across the publish window
        active = torch.nonzero(spikes, as_tuple=False).squeeze(-1).tolist()
        frame_buffer.extend(active)

        if tick_count % PUBLISH_EVERY_MS == 0:
            # Deduplicate indices in this window (if the same cell fired twice)
            if frame_buffer:
                uniq = sorted(set(frame_buffer))
            else:
                uniq = []
            state.publish(state.t_ms, uniq)
            frame_buffer.clear()

        # Track real tick rate
        if bench_ticks >= 200:
            dt = time.perf_counter() - bench_t0
            state.wall_hz = bench_ticks / dt
            bench_t0 = time.perf_counter()
            bench_ticks = 0

        elapsed = time.perf_counter() - tick_start
        slack = STEP_INTERVAL_S - elapsed
        if slack > 0:
            await asyncio.sleep(slack)
        else:
            await asyncio.sleep(0)  # yield


# ------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------

app = FastAPI(title="Fly VNC live")


@app.on_event("startup")
async def startup():
    state.load()
    state.loop_task = asyncio.create_task(sim_loop())


@app.get("/meta")
async def meta():
    return {
        "dataset": "male-cns:v1.0",
        "n_neurons": state.N,
        "n_edges": state.brain.W.values().numel() if state.brain else 0,
        "sim_time_ms": state.t_ms,
        "drive": state.drive_name,
        "wall_tick_hz": round(state.wall_hz, 1),
        "command_idx": {k: v for k, v in state.command_idx.items()},
    }


@app.get("/neurons")
async def neurons():
    return JSONResponse(state.neuron_meta)


class DriveBody(BaseModel):
    command: str


@app.post("/drive")
async def drive(body: DriveBody):
    state.set_drive(body.command)
    return {"ok": True, "drive": state.drive_name}


@app.post("/reset")
async def reset():
    if state.brain:
        state.brain.reset()
        state.t_ms = 0
    return {"ok": True}


@app.websocket("/spikes")
async def spikes_ws(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    state.subscribers.add(q)
    try:
        while True:
            frame = await q.get()
            await ws.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(q)


# Static viewer last so API routes take priority.
app.mount("/", StaticFiles(directory=str(VIEWER_DIR), html=True), name="viewer")


def serve():
    import uvicorn
    uvicorn.run("src.server.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    serve()
