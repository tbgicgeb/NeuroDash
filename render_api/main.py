"""
main.py — NeuroDash Render API
==============================
FastAPI application serving 3D Plotly volume renders for MRI .npy files.

Endpoints
─────────
POST   /api/session              Upload data.npy (+ optional seg.npy) → session_id
GET    /api/render/{sid}/single/{view}   Single view on demand → { figure: "<json>" }
GET    /api/render/{sid}/stream  SSE stream: all views rendered in parallel
DELETE /api/session/{sid}        Free session memory
GET    /health                   Liveness probe
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from renderer import (
    SessionVolume,
    all_view_types,
    preprocess_session,
    render_view,
)

# ─── App setup ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="NeuroDash Render API",
    version="1.0.0",
    description="3D volume rendering for multi-modal MRI .npy files",
)

# Allow the Vite dev server (any origin in dev; lock down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CPU-bound render tasks run in a separate process pool to avoid blocking uvicorn
# Adjust max_workers based on your CPU count; 4 is a safe default
_executor = ProcessPoolExecutor(max_workers=4)

# ─── In-memory session store ──────────────────────────────────────────────────
# For production, replace with Redis (value-serialize SessionVolume → store
# downsampled arrays as bytes, or keep in-proc behind a session proxy).

_SESSION_TTL_SECONDS = 3600  # 1 hour

class SessionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def put(self, session_id: str, sv: SessionVolume) -> None:
        self._store[session_id] = {"sv": sv, "created_at": time.time()}
        logger.info(f"Session {session_id[:8]}… stored  (total={len(self._store)})")

    def get(self, session_id: str) -> SessionVolume:
        entry = self._store.get(session_id)
        if entry is None:
            raise HTTPException(404, f"Session {session_id!r} not found or expired.")
        return entry["sv"]

    def delete(self, session_id: str) -> bool:
        entry = self._store.pop(session_id, None)
        if entry:
            entry["sv"].free()
            logger.info(f"Session {session_id[:8]}… deleted")
            return True
        return False

    def evict_expired(self) -> None:
        now = time.time()
        stale = [sid for sid, e in self._store.items()
                 if now - e["created_at"] > _SESSION_TTL_SECONDS]
        for sid in stale:
            self.delete(sid)
        if stale:
            logger.info(f"Evicted {len(stale)} expired session(s)")


_sessions = SessionStore()


# ─── Background TTL eviction ──────────────────────────────────────────────────

@app.on_event("startup")
async def _start_eviction_loop() -> None:
    async def _loop():
        while True:
            await asyncio.sleep(600)   # check every 10 min
            _sessions.evict_expired()
    asyncio.create_task(_loop())


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _load_npy(upload: UploadFile) -> np.ndarray:
    """Read an uploaded .npy file into a numpy array (async I/O)."""
    raw = await upload.read()
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse .npy file {upload.filename!r}: {exc}")


def _render_sync(sv: SessionVolume, view_type: str) -> str:
    """
    Thin wrapper called inside ProcessPoolExecutor.
    Must be a module-level function (picklable).
    NOTE: SessionVolume is passed by reference inside the same process;
    for true multi-process isolation pass raw arrays instead (see comment below).
    """
    return render_view(sv, view_type)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions._store)}


@app.post("/api/session")
async def create_session(
    data_file: UploadFile = File(..., description="4-channel MRI .npy (FLAIR, T1, T1ce, T2)"),
    seg_file: Optional[UploadFile] = File(None, description="Segmentation label map .npy (optional)"),
):
    """
    Upload .npy files, preprocess (downsample + normalise), and return a session ID
    along with the list of available view types ready to render.

    The preprocessing runs once so every subsequent render is fast (~50–300 ms each).
    """
    t0 = time.perf_counter()

    raw_data = await _load_npy(data_file)
    raw_seg: Optional[np.ndarray] = None
    if seg_file and seg_file.filename:
        raw_seg = await _load_npy(seg_file)

    # Preprocess in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    sv: SessionVolume = await loop.run_in_executor(
        _executor,
        preprocess_session,
        raw_data,
        raw_seg,
    )

    session_id = str(uuid.uuid4())
    _sessions.put(session_id, sv)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Session {session_id[:8]}… created  "
        f"shape_ds={sv.shape_ds}  has_seg={sv.has_seg}  t={elapsed:.2f}s"
    )

    return {
        "session_id": session_id,
        "shape_downsampled": list(sv.shape_ds),
        "modalities": list(sv.data_ds.keys()),
        "has_segmentation": sv.has_seg,
        "available_views": all_view_types(sv.has_seg),
    }


@app.get("/api/render/{session_id}/single/{view_type}")
async def render_single(session_id: str, view_type: str):
    """
    Render one view on demand and return its Plotly JSON.
    Use this for interactive on-click rendering of a specific combination.

    Response: { "view": "T1_FLAIR", "figure": "<plotly json string>" }
    """
    sv = _sessions.get(session_id)

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    figure_json: str = await loop.run_in_executor(
        _executor, render_view, sv, view_type
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"single render  view={view_type}  t={elapsed:.2f}s")

    return JSONResponse({"view": view_type, "figure": figure_json})


@app.get("/api/render/{session_id}/stream")
async def stream_all_renders(session_id: str):
    """
    SSE endpoint that streams all view renders in parallel.

    The executor renders all views concurrently; each result is sent to the
    client as soon as it's ready (out-of-order delivery is fine — the client
    stores them by name).

    Event format:
        data: {"view": "<view_type>", "figure": "<plotly_json>"}\\n\\n
    Terminal event:
        data: {"done": true, "total": <n>}\\n\\n
    Error event:
        data: {"error": "<view_type>", "message": "<msg>"}\\n\\n
    """
    sv = _sessions.get(session_id)
    views = all_view_types(sv.has_seg)

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()

        # Wrap each render in a tagged coroutine so we know which view it is
        async def render_tagged(view: str) -> tuple[str, str]:
            result = await loop.run_in_executor(_executor, render_view, sv, view)
            return view, result

        tasks = [asyncio.create_task(render_tagged(v)) for v in views]
        completed = 0

        # Yield results AS they complete (fastest views arrive first)
        for coro in asyncio.as_completed(tasks):
            try:
                view, figure_json = await coro
                completed += 1
                logger.info(f"stream render  view={view}  [{completed}/{len(views)}]")
                payload = json.dumps({"view": view, "figure": figure_json})
                yield f"data: {payload}\n\n"
            except Exception as exc:
                view_name = "unknown"
                payload = json.dumps({"error": view_name, "message": str(exc)})
                yield f"data: {payload}\n\n"
                logger.error(f"Render error: {exc}", exc_info=True)

        yield f"data: {json.dumps({'done': True, 'total': len(views)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",    # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Free session memory when the user navigates away."""
    deleted = _sessions.delete(session_id)
    if not deleted:
        raise HTTPException(404, "Session not found.")
    return {"deleted": session_id}