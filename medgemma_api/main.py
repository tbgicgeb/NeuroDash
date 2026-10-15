"""
main.py — NeuroDash MedGemma Report API
========================================
FastAPI service that accepts MRI .npy uploads, extracts diagnostically rich
2-D slices from all modalities + segmentation, and streams a structured
radiology report from a locally-running MedGemma model.

Endpoints
─────────
POST   /api/report/generate         Upload .npy files → report_id + SSE stream URL
GET    /api/report/{rid}/stream      SSE: streams report tokens + metadata events
GET    /api/report/{rid}             Get completed report (after stream finishes)
GET    /api/report/{rid}/slices      Get the extracted slice images (base64 PNGs)
DELETE /api/report/{rid}             Free memory
GET    /health                       Liveness + backend probe
GET    /api/backend/status           Detailed backend health

Run
───
    cd medgemma_api
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Environment variables
─────────────────────
    MEDGEMMA_BACKEND   = ollama | transformers | mock   (default: ollama)
    MEDGEMMA_MODEL     = medgemma3:4b                   (default; Ollama model tag)
    OLLAMA_BASE_URL    = http://localhost:11434          (default)
    HF_MODEL_ID        = google/medgemma-4b-it          (transformers backend)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from slicer import extract_slices, SlicePackage
from report_generator import stream_report, check_backend_health

# ─── App ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
logger = logging.getLogger("report_api")

app = FastAPI(
    title="NeuroDash MedGemma Report API",
    version="1.0.0",
    description="LLM-powered radiology report generation from multi-modal MRI slices",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory report store ───────────────────────────────────────────────────

_REPORT_TTL = 3600   # 1 hour

class ReportStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def create(self, rid: str, pkg: SlicePackage) -> None:
        self._store[rid] = {
            "pkg":        pkg,
            "report_md":  "",         # filled as stream completes
            "status":     "pending",  # pending | streaming | done | error
            "error":      None,
            "created_at": time.time(),
        }

    def get(self, rid: str) -> dict:
        entry = self._store.get(rid)
        if not entry:
            raise HTTPException(404, f"Report {rid!r} not found or expired.")
        return entry

    def append_text(self, rid: str, text: str) -> None:
        self._store[rid]["report_md"] += text

    def set_status(self, rid: str, status: str, error: Optional[str] = None) -> None:
        self._store[rid]["status"] = status
        if error:
            self._store[rid]["error"] = error

    def delete(self, rid: str) -> bool:
        return self._store.pop(rid, None) is not None

    def evict_expired(self) -> None:
        now = time.time()
        stale = [rid for rid, e in self._store.items()
                 if now - e["created_at"] > _REPORT_TTL]
        for rid in stale:
            self._store.pop(rid, None)
        if stale:
            logger.info(f"Evicted {len(stale)} expired report(s)")


_reports = ReportStore()


@app.on_event("startup")
async def _start_eviction():
    async def _loop():
        while True:
            await asyncio.sleep(600)
            _reports.evict_expired()
    asyncio.create_task(_loop())


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _load_npy(upload: UploadFile) -> np.ndarray:
    raw = await upload.read()
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse .npy file '{upload.filename}': {exc}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    backend = await check_backend_health()
    return {
        "status": "ok",
        "reports_in_memory": len(_reports._store),
        "backend": backend,
    }


@app.get("/api/backend/status")
async def backend_status():
    """Detailed check of the MedGemma backend."""
    return await check_backend_health()


@app.post("/api/report/generate")
async def generate_report(
    data_file: UploadFile = File(..., description="4-channel MRI .npy (FLAIR, T1, T1ce, T2)"),
    seg_file:  Optional[UploadFile] = File(None, description="Segmentation label map .npy"),
):
    """
    Upload MRI .npy files → extract slices → return report_id.

    The actual report is generated via the SSE stream endpoint.
    This endpoint does the heavy slice extraction synchronously then returns
    a report_id you can poll/stream.

    Response
    ────────
    {
      "report_id":   "<uuid>",
      "stream_url":  "/api/report/<uuid>/stream",
      "n_images":    12,
      "meta":        { "centroid": ..., "labels": ..., ... }
    }
    """
    t0 = time.perf_counter()

    raw_data = await _load_npy(data_file)

    raw_seg: Optional[np.ndarray] = None
    if seg_file and seg_file.filename:
        raw_seg = await _load_npy(seg_file)

    # Extract slices (CPU-bound, but fast enough synchronously for typical MRI sizes)
    try:
        pkg = extract_slices(raw_data, raw_seg, n_offset_slices=1)
    except Exception as exc:
        logger.error(f"Slice extraction failed: {exc}", exc_info=True)
        raise HTTPException(500, f"Slice extraction error: {exc}")

    rid = str(uuid.uuid4())
    _reports.create(rid, pkg)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Report {rid[:8]}… created  "
        f"n_images={len(pkg.images)}  t={elapsed:.2f}s"
    )

    return {
        "report_id":  rid,
        "stream_url": f"/api/report/{rid}/stream",
        "n_images":   len(pkg.images),
        "meta":       pkg.meta,
    }


@app.get("/api/report/{report_id}/stream")
async def stream_report_sse(report_id: str):
    """
    Server-Sent Events stream for a generated report.

    Event types
    ───────────
    data: {"type": "token",    "text": "..."}       — incremental text token
    data: {"type": "progress", "section": "..."}    — section heading detected
    data: {"type": "done",     "report_md": "..."}  — final complete report
    data: {"type": "error",    "message": "..."}    — error
    """
    entry = _reports.get(report_id)
    pkg: SlicePackage = entry["pkg"]

    async def _event_gen() -> AsyncGenerator[str, None]:
        _reports.set_status(report_id, "streaming")
        buffer = ""

        # Known section headings to emit progress events
        section_markers = {
            "## Clinical Summary":        "clinical_summary",
            "## Tumour Characterisation": "tumour_characterisation",
            "## Modality":                "modality_findings",
            "## Segmentation":            "segmentation_analysis",
            "## Differential":            "differential_diagnosis",
            "## Impression":              "impression",
        }

        try:
            async for token in stream_report(pkg):
                buffer += token
                _reports.append_text(report_id, token)

                # Yield token event
                payload = json.dumps({"type": "token", "text": token})
                yield f"data: {payload}\n\n"

                # Check for section markers in the buffered text
                for marker, section_id in section_markers.items():
                    if marker in buffer:
                        prog_payload = json.dumps({
                            "type":    "progress",
                            "section": section_id,
                            "title":   marker.lstrip("# "),
                        })
                        yield f"data: {prog_payload}\n\n"
                        buffer = ""   # reset after match to avoid re-emitting

            # All done
            _reports.set_status(report_id, "done")
            done_payload = json.dumps({
                "type":      "done",
                "report_md": _reports.get(report_id)["report_md"],
                "meta":      pkg.meta,
            })
            yield f"data: {done_payload}\n\n"

        except Exception as exc:
            logger.error(f"Report generation error: {exc}", exc_info=True)
            _reports.set_status(report_id, "error", str(exc))
            err_payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":              "no-cache",
            "Connection":                 "keep-alive",
            "X-Accel-Buffering":          "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/report/{report_id}")
def get_report(report_id: str):
    """
    Retrieve a completed report.

    Response
    ────────
    {
      "report_id":  "...",
      "status":     "done" | "streaming" | "pending" | "error",
      "report_md":  "# Clinical Summary\n...",
      "meta":       { ... }
    }
    """
    entry = _reports.get(report_id)
    return JSONResponse({
        "report_id":  report_id,
        "status":     entry["status"],
        "report_md":  entry["report_md"],
        "meta":       entry["pkg"].meta,
        "error":      entry.get("error"),
    })


@app.get("/api/report/{report_id}/slices")
def get_slices(report_id: str):
    """
    Return the extracted slice images as base64 PNGs for inspection.

    Response
    ────────
    {
      "report_id": "...",
      "images": [
        { "label": "Axial Z=64 | 4-Modality Mosaic", "b64_png": "...", "description": "..." },
        ...
      ]
    }
    """
    entry = _reports.get(report_id)
    pkg: SlicePackage = entry["pkg"]
    return JSONResponse({
        "report_id": report_id,
        "images": [
            {
                "label":       img.label,
                "description": img.description,
                "b64_png":     img.b64_png,
            }
            for img in pkg.images
        ],
    })


@app.delete("/api/report/{report_id}")
def delete_report(report_id: str):
    """Free memory for a report session."""
    if not _reports.delete(report_id):
        raise HTTPException(404, "Report not found.")
    return {"deleted": report_id}
