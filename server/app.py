"""FastAPI worker for the lingbot-map 3D scan beta.

Receives a video, runs CPU inference, exports a colored point-cloud GLB.

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from server.pipeline import run_scan

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LINGBOT_SCANS_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="lingbot-map scan worker", version="0.1.0")

# Single CPU pipeline → serialize all jobs through one lock.
_pipeline_lock = threading.Lock()


def _scan_dir(scan_id: str) -> Path:
    return DATA_DIR / scan_id


def _meta_path(scan_id: str) -> Path:
    return _scan_dir(scan_id) / "meta.json"


def _read_meta(scan_id: str) -> Optional[dict]:
    p = _meta_path(scan_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _write_meta(scan_id: str, meta: dict) -> None:
    _meta_path(scan_id).write_text(json.dumps(meta, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process(scan_id: str) -> None:
    """Run inference for one scan. Serialized via _pipeline_lock."""
    with _pipeline_lock:
        meta = _read_meta(scan_id) or {}
        meta["status"] = "processing"
        meta["started_at"] = _now()
        _write_meta(scan_id, meta)
        try:
            video = _scan_dir(scan_id) / "input.mp4"
            glb = _scan_dir(scan_id) / "scan.glb"
            progress = _scan_dir(scan_id) / "progress.json"
            result = run_scan(video, glb, progress_path=progress)
            meta["status"] = "done"
            meta["finished_at"] = _now()
            meta["result"] = result
        except Exception as exc:
            meta["status"] = "failed"
            meta["finished_at"] = _now()
            meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["traceback"] = traceback.format_exc()
        _write_meta(scan_id, meta)


@app.post("/scans")
async def create_scan(
    background: BackgroundTasks,
    video: UploadFile = File(...),
) -> JSONResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    scan_id = uuid.uuid4().hex[:12]
    scan_dir = _scan_dir(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    target = scan_dir / "input.mp4"
    with target.open("wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    meta = {
        "id": scan_id,
        "status": "queued",
        "created_at": _now(),
        "input_bytes": target.stat().st_size,
        "input_filename": video.filename,
    }
    _write_meta(scan_id, meta)
    background.add_task(_process, scan_id)
    return JSONResponse(meta, status_code=202)


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str) -> JSONResponse:
    meta = _read_meta(scan_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="scan not found")
    meta = dict(meta)
    if meta.get("status") == "done":
        # Prefer MP4 (official rgbd_render output); GLB only exists on legacy scans.
        scan_dir = _scan_dir(scan_id)
        if (scan_dir / "scan.mp4").exists():
            meta["mp4_url"] = f"/scans/{scan_id}/scan.mp4"
        if (scan_dir / "scan.glb").exists():
            meta["glb_url"] = f"/scans/{scan_id}/scan.glb"
    if meta.get("status") == "processing":
        progress_path = _scan_dir(scan_id) / "progress.json"
        if progress_path.exists():
            try:
                meta["progress"] = json.loads(progress_path.read_text())
            except Exception:
                pass
    return JSONResponse(meta)


@app.get("/scans/{scan_id}/scan.glb")
def get_glb(scan_id: str) -> FileResponse:
    glb = _scan_dir(scan_id) / "scan.glb"
    if not glb.exists():
        raise HTTPException(status_code=404, detail="glb not ready")
    return FileResponse(glb, media_type="model/gltf-binary", filename="scan.glb")


@app.get("/scans/{scan_id}/scan.mp4")
def get_mp4(scan_id: str) -> FileResponse:
    mp4 = _scan_dir(scan_id) / "scan.mp4"
    if not mp4.exists():
        raise HTTPException(status_code=404, detail="mp4 not ready")
    return FileResponse(mp4, media_type="video/mp4", filename="scan.mp4")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "queue_locked": _pipeline_lock.locked()}
