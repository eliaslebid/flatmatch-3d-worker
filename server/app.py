"""FastAPI worker for the Modula 3D scan beta (Gaussian Splatting variant).

Receives a video, runs splatfacto training, renders an MP4 flythrough.

Run:
    setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > worker.log 2>&1 < /dev/null
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
DATA_DIR = Path(os.environ.get("SCAN_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="modula 3d scan worker (gsplat)", version="0.2.0")

_pipeline_lock = threading.Lock()


def _scan_dir(scan_id: str) -> Path:
    return DATA_DIR / scan_id


def _meta_path(scan_id: str) -> Path:
    return _scan_dir(scan_id) / "meta.json"


def _read_meta(scan_id: str) -> Optional[dict]:
    p = _meta_path(scan_id)
    return json.loads(p.read_text()) if p.exists() else None


def _write_meta(scan_id: str, meta: dict) -> None:
    _meta_path(scan_id).write_text(json.dumps(meta, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process(scan_id: str) -> None:
    with _pipeline_lock:
        meta = _read_meta(scan_id) or {}
        meta["status"] = "processing"
        meta["started_at"] = _now()
        _write_meta(scan_id, meta)
        try:
            video = _scan_dir(scan_id) / "input.mp4"
            mp4 = _scan_dir(scan_id) / "scan.mp4"
            progress = _scan_dir(scan_id) / "progress.json"
            result = run_scan(video, mp4, progress_path=progress)
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
async def create_scan(background: BackgroundTasks, video: UploadFile = File(...)) -> JSONResponse:
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


@app.get("/scans")
def list_scans() -> JSONResponse:
    """List every scan, newest first. Includes status, timing, and result urls."""
    items = []
    for d in sorted(
        (p for p in DATA_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            m = json.loads((d / "meta.json").read_text())
        except Exception:
            continue
        m = dict(m)
        sid = m["id"]
        if m.get("status") == "done":
            if (d / "scan.compressed.ply").exists() or (d / "scan.ply").exists():
                m["ply_url"] = f"/scans/{sid}/scan.ply"
            if (d / "scan.mp4").exists():
                m["mp4_url"] = f"/scans/{sid}/scan.mp4"
        items.append(m)
    return JSONResponse(items)


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str) -> JSONResponse:
    meta = _read_meta(scan_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="scan not found")
    meta = dict(meta)
    scan_dir = _scan_dir(scan_id)
    if meta.get("status") == "done":
        if (scan_dir / "scan.spz").exists():
            meta["spz_url"] = f"/scans/{scan_id}/scan.spz"
        if (scan_dir / "scan.ply").exists():
            meta["ply_url"] = f"/scans/{scan_id}/scan.ply"
        if (scan_dir / "scan.mp4").exists():
            meta["mp4_url"] = f"/scans/{scan_id}/scan.mp4"
    if meta.get("status") == "processing":
        progress_path = scan_dir / "progress.json"
        if progress_path.exists():
            try:
                meta["progress"] = json.loads(progress_path.read_text())
            except Exception:
                pass
    return JSONResponse(meta)


@app.get("/scans/{scan_id}/scan.mp4")
def get_mp4(scan_id: str) -> FileResponse:
    mp4 = _scan_dir(scan_id) / "scan.mp4"
    if not mp4.exists():
        raise HTTPException(status_code=404, detail="mp4 not ready")
    return FileResponse(mp4, media_type="video/mp4", filename="scan.mp4")


@app.get("/scans/{scan_id}/scan.ply")
def get_ply(scan_id: str) -> FileResponse:
    ply = _scan_dir(scan_id) / "scan.ply"
    if not ply.exists():
        raise HTTPException(status_code=404, detail="ply not ready")
    return FileResponse(ply, media_type="application/octet-stream", filename="scan.ply")


@app.get("/scans/{scan_id}/scan.spz")
def get_spz(scan_id: str) -> FileResponse:
    spz = _scan_dir(scan_id) / "scan.spz"
    if not spz.exists():
        raise HTTPException(status_code=404, detail="spz not ready")
    return FileResponse(spz, media_type="application/octet-stream", filename="scan.spz")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "queue_locked": _pipeline_lock.locked()}
