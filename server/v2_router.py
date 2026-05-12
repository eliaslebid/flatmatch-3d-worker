"""v2 endpoints — Gaussian Splatting from iPhone guided-capture.

Contract (vs the v1 `/scans` video-input route): the mobile client runs an
ARKit-driven capture session, blur-rejects frames in real time, and uploads
a zip containing `images/*.jpg` + `transforms.json` (nerfstudio format with
camera intrinsics + per-frame poses). The worker skips frame extraction /
SfM entirely and feeds straight into splatfacto training.

This module currently ships only the *stub*: it accepts the upload, walks
through realistic status transitions, and emits a synthesized 8-gaussian
sample PLY so the mobile + WebView plumbing can be verified end-to-end
before the real splatfacto pipeline lands.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from server.sample_ply import write_sample_ply

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SCAN_DATA_DIR", ROOT / "data")) / "v2"
DATA_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/v2", tags=["v2"])

_lock = threading.Lock()


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


_STUB_STEPS = [
    ("unpack", 10),
    ("validate_poses", 25),
    ("train_splatfacto", 70),
    ("render_flythrough", 90),
    ("export", 100),
]


def _process_stub(scan_id: str) -> None:
    with _lock:
        meta = _read_meta(scan_id) or {}
        meta["status"] = "processing"
        meta["started_at"] = _now()
        _write_meta(scan_id, meta)
        try:
            for step, pct in _STUB_STEPS:
                meta["progress"] = {"step": step, "pct": pct}
                _write_meta(scan_id, meta)
                time.sleep(1.5)
            write_sample_ply(_scan_dir(scan_id) / "scene.ply")
            meta["status"] = "done"
            meta["finished_at"] = _now()
            meta["stub"] = True
        except Exception as exc:
            meta["status"] = "failed"
            meta["finished_at"] = _now()
            meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["traceback"] = traceback.format_exc()
        _write_meta(scan_id, meta)


@router.post("/scan")
async def create_scan(
    background: BackgroundTasks,
    archive: UploadFile = File(
        ..., description="zip containing images/ and transforms.json"
    ),
) -> JSONResponse:
    if not archive.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    scan_id = uuid.uuid4().hex[:12]
    scan_dir = _scan_dir(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    target = scan_dir / "input.zip"
    with target.open("wb") as f:
        while chunk := await archive.read(1024 * 1024):
            f.write(chunk)
    meta = {
        "id": scan_id,
        "engine": "v2",
        "status": "queued",
        "created_at": _now(),
        "input_bytes": target.stat().st_size,
        "input_filename": archive.filename,
    }
    _write_meta(scan_id, meta)
    background.add_task(_process_stub, scan_id)
    return JSONResponse(meta, status_code=202)


@router.get("/scan")
def list_scans() -> JSONResponse:
    items: list[dict] = []
    if not DATA_DIR.exists():
        return JSONResponse(items)
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
            if (d / "scene.ply").exists():
                m["ply_url"] = f"/v2/scan/{sid}/scene.ply"
            if (d / "scene.mp4").exists():
                m["mp4_url"] = f"/v2/scan/{sid}/scene.mp4"
        items.append(m)
    return JSONResponse(items)


@router.get("/scan/{scan_id}")
def get_scan(scan_id: str) -> JSONResponse:
    meta = _read_meta(scan_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="scan not found")
    meta = dict(meta)
    sd = _scan_dir(scan_id)
    if meta.get("status") == "done":
        if (sd / "scene.ply").exists():
            meta["ply_url"] = f"/v2/scan/{scan_id}/scene.ply"
        if (sd / "scene.mp4").exists():
            meta["mp4_url"] = f"/v2/scan/{scan_id}/scene.mp4"
    return JSONResponse(meta)


@router.get("/scan/{scan_id}/scene.ply")
def get_ply(scan_id: str) -> FileResponse:
    p = _scan_dir(scan_id) / "scene.ply"
    if not p.exists():
        raise HTTPException(status_code=404, detail="ply not ready")
    return FileResponse(
        p, media_type="application/octet-stream", filename="scene.ply"
    )


@router.get("/scan/{scan_id}/scene.mp4")
def get_mp4(scan_id: str) -> FileResponse:
    p = _scan_dir(scan_id) / "scene.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="mp4 not ready")
    return FileResponse(p, media_type="video/mp4", filename="scene.mp4")
