"""Gaussian Splatting pipeline: video → COLMAP → splatfacto → MP4 flythrough.

End-to-end:
  1. ffmpeg extract frames at fps=5
  2. ns-process-data images (runs COLMAP feature matching + SfM)
  3. ns-train splatfacto (5-15 min on a 4090/L40S)
  4. ns-render interpolate → MP4 of the trained gaussians
  5. ns-export gaussian-splat → .ply of the trained gaussians (best-effort)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from server import progress as progress_mod


def _stream_run(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    on_line: Optional[Callable[[str], None]] = None,
) -> None:
    """Run cmd; stream stdout to log + optional callback; raise on non-zero."""
    if log_path:
        with log_path.open("ab") as logf:
            logf.write(f"\n$ {' '.join(cmd)}\n".encode())
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if log_path:
            with log_path.open("a") as logf:
                logf.write(line)
        if on_line:
            try:
                on_line(line)
            except Exception:
                pass
    proc.wait()
    if proc.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
        raise RuntimeError(
            f"`{cmd[0]}` exited {proc.returncode}\n--- log tail ---\n{tail}"
        )


def _find_config_yml(train_root: Path) -> Path:
    candidates = sorted(
        train_root.glob("outputs/**/config.yml"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise RuntimeError(f"no nerfstudio config.yml under {train_root}/outputs/")
    return candidates[-1]


_STEP_RE = re.compile(r"Step\s+(\d+)\s*/\s*(\d+)")


def run_scan(
    video_path: Path,
    output_mp4: Path,
    *,
    fps: int = 5,
    max_iterations: int = 7000,
    progress_path: Optional[Path] = None,
) -> dict:
    t0 = time.time()
    scan_dir = output_mp4.parent
    scan_dir.mkdir(parents=True, exist_ok=True)
    log_path = scan_dir / "gsplat.log"
    progress_mod.set_path(progress_path)

    work_dir = scan_dir / "ns_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = work_dir / "raw_frames"
    raw_frames.mkdir(parents=True, exist_ok=True)
    processed = work_dir / "processed"

    try:
        progress_mod.set_phase("extracting")
        _stream_run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
             str(raw_frames / "frame_%05d.jpg")],
            log_path=log_path,
        )
        n_raw = len(list(raw_frames.glob("frame_*.jpg")))
        if n_raw < 10:
            raise ValueError(f"Need >=10 frames, got {n_raw}")

        progress_mod.set_phase("colmap")
        _stream_run(
            ["ns-process-data", "images",
             "--data", str(raw_frames),
             "--output-dir", str(processed),
             "--matching-method", "sequential",
             "--num-downscales", "0",
             "--verbose"],
            log_path=log_path,
        )

        progress_mod.set_phase("training", current=0, total=max_iterations)

        def _on_train_line(line: str) -> None:
            m = _STEP_RE.search(line)
            if m:
                progress_mod.set_phase(
                    "training", current=int(m.group(1)), total=int(m.group(2))
                )

        train_outputs = work_dir / "outputs"
        train_outputs.mkdir(parents=True, exist_ok=True)
        _stream_run(
            ["ns-train", "splatfacto",
             "--data", str(processed),
             "--output-dir", str(train_outputs),
             "--max-num-iterations", str(max_iterations),
             "--viewer.quit-on-train-completion", "True",
             "--vis", "tensorboard"],
            cwd=work_dir,
            log_path=log_path,
            on_line=_on_train_line,
        )

        progress_mod.set_phase("rendering")
        config_yml = _find_config_yml(work_dir)
        _stream_run(
            ["ns-render", "interpolate",
             "--load-config", str(config_yml),
             "--output-path", str(output_mp4),
             "--frame-rate", "30",
             "--interpolation-steps", "30"],
            log_path=log_path,
        )

        ply_path = scan_dir / "scan.ply"
        try:
            _stream_run(
                ["ns-export", "gaussian-splat",
                 "--load-config", str(config_yml),
                 "--output-dir", str(scan_dir)],
                log_path=log_path,
            )
            default_ply = scan_dir / "splat.ply"
            if default_ply.exists():
                default_ply.rename(ply_path)
        except Exception:
            ply_path = None

        return {
            "frames": n_raw,
            "iterations": max_iterations,
            "total_seconds": round(time.time() - t0, 1),
            "mp4_bytes": output_mp4.stat().st_size if output_mp4.exists() else 0,
            "ply_bytes": ply_path.stat().st_size if ply_path and ply_path.exists() else 0,
        }
    finally:
        progress_mod.set_path(None)
        shutil.rmtree(raw_frames, ignore_errors=True)
