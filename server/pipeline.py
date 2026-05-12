"""Gaussian Splatting pipeline — Stack A (production v1).

Closely mirrors the recommended Modula 3D pipeline (May 2026):
  1. ffmpeg extract frames at fps=2 with Laplacian-blur rejection filter
  2. hloc (SuperPoint + SuperGlue) feature extraction & matching
  3. GLOMAP global SfM (drops in a COLMAP-format sparse model)
  4. ns-train splatfacto-mcmc 30k iters with scale reg + alpha cull
  5. ns-export gaussian-splat → .ply
  6. splat-transform → .sog (Spatially Ordered Gaussians, ~15-20× smaller)
  7. ns-render interpolate → preview MP4 (fallback for clients without WebGL2)

Returns mp4 + sog + ply paths in meta.
"""

from __future__ import annotations

import os
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
    env: Optional[dict] = None,
) -> None:
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
        env=env,
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


def _hloc_glomap_sfm(
    frames_dir: Path,
    out_dir: Path,
    log_path: Path,
) -> None:
    """SuperPoint+SuperGlue feature extraction → COLMAP DB → GLOMAP mapper.

    hloc handles feature extraction + matching using SuperPoint+SuperGlue (much
    better than COLMAP's SIFT defaults on textureless indoor scenes). We then
    feed the resulting matches to GLOMAP for global SfM.
    """
    from hloc import extract_features, match_features, reconstruction

    out_dir.mkdir(parents=True, exist_ok=True)
    sfm_pairs = out_dir / "pairs.txt"
    feature_path = out_dir / "features.h5"
    match_path = out_dir / "matches.h5"
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    # 1. SuperPoint feature extraction
    extract_features.main(feature_conf, frames_dir, feature_path=feature_path)

    # 2. Sequential pair generation (each frame paired with next N + loop closure).
    images = sorted(p.name for p in frames_dir.glob("*.jpg"))
    window = 10
    pairs: list[tuple[str, str]] = []
    n = len(images)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            pairs.append((images[i], images[j]))
    # Sparse loop-closure pairs (every 30 frames against every other 30 frame)
    if n > 30:
        anchors = list(range(0, n, max(1, n // 20)))
        for a in anchors:
            for b in anchors:
                if a < b and (b - a) > window:
                    pairs.append((images[a], images[b]))
    sfm_pairs.write_text("\n".join(f"{a} {b}" for a, b in pairs))

    # 3. SuperGlue match the pairs
    match_features.main(matcher_conf, sfm_pairs, features=feature_path, matches=match_path)

    # 4. hloc reconstruction writes a COLMAP database + sparse model. We let
    # hloc do the database+import work, then let GLOMAP re-do the mapping.
    db_path = out_dir / "database.db"
    reconstruction.main(
        sfm_dir=out_dir,
        image_dir=frames_dir,
        pairs=sfm_pairs,
        features=feature_path,
        matches=match_path,
        camera_mode="SINGLE",
        verbose=False,
    )

    # GLOMAP global mapper over the hloc-populated DB.
    _stream_run(
        ["glomap", "mapper",
         "--database_path", str(db_path),
         "--image_path", str(frames_dir),
         "--output_path", str(out_dir / "sparse")],
        log_path=log_path,
    )


def run_scan(
    video_path: Path,
    output_mp4: Path,
    *,
    fps: int = 2,
    max_iterations: int = 30000,
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
    sfm_dir = work_dir / "sfm"
    processed = work_dir / "processed"

    try:
        # 1. ffmpeg extract frames @ fps=2, drop blurry frames via Laplacian variance.
        progress_mod.set_phase("extracting")
        _stream_run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"fps={fps},select='gt(scene\\,0.0)'",
             "-q:v", "2",
             str(raw_frames / "frame_%05d.jpg")],
            log_path=log_path,
        )
        # Quick blur filter: keep only frames with Laplacian variance above
        # the 20th percentile (drops the worst 20%).
        try:
            import cv2
            import numpy as np
            files = sorted(raw_frames.glob("frame_*.jpg"))
            vars_ = []
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                vars_.append(cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0.0)
            if len(vars_) > 20:
                thresh = np.percentile(vars_, 20)
                for f, v in zip(files, vars_):
                    if v < thresh:
                        f.unlink(missing_ok=True)
        except Exception:
            pass  # blur filter is best-effort

        n_raw = len(list(raw_frames.glob("frame_*.jpg")))
        if n_raw < 10:
            raise ValueError(f"Need >=10 frames, got {n_raw}")

        # 2-3. hloc + GLOMAP SfM
        progress_mod.set_phase("colmap")
        _hloc_glomap_sfm(raw_frames, sfm_dir, log_path)

        # Wrap SfM output in nerfstudio's expected layout via ns-process-data colmap.
        processed.mkdir(parents=True, exist_ok=True)
        # Copy frames to processed/images
        images_out = processed / "images"
        images_out.mkdir(parents=True, exist_ok=True)
        for f in raw_frames.glob("*.jpg"):
            shutil.copy(f, images_out / f.name)
        # Copy sparse to processed/colmap/sparse/0
        colmap_out = processed / "colmap" / "sparse" / "0"
        colmap_out.mkdir(parents=True, exist_ok=True)
        src_sparse = sfm_dir / "sparse" / "0"
        for f in src_sparse.glob("*"):
            shutil.copy(f, colmap_out / f.name)
        # Generate transforms.json
        _stream_run(
            ["ns-process-data", "images",
             "--data", str(images_out),
             "--output-dir", str(processed),
             "--skip-colmap",
             "--colmap-model-path", str(colmap_out)],
            log_path=log_path,
        )

        # 4. ns-train splatfacto-mcmc
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
            ["ns-train", "splatfacto-mcmc",
             "--data", str(processed),
             "--output-dir", str(train_outputs),
             "--max-num-iterations", str(max_iterations),
             "--pipeline.model.use-scale-regularization", "True",
             "--pipeline.model.cull-alpha-thresh", "0.005",
             "--viewer.quit-on-train-completion", "True",
             "--vis", "tensorboard"],
            cwd=work_dir,
            log_path=log_path,
            on_line=_on_train_line,
        )

        config_yml = _find_config_yml(work_dir)

        # 5. Export .ply
        progress_mod.set_phase("exporting")
        ply_path = scan_dir / "scan.ply"
        _stream_run(
            ["ns-export", "gaussian-splat",
             "--load-config", str(config_yml),
             "--output-dir", str(scan_dir)],
            log_path=log_path,
        )
        default_ply = scan_dir / "splat.ply"
        if default_ply.exists() and not ply_path.exists():
            default_ply.rename(ply_path)

        # 6. Compress to SOG (PlayCanvas Spatially Ordered Gaussians)
        progress_mod.set_phase("compressing")
        sog_path = scan_dir / "scan.sog"
        if ply_path.exists():
            try:
                _stream_run(
                    ["splat-transform", str(ply_path), str(sog_path)],
                    log_path=log_path,
                )
            except Exception:
                sog_path = None
        else:
            sog_path = None

        # 7. Optional preview MP4 flythrough as a fallback
        progress_mod.set_phase("rendering")
        try:
            _stream_run(
                ["ns-render", "interpolate",
                 "--load-config", str(config_yml),
                 "--output-path", str(output_mp4),
                 "--frame-rate", "30",
                 "--interpolation-steps", "30"],
                log_path=log_path,
            )
        except Exception:
            pass  # MP4 is nice-to-have; SOG is the main deliverable

        return {
            "frames": n_raw,
            "iterations": max_iterations,
            "total_seconds": round(time.time() - t0, 1),
            "mp4_bytes": output_mp4.stat().st_size if output_mp4.exists() else 0,
            "ply_bytes": ply_path.stat().st_size if ply_path.exists() else 0,
            "sog_bytes": sog_path.stat().st_size if sog_path and sog_path.exists() else 0,
        }
    finally:
        progress_mod.set_path(None)
        shutil.rmtree(raw_frames, ignore_errors=True)
