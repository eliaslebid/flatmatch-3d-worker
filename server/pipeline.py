"""Run lingbot-map inference and export a colored point-cloud GLB.

This is the v0 "GLB beta" path: model inference + depth unprojection +
voxel-grid downsampling + trimesh export. Works on either CPU or CUDA.

The official rgbd_render MP4 pipeline (in demo_render/) was attempted but
silently segfaults inside Kaolin's octree builder on our pod env. Tracked
for v2 along with a Gaussian Splatting alternative — see
https://github.com/eliaslebid/flatmatch-3d-worker/blob/main/ARCHITECTURE.md
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import (  # noqa: E402
    load_images,
    load_model,
    postprocess,
    prepare_for_visualization,
)
from lingbot_map.utils.geometry import (  # noqa: E402
    unproject_depth_map_to_point_map,
)
from server import progress as progress_mod  # noqa: E402


@dataclass
class InferenceArgs:
    """Minimal stand-in for the argparse Namespace that load_model expects."""

    model_path: str
    mode: str = "streaming"
    image_size: int = 518
    patch_size: int = 14
    enable_3d_rope: bool = True
    max_frame_num: int = 320
    kv_cache_sliding_window: int = 320
    num_scale_frames: int = 8
    use_sdpa: bool = True
    camera_num_iterations: int = 4


CHECKPOINT_PATH = os.environ.get(
    "LINGBOT_CHECKPOINT",
    str(REPO_ROOT / "checkpoints" / "lingbot-map-long.pt")
    if (REPO_ROOT / "checkpoints" / "lingbot-map-long.pt").exists()
    else str(REPO_ROOT / "checkpoints" / "lingbot-map.pt"),
)

_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        args = InferenceArgs(model_path=CHECKPOINT_PATH)
        _MODEL = load_model(args, _DEVICE)
        if _DEVICE.type == "cuda" and getattr(_MODEL, "aggregator", None) is not None:
            _MODEL.aggregator = _MODEL.aggregator.to(dtype=torch.bfloat16)
    return _MODEL


def run_scan(
    video_path: Path,
    output_glb: Path,
    *,
    fps: int = 5,
    first_k: Optional[int] = 300,
    conf_threshold: float = 1.5,
    max_points: int = 500_000,
    progress_path: Optional[Path] = None,
) -> dict:
    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    t0 = time.time()
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    progress_mod.set_path(progress_path)

    try:
        progress_mod.set_phase("extracting")
        images, _paths, _folder = load_images(
            video_path=str(video_path),
            fps=fps,
            first_k=first_k,
            image_size=518,
            patch_size=14,
        )
        num_frames = int(images.shape[0])
        if num_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {num_frames}")

        progress_mod.set_phase("loading_model")
        model = _get_model()
        images_dev = images.to(_DEVICE)

        progress_mod.set_phase("inferring", current=0, total=num_frames)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if _DEVICE.type == "cuda"
            else contextlib.nullcontext()
        )
        t_infer = time.time()
        with torch.no_grad(), amp_ctx:
            predictions = model.inference_streaming(
                images_dev,
                num_scale_frames=8,
                keyframe_interval=1,
                output_device=torch.device("cpu"),
            )
        t_infer = time.time() - t_infer

        progress_mod.set_phase("exporting")
        predictions, images_cpu = postprocess(predictions, images_dev)
        vis = prepare_for_visualization(predictions, images_cpu)

        depth = vis["depth"]
        depth_conf = vis["depth_conf"]
        extrinsic = vis["extrinsic"]
        intrinsic = vis["intrinsic"]
        imgs = vis["images"]

        world = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        pts = world.reshape(-1, 3)
        colors = imgs.transpose(0, 2, 3, 1).reshape(-1, 3)
        conf = depth_conf.reshape(-1)

        mask = np.isfinite(pts).all(axis=1) & (conf >= conf_threshold)
        pts = pts[mask]
        colors = colors[mask]

        # Voxel-grid downsample preserves spatial structure better than random.
        if len(pts) > max_points:
            lo = np.percentile(pts, 1, axis=0)
            hi = np.percentile(pts, 99, axis=0)
            scene_diag = float(np.linalg.norm(hi - lo))
            voxel_size = max(scene_diag / 500.0, 1e-4)
            for _ in range(6):
                grid = np.floor(pts / voxel_size).astype(np.int64)
                _, idx = np.unique(grid, axis=0, return_index=True)
                if len(idx) <= max_points * 1.1:
                    break
                voxel_size *= 1.4
            idx.sort()
            pts = pts[idx]
            colors = colors[idx]

        colors_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        cloud = trimesh.PointCloud(vertices=pts.astype(np.float32), colors=colors_u8)
        trimesh.Scene([cloud]).export(str(output_glb), file_type="glb")

        return {
            "frames": num_frames,
            "points": int(len(pts)),
            "inference_seconds": round(t_infer, 1),
            "total_seconds": round(time.time() - t0, 1),
            "glb_bytes": output_glb.stat().st_size,
        }
    finally:
        progress_mod.set_path(None)
