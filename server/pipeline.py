"""Run lingbot-map inference on a video and export a colored point-cloud GLB.

Wraps the demo.py functions (load_images, load_model, postprocess,
prepare_for_visualization) so the FastAPI worker can call into them without
shelling out to demo.py.
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

from demo import (  # noqa: E402  (demo.py is the script in the repo root)
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
    # Streaming is the paper-spec mode for ≤320 frame scans. We cap input
    # at 300 frames upstream, so streaming is always fine.
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
    # Prefer the long-sequence checkpoint when available — it's tuned for
    # indoor walkthroughs and large-scale scenes per the model card.
    str(REPO_ROOT / "checkpoints" / "lingbot-map-long.pt")
    if (REPO_ROOT / "checkpoints" / "lingbot-map-long.pt").exists()
    else str(REPO_ROOT / "checkpoints" / "lingbot-map.pt"),
)

# Lazy global model — first request pays the load cost (~6s), subsequent reuse.
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        args = InferenceArgs(model_path=CHECKPOINT_PATH)
        _MODEL = load_model(args, _DEVICE)
        # On CUDA, cast the DINOv2-style aggregator to bf16 to drop a redundant
        # fp32 master weight copy (~2-3 GB saved). Heads stay fp32 internally
        # under autocast(enabled=False).
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
        # Release any cached blocks from a previous (possibly failed) run.
        torch.cuda.empty_cache()
    """Run inference on a video and write a colored point-cloud GLB.

    Returns metadata about the scan: timing, frame count, point count.
    """
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
        # Keep input images in fp32 — autocast handles bf16 internally. Casting
        # the input to bf16 causes downstream `.numpy()` failures (numpy lacks
        # a native bf16 dtype).
        images_dev = images.to(_DEVICE)

        # Paper-spec streaming: every frame cached, 8 scale frames.
        # On the 48 GB L40S this comfortably fits up to 300 frames in bf16.
        progress_mod.set_phase("inferring", current=0, total=num_frames)
        if _DEVICE.type == "cuda":
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            amp_ctx = contextlib.nullcontext()
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

        # The streaming model returns depth + extrinsic + intrinsic; unproject to world.
        depth = vis["depth"]              # (S, H, W, 1)
        depth_conf = vis["depth_conf"]    # (S, H, W)
        extrinsic = vis["extrinsic"]      # (S, 3, 4) c2w after postprocess
        intrinsic = vis["intrinsic"]      # (S, 3, 3)
        imgs = vis["images"]              # (S, 3, H, W) in [0,1]

        world = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)  # (S, H, W, 3)

        pts = world.reshape(-1, 3)
        colors = imgs.transpose(0, 2, 3, 1).reshape(-1, 3)
        conf = depth_conf.reshape(-1)

        mask = np.isfinite(pts).all(axis=1) & (conf >= conf_threshold)
        pts = pts[mask]
        colors = colors[mask]

        # Voxel-grid downsample: keep one point per voxel. Preserves spatial
        # structure way better than random subsampling (which leaves sparse
        # areas sparse and over-represents dense regions). We auto-tune the
        # voxel size to hit roughly max_points.
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
        scene = trimesh.Scene([cloud])
        scene.export(str(output_glb), file_type="glb")

        return {
            "frames": num_frames,
            "points": int(len(pts)),
            "inference_seconds": round(t_infer, 1),
            "total_seconds": round(time.time() - t0, 1),
            "glb_bytes": output_glb.stat().st_size,
        }
    finally:
        progress_mod.set_path(None)
