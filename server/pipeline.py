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
    str(REPO_ROOT / "checkpoints" / "lingbot-map.pt"),
)

# Lazy global model — first request pays the load cost (~6s), subsequent reuse.
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        args = InferenceArgs(model_path=CHECKPOINT_PATH)
        _MODEL = load_model(args, _DEVICE)
    return _MODEL


def run_scan(
    video_path: Path,
    output_glb: Path,
    *,
    fps: int = 5,
    first_k: Optional[int] = 160,
    conf_threshold: float = 1.5,
    max_points: int = 500_000,
    progress_path: Optional[Path] = None,
) -> dict:
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
        images_dev = images.to(_DEVICE)

        # Streaming inference. On GPU use bf16 autocast (matches training);
        # on CPU autocast is a no-op via nullcontext.
        progress_mod.set_phase("inferring", current=0, total=num_frames)
        if _DEVICE.type == "cuda":
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            num_scale_frames = 4
            keyframe_interval = 2
        else:
            amp_ctx = contextlib.nullcontext()
            num_scale_frames = 8
            keyframe_interval = 1
        t_infer = time.time()
        with torch.no_grad(), amp_ctx:
            predictions = model.inference_streaming(
                images_dev,
                num_scale_frames=num_scale_frames,
                keyframe_interval=keyframe_interval,
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

        # Cap output at ~max_points so the mobile viewer stays responsive.
        # 500k points -> ~8MB GLB, ~60fps on iPhone. Above 1M, three.js
        # on-device gets sluggish.
        if len(pts) > max_points:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(pts), size=max_points, replace=False)
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
