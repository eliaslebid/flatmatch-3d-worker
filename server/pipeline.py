"""Run lingbot-map inference and render the official rgbd_render MP4 flythrough.

This is the same pipeline that produces the demo videos on the repo:
  video -> frames -> model.inference_streaming -> NPZ -> SceneBuilder
       -> CUDA voxelization (NVIDIA Kaolin) -> OfflinePipeline -> MP4

Requires the pod to have:
  - torch 2.8.0 + cu128 wheels
  - lingbot-map[vis,render] (open3d, pyyaml, viser, trimesh)
  - kaolin built for torch 2.8 / cu128
  - demo_render/render_cuda_ext built in-place (voxel_morton_ext + frustum_cull_ext)
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_RENDER_DIR = REPO_ROOT / "demo_render"
RENDER_EXT_DIR = DEMO_RENDER_DIR / "render_cuda_ext"

# IMPORTANT: only add REPO_ROOT to sys.path here. Adding demo_render/ globally
# would shadow the top-level demo.py with demo_render/demo.py (different
# load_images signature). We import rgbd_render lazily via importlib in
# _render_to_mp4().
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import (  # noqa: E402  (demo.py is the script in the repo root)
    load_images,
    load_model,
    postprocess,
    prepare_for_visualization,
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

INDOOR_YAML = REPO_ROOT / "demo_render" / "config" / "indoor.yaml"

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


def _save_predictions_npz(predictions, npz_dir: Path) -> str:
    """Wrapper around batch_demo.save_predictions_npz that takes a Path."""
    from demo_render.batch_demo import save_predictions_npz

    return save_predictions_npz(predictions, str(npz_dir))


def _render_to_mp4(npz_dir: Path, output_mp4: Path) -> None:
    """Run the official rgbd_render pipeline on saved NPZ predictions.

    Loads the indoor preset (follow-cam, sky off, indoor scale), then
    SceneBuilder -> voxelize -> OfflinePipeline -> MP4.
    """
    import multiprocessing

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # rgbd_render lives under demo_render/. We add demo_render/ and
    # render_cuda_ext/ to sys.path *here* (after demo has already been
    # imported and cached in sys.modules) so we don't shadow the top-level
    # demo.py with demo_render/demo.py.
    for p in (DEMO_RENDER_DIR, RENDER_EXT_DIR):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from rgbd_render.config import PipelineConfig
    from rgbd_render.camera import build_camera_path
    from rgbd_render.overlay import build_overlays
    from rgbd_render.pipeline.builder import SceneBuilder
    from rgbd_render.pipeline.offline import OfflinePipeline

    cfg = (
        PipelineConfig.from_yaml(str(INDOOR_YAML))
        if INDOOR_YAML.exists()
        else PipelineConfig()
    )
    cfg.input = str(npz_dir)
    cfg.output = str(output_mp4)
    cfg.fast_review = 0
    # Force single-process rendering: the default num_workers=16 spawns
    # multiprocessing children, which kill the uvicorn parent when running
    # inside a FastAPI BackgroundTask.
    cfg.num_workers = 1
    # Sky masking is for outdoor scenes; off for indoor.
    cfg.preprocess.mask_sky = False

    scene = SceneBuilder(cfg).load().preprocess().voxelize().build()
    try:
        camera_path = build_camera_path(cfg.camera, scene)
        overlays, overlay_specs = build_overlays(cfg, scene)
        OfflinePipeline(
            scene, camera_path, overlays, cfg, overlay_specs=overlay_specs
        ).run()
    finally:
        scene.destroy()


def run_scan(
    video_path: Path,
    output_mp4: Path,
    *,
    fps: int = 5,
    first_k: Optional[int] = 300,
    progress_path: Optional[Path] = None,
) -> dict:
    """Run inference on a video and render the official MP4 flythrough.

    Returns metadata about the scan: timing, frame count, mp4 size.
    """
    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    t0 = time.time()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    npz_dir = output_mp4.parent / "predictions"
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
        _save_predictions_npz(vis, npz_dir)

        progress_mod.set_phase("rendering")
        _render_to_mp4(npz_dir, output_mp4)

        return {
            "frames": num_frames,
            "inference_seconds": round(t_infer, 1),
            "total_seconds": round(time.time() - t0, 1),
            "mp4_bytes": output_mp4.stat().st_size if output_mp4.exists() else 0,
        }
    finally:
        progress_mod.set_path(None)
