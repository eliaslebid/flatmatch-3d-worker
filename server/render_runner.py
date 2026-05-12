"""Standalone renderer entry point.

Run as a subprocess so any CUDA/Kaolin crash in the renderer doesn't take
down the FastAPI uvicorn parent process. Invoked by pipeline.py as:

    python3 -m server.render_runner <npz_dir> <output_mp4>
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

# Dump Python traceback on SIGSEGV / SIGABRT so we can see where the
# native crash happens (renderer uses Kaolin CUDA extensions that can
# segfault silently).
faulthandler.enable(file=sys.stderr, all_threads=True)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_runner <npz_dir> <output_mp4>", file=sys.stderr)
        return 2

    npz_dir = Path(sys.argv[1])
    output_mp4 = Path(sys.argv[2])

    # Add demo_render/ + render_cuda_ext/ to sys.path so rgbd_render imports.
    repo_root = Path(__file__).resolve().parent.parent
    for p in (repo_root, repo_root / "demo_render", repo_root / "demo_render" / "render_cuda_ext"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from rgbd_render.camera import build_camera_path
    from rgbd_render.config import PipelineConfig
    from rgbd_render.overlay import build_overlays
    from rgbd_render.pipeline.builder import SceneBuilder
    from rgbd_render.pipeline.offline import OfflinePipeline

    indoor_yaml = repo_root / "demo_render" / "config" / "indoor.yaml"
    cfg = (
        PipelineConfig.from_yaml(str(indoor_yaml))
        if indoor_yaml.exists()
        else PipelineConfig()
    )
    cfg.input = str(npz_dir)
    cfg.output = str(output_mp4)
    cfg.fast_review = 0
    cfg.num_workers = 1
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
