"""Synthesize a tiny valid Gaussian-Splat PLY for the v2 stub processor.

Generated on demand (no binary checked into git). Layout: a 2x2x2 grid of
8 colored gaussians at the origin so the in-app WebView viewer renders
*something* recognizable when wired against the stub.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

_HEADER = """ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

_SH_C0 = 0.28209479177387814


def _rgb_to_sh(c: float) -> float:
    return (c - 0.5) / _SH_C0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def write_sample_ply(path: Path) -> None:
    points = [
        (-0.5, -0.5, -0.5, 1.0, 0.2, 0.2),
        ( 0.5, -0.5, -0.5, 1.0, 0.6, 0.2),
        (-0.5,  0.5, -0.5, 1.0, 1.0, 0.2),
        ( 0.5,  0.5, -0.5, 0.2, 1.0, 0.2),
        (-0.5, -0.5,  0.5, 0.2, 1.0, 1.0),
        ( 0.5, -0.5,  0.5, 0.2, 0.4, 1.0),
        (-0.5,  0.5,  0.5, 0.8, 0.2, 1.0),
        ( 0.5,  0.5,  0.5, 1.0, 0.2, 0.8),
    ]
    log_scale = math.log(0.18)
    opacity = _logit(0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(_HEADER.format(n=len(points)).encode("ascii"))
        for x, y, z, r, g, b in points:
            f.write(struct.pack(
                "<17f",
                x, y, z,
                0.0, 0.0, 0.0,
                _rgb_to_sh(r), _rgb_to_sh(g), _rgb_to_sh(b),
                opacity,
                log_scale, log_scale, log_scale,
                1.0, 0.0, 0.0, 0.0,
            ))


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.ply")
    write_sample_ply(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
