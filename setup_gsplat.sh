#!/usr/bin/env bash
# Full setup for the Modula 3D-scan worker (Stack A: Gaussian Splatting).
#
# Tested on RunPod community-cloud L40S 48GB starting from:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# What this installs / configures (every step captures a real fix we hit
# during May 2026 debugging — do not reorder lightly):
#   1. apt: ffmpeg, COLMAP, build tools
#   2. blinker distutils conflict resolution
#   3. PyTorch 2.5+ (nerfstudio needs torch.nn.attention.flex_attention)
#   4. tinycudann (~10 min CUDA compile)
#   5. nerfstudio + FastAPI/uvicorn deps
#   6. gsplat REBUILT against this exact CUDA (the pre-built wheel ships
#      without CUDA kernels → "AttributeError: 'NoneType' has no attr
#      CameraModelType")
#   7. hloc + SuperGluePretrainedNetwork (hloc doesn't bundle it)
#   8. Node 20 + @playcanvas/splat-transform (PlayCanvas package needs
#      Node >=18; pod's apt nodejs is 12)
#   9. torch.load weights_only monkey-patch via sitecustomize.py (PyTorch
#      2.6+ default broke nerfstudio checkpoint loading)
#   10. Pull worker code into /workspace/server
#
# After this script, start the worker with:
#   cd /workspace && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null
#
# Total install time ~25-35 min on a fresh pod, dominated by tinycudann
# and gsplat CUDA kernel compilation.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
# L40S = SM 8.9. For A100 use 8.0; H100 use 9.0; 4090 = 8.9.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"

echo "==> [1/10] apt: ffmpeg, COLMAP, build tools"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ffmpeg colmap git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 xz-utils curl

echo "==> [2/10] resolve blinker distutils conflict"
# Ubuntu 22.04 ships blinker as a distutils package; pip can't uninstall it
# but does need a newer version for some deps. Manual clean-up.
rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true

echo "==> [3/10] PyTorch 2.5+ (cu124 base image)"
pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> [4/10] tinycudann (~10 min CUDA kernel compile)"
pip install --quiet ninja
pip install --quiet "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" 2>&1 | tail -5

echo "==> [5/10] nerfstudio + FastAPI/uvicorn"
pip install --quiet nerfstudio fastapi uvicorn python-multipart

echo "==> [6/10] gsplat REBUILD against our CUDA"
# The PyPI wheel ships without CUDA kernels for our env. Forcing a source
# build with CUDA_HOME set produces the _C extension splatfacto needs.
pip uninstall -y gsplat 2>/dev/null || true
pip install --quiet --no-build-isolation gsplat
python3 -c "
import gsplat
from gsplat.cuda._wrapper import _make_lazy_cuda_obj
_ = _make_lazy_cuda_obj('CameraModelType')
print('gsplat', gsplat.__version__, 'CUDA kernels OK')
"

echo "==> [7/10] hloc + SuperGluePretrainedNetwork"
pip install --quiet "git+https://github.com/cvg/Hierarchical-Localization.git"
# hloc's SuperGlue matcher imports `SuperGluePretrainedNetwork` as a
# top-level module; the project doesn't bundle it.
HLOC_DIR=$(python3 -c "import hloc, os; print(os.path.dirname(os.path.dirname(hloc.__file__)))")
if [ ! -d "$HLOC_DIR/third_party/SuperGluePretrainedNetwork" ]; then
    git clone --quiet --depth 1 https://github.com/magicleap/SuperGluePretrainedNetwork.git \
        "$HLOC_DIR/third_party/SuperGluePretrainedNetwork"
fi
ln -sf "$HLOC_DIR/third_party/SuperGluePretrainedNetwork" \
    /usr/local/lib/python3.11/dist-packages/SuperGluePretrainedNetwork
python3 -c "import SuperGluePretrainedNetwork; print('SuperGluePretrainedNetwork importable')"
python3 -c "from hloc import match_features, extract_features, reconstruction; print('hloc OK')"

echo "==> [8/10] Node 20 + splat-transform"
# Ubuntu apt ships Node 12 which is too old for @playcanvas/splat-transform
# (requires >=18). Use the official tarball.
if [ "$(node --version 2>/dev/null | head -c 3)" != "v20" ] && [ "$(node --version 2>/dev/null | head -c 3)" != "v21" ]; then
    cd /tmp && curl -fsSLO https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz
    tar -xf node-v20.18.0-linux-x64.tar.xz
    cp -rf node-v20.18.0-linux-x64/* /usr/local/
    cd "$WORKDIR"
fi
npm install -g @playcanvas/splat-transform 2>&1 | tail -3
splat-transform --version || true

echo "==> [9/10] torch.load weights_only=False monkey-patch"
# PyTorch >=2.6 flipped the default of weights_only to True. Nerfstudio's
# checkpoint files contain pickled non-tensor objects, which breaks the
# stricter loader. Patch via sitecustomize so every Python invocation gets it.
cat > /usr/local/lib/python3.11/sitecustomize.py <<'EOF'
# nerfstudio checkpoints contain pickled Python objects; force the legacy
# torch.load(weights_only=False) behavior so ns-export / ns-render work.
try:
    import torch
    _orig_load = torch.load
    def _safe_load(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig_load(*a, **kw)
    torch.load = _safe_load
except Exception:
    pass
EOF
python3 -c "import torch; print('torch.load patched:', torch.load.__name__ == '_safe_load')"

echo "==> [10/10] worker code → /workspace/server"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/server"
mkdir -p /workspace/server
cp -f "$SRC_DIR"/*.py /workspace/server/

echo ""
echo "================================================================"
echo "Setup complete. Start the worker:"
echo "  cd /workspace && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null"
echo "Then verify:"
echo "  curl -s http://localhost:8765/health"
echo "================================================================"
