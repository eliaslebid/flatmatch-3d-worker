#!/usr/bin/env bash
# Install nerfstudio + COLMAP + our worker on a fresh RunPod CUDA pod.
# Use the runpod/pytorch:2.4.0-py3.11-cuda12.4.1 base image.
#
# Long install (~15-25 min, mostly tinycudann compile + COLMAP).
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"

echo "==> [1/7] apt: ffmpeg, colmap, build tools"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ffmpeg colmap git build-essential cmake ninja-build

echo "==> [2/7] upgrade torch to 2.5 (needed for nerfstudio)"
rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true
pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> [3/7] tinycudann (compiles CUDA kernels, ~10 min)"
pip install --quiet ninja
pip install --quiet "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" 2>&1 | tail -5

echo "==> [4/7] nerfstudio"
pip install --quiet nerfstudio 2>&1 | tail -5

echo "==> [5/7] FastAPI worker deps"
pip install --quiet fastapi uvicorn python-multipart

echo "==> [6/7] worker code"
cp -f "$(dirname "$0")"/server/*.py "$WORKDIR/" 2>/dev/null || true
mkdir -p "$WORKDIR/server"
cp -f "$(dirname "$0")"/server/*.py "$WORKDIR/server/"

echo "==> [7/7] sanity check"
which colmap
python3 -c "import torch, tinycudann, nerfstudio; print('torch', torch.__version__, 'tcnn ok', 'nerfstudio ok')"
ns-train --help > /dev/null && echo "ns-train CLI ok"

echo ""
echo "================================================================"
echo "Setup complete. To start the worker:"
echo "  cd $WORKDIR"
echo "  setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > worker.log 2>&1 < /dev/null"
echo "================================================================"
