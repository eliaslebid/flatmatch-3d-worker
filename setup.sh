#!/usr/bin/env bash
# One-shot installer for the Modula 3D scan worker on a RunPod CUDA pod.
# Run from /workspace (RunPod's persistent volume mount).
#
# Assumes: PyTorch + CUDA 12.4/12.8 base image, network volume at /workspace.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
LINGBOT_DIR="$WORKDIR/lingbot-map"

echo "==> [1/6] apt: ffmpeg"
apt-get update -qq
apt-get install -y -qq ffmpeg git

echo "==> [2/6] clone lingbot-map"
if [ ! -d "$LINGBOT_DIR" ]; then
    git clone --depth 1 https://github.com/robbyant/lingbot-map.git "$LINGBOT_DIR"
fi

echo "==> [3/6] copy worker code into lingbot-map"
mkdir -p "$LINGBOT_DIR/server"
cp -f "$(dirname "$0")"/server/*.py "$LINGBOT_DIR/server/"

echo "==> [4/6] patch demo.py for CPU/GPU autocast portability"
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/lingbot-map/demo.py")
src = p.read_text()
old = '    output_device = torch.device("cpu") if args.offload_to_cpu else None\n\n    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):'
new = '''    output_device = torch.device("cpu") if args.offload_to_cpu else None

    import contextlib
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_ctx:'''
if old in src:
    p.write_text(src.replace(old, new, 1))
    print("  patched.")
else:
    print("  patch site not found (maybe already patched). continuing.")
PY

echo "==> [5/6] python deps"
cd "$LINGBOT_DIR"
pip install --quiet -e ".[vis]"
pip install --quiet --index-url https://pypi.org/simple flashinfer-python
pip install --quiet fastapi uvicorn python-multipart

echo "==> [6/6] model checkpoint"
mkdir -p checkpoints
if [ ! -s checkpoints/lingbot-map.pt ]; then
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('robbyant/lingbot-map', 'lingbot-map.pt', local_dir='checkpoints')"
else
    echo "  checkpoint already present"
fi

echo ""
echo "================================================================"
echo "Setup complete. To start the worker:"
echo "  cd $LINGBOT_DIR"
echo "  uvicorn server.app:app --host 0.0.0.0 --port 8765"
echo "================================================================"
