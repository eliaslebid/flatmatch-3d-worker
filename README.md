# Modula 3D scan worker

FastAPI worker that wraps [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map)
to turn a video upload into a colored point-cloud GLB. Designed to run on a CUDA
GPU (RunPod RTX 4090 recommended) and serve a single mobile beta user.

## Quick start on RunPod

1. **Launch a pod**
   - Template: any "PyTorch 2.x + CUDA 12.x" image
   - GPU: RTX 4090 (24GB) is plenty
   - Network volume: 50GB mounted at `/workspace` (so the model survives stops)
   - Expose HTTP port: 8765

2. **Open the web terminal** (RunPod → Pod → Connect → Web Terminal) and run:

   ```bash
   cd /workspace
   git clone https://github.com/eliaslebid/flatmatch-3d-worker.git
   cd flatmatch-3d-worker
   bash setup.sh
   ```

   This installs ffmpeg, FlashInfer, our FastAPI wrapper, downloads the 4.6GB
   checkpoint, and patches `demo.py`. ~5 min first time, ~30s on subsequent
   pod starts because everything lives on the persistent network volume.

3. **Start the worker**

   ```bash
   cd /workspace/lingbot-map
   nohup uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 &
   ```

4. **Get the public URL**: in the RunPod pod dashboard, find the proxy URL for
   port 8765 (looks like `https://<pod-id>-8765.proxy.runpod.net`).

5. **Point the mobile app at it**: in `apps/mobile/.env` set

   ```
   EXPO_PUBLIC_SCAN_WORKER_URL=https://<pod-id>-8765.proxy.runpod.net
   ```

   Restart Metro (`r` in the terminal) and reload the app on the phone.

## Endpoints

- `POST /scans` — multipart `video=@scan.mp4`. Returns `{ id, status: "queued", ... }`.
- `GET /scans/{id}` — full meta including `status`, `progress`, `glb_url`.
- `GET /scans/{id}/scan.glb` — the colored point-cloud GLB.
- `GET /health` — `{ ok: true, queue_locked: bool }`.
