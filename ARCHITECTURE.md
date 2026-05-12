# Modula 3D scan — architecture & runbook

Snapshot of what was built, what's running, and how to operate it. Written
2026-05-12 at the end of a long debugging session — captures every gotcha we
hit so the next person (or future you) doesn't relearn them.

## Shipped state (v0 "GLB beta")

After a long debugging session ending 2026-05-12, the shipped configuration is:

- **Compute**: Mac Studio CPU (no rented GPU). Inference at ~3 s/frame,
  total ~5–15 min per 30 s phone video. **No ongoing cost.**
- **Output**: colored point-cloud GLB, voxel-grid downsampled to ~500k
  points, rendered in-app via expo-gl + three.js.
- **Quality**: fragmented but functional. The official MP4 renderer was
  attempted and got close — see "Why no official MP4 renderer (yet)" below.
- **Pod**: terminated. To resume cloud-GPU mode, follow the runbook below.

## Pipeline at 30,000 ft (current GLB beta path)

```
iPhone (Modula app, Profile → 3D Scan Beta)
   │  axios POST multipart/form-data, no body limit (LAN)
   ▼
Mac Studio :8765
   └── uvicorn server.app  →  FastAPI worker
       ├── lingbot-map (cloned to ~/code/lingbot-map)
       ├── lingbot-map-long.pt checkpoint (4.4 GB)
       └── pipeline.py → trimesh GLB export
```

## Cloud-GPU pipeline (when needed, currently OFF)

```
iPhone → Mac :8765 (autossh -L) → RunPod pod :8765 (L40S 48 GB, $0.79/hr)
   └── uvicorn server.app  →  FastAPI worker
       └── (same pipeline.py, picks CUDA + bf16 automatically)
```

Per scan:
1. Phone records 10–60 s video (HEVC, up to ~100 MB).
2. axios POSTs to Mac `:8765`. The Mac SSH-tunnels every byte to the pod's
   `:8765`. No size cap on either hop.
3. Worker accepts the upload, returns `{id, status:"queued"}`, queues a
   FastAPI BackgroundTask. Per-process lock serializes scans.
4. Worker pipeline (see [`server/pipeline.py`](server/pipeline.py)):
   - `load_images` extracts frames from MP4 at 5 fps (cap `first_k=300`).
   - `load_model` once, lazy global, aggregator cast to bf16 on CUDA.
   - `model.inference_streaming(num_scale_frames=8, keyframe_interval=1)`
     under bf16 autocast.
   - `save_predictions_npz` writes per-frame `.npz` to `predictions/`.
   - `rgbd_render.OfflinePipeline` → MP4 flythrough at `scan.mp4`.
5. Mobile polls `GET /scans/{id}`. When `status:"done"`, response contains
   `mp4_url`. Mobile renders the video in a `WebView` (HTML5 `<video>`).
6. `expo-notifications` fires a local notification when status flips to done.

## What's running right now

| Component | Where | How to find it |
|---|---|---|
| RunPod pod | `xgiotw11pmmf8e`, L40S, Taiwan, $0.79/hr | `curl -H "Authorization: Bearer <KEY>" https://rest.runpod.io/v1/pods` |
| Pod SSH | `root@193.183.22.51:1883`, key `~/.ssh/id_ed25519` | `cat ~/.ssh/id_ed25519.pub` is in pod's `PUBLIC_KEY` env |
| Pod worker | `uvicorn server.app:app --host 0.0.0.0 --port 8765` in `/workspace/lingbot-map` | `ssh ... 'pgrep -af uvicorn'` |
| Mac tunnel | `autossh -M 0 -N -L 0.0.0.0:8765:localhost:8765 -p 1883 -i ~/.ssh/id_ed25519 root@193.183.22.51` | `pgrep -af autossh` |
| Mobile config | `apps/mobile/.env`: `EXPO_PUBLIC_SCAN_WORKER_URL=http://192.168.31.99:8765` | `cat apps/mobile/.env \| grep SCAN` |
| Worker source | https://github.com/eliaslebid/flatmatch-3d-worker | this repo |

## Operating the system

### Start everything from cold

```bash
# 1. Create pod (one-time per session — pod is ephemeral, model on container disk)
curl -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer <RUNPOD_KEY>" -H "Content-Type: application/json" \
  -d '{"name":"modula-3d-worker","imageName":"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04","gpuTypeIds":["NVIDIA L40S"],"gpuCount":1,"containerDiskInGb":50,"ports":["8765/http","22/tcp"],"interruptible":false,"cloudType":"COMMUNITY"}'

# 2. Wait for SSH port mapping (GraphQL `runtime{ports{...}}` until publicPort != null)

# 3. SSH in, clone worker repo, run setup + renderer upgrade
ssh -p <SSH_PORT> root@<SSH_IP>
cd /workspace
git clone https://github.com/eliaslebid/flatmatch-3d-worker.git
cd flatmatch-3d-worker && bash setup.sh
# Then run the renderer-extras install (manual — see "Pod setup gotchas")
rm -rf /usr/lib/python3/dist-packages/blinker*
pip install --quiet open3d pyyaml onnxruntime-gpu
pip install --quiet --upgrade torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --quiet --index-url https://pypi.org/simple kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
cd /workspace/lingbot-map/demo_render/render_cuda_ext && python setup.py build_ext --inplace
# Also download the long-sequence checkpoint
cd /workspace/lingbot-map && python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('robbyant/lingbot-map', 'lingbot-map-long.pt', local_dir='checkpoints')"

# 4. Start the worker (must use setsid — see "Why setsid")
ssh -p <SSH_PORT> root@<SSH_IP> 'cd /workspace/lingbot-map && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null'

# 5. Start Mac autossh tunnel
AUTOSSH_GATETIME=0 nohup autossh -M 0 -N \
  -L 0.0.0.0:8765:localhost:8765 \
  -p <SSH_PORT> -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/tmp/runpod-known-hosts \
  root@<SSH_IP> > /tmp/scan-tunnel.log 2>&1 &
disown

# 6. Update apps/mobile/.env if SSH_IP/SSH_PORT changed (it doesn't; URL is Mac's LAN IP)

# 7. Rebuild iOS app only if .env changed:
#    APP_ENV=production pnpm exec expo run:ios --device --configuration Release
```

### Stop everything

```bash
# Stop tunnel
pkill -f autossh

# Stop pod (kills billing)
curl -X DELETE -H "Authorization: Bearer <RUNPOD_KEY>" https://rest.runpod.io/v1/pods/<POD_ID>
```

### Cancel in-flight scans

```bash
ssh -p <SSH_PORT> root@<SSH_IP> 'python3 -c "
import json, datetime, glob
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for f in glob.glob(\"/workspace/lingbot-map/server/data/*/meta.json\"):
    d = json.load(open(f))
    if d.get(\"status\") in (\"processing\", \"queued\"):
        d[\"status\"] = \"failed\"; d[\"error\"] = \"Cancelled\"; d[\"finished_at\"] = now
        json.dump(d, open(f, \"w\"), indent=2)
        print(\"cancelled\", d.get(\"id\"))
"; pkill -9 -f uvicorn; sleep 2; cd /workspace/lingbot-map && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null'
```

### Tail the worker log live

```bash
ssh -p <SSH_PORT> root@<SSH_IP> 'tail -f /workspace/worker.log'
```

### Verify everything works

```bash
# From Mac, after tunnel + worker are up
curl -s http://192.168.31.99:8765/health
# → {"ok":true,"queue_locked":false}

# End-to-end test upload
curl -X POST -F "video=@/tmp/test.mp4" http://192.168.31.99:8765/scans
# Returns {id, status:"queued"}, then poll /scans/{id}.
```

## Gotchas we hit (in order of pain)

### Networking

1. **RunPod's `*.proxy.runpod.net` URL silently truncates POST bodies at ~5 MB.**
   Real phone videos die mid-upload. Direct TCP (port-mapped 8765) is the only
   reliable path, but it's plain HTTP — requires either ATS exemption (we use
   the Mac as a LAN frontend) or a separate HTTPS terminator (Caddy/Cloudflare).

2. **Cloudflare quick-tunnels (`*.trycloudflare.com`) are blocked from at
   least one of our networks** (DNS resolves, TCP connect to those specific
   Cloudflare IPs hangs). Mac couldn't reach the tunnel; the pod could.

3. **localhost.run tunnels work but truncate large POSTs mid-stream too** —
   30 MB completed from the Mac, but the phone got an "moov atom not found"
   from a truncated upload. Suspect SSH tunnel back-pressure.

4. **Mac as relay is the cleanest setup.** Phone → Mac LAN (no body cap, no
   ATS issues, WPA-encrypted at the WiFi layer) → SSH tunnel to pod.

5. **iOS App Transport Security blocks plain HTTP to public IPs even with
   `NSAllowsLocalNetworking=true`** — that key only covers RFC1918 LAN IPs.

6. **iOS Local Network permission is silent.** Without
   `NSLocalNetworkUsageDescription` in Info.plist, iOS denies connections to
   `192.168.x.x` without ever prompting.

### iOS

7. **`expo-notifications` requires `aps-environment` entitlement, which a
   free Apple Developer Team can't sign.** Strip from `*.entitlements` to
   install on personal-team builds. Local notifications still work.

8. **`ENABLE_USER_SCRIPT_SANDBOXING=YES` (Xcode 15+ default) breaks CocoaPods
   builds** and lies about it ("The sandbox is not in sync with the
   Podfile.lock"). Set to `NO` at project level.

9. **`0xe800801f Attempted to install a Beta profile without the proper
   entitlement` = wrong profile type, not wrong device OS.** Switching from
   manual AppStore signing to "Automatically manage signing" fixes it.

10. **`.env` (and any `EXPO_PUBLIC_*`) is baked into the bundle at Metro
    start time.** Reload-only (`r`) does NOT pick up `.env` changes. Need a
    full Metro restart, or for `--configuration Release` builds, a full
    `expo run:ios` rebuild.

11. **Gallery picker is slow because of HEVC → sandbox copy/transcode.**
    Recorded videos upload instantly; gallery picks have a 5–60s
    invisible "preparing" phase before axios sees any bytes.

### Pod setup

12. **`pip install` on the pod fights `distutils`-installed `blinker`.**
    Remove `/usr/lib/python3/dist-packages/blinker*` first.

13. **`pip install -e ".[render]"` doesn't work** — the `render` extra
    isn't declared in `pyproject.toml`. Install `open3d pyyaml
    onnxruntime-gpu` manually.

14. **Adding `demo_render/` to `sys.path` shadows the top-level `demo.py`**
    with `demo_render/demo.py` (different `load_images` signature, returns
    2 values instead of 3). Fix: import `demo` first (let it cache in
    `sys.modules`), THEN add `demo_render/` to `sys.path` lazily, only
    when calling the renderer.

15. **`nohup ... &` inside `ssh` dies on SSH disconnect** if the spawned
    process inherits SSH's controlling terminal. Use `setsid -f` instead.

16. **Renderer's default `num_workers=16` spawns multiprocessing children
    that kill the FastAPI uvicorn parent** inside BackgroundTasks. Force
    `cfg.num_workers = 1` for serial rendering.

### Model / inference

17. **Hardcoded `_DEVICE = torch.device("cpu")`** in `pipeline.py` made
    everything run on CPU even on a $0.79/hr 4090 box for an hour. Always
    `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

18. **Lingbot-map needs `torch>=2.5`** for `torch.nn.attention.flex_attention`.
    PyTorch base image with 2.4 gives `ModuleNotFoundError`. Upgrade to 2.8 +
    cu128 wheels for the official renderer (Kaolin needs it).

19. **24 GB VRAM is not enough** for paper-spec settings (`window_size=128`,
    `keyframe_interval=2`, `num_scale_frames=8`). 4090 OOMs at ~22.5 GB. The
    L40S 48 GB handles it. The paper benchmarks on A100.

20. **bf16 input tensor breaks downstream `.numpy()`** (numpy has no bf16
    dtype) during GLB/MP4 export. Cast model weights to bf16, but keep
    input images in fp32; let autocast handle bf16 math internally.

21. **`lingbot-map.pt` ("balanced") vs `lingbot-map-long.pt` ("long-sequence,
    recommended for indoor")** produce very different quality. Use long.

22. **Random downsampling of per-pixel point cloud throws away spatial
    structure.** Voxel-grid downsampling preserves dense surfaces. Even
    better: skip it entirely and use the official `rgbd_render` CUDA
    voxelizer, which produces the demo-quality MP4s.

23. **`windowed` mode is "for >3000 frames" per README**, not >320. Using it
    for short scans introduces window-boundary discontinuities and degrades
    quality. Use `streaming` mode for any scan ≤320 frames.

24. **Quality is bounded by recording technique.** Feed-forward 3D models
    need multi-view triangulation; handheld pans without revisiting surfaces
    give the model very little parallax. Slow, deliberate walks revisiting
    feature-dense areas yield dramatically better output.

## Pipeline knobs (server/pipeline.py)

| Knob | Default | Effect |
|---|---|---|
| `fps` | 5 | Frame extraction rate from input video |
| `first_k` | 300 | Hard cap on extracted frames (60 s at 5 fps) |
| `num_scale_frames` | 8 | Paper default; bidirectional scale frames |
| `keyframe_interval` | 1 | Cache every frame (paper default for ≤320 frames) |
| `cfg.num_workers` | 1 | Force serial render — DO NOT raise inside FastAPI |
| `cfg.preprocess.mask_sky` | false | Sky masking off for indoor; on for outdoor |

To experiment with higher quality, edit `_render_to_mp4()` in `pipeline.py`
and load `outdoor_large.yaml` or `indoor_overview.yaml` instead of
`indoor.yaml`. The renderer respects all the YAML camera/scene knobs.

## Cost notes

- L40S 48 GB on RunPod community: $0.79/hr (~$19/day if left running).
- Container disk loses model on pod termination; first scan on a fresh pod
  pays ~5 min of setup (model re-download + extension rebuild).
- For "always-on" beta usage, attach a **50 GB network volume** at
  `/workspace` so the model + Python env persist across pod stops.

## Why no official MP4 renderer (yet)

We got the official `demo_render/rgbd_render` pipeline 95% of the way
working on the L40S pod:

- ✅ Upgraded torch 2.4 → 2.8.0+cu128 (Kaolin's required version)
- ✅ Installed Kaolin 0.18.0 for torch 2.8/cu128 (after fighting blinker
  distutils conflicts)
- ✅ Built `render_cuda_ext` (`voxel_morton_ext`, `frustum_cull_ext`)
- ✅ Pipeline runs through inference → NPZ save → SceneBuilder → unproject
- ❌ **Silent SIGSEGV inside `OctreeSPC.build`** (the next step after unproject)

The crash happens with **no Python traceback even with `faulthandler.enable`**
— it's a hard native fault in Kaolin's SPC code. Kaolin's basic SPC test
(`unbatched_points_to_octree`) works fine in isolation, so the bug is
specific to how rgbd_render uses it.

Subprocess isolation (`server/render_runner.py` in earlier commits) was
added so the segfault doesn't kill the FastAPI worker — but we couldn't
get past the segfault itself.

Next time, two paths to try:
1. Rebuild Kaolin from source against our exact torch/CUDA combo (could be
   a kernel ABI mismatch with the prebuilt wheel).
2. Skip Kaolin entirely: rewrite voxelization with `open3d` + GPU CUDA
   reduction. Or pivot to **Gaussian Splatting** (`nerfstudio` /
   `splatfacto`), which gives 9/10 quality vs lingbot-map's 3/10 on
   handheld video.

## What's NOT done yet

- **Quality is fragmented point cloud, not the photographable demo output.**
  The fundamental fix is gaussian splatting — see "Why no official MP4
  renderer".
- **Mac must be awake on the same WiFi as the phone.** Cellular / hotel
  WiFi scans need either a TLS-fronted cloud endpoint (Caddy + a custom
  domain) or a Tailscale exit.
- **No queue UX in mobile** — multiple uploads serialize silently.
- **No network volume on the pod** — every pod restart wipes the model.

## Mobile config (snapshot)

- `apps/mobile/.env`:
  - `EXPO_PUBLIC_SCAN_WORKER_URL=http://192.168.31.99:8765` (Mac LAN)
  - `EXPO_PUBLIC_SCAN_BETA=1`
- `apps/mobile/app.config.ts` `ios.infoPlist`:
  - `NSAppTransportSecurity.NSAllowsLocalNetworking=true`
  - `NSLocalNetworkUsageDescription="…"`
- `apps/mobile/ios/HotlineFlat/HotlineFlat.entitlements`: empty `<dict>`
  (no `aps-environment`, so a free Apple Dev Team can sign).
- `apps/mobile/ios/HotlineFlat.xcodeproj/project.pbxproj`:
  `ENABLE_USER_SCRIPT_SANDBOXING=NO` at project level.
