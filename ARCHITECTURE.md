# Modula 3D scan — architecture & runbook

Snapshot of what was built, what's running, and how to operate it. Last
updated 2026-05-12.

## Shipped state (v1 "Gaussian Splatting beta", Stack A)

3D reconstruction is **3D Gaussian Splatting** via hloc + COLMAP +
nerfstudio's `splatfacto`. Output is a compressed `.ply` rendered in the
mobile app's WebView with custom first-person controls.

- **Compute**: RunPod community-cloud **L40S 48 GB** at $0.79/hr.
  Per-scan wallclock ~12–25 min, GPU cost ~$0.10–0.20.
- **Output**:
  - `scan.ply` — mkkellogg-native compressed PLY (~30–80 MB,
    300k–500k gaussians after cleanup).
  - `scan.original.ply` — pre-compression debug copy.
  - `scan.mp4` — fallback flythrough from `ns-render interpolate`.
- **Viewer**: `@mkkellogg/gaussian-splats-3d` in a `react-native-webview`
  WebView, loaded from CDN via importmap. Custom controls drive the
  camera directly (joystick + look + pinch-fly).
- **Pod**: currently terminated (last pod `pqbblmkyk9kg41`). Recreate
  via runbook below; setup is fully scripted in `setup_gsplat.sh`.

## Pipeline at 30,000 ft

```
iPhone (Modula app, Profile → 3D Scan Beta)
   │  axios POST multipart/form-data, no body limit (LAN)
   ▼
Mac Studio :8765 (autossh -L 0.0.0.0:8765 → pod :8765)
   ▼
RunPod L40S pod :8765
   └── uvicorn server.app  →  FastAPI worker
       └── pipeline.py:
           1. ffmpeg extract @ 2 fps + Laplacian blur filter (drop 20%)
           2. hloc: SuperPoint features → custom sequential pairs →
              SuperGlue matching → pycolmap incremental mapping (SfM)
           3. ns-process-data images --skip-colmap (consumes hloc output)
           4. ns-train splatfacto (--max-num-iterations 30000,
              --pipeline.model.use-scale-regularization True)
           5. ns-export gaussian-splat → raw .ply
           6. splat-transform cleanup:
                --filter-value opacity,gt,0.1
                --filter-floaters 0.05,0.2,0.02
           7. splat-transform compress → scan.compressed.ply
           8. ns-render interpolate → scan.mp4 (fallback)
```

Per scan:
1. Phone records 10–60 s video (HEVC, up to ~150 MB).
2. axios POSTs to Mac `:8765`. The Mac SSH-tunnels every byte to the pod.
   No size cap on either hop.
3. Worker accepts, returns `{id, status:"queued"}`, queues a FastAPI
   `BackgroundTask`. Per-process lock serializes scans.
4. Mobile polls `GET /scans/{id}`. When `status:"done"`, response
   contains `ply_url` (and `mp4_url` as fallback).
5. `expo-notifications` fires a local notification on completion.

## What's running (and how to find it)

| Component | Where | How to find / start |
|---|---|---|
| RunPod pod | _terminated_; recreate as `modula-gsplat-worker`, L40S, $0.79/hr | `curl -H "Authorization: Bearer <KEY>" https://rest.runpod.io/v1/pods` |
| Pod SSH | port mapped from `22/tcp` to a public port | RunPod GraphQL `runtime{ports{...}}` |
| Pod worker | `uvicorn server.app:app --host 0.0.0.0 --port 8765` in `/workspace` | `ssh ... 'pgrep -af uvicorn'` |
| Mac tunnel | `autossh -M 0 -N -L 0.0.0.0:8765:localhost:8765 -p <SSH_PORT> -i ~/.ssh/id_ed25519 root@<SSH_IP>` | `pgrep -af autossh` |
| Mobile config | `apps/mobile/.env`: `EXPO_PUBLIC_SCAN_WORKER_URL=http://192.168.31.99:8765` | `cat apps/mobile/.env \| grep SCAN` |
| Worker source | https://github.com/eliaslebid/flatmatch-3d-worker | this repo |
| Scan history | survives app reinstall — `GET /scans` lists everything in `/workspace/scans/` | open Profile → 3D Scan Beta → "Минулі сканування" |

## Operating the system

### Start everything from cold (post-termination)

```bash
# 1. Create pod (one-time per session — container disk is ephemeral; attach
#    a network volume at /workspace for persistence if needed)
curl -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer <RUNPOD_KEY>" -H "Content-Type: application/json" \
  -d '{"name":"modula-gsplat-worker","imageName":"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04","gpuTypeIds":["NVIDIA L40S"],"gpuCount":1,"containerDiskInGb":80,"ports":["8765/http","22/tcp"],"interruptible":false,"cloudType":"COMMUNITY"}'

# 2. Wait for SSH port mapping (GraphQL until publicPort != null)

# 3. SSH in, clone worker repo, run setup
ssh -p <SSH_PORT> root@<SSH_IP>
cd /workspace
git clone https://github.com/eliaslebid/flatmatch-3d-worker.git
cd flatmatch-3d-worker && bash setup_gsplat.sh
# ~25–35 min total (tinycudann + gsplat CUDA compile)

# 4. Start the worker (must use setsid — see gotchas)
ssh -p <SSH_PORT> root@<SSH_IP> 'cd /workspace && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null'

# 5. Start Mac autossh tunnel
AUTOSSH_GATETIME=0 nohup autossh -M 0 -N \
  -L 0.0.0.0:8765:localhost:8765 \
  -p <SSH_PORT> -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/tmp/runpod-known-hosts \
  root@<SSH_IP> > /tmp/scan-tunnel.log 2>&1 &
disown

# 6. apps/mobile/.env stays pointed at the Mac's LAN IP — no rebuild needed
#    unless the LAN IP itself changed.
```

### Stop everything

```bash
pkill -f autossh
curl -X DELETE -H "Authorization: Bearer <RUNPOD_KEY>" \
     https://rest.runpod.io/v1/pods/<POD_ID>
```

### Cancel in-flight scans / restart worker

```bash
ssh -p <SSH_PORT> root@<SSH_IP> '
  python3 -c "
import json, datetime, glob
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for f in glob.glob(\"/workspace/scans/*/meta.json\"):
    d = json.load(open(f))
    if d.get(\"status\") in (\"processing\", \"queued\"):
        d[\"status\"] = \"failed\"; d[\"error\"] = \"Cancelled\"; d[\"finished_at\"] = now
        json.dump(d, open(f, \"w\"), indent=2)
        print(\"cancelled\", d.get(\"id\"))
"
  pkill -9 -f uvicorn; sleep 2
  cd /workspace && setsid -f uvicorn server.app:app --host 0.0.0.0 --port 8765 > /workspace/worker.log 2>&1 < /dev/null
'
```

### Tail logs

```bash
ssh -p <SSH_PORT> root@<SSH_IP> 'tail -f /workspace/worker.log'
```

### Verify end-to-end

```bash
curl -s http://192.168.31.99:8765/health           # {"ok":true,...}
curl -X POST -F "video=@/tmp/test.mp4" http://192.168.31.99:8765/scans
curl -s http://192.168.31.99:8765/scans            # list
```

## Gotchas we hit (in order of pain)

### gsplat / nerfstudio install

1. **Stock `gsplat` PyPI wheel ships without CUDA kernels for our env.**
   Splatfacto fails at runtime with `AttributeError: 'NoneType' object has no
   attribute 'CameraModelType'`. Fix: reinstall from source with
   `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 pip install
   --no-build-isolation gsplat`. Captured in `setup_gsplat.sh` step 6.

2. **hloc doesn't bundle SuperGluePretrainedNetwork.** The SuperGlue matcher
   imports it as a top-level module. Fix: clone
   `magicleap/SuperGluePretrainedNetwork` into hloc's `third_party/` and
   symlink to `site-packages`. Captured in setup step 7.

3. **`splatfacto-mcmc` is not in nerfstudio's installed model list** despite
   docs implying it is. Use plain `splatfacto` with
   `--pipeline.model.use-scale-regularization True` instead — same paper, no
   missing model error.

4. **PyTorch ≥2.6 flips `torch.load(weights_only=True)` default**, breaking
   nerfstudio checkpoint loading at `ns-export` / `ns-render`. Fix: write
   `/usr/local/lib/python3.11/sitecustomize.py` that monkey-patches
   `torch.load`. Captured in setup step 9.

5. **Ubuntu's apt nodejs is v12, too old for `@playcanvas/splat-transform`
   (needs ≥18).** apt's nodesource setup script fails on 22.04. Fix: install
   Node 20 from the official tarball. Captured in setup step 8.

6. **`splat-transform --filter-sphere` rejects float values < 1** — bug in
   the package's `parseNumber`. Pivoted to `--filter-value opacity,gt,0.1`
   and `--filter-floaters 0.05,0.2,0.02` (voxel-based) for floater cleanup.

7. **mkkellogg/gaussian-splats-3d 0.4.7 doesn't support SOG or SPZ formats**
   despite being mentioned in some PRs — only `.ply` and
   `.compressed.ply`. We use the latter (~3× smaller than raw .ply, with
   identical visual quality at our point counts).

### Pipeline / SfM

8. **`pairs_from_sequential` missing in installed hloc version.** Wrote an
   inline sequential pair generator: each frame paired with next 10 + sparse
   loop-closure pairs every `n/20` frames.

9. **GLOMAP's DB schema is incompatible with hloc/pycolmap output**, so
   feeding hloc's matches into GLOMAP fails at the mapper step. Dropped
   GLOMAP entirely; use `hloc.reconstruction.main()` which calls pycolmap's
   incremental mapper directly. Quality difference is invisible at our
   scale (300–1500 input matches).

10. **hloc writes COLMAP `cameras.bin/images.bin/points3D.bin` directly to
    `sfm/`, NOT to `sfm/sparse/0/`.** Copy logic must handle both shapes.

11. **`ns-process-data images --colmap-model-path` needs a path RELATIVE to
    the data directory.** Absolute paths silently produce empty results.

12. **`ns-train` progress line is a printed table, not tqdm.** Regex parser:
    `re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")` after stripping ANSI
    sequences. tqdm-style regex matches zero lines.

### Networking

13. **RunPod's `*.proxy.runpod.net` URL silently truncates POST bodies at
    ~5 MB.** Phone videos die mid-upload. Direct port-mapped TCP is the
    only reliable path.

14. **Cloudflare quick-tunnels (`*.trycloudflare.com`) are blocked from at
    least one of our networks.** DNS resolves; TCP connect hangs.

15. **localhost.run tunnels truncate large POSTs mid-stream too** (we saw
    "moov atom not found" from a truncated 30 MB upload).

16. **Mac LAN relay is the only setup that doesn't break.** Phone → Mac LAN
    (no body cap, no ATS issues) → autossh tunnel to pod.

### iOS

17. **`expo-notifications` requires `aps-environment` entitlement, which a
    free Apple Developer Team can't sign.** Strip from `*.entitlements` —
    local notifications still work without it.

18. **`ENABLE_USER_SCRIPT_SANDBOXING=YES` (Xcode 15+ default) breaks
    CocoaPods builds** and lies about Podfile.lock sync. Set to `NO` in the
    pbxproj at project level.

19. **`0xe800801f Attempted to install a Beta profile…` is actually a
    wrong-profile-type error** (AppStore distribution vs Development), not
    a beta-OS error. Switch to "Automatically manage signing".

20. **`.env` and any `EXPO_PUBLIC_*` is baked at Metro start time.** Reload
    (`r`) does NOT pick up changes. Full rebuild required for release.

21. **Gallery picker has invisible 5–60 s HEVC transcode** before axios
    sees any bytes. Recorded videos upload instantly; picked ones look
    "frozen" for a while.

### Pod / runtime

22. **Ubuntu 22.04's `distutils`-installed `blinker` blocks pip upgrades.**
    Remove `/usr/lib/python3/dist-packages/blinker*` first. Captured in
    setup step 2.

23. **`nohup … &` inside `ssh` dies on SSH disconnect** when the child
    inherits SSH's controlling terminal. Use `setsid -f` instead.

24. **WebView shows "100% 100%" blank on the 2nd visit** to the same scan
    because mkkellogg's worker state persists across navigations. Fix:
    `key={url}`, `incognito`, `cacheEnabled={false}`, and
    `progressiveLoad: false`. See `apps/mobile/app/scan/[id].tsx`.

25. **Importing `splat-viewer.html.ts` breaks Metro module resolution** —
    the `.html.` infix in the name confuses the resolver. Renamed to
    `splatViewerHtml.ts`.

## Pipeline knobs (`server/pipeline.py`)

| Knob | Default | Effect |
|---|---|---|
| `fps` (ffmpeg) | 2 | Frame extraction rate |
| Laplacian blur drop | 20% | Drop worst 20% frames by Laplacian variance |
| `--max-num-iterations` | 30000 | splatfacto training steps (~10–15 min on L40S) |
| `--pipeline.model.use-scale-regularization` | True | Limits very thin/elongated gaussians (reduces floaters) |
| `splat-transform --filter-value opacity,gt,0.1` | 0.1 | Drop low-opacity gaussians (~3–5%) |
| `splat-transform --filter-floaters 0.05,0.2,0.02` | as shown | Voxel-occupancy floater removal |

## Cost notes

- L40S 48 GB community: $0.79/hr → ~$0.15 per scan, ~$19/day if idle.
- Container disk is wiped on termination. For "always-on" beta usage,
  attach a 50 GB network volume at `/workspace`.

## Mobile viewer (`apps/mobile/components/scan/splatViewerHtml.ts`)

WebView-based first-person viewer:

- Left third of the screen = virtual joystick (forward/back/strafe).
- Right two thirds = drag to look (yaw + pitch).
- Pinch with two fingers on the right = vertical fly.
- "↺" button recenters at the scene's median position.
- `cameraUp = [0, -1, 0]` (mkkellogg / nerfstudio convention).
- In-page status overlay catches `window.error` /
  `unhandledrejection` for on-device debugging without a desktop.

## Scan history (survives app reinstall)

- `GET /scans` lists every scan dir under `/workspace/scans/`.
- `apps/mobile/app/scan/history.tsx` calls it, prepopulates the Zustand
  store, navigates to `/scan/[id]`.
- Independent of MMKV state, so a fresh app install still sees old scans.

## What's NOT done yet

- **No queue UX in mobile** — multiple uploads serialize silently on the
  worker side.
- **No network volume** — every pod recreation re-pays the ~30 min setup.
- **Floaters remain visible** in low-texture areas (large white walls)
  even after cleanup. Better recording technique (slow walk, revisit
  surfaces) helps more than any post-filter.
- **Mac must be on the same WiFi as the phone.** Cellular scans need a
  TLS-fronted endpoint (Caddy + domain) or a Tailscale exit.

## Mobile config (snapshot)

- `apps/mobile/.env`:
  - `EXPO_PUBLIC_SCAN_WORKER_URL=http://192.168.31.99:8765` (Mac LAN)
  - `EXPO_PUBLIC_SCAN_BETA=1`
- `apps/mobile/app.config.ts` `ios.infoPlist`:
  - `NSAppTransportSecurity.NSAllowsLocalNetworking=true`
  - `NSLocalNetworkUsageDescription="…"`
- `apps/mobile/ios/HotlineFlat/HotlineFlat.entitlements`: empty `<dict>`
  (no `aps-environment`, free Apple Dev Team-friendly).
- `apps/mobile/ios/HotlineFlat.xcodeproj/project.pbxproj`:
  `ENABLE_USER_SCRIPT_SANDBOXING=NO` at project level.
