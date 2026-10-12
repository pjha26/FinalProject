# Real-Time Monocular Depth Estimation

Final-year project: Monodepth2 RGB-only relative-depth inference with CARLA 0.9.13, following the supplied project PDF. Development is organized into five two-day phases.

## Phase 1 status

**PARTIAL: software foundation and CUDA baseline verified; live simulator gates are pending.**

- Python 3.8.20, PyTorch 2.0.1 + CUDA 11.8, torchvision 0.15.2, and CARLA client 0.9.13 are installed in the project-local `.venv`.
- Official monocular `mono_640x192` weights were downloaded and matched the upstream archive checksum. Source revision: `b017a8164fae339b0ce01f3a9e5048104d5abd0e`.
- Inference succeeds on the NVIDIA GTX 1650. The single warmed-up baseline observation was about 23 ms model time; this is **not** an integrated FPS benchmark or a KITTI accuracy result.
- The 25-test suite covers numerical decoding/metrics, resize geometry, frame synchronization, archive/checkpoint safety, replay, cleanup, and real-checkpoint repeatability.
- The official Windows CARLA server download returned HTTP 403 at the release CDN and S3 origin. The server is not installed; Town04/05 pilots, real sensor timing, interruption behavior, and the static-wall depth-convention check remain unverified.
- The pinned source contains the official 697-frame Eigen list. KITTI raw image/LiDAR assets, ResNet-50 comparison weights, and NYU samples are not downloaded. KITTI's current raw-data page requires a user login; prepare those assets before the relevant later phases.

Generated previews, numerical arrays, environment diagnostics, source provenance, and baseline weight manifests remain local under ignored `runs/`, `third_party/`, `data/`, and `checkpoints/` directories. They are not staged as source changes.

## Install and run

Run from this project directory in PowerShell. Python 3.8 is deliberate for the pinned CARLA client; do not install these packages into the desktop app's bundled Python.

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.runtime\python'
uv venv --python 3.8 --managed-python --cache-dir .cache\uv .venv
uv pip install --python .venv\Scripts\python.exe --cache-dir .cache\uv --index-strategy unsafe-best-match -r requirements.txt
.\.venv\Scripts\python.exe project.py assets
.\.venv\Scripts\python.exe project.py check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe project.py infer --image data\fixtures\test_image.jpg --output runs\baseline_image
```

`assets` downloads the fixed upstream revision, retains its license, applies one recorded torchvision constructor compatibility patch, verifies the official model ZIP checksum, and records SHA-256 file identities. Image tensors stay in `[0,1]`; the upstream encoder performs its own normalization exactly once. The official test image is a baseline fixture, not a CARLA recording.

Output directories for inference, capture, and replay must be new; completed results and baseline weights are never overwritten implicitly. Use a different `--output` path for another run.

## CARLA server and pilot capture

The Python client is not the simulator server. Obtain the **official Windows 0.9.13 archive** from the [CARLA release](https://github.com/carla-simulator/carla/releases/tag/0.9.13). A local archive can be installed when network downloads are blocked:

```powershell
.\.venv\Scripts\python.exe project.py install-server
# Or install an official release ZIP already downloaded on another machine:
.\.venv\Scripts\python.exe project.py install-server --archive 'C:\path\CARLA_0.9.13.zip'
Start-Process -FilePath '.runtime\carla\CarlaUE4.exe' -ArgumentList '-quality-level=Low','-RenderOffScreen','-carla-rpc-port=2000' -WindowStyle Hidden
.\.venv\Scripts\python.exe project.py check
.\.venv\Scripts\python.exe project.py collect --map Town04 --frames 100 --output data\carla\scenes\Town04_pilot
.\.venv\Scripts\python.exe project.py collect --map Town05 --frames 100 --output data\carla\scenes\Town05_pilot
```

The local GPU has 4 GB VRAM. CARLA plus inference must be tested, not assumed feasible; a department GPU is the preferred fallback. Collection and training should not compete on this GPU. Package installation needs substantial free storage and may fail if the official host denies the download.

Capture uses synchronous 0.1-second steps, 640x480 cameras, ClearNoon, an ego autopilot, and identical RGB/depth camera mounts. It checks server/client versions, matches frames and snapshots by ID, records decoded depth and pose metadata, and marks failed recordings partial. It owns a Traffic Manager on port 8001. Exit stops/destroys owned actors, disables that manager, and restores the loaded map's previous world settings/weather. Loading Town04/05 changes the map; use a dedicated simulator session.

No live depth measurement claim is made until the sensor pilot and static-wall depth-convention check pass. Depth-camera ground truth is for evaluation, never the model input.

## Replay and live display

```powershell
.\.venv\Scripts\python.exe project.py demo --source replay --scene data\carla\scenes\Town04_pilot --frames 100 --output runs\pilot_replay
.\.venv\Scripts\python.exe project.py demo --source replay --scene data\carla\scenes\Town04_pilot --frames 100 --headless --output runs\pilot_replay_headless
.\.venv\Scripts\python.exe project.py demo --source carla --map Town04 --frames 100 --output runs\pilot_live
```

All prediction commands accept `--checkpoint <directory>` and `--device auto|cpu|cuda`. Closing the display exits cleanly. Numerical depth arrays remain separate from the fixed-scale plasma color display. Replay writes a preview, `runtime.csv`, diagnostic metrics, and summary. Live/replay use the same preprocessing and model functions.

The primary output is **relative depth**, not calibrated metres. Optional CARLA errors use explicitly labeled per-frame GT median alignment. These diagnostic metrics are not the final KITTI Eigen evaluator: the full split/crop/protocol is a later-phase deliverable. Headless throughput is distinguished from displayed throughput.

## Stable interfaces and phase continuity

To meet the requested ten-file Phase 1 limit, commands are consolidated in `project.py` instead of separate thin `scripts/*.py` wrappers. Configuration defaults live in ordinary arguments and `CaptureSettings`; the actual run configuration is written to manifests/reports. Shared source module names remain compatible with the plan. This documented C01/C09 layout adjustment must be retained in later phases, rather than adding a second pipeline.

| Planned action | Implemented command |
|---|---|
| Check environment | `project.py check` |
| Acquire source and baseline | `project.py assets` |
| Single-image prediction | `project.py infer` |
| CARLA scene collection | `project.py collect` |
| Live/replay visualization | `project.py demo` |
| Tests | `python -m unittest discover -s tests -v` |

Shared contract v1:

- `preprocess_rgb(rgb, camera)` receives native HWC uint8 **RGB**, returning float32 `1x3x192x640` values in `[0,1]` and resized 3x3 intrinsics. Its ID is `rgb_bilinear_192x640_monodepth2_internal_v1`.
- `load_model(checkpoint_dir, device)` strictly loads ResNet-18 encoder/depth files; `predict(model, tensor)` accepts batch dimension B and returns `inverse_depth` / `relative_depth` tensors `Bx1x192x640`. GT is not accepted.
- Sigmoid output converts to bounded inverse depth with model bounds 0.1/100; evaluation's 80 m limit is separate. Display uses fixed inverse-depth bounds 0.01/10, without changing numerical predictions.
- `CaptureSettings` records host, map, seed, camera geometry, timeout, and Traffic Manager port. `metadata.json` schema 1 plus `frames.csv`, `rgb/*.png`, and float32 `depth/*.npy` form a scene.
- Intrinsics follow the actual resize sampling convention. Default 90-degree image intrinsics are a diagnostic assumption when camera metadata is absent, not KITTI calibration.
- `compute_metrics` implements AbsRel, SqRel, RMSE, natural-log RMSE, and a1/a2/a3 on valid numerical depth, with explicit median alignment. Empty/invalid inputs are errors rather than zero scores.
- Baseline and future trained checkpoints live in different directories. Never replace original weights, regenerate frozen splits implicitly, or change the normalization/units without updating every caller and regression fixture.
- Run all earlier tests and reproduce the baseline before accepting each phase. Actual GPU/simulator integration checks are separate from mocks and synthetic replay tests.

## Phase 1 handoff

| Gate | Actual state |
|---|---|
| Local Python/pinned packages/CUDA | Verified |
| Trusted pretrained checkpoint and finite/repeatable prediction | Verified |
| Numerical and mock lifecycle/replay regression checks | Verified |
| CARLA server and real Town04/05 synchronized pilots | Blocked by server package download |
| Static-wall depth convention and real interruption cleanup | Not run |
| Full KITTI assets/benchmark and training | Not created; later phases |
| Independent teammate reproduction / lab GPU booking | Requires the project team |

Handoff identities are in `third_party/monodepth2/provenance.json`, `checkpoints/baseline_mono_640x192/manifest.json`, and `runs/phase1_environment.json`; the baseline observation is in `runs/phase1_baseline/summary.json`. Keep these with the local snapshot. Phase 2 may prepare independent work, but should not collect a full dataset until the real sensor/depth-geometry gate passes.

Only the ten Phase 1 implementation files are intended to be staged in this change. Existing local planning documents are left unstaged. No commit or GitHub push is performed by setup or any project command.

## Attribution

This project uses the [official Monodepth2 implementation](https://github.com/nianticlabs/monodepth2), Godard et al., ICCV 2019. The downloaded source retains its original LICENSE and copyright notices; review its non-commercial terms before redistribution. The supplied SemanticDepth repository is conceptual reference only. KITTI downloads follow the [official raw-data access requirements](https://www.cvlibs.net/datasets/kitti/raw_data.php).
