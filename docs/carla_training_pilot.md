# CARLA pilot and temporal training-data preparation

The next original-scope milestone is a native CARLA 0.9.13 RGB/depth pilot on this laptop, followed by map-separated temporal inputs for self-supervised training. Simulator capture, input preparation and actual fine-tuning are separate acceptance steps.

## RGB triplet preparation

`tools/prepare_carla_triplets.py` uses the existing stored-scene audit/replay contracts. It requires complete captures, 640x480 RGB and numerical depth, valid camera intrinsics/poses, consecutive frame IDs and 0.1-second simulation intervals. Training and validation must use different maps; this conservative pilot policy prevents neighboring frames from one map being split across both sets. Scene names and source directories must also be unique.

The tool groups previous/current/next RGB frames, retaining only triplets with at least 0.01 m of recorded ego translation on each adjacent step. The threshold is a pilot parallax filter, not an accuracy metric. Recorded ego poses are used only for this filter, not as training targets. Each split must retain moving triplets. Incomplete, stationary or gapped recordings fail instead of becoming accepted training inputs.

Only RGB files are copied to the prepared dataset. `triplets.json` stores frame IDs, simulation times, camera intrinsics, image SHA256 hashes, source metadata/index fingerprints and retained/rejected counts. No reference depth arrays are included in the prepared model inputs. The source RGB copy is verified; metadata/index changes during preparation are rejected. Outputs are staged atomically, and existing datasets are never overwritten.

After native captures have passed, run:

```powershell
.\.venv\Scripts\python.exe tools/prepare_carla_triplets.py --train-scene data/carla/train_town04_01 --validation-scene data/carla/validation_town05_01 --output data/training/carla_pilot_01
```

This command prepares data only. It does not train a model, add augmentation, compute a photometric loss, recover metric scale or create new weights. Those training components and real-data acceptance remain later steps. KITTI validation remains evaluation-only and is not copied into the training split.

## Native pilot procedure

The [official 0.9.13 Windows server](https://github.com/carla-simulator/carla/releases/tag/0.9.13) is separate from the already installed Python client. The server archive is about 5.57 GiB; it and its extracted runtime remain ignored. Use the existing `project.py install-server` with the official archive, then start the local server before capture. Run only one server/model workload at a time on the 16 GB / RTX 4050 6 GB laptop.

For the first bounded smoke test, use off-screen rendering, Low quality and no sound. [CARLA's rendering documentation](https://carla.readthedocs.io/en/0.9.13/adv_rendering_options/) distinguishes off-screen rendering (camera sensors still produce images) from no-rendering mode (cameras return empty data). Do not enable no-rendering mode. Low quality has a reduced drawing distance and disabled shadows, so a successful low-quality pilot does not establish photorealistic training coverage.

Existing commands:

```powershell
.\.venv\Scripts\python.exe project.py check --carla --output runs/carla_environment_01.json
.\.venv\Scripts\python.exe project.py calibrate-depth --map Town04 --output runs/carla_depth_geometry_01
.\.venv\Scripts\python.exe project.py collect --map Town04 --frames 100 --output data/carla/train_town04_01
.\.venv\Scripts\python.exe project.py collect --map Town05 --frames 100 --output data/carla/validation_town05_01
.\.venv\Scripts\python.exe project.py audit-scene --scene data/carla/train_town04_01 --output runs/carla_train_audit_01.json
.\.venv\Scripts\python.exe project.py audit-scene --scene data/carla/validation_town05_01 --output runs/carla_validation_audit_01.json
```

Use new paths after a failed attempt. The depth geometry probe distinguishes ray distance from camera-axis depth with static off-axis facade measurements at 5/10/20 m; source sensor depth is not the model's predicted metric scale. Capture requires exact client/server 0.9.13, synchronized RGB/depth IDs, recorded poses and actor/world cleanup.

## Initial implementation acceptance

Three new regression tests cover ordered temporal grouping/calibration/hash preservation, no depth files in model inputs, map/directory separation, refusal to overwrite, and rejection of stationary/partial/gapped captures. These use explicitly synthetic fixtures and do not prove native simulator operation. The full suite passes 95 tests. Native simulator/data acceptance and project-specific training are pending until separately measured below.

## Project-local client cache fix

The first native attempt reached the server but map access tried to create `USERPROFILE/carlaCache`, which was not writable in this execution environment. Both capture and optional `project.py check --carla` now use one shared client factory that creates `.cache/carla` inside the repository, calls CARLA 0.9.13's supported `set_files_base_folder` before any map RPC, and preserves the requested host, port and timeout. A rejected cache setting fails explicitly; it never silently falls back to the home directory.

Four regression tests cover cache path/call order, cache failure, command timeout forwarding and invalid timeout rejection. The full suite passes **99 tests**. A native retry reported server version 0.9.13 and resolved the initial map through the project cache, confirming that this specific permission failure was fixed. That retry then ended in native client exit `0xC0000409` before camera frames; its zero-frame recording is not accepted. An isolated 60-second-timeout probe subsequently loaded Town04 and initialized its traffic manager successfully. A separate capture retry then timed out waiting for RGB frame 9 before cleanup aborted; delayed-camera synchronization is being investigated. No native capture or trained weights are claimed by this cache fix.


## Setup diagnostics - 25 October 2026

The official Windows 0.9.13 package is installed locally (5977917518 archive bytes, about 13.45 GiB extracted). Its locally computed archive SHA256 is `00cd8aeb56a6bbaba25fe4351918e0d0bcbd3c62eef2709337688c2e51b25c92`. This is a provenance fingerprint, not an independently published upstream checksum. The running API reported server/client 0.9.13 and listed Town04/Town05. Native connection, map loading and traffic-manager initialization have passed a bounded diagnostic; camera capture and the facade-depth gate have not yet passed. Zero-frame and interrupted recordings remain unaccepted.

The existing `CaptureSettings.timeout` is now exposed as `--timeout` on `collect`, CARLA `demo`, and `calibrate-depth`. Its default remains 15 seconds; use `--timeout 60` for the measured cold-map diagnostic. Nonpositive/nonfinite values fail validation. Increasing the timeout did not by itself produce camera frames, and the cause of the earlier native abort is not established by the successful map-only probe. The shared cache fix and timeout option preserve all model/input contracts.

## Delayed-camera fix and native evidence - 25 October 2026

GPU camera callbacks can arrive after the world has advanced. Capture now polls bounded callback queues while ticking, retaining the world snapshot and ego pose for each frame ID. Returned RGB/depth timestamps must match that historical snapshot. Startup gaps may be skipped only during explicit unrecorded warmup; recorded frames must remain consecutive. Queue overflow, duplicate callbacks, missing snapshots, missing recorded frames and timestamp mismatches fail explicitly. The facade probe also waits for images whose stamped camera pose matches the requested probe position.

Cleanup unregisters the ego vehicle from autopilot before actor destruction, then restores traffic-manager, world and weather settings. The earlier 100-frame attempt (`native_carla_pilot_08`) aborted in native world-settings restoration; it is excluded from acceptance despite having written frames. After this change, the Town04 capture returned normally and its separate audit passed. A single successful cleanup does not establish long-run native stability.

Measured on the i7-13620H / 16 GB / RTX 4050 6 GB laptop, CARLA client/server 0.9.13, off-screen Low quality, ClearNoon, seed 42, 640x480 RGB/depth and 0.1-second simulation steps:

| Check | Measured result | Local evidence (ignored) |
|---|---|---|
| Static facade geometry | Pass: 6 samples spanning 5/10/20 m; camera-axis Z maximum error 0.0615673 m against a 0.15 m tolerance; ray-distance interpretation rejected | `runs/native_carla_pilot_09/depth_geometry/summary.json` |
| Town04 capture and cleanup | Pass: 100 paired frames, IDs 281-380; all RGBs nonuniform, first/last images differ; inspected preview shows a rendered roadway | `data/carla/train_town04_09/`; `runs/native_carla_pilot_09/train_capture.log` |
| Town04 stored-scene audit | Pass: 9.90000015 simulation seconds, consecutive 10 Hz IDs/timestamps, 21,293,006 valid depth pixels below 80 m | `runs/native_carla_pilot_09/train_audit.json` |
| Town05 after Town04 | Failed: simulator connection lost, no accepted validation frames | `runs/native_carla_pilot_09/validation_capture.log` |
| Town05 on a fresh server | Failed: server connection lost before any recorded frames; map/traffic-manager files had been fetched | `runs/native_carla_pilot_10/` |
| Software regression | All 104 tests pass, including five new delayed-frame/cleanup tests | `runs/carla_latency_validation_01/tests.log` |
| Preserved inference paths | Frozen demo manifest verified; CUDA image inference on the captured Town04 RGB and 10-frame headless video smoke test completed | `runs/carla_latency_validation_01/` |

The Town04 metadata SHA256 is `779b7dafda6fcf993d09128ab3f16d9c26278fcd884c842ab54877b5b01b2bfb`; its frame-index SHA256 is `547fece7c855b294d7f61d607aa594fca5a791027be326acc9e91c7af21165c7`. These fingerprints identify local evidence, not downloadable Git assets. Capture metadata retains its original generic wall-check-pending text; the separate geometry report above establishes the measured convention for this probe without rewriting recorded source evidence.

Both multi-stage pilot summaries remain failed overall because Town05 did not complete. No map-separated triplet dataset or new trained weights were accepted. The cause of the Town05 native server exit is unresolved; longer timeouts did not fix it. Diagnose that failure before broadening capture, then audit both maps and run the existing triplet preparer with fresh output paths. Do not substitute KITTI validation examples into training or relax the map-separation requirement to bypass this gate. Low-quality daylight samples and a short facade probe do not establish weather/night robustness, long-run reliability or calibrated model predictions. Simulation Hz is not wall-clock model FPS.

## Map-disjoint native pilot accepted - 25 October 2026

An isolated Town05 probe passed map loading, traffic-manager initialization, synchronous mode and seed setup. A traced seed-42 capture then reached vehicle/camera creation and failed during warmup; the owned server exited with code 3. The precise native failure is unresolved. Changing the existing capture seed to **43** produced 100 consecutive Town05 RGB/depth frames with clean capture exit and audit. This is a measured working configuration, not a general seed-42 or server-stability fix; application source and default seed remain unchanged.

Accepted source scenes are `data/carla/train_town04_09` (Town04, seed 42, 100 frames) and `data/carla/validation_town05_13` (Town05, seed 43, 100 frames). Town05 has 17,837,428 valid depth pixels below 80 m and a 9.90000015-second simulation span. All 100 RGBs were checked for native 640x480 dimensions/nonuniformity, first/last files differ, and the first RGB was visually inspected as a rendered road scene. Its metadata SHA256 is `ffacb607d049619e7de8f69c41f27a903193660458e80c0f74baf24e551bc2db`; index SHA256 is `c96bee28f43621453bd1f717e48f52a1f4a8b7a53f70b7be85416d62cf7296f5`. Full run evidence is `runs/native_carla_pilot_13/`; the captured scene metadata records the actual seed.

The unchanged triplet preparer completed `data/training/carla_pilot_13`: **98 training triplets, 25 validation triplets, 127 unique RGB images**. It rejected 73 stationary Town05 windows and zero Town04 windows. Both adjacent steps of each retained triple pass the 0.01 m translation filter. Reference depth is absent from the prepared training inputs. Only one short route per map has been accepted; correlated windows are not independent samples, and this Low-quality daylight pilot is too small to establish training benefit or generalization. Broader data collection, pose/loss training components, fine-tuning, full benchmark evaluation and integrated CARLA inference/FPS remain pending.

Reproduce on a running server with new paths (retain the explicit seeds):

```powershell
.\.venv\Scripts\python.exe project.py collect --map Town04 --seed 42 --timeout 60 --frames 100 --output data/carla/train_town04_next
.\.venv\Scripts\python.exe project.py collect --map Town05 --seed 43 --timeout 60 --frames 100 --output data/carla/validation_town05_next
.\.venv\Scripts\python.exe tools/prepare_carla_triplets.py --train-scene data/carla/train_town04_next --validation-scene data/carla/validation_town05_next --output data/training/carla_pilot_next
```

The successful diagnostic used a fresh server for Town05, off-screen Low rendering, no sound, `-stdout -FullStdOutLogOutput`, and bounded capture. Do not assume a long-running mixed-map session has been validated. The separate [reference outputs guide](reference_outputs.md) describes the copied local evidence folder.
