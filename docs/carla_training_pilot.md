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
