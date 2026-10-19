# Final demonstration and laptop handoff

The project estimates **relative depth from RGB alone** using the official pretrained Monodepth2 ResNet-18 `mono_640x192` model. The supported practical demonstration is real driving-video replay with depth computed for each frame. It displays RGB, estimated depth, overlay and measured FPS. It is a perception prototype, without vehicle control or calibrated distances.

## Architecture and completed phases

```mermaid
flowchart LR
    A[Phase 2: verified driving video / image fallback] --> B[Phase 1: RGB preprocessing, 640 x 192]
    B --> C[Phase 1: pretrained depth inference]
    C --> D[Phase 3: shared optimized visualization]
    D --> E[RGB / depth / overlay + numerical maps]
    C --> F[Phase 4: paired-depth diagnostic]
    D --> G[Phase 3/4: repeatability / runtime / review snapshots]
    E --> H[Phase 5: integrated validation and presentation]
    F --> H
    G --> H
```

The existing image, webcam, video and optional CARLA/replay commands keep their interfaces. Phase 5 runs those accepted components rather than introducing another inference pipeline.

## Prepare the presentation laptop

1. Clone the published project and use the same commit as the development machine. Install **Python 3.8.20 x64** and a compatible NVIDIA driver. Plug the laptop into power.
2. Rebuild the environment from PowerShell in the project root. Run each line only after the preceding command succeeds:

   ```powershell
   py -3.8 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe project.py assets
   .\.venv\Scripts\python.exe project.py prepare-demo
   .\.venv\Scripts\python.exe project.py prepare-reference-eval
   ```

3. Complete downloads before the demonstration. Alternatively, copy these **complete folders** from this PC into matching locations after cloning:

   | Folder | Purpose |
   |---|---|
   | `checkpoints/baseline_mono_640x192/` | Original weights and hash manifest |
   | `third_party/monodepth2/` | Pinned model source, provenance and original license |
   | `data/demo/` | Driving clip, image fallback, manifest and Udacity license |
   | `data/fixtures/` | Official saved-image fixture used by the regression suite |
   | `data/evaluation/redwood/` | Frozen five-pair reference data and evaluation manifest |

   Keep all provenance/license files. Do not copy `.venv`, `.runtime`, `.cache`, private camera captures or unfinished CARLA scenes. The weights, media and original run folders are deliberately absent from Git. Transfer desired evidence separately, including its full run directory. Offline inference requires prepared assets and a working environment; first-time installation still needs packages or a separately prepared wheel cache.

4. Validate on **that laptop**, using a fresh output directory:

   ```powershell
   .\.venv\Scripts\python.exe tools/validate_delivery.py --device cuda --display
   ```

   This sequential command runs all regression tests, verifies the driving manifest/checkpoint, checks the environment, predicts an image, repeats a short profile, evaluates the five reference pairs, performs a headless video smoke test and requires at least 60 seconds of displayed video. It uses the interpreter that launches it, makes no downloads and writes timestamped outputs under `runs/`. Use `--output runs/my_delivery_01` to choose an unused directory. Closing the window early fails the displayed gate. Without `--display`, software checks can pass while the display gate remains **pending**.

5. Check the final `summary.json`: `status` and `displayed_rehearsal` should both be `pass`. Inspect `tests.log`, `environment.json`, `profile/summary.json`, `evaluation/summary.json` and `display/summary.json`. A failure stops subsequent steps and preserves logs. Fix the failure and retry into a new directory. Source/requirements hashes must remain unchanged during the run.

On the local GTX 1650, a separate Python 3.8.20 environment was rebuilt from the pinned requirements using the local package cache. The friend's RTX 3050 results must be recorded separately; local measurements do not establish its acceptance.

**Recorded local acceptance (19 October 2026):** `runs/phase5_delivery/summary.json` records all eight steps and displayed acceptance as passing. All 61 regression tests passed; the short profile's predictions matched exactly across repeats; image/video inference and the five-pair evaluator completed. The displayed replay processed 650 frames in 82.55 seconds at 7.87 FPS under a 10 FPS cap. Peak process working set was 540.00 MiB and PyTorch peak GPU allocation was 106.34 MiB; neither measures total system memory. Source hashes were unchanged during validation. This establishes local software acceptance without claiming 10 FPS for this displayed run.

## Run the actual presentation

Prepare the environment and assets first. Use unused output names; the existing commands refuse to overwrite a run.

```powershell
.\.venv\Scripts\python.exe project.py demo --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --min-seconds 60 --review-every 200 --output runs/presentation_video_01
```

Show the moving RGB view, changing depth map and overlay. Explain that the colours use a fixed scale of predicted inverse depth: they do not represent measured metres. Sky, reflective objects and fine boundaries can be inaccurate. At the 10 FPS cap the 25 FPS source plays more slowly; the measured displayed rate is in `summary.json`, including display-loop work and pacing. This command processes 26 seconds of source footage over at least a minute of wall time.

Press Escape or close the window to stop. For the acceptance run, allow the command to finish; early closure may fail the minimum-duration gate. To repeat, change the output suffix.

**Optional physical webcam**, only after separately checking permissions and camera index:

```powershell
.\.venv\Scripts\python.exe project.py demo --source webcam --camera 0 --device cuda --frames 650 --fps 10 --min-seconds 60 --output runs/presentation_camera_01
```

Move the camera slowly and show changing predictions. Indoor scenes differ from KITTI driving training data. If the camera cannot open, use the accepted video command. Never describe the video fallback as a successful live webcam or CARLA test.

**Saved-image fallback**, if video decoding/display fails:

```powershell
.\.venv\Scripts\python.exe project.py infer --image data/demo/driving_first_frame.png --device cuda --output runs/presentation_image_01
```

Open the resulting `preview.png`. The output also contains numerical inverse-depth and relative-depth arrays. If CUDA is unavailable, repeat into a new directory with `--device cpu` and expect slower processing. Lower `--fps` for video if needed; do not claim a 10 FPS result under a 5 FPS cap.

## A short explanation for the examiner

“An ordinary camera records colour but does not directly measure distance. My project uses a pretrained monocular depth network to estimate a relative depth map from each RGB frame. I implemented a reproducible, low-memory inference pipeline, verified driving inputs, visualization, profiling and evaluation, then checked the phases together. The demonstration computes depth as the video runs. It supports perception research; it does not control a vehicle or give calibrated distances.”

If asked about contribution, distinguish the pretrained model from your input preparation, integrity checks, integration, visualization optimization, measured validation and handoff tooling. Explain that no project-specific training has been completed.

If asked about accuracy, report the five-frame indoor diagnostic separately: AbsRel about **0.4960**, median-aligned RMSE about **1.4404 m**. These results expose substantial error; they are neither KITTI benchmark scores nor proof of driving accuracy. Ground-truth median alignment is available only for scoring, not deployment. The driving video has no aligned GT. See [evaluation protocol](evaluation_protocol.md).

## Final handoff and remaining work

Deliver the published source, five implementation files, this guide and the implementation PDF. Keep the accepted input/checkpoint manifests and full local validation evidence with the prepared offline assets. Re-run validation after code, model, preprocessing or input changes; retain the accepted baseline while experimenting elsewhere.

The local software delivery track covers Phases 1-5. Still pending: RTX 3050 rehearsal, physical webcam acceptance, successful native CARLA capture, full driving-depth benchmark, project-specific training, calibrated metre-scale inference and final academic submission. Vehicle control is outside this project's scope. Rehearsing on the actual presentation laptop is the last hardware gate, not an assumed result.
