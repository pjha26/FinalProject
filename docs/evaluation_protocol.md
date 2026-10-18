# Phase 4 evaluation protocol

Prepared 18 October 2026. Freeze this protocol before inspecting scores. Evaluation data, predictions and runtime evidence remain under ignored `data/` and `runs/`; this document records the reproducible procedure and limits.

## Selected model and task

The original `mono_640x192` Monodepth2 ResNet-18 checkpoint remains unchanged. It receives only RGB, resized through the existing preprocessing to 640x192. Reference depth is read separately for scoring; it never enters the model. This phase performs evaluation, not fine-tuning or metric-scale recovery.

## Paired reference input

`project.py prepare-reference-eval` downloads the five-frame `SampleRedwoodRGBDImages` archive hosted by Open3D. The official [sample description](https://www.open3d.org/html/python_api/open3d.data.SampleRedwoodRGBDImages.html) identifies five colour/depth pairs from living-room1. The [RGB-D tutorial](https://www.open3d.org/docs/0.17.0/tutorial/geometry/rgbd_image.html) documents registered images and integer-millimetre depth storage. The preparation command reads the PNG with Pillow and divides by 1000 into float32 metre-valued NPY arrays. Open3D itself is not a project dependency.

- Source: `https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/SampleRedwoodRGBDImages.zip`
- Archive SHA256: `4f161ba04d81cbe628fc69dac20552749f1b255b0090fe51c766e5d02355937b`.
- Upstream MD5 cross-check: `43971c5f690c9cfc52dda8c96a0140ee`, declared in [Open3D v0.18.0 source](https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/data/dataset/SampleRedwoodRGBDImages.cpp).
- Sample IDs `00000` through `00004`; one indoor scene; split `diagnostic`; data kind `rgbd_reference`.
- Five correlated reference images are not an independent, representative driving test set. These results cannot be compared directly to KITTI/NYU benchmark targets or used to claim real-world driving accuracy.
- No raw reference data is redistributed in Git. The saved local manifest retains provenance, conversion notes and SHA256 hashes of each RGB/depth asset and selected checkpoint.

## Fixed numerical protocol

1. Validate schema, declared metre units, alignment, safe unique sample IDs, scene labels, asset hashes and selected checkpoint identity. Reject missing, changed, duplicated or out-of-root assets.
2. Require HWC RGB and a matching floating-point 2D reference-depth array. Missing reference values are masked; an image with no valid evaluation pixels fails rather than silently disappearing from the sample count.
3. Run the existing model in evaluation/inference mode. A first-image warmup is excluded from logged inference timing. Save raw relative-depth predictions independently of aligned scores.
4. Resize numerical inverse depth to the reference image size using OpenCV bilinear interpolation, then take its reciprocal. Do not resize a colour map or invert before interpolation.
5. The reference diagnostic uses no spatial crop. The optional `garg` crop uses integer bounds `[0.40810811*H, 0.99189189*H, 0.03594771*W, 0.96405229*W]`, matching the crop in the pinned upstream Monodepth2 Eigen-evaluation code. Supporting the crop does not by itself implement the full official KITTI test protocol.
6. Keep only finite GT pixels with `0.001 < GT < 80` metres, after the spatial crop. Require finite positive predictions at valid pixels.
7. Compute a separate median scaling ratio for each image: `median(GT_valid) / median(prediction_valid)`. Apply it to predictions, then clip predictions to `[0.001, 80]` metres for metric computation.
8. Compute AbsRel, SqRel, RMSE, natural-log RMSE, and delta accuracies at 1.25, 1.25 squared and 1.25 cubed. Average each metric equally across images, not by pooling all valid pixels. Record individual ratios, valid-pixel counts and per-image metrics.
9. A failed run is marked `partial` with no aggregate accuracy. Synthetic unit fixtures are labelled `synthetic_test_fixture` and are never mixed with the reference diagnostic.

Median alignment uses GT information unavailable during ordinary deployment. An aligned RMSE in metres does not mean the model predicts calibrated metre-scale distance.

## Run the diagnostic

```powershell
.\.venv\Scripts\python.exe project.py prepare-reference-eval
.\.venv\Scripts\python.exe project.py evaluate --device cuda --output runs/reference_evaluation_01
```

Outputs: `summary.json`, `per_sample.csv`, numerical relative-depth NPY files and prediction previews. Use a new output directory for every run. To evaluate separately acquired paired data, provide its compatible frozen manifest via `--manifest`; retain source, limitations, split, protocol, sample IDs/scenes, hashes and aligned metre-valued depth arrays. Training data must not be presented as a held-out test set.

## Integrated runtime and visual inspection

```powershell
.\.venv\Scripts\python.exe project.py demo --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --min-seconds 60 --review-every 200 --output runs/driving_evaluation_01
```

The gate requires a displayed run with at least 60 seconds of measured frame-loop time. A headless run cannot satisfy it; ending a nonempty run below the threshold saves `needs_review` and returns a failure. Closing before any frames are processed fails with `No frames processed`. The 10 FPS setting is a ceiling, not a guaranteed minimum. Native video is 25 FPS; processing sequentially at a lower rate slows replay rather than dropping frames.

Runtime summaries retain wall-clock FPS, model/processing/arrival-to-display/full-cycle mean, median and p95 timings, configured cap, device, selected input manifest, process working set, and PyTorch peak allocated memory. Frame-loop time includes initial-frame overhead, periodic snapshot writes and pacing; model loading, manifest verification and final report exports are excluded. Processing includes model time, so those fields overlap. PyTorch allocations are not total GPU memory, and working set is not total system RAM. Use Task Manager for total system/driver memory on the presentation laptop.

`--review-every 200` saves the first two frames, periodic adjacent pairs and the last processed frame. `review.json` points to RGB, predicted inverse-depth arrays and composite PNGs. Inspect broad near/far structure, boundaries, shadows, sky and textureless/reflective surfaces. Predictions can change because the camera and objects move; unregistered adjacent frames are not an objective flicker/accuracy metric. This driving clip has no aligned reference depth, so its accuracy remains unavailable.

## Phase transition gate

Keep the accepted checkpoint and Phase 2 input identity frozen. Run all regression tests, verify-demo, image inference, repeated profiling and a displayed runtime rehearsal before accepting changes. Retain failures and old results. Full KITTI validation, the friend's RTX 3050, physical webcam and native CARLA sensor acceptance are separate gates.
