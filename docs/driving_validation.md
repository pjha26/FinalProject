# Driving-scene validation and presentation readiness

## Immediate objective

Demonstrate repeatable RGB-to-relative-depth inference on the actual presentation laptop, offline, with a rehearsed fallback. A run cannot be guaranteed never to fail; readiness means checking likely failures in advance and retaining a clear recovery path. New training is not a prerequisite for this demonstration.

Local verification on 21 October 2026: 87 tests and all eight baseline delivery checks passed. The GTX 1650 displayed baseline replay completed 650 frames in 71.42 seconds (9.10 FPS at a 10 FPS ceiling). A separate Depth Anything V2 Small-294 displayed replay completed 650 frames in 77.66 seconds (8.37 FPS at the same ceiling), including a 10,000-point first-frame export. Both passed the displayed-duration gate. These are local integration observations, not a controlled speed comparison, driving-accuracy benchmark or acceptance on the friend's laptop.

## 1. Freeze and rehearse the current system

Keep the accepted source revision, weights and prepared input manifests together. Rebuild the environment on the presentation laptop, copy or prepare ignored assets, and run:

```powershell
.\.venv\Scripts\python.exe project.py check --output runs/presentation_hardware.json
.\.venv\Scripts\python.exe project.py verify-demo
.\.venv\Scripts\python.exe tools/validate_delivery.py --device cuda --model monodepth2 --display
```

Inspect CUDA availability, the actual device, and the validator's final status. `check` is diagnostic; it does not itself prove a successful demo. A displayed acceptance run needs both `status: pass` and `displayed_rehearsal: pass`. Run separately with `--model depth-anything --input-size 294` if that model will be presented. Use only a backend that has passed on this laptop, and identify it correctly in the presentation.

Repeat the actual selected demo command after disconnecting internet. Prepare the single-image fallback and retain an actual preview from the accepted run. If the live command fails, show the saved output explicitly as a recorded result. Do not silently change from webcam to video or GPU to CPU while reporting the previous configuration's performance.

Use a unique output directory each time. Demo failures after output creation save `failure.json`; these files and any partial predictions are diagnostic evidence, not accepted results. A known declared video length lets the decoder flag premature termination. Unknown frame counts still require manual/pre-run verification.

## 2. Review driving quality before changing models

Use the pinned daytime highway clip as a repeatable reference. Add legally usable clips covering urban traffic, turning, shadows, nearby vehicles, thin structures and reflective surfaces. Treat night/rain as challenge cases, not supported conditions established by the current result. Keep source attribution, licenses and hashes with each clip. Prefer full short sequences from multiple scenes over many adjacent images from one drive.

Run each model on the same clips and same frame range with separate outputs:

```powershell
.\.venv\Scripts\python.exe project.py demo --model monodepth2 --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --review-every 200 --output runs/driving_baseline_review_01
.\.venv\Scripts\python.exe project.py demo --model depth-anything --input-size 294 --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --review-every 200 --output runs/driving_small_review_01
```

For a custom local clip, replace `--video` and omit the pinned sample's `--manifest`; do not reuse a manifest for different media. A custom clip must be decoded/rehearsed beforehand. Save both terminal output and run evidence. Use `--fps 30` for a separate speed investigation, rather than interpreting a 10 FPS ceiling as the hardware maximum.

For the same reviewed frame IDs, record:

| Scene / frame | Backend | Near/far structure | Vehicle boundaries | Sky / reflections / shadows | Adjacent-frame instability | Completed FPS |
|---|---|---|---|---|---|---|
| Fill from actual saved outputs | Record actual model | Qualitative observation | Qualitative observation | Qualitative observation | Qualitative observation | From completed summary |

Inspect raw inverse-depth arrays as well as coloured previews. A clipped colour scale can hide detail; a prettier overlay cannot establish numerical correctness. Adjacent moving frames are not aligned, so their raw differences are not an objective temporal-consistency metric. Point-cloud distortions expose prediction/intrinsics limitations; do not reshape the geometry merely to resemble poster artwork.

## 3. Obtain quantitative driving-depth evidence

The existing dashcam video has no aligned reference depth. It can establish operation and runtime, but cannot produce valid driving AbsRel or RMSE.

Acquire a permitted held-out paired driving dataset. The [KITTI depth prediction resource](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) supplies a relevant RGB/depth benchmark context. Follow its dataset terms and access requirements. Do not download the complete dataset merely to repeat a small presentation clip.

Before using the existing evaluator:

1. Establish the exact dataset version, split and overlap with pretrained training data. Use identical held-out examples for both backends; do not tune and report on the same final test set.
2. Convert depth according to that dataset's documented encoding into floating-point metres. Preserve invalid/missing pixels; do not assume all PNG depth uses the Redwood millimetre conversion.
3. Verify registration to the RGB grid and matching spatial dimensions. Handle sparse reference pixels with the same valid mask for both models.
4. Create a frozen manifest compatible with `src/evaluation.py`: safe unique sample IDs, scene IDs, RGB/depth paths and SHA256 hashes, checkpoint reference, source, limitations and declared protocol. Reuse [the evaluation protocol](evaluation_protocol.md), including its explicit crop and per-image median alignment.
5. Run `project.py evaluate --manifest <prepared_manifest> --model <backend> --device cuda --output <fresh_run>` for each backend. Preserve raw predictions, per-image scores and failed cases.

Using the `garg` crop alone does not implement the complete official KITTI/Eigen evaluation procedure. Label any custom subset as a local diagnostic, not an official benchmark score. Median-aligned error uses GT during scoring; it does not make deployment outputs metric. Report performance and accuracy separately with their hardware, protocol and sample scope.

Dataset acquisition/conversion and a held-out driving benchmark are **next work**, not completed by this document. The current five-frame indoor comparison cannot decide which model is most accurate for every driving scene.

## 4. Accept improvements only with evidence

Prioritize errors found in the driving review. Compare preprocessing/input resolution and the two existing models before adding another dependency or architecture. Consider domain-specific fine-tuning only after obtaining suitable data, checking its license and splits, and establishing an untouched evaluation set.

Keep baseline weights unchanged. A candidate must pass automated tests, input checks, finite/positive numerical output checks, repeated profiling, image/video inference and a displayed rehearsal on the target laptop. Measure complete displayed FPS separately from model latency. Never keep an unvalidated optimization as the presentation default solely because it is faster in isolation.

Hardware still pending: physical webcam and the friend's RTX 3050/other presentation GPU. Native CARLA validation is optional and is not required for the recorded-video depth demonstration.
