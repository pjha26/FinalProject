# Accuracy investigation and improved depth backend

Date: 19 October 2026. The original pretrained Monodepth2 baseline remains available. `--model depth-anything` selects the new, independently pinned Depth Anything V2 Small backend. No weights are trained or fine-tuned by this project.

## Later driving-depth evidence - 24 October 2026

The [KITTI cropped-validation pilot](kitti_pilot.md) now adds 24 RGB/depth examples from eight drives, with source/split audits and matched scoring. Monodepth2 AbsRel is 0.093627 and Small-294 AbsRel is 0.094203; Small has better delta accuracies. This differs from the older indoor comparison below and does not establish a universal winner. The new result is a local diagnostic, not official Eigen scoring or completed training. The full 92-test suite passed.

## Bugs and limitations investigated

| Finding | Resolution |
|---|---|
| RGB channel order, normalization or reciprocal-depth calculation might explain poor predictions | Checked the existing loader/preprocessor against official Monodepth2 inference. RGB order, encoder normalization, strict weight loading and disparity-to-inverse-depth calculation are consistent. No channel swap or inverted near/far calculation was found. |
| Baseline uses bilinear resizing while the official image inference uses Lanczos | Tested as an isolated exploratory variant. Lanczos helped the five-frame diagnostic, but did not close the gap to the new model. The frozen baseline preprocessing/manifest remains unchanged as the comparison control. |
| Live/replay GT metrics used a different interpolation implementation from the evaluator | Both now call `score_inverse_depth`, sharing inverse-depth resize-before-inversion, masking, median alignment and clipping. A regression test checks equal metrics. |
| The fixed-size live window stretched images | The composite is now fitted into the window with its aspect ratio preserved and unused space filled. Saved preview geometry is unchanged. |
| Stability validation assumed every backend outputs 192x640 maps | Monodepth2 still does. Depth Anything retains native RGB output geometry. Fingerprints now validate matching positive finite 2D maps and include non-baseline shapes; original baseline fingerprints remain compatible. |
| Using the old preprocessing on a new model would invalidate accuracy | The new backend uses official aspect-preserving bicubic resize, multiples of 14 and ImageNet normalization. It receives RGB directly and honors the requested CPU/CUDA device, rather than calling upstream's auto-device image helper. |
| Aspect-preserving resize of an extreme panorama could exhaust a small GPU | The new adapter rejects inputs whose resized long edge would exceed 1024 pixels, before allocating the network tensor. Standard driving and reference inputs fit; try a smaller input size for wider images. |
| Depth Anything can predict exact zero inverse depth for sky | Negative/nonfinite predictions are rejected. Exact zero is floored at 1e-6 solely for finite reciprocals, with the floor and last-frame pixel count recorded. These pixels are not removed from accuracy evaluation. |
| Reusing a frozen manifest could mislabel a comparison checkpoint | The reference/input manifest retains baseline provenance. Every new-backend run separately records its actual weight SHA256, source revision, preprocessing ID, input geometry and backend. GT remains scoring-only. |

The new model does not estimate semantic labels, detect obstacles or control a vehicle. Stronger colours and sharper boundaries do not establish calibrated distance accuracy.

## Reproduce the comparison

Use the existing pinned Python environment; there are no added inference dependencies. First-time preparation needs Internet. Prepared inference/evaluation is offline:

```powershell
.\.venv\Scripts\python.exe project.py assets
.\.venv\Scripts\python.exe project.py prepare-demo
.\.venv\Scripts\python.exe project.py prepare-reference-eval
.\.venv\Scripts\python.exe project.py assets --model depth-anything
.\.venv\Scripts\python.exe tools/compare_accuracy.py --device cuda --output runs/my_accuracy_comparison_01
```

The comparison evaluates both model families on exactly the same five RGB/GT pairs using the existing [protocol](evaluation_protocol.md). It compares Small input settings 294, 392 and 518. Every configuration also profiles the same 30 driving frames twice after three warmups, and predicts the same verified driving frame 200. Runs execute sequentially to avoid keeping multiple networks in GPU memory. Failed/incomplete runs cannot report complete comparison results. Code, requirements, inputs, reference manifest and selected checkpoint identities are recorded.

Input sizes are short-edge targets, with aspect ratio preserved and dimensions rounded to multiples of 14, subject to the 1024-pixel long-edge budget. They do not mean square images. Output arrays are interpolated back to RGB geometry; interpolation does not create additional learned detail. The baseline remains 640x192.

The five samples are correlated indoor reference frames from one scene. They are **an exploratory diagnostic**, not a representative driving test set or an independent test of settings selected from these results. Higher input resolution is not guaranteed to reduce this particular metric. The separate driving video has no GT, so its previews support qualitative inspection only.

## Measured comparison on the local GTX 1650

All four configurations completed in `runs/accuracy_comparison_01/`, with matching repeated prediction fingerprints. Accuracy uses all five pairs, the same valid-pixel mask and per-image GT median alignment. RMSE below is **after alignment**, not a deployed metre-scale prediction.

| Configuration | Actual driving network input (H x W) | AbsRel, lower is better | Aligned RMSE (m), lower is better | Delta1, higher is better | Uncapped headless FPS, two repeats | Peak PyTorch allocation (MiB) |
|---|---|---:|---:|---:|---|---:|
| Original Monodepth2 | 192 x 640 | 0.4960 | 1.4404 | 0.3763 | 17.69 / 18.08 | 106.34 |
| Depth Anything Small, 294 | 294 x 518 | **0.2033** | **0.5202** | **0.5986** | **8.22 / 8.62** | 277.17 |
| Depth Anything Small, 392 | 392 x 700 | 0.2493 | 0.6202 | 0.4670 | 4.99 / 5.30 | 315.07 |
| Depth Anything Small, 518 | 518 x 924 | 0.3463 | 0.8671 | 0.3487 | 2.21 / 2.47 | 413.87 |

The 294 setting reduced diagnostic AbsRel by approximately **59.0%** and aligned RMSE by **63.9%** relative to the original baseline. It was the fastest and best-scoring **new-model configuration in this limited comparison**. This does not establish that it is the most accurate setting on other scenes. The larger settings remain optional experiments, not automatic quality upgrades.

Runtime scope: 30 measured driving frames per repeat, two repeats, three warmups excluded, batch one, two CPU threads, FP32, no xFormers, same source RGB and same shared renderer. Full headless cycles include decode, preprocessing, network prediction, output transfer, rendering and numerical validation; they exclude display/pacing, setup and file writes. Different input sizes change computation. PyTorch allocation is not total VRAM, and process working set is not total system RAM. Friend-laptop performance cannot be inferred from this table.

The driving image comparison is saved locally as `runs/accuracy_comparison_01/model_before_after.png`. Roadside silhouettes and sky separation look better in the inspected new-model frame, but no driving accuracy score is available. Each model's inverse-depth scale is arbitrary; equal colours across model families do not imply equal distance. No GT-guided colour tuning, semantic masking or artificial object outlines were added.

## Final integration acceptance

The final source passed **74 regression tests** and all **eight integrated quality-backend checks** in `runs/accuracy_quality_final/`: input verification, environment, image inference, repeated numerical stability, paired evaluation, video smoke and displayed rehearsal, in addition to the test suite. Source hashes matched throughout. The original baseline separately passed its seven software checks in `runs/accuracy_baseline_delivery/`; its display gate was not requested in that run. Original commands and baseline checkpoint/preprocessing remain available.

The final quality-backend display processed **650 frames in 67.99 seconds at 9.56 FPS**, under a 10 FPS cap, and passed the 60-second minimum. Actual input was 294x518, with 360x640 numerical output corresponding to the bounded source RGB. Peak process working set was **716.24 MiB** and peak PyTorch GPU allocation **277.17 MiB**; these do not measure total system RAM or total VRAM. An earlier rehearsal is retained separately under `runs/accuracy_quality_delivery/`. Runtime varies with system load; these display observations are not a controlled speed comparison with the shorter profiles above.

The final paired evaluator reproduced all seven Small-294 diagnostic scores in the comparison table. Four comparison configurations also passed the explicit reference/driving identity and input-budget checks, recorded in `runs/accuracy_comparison_01/identity_checks.json`. GT was used only for scoring. Private/raw media, model weights, external source and runtime evidence remain ignored by Git; source, tests and documentation are the versioned deliverables.

The actual CPU saved-image path also completed separately in `runs/accuracy_cpu_image/`, with finite reciprocal maps at the source image's 720x1280 output geometry. It is a fallback check, not a CPU real-time acceptance result. No additional inference dependencies were installed.

## Model and asset provenance

- Official source: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
- Model: Small / ViT-S, 24.8M parameters, relative depth. The project retains the upstream Apache-2.0 license.
- Official checkpoint: [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small), revision `03876f8651c73a60fe4c2c48294e09fcb6838fcf`.
- Weight SHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` (99,218,434 bytes).
- Source files and weights are verified against constants **before** downloaded code is imported or weights are loaded. Weights use `torch.load(..., weights_only=True)` and strict state loading.
- Downloads remain in ignored `third_party/depth_anything_v2/` and `checkpoints/depth_anything_v2_small/`; retain licenses/manifests when transferring machines. The default installation uses the existing PyTorch fallback without xFormers.

## Use the improved model

```powershell
.\.venv\Scripts\python.exe project.py demo --model depth-anything --input-size 294 --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --min-seconds 60 --review-every 200 --output runs/quality_demo_01
.\.venv\Scripts\python.exe project.py infer --model depth-anything --image data/demo/driving_first_frame.png --device cuda --output runs/quality_image_01
.\.venv\Scripts\python.exe tools/validate_delivery.py --model depth-anything --input-size 294 --device cuda --display --output runs/quality_delivery_01
```

The existing `infer`, `profile`, `evaluate` and `demo` commands accept the new model selection. Omit `--model` to run the original Monodepth2 baseline. `--checkpoint` continues to select Monodepth2 weights; combining a custom baseline checkpoint with the new model is rejected. All frozen input/reference verification remains in place, with actual selected-model identity recorded separately.

Try 392 or 518 only after measuring the presentation machine. Preserve each run in a fresh directory. The larger settings increase computation and may run slower than the video source. CPU fallback remains available but will be slower. The RTX 3050 laptop and physical camera still require their own rehearsal.

## Remaining accuracy work

Independent multi-scene driving-depth evaluation is still needed. No finite small comparison can establish the globally best model. Dataset contamination/training overlap has not been audited. Relative models can misestimate reflective/transparent surfaces, thin objects and unusual scenes; they do not provide deployment-calibrated metres or temporal consistency guarantees. Calibrated distances, training, native CARLA acceptance and vehicle control are separate work.
