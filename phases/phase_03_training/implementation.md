# Phase 3: Model Stability and Optional Fine-Tuning

Days 5-6 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Previous phase](../phase_02_data/implementation.md) | [Next phase](../phase_04_evaluation/implementation.md)

This is an implementation/handoff plan; acceptance requires recorded evidence.

## Entry

Use Phase 2's immutable baseline and representative inputs. The folder name is retained to preserve earlier links; project-specific training is optional under the revised hardware scope.

## Day 5: profile the existing model

1. Run batch-one 640x192 inference on the actual laptop. Separate model-only latency from preprocessing/display/full-cycle timing.
2. Inspect Task Manager RAM/VRAM while running. Reduce input workload to 5 FPS if needed; avoid loading a second model or CARLA alongside the baseline.
3. Repeat identical input/configuration runs; check finite depth, reproducibility, model.eval/inference_mode behavior and resource release.
4. Fix measured problems before adding an alternative framework or dependency. Preserve the accepted demonstration commands across phase transitions.

## Day 6: choose only a justified experiment

1. If the current route is usable, keep ResNet-18. A smaller model is not required merely because it has fewer parameters.
2. If profiling shows the model remains the bottleneck, evaluate an official lightweight candidate such as Lite-Mono-tiny in isolation. Pin source/license/checkpoint, use its own matching preprocessing/decoder, verify finite outputs, measure the same clip, and compare qualitative quality. Do not load its weights into the ResNet-18 loader.
3. If academic objectives require project-trained weights and suitable data exists, attempt limited fine-tuning in a separate run: batch one, compatible 640x192 inputs, validation split, and checkpoint/optimizer provenance. Self-supervised training additionally needs valid temporal RGB triplets, camera calibration and a matching pose/loss pipeline, which is not yet implemented locally.
4. Abort an optional experiment on memory failure or insufficient validation time. Retain the immutable pretrained baseline. Never claim an epoch count, training improvement or architecture integration without evidence.

## Acceptance and handoff

Deliver a selected compatible checkpoint, configuration, repeatable performance profile and comparison evidence for any attempted experiment. Rerun all previous image/video/replay tests. If no experiment completes, declare pretrained inference and continue to evaluation; do not relabel baseline weights as newly trained.

Phase 4 evaluates the selected frozen model. Any loader/preprocessing change must pass the shared contracts before adoption; rollback uses the original baseline and command.

## Implemented and validated on 18 October 2026

The local profiling/stability track is implemented. Fine-tuning and smaller-model integration were not performed: the existing baseline is usable and no calibrated temporal training dataset/local training pipeline is prepared. The selected checkpoint remains the original monocular ResNet-18, with its existing source/weight hashes and preprocessing.

### Reproducible profiler

```powershell
.\.venv\Scripts\python.exe project.py profile --device cuda --frames 60 --warmup 5 --repeats 2 --output runs/profile_review_01
```

- Verifies the Phase 2 manifest and baseline before loading the model. Bounds the diagnostic to 1-300 measured frames, 0-30 warmup frames and 2-5 repeats.
- Warms decoding, inference, rendering and numerical checks; excludes warmup from statistics. Each repeat processes the same video frames sequentially, without an FPS cap or display window.
- Measures decode/RGB conversion/resize, model time, complete inference processing, composite rendering, validation and total headless cycle. Records mean, median, p95, minimum and maximum. Processing time includes model time; these fields must not be summed as independent stages.
- Requires float32 192x640 positive finite depth/inverse-depth maps with reciprocal consistency. Compares exact per-frame fingerprints across repeats; a mismatch produces `needs_review` and a failing command exit.
- Writes runtime CSV, summary JSON, first measured preview and reference fingerprints to a fresh output directory. Records GPU/software/input identity, process working-set memory on Windows, and PyTorch peak allocated GPU memory.

### Measured change and evidence

Profiling identified composite rendering as a local bottleneck. Cache the palette, font and static labels, use palette lookup instead of a full float64 RGBA image, and blend via installed OpenCV. No model-weight or numerical-depth changes. Overlay rounding may differ by one uint8 level; the colour palette remains identical.

- Isolated warmed rendering check, 20 measured calls: median **55.06 ms before**, **31.49 ms after**. This is one local observation, not a general performance guarantee.
- Final actual CUDA profile: two repeats of 60 measured frames, five warmup frames per repeat; **15.75 and 16.00 FPS uncapped headless**, median model **22.40 ms**, rendering **22.52 ms**, headless cycle **61.30 ms**. All repeated predictions matched exactly; fingerprints also match the pre-optimization profile.
- Final profile process peak working set **631.95 MiB**; PyTorch peak allocated GPU memory **106.34 MiB**. These are not total system RAM/VRAM figures.
- **45 tests passed**, including all Phase 1/2 regression checks. Fresh CUDA image inference and a displayed 30-frame manifest-based video smoke test passed after the rendering change.
- Local evidence: `runs/phase3_profile_final/`, `runs/phase3_tests.json`, `runs/phase3_render_before.json`, `runs/phase3_render_final.json`, and `runs/phase3_transition_image/` / `runs/phase3_transition_display/`.

### Handoff to Phase 4

Keep the accepted baseline, preprocessing and Phase 2 input manifest unchanged. Reuse the profiler to describe runtime, then prepare a documented aligned-GT evaluation before making accuracy claims. Current profile FPS excludes display, startup, output writes, warmup and pacing; it does not prove sustained webcam/CARLA performance or the original integrated 10 FPS target. RTX 3050, physical webcam and native CARLA sensor validation remain pending.

The implementation PDF is `output/pdf/implementation_progress_report.pdf`; its generator is `tools/build_implementation_report.py` and uses a separate documentation Python with ReportLab/Pillow/pypdf. No PDF tooling was added to application requirements. The user authorized staging, committing and pushing this Phase 3 code/report update in no more than three commits.
