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
4. Fix measured problems before adding an alternative framework or dependency. Preserve the working command for the 15 October prototype review.

## Day 6: choose only a justified experiment

1. If the current route is usable, keep ResNet-18. A smaller model is not required merely because it has fewer parameters.
2. If profiling shows the model remains the bottleneck, evaluate an official lightweight candidate such as Lite-Mono-tiny in isolation. Pin source/license/checkpoint, use its own matching preprocessing/decoder, verify finite outputs, measure the same clip, and compare qualitative quality. Do not load its weights into the ResNet-18 loader.
3. If academic objectives require project-trained weights and suitable data exists, attempt limited fine-tuning in a separate run: batch one, compatible 640x192 inputs, validation split, and checkpoint/optimizer provenance. Self-supervised training additionally needs valid temporal RGB triplets, camera calibration and a matching pose/loss pipeline, which is not yet implemented locally.
4. Abort an optional experiment on memory failure or insufficient validation time. Retain the immutable pretrained baseline. Never claim an epoch count, training improvement or architecture integration without evidence.

## Acceptance and handoff

Deliver a selected compatible checkpoint, configuration, repeatable performance profile and comparison evidence for any attempted experiment. Rerun all previous image/video/replay tests. If no experiment completes, declare pretrained inference and continue to evaluation; do not relabel baseline weights as newly trained.

Phase 4 evaluates the selected frozen model. Any loader/preprocessing change must pass the shared contracts before adoption; rollback uses the original baseline and command.
