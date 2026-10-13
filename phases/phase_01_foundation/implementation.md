# Phase 1: Foundation and Low-Memory Inference

Days 1-2 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Next phase](../phase_02_data/implementation.md)

This is an implementation/handoff plan; acceptance requires recorded evidence.

## Entry and decisions

Reuse the existing environment, checksum-checked monocular weights, and preprocessing/visualization. The target is GTX 1650 through RTX 3050 with 4 GB VRAM. The latest user hardware instruction controls scope; CARLA and the earlier PDF dataset/training thresholds are no longer blocking requirements.

## Day 1: preserve the working baseline

1. Check Python, CUDA, GPU memory, driver and free RAM with `project.py check` and Task Manager.
2. Keep Monodepth2 ResNet-18 at 640x192, batch one. Validate checkpoint/source hashes and image output before editing integration code.
3. Preserve RGB input order, [0,1] values, internal encoder normalization, and numerical depth versus display colors.
4. Keep the first baseline preview/arrays and hardware report in ignored run folders. Record the pretrained model provenance in the report.

## Day 2: lightweight live input

1. Add OpenCV webcam/local-video reading to the existing demo; retain CARLA and scene replay choices.
2. Read one frame at a time, bound the display input to 640x480, default to a 10 FPS ceiling, and keep the model at 640x192.
3. Display RGB/depth/overlay, source type and measured completed FPS. Escape/close must release the camera and window.
4. Save first-frame preview and numerical maps, per-frame timing, summary and explicit unavailable-GT status.
5. Test file EOF, invalid paths, RGB channel order, resizing, missing cameras, disconnects, early release, and existing scene replay.
6. Run the actual checkpoint on a decoded test clip. Label any synthetic fixture as such; then rehearse the physical webcam or real driving clip.

## Acceptance and handoff

Run the regression suite plus image/video smoke commands from README. Provide a fresh output directory, immutable baseline manifest, known-working command, test result, measured hardware details, and an explicit list of unverified hardware paths. A physical-camera demo is accepted only after actually running it; a mocked camera test is insufficient.

Do not block this phase on CARLA, training, a full KITTI download or a new model architecture. On failure use the saved image demo while repairing the video/camera route. Do not replace baseline weights. Phase 2 consumes the same CLI and output contract.
