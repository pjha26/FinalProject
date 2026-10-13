# Phase 2: Real Inputs and Demo Data

Days 3-4 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Previous phase](../phase_01_foundation/implementation.md) | [Next phase](../phase_03_training/implementation.md)

This is an implementation/handoff plan; acceptance requires recorded evidence.

## Entry

Phase 1 numerical image/video gates must pass. Independent clip preparation can start earlier. Maintain the same source interface and frozen model/preprocessing identifiers.

## Day 3: prepare representative inputs

1. Obtain a short permitted driving video with road, buildings, vehicles and camera motion; retain source, license, duration, resolution and file SHA256 in a local input manifest.
2. Test camera permission/index and lighting on the actual presentation laptop. Use driving footage to demonstrate the model's training domain; use indoor webcam depth as an explicitly limited generalization demo.
3. Select a repeatable 60-second review sequence. Avoid uploading private webcam footage. Keep video/fixtures under ignored `data/`.
4. Prepare an offline image and local-video fallback. Confirm no CARLA server or network service is needed during the demonstration.

## Day 4: validate the input contract

1. Check EOF, corrupted/unreadable inputs, resolution changes, aspect ratios and unavailable cameras. A video ending normally must produce a complete summary; read failures must produce a clear error and release resources.
2. Check first-frame RGB/depth arrays and display colors against a known image. Ensure colorization never changes numerical maps.
3. Rehearse without CARLA on the 8 GB laptop, then on the friend's 16 GB RTX 3050 if available. Record results separately.
4. If accuracy evaluation is feasible, acquire a small documented aligned RGB/depth subset with calibration and scene-level split. Do not pretend RGB-only videos provide metric GT.

## Acceptance and handoff

Deliver a frozen input manifest, working demo command, qualitative examples, fallback image/video, and a real 60-second camera/video run. Provide any GT dataset protocol and actual sample count, or clearly mark accuracy pending. No mandated tens-of-thousands-frame CARLA collection.

Phase 3 uses the same baseline/input files and output schema. Keep private media out of Git. A new input must not alter checkpoint normalization or model resolution.

## Implemented on 14 October 2026

- `project.py prepare-demo` acquires the pinned Udacity driving clip and repository MIT license, checks trusted file hashes, decodes every frame, writes a first-frame RGB fallback and freezes `data/demo/input_manifest.json`.
- Actual input: 1,260 frames, 1280x720, 25 FPS, 50.4 source seconds. Playback below 25 FPS is intentionally slowed sequential replay; a minute of application wall time does not imply a minute of unique source footage.
- `project.py verify-demo` checks video/license/image identity, actual decode count/rate/dimensions, preprocessing and baseline checkpoint hashes. Relative filenames keep the manifest usable on another laptop.
- `demo --manifest` validates the selected video/model pair before inference and records the manifest checksum. Existing unmanifested webcam/video, image and optional CARLA/replay commands remain compatible.
- Tests cover portable manifest reuse, altered media/model/metadata, unsafe paths, incomplete decoding, resolution changes and a mismatched selected video.

```powershell
.\.venv\Scripts\python.exe project.py prepare-demo
.\.venv\Scripts\python.exe project.py verify-demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe project.py demo --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 600 --fps 10 --output runs/driving_review_01
.\.venv\Scripts\python.exe project.py infer --image data/demo/driving_first_frame.png --device cuda --output runs/driving_fallback_01
```

Transfer current source plus the pinned model/source and `data/demo` folders; rebuild `.venv`, verify inputs and rehearse on the friend's laptop. See README for the transfer checklist. The user authorized staging and a local Phase 2 commit; pushing this phase requires separate authorization.

Local validation passed on 14 October 2026: **39 regression checks**, plus **350 displayed real-video CUDA inferences over 71.58 seconds** on the 8 GB GTX 1650 PC. Observed throughput was **4.89 FPS under a 5 FPS ceiling**; peak process working-set RAM was **620.13 MiB**, and peak PyTorch allocated GPU memory was **106.34 MiB**. These memory figures are not total system RAM/VRAM. The processed frames cover 14 source seconds. Evidence: ignored `runs/phase2_real_driving_rehearsal/validation.json`, runtime CSV, summary, preview and numerical arrays. Reproduce using `--frames 350 --fps 5` with a fresh output folder.

**Acceptance status:** local input preparation, integrity checks and displayed real-video rehearsal passed. Presentation-machine acceptance remains pending: rebuild/verify/rehearse on the friend's RTX 3050 laptop. The physical webcam is also unverified. No aligned GT assets are provided by this video, so metric accuracy is explicitly unavailable.
