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
