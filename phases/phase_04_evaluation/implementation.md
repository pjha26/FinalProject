# Phase 4: Evaluation and Performance Evidence

Days 7-8 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Previous phase](../phase_03_training/implementation.md) | [Next phase](../phase_05_delivery/implementation.md)

This is an implementation/handoff plan; acceptance requires recorded evidence.

## Entry

Freeze the selected model and input manifest. The existing accuracy formulas are tested, but a complete benchmark is not implemented or completed merely by passing unit tests.

## Day 7: functionality and quality

1. Review saved RGB/depth/overlay for near/far ordering, boundaries, sky, shadows, textureless regions and reflective vehicles.
2. Inspect consecutive frames for flicker. Preserve raw numerical arrays for reproducible inspection; distinguish qualitative findings from scored accuracy.
3. Where aligned real GT exists, define mask, depth range, resolution/crop, sample count and median scaling before computing AbsRel, SqRel, RMSE, RMSE-log and delta thresholds. Do not change the protocol after seeing scores.
4. Report pretrained-domain limitations and each failed input. With no GT, mark accuracy unavailable rather than comparing the model to its own predictions.

## Day 8: integrated runtime

1. Run a minimum 60-second physical-camera or real-video demonstration on the presentation laptop, with display enabled. Note frame limit and --fps cap.
2. Retain runtime.csv/summary.json and record model latency, full-cycle distribution, wall-clock FPS, process/system RAM and total GPU memory. PyTorch peak allocated memory is only part of VRAM use.
3. Measure GTX 1650 and RTX 3050 separately if both are available. Do not extrapolate one observed FPS to all laptops.
4. Exercise Escape/close, repeated runs, missing camera and EOF. Confirm a fresh output folder is required and old evidence is preserved.

## Acceptance and handoff

Deliver actual measurements, sample counts, fixed protocols, failures and representative plots/tables. No unsupported metre, real-time, safety or training claims. Native CARLA remains explicitly unverified unless separately demonstrated.

Phase 5 receives frozen results and reproducible commands. Changes after the freeze require the targeted regression and a fresh practical run; retain prior evidence rather than overwriting it.
