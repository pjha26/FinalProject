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

## Implemented on 18 October 2026

- `prepare-reference-eval`: checksum-pinned public Open3D RGB-D reference acquisition, millimetre-to-metre conversion, original metadata retention and paired input/checkpoint manifest.
- `evaluate`: checks asset integrity and RGB/depth alignment; RGB-only model inference; explicit masks/crop/interpolation/median scaling; per-image and equal-image aggregate metrics; numerical predictions and previews; partial runs never receive aggregate accuracy.
- `demo --min-seconds`: display-enabled duration gate, with failure status and exit on a short run. Headless runs cannot meet it.
- `demo --review-every`: first, periodic adjacent and final qualitative frames, RGB/NPY/preview exports, and review index. Mean/median/p95 runtime statistics and process/PyTorch memory scope are retained in the run summary.
- Full numerical/data protocol and source citations: [evaluation protocol](../../docs/evaluation_protocol.md). Model weights, normalization and Phase 1-3 default interfaces remain unchanged.

## Reference-depth result and its limits

Actual CUDA diagnostic: **5 correlated indoor frames, 1 scene, 1,340,711 valid pixels**. Input identity is `Open3D SampleRedwoodRGBDImages`, with per-frame GT median alignment and no spatial crop. Mean metrics:

| AbsRel | SqRel | Aligned RMSE (m) | Log RMSE | Delta1 | Delta2 | Delta3 |
|---|---|---|---|---|---|---|
| 0.4960 | 1.1006 | 1.4404 | 0.5071 | 0.3763 | 0.6680 | 0.8449 |

These substantial indoor errors expose a limitation. They are not comparable to the original KITTI targets, not a NYU benchmark, not evidence of accurate driving perception and not recovered metric scale. The cause of the error has not been isolated. GT never enters model inference. No fine-tuning or score-driven sample filtering was performed.

Evidence: ignored `data/evaluation/redwood/evaluation_manifest.json`, `runs/phase4_reference_evaluation/summary.json`, `per_sample.csv`, relative-depth arrays and prediction previews.

## Displayed runtime and qualitative inspection

The first displayed real-video run passed its 60-second gate: **650 frames over 76.26 seconds**, **8.52 FPS under a 10 FPS cap**, **9 review snapshots**. This covers 26 seconds of the 25 FPS source footage. Process peak working set: **587.38 MiB**; PyTorch peak allocation: **106.34 MiB**. These are not total system/driver memory peaks.

Inspected adjacent pairs **200/201** and **600/601** show broad road-to-horizon structure, with coarse vehicle boundaries. Scene motion changes predictions; snapshots are not an objective flicker score. Sky, shadows, reflective and textureless regions remain unvalidated. The driving clip has no aligned GT, so accuracy is unavailable for that clip.

The full-source run also passed: **1,260 displayed frames over 71.40 seconds**, **17.65 FPS**, configured ceiling **30 FPS** (effectively capped at the source's 25 FPS), and **9 review snapshots**. This processes all 50.4 seconds of source footage in 71.4 seconds, so it is slower than native-speed replay. Process peak working set: **734.73 MiB**; PyTorch peak allocation: **106.34 MiB**. Median model time **19.56 ms**, complete processing **25.26 ms**. This single-machine result demonstrates observed displayed video throughput above 10 FPS, not a webcam/CARLA or cross-hardware guarantee. Do not treat differently capped runs or earlier headless profiles as a controlled speedup comparison.

Run data are in ignored `runs/phase4_driving_rehearsal/` and `runs/phase4_driving_full_clip/`; `review.json` maps frame IDs to review PNG/NPY files. All **55 regression tests passed**, including every Phase 1-3 test, numerical evaluation protocol, input integrity, duplicate/misaligned/wrong-unit data, partial evaluations, snapshot preservation, duration rejection and window cleanup. Local raw evidence remains ignored; published documentation records the observed numbers and their scope.

Fresh transition checks also passed: the existing CUDA image CLI (`runs/phase4_transition_image/`) and the Phase 3 repeated profiler (`runs/phase4_transition_profile/`) with exact repeated predictions. The selected checkpoint, preprocessing and visualization contracts remain unchanged.

## Reproduce and hand off

```powershell
.\.venv\Scripts\python.exe project.py prepare-reference-eval
.\.venv\Scripts\python.exe project.py evaluate --device cuda --output runs/reference_evaluation_01
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe project.py demo --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 1260 --fps 30 --min-seconds 60 --review-every 400 --output runs/driving_full_clip_01
```

Retain sample counts, protocol, manifest/checkpoint hashes and actual scope with every reported score. Phase 5 can consume the accepted local prototype and evidence; it must list full KITTI/CARLA accuracy, physical webcam, native CARLA capture, RTX 3050 validation, fine-tuning and calibrated metric scale as pending. The user authorizes exactly three Phase 4 commits and pushing them to GitHub.
