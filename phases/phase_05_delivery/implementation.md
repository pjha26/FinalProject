# Phase 5: Rehearsal and Final Delivery

Days 9-10 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Previous phase](../phase_04_evaluation/implementation.md)

This delivery track is implemented; local acceptance and presentation-laptop acceptance are recorded separately below.

## Entry

Use the selected frozen checkpoint, input manifest, accepted commands, evaluation report and known limitations. Prioritize a reliable demonstrable pipeline over late model changes.

## Day 9: package and report

1. Rebuild the environment on the presentation laptop from the pinned requirements instead of copying .venv. Verify CUDA availability and asset hashes; prepare downloads before arriving at the review.
2. Document architecture, hardware adaptation, pretrained provenance, input processing, relative scale, actual measurements, qualitative examples and incomplete experiments.
3. Credit official model/source licenses and permitted footage. Cite reference repositories accurately. Include the diagram from the master plan and evidence-backed progress.
4. Prepare camera, video and image commands with fresh output names. Rehearse permissions, power connection, camera index and window controls. Close unused memory-heavy apps after saving work.

## Day 10: freeze and demonstrate

1. Run the final regression suite, then the same accepted displayed demo. Keep network-independent local footage and weights available.
2. Demonstrate input movement, depth-map change, overlay and measured FPS. Explain relative depth and unavailable GT where appropriate.
3. If camera access fails, switch to the prepared driving-video replay; if video decoding fails, show the verified image path and explain the limitation. Do not call these fallbacks successful native CARLA simulation.
4. Review staged files, excluding weights/media/runtimes. Preserve the user's at-most-ten-file staging constraint for the current change set. The user authorizes Phase 5 publication in exactly three commits.

## Delivery acceptance

Deliver source, five phase plans, README commands, selected checkpoint provenance and local evidence, final report/presentation, measured hardware behavior and an explicit incomplete-work list. Reproduce the demo from a clean environment. A model trained elsewhere or an optional simulator that has not been tested must not be presented as a completed project result.

No further phase transition is required. Future training, calibrated metric depth, smaller-model adoption or vehicle control is separate work with new acceptance tests.

## Implemented delivery workflow

`tools/validate_delivery.py` runs the shared commands using its own Python interpreter: complete regression suite, frozen input verification, environment check, image prediction, repeated profile, paired-reference evaluation and short video processing. Optional `--display` adds a 650-frame replay at a 10 FPS cap with a minimum 60-second displayed gate and review snapshots. It makes no downloads, writes exclusive output directories, records per-step logs, stops on failure and rejects source changes during validation. Omitting the display leaves that acceptance gate pending.

```powershell
.\.venv\Scripts\python.exe tools/validate_delivery.py --device cuda --display
```

The [presentation guide](../../docs/demo_guide.md) includes environment reconstruction, offline assets and license transfer, exact demo/fallback commands, explanation script and outstanding work. The generated PDF records all five phases and separates historical Phase 3/4 measurements from final acceptance.

Final local evidence is recorded in `runs/phase5_delivery/`. A separate Python 3.8.20 environment under ignored `tmp/phase5_rebuild/` was installed from the 30 pinned requirements, then used for integrated validation. Keep the original development environment available. Rebuild normally on another machine rather than copying this test environment.

On 19 October 2026 all **61 tests and eight integrated checks passed** on the local GTX 1650. Image inference, exact repeated-profile predictions, the five-pair diagnostic and video processing passed. The final displayed run completed **650 frames in 82.55 seconds at 7.87 FPS**, under a 10 FPS cap, passing the minimum 60-second gate. Peak process working set was 540.00 MiB; PyTorch peak allocated GPU memory was 106.34 MiB, not total RAM/VRAM. Source hashes remained unchanged. The five-frame diagnostic reproduced the Phase 4 scores; it remains a limited indoor diagnostic, not driving benchmark accuracy.

The target RTX 3050, physical webcam and native CARLA gates remain pending. Full KITTI evaluation, new training and calibrated distances are incomplete; finishing the software delivery track does not imply those original targets have passed.
