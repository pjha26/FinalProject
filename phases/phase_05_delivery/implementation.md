# Phase 5: Rehearsal and Final Delivery

Days 9-10 of the existing ten-day schedule.

[Master plan](../../IMPLEMENTATION_PLAN.md) | [Previous phase](../phase_04_evaluation/implementation.md)

This is an implementation/handoff plan; acceptance requires recorded evidence.

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
4. Review staged files, excluding weights/media/runtimes. Preserve the user's at-most-ten-file staging constraint for the current change set. Phase 1 publication is authorized in no more than four commits; later publication requires user authorization.

## Delivery acceptance

Deliver source, five phase plans, README commands, selected checkpoint provenance and local evidence, final report/presentation, measured hardware behavior and an explicit incomplete-work list. Reproduce the demo from a clean environment. A model trained elsewhere or an optional simulator that has not been tested must not be presented as a completed project result.

No further phase transition is required. Future training, calibrated metric depth, smaller-model adoption or vehicle control is separate work with new acceptance tests.
