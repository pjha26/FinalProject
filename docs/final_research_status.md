# Research implementation and presentation status

26 October 2026. Steps 2-9 of the remaining implementation plan now have software and measured evidence within their stated budgets. This does not mean all research accuracy targets passed.

| Steps | Delivered evidence |
|---|---|
| 2-3 | Eight registered condition/route captures; 392 train and 152 development triplets; tested checkpoint/optimizer/RNG resume |
| 4-5 | Three-epoch controlled fine-tuning, matched unaugmented control, loss ablations and seed repeats |
| 6 | Complete 697-frame KITTI Eigen evaluation with frozen pairing/crop/scaling and asset hashes |
| 7 | Frozen 100-frame reserved-route comparison; route now consumed as final testing |
| 8 | Native displayed CARLA: 783 frames, 65.0114 seconds, 12.0440 application FPS |
| 9 | Report supplement, editable slides, local evidence gallery and hashes, launcher, 137 tests and eight integrated checks |

The new presentation launcher separately completed 780 frames over 65.0081 seconds at 11.9985 FPS. All 137 tests passed in the latest integrated delivery, and the displayed video rehearsal passed. Source bytes match that validator snapshot. See [runbook](presentation_runbook.md), [full KITTI evaluation](full_eigen.md), [native runtime](native_runtime.md) and [numerical delivery status](final_research_status.json).

The accepted model remains official pretrained Monodepth2. The three-epoch candidate worsened full KITTI AbsRel from 0.1146166 to 0.1351531. Baseline AbsRel and delta1 meet proposal targets; RMSE 4.8634, SqRel 0.9048 and log-RMSE 0.1928 miss their targets. Scores use GT median alignment only during evaluation.

The original proposal remains unchanged. The new research report supplements it with actual results and failed targets. Physical webcam, another laptop, physically disconnected-network rehearsal, calibrated metre-scale prediction, convergence and broad robustness are not established. Presentation files and selected large outputs remain in the local outputs bundle and require separate backup. Raw datasets, full weights and simulator/runtime are not committed to Git.


Subsequent experimental extension: the latest integrated rerun passes 149 tests and eight checks. A separate simulator-reference controller passed clear-road and two obstacle trials; model calibration failed and learned actuation is disabled. See [extension results](metric_control_results.md). The original depth-demo scope and failed accuracy targets remain unchanged.


27 October accuracy update: a separately supervised replay-trained candidate passes new 200-frame CARLA pixel validation with a 13.01% relative error reduction, while performing slightly worse on full KITTI. It remains an optional research checkpoint; original pretrained inference stays default and learned actuation is disabled. The final baseline delivery passes 170 tests/eight checks. Unseen-object validation is incomplete and one reference integration freshness gate failed. See [validation report](metric_model_validation.md). No PDF or presentation file is changed.
