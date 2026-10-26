# Accuracy iteration 2: frozen holdout and obstacle results

27 October 2026. No presentation or PDF files were changed in this iteration.

The objective-matched scale 14.9965364942 improves mean development AbsRel from 0.27433548 to 0.24866763 (9.3564% relative). Frozen before new data acquisition, it also improves all 200 frames of two new Town05 routes from 0.23694070 to 0.21300889 (10.1003% relative). P95 holdout frame error improves from 0.28704817 to 0.25150180. Both scene scores improve.

An independent saved-prediction audit reproduces scores to within 5.30e-8 over 27,659,610 valid pixels. No exact RGB hashes overlap fit/development scenes. No new frame lies within 1 m of earlier Town05 development trajectories; minimum route distances are 190.09 m and 49.77 m. All routes still share the Town05 map and correlated short captures. The holdout is consumed and cannot be reused for selection.

These are **postprocessing calibration improvements**. Monodepth2 network weights and KITTI median-aligned benchmark results are unchanged. The candidate is not promoted to the vehicle-control default.

A matched stationary-obstacle diagnostic uses the same RGB predictions for both scales across four distances and two weather presets (48 samples each). It holds ego braked throughout, uses reference masks only for scoring, and does not run learned closed-loop control:

| Measure | Original scale | Candidate scale |
|---|---:|---:|
| Matched surface MAE | 7.7569 m | 7.1750 m |
| Maximum surface overestimate | 21.9244 m | 19.7572 m |
| Corridor clearance MAE | 7.6129 m | 7.9300 m |
| Counterfactual missed brake triggers | 12 | 6 |

Both fail the stationary diagnostic gate. Lower whole-image error therefore does not establish reliable obstacle distance. Reference-control trial completion is separate from model validation. Learned actuation remains disabled.

Local evidence: runs/metric_absrel_holdout_01/, runs/metric_obstacle_diagnostic_01/ and runs/metric_accuracy_control_03/. Full source/weight identities, masks, scales and all rejected candidates are retained.

