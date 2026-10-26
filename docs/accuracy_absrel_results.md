# AbsRel-objective calibration: development result

27 October 2026. The weighted-median scale minimizes the reported mean per-frame pixel AbsRel on fit data only. It uses the same 40 Town04 fit frames and hash-verified cached RGB predictions. Network weights are unchanged.

| Development measure | Original multiplier | Objective-matched multiplier |
|---|---:|---:|
| Scale | 16.1623278248 | 14.9965364942 |
| Mean AbsRel | 0.27433548 | 0.24866763 |
| P95 frame AbsRel | 0.35844924 | 0.33119349 |

Relative mean-error improvement is 9.3564%; all four development scenes improve, and the unchanged development gates pass. This is supervised postprocessing calibration, not depth-network retraining or metric certification. A candidate file was frozen before acquisition/scoring of two reserved new-seed Town05 routes. The separate one-time holdout report determines its limited out-of-sample acceptance.

Numerical tests verify the relative-error weighted-median optimum, exact scale recovery, invariance to within-frame pixel replication, invalid shapes/values and overflow. Model actuation remains disabled regardless of pixel-score acceptance.

