# Accuracy validation iteration 2

27 October 2026. Code/results only: do not add or update PDF/PPTX files in this work.

The current frozen multiplier gives 0.274335 mean Town05 development AbsRel and fails the 0.25 gate. Preserve accepted pretrained weights, KITTI results, presentation commands and the model-actuation block.

1. Reproduce the multiplier on the same 40 Town04 fit / 80 Town05 development frames.
2. Fit one two-parameter monotonic power correction, d_metres = scale * d_relative ** exponent, on **fit data only**. This is supervised postprocessing calibration, not new depth-network training or a new architecture. Use deterministic equal-size per-frame log-depth samples and bounded robust regression. Reject invalid/non-monotonic/extrapolated corrections.
3. Compare both frozen mappings on all development pixels in the same 1-50 m crop. Record AbsRel, RMSE, per-distance-bin errors, repeatability and frame/scene counts. Keep the original mean <=0.25 / p95 frame <=0.50 gates, plus require at least 5% relative development AbsRel improvement and no per-scene regression above 10% for promotion to a fresh-test candidate.
4. Reserve new Town05 seed47 ClearNoon and seed48 WetCloudyNoon captures before scoring. They are camera/route development extensions on a shared map, not unseen-map data. Never fit or reselect using these fresh holdout scores. A development-rejected correction does not consume the holdout test.
5. If the candidate passes development selection, freeze its bytes/model identity and score the new captures once without per-frame GT scaling. A failure remains rejected. Learned actuation stays disabled regardless of a pixel-accuracy pass until independent obstacle/clearance and dynamic-control gates pass.
6. Save prediction/evaluation evidence locally, run numerical tests, and publish source and summarized results only. Do not upload PDFs, slides, raw datasets or weights. Validation does not guarantee an accuracy improvement; report measured outcomes.

The next network-training decision depends on error diagnostics. Avoid launching a large unbounded training run while the current metric failure is still unexplained.


## Follow-up: fit the reported error objective directly

The two-parameter mapping failed development gates. Before another evaluation, predeclare a simpler alternative: one scale minimizing mean per-frame pixel AbsRel on the same fit frames. For relative prediction d and reference g, |s*d-g|/g = (d/g)*|s-g/d|. Therefore fit the weighted median of g/d with weights (d/g)/pixels_in_frame. Use all fit pixels, deterministic stable sorting, no development labels in fitting, and keep all promotion thresholds unchanged. Reuse only hash-verified cached RGB predictions. The original power candidate remains rejected. Evaluate this one additional candidate on development; only a passing candidate may consume the reserved holdout.

