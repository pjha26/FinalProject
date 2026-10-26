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


## One bounded supervised metric experiment

The matched obstacle test confirms depth error as well as corridor-selection error. Predeclare a single exploratory network candidate: initialize official Monodepth2, freeze its encoder, train only the depth decoder for 128 updates at Adam1e-5, seed42, fixed metric multiplier14.9965364942. Alternate the 40 original Town04 fit frames and eight Town04 obstacle images saved by the diagnostic. Use only their simulator reference depth as supervision. Loss: valid1-50m log-depth L1 +0.5 matched-object-region log L1 +0.1 frozen-teacher log-depth consistency. No validation/test labels enter training. This is explicitly supervised fine-tuning, not the earlier temporal self-supervised method.

Freeze the final update128 candidate; no checkpoint search. Evaluate on the same 80 Town05 development targets against the best fixed-scale baseline with the original pixel gates, >=5% relative improvement and <=10% scene regression. Also require <=5% KITTI24 development AbsRel regression; never use the full Eigen test to select it. If eligible, use newly reserved Town05 seed49/50 captures for fresh final evaluation; earlier seed47/48 holdout stays consumed. Learned actuation remains disabled regardless of this exploratory model's pixel pass. This is one short test, not a convergence or robustness claim.


## Conservative real-RGB replay variant

Trial1 improves CARLA but fails KITTI24 regression. Predeclare one further128-update final-checkpoint trial at Adam2e-6 with the same CARLA fit set and loss, plus weight1.0 log-inverse-depth distillation on 16 uniformly selected frames from the existing verified Udacity demo clip. The frozen official teacher supplies these real-RGB targets. KITTI RGB/GT and all evaluation scenes remain excluded from training. Record video hash, frame IDs and teacher identity. This is supervised CARLA training with teacher regularization, not a pure temporal self-supervised run. Keep all existing acceptance gates; do not tune against the final Eigen or consumed seed47/48 holdout. This bounded trial changes both rate and replay, so it is not a one-factor ablation.


## Final bounded replay budget

Before execution, extend the same replay recipe once to 256 updates. Keep Adam2e-6, teacher weight1.0, all fit data, seed42 and all acceptance thresholds unchanged. Freeze update256; do not select an intermediate checkpoint. This is a controlled duration extension of the rejected128-update replay candidate. If it fails development acceptance, retain the accepted original model and report the remaining work; no additional sweep is planned in this iteration.


## Final execution status

The weighted scale and final 256-update replay network pass their separately frozen fresh moving-scene tests. The network is not the general default because KITTI is slightly worse. Four unseen-object native attempts fail before scoring; one wet reference integration freshness gate fails. Learned actuation remains disabled. All 170 unit tests and eight original-demo delivery checks pass. The consumed fresh test routes are registered; no further fitting or reselection uses their scores. No PDF or presentation artifacts are committed.
