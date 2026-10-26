# Metric scale experiment: first result

26 October 2026. The separate fixed-scale experiment completed on 40 Town04 calibration frames and 80 Town05 validation frames. The scale is fitted only on calibration data, then frozen. Validation uses no per-image GT alignment.

- Frozen multiplier: 16.1623278248.
- Mean validation AbsRel: 0.2743354775.
- 95th-percentile frame AbsRel: 0.3584492430.
- Predeclared thresholds: mean <= 0.25 and 95th percentile <= 0.50.
- Decision: **rejected** because mean error exceeds the threshold. Model-based actuation remains blocked.

This is a measured calibration attempt, not validated deployment distances. Source RGB/depth, camera-axis convention probe, input geometry, source hashes, model hashes and fit/validation identities are frozen in runs/metric_scale_01/protocol.json. Full per-frame scores are local. Earlier development scenes are reused explicitly; no untouched final-test claim is made.

The new control math is tested independently: positive-depth/geometry validation, fixed scale, acceptance thresholds, corridor distances, steering bounds, low-speed control and full braking on invalid/stale/latched states. The corridor is a flat, straight-road experiment with known 1.7 m camera height; an absence of projected obstacles means no obstacle found within the 50 m range, not a universal free-space guarantee.

Next native trials will compare a clearly labelled simulator-reference controller and model shadow estimates. The reference may command the simulator; rejected learned metric estimates cannot command it. The original presentation demo remains unchanged.

