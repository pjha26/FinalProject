# Metric-distance and vehicle-control extension plan

26 October 2026. Add a separate CARLA-only experiment; preserve tools/present.ps1, accepted pretrained weights and the existing presentation demo. No real-vehicle deployment. Each logical step is tested, recorded, committed and published before the next.

1. **Freeze protocol and roles.** Use existing Town04 development captures for fixed-scale fitting and separate Town05 development captures for validation. Never fit on KITTI or the consumed reserved route. Record all source/model/configuration hashes. These already-seen routes are development calibration evidence, not fresh final tests.
2. **Calibrate metric estimates.** Reuse the verified camera-axis depth convention, RGB preprocessing and pretrained model. Fit one robust positive multiplier from calibration-only predictions/reference depth. Freeze it; evaluate with no per-image GT scaling. Predeclare validation gates: mean per-frame AbsRel <= 0.25 and 95th-percentile per-frame AbsRel <= 0.50 over 1-50 m. A failure is retained and blocks model-based actuation. Intrinsics alone do not resolve monocular scale.
3. **Implement bounded control primitives.** Project optical-axis depth into a central vehicle corridor; exclude ground using fixed camera height and flat-road assumptions. Reject invalid/stale/mismatched readings. Implement low-speed throttle/brake and map-waypoint steering, stopping on invalid data, collision, end of route or obstacle proximity. Unit-test distances, steering, braking, latching and failure cases.
4. **Run staged native experiments.** Own and clean up only our simulator/actors. First test a labelled simulator-reference-depth controller on straight, low-speed routes with stationary obstacles and a clear-road control. Then run calibrated model in shadow mode on the same setup. Model actuation requires the frozen calibration gate and separate obstacle-clearance checks; no hidden GT substitution. Log synchronized frame IDs, estimates/reference, speed, controls, actual sensor delay, collisions, stopping clearance and route completion.
5. **Accept or reject honestly.** Predeclare low-speed experimental success as no collision, at least 1 m vehicle clearance when stopped, movement before braking, and bounded control/sensor delay. A false stop on a clear route is a failure. Reference-controller success is not learned camera-only autonomy. A failing model remains observation-only.
6. **Package presentation evidence.** Save separate previews, CSV/JSON, calibration and controller reports under ignored runs/ and a dated outputs/ folder. Add commands and actual results to README/docs; keep the accepted demo as fallback. Full autonomy, arbitrary maps/weather, traffic-rule reasoning and calibrated deployment outside the validated configuration remain future work.

This is a bounded development plan, not a promise of validated full autonomous driving within sixteen hours.

Primary API references: https://carla.readthedocs.io/en/0.9.13/python_api/ and https://carla.readthedocs.io/en/0.9.13/ref_sensors/.


## Execution record

Steps 1-3 implemented and tested. Step 4 completed three displayed simulator-reference trials with simultaneous learned shadow predictions. Step 5 accepted the bounded reference trials and rejected metric calibration/learned actuation; both shadow brake triggers were missed. Step 6 saves a dated evidence gallery, videos, traces, source/configuration hashes and updated documentation. The 149-test/eight-check baseline delivery rerun passes. Full learned closed-loop driving remains blocked by measured prediction errors, rather than being enabled by bypassing the gates.
