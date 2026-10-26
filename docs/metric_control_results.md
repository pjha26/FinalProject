# CARLA metric-distance and control extension

26 October 2026. This is a separate, low-speed simulator experiment. The original RGB-only presentation launcher and accepted weights are unchanged.

## Implemented and measured

- Fixed-scale calibration: 40 Town04 fit frames, 80 Town05 development-validation frames. Mean AbsRel 0.274335 exceeds the predeclared 0.25 gate; the 95th-percentile frame error 0.358449 is below 0.50. **Calibration is rejected.**
- Low-speed vehicle control: autopilot is disabled; a map-waypoint steering controller and throttle/brake controller command the ego vehicle using **CARLA reference depth**.
- Learned predictions run in shadow on exactly the same captured RGB. Their scaled values are labelled unvalidated and never control the vehicle.
- Collision detection, bounded frame age, lane deviation, route-goal stopping, latched braking and actor/world cleanup are implemented.
- Invalid/stale observations request full braking. GPU inference is warmed before actuation begins.
- All 149 tests pass. An independent audit decodes every evidence-video frame, checks consecutive sensor IDs, verifies control bounds and confirms recorded acceptance outcomes.

| Native displayed trial | Outcome | Frames | Progress | Minimum vehicle gap | Maximum sensor age |
|---|---|---:|---:|---:|---:|
| ClearNoon, clear road, seed 42 | Reached goal and stopped | 212 | 30.3704 m | N/A | 0.2890 s |
| ClearNoon, obstacle, seed 42 | Stopped without collision | 99 | 12.8849 m | 6.1949 m | 0.2839 s |
| WetCloudyNoon, obstacle, seed 44 | Stopped without collision | 97 | 13.1929 m | 6.0153 m | 0.3016 s |

Maximum speed was about 1.62 m/s (5.8 km/h). These short trials use level, straight Town04 road segments, fixed camera geometry, a stationary vehicle obstacle and a map-aware planner. They do not test intersections, traffic-rule decisions, pedestrians, moving obstacles, arbitrary slopes or real vehicles.

Learned obstacle-clearance mean absolute errors were 4.9703 m and 3.7468 m in the two successful obstacle trials. At a saved stopping frame the model estimates about 12.4 m while reference clearance is about 7.2 m. This is a material overestimate, so the rejected scale cannot be presented as reliable calibrated distance or used for actuation.

The first suite is retained at runs/metric_control_suite_01: clear/wet trials braked before motion when cold first inference exceeded the 0.35 s freshness limit. The corrected suite warms inference before enabling control, retains the same limit, and passes all three cases at runs/metric_control_suite_02. This is a fix for startup placement, not a relaxed acceptance threshold.

## Run this extension

From the prepared repository root, using a fresh output path:

    $env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
    .\.venv\Scripts\python.exe tools\run_control_suite.py --output runs/control_presentation_unique

The suite starts its own local CARLA server, refuses an occupied RPC port, runs three displayed trials and stops its owned server. It requires runs/metric_scale_01/calibration.json and the prepared baseline checkpoint. Rebuild that artifact if necessary:

    .\.venv\Scripts\python.exe tools\calibrate_metric_scale.py --output runs/metric_scale_unique

Then pass --calibration runs/metric_scale_unique/calibration.json to the suite. Calibration also needs the six named development captures and the original passed facade-geometry report; these large local assets are not in Git.

Individual experiments can use tools/carla_control_demo.py against an already running local server. --mode model always refuses actuation while the development and independent control gates remain incomplete. Reference control is explicitly labelled in the display. Do not describe it as Monodepth2 steering/braking autonomy.

## Saved outputs and presentation wording

Local reference bundle: ../outputs/metric_control_extension_2026-10-26/. Open index.html for actual video/previews and reports. Each case retains CSV controls and frame IDs, protocol, route and summary. The MP4 uses a 10 Hz simulation timebase; it is a rendered experiment video, not an original wall-clock screen recording or an FPS benchmark.

Suggested explanation: "We added a low-speed CARLA route-following and obstacle-stopping controller, validated using simulator depth. Monodepth2 distance estimates are compared in shadow. The learned metric calibration failed its acceptance gate, so learned vehicle control is blocked."

Original demo: .\tools\present.ps1 -Mode carla remains the accepted RGB-only relative-depth demonstration. Optional reference-control trials are supplemental. Full learned autonomous driving and validated deployment metre distances remain unfinished research, not achieved features.


## Integrated baseline recheck and shadow review

After these additions, runs/metric_control_delivery_01 passed all eight checks and 149 tests, including a displayed 650-frame recorded-input rehearsal lasting 66.9169 seconds at 9.7135 FPS under the original 10-FPS video cap. This is separate from the prior native CARLA throughput measurement. The validator source bytes remained unchanged throughout.

Offline counterfactual command review found no model false-stop commands on the recorded clear route, but the model would have missed the initial reference braking trigger in both obstacle trials. These are counterfactual commands on the reference trajectory, not a learned closed-loop run or proof that a collision would occur. Learned actuation remains blocked. Full details are in metric_control_shadow_results.json.
