# Visible distances in the experimental control demo

26 October 2026. Distances were previously confined to a small top caption. The reference-control demo now has a large panel beneath RGB/depth/overlay and a reference-sample outline on RGB. The original relative-depth presentation launcher is unchanged.

- **Camera clearance - SIMULATOR REFERENCE:** the near-decile axial-depth statistic in the bounded central corridor. It is a camera-based reference statistic, not bumper-to-bumper distance.
- **Model estimate - UNVALIDATED:** the same statistic from the frozen-scale learned map. It is never used for control; calibration remains rejected.
- **Vehicle gap (simulator):** current simulator actor geometry along the straight route, minus vehicle extents. This explains why it differs from camera clearance.
- **Green outline:** the reference pixels contributing to the near-decile statistic. It is not an object detector or a learned bounding box.
- **Speed, brake percentage and state:** explain the vehicle's current reference-controller action.
- On clear roads, no projected obstacle within the bounded range is reported as “No obstacle found within 50 m”; the display does not invent a 50 m obstacle.

The displayed/control CSV values share the same computed measurements. Colour-map units remain relative; the labelled reference/model measurements are separate. Unit tests check metric source labels, numerical formatting, no fake clear-road outline, invalid measurements and agreement between reference sample masks and the scalar statistic.

Native verification: runs/control_distance_display_02 passes clear-road, ClearNoon obstacle and WetCloudyNoon obstacle trials with the enlarged overlay. The first attempt is preserved at runs/control_distance_display_01: clear and wet obstacle trials passed; the clear-weather obstacle trial failed at simulator warm-up with a 60-second RPC timeout before any recorded frame. The native startup failure remains a known limitation, not a successful run.

Full regression suite: 153 tests pass. Large preview/video/log copies are in ../outputs/control_distance_display_2026-10-26/. Rendering and caption changes do not unlock learned actuation or establish calibrated model distances.

