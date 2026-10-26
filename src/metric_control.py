"""Frozen scale calibration and bounded CARLA control math; never real-vehicle control."""
import math
import numpy as np


def positive_map(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected a nonempty finite positive depth map")
    return values


def fixed_scale(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("Paired calibration samples required")
    ratios = []
    for prediction, reference in zip(predictions, references):
        prediction, reference = positive_map(prediction), positive_map(reference)
        if prediction.shape != reference.shape:
            raise ValueError("Calibration geometry mismatch")
        ratios.append(float(np.median(reference / prediction)))
    return float(np.median(ratios))


def scaled_depth(relative, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    return positive_map(relative) * scale


def calibration_gate(rows):
    if not rows:
        raise ValueError("Validation rows required")
    errors = np.asarray([r["abs_rel"] for r in rows], dtype=float)
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Invalid validation errors")
    mean, p95 = float(errors.mean()), float(np.percentile(errors, 95))
    return dict(mean_abs_rel=mean, p95_frame_abs_rel=p95, thresholds=dict(mean_abs_rel=0.25, p95_frame_abs_rel=0.50), passed=mean <= 0.25 and p95 <= 0.50)


def corridor_clearance(depth, intrinsics, camera_height=1.7):
    """Optical-axis Z; flat road, level camera and 2.6 m wide corridor only."""
    depth = positive_map(depth)
    k = np.asarray(intrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(camera_height) or camera_height <= 0:
        raise ValueError("Invalid camera geometry")
    v, u = np.indices(depth.shape)
    lateral = (u - k[0, 2]) * depth / k[0, 0]
    height = camera_height - (v - k[1, 2]) * depth / k[1, 1]
    valid = (np.abs(lateral) < 1.3) & (height > 0.4) & (height < 3.0) & (depth > 0.05) & (depth < 50)
    # shortcut: this narrow corridor assumes a level straight road; validate curved/sloped routes before expanding.
    values = depth[valid]
    if values.size < 25:
        return float(values.min()) if values.size else 50.0
    return float(np.percentile(values, 10))


def steering_to_waypoint(position, yaw_degrees, target):
    values = np.asarray([*position, yaw_degrees, *target], dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Finite planar pose/target required")
    dx, dy = np.asarray(target) - position
    if math.hypot(dx, dy) < 0.01:
        return 0.0
    angle = math.atan2(dy, dx) - math.radians(yaw_degrees)
    angle = math.atan2(math.sin(angle), math.cos(angle))
    return float(np.clip(angle / math.radians(35), -0.5, 0.5))


def bounded_control(speed, clearance, age_seconds, steer=0.0, stop=False):
    """Low-speed experimental controller; invalid readings always request braking."""
    numeric = [speed, age_seconds, steer]
    if not np.isfinite(numeric).all() or speed < 0 or age_seconds < 0:
        return dict(throttle=0.0, brake=1.0, steer=0.0, reason="invalid_state")
    if stop:
        return dict(throttle=0.0, brake=1.0, steer=float(np.clip(steer, -0.5, 0.5)), reason="latched_stop")
    if clearance is None or not np.isfinite(clearance) or clearance <= 0 or age_seconds > 0.35:
        return dict(throttle=0.0, brake=1.0, steer=0.0, reason="invalid_or_stale_depth")
    # Stop with a 6 m camera-range buffer, a 0.5 s delay allowance and 2 m/s^2 nominal deceleration.
    threshold = 6.0 + speed * 0.5 + speed * speed / 4.0
    if clearance <= threshold:
        return dict(throttle=0.0, brake=1.0, steer=float(np.clip(steer, -0.5, 0.5)), reason="obstacle_stop")
    return dict(throttle=float(np.clip((2.0 - speed) * 0.3, 0, 0.35)), brake=0.2 if speed > 2.3 else 0.0, steer=float(np.clip(steer, -0.5, 0.5)), reason="follow_route")


