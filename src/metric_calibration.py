"""Fit-only monotonic metric calibration, separate from depth-network weights."""
import numpy as np
from src.metric_control import positive_map


def apply_power_depth(relative, scale, exponent):
    if not np.isfinite([scale, exponent]).all() or scale <= 0 or not .25 <= exponent <= 4:
        raise ValueError("Finite positive scale and exponent in [0.25,4] required")
    relative = positive_map(relative)
    with np.errstate(over="ignore", invalid="ignore"):
        result = scale * np.power(relative, exponent)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Invalid mapped metric depth")
    return result


def fit_power_depth(predictions, references):
    """Deterministic, equal frame sampling, Huber IRLS in log depth; no validation inputs."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("Paired fit frames required")
    count = min(1024, *(positive_map(p).size for p in predictions))
    if count < 20:
        raise ValueError("At least 20 valid pixels per fit frame required")
    xs, ys = [], []
    for pred, gt in zip(predictions, references):
        pred, gt = positive_map(pred), positive_map(gt)
        if pred.shape != gt.shape:
            raise ValueError("Fit geometry mismatch")
        index = np.linspace(0, pred.size - 1, count, dtype=int)
        xs.append(np.log(pred.ravel()[index]))
        ys.append(np.log(gt.ravel()[index]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(x) < 20 or np.ptp(x) < .05:
        raise ValueError("Insufficient depth variation for a two-parameter fit")
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(10):
        residual = design @ coefficients - y
        weight = .15 / np.maximum(np.abs(residual), .15)
        weighted = np.sqrt(weight)
        coefficients = np.linalg.lstsq(design * weighted[:, None], y * weighted, rcond=None)[0]
    scale, exponent = float(np.exp(coefficients[0])), float(coefficients[1])
    apply_power_depth(np.ones((1, 1)), scale, exponent)
    return dict(scale=scale, exponent=exponent, sampled_fit_pixels=int(len(x)), method="Equal-size deterministic per-frame log samples; 10 Huber IRLS iterations, delta0.15")


def paired_promotion(baseline, candidate):
    """Require absolute pixel gates, >=5% overall gain and no >10% scene regression."""
    from src.metric_control import calibration_gate
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Complete paired validation rows required")
    keys = [(r["scene"], r["frame"]) for r in baseline]
    if len(set(keys)) != len(keys) or keys != [(r["scene"], r["frame"]) for r in candidate]:
        raise ValueError("Paired frame identities mismatch")
    before = calibration_gate(baseline)
    after = calibration_gate(candidate)
    scenes = {}
    for name in sorted(set(k[0] for k in keys)):
        a = np.mean([r["abs_rel"] for r in baseline if r["scene"] == name])
        b = np.mean([r["abs_rel"] for r in candidate if r["scene"] == name])
        if not np.isfinite([a, b]).all() or a <= 0:
            raise ValueError("Invalid baseline scene error")
        scenes[name] = dict(before=float(a), after=float(b), ratio=float(b / a))
    gain = 1 - after["mean_abs_rel"] / before["mean_abs_rel"] if before["mean_abs_rel"] > 0 else 0
    return dict(baseline=before, candidate=after, relative_improvement=float(gain), scenes=scenes, passed=bool(after["passed"] and gain >= .05 and all(s["ratio"] <= 1.10 for s in scenes.values())))


