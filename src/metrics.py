"""Depth metrics on numerical arrays, never on colored images."""
import numpy as np


def compute_metrics(ground_truth, prediction, median_scale=True, max_depth=80.0):
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Ground truth and prediction must be matching 2D maps")
    if not np.isfinite(max_depth) or max_depth <= 1e-3:
        raise ValueError("Maximum depth must exceed 1e-3")
    valid = np.isfinite(gt) & (gt > 1e-3) & (gt < max_depth)
    if not valid.any():
        raise ValueError("No valid ground-truth pixels")
    gt, pred = gt[valid], pred[valid]
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("Invalid predicted depth at valid ground-truth pixels")
    ratio = float(np.median(gt) / np.median(pred)) if median_scale else 1.0
    pred = np.clip(pred * ratio, 1e-3, max_depth)
    threshold = np.maximum(gt / pred, pred / gt)
    difference = gt - pred
    return {
        "valid_pixels": int(gt.size), "scale_ratio": ratio,
        "abs_rel": float(np.mean(np.abs(difference) / gt)),
        "sq_rel": float(np.mean(difference ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))),
        "a1": float(np.mean(threshold < 1.25)),
        "a2": float(np.mean(threshold < 1.25 ** 2)),
        "a3": float(np.mean(threshold < 1.25 ** 3)),
    }
