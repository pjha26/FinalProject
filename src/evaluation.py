"""Frozen paired-depth diagnostics; ground truth is never a model input."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from src.metrics import compute_metrics
from src.model import sha256
from src.preprocess import PREPROCESSING_ID

REFERENCE_URL = "https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/SampleRedwoodRGBDImages.zip"
REFERENCE_SHA256 = "4f161ba04d81cbe628fc69dac20552749f1b255b0090fe51c766e5d02355937b"
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def evaluation_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("preprocessing_id") != PREPROCESSING_ID:
        raise ValueError("Incompatible evaluation manifest")
    for key in ("dataset", "source", "limitations"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("Evaluation manifest requires " + key)
    if manifest.get("data_kind") not in ("rgbd_reference", "real_rgbd", "synthetic_test_fixture") or manifest.get("split") not in ("diagnostic", "test", "validation"):
        raise ValueError("Declare evaluation data kind and held-out/diagnostic split")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("depth_units") != "metres" or protocol.get("aligned_to_rgb") is not True or protocol.get("min_depth") != 0.001 or protocol.get("max_depth") != 80 or protocol.get("scaling") != "per_image_median" or protocol.get("aggregation") != "mean_of_images" or protocol.get("crop") not in ("none", "garg"):
        raise ValueError("Evaluation requires the declared aligned-metre, 0.001-80m median-scaled protocol")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation manifest needs at least one paired sample")
    identifiers, rgb_paths = set(), set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sample["id"]):
            raise ValueError("Evaluation sample requires a safe unique ID")
        if sample["id"] in identifiers or not isinstance(sample.get("scene"), str) or not sample["scene"]:
            raise ValueError("Duplicate sample ID or missing scene identity")
        identifiers.add(sample["id"])
        for key in ("rgb", "depth"):
            relative = sample.get(key)
            if not isinstance(relative, str) or not relative:
                raise ValueError("Missing paired evaluation asset")
            asset = (path.parent / relative).resolve()
            if path.parent not in asset.parents or not asset.is_file() or sha256(asset) != sample.get(key + "_sha256"):
                raise ValueError("Evaluation asset missing, outside dataset or checksum mismatch: " + relative)
            if key == "rgb":
                if asset in rgb_paths:
                    raise ValueError("Duplicate RGB asset cannot inflate sample count")
                rgb_paths.add(asset)
    return manifest


def paired_samples(path, manifest):
    root = Path(path).resolve().parent
    for sample in manifest["samples"]:
        with Image.open(root / sample["rgb"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        depth = np.load(root / sample["depth"], allow_pickle=False)
        if depth.ndim != 2 or depth.dtype.kind != "f" or depth.shape != rgb.shape[:2]:
            raise ValueError("Reference depth must be floating-point metres aligned to RGB: " + sample["id"])
        if not (np.isfinite(depth) & (depth > 0.001) & (depth < 80)).any():
            raise ValueError("Reference sample has no valid depth pixels: " + sample["id"])
        yield sample, rgb, depth


def score_inverse_depth(ground_truth, inverse_depth, crop="none"):
    import cv2
    gt = np.asarray(ground_truth, dtype=np.float64).copy()
    inverse = np.asarray(inverse_depth, dtype=np.float32)
    if gt.ndim != 2 or inverse.ndim != 2 or not np.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError("Expected 2D reference and finite positive inverse depth")
    if crop == "garg":
        height, width = gt.shape
        top, bottom, left, right = np.asarray([0.40810811 * height, 0.99189189 * height, 0.03594771 * width, 0.96405229 * width], dtype=np.int32)
        mask = np.zeros(gt.shape, dtype=bool)
        mask[top:bottom, left:right] = True
        gt[~mask] = np.nan
    elif crop != "none":
        raise ValueError("Unknown evaluation crop")
    resized = cv2.resize(inverse, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return compute_metrics(gt, 1 / resized, median_scale=True, max_depth=80)


def aggregate_scores(rows):
    if not rows:
        raise ValueError("No evaluated samples; accuracy is unavailable")
    values = np.asarray([[row[key] for key in METRICS] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation metric")
    return {key: float(value) for key, value in zip(METRICS, values.mean(axis=0))}


def runtime_evidence(rows, elapsed, headless, minimum_seconds=0):
    from src.profiling import summarize_timings
    if not np.isfinite(elapsed) or elapsed <= 0 or not np.isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("Invalid runtime duration or acceptance threshold")
    gate = "not_requested" if minimum_seconds == 0 else "pass" if not headless and elapsed >= minimum_seconds else "needs_review"
    return dict(timings=summarize_timings(rows, ("model_ms", "processing_ms", "capture_to_display_ms", "full_cycle_ms")),
                runtime_acceptance=dict(status=gate, minimum_seconds=minimum_seconds,
                                        measured_seconds=elapsed, display_enabled=not headless),
                scope="Integrated frame-loop time includes first-frame overhead, periodic review exports and pacing; excludes model loading, manifest verification and final report writes. Processing includes model time.")


def save_review_snapshot(output, frame_id, rgb, maps, preview):
    from src.profiling import prediction_fingerprint
    if not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("Review snapshot requires a nonnegative integer frame ID")
    fingerprint = prediction_fingerprint(maps)
    folder = Path(output) / "review"
    folder.mkdir(exist_ok=True)
    prefix = "{:08d}".format(frame_id)
    if (folder / (prefix + "_preview.png")).exists():
        return None
    Image.fromarray(rgb).save(folder / (prefix + "_rgb.png"))
    Image.fromarray(preview).save(folder / (prefix + "_preview.png"))
    np.save(folder / (prefix + "_inverse_depth.npy"), maps["inverse_depth"], allow_pickle=False)
    return dict(frame_id=frame_id, rgb="review/" + prefix + "_rgb.png", preview="review/" + prefix + "_preview.png",
                inverse_depth="review/" + prefix + "_inverse_depth.npy", prediction_sha256=fingerprint,
                interpretation="Qualitative inspection only; camera/object motion can change adjacent predictions.")
