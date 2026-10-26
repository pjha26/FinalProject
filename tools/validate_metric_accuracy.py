"""Reproduce fixed scaling, fit a frozen monotonic correction and compare development frames."""
import argparse
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import write_json
from src.carla_capture import replay_scene
from src.demo_data import checkpoint_hashes
from src.metric_control import fixed_scale
from src.metric_calibration import fit_power_depth, apply_power_depth, paired_promotion
from src.model import load_model, predict, sha256
from src.preprocess import preprocess_rgb, PREPROCESSING_ID
from tools.calibrate_metric_scale import FIT, VALIDATION


def score(sample, prediction):
    gt = sample["reference"]
    error = np.abs(prediction - gt)
    row = dict(scene=sample["scene"], frame=sample["frame"], valid_pixels=int(gt.size), abs_rel=float(np.mean(error / gt)), mae_m=float(error.mean()), rmse_m=float(np.sqrt(np.mean(error ** 2))))
    for lower, upper in [(1, 5), (5, 10), (10, 20), (20, 50)]:
        mask = (gt >= lower) & (gt < upper if upper < 50 else gt <= upper)
        row["mae_{}_{}m".format(lower, upper)] = float(error[mask].mean()) if mask.any() else None
    return row


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "checkpoints/baseline_mono_640x192"
    geometry = ROOT / "runs/native_carla_pilot_09/depth_geometry/summary.json"
    if json.loads(geometry.read_text())["convention"] != "camera_axis_z":
        raise ValueError("Verified axial depth required")
    source_hashes = {}
    for name in FIT + VALIDATION:
        scene = ROOT / "data/carla" / name
        source_hashes[name] = {p.relative_to(scene).as_posix(): sha256(p) for p in sorted(scene.rglob("*")) if p.is_file()}
    protocol = dict(schema_version=1, fit_scenes=FIT, development_scenes=VALIDATION, checkpoint_sha256=checkpoint_hashes(checkpoint), preprocessing_id=PREPROCESSING_ID, stride=5, method="One monotonic power mapping; equal 1024 samples/frame, log Huber delta0.15, 10 iterations, exponent[0.25,4]", reference_mask="x32:608 y96:456; 1-50m", gates=dict(mean_abs_rel=.25, p95_frame_abs_rel=.50, minimum_relative_improvement=.05, maximum_scene_error_ratio=1.10), reserved_holdout=["metric_holdout_town05_s47_clearnoon_01", "metric_holdout_town05_s48_wetcloudynoon_01"], source_sha256=source_hashes)
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(2)
    model = load_model(checkpoint, "cuda")
    samples = {}
    repeatability = None
    for role, scenes in [("fit", FIT), ("development", VALIDATION)]:
        samples[role] = []
        for name in scenes:
            for index, record in enumerate(replay_scene(ROOT / "data/carla" / name)):
                if index % 5:
                    continue
                tensor, _ = preprocess_rgb(record["rgb"], record["camera"])
                raw = predict(model, tensor)["relative_depth"][0, 0].cpu().numpy()
                if repeatability is None:
                    repeated = [predict(model, tensor)["relative_depth"][0, 0].cpu().numpy() for _ in range(2)]
                    repeatability = dict(exact_equal=all(np.array_equal(raw, r) for r in repeated), max_absolute_difference=max(float(np.max(np.abs(raw-r))) for r in repeated), repeats=3)
                native = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_LINEAR)
                gt = record["depth"]
                mask = np.zeros(gt.shape, dtype=bool)
                mask[96:456, 32:608] = True
                mask &= (gt >= 1) & (gt <= 50)
                if mask.sum() < 100:
                    raise ValueError("Insufficient reference pixels")
                samples[role].append(dict(scene=name, frame=record["frame_id"], prediction=native[mask][None, :], reference=gt[mask][None, :]))
                folder = output / "predictions" / name
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / (str(record["frame_id"]) + ".npy"), native, allow_pickle=False)
            print("Predicted", role, name, flush=True)
    predictions = [s["prediction"] for s in samples["fit"]]
    references = [s["reference"] for s in samples["fit"]]
    baseline_scale = fixed_scale(predictions, references)
    mapping = fit_power_depth(predictions, references)
    artifact = dict(schema_version=2, calibration_type="monotonic_power", **mapping, checkpoint_sha256=protocol["checkpoint_sha256"], preprocessing_id=PREPROCESSING_ID, protocol_sha256=sha256(output / "protocol.json"), fit_frames=len(predictions), supported_reference_range_m=[1, 50], ground_truth_usage="Calibration fit and evaluation only", learned_actuation=False)
    write_json(output / "frozen_mapping.json", artifact)
    rows = {}
    for label in ["fixed_scale", "power"]:
        rows[label] = []
        for sample in samples["development"]:
            mapped = sample["prediction"] * baseline_scale if label == "fixed_scale" else apply_power_depth(sample["prediction"], mapping["scale"], mapping["exponent"])
            rows[label].append(score(sample, mapped))
        write_json(output / (label + "_scores.json"), rows[label])
        with (output / (label + "_scores.csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[label][0]))
            writer.writeheader()
            writer.writerows(rows[label])
    promotion = paired_promotion(rows["fixed_scale"], rows["power"])
    for name, inventory in source_hashes.items():
        scene = ROOT / "data/carla" / name
        if set(inventory) != {p.relative_to(scene).as_posix() for p in scene.rglob("*") if p.is_file()}:
            raise ValueError("Source inventory changed")
        if any(sha256(scene / p) != h for p,h in inventory.items()):
            raise ValueError("Source data changed")
    summary = dict(status="development_pass" if promotion["passed"] else "development_rejected", baseline_scale=baseline_scale, mapping=artifact, selection=promotion, repeatability=repeatability, fit_frames=len(predictions), development_frames=len(rows["power"]), sources_unchanged=True, limits="Already-seen correlated development scenes. Postprocessing calibration only; network weights unchanged. No holdout scoring or learned actuation yet.")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output)

