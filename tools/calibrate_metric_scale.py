"""Fit one scale on Town04 and validate unchanged on Town05 development captures."""
import argparse
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import write_json
from src.carla_capture import replay_scene
from src.demo_data import checkpoint_hashes
from src.metric_control import fixed_scale, calibration_gate
from src.model import load_model, predict, sha256
from src.preprocess import preprocess_rgb, PREPROCESSING_ID

FIT = ["train_town04_09", "train_town04_14"]
VALIDATION = ["validation_town05_13", "validation_town05_s46_clearnoon_01", "validation_town05_s43_wetcloudynoon_01", "validation_town05_s46_wetcloudynoon_01"]


def run(output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    geometry = ROOT / "runs/native_carla_pilot_09/depth_geometry/summary.json"
    report = json.loads(geometry.read_text(encoding="utf-8"))
    if report["status"] != "pass" or report["convention"] != "camera_axis_z":
        raise ValueError("Verified axial-depth reference required")
    torch.set_num_threads(2)
    checkpoint = ROOT / "checkpoints/baseline_mono_640x192"
    protocol = dict(schema_version=1, fit_scenes=FIT, validation_scenes=VALIDATION, stride=5, checkpoint_sha256=checkpoint_hashes(checkpoint), preprocessing_id=PREPROCESSING_ID, geometry_sha256=sha256(geometry), depth_convention="camera_axis_z", reference_bounds_m=[1, 50], crop="x5:95%, y20:95%", scaling="One frozen median of per-frame median GT/prediction ratios; never per-validation-frame scaling", gates=dict(mean_abs_rel=0.25, p95_frame_abs_rel=0.50), source_sha256={})
    for name in FIT + VALIDATION:
        scene = ROOT / "data/carla" / name
        protocol["source_sha256"][name] = {p.relative_to(scene).as_posix(): sha256(p) for p in sorted(scene.rglob("*")) if p.is_file()}
    write_json(output / "protocol.json", protocol)
    model = load_model(checkpoint, device)
    samples = {}
    for role, scenes in [("fit", FIT), ("validation", VALIDATION)]:
        samples[role] = []
        for name in scenes:
            for index, record in enumerate(replay_scene(ROOT / "data/carla" / name)):
                if index % protocol["stride"]:
                    continue
                tensor, _ = preprocess_rgb(record["rgb"], record["camera"])
                relative = predict(model, tensor)["relative_depth"][0, 0].cpu().numpy()
                relative = cv2.resize(relative, (640, 480), interpolation=cv2.INTER_LINEAR)
                gt = record["depth"]
                mask = np.zeros(gt.shape, dtype=bool)
                mask[96:456, 32:608] = True
                mask &= (gt >= 1) & (gt <= 50)
                if mask.sum() < 100:
                    raise ValueError("Insufficient reference pixels")
                samples[role].append(dict(scene=name, frame=record["frame_id"], prediction=relative[mask][None, :], reference=gt[mask][None, :]))
            print("Prepared", role, name, flush=True)
    scale = fixed_scale([s["prediction"] for s in samples["fit"]], [s["reference"] for s in samples["fit"]])
    rows = []
    for s in samples["validation"]:
        pred, gt = s["prediction"] * scale, s["reference"]
        rows.append(dict(scene=s["scene"], frame=s["frame"], valid_pixels=gt.size, abs_rel=float(np.mean(np.abs(pred - gt) / gt)), mae_m=float(np.mean(np.abs(pred - gt))), rmse_m=float(np.sqrt(np.mean((pred - gt) ** 2)))))
    gate = calibration_gate(rows)
    artifact = dict(schema_version=1, scale=scale, status="accepted_development" if gate["passed"] else "rejected_development", gate=gate, checkpoint_sha256=protocol["checkpoint_sha256"], preprocessing_id=PREPROCESSING_ID, protocol_sha256=sha256(output / "protocol.json"), camera=dict(width=640, height=480, fov=90, mount_x=1.5, mount_z=1.7), fit_frames=len(samples["fit"]), validation_frames=len(rows), ground_truth_used="fit and evaluation only; not runtime scaling", limits="Already-seen development scenes; fixed-scale experiment only. Not metric deployment certification or unseen-route final acceptance.")
    write_json(output / "calibration.json", artifact)
    write_json(output / "validation.json", rows)
    with (output / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.output, args.device)

