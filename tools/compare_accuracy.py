"""Compare pinned models on identical RGB/GT samples and driving inputs, sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project import write_json, file_hash, video_frames, BASELINE
from src.demo_data import verify_manifest
from tools.validate_delivery import source_hashes


def check_reports(evaluation, profile, identity):
    if evaluation["status"] != "complete" or profile["status"] != "pass":
        raise RuntimeError("Incomplete evaluation or unstable profile")
    if evaluation["manifest_sha256"] != identity["evaluation_manifest_sha256"] or profile["input_manifest_sha256"] != identity["input_manifest_sha256"]:
        raise RuntimeError("Comparison inputs changed between model runs")


def compare(output, device="auto"):
    from PIL import Image
    from types import SimpleNamespace
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "data/demo/input_manifest.json"
    manifest = verify_manifest(manifest_path, BASELINE)
    evaluation = ROOT / "data/evaluation/redwood/evaluation_manifest.json"
    summary = dict(status="partial", source_sha256=source_hashes(), input_manifest_sha256=file_hash(manifest_path),
                   evaluation_manifest_sha256=file_hash(evaluation), configurations=[],
                   limitations="Five correlated indoor diagnostic frames; resolutions compared on this diagnostic are exploratory, not independently tested. Driving clip has no GT. No calibrated metres or general driving-accuracy claim.")
    try:
        records = video_frames(SimpleNamespace(source="video", video=manifest_path.parent / manifest["video"], frames=200))
        try:
            for record in records:
                pass
        finally:
            records.close()
        if record["frame_id"] != 200:
            raise ValueError("Driving comparison requires frame 200")
        image = output / "shared_driving_frame_200.png"
        Image.fromarray(record["rgb"]).save(image)
        summary["driving_image_sha256"] = file_hash(image)
        for model, size in (("monodepth2", 294), ("depth-anything", 294), ("depth-anything", 392), ("depth-anything", 518)):
            name = model + ("_" + str(size) if model == "depth-anything" else "")
            row = dict(model=model, input_size=size if model == "depth-anything" else [192, 640], status="partial")
            summary["configurations"].append(row)
            folder = output / name
            folder.mkdir()
            commands = [
                ["evaluate", "--manifest", str(evaluation), "--output", str(folder / "evaluation")],
                ["profile", "--manifest", str(manifest_path), "--frames", "30", "--warmup", "3", "--repeats", "2", "--output", str(folder / "profile")],
                ["infer", "--image", str(image), "--output", str(folder / "driving_image")],
            ]
            for command in commands:
                print("Compare:", name, command[0], flush=True)
                with (folder / (command[0] + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, str(ROOT / "project.py")] + command +
                                            ["--model", model, "--input-size", str(size), "--device", device],
                                            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError("Comparison failed: " + name + " " + command[0] + "; inspect its log")
            measured = json.loads((folder / "evaluation/summary.json").read_text(encoding="utf-8"))
            profile = json.loads((folder / "profile/summary.json").read_text(encoding="utf-8"))
            check_reports(measured, profile, summary)
            row.update(status="complete", metrics=measured["mean_metrics"], samples=measured["evaluated_samples"],
                       checkpoint_sha256=measured["checkpoint_sha256"],
                       headless_fps=[run["uncapped_headless_fps"] for run in profile["runs"]],
                       peak_pytorch_mib=profile["pytorch_peak_allocated_mib"], process_memory=profile["process_memory"])
            write_json(output / "summary.json", summary)
        if summary["source_sha256"] != source_hashes():
            raise RuntimeError("Source changed during comparison")
        if file_hash(image) != summary["driving_image_sha256"]:
            raise RuntimeError("Driving comparison image changed")
        summary["status"] = "complete"
    except BaseException as error:
        summary["error"] = str(error) or type(error).__name__
        raise
    finally:
        write_json(output / "summary.json", summary)
    print("Model comparison completed:", output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/accuracy_comparison")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        compare(args.output, args.device)
    except (ValueError, RuntimeError, OSError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
