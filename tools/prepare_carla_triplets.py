"""Prepare RGB-only temporal triplets from audited CARLA captures, with map-disjoint splits."""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import audit_scene, file_hash, write_json
from src.carla_capture import replay_scene, scene_path
from src.preprocess import PREPROCESSING_ID


def prepare(train_scenes, validation_scenes, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh triplet output directory")
    splits = {"train": [Path(p).resolve() for p in train_scenes],
              "validation": [Path(p).resolve() for p in validation_scenes]}
    roots = [p for paths in splits.values() for p in paths]
    if not all(splits.values()) or len(set(roots)) != len(roots):
        raise ValueError("Need separate, nonduplicate train and validation capture directories")
    metadata = {p: json.loads((p / "metadata.json").read_text(encoding="utf-8")) for p in roots}
    names = [m["scene_id"] for m in metadata.values()]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names):
        raise ValueError("Capture scene IDs must be safe and unique")
    maps = {split: {metadata[p]["settings"]["map"] for p in paths} for split, paths in splits.items()}
    if maps["train"] & maps["validation"]:
        raise ValueError("Training and validation maps must be disjoint for this pilot")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, task="self_supervised_rgb_triplets", preprocessing_id=PREPROCESSING_ID,
                    frame_offsets=[-1, 0, 1], ground_truth_usage="none; no depth files copied into training inputs",
                    selection="Consecutive 10 Hz frames; at least 0.01 m ego translation on both adjacent steps",
                    limitations="Preparation only, not trained weights. Map-disjoint pilot, not independent real-world generalization evidence.",
                    splits={"train": [], "validation": []}, images={}, captures=[])
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        for split, paths in splits.items():
            for source in paths:
                meta = metadata[source]
                name = meta["scene_id"]
                audit_path = staged / (name + "_audit.json")
                before = {p: file_hash(source / p) for p in ("metadata.json", "frames.csv")}
                if json.loads((source / "metadata.json").read_text(encoding="utf-8")) != meta:
                    raise ValueError("Capture metadata changed before audit")
                audit_scene(SimpleNamespace(scene=source, output=audit_path))
                folder = staged / split / name
                folder.mkdir(parents=True)
                window = deque(maxlen=3)
                retained, stationary = 0, 0
                for record in replay_scene(source):
                    window.append(record)
                    if len(window) < 3:
                        continue
                    positions = np.asarray([[json.loads(r["ego_transform"])["location"][k] for k in ("x", "y", "z")] for r in window])
                    if np.min(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.01:
                        stationary += 1
                        continue
                    images = []
                    for r in window:
                        original = scene_path(source, r["rgb_path"])
                        target = folder / original.name
                        relative = target.relative_to(staged).as_posix()
                        if relative not in manifest["images"]:
                            expected = file_hash(original)
                            shutil.copyfile(original, target)
                            if file_hash(target) != expected or file_hash(original) != expected:
                                raise ValueError("Capture RGB changed during preparation")
                            manifest["images"][relative] = expected
                        images.append(relative)
                    manifest["splits"][split].append(dict(scene=name, map=meta["settings"]["map"],
                                                         frame_ids=[r["frame_id"] for r in window], rgb=images,
                                                         timestamps=[float(r["simulation_timestamp"]) for r in window],
                                                         intrinsics=meta["intrinsics"]))
                    retained += 1
                if retained == 0:
                    raise ValueError("No moving temporal triplets in capture: " + name)
                if any(file_hash(source / p) != digest for p, digest in before.items()):
                    raise ValueError("Capture metadata/index changed during preparation")
                manifest["captures"].append(dict(scene=name, split=split, map=meta["settings"]["map"],
                                                source_sha256=before, retained_triplets=retained,
                                                rejected_stationary_triplets=stationary,
                                                audit=audit_path.name, audit_sha256=file_hash(audit_path)))
        write_json(staged / "triplets.json", manifest)
        staged.replace(output)
    print("Prepared RGB triplets:", {k: len(v) for k, v in manifest["splits"].items()}, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--output", default="data/training/carla_pilot")
    args = parser.parse_args()
    try:
        prepare(args.train_scene, args.validation_scene, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
