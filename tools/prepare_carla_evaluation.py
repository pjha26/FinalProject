"""Freeze validation-target RGB/depth pairs from an audited map-disjoint CARLA pilot."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import BASELINE, audit_scene, file_hash, write_json
from src.carla_capture import replay_scene, scene_path
from src.demo_data import checkpoint_hashes
from src.evaluation import evaluation_manifest, paired_samples
from src.preprocess import PREPROCESSING_ID


def prepare(triplets, scene, output, checkpoint=BASELINE):
    triplets, scene, output = Path(triplets).resolve(), Path(scene).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh CARLA evaluation output directory")
    triplet_hash = file_hash(triplets)
    training = json.loads(triplets.read_text(encoding="utf-8"))
    if training.get("schema_version") != 1 or training.get("task") != "self_supervised_rgb_triplets" or training.get("frame_offsets") != [-1, 0, 1]:
        raise ValueError("Need a prepared temporal pilot manifest")
    meta = json.loads((scene / "metadata.json").read_text(encoding="utf-8"))
    name = meta["scene_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name):
        raise ValueError("Unsafe evaluation scene identity")
    captures = training["captures"]
    selected = [c for c in captures if c["scene"] == name and c["split"] == "validation"]
    train_maps = {c["map"] for c in captures if c["split"] == "train"}
    if len(selected) != 1 or not train_maps or meta["settings"]["map"] in train_maps or selected[0]["map"] != meta["settings"]["map"]:
        raise ValueError("Evaluation requires the map-disjoint validation capture")
    source_hashes = {p: file_hash(scene / p) for p in ("metadata.json", "frames.csv")}
    if selected[0]["source_sha256"] != source_hashes:
        raise ValueError("Capture differs from prepared triplet source hashes")
    targets = {}
    for row in training["splits"]["validation"]:
        if row["scene"] != name:
            continue
        ids = row["frame_ids"]
        if len(ids) != 3 or ids != list(range(ids[1] - 1, ids[1] + 2)) or ids[1] in targets:
            raise ValueError("Validation targets must be unique consecutive triplets")
        targets[ids[1]] = training["images"][row["rgb"][1]]
    if not targets:
        raise ValueError("No moving validation targets")
    hashes = checkpoint_hashes(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as temporary:
        staged = Path(temporary) / "evaluation"
        staged.mkdir()
        audit_scene(SimpleNamespace(scene=scene, output=staged / "source_audit.json"))
        samples = []
        for record in replay_scene(scene):
            frame = record["frame_id"]
            if frame not in targets:
                continue
            identifier = name + "_{:08d}".format(frame)
            sample = dict(id=identifier, scene=name)
            for key, source_key in (("rgb", "rgb_path"), ("depth", "depth_path")):
                original = scene_path(scene, record[source_key])
                digest = file_hash(original)
                if key == "rgb" and digest != targets[frame]:
                    raise ValueError("Validation RGB differs from frozen training manifest")
                relative = key + "/" + identifier + original.suffix
                copied = staged / relative
                copied.parent.mkdir(exist_ok=True)
                shutil.copyfile(original, copied)
                if file_hash(copied) != digest or file_hash(original) != digest:
                    raise ValueError("Source evaluation asset changed during copy")
                sample[key], sample[key + "_sha256"] = relative, digest
            samples.append(sample)
        if len(samples) != len(targets):
            raise ValueError("Validation targets are missing from source capture")
        manifest = dict(schema_version=1, dataset="CARLA map-disjoint moving-target pilot", data_kind="simulated_rgbd", split="validation",
                        source="CARLA 0.9.13 locally captured RGB/depth sensors", preprocessing_id=PREPROCESSING_ID,
                        checkpoint_sha256=hashes, source_capture_sha256=source_hashes, triplets_manifest_sha256=triplet_hash,
                        selection="All unique middle frames of retained moving validation triplets; fixed before scoring",
                        reference_depth="Unchanged native CARLA numerical depth in metres; sensor geometry probed separately on Town04",
                        intrinsics=meta["intrinsics"], map=meta["settings"]["map"], seed=meta["settings"].get("seed"),
                        limitations="Tiny correlated Low-quality daylight simulator validation from one route/map; not an untouched test set, KITTI benchmark, training benefit or deployment metric-scale evidence.",
                        protocol=dict(depth_units="metres", aligned_to_rgb=True, min_depth=0.001, max_depth=80,
                                      scaling="per_image_median", aggregation="mean_of_images", crop="none"), samples=samples)
        path = staged / "evaluation_manifest.json"
        write_json(path, manifest)
        for _ in paired_samples(path, evaluation_manifest(path)):
            pass
        if file_hash(triplets) != triplet_hash or any(file_hash(scene / p) != digest for p, digest in source_hashes.items()):
            raise ValueError("Source manifests changed during preparation")
        staged.replace(output)
    print("Frozen CARLA validation targets:", len(samples), output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--validation-scene", required=True)
    parser.add_argument("--checkpoint", default=str(BASELINE))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        prepare(args.triplets, args.validation_scene, args.output, args.checkpoint)
    except (ValueError, OSError, RuntimeError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
