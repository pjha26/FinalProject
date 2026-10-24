"""Import a bounded, scene-audited KITTI selected-validation pilot from a local ZIP."""
import argparse
from collections import defaultdict
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import BASELINE, file_hash, write_json
from src.demo_data import checkpoint_hashes
from src.evaluation import evaluation_manifest, paired_samples
from src.model import UPSTREAM_REVISION
from src.preprocess import PREPROCESSING_ID

PREFIX = "depth_selection/val_selection_cropped/"
IMAGE_NAME = re.compile(r"^(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(\d{10})_image_02\.png$")
SPLIT_HASHES = {
    "train_files.txt": "00173a231642151c4693274e701ddd4cae7916520a757c7a94eae8e1173f7b0c",
    "val_files.txt": "64a868b8d2c66c3979a2d74c22ea2e9146850d5b9f5786ed8e5f92201ba1ad04",
}


def excluded_scenes():
    scenes = set()
    for name, expected in SPLIT_HASHES.items():
        path = ROOT / "third_party/monodepth2/splits/eigen_zhou" / name
        if file_hash(path) != expected:
            raise ValueError("Missing or changed pinned Monodepth2 split: " + name)
        scenes.update(line.split()[0].split("/")[-1] for line in path.read_text().splitlines())
    return scenes


def select_pairs(archive, excluded):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP member names")
    if len(entries) > 20000 or sum(entry.file_size for entry in entries) > 4 * 1024 ** 3:
        raise ValueError("Archive exceeds the selected-validation import budget")
    groups = defaultdict(list)
    excluded_found = set()
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in entry.orig_filename or ":" in entry.filename:
            raise ValueError("Unsafe ZIP member path")
        if entry.is_dir() or not entry.filename.startswith(PREFIX + "image/"):
            continue
        match = IMAGE_NAME.fullmatch(path.name)
        if not match:
            continue  # Only left-camera validation RGB; never anonymous test inputs.
        scene, frame = match.groups()
        if scene in excluded:
            excluded_found.add(scene)
        else:
            groups[scene].append((frame, entry.filename))
    if not groups or len(groups) > 32:
        raise ValueError("No eligible validation scenes, or scene budget exceeded")
    selected = []
    for scene, frames in sorted(groups.items()):
        frames.sort()
        if len(frames) < 3:
            raise ValueError("Need three left-camera samples per eligible scene: " + scene)
        for index in (0, len(frames) // 2, len(frames) - 1):
            frame, rgb = frames[index]
            stem = scene + "_image_" + frame + "_image_02"
            depth = PREFIX + "groundtruth_depth/" + scene + "_groundtruth_depth_" + frame + "_image_02.png"
            intrinsics = PREFIX + "intrinsics/" + stem + ".txt"
            for member in (rgb, depth, intrinsics):
                if member not in names or archive.getinfo(member).file_size > 8 * 1024 ** 2:
                    raise ValueError("Missing or oversized paired asset: " + member)
            selected.append(dict(scene=scene, frame=frame, rgb=rgb, depth=depth, intrinsics=intrinsics))
    return selected, sorted(excluded_found)


def prepare(archive_path, output):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    excluded = excluded_scenes()
    hashes = checkpoint_hashes(BASELINE)
    archive_hash = file_hash(archive_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        selected, excluded_found = select_pairs(archive, excluded)
        with tempfile.TemporaryDirectory(dir=str(output.parent)) as temporary:
            staged = Path(temporary) / "dataset"
            staged.mkdir()
            samples = []
            for pair in selected:
                identifier = pair["scene"] + "_" + pair["frame"] + "_02"
                rgb_bytes, depth_bytes = archive.read(pair["rgb"]), archive.read(pair["depth"])
                with Image.open(io.BytesIO(rgb_bytes)) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (1216, 352):
                        raise ValueError("Expected selected-validation 1216x352 RGB PNG")
                    image.load()
                # KITTI PNG depth is uint16 / 256 metres; zero remains invalid.
                if depth_bytes[:8] != b"\x89PNG\r\n\x1a\n" or depth_bytes[24:26] != bytes((16, 0)):
                    raise ValueError("Expected 16-bit grayscale KITTI depth PNG")
                with Image.open(io.BytesIO(depth_bytes)) as image:
                    if image.size != (1216, 352):
                        raise ValueError("Depth geometry/encoding does not match RGB")
                    raw = np.asarray(image)
                    if raw.shape != (352, 1216) or raw.dtype.kind not in "ui" or raw.min() < 0 or raw.max() > 65535:
                        raise ValueError("Depth geometry/encoding does not match RGB")
                    depth = raw.astype(np.float32) / 256
                if not ((depth > 0.001) & (depth < 80)).any():
                    raise ValueError("No valid scoring depth")
                calibration = archive.read(pair["intrinsics"])
                matrix = np.loadtxt(io.BytesIO(calibration)).reshape(3, 3)
                if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                    raise ValueError("Invalid camera intrinsics")
                rgb_path, depth_path = staged / (identifier + ".png"), staged / (identifier + ".npy")
                rgb_path.write_bytes(rgb_bytes)
                np.save(depth_path, depth, allow_pickle=False)
                calibration_path = staged / (identifier + "_intrinsics.txt")
                calibration_path.write_bytes(calibration)
                samples.append(dict(id=identifier, scene=pair["scene"], camera="image_02", frame=int(pair["frame"]),
                                    rgb=rgb_path.name, depth=depth_path.name, rgb_sha256=file_hash(rgb_path),
                                    depth_sha256=file_hash(depth_path), source_members=pair,
                                    source_depth_png_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                                    intrinsics=matrix.tolist(), intrinsics_file=calibration_path.name,
                                    intrinsics_sha256=file_hash(calibration_path)))
            manifest = dict(schema_version=1, dataset="KITTI selected validation cropped pilot", data_kind="real_rgbd", split="validation",
                            source="https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction",
                            source_archive_sha256=archive_hash, preprocessing_id=PREPROCESSING_ID, checkpoint_sha256=hashes,
                            limitations="Small cropped validation diagnostic; not official KITTI/Eigen scoring. Scenes excluded using pinned Monodepth2 Eigen-Zhou training/development lists; Depth Anything pretraining overlap is unknown. Three correlated frames per drive; no training performed.",
                            depth_conversion="16-bit PNG / 256 to float32 metres; zero preserved as invalid; no GT resizing",
                            selection="First, middle and last available left-camera frame in each eligible drive, before scoring",
                            split_audit=dict(source_revision=UPSTREAM_REVISION, split_sha256=SPLIT_HASHES,
                                             excluded_scenes=excluded_found, selected_scenes=sorted({p['scene'] for p in samples}),
                                             depth_anything_overlap="unknown"),
                            protocol=dict(depth_units="metres", aligned_to_rgb=True, min_depth=0.001, max_depth=80,
                                          scaling="per_image_median", aggregation="mean_of_images", crop="none"), samples=samples)
            manifest_path = staged / "evaluation_manifest.json"
            write_json(manifest_path, manifest)
            for _ in paired_samples(manifest_path, evaluation_manifest(manifest_path)):
                pass
            if file_hash(archive_path) != archive_hash:
                raise ValueError("Source ZIP changed during import")
            staged.replace(output)
    print("Prepared", len(samples), "samples from", len(manifest["split_audit"]["selected_scenes"]), "drives:", output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="data/evaluation/kitti_pilot")
    args = parser.parse_args()
    try:
        prepare(args.archive, args.output)
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
