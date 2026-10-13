"""Portable identity and decoding checks for the Phase 2 driving input."""
import json
from pathlib import Path

import numpy as np

from src.model import UPSTREAM_REVISION, sha256
from src.preprocess import PREPROCESSING_ID

SAMPLE_REVISION = "91b080357290da37f533e2d0832f601c3e770d51"
SAMPLE_BASE = "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/" + SAMPLE_REVISION + "/"
SAMPLE_FILES = {
    "driving.mp4": ("project_video.mp4", "651949922042c223375d40bf8929def97dbb62c6ade651d1f5b08eea47127d5f"),
    "UDACITY_LICENSE.txt": ("LICENSE", "404cc661d6b73f01ae0a6a7bb89f3bbf84844c2d993df254a3e0affe3c593dd1"),
}


def checkpoint_hashes(checkpoint):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("preprocessing_id") != PREPROCESSING_ID or manifest.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("Demo requires the compatible pinned baseline checkpoint")
    expected = manifest.get("sha256")
    if not isinstance(expected, dict):
        raise ValueError("Checkpoint manifest has no valid hashes")
    hashes = {}
    for name in ("encoder.pth", "depth.pth"):
        hashes[name] = sha256(checkpoint / name)
        if hashes[name] != expected.get(name):
            raise ValueError("Checkpoint checksum mismatch: " + name)
    return hashes


def inspect_video(path):
    """Decode every frame with bounded memory; do not trust container counts alone."""
    import cv2

    capture = cv2.VideoCapture(str(Path(path).resolve()))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open demo video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("Demo video needs a valid source frame rate")
        count, shape, first = 0, None, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) <= 0:
                raise ValueError("Invalid decoded video frame")
            if shape is not None and frame.shape != shape:
                raise ValueError("Demo video dimensions changed during decoding")
            shape = frame.shape
            if count == 0:
                first = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
        if count == 0:
            raise ValueError("Demo video contains no readable frames")
        if np.isfinite(expected) and expected > 0 and count != round(expected):
            raise ValueError("Demo video ended before its declared frame count")
        return {"decoded_frames": count, "width": shape[1], "height": shape[0],
                "source_fps": fps, "source_duration_seconds": count / fps}, first
    finally:
        capture.release()


def verify_manifest(manifest_path, checkpoint):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("preprocessing_id") != PREPROCESSING_ID or manifest.get("source_revision") != SAMPLE_REVISION:
        raise ValueError("Unsupported or incompatible demo input manifest")
    if manifest.get("video") != "driving.mp4" or manifest.get("ground_truth") != "unavailable":
        raise ValueError("Invalid demo input identity or ground-truth claim")
    if manifest.get("checkpoint_sha256") != checkpoint_hashes(checkpoint):
        raise ValueError("Demo checkpoint differs from the frozen input manifest")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or set(files) != set(SAMPLE_FILES) | {"driving_first_frame.png"}:
        raise ValueError("Unexpected demo asset paths")
    for name, expected in files.items():
        asset = (manifest_path.parent / name).resolve()
        if manifest_path.parent not in asset.parents or not asset.is_file() or sha256(asset) != expected:
            raise ValueError("Demo asset missing or checksum mismatch: " + name)
        if name in SAMPLE_FILES and expected != SAMPLE_FILES[name][1]:
            raise ValueError("Demo sample differs from its pinned source: " + name)
    information, _ = inspect_video(manifest_path.parent / manifest["video"])
    for key, value in information.items():
        if manifest.get(key) != value:
            raise ValueError("Demo video metadata mismatch: " + key)
    return manifest
