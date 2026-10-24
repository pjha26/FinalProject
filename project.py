"""Low-memory RGB depth demos, image inference, and optional CARLA tools."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REVISION = "b017a8164fae339b0ce01f3a9e5048104d5abd0e"
BASELINE = ROOT / "checkpoints" / "baseline_mono_640x192"
MODEL_URL = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip"
MODEL_MD5 = "a964b8356e08a02d009609d9e3928f7c"
SERVER_URL = "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.13.zip"


def selected_model(args):
    if getattr(args, "model", "monodepth2") == "depth-anything":
        from src.depth_anything import load_model
        if Path(args.checkpoint).resolve() != BASELINE.resolve():
            raise ValueError("--checkpoint selects Monodepth2 weights; Depth Anything uses its separately pinned assets")
        return load_model(args.device, getattr(args, "input_size", 294))
    from src.model import load_model
    if getattr(args, "input_size", 294) != 294:
        raise ValueError("--input-size applies only to --model depth-anything")
    return load_model(args.checkpoint, args.device)


def model_details(model):
    if getattr(model, "backend", None) == "depth-anything":
        return model.details()
    from src.preprocess import PREPROCESSING_ID
    return dict(backend="monodepth2", model_name="Monodepth2 ResNet-18 mono_640x192",
                model_input=[192, 640], preprocessing_id=PREPROCESSING_ID)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "FinalYearDepth/phase1"}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        resumed = response.status == 206 and offset > 0
        if resumed and not response.headers.get("Content-Range", "").startswith("bytes {}-".format(offset)):
            raise ValueError("Unexpected download resume range")
        total = offset if resumed else 0
        reported = total
        print("Downloading", url, flush=True)
        with partial.open("ab" if resumed else "wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                total += len(block)
                if total - reported >= 100 * 1024 * 1024:
                    print("  {:.0f} MiB".format(total / 1024 ** 2), flush=True)
                    reported = total
        expected = response.headers.get("Content-Length")
        if expected is not None and total - (offset if resumed else 0) != int(expected):
            raise OSError("Incomplete download; partial file retained for retry")
    partial.replace(destination)
    return destination


def prepare_demo(args):
    from PIL import Image
    from src.demo_data import SAMPLE_BASE, SAMPLE_FILES, SAMPLE_REVISION, checkpoint_hashes, inspect_video, verify_manifest
    from src.preprocess import PREPROCESSING_ID

    output = Path(args.output)
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        verify_manifest(manifest_path, args.checkpoint)
        print("Existing demo input verified:", manifest_path)
        return
    hashes = checkpoint_hashes(args.checkpoint)
    for local, (remote, expected) in SAMPLE_FILES.items():
        path = download(SAMPLE_BASE + remote, output / local)
        if file_hash(path) != expected:
            raise ValueError("Pinned demo checksum mismatch: " + local)
    information, first = inspect_video(output / "driving.mp4")
    image_path = output / "driving_first_frame.png"
    if image_path.exists():
        raise FileExistsError("Unmanifested fallback image exists; choose a fresh output directory")
    Image.fromarray(first).save(image_path)
    files = {name: expected for name, (_, expected) in SAMPLE_FILES.items()}
    files[image_path.name] = file_hash(image_path)
    write_json(manifest_path, dict(information, schema_version=1, video="driving.mp4", files=files,
               source_revision=SAMPLE_REVISION, source_url=SAMPLE_BASE + "project_video.mp4",
               attribution="Udacity, Inc.; CarND-Advanced-Lane-Lines repository sample; MIT license retained",
               input_kind="real driving video replay", ground_truth="unavailable",
               preprocessing_id=PREPROCESSING_ID, checkpoint_sha256=hashes))
    print("Demo input prepared:", manifest_path)


def prepare_reference_evaluation(args):
    import numpy as np
    from PIL import Image
    from src.demo_data import checkpoint_hashes
    from src.evaluation import REFERENCE_URL, REFERENCE_SHA256, evaluation_manifest
    from src.preprocess import PREPROCESSING_ID

    output = Path(args.output)
    if output.exists():
        evaluation_manifest(output / "evaluation_manifest.json")
        print("Existing evaluation input verified:", output)
        return
    checkpoint = checkpoint_hashes(args.checkpoint)
    archive = download(REFERENCE_URL, ROOT / ".cache" / "SampleRedwoodRGBDImages.zip")
    if file_hash(archive) != REFERENCE_SHA256:
        raise ValueError("RGB-D reference archive checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "reference"
        safe_extract(archive, staged, max_bytes=32 * 1024 ** 2)
        samples = []
        for index in range(5):
            identifier = "{:05d}".format(index)
            rgb_path = staged / "color" / (identifier + ".jpg")
            raw_path = staged / "depth" / (identifier + ".png")
            with Image.open(raw_path) as image:
                raw = np.asarray(image)
            if raw.ndim != 2 or raw.dtype.kind not in "ui" or raw.min() < 0 or raw.max() > 65535:
                raise ValueError("Reference depth must be a 16-bit millimetre image")
            depth_path = staged / "depth" / (identifier + ".npy")
            np.save(depth_path, raw.astype(np.float32) / 1000, allow_pickle=False)
            samples.append(dict(id=identifier, scene="redwood_living_room1", rgb=rgb_path.relative_to(staged).as_posix(),
                                depth=depth_path.relative_to(staged).as_posix(), rgb_sha256=file_hash(rgb_path), depth_sha256=file_hash(depth_path)))
        manifest = dict(schema_version=1, dataset="Open3D SampleRedwoodRGBDImages", source=REFERENCE_URL,
                        source_archive_sha256=REFERENCE_SHA256, data_kind="rgbd_reference", split="diagnostic",
                        preprocessing_id=PREPROCESSING_ID, checkpoint_sha256=checkpoint,
                        limitations="Five correlated indoor reference frames from one scene; not KITTI/NYU benchmark evidence or proof of real-world driving accuracy.",
                        depth_conversion="Reference PNG integer millimetres divided by 1000; GT is used only for scoring.",
                        protocol=dict(depth_units="metres", aligned_to_rgb=True, min_depth=0.001, max_depth=80,
                                      crop="none", scaling="per_image_median", aggregation="mean_of_images"), samples=samples)
        write_json(staged / "evaluation_manifest.json", manifest)
        evaluation_manifest(staged / "evaluation_manifest.json")
        staged.replace(output)
    print("Evaluation reference prepared:", output / "evaluation_manifest.json")


def evaluate_model(args):
    import numpy as np
    import torch
    from PIL import Image
    from src.demo_data import checkpoint_hashes
    from src.evaluation import evaluation_manifest, paired_samples, score_inverse_depth, aggregate_scores
    from src.visualize import render_frame

    manifest = evaluation_manifest(args.manifest)
    hashes = checkpoint_hashes(args.checkpoint)
    if hashes != manifest.get("checkpoint_sha256"):
        raise ValueError("Evaluation checkpoint differs from the frozen manifest")
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = selected_model(args)
    rows, complete = [], False
    try:
        with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for sample, rgb, gt in paired_samples(args.manifest, manifest):
                if not rows:
                    timed_prediction(model, rgb)
                maps, _, model_ms, processing_ms = timed_prediction(model, rgb)
                measured = score_inverse_depth(gt, maps["inverse_depth"], manifest["protocol"]["crop"])
                row = dict(id=sample["id"], scene=sample["scene"], model_ms=model_ms, processing_ms=processing_ms, **measured)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                np.save(output / (sample["id"] + "_relative_depth.npy"), maps["relative_depth"], allow_pickle=False)
                Image.fromarray(render_frame(rgb, maps["inverse_depth"], "Relative prediction | {} | GT median-aligned diagnostic".format(sample["id"]))).save(output / (sample["id"] + "_preview.png"))
        complete = True
    finally:
        summary = dict(schema_version=1, status="complete" if complete else "partial", dataset=manifest["dataset"],
                       data_kind=manifest["data_kind"], split=manifest["split"], limitations=manifest["limitations"],
                       evaluated_samples=len(rows), expected_samples=len(manifest["samples"]),
                       scenes=len({row["scene"] for row in rows}), protocol=manifest["protocol"],
                       manifest_sha256=file_hash(args.manifest), checkpoint_sha256=hashes,
                       device=str(model.device), mean_metrics=aggregate_scores(rows) if complete else None,
                       total_valid_pixels=sum(row["valid_pixels"] for row in rows),
                       depth_claim="Predictions remain relative; scores use per-image GT median scaling, not recovered metric scale.")
        summary.update(reference_checkpoint_sha256=hashes, **model_details(model))
        write_json(output / "summary.json", summary)
    print("Evaluation completed:", len(rows), "samples; diagnostic scope:", manifest["limitations"])


def safe_extract(archive, destination, strip_root=False, max_bytes=1024 ** 3):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        if sum(info.file_size for info in zipped.infolist()) > max_bytes:
            raise ValueError("Archive exceeds uncompressed size budget")
        entries = []
        for info in zipped.infolist():
            relative = PurePosixPath(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
                raise ValueError("Unsafe archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are unsupported")
            parts = relative.parts[1:] if strip_root else relative.parts
            if not parts:
                continue
            target = destination.joinpath(*parts).resolve()
            if destination not in target.parents:
                raise ValueError("Archive path escapes extraction directory")
            entries.append((info, target))
        for info, target in entries:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def install_assets():
    from src.preprocess import PREPROCESSING_ID

    cache = ROOT / ".cache"
    source = ROOT / "third_party" / "monodepth2"
    if not (source / "provenance.json").exists():
        if source.exists():
            raise FileExistsError("Inspect incomplete source directory before retrying: " + str(source))
        archive = download("https://codeload.github.com/nianticlabs/monodepth2/zip/" + REVISION, cache / (REVISION + ".zip"))
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
            staged = Path(temporary) / "source"
            safe_extract(archive, staged, strip_root=True)
            encoder = staged / "networks" / "resnet_encoder.py"
            original = encoder.read_text(encoding="utf-8")
            old = "self.encoder = resnets[num_layers](pretrained)"
            if original.count(old) != 1:
                raise ValueError("Pinned encoder compatibility patch no longer matches")
            encoder.write_text(original.replace(old, 'self.encoder = resnets[num_layers](weights="DEFAULT" if pretrained else None)'), encoding="utf-8")
            hashes = {name: file_hash(staged / name) for name in ("networks/resnet_encoder.py", "networks/depth_decoder.py", "layers.py")}
            write_json(staged / "provenance.json", {"revision": REVISION, "archive_sha256": file_hash(archive),
                       "patch": "torchvision keyword-only weights API; architecture and normalization unchanged", "sha256": hashes})
            staged.replace(source)
    if not (BASELINE / "manifest.json").exists():
        if BASELINE.exists():
            raise FileExistsError("Inspect incomplete checkpoint directory before retrying: " + str(BASELINE))
        archive = download(MODEL_URL, cache / "mono_640x192.zip")
        if file_hash(archive, "md5") != MODEL_MD5:
            raise ValueError("Official model archive checksum failed; do not load its contents")
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASELINE.parent) as temporary:
            staged = Path(temporary) / "checkpoint"
            safe_extract(archive, staged)
            hashes = {name: file_hash(staged / name) for name in ("encoder.pth", "depth.pth")}
            write_json(staged / "manifest.json", {"schema_version": 1, "model": "mono_640x192", "modality": "monocular",
                       "preprocessing_id": PREPROCESSING_ID, "upstream_revision": REVISION, "source_url": MODEL_URL,
                       "archive_md5": MODEL_MD5, "archive_sha256": file_hash(archive), "sha256": hashes})
            staged.replace(BASELINE)
    fixture = ROOT / "data" / "fixtures" / "test_image.jpg"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if not fixture.exists():
        shutil.copyfile(source / "assets" / "test_image.jpg", fixture)
    print("Pinned inference source, immutable baseline, and official image ready.")


def install_server(local_archive=None):
    if platform.system() != "Windows":
        raise RuntimeError("This server installation command is for Windows; use the official Linux package on Linux")
    server = ROOT / ".runtime" / "carla"
    if (server / "CarlaUE4.exe").exists():
        print("CARLA server already installed:", server / "CarlaUE4.exe")
        return
    if server.exists():
        raise FileExistsError("Inspect partial CARLA server directory before retrying")
    if shutil.disk_usage(ROOT).free < 32 * 1024 ** 3:
        raise OSError("CARLA archive/extraction requires at least 32 GiB free space")
    archive = Path(local_archive).resolve() if local_archive else download(SERVER_URL, ROOT / ".cache" / "CARLA_0.9.13.zip")
    if not archive.is_file():
        raise FileNotFoundError("CARLA archive does not exist: " + str(archive))
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        nested = "WindowsNoEditor/CarlaUE4.exe" in names
        if nested and any(not name.startswith("WindowsNoEditor/") for name in names):
            raise ValueError("CARLA archive has a conflicting root layout")
    server.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=server.parent) as temporary:
        staged = Path(temporary) / "server"
        print("Extracting CARLA 0.9.13...", flush=True)
        safe_extract(archive, staged, strip_root=nested, max_bytes=32 * 1024 ** 3)
        if not (staged / "CarlaUE4.exe").is_file():
            raise FileNotFoundError("Expected Windows server executable missing from official archive")
        write_json(staged / "download_provenance.json", {"version": "0.9.13", "source_url": SERVER_URL if not local_archive else "user-supplied official archive",
                   "archive_sha256": file_hash(archive)})
        staged.replace(server)
    print("CARLA installed:", server / "CarlaUE4.exe")


def environment_report(args):
    report = {"schema_version": 1, "python": sys.version, "platform": platform.platform(),
              "disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 2), "dependencies": {}}
    for name in ("torch", "torchvision", "numpy", "PIL", "cv2", "pygame"):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = getattr(module, "__version__", "imported")
        except (ImportError, OSError) as error:
            report["dependencies"][name] = {"error": str(error)}
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            torch.ones(1, device="cuda").sum().item()
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_vram_mib"] = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        report["cuda_build"] = torch.version.cuda
    except (ImportError, RuntimeError, OSError) as error:
        report["cuda_error"] = str(error)
    if getattr(args, "carla", False):
        try:
            from src.carla_capture import create_client
            client = create_client(args.host, args.port, 3)
            report["carla_client"] = client.get_client_version()
            report["carla_server"] = client.get_server_version()
            report["maps"] = client.get_available_maps()
        except (ImportError, RuntimeError, OSError) as error:
            report["carla_server_error"] = str(error)
    report["phase1_status"] = "diagnostics only; webcam/video workflow does not require CARLA"
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def timed_prediction(model, rgb, camera=None):
    if getattr(model, "backend", None) == "depth-anything":
        from src.depth_anything import timed_prediction as small_prediction
        return small_prediction(model, rgb, camera)
    import torch
    from src.model import predict
    from src.preprocess import preprocess_rgb

    start = time.perf_counter()
    tensor, intrinsics = preprocess_rgb(rgb, camera)
    tensor = tensor.to(model.device)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    model_start = time.perf_counter()
    result = predict(model, tensor)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    model_ms = (time.perf_counter() - model_start) * 1000
    numerical = {key: value[0, 0].cpu().numpy() for key, value in result.items()}
    return numerical, intrinsics, model_ms, (time.perf_counter() - start) * 1000


def infer_image(args):
    import numpy as np
    from PIL import Image
    from src.model import sha256
    from src.preprocess import PREPROCESSING_ID
    from src.visualize import render_frame

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    model = selected_model(args)
    timed_prediction(model, rgb)  # Warmup is excluded from the reported prediction.
    maps, intrinsics, model_ms, processing_ms = timed_prediction(model, rgb)
    for name, values in maps.items():
        np.save(output / (name + ".npy"), values, allow_pickle=False)
    Image.fromarray(render_frame(rgb, maps["inverse_depth"], "{} | Relative depth | {} | model {:.1f} ms".format(model_details(model)["model_name"], model.device, model_ms))).save(output / "preview.png")
    cloud = None
    if getattr(args, "point_cloud", False):
        from src.point_cloud import export_point_cloud
        cloud = export_point_cloud(output, rgb, maps["relative_depth"], intrinsics, model_details(model))
    write_json(output / "summary.json", {"schema_version": 1, "checkpoint": str(model.checkpoint),
               "encoder_sha256": sha256(model.checkpoint / "encoder.pth") if getattr(model, "backend", None) != "depth-anything" else None,
               "preprocessing_id": PREPROCESSING_ID,
               "input": str(Path(args.image).resolve()), "intrinsics_resized": intrinsics.tolist(),
               "intrinsics_source": "assumed 90-degree FOV for diagnostic image; not used by RGB inference",
               "model_ms": model_ms, "processing_ms": processing_ms, "device": str(model.device),
               "depth_units": "relative; not calibrated metres", "point_cloud": cloud, **model_details(model)})
    print("Inference saved:", output / "preview.png")


def capture_settings(args):
    from src.carla_capture import CaptureSettings
    return CaptureSettings(host=args.host, port=args.port, map=args.map, seed=args.seed, timeout=getattr(args, "timeout", 15.0))


def audit_scene(args):
    """Audit stored sensor identity/timing; this is not a model accuracy benchmark."""
    import numpy as np
    from src.carla_capture import replay_scene
    from src.preprocess import camera_intrinsics

    scene = Path(args.scene)
    metadata = json.loads((scene / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["settings"]["width"], metadata["settings"]["height"]) != (640, 480):
        raise ValueError("CARLA pilot must use 640x480 cameras")
    expected_k = camera_intrinsics(640, 480, metadata["settings"].get("fov", 90))
    matrix = np.asarray(metadata["intrinsics"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix, expected_k, rtol=0, atol=1e-6):
        raise ValueError("Pilot intrinsics do not match the capture configuration")
    if metadata.get("depth_units") != "metres":
        raise ValueError("Pilot depth units must be metres")
    previous_frame = previous_time = None
    frames, finite_valid_pixels, timestamps = 0, 0, []
    for record in replay_scene(scene):
        frame_id, timestamp = record["frame_id"], float(record["simulation_timestamp"])
        if not np.isfinite(timestamp):
            raise ValueError("Nonfinite simulation timestamp")
        if previous_frame is not None:
            if frame_id != previous_frame + 1 or not np.isclose(timestamp - previous_time, 0.1, atol=1e-4, rtol=0):
                raise ValueError("Pilot contains dropped frames or incorrect 10 Hz simulation timing")
        for key in ("rgb_path", "depth_path"):
            if int(Path(record[key]).stem) != frame_id:
                raise ValueError("Sensor filename does not match recorded frame ID")
        for key in ("ego_transform", "sensor_transform"):
            transform = json.loads(record[key])
            values = [transform["location"][k] for k in ("x", "y", "z")] + [transform["rotation"][k] for k in ("pitch", "yaw", "roll")]
            if not np.isfinite(values).all():
                raise ValueError("Invalid pose metadata")
        if record["depth"].max() > 1000.001:
            raise ValueError("Depth exceeds CARLA's raw encoding range")
        finite_valid_pixels += int(((record["depth"] > 1e-3) & (record["depth"] < 80)).sum())
        previous_frame, previous_time = frame_id, timestamp
        timestamps.append(timestamp)
        frames += 1
    if frames == 0 or finite_valid_pixels == 0:
        raise ValueError("Pilot has no frames or no valid evaluation-range ground truth")
    report = {"schema_version": 1, "status": "pass", "scene": str(scene.resolve()), "frames": frames,
              "frame_pairs": "matched IDs and numerical dimensions", "simulation_hz": 10,
              "valid_depth_pixels_below_80m": finite_valid_pixels,
              "simulation_span_seconds": timestamps[-1] - timestamps[0],
              "manifest_sha256": file_hash(scene / "metadata.json"), "frame_index_sha256": file_hash(scene / "frames.csv"),
              "scope": "stored-frame integrity and simulation timing; not wall-clock FPS or model accuracy"}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def video_frames(args):
    """Stream a local video or camera without retaining decoded frames."""
    import cv2
    import numpy as np

    if args.source == "video":
        path = Path(getattr(args, "video", "") or "")
        if not path.is_file():
            raise ValueError("--video must name an existing local video file")
        source = str(path.resolve())
    else:
        source = getattr(args, "camera", 0)
        if source < 0:
            raise ValueError("--camera must be a nonnegative device index")
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open {} input; check the file or camera permission/device index".format(args.source))
        if args.source == "webcam":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, getattr(args, "fps", 10))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        native_fps = native_fps if 0 < native_fps <= 240 else 0
        expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) if args.source == "video" else 0
        expected_frames = round(expected_frames) if np.isfinite(expected_frames) and expected_frames > 0 else 0
        for index in range(args.frames):
            ok, frame = capture.read()
            arrival = time.perf_counter()
            if not ok:
                if args.source == "webcam":
                    raise RuntimeError("Camera stopped supplying frames")
                if index == 0 or index < expected_frames:
                    raise RuntimeError("Video decode stopped at frame {} before its declared end ({}); verify or replace the clip".format(index, expected_frames or "unknown"))
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) == 0:
                raise ValueError("Invalid decoded RGB frame")
            height, width = frame.shape[:2]
            scale = min(640 / width, 480 / height, 1)
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            yield {"frame_id": index + 1, "rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                   "camera": None, "arrival_timestamp": arrival, "input_fps": native_fps}
    finally:
        capture.release()


def demo(args):
    import numpy as np
    from PIL import Image
    from src.carla_capture import CarlaCapture, replay_scene
    from src.preprocess import camera_intrinsics
    from src.visualize import render_frame
    from src.evaluation import runtime_evidence, save_review_snapshot, score_inverse_depth
    from src.profiling import process_memory_mib

    if args.source == "replay" and not args.scene:
        raise ValueError("--scene is required for replay")
    fps_limit = getattr(args, "fps", 10)
    if not 0 < fps_limit <= 60:
        raise ValueError("--fps must be between 0 (exclusive) and 60")
    minimum_seconds, review_every = getattr(args, "min_seconds", 0), getattr(args, "review_every", 0)
    if not np.isfinite(minimum_seconds) or minimum_seconds < 0 or review_every < 0:
        raise ValueError("--min-seconds and --review-every must be nonnegative")
    if minimum_seconds and args.headless:
        raise ValueError("The displayed-runtime acceptance gate cannot run with --headless")
    manifest_path = getattr(args, "manifest", None)
    input_manifest_hash = None
    if manifest_path:
        from src.demo_data import verify_manifest
        if args.source != "video":
            raise ValueError("--manifest is only supported for the prepared driving-video source")
        manifest = verify_manifest(manifest_path, args.checkpoint)
        if not getattr(args, "video", None) or Path(args.video).resolve() != (Path(manifest_path).resolve().parent / manifest["video"]).resolve():
            raise ValueError("--video does not match the verified input manifest")
        input_manifest_hash = file_hash(manifest_path)
    import torch
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    display = capture = records = None
    pygame_started = False
    rows = []
    try:
        print("Loading {} on {}...".format(getattr(args, "model", "monodepth2"), args.device), flush=True)
        model = selected_model(args)
        if getattr(getattr(model, "device", None), "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        if not args.headless:
            import pygame
            pygame.init()
            pygame_started = True
            display = pygame.display.set_mode((1280, 360))
            pygame.display.set_caption(model_details(model)["model_name"] + " - RGB depth demo - Escape to stop")
        if args.source == "carla":
            settings = capture_settings(args)
            capture = CarlaCapture(settings).__enter__()
            camera = {"width": 640, "height": 480, "intrinsics": camera_intrinsics(640, 480, 90).tolist()}
            records = (dict(capture.next_frame(), camera=camera) for _ in range(args.frames))
        elif args.source == "replay":
            records = replay_scene(args.scene)
        else:
            records = video_frames(args)
        rows, metrics, reviews, last_review, start = [], [], [], None, time.perf_counter()
        cloud = None
        cycle_start = start
        for index, record in enumerate(records):
            if index >= args.frames:
                break
            if display is not None and any(event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) for event in pygame.event.get()):
                break
            frame_start = time.perf_counter()
            arrival_start = record.get("arrival_timestamp", frame_start)
            maps, intrinsics, model_ms, processing_ms = timed_prediction(model, record["rgb"], record["camera"])
            gt = record.get("depth")
            measured = None
            if gt is not None:
                try:
                    measured = score_inverse_depth(gt, maps["inverse_depth"])
                    metrics.append(dict(measured, frame_id=record["frame_id"], status="valid"))
                except ValueError as error:
                    metrics.append({"frame_id": record["frame_id"], "status": "invalid", "error": str(error)})
            elapsed = time.perf_counter() - start
            caption = "{} | {} | {} | frame {} | {:.1f} completed FPS".format(model_details(model)["model_name"], args.source, getattr(model, "device", "unknown"), record["frame_id"], len(rows) / max(elapsed, 1e-9))
            if measured:
                caption += " | GT-aligned AbsRel {:.3f}, RMSE {:.2f} m".format(measured["abs_rel"], measured["rmse"])
            else:
                caption += " | GT metrics unavailable"
            preview = render_frame(record["rgb"], maps["inverse_depth"], caption)
            if index == 0:
                Image.fromarray(preview).save(output / "preview.png")
                for name, values in maps.items():
                    np.save(output / (name + ".npy"), values, allow_pickle=False)
                if getattr(args, "point_cloud", False):
                    from src.point_cloud import export_point_cloud
                    details = model_details(model)
                    if record.get("camera"):
                        details["intrinsics_source"] = "Provided camera intrinsics; depth remains model-estimated and relative"
                    cloud = export_point_cloud(output, record["rgb"], maps["relative_depth"], intrinsics, details)
            if display is not None:
                surface = pygame.surfarray.make_surface(preview.transpose(1, 0, 2))
                target = surface.get_rect().fit(display.get_rect())
                display.fill((20, 20, 20))
                display.blit(pygame.transform.smoothscale(surface, target.size), target)
                pygame.display.flip()
            display_end = time.perf_counter()
            if review_every:
                last_review = (record["frame_id"], record["rgb"], maps, preview)
                if index < 2 or record["frame_id"] % review_every in (0, 1):
                    saved = save_review_snapshot(output, *last_review)
                    if saved is not None:
                        reviews.append(saved)
            if args.source in ("webcam", "video"):
                playback_fps = min(fps_limit, record["input_fps"]) if record["input_fps"] else fps_limit
                time.sleep(max(0, 1 / playback_fps - (time.perf_counter() - cycle_start)))
            cycle_end = time.perf_counter()
            rows.append({"frame_id": record["frame_id"], "model_ms": model_ms, "processing_ms": processing_ms,
                         "capture_to_display_ms": (display_end - arrival_start) * 1000,
                         "full_cycle_ms": (cycle_end - cycle_start) * 1000})
            cycle_start = cycle_end
        elapsed = time.perf_counter() - start
        if not rows:
            raise ValueError("No frames processed")
        if last_review is not None:
            saved = save_review_snapshot(output, *last_review)
            if saved is not None:
                reviews.append(saved)
            write_json(output / "review.json", reviews)
        evidence = runtime_evidence(rows, elapsed, args.headless, minimum_seconds)
        with (output / "runtime.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "metrics.json", metrics)
        write_json(output / "summary.json", {"schema_version": 1, "source": args.source, "checkpoint": str(model.checkpoint),
                   "frames": len(rows), "wall_seconds": elapsed, "processed_fps": len(rows) / elapsed,
                   "headless": args.headless, "depth_units": "relative; not calibrated metres",
                   "input_fps": record.get("input_fps"), "fps_limit": fps_limit,
                   "model_input": [192, 640], "batch_size": 1,
                   "device": str(getattr(model, "device", "unknown")),
                   "process_memory": process_memory_mib(), "review_every": review_every,
                   "review_snapshots": len(reviews), **evidence,
                   "input_manifest_sha256": input_manifest_hash,
                   "pytorch_peak_allocated_mib": round(torch.cuda.max_memory_allocated(model.device) / 1024 ** 2, 2) if getattr(getattr(model, "device", None), "type", None) == "cuda" else None,
                   "video_playback": "sequential frames capped at source FPS and --fps; slower playback when processing is slower" if args.source == "video" else None,
                   "metrics_scaling": "GT median alignment; diagnostic only" if metrics else "unavailable: no ground-truth depth", "point_cloud": cloud,
                   **model_details(model)})
        print("Demo completed:", len(rows), "frames; output:", output)
        if evidence["runtime_acceptance"]["status"] == "needs_review":
            raise RuntimeError("Displayed runtime did not meet --min-seconds; evidence saved for inspection")
    except (Exception, KeyboardInterrupt) as error:
        # Keep failed runs inspectable without labelling partial results as completed evidence.
        try:
            write_json(output / "failure.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=str(error) or type(error).__name__, error_type=type(error).__name__,
                       completed_frames=len(rows), last_completed_frame=rows[-1]["frame_id"] if rows else None,
                       source=args.source, model_requested=getattr(args, "model", "monodepth2"),
                       device_requested=args.device, partial_outputs=True,
                       scope="Failure diagnostic only; partial files do not establish completed runtime or accuracy"))
        except OSError as report_error:
            print("Could not save failure diagnostic:", report_error, file=sys.stderr)
        raise
    finally:
        if records is not None:
            records.close()
        if pygame_started:
            pygame.quit()
        if capture is not None:
            capture.__exit__(*sys.exc_info())


def profile_model(args):
    import numpy as np
    import torch
    from PIL import Image
    from src.demo_data import verify_manifest
    from src.profiling import prediction_fingerprint, process_memory_mib, summarize_timings
    from src.visualize import render_frame

    if not 1 <= args.frames <= 300 or not 0 <= args.warmup <= 30 or not 2 <= args.repeats <= 5:
        raise ValueError("Profile requires 1-300 frames, 0-30 warmup frames and 2-5 repeats")
    manifest = verify_manifest(args.manifest, args.checkpoint)
    total = args.frames + args.warmup
    if manifest["decoded_frames"] < total:
        raise ValueError("Prepared video is too short for the requested profile")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = selected_model(args)
    if model.encoder.training or model.decoder.training:
        raise ValueError("Profiling requires encoder and decoder evaluation mode")
    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    video_args = argparse.Namespace(source="video", video=str(Path(args.manifest).resolve().parent / manifest["video"]), frames=total)
    rows, runs, reference = [], [], []
    repeatable, first_preview = True, None
    for repeat in range(args.repeats):
        records = video_frames(video_args)
        measured_start, fingerprints = None, []
        try:
            for index in range(total):
                cycle_start = time.perf_counter()
                try:
                    record = next(records)
                except StopIteration:
                    raise ValueError("Video ended before the requested profile completed")
                decoded = time.perf_counter()
                maps, _, model_ms, processing_ms = timed_prediction(model, record["rgb"])
                render_start = time.perf_counter()
                preview = render_frame(record["rgb"], maps["inverse_depth"], "Phase 3 profile | relative depth | repeat {} | frame {}".format(repeat + 1, record["frame_id"]))
                rendered = time.perf_counter()
                fingerprint = prediction_fingerprint(maps)
                validated = time.perf_counter()
                if index < args.warmup:
                    continue
                if measured_start is None:
                    measured_start = cycle_start
                fingerprints.append((record["frame_id"], fingerprint))
                if first_preview is None:
                    first_preview = preview
                rows.append(dict(repeat=repeat + 1, frame_id=record["frame_id"],
                                 decode_ms=(decoded - cycle_start) * 1000, model_ms=model_ms, processing_ms=processing_ms,
                                 render_ms=(rendered - render_start) * 1000, validation_ms=(validated - rendered) * 1000,
                                 pipeline_ms=(validated - cycle_start) * 1000))
            elapsed = time.perf_counter() - measured_start
        finally:
            records.close()
        if repeat == 0:
            reference = fingerprints
        else:
            repeatable = repeatable and fingerprints == reference
        run_rows = rows[-args.frames:]
        runs.append(dict(repeat=repeat + 1, measured_frames=args.frames, measured_seconds=elapsed,
                         uncapped_headless_fps=args.frames / elapsed, timings=summarize_timings(run_rows)))
    Image.fromarray(first_preview).save(output / "preview.png")
    with (output / "runtime.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "fingerprints.json", [{"frame_id": frame, "sha256": fingerprint} for frame, fingerprint in reference])
    summary = dict(schema_version=1, status="pass" if repeatable else "needs_review", model=model_details(model)["model_name"],
                   checkpoint_sha256=manifest["checkpoint_sha256"], input_manifest_sha256=file_hash(args.manifest),
                   source_revision=manifest["source_revision"], device=str(model.device),
                   gpu=torch.cuda.get_device_name(model.device) if cuda else None,
                   python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
                   model_input=[192, 640], batch_size=1, cpu_threads=2, repeats=args.repeats,
                   warmup_frames_per_repeat=args.warmup, measured_frames_per_repeat=args.frames,
                   exact_prediction_repeatability=repeatable, timings=summarize_timings(rows), runs=runs,
                   process_memory=process_memory_mib(),
                   pytorch_peak_allocated_mib=torch.cuda.max_memory_allocated(model.device) / 1024 ** 2 if cuda else None,
                   depth_units="relative; not calibrated metres", ground_truth_accuracy="unavailable",
                   scope="uncapped headless decode + inference + composite rendering + numerical validation; excludes display, pacing, setup, warmup and output file writes")
    summary.update(reference_checkpoint_sha256=manifest["checkpoint_sha256"], **model_details(model))
    write_json(output / "summary.json", summary)
    print("Profile", summary["status"], "| exact repeated predictions:", repeatable, "| output:", output)
    if not repeatable:
        raise RuntimeError("Repeated predictions differ; inspect the profile before adopting changes")


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    commands = command.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets", help="Download pinned source and selected model weights")
    assets.add_argument("--model", choices=("monodepth2", "depth-anything"), default="monodepth2")
    preparation = commands.add_parser("prepare-demo", help="Prepare a pinned real driving clip, license, fallback image and portable manifest")
    preparation.add_argument("--output", default="data/demo")
    preparation.add_argument("--checkpoint", default=str(BASELINE))
    verification = commands.add_parser("verify-demo", help="Verify local demo inputs and checkpoint identity before moving laptops")
    verification.add_argument("--manifest", default="data/demo/input_manifest.json")
    verification.add_argument("--checkpoint", default=str(BASELINE))
    preparation_eval = commands.add_parser("prepare-reference-eval", help="Prepare the pinned five-frame indoor RGB-D diagnostic input")
    preparation_eval.add_argument("--output", default="data/evaluation/redwood")
    preparation_eval.add_argument("--checkpoint", default=str(BASELINE))
    evaluation = commands.add_parser("evaluate", help="Score a frozen paired RGB/depth manifest; never infer accuracy from RGB-only footage")
    evaluation.add_argument("--manifest", default="data/evaluation/redwood/evaluation_manifest.json")
    evaluation.add_argument("--output", default="runs/phase4_reference_evaluation")
    server = commands.add_parser("install-server", help="Download/extract official Windows CARLA 0.9.13")
    server.add_argument("--archive", help="Use an already downloaded official Windows release ZIP")
    check = commands.add_parser("check", help="Save environment/CUDA diagnostics; CARLA is optional")
    check.add_argument("--output", default="runs/phase1_environment.json")
    check.add_argument("--carla", action="store_true", help="Also probe an optional CARLA server")
    inference = commands.add_parser("infer", help="Predict one RGB image and save numerical maps and preview")
    inference.add_argument("--image", required=True)
    inference.add_argument("--output", default="runs/baseline_image")
    inference.add_argument("--point-cloud", action="store_true", help="Also export a qualitative relative point-cloud snapshot (PLY, PNG and interactive HTML)")
    profile = commands.add_parser("profile", help="Repeatable uncapped headless timing and numerical-stability profile")
    profile.add_argument("--manifest", default="data/demo/input_manifest.json")
    profile.add_argument("--frames", type=int, default=30)
    profile.add_argument("--warmup", type=int, default=3)
    profile.add_argument("--repeats", type=int, default=2)
    profile.add_argument("--output", default="runs/phase3_profile")
    collect = commands.add_parser("collect", help="Record an exclusive, synchronized CARLA scene")
    collect.add_argument("--output", required=True)
    playback = commands.add_parser("demo", help="Low-memory webcam/video depth demo; optional CARLA/replay")
    playback.add_argument("--source", choices=("webcam", "video", "carla", "replay"), default="webcam")
    playback.add_argument("--camera", type=int, default=0, help="Local webcam index")
    playback.add_argument("--video", help="Local video file (required for --source video)")
    playback.add_argument("--manifest", help="Verify prepared driving clip/checkpoint and record its identity")
    playback.add_argument("--fps", type=float, default=10, help="Webcam/video processing ceiling (default 10 FPS)")
    playback.add_argument("--scene")
    playback.add_argument("--headless", action="store_true")
    playback.add_argument("--min-seconds", type=float, default=0, help="Require at least this many seconds of displayed processing; zero disables the gate")
    playback.add_argument("--review-every", type=int, default=0, help="Save first, periodic adjacent and last review frames; zero disables snapshots")
    playback.add_argument("--point-cloud", action="store_true", help="Export the first predicted frame as a qualitative relative point cloud")
    playback.add_argument("--output", default="runs/demo")
    audit = commands.add_parser("audit-scene", help="Check pilot frame IDs, numerical depth, poses and 10 Hz timing")
    audit.add_argument("--scene", required=True)
    audit.add_argument("--output", required=True)
    geometry = commands.add_parser("calibrate-depth", help="Check static-facade sensor depth at 5/10/20 metres")
    geometry.add_argument("--output", required=True)
    for child in (check, collect, playback, geometry):
        child.add_argument("--host", default="127.0.0.1")
        child.add_argument("--port", type=int, default=2000)
    for child in (collect, playback, geometry):
        child.add_argument("--timeout", type=float, default=15.0, help="CARLA RPC/sensor wait timeout in seconds")
    for child in (collect, playback):
        child.add_argument("--map", choices=("Town04", "Town05"), default="Town04")
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--frames", type=int, default=100)
    geometry.add_argument("--map", choices=("Town04", "Town05"), default="Town04")
    geometry.add_argument("--seed", type=int, default=42)
    for child in (inference, playback, profile, evaluation):
        child.add_argument("--model", choices=("monodepth2", "depth-anything"), default="monodepth2")
        child.add_argument("--input-size", type=int, choices=(294, 392, 518), default=294, help="Depth Anything input short edge; aspect preserved, multiple of 14")
        child.add_argument("--checkpoint", default=str(BASELINE))
        child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return command


def main():
    args = parser().parse_args()
    if getattr(args, "frames", 1) <= 0:
        raise ValueError("--frames must be positive")
    if args.command == "assets":
        if args.model == "depth-anything":
            from src.depth_anything import install_assets as install_small_assets
            install_small_assets()
        else:
            install_assets()
    elif args.command == "prepare-demo":
        prepare_demo(args)
    elif args.command == "prepare-reference-eval":
        prepare_reference_evaluation(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "verify-demo":
        from src.demo_data import verify_manifest
        manifest = verify_manifest(args.manifest, args.checkpoint)
        print("Demo input verified:", manifest["decoded_frames"], "frames;", manifest["source_duration_seconds"], "source seconds")
    elif args.command == "install-server":
        install_server(args.archive)
    elif args.command == "check":
        environment_report(args)
    elif args.command == "infer":
        infer_image(args)
    elif args.command == "profile":
        profile_model(args)
    elif args.command == "collect":
        from src.carla_capture import collect_scene
        print(json.dumps(collect_scene(args.output, capture_settings(args), args.frames), indent=2))
    elif args.command == "demo":
        demo(args)
    elif args.command == "audit-scene":
        audit_scene(args)
    elif args.command == "calibrate-depth":
        from src.carla_capture import check_depth_geometry
        print(json.dumps(check_depth_geometry(capture_settings(args), args.output), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
