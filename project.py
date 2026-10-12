"""Phase 1 commands: assets, environment check, inference, capture, and replay."""
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
SERVER_URL = "https://carla-releases.b-cdn.net/Windows/CARLA_0.9.13.zip"


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
    server.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=server.parent) as temporary:
        staged = Path(temporary) / "server"
        print("Extracting CARLA 0.9.13...", flush=True)
        safe_extract(archive, staged, max_bytes=32 * 1024 ** 3)
        if not (staged / "CarlaUE4.exe").is_file():
            raise FileNotFoundError("Expected Windows server executable missing from official archive")
        write_json(staged / "download_provenance.json", {"version": "0.9.13", "source_url": SERVER_URL if not local_archive else "user-supplied official archive",
                   "archive_sha256": file_hash(archive)})
        staged.replace(server)
    print("CARLA installed:", server / "CarlaUE4.exe")


def environment_report(args):
    report = {"schema_version": 1, "python": sys.version, "platform": platform.platform(),
              "disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 2), "dependencies": {}}
    for name in ("torch", "torchvision", "numpy", "PIL", "cv2", "pygame", "carla"):
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
    except (ImportError, RuntimeError) as error:
        report["cuda_error"] = str(error)
    try:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(3)
        report["carla_client"] = client.get_client_version()
        report["carla_server"] = client.get_server_version()
        report["maps"] = client.get_available_maps()
    except (ImportError, RuntimeError) as error:
        report["carla_server_error"] = str(error)
    report["phase1_status"] = "diagnostics_only; tests, baseline inference and both town pilots are separate gates"
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def timed_prediction(model, rgb, camera=None):
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
    from src.model import load_model, sha256
    from src.preprocess import PREPROCESSING_ID
    from src.visualize import render_frame

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    model = load_model(args.checkpoint, args.device)
    timed_prediction(model, rgb)  # Warmup is excluded from the reported prediction.
    maps, intrinsics, model_ms, processing_ms = timed_prediction(model, rgb)
    for name, values in maps.items():
        np.save(output / (name + ".npy"), values, allow_pickle=False)
    Image.fromarray(render_frame(rgb, maps["inverse_depth"], "Relative depth | {} | model {:.1f} ms".format(model.device, model_ms))).save(output / "preview.png")
    write_json(output / "summary.json", {"schema_version": 1, "checkpoint": str(model.checkpoint),
               "encoder_sha256": sha256(model.checkpoint / "encoder.pth"), "preprocessing_id": PREPROCESSING_ID,
               "input": str(Path(args.image).resolve()), "intrinsics_resized": intrinsics.tolist(),
               "intrinsics_source": "assumed 90-degree FOV for diagnostic image; not used by RGB inference",
               "model_ms": model_ms, "processing_ms": processing_ms, "device": str(model.device),
               "depth_units": "relative; not calibrated metres"})
    print("Inference saved:", output / "preview.png")


def capture_settings(args):
    from src.carla_capture import CaptureSettings
    return CaptureSettings(host=args.host, port=args.port, map=args.map, seed=args.seed)


def demo(args):
    import numpy as np
    from PIL import Image
    from src.carla_capture import CarlaCapture, replay_scene
    from src.metrics import compute_metrics
    from src.model import load_model
    from src.preprocess import camera_intrinsics
    from src.visualize import render_frame

    if args.source == "replay" and not args.scene:
        raise ValueError("--scene is required for replay")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.checkpoint, args.device)
    display = None
    if not args.headless:
        import pygame
        pygame.init()
        display = pygame.display.set_mode((1280, 360))
        pygame.display.set_caption("Monodepth2 - RGB-only relative depth")
    capture = None
    try:
        if args.source == "carla":
            settings = capture_settings(args)
            capture = CarlaCapture(settings).__enter__()
            camera = {"width": 640, "height": 480, "intrinsics": camera_intrinsics(640, 480, 90).tolist()}
            records = (dict(capture.next_frame(), camera=camera) for _ in range(args.frames))
        else:
            records = replay_scene(args.scene)
        rows, metrics, start = [], [], time.perf_counter()
        cycle_start = start
        for index, record in enumerate(records):
            if index >= args.frames:
                break
            if display is not None and any(event.type == pygame.QUIT for event in pygame.event.get()):
                break
            frame_start = time.perf_counter()
            arrival_start = record["arrival_timestamp"] if args.source == "carla" else frame_start
            maps, _, model_ms, processing_ms = timed_prediction(model, record["rgb"], record["camera"])
            gt = record.get("depth")
            measured = None
            if gt is not None:
                inverse = np.asarray(Image.fromarray(maps["inverse_depth"]).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
                try:
                    measured = compute_metrics(gt, 1 / inverse)
                    metrics.append(dict(measured, frame_id=record["frame_id"], status="valid"))
                except ValueError as error:
                    metrics.append({"frame_id": record["frame_id"], "status": "invalid", "error": str(error)})
            elapsed = time.perf_counter() - start
            caption = "Relative depth | frame {} | {} | {:.1f} completed FPS".format(record["frame_id"], model.checkpoint.name, len(rows) / max(elapsed, 1e-9))
            if measured:
                caption += " | GT-aligned AbsRel {:.3f}, RMSE {:.2f} m".format(measured["abs_rel"], measured["rmse"])
            else:
                caption += " | GT metrics unavailable"
            preview = render_frame(record["rgb"], maps["inverse_depth"], caption)
            if index == 0:
                Image.fromarray(preview).save(output / "preview.png")
            if display is not None:
                surface = pygame.surfarray.make_surface(preview.transpose(1, 0, 2))
                display.blit(pygame.transform.smoothscale(surface, display.get_size()), (0, 0))
                pygame.display.flip()
            cycle_end = time.perf_counter()
            rows.append({"frame_id": record["frame_id"], "model_ms": model_ms, "processing_ms": processing_ms,
                         "capture_to_display_ms": (cycle_end - arrival_start) * 1000,
                         "full_cycle_ms": (cycle_end - cycle_start) * 1000})
            cycle_start = cycle_end
        elapsed = time.perf_counter() - start
        if not rows:
            raise ValueError("No frames processed")
        with (output / "runtime.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "metrics.json", metrics)
        write_json(output / "summary.json", {"schema_version": 1, "source": args.source, "checkpoint": str(model.checkpoint),
                   "frames": len(rows), "wall_seconds": elapsed, "processed_fps": len(rows) / elapsed,
                   "headless": args.headless, "depth_units": "relative", "metrics_scaling": "GT median alignment; diagnostic only"})
        print("Demo completed:", len(rows), "frames; output:", output)
    finally:
        if display is not None:
            pygame.quit()
        if capture is not None:
            capture.__exit__(*sys.exc_info())


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    commands = command.add_subparsers(dest="command", required=True)
    commands.add_parser("assets", help="Download pinned upstream source and trusted baseline weights")
    server = commands.add_parser("install-server", help="Download/extract official Windows CARLA 0.9.13")
    server.add_argument("--archive", help="Use an already downloaded official Windows release ZIP")
    check = commands.add_parser("check", help="Save environment, CUDA and CARLA connectivity diagnostics")
    check.add_argument("--output", default="runs/phase1_environment.json")
    inference = commands.add_parser("infer", help="Predict one RGB image and save numerical maps and preview")
    inference.add_argument("--image", required=True)
    inference.add_argument("--output", default="runs/baseline_image")
    collect = commands.add_parser("collect", help="Record an exclusive, synchronized CARLA scene")
    collect.add_argument("--output", required=True)
    playback = commands.add_parser("demo", help="Live CARLA or recorded-scene replay through shared inference")
    playback.add_argument("--source", choices=("carla", "replay"), required=True)
    playback.add_argument("--scene")
    playback.add_argument("--headless", action="store_true")
    playback.add_argument("--output", default="runs/demo")
    for child in (check, collect, playback):
        child.add_argument("--host", default="127.0.0.1")
        child.add_argument("--port", type=int, default=2000)
    for child in (collect, playback):
        child.add_argument("--map", choices=("Town04", "Town05"), default="Town04")
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--frames", type=int, default=100)
    for child in (inference, playback):
        child.add_argument("--checkpoint", default=str(BASELINE))
        child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return command


def main():
    args = parser().parse_args()
    if getattr(args, "frames", 1) <= 0:
        raise ValueError("--frames must be positive")
    if args.command == "assets":
        install_assets()
    elif args.command == "install-server":
        install_server(args.archive)
    elif args.command == "check":
        environment_report(args)
    elif args.command == "infer":
        infer_image(args)
    elif args.command == "collect":
        from src.carla_capture import collect_scene
        print(json.dumps(collect_scene(args.output, capture_settings(args), args.frames), indent=2))
    elif args.command == "demo":
        demo(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
