"""Bounded presentation entry point using prepared assets and an owned simulator."""
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def demo_args(mode, output):
    args = [sys.executable, "project.py", "demo", "--source", mode, "--device", "cuda", "--output", str(output)]
    if mode == "carla":
        return args + ["--map", "Town04", "--seed", "42", "--weather", "ClearNoon", "--timeout", "60", "--frames", "5000", "--stop-after-seconds", "65", "--min-seconds", "60", "--fps", "30", "--review-every", "100"]
    return args + ["--video", "data/demo/driving.mp4", "--manifest", "data/demo/input_manifest.json", "--frames", "650", "--fps", "10", "--min-seconds", "60", "--review-every", "200"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["carla", "video"], default="carla")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    server = ROOT / ".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"
    required = [ROOT / "checkpoints/baseline_mono_640x192" / name for name in ("encoder.pth", "depth.pth")]
    if args.mode == "carla":
        required.append(server)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError("Required local asset missing: " + str(path))
    subprocess.run([sys.executable, "project.py", "verify-demo"], cwd=ROOT, check=True)
    if args.check_only:
        print("Local presentation assets verified:", args.mode)
        return
    import uuid
    output = ROOT / "runs" / ("presentation_" + args.mode + "_" + uuid.uuid4().hex)
    if args.mode == "video":
        subprocess.run(demo_args(args.mode, output), cwd=ROOT, check=True)
    else:
        try:
            with socket.create_connection(("127.0.0.1", 2000), timeout=1):
                raise RuntimeError("Port 2000 occupied. Close the existing CARLA server first.")
        except (ConnectionRefusedError, socket.timeout):
            pass
        sys.path.insert(0, str(ROOT))
        from src.carla_capture import create_client
        with output.with_name(output.name + "_server.log").open("w", encoding="utf-8") as log:
            owned = subprocess.Popen([str(server), "-RenderOffScreen", "-quality-level=Low", "-nosound", "-unattended", "-nosplash", "-stdout", "-carla-rpc-port=2000"], cwd=ROOT / ".runtime/carla", stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 120
                while True:
                    if owned.poll() is not None:
                        raise RuntimeError("CARLA exited during startup. Inspect the server log; use --mode video.")
                    try:
                        create_client("127.0.0.1", 2000, 3).get_server_version()
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(2)
                subprocess.run(demo_args(args.mode, output), cwd=ROOT, check=True, timeout=360)
            finally:
                if owned.poll() is None:
                    owned.terminate()
                    owned.wait(timeout=30)
    print("Saved actual presentation outputs:", output)


if __name__ == "__main__":
    main()

