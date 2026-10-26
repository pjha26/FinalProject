"""Own a local CARLA server and run the bounded reference-control/shadow suite."""
from pathlib import Path
import argparse
import json
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import write_json
from src.carla_capture import create_client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", default="runs/metric_scale_01/calibration.json")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--checkpoint", help="Optional research checkpoint; does not enable learned control")
    a = p.parse_args()
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [dict(name="clear_s42", seed=42, weather="ClearNoon", clear=True), dict(name="obstacle_s42", seed=42, weather="ClearNoon", clear=False), dict(name="obstacle_s44_wet", seed=44, weather="WetCloudyNoon", clear=False)]
    write_json(output / "protocol.json", dict(cases=cases, control="simulator reference depth; learned estimates recorded in shadow", calibration=a.calibration, accepted_original_demo_unchanged=True))
    try:
        with socket.create_connection(("127.0.0.1", 2000), timeout=1):
            raise RuntimeError("Existing server on 2000; do not disturb it")
    except (ConnectionRefusedError, socket.timeout):
        pass
    with (output / "server.log").open("w", encoding="utf-8") as log:
        owned = subprocess.Popen([str(ROOT / ".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"), "-RenderOffScreen", "-quality-level=Low", "-nosound", "-unattended", "-nosplash", "-stdout"], cwd=ROOT / ".runtime/carla", stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        results = []
        try:
            deadline = time.monotonic() + 120
            while True:
                if owned.poll() is not None:
                    raise RuntimeError("Server exited")
                try:
                    create_client("127.0.0.1", 2000, 3).get_server_version()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
            for case in cases:
                command = [sys.executable, "tools/carla_control_demo.py", "--calibration", a.calibration, "--output", str(output / case["name"]), "--seed", str(case["seed"]), "--weather", case["weather"]]
                if a.checkpoint:
                    command += ["--checkpoint", a.checkpoint]
                if case["clear"]:
                    command += ["--clear-road"]
                if a.headless:
                    command += ["--headless"]
                with (output / (case["name"] + ".log")).open("w", encoding="utf-8") as stream:
                    done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=360)
                summary_path = output / case["name"] / "summary.json"
                results.append(dict(name=case["name"], returncode=done.returncode, summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else dict(status="failed")))
                write_json(output / "summary.json", dict(status="pass" if len(results) == len(cases) and all(r["summary"]["status"] == "pass" for r in results) else "incomplete_or_failed", results=results))
                print(case["name"], results[-1]["summary"]["status"], flush=True)
        finally:
            if owned.poll() is None:
                owned.terminate()
                owned.wait(timeout=30)


if __name__ == "__main__":
    main()

