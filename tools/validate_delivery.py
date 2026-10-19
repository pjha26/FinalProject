"""Run all five phases together, offline, with fresh outputs and explicit failure evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    files = [ROOT / "project.py", ROOT / "requirements.txt"]
    for folder in ("src", "tests", "tools"):
        files.extend(sorted((ROOT / folder).glob("*.py")))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def run_delivery(output, device="auto", display=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   python=sys.version, executable=sys.executable, device_requested=device,
                   source_sha256=source_hashes(), steps=[], displayed_rehearsal="pending",
                   limitations=["RGB-only relative depth, not calibrated metres or autonomous control",
                                "Five indoor reference frames are diagnostic, not a driving benchmark",
                                "Physical webcam, native CARLA and other laptops require separate validation"])

    def save():
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "summary.json")

    def step(name, arguments):
        command = [sys.executable] + arguments
        row = dict(name=name, command=command, status="running", log=name + ".log")
        summary["steps"].append(row)
        save()
        print("Delivery:", name, flush=True)
        started = time.perf_counter()
        try:
            with (output / row["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=result.returncode, elapsed_seconds=time.perf_counter() - started,
                       status="pass" if result.returncode == 0 else "failed")
            if result.returncode:
                raise RuntimeError(name + " failed; inspect " + str(output / row["log"]))
        except BaseException:
            row["status"] = "failed"
            raise
        finally:
            save()

    def project_step(name, arguments):
        step(name, ["project.py"] + arguments + ["--output", str(output / name)])

    save()
    try:
        step("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
        step("inputs", ["project.py", "verify-demo"])
        step("environment", ["project.py", "check", "--output", str(output / "environment.json")])
        project_step("image", ["infer", "--image", "data/demo/driving_first_frame.png", "--device", device])
        project_step("profile", ["profile", "--frames", "10", "--warmup", "3", "--repeats", "2", "--device", device])
        project_step("evaluation", ["evaluate", "--device", device])
        video = ["demo", "--source", "video", "--video", "data/demo/driving.mp4",
                 "--manifest", "data/demo/input_manifest.json", "--device", device]
        project_step("video", video + ["--headless", "--frames", "10", "--fps", "30"])
        if display:
            project_step("display", video + ["--frames", "650", "--fps", "10", "--min-seconds", "60", "--review-every", "200"])
            evidence = json.loads((output / "display" / "summary.json").read_text(encoding="utf-8"))
            if evidence["runtime_acceptance"]["status"] != "pass":
                raise RuntimeError("Displayed rehearsal did not pass its duration gate")
            summary["displayed_rehearsal"] = "pass"
        if source_hashes() != summary["source_sha256"]:
            raise RuntimeError("Source changed during validation; repeat with a frozen checkout")
        summary["status"] = "pass"
    except BaseException as error:
        summary.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print("Delivery checks passed:", output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/delivery_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--display", action="store_true", help="Also require 60 seconds of displayed driving replay")
    args = parser.parse_args()
    run_delivery(args.output, args.device, args.display)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
