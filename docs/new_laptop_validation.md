# Presentation laptop validation - 24 October 2026

## Scope and source

The existing implementation was restored from clean `main` at `22e707d`. Both existing pretrained backends passed the repository's integrated delivery workflow on this laptop. Application source, pinned requirements, model defaults, preprocessing, and accepted baseline weights were unchanged during validation. This update records a new-machine acceptance milestone within the existing implementation plan; it does not complete the original report's research scope.

The supplied project report matches the PDF already in the repository byte for byte (SHA256 `62abe34c678d69d02e713f3f4c5b9f35e82f2860dc0e0d6ae225aa02e3eeb4bd`). Its targets and historical descriptions are not evidence of completed training or benchmark performance.

## Hardware and reconstructed environment

- CPU: Intel Core i7-13620H, 16 logical processors.
- System RAM: 15.73 GiB reported by Windows GlobalMemoryStatusEx (16 GB class).
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB class; NVIDIA-SMI reports 6141 MiB, PyTorch reports 6140 MiB.
- Driver: 617.42. Windows build: 10.0.26300 as reported by Python.
- Python: 3.8.20, 64-bit; torch 2.0.1+cu118; torchvision 0.15.2+cu118.
- NumPy 1.24.4, Pillow 9.5.0, OpenCV package 4.8.1.78, Pygame 2.5.2; all project requirements retained at their pinned versions. `pip check` passed.
- CUDA availability was verified with an actual tensor operation. The packaged PyTorch CUDA runtime is 11.8; the driver's reported CUDA capability is a separate value.

No Python 3.8 launcher was available. A fresh local virtual environment was built using the Python Build Standalone 3.8.20 Windows x64 shared runtime from [release 20241002](https://github.com/astral-sh/python-build-standalone/releases/tag/20241002). The archive SHA256 was verified against the release checksum: `41b6709fec9c56419b7de1940d1f87fa62045aff81734480672dcb807eedc47e`.

Windows sandbox temporary-folder permissions initially blocked ensurepip. A workspace-local temporary directory resolved that setup issue:

```powershell
New-Item -ItemType Directory -Force .runtime/temp | Out-Null
$env:TEMP = (Resolve-Path .runtime/temp).Path
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $PWD '.cache/matplotlib'
```

The 2.6 GB PyTorch wheel stalled through the initial download route. The same official pinned wheel was retrieved in bounded HTTP ranges and checked against the official index SHA256 `e58d26a11bd57ac19761c018c3151c15bc71d068afc8ec409bfd9b4cfcc63a52`. No replacement dependency or model was introduced. Runtimes, wheels, checkpoints, media and raw runs remain ignored by Git.

## Acceptance results

Both backends independently passed all eight delivery steps: 87 regression tests, input verification, environment diagnostics, image inference, repeated profiling, five-pair reference evaluation, short headless video, and displayed replay. Both validators confirmed their source hashes stayed unchanged. The runs were sequential.

Each displayed rehearsal used the same verified driving video and frame range, a 10 FPS ceiling, batch one, CUDA, two PyTorch CPU threads, review snapshots every 200 frames, and no point-cloud export. Both passed the minimum 60-second display gate.

| Backend | Frames | Seconds | Completed FPS | Median displayed-loop model ms | Process peak working set MiB | PyTorch peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| Monodepth2 ResNet-18 | 650 | 67.00 | 9.70 | 15.97 | 842.66 | 109.77 |
| Depth Anything V2 Small-294 | 650 | 67.16 | 9.68 | 35.20 | 838.54 | 299.75 |

The ceiling limits the workload; these results do not establish maximum speed or a guaranteed 10 FPS minimum. Timing excludes model loading, input verification and final report writes. Process working set and PyTorch allocation are separate measurements, not total system RAM or total process VRAM. These observations are not a controlled speedup over the previous laptop.

The short uncapped headless profiles used two repeats of ten measured frames, with three warmup frames per repeat:

| Backend | Headless FPS by repeat | Median model ms | Prediction repeatability |
|---|---|---:|---|
| Monodepth2 ResNet-18 | 48.99 / 49.91 | 6.19 | exact match |
| Depth Anything V2 Small-294 | 24.61 / 25.78 | 18.22 | exact match |

These short profiles exclude display, startup, warmup, pacing and output writes; they do not establish sustained displayed throughput.

The existing five correlated indoor reference frames from one Redwood scene produced the following per-image GT-median-aligned diagnostic scores:

| Backend | AbsRel |
|---|---:|
| Monodepth2 ResNet-18 | 0.496026 |
| Depth Anything V2 Small-294 | 0.203328 |

This is not a KITTI or driving benchmark. The driving clip has no aligned reference depth. GT scaling is scoring-only and does not recover deployment metre scale.

## Reproduce and inspect

From the repository root, using fresh output paths:

```powershell
.\.venv\Scripts\python.exe project.py check --output runs/hardware_next.json
.\.venv\Scripts\python.exe project.py verify-demo
.\.venv\Scripts\python.exe tools/validate_delivery.py --device cuda --model monodepth2 --display --output runs/baseline_delivery_next
.\.venv\Scripts\python.exe tools/validate_delivery.py --device cuda --model depth-anything --input-size 294 --display --output runs/small_delivery_next
.\.venv\Scripts\python.exe project.py infer --image data/demo/driving_first_frame.png --device cuda --output runs/image_fallback_next
```

The validator itself performs no downloads. Prepared local assets supported all inference and evaluation steps; the network was not physically disconnected during these rehearsals.

Local evidence is in `runs/new_laptop_hardware_01.json`, `runs/new_laptop_image_01/`, `runs/new_laptop_video_01/`, `runs/new_laptop_baseline_delivery_01/`, and `runs/new_laptop_small_delivery_01/`. Each delivery directory retains step logs, source hashes, previews, numerical maps, runtime CSVs, evaluation and profile summaries, and nine displayed-run review snapshots. These ignored directories must be retained or transferred separately; cloning Git does not reproduce historical run evidence.

## Next work in the existing plan

Monodepth2 remains the default and a validated fallback; the optional Small backend is also accepted for this recorded clip on this laptop. This evidence does not select a universal driving-accuracy winner.

Next, compare both on the same varied, permitted driving clips, then prepare documented held-out driving RGB/aligned-depth examples before designing fine-tuning. Keep scene splits and accepted baseline weights separate from experiments. Native CARLA needs a small successful RGB/depth pilot before larger collection.

Physical webcam validation, a physically disconnected rehearsal, new-machine point-cloud rehearsal, broader driving quality review, held-out driving accuracy, KITTI protocol validation, native CARLA capture, temporal training preparation, fine-tuning and original-scope ablations remain pending. No calibrated distances, vehicle control or newly trained weights are claimed.
