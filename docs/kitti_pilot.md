# KITTI selected-validation pilot

The importer reads a local `data_depth_selection.zip` directly. It copies only selected validation RGB, corresponding depth and camera-intrinsics files into a fresh ignored data directory; it never unpacks the anonymous test folders or modifies the source ZIP.

```powershell
.\.venv\Scripts\python.exe tools/prepare_kitti_subset.py --archive 'C:\path\to\data_depth_selection.zip' --output data/evaluation/kitti_pilot_01
```

Requirements: the existing pinned baseline assets, including the upstream `splits/eigen_zhou/train_files.txt` and `val_files.txt`. Their SHA256 hashes are checked against the pinned Monodepth2 revision, so a changed split cannot silently alter eligibility. Do not reuse an existing output directory.

Selection is fixed before scoring: exclude every drive appearing in either pinned training or development list; use left camera `image_02`; take the first, middle and last available validation frame from each remaining drive. The supplied archive contains 13 validation drives. Five overlap the upstream training list, leaving **24 images from eight drives** in this pilot. Three frames from one drive remain correlated. Depth Anything's pretraining overlap is unknown; this is not a claim that it has never seen these scenes.

RGB and depth must have the selected-validation 1216x352 geometry. Pairing uses exact drive/frame/camera filenames. The importer requires 16-bit grayscale depth PNGs, converts integer values to float32 metres by dividing by 256, and preserves zero as invalid. It checks finite, positive-focal camera intrinsics and validates the prepared data with the existing paired evaluator. These checks rely on KITTI's supplied registration; equal dimensions alone do not independently establish physical alignment.

The [upstream depth loader](https://github.com/nianticlabs/monodepth2/blob/b017a8164fae339b0ce01f3a9e5048104d5abd0e/datasets/kitti_dataset.py) documents the /256 conversion. Source data and access terms are provided by [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

The output `evaluation_manifest.json` records the entire source ZIP's SHA256, each selected archive member, RGB/depth/calibration hashes, source-depth PNG hashes, selection, excluded/selected drive identities, and exact split-file provenance. The source ZIP is hashed again before publishing the prepared directory. Selected ZIP members are CRC-checked when read. Duplicate/unsafe ZIP paths, missing/oversized pairs, invalid image encoding/calibration and empty scoring masks are rejected. Preparation is staged atomically; a failed import does not leave an accepted dataset directory.

## Evaluation protocol

Use the existing RGB-only evaluator for each model, sequentially, with fresh output directories:

```powershell
.\.venv\Scripts\python.exe project.py evaluate --manifest data/evaluation/kitti_pilot_01/evaluation_manifest.json --model monodepth2 --device cuda --output runs/kitti_baseline_01
.\.venv\Scripts\python.exe project.py evaluate --manifest data/evaluation/kitti_pilot_01/evaluation_manifest.json --model depth-anything --input-size 294 --device cuda --output runs/kitti_small_01
```

Ground truth is used only for scoring. The selected images are already cropped: use no additional Garg crop. The frozen existing protocol retains finite GT pixels between 0.001 and 80 metres, resizes predicted inverse depth to the GT grid before reciprocal conversion, applies per-image GT median alignment and averages metrics equally over images. All drives contribute three samples.

This is a small **local cropped-validation diagnostic**, not an official KITTI leaderboard or full Eigen benchmark. Per-image median alignment does not recover metre-scale deployment predictions. The pretrained Monodepth2 checkpoint remains unchanged. No training examples or fine-tuned weights are created by this importer.

Raw KITTI files, prepared inputs, numerical outputs and run logs remain ignored by Git. Retain them locally with the source manifest and dataset terms; only source/tests/documentation are committed.


## Measured pilot results - 24 October 2026

The importer was published as `4e0d870` before evaluating the models. Both sequential CUDA evaluations completed all 24 images from eight drives. The 92-test suite passed, including five importer tests covering conversion, masking, deterministic selection, input integrity and failure cases. Raw predictions from all 48 model/sample combinations independently reproduced their recorded scores within float32 tolerance. Paired sample IDs and valid-pixel counts matched exactly; retained calibration hashes also matched.

| Backend | AbsRel | SqRel | Aligned RMSE (m) | Log RMSE | Delta1 | Delta2 | Delta3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monodepth2 ResNet-18 mono_640x192 | 0.093627 | 0.689646 | 4.219610 | 0.149034 | 0.912854 | 0.975248 | 0.990976 |
| Depth Anything V2 Small | 0.094203 | 0.722033 | 4.258988 | 0.138601 | 0.927902 | 0.979386 | 0.992438 |

AbsRel, SqRel, RMSE and log RMSE are lower-is-better; delta accuracies are higher-is-better. RMSE is measured after per-image GT median alignment. These figures do not establish calibrated depth during deployment, and must not be presented as achievement of the original report's official KITTI/Eigen targets.

Monodepth2 has slightly lower aggregate AbsRel/SqRel/RMSE here; Small has better log RMSE and delta accuracies. The AbsRel difference is about 0.00058. Per-drive performance is mixed, so this pilot does not establish a universal winner. Monodepth2 remains the default; no candidate weights or training hyperparameters were selected using these validation scores.

| Drive | Monodepth2 AbsRel | Small-294 AbsRel |
|---|---:|---:|
| 2011_09_26_drive_0002_sync | 0.064781 | 0.069919 |
| 2011_09_26_drive_0013_sync | 0.107718 | 0.088954 |
| 2011_09_26_drive_0020_sync | 0.060432 | 0.125545 |
| 2011_09_26_drive_0023_sync | 0.071283 | 0.130160 |
| 2011_09_26_drive_0036_sync | 0.111334 | 0.070947 |
| 2011_09_28_drive_0037_sync | 0.134836 | 0.116265 |
| 2011_09_30_drive_0016_sync | 0.079555 | 0.064449 |
| 2011_10_03_drive_0047_sync | 0.119076 | 0.087389 |
The small, deterministic sample has three correlated frames per drive, all from selected cropped validation. It is not the full Eigen split, an official benchmark submission, an independent test of every model's pretraining, or a fine-tuning result. Depth Anything overlap remains unknown. Any later tuning requires a separate untouched test set.

### Evidence and provenance

- Supplied ZIP: `data_depth_selection.zip`; locally computed SHA256 `11028c5f9ea21afdba20d00baeae023a9dbe80183765938816e47bfce1dd0f70`. This fingerprints the supplied file; it is not a checksum independently published by KITTI.
- Prepared manifest SHA256: `a169c0571f4fd4023d3840497fba0aa7bbd4e3d322f98292da65f50e6ce5b492`.
- Evaluated valid reference pixels across the 24 images: **1782495** for each backend; equal-image aggregation, not pooled-pixel scoring.
- Prepared subset: approximately 55.53 MiB; the 2 GB ZIP was not fully extracted. Only selected RGB/depth/intrinsics were retained.
- Baseline input: 192x640. Small network input: 294x1022, with output returned to the supplied RGB geometry. Neither setting was changed during evaluation.
- Local inputs: `data/evaluation/kitti_pilot_01/`. The original source ZIP remains unchanged at its user-supplied location.
- Local results: `runs/kitti_baseline_01/`, `runs/kitti_small_01/`, and `runs/kitti_import_validation_01/` (tests, run logs, source inventory and recomputed-score audit). These remain ignored.

The first sample and each backend's highest-AbsRel sample were inspected visually after scoring, without changing subset membership. They include street, bicycle/pedestrian and shaded parked-car scenes. Appearance and raw preview colours alone do not establish depth correctness. Preview maps remain uncalibrated relative predictions; median alignment is applied only in scoring.

### Remaining work

Preserve this baseline. Broader driving evaluation and an untouched test split remain necessary before claiming generalization or selecting a fine-tuned candidate. The original CARLA temporal RGB/calibration collection, pose/loss training components, project-specific fine-tuning, full Eigen evaluation and ablations remain unfinished. The KITTI pilot is evaluation data, not training input.
