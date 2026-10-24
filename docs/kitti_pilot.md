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
