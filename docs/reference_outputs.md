# Saved project reference outputs

The local folder `../outputs/project_reference_2026-10-25_01/` (relative to the Git checkout) contains selected actual historical outputs. Open `index.html` for previews and `README.md` for scope. `manifest.json` records each copied file's original repository-relative source, bytes and SHA256; copied bytes were checked against their originals.

Included: Town04/Town05 RGB examples, a pretrained Monodepth2 Town04 prediction and numerical relative-depth map, both capture audits, the facade geometry report, the prepared RGB-triplet manifest, paired KITTI example previews and 24-image metric summaries, regression log, video smoke summary and dated guide snapshots. Later implementations can add separate dated bundles without replacing these references.

This folder is outside the Git checkout and must be backed up separately. Source, documentation and summarized results are committed; raw runs, datasets, images, runtimes and weights remain local. These selected files are not a full reproducibility backup. Retain original data/runs/checkpoints, asset licenses and the compatible environment as well.

Predicted depth is relative. KITTI pilot scores use per-image GT-median alignment and are not official Eigen results. The facade probe measures simulator sensor geometry, not model metric scale. Training has not yet been completed. Respect original dataset/asset terms before redistributing examples publicly.


A second local folder, `../outputs/carla_baseline_reference_2026-10-25_01/`, contains the frozen 25-target CARLA baseline metrics/CSV, evaluation manifest, source audit, reproduced-score check, 107-test log, and first/highest-AbsRel prediction previews plus selected numerical prediction/reference arrays. Its `index.html`, `README.md` and SHA256 `manifest.json` explain selection, protocol and limits. Highest-error previews are chosen after scoring for review only. Both reference folders remain separate from Git and from training inputs.
