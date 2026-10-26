# Saved project reference outputs

The local folder `../outputs/project_reference_2026-10-25_01/` (relative to the Git checkout) contains selected actual historical outputs. Open `index.html` for previews and `README.md` for scope. `manifest.json` records each copied file's original repository-relative source, bytes and SHA256; copied bytes were checked against their originals.

Included: Town04/Town05 RGB examples, a pretrained Monodepth2 Town04 prediction and numerical relative-depth map, both capture audits, the facade geometry report, the prepared RGB-triplet manifest, paired KITTI example previews and 24-image metric summaries, regression log, video smoke summary and dated guide snapshots. Later implementations can add separate dated bundles without replacing these references.

This folder is outside the Git checkout and must be backed up separately. Source, documentation and summarized results are committed; raw runs, datasets, images, runtimes and weights remain local. These selected files are not a full reproducibility backup. Retain original data/runs/checkpoints, asset licenses and the compatible environment as well.

Predicted depth is relative. KITTI pilot scores use per-image GT-median alignment and are not official Eigen results. The facade probe measures simulator sensor geometry, not model metric scale. Training has not yet been completed. Respect original dataset/asset terms before redistributing examples publicly.


A second local folder, `../outputs/carla_baseline_reference_2026-10-25_01/`, contains the frozen 25-target CARLA baseline metrics/CSV, evaluation manifest, source audit, reproduced-score check, 107-test log, and first/highest-AbsRel prediction previews plus selected numerical prediction/reference arrays. Its `index.html`, `README.md` and SHA256 `manifest.json` explain selection, protocol and limits. Highest-error previews are chosen after scoring for review only. Both reference folders remain separate from Git and from training inputs.


The third folder, `../outputs/self_supervised_training_reference_2026-10-25_01/`, records the first ten-step self-supervised sanity experiment. Open `index.html` for the recorded loss/metric plot, actual target/reconstruction images and matched baseline/candidate CARLA/KITTI previews. It also holds selected numerical arrays, configuration, training CSV, 114-test log, checkpoint/state audit, comparison metrics and a SHA256 manifest. Candidate weights stay in the ignored experiment directory. KITTI regressed, so this candidate is retained as experimental and the accepted default remains pretrained.


`../outputs/encoder_freeze_reference_2026-10-25_01/` records the next two steps: expanded training-route preparation/reserved-test integrity, and the matched encoder-freezing trial. The `index.html` gallery shows the added training RGB, identical-sample baseline/unfrozen/frozen previews, and recorded loss/metric comparisons. JSON/CSV files, selected arrays, the 116-test log, numerical-state audit and SHA256 manifest are included. No reserved-test model predictions or candidate weights are included. The pretrained baseline remains default because both candidates regress on the KITTI pilot.


`../outputs/training_controls_2026-10-25_01/` contains the predeclared lower-rate, expanded-data and duration-control reference cases as they complete. Open `index.html` for status and per-case galleries. Each completed case retains actual first-sample previews, selected prediction arrays, configuration, state/scoring audits, metric tables and source hashes; a cumulative manifest checks saved files. No reserved-test predictions or full checkpoints are included.


`../outputs/temporal_augmentation_reference_2026-10-25_01/` records Step 1 of the remaining plan. Open `index.html` for actual paired-control prediction previews, selected arrays, the numerical comparison plot and a labelled replay of a logged shared colour transform. Configuration, 119-test log, checkpoint/state/scoring audits and SHA256 manifest identify 40 selected files. The colour example is a CPU illustration of actual logged parameters, not a new metric result. Full weights/datasets remain in ignored repository directories; reserved-test predictions are excluded.


The completed steps 2-9 bundle is ../outputs/research_completion_2026-10-25_01/. Open index.html for saved previews, the report supplement, editable presentation and per-step evidence. manifest.json hashes selected files. Step 9 includes the 137-test delivery log, source-hash-verified integrated summary and fresh live launcher rehearsal. This supersedes historical statements above that training/full Eigen/reserved testing were pending. Large raw assets and full checkpoints still require separate backup.


../outputs/metric_control_extension_2026-10-26/ contains the separate rejected-scale calibration and passing simulator-reference control trials. Open index.html for actual rendered experiment videos and shadow estimates; the videos use simulation time. A SHA256 manifest covers selected files. Learned vehicle actuation is disabled, and these clips must not be presented as learned camera-only autonomous driving.


../outputs/accuracy_validation_2026-10-27/ contains the new metric-validation reports, per-frame scores, selected RGB/reference/prediction arrays, rejected-trial summaries, full-Eigen results and research checkpoint backup. It contains no presentation/PDF artifacts. The candidate is optional and not qualified for learned actuation.
