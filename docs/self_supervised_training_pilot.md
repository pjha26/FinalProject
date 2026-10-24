# First self-supervised CARLA training sanity experiment

Measured 25 October 2026. This implements the previously missing RGB temporal loader, pose path, view reconstruction and training losses, then exercises them with **10 optimizer steps**. It does not complete representative training, final model selection, full benchmark evaluation or the PDF's research targets. The pretrained baseline remains the default.

## Implementation and input boundaries

`src/training.py` validates the prepared triplet contract, image hashes, temporal IDs/timestamps, map separation and nonoverlapping RGB paths. It loads only PNG RGB frames and camera intrinsics. Numerical reference depth and recorded ego poses are never read by the trainer. The 640x480 frames use the existing RGB preprocessing to 640x192, including its half-pixel intrinsic resize geometry and checkpoint-internal normalization.

The official pinned Monodepth2 ResNet-18 depth encoder/decoder initialize the depth path. The separate two-image ResNet-18 pose encoder and pose decoder initialize from the pose weights in the same official `mono_640x192` archive. Pose source/weight SHA256 values are checked before training; the installed pose decoder and weights were checked byte-for-byte against the already downloaded pinned archives. Existing upstream licenses remain applicable. No dependency was added.

The previous/target pair is encoded in temporal order and its predicted transform inverted; the target/next transform is used directly. Every one of the four decoder disparities is upsampled to the full training resolution for calibrated backprojection/projective image sampling. `align_corners=True` matches the upstream Project3D normalization by width/height minus one. The objective averages four scale losses: minimum reprojection across adjacent sources, 0.85 SSIM plus 0.15 L1, identity automasking, and edge-aware mean-normalized disparity smoothness weighted by `1e-3 / 2**scale`. Training identity ties use the upstream-style 1e-5 noise; validation ties are deterministic.

## Bounded experiment

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_pilot_13/triplets.json --output runs/carla_training_sanity_next --steps 10 --device cuda
```

Use fresh output paths. The runner rejects existing outputs and paths inside the accepted baseline checkpoint. Failed experiments retain a failed summary. Final checkpoints are staged and include depth and pose states, file hashes, parent checkpoint hashes and the training-manifest hash. The existing inference loader can load the separate candidate using an explicit `--checkpoint`; the default model/checkpoint is unchanged. This runner has no resume support and does not save optimizer state.

Configuration: batch one, Adam learning rate 1e-5, seed 42, global gradient-norm clipping at 10, two PyTorch CPU threads, fixed pretrained BatchNorm running statistics with trainable affine weights, and no augmentation or learning-rate schedule. CARLA was stopped during training. Sampling is seeded; CUDA grid_sample backward may remain nondeterministic. These are explicit pilot choices, not an exact reproduction of every upstream training option.

Available data: 98 correlated Town04 training triplets and 25 correlated Town05 validation triplets. **Only ten unique training targets were optimized**, one per step; this was not a full epoch or a ten-epoch run. Town05 and KITTI reference data were used for validation/scoring only, never for the optimizer. These reused validation pilots are not untouched final test sets.

Local run: `runs/carla_training_sanity_01/`. All ten updates had finite, nonzero depth and pose decoder gradients. All four model state dictionaries changed; all baseline checkpoint bytes and BatchNorm running buffers stayed unchanged. The mean self-supervised Town05 validation objective increased from **0.0237301 to 0.0243978**. Training losses were 0.0494671 at step one and 0.0382796 at step ten, but those steps used different examples and do not establish convergence. The measured validation/train/export section took 6.26 seconds, excluding model/data setup. Peak PyTorch allocated CUDA memory was 906,529,792 bytes (about 864.5 MiB); this is not total process/system VRAM.

## Candidate versus preserved baseline

The existing evaluator scored both models on identical RGB/reference examples and masks in each domain. Reference depth remains native, masks use 0.001-80 m, there is no crop, predictions use the existing inverse-depth interpolation, scores use per-image GT-median alignment and aggregation is the arithmetic mean over images. The frozen candidate manifests change checkpoint identity while retaining asset hashes and the original reference-manifest fingerprint.

| Dataset / metric | Pretrained baseline | 10-step candidate |
|---|---:|---:|
| CARLA 25 targets: AbsRel | 0.181408 | 0.145156 |
| CARLA: aligned RMSE | 8.352287 m | 7.032227 m |
| CARLA: delta1 | 0.764108 | 0.833041 |
| KITTI cropped 24-image pilot: AbsRel | 0.093627 | 0.119681 |
| KITTI: aligned RMSE | 4.219610 m | 5.299157 m |
| KITTI: delta1 | 0.912854 | 0.895992 |

**Decision: keep the unchanged pretrained baseline as the default.** The CARLA diagnostic improved, while KITTI regressed and the self-supervised validation objective rose. This does not establish a broadly better model. CARLA covers one short Low-quality daylight route; KITTI covers 24 cropped-validation images from eight drives and is not official Eigen evaluation. Predictions remain relative, not deployment metre-scale estimates.

## Verification and reference outputs

Seven new component tests cover RGB-only loading, calibration/temporal order, hash mutation, split leakage/time gaps, identity and translated projective sampling, temporal transform direction/inversion, identity automasking and finite depth/pose gradient propagation. All **114 tests** pass. Frozen-demo verification plus fresh default-checkpoint CUDA image and ten-frame video smoke tests pass. Both candidate scoring jobs completed, and all 49 saved predictions reproduced per-image/aggregate metrics within rtol/atol 1e-6, with matching baseline IDs and valid-pixel counts.

[Tracked experiment metadata](training_pilot_results.json) contains the configuration, objective, checkpoint/source hashes, state audit, metrics and decision. Local evidence: `runs/training_component_validation_01/`, `runs/carla_training_sanity_01/`, `runs/carla_sanity_candidate_01/`, `runs/kitti_sanity_candidate_01/`, and each scoring dataset's `candidate_sanity_01_manifest.json`. Raw evidence and candidate weights remain ignored. The separate [reference-output folder](reference_outputs.md) contains actual labelled previews, numerical arrays, training trace, metrics and hashes; the trace is descriptive, not proof of convergence.

Next: expand route/scene diversity and reserve untouched testing routes; design a controlled short experiment to investigate the KITTI regression before longer fine-tuning. Evaluate one change at a time (for example, freezing the encoder or a smaller learning rate), without training on evaluation examples or replacing the baseline. Broader training, augmentation, loss ablations, full Eigen evaluation and integrated CARLA inference/FPS remain incomplete.
