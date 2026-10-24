# Step 2: controlled depth-encoder freezing experiment

Measured 25 October 2026. The first ten-step CARLA adaptation improved its small simulator validation result but regressed on KITTI. This experiment changes **one training policy: freeze the depth encoder**. It does not replace the pretrained default or claim converged training.

## Implementation and control

`TrainableModel` now accepts `freeze_encoder=False`. Opting in disables depth-encoder gradients; the depth decoder, pose encoder and pose decoder remain trainable. The runner exposes `--freeze-encoder`, excludes frozen parameters from Adam, and records the policy and gradient clipping in its summary. The default retains the original behavior. Dataset size in the limitation message is now derived from the manifest rather than hardcoded to 98.

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_pilot_13/triplets.json --output runs/carla_encoder_frozen_next --steps 10 --device cuda --freeze-encoder
```

The controlled run deliberately uses the original **98 training / 25 validation** manifest, not the newly expanded 196/25 dataset. It starts from the same official depth/pose weights, uses the exact same ten training target IDs and order, seed 42, batch one, Adam 1e-5, ten updates, four-scale objective, clip norm 10 and fixed BatchNorm running statistics. CUDA grid_sample backward may remain nondeterministic; this is a matched-protocol sanity control, not a repeated-seed statistical study. No reference depth, ego pose or reserved-test data enters training.

State-dictionary audit: **all 122 depth-encoder tensors remain equal to the baseline**; all 28 depth-decoder tensors, 60 pose-encoder tensors and eight pose-decoder tensors changed. Original baseline checkpoint bytes and BatchNorm running buffers stayed unchanged. Finite nonzero depth-decoder and pose-decoder gradients were recorded at each step. Frozen encoder serialization has a different file hash because checkpoint packaging/metadata differs; numerical state equality is the relevant freezing check.

## Validation results

All three models use the same frozen sample IDs, masks and scoring protocol within each dataset. Reference range is 0.001-80 m; no crop; existing inverse-depth interpolation; per-image GT-median scaling; mean over images. These are reused development-validation diagnostics, not untouched final test results.

| Dataset / metric | Pretrained baseline | Unfrozen 10-step | Frozen 10-step |
|---|---:|---:|---:|
| CARLA, 25 targets: AbsRel | 0.181408 | 0.145156 | 0.151677 |
| CARLA: aligned RMSE | 8.352287 m | 7.032227 m | 7.584519 m |
| CARLA: delta1 | 0.764108 | 0.833041 | 0.813031 |
| KITTI cropped pilot, 24 images: AbsRel | 0.093627 | 0.119681 | 0.109467 |
| KITTI: aligned RMSE | 4.219610 m | 5.299157 m | 4.857251 m |
| KITTI: delta1 | 0.912854 | 0.895992 | 0.906212 |

Mean self-supervised Town05 validation loss is 0.0237301 before either experiment; after training it is 0.0243978 unfrozen and 0.0241177 frozen. Both increased. Freezing reduces the KITTI regression relative to the unfrozen candidate but still underperforms the pretrained baseline there, while giving less CARLA improvement. **Keep the pretrained default.** Neither candidate establishes broad improvement, final PDF targets, official Eigen performance or calibrated predicted metres.

The frozen run's validation/train/export section took 5.64 seconds, excluding setup. Peak PyTorch allocated CUDA memory was 772,021,248 bytes (about 736.3 MiB), not total process/system VRAM. The measured scope is ten target updates, not a full epoch or a throughput benchmark.

## Verification and evidence

Two new tests verify encoder-only freezing, preserved trainability elsewhere and frozen BatchNorm behavior after switching to train mode; the default remains trainable. All **116 tests** pass. Fresh frozen-asset verification, default-checkpoint CUDA image inference and ten-frame headless video checks pass. Both candidate scoring jobs complete. All 49 saved predictions reproduce metrics within rtol/atol 1e-6, with matching valid-pixel counts and IDs across baseline/unfrozen/frozen runs.

The additional Town05 seed-45 route remains unscored and excluded from both training manifests and these evaluation manifests. Its source metadata/index hashes still match the [route registry](carla_route_registry.json). It shares the validation map, so it is not independent-map generalization evidence. The seed-44 native failure remains recorded separately.

[Tracked experiment metadata](encoder_freeze_results.json) records the complete configuration, source/checkpoint hashes, numerical-state audit and seven metrics per model/domain. Local evidence: `runs/carla_encoder_frozen_01/`, `runs/carla_encoder_frozen_eval_01/`, `runs/kitti_encoder_frozen_eval_01/` and `runs/encoder_freeze_validation_01/`. Candidate weights remain ignored. The [reference gallery](reference_outputs.md) contains matched actual previews, arrays, the controlled loss/metric plot, capture/registry evidence and verified source hashes.

Next, predeclare a lower-learning-rate control with the same data/order/steps before longer training. Separately assess expanded-data training under a fixed policy. Do not change learning rate, dataset and duration together and describe the result as an encoder ablation. Reserve the new route until a model/protocol is frozen for final evaluation. Augmentation, broader conditions/routes, longer training, full Eigen evaluation and integrated CARLA runtime acceptance remain incomplete.
