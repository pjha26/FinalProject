# Step 1: synchronized temporal colour augmentation

Measured 25 October 2026. Implements the first gate of the [remaining-work plan](remaining_implementation_plan.md). Implementation and a matched one-epoch comparison are complete; accuracy improvement, convergence and the remaining research gates are not claimed.

## Behaviour and boundaries

`--color-augmentation` is opt-in on `tools/train_carla_pilot.py`. The default remains disabled. Training uses the installed torchvision tensor colour operations after the existing 640x192 RGB preprocessing. A dedicated CPU generator, seeded independently of sampling, applies a transform with probability 0.5. Brightness, contrast and saturation factors are uniform in 0.8-1.2; hue is -0.1 to 0.1. The sampled factors and random operation order are shared across previous/current/next frames; contrast uses the standard per-image mean with the shared factor. Each step logs its applied flag and parameters.

Only network inputs are augmented. The depth/pose networks see transformed RGB, while photometric reconstruction, identity comparisons and edge-aware smoothness use original RGB. Camera intrinsics and temporal order are unchanged; no flips, crops or geometric augmentation are added. Validation, scoring and saved validation reconstructions stay unaugmented. No reference depth/ego poses are loaded by training. No dependency was added.

`augment_triplet` checks finite float32 RGB in [0,1], returns a new tensor and does not modify targets. `training_objective` routes augmented network inputs and original loss targets explicitly. Three regression tests cover seeded/shared transforms and unchanged originals, skip/invalid boundaries, and preservation of loss-target/camera objects and disabled behaviour. All **119 tests** pass.

## Predeclared matched comparison

The same-source control and treatment each start independently from the same official depth/pose weights. Both use the expanded 196-training/25-validation manifest, 196 updates on all 196 unique targets, identical target order, seed 42, batch one, Adam 1e-6, frozen depth encoder, fixed BatchNorm statistics, four-scale losses and norm clipping at 10. The only policy difference is input colour augmentation. The first 196-step unaugmented run from earlier work remains a historical reference; the fresh same-source control avoids attributing code-version changes to augmentation. CUDA grid_sample backward can remain nondeterministic. These are single-seed pilots, not a statistical significance study.

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_expanded_01/triplets.json --output runs/augmentation_control_next --steps 196 --learning-rate 0.000001 --device cuda --freeze-encoder
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_expanded_01/triplets.json --output runs/augmentation_candidate_next --steps 196 --learning-rate 0.000001 --device cuda --freeze-encoder --color-augmentation
```

Augmentation was applied to **100 of 196 triplets**; all frames in an applied triplet share parameters. Both state audits confirm the original baseline bytes and all 122 depth-encoder tensors are preserved, with decoder/pose updates and finite nonzero gradients. Target-order and validation-row checks pass.

## Results

Protocol: the same 25-target CARLA and 24-image/eight-drive KITTI development pilots, identical source hashes/sample IDs/valid-pixel masks, native reference depth, 0.001-80 m range, no crop, existing inverse-depth interpolation, per-image GT-median alignment and mean over images. Predictions remain relative. KITTI cropped validation is not full Eigen; the reserved route remains unscored.

| Dataset / metric | Pretrained baseline | Unaugmented control | Augmented |
|---|---:|---:|---:|
| CARLA: AbsRel | 0.181408 | 0.147681 | 0.147788 |
| CARLA: aligned RMSE | 8.352287 m | 7.350936 m | 7.354224 m |
| CARLA: delta1 | 0.764108 | 0.820380 | 0.819606 |
| KITTI pilot: AbsRel | 0.093627 | 0.108975 | 0.108827 |
| KITTI pilot: aligned RMSE | 4.219610 m | 4.723708 m | 4.718729 m |
| KITTI pilot: delta1 | 0.912854 | 0.905984 | 0.906098 |

Mean self-supervised validation objective: 0.023730101 initially; 0.022280429 after the unaugmented epoch; 0.022274111 after the augmented epoch. Validation/train/export sections took 25.62 s and 27.38 s, excluding setup. Peak PyTorch allocated CUDA memory was 774337024 / 782303744 bytes; this is not total process/system VRAM.

**Decision: keep the pretrained default.** The paired differences are tiny and do not establish a useful benefit. Both candidates still regress on KITTI relative to pretrained. This does not reject augmentation in general; longer/repeated experiments on broader data are needed. Do not select a production model from these pilots or treat loss reduction as generalization evidence.

## Evidence and next gate

All 98 candidate predictions reproduce per-image/aggregate scores within rtol/atol 1e-6, with unchanged baseline/reference sample IDs and valid-pixel counts. Fresh frozen-demo verification, default CUDA image inference and ten-frame headless video smoke checks pass. Source/manifest/checkpoint hashes and the original predeclared protocol are in [tracked metadata](temporal_augmentation_results.json). Local evidence: `runs/augmentation_validation_01/`, `runs/carla_augmentation_control_01/`, `runs/carla_color_aug_epoch_01/` and each run's `_carla_eval` / `_kitti_eval` directories. Weights and raw datasets remain ignored.

The [separate reference folder](reference_outputs.md) contains actual saved predictions/arrays, audit logs, a metric plot and a labelled colour-transform example. The colour example replays actual logged parameters on source pixels with the same helper on CPU; it illustrates the transformation, and CPU/GPU rounding may differ. It is not an extra training/scoring result.

Next execute **Step 2: broader CARLA conditions/routes** with explicit capture settings and role/hash registration before training. Then add epoch/resume support and controlled longer training. The reserved `test_town05_16` metadata/index hashes remain unchanged, and the route stays excluded from training, validation and these scoring manifests. Final testing, full Eigen, ablations, >=10 FPS native integration and final research delivery remain pending.
