# Learning-rate, dataset and duration controls

Measured 25 October 2026. These runs reuse the existing training runner with no new dependency or application-source change. The protocol below was saved before execution; each candidate starts independently from the same official depth/pose weights. The pretrained baseline remains the accepted default.

## Frozen protocol

- Depth encoder frozen; BatchNorm running statistics fixed; train depth decoder and pose networks. Adam 1e-6, seed 42, batch one, gradient norm clipped at 10; same four-scale self-supervised objective, no augmentation or LR schedule.
- Control 1 changes only learning rate from the prior frozen-encoder 1e-5 run, on the original 98/25 manifest and identical ten target IDs/order.
- Control 2 changes only training dataset to 196 triplets while retaining ten updates, rate and policy; shuffled target membership/order naturally changes with dataset size. Validation rows remain identical.
- Control 3 changes only duration to 196 updates on the expanded dataset; one complete shuffled pass with no repeated target. It proceeds only after short-run gradient/checkpoint gates pass.
- Development scoring uses the same 25 Town05 and 24 KITTI cropped-validation samples/masks. Reference depth is scoring-only. Test route `test_town05_16` remains excluded and unscored.
- CUDA grid_sample backward may remain nondeterministic despite seeds; these single-seed pilots do not provide statistical significance.

## Results

| Run | Training targets seen | CARLA AbsRel | CARLA aligned RMSE | KITTI AbsRel | KITTI aligned RMSE |
|---|---:|---:|---:|---:|---:|
| Pretrained baseline | 0 project updates | 0.181408 | 8.352287 m | 0.093627 | 4.219610 m |
| Earlier frozen encoder, 1e-5, original data | 10 | 0.151677 | 7.584519 m | 0.109467 | 4.857251 m |
| Lower learning rate; original data; 10 updates | 10 | 0.178996 | 8.283810 m | 0.094239 | 4.231347 m |
| Expanded data; same rate; 10 updates | 10 | 0.179563 | 8.296955 m | 0.094061 | 4.224401 m |
| Expanded data; same rate; one 196-update epoch | 196 | 0.147682 | 7.350631 m | 0.108976 | 4.723875 m |

Protocol: native reference depth, 0.001-80 m mask, no crop, existing inverse-depth interpolation, per-image GT-median alignment and arithmetic mean over images. Scores do not recover predicted metric scale. CARLA covers one short Low-quality daylight validation route; KITTI is a 24-image/eight-drive diagnostic, not official Eigen or untouched final testing.

## Lower learning rate; original data; 10 updates

Learning rate 1e-5 -> 1e-6 relative to frozen-encoder control.

Completed 10 updates on 10 unique training targets from 98 available triplets. Mean self-supervised validation objective: 0.023730101 before, 0.023442584 after. Validation/train/export section: 5.69 s excluding setup; peak PyTorch allocated CUDA memory 772021248 bytes, not total process/system VRAM.

Training target counts by scene: {'train_town04_09': 10}.

State and provenance audits confirm unchanged depth-encoder tensors and original baseline bytes, updated depth/pose paths, finite gradients/checkpoints, unchanged validation rows and matching manifest hashes. All 49 candidate predictions reproduce metrics within rtol/atol 1e-6 with the same sample IDs and valid-pixel counts as reference/baseline. Raw evidence: `runs/carla_frozen_lr1e6_01/`, `runs/carla_frozen_lr1e6_01_carla_eval/` and `runs/carla_frozen_lr1e6_01_kitti_eval/` (the low-rate KITTI description-corrected rerun has suffix `_corrected`; its metrics are identical).

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_pilot_13/triplets.json --output runs/carla_frozen_lr1e6_01_next --steps 10 --learning-rate 0.000001 --device cuda --freeze-encoder
```

## Expanded data; same rate; 10 updates

Training dataset 98 -> 196 relative to low-LR original-data control.

Completed 10 updates on 10 unique training targets from 196 available triplets. Mean self-supervised validation objective: 0.023730101 before, 0.023584328 after. Validation/train/export section: 5.44 s excluding setup; peak PyTorch allocated CUDA memory 772021248 bytes, not total process/system VRAM.

Training target counts by scene: {'train_town04_09': 5, 'train_town04_14': 5}.

State and provenance audits confirm unchanged depth-encoder tensors and original baseline bytes, updated depth/pose paths, finite gradients/checkpoints, unchanged validation rows and matching manifest hashes. All 49 candidate predictions reproduce metrics within rtol/atol 1e-6 with the same sample IDs and valid-pixel counts as reference/baseline. Raw evidence: `runs/carla_expanded_lr1e6_01/`, `runs/carla_expanded_lr1e6_01_carla_eval/` and `runs/carla_expanded_lr1e6_01_kitti_eval/` (the low-rate KITTI description-corrected rerun has suffix `_corrected`; its metrics are identical).

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_expanded_01/triplets.json --output runs/carla_expanded_lr1e6_01_next --steps 10 --learning-rate 0.000001 --device cuda --freeze-encoder
```

## Expanded data; same rate; one 196-update epoch

Duration 10 -> 196 updates, one complete shuffled pass through expanded data.

Completed 196 updates on 196 unique training targets from 196 available triplets. Mean self-supervised validation objective: 0.023730101 before, 0.022279437 after. Validation/train/export section: 26.23 s excluding setup; peak PyTorch allocated CUDA memory 774337024 bytes, not total process/system VRAM.

Training target counts by scene: {'train_town04_09': 98, 'train_town04_14': 98}.

State and provenance audits confirm unchanged depth-encoder tensors and original baseline bytes, updated depth/pose paths, finite gradients/checkpoints, unchanged validation rows and matching manifest hashes. All 49 candidate predictions reproduce metrics within rtol/atol 1e-6 with the same sample IDs and valid-pixel counts as reference/baseline. Raw evidence: `runs/carla_expanded_epoch_01/`, `runs/carla_expanded_epoch_01_carla_eval/` and `runs/carla_expanded_epoch_01_kitti_eval/` (the low-rate KITTI description-corrected rerun has suffix `_corrected`; its metrics are identical).

```powershell
.\.venv\Scripts\python.exe tools/train_carla_pilot.py --manifest data/training/carla_expanded_01/triplets.json --output runs/carla_expanded_epoch_01_next --steps 196 --learning-rate 0.000001 --device cuda --freeze-encoder
```

## Decision and remaining work

Keep the pretrained default. The completed one-epoch pilot (when listed above) is a single pass, not convergence. It improves CARLA validation but regresses on KITTI; a lower self-supervised objective does not establish better cross-domain accuracy. Next expand condition/scene coverage and implement synchronized temporal augmentation before committing to longer training. These controls measure sensitivity and early adaptation, not a converged or generally better depth model. Do not select a production checkpoint from isolated validation improvements. Reserve the new route until model/protocol choices are frozen. Broader training conditions, augmentation controls, repeated seeds, longer training, full Eigen evaluation and integrated CARLA FPS remain incomplete.

All 116 regression tests pass. Existing asset verification and baseline image/video smoke checks are recorded under `runs/lr_data_controls_01/`. [Complete numerical metadata](lr_data_training_controls_results.json) records hashes and metrics; [reference outputs](reference_outputs.md) describe the local gallery and backup boundary. Weights/raw datasets remain ignored and are not uploaded to Git.
