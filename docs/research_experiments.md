# Steps 4-5: declared training experiments

25 October 2026. The protocol was frozen before execution in `runs/research_matrix_01/protocol.json`: Town04 training 392 moving triplets, Town05 development validation 152 targets from four condition/route captures, frozen depth encoder, shared colour augmentation, Adam 1e-6, batch one and seed 42. The main candidate is fixed at the final third epoch, not selected by searching test results. Three complete epochs give **1,176 updates on 392 distinct targets**, each visited once per epoch. All model/checkpoint states remain finite and hashed; the encoder and original baseline bytes remain unchanged.

| Development diagnostic | Pretrained | Fixed epoch-3 candidate |
|---|---:|---:|
| Broader CARLA AbsRel, 152 targets | 0.317736 | 0.291137 |
| KITTI cropped pilot AbsRel, 24 images | 0.093627 | 0.116876 |

**Promotion guard failed:** the KITTI pilot regressed. The default remains pretrained. This is a completed bounded longer-training experiment, not convergence or a generally superior model. Reference depth is scoring-only; median-aligned metrics do not provide calibrated predicted distances. The new reserved route was not consulted in training or development scoring.

[Complete training/protocol/per-route metadata](longer_training_results.json) records hashes, loss histories, metrics and counts. Epoch checkpoints include optimizer/RNG state; resuming restarts from a completed epoch in a fresh run. The matched unaugmented run, one-factor SSIM/automasking/smoothness ablations and seeds 43/44 were also predeclared; their scoring is recorded separately as Step 5 completes. All ablations use one epoch and compare with main epoch 1, not the longer three-epoch candidate.
