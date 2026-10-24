# Steps 3 and 5 infrastructure: epoch checkpoints, resume and explicit loss controls

25 October 2026. `tools/train_carla_epochs.py` reuses the accepted RGB-only model/objective, adds bounded full epochs and per-epoch validation, and atomically saves each completed epoch. Each checkpoint contains the four model states plus Adam state and Python/NumPy/PyTorch/CUDA, sampler and augmentation RNG states. Hashes and configuration are checked before resume. Resume starts the next epoch in a fresh output directory; partial mid-epoch work is discarded. Incompatible data/configuration, corrupt state, completed target epoch or existing output are rejected. Accepted baseline weights stay untouched.

```powershell
.\.venv\Scripts\python.exe tools/train_carla_epochs.py --manifest data/training/carla_conditions_01/triplets.json --output runs/epochs_next --epochs 1
.\.venv\Scripts\python.exe tools/train_carla_epochs.py --manifest data/training/carla_conditions_01/triplets.json --output runs/resumed_next --epochs 3 --resume runs/epochs_next/epoch_001
```

Tests use labelled synthetic CPU fixtures, not research results. Uninterrupted versus resumed execution reproduces weights, validation losses, optimizer tensors and CPU RNG state exactly. A deliberate mid-epoch exception retains the last valid checkpoint and resumes to the same final CPU weights. Config/checksum failures and no-overwrite behavior are covered. CUDA grid-sample backward may remain nondeterministic; GPU bitwise equality is not claimed.

Loss controls `--no-ssim`, `--no-automasking`, and `--no-smoothness` disable one objective component for predeclared ablations. Default objective behavior is preserved. Tests verify L1-only behavior, explicit automasking disable and zero smoothness, plus invalid-weight rejection. The epoch budget, loss settings, data hash, device, seed, frozen encoder and augmentation policy are all checkpoint configuration and must match on resume.

The validation preparer also accepts repeated `--validation-scene` arguments and combines frozen native pairs from audited validation captures. It rejects duplicate scenes/targets, preserves hashes and verifies the combined manifest. Single-scene behavior remains available. The broadened validation set is **152 targets from four condition/route captures**, separate from the reserved route.

All **126 regression tests** pass in `runs/research_completion_01/tests_final.log`. This completes implementation/CPU-resume acceptance; real longer training and loss/seed experiments are separate measured steps. No new dependency, default-model change or model-accuracy improvement is claimed by this implementation.
