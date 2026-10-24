# CARLA data expansion and reserved test route

Measured 25 October 2026, using the unchanged capture/audit/preparation commands. Off-screen Low-quality ClearNoon, CARLA 0.9.13, 640x480 RGB/depth, 0.1-second simulation steps and 100 frames per capture. Each new capture used a fresh server; servers were stopped before training.

- Added `data/carla/train_town04_14` (Town04 seed 44): 100 frames, clean capture exit and passing audit; all 98 temporal windows pass the existing movement filter. This route is 671.52 m from the original route at the nearest sampled ego XY positions; this is a pose-sampling diagnostic, not proof of independent visual coverage.
- Prepared `data/training/carla_expanded_01` from both Town04 training scenes and the unchanged Town05 seed-43 validation scene: **196 training triplets and 25 validation triplets**. The original `carla_pilot_13` manifest remains unchanged for controlled comparisons.
- The proposed Town05 seed-44 reserved-test capture failed before frames, with native server exit 3. Its zero-frame scene is excluded; evidence remains in `runs/native_carla_pilot_15/`.
- A bounded Town05 seed-45 retry, `data/carla/test_town05_16`, passed capture/cleanup and audit with **100 frames**. Every RGB was checked for 640x480 dimensions/nonuniformity and first/last files differ. This route is reserved without model scoring. It is excluded from prepared training/validation RGBs; no copied image hashes overlap. No prediction or model-accuracy metric was computed for it in these steps.

The [route registry](carla_route_registry.json) freezes roles, seeds, source metadata/index hashes, expanded-manifest identity, pose separation diagnostics and the failed attempt. Town05 validation and reserved testing still share a map; neither this short route nor the expanded two-route training set establishes broad generalization. Native instability at some Town05 seeds remains unresolved.

Reproduce preparation with fresh outputs:

```powershell
.\.venv\Scripts\python.exe tools/prepare_carla_triplets.py --train-scene data/carla/train_town04_09 --train-scene data/carla/train_town04_14 --validation-scene data/carla/validation_town05_13 --output data/training/carla_expanded_next
```

Local evidence: `runs/native_carla_pilot_14/`, `runs/native_carla_pilot_15/`, `runs/native_carla_pilot_16/`, source scenes and the expanded manifest. These remain ignored and require separate backup. The next encoder-freezing experiment deliberately uses the original 98/25 manifest, seed, ten target IDs, learning rate and step count to isolate the training-policy change. Broader-data experiments should be reported separately.
