# Step 2: explicit capture conditions and broader routes

25 October 2026. Capture now accepts validated `--weather ClearNoon|WetCloudyNoon` on collect, simulator demo and depth-probe commands. Default ClearNoon behavior is preserved. Named settings are stored in scene metadata and prepared capture records. Invalid names fail before capture; weather forwarding/default/rejection tests pass. No model or inference preprocessing changed.

Five fresh-server native captures completed and passed the existing stored-scene audits. Combined with existing clear-weather routes, there are eight 100-frame source recordings: Town04 training seeds 42/44 and Town05 validation seeds 43/46 under both presets. All newly captured RGBs are native 640x480/nonuniform, and first/last frames differ. These are 10 Hz simulation sequences; this is not an application-FPS result. Previous failed seeds stay recorded.

The prepared `data/training/carla_conditions_01/triplets.json` contains **392 training and 152 validation triplets**. Stationary windows are rejected using the existing movement filter. [Registered metadata](carla_conditions_results.json) freezes per-capture condition, role, hashes, moving/rejected counts and manifest identity. Depth remains absent from training inputs. The reserved seed-45 test route remains unscored and excluded by scene and image hashes; its nearest newly sampled validation ego position is 246.01 m away. This does not establish independent-map testing.

```powershell
.\.venv\Scripts\python.exe project.py collect --map Town04 --seed 42 --weather WetCloudyNoon --timeout 60 --frames 100 --output data/carla/wet_training_next
```

Local evidence is `runs/carla_conditions_01/` and the source/prepared data folders. Low rendering and short daytime routes limit claims; a named preset does not establish rain/night robustness. Step 2's bounded coverage gate is complete, while representative data remains a research limitation. Next: verified epoch/resume support and predeclared longer/ablation experiments.
