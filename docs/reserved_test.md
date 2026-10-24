# Step 7: frozen-model reserved CARLA test

25 October 2026. The final research candidate was fixed at the predeclared augmented seed-42 epoch 3. A protocol froze both model identities and every source-scene file hash before any reserved scoring. The evaluator rejects unregistered models or incomplete/changed source inventories. Tests cover successful synthetic plumbing, changed/unregistered sources/models and partial Eigen-manifest rejection.

The full 100-frame Town05 seed-45 route was scored without training or checkpoint reselection afterward. Both models use identical native RGB/reference arrays, masks and per-image GT median scaling. Source hashes were rechecked for this report.

| Model | AbsRel | Aligned RMSE |
|---|---:|---:|
| Pretrained baseline | 0.256899 | 9.400815 m |
| Fixed research candidate | 0.264372 | 8.750513 m |

Metrics trade off; this does not establish dominance. Keep the pretrained default under the prior development guard. These are 100 correlated frames from one short clear-weather route on the same Town05 map used for development, not independent-map or broad-condition testing. Relative predictions remain uncalibrated. This route is now consumed final-test evidence and must not be reused as untouched model-selection data.

[Results/protocol identity](reserved_test_results.json) and `runs/research_completion_01/frozen_test_protocol.json`, `reserved_baseline/`, `reserved_candidate/` preserve evidence. The original full scene/weights remain ignored. A new untouched route is required for any later iterative tuning/final-test claim.
