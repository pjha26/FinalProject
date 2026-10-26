# Metric model validation: final replay candidate

27 October 2026. Code, numerical results and local evidence only; no PDF/PPTX files were added or changed.

## Measured improvement

The research candidate uses Monodepth2 with a frozen encoder and a depth decoder fine-tuned for 256 updates at Adam 2e-6. Training uses simulator metric GT for 40 Town04 road images and eight Town04 obstacle images, with frozen-teacher regularization on 16 verified Udacity driving frames. This is **supervised metric fine-tuning with teacher replay**, separate from the earlier temporal self-supervised work.

| Dataset / protocol | Pretrained with best fitted scale | Research network, same scale |
|---|---:|---:|
| Town05 development, 80 frames, fixed scale | AbsRel 0.24866763 | AbsRel 0.22695964 |
| New seed49/50 Town05 holdout, 200 frames, fixed scale | AbsRel 0.20083652 | AbsRel 0.17470693 |
| Holdout p95 frame AbsRel | 0.26968911 | 0.25213647 |
| KITTI24 development, GT-median scoring | AbsRel 0.09362687 | AbsRel 0.09803287 |
| Final KITTI Eigen697, GT-median scoring | AbsRel 0.11461662 | AbsRel 0.11818809 |

The network reduces fresh CARLA mean error by **13.0104% relative**, improving both routes. CARLA uses the declared 1-50 m crop and no per-frame GT alignment. An independent audit reproduces all 200 scores within 6.98e-8 over 27,869,421 valid pixels, with no exact RGB-file overlap with earlier CARLA fit/development/scale-test captures.

The candidate was frozen before new-route acquisition; that test is now consumed. Final Eigen results do not drive retraining/reselection. All routes share Town05 and are short, correlated, Low-rendering captures. This is not unseen-map or real-world metric certification.

## Default and control decision

The official pretrained model remains the presentation/general default because it is better on KITTI. The research checkpoint is explicitly selectable. Accepted checkpoint bytes and frozen encoder values remain unchanged.

Learned actuation remains disabled. Earlier matched-object tests show that whole-image gains do not guarantee reliable clearance. The new Audi/Town05 stationary test failed natively before scoring in four retained attempts, including CPU inference and altered startup warmup. Unseen-object metric validation is **not completed**.

The candidate's observational reference-control rerun passes clear-road and ClearNoon obstacle cases. Wet control stops without collision and leaves 6.08 m gap, but fails the unchanged 0.35 s freshness gate (0.36634 s maximum). Familiar Town04 integration scenes are not held-out learned-control tests. No acceptance thresholds are relaxed.

## Commands and evidence

Bounded training, with the original local data/assets available:

    .\.venv\Scripts\python.exe tools\train_metric_decoder.py --real-replay --steps 256 --output runs/new_metric_trial

Optional research observation:

    .\.venv\Scripts\python.exe tools\run_control_suite.py --checkpoint runs/metric_decoder_replay_256_01/checkpoint --calibration runs/metric_decoder_replay_256_01/calibration.json --output runs/research_metric_demo_unique

Reference depth controls the vehicle and model distances stay labelled UNVALIDATED for control. Native/freshness checks can fail as recorded. The accepted original tools/present.ps1 relative-depth demo remains available.

Selected evidence and the research checkpoint backup are local under ../outputs/accuracy_validation_2026-10-27/. Weights, raw data and runtime remain outside Git. No presentation artifacts are modified.

