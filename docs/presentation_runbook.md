# Presentation runbook

Updated 26 October 2026. Use this laptop, plugged into power, with other GPU applications closed. Run PowerShell from the repository root. Prepared assets are already local; do not download or train during the presentation.

## Live simulator demonstration

    .\tools\present.ps1 -Mode carla -CheckOnly
    .\tools\present.ps1 -Mode carla

The launcher verifies the prepared demo assets, starts its own CARLA 0.9.13 server in Low quality, waits for readiness, and displays 65 seconds of live moving Town04 RGB with Monodepth2 depth and overlay. It uses ClearNoon, seed 42 and the accepted official pretrained checkpoint. Each run has a unique folder under runs/presentation_carla_*. The launcher stops its own server on exit. Close any previously running CARLA instance first. If PowerShell blocks script execution, use the individual commands in native_runtime.md or a process-scoped execution-policy setting permitted by your institution.

Show the RGB, relative depth colours, overlay and completed FPS. Purple means farther and yellow means nearer. The simulator's reference depth is for diagnostic scoring; the network receives only RGB. Pressing Escape closes the view, but early closure fails the 60-second acceptance gate. Allow the timed rehearsal to complete.

The measured historical native run processed 783 frames in 65.0114 seconds at 12.0440 application FPS on this RTX 4050 6 GB laptop. Startup is excluded. Performance is configuration dependent.

## Recorded-input fallback

    .\tools\present.ps1 -Mode video

This computes predictions while replaying verified driving footage. Describe it as recorded-input inference, separately from the live simulator. If the simulator fails, preserve its failure log and use this fallback.

For a still-image fallback and optional point cloud:

    .\.venv\Scripts\python.exe project.py image --image data/demo/driving_first_frame.png --device cuda --point-cloud --output runs/presentation_image_unique

Choose an unused output name. Open its preview and point_cloud/viewer.html. The first-frame point cloud uses relative depth and assumed camera intrinsics.

## Suggested 5-minute demonstration

1. Explain RGB-only depth inference and show the Monodepth2 architecture slide.
2. Run the live simulator for the complete 65 seconds. Explain near/far colours and live FPS.
3. Show the saved full KITTI evaluation and training comparison. The pretrained model remains default because fine-tuning regressed on KITTI.
4. Show one numerical map or the optional point cloud. Explain scale ambiguity.
5. State limits and future work: more diverse training, better cross-domain accuracy and physical hardware rehearsals.

## Evidence and answers

- Original PDF alignment: CARLA 0.9.13, RGB capture, ResNet-18 depth decoder, pose/temporal self-supervision, loss ablations, KITTI/CARLA evaluation and a measured >=10 FPS native configuration are implemented.
- Training: three bounded epochs, 392 triplets per epoch, frozen encoder, Adam 1e-6. This is a fine-tuning experiment, not training from scratch or established convergence.
- KITTI Eigen: all 697 prescribed frames, Garg crop, 80 m maximum, per-image ground-truth median alignment, no flip postprocessing. Baseline AbsRel 0.1146166, RMSE 4.8634129, delta1 0.8774286. AbsRel/delta1 targets pass; RMSE, SqRel and log-RMSE targets miss.
- Ground-truth median alignment is evaluation-only. Predicted distances in the live demo are not calibrated metres.
- The held-out route is now consumed by final testing and shares the Town05 development map.
- The integrated validator passed all eight checks and 133 tests. Physical webcam, another laptop and a physically disconnected-network run remain unverified.

## Before leaving

Keep checkpoints, third_party model sources/licenses, data/demo, the prepared environment, CARLA installation and selected evidence together on this laptop. GitHub stores code and summarized results, not the large runtime/data/weights. On another laptop follow demo_guide.md and rerun tools/validate_delivery.py --device cuda --display. Never assume another machine inherits these measurements.
