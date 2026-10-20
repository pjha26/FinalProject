# Poster alignment review — 21 October 2026

The project implements the poster's central task: estimating **relative scene depth from a single RGB image or each frame of a video**. It uses pretrained weights; it does not train a new model or measure physical distances. The poster needs evidence-based wording in several places. Adding features cannot establish an unmeasured performance or accuracy claim.

## Claim-by-claim comparison

| Poster claim | Actual implementation and evidence | Required presentation change |
|---|---|---|
| Monodepth2 with ResNet-18, 640 × 192 | Implemented as the default backend, using the official monocular checkpoint. | Keep this when demonstrating `--model monodepth2`. |
| Image, driving video and webcam inputs | Image and recorded driving video run successfully. Webcam capture exists, but the physical camera-index-0 probe failed to open on this laptop. | Say “image/video demonstrated; webcam supported in software, physical validation pending.” A recorded video is not a successful live-camera test. |
| Resize, normalization, encoder/decoder, depth visualization | Implemented through each selected backend's preprocessing and inference contract. Baseline normalization is handled by its model implementation. | Keep the baseline pipeline diagram. Draw post-processing **after the depth prediction**; RGB also supplies colours for the optional cloud. |
| Relative depth map | Implemented. Display uses inverse-depth colours: yellow is nearer, purple farther. Numerical relative-depth and inverse-depth arrays are saved. | Keep; label the legend “relative”, without metre distances. Stronger colours do not mean better numerical accuracy. |
| Optional 3D point cloud | Added in this change through `infer --point-cloud` and `demo --point-cloud`. Saves real back-projected predicted points, RGB colours, PLY, PNG, metadata and an offline interactive viewer. | Describe as an **estimated relative point-cloud snapshot**, not a measured 3D reconstruction. Demo export is the first processed frame only. |
| Trained on KITTI | The supplied baseline was pretrained by the original authors on KITTI. This project has not trained weights. The selected checkpoint is monocular self-supervised, not a project model trained with stereo/depth labels. | Use “Pretrained Monodepth2 checkpoint, trained by its authors on monocular KITTI sequences.” Distinguish evaluation GT from training supervision. |
| Diverse lighting/weather | No project robustness evaluation establishes this. | Remove the implied robustness result; mention varied conditions as future evaluation. |
| CPU: 50–70 ms and 14–20 FPS | Not reproduced. Today's CPU profile measured approximately 578 ms median model inference and 1.56–1.57 FPS for the headless pipeline. | Replace the performance table with measured, device-specific results below. |
| Accurate/high-quality results on real-world scenes | A small five-frame indoor diagnostic exists; the driving demo has no aligned ground-truth depth. No full KITTI benchmark was run locally. | Say “produces relative-depth visualizations”; qualify accuracy with the actual diagnostic and its scope. |
| Vehicle boxes, impressive street point clouds and webcam result pictures | The pipeline does not detect or track objects. Poster artwork is not evidence of these functions or of physical webcam validation. | Replace result illustrations with actual saved outputs; otherwise label them “illustrative”. Remove vehicle boxes from claimed project outputs. |
| Navigation, obstacle detection, AR/VR | Potential downstream applications; no obstacle detector, vehicle controller, SLAM or calibrated range output is implemented. | Present these as applications/future work, not delivered functions. |

Official checkpoint/training reference: [Monodepth2 repository and pretrained model table](https://github.com/nianticlabs/monodepth2). Publisher benchmark results are not this project's measured results.

## Which model is being shown?

The default remains **Monodepth2 ResNet-18** so existing phase commands retain their baseline behavior. An optional **Depth Anything V2 Small** backend was added during accuracy work; it is a different architecture and must not be called Monodepth2 in the presentation. The window title, preview caption and saved run metadata now identify the actual backend.

The five correlated indoor reference pairs gave GT-median-aligned AbsRel approximately **0.4960** for Monodepth2 and **0.2033** for Depth Anything V2 Small at input size 294. This supports the optional model for that diagnostic only; it does not establish driving accuracy, metre-scale deployment, or a universal improvement. See [accuracy report](accuracy_report.md) and [evaluation protocol](evaluation_protocol.md).

If the optional backend is your main demonstration, use a title such as **“Monocular Relative Depth Estimation with Lightweight Pretrained Models”**, and show the two backends separately. Retain the current poster title only when presenting the Monodepth2 baseline.

## Measured performance for the replacement table

These are local measurements from 21 October 2026, not a guaranteed minimum. Different pipelines include different work; do not equate model latency with displayed FPS.

| Configuration | Measurement | Local evidence (ignored by Git) |
|---|---|---|
| Monodepth2, CPU, 640 × 192, batch 1, two CPU threads, 30 frames/repeat, three warmups, two repeats | Median model time **578.29 ms**; median full-pipeline time **624.26 ms**; repeat headless rates **1.57 / 1.56 FPS** | `runs/poster_cpu_profile/summary.json` |
| Monodepth2, GTX 1650 4 GB, displayed driving video, 650 frames, configured cap 30 FPS/source 25 FPS, optional first-frame cloud | **65.27 s**, **9.96 completed FPS**; median model **28.49 ms**, median capture-to-display **76.43 ms** | `runs/poster_baseline_display/summary.json` |

The displayed run's peak process working set was approximately **707.80 MiB**, and peak PyTorch GPU allocation **106.34 MiB**. These exclude total operating-system and GPU-driver usage. The friend's RTX 3050 is still untested. Camera index 0 failed to open during this review; no webcam FPS is reported.

## Added point-cloud demonstration

Run from the repository root after setting up the existing environment and baseline assets. Use a fresh output directory each time:

```powershell
.\.venv\Scripts\python.exe project.py demo --model monodepth2 --source video --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json --device cuda --frames 650 --fps 10 --point-cloud --output runs/poster_presentation_01
```

The moving window still displays RGB, predicted relative inverse depth and the colour overlay. Open `runs/poster_presentation_01/point_cloud/viewer.html` separately in a browser: drag or use buttons to rotate, and scroll or use buttons to zoom. The file works offline. It shows a **static snapshot of the first frame**, not a continuously updated 3D video. For one image:

```powershell
.\.venv\Scripts\python.exe project.py infer --model monodepth2 --image data/demo/driving_first_frame.png --device cuda --point-cloud --output runs/poster_image_01
```

The cloud folder contains `cloud.ply`, `viewer.html`, `preview.png` and `metadata.json`. Original RGB colours are aligned to the model's depth grid. Export is bounded to 10,000 points, excludes invalid/inverse-floor values and depths beyond 20 scene medians, and records those counts. It uses returned camera intrinsics at prediction resolution; ordinary image/video inputs use assumed intrinsics, whereas compatible recorded sensor scenes can supply them. Depth is normalized by its valid median. Neither supplied intrinsics nor this normalization recovers physical scale or fixes model errors. A cloudy, stretched or incomplete scene reflects the predicted geometry; there is no hidden LiDAR or complete scene reconstruction.

Without `--point-cloud`, no cloud is generated. It does not alter saved numerical depth maps or model weights. No dependencies were added. Both model backends retain their existing numerical output dimensions.

## Suggested replacement wording

**Introduction/contribution:** “We integrated pretrained monocular depth networks into an RGB image and video pipeline with relative-depth visualization, optional point-cloud export, runtime profiling and reproducible evaluation. Monodepth2 ResNet-18 is the baseline; Depth Anything V2 Small is an optional comparison model.”

**Dataset:** “The baseline uses author-provided KITTI-pretrained weights. Demonstration uses a recorded driving clip. A separate five-frame indoor reference set supports diagnostic evaluation with ground-truth median alignment. No project-specific training or full driving-depth benchmark has been completed.”

**Results:** “Recorded driving-video inference and relative-depth overlays were demonstrated on the GTX 1650. An optional first-frame relative point-cloud snapshot can be inspected interactively. Runtime depends on hardware, model, display and export work. Physical webcam validation remains pending.”

**Conclusion:** “The implemented pipeline estimates relative scene depth from a single RGB image or successive video frames on limited hardware. It demonstrates perception and visualization, with device-specific runtime measurements. Calibrated distances, robust driving evaluation and downstream navigation remain future work.”

## Validation of this change

- **81 automated tests passed**, including seven point-cloud tests for geometry, colour alignment, scale ambiguity, filtering, point budgets, invalid inputs, safe offline export and CLI integration.
- Actual baseline image inference and a 650-frame displayed video run exported the optional cloud successfully. The saved PNG was inspected. The generated viewer loaded in the browser; rotate/zoom/reset buttons executed without browser errors. Browser screenshot capture was unavailable, so the interactive canvas was not visually verified through a browser screenshot.
- Actual Depth Anything V2 Small inference also exported a 10,000-point cloud from its native 360 × 640 prediction grid (`runs/poster_quality_image/`). Its metadata records 105,248 invalid/floor pixels and 9,439 far pixels excluded for this image. This validates backend integration, not geometry accuracy; the inspected PNG is sparse and does not reproduce the poster's polished street reconstruction.
- CPU profiling completed with repeatable predictions. Physical webcam capture failed and remains unresolved.
- Generated evidence and outputs stay under ignored `runs/`; they are not included in source commits.
