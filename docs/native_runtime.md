# Step 8: native CARLA displayed inference

25 October 2026. The accepted pretrained Monodepth2 default processed a moving native CARLA RGB feed with reference scoring and the RGB/depth/overlay display. The owned server was stopped after clean client completion. Settings: CARLA 0.9.13, Town04, seed 42, ClearNoon, Low off-screen rendering, no sound, 640x480 camera and 640x192 inference, i7-13620H / 16 GB / RTX4050 6 GB.

**783 consecutive matched frames in 65.0114 processing seconds: 12.0440 application FPS.** Displayed-duration and >=10 FPS gates pass for this configuration. Processing includes capture, model prediction, scoring/rendering and display; server/model startup and unrecorded warmup are excluded. Simulation timing is 10 Hz and is not substituted for wall-clock FPS. Per-frame CSV and review maps are preserved.

`--stop-after-seconds` now optionally ends demo processing after a wall-duration while preserving the existing frame-limit/default behavior. It validates finite nonnegative values before loading. The summary records duration and actual capture settings. Tests cover early bounded stop and invalid values; headless operation still cannot claim displayed acceptance.

```powershell
.\.venv\Scripts\python.exe project.py demo --source carla --map Town04 --seed 42 --weather ClearNoon --timeout 60 --frames 5000 --stop-after-seconds 65 --min-seconds 60 --fps 30 --device cuda --review-every 100 --output runs/native_demo_next
```

The fps argument caps video/webcam pacing; CARLA remains synchronously driven and this run reports observed application FPS. Low quality reduces visual fidelity/drawing range. This measured result is not a guarantee on other maps/conditions/hardware or a vehicle-control latency bound. GT-median scoring is diagnostic; predictions remain relative. Physical webcam, other-laptop and physically network-disconnected rehearsals are not claimed.

[Complete result metadata](native_runtime_results.json), `runs/research_completion_01/native_display/` and `native_display.log` record the run, snapshots, timing, allocation/memory and acceptance. No candidate was promoted because of runtime; the research checkpoint had already failed development quality guards.
