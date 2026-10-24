# Paired driving review - 24 October 2026

## What was completed

Both existing models processed identical 300-frame prefixes from three pinned, permitted daytime driving clips: the accepted highway sample, an additional freeway/overpass clip, and a winding tree-lined road clip. All six sequential CUDA runs completed. All 42 saved review maps were finite, positive float32 arrays. Both models saved the same review frame IDs per clip: 1, 2, 100, 101, 200, 201 and 300. All six metrics files were empty because there is no reference depth.

Application source, model defaults, requirements, preprocessing and accepted weights were unchanged from `76a089d`. The full 87-test suite passed after these runs. This milestone implements the additional-footage review in the [driving validation plan](driving_validation.md); it does not establish driving-depth accuracy or complete the original research scope.

## Input provenance

The [frozen input catalogue](driving_review_inputs.json) records source URLs, source revision, SHA256 hashes, retained license, full decoded frame counts, geometry and frame rates. The two new clips total 32,343,769 bytes (about 30.85 MiB); no large dataset was downloaded.

All three come from Udacity's Advanced Lane Lines repository at revision `91b080357290da37f533e2d0832f601c3e770d51`. The upstream MIT license was retained and matched the existing pinned license checksum. The two new files also matched the Git blob SHA1s in the pinned upstream tree. Full decoding succeeded before inference: highway 1260 frames, challenge 484 frames, harder 1199 frames, all 1280x720. No independent-scene or training-overlap guarantee is established by using different filenames from one collection.

## Measured runtime

Hardware/environment: the [accepted RTX 4050 laptop](new_laptop_validation.md), CUDA, batch one, two PyTorch CPU threads, Monodepth2 640x192 or Small with a short-edge target of 294. Each run used the existing headless demo, 300 frames, a 30 FPS ceiling, snapshots every 100 frames, no point cloud, and no smoothing or model changes.

The video loop also caps at the source rate: 25 FPS for highway/harder and approximately 29.97 FPS for challenge. Reported rates include decoding, preprocessing, inference, composite rendering, first-frame outputs, review exports and pacing. Model loading, full input verification and final report writes are excluded. No initial frame warmup was excluded in these video measurements.

| Clip | Backend | Seconds | Completed headless FPS | Median model ms |
|---|---|---:|---:|---:|
| highway | Monodepth2 ResNet-18 mono_640x192 | 14.70 | 20.40 | 12.42 |
| highway | Depth Anything V2 Small | 14.71 | 20.39 | 17.67 |
| challenge | Monodepth2 ResNet-18 mono_640x192 | 12.68 | 23.66 | 9.74 |
| challenge | Depth Anything V2 Small | 16.44 | 18.25 | 19.30 |
| harder | Monodepth2 ResNet-18 mono_640x192 | 20.75 | 14.46 | 15.15 |
| harder | Depth Anything V2 Small | 37.71 | 7.96 | 40.38 |

These are one sequential trial per configuration under ordinary workstation conditions, not repeated controlled speed benchmarks. In particular, the Small harder run achieved 7.96 FPS despite a 30 FPS ceiling. This does not establish a 10 FPS minimum. Nor does it prove that scene content caused the slowdown: system load, GPU clocks, export cost and run order were not controlled. The prior 60-second displayed acceptance applies to its original highway workload, not automatically to these extra clips.

## Qualitative inspection

Paired previews at frames 1, 100 and 300, plus adjacent frames 200/201, were visually inspected for each clip. These observations describe appearance, not measured depth correctness:

| Clip / frames | Observed behavior |
|---|---|
| Highway, 1/100/300 | Small gives a more distinct sky/vegetation silhouette. Monodepth2 shows broad warm-coloured sky regions and softer horizon boundaries. The white car at frame 300 appears as a near-region blob in both; Small's outline is more distinct in this view. |
| Challenge, 1/100/300 | Small retains a more distinct outline of the overpass and its visible opening at frame 100. Monodepth2 blurs these structures more strongly. Distant vehicles remain small and lack independent depth validation in both. |
| Harder, 1/100/300 | Both show a broad road-relative gradient on bends. Small outlines the motorcycle at frame 100 more distinctly and separates small sky gaps through vegetation. Fine tree/guardrail structure and shadowed road surfaces are not verified geometry. |
| Adjacent 200/201 in all clips | Broad near/far structure persists across these inspected pairs, with contour changes. The images move with the camera and objects; this is not a motion-compensated temporal-consistency metric or proof of flicker-free playback. |

Backend outputs have different relative scales, so the same display hue is not evidence of the same distance across models. Small's upstream zero inverse-depth values are floored to 1e-6 for a finite reciprocal; dark sky regions must not be interpreted as measured distant geometry. Among the seven saved Small snapshots per clip, floor-valued fractions were 43.92-46.84% for highway, 27.86-45.33% for challenge, and 0.10-3.78% for harder. These are whole-image fractions, not semantic sky masks. At most 1.18% of pixels in these saved Small maps reached the fixed display's upper inverse-depth clip of 10. Numerical maps remain available before colour clipping.

Monodepth2 stays the default and accepted fallback. Small is an optional qualitative candidate; these RGB-only observations do not establish a universal accuracy winner, calibrated metres or safer vehicle behavior.

## Reproduce using existing commands

Prepare the baseline model/input normally and install optional Small assets first. Run from the repository root, with unused output names. Download only missing files, then reject any checksum mismatch instead of overwriting an existing file:

```powershell
$inputs = Get-Content docs/driving_review_inputs.json -Raw | ConvertFrom-Json
$assets = @([pscustomobject]@{file=$inputs.license_file; url=$inputs.license_url; sha256=$inputs.license_sha256}) + $inputs.clips
foreach ($asset in $assets) {
    if (-not (Test-Path -LiteralPath $asset.file)) {
        New-Item -ItemType Directory -Force (Split-Path $asset.file) | Out-Null
        Invoke-WebRequest -Uri $asset.url -OutFile $asset.file
    }
    if ((Get-FileHash -LiteralPath $asset.file -Algorithm SHA256).Hash.ToLower() -ne $asset.sha256) {
        throw "Input checksum mismatch: $($asset.file)"
    }
}
.\.venv\Scripts\python.exe -c "import json; from src.demo_data import inspect_video; d=json.load(open('docs/driving_review_inputs.json')); [print(c['id'], inspect_video(c['file'])[0]) for c in d['clips']]"
foreach ($clip in $inputs.clips) {
    foreach ($model in @('monodepth2', 'depth-anything')) {
        .\.venv\Scripts\python.exe project.py demo --source video --video $clip.file --model $model --input-size 294 --device cuda --frames 300 --fps 30 --review-every 100 --headless --output "runs/driving_review_next/$($clip.id)_$model"
        if ($LASTEXITCODE -ne 0) { throw "Review failed: $($clip.id) $model" }
    }
}
```

Recheck input hashes after the runs. Keep the same frame range and configuration for each paired model. A full decode failure, nonzero exit, `failure.json`, mismatched completed frame count, modified input or missing summary invalidates that run. Do not aggregate a partially completed comparison as success.

Local evidence: `runs/driving_review_01/summary.json`, `review_statistics.json`, `tests.log`, six job logs and six run subdirectories with previews, raw review maps, timing CSVs and run summaries. Local comparison sheets are generated from these actual saved previews; no geometry was fabricated. All footage, weights and raw runs remain ignored and must be transferred separately from Git.

## Next step: aligned driving-depth evidence

On 24 October 2026 the [official KITTI depth prediction page](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) lists the selected validation/test download at 2 GB and the full annotated-depth download at 14 GB; its download links require login. The user reports no local KITTI data. Dataset acquisition is therefore pending account access/download, not a completed benchmark.

Use validation examples with published GT, not hidden-GT test frames, for the local pilot. Before inference, inspect the actual archive layout, documented depth encoding, RGB registration, camera/scene identity and potential overlap with each pretrained model's training data. Freeze a small scene-separated evaluation subset and its protocol. Label it as a local diagnostic until the full appropriate KITTI protocol and provenance have been established. The existing evaluator's Garg crop alone is not the official Eigen evaluation.

No large CARLA collection, training, calibrated-distance output, model replacement, new dependency or new inference framework was introduced for this review.
