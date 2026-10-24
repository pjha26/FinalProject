# Step 6: full KITTI Eigen evaluation

Completed 25 October 2026. All **697 pinned test identities** were selectively acquired from the public raw archive endpoints referenced by upstream tooling. HTTPS byte ranges pin archive ETags; selected members pass ZIP CRC and file SHA256 checks. Calibration, selected native RGB/LiDAR and generated reference arrays remain local. No full-archive checksum is claimed because only selected members were transferred.

The original pinned Monodepth2 projection is reused with only removed `np.int` aliases changed to builtin `int`; the source/compatibility hashes and a duplicate-point fixture test are recorded. Evaluator rejects partial/unpinned identities and changed source hashes. Native reference projection uses camera 2 and upstream `vel_depth=True`, collision minimum and native dimensions. Input uses PIL LANCZOS 640x192, internal checkpoint normalization, Garg crop, 0.001-80 m mask, per-image GT median scaling, no flip post-processing and mean over images.

| Metric | Pretrained | Fixed epoch-3 candidate |
|---|---:|---:|
| abs_rel | 0.114617 | 0.135153 |
| sq_rel | 0.904808 | 1.192996 |
| rmse | 4.863413 | 5.253816 |
| rmse_log | 0.192775 | 0.207001 |
| a1 | 0.877429 | 0.855952 |
| a2 | 0.959219 | 0.954888 |
| a3 | 0.980870 | 0.978864 |

Both models score 11,817,275 valid pixels on identical examples. The 24-image cropped development pilot shares **0 exact frames and 7 drives** with this full split. Thus there is no exact-frame reuse, but this is not wholly untouched development at drive level. No retraining or checkpoint reselection followed Eigen results.

The pretrained baseline meets the proposal AbsRel and delta1 thresholds in this run, but SqRel 0.904808, RMSE 4.863413 m and log RMSE 0.192775 miss the strict 0.9/4.8/0.190 targets. The experimental candidate is worse overall and remains rejected as default. Report all thresholds honestly, without rounding failures into passes.

[Complete metadata](full_eigen_results.json) and `runs/research_completion_01/eigen_baseline/` / `eigen_candidate/` contain numerical evidence. All 697 predictions per model are saved locally. Acquisition supports incomplete-cache resume and never publishes an incomplete dataset. Range-bound/identity/CRC and projection tests pass. Dataset/source licenses remain applicable.

Primary protocol sources: [Monodepth2](https://github.com/nianticlabs/monodepth2), [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php). This is classical Eigen sparse-LiDAR evaluation, not the improved 652-image benchmark. Raw assets/weights stay ignored and require separate backup.
