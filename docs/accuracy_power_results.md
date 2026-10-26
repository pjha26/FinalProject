# Accuracy validation: monotonic correction result

27 October 2026. The original 40/80-frame split, crop and model were reproduced. The baseline fixed scale is exactly 16.162327824804848. A two-parameter monotonic power mapping fitted only on Town04 reduces mean Town05 development AbsRel from 0.27433548 to 0.26106880 (4.8359% relative improvement), but increases p95 frame AbsRel from 0.35844924 to 0.54648827.

The candidate fails the original 0.25/0.50 accuracy gates and the predeclared minimum 5% relative improvement. It is rejected, and no new holdout examples are scored. Three repeated predictions of one fit image were exactly equal, so this example's problem is prediction accuracy rather than random inference variation.

This mapping is supervised postprocessing calibration, not retraining of Monodepth2. Existing pretrained weights, relative-depth demo and all vehicle-control guards remain unchanged. Source files, all source data hashes, predictions, frozen mapping, complete per-frame/binned scores and repeatability records are in runs/metric_accuracy_02/.

Six numerical tests cover known mapping recovery, robustness to outliers, equal frame sampling, invalid fits, numerical overflow and paired promotion gates. No PDF or presentation file is changed or uploaded.

