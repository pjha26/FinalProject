# Supervised metric decoder trial 1

27 October 2026. A separate Monodepth2 depth decoder was trained for 128 updates on 40 Town04 road images and eight Town04 controlled-obstacle images, using simulator metric-depth labels. The encoder is frozen and baseline checkpoint bytes are unchanged. This is supervised metric fine-tuning, not the project's earlier temporal self-supervised method.

On 80 Town05 development frames, mean metric AbsRel improves from 0.24866763 to 0.17027507 (31.5250% relative); p95 frame error is 0.27322195. However, KITTI24 development AbsRel worsens from 0.09362687 to 0.15215403, failing the fixed <=5% regression guard. Therefore the candidate is rejected for default promotion. No fresh seed49/50 test is consumed.

The fixed final128-update checkpoint is retained at runs/metric_decoder_supervised_02/checkpoint/; development evidence is in runs/metric_decoder_validation_01/. The first attempt's network updates finished but serialization failed before a manifest was created; that failed output remains separate. A fresh rerun fixed manifest publication and completed. No accepted checkpoint was modified.

Four numerical loss tests verify exact-target zero loss, decoder gradient flow, teacher isolation, explicit object-region weighting and invalid supervision rejection. The small fit set does not establish convergence, generalization or control reliability.

