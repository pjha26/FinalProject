# Real-RGB teacher replay: 128 updates

27 October 2026. The predeclared replay trial uses Adam2e-6, the same supervised CARLA fit images, and teacher regularization on 16 hash-verified Udacity driving frames. KITTI images and ground truth are not training inputs.

CARLA mean development AbsRel improves from 0.24866763 to 0.23633542, a 4.9593% relative gain. The absolute pixel thresholds pass, but the strict >=5% improvement requirement misses. KITTI24 AbsRel is 0.09549767 versus baseline0.09362687, within the <=5% regression guard. The candidate remains rejected; the gain is not rounded into a pass.

Weights, exact clip hash/frame identities, teacher identity, training source hashes and the final128-update model remain in runs/metric_decoder_replay_01/. Development scores are in runs/metric_decoder_replay_validation_01/. No final Eigen or fresh network holdout is used for selection, and learned actuation remains disabled.

