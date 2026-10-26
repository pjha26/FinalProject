# Monocular Depth Estimation for Autonomous Driving
Presentation pitch and viva notes | 26 October 2026

## Opening: the problem and our aim
Good morning. Our project explores how a single RGB camera can estimate the relative depth of a driving scene. An ordinary image records colour, but a perception system also needs to understand which surfaces appear nearer or farther away. Our aim is to build and evaluate a practical Monodepth2 pipeline that captures live CARLA camera frames, predicts dense relative depth, and displays the result at more than ten frames per second on a commodity GPU.

## Why we chose this project
Camera-based perception is accessible and relevant to autonomous-driving research. LiDAR provides valuable geometric measurements but adds sensor cost and integration needs. We chose monocular depth to study what a single camera can infer, including its limitations. We chose Monodepth2 because its established architecture and temporal self-supervision let us investigate depth learning without requiring depth labels for the training loss. CARLA provides a repeatable environment and synchronized reference depth for evaluation. This prototype does not establish that a camera can replace LiDAR in a safety-critical vehicle.

## How the system works
The pipeline is: CARLA RGB capture, resize to 640 x 192, pretrained ResNet-18 encoder and depth decoder, numerical depth prediction, then colour map and overlay. Purple indicates estimated farther regions and yellow indicates nearer regions. During fine-tuning, adjacent RGB frames and a pose network support photometric reconstruction, SSIM, automasking and smoothness losses. Simulator reference depth is used for scoring, not as a network input or training target.

## What we built and measured
Our contribution is the simulator integration, reproducible input pipeline, bounded fine-tuning experiments, checkpoint/resume support, visualization, evaluation and presentation tooling. We use the published model architecture and official pretrained weights; we do not claim a new architecture. The training experiment used 392 triplets per epoch for three epochs, with a frozen depth encoder. We also ran matched controls, loss ablations and seed repeats.
The fresh live-demo rehearsal processed 780 frames in 65.01 seconds at 12.00 application FPS on an RTX 4050 laptop. All 137 tests and eight delivery checks passed. On all 697 KITTI Eigen frames, the pretrained baseline achieved AbsRel 0.114617, RMSE 4.863413 and delta1 0.877429, using ground-truth median alignment for scoring. AbsRel and delta1 meet the proposal targets; RMSE, SqRel and log-RMSE do not. The CARLA fine-tuned candidate regressed on KITTI, so the demo retains pretrained weights.

## Demonstration, limitations and next steps
We will show moving simulator RGB, predicted depth, overlay and FPS, followed by saved evaluation results. Recorded driving input is the fallback. Predictions describe relative structure, not calibrated metres, and the project does not steer or brake a vehicle. Next steps are more representative training data, controlled longer training, fresh independent test routes and improved cross-domain accuracy. Our result is a working and evaluated research prototype, with its remaining limitations made explicit.

---PAGE---

# Likely viva questions and answers

## 1. What is monocular depth estimation?
It predicts scene depth from one RGB image. Our model estimates relative near/far structure; absolute metric scale is ambiguous without additional information.

## 2. Why Monodepth2 instead of designing a new network?
It provides an established ResNet encoder, depth decoder and self-supervised framework. We can focus on integration, controlled experiments and evaluation instead of claiming an untested new architecture.

## 3. How can training work without depth labels?
A pose network estimates relative camera motion. Predicted depth and motion warp adjacent frames toward the target image; reconstruction error supplies the learning signal. Depth reference data is kept for evaluation.

## 4. What do SSIM, automasking and smoothness do?
SSIM compares local image structure. Automasking suppresses pixels where reconstruction does not improve over unchanged source images. Edge-aware smoothness encourages consistent depth while respecting image boundaries. These mechanisms reduce errors but do not solve every moving-object failure.

## 5. Did you train the model yourself?
We fine-tuned official pretrained weights in bounded CARLA experiments: three epochs, 392 triplets per epoch, with the depth encoder frozen. We did not train from scratch, and we have not established convergence.

## 6. Why does the demo still use pretrained weights?
The candidate improved CARLA development AbsRel but worsened full KITTI AbsRel from 0.114617 to 0.135153. Our quality guard rejected promotion. Lower training loss alone does not prove better generalization.

## 7. How did you evaluate accuracy?
We evaluated all 697 pinned KITTI Eigen frames with a Garg crop, an 80-metre cap and per-image ground-truth median scaling. We also compared frozen models on 100 reserved CARLA frames. Reference depth never enters the inference network.

## 8. Can you claim that every PDF target was achieved?
No. Baseline AbsRel and delta1 passed. RMSE 4.863413, SqRel 0.904808 and log-RMSE 0.192775 missed the strict 4.8, 0.9 and 0.190 targets. The measured native runtime passed the ten-FPS objective on this laptop.

## 9. What exactly does your FPS measurement include?
The fresh native run achieved about 12.00 processed frames per second over 65 seconds, including capture, inference, diagnostic scoring and display. Startup is excluded. This is application throughput, not simulator frequency or a guarantee on another computer.

## 10. Can it measure exact distances or drive a car?
No. Median alignment uses reference depth during evaluation and is unavailable in deployment. The live predictions remain relative. Vehicle control, collision avoidance and safety certification are outside this prototype.

## 11. What is original in your work, and what is still weak?
Our work integrates capture, data checks, training controls, visualization, benchmark evaluation and repeatable delivery. The model is published research. Short correlated routes, limited weather coverage, domain shift and scale ambiguity limit our claims.

## 12. What if the live demonstration fails?
We preserve the failure logs and run the prepared driving-video fallback, describing it accurately as recorded-input inference. We rehearse both paths beforehand. Another laptop requires its own environment, assets and performance validation.

Sources: original project proposal; repository docs/final_research_status.md, full_eigen.md and presentation_runbook.md. Code and protocols: https://github.com/pjha26/FinalProject

