"""CARLA-only straight-route control experiment; reference control and learned shadow."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project import write_json
from src.carla_capture import CarlaCapture, CaptureSettings, FrameSynchronizer
from src.demo_data import checkpoint_hashes
from src.metric_control import corridor_clearance, scaled_depth, steering_to_waypoint, bounded_control
from src.model import load_model, predict, sha256
from src.preprocess import preprocess_rgb, camera_intrinsics, PREPROCESSING_ID
from src.visualize import render_frame


def load_calibration(path, checkpoint, actuate=False):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("preprocessing_id") != PREPROCESSING_ID or data.get("checkpoint_sha256") != checkpoint_hashes(checkpoint):
        raise ValueError("Calibration/model/preprocessing identity mismatch")
    if data.get("camera") != dict(width=640, height=480, fov=90, mount_x=1.5, mount_z=1.7):
        raise ValueError("Calibration camera mismatch")
    if not np.isfinite(data["scale"]) or data["scale"] <= 0:
        raise ValueError("Invalid calibration scale")
    if actuate:
        # Model actuation additionally needs independently passed obstacle/clear-road shadow trials.
        raise ValueError("Learned actuation is disabled: development calibration/independent control acceptance is incomplete. Use reference mode or shadow logging.")
    return data


def straight_route(world_map, origin):
    candidates = sorted(world_map.get_spawn_points(), key=lambda t: t.location.distance(origin))
    for index, transform in enumerate(candidates):
        waypoint = world_map.get_waypoint(transform.location)
        route = [waypoint]
        for _ in range(10):
            next_points = [w for w in route[-1].next(4) if w.road_id == waypoint.road_id and w.lane_id == waypoint.lane_id and not w.is_junction]
            if not next_points:
                break
            chosen = min(next_points, key=lambda w: abs((w.transform.rotation.yaw - waypoint.transform.rotation.yaw + 180) % 360 - 180))
            yaw_delta = abs((chosen.transform.rotation.yaw - waypoint.transform.rotation.yaw + 180) % 360 - 180)
            if yaw_delta > 3 or abs(chosen.transform.rotation.pitch) > 1:
                break
            route.append(chosen)
        if len(route) == 11:
            return transform, route
    raise RuntimeError("No level straight 40 m route found; preserve failure instead of changing acceptance")


def run(output, calibration_path, mode="reference", obstacle=True, seed=42, weather="ClearNoon", display=True):
    import carla
    checkpoint = ROOT / "checkpoints/baseline_mono_640x192"
    calibration = load_calibration(calibration_path, checkpoint, actuate=mode == "model")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=1, mode=mode, obstacle=obstacle, seed=seed, weather=weather, target_speed_mps=2, max_steps=400, max_sensor_age_seconds=.35, stop_clearance_gate_m=1, max_center_lane_error_m=1.3, route_length_m=40, goal_progress_m=30, map="Town04", calibration_sha256=sha256(calibration_path), calibration_status=calibration["status"], checkpoint_sha256=checkpoint_hashes(checkpoint), controller_input="CARLA reference depth" if mode == "reference" else "rejected calibrated model estimates; simulator vehicle held braked", notes="Model prediction never receives GT. Map waypoints/ego telemetry are simulator privileged inputs. Reference mode is not learned autonomous driving.")
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(2)
    model = load_model(checkpoint, "cuda")
    for _ in range(3):
        predict(model, torch.zeros((1, 3, 192, 640), dtype=torch.float32))
    torch.cuda.synchronize()
    rows, collisions = [], []
    writer_video = None
    window = None
    status, reason, completed = "failed", "not_started", False
    started = time.perf_counter()
    try:
        if display:
            import pygame
            pygame.init()
            window = pygame.display.set_mode((1280, 430))
            pygame.display.set_caption("Experimental CARLA control - simulator reference / learned shadow")
        with CarlaCapture(CaptureSettings(map="Town04", seed=seed, timeout=60, weather=weather)) as capture:
            vehicle, world = capture.vehicle, capture.world
            vehicle.set_autopilot(False, capture.settings.traffic_manager_port)
            vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            spawn, route = straight_route(world.get_map(), vehicle.get_location())
            vehicle.set_transform(spawn)
            vehicle.set_target_velocity(carla.Vector3D())
            vehicle.set_target_angular_velocity(carla.Vector3D())
            obstacle_actor = None
            if obstacle:
                target = route[6].transform
                target.location.z += .5
                obstacle_actor = world.try_spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"), target)
                if obstacle_actor is None:
                    raise RuntimeError("Obstacle spawn failed")
                capture.actors.append(obstacle_actor)
                obstacle_actor.set_simulate_physics(False)
                obstacle_actor.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            collision_sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
            capture.actors.append(collision_sensor)
            collision_sensor.listen(lambda event: collisions.append(dict(frame=int(event.frame), other_actor=int(event.other_actor.id))))
            # Drop pre-teleport queued frames; callbacks resolve the replacement synchronizer dynamically.
            capture.sync = FrameSynchronizer()
            capture.snapshots = {}
            capture.expected_frame = None
            for _ in range(12):
                capture.next_frame(warmup=True)
            capture.expected_frame = None
            route_data = [dict(x=w.transform.location.x, y=w.transform.location.y, z=w.transform.location.z, road_id=w.road_id, lane_id=w.lane_id) for w in route]
            write_json(output / "route.json", route_data)
            stop_latched, stop_ticks = False, 0
            max_speed, max_progress, min_gap = 0.0, 0.0, None
            lane_errors, clearance_errors = [], []
            origin = route[0].transform.location
            forward = route[0].transform.get_forward_vector()
            reason = "step_limit"
            with (output / "runtime.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = None
                for step in range(400):
                    if window is not None and any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE) for e in pygame.event.get()):
                        reason = "user_closed_early"
                        break
                    record = capture.next_frame()
                    frame_started = time.perf_counter()
                    tensor, _ = preprocess_rgb(record["rgb"])
                    maps = predict(model, tensor)
                    relative = maps["relative_depth"][0, 0].cpu().numpy()
                    inverse = maps["inverse_depth"][0, 0].cpu().numpy()
                    metric = cv2.resize(scaled_depth(relative, calibration["scale"]), (640, 480), interpolation=cv2.INTER_LINEAR)
                    k = camera_intrinsics(640, 480)
                    predicted_distance = corridor_clearance(metric, k)
                    reference_distance = corridor_clearance(record["depth"], k)
                    now = world.get_snapshot()
                    lag_frames = now.frame - record["frame_id"]
                    sensor_age = max(0, now.timestamp.elapsed_seconds - record["simulation_timestamp"]) + time.perf_counter() - frame_started
                    if lag_frames < 0:
                        raise RuntimeError("Camera frame is newer than current world snapshot")
                    location, transform = vehicle.get_location(), vehicle.get_transform()
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    delta = location - origin
                    progress = delta.x * forward.x + delta.y * forward.y
                    lane_error = abs(-delta.x * forward.y + delta.y * forward.x)
                    nearest = min(range(len(route)), key=lambda i: route[i].transform.location.distance(location))
                    target = route[min(nearest + 1, len(route) - 1)].transform.location
                    steer = steering_to_waypoint((location.x, location.y), transform.rotation.yaw, (target.x, target.y))
                    if abs(transform.rotation.pitch) > 2 or abs(transform.rotation.roll) > 2:
                        stop_latched, reason = True, "unsupported_camera_attitude"
                    if collisions or lane_error > 1.3:
                        stop_latched, reason = True, "collision_or_lane_departure"
                    if not obstacle and progress >= 30:
                        stop_latched, reason = True, "route_goal"
                    command = bounded_control(speed, reference_distance, sensor_age, steer, stop_latched)
                    if mode == "shadow":
                        command = bounded_control(speed, reference_distance, sensor_age, steer, stop=True)
                    if command["reason"] in ("invalid_state", "invalid_or_stale_depth"):
                        reason, stop_latched = command["reason"], True
                    if command["reason"] == "obstacle_stop":
                        stop_latched = True
                        reason = "obstacle_stop" if obstacle else "false_stop_on_clear_route"
                    vehicle.apply_control(carla.VehicleControl(throttle=command["throttle"], brake=command["brake"], steer=command["steer"]))
                    gap = None
                    if obstacle_actor is not None:
                        offset = obstacle_actor.get_location() - location
                        gap = offset.x * forward.x + offset.y * forward.y - vehicle.bounding_box.extent.x - obstacle_actor.bounding_box.extent.x
                        min_gap = gap if min_gap is None else min(min_gap, gap)
                    max_speed, max_progress = max(max_speed, speed), max(max_progress, progress)
                    lane_errors.append(lane_error)
                    if reference_distance < 40:
                        clearance_errors.append(abs(predicted_distance - reference_distance))
                    row = dict(step=step, frame=record["frame_id"], simulation_seconds=record["simulation_timestamp"], simulator_lag_frames=lag_frames, sensor_age_seconds=sensor_age, reference_clearance_m=reference_distance, model_estimate_unvalidated_m=predicted_distance, speed_mps=speed, progress_m=progress, lane_error_m=lane_error, vehicle_gap_m=gap, throttle=command["throttle"], brake=command["brake"], steer=command["steer"], reason=command["reason"])
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    caption = "SIMULATOR REFERENCE CONTROL | model estimate UNVALIDATED {:.1f}m | reference {:.1f}m | speed {:.1f}m/s | {}".format(predicted_distance, reference_distance, speed, reason)
                    preview = render_frame(record["rgb"], inverse, caption)
                    video_frame = cv2.resize(preview, (1280, round(preview.shape[0] * 1280 / preview.shape[1])))
                    if writer_video is None:
                        writer_video = cv2.VideoWriter(str(output / "simulated_time_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (video_frame.shape[1], video_frame.shape[0]))
                        if not writer_video.isOpened():
                            raise RuntimeError("Could not open evidence video encoder")
                    writer_video.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
                    if step == 0 or step % 25 == 0 or stop_latched:
                        Image.fromarray(preview).save(output / "preview.png")
                    if window is not None:
                        surface = pygame.surfarray.make_surface(preview.transpose(1, 0, 2))
                        target_rect = surface.get_rect().fit(window.get_rect())
                        window.fill((20, 20, 20))
                        window.blit(pygame.transform.smoothscale(surface, target_rect.size), target_rect)
                        pygame.display.flip()
                    stop_ticks = stop_ticks + 1 if stop_latched and speed < .1 else 0
                    if stop_ticks >= 10 or mode == "shadow" and step >= 49:
                        break
                vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                maximum_age = max([r["sensor_age_seconds"] for r in rows], default=float("inf"))
                checks = dict(no_collision=not collisions, moved=max_progress >= 2, sensor_age_bounded=maximum_age <= .35, lane_bounded=max(lane_errors, default=999) <= 1.3, stopped=bool(rows and rows[-1]["speed_mps"] < .1), correct_outcome=reason == ("obstacle_stop" if obstacle else "route_goal"), clearance=not obstacle or min_gap is not None and min_gap >= 1)
                status = "pass" if all(checks.values()) and mode == "reference" else "shadow_only" if mode == "shadow" else "failed"
                completed = True
                write_json(output / "summary.json", dict(schema_version=1, status=status, reason=reason, mode=mode, obstacle=obstacle, frames=len(rows), wall_seconds=time.perf_counter() - started, checks=checks, maximum_speed_mps=max_speed, maximum_progress_m=max_progress, minimum_vehicle_gap_m=min_gap, max_sensor_age_seconds=maximum_age, collisions=collisions, model_clearance_mae_m=float(np.mean(clearance_errors)) if clearance_errors else None, calibration_accepted=calibration["gate"]["passed"], headless=not display, video_timebase="10 Hz simulation frames; not wall-clock screen recording", limitations="Straight level route only. Reference depth controls the car; learned depth is shadow-only and unvalidated. No learned camera-only autonomous-driving claim."))
    except BaseException as error:
        write_json(output / "failure.json", dict(error=str(error), frames=len(rows), reason=reason))
        raise
    finally:
        if writer_video is not None:
            writer_video.release()
        if window is not None:
            pygame.quit()
        if not completed:
            write_json(output / "summary.json", dict(status="failed", frames=len(rows), reason=reason, collisions=collisions))
    print("CONTROL EXPERIMENT", status, reason, output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["reference", "shadow", "model"], default="reference")
    p.add_argument("--clear-road", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weather", choices=["ClearNoon", "WetCloudyNoon"], default="ClearNoon")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()
    run(a.output, a.calibration, a.mode, not a.clear_road, a.seed, a.weather, not a.headless)


