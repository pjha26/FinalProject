"""Synchronized sensor acquisition, recoverable recording, and validated replay."""
import csv
import json
from pathlib import Path
import queue
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from src.preprocess import camera_intrinsics, decode_rgb, decode_depth, PREPROCESSING_ID


@dataclass
class CaptureSettings:
    host: str = "127.0.0.1"
    port: int = 2000
    map: str = "Town04"
    width: int = 640
    height: int = 480
    fov: float = 90.0
    seed: int = 42
    timeout: float = 15.0
    traffic_manager_port: int = 8001
    warmup_frames: int = 10

    def __post_init__(self):
        if self.map not in ("Town04", "Town05"):
            raise ValueError("Phase 1 supports Town04 and Town05")
        if (self.width, self.height) != (640, 480) or self.timeout <= 0 or self.warmup_frames < 0:
            raise ValueError("Expected 640x480 camera, positive timeout, and nonnegative warmup")
        camera_intrinsics(self.width, self.height, self.fov)


class FrameSynchronizer:
    """Bound sensor queues and match identity, regardless of arrival order."""
    def __init__(self, capacity=64):
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive")
        self.capacity = capacity
        self.queues = {name: queue.Queue(capacity) for name in ("rgb", "depth")}
        self.pending = {name: {} for name in self.queues}
        self.overflowed = False

    def put(self, name, measurement):
        try:
            self.queues[name].put_nowait((measurement, time.perf_counter()))
        except queue.Full:
            self.overflowed = True

    def get_pair(self, frame_id, timeout):
        deadline = time.monotonic() + timeout
        for name in self.queues:
            while frame_id not in self.pending[name]:
                if self.overflowed:
                    raise RuntimeError("Sensor queue overflow; recording must not silently lose frames")
                try:
                    measurement, arrived_at = self.queues[name].get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty:
                    raise TimeoutError("Timed out waiting for {} frame {}".format(name, frame_id))
                if measurement.frame < frame_id:
                    continue
                if measurement.frame in self.pending[name]:
                    raise ValueError("Duplicate sensor frame: {} {}".format(name, measurement.frame))
                self.pending[name][measurement.frame] = (measurement, arrived_at)
                if len(self.pending[name]) > self.capacity:
                    raise RuntimeError("Too many unmatched sensor frame IDs")
        if self.overflowed:
            raise RuntimeError("Sensor queue overflow")
        paired = {name: self.pending[name].pop(frame_id) for name in self.queues}
        return {"rgb": paired["rgb"][0], "depth": paired["depth"][0], "arrival_timestamp": paired["rgb"][1]}


def transform_dict(transform):
    return {"location": {k: getattr(transform.location, k) for k in ("x", "y", "z")},
            "rotation": {k: getattr(transform.rotation, k) for k in ("pitch", "yaw", "roll")}}


class CarlaCapture:
    def __init__(self, settings):
        self.settings = settings
        self.actors = []
        self.world = self.previous_settings = self.traffic_manager = None
        self.previous_weather = None
        self.sync = FrameSynchronizer()

    def __enter__(self):
        import carla

        try:
            cfg = self.settings
            self.client = carla.Client(cfg.host, cfg.port)
            self.client.set_timeout(cfg.timeout)
            if self.client.get_server_version().split("-")[0] != "0.9.13" or self.client.get_client_version().split("-")[0] != "0.9.13":
                raise RuntimeError("Both CARLA server and client must be 0.9.13")
            self.world = self.client.load_world(cfg.map)
            self.previous_settings = self.world.get_settings()
            self.previous_weather = self.world.get_weather()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            self.traffic_manager = self.client.get_trafficmanager(cfg.traffic_manager_port)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(cfg.seed)
            blueprints = self.world.get_blueprint_library()
            vehicle_bp = blueprints.find("vehicle.tesla.model3")
            positions = list(self.world.get_map().get_spawn_points())
            random.Random(cfg.seed).shuffle(positions)
            self.vehicle = None
            for position in positions:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, position)
                if self.vehicle is not None:
                    break
            if self.vehicle is None:
                raise RuntimeError("No available vehicle spawn point")
            self.actors.append(self.vehicle)
            self.vehicle.set_autopilot(True, cfg.traffic_manager_port)
            mount = carla.Transform(carla.Location(x=1.5, z=1.7))
            for name in ("rgb", "depth"):
                blueprint = blueprints.find("sensor.camera." + name)
                for key, value in {"image_size_x": cfg.width, "image_size_y": cfg.height, "fov": cfg.fov,
                                   "sensor_tick": 0.1, "lens_k": 0.0, "lens_kcube": 0.0}.items():
                    blueprint.set_attribute(key, str(value))
                sensor = self.world.spawn_actor(blueprint, mount, attach_to=self.vehicle)
                self.actors.append(sensor)
                sensor.listen(lambda image, sensor_name=name: self.sync.put(sensor_name, image))
            for _ in range(cfg.warmup_frames):
                self.next_frame()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def next_frame(self):
        frame_id = self.world.tick(self.settings.timeout)
        snapshot = self.world.get_snapshot()
        pair = self.sync.get_pair(frame_id, self.settings.timeout)
        if snapshot.frame != frame_id or abs(pair["rgb"].timestamp - pair["depth"].timestamp) > 1e-6:
            raise RuntimeError("Sensor/snapshot timestamps do not match")
        rgb, depth = pair["rgb"], pair["depth"]
        vehicle_snapshot = snapshot.find(self.vehicle.id)
        if vehicle_snapshot is None:
            raise RuntimeError("Ego vehicle missing from matched snapshot")
        cfg = self.settings
        return {"frame_id": frame_id, "simulation_timestamp": rgb.timestamp,
                "arrival_timestamp": pair["arrival_timestamp"],
                "rgb": decode_rgb(rgb.raw_data, cfg.width, cfg.height),
                "depth": decode_depth(depth.raw_data, cfg.width, cfg.height),
                "ego_transform": transform_dict(vehicle_snapshot.get_transform()),
                "sensor_transform": transform_dict(rgb.transform)}

    def __exit__(self, exc_type, exc, traceback):
        errors = []
        for actor in reversed(self.actors):
            try:
                if actor.type_id.startswith("sensor.") and actor.is_listening:
                    actor.stop()
            except Exception as error:
                errors.append(str(error))
            try:
                actor.destroy()
            except Exception as error:
                errors.append(str(error))
        self.actors.clear()
        restore = []
        if self.traffic_manager is not None:
            restore.append(lambda: self.traffic_manager.set_synchronous_mode(False))
        if self.world is not None and self.previous_settings is not None:
            restore.append(lambda: self.world.apply_settings(self.previous_settings))
        if self.world is not None and self.previous_weather is not None:
            restore.append(lambda: self.world.set_weather(self.previous_weather))
        for action in restore:
            try:
                action()
            except Exception as error:
                errors.append(str(error))
        if errors:
            message = "CARLA cleanup errors: " + "; ".join(errors)
            if exc_type is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)


def collect_scene(output, settings, frames=100):
    if frames <= 0:
        raise ValueError("Frame count must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("rgb", "depth"):
        (output / folder).mkdir()
    metadata = {"schema_version": 1, "scene_id": output.name, "status": "recording", "captured_frames": 0,
                "preprocessing_id": PREPROCESSING_ID, "settings": asdict(settings),
                "intrinsics": camera_intrinsics(settings.width, settings.height, settings.fov).tolist(),
                "depth_units": "metres", "depth_convention": "CARLA raw depth buffer; wall check pending"}
    fields = ["scene_id", "frame_id", "simulation_timestamp", "arrival_timestamp", "rgb_path", "depth_path", "ego_transform", "sensor_transform"]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream, CarlaCapture(settings) as capture:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for _ in range(frames):
                record = capture.next_frame()
                stem = "{:08d}".format(record["frame_id"])
                rgb_path, depth_path = "rgb/" + stem + ".png", "depth/" + stem + ".npy"
                Image.fromarray(record.pop("rgb")).save(output / rgb_path)
                np.save(output / depth_path, record.pop("depth"), allow_pickle=False)
                for key in ("ego_transform", "sensor_transform"):
                    record[key] = json.dumps(record[key], separators=(",", ":"))
                writer.writerow(dict(record, scene_id=output.name, rgb_path=rgb_path, depth_path=depth_path))
                stream.flush()
                metadata["captured_frames"] += 1
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="partial", error=str(error))
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def scene_path(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Scene asset is missing or outside its scene: " + relative)
    return path


def replay_scene(scene):
    root = Path(scene).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete" or metadata.get("preprocessing_id") != PREPROCESSING_ID:
        raise ValueError("Replay requires a complete compatible scene manifest")
    camera = dict(metadata["settings"], intrinsics=metadata["intrinsics"])
    previous_id, count = -1, 0
    with (root / "frames.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_id = int(row["frame_id"])
            if frame_id <= previous_id or row["scene_id"] != metadata["scene_id"]:
                raise ValueError("Scene IDs or frame ordering are invalid")
            previous_id = frame_id
            rgb = np.asarray(Image.open(scene_path(root, row["rgb_path"])).convert("RGB"))
            depth = np.load(scene_path(root, row["depth_path"]), allow_pickle=False)
            if rgb.shape[:2] != (camera["height"], camera["width"]) or depth.dtype != np.float32 or depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError("Recorded numerical depth shape/values are invalid")
            count += 1
            yield dict(row, frame_id=frame_id, rgb=rgb, depth=depth, camera=camera)
    if count != metadata["captured_frames"]:
        raise ValueError("Scene frame count does not match its manifest")
