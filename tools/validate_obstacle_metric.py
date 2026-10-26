"""Matched stationary-obstacle depth diagnostic; ego is held braked throughout."""
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import write_json
from src.carla_capture import CarlaCapture,CaptureSettings,FrameSynchronizer,create_client
from src.demo_data import checkpoint_hashes
from src.metric_control import corridor_clearance,clearance_sample_mask,bounded_control
from src.model import load_model,predict,sha256
from src.preprocess import preprocess_rgb,camera_intrinsics
from tools.carla_control_demo import load_calibration,straight_route


def run(calibration,output):
    import carla
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/"checkpoints/baseline_mono_640x192"
    mapping=load_calibration(calibration,checkpoint)
    protocol=dict(scale=mapping["scale"],baseline_scale=16.162327824804848,calibration_sha256=sha256(calibration),checkpoint_sha256=checkpoint_hashes(checkpoint),weather=["ClearNoon","WetCloudyNoon"],route_waypoint_indices=[2,3,5,8],samples_per_pose=6,gate=dict(surface_mae_m=2,maximum_surface_overestimate_m=2,missed_brake_triggers=0),ego_control="Handbrake throughout; no model actuation",comparison="Both scales on same RGB predictions/reference sample mask. GT masks are diagnostics only, never inference inputs.")
    write_json(output/"protocol.json",protocol)
    torch.set_num_threads(2);model=load_model(checkpoint,"cuda")
    rows=[]
    for weather in protocol["weather"]:
        with CarlaCapture(CaptureSettings(map="Town04",seed=42,weather=weather,timeout=60)) as capture:
            vehicle,world=capture.vehicle,capture.world
            vehicle.set_autopilot(False,capture.settings.traffic_manager_port)
            vehicle.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            spawn,route=straight_route(world.get_map(),vehicle.get_location())
            vehicle.set_transform(spawn);vehicle.set_target_velocity(carla.Vector3D());vehicle.set_target_angular_velocity(carla.Vector3D())
            obstacle=None
            for index in protocol["route_waypoint_indices"]:
                pose=route[index].transform;pose.location.z+=.5
                if obstacle is None:
                    obstacle=world.try_spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"),pose)
                    if obstacle is None:raise RuntimeError("Obstacle spawn failed")
                    capture.actors.append(obstacle);obstacle.set_simulate_physics(False)
                else:obstacle.set_transform(pose)
                capture.sync=FrameSynchronizer();capture.snapshots={};capture.expected_frame=None
                for _ in range(12):capture.next_frame(warmup=True)
                capture.expected_frame=None
                for sample in range(6):
                    record=capture.next_frame();gt=record["depth"];k=camera_intrinsics(640,480)
                    sensor=record["sensor_transform"];location=sensor["location"];yaw=math.radians(sensor["rotation"]["yaw"])
                    delta=obstacle.get_location()
                    expected=(delta.x-location["x"])*math.cos(yaw)+(delta.y-location["y"])*math.sin(yaw)-obstacle.bounding_box.extent.x
                    reference=corridor_clearance(gt,k)
                    if abs(reference-expected)>1:
                        raise RuntimeError("Reference sample is not on the intended stationary obstacle")
                    mask=clearance_sample_mask(gt,k)
                    if mask.sum()<10:raise ValueError("Insufficient matched surface pixels")
                    tensor,_=preprocess_rgb(record["rgb"]);raw=predict(model,tensor)["relative_depth"][0,0].cpu().numpy()
                    native=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR)
                    for label,scale in [("original",protocol["baseline_scale"]),("candidate",mapping["scale"])]:
                        metric=native*scale
                        estimated=corridor_clearance(metric,k)
                        surface_error=metric[mask]-gt[mask]
                        ref_stop=bounded_control(1.6,reference,.1)["brake"]==1
                        model_stop=bounded_control(1.6,estimated,.1)["brake"]==1
                        rows.append(dict(weather=weather,waypoint_index=index,sample=sample,frame=record["frame_id"],model=label,reference_clearance_m=reference,expected_obstacle_plane_m=float(expected),corridor_estimate_m=estimated,corridor_abs_error_m=abs(estimated-reference),matched_surface_mae_m=float(np.abs(surface_error).mean()),matched_surface_signed_error_m=float(surface_error.mean()),missed_brake_trigger=bool(ref_stop and not model_stop)))
                    if sample==0:
                        folder=output/(weather+"_"+str(index));folder.mkdir()
                        from PIL import Image
                        Image.fromarray(record["rgb"]).save(folder/"rgb.png")
                        np.save(folder/"relative_depth.npy",native,allow_pickle=False)
                        np.save(folder/"reference_depth.npy",gt,allow_pickle=False)
                print("Matched obstacle",weather,index,flush=True)
    aggregate={}
    for label in ["original","candidate"]:
        data=[r for r in rows if r["model"]==label]
        mae=float(np.mean([r["matched_surface_mae_m"] for r in data]))
        over=max(r["matched_surface_signed_error_m"] for r in data)
        missed=sum(r["missed_brake_trigger"] for r in data)
        aggregate[label]=dict(samples=len(data),surface_mae_m=mae,maximum_surface_overestimate_m=over,corridor_mae_m=float(np.mean([r["corridor_abs_error_m"] for r in data])),missed_brake_triggers=missed,passes_diagnostic_gate=mae<=2 and over<=2 and missed==0)
    write_json(output/"scores.json",rows)
    summary=dict(status="diagnostic_complete",summary=aggregate,learned_actuation=False,limits="Stationary ego, fixed obstacles, one straight Town04 route, two weather presets. GT region selection is evaluation-only. Counterfactual brakes at synthetic1.6m/s are not closed-loop trials.")
    write_json(output/"summary.json",summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--calibration",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    try:
        with socket.create_connection(("127.0.0.1",2000),timeout=1):raise RuntimeError("Existing CARLA server; do not disturb it")
    except (ConnectionRefusedError,socket.timeout):pass
    log_path=Path(a.output).resolve().with_suffix(".server.log");log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("w",encoding="utf-8") as log:
        server=subprocess.Popen([str(ROOT/".runtime/carla/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"),"-RenderOffScreen","-quality-level=Low","-nosound","-unattended","-nosplash","-stdout"],cwd=ROOT/".runtime/carla",stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            deadline=time.monotonic()+120
            while True:
                if server.poll() is not None:raise RuntimeError("Simulator exited")
                try:create_client("127.0.0.1",2000,3).get_server_version();break
                except RuntimeError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(2)
            run(a.calibration,a.output)
        finally:
            if server.poll() is None:server.terminate();server.wait(timeout=30)


if __name__=="__main__":main()

