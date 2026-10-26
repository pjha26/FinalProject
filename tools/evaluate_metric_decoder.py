"""Evaluate a frozen metric network candidate; development data only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import write_json
from src.carla_capture import replay_scene
from src.demo_data import checkpoint_hashes
from src.metric_calibration import paired_promotion
from src.model import load_model,predict,sha256
from src.preprocess import preprocess_rgb
from tools.calibrate_metric_scale import VALIDATION
from tools.validate_metric_accuracy import score
from tools.carla_control_demo import load_calibration


def run(checkpoint,calibration,output):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    mapping=load_calibration(calibration,checkpoint)
    output.mkdir(parents=True,exist_ok=False)
    source=json.loads((ROOT/"runs/metric_accuracy_02/protocol.json").read_text(encoding="utf-8"))
    protocol=dict(checkpoint_sha256=checkpoint_hashes(checkpoint),calibration_sha256=sha256(calibration),scale=mapping["scale"],scenes=VALIDATION,stride=5,pixel_gates=dict(mean_abs_rel=.25,p95_frame_abs_rel=.5,relative_improvement=.05,scene_ratio=1.10),kitti24_max_absrel_ratio=1.05,full_eigen_used=False,learned_actuation=False)
    write_json(output/"protocol.json",protocol)
    torch.set_num_threads(2);model=load_model(checkpoint,"cuda");rows=[]
    for name in VALIDATION:
        scene=ROOT/"data/carla"/name
        if any(sha256(scene/p)!=h for p,h in source["source_sha256"][name].items()):raise ValueError("Development data changed")
        for index,r in enumerate(replay_scene(scene)):
            if index%5:continue
            tensor,_=preprocess_rgb(r["rgb"],r["camera"])
            raw=predict(model,tensor)["relative_depth"][0,0].cpu().numpy()
            native=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR)
            gt=r["depth"];mask=np.zeros(gt.shape,dtype=bool);mask[96:456,32:608]=True;mask&=(gt>=1)&(gt<=50)
            sample=dict(scene=name,frame=r["frame_id"],reference=gt[mask][None,:])
            rows.append(score(sample,native[mask][None,:]*mapping["scale"]))
        print("Metric candidate development",name,flush=True)
    baseline=json.loads((ROOT/"runs/metric_absrel_scale_01/scores.json").read_text(encoding="utf-8"))
    pixel=paired_promotion(baseline,rows)
    write_json(output/"metric_scores.json",rows)
    dataset=ROOT/"data/evaluation/kitti_pilot_01"
    manifest_path=dataset/"evaluation_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"]=protocol["checkpoint_sha256"]
    manifest["comparison_reference_manifest_sha256"]=sha256(manifest_path)
    manifest["limitations"]="KITTI24 development only; supervised CARLA metric candidate; no final Eigen use or default promotion."
    candidate_manifest=dataset/(output.name+"_manifest.json")
    if candidate_manifest.exists():raise FileExistsError("Use a new validation name")
    write_json(candidate_manifest,manifest)
    with (output/"kitti.log").open("w",encoding="utf-8") as stream:
        done=subprocess.run([sys.executable,"project.py","evaluate","--manifest",str(candidate_manifest),"--checkpoint",str(checkpoint),"--device","cuda","--output",str(output/"kitti24")],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    if done.returncode:raise RuntimeError("KITTI development evaluation failed")
    old=json.loads((ROOT/"runs/kitti_baseline_01/summary.json").read_text(encoding="utf-8"))["mean_metrics"]
    new=json.loads((output/"kitti24/summary.json").read_text(encoding="utf-8"))["mean_metrics"]
    guard=new["abs_rel"]<=old["abs_rel"]*1.05
    eligible=pixel["passed"] and guard
    result=dict(status="eligible_for_fresh_test" if eligible else "development_rejected",metric_selection=pixel,kitti24_baseline=old,kitti24_candidate=new,kitti_guard_passed=guard,checkpoint_sha256=protocol["checkpoint_sha256"],network_supervision="Explicit supervised metric CARLA fine-tuning",learned_actuation=False,limits="Seen development scenes; exploratory128-update model, not convergence. Do not use previous seed47/48 holdout for this new candidate.")
    if checkpoint_hashes(checkpoint)!=protocol["checkpoint_sha256"]:raise ValueError("Candidate changed during evaluation")
    write_json(output/"summary.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--calibration",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.checkpoint,a.calibration,a.output)

