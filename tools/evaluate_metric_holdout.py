"""One-time evaluation of a frozen metric scale on reserved fresh CARLA routes."""
import argparse
import csv
import json
from pathlib import Path
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
from tools.validate_metric_accuracy import score


def run(freeze_path, output):
    freeze_path=Path(freeze_path).resolve();output=Path(output).resolve()
    frozen=json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint=ROOT/frozen.get("checkpoint_path","checkpoints/baseline_mono_640x192")
    baseline_checkpoint=ROOT/frozen.get("baseline_checkpoint_path","checkpoints/baseline_mono_640x192")
    mapping=ROOT/frozen["candidate_path"]
    if frozen.get("status")!="frozen_before_holdout_acquisition" or sha256(mapping)!=frozen["candidate_sha256"] or checkpoint_hashes(checkpoint)!=frozen["checkpoint_sha256"]:
        raise ValueError("Frozen candidate/model identity mismatch")
    if checkpoint_hashes(baseline_checkpoint)!=frozen.get("baseline_checkpoint_sha256",frozen["checkpoint_sha256"]):
        raise ValueError("Frozen baseline checkpoint changed")
    ledger=freeze_path.parent/"holdout_consumed.json"
    if ledger.exists():
        raise ValueError("This holdout is already consumed; do not use it for tuning/reselection")
    output.mkdir(parents=True,exist_ok=False)
    inventory={}
    for item in frozen["scenes"]:
        scene=ROOT/"data/carla"/item["name"]
        meta=json.loads((scene/"metadata.json").read_text(encoding="utf-8"))
        settings=meta["settings"]
        if meta["status"]!="complete" or meta["captured_frames"]!=100 or settings["map"]!="Town05" or settings["seed"]!=item["seed"] or settings["weather"]!=item["weather"]:
            raise ValueError("Fresh source identity/completeness mismatch")
        inventory[item["name"]]={p.relative_to(scene).as_posix():sha256(p) for p in sorted(scene.rglob("*")) if p.is_file()}
    protocol=dict(frozen_candidate=frozen,source_sha256=inventory,freeze_sha256=sha256(freeze_path),scaling="Frozen constant only; never GT-aligned per image")
    write_json(output/"protocol.json",protocol)
    write_json(ledger,dict(status="consumed_started",protocol_sha256=sha256(output/"protocol.json"),output=str(output)))
    torch.set_num_threads(2);rows={"fixed_scale":[],"candidate":[]}
    complete=False
    try:
        model=load_model(checkpoint,"cuda")
        baseline_model=model if baseline_checkpoint.resolve()==checkpoint.resolve() else load_model(baseline_checkpoint,"cuda")
        for item in frozen["scenes"]:
            scene=ROOT/"data/carla"/item["name"];count=0
            for record in replay_scene(scene):
                tensor,_=preprocess_rgb(record["rgb"],record["camera"])
                raw=predict(model,tensor)["relative_depth"][0,0].cpu().numpy()
                native=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR)
                baseline_raw=raw if baseline_model is model else predict(baseline_model,tensor)["relative_depth"][0,0].cpu().numpy()
                baseline_native=cv2.resize(baseline_raw,(640,480),interpolation=cv2.INTER_LINEAR)
                gt=record["depth"];mask=np.zeros(gt.shape,dtype=bool);mask[96:456,32:608]=True;mask&=(gt>=1)&(gt<=50)
                if mask.sum()<100:raise ValueError("Insufficient holdout GT")
                sample=dict(scene=item["name"],frame=record["frame_id"],prediction=native[mask][None,:],reference=gt[mask][None,:])
                for label,scale in [("fixed_scale",frozen["baseline_scale"]),("candidate",frozen["candidate_scale"])]:
                    values=baseline_native[mask][None,:] if label=="fixed_scale" else sample["prediction"]
                    rows[label].append(score(sample,values*scale))
                folder=output/"predictions"/item["name"];folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/(str(record["frame_id"])+".npy"),native,allow_pickle=False)
                count+=1
            if count!=100:raise ValueError("Holdout frame count changed")
            print("Scored fresh holdout",item["name"],count,flush=True)
        result=paired_promotion(rows["fixed_scale"],rows["candidate"])
        for name,files in inventory.items():
            scene=ROOT/"data/carla"/name
            if set(files)!={p.relative_to(scene).as_posix() for p in scene.rglob("*") if p.is_file()} or any(sha256(scene/p)!=h for p,h in files.items()):
                raise ValueError("Holdout source changed")
        if checkpoint_hashes(checkpoint)!=frozen["checkpoint_sha256"] or checkpoint_hashes(baseline_checkpoint)!=frozen.get("baseline_checkpoint_sha256",frozen["checkpoint_sha256"]):raise ValueError("Checkpoint changed")
        for label,data in rows.items():
            write_json(output/(label+"_scores.json"),data)
            with (output/(label+"_scores.csv")).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        summary=dict(status="holdout_pass" if result["passed"] else "holdout_rejected",samples=len(rows["candidate"]),scenes=len(inventory),selection=result,protocol_sha256=sha256(output/"protocol.json"),candidate_sha256=frozen["candidate_sha256"],learned_actuation=False,limits="Only two new-seed routes on shared Town05, Low rendering, 1-50m crop. No per-frame scaling. Test is now consumed; no reselecting against it. Only the frozen candidate in this protocol is tested; no autonomous-control claim.")
        write_json(output/"summary.json",summary);complete=True
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        write_json(ledger,dict(status="consumed_complete" if complete else "consumed_partial_failure",protocol_sha256=sha256(output/"protocol.json"),scored_frames=len(rows["candidate"]),output=str(output)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--freeze",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.freeze,a.output)

