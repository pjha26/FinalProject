"""Fit objective-matched scale using hash-verified cached RGB predictions only."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project import write_json
from src.demo_data import checkpoint_hashes
from src.metric_calibration import fit_absrel_scale, paired_promotion
from src.model import sha256
from src.preprocess import PREPROCESSING_ID
from tools.validate_metric_accuracy import score


def run(cache, output):
    cache,output=Path(cache).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((cache/"protocol.json").read_text(encoding="utf-8"))
    current=checkpoint_hashes(ROOT/"checkpoints/baseline_mono_640x192")
    if current!=prior["checkpoint_sha256"] or prior["preprocessing_id"]!=PREPROCESSING_ID:
        raise ValueError("Prediction-cache model or preprocessing mismatch")
    frozen=json.loads((cache/"summary.json").read_text(encoding="utf-8"))
    index={}
    selected={}
    for role,scenes in [("fit",prior["fit_scenes"]),("development",prior["development_scenes"])]:
        selected[role]=[]
        for name in scenes:
            scene=ROOT/"data/carla"/name
            inventory=prior["source_sha256"][name]
            if any(sha256(scene/p)!=h for p,h in inventory.items()):
                raise ValueError("Source data changed")
            with (scene/"frames.csv").open(encoding="utf-8") as stream:
                rows=list(csv.DictReader(stream))
            for row in rows[::prior["stride"]]:
                frame=int(row["frame_id"])
                pred_path=cache/"predictions"/name/(str(frame)+".npy")
                index[str(pred_path)]=sha256(pred_path)
                pred=np.load(pred_path,allow_pickle=False)
                depth_path=(scene/row["depth_path"]).resolve()
                if scene.resolve() not in depth_path.parents:
                    raise ValueError("Reference asset path escaped scene")
                gt=np.load(depth_path,allow_pickle=False)
                if pred.shape!=(480,640) or gt.shape!=pred.shape:
                    raise ValueError("Cached geometry mismatch")
                mask=np.zeros(gt.shape,dtype=bool);mask[96:456,32:608]=True
                mask&=(gt>=1)&(gt<=50)
                selected[role].append(dict(scene=name,frame=frame,prediction=pred[mask][None,:],reference=gt[mask][None,:]))
    protocol=dict(schema_version=1,cache_protocol_sha256=sha256(cache/"protocol.json"),cache_model_sha256=current,prediction_sha256=index,fit_scenes=prior["fit_scenes"],development_scenes=prior["development_scenes"],method="Exact frame-balanced AbsRel-optimal weighted median; all fit pixels",gates=prior["gates"],holdout=prior["reserved_holdout"])
    write_json(output/"protocol.json",protocol)
    scale=fit_absrel_scale([s["prediction"] for s in selected["fit"]],[s["reference"] for s in selected["fit"]])
    mapping=dict(schema_version=1,scale=scale,status="frozen_before_development",method=protocol["method"],checkpoint_sha256=current,preprocessing_id=PREPROCESSING_ID,protocol_sha256=sha256(output/"protocol.json"),camera=dict(width=640,height=480,fov=90,mount_x=1.5,mount_z=1.7),learned_actuation=False)
    write_json(output/"frozen_mapping.json",mapping)
    rows=[score(s,s["prediction"]*scale) for s in selected["development"]]
    baseline=json.loads((cache/"fixed_scale_scores.json").read_text(encoding="utf-8"))
    gate=paired_promotion(baseline,rows)
    write_json(output/"scores.json",rows)
    with (output/"scores.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mapping.update(status="accepted_development" if gate["passed"] else "rejected_development",gate=gate["candidate"],selection=gate,fit_frames=len(selected["fit"]),validation_frames=len(rows))
    write_json(output/"calibration.json",mapping)
    if any(sha256(p)!=h for p,h in index.items()):
        raise ValueError("Cached predictions changed")
    summary=dict(status=mapping["status"],scale=scale,selection=gate,fit_frames=len(selected["fit"]),development_frames=len(rows),cache_integrity_verified=True,learned_actuation=False,limits="Seen development data; postprocessing only; no new holdout scoring or network-weight change.")
    write_json(output/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.cache,a.output)

