"""One bounded supervised CARLA metric-decoder trial; never replaces accepted weights."""
import argparse
import copy
import csv
import json
import random
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import write_json
from src.carla_capture import replay_scene
from src.demo_data import checkpoint_hashes
from src.metric_control import corridor_clearance,_corridor_geometry
from src.metric_training import metric_supervision_loss
from src.model import load_model,sha256,UPSTREAM_REVISION
from src.preprocess import preprocess_rgb,camera_intrinsics,PREPROCESSING_ID


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    base=ROOT/"checkpoints/baseline_mono_640x192";original=checkpoint_hashes(base)
    data=[]
    for name in ["train_town04_09","train_town04_14"]:
        for i,r in enumerate(replay_scene(ROOT/"data/carla"/name)):
            if i%5:continue
            data.append(dict(id=name+"/"+str(r["frame_id"]),rgb=r["rgb"],gt=r["depth"],object=False))
    for folder in sorted((ROOT/"runs/metric_obstacle_diagnostic_01").iterdir()):
        if folder.is_dir() and (folder/"rgb.png").exists():
            data.append(dict(id=str(folder.relative_to(ROOT)),rgb=np.asarray(Image.open(folder/"rgb.png").convert("RGB")),gt=np.load(folder/"reference_depth.npy",allow_pickle=False),object=True))
    roads=[s for s in data if not s["object"]];objects=[s for s in data if s["object"]]
    if len(roads)!=40 or len(objects)!=8:raise ValueError("Exact48-frame supervised fit set required")
    dataset_hash={}
    for sample in data:
        dataset_hash[sample["id"]]=dict(rgb=__import__("hashlib").sha256(sample["rgb"].tobytes()).hexdigest(),depth=__import__("hashlib").sha256(sample["gt"].tobytes()).hexdigest())
    config=dict(schema_version=1,supervision="CARLA GT metric depth, not temporal self-supervision",steps=128,learning_rate=1e-5,seed=42,scale=14.996536494241578,freeze_encoder=True,roads=40,obstacle_images=8,loss="logL1metric +0.5 objectlogL1 +0.1 frozen-teacher consistency",dataset_sha256=dataset_hash,parent_checkpoint_sha256=original,preprocessing_id=PREPROCESSING_ID,selection="Final128th update, no checkpoint search",learned_actuation=False)
    write_json(output/"protocol.json",config)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    random.shuffle(roads);random.shuffle(objects)
    model=load_model(base,"cuda");model.encoder.eval();model.decoder.train()
    for p in model.encoder.parameters():p.requires_grad_(False)
    frozen_encoder={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    teacher=copy.deepcopy(model.decoder).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.decoder.parameters(),lr=1e-5)
    rows=[];started=time.perf_counter();complete=False
    try:
        for step in range(128):
            s=objects[(step//2)%len(objects)] if step%2==0 else roads[(step//2)%len(roads)]
            rgb=s["rgb"];gt=s["gt"];tensor,_=preprocess_rgb(rgb);tensor=tensor.to(model.device)
            target=cv2.resize(gt,(640,192),interpolation=cv2.INTER_NEAREST)
            valid=np.zeros(target.shape,dtype=bool);valid[38:182,32:608]=True;valid&=(target>=1)&(target<=50)
            obj=np.zeros(gt.shape,dtype=bool)
            if s["object"]:
                _,obj=_corridor_geometry(gt,camera_intrinsics(640,480));obj&=gt<corridor_clearance(gt,camera_intrinsics(640,480))+.75
            obj=cv2.resize(obj.astype(np.uint8),(640,192),interpolation=cv2.INTER_NEAREST).astype(bool)
            target=torch.from_numpy(target)[None,None].to(model.device)
            valid=torch.from_numpy(valid)[None,None].to(model.device)
            obj=torch.from_numpy(obj)[None,None].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features=model.encoder(tensor)
                teacher_disp=teacher(features)[("disp",0)]
                teacher_inverse=.01+9.99*teacher_disp
            disparity=model.decoder(features)[("disp",0)];inverse=.01+9.99*disparity
            objective,parts=metric_supervision_loss(inverse,target,valid,obj,teacher_inverse,config["scale"])
            if not torch.isfinite(objective):raise RuntimeError("Nonfinite supervised loss")
            objective.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.0)
            if not torch.isfinite(grad) or grad<=0:raise RuntimeError("Invalid supervised gradient")
            optimizer.step()
            row=dict(step=step+1,sample=s["id"],loss=float(objective.detach()),gradient_norm=float(grad),**{k:float(v.detach()) for k,v in parts.items()})
            rows.append(row)
            if (step+1)%32==0:print("Supervised metric update",step+1,row["loss"],flush=True)
        if not all(torch.equal(v.detach().cpu(),frozen_encoder[k]) for k,v in model.encoder.state_dict().items()):raise RuntimeError("Frozen encoder changed")
        if checkpoint_hashes(base)!=original:raise RuntimeError("Accepted checkpoint changed")
        candidate=output/"checkpoint";candidate.mkdir()
        states={"encoder.pth":dict(model.encoder.state_dict(),height=192,width=640,use_stereo=False),"depth.pth":model.decoder.state_dict()}
        for name,state in states.items():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise RuntimeError("Nonfinite candidate")
            torch.save(state,candidate/name)
        hashes={name:sha256(candidate/name) for name in states}
        write_json(candidate/"manifest.json",dict(schema_version=1,preprocessing_id=PREPROCESSING_ID,upstream_revision=UPSTREAM_REVISION,sha256=hashes,scope="Experimental supervised metric CARLA decoder; not accepted default",protocol_sha256=sha256(output/"protocol.json")))
        write_json(output/"calibration.json",dict(schema_version=1,scale=config["scale"],checkpoint_sha256=hashes,preprocessing_id=PREPROCESSING_ID,camera=dict(width=640,height=480,fov=90,mount_x=1.5,mount_z=1.7),status="unvalidated_supervised_candidate",gate=dict(passed=False),learned_actuation=False))
        complete=True
    finally:
        write_json(output/"training.json",rows)
        write_json(output/"summary.json",dict(status="completed" if complete else "failed",updates=len(rows),seconds=time.perf_counter()-started,frozen_encoder_unchanged=complete,baseline_bytes_unchanged=checkpoint_hashes(base)==original,supervision=config["supervision"],learned_actuation=False))
    print("Bounded metric training complete",output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);a=p.parse_args();run(a.output)


