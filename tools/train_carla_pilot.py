"""Run a batch-one self-supervised sanity experiment; never replace accepted weights."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from project import BASELINE, write_json
from src.demo_data import checkpoint_hashes
from src.model import UPSTREAM_REVISION, sha256
from src.preprocess import PREPROCESSING_ID
from src.training import POSE_WEIGHT_SHA256, POSE_SOURCE_SHA256, TripletDataset, TrainableModel, SelfSupervisedLoss


def scalars(result):
    return {key:float(result[key].detach()) for key in ("loss","photometric","smoothness","warp_fraction")}


def validate(model, objective, dataset, device):
    model.eval();objective.eval()
    rows=[]
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0):
            colors,k=batch["rgb"].to(device),batch["K"].to(device)
            disparities,transforms=model(colors)
            row=scalars(objective(colors,k,disparities,transforms))
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError("Nonfinite validation objective")
            rows.append(row)
    return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run(manifest, output, steps=10, learning_rate=1e-5, device="cuda", seed=42):
    if not isinstance(steps,int) or steps<=0 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("Need positive steps and finite positive learning rate")
    output=Path(output).resolve()
    if output.exists() or BASELINE.resolve() in output.parents or output in BASELINE.resolve().parents:
        raise FileExistsError("Use a fresh experiment directory outside the baseline checkpoint")
    train=TripletDataset(manifest,"train");val=TripletDataset(manifest,"validation")
    initial=checkpoint_hashes(BASELINE)
    initial.update({name:sha256(BASELINE/name) for name in POSE_WEIGHT_SHA256})
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    device=torch.device(device)
    if device.type=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
    model=TrainableModel(BASELINE,device)
    objective=SelfSupervisedLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    order=torch.Generator().manual_seed(seed)
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    record=dict(schema_version=1,status="running",started_utc=datetime.now(timezone.utc).isoformat(),device=str(device),
                steps_requested=steps,steps_completed=0,seed=seed,batch_size=1,learning_rate=learning_rate,
                manifest_sha256=train.fingerprint,train_triplets=len(train),validation_triplets=len(val),
                initial_checkpoint_sha256=initial,pose_source_sha256=POSE_SOURCE_SHA256,
                objective="Four-scale minimum reprojection; 0.85 SSIM + 0.15 L1; identity automasking; normalized edge-aware smoothness 1e-3 / 2**scale",
                adaptations="Batch-one pilot; frozen BatchNorm running statistics; half-pixel resize intrinsics; align_corners=True projection; no augmentation, LR schedule or resume",
                reproducibility="Seeded sampling/model/tie noise; CUDA grid_sample backward may remain nondeterministic",
                limitations="Training sanity check on 98 correlated Town04 triplets, not converged training, generalization, official benchmark or calibrated depth evidence.")
    write_json(output/"summary.json",record)
    try:
        record["validation_before"]=validate(model,objective,val,device)
        loader=torch.utils.data.DataLoader(train,batch_size=1,shuffle=True,num_workers=0,generator=order)
        iterator=iter(loader)
        with (output/"training.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=None
            for step in range(steps):
                try:batch=next(iterator)
                except StopIteration:iterator=iter(loader);batch=next(iterator)
                model.train();objective.train();optimizer.zero_grad(set_to_none=True)
                colors,k=batch["rgb"].to(device),batch["K"].to(device)
                disparities,transforms=model(colors)
                result=objective(colors,k,disparities,transforms)
                if not torch.isfinite(result["loss"]):raise RuntimeError("Nonfinite training objective")
                result["loss"].backward()
                norms={}
                for name,module in (("depth",model.depth),("pose",model.pose)):
                    grads=[p.grad for p in module.parameters() if p.grad is not None]
                    if not grads or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError("Missing/nonfinite "+name+" gradients")
                    norms[name+"_grad_norm"]=float(torch.sqrt(sum(g.square().sum() for g in grads)))
                    if norms[name+"_grad_norm"]<=0:raise RuntimeError("Zero "+name+" gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True)
                optimizer.step()
                row=dict(step=step+1,target=batch["id"][0],**scalars(result),**norms)
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush()
                record["steps_completed"]=step+1
                print("STEP",step+1,"loss",row["loss"],flush=True)
        record["validation_after"]=validate(model,objective,val,device)
        if sha256(train.path)!=train.fingerprint or checkpoint_hashes(BASELINE)!={k:initial[k] for k in ("encoder.pth","depth.pth")} or any(sha256(BASELINE/k)!=initial[k] for k in POSE_WEIGHT_SHA256):
            raise ValueError("Input manifest or accepted baseline changed during experiment")
        model.eval();objective.eval()
        batch=next(iter(torch.utils.data.DataLoader(val,batch_size=1,shuffle=False)))
        with torch.no_grad():
            colors,k=batch["rgb"].to(device),batch["K"].to(device)
            disparities,transforms=model(colors)
            result=objective(colors,k,disparities,transforms)
            np.save(output/"validation_reconstruction.npy",result["reconstruction"].cpu().numpy(),allow_pickle=False)
            for label,tensor in (("target",colors[:,1]),("reconstruction",result["reconstruction"])):
                pixels=(tensor[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
                Image.fromarray(pixels).save(output/("validation_"+label+".png"))
        with tempfile.TemporaryDirectory(dir=str(output)) as temporary:
            staged=Path(temporary)/"checkpoint";staged.mkdir()
            encoder=dict(model.encoder.state_dict(),height=192,width=640,use_stereo=False)
            states={"encoder.pth":encoder,"depth.pth":model.depth.state_dict(),"pose_encoder.pth":model.pose_encoder.state_dict(),"pose.pth":model.pose.state_dict()}
            for name,state in states.items():
                if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise RuntimeError("Nonfinite candidate checkpoint")
                torch.save(state,staged/name)
            hashes={name:sha256(staged/name) for name in states}
            write_json(staged/"manifest.json",dict(schema_version=1,model="monodepth2_carla_sanity_candidate",modality="monocular",
                       preprocessing_id=PREPROCESSING_ID,upstream_revision=UPSTREAM_REVISION,sha256=hashes,
                       parent_checkpoint_sha256=initial,training_manifest_sha256=train.fingerprint,training_steps=steps,
                       scope="Experimental sanity checkpoint; not selected for deployment or an accuracy improvement"))
            staged.replace(output/"checkpoint")
        record.update(status="sanity_complete",candidate_checkpoint_sha256=hashes,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
    except BaseException as error:
        record.update(status="failed",error=str(error))
        raise
    finally:
        record["elapsed_seconds"]=time.perf_counter()-started
        write_json(output/"summary.json",record)
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=10)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    run(args.manifest,args.output,args.steps,args.learning_rate,args.device)
