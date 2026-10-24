"""Bounded epoch training with verified epoch-boundary optimizer/RNG resume."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project import BASELINE,write_json
from src.demo_data import checkpoint_hashes
from src.model import sha256,UPSTREAM_REVISION
from src.preprocess import PREPROCESSING_ID
from src.training import TripletDataset,TrainableModel,SelfSupervisedLoss,training_objective,POSE_WEIGHT_SHA256
from tools.train_carla_pilot import scalars,validate


def rng_state(order,augmentation):
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=n[0],keys=torch.tensor(n[1].astype(np.int64)),position=n[2],gaussian=n[3],cached=n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],order=order.get_state(),augmentation=augmentation.get_state())


def restore_rng(state,order,augmentation):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['name'],n['keys'].numpy().astype(np.uint32),n['position'],n['gaussian'],n['cached']))
    torch.set_rng_state(state['torch']);order.set_state(state['order']);augmentation.set_state(state['augmentation'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_epoch(folder,model,optimizer,config,epoch,order,augmentation,initial):
    with tempfile.TemporaryDirectory(dir=str(folder.parent)) as temporary:
        staged=Path(temporary)/'checkpoint';staged.mkdir()
        states={'encoder.pth':dict(model.encoder.state_dict(),height=192,width=640,use_stereo=False),'depth.pth':model.depth.state_dict(),'pose_encoder.pth':model.pose_encoder.state_dict(),'pose.pth':model.pose.state_dict()}
        for name,state in states.items():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise RuntimeError('Nonfinite checkpoint')
            torch.save(state,staged/name)
        torch.save(dict(schema_version=1,epoch_completed=epoch,config=config,optimizer=optimizer.state_dict(),rng=rng_state(order,augmentation)),staged/'training_state.pth')
        hashes={name:sha256(staged/name) for name in list(states)+['training_state.pth']}
        write_json(staged/'manifest.json',dict(schema_version=1,model='monodepth2_carla_epoch_candidate',modality='monocular',preprocessing_id=PREPROCESSING_ID,upstream_revision=UPSTREAM_REVISION,sha256=hashes,training_manifest_sha256=config['manifest_sha256'],configuration=config,epoch_completed=epoch,parent_checkpoint_sha256=initial,scope='Experimental epoch checkpoint; no automatic default promotion'))
        staged.replace(folder)


def resume_epoch(folder,model,optimizer,config,order,augmentation,epochs):
    folder=Path(folder).resolve();meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('configuration')!=config or meta.get('preprocessing_id')!=PREPROCESSING_ID or meta.get('upstream_revision')!=UPSTREAM_REVISION:raise ValueError('Resume configuration/manifest is incompatible')
    for name in ['encoder.pth','depth.pth','pose_encoder.pth','pose.pth','training_state.pth']:
        if sha256(folder/name)!=meta['sha256'].get(name):raise ValueError('Resume checksum mismatch: '+name)
    state=torch.load(folder/'training_state.pth',map_location='cpu',weights_only=True)
    if state.get('schema_version')!=1 or state['config']!=config or state['epoch_completed']!=meta['epoch_completed'] or epochs<=state['epoch_completed']:raise ValueError('Resume epoch/configuration is invalid')
    for module,name in [(model.encoder,'encoder.pth'),(model.depth,'depth.pth'),(model.pose_encoder,'pose_encoder.pth'),(model.pose,'pose.pth')]:
        values=torch.load(folder/name,map_location='cpu',weights_only=True)
        for key in ['height','width','use_stereo']:values.pop(key,None)
        module.load_state_dict(values,strict=True)
    optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],order,augmentation)
    return state['epoch_completed']


def run(manifest,output,epochs=2,learning_rate=1e-6,device='cuda',seed=42,freeze_encoder=True,color_augmentation=True,resume=None,no_ssim=False,no_automasking=False,no_smoothness=False):
    if epochs<=0 or not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Positive epochs and finite learning rate required')
    output=Path(output).resolve()
    if output.exists() or BASELINE.resolve() in output.parents or output in BASELINE.resolve().parents:raise FileExistsError('Use a fresh experiment outside accepted weights')
    train=TripletDataset(manifest,'train');val=TripletDataset(manifest,'validation')
    initial=checkpoint_hashes(BASELINE);initial.update({name:sha256(BASELINE/name) for name in POSE_WEIGHT_SHA256})
    config=dict(manifest_sha256=train.fingerprint,learning_rate=learning_rate,device=str(device),seed=seed,batch_size=1,freeze_encoder=freeze_encoder,color_augmentation=color_augmentation,no_ssim=no_ssim,no_automasking=no_automasking,no_smoothness=no_smoothness)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    device=torch.device(device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
    model=TrainableModel(BASELINE,device,freeze_encoder=freeze_encoder)
    objective=SelfSupervisedLoss(use_ssim=not no_ssim,automasking=not no_automasking,smoothness_weight=0 if no_smoothness else 1e-3).to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    order=torch.Generator().manual_seed(seed);augmentation=torch.Generator().manual_seed(seed)
    completed=resume_epoch(resume,model,optimizer,config,order,augmentation,epochs) if resume else 0
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema_version=1,status='running',configuration=config,epochs_requested=epochs,epochs_completed=completed,started_from_epoch=completed,train_triplets=len(train),validation_triplets=len(val),initial_checkpoint_sha256=initial,resume_from=str(Path(resume).resolve()) if resume else None,epoch_results=[],limitations='Small correlated CARLA dataset. Epoch-boundary resume only; failed mid-epoch work restarts from last completed checkpoint. CUDA sampling backward may be nondeterministic. No automatic checkpoint selection/default promotion.')
    start=time.perf_counter();write_json(output/'summary.json',summary)
    try:
        for epoch in range(completed+1,epochs+1):
            model.train();objective.train();losses=[];targets=[]
            with (output/('epoch_{:03d}.csv'.format(epoch))).open('w',newline='',encoding='utf-8') as stream:
                writer=None
                for step,batch in enumerate(torch.utils.data.DataLoader(train,batch_size=1,shuffle=True,num_workers=0,generator=order),1):
                    optimizer.zero_grad(set_to_none=True);colors,k=batch['rgb'].to(device),batch['K'].to(device)
                    result,parameters=training_objective(model,objective,colors,k,augmentation if color_augmentation else None)
                    if not torch.isfinite(result['loss']):raise RuntimeError('Nonfinite training objective')
                    result['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                    optimizer.step();row=dict(epoch=epoch,step=step,target=batch['id'][0],augmentation=json.dumps(parameters,sort_keys=True),gradient_norm=float(norm),**scalars(result))
                    if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);stream.flush();losses.append(row['loss']);targets.append(row['target'])
            if len(targets)!=len(train) or len(set(targets))!=len(train):raise RuntimeError('Epoch must visit each target exactly once')
            measured=validate(model,objective,val,device)
            if sha256(train.path)!=train.fingerprint or any(sha256(BASELINE/name)!=value for name,value in initial.items()):raise ValueError('Manifest or accepted baseline changed')
            checkpoint=output/('epoch_{:03d}'.format(epoch));save_epoch(checkpoint,model,optimizer,config,epoch,order,augmentation,initial)
            summary['epochs_completed']=epoch;summary['checkpoint']=checkpoint.name
            summary['epoch_results'].append(dict(epoch=epoch,updates=len(targets),unique_targets=len(set(targets)),mean_train_loss=float(np.mean(losses)),validation=measured,checkpoint=checkpoint.name))
            write_json(output/'summary.json',summary);print('EPOCH',epoch,'updates',len(targets),'validation',measured['loss'],flush=True)
        summary.update(status='complete',peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    except BaseException as error:summary.update(status='failed',error=str(error));raise
    finally:summary['elapsed_seconds']=time.perf_counter()-start;write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=2);p.add_argument('--learning-rate',type=float,default=1e-6);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--seed',type=int,default=42);p.add_argument('--unfreeze-encoder',action='store_true');p.add_argument('--no-color-augmentation',action='store_true');p.add_argument('--resume');p.add_argument('--no-ssim',action='store_true');p.add_argument('--no-automasking',action='store_true');p.add_argument('--no-smoothness',action='store_true')
    a=p.parse_args();run(a.manifest,a.output,a.epochs,a.learning_rate,a.device,a.seed,not a.unfreeze_encoder,not a.no_color_augmentation,a.resume,a.no_ssim,a.no_automasking,a.no_smoothness)
