"""Full pinned Eigen evaluation: LANCZOS input, Garg crop, GT median scaling, no flip PP."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import write_json
from src.model import load_model,predict,sha256
from src.demo_data import checkpoint_hashes
from src.evaluation import score_inverse_depth,aggregate_scores
from src.visualize import render_frame
from tools.prepare_kitti_eigen import SPLIT_SHA,UTILS_SHA


def evaluate(manifest_path,checkpoint,output,device='cuda'):
    path=Path(manifest_path).resolve();manifest=json.loads(path.read_text(encoding='utf-8'));output=Path(output).resolve()
    if manifest.get('split_sha256')!=SPLIT_SHA or manifest.get('source_utils_sha256')!=UTILS_SHA or len(manifest.get('samples',[]))!=697 or len({s['id'] for s in manifest['samples']})!=697:raise ValueError('Complete pinned Eigen 697 manifest required')
    split=ROOT/'third_party/monodepth2/splits/eigen/test_files.txt'
    if sha256(split)!=SPLIT_SHA:raise ValueError('Pinned Eigen split source changed')
    expected=[drive.split('/')[1]+'_'+str(int(frame)).zfill(10) for drive,frame,side in [line.split() for line in split.read_text().splitlines()]]
    if [sample['id'] for sample in manifest['samples']]!=expected:raise ValueError('Eigen sample identities/order differ from pinned split')
    required=dict(min_depth=0.001,max_depth=80,crop='garg',scaling='per_image_median',post_processing=False)
    if any(manifest.get('protocol',{}).get(k)!=v for k,v in required.items()):raise ValueError('Eigen scoring protocol differs from implementation')
    if output.exists():raise FileExistsError('Use a fresh Eigen output')
    torch.set_num_threads(2);model=load_model(checkpoint,device);output.mkdir(parents=True);rows=[];complete=False;worst=None
    try:
        with (output/'per_sample.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=None
            for sample in manifest['samples']:
                for key in ['rgb','depth']:
                    asset=(path.parent/sample[key]).resolve()
                    if path.parent not in asset.parents or sha256(asset)!=sample[key+'_sha256']:raise ValueError('Eigen source asset/hash mismatch')
                with Image.open(path.parent/sample['rgb']) as image:
                    rgb=np.asarray(image.convert('RGB'));resized=np.asarray(image.convert('RGB').resize((640,192),Image.Resampling.LANCZOS),dtype=np.float32)/255
                gt=np.load(path.parent/sample['depth'],allow_pickle=False)
                if gt.shape!=rgb.shape[:2]:raise ValueError('Eigen reference/RGB mismatch')
                maps=predict(model,torch.from_numpy(resized.transpose(2,0,1).copy()).unsqueeze(0));inverse=maps['inverse_depth'][0,0].cpu().numpy();measured=score_inverse_depth(gt,inverse,'garg');row=dict(id=sample['id'],scene=sample['scene'],**measured)
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush();rows.append(row);np.save(output/(sample['id']+'_inverse_depth.npy'),inverse,allow_pickle=False)
                if len(rows)==1:Image.fromarray(render_frame(rgb,inverse,'KITTI Eigen | relative prediction | scoring-only median alignment')).save(output/'first_preview.png')
                if worst is None or measured['abs_rel']>worst[0]:worst=(measured['abs_rel'],rgb.copy(),inverse.copy())
                if len(rows)%100==0:print('EIGEN SCORED',len(rows),flush=True)
        Image.fromarray(render_frame(worst[1],worst[2],'KITTI Eigen | highest AbsRel | relative prediction')).save(output/'worst_preview.png');complete=True
    finally:
        write_json(output/'summary.json',dict(schema_version=1,status='complete' if complete else 'partial',dataset='KITTI Eigen 697',evaluated_samples=len(rows),expected_samples=697,manifest_sha256=sha256(path),checkpoint_sha256=checkpoint_hashes(checkpoint),protocol=manifest['protocol'],mean_metrics=aggregate_scores(rows) if complete else None,total_valid_pixels=sum(row['valid_pixels'] for row in rows),limits='Pinned full Eigen split with original sparse-LiDAR projection and LANCZOS input; no flip post-processing. Per-image GT median scaling is scoring only. Prior cropped-pilot overlap is reported separately; no test-driven retraining.'))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(a.manifest,a.checkpoint,a.output,a.device)
