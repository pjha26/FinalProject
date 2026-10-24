"""Score a complete CARLA scene only against an already frozen source/model protocol."""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import audit_scene,write_json
from src.carla_capture import replay_scene
from src.demo_data import checkpoint_hashes
from src.model import load_model,predict,sha256
from src.preprocess import preprocess_rgb
from src.evaluation import score_inverse_depth,aggregate_scores
from src.visualize import render_frame


def evaluate(scene,checkpoint,protocol_path,role,output,device='cuda'):
    scene=Path(scene).resolve();checkpoint=Path(checkpoint).resolve();protocol_path=Path(protocol_path).resolve();output=Path(output).resolve()
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status')!='frozen_before_test' or role not in protocol['models'] or checkpoint_hashes(checkpoint)!=protocol['models'][role]:raise ValueError('Model is not registered in frozen test protocol')
    actual={p.relative_to(scene).as_posix() for p in scene.rglob('*') if p.is_file()}
    if set(protocol['scene_files'])!=actual or not {'metadata.json','frames.csv'}<=actual:raise ValueError('Frozen test protocol must cover complete source inventory')
    for relative,digest in protocol['scene_files'].items():
        path=(scene/relative).resolve()
        if scene not in path.parents or sha256(path)!=digest:raise ValueError('Frozen test source changed or escaped scene')
    if output.exists():raise FileExistsError('Use a fresh frozen-test output')
    output.mkdir(parents=True);audit_scene(SimpleNamespace(scene=scene,output=output/'source_audit.json'))
    torch.set_num_threads(2);model=load_model(checkpoint,device);rows=[];complete=False;worst=None
    try:
        with (output/'per_sample.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=None
            for record in replay_scene(scene):
                rgb=record['rgb'];tensor,_=preprocess_rgb(rgb,record['camera']);maps=predict(model,tensor);inverse=maps['inverse_depth'][0,0].cpu().numpy()
                measured=score_inverse_depth(record['depth'],inverse,'none');row=dict(id=str(record['frame_id']),scene=scene.name,**measured)
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush();rows.append(row);np.save(output/(row['id']+'_relative_depth.npy'),maps['relative_depth'][0,0].cpu().numpy(),allow_pickle=False)
                if len(rows)==1:Image.fromarray(render_frame(rgb,inverse,'Frozen reserved test | relative prediction')).save(output/'first_preview.png')
                if worst is None or row['abs_rel']>worst[0]:worst=(row['abs_rel'],rgb.copy(),inverse.copy())
        Image.fromarray(render_frame(worst[1],worst[2],'Frozen reserved test | highest AbsRel | relative prediction')).save(output/'worst_preview.png');complete=True
    finally:
        write_json(output/'summary.json',dict(schema_version=1,status='complete' if complete else 'partial',role=role,scene=scene.name,frames=len(rows),protocol_sha256=sha256(protocol_path),checkpoint_sha256=checkpoint_hashes(checkpoint),mean_metrics=aggregate_scores(rows) if complete else None,total_valid_pixels=sum(r['valid_pixels'] for r in rows),limits='One reserved route, same Town05 map as development; GT-median scoring only, not calibrated output or unseen-map generalization. Do not tune on this test after scoring.'))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--protocol',required=True);p.add_argument('--role',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(a.scene,a.checkpoint,a.protocol,a.role,a.output,a.device)
