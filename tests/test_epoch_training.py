"""Synthetic CPU fixtures verify optimizer/RNG resume, not research accuracy."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from project import ROOT,write_json
from src.model import sha256,UPSTREAM_REVISION
from src.preprocess import PREPROCESSING_ID
from tools import train_carla_epochs as trainer

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self,path,split):
        self.path=Path(path);self.fingerprint=sha256(path);self.split=split;self.rows=list(range(2 if split=='train' else 1))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return dict(rgb=torch.full((3,3,8,12),0.2+i*.1),K=torch.eye(4),id=self.split+'_'+str(i))
class TinyModel(nn.Module):
    def __init__(self,path,device,freeze_encoder=False):
        super().__init__()
        for name in ['encoder','depth','pose_encoder','pose']:setattr(self,name,nn.Linear(1,1,bias=False))
        self.encoder.requires_grad_(not freeze_encoder);self.to(device)
    def forward(self,colors):return {'signal':sum(m.weight.sum() for m in [self.depth,self.pose_encoder,self.pose])*colors[:,1].mean()},[]
class TinyLoss(nn.Module):
    def __init__(self,**kwargs):super().__init__()
    def forward(self,colors,k,disparities,transforms):
        noise=(random.random()+np.random.random()+float(torch.rand(())))*.01 if self.training else 0
        loss=(disparities['signal']-.7+noise).square()
        return dict(loss=loss,photometric=loss,smoothness=loss*0,warp_fraction=loss*0+1)

class EpochResumeTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=str(ROOT/'tmp'));self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.manifest=self.root/'data.json';self.manifest.write_text('{"fixture":true}',encoding='utf-8')
        base=self.root/'baseline';base.mkdir()
        for name in ['encoder.pth','depth.pth','pose_encoder.pth','pose.pth']:torch.save({'weight':torch.tensor([.1])},base/name)
        write_json(base/'manifest.json',dict(preprocessing_id=PREPROCESSING_ID,upstream_revision=UPSTREAM_REVISION,sha256={n:sha256(base/n) for n in ['encoder.pth','depth.pth']}))
        for target,value in [('BASELINE',base),('TripletDataset',TinyDataset),('TrainableModel',TinyModel),('SelfSupervisedLoss',TinyLoss)]:
            p=patch.object(trainer,target,value);p.start();self.addCleanup(p.stop)
    def run_case(self,name,epochs,resume=None,lr=1e-3):return trainer.run(self.manifest,self.root/name,epochs,lr,'cpu',42,True,False,resume)
    def test_epoch_boundary_resume_reproduces_cpu_weights_losses_and_optimizer(self):
        full=self.run_case('full',2);self.run_case('first',1);resumed=self.run_case('resumed',2,self.root/'first/epoch_001')
        self.assertEqual(full['epoch_results'][1]['validation'],resumed['epoch_results'][0]['validation'])
        for name in ['encoder.pth','depth.pth','pose_encoder.pth','pose.pth']:
            a=torch.load(self.root/'full/epoch_002'/name,weights_only=True);b=torch.load(self.root/'resumed/epoch_002'/name,weights_only=True)
            for key in a:
                if torch.is_tensor(a[key]):self.assertTrue(torch.equal(a[key],b[key]))
        a=torch.load(self.root/'full/epoch_002/training_state.pth',weights_only=True);b=torch.load(self.root/'resumed/epoch_002/training_state.pth',weights_only=True)
        for key,state in a['optimizer']['state'].items():
            for name,value in state.items():self.assertTrue(torch.equal(value,b['optimizer']['state'][key][name]))
        self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
    def test_resume_rejects_incompatible_config_and_corrupted_state(self):
        self.run_case('first',1);checkpoint=self.root/'first/epoch_001'
        with self.assertRaisesRegex(ValueError,'incompatible'):self.run_case('bad_lr',2,checkpoint,lr=1e-4)
        self.assertFalse((self.root/'bad_lr').exists())
        with (checkpoint/'training_state.pth').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_case('corrupt',2,checkpoint)
        self.assertFalse((self.root/'corrupt').exists())
    def test_mid_epoch_failure_retains_resumable_last_completed_epoch(self):
        self.run_case('full',2)
        original=trainer.training_objective;calls=[0]
        def interrupted(*args,**kwargs):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('synthetic interruption')
            return original(*args,**kwargs)
        with patch.object(trainer,'training_objective',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):self.run_case('interrupted',2)
        self.assertEqual(json.loads((self.root/'interrupted/summary.json').read_text())['status'],'failed')
        self.assertTrue((self.root/'interrupted/epoch_001/training_state.pth').exists())
        self.assertFalse((self.root/'interrupted/epoch_002').exists())
        self.run_case('resumed',2,self.root/'interrupted/epoch_001')
        a=torch.load(self.root/'full/epoch_002/depth.pth',weights_only=True)
        b=torch.load(self.root/'resumed/epoch_002/depth.pth',weights_only=True)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]))

    def test_existing_output_is_preserved(self):
        folder=self.root/'existing';folder.mkdir();(folder/'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):self.run_case('existing',1)
        self.assertEqual((folder/'keep.txt').read_text(),'keep')

class LossAblationTests(unittest.TestCase):
    def test_l1_and_disabled_smoothness_and_automasking_are_explicit(self):
        from src.training import SelfSupervisedLoss
        objective=SelfSupervisedLoss(32,64,use_ssim=False,automasking=False,smoothness_weight=0).eval()
        a=torch.rand(1,3,32,64);b=torch.rand_like(a)
        self.assertTrue(torch.equal(objective.photometric(a,b),(a-b).abs().mean(1,True)))
        colors=torch.stack([a,b,a],1);k=torch.eye(4).unsqueeze(0)
        disparity={('disp',s):torch.rand(1,1,32//2**s,64//2**s) for s in range(4)}
        result=objective(colors,k,disparity,[torch.eye(4).unsqueeze(0)]*2)
        self.assertEqual(float(result['smoothness']),0);self.assertEqual(float(result['warp_fraction']),1)
        with self.assertRaises(ValueError):SelfSupervisedLoss(smoothness_weight=float('nan'))
