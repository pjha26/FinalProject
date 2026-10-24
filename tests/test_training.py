"""Geometry, split integrity and gradient checks for RGB-only training."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from tests import test_temporal_data as fixtures
from tools.prepare_carla_triplets import prepare
from src.preprocess import resize_intrinsics
from src.training import TripletDataset, SelfSupervisedLoss, temporal_transforms


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TemporalPreparationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        a=self.fixture.scene("train","Town04");b=self.fixture.scene("val","Town05")
        prepare([a],[b],self.root/"prepared")
        self.path=self.root/"prepared/triplets.json"

    def test_rgb_only_temporal_order_and_half_pixel_calibration(self):
        dataset=TripletDataset(self.path,"train")
        with patch("numpy.load",side_effect=AssertionError("Training must not load numerical GT")):
            sample=dataset[0]
        self.assertEqual(tuple(sample["rgb"].shape),(3,3,192,640))
        self.assertTrue(torch.all((sample["rgb"]>=0)&(sample["rgb"]<=1)))
        expected=resize_intrinsics(dataset.rows[0]["intrinsics"],(640,480),(640,192))
        np.testing.assert_allclose(sample["K"][:3,:3],expected,rtol=1e-6)
        self.assertAlmostEqual(float(sample["K"][1,2]),95.7,places=4)
        self.assertLess(float(sample["rgb"][0].mean()),float(sample["rgb"][2].mean()))
        relative=dataset.rows[0]["rgb"][0]
        from PIL import Image
        Image.new("RGB",(640,480),(100,100,100)).save(self.path.parent/relative)
        with self.assertRaisesRegex(ValueError,"changed"):
            dataset[0]

    def test_split_leakage_and_time_gaps_are_rejected(self):
        m=json.loads(self.path.read_text(encoding="utf-8"))
        m["splits"]["validation"][0]["map"]="Town04"
        self.path.write_text(json.dumps(m),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"disjoint"):
            TripletDataset(self.path,"train")
        m["splits"]["validation"][0]["map"]="Town05"
        m["splits"]["train"][0]["timestamps"][1]+=0.03
        self.path.write_text(json.dumps(m),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"10 Hz"):
            TripletDataset(self.path,"train")


class TrainingGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        self.loss=SelfSupervisedLoss(32,64).eval()
        self.k=torch.eye(4).unsqueeze(0);self.k[:,0,0]=20;self.k[:,1,1]=20
        self.k[:,0,2]=31.5;self.k[:,1,2]=15.5

    def test_identity_warp_preserves_pixels(self):
        source=torch.rand(1,3,32,64)
        result=self.loss.warp(source,torch.ones(1,1,32,64)*2,self.k,torch.eye(4).unsqueeze(0))
        self.assertTrue(torch.allclose(result,source,atol=1e-5,rtol=1e-5))

    def test_translation_warp_has_expected_direction_and_pixel_shift(self):
        source=torch.arange(64).float().view(1,1,1,64).expand(1,3,32,64)/64
        t=torch.eye(4).unsqueeze(0);t[:,0,3]=0.1
        result=self.loss.warp(source,torch.ones(1,1,32,64)*2,self.k,t)
        self.assertTrue(torch.allclose(result[:,:,:,:-1],source[:,:,:,1:],atol=1e-5,rtol=1e-5))

    def test_temporal_pose_order_and_previous_transform_inversion(self):
        seen=[]
        def encoder(pair):seen.append((float(pair[:,0].mean()),float(pair[:,3].mean())));return pair
        def decoder(features):return torch.zeros(1,2,1,3),torch.tensor([[[[0.2,0.,0.]],[[0.,0.,0.]]]])
        colors=torch.stack([torch.full((1,3,32,64),v) for v in (0.1,0.4,0.9)],1)
        prev,nxt=temporal_transforms(encoder,decoder,colors)
        np.testing.assert_allclose(seen,[(0.1,0.4),(0.4,0.9)],atol=1e-6)
        self.assertAlmostEqual(float(prev[0,0,3]),-0.2,places=6)
        self.assertAlmostEqual(float(nxt[0,0,3]),0.2,places=6)
        self.assertTrue(torch.allclose(prev@nxt,torch.eye(4).unsqueeze(0),atol=1e-6))

    def test_identity_automasking_and_constant_disparity_zero_loss(self):
        image=torch.rand(1,3,32,64)
        colors=torch.stack((image,image,image),1)
        disparities={("disp",s):torch.full((1,1,32//2**s,64//2**s),0.4) for s in range(4)}
        result=self.loss(colors,self.k,disparities,[torch.eye(4).unsqueeze(0)]*2)
        self.assertAlmostEqual(float(result["loss"]),0,places=6)
        self.assertEqual(float(result["warp_fraction"]),0)

    def test_reprojection_propagates_finite_depth_and_pose_gradients(self):
        source=torch.rand(1,3,32,64)
        target=torch.roll(source,-2,dims=3)
        colors=torch.stack((source,target,source),1)
        disp=torch.nn.Parameter(torch.full((1,1,4,8),0.4))
        dx=torch.nn.Parameter(torch.tensor(0.01))
        t=torch.eye(4).unsqueeze(0);t[:,0,3]=dx
        disparities={("disp",s):F.interpolate(disp,(32//2**s,64//2**s),mode="bilinear",align_corners=False) for s in range(4)}
        result=self.loss(colors,self.k,disparities,[t,t])
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        for gradient in (disp.grad,dx.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()),0)
        self.assertGreater(float(result["warp_fraction"]),0)


class EncoderFreezeTests(unittest.TestCase):
    def test_freezing_only_depth_encoder_preserves_other_trainable_paths(self):
        from project import BASELINE
        from src.training import TrainableModel
        model=TrainableModel(BASELINE,"cpu",freeze_encoder=True)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        for module in (model.depth,model.pose_encoder,model.pose):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))
        model.train()
        self.assertTrue(all(not module.training for module in model.modules() if isinstance(module,torch.nn.BatchNorm2d)))
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_default_keeps_depth_encoder_trainable(self):
        from project import BASELINE
        from src.training import TrainableModel
        model=TrainableModel(BASELINE,"cpu")
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))


class TemporalAugmentationTests(unittest.TestCase):
    def test_shared_transform_is_seeded_and_preserves_original_rgb(self):
        from src.training import augment_triplet
        image=torch.linspace(0.1,0.9,3*8*12).reshape(1,3,8,12)
        colors=torch.stack((image,image,image),1)
        original=colors.clone()
        result,parameters=augment_triplet(colors,torch.Generator().manual_seed(0))
        repeated,again=augment_triplet(colors,torch.Generator().manual_seed(0))
        self.assertTrue(parameters['applied'])
        self.assertEqual(set(parameters['order']),{'brightness','contrast','saturation','hue'})
        self.assertTrue(torch.equal(result,repeated));self.assertEqual(parameters,again)
        self.assertTrue(torch.equal(result[:,0],result[:,1]))
        self.assertTrue(torch.equal(result[:,1],result[:,2]))
        self.assertTrue(torch.equal(colors,original))
        self.assertFalse(torch.equal(result,colors))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result>=0)&(result<=1)).all())

    def test_skip_and_invalid_input_boundaries(self):
        from src.training import augment_triplet
        colors=torch.full((1,3,3,8,12),0.4)
        result,parameters=augment_triplet(colors,torch.Generator().manual_seed(42))
        self.assertFalse(parameters['applied']);self.assertTrue(torch.equal(result,colors))
        for bad in (colors.double(),colors[:,0],torch.full_like(colors,float('nan')),colors+2):
            with self.assertRaises(ValueError):augment_triplet(bad,torch.Generator().manual_seed(0))

    def test_augmented_network_input_uses_unaugmented_loss_targets_and_camera(self):
        from src.training import training_objective
        colors=torch.linspace(0.1,0.9,1*3*3*8*12).reshape(1,3,3,8,12)
        original=colors.clone();k=torch.eye(4).unsqueeze(0);before_k=k.clone()
        seen={}
        def model(inputs):seen['input']=inputs.clone();return 'disparities','transforms'
        def objective(targets,camera,disparities,transforms):
            self.assertIs(targets,colors);self.assertIs(camera,k)
            self.assertTrue(torch.equal(targets,original));self.assertTrue(torch.equal(camera,before_k))
            self.assertEqual((disparities,transforms),('disparities','transforms'))
            return {'loss':torch.tensor(0.)}
        result,parameters=training_objective(model,objective,colors,k,torch.Generator().manual_seed(0))
        self.assertTrue(parameters['applied']);self.assertFalse(torch.equal(seen['input'],original))
        self.assertTrue(torch.equal(colors,original));self.assertEqual(float(result['loss']),0.)
        _,disabled=training_objective(model,objective,colors,k)
        self.assertFalse(disabled['applied']);self.assertTrue(torch.equal(seen['input'],original))


if __name__=="__main__":unittest.main()
