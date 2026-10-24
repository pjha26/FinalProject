"""Synthetic fixture checks for frozen test-source/model enforcement."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from tests import test_temporal_data as fixtures
from project import write_json
from src.model import sha256
from tools import evaluate_frozen_carla as frozen
from tools.evaluate_kitti_eigen import evaluate as evaluate_eigen

class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TemporalPreparationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root
        self.scene=self.fixture.scene('reserved','Town05');self.protocol=self.root/'protocol.json';self.hashes={'encoder.pth':'fixture','depth.pth':'fixture'}
        write_json(self.protocol,dict(status='frozen_before_test',models={'baseline':self.hashes},scene_files={p.relative_to(self.scene).as_posix():sha256(p) for p in self.scene.rglob('*') if p.is_file()}))
    def test_scores_frozen_complete_scene_with_reference_only_in_scoring(self):
        with patch.object(frozen,'checkpoint_hashes',return_value=self.hashes),patch.object(frozen,'load_model',return_value=object()),patch.object(frozen,'predict',return_value={'inverse_depth':torch.full((1,1,192,640),0.1),'relative_depth':torch.full((1,1,192,640),10.)}):
            rows=frozen.evaluate(self.scene,self.root/'checkpoint',self.protocol,'baseline',self.root/'scores','cpu')
        self.assertEqual(len(rows),5);self.assertTrue(all(row['abs_rel']<1e-6 for row in rows))
    def test_changed_scene_or_unregistered_model_is_rejected_before_inference(self):
        with patch.object(frozen,'checkpoint_hashes',return_value={'encoder.pth':'different'}):
            with self.assertRaisesRegex(ValueError,'registered'):frozen.evaluate(self.scene,self.root/'checkpoint',self.protocol,'baseline',self.root/'bad','cpu')
        with (self.scene/'frames.csv').open('a') as f:f.write('changed')
        with patch.object(frozen,'checkpoint_hashes',return_value=self.hashes):
            with self.assertRaisesRegex(ValueError,'source'):frozen.evaluate(self.scene,self.root/'checkpoint',self.protocol,'baseline',self.root/'bad2','cpu')
        self.assertFalse((self.root/'bad2').exists())
    def test_partial_or_unpinned_eigen_manifest_cannot_claim_full_evaluation(self):
        path=self.root/'incomplete.json';path.write_text(json.dumps({'samples':[]}))
        with self.assertRaisesRegex(ValueError,'697'):evaluate_eigen(path,self.root/'checkpoint',self.root/'bad','cpu')
        self.assertFalse((self.root/'bad').exists())
