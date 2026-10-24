"""Synthetic fixtures verify split/hash contracts, not simulator/model accuracy."""
import json
import unittest
from unittest.mock import patch
from tests import test_temporal_data as fixtures
from project import file_hash, write_json
from tools.prepare_carla_triplets import prepare as prepare_triplets
from tools.prepare_carla_evaluation import prepare
from src.evaluation import evaluation_manifest, paired_samples


class CarlaEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TemporalPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.train = self.fixture.scene("train", "Town04")
        self.val = self.fixture.scene("val", "Town05")
        prepare_triplets([self.train], [self.val], self.root / "triplets")
        self.triplets = self.root / "triplets/triplets.json"
        self.hashes = patch("tools.prepare_carla_evaluation.checkpoint_hashes", return_value={"encoder.pth":"fixture", "depth.pth":"fixture"})
        self.hashes.start()
        self.addCleanup(self.hashes.stop)

    def test_freezes_only_validation_targets_and_preserves_native_gt(self):
        out = self.root / "evaluation"
        result = prepare(self.triplets, self.val, out)
        self.assertEqual(result["data_kind"], "simulated_rgbd")
        self.assertEqual([s["id"] for s in result["samples"]], ["val_00000002", "val_00000003", "val_00000004"])
        for sample in result["samples"]:
            frame = int(sample["id"].rsplit("_",1)[1])
            self.assertEqual(file_hash(out / sample["depth"]), file_hash(self.val / "depth/{:08d}.npy".format(frame)))
        self.assertEqual(len(list(paired_samples(out / "evaluation_manifest.json", evaluation_manifest(out / "evaluation_manifest.json")))), 3)
        self.assertFalse(list((self.root / "triplets").rglob("*.npy")))
        with self.assertRaises(FileExistsError):
            prepare(self.triplets, self.val, out)

    def test_rejects_training_scene_and_changed_source_manifest(self):
        out = self.root / "evaluation"
        with self.assertRaisesRegex(ValueError,"map-disjoint"):
            prepare(self.triplets, self.train, out)
        self.assertFalse(out.exists())
        p=self.val / "metadata.json";meta=json.loads(p.read_text());meta["seed"]=999;write_json(p,meta)
        with self.assertRaisesRegex(ValueError,"hashes"):
            prepare(self.triplets,self.val,out)
        self.assertFalse(out.exists())

    def test_rgb_mutation_cannot_publish_evaluation(self):
        p=self.val / "rgb/00000002.png"
        from PIL import Image
        Image.new("RGB",(640,480),(90,80,70)).save(p)
        out=self.root / "evaluation"
        with self.assertRaisesRegex(ValueError,"RGB differs"):
            prepare(self.triplets,self.val,out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
