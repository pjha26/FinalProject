"""Synthetic capture fixtures test triplet plumbing, never simulator acceptance."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import project
from src.preprocess import PREPROCESSING_ID, camera_intrinsics
from tools.prepare_carla_triplets import prepare


class TemporalPreparationTests(unittest.TestCase):
    def setUp(self):
        (project.ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(project.ROOT / "tmp"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def scene(self, name, town, moving=True):
        root = self.root / name
        (root / "rgb").mkdir(parents=True)
        (root / "depth").mkdir()
        meta = dict(schema_version=1, status="complete", preprocessing_id=PREPROCESSING_ID, scene_id=name,
                    settings=dict(width=640, height=480, fov=90, map=town), captured_frames=5,
                    intrinsics=camera_intrinsics(640, 480).tolist(), depth_units="metres")
        project.write_json(root / "metadata.json", meta)
        with (root / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["scene_id","frame_id","simulation_timestamp","rgb_path","depth_path","ego_transform","sensor_transform"])
            writer.writeheader()
            for i in range(1, 6):
                rgb, depth = "rgb/{:08d}.png".format(i), "depth/{:08d}.npy".format(i)
                Image.new("RGB", (640, 480), (i, 2, 3)).save(root / rgb)
                np.save(root / depth, np.ones((480, 640), dtype=np.float32), allow_pickle=False)
                pose = json.dumps(dict(location=dict(x=i * 0.1 if moving else 0, y=0, z=0), rotation=dict(pitch=0,yaw=0,roll=0)))
                writer.writerow(dict(scene_id=name,frame_id=i,simulation_timestamp=i * 0.1,rgb_path=rgb,depth_path=depth,ego_transform=pose,sensor_transform=pose))
        return root

    def test_map_separation_order_calibration_and_no_gt_in_inputs(self):
        train, val = self.scene("train", "Town04"), self.scene("val", "Town05")
        output = self.root / "prepared"
        data = prepare([train], [val], output)
        self.assertEqual([r["frame_ids"] for r in data["splits"]["train"]], [[1,2,3],[2,3,4],[3,4,5]])
        self.assertEqual(len(data["splits"]["validation"]), 3)
        self.assertEqual(len(data["images"]), 10)
        self.assertEqual(data["splits"]["train"][0]["intrinsics"], camera_intrinsics(640,480).tolist())
        self.assertFalse(list(output.rglob("*.npy")))
        for name, digest in data["images"].items():
            self.assertEqual(project.file_hash(output / name), digest)
        with self.assertRaises(FileExistsError):
            prepare([train], [val], output)

    def test_same_map_or_duplicate_scene_cannot_leak_between_splits(self):
        train, val = self.scene("train", "Town04"), self.scene("val", "Town04")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare([train], [val], self.root / "prepared")
        with self.assertRaisesRegex(ValueError, "nonduplicate"):
            prepare([train], [train], self.root / "prepared")

    def test_stationary_partial_and_gapped_captures_cannot_publish_dataset(self):
        train, val = self.scene("train", "Town04", moving=False), self.scene("val", "Town05")
        output=self.root/"prepared"
        with self.assertRaisesRegex(ValueError, "moving"):
            prepare([train], [val], output)
        self.assertFalse(output.exists())
        p=val/"metadata.json"
        meta=json.loads(p.read_text()); meta["status"]="partial"; project.write_json(p,meta)
        with self.assertRaises(ValueError):
            prepare([val], [train], output)
        meta["status"]="complete"; project.write_json(p,meta)
        p=val/"frames.csv"; lines=p.read_text().splitlines(); p.write_text("\n".join(lines[:2]+lines[3:])+"\n")
        with self.assertRaises(ValueError):
            prepare([val], [train], output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
