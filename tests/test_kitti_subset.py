"""Local ZIP import checks; synthetic PNGs are never model-accuracy evidence."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from PIL import Image

from tools import prepare_kitti_subset as kitti
from src.evaluation import evaluation_manifest, paired_samples


class KittiSubsetTests(unittest.TestCase):
    def setUp(self):
        (kitti.ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(kitti.ROOT / "tmp"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "input.zip"
        self.output = self.root / "dataset"
        self.scene = "2011_09_26_drive_0002_sync"
        self.excluded = "2011_09_26_drive_0005_sync"

    def png(self, mode="depth", size=(1216, 352)):
        buffer = io.BytesIO()
        if mode == "rgb":
            image = Image.new("RGB", size, (20, 40, 60))
        else:
            array = np.full((size[1], size[0]), 512 if mode == "depth" else 2,
                            dtype=np.uint16 if mode == "depth" else np.uint8)
            array[0, 0] = 0
            image = Image.fromarray(array)
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def fixture(self, bad_depth=False, missing=False, bad_geometry=False, bad_intrinsics=False):
        members = {}
        for scene in (self.scene, self.excluded):
            for frame in (5, 11, 17, 23, 29):
                stem = scene + "_image_{:010d}_image_02".format(frame)
                members[kitti.PREFIX + "image/" + stem + ".png"] = self.png("rgb")
                depth = kitti.PREFIX + "groundtruth_depth/" + scene + "_groundtruth_depth_{:010d}_image_02.png".format(frame)
                if not missing or scene != self.scene or frame != 5:
                    members[depth] = self.png("8bit" if bad_depth else "depth", (32, 16) if bad_geometry else (1216, 352))
                members[kitti.PREFIX + "intrinsics/" + stem + ".txt"] = b"nan 0 600 0 700 150 0 0 1" if bad_intrinsics else b"700 0 600 0 700 150 0 0 1"
        members["depth_selection/test_depth_prediction_anonymous/image/0000000000.png"] = b"never read"
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        return members

    def import_fixture(self):
        with patch.object(kitti, "excluded_scenes", return_value={self.excluded}), \
                patch.object(kitti, "checkpoint_hashes", return_value={"encoder.pth": "fixture"}):
            return kitti.prepare(self.archive, self.output)

    def test_selection_conversion_zero_mask_and_real_manifest_contract(self):
        self.fixture()
        original = self.archive.read_bytes()
        manifest = self.import_fixture()
        self.assertEqual([s["frame"] for s in manifest["samples"]], [5, 17, 29])
        self.assertEqual(manifest["split_audit"]["excluded_scenes"], [self.excluded])
        self.assertEqual(manifest["split_audit"]["selected_scenes"], [self.scene])
        path = self.output / "evaluation_manifest.json"
        samples = list(paired_samples(path, evaluation_manifest(path)))
        self.assertEqual(len(samples), 3)
        for _, rgb, depth in samples:
            self.assertEqual(rgb.shape, (352, 1216, 3))
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(depth[0, 0], 0)
            self.assertEqual(depth[1, 1], 2)  # 512 / 256, not Redwood's /1000.
        self.assertEqual(self.archive.read_bytes(), original)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.import_fixture()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_bad_depth_encoding_geometry_or_calibration_cannot_publish_dataset(self):
        for kwargs in ({"bad_depth": True}, {"bad_geometry": True}, {"bad_intrinsics": True}):
            with self.subTest(kwargs=kwargs):
                self.fixture(**kwargs)
                with self.assertRaises(ValueError):
                    self.import_fixture()
                self.assertFalse(self.output.exists())

    def test_missing_pair_and_all_excluded_scenes_are_rejected(self):
        self.fixture(missing=True)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.import_fixture()
        self.fixture()
        with zipfile.ZipFile(self.archive) as archive:
            with self.assertRaisesRegex(ValueError, "No eligible"):
                kitti.select_pairs(archive, {self.scene, self.excluded})

    def test_duplicate_and_unsafe_members_are_rejected_without_extraction(self):
        for name in ("../escape.png", "C:/escape.png", "folder\\escape.png", "duplicate"):
            self.fixture()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(self.archive, "a") as archive:
                    info = zipfile.ZipInfo(name)
                    info.filename = name  # Preserve the unsafe spelling in the ZIP fixture.
                    archive.writestr(info, b"bad")
                    if name == "duplicate":
                        archive.writestr(name, b"again")
            with self.assertRaises(ValueError):
                self.import_fixture()
            self.assertFalse(self.output.exists())
            self.assertFalse((self.root / "escape.png").exists())

    def test_insufficient_frames_rejected_and_pinned_split_edits_detected(self):
        name = kitti.PREFIX + "image/" + self.scene + "_image_0000000005_image_02.png"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(name, b"not reached")
        with self.assertRaisesRegex(ValueError, "three"):
            self.import_fixture()
        split = self.root / "third_party/monodepth2/splits/eigen_zhou"
        split.mkdir(parents=True)
        (split / "train_files.txt").write_text("changed")
        with patch.object(kitti, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "pinned"):
                kitti.excluded_scenes()


if __name__ == "__main__":
    unittest.main()
