"""Bounded RGB-only Monodepth2 adaptation; native reference depth is never loaded."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from src.model import ROOT, UPSTREAM_REVISION, load_model, sha256, upstream_networks
from src.preprocess import HEIGHT, WIDTH, PREPROCESSING_ID, preprocess_rgb

POSE_SOURCE_SHA256 = "e369613794a6e03d4ddb280d9530018c26581aae93eac583cec2771a55e992cf"
POSE_WEIGHT_SHA256 = {"pose_encoder.pth": "bc9e85d241997d945cce6c342c0fe7d003772d13b16a59f77c86b82ff2f40e1f",
                      "pose.pth": "bf8fa3fa475c5cc801fc74e8f6380a1751e940b1698949116f1bfc690e8aad9a"}


def training_layers():
    networks = upstream_networks()
    if sha256(ROOT / "third_party/monodepth2/networks/pose_decoder.py") != POSE_SOURCE_SHA256:
        raise ValueError("Pinned pose decoder source changed")
    import layers
    return networks, layers


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split):
        self.path = Path(manifest).resolve()
        self.fingerprint = sha256(self.path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        if split not in ("train", "validation") or m.get("schema_version") != 1 or m.get("task") != "self_supervised_rgb_triplets" or m.get("preprocessing_id") != PREPROCESSING_ID or m.get("frame_offsets") != [-1, 0, 1]:
            raise ValueError("Need the compatible prepared RGB triplet manifest")
        maps, paths = {}, {}
        for kind in ("train", "validation"):
            rows = m["splits"][kind]
            if not rows:
                raise ValueError("Both temporal splits must be nonempty")
            maps[kind], paths[kind], identities = set(), set(), set()
            for row in rows:
                ids, times = row["frame_ids"], row["timestamps"]
                if len(ids) != 3 or ids != list(range(ids[1]-1, ids[1]+2)) or len(times) != 3 or not np.isfinite(times).all() or not np.allclose(np.diff(times), 0.1, atol=1e-4, rtol=0):
                    raise ValueError("Training requires consecutive 10 Hz triplets")
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["scene"]):
                    raise ValueError("Unsafe training scene identity")
                identity = (row["scene"], ids[1])
                if identity in identities or len(row["rgb"]) != 3:
                    raise ValueError("Duplicate target or malformed RGB triplet")
                identities.add(identity); maps[kind].add(row["map"])
                for relative in row["rgb"]:
                    path = (self.path.parent / relative).resolve()
                    if self.path.parent not in path.parents or path.suffix.lower() != ".png" or not path.is_file() or sha256(path) != m["images"].get(relative):
                        raise ValueError("Missing, changed or unsafe RGB training asset")
                    paths[kind].add(path)
        if maps["train"] & maps["validation"] or paths["train"] & paths["validation"]:
            raise ValueError("Train and validation must have disjoint maps and RGB assets")
        self.rows = m["splits"][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        colors = []
        for relative in row["rgb"]:
            path = self.path.parent / relative
            if sha256(path) != self.manifest["images"][relative]:
                raise ValueError("RGB changed after manifest validation")
            with Image.open(path) as image:
                if image.size != (640, 480) or image.mode != "RGB":
                    raise ValueError("Training requires native 640x480 RGB")
                pixels = np.asarray(image)
            tensor, resized_k = preprocess_rgb(pixels, dict(width=640, height=480, intrinsics=row["intrinsics"]))
            colors.append(tensor[0])
        k = np.eye(4, dtype=np.float32)
        k[:3, :3] = resized_k
        if not np.isfinite(k).all() or np.linalg.det(k[:3,:3]) <= 0:
            raise ValueError("Invalid training intrinsics")
        return dict(rgb=torch.stack(colors), K=torch.from_numpy(k), id=row["scene"]+"_"+str(row["frame_ids"][1]))


def temporal_transforms(encoder, decoder, colors):
    _, layers = training_layers()
    transforms = []
    for source, invert in ((0, True), (2, False)):
        pair = (colors[:,source], colors[:,1]) if invert else (colors[:,1], colors[:,source])
        axis, translation = decoder([encoder(torch.cat(pair, 1))])
        transforms.append(layers.transformation_from_parameters(axis[:,0], translation[:,0], invert=invert))
    return transforms


class TrainableModel(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        networks, _ = training_layers()
        base = load_model(checkpoint, device)
        self.encoder, self.depth = base.encoder, base.decoder
        for name, expected in POSE_WEIGHT_SHA256.items():
            if sha256(Path(checkpoint) / name) != expected:
                raise ValueError("Pinned initial pose checkpoint changed: " + name)
        self.pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose = networks.PoseDecoder(self.pose_encoder.num_ch_enc, 1, num_frames_to_predict_for=2)
        self.pose_encoder.load_state_dict(torch.load(Path(checkpoint)/"pose_encoder.pth", map_location="cpu", weights_only=True), strict=True)
        self.pose.load_state_dict(torch.load(Path(checkpoint)/"pose.pth", map_location="cpu", weights_only=True), strict=True)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        # Batch-one pilot: preserve pretrained running statistics, train affine weights normally.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, colors):
        return self.depth(self.encoder(colors[:,1])), temporal_transforms(self.pose_encoder, self.pose, colors)


class SelfSupervisedLoss(nn.Module):
    def __init__(self, height=HEIGHT, width=WIDTH):
        super().__init__()
        _, layers = training_layers()
        self.backproject = layers.BackprojectDepth(1, height, width)
        self.project = layers.Project3D(1, height, width)
        self.ssim = layers.SSIM()
        self.height, self.width = height, width

    def photometric(self, prediction, target):
        return 0.85*self.ssim(prediction,target).mean(1,True) + 0.15*(prediction-target).abs().mean(1,True)

    def warp(self, source, depth, k, transform):
        points = self.backproject(depth, torch.linalg.inv(k))
        grid = self.project(points, k, transform)
        # Project3D normalizes pixel centers by W-1/H-1.
        return F.grid_sample(source, grid, padding_mode="border", align_corners=True)

    def forward(self, colors, k, disparities, transforms):
        _, layers = training_layers()
        target = colors[:,1]
        if tuple(colors.shape) != (1,3,3,self.height,self.width) or tuple(k.shape) != (1,4,4):
            raise ValueError("The bounded training pilot requires batch one RGB triplets and 4x4 intrinsics")
        identity = torch.cat([self.photometric(colors[:,i], target) for i in (0,2)],1)
        losses, photos, smooths, fractions = [], [], [], []
        reconstruction = None
        for scale in range(4):
            disp = disparities[("disp",scale)]
            full = F.interpolate(disp,(self.height,self.width),mode="bilinear",align_corners=False)
            _, depth = layers.disp_to_depth(full,0.1,100.0)
            warped = [self.warp(colors[:,i],depth,k,t) for i,t in zip((0,2),transforms)]
            reprojection = torch.cat([self.photometric(image,target) for image in warped],1)
            ties = torch.randn_like(identity)*1e-5 if self.training else torch.zeros_like(identity)
            best, chosen = torch.cat((identity+ties,reprojection),1).min(1)
            photo = best.mean()
            color = F.interpolate(target,disp.shape[-2:],mode="bilinear",align_corners=False)
            smooth = layers.get_smooth_loss(disp/(disp.mean((2,3),keepdim=True)+1e-7),color)*1e-3/(2**scale)
            losses.append(photo+smooth);photos.append(photo);smooths.append(smooth);fractions.append((chosen>=2).float().mean())
            if scale == 0:
                reconstruction = warped[0]
        return dict(loss=torch.stack(losses).mean(), photometric=torch.stack(photos).mean(), smoothness=torch.stack(smooths).mean(),
                    warp_fraction=torch.stack(fractions).mean(), reconstruction=reconstruction)
