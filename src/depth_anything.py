"""Pinned Depth Anything V2 Small adapter; no extra inference dependencies."""
from dataclasses import dataclass, field
import sys
import time

import numpy as np

from src.model import ROOT, sha256
from src.preprocess import camera_intrinsics, resize_intrinsics

REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
WEIGHTS_REVISION = "03876f8651c73a60fe4c2c48294e09fcb6838fcf"
WEIGHTS_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
SOURCE = ROOT / "third_party/depth_anything_v2"
CHECKPOINT = ROOT / "checkpoints/depth_anything_v2_small"
WEIGHTS = "depth_anything_v2_vits.pth"
SOURCE_FILES = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "README.md": "987d1ab991c1b149276a0426bec7d8f3472d85c29ec976d96ea657aa960fad50",
    "depth_anything_v2/dinov2.py": "d22df8d0da44c8525ebed898fd9c33b3acfcda919a84d577cc2a08bfd3d50ac1",
    "depth_anything_v2/dinov2_layers/__init__.py": "e3c28c7b775165b6844000eb69f44d104d6c3803ac7da5d59f6ac3ecfb46d9d8",
    "depth_anything_v2/dinov2_layers/attention.py": "0d2410ce8b8b4499a6b0cdab939ae2bceab8971efd8660d999b0f3088f28cbf7",
    "depth_anything_v2/dinov2_layers/block.py": "093266ea08084c7cce7b8ddd779ecfaaa13055735faa8a7d68b5924f385b395b",
    "depth_anything_v2/dinov2_layers/drop_path.py": "949100aac4d9c19c9d65319dbd362886c5d46a72d7586ba4fd500e9f1d51570c",
    "depth_anything_v2/dinov2_layers/layer_scale.py": "356817b5d1390737128cbcc03a5333fd3060b73beae478772282ccb851d0ef85",
    "depth_anything_v2/dinov2_layers/mlp.py": "c18c5b3f021408597a473bb710070ec3fb3b39f5a2e42e88ae83ced425a608b5",
    "depth_anything_v2/dinov2_layers/patch_embed.py": "b3f82e19926c9cfb63f10136aca5e8a723bde1ad62dae1d04f90d405f9439baf",
    "depth_anything_v2/dinov2_layers/swiglu_ffn.py": "df81d9901240214edc1831ce1b77558dcd6227b29a2f000fc5f546df55d2d09a",
    "depth_anything_v2/dpt.py": "7cc9a29b41d31e1628e5c8b8705e25ae95bc6490f779dbb695297ce2341566f5",
    "depth_anything_v2/util/blocks.py": "351aa24b8af4bef923ebe3bed5e5e8503b08e39bf06ee045d6444a485a2036a8",
    "depth_anything_v2/util/transform.py": "a97df1dbb687457a40c387fdc3482edee7ddc5c142c7d2735ca6235a8ae2885f",
}


def verify_assets():
    for relative, expected in SOURCE_FILES.items():
        if not (SOURCE / relative).is_file() or sha256(SOURCE / relative) != expected:
            raise ValueError("Missing or changed pinned source; run project.py assets --model depth-anything: " + relative)
    if not (CHECKPOINT / WEIGHTS).is_file() or sha256(CHECKPOINT / WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("Missing or changed pinned Depth Anything checkpoint; run project.py assets --model depth-anything")


def install_assets():
    from project import download, write_json
    for relative in SOURCE_FILES:
        download("https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/" + REVISION + "/" + relative, SOURCE / relative)
    download("https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/" + WEIGHTS_REVISION + "/" + WEIGHTS, CHECKPOINT / WEIGHTS)
    verify_assets()
    write_json(SOURCE / "provenance.json", dict(revision=REVISION, sha256=SOURCE_FILES))
    write_json(CHECKPOINT / "manifest.json", dict(model="Depth Anything V2 Small", source_revision=REVISION,
               weights_revision=WEIGHTS_REVISION, sha256={WEIGHTS: WEIGHTS_SHA256}, license="Apache-2.0",
               depth_units="relative inverse depth; not metres"))
    print("Depth Anything V2 Small assets verified; license retained:", SOURCE / "LICENSE")


@dataclass
class SmallModel:
    network: object
    transform: object
    device: object
    input_size: int
    checkpoint: object = CHECKPOINT
    backend: str = "depth-anything"
    input_shape: list = field(default_factory=list)
    zero_pixels: int = 0

    @property
    def encoder(self):
        return self.network.pretrained

    @property
    def decoder(self):
        return self.network.depth_head

    def details(self):
        return dict(backend=self.backend, model_name="Depth Anything V2 Small", model_source_revision=REVISION,
                    checkpoint_sha256={WEIGHTS: WEIGHTS_SHA256}, input_size=self.input_size, model_input=self.input_shape,
                    preprocessing_id="dav2_rgb_imagenet_bicubic_aspect_multiple14_" + str(self.input_size),
                    zero_inverse_depth_floor=1e-6, zero_pixels_in_last_frame=self.zero_pixels)


def load_model(device="auto", input_size=294):
    import cv2
    import torch
    from torchvision.transforms import Compose
    if input_size not in (294, 392, 518):
        raise ValueError("Depth Anything input size must be 294, 392 or 518")
    verify_assets()  # Verify before importing downloaded code or deserializing weights.
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    network = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    network.load_state_dict(torch.load(CHECKPOINT / WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    transform = Compose([Resize(input_size, input_size, resize_target=False, keep_aspect_ratio=True,
                                ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
                         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()])
    return SmallModel(network.to(device).eval(), transform, device, input_size)


def numerical_maps(inverse):
    inverse = np.asarray(inverse, dtype=np.float32)
    if inverse.ndim != 2 or not inverse.size or not np.isfinite(inverse).all() or (inverse < 0).any():
        raise ValueError("Depth Anything returned invalid inverse depth")
    # The upstream ReLU may return zero for sky; floor only to make reciprocals finite.
    inverse = np.maximum(inverse, np.float32(1e-6))
    return dict(inverse_depth=inverse, relative_depth=np.reciprocal(inverse))


def timed_prediction(model, rgb, camera=None):
    import torch
    import torch.nn.functional as functional
    start = time.perf_counter()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Depth Anything expects nonempty HWC uint8 RGB")
    height, width = rgb.shape[:2]
    if max(width, height) * model.input_size > 1024 * min(width, height):
        raise ValueError("Aspect-preserving input would exceed the 1024-pixel long-edge budget; use a smaller input size or a less panoramic image")
    camera = camera or {}
    if camera.get("width", width) != width or camera.get("height", height) != height:
        raise ValueError("Camera metadata dimensions do not match RGB")
    intrinsics = resize_intrinsics(camera.get("intrinsics", camera_intrinsics(width, height)), (width, height), (width, height))
    # Our input is already RGB; upstream image2tensor expects BGR and auto-selects its device.
    pixels = model.transform({"image": rgb / 255.0})["image"]
    tensor = torch.from_numpy(pixels).unsqueeze(0).to(model.device)
    model.input_shape = list(tensor.shape[-2:])
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    model_start = time.perf_counter()
    with torch.inference_mode():
        inverse = model.network(tensor)
        inverse = functional.interpolate(inverse[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    model_ms = (time.perf_counter() - model_start) * 1000
    raw = inverse.cpu().numpy()
    model.zero_pixels = int((raw == 0).sum())
    return numerical_maps(raw), intrinsics, model_ms, (time.perf_counter() - start) * 1000
