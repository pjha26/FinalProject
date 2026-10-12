"""Checkpoint-compatible wrapper around pinned, locally downloaded Monodepth2."""
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import dataclass

from src.preprocess import HEIGHT, WIDTH, PREPROCESSING_ID

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "b017a8164fae339b0ce01f3a9e5048104d5abd0e"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_networks():
    source = ROOT / "third_party" / "monodepth2"
    manifest_path = source / "provenance.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run 'python project.py assets' to install pinned Monodepth2 source and weights")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != UPSTREAM_REVISION:
        raise ValueError("Unexpected Monodepth2 source revision")
    for relative in ("networks/resnet_encoder.py", "networks/depth_decoder.py", "layers.py"):
        if sha256(source / relative) != manifest["sha256"][relative]:
            raise ValueError("Pinned inference source changed: " + relative)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import networks
    return networks


@dataclass
class DepthModel:
    encoder: object
    decoder: object
    device: object
    checkpoint: Path
    min_depth: float = 0.1
    max_depth: float = 100.0


def load_model(checkpoint_dir, device="auto"):
    import torch

    checkpoint = Path(checkpoint_dir).resolve()
    for filename in ("encoder.pth", "depth.pth"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError("Missing checkpoint file: " + str(checkpoint / filename))
    manifest_path = checkpoint / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preprocessing_id") != PREPROCESSING_ID:
            raise ValueError("Checkpoint preprocessing is incompatible")
        for filename in ("encoder.pth", "depth.pth"):
            if sha256(checkpoint / filename) != manifest["sha256"][filename]:
                raise ValueError("Checkpoint checksum mismatch: " + filename)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for diagnostics")
    networks = upstream_networks()
    encoder_state = torch.load(checkpoint / "encoder.pth", map_location="cpu", weights_only=True)
    if (encoder_state.pop("height", HEIGHT), encoder_state.pop("width", WIDTH)) != (HEIGHT, WIDTH):
        raise ValueError("Phase 1 requires a 640x192 checkpoint")
    if encoder_state.pop("use_stereo", False):
        raise ValueError("Phase 1 requires monocular weights, not stereo weights")
    encoder = networks.ResnetEncoder(18, False)
    encoder.load_state_dict(encoder_state, strict=True)
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(checkpoint / "depth.pth", map_location="cpu", weights_only=True), strict=True)
    return DepthModel(encoder.to(device).eval(), decoder.to(device).eval(), device, checkpoint)


def predict(model, rgb_tensor):
    import torch

    if rgb_tensor.ndim != 4 or tuple(rgb_tensor.shape[1:]) != (3, HEIGHT, WIDTH) or not rgb_tensor.shape[0]:
        raise ValueError("Expected Bx3x192x640 RGB tensor")
    if rgb_tensor.dtype != torch.float32:
        raise ValueError("RGB tensor must be float32 in [0,1]")
    if (~torch.isfinite(rgb_tensor) | (rgb_tensor < 0) | (rgb_tensor > 1)).any().item():
        raise ValueError("RGB tensor contains invalid values")
    with torch.inference_mode():
        sigmoid_disparity = model.decoder(model.encoder(rgb_tensor.to(model.device)))[("disp", 0)]
        inverse_depth = 1 / model.max_depth + (1 / model.min_depth - 1 / model.max_depth) * sigmoid_disparity
        if not torch.isfinite(inverse_depth).all().item():
            raise RuntimeError("Model produced nonfinite inverse depth")
        return {"inverse_depth": inverse_depth, "relative_depth": inverse_depth.reciprocal()}
