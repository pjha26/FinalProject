"""Shared RGB and camera geometry; normalization belongs to Monodepth2."""
import numpy as np
from PIL import Image

HEIGHT, WIDTH = 192, 640
PREPROCESSING_ID = "rgb_bilinear_192x640_monodepth2_internal_v1"


def camera_intrinsics(width, height, fov=90.0):
    if width <= 0 or height <= 0 or not 0 < fov < 180:
        raise ValueError("Camera dimensions must be positive and FOV between 0 and 180")
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def resize_intrinsics(intrinsics, source_size, target_size):
    """Sizes are (width, height); preserve PIL's half-pixel sampling geometry."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 intrinsic matrix")
    if any(size <= 0 for size in (*source_size, *target_size)):
        raise ValueError("Image sizes must be positive")
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return transform @ matrix


def preprocess_rgb(rgb, camera=None):
    import torch

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HWC uint8 RGB, not BGR/BGRA or normalized floats")
    height, width = rgb.shape[:2]
    if not height or not width:
        raise ValueError("Empty RGB image")
    camera = camera or {}
    if camera.get("width", width) != width or camera.get("height", height) != height:
        raise ValueError("Camera metadata dimensions do not match RGB")
    native_k = camera.get("intrinsics", camera_intrinsics(width, height, camera.get("fov", 90)))
    resized = Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor, resize_intrinsics(native_k, (width, height), (WIDTH, HEIGHT))


def decode_bgra(raw_data, width, height):
    if width <= 0 or height <= 0 or len(raw_data) != width * height * 4:
        raise ValueError("BGRA byte length does not match camera dimensions")
    return np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)


def decode_rgb(raw_data, width, height):
    return decode_bgra(raw_data, width, height)[:, :, 2::-1].copy()


def decode_depth(raw_data, width, height):
    channels = decode_bgra(raw_data, width, height).astype(np.uint32)
    encoded = channels[:, :, 2] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 0]
    return (encoded.astype(np.float64) * (1000.0 / 16777215)).astype(np.float32)
