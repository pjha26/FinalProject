"""Fixed-scale inverse-depth colors and a single shared RGB/depth/overlay view."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def render_frame(rgb, inverse_depth, caption="Relative depth - near: yellow, far: purple"):
    from matplotlib import colormaps
    from matplotlib import get_data_path

    rgb = np.asarray(rgb)
    inverse_depth = np.asarray(inverse_depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Display expects uint8 RGB")
    if inverse_depth.ndim != 2 or not np.isfinite(inverse_depth).all() or (inverse_depth <= 0).any():
        raise ValueError("Display expects finite positive numerical inverse depth")
    height = max(1, round(rgb.shape[0] * 640 / rgb.shape[1]))
    size = (640, height)
    original = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    resized = np.asarray(Image.fromarray(inverse_depth).resize(size, Image.Resampling.BILINEAR))
    # Fixed display bounds prevent per-frame color scaling from manufacturing flicker.
    normalized = np.clip((resized - 0.01) / (10.0 - 0.01), 0, 1)
    colors = (colormaps["plasma"](normalized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (original * 0.6 + colors * 0.4).astype(np.uint8)
    canvas = Image.new("RGB", (1920, height + 44), (20, 20, 20))
    font = ImageFont.truetype(str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), 14)
    for offset, panel, label in zip((0, 640, 1280), (original, colors, overlay), ("RGB", "Relative inverse depth", "Overlay")):
        canvas.paste(Image.fromarray(panel), (offset, 44))
        ImageDraw.Draw(canvas).text((offset + 8, 3), label, fill="white", font=font)
    ImageDraw.Draw(canvas).text((8, 23), caption, fill="white", font=font)
    return np.asarray(canvas).copy()
