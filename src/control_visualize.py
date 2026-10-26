"""Readable measurement overlay for the separate CARLA reference-control experiment."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.metric_control import clearance_sample_mask
from src.visualize import render_frame


@lru_cache(maxsize=1)
def _fonts():
    from matplotlib import get_data_path
    from pathlib import Path
    path = str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    return {size: ImageFont.truetype(path, size) for size in (22, 26, 30, 48)}


def render_control_frame(rgb, inverse_depth, reference_depth, intrinsics, reference_m, model_m, speed_mps, brake, state, vehicle_gap_m=None):
    numbers = [reference_m, model_m, speed_mps, brake]
    if not np.isfinite(numbers).all() or reference_m <= 0 or model_m <= 0 or speed_mps < 0 or not 0 <= brake <= 1:
        raise ValueError("Invalid control-display measurements")
    if vehicle_gap_m is not None and not np.isfinite(vehicle_gap_m):
        raise ValueError("Invalid simulator vehicle gap")
    preview = Image.fromarray(render_frame(rgb, inverse_depth, "SIMULATOR REFERENCE CONTROL | learned estimates are unvalidated and never actuate"))
    native_height = reference_depth.shape[0]
    native_width = reference_depth.shape[1]
    if rgb.shape[:2] != (native_height, native_width):
        raise ValueError("Measurement overlay requires aligned RGB/reference geometry")
    mask = clearance_sample_mask(reference_depth, intrinsics)
    font = _fonts()
    draw = ImageDraw.Draw(preview)
    if mask.any():
        ys, xs = np.where(mask)
        factor = 640 / native_width
        x1, y1 = int(xs.min() * factor), 44 + int(ys.min() * factor)
        x2, y2 = int(xs.max() * factor), 44 + int(ys.max() * factor)
        draw.rectangle((x1, y1, max(x1 + 6, x2), max(y1 + 6, y2)), outline="#46FF94", width=4)
        label = "REFERENCE {:.1f} m".format(reference_m)
        box = draw.textbbox((0, 0), label, font=font[26])
        label_y = max(44, y1 - 38)
        label_x = min(x1, 620 - (box[2] - box[0]))
        draw.rectangle((label_x, label_y, label_x + box[2] - box[0] + 12, label_y + 36), fill="#10231B")
        draw.text((label_x + 5, label_y + 2), label, font=font[26], fill="#46FF94")
    canvas = Image.new("RGB", (preview.width, preview.height + 170), "#121C28")
    canvas.paste(preview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    top = preview.height
    draw.line((0, top, 1920, top), fill="#A4B1BE", width=2)
    draw.text((22, top + 10), "Camera clearance - SIMULATOR REFERENCE", font=font[26], fill="#83FFB0")
    draw.text((990, top + 10), "Model estimate - UNVALIDATED", font=font[26], fill="#FFC46B")
    ref_text = "{:.1f} m".format(reference_m) if reference_m < 50 else "No obstacle found within 50 m"
    draw.text((22, top + 42), ref_text, font=font[48] if reference_m < 50 else font[30], fill="white")
    draw.text((990, top + 42), "{:.1f} m  (not used for control)".format(model_m), font=font[48], fill="#FFC46B")
    gap = "n/a" if vehicle_gap_m is None else "{:.1f} m".format(vehicle_gap_m)
    draw.text((22, top + 108), "Speed {:.1f} km/h | Brake {:.0f}% | Vehicle gap (simulator): {}".format(speed_mps * 3.6, brake * 100, gap), font=font[26], fill="white")
    draw.text((22, top + 140), "State: {} | Green outline = reference-depth sample region, not object detection".format(state), font=font[22], fill="#CBDAE7")
    return np.asarray(canvas).copy()

