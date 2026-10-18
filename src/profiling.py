"""Numerical stability checks and timing summaries, independent of GPU hardware."""
import hashlib
import os

import numpy as np

TIMING_FIELDS = ("decode_ms", "model_ms", "processing_ms", "render_ms", "validation_ms", "pipeline_ms")


def prediction_fingerprint(maps):
    digest = hashlib.sha256()
    for key in ("inverse_depth", "relative_depth"):
        values = np.asarray(maps[key])
        if values.shape != (192, 640) or values.dtype != np.float32 or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Profile requires finite positive float32 192x640 depth maps")
        digest.update(values.tobytes())
    if not np.allclose(maps["inverse_depth"] * maps["relative_depth"], 1, rtol=1e-5, atol=1e-6):
        raise ValueError("Depth and inverse-depth outputs are inconsistent")
    return digest.hexdigest()


def summarize_timings(rows, fields=TIMING_FIELDS):
    if not rows:
        raise ValueError("Cannot summarize an empty profile")
    result = {}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid profile timing: " + field)
        result[field] = dict(mean=float(values.mean()), median=float(np.median(values)),
                             p95=float(np.percentile(values, 95)), minimum=float(values.min()), maximum=float(values.max()))
    return result


def process_memory_mib():
    """Windows process working set, not total system RAM or committed memory."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(key, ctypes.c_size_t) for key in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return {"working_set_mib": counters.WorkingSetSize / 1024 ** 2,
            "peak_working_set_mib": counters.PeakWorkingSetSize / 1024 ** 2}
