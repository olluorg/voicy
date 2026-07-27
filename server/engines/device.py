"""Pick the device once, so the same image runs with or without a GPU.

In a container the GPU may simply not be passed through, and both engines should
degrade to CPU rather than crash on start. faster-whisper additionally cannot use
float16 on CPU, so the compute type follows from the same decision.
"""
from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def has_cuda() -> bool:
    if os.environ.get("FORCE_CPU") == "1":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_device() -> str:
    return os.environ.get("TTS_DEVICE") or ("cuda:0" if has_cuda() else "cpu")


def whisper_device() -> tuple[str, str]:
    """(device, compute_type) — float16 is a GPU-only option."""
    if os.environ.get("STT_DEVICE"):
        dev = os.environ["STT_DEVICE"]
    else:
        dev = "cuda" if has_cuda() else "cpu"
    ct = os.environ.get("STT_COMPUTE_TYPE") or ("float16" if dev.startswith("cuda") else "int8")
    return dev, ct


def describe() -> str:
    if not has_cuda():
        return "cpu"
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"
