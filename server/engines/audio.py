"""Decoding what the server's own decoder cannot read.

The Rust server reads wav, mp3, flac, ogg, webm and mp4 itself; whatever else
a browser or a phone produces comes here (`audio.decode` in host.py), as do
files the training recogniser is handed by path.
"""
from __future__ import annotations

import io

import numpy as np


def decode(raw: bytes, sr: int) -> np.ndarray:
    """Any container a browser or a phone produces → mono float32 at `sr`.

    Goes through PyAV, so it works without the system ffmpeg. The same steps
    as faster-whisper's decode_audio.
    """
    import av

    try:
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sr)
        chunks = []
        with av.open(io.BytesIO(raw), mode="r", metadata_errors="ignore") as container:
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    except Exception as e:                  # noqa: BLE001 — формат приходит от клиента
        raise ValueError(f"cannot decode audio: {e}") from None
    if not chunks:
        raise ValueError("cannot decode audio: no audio stream")
    return np.concatenate(chunks).astype(np.float32) / 32768.0
