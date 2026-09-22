"""Encoding to the formats the OpenAI audio API advertises.

wav and pcm are written directly; everything else goes through ffmpeg, which is
already required by the project. Opus is the interesting one — at 24 kbps it is
half the size of 32 kbps mp3 and sounds better on speech.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave

import numpy as np

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "application/octet-stream",
}

_FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k"],
    "opus": ["-c:a", "libopus", "-b:a", "24k"],
    "flac": ["-c:a", "flac"],
    "aac": ["-c:a", "aac", "-b:a", "96k"],
}


def to_pcm16(x: np.ndarray) -> bytes:
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1.0:                       # синтез иногда выходит за единицу
        x = x / peak
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def to_wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_pcm16(x))
    return buf.getvalue()


def encode(x: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "wav").lower()
    if fmt == "pcm":
        return to_pcm16(x), CONTENT_TYPES["pcm"]
    wav = to_wav(x, sr)
    if fmt == "wav":
        return wav, CONTENT_TYPES["wav"]
    if fmt not in _FFMPEG_ARGS:
        raise ValueError(f"unsupported response_format: {fmt}")
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, f"out.{fmt}")
        with open(src, "wb") as f:
            f.write(wav)
        _ffmpeg(["-i", src, *_FFMPEG_ARGS[fmt], dst], f"the {fmt} format")
        with open(dst, "rb") as f:
            return f.read(), CONTENT_TYPES[fmt]


def _ffmpeg(args: list[str], purpose: str) -> None:
    """Run ffmpeg; if it is missing, say so by name. The OS alone would not —
    on Windows the error is "the system cannot find the file specified"."""
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg is not installed on the server; it is needed for {purpose}. "
                           "wav and pcm work without it") from None


def stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Change tempo without pitch, via ffmpeg's WSOLA.

    Applied to finished audio rather than to the model's duration prediction:
    the model's own speed control was measured to raise the error rate fivefold
    at the same tempo, while post-hoc stretching leaves it flat.
    """
    if abs(factor - 1.0) < 1e-3:
        return x
    factor = max(0.5, min(2.0, factor))
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, x, sr)
        _ffmpeg(["-i", a, "-filter:a", f"atempo={factor}", b], "changing the tempo")
        y, _ = sf.read(b, dtype="float32")
    return y


def decode(raw: bytes, sr: int) -> np.ndarray:
    """Any container a browser or a phone produces → mono float32 at `sr`.

    Goes through PyAV, so it works without the system ffmpeg that a source
    install may lack. The same steps as faster-whisper's decode_audio, which
    this used to call — but the server should not need a recogniser's package
    to read a file.
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


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)
