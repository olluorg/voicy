"""Silero voice activity detector behind the VAD interface (base.VADEngine).

The model comes bundled with faster-whisper as an ONNX file and runs on the CPU
in 32 ms frames, so live speech is scored as it arrives and never waits for the
recogniser. Taking it from there saves a download, not a dependency: with a
different recogniser this module is what needs a new source for the weights.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 512                 # отсчётов на кадр, 32 мс
CONTEXT = 64                # столько предыдущих отсчётов видит кадр


class SileroVAD:
    name = "silero"
    sample_rate = SR

    def stream(self) -> "SileroStream":
        return SileroStream()


class SileroStream:
    """Frame by frame. Each frame sees only 64 samples before it, so frames can
    be scored as they arrive with the same result as all at once."""

    frame = FRAME

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._tail = np.zeros(CONTEXT, np.float32)
        self._rest = np.zeros(0, np.float32)         # хвост короче кадра

    @property
    def pending(self) -> int:
        return len(self._rest)

    def feed(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._rest, x])
        n = len(x) // FRAME
        self._rest = x[n * FRAME:]
        if n == 0:
            return np.zeros(0, np.float32)
        frames = x[:n * FRAME].reshape(n, FRAME)
        ctx = np.vstack([self._tail[None, :], frames[:-1, -CONTEXT:]])
        self._tail = frames[-1, -CONTEXT:].copy()
        batch = np.concatenate([ctx, frames], axis=1).astype(np.float32)
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        out, _, _ = self._model.session.run(None, {"input": batch, "h": h, "c": c})
        return np.asarray(out, np.float32).reshape(-1)
