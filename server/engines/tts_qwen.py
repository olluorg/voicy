"""Qwen3-TTS behind a small synchronous interface.

The model is loaded once and kept warm; a lock serialises GPU work because a
single 1.7B model does not benefit from concurrent requests and would only run
out of memory trying.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .device import has_cuda, torch_device


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int


class QwenTTS:
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str | None = None):
        self.model_id = model_id
        self.device = device or torch_device()
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel

        with self._lock:
            if self._model is None:
                # bfloat16 на CPU поддержан хуже float32, а выигрыш там всё равно
                # съедается отсутствием тензорных ядер
                dtype = torch.bfloat16 if has_cuda() else torch.float32
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_id, device_map=self.device, dtype=dtype
                )

    def speak(self, text: str, ref_audio: str | Path, ref_text: str,
              language: str = "Russian", seed: int | None = None) -> Synthesis:
        """Clone the reference voice and read `text` with it."""
        import torch

        self.load()
        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            waves, sr = self._model.generate_voice_clone(
                text=text, language=language,
                ref_audio=str(ref_audio), ref_text=ref_text,
            )
        return Synthesis(np.asarray(waves[0], dtype=np.float32), int(sr))
