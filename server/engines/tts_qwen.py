"""Qwen3-TTS behind a small synchronous interface.

The model is loaded once and kept warm; a lock serialises GPU work because a
single 1.7B model does not benefit from concurrent requests and would only run
out of memory trying.

Progress is reported from inside the decoding loop. The public API drops user
kwargs before they reach generation, but the model delegates to
`self.talker.generate`, which is an ordinary transformers model — wrapping that
one call lets a logits processor ride along. It changes nothing about the output
and is invoked exactly once per step, and the model is a 12 Hz codec, so steps
convert to seconds of audio already produced.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .device import has_cuda, torch_device


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int
    steps: int = 0


class QwenTTS:
    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str | None = None):
        self.model_id = model_id
        self.device = device or torch_device()
        self._model = None
        self._lock = threading.Lock()
        self._on_step: Callable[[int], None] | None = None
        self._steps = 0

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
                self._install_step_hook()

    def _install_step_hook(self) -> None:
        from transformers import LogitsProcessor, LogitsProcessorList

        inner = getattr(self._model, "model", self._model)
        talker = getattr(inner, "talker", None)
        if talker is None or not hasattr(talker, "generate"):
            return                                   # прогресса не будет, синтез — будет

        owner = self

        class StepCounter(LogitsProcessor):
            """Не трогает логиты, только считает вызовы."""
            def __call__(self, input_ids, scores):
                owner._steps += 1
                if owner._on_step is not None:
                    owner._on_step(owner._steps)
                return scores

        original = talker.generate
        counter = StepCounter()

        def with_counter(*args, **kwargs):
            processors = kwargs.get("logits_processor") or LogitsProcessorList()
            if counter not in processors:
                processors.append(counter)
            kwargs["logits_processor"] = processors
            return original(*args, **kwargs)

        talker.generate = with_counter

    def speak(self, text: str, ref_audio: str | Path, ref_text: str,
              language: str = "Russian", seed: int | None = None,
              on_step: Callable[[int], None] | None = None) -> Synthesis:
        """Clone the reference voice and read `text` with it.

        `on_step` is called once per decoding step with the running count.
        """
        import torch

        self.load()
        with self._lock:
            self._steps = 0
            self._on_step = on_step
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                waves, sr = self._model.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=str(ref_audio), ref_text=ref_text,
                )
                steps = self._steps
            finally:
                self._on_step = None
        return Synthesis(np.asarray(waves[0], dtype=np.float32), int(sr), steps)
