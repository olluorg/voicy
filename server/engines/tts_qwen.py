"""Qwen3-TTS behind the synthesiser interface (base.TTSEngine).

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
from pathlib import Path
from typing import Callable

import numpy as np

from .base import Progress, Synthesis, Unsupported
from .device import has_cuda, torch_device

# Один шаг декодирования = один аудио-токен. Название модели обещает 12 Гц, но
# на готовом звуке измеряется 12.6–13.0 шага на секунду: несколько шагов уходит
# на промпт и на завершение, и по номинальным 12 счётчик обгоняет реальность
# примерно на 5%. Берём измеренное значение, а не заявленное.
#
#   24 шага → 1.84 с (13.0)   67 → 5.28 (12.7)
#  127 шагов → 10.08 с (12.6) 145 → 11.52 (12.6)
STEPS_PER_SECOND = 12.6

# Модель знает языки по именам — по ним выбирается токен языка в talker_config.
LANGUAGES = {"ru": "Russian", "en": "English", "de": "German", "fr": "French",
             "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
             "ko": "Korean", "zh": "Chinese"}


class QwenTTS:
    name = "qwen3-tts"
    sample_rate = 24000
    needs_reference_text = True     # клонирует по паре «образец + его расшифровка»
    # Короче трёх секунд образец не несёт тембра, длиннее минуты — лишь удлиняет
    # каждый синтез. Лучшие 8–14 с — рекомендация README, остальное принимается
    # с предупреждением.
    reference_seconds = (3.0, 60.0)
    reference_best = (8.0, 14.0)

    def __init__(self, model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str | None = None):
        self.model = model
        self.device = device or torch_device()
        self._model = None
        self._lock = threading.Lock()
        self._on_step: Callable[[int], None] | None = None
        self._steps = 0
        self.fast = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {"engine": self.name, "model": self.model, "loaded": self.loaded,
                "device": self.device, "cuda_graphs": self.fast is not None}

    def language(self, code: str | None) -> str | None:
        """ISO code or the model's own name, any case; None — Russian."""
        if code is None or not code.strip():
            return "ru"
        c = code.strip().lower()
        if c in LANGUAGES:
            return c
        for iso, name in LANGUAGES.items():
            if name.lower() == c:
                return iso
        raise Unsupported(f"language '{code}' is not supported by {self.name}; "
                          f"supported: {', '.join(LANGUAGES)}")

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
                    self.model, device_map=self.device, dtype=dtype
                )
                self._install_step_hook()
                self._install_fast_path()

    def _install_fast_path(self) -> None:
        """Code predictor from CUDA graphs: ×0.55 → ×1.5 on an RTX 3080 (tts_fast.py).

        Graphs are recorded here, at load, so the first request is not the one
        that pays for it. TTS_FAST=0 keeps the original path.
        """
        import os

        if os.environ.get("TTS_FAST", "1") == "0" or not self.device.startswith("cuda"):
            return
        from .tts_fast import install

        self.fast = install(self._model)
        self.fast.capture()

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
              language: str | None = None, seed: int | None = None,
              on_progress: Callable[[Progress], None] | None = None) -> Synthesis:
        """Clone the reference voice and read `text` with it.

        `on_progress` is called once per decoding step with seconds produced so
        far. How many are coming is not known until generation stops.
        """
        import torch

        lang = LANGUAGES[self.language(language)]
        self.load()
        with self._lock:
            self._steps = 0
            self._on_step = ((lambda n: on_progress(Progress(produced=n / STEPS_PER_SECOND)))
                             if on_progress is not None else None)
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                waves, sr = self._model.generate_voice_clone(
                    text=text, language=lang,
                    ref_audio=str(ref_audio), ref_text=ref_text,
                )
                steps = self._steps
            finally:
                self._on_step = None
        return Synthesis(np.asarray(waves[0], dtype=np.float32), int(sr), {"steps": steps})
