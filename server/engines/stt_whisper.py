"""Speech to text via faster-whisper, behind the recogniser interface (base.STTEngine).

Shares the GPU with the synthesiser, so it takes the same kind of lock. Word
timestamps are optional because they cost time and most callers only want text.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .base import BadAudio, Segment, Transcript, Word
from .device import whisper_device


class WhisperSTT:
    name = "faster-whisper"
    features = frozenset({"translate", "prompt", "hotwords", "word_timestamps"})

    def __init__(self, model: str = "large-v3-turbo", device: str | None = None,
                 compute_type: str | None = None):
        auto_dev, auto_ct = whisper_device()
        self.model = model
        self.device = device or auto_dev
        self.compute_type = compute_type or auto_ct
        self._model = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {"engine": self.name, "model": self.model, "loaded": self.loaded,
                "device": self.device, "compute_type": self.compute_type}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        with self._lock:
            if self._model is None:
                self._model = WhisperModel(self.model, device=self.device,
                                           compute_type=self.compute_type)

    def transcribe(self, audio: str | np.ndarray, language: str | None = None,
                   prompt: str | None = None, hotwords: list[str] | None = None,
                   temperature: float = 0.0, word_timestamps: bool = False,
                   task: str = "transcribe", live: bool = False, draft: bool = False,
                   on_segment: Callable[[float, float, str], None] | None = None) -> Transcript:
        """`audio` is a file path or 16 kHz mono float32 samples.

        `on_segment` receives (position, total, text) as each segment lands.
        Both numbers are facts: the file duration is known before decoding starts
        and every segment reports where it ended, so the fraction needs no guessing.
        """
        import av

        prompt, hot = _prompt_and_hotwords(prompt, hotwords)
        self.load()
        out: list[Segment] = []
        with self._lock:
            try:
                segments, info = self._model.transcribe(
                    audio, language=language, initial_prompt=prompt,
                    temperature=temperature, beam_size=2 if draft else 5,
                    word_timestamps=word_timestamps, task=task, vad_filter=live,
                    condition_on_previous_text=not live, hotwords=hot,
                )
                for s in segments:                  # generator — consumed exactly once
                    out.append(Segment(
                        start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip(),
                        words=[Word(round(w.start, 3), round(w.end, 3), w.word)
                               for w in (s.words or [])],
                    ))
                    if on_segment is not None:
                        on_segment(float(s.end), float(info.duration), s.text.strip())
            except av.error.FFmpegError as e:     # не звук или битый файл — вина входа
                raise BadAudio(f"cannot decode audio: {e}") from None
        return Transcript(
            text=" ".join(s.text for s in out).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 3),
            segments=out,
        )


def _prompt_and_hotwords(prompt: str | None, hotwords: list[str] | None) -> tuple[str | None, str | None]:
    """faster-whisper ставит hotwords *перед* подсказкой, и русская фраза,
    оказавшись ближе к декодеру, перетягивает термины обратно в кириллицу:
    62.9% терминов против 85.7% у одних hotwords. Термины после подсказки
    в той же строке возвращают 82.9% (experiments/16-voice-agent)."""
    terms = ", ".join(h for h in (hotwords or []) if h) or None
    if prompt and terms:
        return f"{prompt} {terms}", None
    return prompt or None, terms
