"""Speech to text via faster-whisper.

Shares the GPU with the synthesiser, so it takes the same kind of lock. Word
timestamps are optional because they cost time and most callers only want text.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]


class WhisperSTT:
    def __init__(self, model_id: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        with self._lock:
            if self._model is None:
                self._model = WhisperModel(self.model_id, device=self.device,
                                           compute_type=self.compute_type)

    def transcribe(self, path: str, language: str | None = None, prompt: str | None = None,
                   temperature: float = 0.0, word_timestamps: bool = False,
                   beam_size: int = 5) -> Transcript:
        self.load()
        with self._lock:
            segments, info = self._model.transcribe(
                path, language=language, initial_prompt=prompt,
                temperature=temperature, beam_size=beam_size,
                word_timestamps=word_timestamps,
            )
            out: list[Segment] = []
            for s in segments:                      # generator — consumed exactly once
                out.append(Segment(
                    start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip(),
                    words=[Word(round(w.start, 3), round(w.end, 3), w.word)
                           for w in (s.words or [])],
                ))
        return Transcript(
            text=" ".join(s.text for s in out).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 3),
            segments=out,
        )
