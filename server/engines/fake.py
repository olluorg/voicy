"""Training engines: the contract without a model, for checking the server.

They need no weights, no GPU and nothing beyond numpy, so the API checks
(tests/) run in seconds on any machine CI has — Linux, Windows, macOS, x86-64
or ARM64. They are deterministic, so a check can say exactly what should come
back. And they are a third implementation of the contract (ADR 0021): anything
the server does only because a real model happens to do it shows up here.

    TTS_ENGINE=tone STT_ENGINE=script TURN_ENGINE=pause VAD_ENGINE=energy

  tone   — synthesis: a 220 Hz tone, 60 ms per character, generated in 0.1 s
           pieces at TONE_SPEED × real time, reporting progress after each.
           TONE_KNOWS_LENGTH=1 makes it report like F5 — share of work done
           and the exact length — instead of seconds produced.
  script — recognition: one word per 0.5 s of sound louder than silence,
           taken in turn from a fixed list; silence ends a segment.
  pause  — end of turn: complete if the last 0.2 s are silent.
  energy — voice activity: loudness of each 32 ms frame.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .base import BadAudio, Progress, Segment, Synthesis, Transcript, Unsupported, Word

QUIET = 0.01                        # RMS: тише — тишина


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


class _Loaded:
    """Nothing to load; the flag is there because the contract has it."""
    device = "cpu"

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def status(self) -> dict:
        return {"engine": self.name, "model": self.model, "loaded": self.loaded,
                "device": self.device}


# ---------------------------------------------------------------------- синтез

class ToneTTS(_Loaded):
    name = "tone"
    sample_rate = 24000
    needs_reference_text = False
    reference_seconds = (3.0, 60.0)
    reference_best = (8.0, 14.0)
    SECONDS_PER_CHAR = 0.06
    PIECE = 0.1                     # с — кусок, после которого сообщается прогресс
    LANGUAGES = ("ru", "en")

    def __init__(self, model: str = "tone-220hz"):
        super().__init__()
        self.model = model
        self.speed = float(os.environ.get("TONE_SPEED", "50"))
        self.knows_length = os.environ.get("TONE_KNOWS_LENGTH") == "1"

    def language(self, code: str | None) -> str | None:
        c = (code or "ru").strip().lower() or "ru"
        if c not in self.LANGUAGES:
            raise Unsupported(f"language '{code}' is not supported by {self.name}; "
                              f"supported: {', '.join(self.LANGUAGES)}")
        return c

    def speak(self, text: str, ref_audio: str | Path, ref_text: str,
              language: str | None = None, seed: int | None = None,
              on_progress: Callable[[Progress], None] | None = None) -> Synthesis:
        self.language(language)
        self.load()
        total = max(0.3, len(text.strip()) * self.SECONDS_PER_CHAR)
        n = int(round(total * self.sample_rate))
        step = int(self.PIECE * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for done in range(step, n + step, step):
            time.sleep(self.PIECE / self.speed)
            if on_progress is not None:
                produced = min(done, n) / self.sample_rate
                on_progress(Progress(done=produced / total, total=total) if self.knows_length
                            else Progress(produced=produced))
        return Synthesis(audio, self.sample_rate, {"pieces": -(-n // step)})


# --------------------------------------------------------------- распознавание

class ScriptSTT(_Loaded):
    name = "script"
    features = frozenset({"translate", "prompt", "hotwords", "word_timestamps"})
    WORDS = ("проверка", "связи", "кавка", "раз", "два")
    ENGLISH = ("check", "one", "two")
    FRAME = 0.5                     # с звука на слово
    SR = 16000

    def __init__(self, model: str = "script"):
        super().__init__()
        self.model = model

    def transcribe(self, audio: str | np.ndarray, language: str | None = None,
                   prompt: str | None = None, hotwords: list[str] | None = None,
                   temperature: float = 0.0, word_timestamps: bool = False,
                   task: str = "transcribe", live: bool = False, draft: bool = False,
                   on_segment: Callable[[float, float, str], None] | None = None) -> Transcript:
        self.load()
        if isinstance(audio, (str, Path)):
            import audio_io
            try:
                audio = audio_io.decode(Path(audio).read_bytes(), self.SR)
            except ValueError as e:
                raise BadAudio(str(e)) from None
        vocab = self.ENGLISH if task == "translate" else self.WORDS
        step = int(self.FRAME * self.SR)
        duration = len(audio) / self.SR
        segments: list[Segment] = []
        words: list[Word] = []
        i = 0

        def close() -> None:
            if words:
                seg = Segment(words[0].start, words[-1].end,
                              " ".join(w.word.strip() for w in words),
                              list(words) if word_timestamps else [])
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg.end, round(duration, 3), seg.text)
                words.clear()

        for k in range(0, len(audio) - step + 1, step):
            if _rms(audio[k:k + step]) > QUIET:
                words.append(Word(round(k / self.SR, 3), round((k + step) / self.SR, 3),
                                  " " + vocab[i % len(vocab)]))
                i += 1
            else:
                close()
        close()
        return Transcript(text=" ".join(s.text for s in segments),
                          language=language or "ru", language_probability=1.0,
                          duration=round(duration, 3), segments=segments)


# ---------------------------------------------------------------- конец реплики

class PauseTurn(_Loaded):
    name = "pause"
    sample_rate = 16000
    TAIL = 0.2

    def __init__(self, model: str = "pause-0.2s"):
        super().__init__()
        self.model = model

    def probability(self, audio: np.ndarray) -> float:
        self.load()
        tail = audio[-int(self.TAIL * self.sample_rate):]
        return 0.9 if _rms(tail) <= QUIET else 0.1


# ------------------------------------------------------------- детектор голоса

class EnergyVAD:
    name = "energy"
    sample_rate = 16000

    def stream(self) -> "_EnergyStream":
        return _EnergyStream()


class _EnergyStream:
    frame = 512

    def __init__(self):
        self._rest = np.zeros(0, np.float32)

    @property
    def pending(self) -> int:
        return len(self._rest)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._rest, audio])
        n = len(x) // self.frame
        self._rest = x[n * self.frame:]
        frames = x[:n * self.frame].reshape(n, self.frame)
        rms = np.sqrt(np.mean(np.square(frames), axis=1)) if n else np.zeros(0)
        return np.clip(rms / (2 * QUIET), 0.0, 1.0).astype(np.float32)
