"""What the server asks of a model, and nothing about how the model works.

Four kinds of engine sit behind the server: a synthesiser, a recogniser, a turn
detector and a voice activity detector. The server talks to them only through
the interfaces below. Everything a particular model needs — its codec rate,
its language names, the order it wants a prompt and hotwords in, CUDA graphs —
stays inside that model's engine, so replacing a model means writing one engine
and touching nothing else (docs/adr/0021).

What the interfaces speak in is what the API speaks in: seconds of audio,
ISO 639-1 language codes, plain text. An engine that cannot do something says
so through its `features` rather than failing halfway, and the server turns a
request for a missing feature into a 400 before any work starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


class Unsupported(ValueError):
    """The engine cannot do what was asked: a language, a feature. The caller's fault."""


class BadAudio(ValueError):
    """The input is not audio the engine can decode. The caller's fault too."""


# ---------------------------------------------------------------------- синтез

@dataclass
class Progress:
    """How far a synthesis has got. An engine fills in only what it actually knows.

    A model that generates audio in order (Qwen3-TTS) knows how many seconds are
    ready but not how many are coming. A model that decides the length first and
    refines everything at once (F5) knows the length exactly and the share of
    work done, but has no finished second until the very end.
    """
    produced: float | None = None   # с звука уже готово
    done: float | None = None       # доля работы, 0..1
    total: float | None = None      # с — длительность результата, если известна заранее


@dataclass
class Synthesis:
    audio: np.ndarray               # float32, моно
    sample_rate: int
    info: dict = field(default_factory=dict)   # числа самого движка — для замеров, сервер их не читает


class TTSEngine(Protocol):
    name: str                       # движок: "qwen3-tts"
    model: str                      # веса: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device: str
    sample_rate: int                # частота готового звука — известна до загрузки
    needs_reference_text: bool      # нужна ли движку расшифровка образца голоса
    reference_seconds: tuple[float, float]   # какой длины образец движок принимает
    reference_best: tuple[float, float]      # и с какой клонирует лучше всего

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def status(self) -> dict:
        """What /health shows: engine, model, loaded, device and anything of its own."""

    def language(self, code: str | None) -> str | None:
        """Check a language before the work is queued. Returns it normalised;
        raises Unsupported if the engine cannot speak it. None means the default."""

    def speak(self, text: str, ref_audio: Path, ref_text: str, language: str | None = None,
              seed: int | None = None,
              on_progress: Callable[[Progress], None] | None = None) -> Synthesis:
        """Clone the reference voice and read `text` with it.

        `on_progress` must be called regularly — at least once a second of work:
        it is also where cancellation happens, by raising out of the callback.
        """


# --------------------------------------------------------------- распознавание

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
    language: str                   # ISO 639-1
    language_probability: float
    duration: float
    segments: list[Segment]


# Что распознаватель может уметь сверх текста. Запрошенное, но не умеемое — 400.
STT_FEATURES = frozenset({
    "translate",            # task="translate": речь на любом языке → английский текст
    "prompt",               # подсказка текстом: стиль, начало разговора
    "hotwords",             # список терминов, которые надо услышать
    "word_timestamps",      # время каждого слова
})


def require(stt: "STTEngine", *, task: str = "transcribe", prompt: str | None = None,
            hotwords: list[str] | None = None, word_timestamps: bool = False) -> None:
    """Refuse up front what the recogniser cannot do, instead of quietly ignoring it."""
    wanted = {"translate"} if task == "translate" else set()
    wanted |= {f for f, on in (("prompt", prompt), ("hotwords", hotwords),
                               ("word_timestamps", word_timestamps)) if on}
    missing = wanted - stt.features
    if missing:
        raise Unsupported(f"{stt.name} cannot do: {', '.join(sorted(missing))}")


class STTEngine(Protocol):
    name: str
    model: str
    device: str
    features: frozenset[str]        # подмножество STT_FEATURES

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def status(self) -> dict: ...

    def transcribe(self, audio: str | np.ndarray, language: str | None = None,
                   prompt: str | None = None, hotwords: list[str] | None = None,
                   temperature: float = 0.0, word_timestamps: bool = False,
                   task: str = "transcribe", live: bool = False, draft: bool = False,
                   on_segment: Callable[[float, float, str], None] | None = None) -> Transcript:
        """`audio` is a file path or 16 kHz mono float32 samples.

        `live` — a short window of a live stream rather than a file: nothing
        carries over between readings, silence is skipped. `draft` — the window
        will be read again, so speed matters more than accuracy.

        `on_segment` receives (position, total, text) in seconds as text lands;
        raising out of it cancels the work.
        """


# ---------------------------------------------------------------- конец реплики

class TurnEngine(Protocol):
    name: str
    model: str
    device: str
    sample_rate: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def status(self) -> dict: ...

    def probability(self, audio: np.ndarray) -> float:
        """Mono float32 of the turn so far at `sample_rate`, including the pause
        after it. Returns the probability that the speaker has finished."""


# ------------------------------------------------------------- детектор голоса

class VADStream(Protocol):
    frame: int                      # отсчётов в кадре
    pending: int                    # отсчётов, пришедших, но ещё не оценённых

    def feed(self, audio: np.ndarray) -> np.ndarray:
        """More 16 kHz mono float32; one speech probability per completed frame."""


class VADEngine(Protocol):
    name: str
    sample_rate: int

    def stream(self) -> VADStream:
        """A fresh detector for one session: frames are scored as they arrive."""
