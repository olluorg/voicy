"""Job registry for work that takes long enough to want a progress bar.

Every number that reaches the client here is measured, not predicted:

  synthesis — the talker's decoding loop is counted step by step, and the model
              is a 12 Hz codec, so steps convert to seconds of audio already
              produced. The *total* is not knowable until generation stops, so
              it is carried separately and labelled as an estimate;

  recognition — the audio duration is known up front and each segment reports
              where it ended, so both halves of the fraction are facts.

Events are pushed onto a per-job queue and drained by an SSE endpoint.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    kind: str
    created: float = field(default_factory=time.time)
    state: str = "running"          # running | done | error
    result: Any = None
    error: str | None = None
    last: dict = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None

    def emit(self, **payload) -> None:
        """Called from a worker thread — hands the event to the event loop."""
        payload.setdefault("elapsed", round(time.time() - self.created, 2))
        self.last = payload
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)
        except RuntimeError:                    # цикл уже закрыт — клиент ушёл
            pass


class Registry:
    def __init__(self, ttl_seconds: int = 1800):
        self._jobs: dict[str, Job] = {}
        self.ttl = ttl_seconds

    def create(self, kind: str) -> Job:
        self._sweep()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        try:
            job.loop = asyncio.get_running_loop()
        except RuntimeError:
            job.loop = None
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _sweep(self) -> None:
        cutoff = time.time() - self.ttl
        for k in [k for k, j in self._jobs.items() if j.created < cutoff]:
            self._jobs.pop(k, None)


registry = Registry()


# Один шаг декодирования = один аудио-токен. Название модели обещает 12 Гц, но
# на готовом звуке измеряется 12.6–13.0 шага на секунду: несколько шагов уходит
# на промпт и на завершение, и по номинальным 12 счётчик обгоняет реальность
# примерно на 5%. Берём измеренное значение, а не заявленное.
#
#   24 шага → 1.84 с (13.0)   67 → 5.28 (12.7)
#  127 шагов → 10.08 с (12.6) 145 → 11.52 (12.6)
STEPS_PER_SECOND = 12.6

# Темп выбранного по умолчанию голоса, слогов в секунду. Используется только
# для оценки полной длительности, то есть знаменателя — и помечается как оценка.
SYLLABLES_PER_SECOND = 4.5
VOWELS = set("аеёиоуыэюяaeiouy")


def estimate_seconds(text: str) -> float:
    """Ожидаемая длительность по числу слогов. Именно оценка, а не измерение."""
    syllables = sum(1 for c in text.lower() if c in VOWELS)
    return max(0.5, syllables / SYLLABLES_PER_SECOND)
