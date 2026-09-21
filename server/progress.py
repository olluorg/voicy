"""Job registry for work that takes long enough to want a progress bar.

Every number that reaches the client here is measured, not predicted:

  synthesis — the talker's decoding loop is counted step by step, and the model
              is a 12 Hz codec, so steps convert to seconds of audio already
              produced. The *total* is not knowable until generation stops, so
              it is carried separately and labelled as an estimate;

  recognition — the audio duration is known up front and each segment reports
              where it ended, so both halves of the fraction are facts.

Events fan out to every subscriber — an SSE stream, a waiting synchronous
request — so several clients can watch the same job.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


TERMINAL = ("done", "error", "cancelled")


class JobCancelled(Exception):
    """Raised from inside a progress callback to abandon the work."""


class BadInput(Exception):
    """The job failed because of what it was given, not because of the server.
    Reaches the client as 400 rather than 500."""


@dataclass
class Job:
    id: str
    kind: str
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    state: str = "queued"           # queued | running | done | error | cancelled
    result: Any = None
    error: str | None = None
    error_status: int = 500         # 400, если виноват вход (BadInput)
    last: dict = field(default_factory=dict)
    loop: asyncio.AbstractEventLoop | None = None
    base_url: str = ""
    webhook: dict | None = None     # {"url", "attempts", "delivered", "error"}
    cancel_requested: bool = False
    _subscribers: list = field(default_factory=list)

    # --------------------------------------------------------------- события

    def emit(self, **payload) -> None:
        """Called from a worker thread — hands the event to the event loop."""
        payload.setdefault("elapsed", round(time.time() - self.created, 2))
        self.last = payload
        self._dispatch(payload)

    def finish(self, state: str, **payload) -> None:
        """Terminal event. `last` is set before `state`, so anyone who sees the
        final state also sees the final event."""
        payload["stage"] = state
        payload.setdefault("elapsed", round(time.time() - self.created, 2))
        self.finished = time.time()
        self.last = payload
        self.state = state
        self._dispatch(payload)

    def _dispatch(self, payload: dict) -> None:
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(self._fan_out, payload)
        except RuntimeError:                    # цикл уже закрыт — клиент ушёл
            pass

    def _fan_out(self, payload: dict) -> None:  # только в потоке цикла событий
        for q in self._subscribers:
            q.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue:
        """Only from the event loop. Subscribe first, then look at `state`."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def wait(self) -> None:
        q = self.subscribe()
        try:
            while self.state not in TERMINAL:
                event = await q.get()
                if event.get("stage") in TERMINAL:
                    break
        finally:
            self.unsubscribe(q)

    # ---------------------------------------------------------------- отмена

    def check(self) -> None:
        """Called by the work between steps; aborts it if cancellation was asked."""
        if self.cancel_requested:
            raise JobCancelled()

    @property
    def done(self) -> bool:
        return self.state in TERMINAL


class Registry:
    def __init__(self, ttl_seconds: int = 1800):
        self._jobs: dict[str, Job] = {}
        self.ttl = ttl_seconds

    def create(self, kind: str, base_url: str = "") -> Job:
        self._sweep()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, base_url=base_url.rstrip("/"))
        try:
            job.loop = asyncio.get_running_loop()
        except RuntimeError:
            job.loop = None
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def discard(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created)

    def _sweep(self) -> None:
        """Forget finished jobs after the TTL. Queued and running ones stay."""
        cutoff = time.time() - self.ttl
        for k in [k for k, j in self._jobs.items()
                  if j.finished is not None and j.finished < cutoff]:
            self._jobs.pop(k, None)


registry = Registry(int(os.environ.get("VOICY_JOB_TTL", "1800")))


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
