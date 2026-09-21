"""One GPU, one worker, one line.

Both models share a single card, and each already serialises itself with a lock.
Without a queue that lock *was* the queue — except that nobody could see their
place in it, cancel, or be told there is no room. Here every request, the
synchronous OpenAI-compatible ones included, becomes a job and waits its turn
in submission order.

Live dictation does not come through here: it cannot wait behind a long
synthesis and still be live. It takes the recogniser's lock between jobs.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Callable

import webhooks
from progress import BadInput, Job, JobCancelled

MAX_PENDING = int(os.environ.get("VOICY_QUEUE_MAX", "32"))


class QueueFull(Exception):
    pass


Work = Callable[[Job], dict]        # делает работу, кладёт job.result, отдаёт поля для «done»


class WorkQueue:
    def __init__(self, max_pending: int = MAX_PENDING):
        self.max_pending = max_pending
        self._pending: deque[tuple[Job, Work]] = deque()
        self._cv = threading.Condition()
        self.current: Job | None = None
        self._thread = threading.Thread(target=self._loop, name="voicy-worker", daemon=True)
        self._thread.start()

    def submit(self, job: Job, work: Work) -> int:
        with self._cv:
            if len(self._pending) >= self.max_pending:
                raise QueueFull(f"queue is full ({self.max_pending} jobs waiting)")
            self._pending.append((job, work))
            position = len(self._pending) - 1 + (self.current is not None)
            job.emit(stage="queued", queue_position=position)
            self._cv.notify()
        return position

    def position(self, job: Job) -> int | None:
        """How many jobs run before this one; None once it has started."""
        with self._cv:
            for i, (j, _) in enumerate(self._pending):
                if j is job:
                    return i + (self.current is not None)
        return None

    def pending(self) -> int:
        with self._cv:
            return len(self._pending)

    def cancel(self, job: Job) -> bool:
        """Queued: removed at once. Running: stopped at the next step. Done: False."""
        with self._cv:
            for item in self._pending:
                if item[0] is job:
                    self._pending.remove(item)
                    job.finish("cancelled")
                    self._announce_positions()
                    break
            else:
                if job.done:
                    return False
                job.cancel_requested = True
                return True
        webhooks.notify(job)
        return True

    # ------------------------------------------------------------------ worker

    def _announce_positions(self) -> None:
        for i, (j, _) in enumerate(self._pending):
            j.emit(stage="queued", queue_position=i + (self.current is not None))

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                job, work = self._pending.popleft()
                self.current = job
                self._announce_positions()
            try:
                job.state, job.started = "running", time.time()
                job.emit(stage="running")
                job.check()                         # отменили в зазоре между очередью и стартом
                extra = work(job)
                job.finish("done", **(extra or {}))
            except JobCancelled:
                job.result = None
                job.finish("cancelled")
            except Exception as e:                  # noqa: BLE001 — уходит клиенту
                job.error = str(e) or type(e).__name__
                job.error_status = 400 if isinstance(e, BadInput) else 500
                job.finish("error", error=job.error)
            finally:
                with self._cv:
                    self.current = None
                webhooks.notify(job)


queue = WorkQueue()
