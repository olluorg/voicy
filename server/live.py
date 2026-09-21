"""Live speech over a WebSocket, shaped for a voice agent.

The agent needs three things from the listening side, in this order of urgency:

  1. the user started talking      → stop speaking (barge-in);
  2. the user finished their turn  → answer, and not a moment earlier;
  3. what the user said.

Whisper is not a streaming model and knows nothing about turns, so each of the
three is built separately.

**Speech start and pauses** come from the Silero voice detector, run frame by
frame on the incoming audio (32 ms frames, on the CPU). It never waits for the
recogniser.

**End of turn** is decided at the start of each pause. Pause length alone is a
poor signal: people stop mid-thought to find a word. So after `min_pause` of
silence Smart Turn (engines/turn_smart.py) listens to the turn and says whether
it sounds finished. If it does, the rest of the turn is read, and unless the
text ends on a word that cannot end a phrase — a preposition, a conjunction, a
filler — the turn ends there. If it does not sound finished, the turn ends only
after `max_pause` of silence.

**Text** is read by Whisper every `interval` seconds of new speech. A word is
final once two consecutive readings agree on it (LocalAgreement), and the
window is cut at the end of the last agreed sentence. So long speech without
pauses still yields final text every sentence or two, and a phrase is never cut
mid-word.

Protocol: `/v1/audio/transcriptions/stream?language=ru&sample_rate=48000`

    client → binary frames: PCM s16le, mono, at `sample_rate`
    client → {"type": "end"}                 — no more audio: finish the turn, close
    server → {"type": "ready", ...}
    server → {"type": "speech_start", "at"}
    server → {"type": "partial", "text", "start"}
    server → {"type": "final", "text", "start", "end"}
    server → {"type": "turn_check", "at", "probability", "complete"[, "veto"]}
    server → {"type": "turn_end", "text", "start", "end", "reason", "probability"}
    server → {"type": "done", "text", "duration"}
    server → {"type": "error", "error"}

`at`, `start` and `end` are seconds from the start of the session.

This does not go through the job queue: a turn detector waiting behind a long
synthesis would not be live. It shares the recogniser's lock with the queue's
file jobs instead; the voice and turn detectors run on the CPU and never wait.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import contexts

SR = 16000
FRAME = 512                 # отсчётов на кадр детектора голоса, 32 мс
CONTEXT = 64                # столько предыдущих отсчётов видит кадр
SPEECH_ON, SPEECH_OFF = 0.5, 0.35   # гистерезис, как у Silero по умолчанию
START_FRAMES = 3            # ~100 мс речи подряд — речь началась
PRE_ROLL = 0.5              # с до начала речи, которые идут в реплику
MAX_WINDOW = 15.0           # с непрочитанного окна — дальше закрываем согласованное
SENTENCE_END = re.compile(r"[.!?…]$")

# Слова, на которых русская фраза не кончается: предлоги, союзы, заминки.
# Если распознанная реплика обрывается на таком слове, модели конца реплики
# не верим и ждём паузу до max_pause.
DANGLING = frozenset("""
в во на с со к ко по о об обо от из у за над под про для без до при через между перед
и а но или либо что чтобы как если когда потому поэтому так то ли будто хотя пока
это мой моя мое моё мои твой наш ваш их его её ее какой какая какое какие который
которая которое которые очень самый не ни
э ээ эээ эм мм м ну вот типа короче значит
""".split())


class StreamVAD:
    """Silero, frame by frame. Each frame sees only 64 samples before it, so
    frames can be scored as they arrive with the same result as all at once."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._tail = np.zeros(CONTEXT, np.float32)
        self.rest = np.zeros(0, np.float32)          # хвост короче кадра

    def feed(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.rest, x])
        n = len(x) // FRAME
        self.rest = x[n * FRAME:]
        if n == 0:
            return np.zeros(0, np.float32)
        frames = x[:n * FRAME].reshape(n, FRAME)
        ctx = np.vstack([self._tail[None, :], frames[:-1, -CONTEXT:]])
        self._tail = frames[-1, -CONTEXT:].copy()
        batch = np.concatenate([ctx, frames], axis=1).astype(np.float32)
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        out, _, _ = self._model.session.run(None, {"input": batch, "h": h, "c": c})
        return np.asarray(out, np.float32).reshape(-1)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower().replace("ё", "е"))


def dangling(text: str) -> bool:
    """Does the text stop where a Russian phrase cannot?"""
    text = text.strip()
    if not text:
        return False
    if text[-1] in ",:;-—(«\"" or text.endswith("..."):   # многоточие Whisper ставит на заминке
        return True
    return _norm(text.split()[-1]) in DANGLING


@dataclass
class Word:
    start: float        # с от начала сессии
    end: float
    text: str


@dataclass
class Turn:
    """Where the endpointing stands. Touched from the event loop thread only."""
    active: bool = False
    start: float = 0.0              # начало реплики с учётом PRE_ROLL
    speaking: bool = False
    speech_run: int = 0             # кадров речи подряд
    silence_run: int = 0            # кадров тишины подряд
    last_speech: float = 0.0        # конец последнего кадра речи, с
    pause_id: int = 0               # растёт с каждой новой паузой
    checked: int = -1               # pause_id, для которого модель уже спрошена
    ending: bool = False
    finals: list[str] = field(default_factory=list)
    probability: float | None = None


class Session:
    def __init__(self, ws: WebSocket, stt, turn_model, *, rate: int, language: str | None,
                 ctx: contexts.Resolved, interval: float, threshold: float,
                 min_pause: float, max_pause: float):
        import soxr

        self.ws, self.stt, self.turn_model = ws, stt, turn_model
        self.language, self.ctx = language, ctx
        self.interval, self.threshold = interval, threshold
        self.min_pause, self.max_pause = min_pause, max_pause
        # потоковый пересчёт частоты: кусочки по 100 мс не щёлкают на стыках
        self._resampler = soxr.ResampleStream(rate, SR, 1, dtype="float32") if rate != SR else None
        self.vad = StreamVAD()

        self.audio = np.zeros(0, np.float32)     # 16 кГц, начиная с отсчёта `base`
        self.base = 0
        self.total = 0                           # отсчётов с начала сессии
        self.commit_at = 0                       # отсчёт, с которого текст ещё не окончателен
        self.hyp: list[Word] = []                # прошлое чтение открытого окна
        self.last_partial = ""
        self.said: list[str] = []                # всё окончательное за сессию
        self.turn = Turn()
        self.speech_since_pass = 0               # отсчётов речи после последнего чтения
        self.reading = asyncio.Lock()            # чтение окна и закрытие реплики — по одному
        self.wake = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()

    # --------------------------------------------------------------- вывод

    async def send(self, **event) -> None:
        for k in ("at", "start", "end"):
            if isinstance(event.get(k), float):
                event[k] = round(event[k], 2)
        async with self._send_lock:
            await self.ws.send_json(event)

    def spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    # ------------------------------------------------------------- звук

    def _slice(self, start: int, end: int | None = None) -> np.ndarray:
        end = self.total if end is None else end
        return self.audio[max(0, start - self.base):max(0, end - self.base)]

    def _trim(self) -> None:
        """Keep only what reading and turn detection may still need."""
        keep = min(self.commit_at, self.total - SR)
        if self.turn.active:
            keep = min(keep, max(int(self.turn.start * SR), self.total - 8 * SR))
        if keep > self.base:
            self.audio = self.audio[keep - self.base:]
            self.base = keep

    async def feed(self, pcm: np.ndarray) -> None:
        x = self._resampler.resample_chunk(pcm) if self._resampler else pcm
        if not len(x):
            return
        self.audio = np.concatenate([self.audio, x])
        self.total += len(x)
        await self._endpoint(self.vad.feed(x))
        if not self.turn.active:
            # тишина между репликами: окно чтения идёт следом, с запасом PRE_ROLL
            self.commit_at = max(self.commit_at, self.total - int(PRE_ROLL * SR))
            self.hyp = []
        self._trim()

    async def _endpoint(self, probs: np.ndarray) -> None:
        t = self.turn
        frames_done = (self.total - len(self.vad.rest)) // FRAME     # кадры идут без пропусков
        for i, p in enumerate(probs):
            at = (frames_done - len(probs) + i + 1) * FRAME / SR      # конец этого кадра
            if p >= SPEECH_ON or (t.speaking and p >= SPEECH_OFF):
                t.speech_run += 1
                t.silence_run = 0
                t.last_speech = at
                self.speech_since_pass += FRAME
                if not t.speaking and t.speech_run >= START_FRAMES:
                    t.speaking = True
                    if not t.active and not t.ending:
                        began = at - START_FRAMES * FRAME / SR
                        t.active, t.finals, t.probability = True, [], None
                        t.start = max(0.0, began - PRE_ROLL)
                        self.commit_at = min(self.commit_at, int(t.start * SR))
                        await self.send(type="speech_start", at=began)
                continue
            t.speech_run = 0
            t.silence_run += 1
            if t.speaking:
                t.speaking = False
                t.pause_id += 1
            pause = t.silence_run * FRAME / SR
            if t.active and not t.ending:
                if pause >= self.min_pause and t.checked != t.pause_id:
                    t.checked = t.pause_id
                    self.spawn(self._check_turn(t, t.pause_id))
                if pause >= self.max_pause:
                    t.ending = True
                    self.spawn(self._end_turn(t, "silence"))
        if self.speech_since_pass >= self.interval * SR:
            self.wake.set()

    # ---------------------------------------------------------- реплика

    async def _check_turn(self, t: Turn, pause_id: int) -> None:
        audio = self._slice(int(t.start * SR))
        p = await asyncio.to_thread(self.turn_model.probability, audio)
        complete = p >= self.threshold
        await self.send(type="turn_check", at=self.total / SR, probability=round(p, 3),
                        complete=complete)
        if complete and t.pause_id == pause_id and not t.speaking and t.active and not t.ending:
            t.probability = p
            await self._end_turn(t, "model", pause_id)

    async def _end_turn(self, t: Turn, reason: str, pause_id: int | None = None) -> None:
        """Read what is left of the turn and close it. With reason "model" the
        text may still veto: a turn ending on "и" or "в" is not over."""
        async with self.reading:
            if not t.active or (reason == "model" and (t.pause_id != pause_id or t.speaking)):
                return
            t.ending = True
            upto = self.total
            text = await self._read(self.commit_at, upto)
            if reason == "model":
                if t.pause_id != pause_id or t.speaking:
                    t.ending = False                     # заговорил, пока читали
                    return
                whole = " ".join(t.finals + [text]).strip()
                if dangling(whole):
                    t.ending = False
                    await self.send(type="turn_check", at=self.total / SR,
                                    probability=round(t.probability or 0, 3),
                                    complete=False, veto=whole.split()[-1])
                    return
            start = self.commit_at / SR
            self._commit(upto, text, t)
            if text:
                await self.send(type="final", text=text, start=start, end=upto / SR)
            await self.send(type="turn_end", text=" ".join(t.finals).strip(), start=t.start,
                            end=t.last_speech, reason=reason,
                            probability=None if t.probability is None else round(t.probability, 3))
            t.active = False                             # старый объект больше не закроют
            self.turn = Turn(speaking=t.speaking, speech_run=t.speech_run,
                             silence_run=t.silence_run, last_speech=t.last_speech,
                             pause_id=t.pause_id, checked=t.pause_id)

    def _commit(self, upto: int, text: str, t: Turn) -> None:
        self.commit_at = upto
        self.hyp, self.last_partial = [], ""
        if text:
            t.finals.append(text)
            self.said.append(text)

    # ------------------------------------------------------------- чтение

    def _prompt(self) -> str | None:
        # Хвост уже сказанного подсказкой не идёт: на коротком окне Whisper
        # охотно переписывает подсказку вместо звука, и фраза приходит дважды —
        # а то и раньше, чем прозвучала. Остаётся только контекст пользователя.
        return self.ctx.prompt

    async def _read(self, start: int, end: int) -> str:
        audio = self._slice(start, end)
        if len(audio) < 0.3 * SR:
            return ""
        tr = await asyncio.to_thread(
            self.stt.transcribe, audio, language=self.language, prompt=self._prompt(),
            hotwords=self.ctx.hotwords, beam_size=5,
            condition_on_previous_text=False, vad_filter=True)
        return self.ctx.fix(tr.text.strip())

    async def reader_pass(self) -> None:
        """One reading of the open window; commit what two readings agree on."""
        async with self.reading:
            t = self.turn
            if not t.active or t.ending:
                return
            self.speech_since_pass = 0
            start, end = self.commit_at, self.total
            audio = self._slice(start, end)
            if len(audio) < 0.5 * SR:
                return
            tr = await asyncio.to_thread(
                self.stt.transcribe, audio, language=self.language, prompt=self._prompt(),
                hotwords=self.ctx.hotwords, beam_size=2, word_timestamps=True,
                condition_on_previous_text=False, vad_filter=True)
            if self.commit_at != start or self.turn is not t or not t.active:
                return                                  # реплика закрылась, пока читали
            t0 = start / SR
            words = [Word(t0 + w.start, t0 + w.end, w.word)
                     for s in tr.segments for w in s.words]

            agreed = 0
            for a, b in zip(self.hyp, words):
                if _norm(a.text) != _norm(b.text):
                    break
                agreed += 1
            cut = -1
            for i in range(agreed - 1, -1, -1):          # конец последнего согласованного предложения
                if SENTENCE_END.search(words[i].text.strip()):
                    cut = i
                    break
            if cut < 0 and agreed and (end - start) / SR > MAX_WINDOW:
                cut = agreed - 1                        # пауз и точек нет, а окно растёт
            if cut >= 0:
                nxt = words[cut + 1].start if cut + 1 < len(words) else words[cut].end + 0.1
                at = min(end, int((words[cut].end + nxt) / 2 * SR))
                text = self.ctx.fix("".join(w.text for w in words[:cut + 1]).strip())
                self._commit(at, text, t)
                await self.send(type="final", text=text, start=t0, end=at / SR)
                words = words[cut + 1:]
            self.hyp = words
            partial = self.ctx.fix("".join(w.text for w in words).strip())
            if partial and partial != self.last_partial:
                self.last_partial = partial
                await self.send(type="partial", text=partial, start=self.commit_at / SR)

    async def reader_loop(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            await self.reader_pass()

    async def finish(self) -> None:
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
        if self.turn.active:
            await self._end_turn(self.turn, "end")
        await self.send(type="done", text=" ".join(self.said).strip(),
                        duration=round(self.total / SR, 2))


def attach(app, stt, turn_model):
    @app.websocket("/v1/audio/transcriptions/stream")
    async def live(ws: WebSocket, language: str | None = None, prompt: str | None = None,
                   hotwords: str | None = None, context: str | None = None,
                   sample_rate: int = SR, interval: float = 0.7,
                   turn_threshold: float = 0.5, min_pause: float = 0.2,
                   max_pause: float = 2.0):
        await ws.accept()
        try:
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("sample_rate out of range")
            try:
                ctx = contexts.resolve(context, prompt, hotwords)
            except KeyError:
                raise ValueError(f"unknown context '{context}'") from None
        except ValueError as e:
            await ws.send_json({"type": "error", "error": str(e)})
            await ws.close(code=1003)
            return

        s = Session(ws, stt, turn_model, rate=sample_rate, language=language or None, ctx=ctx,
                    interval=min(5.0, max(0.3, interval)),
                    threshold=min(0.99, max(0.01, turn_threshold)),
                    min_pause=min(2.0, max(0.1, min_pause)),
                    max_pause=min(10.0, max(0.3, max_pause)))
        await ws.send_json({"type": "ready", "sample_rate": sample_rate, "context": context,
                            "turn": {"threshold": s.threshold, "min_pause": s.min_pause,
                                     "max_pause": s.max_pause}})
        # первая сессия ждёт загрузки весов; звук тем временем копится в сокете
        await asyncio.gather(asyncio.to_thread(stt.load), asyncio.to_thread(turn_model.load))

        reading = asyncio.create_task(s.reader_loop())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes"):
                    data = msg["bytes"][:len(msg["bytes"]) // 2 * 2]
                    await s.feed(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"])
                    except ValueError:
                        cmd = {}
                    if cmd.get("type") in ("end", "stop"):
                        break
            reading.cancel()
            await s.finish()
            await ws.close()
        except WebSocketDisconnect:
            pass
        except Exception as e:                          # noqa: BLE001 — уходит клиенту
            try:
                await ws.send_json({"type": "error", "error": str(e)})
                await ws.close(code=1011)
            except Exception:                           # noqa: BLE001
                pass
        finally:
            reading.cancel()
            for t in list(s.tasks):
                t.cancel()
