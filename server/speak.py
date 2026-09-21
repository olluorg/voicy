"""Speech as it is synthesised, for a voice agent that should not go silent.

The synthesiser returns a piece of text only once the whole of it is generated.
Waiting for the whole answer makes the listener wait for its last sentence
before hearing the first, so the text is cut into pieces and each piece is sent
as soon as it is ready. With Qwen3-TTS on an RTX 3080 synthesis runs at ×1.5
(engines/tts_fast.py), so after the first piece the next one is ready before
the current one finishes playing.

The cuts still go where a listener expects a pause. On a card slower than real
time there will be gaps between pieces; at sentence boundaries they sound like
breathing, in the middle of a phrase they sound like a stutter. So:

  - the first piece ends at the first clause boundary (a comma after three
    words or more) — the first sound is what the listener waits for;
  - every later piece ends at the end of a sentence;
  - a short sentence is joined to the next one when both are already known,
    because each call has a fixed cost;
  - a sentence longer than MAX_CHUNK is cut at its last clause boundary.

Two ways in:

  POST /v1/audio/speech with `stream_format` — OpenAI's parameter. "audio" sends
  the bytes as they are made (pcm, wav, mp3), "sse" sends `speech.audio.delta`
  events with base64 audio and ends with `speech.audio.done`.

  WS /v1/audio/speech/stream — for an agent that has the text only as its LLM
  produces it:

    client → {"type": "text", "text": "..."}   — more text, any size, even a token
    client → {"type": "flush"}                 — speak what is buffered now
    client → {"type": "cancel"}                — the user interrupted: drop everything
    client → {"type": "end"}                   — no more text: finish and close
    server → {"type": "ready", "sample_rate", "voice"}
    server → {"type": "synthesizing", "index", "text"}
    server → {"type": "audio", "index", "text", "seconds", "synth_seconds"}
             followed by one binary frame: PCM s16le mono at `sample_rate`
    server → {"type": "cancelled"}
    server → {"type": "done", "chunks", "seconds"}

Streaming does not go through the job queue — an agent's reply waiting behind a
long file would be no reply. It takes the synthesiser's lock piece by piece, so a
running synthesis job still delays it by the rest of that job.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import struct
import time
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import audio_io
import textprep
import voices as voice_registry
from engines.base import Unsupported

MAX_CHUNK = 200             # символов — длиннее режем по запятой
MIN_CHUNK = 25              # символов — короче склеиваем с соседом, если он уже есть
FIRST_CLAUSE_WORDS = 3      # первый кусок можно отрезать по запятой после стольких слов
STREAM_FORMATS = ("pcm", "wav", "mp3")

# Конец предложения — знак, пробел и слово не со строчной буквы: так «т. е.»,
# «см. выше» и «3.5» не режутся. Поток ради этого ждёт первую букву следующего
# предложения — один токен.
SENTENCE = re.compile(r"[.!?…]+[\"»)\]]*(?=\s+[^\sa-zа-яё])|\n")
CLAUSE = re.compile(r"[,;:—–](?=\s)")


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------- нарезка

class Segmenter:
    """Text in any pieces — tokens, lines, whole paragraphs; speakable chunks out."""

    def __init__(self):
        self.buf = ""
        self.emitted = 0            # кусков отдано наружу
        self.cuts = 0               # кусков отрезано, включая ещё не отданные

    def push(self, text: str) -> list[str]:
        self.buf += text
        return self._take(final=False)

    def flush(self) -> list[str]:
        return self._take(final=True)

    def split(self, text: str) -> list[str]:
        """The whole text at once: short sentences join their neighbours on both sides."""
        self.buf += text
        return self._take(final=True)

    def reset(self) -> None:
        self.buf, self.emitted, self.cuts = "", 0, 0

    def _cut(self, final: bool) -> str | None:
        m = SENTENCE.search(self.buf)
        if self.cuts == 0:
            # первый кусок — до первой запятой, если она раньше конца предложения
            for c in CLAUSE.finditer(self.buf, 0, m.start() if m else len(self.buf)):
                if len(self.buf[:c.start()].split()) >= FIRST_CLAUSE_WORDS:
                    return self._split(c.end())
        if m:
            return self._split(m.end())
        if len(self.buf) > MAX_CHUNK:
            head = self.buf[:MAX_CHUNK]
            cuts = [c.end() for c in CLAUSE.finditer(head)]
            return self._split(cuts[-1] if cuts else (head.rfind(" ") + 1 or MAX_CHUNK))
        if final and self.buf.strip():
            return self._split(len(self.buf))
        return None

    def _split(self, at: int) -> str:
        piece, self.buf = self.buf[:at], self.buf[at:]
        if piece.strip():
            self.cuts += 1
        return piece

    def _take(self, final: bool) -> list[str]:
        pieces = []
        while (p := self._cut(final)) is not None:
            if p.strip():
                pieces.append(p.strip())
        out: list[str] = []
        for p in pieces:
            # короткое склеиваем со следующим, раз оно уже есть; первый кусок
            # потока не трогаем — он ради того, чтобы звук начался раньше
            stream_first = self.emitted == 0 and len(out) == 1
            if out and len(out[-1]) < MIN_CHUNK and not stream_first:
                out[-1] = f"{out[-1]} {p}"
            else:
                out.append(p)
        self.emitted += len(out)
        return out


# ---------------------------------------------------------------------- синтез

@dataclass
class Options:
    voice: voice_registry.Voice
    language: str | None = None
    speed: float = 1.0
    prepare: bool = False
    legato: bool = False
    seed: int | None = None


def synthesize(tts, text: str, o: Options, cancelled) -> tuple[np.ndarray, int, float]:
    """One chunk. `cancelled()` is polled whenever the engine reports progress."""
    if o.prepare or o.legato:
        text = textprep.prepare(text, o.prepare, o.legato)["text"]

    def on_progress(_) -> None:
        if cancelled():
            raise Cancelled()

    started = time.perf_counter()
    out = tts.speak(text, o.voice.path, o.voice.text, language=o.language,
                    seed=o.seed, on_progress=on_progress)
    audio = audio_io.stretch(out.audio, out.sample_rate, o.speed) if o.speed != 1.0 else out.audio
    return audio, out.sample_rate, time.perf_counter() - started


def _wav_header(sr: int) -> bytes:
    """A wav header for a stream of unknown length: sizes set to the maximum,
    which players read as "until the data ends"."""
    unknown = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", unknown) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", unknown))


def pick_voice(tts, name: str | None) -> voice_registry.Voice:
    v = voice_registry.get(name) if name else voice_registry.default()
    if v is None:
        raise HTTPException(400, f"unknown voice '{name}'" if name else
                            "no voice available — add one via /v1/voices")
    if tts.needs_reference_text and not v.text:
        raise HTTPException(400, f"voice '{v.name}' has no reference transcript")
    return v


# ------------------------------------------------------------------------ HTTP

def http_stream(tts, req) -> StreamingResponse:
    """`/v1/audio/speech` with `stream_format` set."""
    fmt = (req.response_format or "wav").lower()
    if req.stream_format == "audio" and fmt not in STREAM_FORMATS:
        raise HTTPException(400, f"stream_format=audio supports {', '.join(STREAM_FORMATS)}; "
                                 f"use sse for {fmt}")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    o = Options(voice=pick_voice(tts, req.voice), language=tts.language(req.language),
                speed=req.speed,
                prepare=req.prepare, legato=req.legato, seed=req.seed)
    seg = Segmenter()
    chunks = seg.split(req.input)
    state = {"stop": False}

    async def gen():
        total, t0 = 0.0, time.perf_counter()
        try:
            if req.stream_format == "audio" and fmt == "wav":
                yield _wav_header(tts.sample_rate)
            for text in chunks:
                audio, sr, _ = await asyncio.to_thread(
                    synthesize, tts, text, o, lambda: state["stop"])
                total += len(audio) / sr
                if req.stream_format == "audio":
                    if fmt in ("pcm", "wav"):
                        yield audio_io.to_pcm16(audio)
                    else:
                        yield (await asyncio.to_thread(audio_io.encode, audio, sr, fmt))[0]
                else:
                    data, _ = await asyncio.to_thread(audio_io.encode, audio, sr, fmt)
                    event = {"type": "speech.audio.delta", "text": text,
                             "audio": base64.b64encode(data).decode()}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if req.stream_format == "sse":
                done = {"type": "speech.audio.done", "chunks": len(chunks),
                        "seconds": round(total, 2),
                        "generation_seconds": round(time.perf_counter() - t0, 2)}
                yield f"data: {json.dumps(done)}\n\n"
        finally:
            state["stop"] = True                     # клиент ушёл — дальше не синтезируем

    if req.stream_format == "sse":
        media = "text/event-stream"
    else:
        media = "audio/pcm" if fmt == "pcm" else audio_io.CONTENT_TYPES[fmt]
    return StreamingResponse(gen(), media_type=media, headers={
        "X-Voice": o.voice.name, "X-Chunks": str(len(chunks)),
        "X-Sample-Rate": str(tts.sample_rate), "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------- WebSocket

def attach(app, tts):
    @app.websocket("/v1/audio/speech/stream")
    async def speak(ws: WebSocket, voice: str | None = None, language: str | None = None,
                    speed: float = 1.0, sample_rate: int | None = None,
                    prepare: bool = False, legato: bool = False, seed: int | None = None):
        await ws.accept()
        sample_rate = sample_rate or tts.sample_rate
        try:
            if not 8000 <= sample_rate <= 48000:
                raise HTTPException(400, "sample_rate must be 8000–48000")
            o = Options(voice=pick_voice(tts, voice), language=tts.language(language),
                        speed=min(1.2, max(0.8, speed)) if speed != 1.0 else 1.0,
                        prepare=prepare, legato=legato, seed=seed)
        except (HTTPException, Unsupported) as e:
            await ws.send_json({"type": "error",
                                "error": e.detail if isinstance(e, HTTPException) else str(e)})
            await ws.close(code=1003)
            return

        seg = Segmenter()
        todo: asyncio.Queue = asyncio.Queue()
        epoch = {"n": 0}                 # отмена увеличивает эпоху; старые куски выбрасываются
        send_lock = asyncio.Lock()
        END = object()

        async def send_json(event: dict) -> None:
            async with send_lock:
                await ws.send_json(event)

        async def reader() -> None:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    epoch["n"] += 1
                    await todo.put(None)
                    return
                raw = msg.get("text")
                if raw is None:
                    continue
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    cmd = {"type": "text", "text": raw}   # просто текст — тоже текст
                kind = cmd.get("type")
                if kind == "text":
                    for p in seg.push(str(cmd.get("text", ""))):
                        await todo.put((epoch["n"], p))
                elif kind == "flush":
                    for p in seg.flush():
                        await todo.put((epoch["n"], p))
                elif kind == "cancel":
                    epoch["n"] += 1
                    seg.reset()
                    while not todo.empty():
                        todo.get_nowait()
                    await send_json({"type": "cancelled"})
                elif kind == "end":
                    for p in seg.flush():
                        await todo.put((epoch["n"], p))
                    await todo.put(END)
                    return

        await send_json({"type": "ready", "sample_rate": sample_rate, "voice": o.voice.name})
        reading = asyncio.create_task(reader())
        index, seconds = 0, 0.0
        try:
            while True:
                item = await todo.get()
                if item is None:
                    return                               # клиент ушёл
                if item is END:
                    await send_json({"type": "done", "chunks": index,
                                     "seconds": round(seconds, 2)})
                    await ws.close()
                    return
                mine, text = item
                if mine != epoch["n"]:
                    continue
                await send_json({"type": "synthesizing", "index": index, "text": text})
                try:
                    audio, sr, took = await asyncio.to_thread(
                        synthesize, tts, text, o, lambda: mine != epoch["n"])
                except Cancelled:
                    continue
                if mine != epoch["n"]:
                    continue                             # отменили, пока кодировали
                if sample_rate != sr:
                    import soxr
                    audio = soxr.resample(audio, sr, sample_rate).astype(np.float32)
                dur = len(audio) / sample_rate
                seconds += dur
                async with send_lock:
                    await ws.send_json({"type": "audio", "index": index, "text": text,
                                        "seconds": round(dur, 2),
                                        "synth_seconds": round(took, 2)})
                    await ws.send_bytes(audio_io.to_pcm16(audio))
                index += 1
        except WebSocketDisconnect:
            pass
        except Exception as e:                           # noqa: BLE001 — уходит клиенту
            try:
                await send_json({"type": "error", "error": str(e)})
                await ws.close(code=1011)
            except Exception:                            # noqa: BLE001
                pass
        finally:
            epoch["n"] += 1
            reading.cancel()
