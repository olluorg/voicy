"""Local speech server: OpenAI-compatible audio API plus a web console.

The point is drop-in compatibility — anything that already speaks to
`/v1/audio/speech` and `/v1/audio/transcriptions` should work against this by
changing the base URL and nothing else. On top of that there are a few endpoints
the OpenAI API has no equivalent for: listing and adding cloned voices, and the
text preparation this repository measured into existence.

Everything runs locally. No key leaves the machine, because there is no key.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import audio_io
import auth
import contexts
import errors
import jobs
import live
import speak
import textprep
import transcripts
from schemas import SpeechRequest
import voices as voice_registry
import engines
from engines.device import describe, has_cuda

HERE = Path(__file__).parent

app = FastAPI(title="voicy", version="1.5.0",
              description="Локальный речевой сервер с OpenAI-совместимым API")
# Модели — за интерфейсами engines/base.py; какую взять, решает окружение
# (TTS_ENGINE, STT_ENGINE, …), а сервер про их устройство не знает.
tts = engines.create("tts")
stt = engines.create("stt")
turn = engines.create("turn")
vad = engines.create("vad")


# ----------------------------------------------------------------- OpenAI: TTS
#
# Совместимые маршруты тоже становятся заданиями и ждут в общей очереди —
# снаружи они по-прежнему синхронны, а id задания приходит в X-Job-Id.
# Исключение — синтез с stream_format: он идёт кусками мимо очереди (speak.py).

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request):
    if req.stream_format:
        return speak.http_stream(tts, req)
    job, _ = jobs.submit_speech(req, request)
    await jobs.wait(job)
    r = job.result
    return Response(content=r["data"], media_type=r["content_type"], headers={
        "X-Job-Id": job.id,
        "X-Voice": r["voice"],
        "X-Audio-Seconds": f"{r['seconds']:.2f}",
        "X-Generation-Seconds": f"{job.finished - job.started:.2f}",
        "X-Queue-Seconds": f"{job.started - job.created:.2f}",
    })


# ----------------------------------------------------------------- OpenAI: STT

def _delta_stream(job):
    """`stream=true` in OpenAI's shape: text deltas, then the whole text.

    OpenAI streams only for its newer models; here the recogniser's segments are
    the deltas, each with the time span it covers as an extra field.
    """
    async def gen():
        q = job.subscribe()
        first = True
        try:
            while not job.done:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                stage = event.get("stage")
                if stage == "recognition" and event.get("text"):
                    delta = event["text"] if first else " " + event["text"]
                    first = False
                    payload = {"type": "transcript.text.delta", "delta": delta,
                               "end": event["position"], "duration": event["total"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif stage in ("error", "cancelled"):
                    break
            if job.state == "done":
                payload = {"type": "transcript.text.done", "text": job.result["text"],
                           "language": job.result["language"],
                           "duration": job.result["duration"], "job_id": job.id}
            else:
                payload = {"type": "error", "error": {"message": job.error or job.state}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            job.unsubscribe(q)
            if not job.done:                        # клиент ушёл — работа никому не нужна
                jobs.queue.cancel(job)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "X-Job-Id": job.id})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: str | None = Form(None),
    stream: bool = Form(False),
    context: str | None = Form(None),
    hotwords: str | None = Form(None),
):
    want_words = bool(timestamp_granularities and "word" in timestamp_granularities) \
        or response_format == "verbose_json"
    job = jobs.submit_transcription(await file.read(), file.filename, request,
                                    language=language, prompt=prompt,
                                    temperature=temperature, word_timestamps=want_words,
                                    context=context, hotwords=hotwords)
    if stream:
        return _delta_stream(job)
    await jobs.wait(job)
    resp = transcripts.render(job.result, response_format)
    resp.headers["X-Job-Id"] = job.id
    return resp


@app.post("/v1/audio/translations")
async def translations(request: Request, file: UploadFile = File(...),
                       model: str = Form("whisper-1"), prompt: str | None = Form(None),
                       response_format: str = Form("json"), temperature: float = Form(0.0)):
    """OpenAI translates to English here — for a recogniser with the
    "translate" feature; any other answers 400."""
    job = jobs.submit_transcription(await file.read(), file.filename, request,
                                    task="translate", prompt=prompt,
                                    temperature=temperature)
    await jobs.wait(job)
    resp = transcripts.render(job.result, response_format)
    resp.headers["X-Job-Id"] = job.id
    return resp


# -------------------------------------------------------------- OpenAI: models

@app.get("/v1/models")
def models():
    now = int(time.time())
    ids = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "whisper-1"]
    return {"object": "list",
            "data": [{"id": i, "object": "model", "created": now,
                      "owned_by": "voicy"} for i in ids]}


# ------------------------------------------------------------------ extensions

@app.get("/v1/voices")
def get_voices():
    v = voice_registry.default()
    return {"object": "list", "default": v.name if v else None,
            "data": [x.as_dict() for x in voice_registry.list_voices()]}


@app.post("/v1/voices")
async def add_voice(file: UploadFile = File(...), name: str = Form(...),
                    text: str = Form(""), note: str = Form(""),
                    replace: bool = Form(False)):
    """Register a reference clip. An empty transcript is filled in by the recogniser.

    The transcript matters more than it looks: the same clip with a transcript cut
    mid-phrase measured 7.3% error against 1.0% when the two agreed exactly.
    """
    try:
        voice_registry.check_name(name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if voice_registry.get(name) and not replace:
        raise HTTPException(409, f"voice '{name}' exists — pass replace=true to overwrite")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "file is empty")
    try:
        audio = audio_io.decode(raw, voice_registry.SAMPLE_RATE)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None

    seconds = len(audio) / voice_registry.SAMPLE_RATE
    # пределы — у движка синтеза: образец, годный одной модели, другой может не подойти
    lo, hi = tts.reference_best
    lo_ok, hi_ok = tts.reference_seconds
    if not lo_ok <= seconds <= hi_ok:
        raise HTTPException(400, f"sample is {seconds:.1f} s; {tts.name} needs "
                                 f"{lo_ok:.0f}–{hi_ok:.0f} s, best {lo:.0f}–{hi:.0f}")
    warnings = []
    if not lo <= seconds <= hi:
        warnings.append(f"sample is {seconds:.1f} s; clones are best from {lo:.0f}–{hi:.0f} s")

    if not text.strip():
        pcm16k = audio_io.resample(audio, voice_registry.SAMPLE_RATE, 16000)
        text = (await asyncio.to_thread(stt.transcribe, pcm16k)).text
    v = voice_registry.add(name, audio_io.to_wav(audio, voice_registry.SAMPLE_RATE),
                           text.strip(), note)
    return {**v.as_dict(), "seconds": round(seconds, 2), "warnings": warnings}


@app.get("/v1/contexts")
def get_contexts():
    return {"object": "list", "data": [c.as_dict() for c in contexts.list_contexts()]}


@app.get("/v1/contexts/{name}")
def get_context(name: str):
    c = contexts.get(name)
    if c is None:
        raise HTTPException(404, "unknown context")
    return c.as_dict()


@app.put("/v1/contexts/{name}")
def put_context(name: str, payload: dict):
    """{"prompt": "...", "hotwords": ["Kafka", ...], "replacements": {"кавка": "Kafka"}, "note": "..."}"""
    hot = payload.get("hotwords") or []
    rep = payload.get("replacements") or {}
    if not isinstance(hot, list) or not isinstance(rep, dict):
        raise HTTPException(400, "hotwords must be a list, replacements an object")
    try:
        c = contexts.put(name, note=str(payload.get("note", "")),
                         prompt=str(payload.get("prompt", "")),
                         hotwords=[str(h) for h in hot],
                         replacements={str(k): str(v) for k, v in rep.items()})
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return c.as_dict()


@app.delete("/v1/contexts/{name}")
def delete_context(name: str):
    if name == contexts.BUILTIN:
        raise HTTPException(400, f"'{name}' is built in")
    if not contexts.delete(name):
        raise HTTPException(404, "unknown context")
    return {"deleted": name}


@app.post("/v1/text/prepare")
def prepare_text(payload: dict):
    text = (payload or {}).get("text", "")
    return textprep.prepare(text,
                            use_dictionary=bool(payload.get("dictionary", True)),
                            use_legato=bool(payload.get("legato", False)))


jobs.attach(app, tts, stt)
live.attach(app, stt, turn, vad)
speak.attach(app, tts)
errors.attach(app)
app.add_middleware(auth.Middleware)


@app.get("/health")
def health():
    return {"status": "ok",
            "tts": tts.status(),
            "stt": {**stt.status(), "features": sorted(stt.features)},
            "turn": turn.status(),
            "vad": {"engine": vad.name},
            "cuda": has_cuda(),
            "auth": auth.enabled(),
            "device": describe(),
            "voices": [v.name for v in voice_registry.list_voices()],
            "queue": {"pending": jobs.queue.pending(), "max": jobs.queue.max_pending,
                      "running": jobs.queue.current.id if jobs.queue.current else None}}


# ------------------------------------------------------------------------- UI

@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
