"""Local speech server: OpenAI-compatible audio API plus a web console.

The point is drop-in compatibility — anything that already speaks to
`/v1/audio/speech` and `/v1/audio/transcriptions` should work against this by
changing the base URL and nothing else. On top of that there are a few endpoints
the OpenAI API has no equivalent for: listing and adding cloned voices, and the
text preparation this repository measured into existence.

Everything runs locally. No key leaves the machine, because there is no key.
"""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import audio_io
import textprep
import voices as voice_registry
from engines.device import describe, has_cuda
from engines.stt_whisper import WhisperSTT
from engines.tts_qwen import QwenTTS

HERE = Path(__file__).parent
TTS_MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
STT_MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")

app = FastAPI(title="engineering-tts-lab", version="1.0.0",
              description="Локальный речевой сервер с OpenAI-совместимым API")
tts = QwenTTS(TTS_MODEL)
stt = WhisperSTT(STT_MODEL)


# ----------------------------------------------------------------- OpenAI: TTS

class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    language: str = "Russian"
    # расширения, которых нет у OpenAI
    prepare: bool = Field(default=False, description="применить словарь произношений")
    legato: bool = Field(default=False, description="убрать запятые внутри коротких фраз")
    seed: int | None = None


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    v = voice_registry.get(req.voice) if req.voice else voice_registry.default()
    if v is None:
        raise HTTPException(400, "no voice available — add one via /v1/voices")
    if not v.text:
        raise HTTPException(400, f"voice '{v.name}' has no reference transcript")

    text = req.input
    if req.prepare or req.legato:
        text = textprep.prepare(text, req.prepare, req.legato)["text"]

    started = time.perf_counter()
    out = tts.speak(text, v.path, v.text, language=req.language, seed=req.seed)
    audio = audio_io.stretch(out.audio, out.sample_rate, req.speed) if req.speed != 1.0 else out.audio
    data, content_type = audio_io.encode(audio, out.sample_rate, req.response_format)

    return Response(content=data, media_type=content_type, headers={
        "X-Voice": v.name,
        "X-Audio-Seconds": f"{len(audio) / out.sample_rate:.2f}",
        "X-Generation-Seconds": f"{time.perf_counter() - started:.2f}",
    })


# ----------------------------------------------------------------- OpenAI: STT

def _srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def _format_transcript(tr, response_format: str):
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return Response(tr.text, media_type="text/plain; charset=utf-8")
    if fmt == "srt":
        lines = []
        for i, s in enumerate(tr.segments, 1):
            lines += [str(i), f"{_srt_time(s.start)} --> {_srt_time(s.end)}", s.text, ""]
        return Response("\n".join(lines), media_type="text/plain; charset=utf-8")
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for s in tr.segments:
            lines += [f"{_srt_time(s.start).replace(',', '.')} --> "
                      f"{_srt_time(s.end).replace(',', '.')}", s.text, ""]
        return Response("\n".join(lines), media_type="text/vtt; charset=utf-8")
    if fmt == "verbose_json":
        return JSONResponse({
            "task": "transcribe", "language": tr.language, "duration": tr.duration,
            "text": tr.text,
            "segments": [{"id": i, "start": s.start, "end": s.end, "text": s.text,
                          "words": [{"word": w.word, "start": w.start, "end": w.end}
                                    for w in s.words]}
                         for i, s in enumerate(tr.segments)],
        })
    return JSONResponse({"text": tr.text})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: str | None = Form(None),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "file is empty")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"in{suffix}")
        with open(path, "wb") as f:
            f.write(raw)
        want_words = bool(timestamp_granularities and "word" in timestamp_granularities) \
            or response_format == "verbose_json"
        tr = stt.transcribe(path, language=language, prompt=prompt,
                            temperature=temperature, word_timestamps=want_words)
    return _format_transcript(tr, response_format)


@app.post("/v1/audio/translations")
async def translations(file: UploadFile = File(...), model: str = Form("whisper-1"),
                       prompt: str | None = Form(None), response_format: str = Form("json"),
                       temperature: float = Form(0.0)):
    """OpenAI translates to English here. Whisper does that with task=translate;
    faster-whisper exposes it the same way, so the contract is honoured."""
    raw = await file.read()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "in" + (Path(file.filename or "a.wav").suffix or ".wav"))
        with open(path, "wb") as f:
            f.write(raw)
        stt.load()
        with stt._lock:                                    # noqa: SLF001 — тот же GPU
            segments, info = stt._model.transcribe(path, task="translate",  # noqa: SLF001
                                                   initial_prompt=prompt,
                                                   temperature=temperature, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
    if response_format == "text":
        return Response(text, media_type="text/plain; charset=utf-8")
    return JSONResponse({"text": text})


# -------------------------------------------------------------- OpenAI: models

@app.get("/v1/models")
def models():
    now = int(time.time())
    ids = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "whisper-1"]
    return {"object": "list",
            "data": [{"id": i, "object": "model", "created": now,
                      "owned_by": "engineering-tts-lab"} for i in ids]}


# ------------------------------------------------------------------ extensions

@app.get("/v1/voices")
def get_voices():
    v = voice_registry.default()
    return {"object": "list", "default": v.name if v else None,
            "data": [x.as_dict() for x in voice_registry.list_voices()]}


@app.post("/v1/voices")
async def add_voice(file: UploadFile = File(...), name: str = Form(...),
                    text: str = Form(""), note: str = Form("")):
    """Register a reference clip. An empty transcript is filled in by the recogniser.

    The transcript matters more than it looks: the same clip with a transcript cut
    mid-phrase measured 7.3% error against 1.0% when the two agreed exactly.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "file is empty")
    if not text.strip():
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ref" + (Path(file.filename or "a.wav").suffix or ".wav"))
            with open(p, "wb") as f:
                f.write(raw)
            text = stt.transcribe(p).text
    v = voice_registry.add(name, raw, text.strip(), note)
    return v.as_dict()


@app.post("/v1/text/prepare")
def prepare_text(payload: dict):
    text = (payload or {}).get("text", "")
    return textprep.prepare(text,
                            use_dictionary=bool(payload.get("dictionary", True)),
                            use_legato=bool(payload.get("legato", False)))


@app.get("/health")
def health():
    return {"status": "ok",
            "tts": {"model": TTS_MODEL, "loaded": tts.loaded, "device": tts.device},
            "stt": {"model": STT_MODEL, "loaded": stt.loaded, "device": stt.device,
                    "compute_type": stt.compute_type},
            "cuda": has_cuda(),
            "device": describe(),
            "voices": [v.name for v in voice_registry.list_voices()]}


# ------------------------------------------------------------------------- UI

@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
