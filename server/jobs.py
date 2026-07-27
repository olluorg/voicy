"""Long work as jobs you can watch.

The OpenAI-compatible routes stay synchronous: a caller there waits for bytes and
knows nothing about jobs. The console wants a progress bar, so the same work is
also offered as a job with an event stream attached.

Nothing reported here is invented:

  synthesis — the decoding loop is counted step by step and the model is a 12 Hz
              codec, so the count converts to seconds of audio already produced.
              Only the *total* is unknown until generation stops, so it travels
              as a separate field named `expected` and the UI marks it as such;

  recognition — the file duration is known before decoding starts and each
              segment reports where it ended. Both halves are facts.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import audio_io
import progress as progress_mod
import textprep
import voices as voice_registry
from schemas import SpeechRequest


def attach(app, tts, stt):
    @app.post("/v1/jobs/speech")
    async def job_speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(400, "input is empty")
        v = voice_registry.get(req.voice) if req.voice else voice_registry.default()
        if v is None or not v.text:
            raise HTTPException(400, "no usable voice")

        text = req.input
        if req.prepare or req.legato:
            text = textprep.prepare(text, req.prepare, req.legato)["text"]

        job = progress_mod.registry.create("speech")
        expected = progress_mod.estimate_seconds(text)
        job.emit(stage="queued", produced=0.0, expected=round(expected, 1), steps=0)

        def work():
            try:
                def on_step(n: int) -> None:
                    job.emit(stage="synthesis", steps=n,
                             produced=round(n / progress_mod.STEPS_PER_SECOND, 2),
                             expected=round(expected, 1))

                out = tts.speak(text, v.path, v.text, language=req.language,
                                seed=req.seed, on_step=on_step)
                audio = (audio_io.stretch(out.audio, out.sample_rate, req.speed)
                         if req.speed != 1.0 else out.audio)
                seconds = round(len(audio) / out.sample_rate, 2)
                job.emit(stage="encoding", steps=out.steps, produced=seconds,
                         expected=seconds)
                data, ctype = audio_io.encode(audio, out.sample_rate, req.response_format)
                job.result = {"data": data, "content_type": ctype, "voice": v.name,
                              "seconds": seconds, "steps": out.steps}
                job.state = "done"
                job.emit(stage="done", produced=seconds, expected=seconds,
                         steps=out.steps, seconds=seconds, voice=v.name)
            except Exception as e:                       # noqa: BLE001 — уходит клиенту
                job.state, job.error = "error", str(e)
                job.emit(stage="error", error=str(e))

        threading.Thread(target=work, daemon=True).start()
        return {"id": job.id, "kind": "speech", "expected_seconds": round(expected, 1)}

    @app.post("/v1/jobs/transcribe")
    async def job_transcribe(file: UploadFile = File(...),
                             language: str | None = Form(None),
                             prompt: str | None = Form(None),
                             temperature: float = Form(0.0),
                             word_timestamps: bool = Form(False)):
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "file is empty")
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        job = progress_mod.registry.create("transcribe")
        job.emit(stage="queued", position=0.0, total=0.0)

        def work():
            try:
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, f"in{suffix}")
                    with open(path, "wb") as f:
                        f.write(raw)

                    def on_segment(pos: float, total: float, text: str) -> None:
                        job.emit(stage="recognition", position=round(pos, 2),
                                 total=round(total, 2), text=text)

                    tr = stt.transcribe(path, language=language, prompt=prompt,
                                        temperature=temperature,
                                        word_timestamps=word_timestamps,
                                        on_segment=on_segment)
                job.result = {
                    "text": tr.text, "language": tr.language, "duration": tr.duration,
                    "segments": [{"start": s.start, "end": s.end, "text": s.text,
                                  "words": [{"word": w.word, "start": w.start, "end": w.end}
                                            for w in s.words]}
                                 for s in tr.segments],
                }
                job.state = "done"
                job.emit(stage="done", position=tr.duration, total=tr.duration,
                         language=tr.language)
            except Exception as e:                       # noqa: BLE001
                job.state, job.error = "error", str(e)
                job.emit(stage="error", error=str(e))

        threading.Thread(target=work, daemon=True).start()
        return {"id": job.id, "kind": "transcribe"}

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = progress_mod.registry.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")

        async def stream():
            yield f"data: {json.dumps(job.last, ensure_ascii=False)}\n\n"
            if job.state in ("done", "error"):
                return
            while True:
                try:
                    event = await asyncio.wait_for(job.queue.get(), timeout=25)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"          # чтобы прокси не рвал соединение
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("stage") in ("done", "error"):
                    break

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{job_id}/audio")
    def job_audio(job_id: str):
        job = _finished(job_id)
        return Response(content=job.result["data"], media_type=job.result["content_type"],
                        headers={"X-Voice": job.result["voice"],
                                 "X-Audio-Seconds": str(job.result["seconds"]),
                                 "X-Decoding-Steps": str(job.result["steps"])})

    @app.get("/v1/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = _finished(job_id)
        payload = {k: v for k, v in job.result.items() if k != "data"}
        return JSONResponse(payload)

    def _finished(job_id: str):
        job = progress_mod.registry.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job.state == "error":
            raise HTTPException(500, job.error or "failed")
        if job.state != "done":
            raise HTTPException(409, "not finished")
        return job
