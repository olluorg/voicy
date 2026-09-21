"""Every request is a job with an id and a place in the queue.

The OpenAI-compatible routes still look synchronous to their callers — they
submit a job and wait for it on the caller's behalf. The `/v1/jobs` routes hand
the id back at once instead: follow it by event stream, poll it, cancel it, or
leave a `webhook_url` and be called when it ends.

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
from pathlib import Path

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import audio_io
import contexts
import progress as progress_mod
import speak
import textprep
import transcripts
import webhooks
from progress import BadInput, Job
from schemas import JobSpeechRequest, SpeechRequest
from workqueue import QueueFull, queue

_engines: dict = {}

# Адрес, под которым сервер виден снаружи, — для ссылок в сводке задания
# и в webhook. За обратным прокси адрес запроса — внутренний
# (http://127.0.0.1:8080), и получатель webhook по нему не достучится.
PUBLIC_URL = os.environ.get("VOICY_PUBLIC_URL", "").rstrip("/")


# ------------------------------------------------------------------ задания

def _create(kind: str, request: Request, webhook_url: str | None) -> Job:
    if webhook_url:
        try:
            webhooks.validate(webhook_url)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
    job = progress_mod.registry.create(kind, base_url=PUBLIC_URL or str(request.base_url))
    if webhook_url:
        job.webhook = {"url": webhook_url, "attempts": 0, "delivered": None, "error": None}
    return job


def _submit(job: Job, work) -> Job:
    try:
        queue.submit(job, work)
    except QueueFull as e:
        progress_mod.registry.discard(job.id)
        raise HTTPException(429, str(e), headers={"Retry-After": "30"}) from None
    return job


def submit_speech(req: SpeechRequest, request: Request,
                  webhook_url: str | None = None) -> tuple[Job, float]:
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    v = speak.pick_voice(req.voice)

    text = req.input
    if req.prepare or req.legato:
        text = textprep.prepare(text, req.prepare, req.legato)["text"]
    expected = progress_mod.estimate_seconds(text)
    tts = _engines["tts"]

    def work(job: Job) -> dict:
        def on_step(n: int) -> None:
            job.check()
            job.emit(stage="synthesis", steps=n,
                     produced=round(n / progress_mod.STEPS_PER_SECOND, 2),
                     expected=round(expected, 1))

        out = tts.speak(text, v.path, v.text, language=req.language,
                        seed=req.seed, on_step=on_step)
        job.check()
        audio = (audio_io.stretch(out.audio, out.sample_rate, req.speed)
                 if req.speed != 1.0 else out.audio)
        seconds = round(len(audio) / out.sample_rate, 2)
        job.emit(stage="encoding", steps=out.steps, produced=seconds, expected=seconds)
        data, ctype = audio_io.encode(audio, out.sample_rate, req.response_format)
        job.result = {"data": data, "content_type": ctype, "format": req.response_format,
                      "voice": v.name, "seconds": seconds, "steps": out.steps}
        return {"produced": seconds, "expected": seconds, "steps": out.steps,
                "seconds": seconds, "voice": v.name}

    job = _create("speech", request, webhook_url)
    return _submit(job, work), expected


def submit_transcription(raw: bytes, filename: str | None, request: Request, *,
                         task: str = "transcribe", language: str | None = None,
                         prompt: str | None = None, temperature: float = 0.0,
                         word_timestamps: bool = False,
                         context: str | None = None, hotwords: str | None = None,
                         webhook_url: str | None = None) -> Job:
    if not raw:
        raise HTTPException(400, "file is empty")
    try:
        ctx = contexts.resolve(context, prompt, hotwords)
    except KeyError:
        raise HTTPException(400, f"unknown context '{context}'") from None
    suffix = Path(filename or "audio.wav").suffix or ".wav"
    stt = _engines["stt"]

    def work(job: Job) -> dict:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"in{suffix}")
            with open(path, "wb") as f:
                f.write(raw)

            def on_segment(pos: float, total: float, text: str) -> None:
                job.check()
                job.emit(stage="recognition", position=round(pos, 2),
                         total=round(total, 2), text=ctx.fix(text))

            import av
            try:
                tr = stt.transcribe(path, language=language, prompt=ctx.prompt,
                                    hotwords=ctx.hotwords, temperature=temperature,
                                    word_timestamps=word_timestamps, task=task,
                                    on_segment=on_segment)
            except av.error.FFmpegError as e:     # не звук или битый файл — вина входа
                raise BadInput(f"cannot decode audio: {e}") from None
        job.result = {**transcripts.to_dict(tr, ctx.fix), "task": task}
        return {"position": tr.duration, "total": tr.duration, "language": tr.language}

    job = _create(task, request, webhook_url)
    return _submit(job, work)


# ------------------------------------------------------------------ сводка

def summary(job: Job) -> dict:
    base = f"{job.base_url}/v1/jobs/{job.id}"
    out = {
        "id": job.id, "kind": job.kind, "state": job.state,
        "position": queue.position(job) if job.state == "queued" else None,
        "created": round(job.created, 3),
        "started": round(job.started, 3) if job.started else None,
        "finished": round(job.finished, 3) if job.finished else None,
        "error": job.error,
        "progress": job.last,
        "links": {"self": base, "events": f"{base}/events"},
    }
    if job.cancel_requested and not job.done:
        out["cancel_requested"] = True
    if job.state == "done" and job.result:
        r = job.result
        if job.kind == "speech":
            out["result"] = {k: r[k] for k in ("voice", "seconds", "steps",
                                                "format", "content_type")}
            out["links"]["audio"] = f"{base}/audio"
        else:
            out["result"] = {k: r.get(k) for k in ("text", "language", "duration")}
            out["links"]["result"] = f"{base}/result"
    if job.webhook is not None:
        out["webhook"] = {k: v for k, v in job.webhook.items() if k != "started"}
    return out


def _get(job_id: str) -> Job:
    job = progress_mod.registry.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


def _finished(job_id: str) -> Job:
    job = _get(job_id)
    if job.state == "error":
        raise HTTPException(job.error_status, job.error or "failed")
    if job.state == "cancelled":
        raise HTTPException(410, "job was cancelled")
    if job.state != "done":
        raise HTTPException(409, "not finished")
    return job


async def wait(job: Job) -> Job:
    """For the synchronous routes: wait, then fail the way a plain call would."""
    await job.wait()
    if job.state == "error":
        raise HTTPException(job.error_status, job.error or "failed")
    if job.state == "cancelled":
        raise HTTPException(409, "job was cancelled")
    return job


# ------------------------------------------------------------------ маршруты

def attach(app, tts, stt):
    _engines.update(tts=tts, stt=stt)

    @app.post("/v1/jobs/speech", status_code=202)
    async def job_speech(req: JobSpeechRequest, request: Request):
        job, expected = submit_speech(req, request, req.webhook_url)
        return {**summary(job), "expected_seconds": round(expected, 1)}

    @app.post("/v1/jobs/transcribe", status_code=202)
    async def job_transcribe(request: Request,
                             file: UploadFile = File(...),
                             language: str | None = Form(None),
                             prompt: str | None = Form(None),
                             temperature: float = Form(0.0),
                             word_timestamps: bool = Form(False),
                             task: str = Form("transcribe"),
                             context: str | None = Form(None),
                             hotwords: str | None = Form(None),
                             webhook_url: str | None = Form(None)):
        if task not in ("transcribe", "translate"):
            raise HTTPException(400, "task must be transcribe or translate")
        job = submit_transcription(await file.read(), file.filename, request, task=task,
                                   language=language, prompt=prompt,
                                   temperature=temperature,
                                   word_timestamps=word_timestamps,
                                   context=context, hotwords=hotwords,
                                   webhook_url=webhook_url)
        return summary(job)

    @app.get("/v1/jobs")
    def list_jobs(state: str | None = None):
        items = [summary(j) for j in progress_mod.registry.all()
                 if state is None or j.state == state]
        return {"object": "list", "data": items,
                "queue": {"pending": queue.pending(), "max": queue.max_pending,
                          "running": queue.current.id if queue.current else None}}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return summary(_get(job_id))

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = _get(job_id)
        if not queue.cancel(job):
            raise HTTPException(409, f"job already {job.state}")
        return summary(job)

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = _get(job_id)

        async def stream():
            q = job.subscribe()                   # сначала подписка, потом взгляд на state
            try:
                yield f"data: {json.dumps(job.last, ensure_ascii=False)}\n\n"
                if job.done:
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=25)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"      # чтобы прокси не рвал соединение
                        continue
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("stage") in progress_mod.TERMINAL:
                        break
            finally:
                job.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{job_id}/audio")
    def job_audio(job_id: str):
        job = _finished(job_id)
        if job.kind != "speech":
            raise HTTPException(400, "not a speech job — see /result")
        return Response(content=job.result["data"], media_type=job.result["content_type"],
                        headers={"X-Job-Id": job.id,
                                 "X-Voice": job.result["voice"],
                                 "X-Audio-Seconds": str(job.result["seconds"]),
                                 "X-Decoding-Steps": str(job.result["steps"])})

    @app.get("/v1/jobs/{job_id}/result")
    def job_result(job_id: str, format: str | None = None):
        job = _finished(job_id)
        if job.kind == "speech":
            return JSONResponse({k: v for k, v in job.result.items() if k != "data"})
        if format is None:
            return JSONResponse(job.result)
        if format not in transcripts.FORMATS:
            raise HTTPException(400, f"format must be one of {', '.join(transcripts.FORMATS)}")
        return transcripts.render(job.result, format)
