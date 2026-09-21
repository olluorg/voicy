"""A transcript as plain data, and the formats OpenAI returns it in.

A job has to outlive the recogniser's objects, so the transcript is kept as a
dict. The same dict serves the compatible route, the job result and the webhook.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse, Response

FORMATS = ("json", "text", "srt", "vtt", "verbose_json")


def to_dict(tr, fix=lambda t: t) -> dict:
    """`fix` rewrites text — a context's replacements. Words keep what was heard,
    because their timestamps belong to those words."""
    return {
        "text": fix(tr.text), "language": tr.language, "duration": tr.duration,
        "segments": [{"start": s.start, "end": s.end, "text": fix(s.text),
                      "words": [{"word": w.word, "start": w.start, "end": w.end}
                                for w in s.words]}
                     for s in tr.segments],
    }


def _srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def render(tr: dict, response_format: str):
    fmt = (response_format or "json").lower()
    segments = tr.get("segments", [])
    if fmt == "text":
        return Response(tr["text"], media_type="text/plain; charset=utf-8")
    if fmt == "srt":
        lines = []
        for i, s in enumerate(segments, 1):
            lines += [str(i), f"{_srt_time(s['start'])} --> {_srt_time(s['end'])}", s["text"], ""]
        return Response("\n".join(lines), media_type="text/plain; charset=utf-8")
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for s in segments:
            lines += [f"{_srt_time(s['start']).replace(',', '.')} --> "
                      f"{_srt_time(s['end']).replace(',', '.')}", s["text"], ""]
        return Response("\n".join(lines), media_type="text/vtt; charset=utf-8")
    if fmt == "verbose_json":
        return JSONResponse({
            "task": tr.get("task", "transcribe"), "language": tr.get("language"),
            "duration": tr.get("duration"), "text": tr["text"],
            "segments": [{"id": i, **s} for i, s in enumerate(segments)],
        })
    return JSONResponse({"text": tr["text"]})
