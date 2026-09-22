"""The engines in a process of their own, for a server that is not Python.

    python -m engines.host        (from server/, with the engines' environment)

The Rust server (rust/) starts this as a child and talks to it over stdin and
stdout: no port to pick, no firewall prompt, the same on Linux, Windows and
macOS, and the process dies with its parent. The engines inside are the same
classes the Python server uses, chosen by the same TTS_ENGINE, STT_ENGINE, …

Frames, both ways:

    u32 LE  length of the JSON header
    bytes   JSON header; if it has "bin": n, then
    bytes   n bytes of binary payload (audio as float32 LE, or a file)

Requests:  {"id": 7, "op": "tts.speak", "args": {...}} [+ bin]
Replies:   {"id": 7, "event": "progress", "data": {...}}     — any number
           {"id": 7, "result": {...}} [+ bin]                — the end
           {"id": 7, "error": {"kind": "...", "message": "..."}}
Cancel:    {"id": 8, "op": "cancel", "args": {"target": 7}}  — no reply of its
           own; the target ends with an error of kind "cancelled".

Error kinds: unsupported and bad_audio are the caller's fault (400), cancelled
is what was asked for, internal is ours (500).

Operations mirror the contract in base.py:

    info                          what every engine is and can do, and the device
    status   {kind}               engine.status()
    load     {kind}
    tts.language {code}           → {"language"}
    tts.speak {text, ref_audio, ref_text, language, seed}
                                  progress events → {"sample_rate", "info"} + audio
    stt.transcribe {path | +bin 16 kHz, language, prompt, hotwords, temperature,
                    word_timestamps, task, live, draft}
                                  segment events → transcript
    turn.probability +bin         → {"probability"}
    vad.open                      → {"stream", "frame"}
    vad.feed {stream} +bin        → {"pending"} + probabilities
    vad.close {stream}
    audio.decode {sample_rate} +bin file → audio
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import struct
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import engines
from .base import BadAudio, Unsupported


class Cancelled(Exception):
    pass


class Host:
    def __init__(self, out_fd: int, in_fd: int):
        self._out = os.fdopen(out_fd, "wb", buffering=0)
        self._in = os.fdopen(in_fd, "rb", buffering=0)
        self._write = threading.Lock()
        self._cancel: set[int] = set()
        self._pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="op")
        self.e = {kind: engines.create(kind) for kind in ("tts", "stt", "turn", "vad")}
        self._vad: dict[int, object] = {}
        self._vad_ids = itertools.count(1)

    # --------------------------------------------------------------- кадры

    def send(self, header: dict, payload: bytes = b"") -> None:
        if payload:
            header["bin"] = len(payload)
        raw = json.dumps(header, ensure_ascii=False).encode()
        with self._write:
            self._out.write(struct.pack("<I", len(raw)) + raw + payload)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._in.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return bytes(buf)

    def serve(self) -> None:
        try:
            while True:
                (n,) = struct.unpack("<I", self._read(4))
                header = json.loads(self._read(n))
                payload = self._read(header["bin"]) if header.get("bin") else b""
                if header["op"] == "cancel":
                    self._cancel.add(header["args"]["target"])
                    continue
                self._pool.submit(self._run, header, payload)
        except EOFError:
            os._exit(0)                              # сервер ушёл — уходим и мы

    def _run(self, header: dict, payload: bytes) -> None:
        rid = header["id"]
        try:
            op = getattr(self, "op_" + header["op"].replace(".", "_"))
            result = op(rid, header.get("args") or {}, payload)
            body, data = result if isinstance(result, tuple) else (result, b"")
            self.send({"id": rid, "result": body}, data)
        except Cancelled:
            self.send({"id": rid, "error": {"kind": "cancelled", "message": "cancelled"}})
        except Unsupported as e:
            self.send({"id": rid, "error": {"kind": "unsupported", "message": str(e)}})
        except BadAudio as e:
            self.send({"id": rid, "error": {"kind": "bad_audio", "message": str(e)}})
        except Exception as e:                       # noqa: BLE001 — уходит серверу
            traceback.print_exc()
            self.send({"id": rid, "error": {"kind": "internal",
                                            "message": str(e) or type(e).__name__}})
        finally:
            self._cancel.discard(rid)

    def _check(self, rid: int) -> None:
        if rid in self._cancel:
            raise Cancelled()

    # ------------------------------------------------------------ операции

    def op_info(self, rid, args, payload):
        tts, stt, turn, vad = (self.e[k] for k in ("tts", "stt", "turn", "vad"))
        return {
            "tts": {**tts.status(), "sample_rate": tts.sample_rate,
                    "needs_reference_text": tts.needs_reference_text,
                    "reference_seconds": list(tts.reference_seconds),
                    "reference_best": list(tts.reference_best)},
            "stt": {**stt.status(), "features": sorted(stt.features)},
            "turn": {**turn.status(), "sample_rate": turn.sample_rate},
            "vad": {"engine": vad.name, "sample_rate": vad.sample_rate},
            "device": _describe(),
            "cuda": _has_cuda(),
        }

    def op_status(self, rid, args, payload):
        return self.e[args["kind"]].status()

    def op_load(self, rid, args, payload):
        self.e[args["kind"]].load()
        return {}

    def op_tts_language(self, rid, args, payload):
        return {"language": self.e["tts"].language(args.get("code"))}

    def op_tts_speak(self, rid, args, payload):
        def on_progress(p):
            self._check(rid)
            self.send({"id": rid, "event": "progress",
                       "data": {k: v for k, v in dataclasses.asdict(p).items() if v is not None}})

        out = self.e["tts"].speak(args["text"], args["ref_audio"], args.get("ref_text", ""),
                                  language=args.get("language"), seed=args.get("seed"),
                                  on_progress=on_progress)
        self._check(rid)
        audio = np.ascontiguousarray(out.audio, dtype="<f4")
        return {"sample_rate": out.sample_rate, "info": out.info}, audio.tobytes()

    def op_stt_transcribe(self, rid, args, payload):
        def on_segment(position, total, text):
            self._check(rid)
            self.send({"id": rid, "event": "segment",
                       "data": {"position": position, "total": total, "text": text}})

        audio = args["path"] if "path" in args else np.frombuffer(payload, dtype="<f4")
        tr = self.e["stt"].transcribe(
            audio, language=args.get("language"), prompt=args.get("prompt"),
            hotwords=args.get("hotwords"), temperature=args.get("temperature", 0.0),
            word_timestamps=args.get("word_timestamps", False),
            task=args.get("task", "transcribe"), live=args.get("live", False),
            draft=args.get("draft", False), on_segment=on_segment)
        return dataclasses.asdict(tr)

    def op_turn_probability(self, rid, args, payload):
        return {"probability": float(self.e["turn"].probability(np.frombuffer(payload, "<f4")))}

    def op_vad_open(self, rid, args, payload):
        sid = next(self._vad_ids)
        self._vad[sid] = s = self.e["vad"].stream()
        return {"stream": sid, "frame": s.frame}

    def op_vad_feed(self, rid, args, payload):
        s = self._vad[args["stream"]]
        probs = np.ascontiguousarray(s.feed(np.frombuffer(payload, "<f4").copy()), dtype="<f4")
        return {"pending": s.pending}, probs.tobytes()

    def op_vad_close(self, rid, args, payload):
        self._vad.pop(args["stream"], None)
        return {}

    def op_audio_decode(self, rid, args, payload):
        import audio_io
        try:
            audio = audio_io.decode(payload, int(args["sample_rate"]))
        except ValueError as e:
            raise BadAudio(str(e)) from None
        return {}, np.ascontiguousarray(audio, dtype="<f4").tobytes()


def _describe() -> str:
    from .device import describe
    return describe()


def _has_cuda() -> bool:
    from .device import has_cuda
    return has_cuda()


def main() -> None:
    # Кадры идут по настоящему stdout; всё, что движки печатают — в том числе
    # из C-кода, мимо sys.stdout, — уходит в stderr, иначе порвёт протокол.
    out_fd, in_fd = os.dup(1), os.dup(0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if sys.platform == "win32":                      # иначе CRT превратит \n в \r\n
        import msvcrt
        msvcrt.setmode(out_fd, os.O_BINARY)
        msvcrt.setmode(in_fd, os.O_BINARY)
    Host(out_fd, in_fd).serve()


if __name__ == "__main__":
    main()
