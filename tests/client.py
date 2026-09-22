"""A client that knows nothing about the server but its API.

Only the standard library and `websockets`: the checks must talk to any
implementation of the server — this Python one today, a Rust one later — the
way an outside client does, over HTTP and WebSocket.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

import numpy as np

# свой сервер — мимо прокси (AGENTS.md: прокси ломает обращение к себе же)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes

    def json(self):
        return json.loads(self.body)

    @property
    def error(self) -> str:
        """The message of an error in OpenAI's shape."""
        return self.json()["error"]["message"]


class Client:
    def __init__(self, url: str, key: str | None = None):
        self.url = url.rstrip("/")
        self.key = key

    # ------------------------------------------------------------------ HTTP

    def request(self, method: str, path: str, *, json_body=None, form: dict | None = None,
                files: dict | None = None, headers: dict | None = None,
                timeout: float = 60) -> Response:
        data, h = None, dict(headers or {})
        if self.key and "Authorization" not in h:
            h["Authorization"] = f"Bearer {self.key}"
        if json_body is not None:
            data = json.dumps(json_body).encode()
            h["Content-Type"] = "application/json"
        elif form is not None or files is not None:
            data, ctype = multipart(form or {}, files or {})
            h["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read())
        except urllib.error.HTTPError as e:
            return Response(e.code, {k.lower(): v for k, v in e.headers.items()}, e.read())

    def get(self, path: str, **kw) -> Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw) -> Response:
        return self.request("POST", path, **kw)

    def events(self, path: str, *, method: str = "GET", timeout: float = 60, **kw):
        """Server-sent events as dicts, until the stream ends."""
        h = dict(kw.pop("headers", None) or {})
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        data = None
        if "form" in kw or "files" in kw:
            data, h["Content-Type"] = multipart(kw.pop("form", {}), kw.pop("files", {}))
        elif "json_body" in kw:
            data = json.dumps(kw.pop("json_body")).encode()
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
        with _OPENER.open(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    # ------------------------------------------------------------- WebSocket

    def ws(self, path: str):
        from websockets.sync.client import connect

        url = "ws" + self.url[len("http"):] + path
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        return connect(url, additional_headers=headers, open_timeout=30,
                       max_size=None, proxy=None)

    def wait(self, predicate, timeout: float = 30, every: float = 0.05):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(every)
        raise AssertionError(f"condition not met in {timeout} s")


def multipart(form: dict, files: dict) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in form.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (name, content, ctype) in files.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                  f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
        out.write(content)
        out.write(b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


# ------------------------------------------------------------------- звук

def tone(seconds: float, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), np.float32)


def to_wav(x: np.ndarray, sr: int = 16000) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


def read_wav(data: bytes) -> tuple[int, int, float]:
    """(channels, sample rate, seconds)"""
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getframerate(), w.getnframes() / w.getframerate()
