"""A throwaway server with the training engines, on a free port.

Its voices and contexts live in a temporary directory — the checks add and
delete them, and the repository must not change under them. Started with the
same Python that runs the checks, as a plain process: works the same on
Linux, Windows and macOS.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Какой сервер проверять: python (server/) или rust (rust/, собранный заранее).
IMPL = os.environ.get("VOICY_IMPL", "python")
RUST_BIN = Path(os.environ.get("VOICY_RUST_BIN") or ROOT / "rust" / "target" / "release" /
                ("voicy.exe" if sys.platform == "win32" else "voicy"))
FAKE = {"TTS_ENGINE": "tone", "STT_ENGINE": "script",
        "TURN_ENGINE": "pause", "VAD_ENGINE": "energy"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, **env: str):
        self.env = env
        self.url = ""
        self.proc: subprocess.Popen | None = None
        self.dir = Path(tempfile.mkdtemp(prefix="voicy-tests-"))
        self.log = self.dir / "server.log"

    def start(self, timeout: float = 60) -> "Server":
        voices = self.dir / "voices"
        shutil.copytree(ROOT / "server" / "voices", voices)
        port = _free_port()
        env = {**os.environ, **FAKE, "VOICY_VOICES_DIR": str(voices),
               "VOICY_CONTEXTS_DIR": str(self.dir / "contexts"),
               "NO_PROXY": "127.0.0.1,localhost", "PYTHONUNBUFFERED": "1", **self.env}
        if "VOICY_API_KEY" not in self.env:
            env.pop("VOICY_API_KEY", None)          # ключ из окружения — не для этих проверок
        self.url = f"http://127.0.0.1:{port}"
        if IMPL == "rust":
            # сервер на Rust; движки — в дочернем Python того же окружения, что у проверок
            env.update(VOICY_HOME=str(ROOT), VOICY_PYTHON=sys.executable)
            # свой кэш: сервер на кэш — один (rust/src/instance.rs), а проверки
            # поднимают несколько сразу и не должны задевать настоящий voicy
            env["VOICY_CACHE"] = str(self.dir / "cache")
            cmd = [str(RUST_BIN), "serve", "--host", "127.0.0.1", "--port", str(port)]
        else:
            cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                   "--port", str(port), "--app-dir", str(ROOT / "server"), "--log-level", "warning"]
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log.open("wb"), stderr=subprocess.STDOUT)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited:\n{self.log.read_text(errors='replace')}")
            try:
                with opener.open(self.url + "/health", timeout=2):
                    return self
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"server did not answer in {timeout} s:\n"
                           f"{self.log.read_text(errors='replace')}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)
