"""Checks of the voicy API — one command, one report.

    python -m tests                          # свои серверы с учебными движками
    python -m tests --report report.json     # и отчёт в файл
    python -m tests --url http://host:8080   # уже запущенный сервер
    python -m tests -k Speech                # только то, в чьём имени есть Speech

Runs the same on Linux, Windows and macOS. The report says what machine it
ran on, what the server said about itself, and how every check went — enough
to send back from a machine you cannot log in to.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

from . import common
from .client import Client

HERE = Path(__file__).resolve().parent


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows, self._t = [], {}

    def startTest(self, test):
        self._t[test.id()] = time.perf_counter()
        super().startTest(test)

    def _row(self, test, status, detail=""):
        took = time.perf_counter() - self._t.get(test.id(), time.perf_counter())
        self.rows.append({"test": test.id().removeprefix("tests."), "status": status,
                          "seconds": round(took, 2), "detail": detail[-2000:]})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._row(test, "ok")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._row(test, "fail", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._row(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._row(test, "skip", reason)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine() -> dict:
    """What the report needs to be read without asking: system, CPU, GPUs, memory."""
    info = {"system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
            "cpus": os.cpu_count(), "ffmpeg": bool(shutil.which("ffmpeg"))}
    gpus = []
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"])
        gpus += [f"NVIDIA {line}" for line in out.splitlines() if line]
    if info["system"] == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "ForEach-Object { $_.Name + ', ' + $_.DriverVersion }"])
        gpus += [line for line in out.splitlines() if line]
    elif info["system"] == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        gpus += [line.split(":", 1)[1].strip() for line in out.splitlines()
                 if "Chipset Model" in line]
    elif shutil.which("lspci"):
        gpus += [line for line in _run(["lspci"]).splitlines()
                 if "VGA" in line or "3D controller" in line]
    info["gpus"] = gpus
    try:
        if info["system"] == "Linux":
            kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
            info["ram_gb"] = round(kb / 1024 ** 2, 1)
        elif info["system"] == "Darwin":
            info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3, 1)
        elif info["system"] == "Windows":
            b = _run(["powershell", "-NoProfile", "-Command",
                      "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            info["ram_gb"] = round(int(b) / 1024 ** 3, 1)
    except (StopIteration, ValueError, OSError):
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m tests", description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="уже запущенный сервер; без него — свои с учебными движками")
    ap.add_argument("--key", default=os.environ.get("VOICY_API_KEY"),
                    help="ключ доступа к серверу из --url")
    ap.add_argument("--writes", action="store_true",
                    help="разрешить проверкам добавлять голоса и контексты на сервер из --url")
    ap.add_argument("--report", help="куда записать отчёт JSON")
    ap.add_argument("-k", dest="pattern", action="append",
                    help="только проверки, в имени которых есть это (можно несколько)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    common.CONFIG.update(url=a.url, writes=a.writes, key=a.key)

    loader = unittest.TestLoader()
    if a.pattern:
        loader.testNamePatterns = [f"*{p}*" for p in a.pattern]
    suite = loader.discover(str(HERE), top_level_dir=str(HERE.parent))
    runner = unittest.TextTestRunner(verbosity=2 if a.verbose else 1, resultclass=Recorder)
    started = time.time()
    try:
        result = runner.run(suite)
        server = None
        try:
            server = common.server().get("/health").json()
        except Exception as e:                      # noqa: BLE001 — отчёт важнее
            server = {"error": str(e)}
    finally:
        common.stop_all()

    counts = {s: sum(r["status"] == s for r in result.rows) for s in ("ok", "fail", "error", "skip")}
    if a.report:
        report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                  "seconds": round(time.time() - started, 1),
                  "target": a.url or "own servers with the training engines",
                  "machine": machine(), "server": server, "counts": counts,
                  "tests": result.rows}
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"отчёт: {a.report}")
    print(f"итог: {counts['ok']} прошло, {counts['fail']} не прошло, "
          f"{counts['error']} с ошибкой, {counts['skip']} пропущено")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
