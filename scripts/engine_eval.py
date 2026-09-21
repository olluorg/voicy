"""How intelligible is whatever engine the running server has?

The same ten sentences and three seeds as tts_fast_eval.py, but through the
HTTP API instead of an engine class: synthesise, read back with the server's
own recogniser, count character errors against the text (ADR 0001). Because it
never imports an engine, it measures any of them the same way — start the
server with the engine in question and run this.

    TTS_ENGINE=espeech ./voicy up
    python scripts/engine_eval.py  →  results/results_engine_<engine>.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tts_fast_eval import SEEDS, SENTENCES, cer  # noqa: E402

URL = os.environ.get("VOICY_URL", "http://localhost:8080").rstrip("/")
# свой сервер — мимо прокси
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


def call(path: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(URL + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read(), r.headers


def transcribe(wav: bytes) -> str:
    b = "voicy-eval"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nru\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{b}--\r\n".encode()
    raw, _ = call("/v1/audio/transcriptions", body,
                  {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(raw)["text"]


def main() -> None:
    health = json.loads(call("/health")[0])
    engine = health["tts"]
    rows = []
    for text in SENTENCES:
        for seed in SEEDS:
            t = time.perf_counter()
            wav, h = call("/v1/audio/speech",
                          json.dumps({"input": text, "seed": seed,
                                      "response_format": "wav"}).encode(),
                          {"Content-Type": "application/json"})
            dt = time.perf_counter() - t
            hyp = transcribe(wav)
            rows.append({"text": text, "seed": seed, "audio_s": float(h["X-Audio-Seconds"]),
                         "synth_s": round(dt, 2), "cer": round(100 * cer(text, hyp), 2),
                         "heard": hyp})
            print(f"CER {rows[-1]['cer']:5.1f}%  seed {seed}  {text[:50]}", flush=True)

    report = {
        "engine": engine, "device": health.get("device"),
        "cer_mean": round(statistics.mean(r["cer"] for r in rows), 2),
        "clean": f"{sum(r['cer'] == 0 for r in rows)}/{len(rows)}",
        "speed": round(sum(r["audio_s"] for r in rows) / sum(r["synth_s"] for r in rows), 2),
        "rows": rows,
    }
    out = ROOT / "results" / f"results_engine_{engine['engine']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{engine['engine']}: CER {report['cer_mean']}%, чисто {report['clean']}, "
          f"×{report['speed']} к реальному времени → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
