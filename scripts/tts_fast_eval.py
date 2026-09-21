"""Does the CUDA-graph code predictor sound the same as the original?

Token-for-token equality is not on offer: the fast path attends over a static
cache with an explicit mask, the original over a growing one, and the two pick
different attention kernels. Logits differ by one or two bf16 units, and where
two candidates are nearly tied, a sampled token flips and the rest of the frame
follows. So the question is statistical — does the speech stay as intelligible —
and it is answered with the repository's own metric (ADR 0001): synthesise, read
back with Whisper, count character errors against the text.

Ten sentences, three seeds each, original vs fast, in one process so both share
the same weights. Speed is recorded alongside.

    python scripts/tts_fast_eval.py  →  results/results_tts_fast.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

SENTENCES = [
    "Хорошо, сейчас посмотрю.",
    "Сейчас я проверю расписание поездов на завтра и сразу вам отвечу.",
    "Судя по данным, первый поезд уходит рано утром, а последний около полуночи.",
    "Если нужно, могу забронировать место у окна.",
    "Она была более чувствительна, нежели добра, и до зрелых лет сохранила институтские замашки.",
    "Давайте разберём, как устроен запрос к серверу.",
    "Каждый кадр звука длится примерно восемьдесят миллисекунд.",
    "Готовый звук кодируется в нужный формат и возвращается клиенту.",
    "Распознавание устроено похоже, только в обратную сторону.",
    "Вечером над рекой поднялся туман, и огни на том берегу стали едва видны.",
]
SEEDS = (1, 2, 3)


def norm(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^а-яa-z ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(r))


def main() -> None:
    import numpy as np
    import soxr

    import voices
    from engines import tts_fast
    from engines.stt_whisper import WhisperSTT
    from engines.tts_qwen import QwenTTS

    tts, stt, v = QwenTTS(), WhisperSTT(), voices.default()
    tts.load()
    stt.load()
    tts.speak("Прогрев.", v.path, v.text)

    rows = []
    for variant in ("original", "fast"):
        if variant == "fast":
            tts_fast.install(tts._model)
            tts.speak("Прогрев.", v.path, v.text)            # запись графов — вне замера
        for text in SENTENCES:
            for seed in SEEDS:
                t = time.perf_counter()
                out = tts.speak(text, v.path, v.text, seed=seed)
                dt = time.perf_counter() - t
                audio = soxr.resample(out.audio, out.sample_rate, 16000).astype(np.float32)
                hyp = stt.transcribe(audio, language="ru").text
                rows.append({"variant": variant, "text": text, "seed": seed,
                             "audio_s": round(len(out.audio) / out.sample_rate, 2),
                             "synth_s": round(dt, 2), "ms_per_frame": round(1000 * dt / out.info["steps"]),
                             "cer": round(100 * cer(text, hyp), 2), "heard": hyp})
                print(f"{variant:8} seed {seed} CER {rows[-1]['cer']:5.1f}%  "
                      f"{rows[-1]['ms_per_frame']:3d} мс/кадр  {text[:40]}", flush=True)

    report = {}
    for variant in ("original", "fast"):
        rs = [r for r in rows if r["variant"] == variant]
        report[variant] = {
            "cer_mean_pct": round(statistics.mean(r["cer"] for r in rs), 2),
            "cer_median_pct": round(statistics.median(r["cer"] for r in rs), 2),
            "cer_max_pct": max(r["cer"] for r in rs),
            "ms_per_frame_median": statistics.median(r["ms_per_frame"] for r in rs),
            "speed": round(sum(r["audio_s"] for r in rs) / sum(r["synth_s"] for r in rs), 2),
            "audio_s_total": round(sum(r["audio_s"] for r in rs), 1),
        }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    out = ROOT / "results" / "results_tts_fast.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
