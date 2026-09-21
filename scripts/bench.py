"""Speed and memory of synthesis and recognition on this machine.

    python scripts/bench.py              →  results/bench_<видеокарта>.json + таблица
    python scripts/bench.py --only tts   —  перемерить одну часть, остальное из файла

Run it with the voicy server stopped (`./voicy down`): memory is measured as
the growth of the card's usage, and a running server would be counted too.

Each part runs in its own process, so one model's memory does not leak into
another's figures:

  tts       — synthesis speed on four lengths of text, the first piece of a
              streamed answer, peak video and system memory;
  tts-min   — the least video memory synthesis still works in: PyTorch is
              capped at a smaller and smaller share of the card until the text
              no longer fits. Memory grows with the text, so two lengths: a
              piece of streamed synthesis (up to 200 characters) and 597
              characters in one pass, as a synchronous call;
  stt       — recognition speed on 10 s, 60 s and 5 min of speech, peak memory;
  stt-int8  — the same with int8_float16 weights: less memory, and how much the
              text differs from float16;
  both      — both models in one process, as the server runs them;
  fit       — the whole server on a smaller card: all but 8 or 6 GB of the card
              is taken by a ballast tensor, and both models run in what is left.

Why a search for the minimum rather than the peak: PyTorch keeps freed memory
for reuse, so what a process occupies overstates what it needs. Whisper runs on
CTranslate2, which cannot be capped from outside, so for it the peak is all
there is.

The audio the recogniser reads is the synthesiser's own output. For speed it
does not matter whose voice it is; the text similarity of int8 against float16
is a sanity check, not a quality metric.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
WORK = Path(os.environ.get("BENCH_DIR", "/tmp/voicy-bench"))

TEXTS = {
    "short": "Хорошо, сейчас посмотрю.",
    "medium": "Сейчас я проверю расписание поездов на завтра и сразу вам отвечу.",
    "long": ("Сейчас я проверю расписание поездов на завтра и сразу вам отвечу. "
             "Судя по данным, первый поезд уходит в шесть утра, а последний около полуночи. "
             "Если нужно, могу забронировать место."),
    "xlong": ("Давайте разберём, как устроен запрос к серверу. Сначала клиент отправляет текст, "
              "и сервер ставит задание в очередь. Очередь одна на обе модели, потому что "
              "видеокарта тоже одна. Когда подходит очередь задания, модель синтеза читает "
              "текст и генерирует звук кадр за кадром. Каждый кадр длится примерно восемьдесят "
              "миллисекунд. Готовый звук кодируется в нужный формат и возвращается клиенту. "
              "Если клиент оставил адрес для уведомления, сервер сам сообщит, что всё готово. "
              "Распознавание устроено похоже, только в обратную сторону: звук превращается "
              "в текст, и текст приходит по частям, пока идёт работа."),
}


# ------------------------------------------------------------------ измерители

class VramMonitor:
    """Samples the card's used memory every 50 ms with nvidia-smi."""

    def __init__(self):
        self.base = self._once()
        self.samples: list[int] = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-lms", "50"], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    @staticmethod
    def _once() -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.split()[0])

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                self.samples.append(int(line.strip()))
            except ValueError:
                pass

    def reset(self) -> None:
        self.samples.clear()

    def peak(self) -> float:
        """ГБ сверх того, что было занято до начала, — пик с последнего reset."""
        time.sleep(0.2)
        top = max(self.samples) if self.samples else self._once()
        return round((top - self.base) / 1024, 2)

    def stop(self) -> None:
        self._proc.terminate()


def rss_peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


def machine() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cpu = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    import importlib.metadata as md
    versions = {}
    for pkg in ("torch", "qwen-tts", "faster-whisper", "ctranslate2", "transformers"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {"gpu": gpu, "cpu": cpu, "ram_gb": round(ram, 1), "python": platform.python_version(),
            "versions": versions}


# ---------------------------------------------------------------------- части

def part_tts() -> dict:
    mon = VramMonitor()
    import soundfile as sf
    import torch

    import voices
    from engines.tts_qwen import QwenTTS
    from speak import Segmenter

    tts, v = QwenTTS(), voices.default()
    t = time.perf_counter()
    tts.load()
    load_s = time.perf_counter() - t
    after_load = mon.peak()
    tts.speak("Прогрев.", v.path, v.text)

    runs = {}
    for name, text in TEXTS.items():
        rows = []
        for _ in range(2):
            t = time.perf_counter()
            out = tts.speak(text, v.path, v.text, seed=1)
            dt = time.perf_counter() - t
            audio = len(out.audio) / out.sample_rate
            rows.append({"audio_s": round(audio, 2), "synth_s": round(dt, 2),
                         "speed": round(audio / dt, 2), "ms_per_step": round(1000 * dt / out.info["steps"])})
        runs[name] = {"chars": len(text), "runs": rows}
        if name == "xlong":
            WORK.mkdir(parents=True, exist_ok=True)
            sf.write(WORK / "speech.wav", out.audio, out.sample_rate)

    # поток: сколько ждать первый кусок длинного ответа
    first = Segmenter().split(TEXTS["long"])[0]
    t = time.perf_counter()
    out = tts.speak(first, v.path, v.text, seed=1)
    stream_first = {"chunk": first, "first_audio_s": round(time.perf_counter() - t, 2),
                    "whole_s": runs["long"]["runs"][-1]["synth_s"]}

    res = {"load_s": round(load_s, 1), "vram_after_load_gb": after_load,
           "vram_peak_gb": mon.peak(),
           "torch_allocated_peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "torch_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
           "ram_peak_gb": rss_peak_gb(), "texts": runs, "stream": stream_first}
    mon.stop()
    return res


def part_tts_min() -> dict:
    mon = VramMonitor()
    import torch

    import voices
    from engines.tts_qwen import QwenTTS

    tts, v = QwenTTS(), voices.default()
    tts.load()
    tts.speak("Прогрев.", v.path, v.text)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    weights = torch.cuda.memory_allocated()
    # всё, что процесс держит на карте помимо тензоров PyTorch: контекст CUDA, cuBLAS
    now = (mon._once() - mon.base) / 1024
    overhead = now - torch.cuda.memory_reserved() / 1024 ** 3

    def fits(text: str, limit: float | None) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0 if limit is None else limit / total, 0)
        try:
            tts.speak(text, v.path, v.text, seed=1)
            return True
        except torch.OutOfMemoryError:
            return False
        except RuntimeError as e:                      # cuBLAS сообщает о нехватке по-своему
            if "out of memory" in str(e).lower() or "ALLOC" in str(e):
                return False
            raise
        finally:
            torch.cuda.empty_cache()

    res = {"weights_gb": round(weights / 1024 ** 3, 2), "overhead_gb": round(overhead, 2),
           "texts": {}}
    # «long» — 181 символ, почти предельный кусок потокового синтеза (200);
    # «xlong» — 597 символов одним проходом, как синхронный вызов
    for name in ("long", "xlong"):
        text = TEXTS[name]
        torch.cuda.reset_peak_memory_stats()
        assert fits(text, None)
        # верхняя граница — то, что этот же текст занял без ограничений: она проверена
        lo, hi = float(weights), float(torch.cuda.max_memory_reserved())
        steps = [{"limit_gb": round(hi / 1024 ** 3, 2), "fits": True}]
        while hi - lo > 64 * 1024 ** 2:                  # точность 64 МБ
            mid = (lo + hi) / 2
            ok = fits(text, mid)
            steps.append({"limit_gb": round(mid / 1024 ** 3, 2), "fits": ok})
            if ok:
                hi = mid
            else:
                lo = mid
        res["texts"][name] = {"chars": len(text), "torch_min_gb": round(hi / 1024 ** 3, 2),
                              "vram_min_gb": round(hi / 1024 ** 3 + overhead, 2),
                              "search": steps}
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    mon.stop()
    return res


def _stt(compute_type: str | None) -> dict:
    mon = VramMonitor()
    import numpy as np
    import soundfile as sf

    from engines.stt_whisper import WhisperSTT

    speech, sr = sf.read(WORK / "speech.wav", dtype="float32")
    import soxr
    speech = soxr.resample(speech, sr, 16000).astype(np.float32)
    clips = {"10s": speech[:10 * 16000],
             "60s": np.resize(speech, 60 * 16000),
             "300s": np.resize(speech, 300 * 16000)}

    stt = WhisperSTT(compute_type=compute_type)
    t = time.perf_counter()
    stt.load()
    load_s = time.perf_counter() - t
    after_load = mon.peak()
    stt.transcribe(clips["10s"], language="ru")          # прогрев

    runs, texts = {}, {}
    for name, audio in clips.items():
        t = time.perf_counter()
        tr = stt.transcribe(audio, language="ru")
        dt = time.perf_counter() - t
        runs[name] = {"audio_s": round(len(audio) / 16000, 1), "time_s": round(dt, 2),
                      "speed": round(len(audio) / 16000 / dt, 1)}
        texts[name] = tr.text
    t = time.perf_counter()
    stt.transcribe(clips["60s"], language="ru", word_timestamps=True)
    runs["60s_words"] = {"audio_s": 60.0, "time_s": round(time.perf_counter() - t, 2)}
    runs["60s_words"]["speed"] = round(60 / runs["60s_words"]["time_s"], 1)

    res = {"compute_type": stt.compute_type, "load_s": round(load_s, 1),
           "vram_after_load_gb": after_load, "vram_peak_gb": mon.peak(),
           "ram_peak_gb": rss_peak_gb(), "runs": runs, "text_60s": texts["60s"]}
    mon.stop()
    return res


def part_stt() -> dict:
    return _stt(None)


def part_stt_int8() -> dict:
    return _stt("int8_float16")


def part_both() -> dict:
    mon = VramMonitor()
    import numpy as np
    import soundfile as sf

    import voices
    from engines.stt_whisper import WhisperSTT
    from engines.tts_qwen import QwenTTS
    from engines.turn_smart import SmartTurn

    tts, stt, turn, v = QwenTTS(), WhisperSTT(), SmartTurn(), voices.default()
    tts.load()
    stt.load()
    turn.load()
    speech, sr = sf.read(WORK / "speech.wav", dtype="float32")
    import soxr
    speech = soxr.resample(speech, sr, 16000).astype(np.float32)
    tts.speak(TEXTS["xlong"], v.path, v.text, seed=1)
    stt.transcribe(np.resize(speech, 60 * 16000), language="ru")
    t = time.perf_counter()
    for _ in range(20):
        turn.probability(speech[:4 * 16000])
    res = {"vram_peak_gb": mon.peak(), "ram_peak_gb": rss_peak_gb(),
           "turn_ms": round((time.perf_counter() - t) / 20 * 1000, 1)}
    mon.stop()
    return res


def part_fit() -> dict:
    """Does the whole server fit on a smaller card? BENCH_FIT="8:float16" leaves
    8 GB free — the rest is taken by a ballast tensor — and runs both models in
    it, as the server does. The card's current occupants (a desktop, say) count
    against the ballast, so the result is for a card with nothing else on it."""
    import numpy as np
    import soundfile as sf
    import torch

    card_gb, compute = os.environ["BENCH_FIT"].split(":")
    os.environ["STT_COMPUTE_TYPE"] = compute
    free, total = torch.cuda.mem_get_info()
    # Драйвер под Windows и WSL по умолчанию добирает недостающее из системной
    # памяти (Sysmem Fallback Policy): нехватки не случается, работа лишь
    # замедляется — и балласт ничего не доказывает. Проверяем, прежде чем мерить.
    try:
        probe = torch.empty(int(free + 1024 ** 3), dtype=torch.uint8, device="cuda")
        del probe
        torch.cuda.empty_cache()
        return {"card_gb": float(card_gb), "stt_compute": compute,
                "skipped": "драйвер отдаёт системную память сверх объёма карты — "
                           "выключите Sysmem Fallback в панели NVIDIA и повторите"}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
    ballast_bytes = int(free - float(card_gb) * 1024 ** 3)
    if ballast_bytes < 0:
        return {"skipped": f"на карте свободно меньше {card_gb} ГБ"}
    ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device="cuda")

    import voices
    from engines.stt_whisper import WhisperSTT
    from engines.tts_qwen import QwenTTS

    speech, sr = sf.read(WORK / "speech.wav", dtype="float32")
    import soxr
    speech = np.resize(soxr.resample(speech, sr, 16000).astype(np.float32), 60 * 16000)
    res = {"card_gb": float(card_gb), "stt_compute": compute,
           "ballast_gb": round(ballast.numel() / 1024 ** 3, 2)}
    try:
        tts, stt, v = QwenTTS(), WhisperSTT(), voices.default()
        tts.load()
        stt.load()
        for name in ("long", "xlong"):
            # как на сервере: синтез и распознавание по очереди, кэш PyTorch не чистится
            tts.speak(TEXTS[name], v.path, v.text, seed=1)
            stt.transcribe(speech, language="ru")
            res[name] = "ok"
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and "ALLOC" not in str(e):
            raise
        res["error"] = str(e).splitlines()[0][:160]
    return res


PARTS = {"tts": part_tts, "tts-min": part_tts_min, "stt": part_stt,
         "stt-int8": part_stt_int8, "both": part_both}
FITS = ("8:float16", "8:int8_float16", "6:float16", "6:int8_float16", "5.5:int8_float16")


# ------------------------------------------------------------------- сборка

def run_part(name: str, env: dict | None = None) -> dict:
    print(f"… {name} {env or ''}", file=sys.stderr, flush=True)
    out = subprocess.run([sys.executable, __file__, "--part", name],
                         capture_output=True, text=True, cwd=ROOT / "server",
                         env={**os.environ, **(env or {})})
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"{name} упал:\n{out.stderr[-3000:]}")


def table(r: dict) -> str:
    t, m, s, s8, b = r["tts"], r["tts-min"], r["stt"], r["stt-int8"], r["both"]
    lines = [f"**{r['machine']['gpu']}**, {r['machine']['cpu']}, {r['machine']['ram_gb']} ГБ ОЗУ", "",
             "| Синтез | Символов | Звук | Синтез | Скорость |", "|---|---|---|---|---|"]
    for name, x in t["texts"].items():
        best = max(x["runs"], key=lambda y: y["speed"])
        lines.append(f"| {name} | {x['chars']} | {best['audio_s']} с | {best['synth_s']} с | ×{best['speed']} |")
    lines += ["", f"Поток: первый кусок «{t['stream']['chunk']}» через {t['stream']['first_audio_s']} с "
              f"вместо {t['stream']['whole_s']} с целиком.", "",
              "| Распознавание | Запись | Время | Скорость |", "|---|---|---|---|"]
    for name, x in s["runs"].items():
        lines.append(f"| {s['compute_type']}, {name} | {x['audio_s']} с | {x['time_s']} с | ×{x['speed']} |")
    for name, x in s8["runs"].items():
        lines.append(f"| {s8['compute_type']}, {name} | {x['audio_s']} с | {x['time_s']} с | ×{x['speed']} |")
    lines += ["", "| Память | Видео | Оперативная |", "|---|---|---|",
              *[f"| синтез, минимум, {x['chars']} символов одним куском | {x['vram_min_gb']} ГБ | — |"
                for x in m["texts"].values()],
              f"| синтез, пик | {t['vram_peak_gb']} ГБ | {t['ram_peak_gb']} ГБ |",
              f"| распознавание {s['compute_type']}, пик | {s['vram_peak_gb']} ГБ | {s['ram_peak_gb']} ГБ |",
              f"| распознавание {s8['compute_type']}, пик | {s8['vram_peak_gb']} ГБ | {s8['ram_peak_gb']} ГБ |",
              f"| обе модели в одном процессе, пик | {b['vram_peak_gb']} ГБ | {b['ram_peak_gb']} ГБ |",
              "", f"Smart Turn: {b['turn_ms']} мс на проверку.", "",
              "| Сервер целиком на карте | Распознавание | 181 символ | 597 символов |",
              "|---|---|---|---|",
              *[f"| {x.get('card_gb', '?')} ГБ | {x.get('stt_compute', '')} | "
                f"{x.get('long', 'не влезло')} | {x.get('xlong', 'не влезло')} |"
                for x in r.get("fit", []) if "skipped" not in x],
              *[f"\nПроверка на меньших картах пропущена: {x['skipped']}."
                for x in r.get("fit", [])[:1] if "skipped" in x]]
    return "\n".join(lines)


def _out(r: dict) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in r["machine"]["gpu"].split(",")[0].lower())
    slug = "-".join(x for x in slug.split("-") if x and x not in ("nvidia", "geforce"))
    return ROOT / "results" / f"bench_{slug}.json"


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--part":
        res = {**PARTS, "fit": part_fit}[sys.argv[2]]()
        print("RESULT " + json.dumps(res, ensure_ascii=False))
        return
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    r = {"date": time.strftime("%Y-%m-%d"), "machine": machine()}
    if only:
        # перемерить одну часть, остальное взять из уже сохранённого файла этой машины
        r = json.loads(_out(r).read_text(encoding="utf-8"))
        if only == "fit":
            r["fit"] = [run_part("fit", {"BENCH_FIT": f}) for f in FITS]
        else:
            r[only] = run_part(only)
        r[f"{only}_date"] = time.strftime("%Y-%m-%d")
    else:
        for name in PARTS:
            r[name] = run_part(name)
        r["fit"] = [run_part("fit", {"BENCH_FIT": f}) for f in FITS]
    import difflib
    r["stt-int8"]["similarity_to_float16"] = round(difflib.SequenceMatcher(
        None, r["stt"]["text_60s"], r["stt-int8"]["text_60s"]).ratio(), 3)
    out = _out(r)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(table(r))
    print(f"\nсохранено: {out.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
