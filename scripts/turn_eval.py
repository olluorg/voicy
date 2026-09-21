"""Does Smart Turn tell a finished Russian phrase from one cut mid-thought?

Material: the validation split of Golos crowd — people reading commands to a
voice assistant, which is exactly the agent's situation. Each recording ≥ 4 words
yields two samples, both followed by 0.2 s of pause (the moment a voice detector
first reports one):

  complete   — up to where the voice detector says speech ended, then the
               recording's own trailing silence;
  incomplete — up to where the *next* word starts, in the middle of the phrase,
               then the recording's leading room noise.

Word boundaries come from the running voicy server (verbose_json). Whisper puts
word ends early often enough that cutting at them clips the last phoneme — and a
clipped phrase sounds unfinished. So the end of speech is taken from the voice
detector, and a mid-phrase cut is made where the next word begins.

The live endpoint does not trust the model alone: when it says "complete",
the turn is read, and a turn ending on a preposition, a conjunction or a filler
("в", "и", "ээ", "...") is kept open. Both are scored — the model alone, and the
model with that text veto — on the same samples.

    python scripts/turn_eval.py <golos.json> [limit]  →  results/results_turn.json
"""
import io
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from engines.turn_smart import SmartTurn  # noqa: E402
from live import dangling  # noqa: E402

URL = "http://127.0.0.1:8080"
SR = 16000
PAUSE = 0.2
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def recognise(wav: bytes) -> dict:
    b = "BOUND"
    body = (f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\n'
            f'verbose_json\r\n--{b}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'
            f'ru\r\n--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
            ).encode() + wav + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(OPENER.open(req, timeout=120).read())


def words(wav: bytes) -> list[dict]:
    return [w for s in recognise(wav)["segments"] for w in s["words"]]


def text_of(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return recognise(buf.getvalue())["text"]


def with_pause(audio: np.ndarray, end: float, noise: np.ndarray, real: bool) -> np.ndarray:
    cut, n = int(end * SR), int(PAUSE * SR)
    if real and len(audio) - cut >= n:           # за концом речи — настоящая тишина записи
        return audio[:cut + n].astype(np.float32)
    tail = np.resize(noise, n) if len(noise) >= SR // 20 else np.zeros(n, np.float32)
    return np.concatenate([audio[:cut], tail]).astype(np.float32)


def speech_end(audio: np.ndarray) -> float | None:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=200, speech_pad_ms=0))
    return ts[-1]["end"] / SR if ts else None


def main() -> None:
    items = json.load(open(sys.argv[1], encoding="utf-8"))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(items)
    base = Path(sys.argv[1]).parent
    random.seed(0)
    model = SmartTurn()
    model.load()

    rows, took = [], []
    for it in items:
        if len(rows) >= limit:
            break
        if len(it["text"].split()) < 4:
            continue
        raw = (base / it["file"]).read_bytes()
        a, sr = sf.read(io.BytesIO(raw), dtype="float32")
        a = soxr.resample(a if a.ndim == 1 else a.mean(1), sr, SR)
        ws = words(raw)
        if len(ws) < 4:
            continue
        end = speech_end(a)
        if end is None:
            continue
        noise = a[:max(0, int((ws[0]["start"] - 0.05) * SR))]
        k = random.randint(1, len(ws) - 3)            # не первое и не последнее слово
        cut = max(ws[k]["end"], ws[k + 1]["start"])
        samples = {"complete": (end, True), "incomplete": (cut, False)}
        row = {"file": it["file"], "text": it["text"], "cut_after": ws[k]["word"].strip()}
        for kind, (at, real) in samples.items():
            clip = with_pause(a, at, noise, real)
            t = time.perf_counter()
            row[kind] = round(model.probability(clip), 4)
            took.append(time.perf_counter() - t)
            row[f"{kind}_text"] = text_of(clip)
            row[f"{kind}_veto"] = dangling(row[f"{kind}_text"])
        rows.append(row)

    report = {"n": len(rows), "pause_s": PAUSE, "ms_per_call": round(1000 * float(np.median(took)), 1),
              "thresholds": {}}
    def score(fn: int, fp: int) -> dict:
        return {"accuracy": round(100 * (2 * len(rows) - fn - fp) / (2 * len(rows)), 1),
                "false_complete_pct": round(100 * fp / len(rows), 1),
                "false_incomplete_pct": round(100 * fn / len(rows), 1)}

    for th in (0.3, 0.5, 0.7, 0.9):
        # fn — законченную фразу приняли за незаконченную: агент ждёт max_pause;
        # fp — незаконченную за законченную: агент перебивает на полуслове
        report["thresholds"][str(th)] = {
            "model": score(sum(r["complete"] < th for r in rows),
                           sum(r["incomplete"] >= th for r in rows)),
            "model+text": score(sum(r["complete"] < th or r["complete_veto"] for r in rows),
                                sum(r["incomplete"] >= th and not r["incomplete_veto"]
                                    for r in rows))}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    out = Path(__file__).resolve().parents[1] / "results" / "results_turn.json"
    # Тексты записей остаются локально: в репозиторий идут только номера файлов
    # датасета, вероятности и решения проверки по тексту.
    keep = ("file", "complete", "incomplete", "complete_veto", "incomplete_veto")
    out.write_text(json.dumps({"report": report, "rows": [{k: r[k] for k in keep} for r in rows]},
                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
