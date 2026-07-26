"""Objective 'eats letters' metric: synth -> ASR -> CER against the input text.

This is the same round-trip the TTS leaderboard uses, and it is exactly the guard
the build script needs: a config that swallows syllables shows up as a CER spike,
and no human has to listen to 600 files to notice.
"""
import json, re, sys, glob, pathlib, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from faster_whisper import WhisperModel
from jiwer import cer

PROBE = ("Слушай вопрос: чем опасно объявление внешнего ключа с он делит каскад? "
         "Подумай и ответь вслух. Короткий ответ: удаление одной строки "
         "может рекурсивно удалить тысячи связанных записей.")

samples = {s["id"]: s for s in json.load(open("samples.json", encoding="utf-8"))}


def norm(t: str) -> str:
    """Compare pronunciation, not orthography: lowercase, ё→е, strip punctuation."""
    t = t.lower().replace("ё", "е").replace("+", "")
    t = re.sub(r"[^а-яa-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


print("loading whisper large-v3-turbo...", flush=True)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def transcribe(path: str) -> str:
    segs, _ = model.transcribe(path, language="ru", beam_size=5)
    return " ".join(s.text for s in segs)


rows = []

# 1. parameter sweep (all on the same probe sentence)
for r in json.load(open("results_sweep.json", encoding="utf-8")):
    hyp = transcribe(f"sweep/{r['file']}")
    rows.append({"group": "sweep", "name": f"{r['ref']} nfe{r['nfe']} sp{r['speed']}",
                 "cer": round(cer(norm(PROBE), norm(hyp)) * 100, 1), "hyp": hyp.strip()})
    print(f"{rows[-1]['name']:34} CER {rows[-1]['cer']:5.1f}%", flush=True)

# 1b. human public-domain references (same probe)
if pathlib.Path("results_human.json").exists():
    for r in json.load(open("results_human.json", encoding="utf-8")):
        hyp = transcribe(f"human/{r['file']}")
        rows.append({"group": "human", "name": f"{r['voice']} / {r['model']}",
                     "cer": round(cer(norm(PROBE), norm(hyp)) * 100, 1), "hyp": hyp.strip()})
        print(f"HUMAN {rows[-1]['name']:22} CER {rows[-1]['cer']:5.1f}%", flush=True)

# 2. engine comparison on the full prepared samples
for f in glob.glob("results_*.json"):
    if "sweep" in f or "human" in f:  # probe-based, already handled above
        continue
    for r in json.load(open(f, encoding="utf-8")):
        if r.get("variant") != "audio":
            continue
        p = pathlib.Path("out") / r["file"]
        if not p.exists():
            continue
        ref = samples[r["id"]]["audio"]
        hyp = transcribe(str(p))
        rows.append({"group": "engines", "name": f"{r['engine']} {r['voice']}", "id": r["id"],
                     "cer": round(cer(norm(ref), norm(hyp)) * 100, 1), "hyp": hyp.strip()})
        print(f"{rows[-1]['name']:26} {r['id']:8} CER {rows[-1]['cer']:5.1f}%", flush=True)

json.dump(rows, open("cer.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
