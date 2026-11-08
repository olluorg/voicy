"""Wrong stress in the held-out paragraphs of experiments/29: A, C and the reader.

The detector of scripts/stress_baseline.py (word loudness contour against Silero
readings of the same sentence with the stress on each vowel in turn) is noisy
word by word: 10–12% false alarms. Here the reader's own reading of the same
paragraph is measured alongside, so its rate is the false-alarm floor on exactly
this text, and only what a variant has above it counts. A word flagged in most
seeds of a variant and not in the reader is a candidate to listen to.

    python scripts/speaker_ft_stress.py          # templates and detection, resumable
    python scripts/speaker_ft_stress.py report

Needs the voicy server (word timestamps), Silero v5_ru.pt in work/stress and
RUAccent; both run on the CPU.
"""
import glob, json, os, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
os.environ.setdefault("STRESS_WORK", os.path.join(WORK, "stress"))
sys.path.insert(0, os.path.dirname(__file__))
from stress_baseline import setup, words_of, detect, verdicts

EVAL = os.path.join(WORK, "eval")
ROWS = os.path.join(WORK, "stress", "rows.json")
VARIANTS = ("human", "A", "C")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if re.search(r"[А-Яа-яЁё]", s)]


def main():
    acc, silero_say, templates_in_context = setup()
    rows = json.load(open(ROWS, encoding="utf-8")) if os.path.exists(ROWS) else {}
    for meta in sorted(glob.glob(os.path.join(EVAL, "human", "*.json"))):
        if meta.endswith(".measure.json"):
            continue
        pid = os.path.basename(meta)[:-5]
        if pid in rows:
            continue
        text = json.load(open(meta, encoding="utf-8"))["text"]
        # разметка и эталоны — по предложениям: Silero и RUAccent работают с фразой
        expected, tpl = [], {}
        for s in sentences(text):
            exp = words_of(acc.process_all(s))
            for i, per_k in templates_in_context(exp).items():
                tpl[len(expected) + i] = per_k
            expected += exp
        row = {"text": text, "words": [w for w, _ in expected], "expected": [e for _, e in expected], "files": {}}
        files = [os.path.join(EVAL, "human", pid + ".wav")]
        files += sorted(glob.glob(os.path.join(EVAL, "A", pid + "_s*.wav")))
        files += sorted(glob.glob(os.path.join(EVAL, "C", pid + "_s*.wav")))
        for f in files:
            x, sr = sf.read(f, dtype="float32")
            row["files"][os.path.relpath(f, EVAL).replace("\\", "/")] = detect(x, sr, expected, tpl)
        rows[pid] = row
        json.dump(rows, open(ROWS, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{pid}: {len(expected)} слов, {len(files)} файлов", flush=True)
    report(rows)


def variant(key):
    return key.split("/")[0]


def report(rows):
    print("\nДоля проверенных слов, где детектор спорит с RUAccent (оригинал — потолок ложных тревог):")
    print("порог | " + " | ".join(f"{v:>14}" for v in VARIANTS))
    for m in (0.0, 0.1, 0.2, 0.3):
        cells = []
        for v in VARIANTS:
            bad = [b for r in rows.values() for k, d in r["files"].items() if variant(k) == v
                   for _, b in verdicts(d, r["expected"], m).values()]
            cells.append(f"{np.mean(bad):5.1%} из {len(bad):4}")
        print(f"{m:5.2f} | " + " | ".join(f"{c:>14}" for c in cells))

    m = 0.2
    print(f"\nСлова, где вариант ошибся в большинстве сидов, а чтец — нет (порог {m}):")
    for pid, r in rows.items():
        flagged = defaultdict(lambda: defaultdict(int)); seeds = defaultdict(int)
        for k, d in r["files"].items():
            seeds[variant(k)] += 1
            for i, (best, b) in verdicts(d, r["expected"], m).items():
                if b:
                    flagged[variant(k)][i] += 1
        for v in ("A", "C"):
            words = [r["words"][i] for i, n in sorted(flagged[v].items())
                     if n * 2 > seeds[v] and i not in flagged["human"]]
            if words:
                print(f"  {pid} {v}: {', '.join(words)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open(ROWS, encoding="utf-8")))
    else:
        main()
