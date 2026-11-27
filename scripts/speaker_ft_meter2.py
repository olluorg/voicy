"""Stress in the held-out paragraphs of experiments/29, judged by meter2.

meter2 (siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf, meter/) force-aligns the
known text with a Russian wav2vec2 CTC model and picks each word's stressed vowel
from its hidden states: 98.3% on unseen LibriSpeech readers, where the loudness
detector of stress_baseline.py had 68–77% and on these paragraphs flagged the
reader more often than the synthesis (speaker_ft_stress.py).

Expected stress is what the server marked (/v1/text/prepare): the very marks the
model was given. The reader's own reading is judged too, first: his disagreement
is RUAccent's errors plus the detector's, the floor for A and C.

    METER2_DEVICE=cuda python scripts/speaker_ft_meter2.py [variant ...]   # human A C by default
    python scripts/speaker_ft_meter2.py report

Research only: meter2's head was trained partly on Silero renders (non-commercial).
"""
import ast, glob, json, os, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import requests, soundfile as sf

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
os.environ.setdefault("METER2_HEAD", os.path.join(WORK, "meter2", "meter", "head.pt"))
sys.path.insert(0, os.path.join(WORK, "meter2", "meter"))
from meter2 import judge, words_in, marked_index, vowel_pos

URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080").rstrip("/")
EVAL = os.path.join(WORK, "eval")
OUT = os.environ.get("METER2_ROWS") or os.path.join(WORK, "meter2", "rows.json")
MARKED = re.compile(r"[А-Яа-яЁё́]+(?:-[А-Яа-яЁё́]+)*")


def marks(text):
    """Server's marks for every word of words_in(text): vowel index or None."""
    r = requests.post(f"{URL}/v1/text/prepare", json={"text": text}, timeout=60, proxies={"http": None})
    r.raise_for_status()
    marked = r.json().get("text") or text
    out = [marked_index(w) for w in MARKED.findall(marked)]
    return out if len(out) == len(words_in(text)) else None


def main(variants):
    rows = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for meta in sorted(glob.glob(os.path.join(EVAL, "human", "*.json"))):
        if meta.endswith(".measure.json"):
            continue
        pid = os.path.basename(meta)[:-5]
        text = json.load(open(meta, encoding="utf-8"))["text"]
        row = rows.setdefault(pid, {"text": text, "words": [w for w, _, _ in words_in(text)], "files": {}})
        if "expected" not in row:
            row["expected"] = marks(text)
            if row["expected"] is None:
                print(f"{pid}: слова разметки не совпали со словами текста — пропуск", flush=True)
                continue
        for v in variants:
            pat = pid + ".wav" if v == "human" else pid + "_s*.wav"
            for f in sorted(glob.glob(os.path.join(EVAL, v, pat))):
                key = f"{v}/{os.path.basename(f)}"
                if key in row["files"]:
                    continue
                res = by_sentence(f, text)
                if res is None:
                    print(f"  {key}: предложения не сошлись с текстом — пропуск", flush=True)
                    continue
                row["files"][key] = res
                json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{pid}: {len(row['words'])} слов, {len(row['files'])} файлов", flush=True)
    report(rows)


def by_sentence(path, text):
    """judge() sentence by sentence: on a whole 40–60 s paragraph the forced alignment
    falls apart (confidence ~0.37 on every word, below meter2's 0.5), and meter2 was
    built on single utterances. Sentence spans are the file's own, from Whisper
    (speaker_ft_eval.py, .measure.json)."""
    x, sr = sf.read(path, dtype="float32")
    sents = json.load(open(path[:-4] + ".measure.json", encoding="utf-8"))["sentences"]
    if isinstance(sents, str):
        sents = ast.literal_eval(sents)
    # край, который Whisper не нашёл, — край соседнего предложения
    dur = len(x) / sr
    for i, s in enumerate(sents):
        if s["start"] is None:
            s["start"] = next((p["end"] for p in reversed(sents[:i]) if p["end"] is not None), 0.0)
        if s["end"] is None:
            s["end"] = next((n["start"] for n in sents[i + 1:] if n["start"] is not None), dur)
    out = []
    for s in sents:
        a = int(max(0, s["start"] - 0.15) * sr)
        b = int(min(dur, s["end"] + 0.15) * sr)
        out += [(w["best"], w["margin"]) for w in judge(x[a:b], sr, s["text"])]
    return out if len(out) == len(words_in(text)) else None


def wrong(row, key):
    """{word index: heard vowel} where the word has ≥2 vowels and the audio disagrees."""
    out, n = {}, 0
    for i, ((best, _), exp) in enumerate(zip(row["files"][key], row["expected"])):
        if best is None or exp is None or len(vowel_pos(row["words"][i])) < 2:
            continue
        n += 1
        if best != exp:
            out[i] = best
    return out, n


def report(rows):
    tot = defaultdict(lambda: [0, 0])
    for r in rows.values():
        if r.get("expected") is None:
            continue
        for key in r["files"]:
            w, n = wrong(r, key)
            tot[key.split("/")[0]][0] += len(w); tot[key.split("/")[0]][1] += n
    print("\nСлова с двумя и больше гласными, где прозвучавшее ударение расходится с разметкой сервера:")
    for v, (bad, n) in tot.items():
        print(f"  {v:>6}: {bad / max(n, 1):5.1%}  ({bad} из {n})")

    print("\nСлова, где вариант разошёлся с разметкой в большинстве сидов, а чтец — нет:")
    for pid, r in rows.items():
        if r.get("expected") is None or f"human/{pid}.wav" not in r["files"]:
            continue
        human, _ = wrong(r, f"human/{pid}.wav")
        for v in sorted({k.split("/")[0] for k in r["files"]} - {"human"}):
            keys = [k for k in r["files"] if k.startswith(v + "/")]
            cnt = defaultdict(int)
            for k in keys:
                for i in wrong(r, k)[0]:
                    cnt[i] += 1
            bad = [i for i, c in sorted(cnt.items()) if c * 2 > len(keys) and i not in human]
            if bad:
                print(f"  {pid} {v}: " + ", ".join(show(r["words"][i], r["expected"][i]) for i in bad))
        if human:
            print(f"  {pid} чтец против разметки: " + ", ".join(
                f"{show(r['words'][i], r['expected'][i])} → {show(r['words'][i], b)}" for i, b in human.items()))


def show(word, k):
    vp = vowel_pos(word)
    return word[:vp[k] + 1] + "́" + word[vp[k] + 1:] if k is not None and k < len(vp) else word


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(json.load(open(OUT, encoding="utf-8")))
    else:
        main(sys.argv[1:] or ["human", "A", "C"])
