"""Held-out paragraphs of «Дворянское гнездо»: synthesis against the reader himself.

The model never heard chapters 23–25, and the reader's own reading of each of
their paragraphs is at hand. So every measure here is a distance to the same
text read by the same man, not to a human norm in general — the comparison
experiments/26 could not make.

The server is whatever runs at VOICY_URL: the shipped model for A, the merged
adapter for C (scripts/speaker_ft_gguf.py). The voice is turgenev for both, the
same 10-second sample.

    python scripts/speaker_ft_eval.py synth A      # синтез отложенных абзацев
    python scripts/speaker_ft_eval.py measure      # всё, что ещё не измерено
    train-venv/Scripts/python scripts/speaker_ft_eval.py speaker   # сходство голоса, без сервера
    python scripts/speaker_ft_eval.py report
    python scripts/speaker_ft_eval.py discriminate   # отличит ли классификатор чтеца от варианта

Measures, per variant, against the original:
  cer      — round trip through Whisper (ADR 0001)
  reset    — pitch of the next sentence's first 0.4 s minus the last 0.4 s, semitones (experiments/26)
  fall_sd  — spread of the fall within a sentence: the original varies its endings, synthesis tends not to
  broken   — share of syntheses that cut the text off or lost part of it (CER > MAX_CER)
  pauses   — pauses ≥ PAUSE per minute and share of silence, by sound
  wpm      — words per minute
  spk      — cosine of Qwen's speaker embedding to the original of the same paragraph
"""
import difflib, glob, io, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, requests, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import intonation_eval as ie

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
EVAL = os.path.join(WORK, "eval")
URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080")
ie.URL = URL
N = 12  # абзацев из отложенных глав
SEEDS = (1, 2, 3)
PAUSE = 0.15  # с — тишина, которая слышится паузой
MAX_CER = 0.05  # выше — синтез сорвался: оборвал или пропустил текст


def paragraphs():
    rows = [json.loads(l) for l in open(os.path.join(WORK, "ds", "test.jsonl"), encoding="utf-8")]
    rows = sorted((r for r in rows if 25 <= r["duration"] <= 50), key=lambda r: r["id"])
    # равномерно по отложенным главам, а не первые подряд
    return [rows[int(i * len(rows) / N)] for i in range(min(N, len(rows)))]


def meta(path, text):
    json.dump({"paragraphs": [text], "text": text}, open(path[:-4] + ".json", "w", encoding="utf-8"),
              ensure_ascii=False)


def synth(tag):
    os.makedirs(os.path.join(EVAL, "human"), exist_ok=True)
    os.makedirs(os.path.join(EVAL, tag), exist_ok=True)
    print(requests.get(URL + "/health", proxies=ie.NO_PROXY).json()["tts"]["model"])
    for p in paragraphs():
        h = os.path.join(EVAL, "human", f"{p['id']}.wav")
        if not os.path.exists(h):
            x, sr = sf.read(p["audio"], dtype="float32")
            sf.write(h, x, sr); meta(h, p["plain"])
        for seed in SEEDS:
            out = os.path.join(EVAL, tag, f"{p['id']}_s{seed}.wav")
            if os.path.exists(out):
                continue
            a, sr = ie.speak(p["plain"], "turgenev", seed)
            sf.write(out, a, sr); meta(out, p["plain"])
            print(f"  {tag} {p['id']} сид {seed}: {len(a) / sr:.1f} с (оригинал {p['duration']:.1f} с)", flush=True)


def pauses(text, tr):
    """Indices of the text's words followed by a pause, by matching Whisper's words to the text."""
    asr = [w for s in tr.get("segments", []) for w in s.get("words", [])] or tr.get("words", [])
    ref = ie.words(text)
    hyp = [ie.words(w["word"])[0] if ie.words(w["word"]) else "" for w in asr]
    at = {}
    for b in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks():
        for j in range(b.size):
            at[b.a + j] = b.b + j
    out = set()
    for k in range(len(ref) - 1):
        if k in at and k + 1 in at and at[k + 1] == at[k] + 1 and asr[at[k] + 1]["start"] - asr[at[k]]["end"] >= PAUSE:
            out.add(k)
    return out, len(ref)


def measure():
    # measure_one из 26 распознаёт файл сама; паузам нужна та же расшифровка, а не вторая
    heard, transcribe = {}, ie.transcribe
    ie.transcribe = lambda p: heard[p] if p in heard else heard.setdefault(p, transcribe(p))
    for f in sorted(glob.glob(os.path.join(EVAL, "*", "*.wav"))):
        out = f[:-4] + ".measure.json"
        if os.path.exists(out):
            continue
        m = ie.measure_one(f)
        text = json.load(open(f[:-4] + ".json", encoding="utf-8"))["text"]
        tr = ie.transcribe(f)
        ps, n = pauses(text, tr)
        m.update({"pauses_at": sorted(ps), "n_words": n})
        json.dump(m, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(os.path.relpath(f, EVAL), f"CER {m['cer']:.1%}, пауз {len(ps)}", flush=True)


def speaker():
    """Qwen's own speaker encoder, as in experiments/23: output against the original paragraph."""
    import torch
    from speaker_ft_train import BASE
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16).model
    emb = lambda p: m.extract_speaker_embedding(*sf.read(p, dtype="float32")).float().flatten()
    for f in sorted(glob.glob(os.path.join(EVAL, "*", "*.wav"))):
        if os.sep + "human" + os.sep in f:
            continue
        mj = f[:-4] + ".measure.json"
        if not os.path.exists(mj):
            continue
        d = json.load(open(mj, encoding="utf-8"))
        if "spk" in d:
            continue
        h = os.path.join(EVAL, "human", os.path.basename(f).rsplit("_s", 1)[0] + ".wav")
        d["spk"] = float(torch.nn.functional.cosine_similarity(emb(f), emb(h), dim=0))
        json.dump(d, open(mj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("сходство голоса посчитано")


def acoustic(path, hop=0.01):
    """Pauses by sound, the way speaker_ft_data.py finds them.

    Whisper's word times stick together on this reader (2 pauses a minute in the
    original by them), so its words cannot count pauses. Quiet is the noise floor
    plus a third of the speech range; only pauses between the first and last sound count.
    """
    x, sr = sf.read(path, dtype="float32")
    n = int(hop * sr)
    db = 20 * np.log10(np.sqrt(np.mean(x[:len(x) // n * n].reshape(-1, n) ** 2, axis=1) + 1e-10))
    floor, speech = np.percentile(db, 5), np.percentile(db, 95)
    quiet = db < floor + (speech - floor) / 3
    loud = np.flatnonzero(~quiet)
    runs, run = [], 0
    for q in quiet[loud[0]:loud[-1] + 1]:
        if q:
            run += 1
        elif run:
            runs.append(run * hop); run = 0
    runs = [r for r in runs if r >= PAUSE]
    return len(runs), sum(runs)


def boot(groups, stat, n=2000):
    """95% interval of stat over paragraphs resampled with replacement."""
    rng = np.random.default_rng(29)
    keys = list(groups)
    vals = [stat([v for k in rng.choice(keys, len(keys)) for v in groups[k]]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def report():
    """Prosody only over files that read the whole text: a paragraph cut off has no ending to measure."""
    load = lambda p: json.load(open(p, encoding="utf-8"))
    tags = ["human"] + sorted(d for d in os.listdir(EVAL) if d != "human" and os.path.isdir(os.path.join(EVAL, d)))
    print("| | срывов | CER | сброс, пт | разброс падения, пт | пауз/мин | тишины | слов/мин | голос |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag in tags:
        ms = {p[:-13]: load(p) for p in glob.glob(os.path.join(EVAL, tag, "*.measure.json"))}
        if not ms:
            continue
        whole = {k: m for k, m in ms.items() if m["cer"] <= MAX_CER}
        para = lambda k: os.path.basename(k).rsplit("_s", 1)[0]
        resets, falls = {}, {}
        for k, m in whole.items():
            resets.setdefault(para(k), []).extend(x["reset"] for x in m["bounds"])
            falls.setdefault(para(k), []).extend(x["onset"] - x["coda"] for x in m["sentences"]
                                                 if x["onset"] is not None and x["coda"] is not None)
        r_all = [v for vs in resets.values() for v in vs]
        f_all = [v for vs in falls.values() for v in vs]
        r_lo, r_hi = boot(resets, np.mean)
        f_lo, f_hi = boot(falls, np.std)
        pauses = [acoustic(k + ".wav") for k in whole]
        secs = sum(m["seconds"] for m in whole.values())
        spk = [m["spk"] for m in whole.values() if "spk" in m]
        cells = [tag, f"{len(ms) - len(whole)} из {len(ms)}", f"{np.median([m['cer'] for m in whole.values()]):.1%}",
                 f"{np.mean(r_all):.1f} [{r_lo:.1f}…{r_hi:.1f}]", f"{np.std(f_all):.1f} [{f_lo:.1f}…{f_hi:.1f}]",
                 f"{sum(n for n, _ in pauses) / secs * 60:.0f}", f"{sum(t for _, t in pauses) / secs:.0%}",
                 f"{sum(m['n_words'] for m in whole.values()) / secs * 60:.0f}",
                 f"{np.mean(spk):.3f}" if spk else "—"]
        print("| " + " | ".join(cells) + " |")
    print(f"\nсрыв — CER > {MAX_CER:.0%}; мелодия, паузы и темп — по остальным; в скобках 95% по абзацам")


FEATURES = ("сброс", "разброс сброса", "падение", "разброс падения", "диапазон", "пауз/мин", "тишина",
            "средняя пауза", "слов/мин")
GROUPS = {"всё": range(9), "мелодия": range(5), "темп и паузы": range(5, 9)}


def features(path, m):
    resets = [x["reset"] for x in m["bounds"]]
    falls = [x["onset"] - x["coda"] for x in m["sentences"] if x["onset"] is not None and x["coda"] is not None]
    n, quiet = acoustic(path)
    return [np.mean(resets) if resets else 0.0, np.std(resets) if resets else 0.0,
            np.mean(falls) if falls else 0.0, np.std(falls) if falls else 0.0, m["range"],
            n / m["seconds"] * 60, quiet / m["seconds"], quiet / max(n, 1), m["n_words"] / m["seconds"] * 60]


def discriminate():
    """Can a plain classifier tell the reader from a variant, paragraph by paragraph?

    Logistic regression on paragraph-level prosody, original against one variant,
    every paragraph held out in turn with all its files: a classifier that has
    seen the text cannot win by remembering it. 50% — the variant passes for the
    reader on these features; 100% — it never does. Only files that read the
    whole text: a cut-off paragraph is told apart by its length alone.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    load = lambda p: json.load(open(p, encoding="utf-8"))
    rows = {}
    for p in glob.glob(os.path.join(EVAL, "*", "*.measure.json")):
        m = load(p)
        if m["cer"] <= MAX_CER:
            tag = os.path.basename(os.path.dirname(p))
            para = os.path.basename(p)[:-13].rsplit("_s", 1)[0]
            rows.setdefault(tag, []).append((para, features(p[:-13] + ".wav", m)))
    human = rows.pop("human")
    print("| вариант | признаки | файлов | точность различителя | главные признаки |")
    print("|---|---|---|---|---|")
    for tag, (group, cols) in ((t, g) for t in sorted(rows) for g in GROUPS.items()):
        pick = lambda f: [f[i] for i in cols]
        data = [(p, pick(f), 1) for p, f in human] + [(p, pick(f), 0) for p, f in rows[tag]]
        hits, n = 0, 0
        for para in {p for p, _, _ in data}:
            train = [(f, y) for p, f, y in data if p != para]
            test = [(f, y) for p, f, y in data if p == para]
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            clf.fit([f for f, _ in train], [y for _, y in train])
            # человек и вариант весят поровну, сколько бы файлов ни было у каждого
            for y in (0, 1):
                fs = [f for f, yy in test if yy == y]
                if fs:
                    hits += np.mean(clf.predict(fs) == y); n += 1
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        clf.fit([f for _, f, _ in data], [y for _, _, y in data])
        coef = clf[-1].coef_[0]
        top = ", ".join(f"{FEATURES[cols[i]]} {'↑' if coef[i] > 0 else '↓'}" for i in np.argsort(-np.abs(coef))[:3])
        print(f"| {tag} | {group} | {len(rows[tag])} | {hits / n:.0%} | {top} |")
    print()
    print("↑ — у чтеца больше, ↓ — меньше")


if __name__ == "__main__":
    cmd = sys.argv[1]
    synth(sys.argv[2]) if cmd == "synth" else {"measure": measure, "speaker": speaker, "report": report,
                                               "discriminate": discriminate}[cmd]()
