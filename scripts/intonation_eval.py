"""Does the intonation start over at every sentence, and what makes it carry on?

experiments/26. A listener hears Qwen3-TTS read sentence after sentence as if
each were said on its own: the voice falls at the full stop and the next one
starts again from the top. Each variant reads the same three paragraphs; at
every sentence boundary the pitch is measured on both sides.

  sent   — as the stream synthesises: a pass per piece of the segmenter
           (stream_format=sse gives exactly the production pieces);
  whole  — the paragraph in one pass, as `voicy say`;
  chain  — the stream's pieces, but each is prompted with the voice's
           reference followed by the piece before it: the model continues
           the previous sentence instead of the reference;
  semi   — one pass, full stops inside the paragraph turned into semicolons;
  chain_semi — the chain over the semicolon text.

The human baseline is prose from Russian LibriSpeech, four readers, runs of
consecutive utterances (their pauses are cut by the corpus, so only pitch is
compared there).

Per boundary: reset — pitch at the start of the next sentence minus pitch at
the end of this one, in semitones; gap — silence between them. Per paragraph:
how the sentence onsets drift across it (a reader holding the paragraph lets
them step down; restarting puts every one at the same height).

    python scripts/intonation_eval.py human     # the baseline audio
    python scripts/intonation_eval.py synth     # a server of its own on :8094
    python scripts/intonation_eval.py measure   # transcribe, align, pitch
    python scripts/intonation_eval.py report
"""
import base64, io, json, os, re, subprocess, sys, time, difflib, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, requests, soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "experiments", "26-intonation-across-sentences", "work")
RULS = os.path.join(ROOT, "experiments", "23-stress-finetune", "work", "ruls")
BIN = os.environ.get("VOICY_BIN", os.path.join(ROOT, "rust/target/release/voicy"))
PORT = 8094
URL = f"http://localhost:{PORT}"
NO_PROXY = {"http": None, "https": None}
VOICES = ("turgenev", "dostoevsky")
SEEDS = (1, 2)
VARIANTS = ("sent", "whole", "chain", "semi", "chain_semi")

PARAGRAPHS = [
    "Модель не знает, где кончается мысль. Она видит только точку. Каждая точка для неё конец "
    "высказывания. Поэтому голос опускается и замолкает. Следующая фраза начинается с той же высоты. "
    "Так читают список, а не рассказ. Живой человек держит мысль дольше. Он опускает голос только тогда, "
    "когда договорил.",
    "Утром пошёл мелкий дождь. Мы вышли из дома позже, чем собирались. Дорога к реке шла через старый сад. "
    "Яблони там давно никто не обрезал. Под ногами хрустели ветки и прошлогодние листья. У самой воды "
    "стояла лодка, наполовину вытащенная на берег. Отец молча столкнул её в реку и сел на вёсла.",
    "Очередь заданий живёт в памяти сервера. Когда задание готово, оно хранится ещё полчаса. За это время "
    "клиент может забрать звук или расшифровку. Если сервер перезапустить, очередь пропадёт вместе с ним. "
    "Для долгой работы это неудобно, но зато просто. Ничего не нужно настраивать и нечего чистить руками.",
]

# человек: (шард, книга, номер первой фразы, сколько секунд)
HUMAN = [
    ("train-00002.parquet", "obryv_006_goncharov", 40, 100),
    ("train-00021.parquet", "shortstories_09_garshin", 40, 100),
    ("train-00008.parquet", "vekhi_02_various", 40, 100),
    ("train-00015.parquet", "belkintales_05_pushkin", 40, 100),
    ("train-00007.parquet", "veshniyevody_08_turgenev", 40, 100),
]

SENT_END = re.compile(r"(?<=[.!?…])[»\"')\]]*\s+(?=[^\sа-яёa-z])")


def sentences(text):
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def semi(paragraph):
    ss = sentences(paragraph)
    out = [s[:-1] + ";" if s.endswith(".") else s for s in ss[:-1]] + [ss[-1]]
    out = [out[0]] + [s[0].lower() + s[1:] for s in out[1:]]
    return " ".join(out)


def words(text):
    return [w for w in (re.sub(r"[^а-яёa-z0-9-]", "", t.lower().replace("ё", "е")) for t in text.split()) if w]


def cer(ref, hyp):
    a = re.sub(r"[^а-яa-z]", "", ref.lower().replace("ё", "е"))
    b = re.sub(r"[^а-яa-z]", "", hyp.lower().replace("ё", "е"))
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal") / max(len(a), 1)


# ------------------------------------------------------------------ человек

def human():
    import pyarrow.parquet as pq
    out = os.path.join(WORK, "human")
    os.makedirs(out, exist_ok=True)
    for shard, book, first, seconds in HUMAN:
        # по одному шарду и только нужные строки: память WSL не резиновая
        t = pq.read_table(os.path.join(RULS, shard), columns=["audio_filepath", "text_no_preprocessing", "duration"]).to_pandas()
        rows = t[t.audio_filepath.str.contains(book)].sort_values("audio_filepath")
        rows = rows.iloc[first:]
        take, total = [], 0.0
        for _, r in rows.iterrows():
            take.append(r.audio_filepath)
            total += r.duration
            if total >= seconds:
                break
        tab = pq.read_table(os.path.join(RULS, shard), filters=[("audio_filepath", "in", take)]).to_pandas()
        tab = tab.set_index("audio_filepath").loc[take]
        chunks, texts = [], []
        for _, r in tab.iterrows():
            a, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            chunks.append(a)
            texts.append(r.text_no_preprocessing.strip())
        name = book.split("_")[0] + "_" + book.split("_")[-1]
        sf.write(os.path.join(out, name + ".wav"), np.concatenate(chunks), sr)
        json.dump({"text": " ".join(texts), "paragraphs": [" ".join(texts)]},
                  open(os.path.join(out, name + ".json"), "w"), ensure_ascii=False)
        print(name, f"{total:.0f} с", len(sentences(" ".join(texts))), "предложений")
        del t, tab


# ------------------------------------------------------------------- синтез

def serve():
    vdir = os.path.join(WORK, "voices")
    os.makedirs(vdir, exist_ok=True)
    for f in ("voices.json", "turgenev.wav", "dostoevsky.wav"):
        if not os.path.exists(os.path.join(vdir, f)):
            subprocess.run(["cp", os.path.join(ROOT, "server/voices", f), vdir], check=True)
    env = {**os.environ, "VOICY_VOICES_DIR": vdir, "NO_PROXY": "127.0.0.1,localhost"}
    p = subprocess.Popen([BIN, "serve", "--port", str(PORT)], env=env, cwd=ROOT,
                         stdout=open(os.path.join(WORK, "serve.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if requests.get(URL + "/health", timeout=2, proxies=NO_PROXY).ok:
                return p
        except requests.RequestException:
            pass
        if p.poll() is not None:
            raise RuntimeError("сервер упал, см. serve.log")
        time.sleep(1)
    p.kill()
    raise RuntimeError("сервер не поднялся")


def speak(text, voice, seed):
    r = requests.post(URL + "/v1/audio/speech", proxies=NO_PROXY, timeout=1800,
                      json={"input": text, "voice": voice, "response_format": "wav", "seed": seed})
    r.raise_for_status()
    a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return a, sr


def pieces(text, voice, seed):
    """The stream's own pieces: text and audio of each."""
    r = requests.post(URL + "/v1/audio/speech", proxies=NO_PROXY, timeout=1800, stream=True,
                      json={"input": text, "voice": voice, "response_format": "wav", "seed": seed,
                            "stream_format": "sse"})
    r.raise_for_status()
    r.encoding = "utf-8"  # text/event-stream приходит без charset, requests взял бы latin-1
    out = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev["type"] == "speech.audio.delta":
            a, sr = sf.read(io.BytesIO(base64.b64decode(ev["audio"])), dtype="float32")
            out.append((ev["text"], a, sr))
    return out


def add_voice(name, audio, sr, text):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    r = requests.post(URL + "/v1/voices", proxies=NO_PROXY, timeout=600,
                      files={"file": ("ref.wav", buf.getvalue(), "audio/wav")},
                      data={"name": name, "text": text, "replace": "true"})
    r.raise_for_status()


RETRIES = []


def ref_words(voice):
    """Word timings of the voice's reference, to cut its tail at a word."""
    path = os.path.join(WORK, "voices", voice + ".words.json")
    if not os.path.exists(path):
        json.dump(transcribe(os.path.join(WORK, "voices", voice + ".wav")), open(path, "w"), ensure_ascii=False)
    tr = json.load(open(path))
    return [w for s in tr["segments"] for w in s["words"]]


def chain(text, voice, seed, texts):
    """Piece by piece, each prompted with the tail of the reference plus the
    piece before, no longer than the reference itself.

    The prompt lays the transcript over the reference frames token by token;
    a reference simply made longer puts all of the new text, its end marker
    included, under the prompt, and the model reads nothing or skips (the
    first run: whole sentences lost, a piece of 0 samples)."""
    meta = json.load(open(os.path.join(WORK, "voices", "voices.json")))
    ref, rsr = sf.read(os.path.join(WORK, "voices", voice + ".wav"), dtype="float32")
    budget = len(ref) / rsr
    rw, rt = ref_words(voice), meta[voice]["text"].split()
    assert len(rw) == len(rt), (len(rw), len(rt))
    out, prev = [], None
    for t in texts:
        if prev is None:
            a, sr = speak(t, voice, seed)
        else:
            pa, psr = prev[1], prev[2]
            assert psr == rsr, (psr, rsr)
            room = budget - len(pa) / psr
            # хвост референса с начала слова, чтобы вместе с прошлым куском уложиться в длину референса
            k = next((i for i, w in enumerate(rw) if budget - w["start"] <= room), None)
            if k is None or room < 1.5:
                audio, words = pa, prev[0]
            else:
                cut = max(rw[k]["start"] - 0.05, rw[k - 1]["end"] if k else 0.0)
                audio = np.concatenate([ref[int(cut * rsr):], pa])
                words = " ".join(rt[k:]) + " " + prev[0]
            name = f"chain-{voice}-{seed}"
            add_voice(name, audio, rsr, words)
            # модель изредка ставит конец речи сразу, чаще на длинном промпте: 0.16 с вместо 4
            for attempt in range(4):
                a, sr = speak(t, name, seed + 1000 * attempt)
                if len(a) / sr >= 0.4 * len(t) / 14:
                    break
                RETRIES.append((voice, seed, t, round(len(a) / sr, 2)))
        out.append((t, a, sr))
        prev = (t, a, sr)
    return out


def synth():
    p = serve()
    try:
        for voice in VOICES:
            for seed in SEEDS:
                for k, par in enumerate(PARAGRAPHS):
                    for v in VARIANTS:
                        path = os.path.join(WORK, "synth", f"{voice}_{v}_s{seed}_p{k}")
                        if os.path.exists(path + ".wav"):
                            continue
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        text = semi(par) if v.endswith("semi") else par
                        t0 = time.time()
                        RETRIES.clear()
                        if v in ("whole", "semi"):
                            a, sr = speak(text, voice, seed)
                            cuts = []
                        else:
                            ps = pieces(text, voice, seed)
                            if v.startswith("chain"):
                                ps = chain(text, voice, seed, [t for t, _, _ in ps])
                            sr = ps[0][2]
                            a = np.concatenate([x for _, x, _ in ps])
                            cuts = [t for t, _, _ in ps]
                        sf.write(path + ".wav", a, sr)
                        json.dump({"text": text, "paragraphs": [par], "pieces": cuts, "retries": RETRIES[:]},
                                  open(path + ".json", "w"), ensure_ascii=False)
                        print(os.path.basename(path), f"{len(a) / sr:.1f} с за {time.time() - t0:.0f} с", flush=True)
    finally:
        p.terminate()
        p.wait()


# ------------------------------------------------------------------- замеры

def transcribe(path):
    with open(path, "rb") as f:
        r = requests.post(URL + "/v1/audio/transcriptions", proxies=NO_PROXY, timeout=1800,
                          files={"file": (os.path.basename(path), f.read(), "audio/wav")},
                          data={"language": "ru", "response_format": "verbose_json",
                                "timestamp_granularities[]": "word", "timestamp_granularities": "word"})
    r.raise_for_status()
    return r.json()


def pitch(path):
    import librosa
    a, sr = sf.read(path, dtype="float32")
    a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(a, fmin=60, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    st = 12 * np.log2(np.where(voiced, f0, np.nan) / 100.0)
    # ошибки трекера на октаву: речь не уходит от своей медианы дальше чем на 9 полутонов
    st[np.abs(st - np.nanmedian(st)) > 9] = np.nan
    return st, 0.01  # полутона от 100 Гц, шаг 10 мс


def spans(text, asr_words):
    """(start, end) of every sentence of `text` in the recording, by matching words."""
    ss = sentences(text)
    ref, owner = [], []
    for i, s in enumerate(ss):
        w = words(s)
        ref += w
        owner += [i] * len(w)
    hyp = [re.sub(r"[^а-яёa-z0-9-]", "", w["word"].lower().replace("ё", "е")) for w in asr_words]
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    at = {}
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            at[blk.a + j] = blk.b + j
    out = []
    for i in range(len(ss)):
        idx = [k for k in range(len(ref)) if owner[k] == i and k in at]
        n = sum(1 for k in owner if k == i)
        # края предложения должны совпасть: иначе граница неизвестна
        first_ok = idx and idx[0] == owner.index(i)
        last_ok = idx and idx[-1] == owner.index(i) + n - 1
        out.append((asr_words[at[idx[0]]]["start"] if first_ok else None,
                    asr_words[at[idx[-1]]]["end"] if last_ok else None))
    return ss, out


def stat(st, hop, t0, t1):
    seg = st[int(t0 / hop):int(t1 / hop)]
    seg = seg[~np.isnan(seg)]
    return float(np.median(seg)) if len(seg) >= 5 else None


def measure_one(path):
    meta = json.load(open(path[:-4] + ".json"))
    tr = transcribe(path)
    asr = [w for s in tr.get("segments", []) for w in s.get("words", [])] or tr.get("words", [])
    st, hop = pitch(path)
    ss, sp = spans(" ".join(meta["paragraphs"]), asr)
    win = 0.4
    sents = []
    for s, (a, b) in zip(ss, sp):
        sents.append({"text": s, "start": a, "end": b,
                      "onset": stat(st, hop, a, a + win) if a is not None else None,
                      "coda": stat(st, hop, b - win, b) if b is not None else None,
                      "peak": (float(np.nanpercentile(st[int(a / hop):int(b / hop)], 90))
                               if a is not None and b is not None and np.any(~np.isnan(st[int(a / hop):int(b / hop)])) else None)})
    # границы между проходами синтеза: конец куска совпадает с концом предложения
    piece_ends = set()
    acc = ""
    for t in meta.get("pieces", []):
        acc = (acc + " " + t).strip()
        piece_ends.add(len(words(acc)))
    bounds, n = [], 0
    for i in range(len(sents) - 1):
        n += len(words(sents[i]["text"]))
        x, y = sents[i], sents[i + 1]
        if None in (x["coda"], y["onset"], x["end"], y["start"]):
            continue
        bounds.append({"reset": y["onset"] - x["coda"], "gap": y["start"] - x["end"],
                       "between_passes": n in piece_ends})
    voiced = st[~np.isnan(st)]
    return {"cer": cer(meta["text"], tr["text"]), "transcript": tr["text"], "sentences": sents, "bounds": bounds,
            "range": float(np.percentile(voiced, 90) - np.percentile(voiced, 10)),
            "seconds": len(st) * hop}


def measure():
    p = serve()
    try:
        files = sorted(glob.glob(os.path.join(WORK, "synth", "*.wav")) + glob.glob(os.path.join(WORK, "human", "*.wav")))
        for f in files:
            out = f[:-4] + ".measure.json"
            if os.path.exists(out):
                continue
            m = measure_one(f)
            json.dump(m, open(out, "w"), ensure_ascii=False, indent=1)
            print(os.path.basename(f), f"CER {m['cer']:.1%}", len(m["bounds"]), "границ", flush=True)
    finally:
        p.terminate()
        p.wait()


# ------------------------------------------------------------------- сводка

def slope(sents):
    ys = [s["onset"] for s in sents if s["onset"] is not None]
    if len(ys) < 3:
        return None
    return float(np.polyfit(np.arange(len(ys)), ys, 1)[0])


def summarize(ms):
    b = [x for m in ms for x in m["bounds"]]
    rng = np.mean([m["range"] for m in ms])
    r = np.array([x["reset"] for x in b])
    g = np.array([x["gap"] for x in b])
    sl = [slope(m["sentences"]) for m in ms]
    sl = [x for x in sl if x is not None]
    falls = [s["onset"] - s["coda"] for m in ms for s in m["sentences"] if s["onset"] is not None and s["coda"] is not None]
    bp = [x["reset"] for x in b if x["between_passes"]]
    wp = [x["reset"] for x in b if not x["between_passes"]]
    return {"n": len(b), "reset": float(np.mean(r)), "reset_sd": float(np.std(r)), "reset_rel": float(np.mean(r) / rng),
            "gap": float(np.median(g)) if len(g) else None, "fall": float(np.mean(falls)), "slope": float(np.mean(sl)) if sl else None,
            "range": float(rng), "cer": float(np.mean([m["cer"] for m in ms])),
            "reset_between": float(np.mean(bp)) if bp else None, "reset_within": float(np.mean(wp)) if wp else None,
            "resets": r.tolist()}


def report():
    rows = {}
    for f in sorted(glob.glob(os.path.join(WORK, "human", "*.measure.json"))):
        rows.setdefault(("человек", "RuLS"), []).append(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(WORK, "synth", "*.measure.json"))):
        voice, v = re.match(r"([^_]+)_(.+)_s\d+_p\d+", os.path.basename(f)).groups()
        rows.setdefault((voice, v), []).append(json.load(open(f)))
    out = {}
    print(f"{'голос':<11}{'вариант':<12}{'гран':>5}{'сброс':>7}{'±':>6}{'/диап':>7}{'пад.':>6}{'наклон':>8}"
          f"{'пауза':>7}{'меж':>6}{'внутр':>7}{'CER':>7}")
    for (voice, v), ms in rows.items():
        s = summarize(ms)
        out[f"{voice}/{v}"] = s
        f = lambda x, p=1: "—" if x is None else f"{x:.{p}f}"
        print(f"{voice:<11}{v:<12}{s['n']:>5}{f(s['reset']):>7}{f(s['reset_sd']):>6}{f(s['reset_rel'], 2):>7}"
              f"{f(s['fall']):>6}{f(s['slope'], 2):>8}{f(s['gap'], 2):>7}{f(s['reset_between']):>6}"
              f"{f(s['reset_within']):>7}{s['cer']:>7.1%}")
    json.dump(out, open(os.path.join(WORK, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"human": human, "synth": synth, "measure": measure, "report": report}[sys.argv[1]]()
