"""Training data for one reader: paragraphs of «Дворянское гнездо» read by tovarisch.

The question of experiments/29 is whether a model that has heard an hour of one
reader carries a paragraph the way he does, instead of restarting the melody at
every sentence. So the examples are paragraphs, 20–60 s, cut at the author's
paragraph ends where possible, and not sentences: a model trained on sentences
learns sentences again.

Source: LibriVox, dvoryanskoegnezdo_1906_librivox (public domain), the same
recording the turgenev voice was cut from, and the text of the novel from
az.lib.ru. Whisper alone does not do: on half of the chapters it slid into
lowercase without punctuation, and punctuation is exactly what the model reads
pauses from. So the text is the book's, and Whisper's word timestamps are laid
onto it. A piece is kept only if Whisper heard at least MIN_MATCH of its words
exactly as printed — the round trip of ADR 0001 against the true text, which
also catches the places where the reader departed from the edition.

Chapters 23–25 of the recording are held out: the model never hears them, and
the reader's own reading of their paragraphs is what the synthesis is compared
with. Chapter 22 is validation.

The layout is that of experiments/23: the reference is another stretch of the
same reader (8–11 s, whole sentences of the same chapter) with its text, the
target carries stress marks, exactly as the server marks text before synthesis —
every word, by RUAccent.

    python scripts/speaker_ft_data.py book      # текст романа: главы и абзацы
    python scripts/speaker_ft_data.py transcribe [NN ...]   # Whisper по кускам главы
    python scripts/speaker_ft_data.py segment   # абзацы и образцы по расшифровке
    python scripts/speaker_ft_data.py verify    # каждый кусок ещё раз через Whisper
    python scripts/speaker_ft_data.py mark      # знаки ударения сервером voicy
    train-venv/Scripts/python scripts/speaker_ft_data.py encode   # коды кодека, jsonl

Input: work/audio/wav24/*_NN.wav (chapters at 24 kHz mono), work/text/azlib.html.
Output: work/ds/{train,val,test}.jsonl and 24 kHz wavs of every reference and
of the held-out paragraphs.
"""
import difflib, glob, html, json, os, random, re, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
AUDIO = os.path.join(WORK, "audio")
TEXT = os.path.join(WORK, "text")
DS = os.path.join(WORK, "ds")
SEGMENTS = os.path.join(DS, "segments.json")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots")
URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080")
BOOK_URL = "http://az.lib.ru/t/turgenew_i_s/text_0020.shtml"

TEST_CH = {"23", "24", "25"}
VAL_CH = {"22"}
TRAIN_H = float(os.environ.get("TRAIN_H", 1.0))  # часов в обучение
MIN_S, MAX_S = 20.0, 60.0  # длина цели
REF_MIN, REF_MAX = 8.0, 11.0  # образец: длиннее 11 с модель срывается в пустой звук (experiments/26)
GAP = 0.45  # пауза после предложения, по которой можно резать
MIN_MATCH = 0.95  # доля слов, которые Whisper расслышал как напечатано
MAX_CER = 0.05  # повторное распознавание вырезанного куска против книги
SR = 24000

SENT_END = re.compile(r"(?<=[.!?…])[»\")\]]*\s+(?=[—–-]?\s*[А-ЯЁA-Z«\"(])")


def norm(w):
    return re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))


# ----------------------------------------------------------------------- книга

def book():
    """Chapters and paragraphs of the novel: <dd> is a paragraph, a centred numeral a chapter."""
    path = os.path.join(TEXT, "azlib.html")
    if not os.path.exists(path):
        os.makedirs(TEXT, exist_ok=True)
        urllib.request.urlretrieve(BOOK_URL, path)
    t = open(path, "rb").read().decode("cp1251", errors="replace")
    t = t[t.find("Весенний, светлый") - 200:]
    t = t[:t.find("Комментарии:")]  # подвал страницы
    t = re.sub(r'<div align="center"\s*>\s*<p\s*>\s*([IVXL]+)\s*</p>\s*</div>', r"\n@@\n", t)
    t = re.sub(r"<dd>", "\n@\n", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("--", "—").replace("\xa0", " ")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n@@?\n", t)]
    paragraphs = [p for p in paragraphs if re.search(r"[а-яА-Я]", p)]
    json.dump(paragraphs, open(os.path.join(TEXT, "book.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(paragraphs)} абзацев, {sum(len(p.split()) for p in paragraphs)} слов")


def book_words():
    """Every word of the book: (normalised, paragraph, sentence, text of the sentence)."""
    out, sents = [], []
    for pi, p in enumerate(json.load(open(os.path.join(TEXT, "book.json"), encoding="utf-8"))):
        for s in SENT_END.split(p):
            si = len(sents)
            sents.append({"text": s.strip(), "par": pi})
            out += [(n, si) for n in (norm(w) for w in s.split()) if n]
    return out, sents


# ----------------------------------------------------------------- расшифровка

def hear(x):
    import io
    buf = io.BytesIO(); sf.write(buf, x, SR, format="WAV")
    sep, nl = "voicy29", "\r\n"
    field = lambda name, value: f'--{sep}{nl}Content-Disposition: form-data; name="{name}"{nl}{nl}{value}{nl}'.encode()
    body = b"".join([field("language", "ru"), field("response_format", "verbose_json"),
                     field("timestamp_granularities", "word"),
                     f'--{sep}{nl}Content-Disposition: form-data; name="file"; filename="x.wav"{nl}'
                     f"Content-Type: audio/wav{nl}{nl}".encode(), buf.getvalue(), f"{nl}--{sep}--{nl}".encode()])
    req = urllib.request.Request(f"{URL}/v1/audio/transcriptions", body,
                                 {"Content-Type": f"multipart/form-data; boundary={sep}"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def pieces(quiet, hop, lo=20.0, hi=40.0):
    """Cuts at the middle of the longest pause between lo and hi seconds from the last cut."""
    out, start, n = [], 0.0, len(quiet) * hop
    while n - start > hi:
        a, b = int((start + lo) / hop), int((start + hi) / hop)
        run, best, at = 0, 0, start + hi
        for f in range(a, b):
            run = run + 1 if quiet[f] else 0
            if run > best:
                best, at = run, (f - run / 2 + 0.5) * hop
        out.append((start, at)); start = at
    return out + [(start, n)]


def transcribe(*chapters):
    """Whisper on 20–40 s pieces cut at the reader's pauses, each heard on its own.

    A whole chapter in one request drifts: on two thirds of them Whisper slid into
    lowercase without punctuation, and its word times stuck together — 4 pauses
    over 0.3 s per chapter against about 130 in the chapters that came out right.
    A piece is too short to drift, and its times are anchored at its own start.
    """
    os.makedirs(os.path.join(AUDIO, "asr"), exist_ok=True)
    for wav in sorted(glob.glob(os.path.join(AUDIO, "wav24", "*.wav"))):
        ch = wav[-6:-4]
        out = os.path.join(AUDIO, "asr", f"{ch}.json")
        if chapters and ch not in chapters or not chapters and os.path.exists(out):
            continue
        quiet, hop = silence(ch)
        x = cut(ch, 0, None)
        words, texts = [], []
        for a, b in pieces(quiet, hop):
            tr = hear(x[int(a * SR):int(b * SR)])
            words += [{"word": w["word"], "start": round(w["start"] + a, 2), "end": round(w["end"] + a, 2)}
                      for s in tr["segments"] for w in s.get("words", [])]
            texts.append(tr["text"].strip())
        json.dump({"text": " ".join(texts), "duration": len(x) / SR, "segments": [{"words": words}]},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False)
        gaps = sum(1 for i in range(len(words) - 1) if words[i + 1]["start"] - words[i]["end"] > 0.3)
        print(f"  глава {ch}: {len(words)} слов, пауз > 0.3 с: {gaps}", flush=True)


# --------------------------------------------------------------------- нарезка

def asr_words(ch):
    d = json.load(open(os.path.join(AUDIO, "asr", f"{ch}.json"), encoding="utf-8"))
    ws = d.get("words") or [w for s in d["segments"] for w in s.get("words", [])]
    return [{"n": norm(w["word"]), "s": w["start"], "e": w["end"]} for w in ws if norm(w["word"])]


def align(ch, bw, sents):
    """Sentences of the book found in the chapter's recording, with times and how well they match."""
    aw = asr_words(ch)
    book_n = [n for n, _ in bw]
    # где в книге эта глава записи: первое место, где совпали восемь слов подряд
    joined = " ".join(book_n)
    start = None
    for k in range(20, len(aw) - 8):
        at = joined.find(" ".join(w["n"] for w in aw[k:k + 8]))
        if at >= 0:
            start = joined[:at].count(" ") - k
            break
    if start is None:
        return []
    lo, hi = max(0, start - 50), min(len(bw), start + int(len(aw) * 1.2) + 50)
    sm = difflib.SequenceMatcher(None, book_n[lo:hi], [w["n"] for w in aw], autojunk=False)
    hit = {}
    for b in sm.get_matching_blocks():
        for j in range(b.size):
            hit[lo + b.a + j] = aw[b.b + j]
    out = []
    for si in sorted({bw[i][1] for i in range(lo, hi)}):
        idx = [i for i in range(lo, hi) if bw[i][1] == si]
        got = [i for i in idx if i in hit]
        if not got or got[0] != idx[0] or got[-1] != idx[-1]:
            continue  # края предложения не нашлись — его границы неизвестны
        out.append({**sents[si], "s": hit[idx[0]]["s"], "e": hit[idx[-1]]["e"],
                    "match": len(got) / len(idx), "n": len(idx), "next": si + 1})
    # пауза после предложения — по звуку: метки Whisper тянут слово до начала следующего,
    # и в трети стыков промежуток между ними ровно ноль
    quiet, hop = silence(ch)
    starts = {x["next"] - 1: x["s"] for x in out}
    for x in out:
        nxt = starts.get(x["next"], x["e"] + 1.0)
        # Whisper тянет конец слова в тишину, но не дальше чем на 0.3 с
        a, b = int((x["e"] - 0.3) / hop), int((min(nxt, x["e"] + 1.5) + 0.3) / hop)
        run, best, at = 0, 0, x["e"]
        for f in range(max(a, 0), min(b, len(quiet))):
            run = run + 1 if quiet[f] else 0
            if run > best:
                best, at = run, (f - run / 2 + 0.5) * hop
        x["gap"] = best * hop if x["next"] in starts else 0.0
        if best * hop >= 0.12:
            x["e"] = min(x["e"], at)  # конец предложения — не позже середины паузы
        x["par_end"] = x["next"] >= len(sents) or sents[x["next"]]["par"] != x["par"]
    return out


def silence(ch, hop=0.01):
    """Frames quieter than the chapter's noise floor plus a third of its speech range."""
    x = cut(ch, 0, None)
    n = int(hop * SR)
    rms = np.sqrt(np.mean(x[:len(x) // n * n].reshape(-1, n) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    floor, speech = np.percentile(db, 5), np.percentile(db, 95)
    return db < floor + (speech - floor) / 3, hop


def good(blk):
    words = sum(x["n"] for x in blk)
    return (sum(x["match"] * x["n"] for x in blk) / words >= MIN_MATCH
            and all(blk[k]["next"] == blk[k + 1]["next"] - 1 for k in range(len(blk) - 1))
            and not re.search(r"[0-9a-zA-Z]", " ".join(x["text"] for x in blk)))


def paragraphs(ss):
    """Greedy: from each start, the best cut within 20–60 s — the author's paragraph end
    if there is one, else the reader's longest pause."""
    out, i = [], 0
    while i < len(ss):
        best, j = None, i
        while j < len(ss) and ss[j]["e"] - ss[i]["s"] <= MAX_S:
            if ss[j]["e"] - ss[i]["s"] >= MIN_S and ss[j]["gap"] >= GAP:
                if best is None or (ss[j]["par_end"], ss[j]["gap"]) >= (ss[best]["par_end"], ss[best]["gap"]):
                    best = j
            j += 1
        if best is None or not good(ss[i:best + 1]):
            # окна нет или в нём брак — начинаем со следующей паузы
            i += 1
            while i < len(ss) and ss[i - 1]["gap"] < GAP:
                i += 1
            continue
        out.append(ss[i:best + 1])
        i = best + 1
    return out


def references(ss):
    """Runs of whole sentences of 8–11 s with a pause on both sides."""
    out = []
    for i in range(len(ss)):
        if i and ss[i - 1]["gap"] < GAP:
            continue
        for j in range(i, len(ss)):
            d = ss[j]["e"] - ss[i]["s"]
            if d > REF_MAX:
                break
            if d >= REF_MIN and ss[j]["gap"] >= GAP and good(ss[i:j + 1]) and min(x["match"] for x in ss[i:j + 1]) == 1:
                out.append(ss[i:j + 1])
                break
    return out


def span(blk, prev_gap):
    """Times with the edges of the pauses around: a breath in, not a cut word.

    The end is already the middle of the pause after the last sentence; half the
    pause on from there is the onset of the next word, so the tail stops short of it.
    """
    return (max(0.0, blk[0]["s"] - min(0.15, prev_gap / 2)),
            blk[-1]["e"] + min(0.2, max(0.0, blk[-1]["gap"] / 2 - 0.1)))


def segment():
    rng = random.Random(29)
    bw, sents = book_words()
    items, refs = [], []
    for path in sorted(glob.glob(os.path.join(AUDIO, "asr", "*.json"))):
        ch = os.path.basename(path)[:-5]
        ss = align(ch, bw, sents)
        split = "test" if ch in TEST_CH else "val" if ch in VAL_CH else "train"
        prev = {id(x): (ss[k - 1]["gap"] if k else 1.0) for k, x in enumerate(ss)}
        mine = [{"chapter": ch, "start": round(a, 2), "end": round(b, 2), "text": " ".join(x["text"] for x in r)}
                for r in references(ss) for a, b in [span(r, prev[id(r[0])])]]
        if split == "train":
            refs += mine
        n = 0
        for blk in paragraphs(ss):
            a, b = span(blk, prev[id(blk[0])])
            items.append({"id": f"{ch}_{n:03d}", "chapter": ch, "split": split, "start": round(a, 2),
                          "end": round(b, 2), "text": " ".join(x["text"] for x in blk),
                          "sentences": len(blk), "par_end": blk[-1]["par_end"],
                          "match": round(sum(x["match"] * x["n"] for x in blk) / sum(x["n"] for x in blk), 3)})
            n += 1
        print(f"  глава {ch}: {len(ss)} предложений найдено, {len(mine)} образцов, {n} абзацев")
    # образец — другое место того же чтеца: своя глава, если в ней есть, иначе любая обучающая
    for it in items:
        own = [r for r in refs if r["chapter"] == it["chapter"] and (r["end"] <= it["start"] or r["start"] >= it["end"])]
        it["ref"] = rng.choice(own or [r for r in refs if r["chapter"] != it["chapter"]])
    print(f"образцов всего: {len(refs)}")
    # в обучение — TRAIN_H часов из train-глав, вперемешку по всей книге, с запасом на брак verify
    train = [it for it in items if it["split"] == "train"]
    rng.shuffle(train)
    total, keep = 0.0, set()
    for it in train:
        if total >= TRAIN_H * 3600 * 1.2:
            break
        keep.add(it["id"]); total += it["end"] - it["start"]
    items = [it for it in items if it["split"] != "train" or it["id"] in keep]
    os.makedirs(DS, exist_ok=True)
    json.dump(items, open(SEGMENTS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    report(items)


def report(items):
    for split in ("train", "val", "test"):
        xs = [it["end"] - it["start"] for it in items if it["split"] == split]
        if xs:
            print(f"{split}: {len(xs)} абзацев, {sum(xs) / 3600:.2f} ч, длина {min(xs):.0f}–{max(xs):.0f} с, "
                  f"медиана {np.median(xs):.0f} с, кончаются абзацем автора "
                  f"{np.mean([it['par_end'] for it in items if it['split'] == split]):.0%}")


def verify():
    """Each cut heard once more on its own, against the book: are text and sound the same?

    The times come from Whisper over a whole piece, and where they are off, a
    paragraph starts on a cut word or ends without its last one. Then its first or
    last words are missing from what is heard, and it goes.
    """
    items = json.load(open(SEGMENTS, encoding="utf-8"))
    keep, dropped = [], []
    for it in items:
        if "cer" not in it:
            heard = hear(cut(it["chapter"], it["start"], it["end"]))["text"]
            a, b = [norm(w) for w in it["text"].split() if norm(w)], [norm(w) for w in heard.split() if norm(w)]
            ca, cb = "".join(a), "".join(b)
            sm = difflib.SequenceMatcher(None, ca, cb, autojunk=False)
            it["cer"] = round(sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
                              / max(1, len(ca)), 4)
            # первое и последнее слово — те же, с точностью до ослышки («супу» — «супа»)
            near = lambda x, y: difflib.SequenceMatcher(None, x, y).ratio() >= 0.7
            it["edges"] = bool(b) and near(a[0], b[0]) and near(a[-1], b[-1])
        (keep if it["cer"] <= MAX_CER and it["edges"] else dropped).append(it)
    json.dump(keep, open(SEGMENTS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(dropped, open(os.path.join(DS, "dropped.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"в брак: {len(dropped)} (CER > {MAX_CER:.0%}: {sum(it['cer'] > MAX_CER for it in dropped)}, "
          f"края: {sum(not it['edges'] for it in dropped)}), оставлено {len(keep)}")
    report(keep)


def mark():
    """Stress marks the way the server puts them before synthesis: RUAccent, no dictionary."""
    items = json.load(open(SEGMENTS, encoding="utf-8"))
    for it in items:
        if "marked" in it:
            continue
        req = urllib.request.Request(f"{URL}/v1/text/prepare",
                                     json.dumps({"text": it["text"], "dictionary": False}).encode(),
                                     {"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req).read())
        if not out.get("stress"):
            sys.exit("сервер не ставит знаки: нужна модель с stress.json")
        it["marked"] = out["text"]
    json.dump(items, open(SEGMENTS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"размечено {len(items)} абзацев, знаков {sum(it['marked'].count(chr(0x301)) for it in items)}")


def cut(ch, a, b, cache={}):
    if ch not in cache:
        cache.clear()
        cache[ch] = sf.read(glob.glob(os.path.join(AUDIO, "wav24", f"*_{ch}.wav"))[0], dtype="float32")[0]
    return cache[ch][int(a * SR):None if b is None else int(b * SR)]


def encode():
    import torch
    from qwen_tts import Qwen3TTSTokenizer
    items = json.load(open(SEGMENTS, encoding="utf-8"))
    tok = Qwen3TTSTokenizer.from_pretrained(os.path.join(SNAP, os.listdir(SNAP)[0], "speech_tokenizer"),
                                            device_map="cuda:0")
    os.makedirs(os.path.join(DS, "wav"), exist_ok=True)

    def codes(x):
        with torch.no_grad():
            return tok.encode([x], sr=SR).audio_codes[0].cpu().numpy().astype(np.int16).tolist()

    out = {s: open(os.path.join(DS, f"{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "val", "test")}
    ref_codes = {}
    for k, it in enumerate(sorted(items, key=lambda it: it["chapter"])):
        r = it["ref"]
        key = f"{r['chapter']}_{r['start']:.2f}"
        ref_wav = os.path.join(DS, "wav", f"ref_{key}.wav")
        if key not in ref_codes:
            x = cut(r["chapter"], r["start"], r["end"])
            sf.write(ref_wav, x, SR)
            ref_codes[key] = codes(x)
        x = cut(it["chapter"], it["start"], it["end"])
        row = {"id": it["id"], "text": it["marked"], "plain": it["text"], "codes": codes(x),
               "ref_text": r["text"], "ref_codes": ref_codes[key], "ref_audio": ref_wav,
               "chapter": it["chapter"], "duration": round(it["end"] - it["start"], 2)}
        if it["split"] == "test":
            # оригинал отложенных абзацев — эталон, с которым сравнивается синтез
            row["audio"] = os.path.join(DS, "wav", f"orig_{it['id']}.wav")
            sf.write(row["audio"], x, SR)
        out[it["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
        if k % 20 == 0:
            print(f"  {k}/{len(items)}", flush=True)
    for f in out.values():
        f.close()
    print("готово:", DS)


if __name__ == "__main__":
    {"book": book, "transcribe": transcribe, "segment": segment, "verify": verify, "mark": mark, "encode": encode}[sys.argv[1]](*sys.argv[2:])
