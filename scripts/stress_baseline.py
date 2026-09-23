"""How many words does Qwen3-TTS stress wrongly in running text?

The question decides whether fine-tuning is worth it at all: if the pronunciation
dictionary already leaves almost nothing, training buys nothing.

The detector is the template matcher from ADR 0016, moved from isolated words
into sentences: Whisper gives word boundaries, each word is cut out and its
loudness contour compared with Silero readings of the same sentence in which
that word is stressed on each vowel in turn. Expected stress comes from RUAccent.

Templates are rendered in context on purpose: with isolated-word templates the
first run raised 45% false alarms on correctly stressed Silero speech, which is
as much as it found in Qwen and therefore says nothing about Qwen.

That matcher was validated only on isolated words, so it is validated again here
before its Qwen numbers mean anything. Silero reads every sentence twice, with
the stress marks RUAccent placed and with every mark deliberately moved to
another vowel. On the first reading every «error» is a false alarm; on the
second every word is wrong and the detector should say so. A Qwen error rate is
only informative where it clearly exceeds the false-alarm rate.

Talks to a running voicy server (VOICY_URL) for synthesis and word timestamps;
Silero and RUAccent run locally on the CPU.
"""
import io, json, os, random, re, sys, warnings
from difflib import SequenceMatcher
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, requests

sys.path.insert(0, os.path.dirname(__file__))
from stress_match import contour, SR as SILERO_SR

URL = os.environ.get("VOICY_URL", "http://localhost:8080").rstrip("/")
WORK = os.environ.get("STRESS_WORK", "experiments/23-stress-finetune/work")
V = "аеёиоуыэюя"
SEEDS = (1, 2, 3)
# эталоны — три голоса Silero, проверка — четвёртый (kseniya); xenia слишком
# похожа на kseniya, чтобы честно быть по другую сторону
TEMPLATE_VOICES = ("eugene", "aidar", "baya")
NO_PROXY = {"http": None, "https": None}

# обычная речь и омографы: жалоба на ударения не только про термины
EVERYDAY = [
    "Этот вопрос стоит разобрать подробно.",
    "Большая часть проверок выполняется при сборке.",
    "Мы договорились созвониться в среду вечером.",
    "Документы нужно подписать до конца квартала.",
    "Он включил свет и начал звонить поставщикам.",
    "Новые правила вступают в силу со следующего месяца.",
    "Красивее всего город выглядит ранним утром.",
    "Средства со счёта списали без предупреждения.",
    "Каталог товаров обновляется каждую неделю.",
    "На договоре не хватает подписи директора.",
    "Эксперты обеспечили проекту хорошую репутацию.",
    "Задача оказалась значительно сложнее, чем казалось.",
    "Цены на жильё выросли почти на треть.",
    "Мастер быстро починил замок на двери.",
    "Старинный замок стоит на высоком холме.",
    "Ученики записали домашнее задание в дневник.",
    "Сливовый торт лучше испечь заранее.",
    "Облегчить работу поможет простой шаблон.",
    "Врач посоветовал принимать лекарство дважды в день.",
    "Банк предложил выгодные условия по кредиту.",
    "Жалюзи в офисе давно пора поменять.",
    "Мы углубили анализ и уточнили выводы.",
    "Приговор огласили в пятницу после обеда.",
    "Мусоропровод в доме не работает с весны.",
]


def eligible(word):
    return sum(ch in V for ch in word.lower()) >= 2


def stress_index(marked):
    """Vowel index after the «+» RUAccent places; ё is stressed by itself."""
    idx, n, hit = None, 0, False
    for ch in marked.lower():
        if ch == "+":
            hit = True; continue
        if ch in V:
            if hit or (ch == "ё" and idx is None):
                idx = n
            hit = False; n += 1
    return idx


def place(word, k):
    plain = word.replace("+", "")
    vp = [i for i, ch in enumerate(plain.lower()) if ch in V]
    return plain[:vp[k]] + "+" + plain[vp[k]:]


def norm(w):
    return re.sub(r"[^а-яa-z]", "", w.lower().replace("ё", "е").replace("+", ""))


def load_ruaccent():
    from onnxruntime import InferenceSession as _I
    _o = _I.run

    def _r(self, n, f, *a, **k):  # то же обхождение, что в autolex.py
        mm = {i.name for i in self.get_inputs()} - set(f)
        if mm and "input_ids" in f:
            f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in mm}}
        return _o(self, n, f, *a, **k)

    _I.run = _r
    from ruaccent import RUAccent
    acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    return acc


def words_of(marked_sentence):
    """(surface form with «+», expected vowel index) for every word."""
    out = []
    for w in re.findall(r"[А-Яа-яЁё+\-]+", marked_sentence):
        w = w.strip("-")
        if norm(w):
            out.append((w, stress_index(w)))
    return out


def transcribe_words(x, sr):
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV"); buf.seek(0)
    r = requests.post(f"{URL}/v1/audio/transcriptions", proxies=NO_PROXY, timeout=600,
                      files={"file": ("a.wav", buf, "audio/wav")},
                      data={"language": "ru", "response_format": "verbose_json",
                            "timestamp_granularities[]": "word"})
    r.raise_for_status()
    return [w for s in r.json()["segments"] for w in s.get("words", [])]


def qwen(text, seed):
    r = requests.post(f"{URL}/v1/audio/speech", proxies=NO_PROXY, timeout=1800,
                      json={"input": text, "voice": "turgenev", "response_format": "wav", "seed": seed})
    r.raise_for_status()
    x, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return x, sr


def prepared(text):
    r = requests.post(f"{URL}/v1/text/prepare", proxies=NO_PROXY, timeout=60, json={"text": text})
    r.raise_for_status()
    return r.json().get("text") or text


def cut(x, sr, expected):
    """Loudness contour of every checkable word that Whisper heard as itself."""
    heard = transcribe_words(x, sr)
    a = [norm(w) for w, _ in expected]
    b = [norm(w["word"]) for w in heard]
    out = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            w, exp = expected[i]
            if exp is None or not eligible(w):
                continue
            s, e = heard[j]["start"], heard[j]["end"]
            c = contour(x[max(0, int((s - 0.02) * sr)):int((e + 0.02) * sr)], sr)
            if c is not None:
                out[i] = c
    return out


def detect(x, sr, expected, templates):
    """Distance of every word's contour to each stress position, averaged over
    the template voices; the decision is left to report(), where a margin can be
    swept against the false-alarm rate."""
    out = {}
    for i, c in cut(x, sr, expected).items():
        per_k = templates.get(i, {})
        if per_k:
            out[i] = {k: float(np.mean([np.mean((c - t) ** 2) for t in ts])) for k, ts in per_k.items()}
    return out


def setup():
    import torch
    torch.set_num_threads(12)
    os.makedirs(WORK, exist_ok=True)
    print("загружаю Silero и RUAccent ...", flush=True)
    silero = torch.package.PackageImporter(os.path.join(WORK, "v5_ru.pt")).load_pickle("tts_models", "model")
    acc = load_ruaccent()

    def silero_say(t, speaker):
        a = silero.apply_tts(text=t, speaker=speaker, sample_rate=SILERO_SR,
                             put_accent=False, put_yo=True)
        return a.numpy().astype(np.float32), SILERO_SR

    def templates_in_context(expected):
        """Эталоны во фразе: вариант k ставит ударение на k-й слог каждого слова
        (короткому слову — на последний), и одно распознавание даёт эталоны всем словам."""
        nvs = [sum(ch in V for ch in w.lower()) for w, _ in expected]
        tpl = {i: {} for i in range(len(expected))}
        for k in range(max(nvs, default=0)):
            words = [place(w, min(k, nv - 1)) if nv >= 1 else w
                     for (w, _), nv in zip(expected, nvs)]
            for voice in TEMPLATE_VOICES:
                for i, c in cut(*silero_say(" ".join(words) + ".", voice), expected).items():
                    if k < nvs[i]:
                        tpl[i].setdefault(k, []).append(c)
        return tpl
    return acc, silero_say, templates_in_context


def main():
    acc, silero_say, templates_in_context = setup()
    samples = json.load(open("data/samples.json", encoding="utf-8"))
    sentences = [("everyday", s) for s in EVERYDAY]
    for smp in samples:
        for s in re.split(r"(?<=[.!?])\s+", smp["audio"]):
            if len(re.findall(r"[А-Яа-яЁё]+", s)) >= 3:
                sentences.append((smp["id"], s))

    rng = random.Random(7)
    rows = []
    for n, (src, raw) in enumerate(sentences, 1):
        text = prepared(raw)
        marked = acc.process_all(text)
        expected = words_of(marked)

        # сдвинутая разметка: у каждого проверяемого слова ударение уводится на другой слог
        shifted, moved = [], {}
        for i, (w, exp) in enumerate(expected):
            nv = sum(ch in V for ch in w.lower())
            if exp is not None and nv >= 2:
                k = rng.choice([v for v in range(nv) if v != exp])
                moved[i] = k
                shifted.append(place(w, k))
            else:
                shifted.append(w)
        shifted_text = " ".join(shifted) + "."

        row = {"src": src, "text": text, "marked": marked,
               "words": [w for w, _ in expected], "expected": [e for _, e in expected],
               "moved": moved}
        tpl = templates_in_context(expected)
        row["silero_true"] = detect(*silero_say(marked, "kseniya"), expected, tpl)
        row["silero_shifted"] = detect(*silero_say(shifted_text, "kseniya"), expected, tpl)
        row["qwen"] = {}
        for sd in (() if os.environ.get("STRESS_NO_QWEN") else SEEDS):
            x, sr = qwen(text, sd)
            sf.write(os.path.join(WORK, f"q{n:03}_{sd}.wav"), x, sr)
            row["qwen"][sd] = detect(x, sr, expected, tpl)
        rows.append(row)
        print(f"[{n:3}/{len(sentences)}] {src:9} {text[:70]}", flush=True)
        json.dump(rows, open(os.path.join(WORK, "rows.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    report(rows)


def verdicts(dists, expected, margin):
    """Wrong stress only when the expected position loses to the best by more than
    `margin`: a lower false-alarm rate bought with missed errors."""
    out = {}
    for i, d in dists.items():
        exp = expected[int(i)]
        d = {int(k): v for k, v in d.items()}
        if exp not in d:
            continue
        best = min(d, key=d.get)
        out[int(i)] = (best, d[exp] - d[best] > margin)
    return out


def rates(rows, margin):
    fa = [bad for r in rows for _, bad in verdicts(r["silero_true"], r["expected"], margin).values()]
    hit = [bad for r in rows for _, bad in verdicts(r["silero_shifted"], r["expected"], margin).values()]
    return np.mean(fa) if fa else 0, np.mean(hit) if hit else 0, len(fa), len(hit)


def report(rows, margin=None):
    print("\nДетектор на Silero kseniya, эталоны — " + ", ".join(TEMPLATE_VOICES) + ":")
    print("  порог   ложные тревоги   замечен сдвиг")
    sweep = {}
    for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        fa, hit, n1, n2 = rates(rows, m)
        sweep[m] = (fa, hit)
        print(f"  {m:4.2f}      {fa:5.0%} из {n1:<4}    {hit:5.0%} из {n2}")
    if margin is None:  # самый низкий порог, при котором ложных тревог не больше 10%
        margin = next((m for m, (fa, _) in sweep.items() if fa <= 0.10), 0.5)
    fa, hit, _, _ = rates(rows, margin)
    print(f"  рабочий порог {margin}: ложные {fa:.0%}, чувствительность {hit:.0%}")

    if not any(r["qwen"] for r in rows):
        return
    per_seed, votes = {}, {}
    for ri, r in enumerate(rows):
        for sd, dists in r["qwen"].items():
            for i, (_, bad) in verdicts(dists, r["expected"], margin).items():
                per_seed.setdefault(sd, []).append(bad)
                votes.setdefault((ri, i), []).append(bad)
    print("\nQwen3-TTS (turgenev), доля слов, где детектор видит ударение не там, где RUAccent:")
    for sd, v in per_seed.items():
        m = np.mean(v)
        p = (m - fa) / max(hit - fa, 1e-6)  # поправка на ложные тревоги и пропуски
        print(f"  seed {sd}: {sum(v)}/{len(v)} = {m:.0%}   → оценка истинной доли ≈ {max(p, 0):.0%}")
    stable = [(k, v) for k, v in votes.items() if len(v) >= 2]
    maj = [k for k, v in stable if sum(v) * 2 > len(v)]
    print(f"  большинством сидов: {len(maj)}/{len(stable)}")
    print("\nСлова, где большинство сидов расходится с RUAccent:")
    for ri, i in maj:
        r = rows[ri]
        print(f"  {r['words'][i]:22} в «{r['text'][:60]}»")


def human(n=80):
    """False alarms on human speech: RuLS test readers, stress taken as RUAccent's.

    Audiobook readers stress normatively almost everywhere, so disagreement with
    RUAccent here is overwhelmingly the detector's error, not the reader's.
    """
    import pyarrow.parquet as pq
    acc, silero_say, templates_in_context = setup()
    t = pq.read_table(os.path.join(WORK, "ruls", "test.parquet")).to_pylist()
    random.Random(3).shuffle(t)
    rows = []
    for k, item in enumerate(t[:n], 1):
        text = item["text_no_preprocessing"]
        x, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
        expected = words_of(acc.process_all(text))
        tpl = templates_in_context(expected)
        rows.append({"src": item["audio_filepath"], "text": text,
                     "words": [w for w, _ in expected], "expected": [e for _, e in expected],
                     "human": detect(x, sr, expected, tpl)})
        print(f"[{k:3}/{n}] {text[:70]}", flush=True)
        json.dump(rows, open(os.path.join(WORK, "human.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    report_human(rows)


def report_human(rows):
    print("\nЖивая речь RuLS: доля слов, где детектор спорит с RUAccent")
    for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        v = [bad for r in rows for _, bad in verdicts(r["human"], r["expected"], m).values()]
        print(f"  порог {m:4.2f}: {np.mean(v):5.0%} из {len(v)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "human":
        human()
    elif len(sys.argv) > 1 and sys.argv[1] == "human-report":
        report_human(json.load(open(os.path.join(WORK, "human.json"), encoding="utf-8")))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open(os.path.join(WORK, "rows.json"), encoding="utf-8")))
    else:
        main()
