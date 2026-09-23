"""Did the LoRA teach Qwen3-TTS to obey stress marks, and what did it cost?

Three conditions on the sentences of stress_baseline.py, same model in memory,
the adapter switched on and off:

  base / plain    — the untouched model, text as voicy sends it today
  lora / plain    — adapted model, same text: ordinary reading must not get worse
  lora / marked   — adapted model, every checkable word marked with RUAccent's stress

Measured for each:
  stress   — share of words the detector hears stressed off RUAccent's position,
             margin 0.3; the detector's own 10% false alarms and 71% sensitivity
             (validated in stress_baseline.py) are corrected for in «≈ true»
  CER      — round trip through Whisper, the project's quality metric (ADR 0001)
  speaker  — cosine of Qwen's own speaker embedding of the output to the voice sample
  >8 kHz   — share of energy above 8 kHz: RuLS is 16 kHz audio, and a model that
             learned to sound muffled would show it here first

Runs without the voicy server: Whisper is loaded locally.
"""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.path.dirname(__file__))
import stress_baseline as sb
from stress_ft_data import ACUTE
from stress_ft_train import BASE, OUT

WORK = sb.WORK
TAG = os.path.basename(OUT)  # у каждого варианта адаптера свои файлы оценки
EVAL = os.path.join(WORK, "eval", TAG)
SEEDS = (1, 2)
MARGIN, FA, SENS = 0.3, 0.10, 0.71
VOICE = "server/voices/turgenev.wav"
VOICE_TEXT = json.load(open("server/voices/voices.json", encoding="utf-8"))["turgenev"]["text"]


def mark_all(accented):
    """Every checkable word marked, the way voicy would send it."""
    def one(m):
        w = m.group(0)
        plain = w.replace("+", "")
        if "+" not in w or "ё" in plain.lower() or sum(ch in sb.V for ch in plain.lower()) < 2:
            return plain
        return re.sub(r"\+(.)", lambda v: v.group(1) + ACUTE, w)
    return re.sub(r"[А-Яа-яЁё+\-]+", one, accented)


def hf_share(x, sr):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spec[f > 8000].sum() / max(spec.sum(), 1e-12))


def cer(ref, hyp):
    a = re.sub(r"[^а-яa-z]", "", ref.lower().replace("ё", "е"))
    b = re.sub(r"[^а-яa-z]", "", hyp.lower().replace("ё", "е"))
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return edits / max(len(a), 1)


def main():
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    os.makedirs(EVAL, exist_ok=True)
    print("адаптер:", OUT, flush=True)

    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    last_text = {}

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        segs = list(segs)
        last_text["t"] = " ".join(s.text for s in segs)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]

    sb.transcribe_words = transcribe_words  # cut() берёт распознавание отсюда
    acc, silero_say, templates_in_context = sb.setup()

    tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(OUT, "adapter.json")))
    inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
                                       target_modules=cfg["target_modules"]), tts.model.talker.model)
    set_peft_model_state_dict(tts.model.talker.model, load_file(os.path.join(OUT, "adapter.safetensors")))
    tts.model.eval()

    def adapter(on):
        for m in tts.model.talker.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(on)

    x_ref, sr_ref = sf.read(VOICE, dtype="float32")
    if sr_ref != 24000:
        import librosa
        x_ref = librosa.resample(x_ref, orig_sr=sr_ref, target_sr=24000)
    spk_ref = tts.model.extract_speaker_embedding(x_ref, 24000).float()

    def speak(text, seed):
        torch.manual_seed(seed)
        w, sr = tts.generate_voice_clone(text=text, language="Russian", ref_audio=VOICE, ref_text=VOICE_TEXT)
        return np.asarray(w[0], dtype=np.float32), sr

    samples = json.load(open("data/samples.json", encoding="utf-8"))
    sentences = [("everyday", s) for s in sb.EVERYDAY]
    for smp in samples:
        for s in re.split(r"(?<=[.!?])\s+", smp["audio"]):
            if len(re.findall(r"[А-Яа-яЁё]+", s)) >= 3:
                sentences.append((smp["id"], s))
    n_max = int(os.environ.get("EVAL_N", len(sentences)))

    conds = {"base/plain": (False, False), "lora/plain": (True, False), "lora/marked": (True, True)}
    rows = []
    for n, (src, text) in enumerate(sentences[:n_max], 1):
        marked = acc.process_all(text)
        expected = sb.words_of(marked)
        tpl = templates_in_context(expected)
        row = {"src": src, "text": text, "words": [w for w, _ in expected],
               "expected": [e for _, e in expected], "c": {}}
        for name, (on, with_marks) in conds.items():
            adapter(on)
            say = mark_all(marked) if with_marks else text
            for sd in SEEDS:
                x, sr = speak(say, sd)
                sf.write(os.path.join(EVAL, f"{n:03}_{name.replace('/', '-')}_{sd}.wav"), x, sr)
                d = sb.detect(x, sr, expected, tpl)
                emb = tts.model.extract_speaker_embedding(
                    x if sr == 24000 else __import__("librosa").resample(x, orig_sr=sr, target_sr=24000), 24000).float()
                row["c"].setdefault(name, []).append({
                    "dists": d, "cer": cer(text, last_text["t"]), "heard": last_text["t"],
                    "spk": float(torch.nn.functional.cosine_similarity(emb, spk_ref, dim=0)),
                    "hf": hf_share(x, sr), "sec": len(x) / sr})
        rows.append(row)
        print(f"[{n:3}/{n_max}] {text[:70]}", flush=True)
        json.dump(rows, open(os.path.join(WORK, f"eval_{TAG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    adapter(True)
    report(rows)


def report(rows):
    print(f"\n{'условие':14} {'ударение':>9} {'≈ истинно':>10} {'CER':>6} {'голос':>6} {'>8 кГц':>7}")
    for name in ("base/plain", "lora/plain", "lora/marked"):
        bad, cers, spk, hf = [], [], [], []
        for r in rows:
            for take in r["c"].get(name, []):
                for _, b in sb.verdicts(take["dists"], r["expected"], MARGIN).values():
                    bad.append(b)
                cers.append(take["cer"]); spk.append(take["spk"]); hf.append(take["hf"])
        if not bad:
            continue
        m = float(np.mean(bad))
        p = max(0.0, (m - FA) / (SENS - FA))
        print(f"{name:14} {m:8.1%} {p:9.0%} {np.mean(cers):6.1%} {np.mean(spk):6.3f} {np.mean(hf):7.2%}"
              f"   ({len(bad)} слов)")

    # парное сравнение: одно и то же слово в одной и той же фразе, среднее по сидам;
    # бутстрэп по фразам, потому что слова одной фразы не независимы
    def per_sentence(name):
        out = []
        for r in rows:
            words = {}
            for take in r["c"].get(name, []):
                for i, (_, b) in sb.verdicts(take["dists"], r["expected"], MARGIN).items():
                    words.setdefault(i, []).append(b)
            out.append({i: np.mean(v) for i, v in words.items()})
        return out

    rng = np.random.default_rng(0)
    base = per_sentence("base/plain")
    for name in ("lora/plain", "lora/marked"):
        other = per_sentence(name)
        diffs = [[o[i] - b[i] for i in b.keys() & o.keys()] for b, o in zip(base, other)]
        diffs = [d for d in diffs if d]
        if not diffs:
            continue
        point = np.mean([x for d in diffs for x in d])
        boot = []
        for _ in range(2000):
            pick = rng.integers(0, len(diffs), len(diffs))
            boot.append(np.mean([x for k in pick for x in diffs[k]]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # в долях «истинных» ошибок: сырой сдвиг делится на чувствительность минус ложные
        k = 1 / (SENS - FA)
        print(f"{name} − base/plain: {point * k:+.1%} истинных ошибок, 95% ДИ [{lo * k:+.1%}, {hi * k:+.1%}]")


def control():
    """Does the mark move the stress at all? One word per sentence, the mark put on
    each of its vowels in turn, other words unmarked.

    Natural text is a weak test: the model already stresses ~88% of words right, so
    a mark mostly repeats what it would do anyway. Here most marks contradict the
    model's habit. The ceiling is the detector's own: 68% exact syllable on Silero
    with known marks (stress_baseline.py), chance is one over the vowel count.
    """
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    os.makedirs(os.path.join(EVAL, "control"), exist_ok=True)

    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]

    sb.transcribe_words = transcribe_words
    acc, silero_say, templates_in_context = sb.setup()
    tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(OUT, "adapter.json")))
    inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
                                       target_modules=cfg["target_modules"]), tts.model.talker.model)
    set_peft_model_state_dict(tts.model.talker.model, load_file(os.path.join(OUT, "adapter.safetensors")))
    tts.model.eval()

    def adapter(on):
        for m in tts.model.talker.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(on)

    rows = []
    for n, text in enumerate(sb.EVERYDAY, 1):
        expected = sb.words_of(acc.process_all(text))
        # самое длинное проверяемое слово: у него больше всего слогов, куда увести знак
        cand = [(sum(ch in sb.V for ch in w.lower()), i) for i, (w, e) in enumerate(expected)
                if e is not None and sb.eligible(w) and "ё" not in w.lower()]
        if not cand:
            continue
        nv, i = max(cand)
        tpl = templates_in_context(expected)
        plain_words = re.findall(r"[А-Яа-яЁё\-]+|[^А-Яа-яЁё\-]+", text)
        word_idx = [j for j, t in enumerate(plain_words) if re.match(r"[А-Яа-яЁё]", t)]
        for k in range(nv):
            w = expected[i][0].replace("+", "")
            vp = [p for p, ch in enumerate(w.lower()) if ch in sb.V]
            marked_w = w[:vp[k] + 1] + ACUTE + w[vp[k] + 1:]
            toks = list(plain_words)
            toks[word_idx[i]] = marked_w
            say = "".join(toks)
            for name, on in (("base", False), ("lora", True)):
                adapter(on)
                torch.manual_seed(1)
                wv, sr = tts.generate_voice_clone(text=say, language="Russian", ref_audio=VOICE, ref_text=VOICE_TEXT)
                x = np.asarray(wv[0], dtype=np.float32)
                sf.write(os.path.join(EVAL, "control", f"{n:02}_{k}_{name}.wav"), x, sr)
                d = sb.detect(x, sr, expected, tpl).get(i)
                got = None if not d else min(d, key=d.get)
                rows.append({"text": say, "word": w, "target": k, "normal": expected[i][1],
                             "nv": nv, "model": name, "detected": got})
        print(f"[{n:2}/{len(sb.EVERYDAY)}] {w}", flush=True)
        json.dump(rows, open(os.path.join(WORK, f"control_{TAG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    adapter(True)
    control_report(rows)


def hard():
    """Words the untouched model gets wrong, re-measured on fresh seeds.

    Selection uses the base/plain takes of the main run (seeds 1, 2): a word is
    «hard» when the detector hears it off RUAccent's stress in both. Measuring
    on the same takes would reward any change by regression to the mean, so both
    the untouched model and the adapter with marks are rendered again with seeds
    3 and 4, and only those takes are counted.
    """
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    main_rows = json.load(open(os.path.join(WORK, f"eval_{TAG}.json"), encoding="utf-8"))
    picks = []
    for ri, r in enumerate(main_rows):
        takes = r["c"]["base/plain"]
        vs = [sb.verdicts(t["dists"], r["expected"], MARGIN) for t in takes]
        bad = [i for i in vs[0] if all(i in v and v[i][1] for v in vs)]
        if bad:
            picks.append((ri, bad))
    print(f"трудных слов: {sum(len(b) for _, b in picks)} в {len(picks)} фразах", flush=True)
    os.makedirs(os.path.join(EVAL, "hard"), exist_ok=True)

    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]

    sb.transcribe_words = transcribe_words
    acc, silero_say, templates_in_context = sb.setup()
    tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(OUT, "adapter.json")))
    inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
                                       target_modules=cfg["target_modules"]), tts.model.talker.model)
    set_peft_model_state_dict(tts.model.talker.model, load_file(os.path.join(OUT, "adapter.safetensors")))
    tts.model.eval()

    def adapter(on):
        for m in tts.model.talker.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(on)

    rows = []
    for n, (ri, bad) in enumerate(picks, 1):
        text = main_rows[ri]["text"]
        marked = acc.process_all(text)
        expected = sb.words_of(marked)
        tpl = templates_in_context(expected)
        row = {"text": text, "hard": bad, "words": [expected[i][0] for i in bad],
               "expected": [e for _, e in expected], "c": {}}
        for name, on, say in (("base/plain", False, text), ("lora/marked", True, mark_all(marked))):
            adapter(on)
            for sd in (3, 4):
                torch.manual_seed(sd)
                wv, sr = tts.generate_voice_clone(text=say, language="Russian", ref_audio=VOICE, ref_text=VOICE_TEXT)
                x = np.asarray(wv[0], dtype=np.float32)
                sf.write(os.path.join(EVAL, "hard", f"{n:02}_{name.replace('/', '-')}_{sd}.wav"), x, sr)
                row["c"].setdefault(name, []).append(sb.detect(x, sr, expected, tpl))
        rows.append(row)
        print(f"[{n:2}/{len(picks)}] {', '.join(row['words'])}", flush=True)
        json.dump(rows, open(os.path.join(WORK, f"hard_{TAG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    adapter(True)
    hard_report(rows)


def hard_report(rows):
    print(f"\n{'условие':12} {'трудные слова: ударение не там':>32}")
    per = {}
    for name in ("base/plain", "lora/marked", "silero"):
        v = []
        for r in rows:
            for d in r["c"].get(name, []):
                vd = sb.verdicts(d, r["expected"], MARGIN)
                v += [vd[i][1] for i in r["hard"] if i in vd]
        per[name] = v
        if not v:
            continue
        m = np.mean(v)
        print(f"{name:12} {m:14.0%} из {len(v)}   ≈ истинно {max(0, (m - FA) / (SENS - FA)):.0%}")
    # бутстрэп по фразам для разницы
    rng = np.random.default_rng(0)
    def sent_means(name):
        out = []
        for r in rows:
            v = []
            for d in r["c"].get(name, []):
                vd = sb.verdicts(d, r["expected"], MARGIN)
                v += [vd[i][1] for i in r["hard"] if i in vd]
            out.append(v)
        return out
    a, b = sent_means("base/plain"), sent_means("lora/marked")
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if pairs:
        boot = []
        for _ in range(2000):
            pick = rng.integers(0, len(pairs), len(pairs))
            xs = [v for k in pick for v in pairs[k][0]]; ys = [v for k in pick for v in pairs[k][1]]
            boot.append(np.mean(ys) - np.mean(xs))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pt = np.mean([v for _, y in pairs for v in y]) - np.mean([v for x, _ in pairs for v in x])
        k = 1 / (SENS - FA)
        print(f"lora/marked − base/plain: {pt * k:+.0%} истинных, 95% ДИ [{lo * k:+.0%}, {hi * k:+.0%}]")


def ceiling():
    """The detector's own ceiling on the control words: Silero, which obeys «+»,
    reads each control sentence with the mark on the target vowel. What share of
    these the detector places on the right syllable bounds what any model can score.
    """
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]

    sb.transcribe_words = transcribe_words
    acc, silero_say, templates_in_context = sb.setup()
    rows = [r for r in json.load(open(os.path.join(WORK, f"control_{TAG}.json"), encoding="utf-8"))
            if r["model"] == "lora"]
    out, tpl_cache = [], {}
    for r in rows:
        plain = r["text"].replace(ACUTE, "")
        if plain not in tpl_cache:
            expected = sb.words_of(acc.process_all(plain))
            tpl_cache[plain] = (expected, templates_in_context(expected))
        expected, tpl = tpl_cache[plain]
        i = next(j for j, (w, _) in enumerate(expected) if w.replace("+", "") == r["word"])
        # знак Silero — «+» перед гласной; остальные слова с ударениями RUAccent
        words = [w for w, _ in expected]
        words[i] = sb.place(r["word"], r["target"])
        x, sr = silero_say(" ".join(words) + ".", "kseniya")
        d = sb.detect(x, sr, expected, tpl).get(i)
        got = None if not d else min(d, key=d.get)
        out.append({**r, "model": "silero", "detected": got})
    rows_all = json.load(open(os.path.join(WORK, f"control_{TAG}.json"), encoding="utf-8"))
    rows_all = [r for r in rows_all if r["model"] != "silero"] + out
    json.dump(rows_all, open(os.path.join(WORK, f"control_{TAG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    control_report(rows_all)

    # трудные слова: Silero с верными ударениями. Если детектор ругается и на него,
    # «трудность» слова — его систематическая ошибка, а не модели
    hp = os.path.join(WORK, f"hard_{TAG}.json")
    if os.path.exists(hp):
        hrows = json.load(open(hp, encoding="utf-8"))
        for r in hrows:
            marked = acc.process_all(r["text"])
            expected = sb.words_of(marked)
            tpl = templates_in_context(expected)
            r["c"]["silero"] = [sb.detect(*silero_say(marked, "kseniya"), expected, tpl)]
        json.dump(hrows, open(hp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        hard_report(hrows)


def control_report(rows):
    print(f"\n{'модель':6} {'знак на верном слоге':>22} {'знак на неверном':>18} {'случайно':>9}")
    for name in ("base", "lora", "silero"):
        rs = [r for r in rows if r["model"] == name and r["detected"] is not None]
        if not rs:
            continue
        right = [r["detected"] == r["target"] for r in rs if r["target"] == r["normal"]]
        wrong = [r["detected"] == r["target"] for r in rs if r["target"] != r["normal"]]
        chance = np.mean([1 / r["nv"] for r in rs]) if rs else 0
        print(f"{name:6} {np.mean(right):16.0%} из {len(right):<3} {np.mean(wrong):12.0%} из {len(wrong):<3} {chance:8.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "control":
        control()
    elif len(sys.argv) > 1 and sys.argv[1] == "ceiling":
        ceiling()
    elif len(sys.argv) > 1 and sys.argv[1] == "hard":
        hard()
    elif len(sys.argv) > 1 and sys.argv[1] == "hard-report":
        hard_report(json.load(open(os.path.join(WORK, f"hard_{TAG}.json"), encoding="utf-8")))
    elif len(sys.argv) > 1 and sys.argv[1] == "control-report":
        control_report(json.load(open(os.path.join(WORK, f"control_{TAG}.json"), encoding="utf-8")))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open(os.path.join(WORK, f"eval_{TAG}.json"), encoding="utf-8")))
    else:
        main()
