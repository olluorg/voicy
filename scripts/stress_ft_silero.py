"""Counterfactual training data: speech whose stress contradicts the norm, and marks that say so.

The first LoRA, trained on audiobooks alone, did not learn to obey marks: on
natural text the difference to the untouched model stayed inside the noise. In
audiobook speech a mark almost always repeats what the model would do anyway, so
the loss gives it next to no reason to read the mark.

Silero honours an explicit stress position (checked in ADR 0016 and again by the
detector validation in stress_baseline.py), so it can say a word stressed on any
syllable. Here 1–3 words per sentence are moved off their normative stress, and
those words always carry the mark in the training text: the mark is then the only
way to predict the audio. A share of sentences stays normative so Silero voices do
not come to mean «wrong stress».

Silero renders at 48 kHz, so this data is full band — a counterweight to the
16 kHz audiobooks.

Output: work/ds/silero.jsonl in the same format as train.jsonl.
"""
import json, os, random, re, subprocess, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from stress_baseline import load_ruaccent, stress_index, V, WORK, SILERO_SR
from stress_ft_data import ACUTE, MARK_P, DS, SNAP

VOICES = ("aidar", "baya", "kseniya", "eugene", "xenia")
N_SENT = int(os.environ.get("SILERO_N", 3000))
NORMATIVE_SHARE = 0.3
CHUNK = 400
OUTDIR = os.path.join(DS, "silero")


def plan():
    """Texts, which words move where, and which voice reads what."""
    rng = random.Random(21)
    acc = load_ruaccent()
    items = [r for r in json.load(open(os.path.join(DS, "items.json"), encoding="utf-8"))
             if r["split"] == "train" and 4.0 <= r["duration"] <= 12.0]
    rng.shuffle(items)
    out = []
    for r in items[:N_SENT]:
        accented = acc.process_all(r["text_no_preprocessing"])
        toks = re.findall(r"[А-Яа-яЁё+\-]+|[^А-Яа-яЁё+\-]+", accented)
        words = [j for j, t in enumerate(toks)
                 if re.match(r"[А-Яа-яЁё+]", t) and "+" in t and "ё" not in t.lower()
                 and sum(ch in V for ch in t.lower()) >= 2]
        n_shift = 0 if rng.random() < NORMATIVE_SHARE else rng.randint(1, min(3, max(1, len(words))))
        shifted = set(rng.sample(words, min(n_shift, len(words))))
        silero_toks, qwen_toks = [], []
        for j, t in enumerate(toks):
            if j not in words:
                plain = t.replace("+", "")
                silero_toks.append(t); qwen_toks.append(plain)
                continue
            plain = t.replace("+", "")
            vp = [p for p, ch in enumerate(plain.lower()) if ch in V]
            normal = stress_index(t)
            if normal is None or normal >= len(vp):
                silero_toks.append(t); qwen_toks.append(plain)
                continue
            k = rng.choice([v for v in range(len(vp)) if v != normal]) if j in shifted else normal
            silero_toks.append(plain[:vp[k]] + "+" + plain[vp[k]:])
            mark = j in shifted or rng.random() < MARK_P
            qwen_toks.append(plain[:vp[k] + 1] + ACUTE + plain[vp[k] + 1:] if mark else plain)
        out.append({"id": len(out), "voice": VOICES[len(out) % len(VOICES)],
                    "silero": "".join(silero_toks), "text": "".join(qwen_toks),
                    "plain": r["text_no_preprocessing"], "shifted": len(shifted)})
    json.dump(out, open(os.path.join(OUTDIR, "plan.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(out)} фраз, со сдвигом {sum(1 for o in out if o['shifted'])}, "
          f"сдвинутых слов {sum(o['shifted'] for o in out)}", flush=True)


def render(lo, hi):
    """Silero speech + Qwen codes for plan[lo:hi]; one process per chunk returns its memory."""
    import torch, librosa
    from qwen_tts import Qwen3TTSTokenizer
    torch.set_num_threads(12)
    plan_ = json.load(open(os.path.join(OUTDIR, "plan.json"), encoding="utf-8"))[lo:hi]
    silero = torch.package.PackageImporter(os.path.join(WORK, "v5_ru.pt")).load_pickle("tts_models", "model")
    tok = Qwen3TTSTokenizer.from_pretrained(os.path.join(SNAP, os.listdir(SNAP)[0], "speech_tokenizer"),
                                            device_map="cuda:0")
    codes, batch = {}, []

    def flush():
        enc = tok.encode([x for _, x in batch], sr=24000)
        for (key, _), c in zip(batch, enc.audio_codes):
            codes[key] = c.cpu().numpy().astype(np.int16)
        batch.clear()

    for p in plan_:
        try:
            a = silero.apply_tts(text=p["silero"], speaker=p["voice"], sample_rate=SILERO_SR,
                                 put_accent=False, put_yo=True)
        except Exception as e:  # слишком длинный или странный текст — пропускаем
            print("  пропуск", p["id"], e, flush=True)
            continue
        x = librosa.resample(a.numpy().astype(np.float32), orig_sr=SILERO_SR, target_sr=24000)
        sf.write(os.path.join(OUTDIR, "wav", f"{p['id']:05}.wav"), x, 24000)
        batch.append((str(p["id"]), x))
        if len(batch) == 16:
            flush()
    if batch:
        flush()
    out = os.path.join(OUTDIR, f"codes_{lo:05}.npz")
    np.savez(out + ".tmp.npz", **codes)
    os.replace(out + ".tmp.npz", out)  # незаконченная часть не сойдёт за готовую
    print(f"  {lo}–{hi}: {len(codes)} фраз", flush=True)


def assemble():
    """Reference: another sentence of the same Silero voice, normatively stressed, plain text."""
    import glob
    rng = random.Random(22)
    plan_ = {str(p["id"]): p for p in json.load(open(os.path.join(OUTDIR, "plan.json"), encoding="utf-8"))}
    codes = {}
    for f in glob.glob(os.path.join(OUTDIR, "codes_*.npz")):
        with np.load(f) as z:
            codes.update({k: z[k] for k in z.files})
    refs = {}
    for k, p in plan_.items():
        if k in codes and p["shifted"] == 0 and 50 <= len(codes[k]) <= 125:
            refs.setdefault(p["voice"], []).append(k)
    n = 0
    with open(os.path.join(DS, "silero.jsonl"), "w", encoding="utf-8") as f:
        for k, p in plan_.items():
            if k not in codes or not refs.get(p["voice"]):
                continue
            ref = rng.choice([r for r in refs[p["voice"]] if r != k] or refs[p["voice"]])
            f.write(json.dumps({"text": p["text"], "codes": codes[k].tolist(),
                                "ref_text": plan_[ref]["plain"], "ref_codes": codes[ref].tolist(),
                                "ref_audio": os.path.join(OUTDIR, "wav", f"{int(ref):05}.wav"),
                                "reader": "silero-" + p["voice"], "shifted": p["shifted"]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"silero.jsonl: {n} примеров")


def main():
    os.makedirs(os.path.join(OUTDIR, "wav"), exist_ok=True)
    if not os.path.exists(os.path.join(OUTDIR, "plan.json")):
        plan()
    total = len(json.load(open(os.path.join(OUTDIR, "plan.json"), encoding="utf-8")))
    for lo in range(0, total, CHUNK):
        if not os.path.exists(os.path.join(OUTDIR, f"codes_{lo:05}.npz")):
            subprocess.run([sys.executable, __file__, "render", str(lo), str(min(total, lo + CHUNK))], check=True)
    assemble()


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "render":
        render(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
