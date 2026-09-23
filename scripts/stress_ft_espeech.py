"""ESpeech-TTS as the source of counterfactual stress, in place of Silero.

Silero TTS is CC BY-NC: an adapter trained on its speech cannot honestly be
published under Apache-2.0 with Qwen. ESpeech-TTS-1 (F5-TTS finetune, Apache-2.0)
reads RUAccent's «+» markup and clones any voice from a sample, so it could give
the same kind of data — if it really puts the stress where the «+» says, also
against the norm. That is checked first (`check`), with the same detector and the
same control sentences as the Qwen adapters and Silero.

    python scripts/stress_ft_espeech.py check
    python scripts/stress_ft_espeech.py data      # после проверки
"""
import json, os, random, re, subprocess, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.path.dirname(__file__))
import stress_baseline as sb
from stress_ft_data import ACUTE, MARK_P, DS, SNAP

WORK = sb.WORK
REPO, FNAME = "ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt"
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
CHECK_VOICE = "server/voices/dostoevsky.wav"
CHECK_TEXT = json.load(open("server/voices/voices.json", encoding="utf-8"))["dostoevsky"]["text"]


def load_espeech():
    import torchaudio
    from huggingface_hub import hf_hub_download
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    def _load_wav(path, *a, **kw):  # torchaudio.load ходит через torchcodec, нужны .so FFmpeg
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    torchaudio.load = _load_wav
    model = load_model(DiT, MODEL_CFG, hf_hub_download(REPO, FNAME), vocab_file=hf_hub_download(REPO, "vocab.txt"))
    vocoder = load_vocoder()
    return model.to("cuda"), vocoder.to("cuda")


def speaker(model, vocoder):
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    cache = {}

    def say(text_plus, ref_wav, ref_text_plus, seed=1):
        if ref_wav not in cache:
            cache[ref_wav] = preprocess_ref_audio_text(ref_wav, ref_text_plus)
        ra, rt = cache[ref_wav]
        torch.manual_seed(seed)
        w, sr, _ = infer_process(ra, rt, text_plus, model, vocoder, cross_fade_duration=0.15,
                                 nfe_step=32, speed=1.0, show_info=lambda *a, **k: None)
        return np.asarray(w, dtype=np.float32), sr
    return say


def asr_patch():
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]
    sb.transcribe_words = transcribe_words


def check():
    asr_patch()
    acc, silero_say, templates_in_context = sb.setup()
    say = speaker(*load_espeech())
    ref_plus = acc.process_all(CHECK_TEXT)
    rows = []
    os.makedirs(os.path.join(WORK, "eval", "espeech"), exist_ok=True)
    for n, text in enumerate(sb.EVERYDAY, 1):
        expected = sb.words_of(acc.process_all(text))
        cand = [(sum(ch in sb.V for ch in w.lower()), i) for i, (w, e) in enumerate(expected)
                if e is not None and sb.eligible(w) and "ё" not in w.lower()]
        if not cand:
            continue
        nv, i = max(cand)
        tpl = templates_in_context(expected)
        for k in range(nv):
            words = [w for w, _ in expected]
            words[i] = sb.place(expected[i][0], k)
            x, sr = say(" ".join(words) + ".", CHECK_VOICE, ref_plus)
            sf.write(os.path.join(WORK, "eval", "espeech", f"{n:02}_{k}.wav"), x, sr)
            d = sb.detect(x, sr, expected, tpl).get(i)
            rows.append({"word": expected[i][0].replace("+", ""), "target": k, "normal": expected[i][1],
                         "nv": nv, "model": "espeech", "detected": None if not d else min(d, key=d.get)})
        print(f"[{n:2}/{len(sb.EVERYDAY)}] {rows[-1]['word']}", flush=True)
        json.dump(rows, open(os.path.join(WORK, "control_espeech.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    rs = [r for r in rows if r["detected"] is not None]
    right = [r["detected"] == r["target"] for r in rs if r["target"] == r["normal"]]
    wrong = [r["detected"] == r["target"] for r in rs if r["target"] != r["normal"]]
    print(f"\nESpeech: знак на нормативном {np.mean(right):.0%} из {len(right)}, "
          f"на ненормативном {np.mean(wrong):.0%} из {len(wrong)} (Silero: 79% / 66%)")


OUTDIR = os.path.join(DS, "espeech")
CHUNK = 300


def voices():
    """Reference voices, all public domain: the five RuLS training readers and
    `dostoevsky`. `turgenev` is left out — it is the voice every evaluation uses."""
    items = [r for r in json.load(open(os.path.join(DS, "items.json"), encoding="utf-8"))
             if r["split"] == "train" and 6.0 <= r["duration"] <= 10.0]
    out = {"dostoevsky": (CHECK_VOICE, CHECK_TEXT)}
    for reader in sorted({r["reader"] for r in items}):
        r = next(x for x in items if x["reader"] == reader)
        from stress_ft_data import wav_path
        out[f"ruls-{reader}"] = (wav_path(r["audio_filepath"]), r["text_no_preprocessing"])
    return out


def render(lo, hi):
    """ESpeech speech + Qwen codes for plan[lo:hi]; a process per chunk returns its memory."""
    from qwen_tts import Qwen3TTSTokenizer
    import librosa
    from stress_baseline import load_ruaccent
    plan = json.load(open(os.path.join(DS, "silero", "plan.json"), encoding="utf-8"))[lo:hi]
    vs = voices()
    names = sorted(vs)
    acc = load_ruaccent()
    refs_plus = {n: acc.process_all(t) for n, (_, t) in vs.items()}
    say = speaker(*load_espeech())
    tok = Qwen3TTSTokenizer.from_pretrained(os.path.join(SNAP, os.listdir(SNAP)[0], "speech_tokenizer"),
                                            device_map="cuda:0")
    codes, meta, batch = {}, {}, []

    def flush():
        enc = tok.encode([x for _, x in batch], sr=24000)
        for (key, _), c in zip(batch, enc.audio_codes):
            codes[key] = c.cpu().numpy().astype(np.int16)
        batch.clear()

    for p in plan:
        voice = names[p["id"] % len(names)]
        try:
            x, sr = say(p["silero"], vs[voice][0], refs_plus[voice], seed=p["id"])
        except Exception as e:
            print("  пропуск", p["id"], e, flush=True)
            continue
        if sr != 24000:
            x = librosa.resample(x, orig_sr=sr, target_sr=24000)
        sf.write(os.path.join(OUTDIR, "wav", f"{p['id']:05}.wav"), x, 24000)
        meta[str(p["id"])] = voice
        batch.append((str(p["id"]), x))
        if len(batch) == 16:
            flush()
    if batch:
        flush()
    out = os.path.join(OUTDIR, f"codes_{lo:05}.npz")
    np.savez(out + ".tmp.npz", **codes)
    json.dump(meta, open(os.path.join(OUTDIR, f"voices_{lo:05}.json"), "w"))
    os.replace(out + ".tmp.npz", out)
    print(f"  {lo}–{hi}: {len(codes)} фраз", flush=True)


def assemble():
    import glob
    rng = random.Random(23)
    plan = {str(p["id"]): p for p in json.load(open(os.path.join(DS, "silero", "plan.json"), encoding="utf-8"))}
    codes, voice = {}, {}
    for f in glob.glob(os.path.join(OUTDIR, "codes_*.npz")):
        with np.load(f) as z:
            codes.update({k: z[k] for k in z.files})
        voice.update(json.load(open(f.replace("codes_", "voices_").replace(".npz", ".json"))))
    refs = {}
    for k in codes:
        if plan[k]["shifted"] == 0 and 50 <= len(codes[k]) <= 125:
            refs.setdefault(voice[k], []).append(k)
    n = 0
    with open(os.path.join(DS, "espeech.jsonl"), "w", encoding="utf-8") as f:
        for k, c in codes.items():
            pool = [r for r in refs.get(voice[k], []) if r != k]
            if not pool:
                continue
            ref = rng.choice(pool)
            f.write(json.dumps({"text": plan[k]["text"], "codes": c.tolist(),
                                "ref_text": plan[ref]["plain"], "ref_codes": codes[ref].tolist(),
                                "ref_audio": os.path.join(OUTDIR, "wav", f"{int(ref):05}.wav"),
                                "reader": "espeech-" + voice[k], "shifted": plan[k]["shifted"]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"espeech.jsonl: {n} примеров")


def data():
    os.makedirs(os.path.join(OUTDIR, "wav"), exist_ok=True)
    total = min(int(os.environ.get("ESPEECH_N", 3000)),
                len(json.load(open(os.path.join(DS, "silero", "plan.json"), encoding="utf-8"))))
    for lo in range(0, total, CHUNK):
        if not os.path.exists(os.path.join(OUTDIR, f"codes_{lo:05}.npz")):
            subprocess.run([sys.executable, __file__, "render", str(lo), str(min(total, lo + CHUNK))], check=True)
    assemble()


if __name__ == "__main__":
    if sys.argv[1] == "render":
        render(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"check": check, "data": data, "assemble": assemble}[sys.argv[1]]()
