"""Every F5-family Russian checkpoint on the same benchmark.

Four of these were never tried: the first round stopped as soon as one variant
worked. Same reference clip, same text, same metrics, so the numbers sit next to
the ones already collected.
"""
import json, sys, os, time, pathlib, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch, torchaudio


def _load_wav(p, *a, **k):
    d, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr


torchaudio.load = _load_wav
from onnxruntime import InferenceSession as _I

_o = _I.run


def _r(self, n, f, *a, **k):
    mm = {i.name for i in self.get_inputs()} - set(f)
    if mm and "input_ids" in f:
        f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in mm}}
    return _o(self, n, f, *a, **k)


_I.run = _r

from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

SR = 24000
V = set("аеёиоуыэюя")
OUT = pathlib.Path("bench"); OUT.mkdir(exist_ok=True)
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

MODELS = [
    ("espeech-rlv2",  "ESpeech/ESpeech-TTS-1_RL-V2",    "espeech_tts_rlv2.pt",   "vocab.txt"),
    ("espeech-rlv1",  "ESpeech/ESpeech-TTS-1_RL-V1",    "espeech_tts_rlv1.pt",   "vocab.txt"),
    ("espeech-sft95", "ESpeech/ESpeech-TTS-1_SFT-95K",  "espeech_tts_95k.pt",    "vocab.txt"),
    ("espeech-sft256","ESpeech/ESpeech-TTS-1_SFT-256K", "espeech_tts_256k.pt",   "vocab.txt"),
    ("espeech-podcast","ESpeech/ESpeech-TTS-1_podcaster","espeech_tts_podcaster.pt","vocab.txt"),
    ("f5ru-misha",    "Misha24-10/F5-TTS_RUSSIAN",      "F5TTS_v1_Base/model_240000.pt", "F5TTS_v1_Base/vocab.txt"),
    ("f5ru-misha-v2", "Misha24-10/F5-TTS_RUSSIAN",      "F5TTS_v1_Base_v2/model_last.pt", "F5TTS_v1_Base/vocab.txt"),
]

PHRASE = "И вот причина ради которой этот вопрос и задают."

acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_txt = next(v for k, v in rt.items() if "ref_t2" in k)
ref_a, ref_t = preprocess_ref_audio_text("lv/ref_t2.wav", acc.process_all(ref_txt))

doc = json.load(open("cues_v3.json", encoding="utf-8"))
cues = [c for c in doc["cues"] if "text" in c]
flat = " ".join(c["text"] for c in cues)
SYL = sum(1 for c in flat.lower() if c in V)


def stats(x):
    win = int(SR * 0.005)
    env = np.array([np.abs(x[i:i + win]).max() for i in range(0, len(x) - win, win)])
    sil = env < env.max() * 0.004
    runs, i = [], 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) >= 12:
                runs.append((j - i) * 5)
            i = j
        else:
            i += 1
    d = len(x) / SR
    return d, len(runs), sum(runs) / 10 / d, (int(np.median(runs)) if runs else 0)


def inner_breaks(x):
    win = int(SR * 0.005)
    env = np.array([np.abs(x[i:i + win]).max() for i in range(0, len(x) - win, win)])
    sil = env < env.max() * 0.02
    live = np.where(~sil)[0]
    if not len(live):
        return 0
    sil = sil[live[0]:live[-1] + 1]
    n, i = 0, 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) >= 20:
                n += 1
            i = j
        else:
            i += 1
    return n


results = []
for tag, repo, fname, vname in MODELS:
    try:
        mp = hf_hub_download(repo_id=repo, filename=fname)
        vp = hf_hub_download(repo_id=repo, filename=vname)
        model = load_model(DiT, CFG, mp, vocab_file=vp).to("cuda")
    except Exception as e:
        print(f"{tag:16} — не загрузился: {str(e)[:70]}", flush=True)
        continue

    torch.manual_seed(1234)
    t0 = time.perf_counter()
    wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(flat), model, voc,
                               cross_fade_duration=0.15, nfe_step=48, speed=1.0)
    wall = time.perf_counter() - t0
    x = np.asarray(wav, dtype=np.float32)
    sf.write(str(OUT / f"{tag}.wav"), x, SR)
    d, n, pct, med = stats(x)

    torch.manual_seed(1234)
    pw, _, _ = infer_process(ref_a, ref_t, acc.process_all(PHRASE), model, voc,
                             cross_fade_duration=0.15, nfe_step=48, speed=1.0)
    px = np.asarray(pw, dtype=np.float32)
    sf.write(str(OUT / f"{tag}__phrase.wav"), px, SR)
    br = inner_breaks(px)

    r = {"tag": tag, "repo": repo, "audio_s": round(d, 1), "rtf": round(d / wall, 2),
         "pauses": n, "silence_pct": round(pct, 1), "pause_median_ms": med,
         "wpm": round(SYL / d * 60 / 2.37), "phrase_breaks": br}
    results.append(r)
    print(f"{tag:16} ~{r['wpm']:3} слов/мин  пауз {n:3} ({pct:4.1f}%, медиана {med:4} мс)  "
          f"разрывов во фразе {br}  ×{d/wall:.1f}", flush=True)
    del model
    torch.cuda.empty_cache()

json.dump({"total_syllables": SYL, "results": results},
          open("results_bench_f5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
