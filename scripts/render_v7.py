"""Round 8: no inserted silence at all.

Every pause the listener hears now comes from the model reading punctuation,
not from samples I splice in. Two ways to get there:

  A `solid`  — the whole script synthesised in one pass. Truly "as it was
               before pauses": one rate throughout, every boundary the model's.
  B `joined` — still split into cues, so a term still slows down, but the
               segments are butted together with zero inserted gap.

Rate is reached the only way that survived measurement: stretching the finished
audio, never the model's speed knob.
"""
import json, sys, pathlib, subprocess, tempfile, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")

import numpy as np, soundfile as sf, torch, torchaudio


def _load_wav(p, *a, **k):
    d, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr


torchaudio.load = _load_wav

from onnxruntime import InferenceSession as _I

_o = _I.run


def _r(self, n, f, *a, **k):
    m = {i.name for i in self.get_inputs()} - set(f)
    if m and "input_ids" in f:
        f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in m}}
    return _o(self, n, f, *a, **k)


_I.run = _r

from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

SR = 24000
FADE_MS = 8                      # anti-click only; not a pause
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path("v7"); OUT.mkdir(exist_ok=True)
V = set("аеёиоуыэюя")
RATE = {"narration": 1.02, "term": 0.90, "key": 0.88, "question": 0.94}


def syllables(t):
    return sum(1 for c in t.lower() if c in V)


def fade(x):
    n = min(int(SR * FADE_MS / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = x.copy(); y[:n] *= ramp; y[-n:] *= ramp[::-1]
    return y


def stretch(x, factor):
    if abs(factor - 1.0) < 1e-3:
        return x
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, x, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a,
                        "-filter:a", f"atempo={factor}", b], check=True)
        y, _ = sf.read(b, dtype="float32")
    return y


def silence_stats(x):
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
    if not runs:
        return 0, 0, 0, 0
    return len(runs), sum(runs) / 10 / (len(x) / SR), int(np.median(runs)), int(max(runs))


acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
model = load_model(DiT, CFG,
                   hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="espeech_tts_rlv2.pt"),
                   vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="vocab.txt")).to("cuda")
_rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_a, ref_t = preprocess_ref_audio_text("lv/ref_ys.wav",
                                         acc.process_all(next(v for k, v in _rt.items() if "ref_ys" in k)))

doc = json.load(open("cues.json", encoding="utf-8"))
cues = [c for c in doc["cues"] if c["kind"] != "beat"]
flat = " ".join(c["text"] for c in cues)
TOTAL_SYL = syllables(flat)

print("A. сплошной синтез всего текста...", flush=True)
torch.manual_seed(1234)
wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(flat), model, voc,
                           cross_fade_duration=0.15, nfe_step=48, speed=1.0)
solid = np.asarray(wav, dtype=np.float32)
print(f"   {len(solid)/SR:.1f} с", flush=True)

print("B. по репликам, стык в стык...", flush=True)
segs = []
for c in cues:
    torch.manual_seed(1234)
    w, _sr, _ = infer_process(ref_a, ref_t, acc.process_all(c["text"]), model, voc,
                              cross_fade_duration=0.15, nfe_step=48, speed=RATE[c["kind"]])
    segs.append(fade(np.asarray(w, dtype=np.float32)))
joined = np.concatenate(segs)
print(f"   {len(joined)/SR:.1f} с", flush=True)

results = []
for name, base in (("solid", solid), ("joined", joined)):
    for f in (1.0, 0.78, 0.72, 0.66):
        y = stretch(base, f)
        tag = f"{name}-{f:.2f}"
        sf.write(str(OUT / f"{tag}.wav"), y, SR)
        dur = len(y) / SR
        n, pct, med, mx = silence_stats(y)
        r = {"tag": tag, "mode": name, "stretch": f, "audio_s": round(dur, 1),
             "pauses": n, "silence_pct": round(pct, 1), "pause_median_ms": med, "pause_max_ms": mx,
             "syl_per_s": round(TOTAL_SYL / dur, 2), "wpm": round(TOTAL_SYL / dur * 60 / 2.37)}
        results.append(r)
        print(f"{tag:14} {dur:6.1f}s  ~{r['wpm']:3} слов/мин  пауз {n:2} "
              f"({pct:4.1f}% тишины, медиана {med} мс, макс {mx} мс)", flush=True)

json.dump({"total_syllables": TOTAL_SYL, "results": results},
          open("results_v7.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
