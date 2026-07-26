"""Round 4: pause shape + speaking rate.

Two defects found by ear and confirmed by measurement:

1. Pauses were spliced as digital zero. Amplitude fell from the recording's own
   noise floor (~0.013) to exactly 0.0000 in one sample, so the pause did not
   read as "the speaker stopped" but as "the recording stopped". Fixed with a
   raised-cosine fade plus real room tone lifted from the reference recording.

2. Rate was 5.6-6.3 syllables/s (142-160 wpm) -- entertainment-podcast speed.
   Teaching norms for technical material land at 2.5-4.0 syl/s. `SCALES` renders
   a ladder so the choice is made on measured rate, not on feel.
"""
import json, sys, time, pathlib, warnings
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

STYLE = {
    "narration": {"speed": 1.02, "pad_before": 120, "pad_after": 120},
    "term":      {"speed": 0.88, "pad_before": 420, "pad_after": 260},
    "key":       {"speed": 0.86, "pad_before": 480, "pad_after": 420},
    "question":  {"speed": 0.92, "pad_before": 300, "pad_after": 250},
}
FADE_MS = 45
SCALES = [1.0, 0.85, 0.75, 0.68]
SR_OUT = 24000

CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path("v2"); OUT.mkdir(exist_ok=True)
V = set("аеёиоуыэюя")


def syllables(t: str) -> int:
    return sum(1 for c in t.lower() if c in V)


def fade(x: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Raised-cosine ramp on both ends so a splice never lands on a live sample."""
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def room_tone(ref_path: str, sr: int) -> np.ndarray:
    """Quietest 400 ms of the reference: the pause is filled with the same air
    the voice was recorded in, instead of dead digital silence."""
    a, s = sf.read(ref_path, dtype="float32", always_2d=True)
    a = a[:, 0]
    if s != sr:
        a = np.interp(np.linspace(0, len(a), int(len(a) * sr / s)), np.arange(len(a)), a).astype(np.float32)
    win = int(sr * 0.4)
    best, lo = None, np.inf
    for i in range(0, len(a) - win, win // 4):
        e = float(np.abs(a[i:i + win]).mean())
        if e < lo:
            lo, best = e, a[i:i + win].copy()
    return best if best is not None else np.zeros(win, dtype=np.float32)


def silence(ms: int, sr: int, tone: np.ndarray) -> np.ndarray:
    """Room tone, tiled and cross-faded at the loop point, ramped in and out."""
    n = int(sr * ms / 1000)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    reps = int(np.ceil(n / len(tone)))
    buf = np.tile(tone, reps)[:n].astype(np.float32)
    return fade(buf, sr, min(30, ms // 3))


acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
model = load_model(DiT, CFG,
                   hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="espeech_tts_rlv2.pt"),
                   vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="vocab.txt")).to("cuda")

_rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_txt = next(v for k, v in _rt.items() if "ref_ys" in k)
ref_a, ref_t = preprocess_ref_audio_text("lv/ref_ys.wav", acc.process_all(ref_txt))
TONE = room_tone("lv/ref_ys.wav", SR_OUT)
print(f"room tone: {len(TONE)/SR_OUT:.2f}s, rms={np.sqrt((TONE**2).mean()):.5f}", flush=True)

doc = json.load(open("cues.json", encoding="utf-8"))
flat_text = " ".join(c["text"] for c in doc["cues"] if c["kind"] != "beat")
TOTAL_SYL = syllables(flat_text)

results = []


def render(scale: float, pause_mode: str, tag: str):
    parts, t0 = [], time.perf_counter()
    for cue in doc["cues"]:
        if cue["kind"] == "beat":
            parts.append(("gap", cue["ms"]))
            continue
        st = STYLE[cue["kind"]]
        torch.manual_seed(1234)
        wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(cue["text"]), model, voc,
                                   cross_fade_duration=0.15, nfe_step=48,
                                   speed=round(st["speed"] * scale, 3))
        w = np.asarray(wav, dtype=np.float32)
        if pause_mode != "hard":
            w = fade(w, sr, FADE_MS)
        parts.append(("gap", st["pad_before"]))
        parts.append(("aud", w))
        parts.append(("gap", st["pad_after"]))

    chunks = []
    for kind, val in parts:
        if kind == "aud":
            chunks.append(val)
        elif pause_mode == "tone":
            chunks.append(silence(val, SR_OUT, TONE))
        else:
            chunks.append(np.zeros(int(SR_OUT * val / 1000), dtype=np.float32))
    audio = np.concatenate(chunks)
    name = f"{tag}.wav"
    sf.write(str(OUT / name), audio, SR_OUT)
    dur = len(audio) / SR_OUT
    r = {"tag": tag, "scale": scale, "pause": pause_mode, "file": name,
         "audio_s": round(dur, 2), "syl_per_s": round(TOTAL_SYL / dur, 2),
         "wpm": round(TOTAL_SYL / dur * 60 / 2.37, 0), "wall_s": round(time.perf_counter() - t0, 1)}
    results.append(r)
    print(f"{tag:34} {dur:6.1f}s  {r['syl_per_s']:.2f} слог/с  ~{r['wpm']:.0f} слов/мин", flush=True)


# A. pause shape, rate held at the current setting
for mode in ("hard", "fade", "tone"):
    render(1.0, mode, f"pause-{mode}")

# B. rate ladder, best pause shape
for s in SCALES:
    if s == 1.0:
        continue
    render(s, "tone", f"rate-{s}")

json.dump({"total_syllables": TOTAL_SYL, "results": results},
          open("results_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
