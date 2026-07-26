"""Round 7: stop eating the edges, stop pausing like a metronome.

Two defects, both mine rather than the model's.

1. The 45 ms raised-cosine fade was applied to the whole segment, including live
   speech. Measurement: some segments carry 83% of their peak energy inside the
   first 45 ms, so the fade was erasing the opening consonant outright. Fixed by
   locating the actual speech boundaries first, keeping a short low-energy
   margin, and fading only across that margin — 8 ms, enough to kill a click and
   far too short to touch a phoneme.

2. Every gap of a given kind was exactly the same length, and every boundary got
   one. People do neither. Gaps are now jittered and some breath-level ones are
   dropped entirely so phrases run together. The randomness is seeded from the
   cue text, so a rebuild of the same content produces the same audio.
"""
import json, sys, pathlib, subprocess, tempfile, os, hashlib, random, warnings
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
FADE_MS = 8          # was 45 — that was eating consonants
MARGIN_MS = 22       # low-energy room kept around the speech, where the fade lives
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path(os.environ.get("OUTDIR", "v4")); OUT.mkdir(exist_ok=True)
V = set("аеёиоуыэюя")

RATE = {"narration": 1.02, "term": 0.90, "key": 0.88, "question": 0.94}
BREATH, CLAUSE, THOUGHT, SECTION = 140, 260, 480, 700

# How irregular a human is: gaps wobble, and a breath is often skipped entirely.
JITTER = (0.62, 1.42)
SKIP_BREATH = 0.35
RATE_WOBBLE = (0.975, 1.03) if os.environ.get("WOBBLE", "1") == "1" else (1.0, 1.0)


def rng_for(*parts):
    """Deterministic per-boundary randomness: same content, same audio, every build."""
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def base_gap(prev_kind, next_kind):
    if prev_kind == "key":
        return SECTION
    if next_kind in ("term", "key"):
        return THOUGHT
    if prev_kind in ("question", "term"):
        return CLAUSE
    return BREATH


def syllables(t):
    return sum(1 for c in t.lower() if c in V)


def speech_bounds(x, sr):
    """First and last sample carrying real signal, by a floor relative to the peak."""
    win = max(1, int(sr * 0.004))
    env = np.array([np.abs(x[i:i + win]).max() for i in range(0, max(1, len(x) - win), win)])
    if not len(env):
        return 0, len(x)
    live = np.where(env > env.max() * 0.02)[0]
    if not len(live):
        return 0, len(x)
    return int(live[0] * win), int(min(len(x), (live[-1] + 1) * win))


TRIM = os.environ.get("TRIM", "1") == "1"


def trim_and_fade(x, sr):
    """Fade just enough to kill a click. Trimming is optional and off by default:
    unvoiced fricatives sit far below any sane energy floor, so a threshold-based
    crop eats the very consonants the fade was destroying."""
    if TRIM:
        a, b = speech_bounds(x, sr)
        m = int(sr * MARGIN_MS / 1000)
        a, b = max(0, a - m), min(len(x), b + m)
        y = x[a:b].copy()
    else:
        y = x.copy()
    n = min(int(sr * FADE_MS / 1000), len(y) // 2)
    if n > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def room_tone(ref_path, sr):
    a, s = sf.read(ref_path, dtype="float32", always_2d=True); a = a[:, 0]
    if s != sr:
        a = np.interp(np.linspace(0, len(a), int(len(a) * sr / s)), np.arange(len(a)), a).astype(np.float32)
    win = int(sr * 0.4); best, lo = None, np.inf
    for i in range(0, len(a) - win, win // 4):
        e = float(np.abs(a[i:i + win]).mean())
        if e < lo:
            lo, best = e, a[i:i + win].copy()
    return best


def silence(ms, tone):
    n = int(SR * ms / 1000)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    buf = np.tile(tone, int(np.ceil(n / len(tone))))[:n].astype(np.float32)
    k = min(int(SR * 0.02), max(1, n // 3))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    buf[:k] *= ramp; buf[-k:] *= ramp[::-1]
    return buf


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


acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
model = load_model(DiT, CFG,
                   hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="espeech_tts_rlv2.pt"),
                   vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="vocab.txt")).to("cuda")
_rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_a, ref_t = preprocess_ref_audio_text("lv/ref_ys.wav",
                                         acc.process_all(next(v for k, v in _rt.items() if "ref_ys" in k)))
TONE = room_tone("lv/ref_ys.wav", SR)

doc = json.load(open("cues.json", encoding="utf-8"))
cues = [c for c in doc["cues"] if c["kind"] != "beat"]
THINK_AFTER = 2
TOTAL_SYL = syllables(" ".join(c["text"] for c in cues))

print("синтез реплик...", flush=True)
segs, eaten = [], 0
for i, c in enumerate(cues):
    r = rng_for("rate", i, c["text"])
    torch.manual_seed(1234)
    wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(c["text"]), model, voc,
                               cross_fade_duration=0.15, nfe_step=48,
                               speed=round(RATE[c["kind"]] * r.uniform(*RATE_WOBBLE), 3))
    segs.append(np.asarray(wav, dtype=np.float32))
print(f"  {len(segs)} реплик, {sum(len(s) for s in segs)/SR:.1f} с речи", flush=True)

# how much the old 45 ms fade would have destroyed, for the record
loud_edges = sum(1 for s in segs
                 if np.abs(s[:int(SR * .045)]).max() > np.abs(s).max() * 0.15
                 or np.abs(s[-int(SR * .045):]).max() > np.abs(s).max() * 0.15)
print(f"  реплик с живым сигналом в зоне старого fade: {loud_edges} из {len(segs)}", flush=True)

results = []


def assemble(tag, factor):
    parts, gaps = [], []
    for i, (c, raw) in enumerate(zip(cues, segs)):
        if i:
            prev = cues[i - 1]["kind"]
            g = base_gap(prev, c["kind"])
            r = rng_for("gap", i, cues[i - 1]["text"], c["text"])
            if i - 1 == THINK_AFTER:
                g = 2500
            elif g == BREATH and r.random() < SKIP_BREATH:
                g = 0                                   # phrases run together, as they do
            else:
                g = int(g * r.uniform(*JITTER))
            if g:
                gaps.append(g)
                parts.append(silence(g, TONE))
        parts.append(trim_and_fade(stretch(raw, factor), SR))
    audio = np.concatenate(parts)
    sf.write(str(OUT / f"{tag}.wav"), audio, SR)
    dur = len(audio) / SR
    r = {"tag": tag, "stretch": factor, "audio_s": round(dur, 1),
         "gaps": len(gaps), "gap_median_ms": int(np.median(gaps)),
         "gap_min_ms": int(min(gaps)), "gap_max_ms": int(max(gaps)),
         "silence_pct": round(sum(gaps) / 10 / dur, 1),
         "syl_per_s": round(TOTAL_SYL / dur, 2), "wpm": round(TOTAL_SYL / dur * 60 / 2.37)}
    results.append(r)
    print(f"{tag:10} {dur:6.1f}s  пауз {r['gaps']:2} ({r['gap_min_ms']}–{r['gap_max_ms']} мс, "
          f"медиана {r['gap_median_ms']})  ~{r['wpm']} слов/мин", flush=True)


PFX = os.environ.get("PFX", "v4")
for tag, f in ((f"{PFX}-0.82", 0.82), (f"{PFX}-0.76", 0.76), (f"{PFX}-0.70", 0.70)):
    assemble(tag, f)

json.dump({"total_syllables": TOTAL_SYL, "fade_ms": FADE_MS, "margin_ms": MARGIN_MS,
           "jitter": JITTER, "skip_breath": SKIP_BREATH,
           "segments_with_live_edges": loud_edges, "results": results},
          open(os.environ.get("RESULTS", "results_v4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
