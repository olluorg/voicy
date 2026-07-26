"""Round 6: pause length belongs to the boundary, not to the cue.

The previous model gave every cue a `pad_before` and a `pad_after`, so the gap
between two cues was the *sum* of two pads, plus any explicit beat, and then the
whole thing was multiplied by a global pause scale and finally stretched by
atempo along with the speech. Four independent sources of silence stacked:
34% of the file was silence and the median pause was 2.15 s.

This version fixes the model rather than the numbers:

  * one gap per boundary, looked up from (previous kind, next kind);
  * no global multiplier — rate comes from stretching *speech*, never silence;
  * atempo is applied per speech segment before assembly, so a pause is exactly
    as long as it says it is.
"""
import json, sys, time, pathlib, subprocess, tempfile, os, warnings
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
FADE_MS = 45
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path("v3"); OUT.mkdir(exist_ok=True)
V = set("аеёиоуыэюя")

# Articulation rate per cue kind. Unchanged — this part worked.
RATE = {"narration": 1.02, "term": 0.90, "key": 0.88, "question": 0.94}

# Gap between two neighbouring cues, in ms, by boundary semantics.
# Within one thought a pause is a breath, not a stop.
BREATH, CLAUSE, THOUGHT, SECTION = 140, 260, 480, 700


def gap(prev_kind, next_kind):
    if prev_kind is None or next_kind is None:
        return 0
    if prev_kind == "key":
        return SECTION                      # let the memorable line land
    if next_kind in ("term", "key"):
        return THOUGHT                      # set up what is coming
    if prev_kind == "question":
        return CLAUSE
    if prev_kind == "term":
        return CLAUSE                       # the term's own explanation follows
    return BREATH                           # narration flowing on


def syllables(t):
    return sum(1 for c in t.lower() if c in V)


def fade(x, sr, ms):
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = x.copy(); y[:n] *= ramp; y[-n:] *= ramp[::-1]
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
    return fade(buf, SR, min(30, max(1, ms // 3)))


def stretch(x, factor):
    """Time-stretch speech only, via ffmpeg's WSOLA. Pauses never pass through here."""
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
cues = [c for c in doc["cues"] if c["kind"] != "beat"]          # explicit beats replaced by the gap model
THINK_AFTER = 2                                                 # index of the cue the learner answers after
TOTAL_SYL = syllables(" ".join(c["text"] for c in cues))

print("синтез реплик...", flush=True)
segs = []
for c in cues:
    torch.manual_seed(1234)
    wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(c["text"]), model, voc,
                               cross_fade_duration=0.15, nfe_step=48, speed=RATE[c["kind"]])
    segs.append(np.asarray(wav, dtype=np.float32))
print(f"  {len(segs)} реплик, {sum(len(s) for s in segs)/SR:.1f} с чистой речи", flush=True)

results = []


def assemble(tag, factor):
    parts, sil_ms = [], 0
    for i, (c, raw) in enumerate(zip(cues, segs)):
        if i:
            g = gap(cues[i - 1]["kind"], c["kind"])
            if i - 1 == THINK_AFTER:
                g = 2500                                        # the one deliberate think-pause
            sil_ms += g
            parts.append(silence(g, TONE))
        parts.append(fade(stretch(raw, factor), SR, FADE_MS))
    audio = np.concatenate(parts)
    sf.write(str(OUT / f"{tag}.wav"), audio, SR)
    dur = len(audio) / SR
    r = {"tag": tag, "stretch": factor, "audio_s": round(dur, 1),
         "silence_s": round(sil_ms / 1000, 1), "silence_pct": round(sil_ms / 10 / dur, 1),
         "syl_per_s": round(TOTAL_SYL / dur, 2), "wpm": round(TOTAL_SYL / dur * 60 / 2.37)}
    results.append(r)
    print(f"{tag:10} {dur:6.1f}s  тишины {r['silence_pct']:4.1f}%  "
          f"{r['syl_per_s']:.2f} слог/с  ~{r['wpm']} слов/мин", flush=True)


for tag, f in (("v3-1.00", 1.00), ("v3-0.88", 0.88), ("v3-0.82", 0.82), ("v3-0.76", 0.76), ("v3-0.70", 0.70)):
    assemble(tag, f)

json.dump({"total_syllables": TOTAL_SYL, "gap_model": {"breath": BREATH, "clause": CLAUSE,
           "thought": THOUGHT, "section": SECTION}, "results": results},
          open("results_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
