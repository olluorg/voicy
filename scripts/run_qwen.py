"""Qwen3-TTS on the same benchmark: same text, same reference clip, same metrics.

Qwen3-TTS shipped in January 2026, months after the Russian TTS leaderboard was
last updated, so nothing about it could be taken from the existing comparison.
It clones a voice from a short sample like F5 does, which means the reference
that was chosen by ear stays usable and the two engines are directly comparable.
"""
import json, sys, os, time, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "qwen"; os.makedirs(OUT, exist_ok=True)
V = set("аеёиоуыэюя")
REF = "lv/ref_t2.wav"

rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

doc = json.load(open("cues_v3.json", encoding="utf-8"))
cues = [c for c in doc["cues"] if "text" in c]
flat = " ".join(c["text"] for c in cues)
SYL = sum(1 for c in flat.lower() if c in V)

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
print(f"загружаю {MODEL} ...", flush=True)
model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
print("готово", flush=True)


def stats(x, sr):
    win = int(sr * 0.005)
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
    d = len(x) / sr
    return d, len(runs), sum(runs) / 10 / d, (int(np.median(runs)) if runs else 0)


results = []

# A. whole script in one pass
t0 = time.perf_counter()
wavs, sr = model.generate_voice_clone(text=flat, language="Russian",
                                      ref_audio=REF, ref_text=ref_text)
wall = time.perf_counter() - t0
x = np.asarray(wavs[0], dtype=np.float32)
sf.write(f"{OUT}/qwen-solid.wav", x, sr)
d, n, pct, med = stats(x, sr)
results.append({"tag": "qwen-solid", "audio_s": round(d, 1), "wall_s": round(wall, 1),
                "rtf": round(d / wall, 2), "pauses": n, "silence_pct": round(pct, 1),
                "pause_median_ms": med, "syl_per_s": round(SYL / d, 2),
                "wpm": round(SYL / d * 60 / 2.37)})
print(f"qwen-solid  {d:6.1f}s / {wall:5.1f}s ×{d/wall:.1f}  ~{results[-1]['wpm']} слов/мин  "
      f"пауз {n} ({pct:.1f}%, медиана {med} мс)", flush=True)

# B. cue by cue, butted together — same assembly as the F5 pipeline
segs, t0 = [], time.perf_counter()
for c in cues:
    w, sr2 = model.generate_voice_clone(text=c["text"], language="Russian",
                                        ref_audio=REF, ref_text=ref_text)
    segs.append(np.asarray(w[0], dtype=np.float32))
wall = time.perf_counter() - t0
y = np.concatenate(segs)
sf.write(f"{OUT}/qwen-joined.wav", y, sr2)
d, n, pct, med = stats(y, sr2)
results.append({"tag": "qwen-joined", "audio_s": round(d, 1), "wall_s": round(wall, 1),
                "rtf": round(d / wall, 2), "pauses": n, "silence_pct": round(pct, 1),
                "pause_median_ms": med, "syl_per_s": round(SYL / d, 2),
                "wpm": round(SYL / d * 60 / 2.37)})
print(f"qwen-joined {d:6.1f}s / {wall:5.1f}s ×{d/wall:.1f}  ~{results[-1]['wpm']} слов/мин  "
      f"пауз {n} ({pct:.1f}%, медиана {med} мс)", flush=True)

# C. the control phrase, for the one-breath check
PHRASE = "И вот причина ради которой этот вопрос и задают."
w, sr3 = model.generate_voice_clone(text=PHRASE, language="Russian",
                                    ref_audio=REF, ref_text=ref_text)
sf.write(f"{OUT}/qwen-phrase.wav", np.asarray(w[0], dtype=np.float32), sr3)

json.dump({"total_syllables": SYL, "sample_rate": int(sr), "results": results},
          open("results_qwen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
