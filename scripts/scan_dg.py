"""Same reader, different passage.

Timbre and phrasing are separable: one narrator reads some passages in long
legato lines and others in short chopped ones. The previous clip from this
recording happened to be a chopped stretch — 13.3 breaks per minute. This scans
the whole recording for a window that keeps the voice but loses the stops.
"""
import sys, json, subprocess, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

SRC = "lv/dg_xl.mp3"
V = set("аеёиоуыэюя")
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

print("распознаю запись...", flush=True)
segs, _ = m.transcribe(SRC, language="ru", beam_size=5)
segs = [s for s in segs if s.start > 40]
print(f"  сегментов: {len(segs)}, до {segs[-1].end:.0f} с", flush=True)

wav = "lv/_dg_full.wav"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", SRC, "-ac", "1", "-ar", "24000", wav], check=True)
a, sr = sf.read(wav, dtype="float32", always_2d=True)
a = a[:, 0]
win = int(sr * 0.005)
env = np.array([np.abs(a[i:i + win]).max() for i in range(0, len(a) - win, win)])
thr = np.percentile(env, 95) * 0.02


def window_stats(t0, t1):
    i0, i1 = int(t0 * sr / win), int(t1 * sr / win)
    e = env[i0:i1]
    if len(e) < 10:
        return None
    sil = e < thr
    runs, i = [], 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) >= 24:                 # >=120 ms is a break
                runs.append((j - i) * 5)
            i = j
        else:
            i += 1
    d = t1 - t0
    return len(runs) / d * 60, sum(runs) / 10 / d, (int(np.median(runs)) if runs else 0)


cands = []
for i in range(len(segs)):
    j = i
    while j < len(segs) and segs[j].end - segs[i].start < 13.5:
        j += 1
    if j == i:
        continue
    t0, t1 = segs[i].start, segs[j - 1].end
    dur = t1 - t0
    if not (9.0 <= dur <= 13.5):
        continue
    text = " ".join(s.text.strip() for s in segs[i:j]).strip()
    if not text or text[-1] not in ".!?" or text[0] in "—-–«":
        continue
    st = window_stats(t0, t1)
    if st is None:
        continue
    ppm, pct, med = st
    syl = sum(1 for c in text.lower() if c in V)
    rate = syl / dur
    wpm = rate * 60 / 2.37
    if not (80 <= wpm <= 120):
        continue
    cands.append({"start": round(t0, 2), "dur": round(dur, 2), "wpm": round(wpm),
                  "ppm": round(ppm, 1), "sil_pct": round(pct, 1), "med": med, "text": text})

cands.sort(key=lambda c: (c["ppm"], -c["wpm"]))
print(f"\nкандидатов: {len(cands)}. Самые слитные:\n")
print(f"{'#':>2} {'старт':>7} {'сек':>5} {'слов/мин':>9} {'пауз/мин':>9} {'тишины':>7}")
for k, c in enumerate(cands[:8]):
    print(f"{k:2} {c['start']:7.1f} {c['dur']:5.1f} {c['wpm']:9} {c['ppm']:9.1f} {c['sil_pct']:6.1f}%")
    print(f"     «{c['text'][:96]}»")

picked, texts = [], {}
for k, c in enumerate(cands[:3]):
    dst = f"lv/ref_t{k}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(c["start"]), "-t", str(c["dur"]),
                    "-i", SRC, "-ac", "1", "-ar", "24000", dst], check=True)
    texts[dst] = c["text"]
    picked.append({**c, "file": dst})

json.dump(texts, open("lv/turgenev_texts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(picked, open("results_scan_dg.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.remove(wav)
print(f"\nнарезано: {[p['file'] for p in picked]}")
