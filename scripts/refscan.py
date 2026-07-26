"""Rank reference candidates by rate AND phrasing, not by rate alone.

The Turgenev clip gave the right speaking rate but chopped phrases: the model
copies the reader's habit of stopping, not just the reader's speed. So a
reference now gets two measurements — how fast it is, and how often it breaks —
and the one we want is slow *and* legato.
"""
import sys, json, glob, os, subprocess, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

V = set("аеёиоуыэюя")
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def pause_stats(path):
    """Pauses per minute and silence share inside a clip."""
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a[:, 0]
    win = int(sr * 0.005)
    env = np.array([np.abs(a[i:i + win]).max() for i in range(0, len(a) - win, win)])
    sil = env < env.max() * 0.02
    runs, i = [], 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) >= 24:            # >=120 ms counts as a break
                runs.append((j - i) * 5)
            i = j
        else:
            i += 1
    d = len(a) / sr
    return len(runs) / d * 60, sum(runs) / 10 / d, (int(np.median(runs)) if runs else 0)


rows, texts = [], {}
for src in sorted(glob.glob("lv/*_long.mp3")):
    tag = os.path.basename(src).replace("_long.mp3", "")
    segs, _ = m.transcribe(src, language="ru", beam_size=5)
    segs = [s for s in segs if s.start > 40]
    best = None
    for i in range(len(segs)):
        j = i
        while j < len(segs) and segs[j].end - segs[i].start < 13.5:
            j += 1
        if j == i:
            continue
        dur = segs[j - 1].end - segs[i].start
        if not (9.0 <= dur <= 13.5):
            continue
        text = " ".join(s.text.strip() for s in segs[i:j]).strip()
        if not text or text[-1] not in ".!?" or text[0] in "—-–":
            continue
        syl = sum(1 for c in text.lower() if c in V)
        rate = syl / dur
        if not (2.8 <= rate <= 4.6):     # only slow-to-moderate candidates
            continue
        tmp = f"lv/_probe_{tag}.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(segs[i].start),
                        "-t", str(dur), "-i", src, "-ac", "1", "-ar", "24000", tmp], check=True)
        ppm, pct, med = pause_stats(tmp)
        # want slow but legato: penalise frequent breaks
        score = rate + ppm / 12
        cand = (score, rate, ppm, pct, med, segs[i].start, dur, text)
        if best is None or cand[0] < best[0]:
            best = cand
    if not best:
        print(f"{tag}: кандидатов нет")
        continue
    score, rate, ppm, pct, med, start, dur, text = best
    dst = f"lv/ref_s{tag}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(dur),
                    "-i", src, "-ac", "1", "-ar", "24000", dst], check=True)
    texts[dst] = text
    rows.append((tag, dur, rate, rate * 60 / 2.37, ppm, pct, med))

for f in glob.glob("lv/_probe_*.wav"):
    os.remove(f)

print(f"\n{'чтец':6} {'сек':>5} {'слог/с':>7} {'слов/мин':>9} {'пауз/мин':>9} {'тишины':>7} {'медиана':>8}")
print("-" * 60)
for tag, dur, rate, wpm, ppm, pct, med in sorted(rows, key=lambda r: r[4]):
    print(f"{tag:6} {dur:5.1f} {rate:7.2f} {wpm:9.0f} {ppm:9.1f} {pct:6.1f}% {med:6}мс")

json.dump(texts, open("lv/slow_texts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nтексты:")
for k, v in texts.items():
    print(f"  {os.path.basename(k)}: «{v[:90]}»")
