"""Re-cut references on sentence boundaries and rank them by speaking rate.

The first pass cut clips by a stopwatch, which landed mid-sentence and produced a
garbled ref_text — F5 gets the reference transcript verbatim, so a wrong one hurts
every generated segment. Whisper's own segment boundaries are sentence boundaries,
so the clip and its transcript now agree exactly.
"""
import sys, json, subprocess, warnings, os
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
V = set("аеёиоуыэюя")
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
out = {}
for tag in ("dg", "nu", "kr"):
    src = f"lv/{tag}_long.mp3"
    if not os.path.exists(src):
        continue
    segs, _ = m.transcribe(src, language="ru", beam_size=5)
    segs = [s for s in segs if s.start > 40]          # skip the LibriVox disclaimer
    best = None
    for i in range(len(segs)):
        acc, j = 0.0, i
        while j < len(segs) and segs[j].end - segs[i].start < 13.5:
            j += 1
        if j == i:
            continue
        dur = segs[j - 1].end - segs[i].start
        if not (9.0 <= dur <= 13.5):
            continue
        text = " ".join(s.text.strip() for s in segs[i:j]).strip()
        syl = sum(1 for c in text.lower() if c in V)
        rate = syl / dur
        # prefer a clean sentence end and a slow, well-filled clip
        if not text or text[-1] not in ".!?":
            continue
        cand = (rate, segs[i].start, dur, text)
        if best is None or cand[0] < best[0]:
            best = cand
    if not best:
        print(f"{tag}: подходящего окна не нашлось"); continue
    rate, start, dur, text = best
    dst = f"lv/clean_{tag}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(dur),
                    "-i", src, "-ac", "1", "-ar", "24000", dst], check=True)
    out[dst] = text
    print(f"{tag}: {dur:.1f}с  {rate:.2f} слог/с  ~{rate*60/2.37:.0f} слов/мин")
    print(f"    «{text[:110]}»")
json.dump(out, open("lv/clean_texts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
