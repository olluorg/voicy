"""Transcribe the public-domain reference clips so F5 gets exact ref_text."""
import sys, glob, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
out = {}
for p in sorted(glob.glob("lv/ref_*.wav")):
    segs, _ = m.transcribe(p, language="ru", beam_size=5)
    t = " ".join(s.text for s in segs).strip()
    out[p] = t
    print(f"{p}: {t}", flush=True)
json.dump(out, open("lv/ref_texts.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
