"""Read the whole dictionary aloud, ten terms per clip.

Every entry is a guess until someone hears it, and eighty separate files would be
unlistenable. Terms are batched into short clips with a carrier phrase around
each one, so a single pass through nine clips flags everything that needs fixing.
"""
import json, sys, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "lexicon"; os.makedirs(OUT, exist_ok=True)
REF = "lv/ref_t2.wav"
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

lex = json.load(open("pronunciation.json", encoding="utf-8"))["terms"]
items = [(k, v["say"]) for k, v in lex.items()]
BATCH = 10

print("загружаю Qwen3-TTS ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for bi in range(0, len(items), BATCH):
    chunk = items[bi:bi + BATCH]
    # each term inside a short frame so it is heard in speech, not as a list of labels
    text = " ".join(f"{say}." for _, say in chunk)
    torch.manual_seed(1234)
    w, sr = model.generate_voice_clone(text=text, language="Russian",
                                       ref_audio=REF, ref_text=ref_text)
    x = np.asarray(w[0], dtype=np.float32)
    name = f"batch{bi // BATCH:02d}.wav"
    sf.write(f"{OUT}/{name}", x, sr)
    rows.append({"batch": bi // BATCH, "file": name, "sec": round(len(x) / sr, 1),
                 "terms": [{"term": t, "say": s} for t, s in chunk], "text": text})
    print(f"  batch{bi // BATCH:02d}  {len(x)/sr:5.1f}s  " +
          ", ".join(s for _, s in chunk), flush=True)

json.dump(rows, open("results_lexicon.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
