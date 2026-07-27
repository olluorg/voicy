"""English terms written in English, inside Russian sentences.

Qwen3-TTS covers ten languages, so a Latin-script term is not foreign input — it
is another language the model knows. The question is what it does with one
embedded in a Russian phrase: switch to English pronunciation for that word, read
it with Russian phonetics, or mangle it.

The first probe transcribed with `language="ru"` forced, which maps whatever was
said onto Cyrillic and can invent a Russian word that was never spoken. Here each
render is transcribed twice — forced Russian and free language detection — so the
two readings can disagree and give the answer away.
"""
import json, sys, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "english"; os.makedirs(OUT, exist_ok=True)
REF = "lv/ref_t2.wav"
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

# (id, кириллицей, латиницей) — оба варианта одной и той же фразы
CASES = [
    ("generics",
     "Дженерики говорят компилятору что лежит в коллекции.",
     "Generics говорят компилятору что лежит в коллекции."),
    ("wildcard",
     "Вайлдкард нужен когда из коллекции только читают.",
     "Wildcard нужен когда из коллекции только читают."),
    ("integer",
     "Положить туда Интеджер компилятор не даст.",
     "Положить туда Integer компилятор не даст."),
    ("erasure",
     "Дальше спросят что такое стирание типов.",
     "Дальше спросят что такое type erasure."),
    ("mixed",
     "Дженерики это про типобезопасность а вайлдкард про направление данных.",
     "Generics это про типобезопасность а wildcard про направление данных."),
]

print("загружаю Qwen3-TTS 1.7B Base ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for cid, ru, en in CASES:
    entry = {"id": cid}
    for lab, text in (("cyr", ru), ("lat", en)):
        torch.manual_seed(1234)
        w, sr = model.generate_voice_clone(text=text, language="Russian",
                                           ref_audio=REF, ref_text=ref_text)
        x = np.asarray(w[0], dtype=np.float32)
        name = f"{cid}__{lab}.wav"
        sf.write(f"{OUT}/{name}", x, sr)
        entry[lab] = {"text": text, "sec": round(len(x) / sr, 2), "file": name}
        print(f"  {cid:9} {lab}  {len(x)/sr:5.2f}s  {text[:56]}", flush=True)
    rows.append(entry)

json.dump(rows, open("results_english.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
