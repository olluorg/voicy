"""Forcing the stress on one borrowed term that has no same-register synonym.

«дженерики» is what a Russian backend developer actually says, so replacing it
with «обобщения» fixes the audio by damaging the vocabulary. The model puts the
stress on the third syllable; the target is the first, as in English GENerics.

Each spelling is rendered twice: alone, where the stress is unmistakable, and in
a sentence, where it has to survive context. CER guards against a spelling that
fixes the stress by breaking the word.
"""
import json, sys, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "generics"; os.makedirs(OUT, exist_ok=True)
REF = "lv/ref_t2.wav"
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

SENT = "{} говорят компилятору что лежит в коллекции."

VARIANTS = [
    ("plain",     "Дженерики",   "как сейчас"),
    ("latin",     "Generics",    "латиницей — английское ударение на первый слог"),
    ("caps_1",    "ДЖЕнерики",   "ЗАГЛАВНЫЕ на первом слоге — цель"),
    ("caps_2",    "ДженЕрики",   "ЗАГЛАВНЫЕ на втором — контроль, двигает ли КАПС вообще"),
    ("double",    "Джеенерики",  "удвоенная гласная притягивает ударение"),
    ("form_sg",   "Дженерик",    "другая словоформа"),
    ("form_loc",  "В дженериках", "другая словоформа"),
]

print("загружаю Qwen3-TTS 1.7B Base ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for tag, word, note in VARIANTS:
    entry = {"tag": tag, "word": word, "note": note}
    for mode, text in (("word", word + "."), ("sent", SENT.format(word))):
        torch.manual_seed(1234)
        w, sr = model.generate_voice_clone(text=text, language="Russian",
                                           ref_audio=REF, ref_text=ref_text)
        x = np.asarray(w[0], dtype=np.float32)
        name = f"{tag}__{mode}.wav"
        sf.write(f"{OUT}/{name}", x, sr)
        entry[mode] = {"text": text, "sec": round(len(x) / sr, 2), "file": name}
    rows.append(entry)
    print(f"  {tag:9} {word:14} слово {entry['word']['sec']:4.2f}s  "
          f"фраза {entry['sent']['sec']:4.2f}s   {note}", flush=True)

json.dump(rows, open("results_generics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
