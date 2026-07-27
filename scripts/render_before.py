"""Re-render the spellings the pipeline rejected, so the fixes can be heard.

Three of these are my own typos, confirmed as correct by the broken checker
(«джей-ви-ти» for JWT). Two are what that checker produced when it tried to
«repair» a reading that was already right — a stretched vowel bolted onto a
perfectly good spelling.

The audio for all of them was overwritten by later runs, so the article had a
section about fixes with nothing to listen to.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "before"; os.makedirs(OUT, exist_ok=True)
CARRIER = "Здесь важно понять что такое {}."

CASES = [
    ("JWT_before",  "джей-ви-ти",       "моя опечатка: W прочитан как «ви»"),
    ("AOP_before",  "а-о-пи",           "моя опечатка: A прочитан как «а»"),
    ("URL_before",  "юар-эль",          "моя опечатка: слипшееся «юар»"),
    ("SQL_broken",  "ээс-ку-эль",       "«починка» сломанного измерителя — растянутая гласная"),
    ("CDL_broken",  "Кааунт Даун Лэтч", "то же самое на составном термине"),
]

rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

print("загружаю Qwen3-TTS ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for tag, say, note in CASES:
    torch.manual_seed(1234)
    w, sr = model.generate_voice_clone(text=CARRIER.format(say), language="Russian",
                                       ref_audio="lv/ref_t2.wav", ref_text=ref_text)
    x = np.asarray(w[0], dtype=np.float32)
    sf.write(f"{OUT}/{tag}.wav", x, sr)
    rows.append({"tag": tag, "say": say, "note": note, "sec": round(len(x) / sr, 2)})
    print(f"  {tag:14} «{say:18}» {len(x)/sr:5.2f}s   {note}", flush=True)

json.dump(rows, open("results_before.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
