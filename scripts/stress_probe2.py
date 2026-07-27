"""Notations that do not corrupt the text: do any of them move the stress?

The first probe ruled out the two standard notations — `+` is read aloud as
"плюс", and the U+0301 diacritic garbles the word. What is left are tricks that
survive the tokenizer intact, so the only question is whether they have any
effect on where the stress lands. That part is decided by ear; the script's job
is to prove the text still comes out whole (CER) and to lay the pairs side by side.
"""
import json, sys, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "stress2"; os.makedirs(OUT, exist_ok=True)
REF = "lv/ref_t2.wav"
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

# label -> text. Same sentence, five ways of hinting at the stressed syllable.
CASES = {
    "стоит (сто-И-т)": {
        "plain":  "Этот вопрос стоит разобрать.",
        "caps":   "Этот вопрос стоИт разобрать.",
        "apos":   "Этот вопрос стои'т разобрать.",
        "hyphen": "Этот вопрос сто-ит разобрать.",
        "rewrite":"Этот вопрос надо разобрать.",
    },
    "большая (боль-ША-я)": {
        "plain":  "Большая часть проверок уже пройдена.",
        "caps":   "БольшАя часть проверок уже пройдена.",
        "apos":   "Больша'я часть проверок уже пройдена.",
        "hyphen": "Боль-шая часть проверок уже пройдена.",
        "rewrite":"Основная часть проверок уже пройдена.",
    },
    "дженерики (дже-НЕ-ри-ки)": {
        "plain":  "Дженерики говорят компилятору что лежит в коллекции.",
        "caps":   "ДженЕрики говорят компилятору что лежит в коллекции.",
        "apos":   "Джене'рики говорят компилятору что лежит в коллекции.",
        "hyphen": "Дже-нерики говорят компилятору что лежит в коллекции.",
        "rewrite":"Обобщения говорят компилятору что лежит в коллекции.",
    },
    "параметризован": {
        "plain":  "Список параметризован типом.",
        "caps":   "Список параметризОван типом.",
        "apos":   "Список параметризо'ван типом.",
        "hyphen": "Список параметри-зован типом.",
        "rewrite":"У списка задан тип элемента.",
    },
}

print("загружаю Qwen3-TTS ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for ci, (label, variants) in enumerate(CASES.items()):
    print(f"\n{label}", flush=True)
    entry = {"case": ci, "label": label, "variants": {}}
    for lab, text in variants.items():
        torch.manual_seed(1234)
        w, sr = model.generate_voice_clone(text=text, language="Russian",
                                           ref_audio=REF, ref_text=ref_text)
        x = np.asarray(w[0], dtype=np.float32)
        name = f"case{ci}__{lab}.wav"
        sf.write(f"{OUT}/{name}", x, sr)
        entry["variants"][lab] = {"text": text, "sec": round(len(x) / sr, 2), "file": name}
        print(f"  {lab:8} {len(x)/sr:5.2f}s  {text[:60]}", flush=True)
    rows.append(entry)

json.dump(rows, open("results_stress2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
