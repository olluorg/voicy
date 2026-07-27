"""Does Qwen3-TTS respond to stress marking, and in which notation?

Qwen has no text frontend — the string goes straight into an end-to-end LM — so
whether stress marks work at all is an empirical question. Three notations are
compared against plain text on the same phrases:

  +  before the stressed vowel   — the RUAccent convention, used by the F5 pipeline
  ́   U+0301 after the vowel      — standard Russian typography (сто́ит)
  CAPS on the stressed syllable  — a crude fallback some models pick up

If a notation is ignored the waveform comes back byte-identical; that is the first
thing measured, before anything is judged by ear.
"""
import json, sys, os, warnings, hashlib
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

OUT = "stress"; os.makedirs(OUT, exist_ok=True)
REF = "lv/ref_t2.wav"
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)

# (plain, +notation, U+0301 notation, CAPS) — stress that a reader would get wrong
CASES = [
    ("Этот вопрос стоит разобрать.",
     "Этот вопрос сто+ит разобрать.",
     "Этот вопрос сто́ит разобрать.",
     "Этот вопрос СТОит разобрать."),
    ("Компилятор проверяет типы.",
     "Компил+ятор провер+яет т+ипы.",
     "Компиля́тор проверя́ет ти́пы.",
     "КомпиЛЯтор провеРЯет ТИпы."),
    ("Большая часть проверок уже пройдена.",
     "Больш+ая часть провер+ок уж+е пройдена.",
     "Больша́я часть прове́рок уже́ пройдена.",
     "БольШАя часть проВЕрок уЖЕ пройдена."),
    ("Список параметризован типом.",
     "Спис+ок параметриз+ован т+ипом.",
     "Спи́сок параметризо́ван ти́пом.",
     "СПИсок параметриЗОван ТИпом."),
    ("Типобезопасность рухнула бы.",
     "Типобезоп+асность р+ухнула бы.",
     "Типобезопа́сность ру́хнула бы.",
     "ТипобезоПАСность РУХнула бы."),
]
LABELS = ["plain", "plus", "acute", "caps"]

print("загружаю Qwen3-TTS 1.7B Base ...", flush=True)
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                      device_map="cuda:0", dtype=torch.bfloat16)

rows = []
for i, variants in enumerate(CASES):
    digests = {}
    for lab, text in zip(LABELS, variants):
        torch.manual_seed(1234)
        w, sr = model.generate_voice_clone(text=text, language="Russian",
                                           ref_audio=REF, ref_text=ref_text)
        x = np.asarray(w[0], dtype=np.float32)
        name = f"case{i}__{lab}.wav"
        sf.write(f"{OUT}/{name}", x, sr)
        digests[lab] = (hashlib.sha256(x.tobytes()).hexdigest()[:12], round(len(x) / sr, 2))
        print(f"  case{i} {lab:6} {len(x)/sr:5.2f}s  {digests[lab][0]}", flush=True)
    same = {lab: (digests[lab][0] == digests["plain"][0]) for lab in LABELS[1:]}
    rows.append({"case": i, "text": variants[0], "digests": digests, "identical_to_plain": same})
    print(f"  → совпадает с plain: " +
          ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in same.items()), flush=True)

json.dump(rows, open("results_stress.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
