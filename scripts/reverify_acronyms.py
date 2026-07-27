"""Re-derive every acronym by rule and verify the derived spelling.

Hand-written letter spellings carried my typos — JWT as «джей-ви-ти» instead of
«джей-дабл-ю-ти» — and the checker confirmed them, because it verifies that the
audio matches the spelling, not that the spelling matches the term. Deriving the
spelling removes that whole class of error; only genuinely idiomatic readings
(SQL, CGLIB) stay as named exceptions.
"""
import json, os, re, sys, tempfile, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
sys.path.insert(0, ".")
from autolex import spell_acronym, recognised, candidates
from qwen_tts import Qwen3TTSModel
from faster_whisper import WhisperModel

lex = json.load(open("pronunciation.json", encoding="utf-8"))
terms = lex["terms"]
acr = [t for t in terms if t.isupper() and len(t) <= 5] + ["NoSQL"]

qwen = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                     device_map="cuda:0", dtype=torch.bfloat16)
asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_text = next(v for k, v in rt.items() if "ref_t2" in k)
CARRIER = "Здесь важно понять что такое {}."
os.makedirs("autolex", exist_ok=True)

changed = 0
for t in acr:
    if t not in terms:
        continue
    want = spell_acronym(t)
    if want == terms[t]["say"]:
        continue
    torch.manual_seed(1234)
    w, sr = qwen.generate_voice_clone(text=CARRIER.format(want), language="Russian",
                                      ref_audio="lv/ref_t2.wav", ref_text=ref_text)
    x = np.asarray(w[0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav"); sf.write(p, x, sr)
        segs, _ = asr.transcribe(p, language="ru", beam_size=5)
        heard = " ".join(s.text for s in segs).strip()
    sim = recognised(t, want, heard)
    print(f"  {t:6} «{terms[t]['say']}» → «{want}»  сходство {sim:.2f}  «{heard[:44]}»", flush=True)
    if sim >= 0.62:
        sf.write(f"autolex/{re.sub(r'[^A-Za-z0-9]', '_', t)}.wav", x, sr)
        terms[t].update({"say": want, "heard": heard, "similarity": round(sim, 2),
                         "verified": True, "derived": True})
        changed += 1

lex["policy"]["acronyms"] = ("Побуквенное чтение выводится из самого термина правилом; "
                            "руками задан только список идиоматических исключений "
                            "(джейсон, рест, эс-ку-эль, си-джи-либ).")
json.dump(lex, open("pronunciation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nпереведено на правило: {changed}")
