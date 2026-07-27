"""Rebuild the dictionary from the corrected criterion, regenerating only what needs it.

The first full pass collected a transcript for every term, and the bug was in how
those transcripts were scored, not in how they were produced. So the pass is not
repeated: entries are re-scored from the data already on disk, and the synthesiser
is only called for the handful that still fail.
"""
import json, os, re, sys, tempfile, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

sys.path.insert(0, ".")
from autolex import recognised, candidates

THRESH = 0.62
prev = {r["term"]: r for r in json.load(open("results_autolex.json", encoding="utf-8"))}
lex = json.load(open("pronunciation.json", encoding="utf-8"))
terms = lex["terms"]

passed, failing = {}, []
for term, meta in terms.items():
    r = prev.get(term)
    if not r:
        failing.append(term); continue
    t = r["tries"][0]
    sim = recognised(term, r["proposed"], t["heard"])
    if sim >= THRESH:
        passed[term] = {"say": r["proposed"], "sim": round(sim, 2), "heard": t["heard"]}
    else:
        failing.append(term)

print(f"пересчёт по исправленному критерию: прошло {len(passed)}, осталось {len(failing)}")
print("догенерирую только:", ", ".join(failing), flush=True)

if failing:
    import torch
    from qwen_tts import Qwen3TTSModel
    from faster_whisper import WhisperModel

    qwen = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map="cuda:0", dtype=torch.bfloat16)
    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
    ref_text = next(v for k, v in rt.items() if "ref_t2" in k)
    CARRIER = "Здесь важно понять что такое {}."
    os.makedirs("autolex", exist_ok=True)

    for term in failing:
        say = terms[term]["say"]
        best = None
        for cand in candidates(say):
            torch.manual_seed(1234)
            w, sr = qwen.generate_voice_clone(text=CARRIER.format(cand), language="Russian",
                                              ref_audio="lv/ref_t2.wav", ref_text=ref_text)
            x = np.asarray(w[0], dtype=np.float32)
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "a.wav"); sf.write(p, x, sr)
                segs, _ = asr.transcribe(p, language="ru", beam_size=5)
                heard = " ".join(s.text for s in segs).strip()
            sim = recognised(term, cand, heard)
            if best is None or sim > best[1]:
                best = (cand, sim, heard, x, sr)
            if sim >= THRESH:
                break
        cand, sim, heard, x, sr = best
        sf.write(f"autolex/{re.sub(r'[^A-Za-z0-9]', '_', term)}.wav", x, sr)
        passed[term] = {"say": cand, "sim": round(sim, 2), "heard": heard,
                        "note": "best effort" if sim < THRESH else None}
        flag = " " if sim >= THRESH else "!"
        print(f" {flag} {term:20} → {cand:22} сходство {sim:.2f}  «{heard[:44]}»", flush=True)

for term, meta in terms.items():
    got = passed.get(term)
    if not got:
        continue
    meta["say"] = got["say"]
    meta["verified"] = got.get("note") is None
    meta["heard"] = got["heard"]
    meta["similarity"] = got["sim"]

ok = sum(1 for m in terms.values() if m.get("verified"))
lex["note"] = ("Собран автоматически. Разборчивость проверяется обратным распознаванием: "
               "термин засчитан, если ASR услышал его либо в кириллическом написании, либо "
               "в исходном английском. Человек в цикле не участвует.")
lex["stats"] = {"terms": len(terms), "verified": ok, "best_effort": len(terms) - ok}
json.dump(lex, open("pronunciation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nсловарь собран: {ok} из {len(terms)} подтверждено автоматически")
