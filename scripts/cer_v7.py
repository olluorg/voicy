import json,re,sys,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
doc=json.load(open("cues.json",encoding="utf-8"))
ref=" ".join(c["text"] for c in doc["cues"] if c["kind"]!="beat")
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е").replace("+",""))).strip()
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
d=json.load(open("results_v7.json",encoding="utf-8"))
for r in d["results"]:
    segs,_=m.transcribe(f"v7/{r['tag']}.wav",language="ru",beam_size=5)
    hyp=" ".join(s.text for s in segs)
    r["cer"]=round(cer(norm(ref),norm(hyp))*100,1)
    print(f"{r['tag']:14} ~{r['wpm']:3} слов/мин  пауз {r['pauses']:2} ({r['silence_pct']:4.1f}%)  CER {r['cer']:5.1f}%",flush=True)
json.dump(d,open("results_v7.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
