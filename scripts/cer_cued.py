import json,re,sys,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
def norm(t):
    t=t.lower().replace("ё","е").replace("+","")
    return re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t)).strip()
doc=json.load(open("cues.json",encoding="utf-8"))
ref=" ".join(c["text"] for c in doc["cues"] if c["kind"]!="beat")
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
out={}
for f in ("jc-024__cued.wav","jc-024__flat.wav"):
    segs,_=m.transcribe(f"cued/{f}",language="ru",beam_size=5)
    hyp=" ".join(s.text for s in segs)
    c=round(cer(norm(ref),norm(hyp))*100,1)
    out[f]={"cer":c,"hyp":hyp.strip()}
    print(f"{f:24} CER {c:5.1f}%",flush=True)
json.dump(out,open("cer_cued.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
