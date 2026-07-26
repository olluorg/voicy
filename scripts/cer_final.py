import json,re,sys,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
S={s["id"]:s for s in json.load(open("samples.json",encoding="utf-8"))}
def norm(t):
    t=t.lower().replace("ё","е").replace("+","")
    return re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t)).strip()
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
out=[]
for r in json.load(open("results_final.json",encoding="utf-8")):
    segs,_=m.transcribe(f"final/{r['file']}",language="ru",beam_size=5)
    hyp=" ".join(s.text for s in segs)
    c=round(cer(norm(S[r["id"]]["audio"]),norm(hyp))*100,1)
    out.append({**r,"cer":c,"hyp":hyp.strip()})
    print(f"{r['id']}  speed {r['speed']}   CER {c:5.1f}%",flush=True)
json.dump(out,open("cer_final.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
