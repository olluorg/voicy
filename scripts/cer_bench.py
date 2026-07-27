import json,re,sys,glob,os,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
doc=json.load(open("cues_v3.json",encoding="utf-8"))
ref=" ".join(c["text"] for c in doc["cues"] if "text" in c)
PHRASE="И вот причина ради которой этот вопрос и задают."
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е").replace("+",""))).strip()
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
def score(p, gold):
    segs,_=m.transcribe(p,language="ru",beam_size=5)
    hyp=" ".join(s.text for s in segs)          # генератор, читаем ровно один раз
    return round(cer(norm(gold), norm(hyp))*100,1), hyp
out={}
for p in sorted(glob.glob("qwen/*.wav"))+sorted(glob.glob("bench/*.wav")):
    b=os.path.basename(p)[:-4]
    gold=PHRASE if "phrase" in b else ref
    c,hyp=score(p,gold)
    out[b]={"cer":c,"hyp":hyp[:160]}
    print(f"{b:26} CER {c:5.1f}%",flush=True)
json.dump(out,open("cer_bench.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
