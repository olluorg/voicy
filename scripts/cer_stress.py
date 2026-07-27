import json,re,sys,glob,os,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
d=json.load(open("results_stress.json",encoding="utf-8"))
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е"))).strip()
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
out=[]
for r in d:
    gold=r["text"]
    row={"text":gold}
    for lab in ("plain","plus","acute","caps"):
        p=f"stress/case{r['case']}__{lab}.wav"
        segs,_=m.transcribe(p,language="ru",beam_size=5)
        hyp=" ".join(s.text for s in segs)
        row[lab]={"cer":round(cer(norm(gold),norm(hyp))*100,1),"hyp":hyp.strip()}
    out.append(row)
    print(f"\n{gold}")
    for lab in ("plain","plus","acute","caps"):
        print(f"  {lab:6} CER {row[lab]['cer']:5.1f}%  «{row[lab]['hyp'][:80]}»")
json.dump(out,open("cer_stress.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
