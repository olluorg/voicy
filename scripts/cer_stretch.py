import json,re,sys,glob,os,warnings,subprocess
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
doc=json.load(open("cues.json",encoding="utf-8"))
ref=" ".join(c["text"] for c in doc["cues"] if c["kind"]!="beat")
V=set("аеёиоуыэюя"); SYL=sum(1 for c in ref.lower() if c in V)
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е").replace("+",""))).strip()
dur=lambda p: float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True).stdout)
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
out=[]
for p in sorted(glob.glob("v2/stretch-*.wav")):
    segs,_=m.transcribe(p,language="ru",beam_size=5)
    hyp=" ".join(s.text for s in segs); d=dur(p)
    r={"tag":os.path.basename(p)[:-4],"audio_s":round(d,1),"syl_per_s":round(SYL/d,2),
       "wpm":round(SYL/d*60/2.37),"cer":round(cer(norm(ref),norm(hyp))*100,1)}
    out.append(r); print(f"{r['tag']:16} {r['syl_per_s']:.2f} слог/с  ~{r['wpm']} слов/мин   CER {r['cer']:5.1f}%",flush=True)
json.dump(out,open("results_stretch.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
