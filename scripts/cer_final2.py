import json,re,sys,warnings,subprocess
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
doc=json.load(open("cues.json",encoding="utf-8"))
ref=" ".join(c["text"] for c in doc["cues"] if c["kind"]!="beat")
V=set("аеёиоуыэюя"); SYL=sum(1 for c in ref.lower() if c in V)
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е").replace("+",""))).strip()
d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0","v2/RECOMMENDED.wav"],capture_output=True,text=True).stdout)
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
segs,_=m.transcribe("v2/RECOMMENDED.wav",language="ru",beam_size=5)
hyp=" ".join(s.text for s in segs)
r={"tag":"RECOMMENDED (паузы ×3 + растяжение 0.92)","audio_s":round(d,1),
   "syl_per_s":round(SYL/d,2),"wpm":round(SYL/d*60/2.37),"cer":round(cer(norm(ref),norm(hyp))*100,1)}
print(f"{r['tag']}\n  {r['audio_s']}s  {r['syl_per_s']} слог/с  ~{r['wpm']} слов/мин  CER {r['cer']}%")
json.dump(r,open("results_recommended.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
