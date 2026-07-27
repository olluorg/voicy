import json,re,sys,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
from jiwer import cer
d=json.load(open("results_generics.json",encoding="utf-8"))
GOLD="дженерики говорят компилятору что лежит в коллекции"
norm=lambda t: re.sub(r"\s+"," ",re.sub(r"[^а-яa-z0-9 ]+"," ",t.lower().replace("ё","е"))).strip()
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
print(f"{'вариант':10} {'слово, что услышал Whisper':32} {'фраза CER':>10}  что услышал во фразе")
print("-"*108)
for e in d:
    segs,_=m.transcribe(f"generics/{e['word']['file']}",language="ru",beam_size=5)
    hw=" ".join(s.text for s in segs).strip()
    segs,_=m.transcribe(f"generics/{e['sent']['file']}",language="ru",beam_size=5)
    hs=" ".join(s.text for s in segs).strip()
    c=round(cer(norm(GOLD),norm(hs))*100,1) if e["tag"] not in ("form_sg","form_loc") else None
    e["heard_word"], e["heard_sent"], e["cer"]=hw,hs,c
    cs=f"{c:9.1f}%" if c is not None else "       — "
    print(f"{e['tag']:10} {hw[:32]:32} {cs}  «{hs[:52]}»")
json.dump(d,open("results_generics.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
