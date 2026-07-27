import json,sys,warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel
d=json.load(open("results_english.json",encoding="utf-8"))
m=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
def hear(p, lang):
    segs,info=m.transcribe(p,language=lang,beam_size=5)
    return " ".join(s.text for s in segs).strip(), info.language, round(info.language_probability,2)
for e in d:
    print(f"\n=== {e['id']}")
    for lab in ("cyr","lat"):
        p=f"english/{e[lab]['file']}"
        ru,_,_ = hear(p,"ru")
        auto,lg,pr = hear(p,None)
        e[lab]["heard_ru"]=ru; e[lab]["heard_auto"]=auto; e[lab]["lang"]=lg; e[lab]["lang_p"]=pr
        print(f"  {lab}  как русский: «{ru[:70]}»")
        print(f"       свободно  ({lg} {pr:.2f}): «{auto[:70]}»")
json.dump(d,open("results_english.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
