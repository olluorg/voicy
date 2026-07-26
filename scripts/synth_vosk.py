import json, time, wave, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")
from vosk_tts import Model, Synth
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
model = Model(model_name="vosk-model-tts-ru-0.9-multi")
synth = Synth(model)
res=[]
for spk in (0, 2, 4):
    for s in json.load(open("samples.json", encoding="utf-8")):
        p = OUT / f"{s['id']}__audio__vosk-s{spk}.wav"
        t0=time.perf_counter()
        synth.synth(s["audio"], str(p), speaker_id=spk)
        wall=time.perf_counter()-t0
        with wave.open(str(p)) as w: d=w.getnframes()/w.getframerate()
        res.append({"engine":"vosk","voice":f"speaker {spk}","id":s["id"],"variant":"audio",
                    "chars":len(s["audio"]),"audio_s":round(d,2),"wall_s":round(wall,2),
                    "rtf":round(d/wall,1),"file":p.name})
        print(f"{p.name:44} {d:6.1f}s / {wall:5.1f}s  x{d/wall:.0f}", flush=True)
json.dump(res,open("results_vosk.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
