import json, time, wave, sys, pathlib, re, torch
sys.stdout.reconfigure(encoding="utf-8")
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
torch.set_num_threads(12)
m = torch.package.PackageImporter("v5_ru.pt").load_pickle("tts_models", "model")
SR, MAXLEN = 48000, 800
def chunks(t):
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", t):
        if len(cur)+len(s)+1 > MAXLEN and cur: parts.append(cur); cur = s
        else: cur = f"{cur} {s}".strip()
    if cur: parts.append(cur)
    return parts
res=[]
for spk in ("eugene","kseniya"):
    for s in json.load(open("samples.json",encoding="utf-8")):
        t0=time.perf_counter()
        a=torch.cat([m.apply_tts(text=c,speaker=spk,sample_rate=SR,put_accent=True,put_yo=True) for c in chunks(s["audio"])])
        wall=time.perf_counter()-t0
        p=OUT/f"{s['id']}__audio__silero5-{spk}.wav"
        with wave.open(str(p),"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((a*32767).to(torch.int16).numpy().tobytes())
        d=len(a)/SR
        res.append({"engine":"silero-v5","voice":spk,"id":s["id"],"variant":"audio","chars":len(s["audio"]),
                    "audio_s":round(d,2),"wall_s":round(wall,2),"rtf":round(d/wall,1),"file":p.name})
        print(f"{p.name:48} {d:6.1f}s / {wall:5.1f}s  x{d/wall:.0f}", flush=True)
json.dump(res,open("results_silero5.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
