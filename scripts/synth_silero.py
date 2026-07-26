"""Silero v4 ru sample generation + speed measurement."""
import json, time, wave, sys, pathlib, urllib.request, re
import torch

sys.stdout.reconfigure(encoding="utf-8")
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
MODEL = pathlib.Path("v4_ru.pt")

if not MODEL.exists():
    print("downloading v4_ru.pt ...", flush=True)
    urllib.request.urlretrieve("https://models.silero.ai/models/tts/ru/v4_ru.pt", MODEL)

device = torch.device("cpu")
torch.set_num_threads(12)
model = torch.package.PackageImporter(str(MODEL)).load_pickle("tts_models", "model")
model.to(device)

SR = 48000
SPEAKERS = ["xenia", "eugene", "baya"]
MAXLEN = 800


def chunks(text: str):
    """Silero caps input length; split on sentence boundaries."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(cur) + len(sent) + 1 > MAXLEN and cur:
            parts.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


samples = json.load(open("samples.json", encoding="utf-8"))
results = []

for spk in SPEAKERS:
    for s in samples:
        for variant in ("raw", "audio"):
            text = s[variant]
            t0 = time.perf_counter()
            audio = torch.cat([
                model.apply_tts(text=c, speaker=spk, sample_rate=SR, put_accent=True, put_yo=True)
                for c in chunks(text)
            ])
            wall = time.perf_counter() - t0
            path = OUT / f"{s['id']}__{variant}__silero-{spk}.wav"
            pcm = (audio * 32767).to(torch.int16).numpy().tobytes()
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm)
            dur = len(audio) / SR
            results.append({
                "engine": "silero", "voice": spk, "id": s["id"], "variant": variant,
                "chars": len(text), "audio_s": round(dur, 2),
                "wall_s": round(wall, 2), "rtf": round(dur / wall, 1),
                "file": path.name,
            })
            print(f"{path.name:52} {dur:6.1f}s аудио / {wall:5.1f}s синтез  x{dur/wall:.0f}", flush=True)

json.dump(results, open("results_silero.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
