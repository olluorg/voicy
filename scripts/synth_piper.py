"""Piper TTS sample generation + speed measurement."""
import json, time, wave, sys, pathlib

sys.stdout.reconfigure(encoding="utf-8")
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
samples = json.load(open("samples.json", encoding="utf-8"))

from piper import PiperVoice

VOICES = ["ru_RU-irina-medium", "ru_RU-dmitri-medium", "ru_RU-ruslan-medium"]
results = []

for vname in VOICES:
    voice = PiperVoice.load(f"voices/{vname}.onnx", config_path=f"voices/{vname}.onnx.json")
    for s in samples:
        for variant in ("raw", "audio"):
            text = s[variant]
            path = OUT / f"{s['id']}__{variant}__piper-{vname.split('-')[1]}.wav"
            t0 = time.perf_counter()
            with wave.open(str(path), "wb") as wf:
                voice.synthesize_wav(text, wf)
            wall = time.perf_counter() - t0
            with wave.open(str(path)) as wf:
                dur = wf.getnframes() / wf.getframerate()
            results.append({
                "engine": "piper", "voice": vname, "id": s["id"], "variant": variant,
                "chars": len(text), "audio_s": round(dur, 2),
                "wall_s": round(wall, 2), "rtf": round(dur / wall, 1),
                "file": path.name,
            })
            print(f"{path.name:52} {dur:6.1f}s аудио / {wall:5.1f}s синтез  x{dur/wall:.0f}", flush=True)

json.dump(results, open("results_piper.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
