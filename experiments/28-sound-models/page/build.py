"""Scenes and chosen elements of every run → Ogg Opus beside the page, plus data.json.

    ~/src/MOSS-TTS/moss_soundeffect_v2/.venv/bin/python experiments/28-sound-models/page/build.py

Reads the runs from D: (/mnt/d/ml/voicy-sound), where the candidates and chosen
takes were left; the Opus files committed beside the page are the lasting copy.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).parent
RUNS = {
    "moss2": Path("/mnt/d/ml/voicy-sound/izba-auto5"),
    "sa3m": Path("/mnt/d/ml/voicy-sound/compare/sa3m"),
    "sa3s": Path("/mnt/d/ml/voicy-sound/compare/sa3s"),
    "unite": Path("/mnt/d/ml/voicy-sound/compare/unite"),
    "ezaudio": Path("/mnt/d/ml/voicy-sound/compare/ezaudio"),
}


def opus(src: Path, dst: Path, seconds_cap=None):
    """Opus takes 8–48 kHz only: Stable Audio's 44.1 kHz goes to 48."""
    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if seconds_cap:
        x = x[: int(seconds_cap * sr)]
    peak = np.abs(x).max() or 1.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    seconds = round(len(x) / sr, 1)
    if sr not in (8000, 12000, 16000, 24000, 48000):
        x = resample_poly(x, 48000, sr).astype(np.float32)
        sr = 48000
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, sr, format="OGG", subtype="OPUS", compression_level=0.8)
    return seconds


data = {}
for name, run in RUNS.items():
    plan = json.loads((run / "plan.json").read_text())
    verdict = json.loads((run / "judge.json").read_text())
    takes = sum(len(v["takes"]) for v in verdict.values())
    bad = sum(1 for v in verdict.values() for t in v["takes"] if t["why"])
    scene_len = opus(run / "scene.wav", HERE / "audio" / name / "scene.opus")
    elements = []
    for x in plan["beds"] + plan["events"]:
        v = verdict[x["id"]]
        t = next(t for t in v["takes"] if t["name"] == v["chosen"])
        secs = opus(run / "chosen" / f"{x['id']}.wav", HERE / "audio" / name / f"{x['id']}.opus", seconds_cap=20)
        elements.append({"id": x["id"], "role": "фон" if x in plan["beds"] else "событие",
                         "prompt": x["prompt"], "target": x["target"], "expect": x.get("expect", []),
                         "ok": not v["all_bad"], "heard": t["heard"], "why": t["why"],
                         "synth": v["chosen"].split("__")[1].startswith("dsp"), "seconds": secs})
    data[name] = {"takes": takes, "bad": bad, "failed": [k for k, v in verdict.items() if v["all_bad"]],
                  "scene_seconds": scene_len, "elements": elements}

(HERE / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
files = list((HERE / "audio").rglob("*.opus"))
print(f"{sum(f.stat().st_size for f in files) / 1e6:.1f} MB in {len(files)} files")
