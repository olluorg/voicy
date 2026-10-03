"""Stable Audio 3 (Medium or Small-SFX; Stability AI Community License) through
sa3.cpp — the GGML port, its Windows CUDA build run from WSL — as a generator
of takes.

    SOUND_GEN=sa3m|sa3s python3 scripts/sound_sa3.py OUT_DIR items.json

Same contract as sound_moss.py. sa3.cpp is in the llama.cpp family voicy
already runs, has CUDA, Vulkan, Metal and CPU backends and needs no Flash
Attention; the Windows build is used because WSL here has no CUDA toolkit to
build the Linux one. Prompts get "TrackType: SFX," up front, as the model's own
prompting guide advises for sound effects (experiments/28: without it the
Small model's sounds were "terrible"). Takes are generated at the model's
default sampling: no CFG, so no negative prompt.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SA3_DIR", "/mnt/d/ml/sa3cpp"))
MODEL = {"sa3m": "medium", "sa3s": "small-sfx"}[os.environ.get("SOUND_GEN", "sa3m")]
MAX_SECONDS = 120                                    # больше за раз не делает ни одна из двух


def win(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


out = Path(sys.argv[1]).resolve()
items = json.loads(Path(sys.argv[2]).read_text())
out.mkdir(parents=True, exist_ok=True)
for it in items:
    dst = out / f"{it['name']}.wav"
    if dst.exists():
        continue
    t = time.perf_counter()
    subprocess.run([str(ROOT / "bin" / "sa3-generate.exe"), "--model", MODEL, "--encoding", "f16",
                    "--models-dir", win(ROOT / "models"),
                    "--prompt", "TrackType: SFX, " + it["prompt"],
                    "--duration", str(min(float(it["seconds"]), MAX_SECONDS)),
                    "--seed", str(int(it.get("seed", 0))), "--out", win(dst)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"{it['name']}: {it['seconds']} s in {time.perf_counter() - t:.1f} s", flush=True)
