"""UNITE-AUDIO (Apache-2.0 weights, 118M flow model; its runner reads
TangoFlux-style data, so the training set is probably research-only) as a
generator of takes.

    HF_HOME=/mnt/d/ml/huggingface ~/src/venvs/unite/bin/python scripts/sound_unite.py OUT_DIR items.json

Same contract as sound_moss.py. Their runner loads the model per caption; it
is small, so each take is one call of it.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path.home() / "src" / "Unite-Audio"
MAX_SECONDS = 10                                    # на чём обучена; длинный фон зациклит сведение

out = Path(sys.argv[1]).resolve()
items = json.loads(Path(sys.argv[2]).read_text())
out.mkdir(parents=True, exist_ok=True)
for it in items:
    if (out / f"{it['name']}.wav").exists():
        continue
    t = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        subprocess.run([sys.executable, str(REPO / "inference" / "infer.py"), it["prompt"],
                        "--seconds", str(min(float(it["seconds"]), MAX_SECONDS)),
                        "--seed", str(int(it.get("seed", 0))), "--device", "cuda", "--output-dir", tmp],
                       check=True, cwd=REPO / "inference", stdout=subprocess.DEVNULL)
        wavs = sorted(Path(tmp).rglob("*.wav"))
        if not wavs:
            raise SystemExit(f"UNITE-AUDIO wrote no wav for {it['name']}")
        shutil.move(str(wavs[0]), out / f"{it['name']}.wav")
    print(f"{it['name']}: {min(it['seconds'], MAX_SECONDS)} s in {time.perf_counter() - t:.1f} s", flush=True)
