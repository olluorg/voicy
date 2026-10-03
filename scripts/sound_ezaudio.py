"""EzAudio (MIT weights; trained on AudioCaps/WavCaps — research-only data,
so its commercial standing is doubtful) as a generator of takes.

    HF_HOME=/mnt/d/ml/huggingface ~/src/venvs/ezaudio/bin/python scripts/sound_ezaudio.py OUT_DIR items.json

Same contract as sound_moss.py. 24 kHz mono, at most 10 s a take (what it was
trained on); a longer bed is looped by the mixer anyway. No negative prompt.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

REPO = Path.home() / "src" / "EzAudio"
MAX_SECONDS = 10

out = Path(sys.argv[1]).resolve()
items = json.loads(Path(sys.argv[2]).read_text())
out.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)                                      # их конфиги заданы путями от корня репозитория
sys.path.insert(0, str(REPO))
import api.ezaudio as ez  # noqa: E402
from api.ezaudio import EzAudio  # noqa: E402


T5 = ez.T5EncoderModel


class T5Half:
    """Their loader reads flan-t5-xl (11 GB in fp32) into RAM before moving it
    to the card: more than WSL here can spare. Straight to the card in fp16
    instead — the encoder only reads a prompt."""
    @staticmethod
    def from_pretrained(name, **kw):
        m = T5.from_pretrained(name, dtype=torch.float16, device_map="cuda", **kw)

        def to_fp32(module, args, output):           # дальше сеть ждёт fp32, как было у них
            output.last_hidden_state = output.last_hidden_state.float()
            return output
        m.register_forward_hook(to_fp32)
        return m


ez.T5EncoderModel = T5Half

t0 = time.perf_counter()
model = EzAudio("s3_xl",
                ckpt_path=hf_hub_download("OpenSound/EzAudio", "ckpts/s3/ezaudio_s3_xl.pt"),
                vae_path=(hf_hub_download("OpenSound/EzAudio", "ckpts/vae/config.json"),  # рядом с весами VAE
                          hf_hub_download("OpenSound/EzAudio", "ckpts/vae/1m.pt"))[1], device="cuda")
print(f"load {time.perf_counter() - t0:.1f} s", flush=True)
for it in items:
    if (out / f"{it['name']}.wav").exists():
        continue
    t = time.perf_counter()
    sr, audio = model.generate_audio(it["prompt"], length=min(math.ceil(float(it["seconds"])), MAX_SECONDS),
                                     random_seed=int(it.get("seed", 0)))
    sf.write(out / f"{it['name']}.wav", audio, sr)
    print(f"{it['name']}: {min(it['seconds'], MAX_SECONDS)} s in {time.perf_counter() - t:.1f} s", flush=True)
