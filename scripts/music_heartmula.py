"""HeartMuLa-oss-3B (Apache 2.0) as a generator of takes for experiments/29.

    ~/src/heartlib/.venv/bin/python scripts/music_heartmula.py OUT_DIR prompts.json [TAKES]

Same contract as scripts/music_acestep.py. Unlike ACE-Step this is a language
model over a 12.5 Hz codec: it writes frames until it decides the song is over,
so the length asked for is only a ceiling, and conditioning is short lowercase
tags rather than a sentence. The caption is cut into tags at its commas.
There is no instrumental switch; an empty song gets "[instrumental]" as lyrics.

The 3B model in bf16 and the codec in fp32 do not fit 10 GB together, so the
pipeline loads each for its stage and drops it after (lazy_load).
Weights: HEARTMULA_DIR (D:, see the heartlib README for the three downloads).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from heartlib import HeartMuLaGenPipeline
from heartlib.heartmula import modeling_heartmula as hm
from torchtune.models import llama3_2

CKPT = os.environ.get("HEARTMULA_DIR", "/mnt/d/ml/heartmula/ckpt")
SLACK = 1.3                    # потолок длины: модель сама решает, где песня кончается
# Кэш ключей бэкбона рассчитан на 8192 позиции (10 минут): с CFG это ещё 1.9 ГБ
# к 7.2 ГБ весов, и на 10 ГБ не помещается. 2560 позиций — текст и 3 минуты звука.
MAX_SEQ = int(os.environ.get("HEARTMULA_MAX_SEQ", 2560))
hm.FLAVORS["llama-3B"] = lambda: llama3_2.llama3_2(
    vocab_size=128_256, num_layers=28, num_heads=24, num_kv_heads=8, embed_dim=3072, max_seq_len=MAX_SEQ,
    intermediate_dim=8192, attn_dropout=0.0, norm_eps=1e-5, rope_base=500_000, scale_factor=32)

out = Path(sys.argv[1]).resolve()
prompts = json.loads(Path(sys.argv[2]).read_text())
takes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
out.mkdir(parents=True, exist_ok=True)

t = time.perf_counter()
pipe = HeartMuLaGenPipeline.from_pretrained(
    CKPT, device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
    dtype={"mula": torch.bfloat16, "codec": torch.float32}, version="3B", lazy_load=True)
print(f"ready in {time.perf_counter() - t:.1f} s", flush=True)

for p in prompts:
    for k in range(takes):
        dst = out / f"{p['id']}.{k}.wav"
        if dst.exists():
            continue
        seed = 1000 + k
        random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
        tags = ",".join(s.strip() for s in p["caption"].split(","))
        lyrics = p.get("lyrics") or "[instrumental]"
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.no_grad():
            inp = pipe.preprocess({"tags": tags, "lyrics": lyrics}, cfg_scale=1.5)
            frames = pipe._forward(inp, max_audio_length_ms=int(p["seconds"] * SLACK * 1000),
                                   temperature=1.0, topk=50, cfg_scale=1.5)["frames"]
            wav = pipe.codec.detokenize(frames.to(pipe.codec_device))
            pipe._unload()
        wall = time.perf_counter() - t
        audio = wav.to(torch.float32).cpu().numpy().T
        sf.write(dst, audio, 48000)
        meta = {"model": "heartmula/oss-3B-happy-new-year", "seed": seed, "seconds": len(audio) / 48000,
                "wall": round(wall, 2), "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "sample_rate": 48000, "tags": tags}
        dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"{dst.name}: {meta['seconds']:.0f} s in {wall:.1f} s, {meta['vram_peak_gb']} GB", flush=True)
