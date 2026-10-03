"""ACE-Step 1.5 (MIT; trained on licensed, royalty-free and synthetic music) as
a generator of takes for experiments/29.

    ~/src/ACE-Step-1.5/.venv/bin/python scripts/music_acestep.py OUT_DIR prompts.json [TAKES]

The contract every scripts/music_<model>.py keeps, so one judge reads them all:
for each prompt and take k it writes OUT_DIR/<id>.<k>.wav and
OUT_DIR/<id>.<k>.json — model, seed, seconds of music, wall time of the call
and peak VRAM. Prompts are experiments/29-music-models/prompts.json: caption,
seconds, and for songs lyrics and language; no lyrics means instrumental.

The model is two stages: a 5 Hz LM plans the song (metadata, codes) and a DiT
renders it. ACESTEP_DIT and ACESTEP_LM pick the weights (turbo DiT and the
1.7B LM — what the authors advise for 10 GB); weights live on D: in
ACESTEP_CHECKPOINTS_DIR and are downloaded on the first run.
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)
os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", "/mnt/d/ml/acestep")
SRC = Path(os.environ.get("ACESTEP_SRC", Path.home() / "src/ACE-Step-1.5"))
sys.path.insert(0, str(SRC))
os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(SRC))     # иначе .cache/ появляется в текущем каталоге

import soundfile as sf                                       # noqa: E402
import torch                                                 # noqa: E402
from acestep.handler import AceStepHandler                   # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler                 # noqa: E402

DIT = os.environ.get("ACESTEP_DIT", "acestep-v15-turbo")
LM = os.environ.get("ACESTEP_LM", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")

out = Path(sys.argv[1]).resolve()
prompts = json.loads(Path(sys.argv[2]).read_text())
takes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
out.mkdir(parents=True, exist_ok=True)

t = time.perf_counter()
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=str(SRC), config_path=DIT, device="auto", offload_to_cpu=False)
if not ok:
    sys.exit(f"DiT: {msg}")
lm = LLMHandler()
msg, ok = lm.initialize(checkpoint_dir=os.environ["ACESTEP_CHECKPOINTS_DIR"], lm_model_path=LM,
                        backend=LM_BACKEND, device="auto", offload_to_cpu=False, dtype=None)
if not ok:
    sys.exit(f"LM: {msg}")
print(f"loaded {DIT} + {LM} ({LM_BACKEND}) in {time.perf_counter() - t:.1f} s", flush=True)

for p in prompts:
    for k in range(takes):
        dst = out / f"{p['id']}.{k}.wav"
        if dst.exists():
            continue
        seed = 1000 + k
        lyrics = p.get("lyrics", "")
        params = GenerationParams(task_type="text2music", thinking=True, caption=p["caption"],
                                  lyrics=lyrics, instrumental=not lyrics,
                                  vocal_language=p.get("language", "unknown") if lyrics else "unknown",
                                  duration=float(p["seconds"]), seed=seed)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        r = generate_music(dit, lm, params=params,
                           config=GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed],
                                                   audio_format="wav"),
                           save_dir=str(out / "_raw"))
        wall = time.perf_counter() - t
        if not r.success:
            print(f"{dst.name}: FAILED {r.error or r.status_message}", flush=True)
            continue
        a = r.audios[0]
        audio = a["tensor"].numpy().T
        sf.write(dst, audio, a["sample_rate"])
        meta = {"model": f"ace-step/{DIT}+{LM}", "seed": seed, "seconds": len(audio) / a["sample_rate"],
                "wall": round(wall, 2), "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "sample_rate": a["sample_rate"],
                "lm": {k2: v for k2, v in (r.extra_outputs.get("lm_metadata") or {}).items()
                       if isinstance(v, (str, int, float))}}
        dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"{dst.name}: {meta['seconds']:.0f} s in {wall:.1f} s, {meta['vram_peak_gb']} GB", flush=True)
