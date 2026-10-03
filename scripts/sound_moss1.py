"""MOSS-SoundEffect v1 (8B, MossTTSDelay, Apache-2.0) as a generator of takes.

    HF_HOME=/mnt/d/ml/huggingface ~/src/venvs/ezaudio/bin/python \\
        scripts/sound_moss1.py OUT_DIR items.json

Same contract as sound_moss.py. In bf16 the backbone needs 16 GB of VRAM; on a
10 GB card it runs in 4-bit NF4 (bitsandbytes) — a quality cost of its own,
so this is the 8B model as it fits here, not as its authors ran it. An
autoregressive model of audio tokens: length is asked for in tokens, 12.5 a
second; no negative prompt.
"""
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

REPO = "OpenMOSS-Team/MOSS-SoundEffect"
TOKENS_PER_SECOND = 12.5
# ~14 с счёта на секунду звука: фон длиннее не нужен, сведение его зациклит
MAX_SECONDS = 12
torch.backends.cuda.enable_cudnn_sdp(False)          # как советуют авторы: cuDNN SDPA у них ломается

out = Path(sys.argv[1]).resolve()
items = json.loads(Path(sys.argv[2]).read_text())
out.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
# аудиотокенизатор остаётся на процессоре: на карте он вместе с 4-битной 8B
# занимал 13 ГБ из 10, остаток уходил в общую память Windows, и 4 с звука
# считались 88 с. Он нужен один раз на звук — при декодировании
n_tok = sum(p.numel() for p in processor.audio_tokenizer.parameters())
model = AutoModel.from_pretrained(
    REPO, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="cuda")
model.eval()
sr = processor.model_config.sampling_rate
print(f"load {time.perf_counter() - t0:.1f} s, vram {torch.cuda.memory_allocated() / 2**30:.2f} GB, {sr} Hz, "
      f"audio tokenizer {n_tok / 1e9:.2f}B params on CPU", flush=True)

with torch.no_grad():
    for it in items:
        if (out / f"{it['name']}.wav").exists():
            continue
        t = time.perf_counter()
        torch.manual_seed(int(it.get("seed", 0)))
        tokens = int(round(min(float(it["seconds"]), MAX_SECONDS) * TOKENS_PER_SECOND))
        batch = processor([[processor.build_user_message(ambient_sound=it["prompt"], tokens=tokens)]],
                          mode="generation")
        outputs = model.generate(input_ids=batch["input_ids"].to("cuda"),
                                 attention_mask=batch["attention_mask"].to("cuda"),
                                 max_new_tokens=int(tokens * 1.4) + 50)
        audio = next(iter(processor.decode(outputs))).audio_codes_list[0]
        sf.write(out / f"{it['name']}.wav", audio.float().cpu().numpy(), sr)
        print(f"{it['name']}: {it['seconds']} s in {time.perf_counter() - t:.1f} s", flush=True)
