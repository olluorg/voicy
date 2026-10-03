"""Эталон для сверки родного HeartMuLa (rust/core/examples/heartmula_probe.rs):
логиты кодбука 0 после подсказки и кодбука 1 при заданном коде 0, условная
ветвь, без выборки — heartlib в bf16 на видеокарте.

    ~/src/heartlib/.venv/bin/python scripts/heartmula_probe_ref.py CKPT "теги" "текст" C0 OUT_PREFIX
"""
import sys

import numpy as np
import torch
from heartlib import HeartMuLaGenPipeline
from heartlib.heartmula import modeling_heartmula as hm
from torchtune.models import llama3_2

hm.FLAVORS["llama-3B"] = lambda: llama3_2.llama3_2(
    vocab_size=128_256, num_layers=28, num_heads=24, num_kv_heads=8, embed_dim=3072, max_seq_len=2048,
    intermediate_dim=8192, attn_dropout=0.0, norm_eps=1e-5, rope_base=500_000, scale_factor=32)

ckpt, tags, lyrics, c0, out = sys.argv[1], sys.argv[2], sys.argv[3].replace("\\n", "\n"), int(sys.argv[4]), sys.argv[5]
pipe = HeartMuLaGenPipeline.from_pretrained(ckpt, device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
                                            dtype={"mula": torch.bfloat16, "codec": torch.float32}, version="3B",
                                            lazy_load=True)
inp = pipe.preprocess({"tags": tags, "lyrics": lyrics}, cfg_scale=1.0)
m = pipe.mula
dev = next(m.parameters()).device
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    m.setup_caches(1)
    tokens, mask, pos = inp["tokens"].to(dev), inp["tokens_mask"].to(dev), inp["pos"].to(dev)
    embeds = m._embed_tokens(tokens, uncond_mask=None)
    h = (embeds * mask.unsqueeze(-1)).sum(dim=2)
    h[0, inp["muq_idx"][0]] = m.muq_linear(inp["muq_embed"].to(dev)[0])
    mask_b = hm._index_causal_mask(m.backbone_causal_mask, pos)
    hb = m.backbone(h, input_pos=pos, mask=mask_b)
    last = hb[:, -1, :]
    l0 = m.codebook0_head(last).float()
    e0 = m._embed_audio(0, torch.tensor([[c0]], device=dev))
    cur = torch.cat([last.unsqueeze(1), e0], 1)
    m.decoder.reset_caches()
    cpos = torch.arange(2, device=dev).unsqueeze(0)
    dh = m.decoder(m.projection(cur), input_pos=cpos, mask=hm._index_causal_mask(m.decoder_causal_mask, cpos))
    l1 = torch.mm(dh[:, -1, :], m.audio_head[0]).float()
np.asarray(l0[0].cpu(), dtype=np.float32).tofile(out + ".c0.f32")
np.asarray(l1[0].cpu(), dtype=np.float32).tofile(out + ".c1.f32")
print("c0 argmax", int(l0.argmax()), "c1 argmax", int(l1.argmax()), "prompt", tokens.shape[1])
