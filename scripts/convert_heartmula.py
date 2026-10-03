"""HeartMuLa-oss-3B → GGUF и таблицы: то, что исполняет родной движок (native/heartmula.rs).

    ~/src/heartlib/.venv/bin/python scripts/convert_heartmula.py CKPT OUT

CKPT — каталог heartlib (`HeartMuLa-oss-3B/`, `HeartCodec-oss/`, `tokenizer.json`,
`gen_config.json`), OUT — куда положить результат.

Устройство то же, что у Qwen3-TTS в voicy (native/qwen.rs): бэкбон Llama-3.2 3B
ведёт цикл кадров и предсказывает кодбук 0, малый декодер (3 слоя) — кодбуки 1–7.
Обе части — обычная архитектура `llama` в GGUF, на вход им подаются эмбеддинги.
Словарь у обеих фиктивный, как у предсказателя Qwen: у бэкбона это 8197 кодов
кодбука 0 и голова `codebook0_head`; у декодера — семь голов `audio_head`,
склеенных подряд, и нужный кусок логитов берёт сэмплер.

RoPE torchtune хранит в мета-раскладке (соседние пары), которую и ждёт llama.cpp
для `llama`, поэтому q и k не переставляются; масштаб llama3 (×32) — тензором
`rope_freqs`, как его строит convert_hf_to_gguf.

Таблицы (.npy, f16): эмбеддинги кодов для входа бэкбона, те же эмбеддинги,
уже прошедшие проекцию в декодер, проекция для скрытого состояния бэкбона,
эмбеддинги текста, безусловный эмбеддинг для CFG и сдвиг muq_linear (вход MuQ
у heartlib пока нулевой, так что от слоя остаётся сдвиг).

Кодек (HeartCodec) — отдельно, scripts/convert_heartcodec.py.
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from safetensors import safe_open

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
mula = ckpt / "HeartMuLa-oss-3B"
cfg = json.loads((mula / "config.json").read_text())
V = cfg["audio_vocab_size"]          # 8197
NQ = cfg["audio_num_codebooks"]      # 8

index = json.loads((mula / "model.safetensors.index.json").read_text())["weight_map"]
handles = {}


def get(name: str) -> torch.Tensor:
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(str(mula / f), framework="pt")
    return handles[f].get_tensor(name)


def f16(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float16).numpy()


# torchtune llama3_2: 3B — 28 слоёв, 24 головы, 8 kv; 300M — 3 слоя, 8 голов, 4 kv; оба 3072/8192
FLAVOR = {"llama-3B": dict(layers=28, heads=24, kv=8), "llama-300M": dict(layers=3, heads=8, kv=4)}
EMBD, FFN, ROPE_BASE, EPS = 3072, 8192, 500_000.0, 1e-5


def rope_freqs(dim: int) -> np.ndarray:
    """Множители частот RoPE для масштаба llama3 — как в convert_hf_to_gguf (LlamaModel)."""
    factor, low, high, old = 32.0, 1.0, 4.0, 8192
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    lo_wave, hi_wave = old / low, old / high
    out = []
    for f in freqs:
        wave = 2 * math.pi / f
        if wave < hi_wave:
            out.append(1.0)
        elif wave > lo_wave:
            out.append(factor)
        else:
            s = (old / wave - low) / (high - low)
            out.append(1.0 / ((1 - s) / factor + s))
    return np.array(out, dtype=np.float32)


def write_llama(path: Path, prefix: str, flavor: str, n_ctx: int, head: np.ndarray, embd: np.ndarray, name: str):
    f = FLAVOR[flavor]
    n_vocab = head.shape[0]
    w = GGUFWriter(str(path), "llama")
    w.add_name(name)
    w.add_context_length(n_ctx)
    w.add_embedding_length(EMBD)
    w.add_block_count(f["layers"])
    w.add_feed_forward_length(FFN)
    w.add_head_count(f["heads"])
    w.add_head_count_kv(f["kv"])
    w.add_rope_dimension_count(EMBD // f["heads"])
    w.add_rope_freq_base(ROPE_BASE)
    w.add_layer_norm_rms_eps(EPS)
    w.add_vocab_size(n_vocab)
    w.add_file_type(1)  # f16
    # словарь фиктивный: на вход идут эмбеддинги, а выход — коды
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list([f"<c{i}>" for i in range(n_vocab)])
    w.add_token_types([1] * n_vocab)
    w.add_token_merges(["<c 0>"])
    w.add_tensor("token_embd.weight", embd)
    w.add_tensor("rope_freqs.weight", rope_freqs(EMBD // f["heads"]), raw_dtype=GGMLQuantizationType.F32)
    for i in range(f["layers"]):
        p = f"{prefix}.layers.{i}"
        w.add_tensor(f"blk.{i}.attn_norm.weight", get(f"{p}.sa_norm.scale").float().numpy())
        w.add_tensor(f"blk.{i}.attn_q.weight", f16(get(f"{p}.attn.q_proj.weight")))
        w.add_tensor(f"blk.{i}.attn_k.weight", f16(get(f"{p}.attn.k_proj.weight")))
        w.add_tensor(f"blk.{i}.attn_v.weight", f16(get(f"{p}.attn.v_proj.weight")))
        w.add_tensor(f"blk.{i}.attn_output.weight", f16(get(f"{p}.attn.output_proj.weight")))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", get(f"{p}.mlp_norm.scale").float().numpy())
        w.add_tensor(f"blk.{i}.ffn_gate.weight", f16(get(f"{p}.mlp.w1.weight")))
        w.add_tensor(f"blk.{i}.ffn_down.weight", f16(get(f"{p}.mlp.w2.weight")))
        w.add_tensor(f"blk.{i}.ffn_up.weight", f16(get(f"{p}.mlp.w3.weight")))
    w.add_tensor("output_norm.weight", get(f"{prefix}.norm.scale").float().numpy())
    w.add_tensor("output.weight", head)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"{path.name}: {path.stat().st_size / 1e9:.2f} GB", flush=True)


audio = get("audio_embeddings.weight")                      # [NQ*V, 3072]
proj = get("projection.weight")                             # [3072, 3072]: y = x @ W.T

# бэкбон: выход — голова кодбука 0; входной словарь — коды кодбука 0 (на вход идут эмбеддинги)
write_llama(out / "heartmula_backbone.f16.gguf", "backbone", "llama-3B", 8192,
            f16(get("codebook0_head.weight")), f16(audio[:V]), "HeartMuLa backbone")

# декодер: семь голов [3072, V] подряд → [7V, 3072]
heads = get("audio_head")                                   # [NQ-1, 3072, V]
head = torch.cat([heads[i].T for i in range(NQ - 1)], 0)
write_llama(out / "heartmula_decoder.f16.gguf", "decoder", "llama-300M", 64, f16(head), f16(head), "HeartMuLa decoder")
del heads, head

np.save(out / "audio_embeddings.npy", f16(audio))
np.save(out / "audio_embeddings_projected.npy", f16(audio @ proj.T))
np.save(out / "projection.npy", f16(proj))
np.save(out / "text_embeddings.npy", f16(get("text_embeddings.weight")))
np.save(out / "unconditional.npy", f16(get("unconditional_text_embedding.weight")[0]))
np.save(out / "muq_bias.npy", f16(get("muq_linear.bias")))
for f in ("tokenizer.json", "gen_config.json"):
    shutil.copy(ckpt / f, out / f)
(out / "heartmula.json").write_text(json.dumps({
    "audio_vocab_size": V, "codebooks": NQ, "frame_rate": 12.5, "source": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
    **json.loads((ckpt / "gen_config.json").read_text())}, indent=1))
print("готово:", out, flush=True)
