"""Merge a stress LoRA into the Qwen3-TTS weights: a checkpoint of the same shape.

    python scripts/stress_ft_merge.py experiments/23-stress-finetune/work/lora_v3 <out_dir>

Only talker.model.layers.* change, W += B·A·alpha/r, computed in fp32 and stored in
the original bf16. Every other file of the official snapshot is linked unchanged,
so the result loads with Qwen3TTSModel.from_pretrained and converts with
scripts/convert_qwen.py exactly like the official weights: same shapes, same
speed — only the talker GGUF differs.

Holds the whole checkpoint in memory once (~4 GB): run it when nothing heavy runs.
"""
import json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots")
BASE = os.path.join(SNAP, os.listdir(SNAP)[0])


def main(adapter_dir, out_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter.json")))
    scale = cfg["alpha"] / cfg["r"]
    lora = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
    pairs = {}
    for k, v in lora.items():
        stem, part = k.rsplit(".lora_", 1)
        pairs.setdefault(stem, {})[part.split(".")[0]] = v

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(BASE):
        if name != "model.safetensors" and not os.path.exists(os.path.join(out_dir, name)):
            os.symlink(os.path.realpath(os.path.join(BASE, name)), os.path.join(out_dir, name))

    weights, meta = {}, None
    with safe_open(os.path.join(BASE, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    merged = 0
    for stem, ab in pairs.items():
        key = f"talker.model.{stem}.weight"
        w = weights[key]
        delta = (ab["B"].float() @ ab["A"].float()) * scale
        assert delta.shape == w.shape, (key, delta.shape, w.shape)
        weights[key] = (w.float() + delta).to(w.dtype)
        merged += 1
    save_file(weights, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    json.dump({"base": BASE, "adapter": os.path.abspath(adapter_dir), "merged_matrices": merged,
               "scale": scale}, open(os.path.join(out_dir, "merge.json"), "w"), indent=1)
    print(f"влито матриц: {merged}, масштаб {scale} → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
