"""Merge a stress LoRA into the Qwen3-TTS weights: a checkpoint of the same shape.

    python scripts/stress_ft_merge.py experiments/23-stress-finetune/work/lora_v3 <out_dir>
    python scripts/stress_ft_merge.py <lora_v3>,<lora_speaker> <out_dir>   # по очереди
    python scripts/stress_ft_merge.py <lora_v3>,<lora_speaker>*0.5 <out_dir>   # второй вполсилы

Only talker.model.layers.* change, W += B·A·alpha/r, computed in fp32 and stored in
the original bf16. Every other file of the official snapshot is linked unchanged,
so the result loads with Qwen3TTSModel.from_pretrained and converts with
scripts/convert_qwen.py exactly like the official weights: same shapes, same
speed — only the talker GGUF differs.

Several adapters through a comma are added one after another, each over the sum
of the previous ones: a speaker LoRA trained on top of v3 (experiments/29) is
merged as "v3,speaker".

Holds the whole checkpoint in memory once (~4 GB): run it when nothing heavy runs.
"""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots")
BASE = os.path.join(SNAP, os.listdir(SNAP)[0])


def deltas(adapter_dir):
    """(key in model.safetensors, B·A·alpha/r in fp32) for every adapted matrix."""
    cfg = json.load(open(os.path.join(adapter_dir, "adapter.json")))
    scale = cfg["alpha"] / cfg["r"]
    pairs = {}
    for k, v in load_file(os.path.join(adapter_dir, "adapter.safetensors")).items():
        stem, part = k.rsplit(".lora_", 1)
        pairs.setdefault(stem, {})[part.split(".")[0]] = v
    for stem, ab in pairs.items():
        yield f"talker.model.{stem}.weight", (ab["B"].float() @ ab["A"].float()) * scale


def link(src, dst):
    # symlink на Windows требует прав, жёсткая ссылка — того же диска
    for f in (os.symlink, os.link):
        try:
            return f(src, dst)
        except OSError:
            pass
    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)


def main(adapters, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(BASE):
        if name != "model.safetensors" and not os.path.exists(os.path.join(out_dir, name)):
            link(os.path.realpath(os.path.join(BASE, name)), os.path.join(out_dir, name))

    weights, meta = {}, None
    with safe_open(os.path.join(BASE, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    merged = []
    for spec in adapters.split(","):
        # «путь*0.5» — адаптер с половинным весом
        adapter_dir, _, weight = spec.partition("*")
        weight = float(weight or 1)
        n = 0
        for key, delta in deltas(adapter_dir):
            w = weights[key]
            assert delta.shape == w.shape, (key, delta.shape, w.shape)
            weights[key] = (w.float() + delta * weight).to(w.dtype)
            n += 1
        merged.append({"adapter": os.path.abspath(adapter_dir), "weight": weight, "matrices": n})
    save_file(weights, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    json.dump({"base": BASE, "merged": merged}, open(os.path.join(out_dir, "merge.json"), "w"), indent=1)
    print(f"влито: {', '.join(str(m['matrices']) for m in merged)} матриц → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
