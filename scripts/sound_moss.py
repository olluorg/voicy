"""MOSS-SoundEffect v2.0: load once, render a list of prompts, report time and memory.

    TORCHDYNAMO_DISABLE=1 ~/src/MOSS-TTS/moss_soundeffect_v2/.venv/bin/python \\
        scripts/sound_moss.py OUT_DIR items.json [steps] [cfg]

items.json: [{"name", "prompt", "seconds", "seed"?, "negative"?}, ...]
Weights are read from the Hugging Face cache (on D:, experiments/28).
"""
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path.home() / "src" / "MOSS-TTS"))
from moss_soundeffect_v2 import MossSoundEffectPipeline  # noqa: E402
from moss_soundeffect_v2.diffsynth.pipelines import wan_audio as W  # noqa: E402


def load_fitting_10gb(model_dir: str) -> MossSoundEffectPipeline:
    """Their from_pretrained, rearranged for a 10 GB card and 16 GB of RAM.

    Theirs builds the fp32 DiT on the CPU beside its loaded weights (two copies,
    >12 GB of RAM) and keeps everything on the GPU (~11 GB): with CFG it spills
    into Windows' shared memory and crawls. Here the DiT is built on the GPU and
    its weights go to bf16 (the complex RoPE buffers stay as they are); the
    Qwen3 text encoder lives in RAM and visits the GPU only to read a prompt.
    """
    d = Path(model_dir)
    dit_cfg = json.loads((d / "transformer" / "config.json").read_text())
    sched = json.loads((d / "scheduler" / "scheduler_config.json").read_text())
    index = json.loads((d / "model_index.json").read_text())
    sd = W._convert_hf_dit_state_dict(W.load_file(str(d / "transformer" / "diffusion_pytorch_model.safetensors")))
    with torch.device("cuda"):
        dit = W.WanAudioModel(
            in_dim=dit_cfg["in_dim"], out_dim=dit_cfg["out_dim"], text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"], eps=dit_cfg["eps"], patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"], dim=dit_cfg["dim"], ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"], num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"))
    r = dit.load_state_dict(sd)
    assert not r.missing_keys and not r.unexpected_keys, r
    del sd
    if os.environ.get("DIT_DTYPE", "bf16") == "bf16":
        for prm in dit.parameters():
            prm.data = prm.data.to(torch.bfloat16)
        torch.cuda.empty_cache()
    vae = W.DAC.load(str(d / "vae" / "vae_128d_48k.pth")).to("cuda")
    te = W.Qwen3TextEncoder(str(d / "text_encoder"), torch_dtype=torch.bfloat16)
    prompter = W.WanPrompter(tokenizer_path=str(d / "tokenizer"))
    prompter.fetch_models(te)
    encode = prompter.encode_prompt
    seen = {}

    def on_loan(prompt, positive=True, device="cuda"):
        """On the CPU bf16 Qwen3 takes 16 s a prompt; on the GPU, 1.5 s.
        The same prompt (the empty negative one) is read once."""
        key = (prompt, positive)
        if key not in seen:
            te.to(device)
            seen[key] = encode(prompt, positive=positive, device=device)
            te.to("cpu")
            torch.cuda.empty_cache()
        return seen[key]

    prompter.encode_prompt = on_loan
    pipe = W.WanAudioPipeline(device="cuda", torch_dtype=torch.bfloat16, flow_shift=sched.get("shift", 5.0))
    object.__setattr__(pipe, "text_encoder", te)       # не ребёнок модуля: никакой .to его не тронет
    pipe.prompter = prompter
    pipe.vae = vae
    pipe.dit = dit
    pipe.audio_latent_dim = dit_cfg["in_dim"]
    pipe.num_samples_division_factor = vae.hop_length
    pipe.dit_variant = index.get("dit_variant")
    return MossSoundEffectPipeline(engine=pipe, sample_rate=int(index.get("sample_rate", 48000)),
                                   max_inference_seconds=int(index.get("max_inference_seconds", 30)))


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(sys.argv[2]).read_text())
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    cfg = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

    t0 = time.perf_counter()
    pipe = load_fitting_10gb(snapshot_download("OpenMOSS-Team/MOSS-SoundEffect-v2.0", local_files_only=True))
    torch.cuda.synchronize()
    print(f"load {time.perf_counter() - t0:.1f} s, vram {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    for it in items:
        if (out / f"{it['name']}.wav").exists():
            continue                                    # прерванный прогон продолжается с места
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        audio = pipe(prompt=it["prompt"], seconds=float(it["seconds"]), num_inference_steps=steps,
                     cfg_scale=cfg, seed=int(it.get("seed", 0)), negative_prompt=it.get("negative", ""))
        torch.cuda.synchronize()
        spent = time.perf_counter() - t
        wav = audio[0].detach().float().cpu().numpy().T
        sf.write(out / f"{it['name']}.wav", wav, pipe.sample_rate)
        print(f"{it['name']}: {it['seconds']} s in {spent:.1f} s, peak vram "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB, rms {float((wav ** 2).mean() ** 0.5):.3f}", flush=True)
