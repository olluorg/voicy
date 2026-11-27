"""LoRA fine-tuning of Qwen3-TTS on one reader's paragraphs (experiments/29, variant C).

The layout, the losses and the LoRA are those of scripts/stress_ft_train.py: ICL
cloning, the reference is another stretch of the same reader with its text. What
differs is the data — 20–60 s paragraphs instead of 4–10 s phrases — and the start:
training goes on top of the released stress model (v3 merged in memory), so the
merged result keeps obeying U+0301.

Long examples: batch 1 with accumulation and gradient checkpointing; one
hour of audio is only ~90 paragraphs, so several epochs.

    python scripts/speaker_ft_train.py check    # loss of v3 on the paragraphs
    python scripts/speaker_ft_train.py probe    # memory and step time by length, synthetic
    python scripts/speaker_ft_train.py train

Merge for the server: scripts/stress_ft_merge.py <lora_v3>,<lora_c> <out_dir>.
"""
import json, math, os, random, sys, time, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.path.dirname(__file__))
from stress_ft_train import Builder, batch_loss, load, BASE, SUB_W
from stress_ft_merge import deltas

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
DS = os.path.join(WORK, "ds")
OUT = os.path.join(WORK, os.environ.get("FT_OUT", "lora_c"))
# адаптеры, поверх которых учимся, через запятую; пусто — исходная Qwen
START = [p for p in os.environ.get("FT_START", os.path.join(WORK, "lora_v3")).split(",") if p]

LR = float(os.environ.get("FT_LR", 1e-4))
RANK = int(os.environ.get("FT_RANK", 16))
ACCUM = int(os.environ.get("FT_ACCUM", 4))
EPOCHS = float(os.environ.get("FT_EPOCHS", 3))
EVAL_EVERY = int(os.environ.get("FT_EVAL_EVERY", 10))
TARGETS = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def start():
    """The official weights with the START adapters added, as stress_ft_merge would."""
    tts = load()
    with torch.no_grad():
        for adapter in START:
            n = 0
            for key, delta in deltas(adapter):
                p = tts.model.get_parameter(key)
                p.copy_((p.float() + delta.to(p.device)).to(p.dtype))
                n += 1
            print(f"влито {n} матриц из {adapter}", flush=True)
    return tts


def with_lora(tts):
    from peft import LoraConfig, inject_adapter_in_model
    for p in tts.model.parameters():
        p.requires_grad_(False)
    talker = tts.model.talker
    lcfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, target_modules=TARGETS)
    inject_adapter_in_model(lcfg, talker.model)
    talker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return talker, [p for p in talker.parameters() if p.requires_grad]


def read(split):
    return [json.loads(l) for l in open(os.path.join(DS, f"{split}.jsonl"), encoding="utf-8")]


def evaluate(talker, bld, val):
    talker.eval()
    with torch.no_grad():
        v = [batch_loss(talker, [bld.build(e)]) for e in val]
    talker.train()
    return float(np.mean([a.item() for a, _ in v])), float(np.mean([b.item() for _, b in v]))


def check():
    tts = start(); tts.model.eval()
    print("val c0 %.3f, остальные кодбуки %.3f" % evaluate(tts.model.talker, Builder(tts), read("val")))


def probe():
    """One forward+backward per target length on random codes: peak memory and time."""
    tts = start()
    bld = Builder(tts)
    talker, params = with_lora(tts)
    talker.train()
    rng = np.random.default_rng(0)
    ref = os.path.join(WORK, "probe_ref.wav")
    sf.write(ref, (rng.standard_normal(24000 * 10) * 0.05).astype(np.float32), 24000)
    letters = "абвгдеёжзийклмнопрстуфхцчшщыьэюя"
    words = lambda sec: " ".join("".join(rng.choice(list(letters), 6)) for _ in range(int(sec * 15 / 7)))
    codes = lambda sec: rng.integers(0, 2048, (int(sec * 12.5), 16)).tolist()
    ex_ref = {"ref_text": words(10), "ref_codes": codes(10), "ref_audio": ref}
    for sec in [int(s) for s in os.environ.get("PROBE_SECONDS", "30,45,60,75,90").split(",")]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            it = bld.build({**ex_ref, "text": words(sec), "codes": codes(sec)})
            c0, sub = batch_loss(talker, [it])
            (c0 + SUB_W * sub).backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            for p in params:
                p.grad = None
            print(f"{sec} с цели: позиций {it['x'].shape[0]}, пик {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ "
                  f"(резерв {torch.cuda.max_memory_reserved() / 2**30:.2f}), шаг {dt:.2f} с", flush=True)
        except torch.OutOfMemoryError:
            print(f"{sec} с цели: не влезает", flush=True)
            for p in params:
                p.grad = None
            break


def train():
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    tts = start()
    bld = Builder(tts)
    talker, params = with_lora(tts)
    print(f"обучаемых параметров: {sum(p.numel() for p in params) / 1e6:.1f} млн", flush=True)

    def save():
        os.makedirs(OUT, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(talker.model).items()}
        save_file(sd, os.path.join(OUT, "adapter.safetensors"))
        json.dump({"r": RANK, "alpha": 2 * RANK, "target_modules": TARGETS, "base": BASE,
                   "applied_to": "talker.model", "on_top_of": START}, open(os.path.join(OUT, "adapter.json"), "w"),
                  indent=1)

    train_set, val = read("train"), read("val")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = math.ceil(len(train_set) * EPOCHS / ACCUM)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    sec = sum(e["duration"] for e in train_set) / 3600
    print(f"{len(train_set)} примеров, {sec:.2f} ч, {steps} шагов по {ACCUM}, {EPOCHS} эпох, lr {LR}, "
          f"rank {RANK}, поверх {START or 'исходной'} → {OUT}", flush=True)

    rng = random.Random(5)
    order = []
    while len(order) < steps * ACCUM:
        chunk = list(range(len(train_set))); rng.shuffle(chunk); order += chunk
    print("до обучения: val c0 %.3f sub %.3f" % evaluate(talker, bld, val), flush=True)
    os.makedirs(OUT, exist_ok=True)
    talker.train()
    # сто абзацев переучить легко: сохраняется лучший по val c0, а не последний
    t0, k, log, best = time.time(), 0, [], float("inf")
    for step in range(steps):
        acc_c0 = acc_sub = 0.0
        for _ in range(ACCUM):
            c0, sub = batch_loss(talker, [bld.build(train_set[order[k]])])
            k += 1
            ((c0 + SUB_W * sub) / ACCUM).backward()
            acc_c0 += c0.item() / ACCUM; acc_sub += sub.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        row = {"step": step + 1, "c0": round(acc_c0, 4), "sub": round(acc_sub, 4)}
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == steps:
            row["val_c0"], row["val_sub"] = evaluate(talker, bld, val)
            if row["val_c0"] < best:
                best = row["val_c0"]; row["saved"] = True
                save()
        log.append(row)
        el = time.time() - t0
        print(f"шаг {step + 1}/{steps}  c0 {acc_c0:.3f}  sub {acc_sub:.3f}"
              + (f"  val c0 {row['val_c0']:.3f} sub {row['val_sub']:.3f}" if "val_c0" in row else "")
              + f"  {el / 60:.1f} мин, ещё ≈{el / (step + 1) * (steps - step - 1) / 60:.0f}", flush=True)
        json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=1)
    print(f"готово: {OUT}, лучший val c0 {best:.3f}")


if __name__ == "__main__":
    {"check": check, "probe": probe, "train": train}[sys.argv[1]]()
