"""LoRA fine-tuning of Qwen3-TTS to obey stress marks (U+0301 after the vowel).

The layout is the one voicy generates with — ICL cloning in streaming mode, as in
Qwen3TTSForConditionalGeneration.generate and rust/src/native/qwen.rs: role
header, codec prefix with language and speaker embedding, then reference
transcript + target text fused step by step with codec bos + reference codes,
then the target frames, each carrying the leftover text or tts_pad.

The official finetuning/sft_12hz.py trains a different layout (no reference,
text before codes) and turns the model into a single fixed speaker; this keeps
cloning of arbitrary voices intact by conditioning every example on a reference
of its own reader.

Only LoRA on the talker's transformer layers trains. Embeddings, the codec head
and the code predictor are frozen, so a merged checkpoint has the same shapes and
the same speed as the original, and converts to GGUF by the same script.

    python scripts/stress_ft_train.py check     # loss of the untouched model
    python scripts/stress_ft_train.py train
"""
import json, math, os, random, sys, time, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from stress_baseline import WORK

DS = os.path.join(WORK, "ds")
OUT = os.path.join(WORK, os.environ.get("FT_OUT", "lora"))
# наборы для обучения через запятую: train — аудиокниги RuLS, silero — сдвинутые ударения
DATA = os.environ.get("FT_DATA", "train").split(",")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots")
BASE = os.path.join(SNAP, os.listdir(SNAP)[0])

LR = float(os.environ.get("FT_LR", 1e-4))
RANK = int(os.environ.get("FT_RANK", 16))
BATCH = int(os.environ.get("FT_BATCH", 2))
ACCUM = int(os.environ.get("FT_ACCUM", 8))
EPOCHS = float(os.environ.get("FT_EPOCHS", 1))
SUB_W = 0.3


def load():
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16)
    return tts


class Builder:
    """Turns one example into talker inputs exactly as generate() would build them."""

    def __init__(self, tts):
        self.tts, self.m = tts, tts.model
        self.cfg = self.m.config
        self.tc = self.cfg.talker_config
        self.lang = self.tc.codec_language_id["russian"]
        self.spk_cache = {}

    def text_emb(self, ids):
        t = self.m.talker
        return t.text_projection(t.get_text_embeddings()(ids))

    def codec_emb(self, ids):
        return self.m.talker.get_input_embeddings()(ids)

    def frames_emb(self, codes):
        """Sum over the 16 codebooks of each frame: (T, 16) → (T, D)."""
        t = self.m.talker
        e = t.get_input_embeddings()(codes[:, 0])
        for i in range(1, codes.shape[1]):
            e = e + t.code_predictor.get_input_embeddings()[i - 1](codes[:, i])
        return e

    def speaker(self, path):
        if path not in self.spk_cache:
            x, sr = sf.read(path, dtype="float32")
            self.spk_cache[path] = self.m.extract_speaker_embedding(x, sr).detach()
        return self.spk_cache[path]

    @torch.no_grad()
    def build(self, ex):
        dev = self.m.talker.device
        tok = self.tts._tokenize_texts
        ids = tok([self.tts._build_assistant_text(ex["text"])])[0].to(dev)
        ref = tok([self.tts._build_ref_text(ex["ref_text"])])[0].to(dev)
        text_id, ref_id, role = ids[:, 3:-5], ref[:, 3:-2], ids[:, :3]
        codes = torch.tensor(ex["codes"], device=dev)
        ref_codes = torch.tensor(ex["ref_codes"], device=dev)

        special = torch.tensor([[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id,
                                 self.cfg.tts_pad_token_id]], device=dev)
        bos, eos, pad = self.text_emb(special).chunk(3, dim=1)

        pre = self.codec_emb(torch.tensor([[self.tc.codec_think_id, self.tc.codec_think_bos_id,
                                            self.lang, self.tc.codec_think_eos_id]], device=dev))
        tail = self.codec_emb(torch.tensor([[self.tc.codec_pad_id, self.tc.codec_bos_id]], device=dev))
        codec = torch.cat([pre, self.speaker(ex["ref_audio"]).view(1, 1, -1).to(pre.dtype), tail], dim=1)
        prefix = torch.cat([pad.expand(-1, codec.shape[1] - 2, -1), bos], dim=1) + codec[:, :-1]

        text = torch.cat([self.text_emb(torch.cat([ref_id, text_id], dim=-1)), eos], dim=1)
        icl_codec = torch.cat([self.codec_emb(torch.tensor([[self.tc.codec_bos_id]], device=dev)),
                               self.frames_emb(ref_codes).unsqueeze(0)], dim=1)
        n = icl_codec.shape[1]
        if text.shape[1] > n:
            icl, trailing = text[:, :n] + icl_codec, text[:, n:]
        else:
            icl = torch.cat([text, pad.expand(-1, n - text.shape[1], -1)], dim=1) + icl_codec
            trailing = pad[:, :0]

        m = codes.shape[0]
        add = torch.cat([trailing[:, :m], pad.expand(-1, max(0, m - trailing.shape[1]), -1)], dim=1)
        target = self.frames_emb(codes).unsqueeze(0) + add

        x = torch.cat([self.text_emb(role), prefix, icl, target], dim=1)[0]
        # позиция, выход которой предсказывает первый кадр, — последняя позиция ICL
        first = x.shape[0] - m - 1
        labels = torch.cat([codes[:, 0], torch.tensor([self.tc.codec_eos_token_id], device=dev)])
        return {"x": x, "first": first, "labels": labels, "codes": codes,
                "text_left": max(0, trailing.shape[1] - m)}


def batch_loss(model_talker, items, sub_shift=0):
    dev = items[0]["x"].device
    T = max(it["x"].shape[0] for it in items)
    D = items[0]["x"].shape[1]
    x = torch.zeros(len(items), T, D, dtype=items[0]["x"].dtype, device=dev)
    mask = torch.zeros(len(items), T, dtype=torch.long, device=dev)
    for b, it in enumerate(items):
        x[b, :it["x"].shape[0]] = it["x"]
        mask[b, :it["x"].shape[0]] = 1
    out = model_talker(inputs_embeds=x, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[0][-1]
    logits = out.logits

    c0_logits, c0_labels, sub_h, sub_codes = [], [], [], []
    for b, it in enumerate(items):
        m = it["codes"].shape[0]
        pos = torch.arange(it["first"], it["first"] + m + 1, device=dev)
        c0_logits.append(logits[b, pos]); c0_labels.append(it["labels"])
        # скрытое состояние, которое предсказало нулевой кодбук кадра, — как в generate()
        sub_h.append(h[b, pos[:m] + sub_shift]); sub_codes.append(it["codes"])
    c0 = F.cross_entropy(torch.cat(c0_logits).float(), torch.cat(c0_labels))
    # логиты forward_finetune уже выровнены (позиция i → кодбук i), а его собственный
    # loss_function сдвигает метки ещё раз; на нетронутой модели это давало 11.9 —
    # хуже случайного угадывания. Потери считаются здесь, без второго сдвига.
    codes = torch.cat(sub_codes)
    sub_logits, _ = model_talker.forward_sub_talker_finetune(codes, torch.cat(sub_h))
    sub = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]), codes[:, 1:].reshape(-1))
    return c0, sub


def read(split):
    return [json.loads(l) for l in open(os.path.join(DS, f"{split}.jsonl"), encoding="utf-8")]


def check():
    """Sanity of the layout on the untouched model: correct alignment gives low loss.

    Also: how much the marks cost the untouched model — it never saw them.
    """
    tts = load(); tts.model.eval()
    bld = Builder(tts)
    val = read("val")[:48]
    strip = lambda s: s.replace("́", "")
    with torch.no_grad():
        for shift in (0, 1):
            c0s, subs = [], []
            for i in range(0, len(val), 4):
                c0, sub = batch_loss(tts.model.talker, [bld.build(e) for e in val[i:i + 4]], shift)
                c0s.append(c0.item()); subs.append(sub.item())
            print(f"сдвиг {shift}: c0 {np.mean(c0s):.3f}, остальные кодбуки {np.mean(subs):.3f}")
        c0s = []
        for i in range(0, len(val), 4):
            c0, _ = batch_loss(tts.model.talker, [bld.build({**e, "text": strip(e["text"])}) for e in val[i:i + 4]])
            c0s.append(c0.item())
        print(f"без знаков ударения: c0 {np.mean(c0s):.3f}")


def train():
    from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    tts = load()
    bld = Builder(tts)
    for p in tts.model.parameters():
        p.requires_grad_(False)
    talker = tts.model.talker
    # только слои основного трансформера; предсказатель остальных кодбуков не трогаем
    targets = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    lcfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, target_modules=targets)
    # адаптер встраивается в сам модуль: вызовы talker остаются прежними
    inject_adapter_in_model(lcfg, talker.model)
    n_train = sum(p.numel() for p in talker.model.parameters() if p.requires_grad)
    print(f"обучаемых параметров: {n_train / 1e6:.1f} млн", flush=True)
    try:
        talker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:
        print("без checkpointing:", e)

    def save():
        os.makedirs(OUT, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(talker.model).items()}
        save_file(sd, os.path.join(OUT, "adapter.safetensors"))
        json.dump({"r": RANK, "alpha": 2 * RANK, "target_modules": targets, "base": BASE,
                   "applied_to": "talker.model"}, open(os.path.join(OUT, "adapter.json"), "w"), indent=1)

    os.makedirs(OUT, exist_ok=True)
    train_set = [e for name in DATA for e in read(name)]
    val = read("val")[:64]
    params = [p for p in talker.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = math.ceil(len(train_set) * EPOCHS / (BATCH * ACCUM))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    print(f"{len(train_set)} примеров из {DATA}, {steps} шагов, lr {LR}, rank {RANK}, → {OUT}", flush=True)

    def evaluate():
        talker.eval()
        with torch.no_grad():
            v = [batch_loss(talker, [bld.build(e) for e in val[i:i + 4]]) for i in range(0, len(val), 4)]
            # те же фразы без знаков: обычное чтение не должно портиться
            u = [batch_loss(talker, [bld.build({**e, "text": e["text"].replace("\u0301", "")})
                                     for e in val[i:i + 4]])[0] for i in range(0, len(val), 4)]
        talker.train()
        return (np.mean([a.item() for a, _ in v]), np.mean([b.item() for _, b in v]),
                np.mean([a.item() for a in u]))

    rng = random.Random(5)
    order = []
    while len(order) < steps * BATCH * ACCUM:
        chunk = list(range(len(train_set))); rng.shuffle(chunk); order += chunk
    print("до обучения: val c0 %.3f sub %.3f, без знаков c0 %.3f" % evaluate(), flush=True)
    talker.train()
    t0, k = time.time(), 0
    log = []
    for step in range(steps):
        acc_c0 = acc_sub = 0.0
        for _ in range(ACCUM):
            items = [bld.build(train_set[order[k + j]]) for j in range(BATCH)]
            k += BATCH
            c0, sub = batch_loss(talker, items)
            ((c0 + SUB_W * sub) / ACCUM).backward()
            acc_c0 += c0.item() / ACCUM; acc_sub += sub.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        row = {"step": step + 1, "c0": round(acc_c0, 4), "sub": round(acc_sub, 4)}
        if (step + 1) % 25 == 0 or step + 1 == steps:
            row["val_c0"], row["val_sub"], row["val_plain"] = map(float, evaluate())
            save()
        log.append(row)
        el = time.time() - t0
        print(f"шаг {step + 1}/{steps}  c0 {acc_c0:.3f}  sub {acc_sub:.3f}"
              + (f"  val c0 {row['val_c0']:.3f} sub {row['val_sub']:.3f} без знаков {row['val_plain']:.3f}"
                 if "val_c0" in row else "")
              + f"  {el / 60:.1f} мин, ещё ≈{el / (step + 1) * (steps - step - 1) / 60:.0f}", flush=True)
        json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=1)
    save()
    print("готово:", OUT)


if __name__ == "__main__":
    {"check": check, "train": train}[sys.argv[1]]()
