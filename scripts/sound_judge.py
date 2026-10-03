"""Pick one candidate per scene element with no human in the loop.

    python scripts/sound_judge.py plan.json CAND_DIR CHOSEN_DIR REPORT.md

REPORT.md is for people; beside it REPORT.json tells the scene runner which
elements had no acceptable take and what AST heard in them instead.

Candidates are CAND_DIR/<element id>__<variant>.wav. Rejected: what AST
(AudioSet) hears as a forbidden group — an engine in the wind, rain in a fire,
a cat in a door, silence where a sound should be; what CLAP puts closer to a
confuser than to the target; near-silence. Of the rest the one closest to its
target in CLAP wins, production quality (Audiobox Aesthetics PQ) breaking ties.
If all are rejected, the least bad goes in and is flagged.

On 28 sounds the user rated (experiments/28) these rules caught 12 of 12
"wrong sound" verdicts with no false alarm on 11 good ones. They do not catch
an unnatural sound of the right kind; and the thresholds were fitted on those
same files.
"""
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ClapModel, ClapProcessor

import audiobox_aesthetics.infer as abx

plan = json.loads(Path(sys.argv[1]).read_text())
cand, chosen, report = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
chosen.mkdir(parents=True, exist_ok=True)
dev = "cuda"

ELEMENTS = plan["beds"] + plan["events"]
# у фона цель сверяется ещё и с типовой фразой его вида: CLAP слишком зависит
# от формулировки, а её пишет LLM (синтезированную вьюгу, одобренную на слух,
# «wind and snow howling outside» забраковало как шум)
KIND_TEXT = {"wind": "strong winter blizzard wind blowing", "rain": "steady rain falling",
             "water": "water flowing", "fire": "a wood fire crackling", "room": "the quiet interior of a room",
             "crowd": "a crowd of people talking", "nature": "sounds of nature outdoors"}
TARGET = {x["id"]: [x["target"]] + ([KIND_TEXT[x["kind"]]] if x.get("kind") in KIND_TEXT else []) for x in ELEMENTS}
# событие обязано звучать как надо, а не только не звучать как запрещённое (вздох вместо кашля)
EXPECT = {x["id"]: [c for c in x.get("expect", []) if "music" not in c.lower()] for x in ELEMENTS}
# ожидаемый класс должен быть сопоставим с самым сильным классом-источником:
# «в первой десятке» пропустило отрыжку 0.30 при Throat clearing 0.14 — на слух рычание
EXPECT_SHARE, EXPECT_MIN = 0.6, 0.10
# классы, которые говорят не о том, что звучит, а о записи и месте
GENERIC = {"Sound effect", "Inside, small room", "Inside, large room or hall", "Inside, public space",
           "Outside, rural or natural", "Outside, urban or manmade", "Field recording", "Silence",
           "Noise", "Static", "Echo", "Reverberation", "Environmental noise"}
# тишина комнаты и есть тишина; остальное — что разрешил план
ALLOW = {x["id"]: set(x.get("allow", [])) | ({"silence"} if x.get("kind") == "room" else set()) for x in ELEMENTS}
CONFUSERS = ["a car engine revving", "hail hitting a roof", "rain falling", "rustling of dry hay",
             "distorted white noise", "a cat meowing", "music playing", "a person speaking"]
FORBID = {  # группа AST: (классы, порог)
    "engine": (["Engine", "Vehicle", "Car", "Accelerating, revving, vroom", "Motor vehicle (road)",
                "Race car, auto racing", "Idling", "Medium engine (mid frequency)",
                "Heavy engine (low frequency)", "Light engine (high frequency)", "Aircraft",
                "Fixed-wing aircraft, airplane", "Jet engine"], 0.20),
    "rain": (["Rain", "Raindrop", "Rain on surface", "Hail"], 0.15),
    "animal": (["Animal", "Cat", "Meow", "Dog", "Bird", "Domestic animals, pets"], 0.20),
    "music": (["Music", "Musical instrument"], 0.30),
    "speech": (["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                "Conversation", "Narration, monologue"], 0.30),
    # «дуновение вместо вьюги»: звук верный, но его почти нет
    "silence": (["Silence"], 0.30),
    # дефект записи, а не звук сцены: ветер бьёт в микрофон
    "mic_wind": (["Wind noise (microphone)"], 0.30),
}
QUIET = 0.002                            # RMS: тише — пусто

fx = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
lab = {v: k for k, v in ast.config.id2label.items()}
cp = ClapProcessor.from_pretrained("laion/larger_clap_general")
clap = ClapModel.from_pretrained("laion/larger_clap_general").to(dev).eval()


def read_wav(meta):
    """soundfile instead of torchaudio: the latter wants ffmpeg here."""
    x, sr = sf.read(meta["path"], dtype="float32", always_2d=True)
    return torch.from_numpy(x.mean(1, keepdims=True).T.copy()), sr


abx.read_wav = read_wav
aes = abx.initialize_predictor()


def mono(path, sr):
    x, r = sf.read(path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    return resample_poly(x, sr, r).astype(np.float32) if r != sr else x


def windows(x, sr, sec=10.0):
    n = int(sec * sr)
    return [x[i:i + n] for i in range(0, max(1, len(x) - n // 2), n)] or [x]


@torch.no_grad()
def judge(path, element):
    x16 = mono(path, 16000)
    p = torch.sigmoid(ast(**fx(windows(x16, 16000), sampling_rate=16000, return_tensors="pt").to(dev)).logits).mean(0).cpu().numpy()
    a = clap.get_audio_features(**cp(audio=windows(mono(path, 48000), 48000), sampling_rate=48000, return_tensors="pt").to(dev))
    a = torch.nn.functional.normalize(torch.nn.functional.normalize(a, dim=-1).mean(0), dim=-1)
    targets = TARGET[element]
    texts = targets + CONFUSERS
    t = torch.nn.functional.normalize(clap.get_text_features(**cp(text=texts, return_tensors="pt", padding=True).to(dev)), dim=-1)
    sims = (t @ a).cpu().numpy()
    best_t = int(np.argmax(sims[: len(targets)]))
    keep = [best_t] + list(range(len(targets), len(texts)))   # лучшая из целей против ложных
    texts, sims = [texts[i] for i in keep], sims[keep]
    probs = torch.softmax(torch.tensor(sims * 100), 0).numpy()
    pq = aes.forward([{"path": str(path)}])[0]["PQ"]
    rms = float(np.sqrt(np.mean(x16 ** 2)))
    why = []
    for g, (names, thr) in FORBID.items():
        v = max(p[lab[n]] for n in names if n in lab)
        if g not in ALLOW[element] and v > thr:
            why.append(f"AST: {g} {v:.2f}")
    if EXPECT[element]:
        lead = next(i for i in np.argsort(-p) if ast.config.id2label[i] not in GENERIC)
        got = max((n for n in EXPECT[element] if n in lab), key=lambda n: p[lab[n]])
        if p[lab[got]] < max(EXPECT_MIN, EXPECT_SHARE * p[lead]):
            why.append(f"AST: слышно {ast.config.id2label[lead]} {p[lead]:.2f}, а {got} только {p[lab[got]]:.2f}")
    k = int(np.argmax(probs[1:])) + 1
    if probs[k] > 0.5:
        why.append(f"CLAP: «{texts[k]}» {probs[k]:.2f}")
    if rms < QUIET:
        why.append(f"тихо: RMS {rms:.4f}")
    top = ", ".join(f"{ast.config.id2label[i]} {p[i]:.2f}" for i in np.argsort(-p)[:3])
    badness = len(why) + max(0.0, probs[k] - 0.5)
    return dict(path=path, sim=float(sims[0]), pq=float(pq), rms=rms, why=why, top=top, badness=badness,
                score=float(sims[0]) + 0.02 * float(pq))


def tag(x):
    """The same print of a description as sound_scene.py puts in a take's name."""
    seconds = 30 if x in plan["beds"] else x["seconds"]
    return hashlib.sha1(f"{x['prompt']}|{x.get('negative', '')}|{seconds}".encode()).hexdigest()[:6]


# только кандидаты нынешнего описания (и синтезированные): после исправления
# плана старые, отвергнутые вместе с описанием, в отбор не идут — иначе
# снова выигрывала дверь «на железных петлях»
current = {x["id"]: tag(x) for x in ELEMENTS}
groups = defaultdict(list)
for f in sorted(cand.glob("*__*.wav")):
    el, variant = f.stem.split("__", 1)
    if el in current and (variant.startswith(current[el]) or variant.startswith("dsp")):
        groups[el].append(f)

lines = ["# Машинный отбор", "", "| элемент | выбран | CLAP | PQ | забраковано |", "|---|---|---|---|---|"]
detail = []
verdict = {}
for el in TARGET:
    if not groups[el]:
        raise SystemExit(f"нет кандидатов для {el}")
    rs = [judge(f, el) for f in groups[el]]
    ok = [r for r in rs if not r["why"]]
    best = max(ok, key=lambda r: r["score"]) if ok else min(rs, key=lambda r: r["badness"])
    flag = "" if ok else " ⚠ все плохи, взят наименее плохой"
    shutil.copy(best["path"], chosen / f"{el}.wav")
    lines.append(f"| {el} | {best['path'].stem.split('__')[1]}{flag} | {best['sim']:.3f} | {best['pq']:.2f} | "
                 f"{len(rs) - len(ok)} из {len(rs)} |")
    verdict[el] = {"all_bad": not ok, "chosen": best["path"].stem,
                   "takes": [{"name": r["path"].stem, "why": r["why"], "heard": r["top"]} for r in rs]}
    detail += ["", f"## {el} — цель: {' | '.join(TARGET[el])}" + (f"; ждём: {', '.join(EXPECT[el])}" if EXPECT[el] else "")]
    for r in sorted(rs, key=lambda r: -r["score"]):
        mark = "✔" if r is best else ("✘" if r["why"] else "·")
        detail.append(f"- {mark} `{r['path'].stem}` CLAP {r['sim']:.3f}, PQ {r['pq']:.2f}, RMS {r['rms']:.3f} — "
                      f"{'; '.join(r['why']) or 'годен'}. AST: {r['top']}")
    print(lines[-1], flush=True)

report.write_text("\n".join(lines + detail) + "\n")
report.with_suffix(".json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
