"""Judge music takes with no human in the loop and pick the best per prompt.

    python scripts/music_judge.py prompts.json TAKES_DIR [TAKES_DIR ...]

Every TAKES_DIR is the output of one scripts/music_<model>.py: <id>.<k>.wav
with <id>.<k>.json beside it. Writes TAKES_DIR/judge.json (every take, every
number) and TAKES_DIR/judge.md (for people). Run it in the environment of
scripts/sound_judge.py (transformers, audiobox_aesthetics) with demucs added
(TORCH_HOME on D: for its weights); lyrics are heard
by the voicy server (./voicy hear), which must be up — and alone on the GPU,
so judge after generating, not beside it.

What is measured, per take:

- vocals — the share of seconds where the vocal stem Demucs (htdemucs)
  pulls out of the mix is within 15 dB of the mix. An instrumental bed with a
  voice in it is useless under narration; a song without one is no song.
  AST was tried first and could not tell a synth lead or a pad from humming
  (experiments/29); its vocal share is kept beside, as `vocal_ast`.
- expect / forbid — AST classes the prompt names, averaged over the windows.
- style — CLAP (larger_clap_music): the take against its own caption and the
  captions of the other prompts in the set; rank 1 means it sounds most like
  what was asked of it.
- lyrics — CER of the lyrics against the best-matching stretch of what
  Whisper hears, section tags and punctuation dropped. Only that stretch
  counts: songs repeat the chorus, and Whisper ends a song with a credit line
  it never heard («Субтитры создавал …»); plain CER read both as errors
  and went over 100% on lyrics sung word for word.
- quality — Audiobox Aesthetics: CE, CU, PC, PQ (1–10).
- length — seconds against the seconds asked.

Thresholds below are a first guess, not fitted on anyone's ratings yet
(experiments/28 fitted its own on 28 rated sounds — this needs the same).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ClapModel, ClapProcessor

import audiobox_aesthetics.infer as abx
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parent.parent
dev = "cuda"
VOCAL = {"Singing", "Male singing", "Female singing", "Choir", "Rapping", "Speech", "Child singing",
         "Vocal music", "A capella", "Humming", "Chant"}
VOCAL_TOP = 5                 # голос «есть» в окне, если вокальный класс в пятёрке сильнейших
MAX_VOCAL_IN_BED = 0.10       # доля секунд с голосом, после которой фон под речь — брак
MIN_VOCAL_IN_SONG = 0.25      # вступление, проигрыш и концовка — без голоса
STEM_DB = -15                 # голос «есть» в секунде, если стем не тише смеси на 15 дБ
MAX_CER = 0.5
FORBID_MAX = 0.15

prompts = json.loads(Path(sys.argv[1]).read_text())
dirs = [Path(d) for d in sys.argv[2:]]

fx = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
LABELS = ast.config.id2label
cp = ClapProcessor.from_pretrained("laion/larger_clap_music")
clap = ClapModel.from_pretrained("laion/larger_clap_music").to(dev).eval()
def read_wav(meta):
    """soundfile instead of torchaudio: the latter wants ffmpeg here."""
    x, sr = sf.read(meta["path"], dtype="float32", always_2d=True)
    return torch.from_numpy(x.mean(1, keepdims=True).T.copy()), sr


abx.read_wav = read_wav
aes = abx.initialize_predictor()
DEMUCS = get_model("htdemucs").to(dev).eval()
with torch.no_grad():
    t = cp(text=[p["caption"] for p in prompts], return_tensors="pt", padding=True).to(dev)
    TEXT = torch.nn.functional.normalize(clap.get_text_features(**t), dim=-1)


def mono(x, sr, to):
    x = x.mean(axis=1) if x.ndim == 2 else x
    return resample_poly(x, to, sr).astype(np.float32) if sr != to else x.astype(np.float32)


def windows(x, sr, sec=10.0):
    n = int(sr * sec)
    return [x[i:i + n] for i in range(0, max(len(x) - n // 2, 1), n)] or [x]


@torch.no_grad()
def ast_scores(x16):
    ws = windows(x16, 16000)
    inp = fx(ws, sampling_rate=16000, return_tensors="pt").to(dev)
    return torch.sigmoid(ast(**inp).logits).cpu().numpy()          # окна × классы


@torch.no_grad()
def clap_rank(x48, idx):
    ws = windows(x48, 48000)
    inp = cp(audios=ws, sampling_rate=48000, return_tensors="pt").to(dev)
    a = torch.nn.functional.normalize(clap.get_audio_features(**inp).mean(0, keepdim=True), dim=-1)
    sims = (a @ TEXT.T)[0].cpu().numpy()
    return int((sims > sims[idx]).sum()) + 1, float(sims[idx]), float(np.sort(sims)[-2] if sims.argmax() == idx else sims.max())


@torch.no_grad()
def vocal_share(x, sr):
    mix = torch.from_numpy(resample_poly(x, DEMUCS.samplerate, sr, axis=0).T.copy()).float()
    if mix.shape[0] == 1:
        mix = mix.repeat(2, 1)
    ref = mix.mean(0)
    stems = apply_model(DEMUCS, ((mix - ref.mean()) / (ref.std() + 1e-8))[None].to(dev), split=True, overlap=0.1)[0]
    voc = stems[DEMUCS.sources.index("vocals")].mean(0).cpu() * (ref.std() + 1e-8)
    n = DEMUCS.samplerate
    secs = [(voc[i:i + n].pow(2).mean().sqrt(), ref[i:i + n].pow(2).mean().sqrt()) for i in range(0, len(ref) - n + 1, n)]
    on = [float(v) > 1e-3 and 20 * np.log10(float(v) / (float(m) + 1e-9)) > STEM_DB for v, m in secs]
    return float(np.mean(on)) if on else 0.0


def norm_text(s):
    s = re.sub(r"\[[^\]]*\]", " ", s.lower()).replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def cer(ref, hyp):
    """Edit distance of ref to its best-matching substring of hyp, per ref char."""
    r, h = norm_text(ref).replace(" ", ""), norm_text(hyp).replace(" ", "")
    d = [0] * (len(h) + 1)                  # начало эталона — где угодно в расшифровке
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return min(d) / max(len(r), 1)       # и конец — тоже


def hear(path, lang):
    r = subprocess.run([str(ROOT / "voicy"), "hear", str(path), "--language", lang],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"voicy hear: {r.stderr.strip()}")
    return r.stdout.strip()


def judge(path, p, idx):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    meta = json.loads(path.with_suffix(".json").read_text())
    rms = float(np.sqrt((x ** 2).mean()))
    s = ast_scores(mono(x, sr, 16000))
    top = np.argsort(-s, axis=1)[:, :VOCAL_TOP]
    vocal_ast = float(np.mean([any(LABELS[i] in VOCAL for i in row) for row in top]))
    vocal = vocal_share(x, sr)
    mean = s.mean(0)
    by = {LABELS[i]: float(mean[i]) for i in range(len(mean))}
    rank, sim, rival = clap_rank(mono(x, sr, 48000), idx)
    q = aes.forward([{"path": str(path)}])[0]
    j = {"take": path.stem, "model": meta.get("model"), "wall": meta.get("wall"),
         "vram_peak_gb": meta.get("vram_peak_gb"), "seconds": round(len(x) / sr, 1),
         "asked": p["seconds"], "rms": round(rms, 4), "vocal": round(vocal, 2),
         "vocal_ast": round(vocal_ast, 2),
         "expect": {c: round(by.get(c, 0), 3) for c in p.get("expect", [])},
         "forbid": {c: round(by.get(c, 0), 3) for c in p.get("forbid", [])},
         "heard": [LABELS[i] for i in np.argsort(-mean)[:5]],
         "clap_rank": rank, "clap": round(sim, 3), "clap_margin": round(sim - rival, 3),
         "aes": {k: round(float(v), 2) for k, v in q.items()}}
    why = []
    if rms < 0.005:
        why.append("тишина")
    if p.get("lyrics"):
        j["transcript"] = hear(path, p.get("language", "ru"))
        j["cer"] = round(cer(p["lyrics"], j["transcript"]), 3)
        if vocal < MIN_VOCAL_IN_SONG:
            why.append(f"нет голоса ({vocal:.0%} секунд)")
        if j["cer"] > MAX_CER:
            why.append(f"текст не разобрать (CER {j['cer']:.0%})")
    elif vocal > MAX_VOCAL_IN_BED:
        why.append(f"голос в инструментале ({vocal:.0%} секунд)")
    why += [f"{c} {v:.2f}" for c, v in j["forbid"].items() if v > FORBID_MAX and not (c in VOCAL and p.get("lyrics"))]
    if j["seconds"] < 0.9 * p["seconds"]:
        why.append(f"короче на {p['seconds'] - j['seconds']:.0f} с")
    j["reject"] = why
    # из годных — ближе всех к своему описанию, ничьи решает качество записи
    j["score"] = round(-rank + 10 * j["clap_margin"] + j["aes"]["PQ"] / 10 - (j.get("cer", 0)), 3)
    return j


for d in dirs:
    rows = []
    for idx, p in enumerate(prompts):
        takes = sorted(d.glob(f"{p['id']}.*.wav"))
        js = [judge(t, p, idx) for t in takes]
        for j in js:
            print(f"{d.name}/{j['take']}: rank {j['clap_rank']} vocal {j['vocal']:.0%} "
                  f"cer {j.get('cer', '-')} PQ {j['aes']['PQ']} {'; '.join(j['reject']) or 'ok'}", flush=True)
        ok = [j for j in js if not j["reject"]]
        best = max(ok or js, key=lambda j: j["score"]) if js else None
        rows.append({"id": p["id"], "task": p["task"], "best": best and best["take"],
                     "accepted": bool(ok), "takes": js})
    (d / "judge.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    md = [f"# {d.name}", "", "| запрос | лучший | ранг CLAP | голос | CER | PQ | время | брак |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for j in r["takes"]:
            mark = "**" if j["take"] == r["best"] else ""
            md.append(f"| {r['id']} | {mark}{j['take']}{mark} | {j['clap_rank']} | {j['vocal']:.0%} | "
                      f"{j.get('cer', '—')} | {j['aes']['PQ']} | {j['wall']} с | {'; '.join(j['reject']) or '—'} |")
    (d / "judge.md").write_text("\n".join(md) + "\n")
