"""Does the phrase get read in one breath?

Two suspects for the chopping: the reference speaker's own habit of stopping,
and the comma, which the model reads as an instruction to pause. Both are tested
on the exact phrase that was reported.
"""
import json, sys, os, glob, warnings, itertools
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch, torchaudio


def _load_wav(p, *a, **k):
    d, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr


torchaudio.load = _load_wav
from onnxruntime import InferenceSession as _I

_o = _I.run


def _r(self, n, f, *a, **k):
    mm = {i.name for i in self.get_inputs()} - set(f)
    if mm and "input_ids" in f:
        f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in mm}}
    return _o(self, n, f, *a, **k)


_I.run = _r

from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

SR = 24000
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = "probe"; os.makedirs(OUT, exist_ok=True)

PHRASE_COMMA = "И вот причина, ради которой этот вопрос и задают."
PHRASE_PLAIN = "И вот причина ради которой этот вопрос и задают."

acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
model = load_model(DiT, CFG,
                   hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="espeech_tts_rlv2.pt"),
                   vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="vocab.txt")).to("cuda")

slow = json.load(open("lv/slow_texts.json", encoding="utf-8"))
old = json.load(open("lv/ref_texts.json", encoding="utf-8"))
refs = {}
for k, v in slow.items():
    refs[os.path.basename(k).replace("ref_s", "s").replace(".wav", "")] = (k.replace("\\", "/"), v)
for k, v in old.items():
    b = os.path.basename(k)
    if b == "ref_cdg.wav":
        refs["cdg (текущий)"] = (k.replace("\\", "/"), v)


def inner_pauses(x):
    win = int(SR * 0.005)
    env = np.array([np.abs(x[i:i + win]).max() for i in range(0, len(x) - win, win)])
    thr = env.max() * 0.02
    sil = env < thr
    # ignore leading and trailing silence — only breaks *inside* the phrase count
    live = np.where(~sil)[0]
    if not len(live):
        return 0, []
    sil = sil[live[0]:live[-1] + 1]
    runs, i, out = [], 0, []
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) >= 20:            # >=100 ms reads as a break
                out.append((j - i) * 5)
            i = j
        else:
            i += 1
    return len(out), out


print(f"{'референс':16} {'вариант':10} {'сек':>5} {'разрывов':>9}  длительности")
print("-" * 68)
rows = []
for name, (path, rtext) in sorted(refs.items()):
    ref_a, ref_t = preprocess_ref_audio_text(path, acc.process_all(rtext))
    for label, phrase in (("с запятой", PHRASE_COMMA), ("без запятой", PHRASE_PLAIN)):
        torch.manual_seed(1234)
        wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(phrase), model, voc,
                                   cross_fade_duration=0.15, nfe_step=48, speed=1.0)
        x = np.asarray(wav, dtype=np.float32)
        n, durs = inner_pauses(x)
        tag = f"{name}__{'comma' if label == 'с запятой' else 'plain'}".replace(" ", "_")
        sf.write(f"{OUT}/{tag}.wav", x, SR)
        rows.append({"ref": name, "variant": label, "sec": round(len(x) / SR, 1),
                     "breaks": n, "durs": durs, "file": f"{tag}.wav"})
        print(f"{name:16} {label:10} {len(x)/SR:5.1f} {n:9}  {durs}", flush=True)

json.dump(rows, open("results_probe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
