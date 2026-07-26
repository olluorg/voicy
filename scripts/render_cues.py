"""Cue-driven rendering: rhythm comes from the script, not from the model.

Each cue carries its own rate and the silence around it, so the narration slows
down on terms and on the sentence worth memorising, and speeds back up on the
connective tissue. Silence is inserted as real samples, so the pause is exact
and independent of whatever the model felt like doing.
"""
import json, sys, time, pathlib, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")

import numpy as np, soundfile as sf, torch, torchaudio


def _load_wav(p, *a, **k):
    d, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr


torchaudio.load = _load_wav

from onnxruntime import InferenceSession as _I

_o = _I.run


def _r(self, n, f, *a, **k):
    m = {i.name for i in self.get_inputs()} - set(f)
    if m and "input_ids" in f:
        f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in m}}
    return _o(self, n, f, *a, **k)


_I.run = _r

from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

# rate per cue kind: slower where the listener needs to catch a new idea
STYLE = {
    "narration": {"speed": 1.02, "pad_before": 120, "pad_after": 120},
    "term":      {"speed": 0.88, "pad_before": 420, "pad_after": 260},
    "key":       {"speed": 0.86, "pad_before": 480, "pad_after": 420},
    "question":  {"speed": 0.92, "pad_before": 300, "pad_after": 250},
}

CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path("cued"); OUT.mkdir(exist_ok=True)

acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
voc = load_vocoder().to("cuda")
model = load_model(CFG and DiT, CFG,
                   hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="espeech_tts_rlv2.pt"),
                   vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2", filename="vocab.txt")).to("cuda")

_rt = json.load(open("lv/ref_texts.json", encoding="utf-8"))
ref_txt = next(v for k, v in _rt.items() if "ref_ys" in k)
ref_a, ref_t = preprocess_ref_audio_text("lv/ref_ys.wav", acc.process_all(ref_txt))

doc = json.load(open("cues.json", encoding="utf-8"))
SR = None
parts = []
t0 = time.perf_counter()

for i, cue in enumerate(doc["cues"]):
    if cue["kind"] == "beat":
        parts.append(("silence", cue["ms"]))
        continue
    st = STYLE[cue["kind"]]
    torch.manual_seed(1234)
    wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(cue["text"]), model, voc,
                               cross_fade_duration=0.15, nfe_step=48, speed=st["speed"])
    SR = sr
    parts.append(("pad", st["pad_before"]))
    parts.append(("audio", np.asarray(wav, dtype=np.float32)))
    parts.append(("pad", st["pad_after"]))
    print(f"  [{i:2}] {cue['kind']:9} speed {st['speed']}  {len(wav)/sr:5.1f}s  {cue['text'][:52]}", flush=True)

chunks = []
for kind, val in parts:
    if kind == "audio":
        chunks.append(val)
    else:
        chunks.append(np.zeros(int(SR * val / 1000), dtype=np.float32))

audio = np.concatenate(chunks)
sf.write(str(OUT / "jc-024__cued.wav"), audio, SR)
wall = time.perf_counter() - t0
print(f"\njc-024__cued.wav  {len(audio)/SR:.1f}s аудио / {wall:.1f}s синтез  ×{len(audio)/SR/wall:.1f}")

# flat control: same text, one voice rate, no engineered pauses
flat = " ".join(c["text"] for c in doc["cues"] if c["kind"] != "beat")
torch.manual_seed(1234)
t0 = time.perf_counter()
wav, sr, _ = infer_process(ref_a, ref_t, acc.process_all(flat), model, voc,
                           cross_fade_duration=0.15, nfe_step=48, speed=1.0)
sf.write(str(OUT / "jc-024__flat.wav"), wav, sr)
print(f"jc-024__flat.wav  {len(wav)/sr:.1f}s аудио / {time.perf_counter()-t0:.1f}s синтез")
json.dump({"flat_text": flat}, open("cued/flat.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
