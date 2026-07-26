"""ESpeech (F5-TTS DiT finetune, ru) sample generation + speed measurement."""
import json, time, sys, pathlib, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import torch
import torchaudio

# torchaudio.load now routes through torchcodec, which needs FFmpeg *shared* libs
# (the local ffmpeg is a static build). Our refs are plain WAV — read them with soundfile.
def _load_wav(path, *a, **kw):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load_wav
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODELS = {
    "podcaster": ("ESpeech/ESpeech-TTS-1_podcaster", "espeech_tts_podcaster.pt"),
    "rlv2": ("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt"),
}
REF_AUDIO = "ref_eugene.wav"
REF_TEXT = open("ref_text.txt", encoding="utf-8").read().strip()

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else ''})", flush=True)

print("loading RUAccent...", flush=True)

# ruaccent's onnx graphs still require token_type_ids, but the bundled tokenizer
# (newer transformers) no longer emits it. Inject zeros for any required-but-missing input.
from onnxruntime import InferenceSession as _ISess
_orig_run = _ISess.run


def _run(self, out_names, feed, *a, **kw):
    want = {i.name for i in self.get_inputs()}
    missing = want - set(feed)
    if missing and "input_ids" in feed:
        feed = {**feed, **{m: np.zeros_like(feed["input_ids"]) for m in missing}}
    return _orig_run(self, out_names, feed, *a, **kw)


_ISess.run = _run

acc = RUAccent()
acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)

print("loading vocoder...", flush=True)
vocoder = load_vocoder()
if device == "cuda":
    vocoder.to(device)

samples = json.load(open("samples.json", encoding="utf-8"))
results = []

for tag, (repo, fname) in MODELS.items():
    print(f"\n=== {tag} ===", flush=True)
    mp = hf_hub_download(repo_id=repo, filename=fname)
    vp = hf_hub_download(repo_id=repo, filename="vocab.txt")
    model = load_model(DiT, MODEL_CFG, mp, vocab_file=vp)
    if device == "cuda":
        model.to(device)

    ref_a, ref_t = preprocess_ref_audio_text(REF_AUDIO, acc.process_all(REF_TEXT))

    for s in samples:
        text = acc.process_all(s["audio"])
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        wave_, sr, _ = infer_process(ref_a, ref_t, text, model, vocoder,
                                     cross_fade_duration=0.15, nfe_step=32, speed=1.0)
        wall = time.perf_counter() - t0
        path = OUT / f"{s['id']}__audio__espeech-{tag}.wav"
        sf.write(str(path), wave_, sr)
        dur = len(wave_) / sr
        results.append({"engine": "espeech", "voice": tag, "id": s["id"], "variant": "audio",
                        "chars": len(s["audio"]), "audio_s": round(dur, 2),
                        "wall_s": round(wall, 2), "rtf": round(dur / wall, 2), "file": path.name})
        print(f"{path.name:48} {dur:6.1f}s / {wall:6.1f}s  x{dur/wall:.1f}", flush=True)

    del model
    torch.cuda.empty_cache()

json.dump(results, open("results_espeech.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
