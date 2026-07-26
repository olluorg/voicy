"""Parameter sweep for ESpeech: attack the 'eats letters' problem.

Levers, in expected order of impact:
  ref      — F5 clones articulation from the reference, not just timbre
  nfe_step — diffusion steps; more steps = cleaner consonants
  speed    — slower gives each phoneme more room (and is easier to follow on a walk)
"""
import json, time, sys, pathlib, itertools, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load_wav(path, *a, **kw):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load_wav

from onnxruntime import InferenceSession as _ISess

_orig_run = _ISess.run


def _run(self, out_names, feed, *a, **kw):
    missing = {i.name for i in self.get_inputs()} - set(feed)
    if missing and "input_ids" in feed:
        feed = {**feed, **{m: np.zeros_like(feed["input_ids"]) for m in missing}}
    return _orig_run(self, out_names, feed, *a, **kw)


_ISess.run = _run

from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
OUT = pathlib.Path("sweep"); OUT.mkdir(exist_ok=True)

# short probe containing exactly the phrase that broke ("Слушай вопрос")
PROBE = ("Слушай вопрос: чем опасно объявление внешнего ключа с он делит каскад? "
         "Подумай и ответь вслух. Короткий ответ: удаление одной строки "
         "может рекурсивно удалить тысячи связанных записей.")

REFS = {
    "silero4-short": ("ref_eugene.wav", "ref_text.txt"),
    "vosk-clear": ("ref_vosk.wav", "ref_text2.txt"),
    "silero5": ("ref_silero5.wav", "ref_text2.txt"),
}

acc = RUAccent()
acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
print("accented probe:", acc.process_all(PROBE)[:120], flush=True)

vocoder = load_vocoder().to("cuda")
mp = hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_podcaster", filename="espeech_tts_podcaster.pt")
vp = hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_podcaster", filename="vocab.txt")
model = load_model(DiT, MODEL_CFG, mp, vocab_file=vp).to("cuda")

gen_text = acc.process_all(PROBE)
results = []

for ref_tag, (ref_wav, ref_txt) in REFS.items():
    ref_a, ref_t = preprocess_ref_audio_text(
        ref_wav, acc.process_all(open(ref_txt, encoding="utf-8").read().strip()))
    for nfe, speed in itertools.product((32, 64), (1.0, 0.9)):
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        wav, sr, _ = infer_process(ref_a, ref_t, gen_text, model, vocoder,
                                   cross_fade_duration=0.15, nfe_step=nfe, speed=speed)
        wall = time.perf_counter() - t0
        name = f"probe__{ref_tag}__nfe{nfe}__sp{speed}.wav"
        sf.write(str(OUT / name), wav, sr)
        dur = len(wav) / sr
        results.append({"ref": ref_tag, "nfe": nfe, "speed": speed, "file": name,
                        "audio_s": round(dur, 2), "wall_s": round(wall, 2), "rtf": round(dur / wall, 2)})
        print(f"{name:46} {dur:5.1f}s / {wall:5.1f}s  ×{dur/wall:.1f}", flush=True)

json.dump(results, open("results_sweep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
