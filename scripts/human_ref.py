"""ESpeech with human (public-domain LibriVox) references.

Both model variants, since RL-V2 is the one tuned for intelligibility (CER 1.4 vs 2.5).
"""
import json, time, sys, pathlib, warnings
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
MODELS = {"podcaster": ("ESpeech/ESpeech-TTS-1_podcaster", "espeech_tts_podcaster.pt"),
          "rlv2": ("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt")}
NFE, SPEED = 48, 0.95

PROBE = ("Слушай вопрос: чем опасно объявление внешнего ключа с он делит каскад? "
         "Подумай и ответь вслух. Короткий ответ: удаление одной строки "
         "может рекурсивно удалить тысячи связанных записей.")

OUT = pathlib.Path("human"); OUT.mkdir(exist_ok=True)
refs = json.load(open("lv/ref_texts.json", encoding="utf-8"))

acc = RUAccent()
acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
vocoder = load_vocoder().to("cuda")
gen_text = acc.process_all(PROBE)

results = []
for tag, (repo, fname) in MODELS.items():
    model = load_model(DiT, MODEL_CFG,
                       hf_hub_download(repo_id=repo, filename=fname),
                       vocab_file=hf_hub_download(repo_id=repo, filename="vocab.txt")).to("cuda")
    for path, text in refs.items():
        voice = pathlib.Path(path).stem.replace("ref_", "")
        try:
            ref_a, ref_t = preprocess_ref_audio_text(path, acc.process_all(text))
            torch.manual_seed(1234)
            t0 = time.perf_counter()
            wav, sr, _ = infer_process(ref_a, ref_t, gen_text, model, vocoder,
                                       cross_fade_duration=0.15, nfe_step=NFE, speed=SPEED)
            wall = time.perf_counter() - t0
            name = f"human__{voice}__{tag}.wav"
            sf.write(str(OUT / name), wav, sr)
            dur = len(wav) / sr
            results.append({"voice": voice, "model": tag, "file": name,
                            "audio_s": round(dur, 2), "wall_s": round(wall, 2),
                            "rtf": round(dur / wall, 2)})
            print(f"{name:34} {dur:5.1f}s / {wall:5.1f}s  ×{dur/wall:.1f}", flush=True)
        except Exception as e:
            print(f"{voice}/{tag} FAILED: {e}", flush=True)
    del model
    torch.cuda.empty_cache()

json.dump(results, open("results_human.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
