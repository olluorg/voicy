"""ESpeech (F5-TTS, flow matching) behind the synthesiser interface (base.TTSEngine).

The engine voicy used before Qwen3-TTS (ADR 0006, ADR 0014), kept as the proof
that the engine contract is not Qwen's shape: it works nothing like Qwen.

F5 does not generate audio token by token. The length of the result is decided
before anything is generated — from the reference's rate of characters per
second — and then the whole mel spectrogram is refined from noise at once in
`nfe_step` steps of an ODE solver. So there are no "seconds produced so far":
progress is the share of solver steps done, and the total duration is known
exactly from the start. Both go to the server as such (base.Progress).

Russian only. The model reads stress from `+` before the vowel, and RUAccent
places it; without it the model guesses and guesses wrong on homographs.
Long text is cut by F5 itself into pieces that fit its 22-second window with
the reference, and the pieces are generated side by side on the same card.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .base import Progress, Synthesis, Unsupported
from .device import torch_device

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
CHECKPOINTS = {
    "ESpeech/ESpeech-TTS-1_RL-V2": "espeech_tts_rlv2.pt",
    "ESpeech/ESpeech-TTS-1_podcaster": "espeech_tts_podcaster.pt",
}
NFE_STEP = int(os.environ.get("ESPEECH_NFE_STEP", "32"))   # ADR 0006 брал 48; 32 — по умолчанию F5
CROSS_FADE = 0.15           # с — перекрытие соседних кусков


class ESpeechTTS:
    name = "espeech"
    sample_rate = 24000
    needs_reference_text = True     # F5 тоже клонирует по паре «образец + расшифровка»
    # Образец и кусок текста делят окно в 22 с, и сама F5 режет образец до 12 —
    # но тогда расшифровка перестаёт ему соответствовать. Режем не мы: длиннее
    # 12 с образец не принимается.
    reference_seconds = (3.0, 12.0)
    reference_best = (8.0, 12.0)

    def __init__(self, model: str = "ESpeech/ESpeech-TTS-1_RL-V2", device: str | None = None):
        if model not in CHECKPOINTS:
            raise SystemExit(f"TTS_MODEL={model}: espeech knows {', '.join(CHECKPOINTS)}")
        self.model = model
        self.device = device or torch_device()
        self._model = None
        self._vocoder = None
        self._accent = None
        self._lock = threading.Lock()
        self._on_call: Callable[[], None] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {"engine": self.name, "model": self.model, "loaded": self.loaded,
                "device": self.device, "nfe_step": NFE_STEP}

    def language(self, code: str | None) -> str | None:
        if code is None or not code.strip() or code.strip().lower() in ("ru", "russian"):
            return "ru"
        raise Unsupported(f"language '{code}' is not supported by {self.name}; supported: ru")

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
            from huggingface_hub import hf_hub_download
            from ruaccent import RUAccent

            ckpt = hf_hub_download(self.model, CHECKPOINTS[self.model])
            vocab = hf_hub_download(self.model, "vocab.txt")
            model = load_model(DiT, MODEL_CFG, ckpt, vocab_file=vocab, device=self.device)
            self._vocoder = load_vocoder(device=self.device)
            accent = RUAccent()
            accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
            self._accent = accent

            # Один проход трансформера — один шаг решателя (euler, CFG в одном
            # батче). Счётчик ничего не меняет в выводе.
            def count(*_):
                if self._on_call is not None:
                    self._on_call()
            model.transformer.register_forward_hook(count)
            self._model = model

    def speak(self, text: str, ref_audio: str | Path, ref_text: str,
              language: str | None = None, seed: int | None = None,
              on_progress: Callable[[Progress], None] | None = None) -> Synthesis:
        import soundfile as sf
        import torch
        from f5_tts.infer.utils_infer import (chunk_text, hop_length, infer_batch_process,
                                              target_sample_rate)

        self.language(language)
        self.load()
        with self._lock:
            ref_text = self._accent.process_all(ref_text.strip())
            if not ref_text.endswith(". "):
                ref_text = ref_text.rstrip(".") + ". "
            gen = self._accent.process_all(text.strip())

            data, sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T.copy())
            ref_seconds = audio.shape[-1] / sr
            lo, hi = self.reference_seconds
            if not lo <= ref_seconds <= hi:
                raise Unsupported(f"voice sample is {ref_seconds:.1f} s; "
                                  f"{self.name} takes {lo:.0f}–{hi:.0f} s")
            # так же режет infer_process: кусок вместе с образцом — до 22 с
            max_chars = int(len(ref_text.encode()) / ref_seconds * (22 - ref_seconds))
            pieces = chunk_text(gen, max_chars=max_chars)

            # Длительность F5 назначает заранее, по темпу образца. Расчёт — тот же,
            # что в infer_batch_process, включая пробел, который она дописывает
            # к расшифровке, и медленный темп для кусков короче 10 байт; минус
            # перекрытие на стыках кусков.
            rt = ref_text + " " if len(ref_text[-1].encode()) == 1 else ref_text
            ref_frames = int(ref_seconds * target_sample_rate) // hop_length
            frames = sum(int(ref_frames / len(rt.encode()) * len(p.encode())
                             / (1.0 if len(p.encode()) >= 10 else 0.3)) for p in pieces)
            total = (frames * hop_length / target_sample_rate
                     - CROSS_FADE * (len(pieces) - 1))
            calls, state = NFE_STEP * len(pieces), {"n": 0}

            def on_call() -> None:
                state["n"] += 1
                if on_progress is not None:
                    on_progress(Progress(done=min(1.0, state["n"] / calls), total=total))

            self._on_call = on_call
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                wave, out_sr, _ = next(infer_batch_process(
                    (audio, sr), ref_text, pieces, self._model, self._vocoder,
                    progress=None, nfe_step=NFE_STEP, cross_fade_duration=CROSS_FADE,
                    device=self.device))
            finally:
                self._on_call = None
        return Synthesis(np.asarray(wave, dtype=np.float32), int(out_sr),
                         {"pieces": len(pieces), "solver_steps": state["n"]})

