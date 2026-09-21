"""Has the speaker finished? Smart Turn v3 answers from the sound itself.

A pause alone is a poor signal for a voice agent: people stop mid-thought to
pick a word, and a fixed threshold either interrupts them or makes every reply
late. Smart Turn listens to the last eight seconds of the turn — intonation,
rhythm, the shape of the final syllable — and returns the probability that the
turn is complete.

It is a Whisper-tiny encoder with a linear head, 8 MB quantised, and runs on
the CPU in about 20 ms. That matters more than size: it never waits for the
recogniser's lock or the GPU, so an agent learns about the end of a turn even
while a long file is being transcribed.

Model card: https://huggingface.co/pipecat-ai/smart-turn-v3 (BSD-2-Clause).
The vendor's benchmark puts Russian at 94.4% accuracy, 3.4% false "complete".
"""
from __future__ import annotations

import os
import threading

import numpy as np

REPO = "pipecat-ai/smart-turn-v3"
FILE = os.environ.get("TURN_MODEL_FILE", "smart-turn-v3.2-cpu.onnx")
SR = 16000
WINDOW = 8                              # с — столько модель видит, считая от конца


class SmartTurn:
    name = "smart-turn"
    device = "cpu"
    sample_rate = SR

    def __init__(self, repo: str = REPO, filename: str = FILE):
        self.repo, self.filename = repo, filename
        self.model = f"{repo}/{filename}"
        self._session = None
        self._features = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def status(self) -> dict:
        return {"engine": self.name, "model": self.model, "loaded": self.loaded,
                "device": self.device}

    def load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor

            path = hf_hub_download(self.repo, self.filename)
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 2     # 20 мс хватает и так; ядра нужнее серверу
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._features = WhisperFeatureExtractor(chunk_length=WINDOW)
            self._session = ort.InferenceSession(path, sess_options=so,
                                                 providers=["CPUExecutionProvider"])

    def probability(self, audio: np.ndarray) -> float:
        """16 kHz mono float32 of the turn so far, including the pause after it.

        Only the last eight seconds are looked at.
        """
        self.load()
        n = WINDOW * SR
        # Короткий звук добиваем нулями *в начале*: так модель обучали. Добивка
        # в конце, как по умолчанию делает извлекатель признаков, показывает ей
        # фразу и затем секунды тишины — и любая фраза выглядит законченной.
        audio = audio[-n:]
        if len(audio) < n:
            audio = np.pad(audio, (n - len(audio), 0))
        feats = self._features(audio, sampling_rate=SR, return_tensors="np",
                               padding="max_length", max_length=WINDOW * SR,
                               truncation=True, do_normalize=True).input_features
        out = self._session.run(None, {"input_features": feats.astype(np.float32)})[0]
        return float(np.asarray(out).reshape(-1)[0])     # выход уже после сигмоиды
