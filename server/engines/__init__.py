"""Engines by name. The server picks one of each kind from the environment.

A new model is a new module here with a class that satisfies base.py, plus one
line below. Modules are imported only when chosen, so an engine's dependencies
are needed only by those who run it.
"""
from __future__ import annotations

import importlib
import os

REGISTRY = {
    "tts": {"qwen3-tts": "engines.tts_qwen:QwenTTS",
            "espeech": "engines.tts_espeech:ESpeechTTS"},
    "stt": {"faster-whisper": "engines.stt_whisper:WhisperSTT"},
    "turn": {"smart-turn": "engines.turn_smart:SmartTurn"},
    "vad": {"silero": "engines.vad_silero:SileroVAD"},
}
DEFAULT = {"tts": "qwen3-tts", "stt": "faster-whisper", "turn": "smart-turn", "vad": "silero"}
MODEL_ENV = {"tts": "TTS_MODEL", "stt": "STT_MODEL"}


def create(kind: str):
    """`TTS_ENGINE`, `STT_ENGINE`, `TURN_ENGINE`, `VAD_ENGINE` choose the engine;
    `TTS_MODEL` and `STT_MODEL` its weights, when it has a choice of them."""
    name = os.environ.get(f"{kind.upper()}_ENGINE") or DEFAULT[kind]
    known = REGISTRY[kind]
    if name not in known:
        raise SystemExit(f"{kind.upper()}_ENGINE={name}: unknown engine; "
                         f"known: {', '.join(known)}")
    module, cls = known[name].split(":")
    kwargs = {}
    model = os.environ.get(MODEL_ENV.get(kind, ""), "")
    if model:
        kwargs["model"] = model
    return getattr(importlib.import_module(module), cls)(**kwargs)
