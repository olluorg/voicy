"""Named voices.

A voice is a short reference clip plus its exact transcript — the model clones
from that pair, and a wrong transcript degrades every sentence it generates
(measured: 7.3% CER against 1.0% for the same clip cut on phrase boundaries).

Adding a voice means dropping a wav next to a json entry. If the transcript is
missing it is filled in by the speech recogniser rather than by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VOICES_DIR = Path(__file__).parent / "voices"
INDEX = VOICES_DIR / "voices.json"


@dataclass
class Voice:
    name: str
    path: Path
    text: str
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "note": self.note,
                "reference_text": self.text, "file": self.path.name}


def _load_index() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return {}


def list_voices() -> list[Voice]:
    idx = _load_index()
    out = []
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        meta = idx.get(wav.stem, {})
        out.append(Voice(name=wav.stem, path=wav,
                         text=meta.get("text", ""), note=meta.get("note", "")))
    return out


def get(name: str) -> Voice | None:
    for v in list_voices():
        if v.name == name:
            return v
    return None


def default() -> Voice | None:
    idx = _load_index()
    wanted = idx.get("_default")
    if isinstance(wanted, str):
        v = get(wanted)
        if v:
            return v
    voices = list_voices()
    return voices[0] if voices else None


def add(name: str, wav_bytes: bytes, text: str = "", note: str = "") -> Voice:
    """Register a clip. An empty transcript is filled in by the recogniser later."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    path = VOICES_DIR / f"{name}.wav"
    path.write_bytes(wav_bytes)
    idx = _load_index()
    idx[name] = {"text": text, "note": note}
    INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return Voice(name=name, path=path, text=text, note=note)


def set_text(name: str, text: str) -> None:
    idx = _load_index()
    entry = idx.setdefault(name, {})
    entry["text"] = text
    INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
