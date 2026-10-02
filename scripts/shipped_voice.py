"""The default shipped voice, for scripts that drive the Python engines directly.

The server keeps voices in its own directory; a measurement wants the same
reference every time, so it reads the one shipped with the engines crate.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

VOICES = Path(__file__).resolve().parents[1] / "rust" / "core" / "assets" / "voices"


def default_voice() -> SimpleNamespace:
    """`.path` — the reference clip, `.text` — its exact transcript."""
    index = json.loads((VOICES / "voices.json").read_text(encoding="utf-8"))
    name = index["_default"]
    return SimpleNamespace(name=name, path=VOICES / f"{name}.wav", text=index[name]["text"])
