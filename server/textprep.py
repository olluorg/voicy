"""Optional text preparation before synthesis.

Everything here was established by measurement earlier in this repository:

  * terms are replaced with the spelling the engine reads correctly, taken from
    `data/pronunciation.json` (built automatically, 79 entries verified by
    round-trip recognition);
  * punctuation inside a phrase is what makes the model stop mid-sentence, so
    a "legato" pass removes commas the author did not intend as a pause.

It is off by default: a caller feeding plain text through the OpenAI-compatible
endpoint expects their string back, not a rewritten one.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "pronunciation.json"


@lru_cache(maxsize=1)
def _terms() -> list[tuple[str, re.Pattern, str]]:
    if not DICT_PATH.exists():
        return []
    raw = json.loads(DICT_PATH.read_text(encoding="utf-8")).get("terms", {})
    # длинные термины первыми, иначе "SQL" съест часть "NoSQL"
    items = sorted(raw.items(), key=lambda kv: -len(kv[0]))
    out = []
    for term, meta in items:
        say = meta.get("say")
        if not say:
            continue
        out.append((term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)"), say))
    return out


def apply_dictionary(text: str) -> tuple[str, list[str]]:
    """Replace known terms with their verified spellings."""
    hits: list[str] = []
    for term, pat, say in _terms():
        text, n = pat.subn(say, text)
        if n:
            hits.append(f"{term} → {say}" + (f" ×{n}" if n > 1 else ""))
    return text, hits


def legato(text: str) -> str:
    """Drop commas inside short sentences so the phrase is read in one breath."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[^\W\d_]+", sentence)
        if len(words) <= 12:
            sentence = re.sub(r",\s+", " ", sentence)
        out.append(sentence.strip())
    return " ".join(s for s in out if s)


def prepare(text: str, use_dictionary: bool = True, use_legato: bool = False) -> dict:
    original = text
    hits: list[str] = []
    if use_dictionary:
        text, hits = apply_dictionary(text)
    if use_legato:
        text = legato(text)
    return {"text": text, "changed": text != original, "replacements": hits}
