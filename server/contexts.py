"""Recognition contexts: what the speaker is likely to say, and how to write it.

Whisper decodes better when it knows the vocabulary in advance. A context gives
it that in two ways, both native to faster-whisper:

  hotwords — terms put in front of the decoder's context window. They pull the
             decoding towards those spellings: "Kafka" rather than "кавка";
  prompt   — a sentence or two of the kind of speech expected. It sets style and
             punctuation more than vocabulary; measured, it barely moves terms.

After recognition, `replacements` rewrite what the model still gets wrong, word
for word. They are exact and never guessed, so a context cannot turn something
said into something expected.

A context is named and stored, like a voice, so a client passes `context=name`
instead of the whole list every time. `engineering` is built in: its hotwords
are the terms of the pronunciation dictionary (data/pronunciation.json).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import textprep
from voices import NAME

CONTEXTS_DIR = Path(os.environ.get("VOICY_CONTEXTS_DIR") or Path(__file__).parent / "contexts")
INDEX = CONTEXTS_DIR / "contexts.json"
BUILTIN = "engineering"


@dataclass
class Context:
    name: str
    note: str = ""
    prompt: str = ""
    hotwords: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    builtin: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _builtin() -> Context:
    terms = [t for t, _, _ in textprep._terms()]                     # noqa: SLF001
    return Context(name=BUILTIN, builtin=True,
                   note="термины словаря произношений: Java, Kafka, Spring, SQL…",
                   hotwords=sorted(terms, key=str.lower))


def _load() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return {}


def _save(idx: dict) -> None:
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")


def list_contexts() -> list[Context]:
    return [_builtin()] + [get(n) for n in sorted(_load())]


def get(name: str) -> Context | None:
    if name == BUILTIN:
        return _builtin()
    entry = _load().get(name)
    if entry is None:
        return None
    return Context(name=name, **{k: entry[k] for k in
                                 ("note", "prompt", "hotwords", "replacements") if k in entry})


def put(name: str, note: str = "", prompt: str = "", hotwords: list[str] | None = None,
        replacements: dict[str, str] | None = None) -> Context:
    if not NAME.match(name or ""):
        raise ValueError("name: 1–40 letters, digits, '-' or '_', not starting with '_'")
    if name == BUILTIN:
        raise ValueError(f"'{BUILTIN}' is built in and read-only")
    hotwords = [h.strip() for h in (hotwords or []) if h.strip()]
    replacements = {k.strip(): v for k, v in (replacements or {}).items() if k.strip()}
    idx = _load()
    idx[name] = {"note": note, "prompt": prompt, "hotwords": hotwords,
                 "replacements": replacements}
    _save(idx)
    return get(name)


def delete(name: str) -> bool:
    idx = _load()
    if name not in idx:
        return False
    del idx[name]
    _save(idx)
    return True


# ------------------------------------------------------------------ применение

@dataclass
class Resolved:
    """What one recognition call actually uses, after merging request and context.

    Prompt and terms stay apart: how to hand them to the model — which goes
    first, whether terms become part of the prompt — is the engine's business."""
    prompt: str | None = None
    hotwords: list[str] | None = None
    replacements: tuple = ()

    def fix(self, text: str) -> str:
        return _apply(text, self.replacements) if self.replacements and text else text


def resolve(context: str | None = None, prompt: str | None = None,
            hotwords: str | list[str] | None = None) -> Resolved:
    """Merge a named context with per-request `prompt` and `hotwords`.

    Raises KeyError for an unknown context name.
    """
    ctx = None
    if context:
        ctx = get(context)
        if ctx is None:
            raise KeyError(context)
    if isinstance(hotwords, str):
        hotwords = [h for h in re.split(r"[,\n]", hotwords)]
    fix = _compile(tuple(sorted((ctx.replacements if ctx else {}).items())))
    words: list[str] = []
    for h in (ctx.hotwords if ctx else []) + [h.strip() for h in (hotwords or [])]:
        if h and h not in words:
            words.append(h)
    prompts = " ".join(p.strip() for p in ((ctx.prompt if ctx else ""), prompt or "")
                       if p and p.strip())
    return Resolved(prompt=prompts or None, hotwords=words or None, replacements=fix)


@lru_cache(maxsize=64)
def _compile(pairs: tuple) -> tuple:
    # длинные первыми, чтобы «эс-ку-эль» не съел часть «ноу эс-ку-эль»
    ordered = sorted(pairs, key=lambda kv: -len(kv[0]))
    return tuple((re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst)
                 for src, dst in ordered)


def _apply(text: str, compiled: tuple) -> str:
    for pattern, dst in compiled:
        text = pattern.sub(dst, text)
    return text
