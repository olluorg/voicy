"""Request models shared by the compatible routes and the job routes.

They live here rather than next to one of the two because `from __future__ import
annotations` turns a parameter annotation into a string, and FastAPI resolves that
string against the *defining module's* globals. A model passed in as an argument
is invisible there, and every request silently degrades to query parameters.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    language: str = "Russian"
    # расширения, которых нет у OpenAI
    prepare: bool = Field(default=False, description="применить словарь произношений")
    legato: bool = Field(default=False, description="убрать запятые внутри коротких фраз")
    seed: int | None = None
