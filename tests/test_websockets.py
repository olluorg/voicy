from __future__ import annotations

import json
import time

import numpy as np

from .client import silence, tone
from .common import Case, training


def _events(ws, until: str, timeout: float = 30):
    """JSON events up to and including `until`; binary frames are attached to
    the event before them as `_pcm`."""
    out, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        m = ws.recv(timeout=max(0.1, deadline - time.monotonic()))
        if isinstance(m, bytes):
            out[-1]["_pcm"] = out[-1].get("_pcm", b"") + m
            continue
        e = json.loads(m)
        out.append(e)
        if e["type"] in (until, "error"):
            return out
    raise AssertionError(f"no '{until}' in {timeout} s: {[e['type'] for e in out]}")


class SpeechSocket(Case):
    def test_tokens_in_audio_out(self):
        with self.c.ws("/v1/audio/speech/stream?sample_rate=16000") as ws:
            ready = json.loads(ws.recv(timeout=30))
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["sample_rate"], 16000)
            for token in "Сейчас проверю, работает ли поток. Второе предложение.".split(" "):
                ws.send(json.dumps({"type": "text", "text": token + " "}))
            ws.send(json.dumps({"type": "end"}))
            events = _events(ws, "done")
        audio = [e for e in events if e["type"] == "audio"]
        self.assertGreaterEqual(len(audio), 2)
        for e in audio:
            self.assertAlmostEqual(len(e["_pcm"]) / 2 / 16000, e["seconds"], delta=0.01)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["chunks"], len(audio))

    def test_cancel_drops_what_is_left(self):
        with self.c.ws("/v1/audio/speech/stream") as ws:
            ws.recv(timeout=30)
            ws.send(json.dumps({"type": "text", "text": "Очень длинная фраза. " * 40}))
            ws.send(json.dumps({"type": "flush"}))
            _events(ws, "synthesizing")
            ws.send(json.dumps({"type": "cancel"}))
            events = _events(ws, "cancelled")
            self.assertEqual(events[-1]["type"], "cancelled")
            ws.send(json.dumps({"type": "text", "text": "После отмены."}))
            ws.send(json.dumps({"type": "end"}))
            after = _events(ws, "done")
        spoken = [e["text"] for e in after if e["type"] == "audio"]
        self.assertIn("После отмены.", spoken)

    def test_bad_language_is_reported(self):
        with self.c.ws("/v1/audio/speech/stream?language=xx") as ws:
            e = json.loads(ws.recv(timeout=30))
        self.assertEqual(e["type"], "error")
        self.assertIn("xx", e["error"])


class LiveSocket(Case):
    def spoken(self, text: str) -> np.ndarray:
        """Speech from the server's own synthesiser, at 16 kHz: a real voice
        detector does not take a tone for speech."""
        with self.c.ws("/v1/audio/speech/stream?sample_rate=16000") as ws:
            ws.recv(timeout=30)
            ws.send(json.dumps({"type": "text", "text": text}))
            ws.send(json.dumps({"type": "end"}))
            events = _events(ws, "done", timeout=120)
        pcm = b"".join(e.get("_pcm", b"") for e in events)
        return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768

    def test_turn_from_start_to_end(self):
        speech = self.spoken("Проверка связи.")
        audio = np.concatenate([silence(0.5), speech, silence(1.5)])
        pcm = (audio * 32767).astype("<i2").tobytes()
        with self.c.ws("/v1/audio/transcriptions/stream?sample_rate=16000&language=ru") as ws:
            self.assertEqual(json.loads(ws.recv(timeout=30))["type"], "ready")
            step = 1600 * 2                             # 100 мс
            for i in range(0, len(pcm), step):
                ws.send(pcm[i:i + step])
                time.sleep(0.02)
            ws.send(json.dumps({"type": "end"}))
            events = _events(ws, "done", timeout=60)
        kinds = [e["type"] for e in events]
        self.assertIn("speech_start", kinds)
        self.assertIn("turn_end", kinds)
        start = next(e for e in events if e["type"] == "speech_start")
        self.assertTrue(0.4 <= start["at"] <= 0.5 + len(speech) / 16000, start)
        self.assertEqual(events[-1]["type"], "done")
        self.assertAlmostEqual(events[-1]["duration"], len(audio) / 16000, delta=0.05)
        self.assertTrue(events[-1]["text"])

    @training
    def test_training_turn(self):
        audio = np.concatenate([silence(0.5), tone(1.5), silence(1.5)])
        pcm = (audio * 32767).astype("<i2").tobytes()
        with self.c.ws("/v1/audio/transcriptions/stream?sample_rate=16000") as ws:
            ws.recv(timeout=30)
            for i in range(0, len(pcm), 3200):
                ws.send(pcm[i:i + 3200])
                time.sleep(0.02)
            ws.send(json.dumps({"type": "end"}))
            events = _events(ws, "done", timeout=60)
        turn = next(e for e in events if e["type"] == "turn_end")
        self.assertEqual(turn["reason"], "model")
        self.assertTrue(turn["text"])
        self.assertEqual(events[-1]["text"], turn["text"])

    def test_bad_context_is_reported(self):
        with self.c.ws("/v1/audio/transcriptions/stream?context=no-such-context") as ws:
            e = json.loads(ws.recv(timeout=30))
        self.assertEqual(e["type"], "error")
        self.assertIn("no-such-context", e["error"])
