from __future__ import annotations

import base64

from .client import read_wav
from .common import Case, training


def _needs_ffmpeg(test, r):
    if r.status >= 500 and "ffmpeg" in r.body.decode(errors="replace"):
        test.skipTest("the server has no ffmpeg")


class Speech(Case):
    def speak(self, **body):
        return self.c.post("/v1/audio/speech", json_body={"input": "Проверка.", **body})

    def test_wav(self):
        r = self.speak(response_format="wav")
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(r.headers["content-type"], "audio/wav")
        channels, _, seconds = read_wav(r.body)
        self.assertEqual(channels, 1)
        self.assertAlmostEqual(seconds, float(r.headers["x-audio-seconds"]), delta=0.01)
        self.assertTrue(r.headers["x-job-id"])
        self.assertTrue(r.headers["x-voice"])

    @training
    def test_length_follows_the_text(self):
        r = self.speak(response_format="wav")
        self.assertAlmostEqual(read_wav(r.body)[2], len("Проверка.") * 0.06, delta=0.01)

    def test_pcm_is_raw_samples(self):
        sr = read_wav(self.speak(response_format="wav").body)[1]
        pcm = self.speak(response_format="pcm")      # другой синтез — сверяем с его же длиной
        self.assertEqual(pcm.status, 200)
        self.assertAlmostEqual(len(pcm.body) / 2 / sr, float(pcm.headers["x-audio-seconds"]),
                               delta=0.01)

    def test_opus(self):
        r = self.speak(response_format="opus")
        _needs_ffmpeg(self, r)
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(r.body[:4], b"OggS")

    def test_speed_stretches(self):
        # один сид — один и тот же синтез до растяжения, иначе длины не сравнить
        base = read_wav(self.speak(response_format="wav", seed=7).body)[2]
        r = self.speak(response_format="wav", speed=0.8, seed=7)
        _needs_ffmpeg(self, r)
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertAlmostEqual(read_wav(r.body)[2], base / 0.8, delta=0.05)

    def test_language_as_code(self):
        self.assertEqual(self.speak(response_format="wav", language="en").status, 200)

    def test_unsupported_language_is_400(self):
        self.assertOpenAIError(self.speak(language="xx"), 400, "xx")

    def test_empty_input_is_400(self):
        r = self.c.post("/v1/audio/speech", json_body={"input": "   "})
        self.assertOpenAIError(r, 400, "empty")

    def test_unknown_voice_is_400(self):
        self.assertOpenAIError(self.speak(voice="no-such-voice"), 400, "no-such-voice")


class SpeechStream(Case):
    def test_audio_stream_as_wav(self):
        r = self.c.post("/v1/audio/speech", json_body={
            "input": "Первое предложение, с запятой. Второе предложение.",
            "response_format": "wav", "stream_format": "audio"})
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(r.body[:4], b"RIFF")
        self.assertGreaterEqual(int(r.headers["x-chunks"]), 2)
        sr = int(r.headers["x-sample-rate"])
        self.assertGreater(len(r.body) - 44, sr * 2 * 0.5)          # больше полусекунды звука

    def test_sse_deltas_then_done(self):
        events = list(self.c.events("/v1/audio/speech", method="POST", json_body={
            "input": "Первое предложение. Второе предложение.",
            "response_format": "wav", "stream_format": "sse"}))
        deltas = [e for e in events if e["type"] == "speech.audio.delta"]
        self.assertGreaterEqual(len(deltas), 1)
        self.assertEqual(base64.b64decode(deltas[0]["audio"])[:4], b"RIFF")
        self.assertEqual(events[-1]["type"], "speech.audio.done")
        self.assertEqual(events[-1]["chunks"], len(deltas))

    def test_audio_stream_refuses_container_formats(self):
        r = self.c.post("/v1/audio/speech", json_body={
            "input": "Текст.", "response_format": "opus", "stream_format": "audio"})
        self.assertOpenAIError(r, 400, "sse")
