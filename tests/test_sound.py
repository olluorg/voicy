"""Sounds that are not speech: /v1/sound-generation in ElevenLabs' shape, and
the same request as a job. The training engine makes white noise, at most
10 s a call; anything longer is a loop the server makes from it."""
from __future__ import annotations

import io
import wave

import numpy as np

from .client import read_wav
from .common import Case, training


def samples(wav: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


class Sound(Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not cls.c.get("/health").json().get("sound"):
            raise cls.skipTest(cls, "no sound engine on this server")

    def make(self, query: str = "", **body):
        return self.c.post("/v1/sound-generation" + query, json_body={"text": "wind in the pines", **body})

    def test_health_names_the_engine(self):
        sound = self.c.get("/health").json()["sound"]
        self.assertIn("engine", sound)
        self.assertIn("loaded", sound)

    def test_wav_of_the_asked_length(self):
        r = self.make(duration_seconds=3, response_format="wav")
        self.assertEqual(r.status, 200, r.body[:300])
        channels, rate, seconds = read_wav(r.body)
        self.assertEqual(channels, 1)
        self.assertAlmostEqual(seconds, 3.0, delta=0.01)
        self.assertAlmostEqual(float(r.headers["x-audio-seconds"]), 3.0, delta=0.01)
        self.assertIn("x-job-id", r.headers)

    def test_mp3_by_default_as_elevenlabs(self):
        r = self.make(duration_seconds=1)
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(r.headers["content-type"], "audio/mpeg")

    def test_output_format_sets_the_rate(self):
        r = self.make("?output_format=pcm_16000", duration_seconds=2)
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertAlmostEqual(len(r.body) / 2 / 16000, 2.0, delta=0.01)

    def test_opus_takes_the_sound_rate(self):
        r = self.make(duration_seconds=1, response_format="opus")
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(r.body[:4], b"OggS")

    @training
    def test_longer_than_the_engine_is_a_faded_loop(self):
        r = self.make(duration_seconds=25, response_format="wav")
        self.assertEqual(r.status, 200, r.body[:300])
        x = samples(r.body)
        self.assertAlmostEqual(len(x) / 44100, 25.0, delta=0.01)
        edge, body = np.abs(x[:100]).max(), np.abs(x[44100:88200]).max()
        self.assertLess(edge, body / 10, "края не затухают")

    @training
    def test_loop_is_whole_repeats(self):
        r = self.make(duration_seconds=25, response_format="wav", loop=True)
        self.assertEqual(r.status, 200, r.body[:300])
        x = samples(r.body)
        self.assertAlmostEqual(len(x) / 44100, 25.0, delta=0.01)
        # 25 с из кусков не длиннее 9.5 с — три повтора одной петли
        third = len(x) // 3
        self.assertTrue(np.array_equal(x[:third], x[third:2 * third]))
        self.assertGreater(np.abs(x[:100]).max(), 0.05, "у петли края не гасятся")

    @training
    def test_seed_repeats_the_sound(self):
        a, b, c = (self.make(duration_seconds=1, response_format="wav", seed=s).body for s in (1, 1, 2))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    @training
    def test_russian_description_is_refused_by_an_english_engine(self):
        self.assertOpenAIError(self.c.post("/v1/sound-generation", json_body={"text": "скрип двери"}),
                               400, "English")

    def test_refused_up_front(self):
        self.assertOpenAIError(self.make(prompt_influence=0.5), 400, "prompt_influence")
        self.assertOpenAIError(self.make(duration_seconds=0.1), 400, "duration_seconds")
        self.assertOpenAIError(self.make(text=" "), 400, "empty")
        self.assertOpenAIError(self.make("?output_format=ulaw_8000"), 400, "ulaw")

    def test_job(self):
        r = self.c.post("/v1/jobs/sound", json_body={"text": "rain on a roof", "duration_seconds": 2,
                                                      "response_format": "wav"})
        self.assertEqual(r.status, 202, r.body[:300])
        job = r.json()
        self.assertEqual(job["kind"], "sound")
        events = list(self.c.events(f"/v1/jobs/{job['id']}/events"))
        self.assertEqual(events[-1]["stage"], "done")
        gen = [e for e in events if e.get("stage") == "generation"]
        self.assertTrue(gen and gen[-1]["done"] == 1.0 and gen[-1]["expected_exact"], events)
        done = self.c.get(f"/v1/jobs/{job['id']}").json()
        self.assertEqual(set(done["result"]), {"seconds", "format", "content_type"})
        audio = self.c.get(f"/v1/jobs/{job['id']}/audio")
        self.assertEqual(audio.status, 200)
        self.assertAlmostEqual(read_wav(audio.body)[2], 2.0, delta=0.01)
