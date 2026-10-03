"""Music: /v1/music in ElevenLabs' shape, the model picked by the request.

Music has no training engine: the models are real and gigabytes, so these
checks see what a server without them says, and what it refuses before the
queue. With `--url` against a server that has models, the generating checks
run too (MusicWithModel).
"""
from __future__ import annotations

import io
import wave

from .common import Case


class Music(Case):
    def models(self) -> dict:
        r = self.c.get("/v1/music/models")
        self.assertEqual(r.status, 200, r.body[:300])
        return r.json()

    def test_models_listed_with_their_cards(self):
        m = self.models()
        self.assertIn("models", m)
        self.assertIn("default", m)
        for card in m["models"]:
            for k in ("id", "sample_rate", "channels", "min_seconds", "max_seconds", "length",
                      "vocals", "instrumental", "languages", "license", "commercial", "default", "loaded"):
                self.assertIn(k, card, card.get("id"))

    def test_health_names_the_models(self):
        music = self.c.get("/health").json()["music"]
        self.assertIn("models", music)
        self.assertIn("default", music)

    def test_without_models_it_says_how_to_get_them(self):
        if self.models()["models"]:
            self.skipTest("this server has music models")
        r = self.c.post("/v1/music", json_body={"prompt": "calm piano"})
        self.assertEqual(r.status, 501)
        self.assertIn("setup music", r.body.decode())


class MusicWithModel(Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not cls.c.get("/v1/music/models").json()["models"]:
            raise cls.skipTest(cls, "no music model on this server")

    def make(self, **body):
        return self.c.post("/v1/music", json_body={"prompt": "calm ambient piano", **body})

    def test_refused_before_the_queue(self):
        for body, field in [({"prompt": " "}, "prompt"),
                            ({"model": "no-such-model"}, "model"),
                            ({"music_length_ms": 1000}, "music_length_ms"),
                            ({"language": "xx", "lyrics": "la la"}, "language"),
                            ({"composition_plan": {}}, "composition_plan")]:
            r = self.make(**body)
            self.assertEqual(r.status, 400, (body, r.body[:200]))
            self.assertEqual(r.json()["error"].get("param"), field, body)
        r = self.make(lyrics="la la", force_instrumental=True)
        self.assertEqual(r.status, 400)

    def test_stereo_wav_of_the_asked_length(self):
        r = self.make(duration_seconds=10, force_instrumental=True, response_format="wav", seed=1)
        self.assertEqual(r.status, 200, r.body[:300])
        with wave.open(io.BytesIO(r.body)) as w:
            self.assertEqual(w.getnchannels(), 2)
            self.assertEqual(w.getframerate(), 48000)
            self.assertAlmostEqual(w.getnframes() / 48000, 10.0, delta=0.1)
        self.assertIn("x-model", r.headers)
        self.assertIn("x-model-license", r.headers)
