from __future__ import annotations

from .client import tone, to_wav
from .common import Case, training, writes

SAMPLE = to_wav(tone(9.0, sr=24000), sr=24000)


class Voices(Case):
    def add(self, name, audio=SAMPLE, **form):
        return self.c.post("/v1/voices", form={"name": name, **form},
                           files={"file": ("sample.wav", audio, "audio/wav")})

    def test_list_has_a_default(self):
        r = self.c.get("/v1/voices").json()
        names = [v["name"] for v in r["data"]]
        self.assertTrue(names)
        self.assertIn(r["default"], names)

    @writes
    def test_add_transcribes_the_sample_itself(self):
        r = self.add("selftest-voice")
        self.assertEqual(r.status, 200, r.body[:300])
        v = r.json()
        self.assertAlmostEqual(v["seconds"], 9.0, delta=0.05)
        self.assertTrue(v["reference_text"])            # распознаватель заполнил сам
        self.assertEqual(v["warnings"], [])
        names = [x["name"] for x in self.c.get("/v1/voices").json()["data"]]
        self.assertIn("selftest-voice", names)
        speech = self.c.post("/v1/audio/speech", json_body={
            "input": "Новым голосом.", "voice": "selftest-voice", "response_format": "wav"})
        self.assertEqual(speech.status, 200, speech.body[:300])
        self.assertEqual(speech.headers["x-voice"], "selftest-voice")

    @writes
    def test_existing_name_needs_replace(self):
        self.add("selftest-twice", text="раз")
        self.assertOpenAIError(self.add("selftest-twice", text="два"), 409, "replace")
        r = self.add("selftest-twice", text="два", replace="true")
        self.assertEqual(r.json()["reference_text"], "два")

    @writes
    def test_outside_best_length_warns(self):
        r = self.add("selftest-short", audio=to_wav(tone(4.0, sr=24000), sr=24000), text="x")
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertTrue(r.json()["warnings"])

    @training
    def test_too_short_is_400(self):
        r = self.add("selftest-tiny", audio=to_wav(tone(1.0, sr=24000), sr=24000), text="x")
        self.assertOpenAIError(r, 400, "1.0 s")

    def test_bad_name_is_400(self):
        self.assertOpenAIError(self.add("_hidden", text="x"), 400, "name")

    def test_not_audio_is_400(self):
        self.assertOpenAIError(self.add("selftest-junk", audio=b"junk", text="x"), 400, "decode")
