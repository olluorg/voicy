from __future__ import annotations

import re

import numpy as np

from .client import silence, to_wav, tone
from .common import Case, training, writes

# учебный распознаватель: слово на каждые полсекунды звука, тишина кончает сегмент
CLIP = to_wav(np.concatenate([tone(1.0), silence(1.0), tone(1.0)]))
HEARD = "проверка связи кавка раз"


class Transcription(Case):
    def hear(self, audio: bytes = CLIP, **form):
        return self.c.post("/v1/audio/transcriptions", form={"model": "whisper-1", **form},
                           files={"file": ("clip.wav", audio, "audio/wav")})

    def test_json(self):
        r = self.hear()
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertIsInstance(r.json()["text"], str)
        self.assertTrue(r.headers["x-job-id"])

    @training
    def test_training_text(self):
        self.assertEqual(self.hear().json()["text"], HEARD)

    def test_text(self):
        r = self.hear(response_format="text")
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        self.assertFalse(r.body.startswith(b"{"))

    def test_verbose_json(self):
        r = self.hear(response_format="verbose_json").json()
        self.assertEqual(r["task"], "transcribe")
        self.assertAlmostEqual(r["duration"], 3.0, delta=0.05)
        for i, s in enumerate(r["segments"]):
            self.assertEqual(s["id"], i)
            self.assertLessEqual(s["start"], s["end"])

    @training
    def test_segments_follow_pauses(self):
        r = self.hear(response_format="verbose_json").json()
        self.assertEqual([(s["start"], s["end"]) for s in r["segments"]], [(0.0, 1.0), (2.0, 3.0)])

    def test_srt(self):
        body = self.hear(response_format="srt").body.decode()
        self.assertRegex(body, r"^1\n\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}\n")

    def test_vtt(self):
        r = self.hear(response_format="vtt")
        self.assertTrue(r.headers["content-type"].startswith("text/vtt"))
        body = r.body.decode()
        self.assertTrue(body.startswith("WEBVTT\n"))
        self.assertRegex(body, r"\d\d:\d\d:\d\d\.\d{3} --> ")

    def test_word_timestamps(self):
        r = self.hear(response_format="verbose_json", timestamp_granularities="word").json()
        words = [w for s in r["segments"] for w in s["words"]]
        self.assertTrue(words)
        self.assertTrue(all(w["start"] <= w["end"] for w in words))

    def test_stream_deltas_then_done(self):
        events = list(self.c.events("/v1/audio/transcriptions", method="POST",
                                    form={"stream": "true"},
                                    files={"file": ("clip.wav", CLIP, "audio/wav")}))
        deltas = [e for e in events if e["type"] == "transcript.text.delta"]
        done = events[-1]
        self.assertEqual(done["type"], "transcript.text.done")
        self.assertEqual("".join(d["delta"] for d in deltas), done["text"])
        self.assertTrue(done["job_id"])

    def test_translation(self):
        r = self.c.post("/v1/audio/translations", form={"model": "whisper-1"},
                        files={"file": ("clip.wav", CLIP, "audio/wav")})
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertTrue(re.fullmatch(r"[A-Za-z ,.'!?-]*", r.json()["text"]), r.json()["text"])

    def test_not_audio_is_400(self):
        self.assertOpenAIError(self.hear(audio=b"definitely not audio"), 400, "decode")

    def test_empty_file_is_400(self):
        self.assertOpenAIError(self.hear(audio=b""), 400, "empty")

    def test_unknown_context_is_400(self):
        self.assertOpenAIError(self.hear(context="no-such-context"), 400, "no-such-context")

    def test_hotwords_accepted(self):
        self.assertEqual(self.hear(hotwords="Kafka, Helm").status, 200)


class Contexts(Case):
    def test_engineering_is_built_in(self):
        names = {c["name"]: c for c in self.c.get("/v1/contexts").json()["data"]}
        self.assertTrue(names["engineering"]["builtin"])
        self.assertIn("Kafka", names["engineering"]["hotwords"])

    def test_built_in_cannot_be_deleted(self):
        self.assertOpenAIError(self.c.request("DELETE", "/v1/contexts/engineering"), 400)

    @writes
    def test_create_use_delete(self):
        body = {"prompt": "Созвон.", "hotwords": ["Kafka"], "replacements": {"кавка": "Kafka"}}
        r = self.c.request("PUT", "/v1/contexts/selftest", json_body=body)
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(self.c.get("/v1/contexts/selftest").json()["replacements"],
                         {"кавка": "Kafka"})
        if self.c.get("/health").json()["stt"]["engine"] == "script":
            text = self.c.post("/v1/audio/transcriptions", form={"context": "selftest"},
                               files={"file": ("clip.wav", CLIP, "audio/wav")}).json()["text"]
            self.assertIn("Kafka", text)
            self.assertNotIn("кавка", text)
        self.assertEqual(self.c.request("DELETE", "/v1/contexts/selftest").status, 200)
        self.assertOpenAIError(self.c.get("/v1/contexts/selftest"), 404)

    @writes
    def test_bad_shape_is_400(self):
        r = self.c.request("PUT", "/v1/contexts/selftest-bad", json_body={"hotwords": "Kafka"})
        self.assertOpenAIError(r, 400, "hotwords")
