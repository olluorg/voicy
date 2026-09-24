from __future__ import annotations

from .common import Case, training


class Health(Case):
    def test_health_names_every_engine(self):
        h = self.c.get("/health").json()
        self.assertEqual(h["status"], "ok")
        for kind in ("tts", "stt", "turn", "vad"):
            self.assertIn("engine", h[kind], kind)
        for kind in ("tts", "stt", "turn"):
            self.assertIn("loaded", h[kind])
            self.assertIn("device", h[kind])
        self.assertIsInstance(h["stt"]["features"], list)
        self.assertIsInstance(h["voices"], list)
        self.assertIn("pending", h["queue"])

    @training
    def test_training_engines_are_on(self):
        h = self.c.get("/health").json()
        self.assertEqual([h[k]["engine"] for k in ("tts", "stt", "turn", "vad")],
                         ["tone", "script", "pause", "energy"])

    def test_models_list_like_openai(self):
        r = self.c.get("/v1/models").json()
        self.assertEqual(r["object"], "list")
        ids = {m["id"] for m in r["data"]}
        self.assertTrue({"tts-1", "whisper-1"} <= ids)


class Errors(Case):
    def test_unknown_route_in_openai_shape(self):
        self.assertOpenAIError(self.c.get("/v1/nothing-here"), 404)

    def test_validation_is_400_not_422(self):
        r = self.c.post("/v1/audio/speech", json_body={"voice": "turgenev"})
        self.assertOpenAIError(r, 400, "input")
        self.assertEqual(r.json()["error"]["param"], "input")

    def test_console_is_served(self):
        r = self.c.get("/")
        self.assertEqual(r.status, 200)
        self.assertIn(b"<html", r.body[:500].lower())


class TextPrepare(Case):
    def test_dictionary_rewrites_terms(self):
        r = self.c.post("/v1/text/prepare", json_body={"text": "API и CI/CD"}).json()
        self.assertTrue(r["changed"])
        self.assertTrue(r["replacements"])
        self.assertNotEqual(r["text"], "API и CI/CD")

    def test_nothing_to_do_changes_nothing(self):
        r = self.c.post("/v1/text/prepare", json_body={"text": "Просто текст."}).json()
        self.assertFalse(r["changed"])
        self.assertEqual(r["text"], "Просто текст.")

    def test_no_stress_marks_for_a_model_that_does_not_read_them(self):
        # знак U+0301 коверкает слово у модели, не обученной на нём (ADR 0015):
        # сервер ставит его, только если модель объявила, что его понимает (ADR 0023)
        health = self.c.get("/health").json()
        if health.get("tts", {}).get("stress_marks"):
            self.skipTest("модель понимает знаки ударения")
        r = self.c.post("/v1/text/prepare", json_body={"text": "Этот вопрос стоит разобрать."}).json()
        self.assertNotIn("́", r["text"])
        self.assertFalse(r.get("stress", False))
