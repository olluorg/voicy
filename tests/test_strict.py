"""A second server: an API key, and synthesis that knows its length in advance."""
from __future__ import annotations

from .client import Client
from .common import Case


class ApiKey(Case):
    variant = "strict"

    def anonymous(self) -> Client:
        return Client(self.c.url)

    def test_no_key_is_401(self):
        r = self.anonymous().get("/v1/models")
        self.assertOpenAIError(r, 401, "Missing API key")
        self.assertEqual(r.headers["www-authenticate"], "Bearer")

    def test_wrong_key_is_401(self):
        self.assertOpenAIError(Client(self.c.url, "nope").get("/v1/models"), 401, "Incorrect")

    def test_every_listed_key_works(self):
        for key in ("k1", "k2"):
            self.assertEqual(Client(self.c.url, key).get("/v1/models").status, 200, key)

    def test_key_in_query_for_browsers(self):
        self.assertEqual(self.anonymous().get("/v1/models?api_key=k2").status, 200)

    def test_health_and_console_stay_open(self):
        self.assertEqual(self.anonymous().get("/health").status, 200)
        self.assertEqual(self.anonymous().get("/").status, 200)

    def test_websocket_needs_the_key(self):
        from websockets.exceptions import InvalidStatus

        with self.assertRaises(InvalidStatus) as ctx:
            with self.anonymous().ws("/v1/audio/speech/stream"):
                pass
        self.assertEqual(ctx.exception.response.status_code, 403)
        with self.c.ws("/v1/audio/speech/stream") as ws:
            self.assertIn('"ready"', ws.recv(timeout=30))


class KnownLength(Case):
    """An engine that decides the length first (like F5) reports share of work
    and an exact length instead of seconds produced."""
    variant = "strict"

    def test_progress_is_share_and_exact_length(self):
        job = self.c.post("/v1/jobs/speech", json_body={
            "input": "Движок знает длину заранее. " * 10, "response_format": "wav"}).json()
        events = list(self.c.events(f"/v1/jobs/{job['id']}/events"))
        synth = [e for e in events if e.get("stage") == "synthesis"]
        self.assertTrue(synth)
        for e in synth:
            self.assertNotIn("produced", e)
            self.assertTrue(e["expected_exact"])
            self.assertLessEqual(e["done"], 1.0)
        self.assertEqual([e["done"] for e in synth], sorted(e["done"] for e in synth))
        self.assertAlmostEqual(synth[-1]["expected"], events[-1]["seconds"], delta=0.1)
