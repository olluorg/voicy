from __future__ import annotations

import hashlib
import hmac
import http.server
import json
import threading

import numpy as np

from .client import read_wav, silence, to_wav, tone
from .common import Case, own

LONG = "Длинный текст, чтобы задание успело поработать. " * 60


class Hook(http.server.BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Hook.received.append((dict(self.headers), body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


class Jobs(Case):
    def submit(self, text="Проверка задания.", **extra):
        r = self.c.post("/v1/jobs/speech", json_body={"input": text, "response_format": "wav", **extra})
        self.assertEqual(r.status, 202, r.body[:300])
        return r.json()

    def finished(self, job_id):
        return self.c.wait(lambda: (j := self.c.get(f"/v1/jobs/{job_id}").json())["state"]
                           in ("done", "error", "cancelled") and j)

    def test_speech_job_lifecycle(self):
        job = self.submit()
        self.assertIn(job["state"], ("queued", "running", "done"))
        self.assertIn("expected_seconds", job)
        self.assertTrue(job["links"]["events"].endswith(f"/v1/jobs/{job['id']}/events"))
        done = self.finished(job["id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual(set(done["result"]), {"voice", "seconds", "format", "content_type"})
        audio = self.c.get(f"/v1/jobs/{job['id']}/audio")
        self.assertEqual(audio.status, 200)
        self.assertAlmostEqual(read_wav(audio.body)[2], done["result"]["seconds"], delta=0.01)
        self.assertNotIn("x-decoding-steps", audio.headers)
        ids = [j["id"] for j in self.c.get("/v1/jobs").json()["data"]]
        self.assertIn(job["id"], ids)

    def test_events_measure_progress(self):
        job = self.submit(LONG[:600])
        events = list(self.c.events(f"/v1/jobs/{job['id']}/events"))
        self.assertEqual(events[-1]["stage"], "done")
        synth = [e for e in events if e.get("stage") == "synthesis"]
        self.assertTrue(synth, events[:3])
        for e in synth:
            self.assertNotIn("steps", e)
            self.assertIn("expected", e)
            self.assertIn("expected_exact", e)
            self.assertTrue("produced" in e or "done" in e, e)
        encoding = [e for e in events if e.get("stage") == "encoding"]
        self.assertTrue(encoding and encoding[0]["expected_exact"])

    def test_cancel_mid_synthesis(self):
        job = self.submit(LONG)
        self.c.wait(lambda: self.c.get(f"/v1/jobs/{job['id']}").json()["state"] == "running")
        r = self.c.request("DELETE", f"/v1/jobs/{job['id']}")
        self.assertEqual(r.status, 200, r.body[:300])
        self.assertEqual(self.finished(job["id"])["state"], "cancelled")
        self.assertOpenAIError(self.c.get(f"/v1/jobs/{job['id']}/audio"), 410)
        self.assertOpenAIError(self.c.request("DELETE", f"/v1/jobs/{job['id']}"), 409)

    def test_unknown_job_is_404(self):
        self.assertOpenAIError(self.c.get("/v1/jobs/nope"), 404)

    def test_transcription_job(self):
        clip = to_wav(np.concatenate([tone(1.0), silence(0.5)]))
        r = self.c.post("/v1/jobs/transcribe", form={"word_timestamps": "true"},
                        files={"file": ("clip.wav", clip, "audio/wav")})
        self.assertEqual(r.status, 202, r.body[:300])
        job = self.finished(r.json()["id"])
        self.assertEqual(job["state"], "done")
        srt = self.c.get(f"/v1/jobs/{job['id']}/result?format=srt")
        self.assertTrue(srt.body.startswith(b"1\n"))
        self.assertOpenAIError(self.c.get(f"/v1/jobs/{job['id']}/audio"), 400)
        self.assertOpenAIError(self.c.get(f"/v1/jobs/{job['id']}/result?format=docx"), 400)

    def test_bad_task_is_400(self):
        r = self.c.post("/v1/jobs/transcribe", form={"task": "summarize"},
                        files={"file": ("clip.wav", to_wav(tone(0.5)), "audio/wav")})
        self.assertOpenAIError(r, 400, "task")

    def test_bad_webhook_is_400(self):
        r = self.c.post("/v1/jobs/speech", json_body={"input": "x", "webhook_url": "ftp://x"})
        self.assertOpenAIError(r, 400, "webhook_url")

    def test_webhook_is_called_and_signed(self):
        if not own():
            self.skipTest("the server must reach back to this machine")
        httpd = http.server.HTTPServer(("127.0.0.1", 0), Hook)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            Hook.received.clear()
            job = self.submit(webhook_url=f"http://127.0.0.1:{httpd.server_port}/hook")
            self.c.wait(lambda: Hook.received)
            headers, body = Hook.received[0]
            payload = json.loads(body)
            self.assertEqual(payload["event"], "job.done")
            self.assertEqual(payload["id"], job["id"])
            want = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            self.assertEqual(headers["X-Voicy-Signature"], f"sha256={want}")
            summary = self.c.wait(lambda: (j := self.c.get(f"/v1/jobs/{job['id']}").json())
                                  ["webhook"]["delivered"] and j)
            self.assertEqual(summary["webhook"]["attempts"], 1)
        finally:
            httpd.shutdown()
