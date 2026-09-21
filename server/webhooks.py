"""Tell someone a job has finished, instead of making them hold a connection.

The call carries the job's summary and links, never the audio itself: a webhook
receiver is often a small service that wants to know *that* something is ready,
and megabytes in a notification make every retry expensive. The audio is fetched
from `links.audio` when it is wanted.

Delivery is retried with growing pauses. A receiver that answers 2xx has it;
anything else — a refusal, a timeout, nobody listening — counts as a miss.
With `VOICY_WEBHOOK_SECRET` set, the body is signed:

    X-Voicy-Signature: sha256=<hex HMAC-SHA256 of the raw body>
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from progress import Job

SECRET = os.environ.get("VOICY_WEBHOOK_SECRET", "")
PAUSES = (0, 2, 10, 30, 120)            # перед каждой попыткой, секунд
TIMEOUT = 10.0
LOCAL = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def validate(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("webhook_url must be an absolute http(s) URL")
    return url


def notify(job: Job) -> None:
    if job.webhook is None or job.webhook.get("started"):
        return
    job.webhook["started"] = True
    threading.Thread(target=_deliver, args=(job,), name=f"webhook-{job.id}",
                     daemon=True).start()


def _opener(url: str):
    # Локальный получатель — мимо прокси, иначе HTTP_PROXY превратит вызов в 503.
    if (urlsplit(url).hostname or "") in LOCAL:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _deliver(job: Job) -> None:
    import jobs                                     # сводка живёт рядом с маршрутами

    hook = job.webhook
    body = json.dumps({"event": f"job.{job.state}", **jobs.summary(job)},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "voicy-webhook",
               "X-Voicy-Event": f"job.{job.state}", "X-Voicy-Job": job.id}
    if SECRET:
        digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Voicy-Signature"] = f"sha256={digest}"

    for pause in PAUSES:
        time.sleep(pause)
        hook["attempts"] = hook.get("attempts", 0) + 1
        req = urllib.request.Request(hook["url"], data=body, headers=headers, method="POST")
        try:
            with _opener(hook["url"]).open(req, timeout=TIMEOUT) as r:
                if 200 <= r.status < 300:
                    hook["delivered"], hook["error"] = True, None
                    return
                hook["error"] = f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            hook["error"] = f"HTTP {e.code}"
        except Exception as e:                      # noqa: BLE001
            hook["error"] = str(e) or type(e).__name__
    hook["delivered"] = False
