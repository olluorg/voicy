"""What every check shares: the server under test and how to reach it.

Two ways to run:

  - own servers (default): the checks start the training-engine server
    themselves and may change anything on it;
  - `--url`: an already running server, maybe with real models on someone
    else's machine. Checks that add voices or contexts are skipped there
    unless `--writes` is given, and checks that need the training engines'
    exact answers are skipped too.
"""
from __future__ import annotations

import functools
import unittest

from .client import Client
from .server import Server

CONFIG = {"url": None, "writes": False, "key": None}
_servers: dict[str, Server] = {}


def own() -> bool:
    return CONFIG["url"] is None


def server(variant: str = "main") -> Client:
    """`main` — plain; `strict` — with an API key and length-aware synthesis."""
    if not own():
        if variant != "main":
            raise unittest.SkipTest("needs its own server")
        return Client(CONFIG["url"], CONFIG["key"])
    if variant not in _servers:
        env = {"TONE_SPEED": "50", "VOICY_WEBHOOK_SECRET": "test-secret"}
        if variant == "strict":
            env.update(VOICY_API_KEY="k1,k2", TONE_KNOWS_LENGTH="1")
        _servers[variant] = Server(**env).start()
    s = _servers[variant]
    return Client(s.url, "k1" if variant == "strict" else None)


def stop_all() -> None:
    for s in _servers.values():
        s.stop()
    _servers.clear()


def writes(fn):
    """The check changes server state — not on someone else's server by default."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        if not own() and not CONFIG["writes"]:
            self.skipTest("changes server state; pass --writes to allow")
        return fn(self, *a, **kw)
    return wrapper


def training(fn):
    """The check knows the training engines' exact answers."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        if not own():
            self.skipTest("expects the training engines")
        return fn(self, *a, **kw)
    return wrapper


class Case(unittest.TestCase):
    variant = "main"

    @classmethod
    def setUpClass(cls):
        cls.c = server(cls.variant)

    def assertOpenAIError(self, r, status: int, contains: str = ""):
        self.assertEqual(r.status, status, r.body[:300])
        err = r.json().get("error")
        self.assertIsInstance(err, dict, r.body[:300])
        self.assertIn("message", err)
        self.assertIn("type", err)
        if contains:
            self.assertIn(contains, err["message"])
