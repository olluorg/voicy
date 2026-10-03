"""A plan and one chosen sound per element → the scene, by rules only.

    python scripts/sound_render.py plan.json CHOSEN_DIR OUT.wav REPORT.md

Rules taken from the hand-made mix the user preferred (experiments/28):
- time is the renderer's, not the planner's: 20 s of beds before anything
  happens, at least 2 s between actions unless the plan overlaps them on
  purpose, 45 s of beds after the last one;
- events are set by peak, not loudness: a short thud stays a thud instead of
  being pumped up to the loudness of a long creak;
- beds by LUFS, before perspective, so a wall takes off what a wall takes off;
  an "open" interval swaps the walled bed for the bare one;
- the scene is scaled to a -1 dBFS peak and keeps its dynamics; no target
  loudness, no limiter.
Each event is checked against the beds in its own window and lifted if it
drowns.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 48000
BED_LUFS = {"quiet": -44, "medium": -32, "loud": -24}
EVENT_PEAK = {"quiet": -15, "medium": -10, "loud": -6}
ESTABLISH, MIN_GAP, TAIL = 20.0, 2.0, 45.0
MARGIN_LU = {"quiet": 2, "medium": 5, "loud": 8}     # насколько событие должно выступать над фоном
MAX_LIFT_DB = 12                                      # больше событие не поднимается: значит, план плох
WALL_DB = -10
PEAK_DB = -1
meter = pyln.Meter(SR)
rng = np.random.default_rng(3)


def db(v):
    return 10 ** (v / 20)


def lufs(x):
    if len(x) < int(0.5 * SR):                       # измеритель берёт блоки по 400 мс
        x = np.concatenate([x, np.zeros(int(0.5 * SR) - len(x), np.float32)])
    v = meter.integrated_loudness(x)
    return v if np.isfinite(v) else -70.0


def to_lufs(x, target):
    return (x * db(target - lufs(x))).astype(np.float32)


def fade(x, a, b):
    a, b = int(a * SR), int(b * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def trim(x, floor_db=-45):
    """An event without the silence the model left around it."""
    env = np.convolve(np.abs(x), np.ones(480) / 480, mode="same")
    on = np.flatnonzero(env > np.abs(x).max() * db(floor_db))
    if not len(on):
        return x
    y = x[max(0, on[0] - 480): min(len(x), on[-1] + 4800)].copy()
    return fade(y, 0.005, 0.08)


def seamless(x, f=0.5):
    """Equal-power tail-into-head loop, as voicy-core's ambience::seamless."""
    n, f = len(x), int(f * SR)
    y = x[: n - f].copy()
    t = (np.arange(f) + 0.5) / f * np.pi / 2
    y[:f] = x[:f] * np.sin(t) + x[n - f:] * np.cos(t)
    return y


def lowpass(x, hz):
    return sosfilt(butter(4, hz, btype="low", fs=SR, output="sos"), x).astype(np.float32)


def room(x, wet_db=-16, tail=0.35):
    t = np.arange(int(tail * SR)) / SR
    ir = lowpass(rng.standard_normal(len(t)) * np.exp(-t / (tail / 3.5)), 3000)
    wet = fftconvolve(x, ir)[: len(x)]
    wet *= np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-12)
    return (x + wet * db(wet_db)).astype(np.float32)


def wall(x):
    """Log walls take the highs (a shelf, not silence above it — a lowpass
    turned wind into a rumbling engine) and a few dB; the room adds a tail."""
    low = lowpass(x, 400)
    return room(low + (x - low) * db(-24), wet_db=-14, tail=0.45) * db(WALL_DB)


def render(plan, src: Path):
    def load(name):
        x, sr = sf.read(src / f"{name}.wav", dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x

    events, ends = {}, {}
    for e in plan["events"]:
        x = trim(load(e["id"]))
        if len(x) > 6 * SR:
            fade(x, 0, 2.0)                           # длинное событие растворяется, а не обрывается
        x = x * (db(EVENT_PEAK[e["level"]]) / max(np.abs(x).max(), 1e-9))
        x = {"behind_wall": wall, "room": room, "near": lambda v: v}[e["perspective"]](x)
        if e["after"] is None:
            start = max(e["gap"], ESTABLISH)
        else:
            gap = e["gap"] if e["gap"] < 0 else max(e["gap"], MIN_GAP)   # наложение — только задуманное
            start = max(0.0, ends[e["after"]] + gap)
        events[e["id"]] = (start, x)
        ends[e["id"]] = start + len(x) / SR
    N = int((max(ends.values()) + TAIL) * SR)

    def place(track, x, at):
        a = int(at * SR)
        b = min(N, a + len(x))
        if b > a:
            track[a:b] += x[: b - a]

    beds = {}
    for b in plan["beds"]:
        start = max(0.0, (ends[b["start_after"]] if b["start_after"] else 0.0) + b["start_offset"])
        length = N - int(start * SR)
        if length <= SR:
            continue
        raw = load(b["id"])[int(1.0 * SR): -int(0.5 * SR)]
        bare = to_lufs(np.resize(seamless(raw), length).astype(np.float32), BED_LUFS[b["level"]])
        x = {"behind_wall": wall, "room": room, "near": lambda v: v}[b["perspective"]](bare)
        if b.get("open") and b["perspective"] == "behind_wall":
            # пока «открыто» — тот же фон без стены, переход за секунду
            t = np.arange(length) / SR
            mask = np.zeros(length, np.float32)
            for o in b["open"]:
                s_ev = events[o["from_event"]][0]
                a = s_ev + 0.4 * (ends[o["from_event"]] - s_ev) - start
                z = events[o["to_event"]][0] + 0.3 - start
                mask = np.maximum(mask, np.clip(np.minimum((t - a) / 1.0, (z - t) / 0.15), 0, 1))
            x = x * (1 - mask) + bare * mask
        fade(x, 3.0 if start == 0 else 2.0, 6.0)
        beds[b["id"]] = (start, x)

    bed_mix = np.zeros(N, np.float32)
    ev_mix = np.zeros(N, np.float32)
    for start, x in beds.values():
        place(bed_mix, x, start)

    def margin(e):
        """Event over the beds already sounding when it starts: a bed that
        begins inside the event (the fire after the fire catching) continues it."""
        start, x = events[e["id"]]
        a = int(start * SR)
        b = min(N, a + len(x))
        under = np.zeros(b - a, np.float32)
        for s0, y in beds.values():
            if s0 <= start:
                o = a - int(s0 * SR)
                seg = y[max(0, o): max(0, o) + (b - a)]
                under[: len(seg)] += seg
        return lufs(x[: b - a]) - lufs(under)

    # событие, которое тонет в фоне в своём окне, поднимается до нужного запаса
    lifted = {}
    for e in plan["events"]:
        short = MARGIN_LU[e["level"]] + 0.5 - margin(e)
        if short > 0:
            lift = min(short, MAX_LIFT_DB)
            start, x = events[e["id"]]
            events[e["id"]] = (start, x * db(lift))
            lifted[e["id"]] = lift
    for start, x in events.values():
        place(ev_mix, x, start)
    mix = bed_mix + ev_mix
    mix = (mix * (db(PEAK_DB) / max(np.abs(mix).max(), 1e-9))).astype(np.float32)

    lines = [f"# {plan['title']}", "",
             f"{N / SR:.0f} с, {lufs(mix):.1f} LUFS, пик {20 * np.log10(np.abs(mix).max()):.1f} dBFS",
             "", "| событие | с | до | над фоном, LU | нужно | поднято, дБ | |", "|---|---|---|---|---|---|---|"]
    for e in plan["events"]:
        start = events[e["id"]][0]
        m = margin(e)
        ok = m >= MARGIN_LU[e["level"]]
        lines.append(f"| {e['id']} | {start:.1f} | {ends[e['id']]:.1f} | {m:+.1f} | {MARGIN_LU[e['level']]} | "
                     f"{lifted.get(e['id'], 0):.1f} | {'✔' if ok else '⚠ тонет в фоне'} |")
    lines += ["", "| фон | с | перспектива | уровень |", "|---|---|---|---|"]
    for b in plan["beds"]:
        if b["id"] in beds:
            lines.append(f"| {b['id']} | {beds[b['id']][0]:.1f} | {b['perspective']} | {b['level']} |")
    return mix, "\n".join(lines) + "\n"


if __name__ == "__main__":
    plan = json.loads(Path(sys.argv[1]).read_text())
    mix, text = render(plan, Path(sys.argv[2]))
    sf.write(sys.argv[3], mix, SR, subtype="PCM_16")
    Path(sys.argv[4]).write_text(text)
    print(text)
