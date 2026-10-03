"""A blizzard without a model: noise whose band and loudness follow the gusts.
In experiments/28 every MOSS wind came out as an engine, hay, a breath or
noise; this one the user called fine, and the judge picked it on its own.

    python scripts/sound_wind.py OUT.wav [seconds] [seed]
"""
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import istft, stft

SR = 48000
NPER, HOP = 2048, 512


def gusts(frames: int, rng) -> np.ndarray:
    """0..1: a slow swell (tens of seconds) and gusts on top (a few seconds)."""
    fps = SR / HOP
    slow = gaussian_filter1d(rng.standard_normal(frames), 8 * fps)
    fast = gaussian_filter1d(rng.standard_normal(frames), 1.2 * fps)
    g = slow / np.abs(slow).max() * 0.6 + fast / np.abs(fast).max() * 0.6
    g = (g - g.min()) / (g.max() - g.min())
    return g ** 1.4                                  # порывы реже, затишья длиннее


def bump(f, center, octaves):
    with np.errstate(divide="ignore"):
        d = np.log2(np.maximum(f, 1.0)[:, None] / center[None, :])
    return np.exp(-0.5 * (d / octaves) ** 2)


def blizzard(seconds: float, seed: int, howl: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    f, _, z = stft(rng.standard_normal(n), SR, nperseg=NPER, noverlap=NPER - HOP)
    g = gusts(z.shape[1], rng)
    # широкий гул: середина полосы ходит 160→520 Гц вслед за порывом
    shape = bump(f, 160 * 2 ** (1.7 * g), 1.3) * (0.3 + 0.7 * g)[None, :]
    # низ, который не стихает: давление ветра на стены
    shape = shape + bump(f, np.full_like(g, 70.0), 0.9) * 0.35
    if howl:
        # свист в щелях: узкая полоса, громче на пике порыва
        shape = shape + bump(f, 520 * 2 ** (1.0 * g), 0.12) * (howl * g ** 2)[None, :]
    _, x = istft(z * shape, SR, nperseg=NPER, noverlap=NPER - HOP)
    x = x[:n].astype(np.float32)
    return x / np.abs(x).max() * 0.5


if __name__ == "__main__":
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    sf.write(sys.argv[1], blizzard(seconds, seed), SR, subtype="FLOAT")
