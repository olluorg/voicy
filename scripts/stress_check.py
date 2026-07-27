"""Measure where the stress actually landed.

Wrong stress is audible but not countable, and every previous round of this
project got unstuck by turning a perception into a number. Russian marks stress
by making the vowel longer and louder, so:

  1. Whisper gives word boundaries (word_timestamps).
  2. Inside each word the energy envelope is split into vowel nuclei.
  3. The nucleus carrying the most energy-weighted duration is the stressed one.
  4. RUAccent says which vowel *should* carry it; mismatches get listed.

The detector is approximate — it only sees loudness and length, not vowel
quality — so it is used to rank suspects for listening, not to declare a verdict.
"""
import json, re, sys, os, glob, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

V = "аеёиоуыэюя"


def accent_index(marked: str):
    """Position of the stressed vowel among the word's vowels, from '+' marks."""
    vs, idx, hit = 0, None, False
    for i, ch in enumerate(marked):
        if ch == "+":
            hit = True
            continue
        if ch.lower() in V:
            if hit:
                idx = vs
                hit = False
            vs += 1
    return idx, vs


def nuclei(x, sr, n_expected):
    """Split a word's audio into n energy peaks — a stand-in for vowel nuclei."""
    if n_expected <= 0 or len(x) < sr * 0.02:
        return []
    win = max(1, int(sr * 0.01))
    env = np.array([np.sqrt(np.mean(x[i:i + win] ** 2)) for i in range(0, len(x) - win, win)])
    if not len(env):
        return []
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    # cut into n_expected equal-ish segments by energy mass, then score each
    total = env.sum()
    if total <= 0:
        return []
    cuts, acc, seg = [], 0.0, 1
    for i, e in enumerate(env):
        acc += e
        if acc >= total * seg / n_expected and seg < n_expected:
            cuts.append(i); seg += 1
    bounds = [0] + cuts + [len(env)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = env[a:b]
        if not len(part):
            out.append(0.0); continue
        out.append(float(part.max() * len(part)))       # loudness × length
    return out


def main():
    audio = sys.argv[1]
    from ruaccent import RUAccent
    from onnxruntime import InferenceSession as _I
    _o = _I.run

    def _r(self, n, f, *a, **k):
        mm = {i.name for i in self.get_inputs()} - set(f)
        if mm and "input_ids" in f:
            f = {**f, **{x: np.zeros_like(f["input_ids"]) for x in mm}}
        return _o(self, n, f, *a, **k)

    _I.run = _r
    acc = RUAccent(); acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)

    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    segs, _ = m.transcribe(audio, language="ru", beam_size=5, word_timestamps=True)
    words = [w for s in segs for w in (s.words or [])]

    x, sr = sf.read(audio, dtype="float32", always_2d=True)
    x = x[:, 0]

    checked, wrong = 0, []
    for w in words:
        token = re.sub(r"[^А-Яа-яЁё-]", "", w.word)
        if len(token) < 3:
            continue
        marked = acc.process_all(token)
        exp, nv = accent_index(marked)
        if exp is None or nv < 2:
            continue
        a, b = int(w.start * sr), int(w.end * sr)
        seg = x[a:b]
        sc = nuclei(seg, sr, nv)
        if len(sc) != nv or max(sc) <= 0:
            continue
        got = int(np.argmax(sc))
        checked += 1
        if got != exp:
            wrong.append({"word": token, "marked": marked, "expected": exp, "got": got,
                          "at": round(w.start, 2)})

    rate = len(wrong) / checked * 100 if checked else 0
    print(f"{os.path.basename(audio):34} слов проверено {checked:4}  "
          f"расхождений {len(wrong):3}  ({rate:.0f}%)")
    for r in wrong[:12]:
        print(f"    {r['at']:6.1f}s  {r['word']:20} ждали слог {r['expected']+1}, "
              f"услышали {r['got']+1}   ({r['marked']})")
    return {"file": os.path.basename(audio), "checked": checked,
            "wrong": len(wrong), "rate": round(rate, 1), "items": wrong}


if __name__ == "__main__":
    out = [main()]
    prev = []
    if os.path.exists("results_stresscheck.json"):
        prev = json.load(open("results_stresscheck.json", encoding="utf-8"))
    names = {r["file"] for r in out}
    json.dump([r for r in prev if r["file"] not in names] + out,
              open("results_stresscheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
