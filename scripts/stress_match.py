"""Stress by template matching instead of syllable hunting.

Two attempts at locating the stressed syllable inside a signal have now failed —
energy peaks track plosive bursts and vowel openness, not prominence. This tries
the problem from the other side: never segment anything, just ask which of
several known renderings the unknown one resembles.

Silero honours an explicit stress position, so a word can be rendered with the
stress on each syllable in turn. Those become templates. The candidate is scored
against each by the shape of its loudness contour alone — resampled to a common
length and z-scored, so timbre and speaking rate drop out and only the rhythm of
prominence is left.

Validation is cross-speaker by construction: templates come from one Silero voice
and the test items from another, which is the same mismatch that will exist
between Silero templates and Qwen output. A method that cannot survive that is
not worth pointing at Qwen.
"""
import sys, json, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.signal import butter, sosfiltfilt

V = "аеёиоуыэюя"
SR = 48000
NPOINTS = 64


def vowels(word):
    return [i for i, ch in enumerate(word.lower()) if ch in V]


def contour(x, sr=SR):
    """Loudness contour: band-limited RMS, trimmed, resampled, z-scored."""
    sos = butter(4, [300, 3000], btype="band", fs=sr, output="sos")
    b = sosfiltfilt(sos, x)
    hop = max(1, int(sr * 0.005))
    env = np.array([np.sqrt(np.mean(b[i:i + hop] ** 2)) for i in range(0, len(b) - hop, hop)])
    if len(env) < 4 or env.max() <= 0:
        return None
    env = env / env.max()
    live = np.where(env > 0.10)[0]
    if len(live) < 4:
        return None
    env = env[live[0]:live[-1] + 1]
    env = np.interp(np.linspace(0, len(env) - 1, NPOINTS), np.arange(len(env)), env)
    s = env.std()
    return (env - env.mean()) / s if s > 1e-6 else None


def marked(word, k):
    vp = vowels(word)
    return word[:vp[k]] + "+" + word[vp[k]:]


def build_templates(model, word, speaker):
    out = {}
    for k in range(len(vowels(word))):
        a = model.apply_tts(text=marked(word, k) + ".", speaker=speaker,
                            sample_rate=SR, put_accent=False, put_yo=True)
        c = contour(a.numpy().astype(np.float32))
        if c is not None:
            out[k] = c
    return out


def classify(cand_contour, templates):
    if cand_contour is None or not templates:
        return None
    best, bd = None, np.inf
    for k, t in templates.items():
        d = float(np.mean((cand_contour - t) ** 2))
        if d < bd:
            best, bd = k, d
    return best


def main():
    import torch
    torch.set_num_threads(12)
    model = torch.package.PackageImporter("v5_ru.pt").load_pickle("tts_models", "model")

    WORDS = ["дженерики", "компилятор", "коллекция", "параметры", "стирание",
             "граница", "безопасность", "проверка", "элементы", "хранилище",
             "переменная", "исключение", "состояние", "разработка", "поведение"]
    TEMPLATE_SPK, TEST_SPK = "eugene", "kseniya"

    ok, total, rows = 0, 0, []
    for w in WORDS:
        tpl = build_templates(model, w, TEMPLATE_SPK)
        for k in range(len(vowels(w))):
            a = model.apply_tts(text=marked(w, k) + ".", speaker=TEST_SPK,
                                sample_rate=SR, put_accent=False, put_yo=True)
            got = classify(contour(a.numpy().astype(np.float32)), tpl)
            if got is None:
                continue
            total += 1
            hit = got == k
            ok += hit
            rows.append({"word": w, "placed": k, "detected": got, "hit": bool(hit)})
        print(f"  {w:14} {sum(1 for r in rows if r['word'] == w and r['hit'])}/"
              f"{sum(1 for r in rows if r['word'] == w)}", flush=True)

    acc = ok / total * 100 if total else 0
    chance = np.mean([1 / len(vowels(w)) for w in WORDS]) * 100
    print(f"\nмежголосовая точность: {ok}/{total} = {acc:.0f}%   (случайное угадывание ≈ {chance:.0f}%)")
    json.dump({"accuracy": round(acc, 1), "chance": round(chance, 1), "rows": rows},
              open("results_stressmatch.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return acc


if __name__ == "__main__":
    main()
