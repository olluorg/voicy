"""A stress detector that is validated before it is believed.

The previous attempt split a word's energy into equal-mass chunks and called the
loudest one the stress. It reported 70% errors on a human reading where the
stress is correct by definition — it was measuring the plosive burst at word
onset, not prominence.

This one differs in two ways. It finds vowel nuclei as peaks of a band-limited
envelope (vowels carry 300–3000 Hz) instead of cutting by cumulative energy, and
it scores prominence as peak loudness times the width of the peak, which is what
Russian stress actually marks.

More importantly it comes with a validation harness. Silero accepts an explicit
stress position via `+`, so a word can be synthesised with the stress deliberately
placed on each syllable in turn; a detector worth using must recover the position
it was given. Accuracy on that set is printed before any verdict is issued about
any other engine.
"""
import sys, json, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

V = "аеёиоуыэюя"


def vowel_positions(word: str):
    return [i for i, ch in enumerate(word.lower()) if ch in V]


def prominence_profile(x: np.ndarray, sr: int, n_vowels: int):
    """Per-nucleus prominence: peak loudness × peak width, in time order."""
    if n_vowels <= 0 or len(x) < sr * 0.05:
        return []
    sos = butter(4, [300, 3000], btype="band", fs=sr, output="sos")
    band = sosfiltfilt(sos, x)
    hop = max(1, int(sr * 0.005))
    env = np.array([np.sqrt(np.mean(band[i:i + hop] ** 2)) for i in range(0, len(band) - hop, hop)])
    if env.max() <= 0:
        return []
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    env = env / env.max()

    # drop leading/trailing silence so onsets do not masquerade as nuclei
    live = np.where(env > 0.12)[0]
    if len(live) < 3:
        return []
    env = env[live[0]:live[-1] + 1]

    peaks, props = find_peaks(env, height=0.15, distance=max(2, int(0.05 / 0.005)), width=1)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1][:n_vowels]
    sel = sorted(int(peaks[i]) for i in order)
    idx = {int(peaks[i]): k for k, i in enumerate(range(len(peaks)))}
    out = []
    for p in sel:
        j = int(np.where(peaks == p)[0][0])
        out.append(float(props["peak_heights"][j] * props["widths"][j]))
    return out


def detect(x: np.ndarray, sr: int, word: str):
    n = len(vowel_positions(word))
    prof = prominence_profile(x, sr, n)
    if len(prof) != n or n < 2:
        return None
    return int(np.argmax(prof))


# ---------------------------------------------------------------- validation

def validate():
    import torch, wave, io, tempfile, os
    import soundfile as sf
    torch.set_num_threads(12)
    model = torch.package.PackageImporter("v5_ru.pt").load_pickle("tts_models", "model")

    WORDS = ["дженерики", "компилятор", "коллекция", "параметры", "стирание",
             "граница", "безопасность", "проверка", "элементы", "хранилище"]
    ok, total, rows = 0, 0, []
    for w in WORDS:
        vp = vowel_positions(w)
        for k in range(len(vp)):
            marked = w[:vp[k]] + "+" + w[vp[k]:]
            try:
                a = model.apply_tts(text=marked + ".", speaker="eugene",
                                    sample_rate=48000, put_accent=True, put_yo=True)
            except Exception:
                continue
            x = a.numpy().astype(np.float32)
            got = detect(x, 48000, w)
            total += 1
            hit = (got == k)
            ok += hit
            rows.append({"word": w, "placed": k, "detected": got, "hit": bool(hit)})
    acc = ok / total * 100 if total else 0
    print(f"валидация детектора: {ok}/{total} = {acc:.0f}% верно восстановленных позиций")
    by_word = {}
    for r in rows:
        by_word.setdefault(r["word"], []).append(r["hit"])
    for w, hits in by_word.items():
        print(f"   {w:16} {sum(hits)}/{len(hits)}")
    json.dump({"accuracy": round(acc, 1), "rows": rows},
              open("results_detector_validation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return acc


if __name__ == "__main__":
    validate()
