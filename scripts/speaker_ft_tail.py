"""Read the tail a take left out: prototype outside the server (experiments/29).

C stops early: in 9 of its 12 broken takes the paragraph's last 5–42 words are
missing, and the sound is not short enough for a length rule to notice (C reads
slower than the reader). Here the take's own transcript (.measure.json) is
aligned with the text word by word; when the last MISSING or more words were
not heard, the rest of the text is synthesized on its own, same voice and seed,
and appended after a short pause. Takes that read to the end are copied as they
are, so the variant keeps all its files.

    python scripts/speaker_ft_tail.py Cg      # → eval/Cg_t, then speaker_ft_eval.py measure
"""
import io, json, os, re, shutil, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, requests, soundfile as sf

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
EVAL = os.path.join(WORK, "eval")
URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080").rstrip("/")
MISSING = 3      # слов в хвосте, с которых дочитываем
PAUSE = 0.25     # с тишины перед дочитанным: граница предложения или фразы
WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:-[А-Яа-яЁё]+)*")


def norm(w):
    return w.lower().replace("ё", "е")


def heard_until(text, transcript):
    """Index (in text words) after the last word the transcript has, and the words."""
    ws = list(WORD.finditer(text))
    a = [norm(m.group(0)) for m in ws]
    b = [norm(w) for w in WORD.findall(transcript)]
    last = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            last = i2
    return last, ws


def heard_words(path):
    """Whisper's words with times, from the server."""
    with open(path, "rb") as f:
        r = requests.post(URL + "/v1/audio/transcriptions", proxies={"http": None}, timeout=600,
                          files={"file": ("a.wav", f, "audio/wav")},
                          data={"language": "ru", "response_format": "verbose_json",
                                "timestamp_granularities[]": "word"})
    r.raise_for_status()
    return [w for s in r.json()["segments"] for w in s.get("words", [])]


def sentence_cut(text, ws, last, path):
    """Restart at the sentence the gap begins in, not mid-phrase: (first text word to
    read again, where to cut the take in seconds), or (last, None) when that
    sentence's first word cannot be found in the take."""
    s = max((m.end() for m in re.finditer(r"[.!?…]+[»\"]?\s", text[:ws[last].start()])), default=0)
    k0 = next(k for k, m in enumerate(ws) if m.start() >= s)
    if k0 == last:
        return last, None
    heard = heard_words(path)
    a = [norm(m.group(0)) for m in ws]
    b = [norm(w["word"].strip(" .,;:!?—–-«»\"'")) for w in heard]
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" and i1 <= k0 < i2:
            return k0, max(0.0, heard[j1 + k0 - i1]["start"] - 0.05)
    return last, None


def speak(text, seed):
    r = requests.post(URL + "/v1/audio/speech", proxies={"http": None}, timeout=1800,
                      json={"input": text, "voice": "turgenev", "response_format": "wav", "seed": seed})
    r.raise_for_status()
    return sf.read(io.BytesIO(r.content), dtype="float32")


def main(variant):
    out_dir = os.path.join(EVAL, variant + os.environ.get("TAIL_SUFFIX", "_t"))
    os.makedirs(out_dir, exist_ok=True)
    fixed = 0
    for meta in sorted(f for f in os.listdir(os.path.join(EVAL, variant)) if f.endswith(".measure.json")):
        base = meta[:-len(".measure.json")]
        src = os.path.join(EVAL, variant, base)
        text = json.load(open(src + ".json", encoding="utf-8"))["text"]
        m = json.load(open(src + ".measure.json", encoding="utf-8"))
        last, ws = heard_until(text, m["transcript"])
        shutil.copy(src + ".json", os.path.join(out_dir, base + ".json"))
        if len(ws) - last < MISSING:
            shutil.copy(src + ".wav", os.path.join(out_dir, base + ".wav"))
            continue
        k0, cut = sentence_cut(text, ws, last, src + ".wav")
        rest = text[ws[k0].start():]
        x, sr = sf.read(src + ".wav", dtype="float32")
        if cut is not None:
            x = x[:int(cut * sr)]
        y, sr2 = speak(rest, int(base.rsplit("_s", 1)[1]))
        assert sr == sr2
        sf.write(os.path.join(out_dir, base + ".wav"), np.concatenate([x, np.zeros(int(PAUSE * sr), np.float32), y]), sr)
        fixed += 1
        print(f"  {base}: не хватало {len(ws) - last} слов из {len(ws)}, дочитано с "
              f"{'начала предложения' if cut is not None else 'места обрыва'}: {len(ws) - k0} слов, {len(y) / sr:.1f} с",
              flush=True)
    print(f"{variant}: дочитано {fixed} дублей → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
