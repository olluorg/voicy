"""«не́ было» or «не бы́ло»: does a mark on «не» move the stress, and toward the reader?

Russian stresses the particle in «не́ был», «не́ было», «не́ были» (but «не была́»).
RUAccent marks the verb («не бы́ло»), and the model says so. Neither meter2 (it
judges words of two vowels and up, and here «было» is the unstressed one) nor
Whisper can hear the difference, so it is measured directly: loudness (dB) and
length of «не» against the verb that follows, from Whisper's word times.

The reader of «Дворянское гнездо» gives the target: every «не был/было/были»
in the paragraphs of experiments/29. Each such sentence is then synthesized
twice, seeds 1–3: as the server marks it, and with the mark moved to «не»
(sent with stress off, so nothing is re-marked).

    python scripts/speaker_ft_nebylo.py        # needs the voicy server
"""
import io, json, os, re, sys
from statistics import median
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, requests, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from speaker_ft_ruaccent import WORK, sentences, spans

URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080").rstrip("/")
OUT = os.path.join(WORK, "nebylo")
NE = re.compile(r"(?<![А-Яа-яЁё])([Нн]е)\s+(был[оиа]?)(?![А-Яа-яЁё])")
ACUTE = "́"


def post(path, **kw):
    r = requests.post(URL + path, proxies={"http": None}, timeout=600, **kw)
    r.raise_for_status()
    return r


def words(x, sr):
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV"); buf.seek(0)
    j = post("/v1/audio/transcriptions", files={"file": ("a.wav", buf, "audio/wav")},
             data={"language": "ru", "response_format": "verbose_json", "timestamp_granularities[]": "word"}).json()
    return [w for s in j["segments"] for w in s.get("words", [])]


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def features(x, sr, heard, verb):
    """(dB of «не» minus dB of the verb, length of «не» / length of the verb) or None."""
    w = [(re.sub(r"[^а-яё]", "", h["word"].lower()), h["start"], h["end"]) for h in heard]
    for k in range(len(w) - 1):
        if w[k][0] == "не" and w[k + 1][0] == verb.lower():
            ne = x[int(w[k][1] * sr):int(w[k][2] * sr)]
            vb = x[int(w[k + 1][1] * sr):int(w[k + 1][2] * sr)]
            if len(ne) > sr * 0.03 and len(vb) > sr * 0.05:
                return float(db(ne) - db(vb)), len(ne) / len(vb)
    return None


def moved(marked):
    """«не бы́ло» → «не́ было»: the mark leaves the verb for the particle (не была́ stays)."""
    def fix(m):
        verb = m.group(2)
        return m.group(0) if verb.replace(ACUTE, "").lower() == "была" else f"{m.group(1)}{ACUTE} {verb.replace(ACUTE, '')}"
    return re.sub(r"(?<![А-Яа-яЁё])([Нн]е)\s+(б́?ы́?л[оиа]?́?)(?![А-Яа-яЁё])", fix, marked)


def main():
    os.makedirs(OUT, exist_ok=True)
    segs = json.load(open(os.path.join(WORK, "ds", "segments.json"), encoding="utf-8"))
    human, items, audio = [], [], {}
    for seg in segs:
        sents = sentences(seg["text"])
        hits = [s for s in sents if NE.search(s) and NE.search(s).group(2).lower() != "была"]
        if not hits:
            continue
        ch = seg["chapter"]
        if ch not in audio:
            audio = {ch: sf.read(os.path.join(WORK, "audio", "wav24", f"dvoryanskoegnezdo_{ch}.wav"), dtype="float32")}
        x, sr = audio[ch]
        heard = [w for s in json.load(open(os.path.join(WORK, "audio", "asr", f"{ch}.json"), encoding="utf-8"))["segments"]
                 for w in s.get("words", []) if seg["start"] - 1 <= w["start"] <= seg["end"] + 1]
        for s, sp in zip(sents, spans(sents, heard)):
            if s not in hits or sp is None:
                continue
            verb = NE.search(s).group(2)
            hw = [w for w in heard if sp[0] - 0.05 <= w["start"] <= sp[1] + 0.05]
            f = features(x, sr, hw, verb)
            if f:
                human.append(f)
            items.append((s, verb))
    print(f"чтец: {len(human)} мест, «не» громче глагола на {median(f[0] for f in human):+.1f} дБ, "
          f"длина «не»/глагол {median(f[1] for f in human):.2f}")
    res = {"server": [], "moved": []}
    for n, (s, verb) in enumerate(items):
        marked = post("/v1/text/prepare", json={"text": s}).json()["text"]
        for variant, text in (("server", marked), ("moved", moved(marked))):
            for seed in (1, 2, 3):
                r = post("/v1/audio/speech", json={"input": text, "voice": "turgenev", "response_format": "wav",
                                                   "seed": seed, "stress": False})
                y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
                sf.write(os.path.join(OUT, f"{n:02}_{variant}_s{seed}.wav"), y, sr)
                f = features(y, sr, words(y, sr), verb)
                if f:
                    res[variant].append(f)
        print(f"  {n:02}: {moved(marked)[:70]}", flush=True)
    for v, fs in res.items():
        print(f"{v:6}: {len(fs)} дублей, «не» громче глагола на {median(f[0] for f in fs):+.1f} дБ, "
              f"длина «не»/глагол {median(f[1] for f in fs):.2f}")
    json.dump({"human": human, **res}, open(os.path.join(OUT, "features.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
