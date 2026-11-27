"""Does the model follow a stress mark, also against the dictionary? The control test.

The sentences are siriusfreak's control set (qwen3-tts-12hz-1.7b-ru-stress-cf,
scripts/sentences_big.py: GENERIC and HOMOGRAPH) plus EVERYDAY of
stress_baseline.py, 130 in all. In each, one word gets the mark on each of its
vowels in turn and the rest of the sentence none; about two thirds of the marks
contradict the dictionary. The server synthesizes with its own marking off, so
the mark under test is the only one; meter2 judges which vowel was stressed.

Through the running server: the model is whatever it was started with
(VOICY_TTS_GGUF_DIR), so one run per model, each under its own name.

    PYTHONUTF8=1 METER2_DEVICE=cuda python scripts/stress_control.py v3
    python scripts/stress_control.py report v3 cf v4
"""
import ast, io, json, os, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import requests, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from speaker_ft_meter2 import WORK, marks, judge, words_in, vowel_pos

URL = os.environ.get("VOICY_URL", "http://127.0.0.1:8080").rstrip("/")
SETS = os.environ.get("CTRL_SENTENCES", os.path.join(WORK, "cf_src", "scripts", "sentences_big.py"))
OUT = os.path.join(WORK, "control")
SEEDS = (1, 2)
ACUTE = "́"


def lists(path):
    out = {}
    for n in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass  # не список предложений: вызовы, выражения
    return out


def sentences():
    ours = lists(os.path.join(os.path.dirname(__file__), "stress_baseline.py"))
    theirs = lists(SETS)
    return ([("everyday", t) for t in ours["EVERYDAY"]] + [("generic", t) for t in theirs["GENERIC"]]
            + [("homograph", t) for t in theirs["HOMOGRAPH"]])


def rows():
    """One row per (sentence, vowel of its test word): the text with only that mark."""
    out = []
    for sid, (block, text) in enumerate(sentences()):
        plain = text.replace("*", "")
        ws = words_in(plain)
        normal = marks(plain)
        if normal is None:
            continue
        if "*" in text:
            i = len(words_in(text.split("*")[0]))
        else:  # самое длинное размечаемое слово без ё — правило теста
            cand = [(len(vowel_pos(w)), j) for j, (w, _, _) in enumerate(ws)
                    if normal[j] is not None and len(vowel_pos(w)) > 1 and "ё" not in w.lower()]
            if not cand:
                continue
            i = max(cand)[1]
        w, a, b = ws[i]
        if normal[i] is None:
            continue
        for k, p in enumerate(vowel_pos(w)):
            out.append({"sid": sid, "block": block, "i": i, "word": w, "target": k, "normal": normal[i],
                        "text": plain[:a] + w[:p + 1] + ACUTE + w[p + 1:] + plain[b:]})
    return out


def main(name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    have = {(r["text"], r["seed"]) for r in done}
    for n, r in enumerate(rows()):
        for seed in SEEDS:
            if (r["text"], seed) in have:
                continue
            resp = requests.post(URL + "/v1/audio/speech", proxies={"http": None}, timeout=600,
                                 json={"input": r["text"], "voice": "turgenev", "response_format": "wav",
                                       "seed": seed, "stress": False})
            resp.raise_for_status()
            x, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
            heard = judge(x, sr, r["text"])
            done.append({**r, "seed": seed, "heard": heard[r["i"]]["best"] if r["i"] < len(heard) else None})
        if n % 20 == 0:
            json.dump(done, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {n}", flush=True)
    json.dump(done, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    report([name])


def report(names):
    print("модель    | знак по словарю | знак против словаря | омографы против | n против")
    for name in names:
        rs = [r for r in json.load(open(os.path.join(OUT, f"{name}.json"), encoding="utf-8")) if r["heard"] is not None]
        cell = defaultdict(lambda: [0, 0])
        for r in rs:
            key = "dict" if r["target"] == r["normal"] else ("homo" if r["block"] == "homograph" else "other")
            cell[key][0] += r["heard"] == r["target"]; cell[key][1] += 1
        against = [cell["homo"][0] + cell["other"][0], cell["homo"][1] + cell["other"][1]]
        f = lambda c: f"{c[0] / max(c[1], 1):6.1%}"
        print(f"{name:9} | {f(cell['dict']):>15} | {f(against):>19} | {f(cell['homo']):>15} | {against[1]}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        main(sys.argv[1])
