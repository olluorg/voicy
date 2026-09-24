"""The stress numbers of what is shipped: the Rust server, the q5_k GGUF, RUAccent in Rust.

experiments/23 measured the adapter in Python (bf16, adapter switched on and
off). What users run is different: the talker merged, quantised to q5_k and
executed by llama.cpp, with the marks placed by the Rust port of RUAccent. The
same two tests go through HTTP, once per model — the official weights and the
stress model — each on a server of its own:

  control — the longest word of each everyday sentence, the mark put by hand on
            each vowel in turn, the server's own marking off (stress=false);
  text    — the 79 sentences of stress_ft_eval.py as a user would send them,
            the server marking them itself (it does not for the official
            weights: they carry no stress.json).

The detector and its templates are those of stress_baseline.py; templates
depend only on the sentence, so they are made once and serve both models.

    python scripts/stress_server_eval.py
"""
import io, json, os, re, subprocess, sys, time, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, requests, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import stress_baseline as sb
from stress_ft_eval import cer, hf_share, MARGIN, FA, SENS

WORK = sb.WORK
BIN = os.environ.get("VOICY_BIN", "rust/target/release/voicy")
PORT = 8094
URL = f"http://localhost:{PORT}"
MODELS = os.path.expanduser("~/.cache/voicy/models")
RUNS = {"base": f"{MODELS}/qwen3-tts-12hz-1.7b-base-gguf", "v3": f"{MODELS}/qwen3-tts-12hz-1.7b-ru-stress-gguf"}
SEEDS = (1, 2)
OUT = os.path.join(WORK, "server_eval")
NO_PROXY = {"http": None, "https": None}
ACUTE = "́"


def serve(model_dir, talker="q5_k"):
    env = {**os.environ, "VOICY_TTS_GGUF_DIR": model_dir, "TTS_GGUF_TALKER": talker}
    p = subprocess.Popen([BIN, "serve", "--port", str(PORT)], env=env,
                         stdout=open(os.path.join(OUT, "serve.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if requests.get(URL + "/health", timeout=2, proxies=NO_PROXY).ok:
                return p
        except requests.RequestException:
            pass
        time.sleep(1)
    p.kill()
    raise RuntimeError("сервер не поднялся")


def speak(text, seed, stress):
    r = requests.post(URL + "/v1/audio/speech", proxies=NO_PROXY, timeout=1800,
                      json={"input": text, "voice": "turgenev", "response_format": "wav", "seed": seed, "stress": stress})
    r.raise_for_status()
    return sf.read(io.BytesIO(r.content), dtype="float32")


def heard_text(x, sr):
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV"); buf.seek(0)
    r = requests.post(URL + "/v1/audio/transcriptions", proxies=NO_PROXY, timeout=600,
                      files={"file": ("a.wav", buf, "audio/wav")}, data={"language": "ru"})
    r.raise_for_status()
    return r.json()["text"]


def sentences():
    out = [("everyday", s) for s in sb.EVERYDAY]
    for smp in json.load(open("data/samples.json", encoding="utf-8")):
        for s in re.split(r"(?<=[.!?])\s+", smp["audio"]):
            if len(re.findall(r"[А-Яа-яЁё]+", s)) >= 3:
                out.append((smp["id"], s))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sb.URL = URL  # распознавание для детектора — через тот же сервер
    acc, silero_say, templates_in_context = sb.setup()
    tpl_cache = {}
    results = {}
    for name, model_dir in RUNS.items():
        proc = serve(model_dir)
        tts = requests.get(URL + "/health", proxies=NO_PROXY).json()["tts"]
        print(f"== {name}: {tts['model']}, знаки: {tts.get('stress_marks')}", flush=True)
        try:
            control, text = [], []
            for n, s in enumerate(sb.EVERYDAY, 1):
                expected = sb.words_of(acc.process_all(s))
                cand = [(sum(ch in sb.V for ch in w.lower()), i) for i, (w, e) in enumerate(expected)
                        if e is not None and sb.eligible(w) and "ё" not in w.lower()]
                if not cand:
                    continue
                nv, i = max(cand)
                if s not in tpl_cache:
                    tpl_cache[s] = templates_in_context(expected)
                word = expected[i][0].replace("+", "")
                for k in range(nv):
                    vp = [p for p, ch in enumerate(word.lower()) if ch in sb.V]
                    marked = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", word[:vp[k] + 1] + ACUTE + word[vp[k] + 1:], s, count=1)
                    x, sr = speak(marked, 1, False)
                    sf.write(os.path.join(OUT, f"{name}_control_{n:02}_{k}.wav"), x, sr)
                    d = sb.detect(x, sr, expected, tpl_cache[s]).get(i)
                    control.append({"word": word, "target": k, "normal": expected[i][1], "nv": nv,
                                    "detected": None if not d else min(d, key=d.get)})
                print(f"  [{n:2}] {word}", flush=True)
            for n, (src, s) in enumerate(sentences(), 1):
                expected = sb.words_of(acc.process_all(s))
                if s not in tpl_cache:
                    tpl_cache[s] = templates_in_context(expected)
                for sd in SEEDS:
                    x, sr = speak(s, sd, True)
                    sf.write(os.path.join(OUT, f"{name}_text_{n:03}_{sd}.wav"), x, sr)
                    d = sb.detect(x, sr, expected, tpl_cache[s])
                    text.append({"n": n, "seed": sd, "expected": [e for _, e in expected], "dists": d,
                                 "cer": cer(s, heard_text(x, sr)), "hf": hf_share(x, sr)})
                if n % 10 == 0:
                    print(f"  текст {n}", flush=True)
            results[name] = {"model": tts["model"], "control": control, "text": text}
            json.dump(results, open(os.path.join(WORK, "server_eval.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        finally:
            proc.terminate(); proc.wait(30)
    report(results)


def report(results):
    from scipy.stats import fisher_exact
    print(f"\n{'модель':6} {'знак на ненорм. слоге':>22} {'на нормативном':>15} {'ударение не там':>16} {'≈ истинно':>10} {'CER':>6} {'>8 кГц':>7}")
    ctrl = {}
    for name, r in results.items():
        rs = [c for c in r["control"] if c["detected"] is not None]
        wrong = [c["detected"] == c["target"] for c in rs if c["target"] != c["normal"]]
        right = [c["detected"] == c["target"] for c in rs if c["target"] == c["normal"]]
        ctrl[name] = (sum(wrong), len(wrong))
        bad = [b for t in r["text"] for _, b in sb.verdicts(t["dists"], t["expected"], MARGIN).values()]
        m = float(np.mean(bad)) if bad else 0
        print(f"{name:6} {np.mean(wrong):14.0%} из {len(wrong):<4} {np.mean(right):9.0%} из {len(right):<3}"
              f" {m:11.1%} {max(0, (m - FA) / (SENS - FA)):9.0%} {np.mean([t['cer'] for t in r['text']]):6.1%}"
              f" {np.mean([t['hf'] for t in r['text']]):7.2%}")
    if "base" in ctrl and "v3" in ctrl:
        (a, n1), (b, n2) = ctrl["v3"], ctrl["base"]
        p = fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="greater").pvalue
        print(f"управляемость v3 против исходной: {a}/{n1} против {b}/{n2}, p = {p:.2g}")


def control_only(name, talker, seeds=(1, 2, 3)):
    """The control test alone, for one model and one talker variant, over several
    seeds: is it the q5_k quantisation that weakens the adapter?"""
    os.makedirs(OUT, exist_ok=True)
    sb.URL = URL
    acc, silero_say, templates_in_context = sb.setup()
    proc = serve(RUNS[name], talker)
    rows = []
    try:
        print("==", requests.get(URL + "/health", proxies=NO_PROXY).json()["tts"]["model"], flush=True)
        for n, s in enumerate(sb.EVERYDAY, 1):
            expected = sb.words_of(acc.process_all(s))
            cand = [(sum(ch in sb.V for ch in w.lower()), i) for i, (w, e) in enumerate(expected)
                    if e is not None and sb.eligible(w) and "ё" not in w.lower()]
            if not cand:
                continue
            nv, i = max(cand)
            tpl = templates_in_context(expected)
            word = expected[i][0].replace("+", "")
            vp = [p for p, ch in enumerate(word.lower()) if ch in sb.V]
            for k in range(nv):
                marked = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", word[:vp[k] + 1] + ACUTE + word[vp[k] + 1:], s, count=1)
                for sd in seeds:
                    x, sr = speak(marked, sd, False)
                    d = sb.detect(x, sr, expected, tpl).get(i)
                    rows.append({"word": word, "target": k, "normal": expected[i][1], "seed": sd,
                                 "detected": None if not d else min(d, key=d.get)})
            print(f"  [{n:2}] {word}", flush=True)
    finally:
        proc.terminate(); proc.wait(30)
    json.dump(rows, open(os.path.join(WORK, f"server_control_{name}_{talker}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    rs = [r for r in rows if r["detected"] is not None]
    wrong = [r["detected"] == r["target"] for r in rs if r["target"] != r["normal"]]
    print(f"{name} {talker}: знак на ненормативном слоге — {sum(wrong)}/{len(wrong)} = {np.mean(wrong):.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "control":
        control_only(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open(os.path.join(WORK, "server_eval.json"), encoding="utf-8")))
    else:
        main()
