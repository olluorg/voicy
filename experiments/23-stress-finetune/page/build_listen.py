"""Страница для прослушивания: пары «исходная Qwen / v3» из теста управляемости и трудных слов.

Отбор случайный (сид 1), не лучшие случаи. Звук — opus 32 кбит/с, встроен в страницу.
"""
import base64, json, os, random, re, subprocess, sys
sys.path.insert(0, "scripts")
import stress_baseline as sb

W = "experiments/23-stress-finetune/work"
EV = f"{W}/eval/lora_v3"
ACUTE = "́"
V = sb.V


def opus(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-c:a", "libopus", "-b:a", "32k",
                          "-f", "ogg", "-"], capture_output=True, check=True).stdout
    return "data:audio/ogg;base64," + base64.b64encode(out).decode()


def stressed(word, k):
    vp = [i for i, ch in enumerate(word.lower()) if ch in V]
    if k is None or k >= len(vp):
        return word
    i = vp[k]
    return word[:i + 1] + ACUTE + word[i + 1:]


def syl(k):
    return "—" if k is None else f"{k + 1}-й"


rng = random.Random(1)
ctrl = json.load(open(f"{W}/control_lora_v3.json", encoding="utf-8"))
by = {}
for r in ctrl:
    if r["model"] in ("base", "lora"):
        by.setdefault((r["text"], r["target"]), {})[r["model"]] = r
items = [(t, k, v) for (t, k), v in by.items() if "base" in v and "lora" in v and v["lora"]["target"] != v["lora"]["normal"]]
items.sort(key=lambda x: (x[0], x[1]))
pick = rng.sample(items, 8)
everyday = [s for s in sb.EVERYDAY]

control = []
for text, k, v in pick:
    plain = text.replace(ACUTE, "")
    n = everyday.index(plain) + 1
    w = v["lora"]["word"]
    control.append({
        "sentence": text, "word": w, "target": stressed(w, k), "normal": stressed(w, v["lora"]["normal"]),
        "base": {"audio": opus(f"{EV}/control/{n:02}_{k}_base.wav"), "det": v["base"]["detected"]},
        "v3": {"audio": opus(f"{EV}/control/{n:02}_{k}_lora.wav"), "det": v["lora"]["detected"]},
        "k": k,
    })

hard_rows = json.load(open(f"{W}/hard_lora_v3.json", encoding="utf-8"))
hpick = sorted(rng.sample(range(len(hard_rows)), 8))
hard = []
for idx in hpick:
    r = hard_rows[idx]
    n = idx + 1
    i = r["hard"][0]
    exp = r["expected"][i]
    w = r["words"][0].replace("+", "")
    def verdict(name):
        d = r["c"][name][0]
        v = sb.verdicts(d, r["expected"], 0.3).get(i)
        return None if v is None else {"best": v[0], "bad": bool(v[1])}
    hard.append({
        "sentence": r["text"], "word": w, "expected": stressed(w, exp), "k": exp,
        "base": {"audio": opus(f"{EV}/hard/{n:02}_base-plain_3.wav"), "v": verdict("base/plain")},
        "v3": {"audio": opus(f"{EV}/hard/{n:02}_lora-marked_3.wav"), "v": verdict("lora/marked")},
    })

json.dump({"control": control, "hard": hard}, open(f"{W}/listen_data.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(control), len(hard), os.path.getsize(f"{W}/listen_data.json") // 1024, "КБ")
