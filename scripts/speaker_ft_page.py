"""Blind listening page for experiments/29: the reader's own paragraph, then A and C as X and Y.

Which of X and Y is which is shuffled per paragraph (seed 29) and kept out of
the page: the key is work/page/key.json. Answers go to the page's db, collection
votes, one document per paragraph: {flow, voice} each "x", "y" or "same".

Each variant is its first seed that read the whole text (round-trip CER at most
MAX_CER), not the best of three: a paragraph cut off halfway would give away
which model it is and say nothing about how a paragraph is carried. How often
a variant breaks off is counted by speaker_ft_eval.py, not hidden here.
Sound is opus 24 kbit/s, embedded.

    python scripts/speaker_ft_page.py    # → work/page/listen.html, key.json

Another pair, template or place through PAGE_VARIANTS, PAGE_TEMPLATE, PAGE_DIR:
the stress page is PAGE_VARIANTS=A,CF PAGE_TEMPLATE=scripts/speaker_ft_page_stress.html
PAGE_DIR=work/page_cf. When work/meter2/rows.json is there (speaker_ft_meter2.py),
each paragraph carries the words where X or Y left the server's mark and the
reader did not, without saying which of the two.
"""
import base64, glob, json, os, random, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")

WORK = os.environ.get("SPEAKER_WORK", "experiments/29-speaker-lora/work")
EVAL = os.path.join(WORK, "eval")
OUT = os.path.join(os.environ.get("PAGE_DIR") or os.path.join(WORK, "page"), "listen.html")
TEMPLATE = os.environ.get("PAGE_TEMPLATE") or os.path.join(os.path.dirname(__file__), "speaker_ft_page.html")
VARIANTS = tuple(os.environ.get("PAGE_VARIANTS", "A,C").split(","))
METER = os.path.join(WORK, "meter2", "rows.json")
SEEDS = (1, 2, 3)
MAX_CER = 0.05


def opus(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-c:a", "libopus", "-b:a", "24k",
                          "-f", "ogg", "-"], capture_output=True, check=True).stdout
    return "data:audio/ogg;base64," + base64.b64encode(out).decode()


def first_whole(variant, pid):
    for seed in SEEDS:
        p = os.path.join(EVAL, variant, f"{pid}_s{seed}.wav")
        m = p[:-4] + ".measure.json"
        if os.path.exists(m) and json.load(open(m, encoding="utf-8"))["cer"] <= MAX_CER:
            return p
    return None


def misses(rows, pid, paths):
    """Words where any of these files left the server's mark and the reader kept it."""
    r = rows.get(pid)
    if not r or r.get("expected") is None:
        return []
    def wrong(key):
        res = r["files"].get(key)
        return set() if res is None else {i for i, ((best, _), exp) in enumerate(zip(res, r["expected"]))
                                         if best is not None and exp is not None and best != exp
                                         and sum(ch in "аеёиоуыэюя" for ch in r["words"][i].lower()) > 1}
    human = wrong(f"human/{pid}.wav")
    bad = set().union(*(wrong(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}") for p in paths)) - human
    return [r["words"][i] for i in sorted(bad)]


def main():
    rows = json.load(open(METER, encoding="utf-8")) if os.path.exists(METER) else {}
    rng = random.Random(29)
    items, key = [], {}
    for h in sorted(glob.glob(os.path.join(EVAL, "human", "*.wav"))):
        pid = os.path.basename(h)[:-4]
        synth = {v: first_whole(v, pid) for v in VARIANTS}
        if not all(synth.values()):
            print(f"  {pid}: пропущен — ни один сид не прочитал текст целиком", {v: bool(p) for v, p in synth.items()})
            continue
        order = list(VARIANTS); rng.shuffle(order)
        text = json.load(open(h[:-4] + ".json", encoding="utf-8"))["text"]
        items.append({"id": pid, "text": text, "orig": opus(h), "x": opus(synth[order[0]]),
                      "y": opus(synth[order[1]]), "words": misses(rows, pid, list(synth.values()))})
        key[pid] = {"x": order[0], "y": order[1]}
        print(f"  {pid}: " + " / ".join(os.path.basename(synth[v]) for v in VARIANTS))
    page = open(TEMPLATE, encoding="utf-8").read().replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(page)
    json.dump(key, open(os.path.join(os.path.dirname(OUT), "key.json"), "w"), indent=1)
    print(f"{len(items)} абзацев → {OUT}, {os.path.getsize(OUT) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
