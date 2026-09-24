"""Does the Rust RUAccent (rust/src/native/accent.rs) say what Python RUAccent says?

Sentences come from RuLS transcripts (experiments/23 work data), the project's
own texts and a handful of edge cases; each is cut into sentences the way
RUAccent cuts (razdel), since the Rust port takes one sentence at a time.
Python's `process_all` is the reference; `voicy probe-accent` answers the same
lines. Reported: exact matches, and for the rest the words that differ.

    python scripts/accent_parity.py [N]
"""
import json, os, random, re, subprocess, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from stress_baseline import load_ruaccent, EVERYDAY

WORK = "experiments/23-stress-finetune/work"
BIN = os.environ.get("VOICY_BIN", "rust/target/release/voicy")

EDGE = [
    "Замок стоит на горе, а мастер починил замок.",
    "Всё хорошо — CI/CD работает на 5%, т.е. почти всегда…",
    "Он сказал: «Это правда?» - и ушёл.",
    "Большая часть проверок выполняется при сборке.",
    "Коса косу косой косит.",
    "Мы пьём чай; ёлка стоит в углу, а все ее игрушки на месте.",
    "ВСЕ ЗАГЛАВНЫМИ БУКВАМИ НАПИСАНО.",
    "Слово-через-дефис и ещё одно - через тире.",
]


def corpus(n):
    from ruaccent.text_preprocessor import TextPreprocessor as T
    rng = random.Random(4)
    texts = list(EDGE) + list(EVERYDAY)
    for smp in json.load(open("data/samples.json", encoding="utf-8")):
        texts.append(smp["audio"])
    items = json.load(open(os.path.join(WORK, "ds", "items.json"), encoding="utf-8"))
    rng.shuffle(items)
    texts += [r["text_no_preprocessing"] for r in items[:n]]
    out = []
    for t in texts:
        for s in T.split_by_sentences(t):
            s = s.strip()
            if s and "\n" not in s:
                out.append(s)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    sents = corpus(n)
    acc = load_ruaccent()
    ref = [acc.process_all(s) for s in sents]
    r = subprocess.run([BIN, "probe-accent"], input="\n".join(sents) + "\n", capture_output=True, text=True, check=True)
    got = r.stdout.split("\n")[:len(sents)]
    same = sum(a == b for a, b in zip(ref, got))
    words_total = words_diff = 0
    diffs = []
    for s, a, b in zip(sents, ref, got):
        wa, wb = a.split(), b.split()
        words_total += len(wa)
        if a != b:
            d = [(x, y) for x, y in zip(wa, wb) if x != y] or [(a, b)]
            words_diff += len(d)
            diffs.append((s, d))
    print(f"предложений: {len(sents)}, совпало целиком: {same} ({same / len(sents):.2%})")
    print(f"слов: {words_total}, различий: {words_diff} ({words_diff / max(words_total, 1):.3%})")
    for s, d in diffs[:25]:
        print("  ", s[:70], "|", "; ".join(f"{x} ≠ {y}" for x, y in d[:4]))
    json.dump([{"sentence": s, "diff": d} for s, d in diffs],
              open(os.path.join(WORK, "accent_parity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
