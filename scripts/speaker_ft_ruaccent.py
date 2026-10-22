"""How often does the server's stress marking disagree with what the reader said?

The model now follows a mark almost always (CF: 0.3% of words off the mark), so
what is left is the marking itself. Here RUAccent's marks, through the server's
/v1/text/prepare, are checked against the reader of «Дворянское гнездо»: every
paragraph of experiments/29 (ds/segments.json), each sentence cut out of the
chapter by the chapter's own word timestamps (audio/asr) and judged by meter2.

Two markings are compared: the text as written, and the text with Title-case
words lowercased first. siriusfreak's model card reports RUAccent right on 95.2%
of capitalized words as written and on 97.4% lowercased (Russian LibriSpeech).
The reader's disagreement also holds meter2's errors (~2%), the same for both.

    PYTHONUTF8=1 METER2_DEVICE=cuda python scripts/speaker_ft_ruaccent.py
    python scripts/speaker_ft_ruaccent.py report
"""
import json, os, re, sys
from collections import Counter
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from speaker_ft_meter2 import WORK, marks, judge, words_in, vowel_pos, show

OUT = os.path.join(WORK, "meter2", "ruaccent.json")
TITLE = re.compile(r"\b[А-ЯЁ][а-яё]+\b")


def lower_titles(text):
    return TITLE.sub(lambda m: m.group(0).lower(), text)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if re.search(r"[А-Яа-яЁё]", s)]


def norm(w):
    return w.lower().replace("ё", "е").strip(" .,;:!?—–-«»\"'()")


def spans(sents, heard):
    """(start, end) of every sentence from the chapter's words, None where unmatched."""
    ours = [(k, norm(w)) for k, s in enumerate(sents) for w, _, _ in words_in(s)]
    theirs = [norm(w["word"]) for w in heard]
    first, last = {}, {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, [w for _, w in ours], theirs, autojunk=False).get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                k = ours[i][0]
                first.setdefault(k, heard[j]["start"]); last[k] = heard[j]["end"]
    return [(first[k], last[k]) if k in first else None for k in range(len(sents))]


def main():
    segs = json.load(open(os.path.join(WORK, "ds", "segments.json"), encoding="utf-8"))
    rows = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    audio = {}
    for seg in segs:
        if seg["id"] in rows:
            continue
        ch = seg["chapter"]
        if ch not in audio:
            audio = {ch: sf.read(os.path.join(WORK, "audio", "wav24", f"dvoryanskoegnezdo_{ch}.wav"), dtype="float32")}
        x, sr = audio[ch]
        heard = [w for s in json.load(open(os.path.join(WORK, "audio", "asr", f"{ch}.json"), encoding="utf-8"))["segments"]
                 for w in s.get("words", []) if seg["start"] - 1 <= w["start"] <= seg["end"] + 1]
        text = seg["text"]
        as_is, low = marks(text), marks(lower_titles(text))
        if as_is is None or low is None:
            print(f"{seg['id']}: разметка не сошлась со словами — пропуск", flush=True)
            continue
        said = []
        for s, sp in zip(sentences(text), spans(sentences(text), heard)):
            n = len(words_in(s))
            if sp is None:
                said += [None] * n
                continue
            a, b = int(max(0, sp[0] - 0.15) * sr), int((sp[1] + 0.15) * sr)
            said += [w["best"] for w in judge(x[a:b], sr, s)]
        if len(said) != len(as_is):
            print(f"{seg['id']}: предложения не сошлись с текстом — пропуск", flush=True)
            continue
        rows[seg["id"]] = {"words": [w for w, _, _ in words_in(text)], "said": said, "as_is": as_is, "low": low}
        json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{seg['id']}: {len(said)} слов", flush=True)
    report(rows)


def report(rows):
    tot, cap, changed = Counter(), Counter(), []
    for pid, r in rows.items():
        for w, said, a, b in zip(r["words"], r["said"], r["as_is"], r["low"]):
            if said is None or a is None or b is None or len(vowel_pos(w)) < 2:
                continue
            for key, m in (("as_is", a), ("low", b)):
                tot[key, m == said] += 1
                if w[0].isupper():
                    cap[key, m == said] += 1
            if a != b:
                changed.append((w, a, b, said))
    acc = lambda c, k: c[k, True] / max(c[k, True] + c[k, False], 1)
    n = tot["as_is", True] + tot["as_is", False]
    nc = cap["as_is", True] + cap["as_is", False]
    print(f"\nРазметка совпадает с тем, что сказал чтец (слова с 2+ гласными, {n}, из них с заглавной {nc}):")
    print(f"  как написано:            все {acc(tot, 'as_is'):.1%}, с заглавной {acc(cap, 'as_is'):.1%}")
    print(f"  заглавные → строчные:    все {acc(tot, 'low'):.1%}, с заглавной {acc(cap, 'low'):.1%}")
    win = sum(1 for _, a, b, s in changed if b == s and a != s)
    lose = sum(1 for _, a, b, s in changed if a == s and b != s)
    print(f"\nСлов, где перевод в строчные сменил знак: {len(changed)}; стало верно {win}, стало неверно {lose}, "
          f"неверно оба раза {len(changed) - win - lose}")
    for w, a, b, s in changed[:40]:
        tag = "лучше" if b == s and a != s else "хуже" if a == s and b != s else "оба мимо"
        print(f"  {show(w, a)} → {show(w, b)}  (чтец: {show(w, s)})  {tag}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(json.load(open(OUT, encoding="utf-8")))
    else:
        main()
