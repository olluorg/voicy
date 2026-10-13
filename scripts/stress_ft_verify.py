"""Did ESpeech really say the shifted stress? Noise in the v3 training data.

ESpeech follows a «+» on a non-normative syllable in about 59% of cases
(stress_ft_espeech.py check). Every miss in espeech.jsonl is an example where
the mark says one syllable and the audio stresses another — it teaches the
model to ignore the mark. Both adapters took the same share of their source's
obedience (v2 55/66, v3 49/59, ≈83%), which is what label noise would give.

Two detectors judge every word of a training phrase:

  silero  — the validated one: templates from three Silero voices
            (10% false alarms, 71% sensitivity). Silero is CC BY-NC, so it only
            measures here and decides nothing about the published data.
  espeech — templates from ESpeech itself (Apache-2.0): the same phrase with
            every word moved to syllable k, in the example's own voice and two
            others. Usable as a filter if it agrees with the first one.

The pilot (100 phrases) showed the espeech detector judging itself: where ESpeech
did not move the stress in the example, it often does not in the template either,
so it says «yes» to shifted words more readily than to normative ones. Only the
Silero detector filters; that makes the cleaned adapter (v4) a research one, like v2.

    python scripts/stress_ft_verify.py pilot [N]     # N phrases with shifts
    python scripts/stress_ft_verify.py report
    python scripts/stress_ft_verify.py full          # Silero verdict for every phrase with shifts
    python scripts/stress_ft_verify.py clean         # → ds/espeech_clean.jsonl
    python scripts/stress_ft_verify.py focus         # + кадры сдвинутых слов → ds/espeech_focus.jsonl
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import stress_baseline as sb
import stress_ft_espeech as se

WORK = sb.WORK
PLAN = os.path.join(se.DS, "silero", "plan.json")
PILOT = os.path.join(WORK, "verify_pilot.json")
FULL = os.path.join(WORK, "verify_full.json")
N_TPL_VOICES = 3


def espeech_voice_of():
    import glob
    out = {}
    for f in glob.glob(os.path.join(se.OUTDIR, "voices_*.json")):
        out.update(json.load(open(f)))
    return out


def judge(dists):
    return {int(i): int(min(d, key=d.get)) for i, d in dists.items()}


def pilot(n=120):
    se.asr_patch()
    acc, _, silero_templates = sb.setup()
    say = se.speaker(*se.load_espeech())
    vs = se.voices()
    names = sorted(vs)
    refs_plus = {k: acc.process_all(t) for k, (_, t) in vs.items()}
    voice_of = espeech_voice_of()

    def espeech_templates(expected, own):
        nvs = [sum(ch in sb.V for ch in w.lower()) for w, _ in expected]
        others = [v for v in names if v != own]
        tvoices = [own] + [others[(len(own) + j) % len(others)] for j in range(N_TPL_VOICES - 1)]
        tpl = {i: {} for i in range(len(expected))}
        for k in range(max(nvs, default=0)):
            words = [sb.place(w, min(k, nv - 1)) if nv >= 1 else w for (w, _), nv in zip(expected, nvs)]
            for v in tvoices:
                x, sr = say(" ".join(words) + ".", vs[v][0], refs_plus[v], seed=100 + k)
                for i, c in sb.cut(x, sr, expected).items():
                    if k < nvs[i]:
                        tpl[i].setdefault(k, []).append(c)
        return tpl

    plan = json.load(open(PLAN, encoding="utf-8"))[:3000]
    todo = [p for p in plan if p["shifted"] and str(p["id"]) in voice_of][:n]
    rows = json.load(open(PILOT, encoding="utf-8")) if os.path.exists(PILOT) else []
    done = {r["id"] for r in rows}
    for p in todo:
        if p["id"] in done:
            continue
        target = sb.words_of(p["silero"])
        normal = sb.words_of(acc.process_all(p["plain"]))
        if [sb.norm(w) for w, _ in target] != [sb.norm(w) for w, _ in normal]:
            print("  слова не совпали, пропуск", p["id"], flush=True)
            continue
        x, sr = sf.read(os.path.join(se.OUTDIR, "wav", f"{p['id']:05}.wav"), dtype="float32")
        own = voice_of[str(p["id"])]
        d_sil = sb.detect(x, sr, target, silero_templates(target))
        d_esp = sb.detect(x, sr, target, espeech_templates(target, own))
        rows.append({"id": p["id"], "voice": own, "text": p["text"],
                     "words": [w for w, _ in target],
                     "target": [k for _, k in target], "normal": [k for _, k in normal],
                     "silero": d_sil, "espeech": d_esp})
        json.dump(rows, open(PILOT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{len(rows)}/{len(todo)}] {p['text'][:70]}", flush=True)
    report()


def report():
    rows = json.load(open(PILOT, encoding="utf-8"))
    stats = {}
    for r in rows:
        js, je = judge(r["silero"]), judge(r["espeech"])
        for i, (t, nrm) in enumerate(zip(r["target"], r["normal"])):
            if t is None or not sb.eligible(r["words"][i]):
                continue
            kind = "сдвинуто" if t != nrm else "норма"
            s, e = js.get(i), je.get(i)
            st = stats.setdefault(kind, {"sil": [], "esp": [], "agree": [], "both": []})
            if s is not None:
                st["sil"].append(s == t)
            if e is not None:
                st["esp"].append(e == t)
            if s is not None and e is not None:
                st["agree"].append((s == t) == (e == t))
                st["both"].append((s == t, e == t))
    print(f"\n{len(rows)} фраз ESpeech. Доля слов, где детектор слышит ударение там, где разметка:")
    print("              эталоны Silero    эталоны ESpeech   согласие")
    for kind, st in stats.items():
        f = lambda v: f"{np.mean(v):5.0%} из {len(v):<4}" if v else "  —"
        print(f"  {kind:9}   {f(st['sil'])}      {f(st['esp'])}      {f(st['agree'])}")
    sh = stats.get("сдвинуто", {}).get("both", [])
    if sh:
        tt = sum(a and b for a, b in sh); ff = sum(not a and not b for a, b in sh)
        tf = sum(a and not b for a, b in sh); ft = sum(not a and b for a, b in sh)
        print(f"\nсдвинутые слова, Silero × ESpeech: оба «да» {tt}, оба «нет» {ff}, "
              f"только Silero {tf}, только ESpeech {ft}")


def full():
    """Silero verdicts for every rendered phrase with shifts; the pilot's are reused."""
    se.asr_patch()
    acc, _, silero_templates = sb.setup()
    voice_of = espeech_voice_of()
    plan = json.load(open(PLAN, encoding="utf-8"))[:3000]
    todo = [p for p in plan if p["shifted"] and str(p["id"]) in voice_of]
    rows = json.load(open(FULL, encoding="utf-8")) if os.path.exists(FULL) else []
    if not rows and os.path.exists(PILOT):
        rows = [{k: r[k] for k in ("id", "voice", "text", "words", "target", "normal", "silero")}
                for r in json.load(open(PILOT, encoding="utf-8"))]
    done = {r["id"] for r in rows}
    for n, p in enumerate(todo, 1):
        if p["id"] in done:
            continue
        target = sb.words_of(p["silero"])
        normal = sb.words_of(acc.process_all(p["plain"]))
        if [sb.norm(w) for w, _ in target] != [sb.norm(w) for w, _ in normal]:
            print("  слова не совпали, пропуск", p["id"], flush=True)
            continue
        x, sr = sf.read(os.path.join(se.OUTDIR, "wav", f"{p['id']:05}.wav"), dtype="float32")
        rows.append({"id": p["id"], "voice": voice_of[str(p["id"])], "text": p["text"],
                     "words": [w for w, _ in target],
                     "target": [k for _, k in target], "normal": [k for _, k in normal],
                     "silero": sb.detect(x, sr, target, silero_templates(target))})
        if len(rows) % 10 == 0:
            json.dump(rows, open(FULL, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{n}/{len(todo)}] {p['text'][:70]}", flush=True)
    json.dump(rows, open(FULL, "w", encoding="utf-8"), ensure_ascii=False)


def clean():
    """espeech.jsonl with the mark taken off every shifted word Silero did not hear
    moved: the example stays, but no longer pairs a mark with audio that ignores it.
    Words Whisper did not catch keep their mark."""
    import re
    verdict = {r["id"]: r for r in json.load(open(FULL, encoding="utf-8"))}
    by_text = {p["text"]: p["id"] for p in json.load(open(PLAN, encoding="utf-8"))}
    word_re = re.compile(r"[А-Яа-яЁё\u0301\-]+")
    kept = stripped = unjudged = 0
    src = os.path.join(se.DS, "espeech.jsonl")
    with open(os.path.join(se.DS, "espeech_clean.jsonl"), "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            ex = json.loads(line)
            r = verdict.get(by_text.get(ex["text"]))
            if r is not None:
                js = judge(r["silero"])
                bad = {i for i, (t, nrm) in enumerate(zip(r["target"], r["normal"]))
                       if t is not None and t != nrm and sb.eligible(r["words"][i])
                       and js.get(i) is not None and js[i] != t}
                for i, (t, nrm) in enumerate(zip(r["target"], r["normal"])):
                    if t is not None and t != nrm and sb.eligible(r["words"][i]):
                        if i in bad:
                            stripped += 1
                        elif i in js:
                            kept += 1
                        else:
                            unjudged += 1
                # слова текста в том же порядке, что words_of: дефис по краям не в счёт
                spans = [m for m in word_re.finditer(ex["text"]) if sb.norm(m.group().strip("-"))]
                if len(spans) == len(r["words"]) and bad:
                    t = ex["text"]
                    for i in sorted(bad, reverse=True):
                        a, b = spans[i].span()
                        t = t[:a] + t[a:b].replace("\u0301", "") + t[b:]
                    ex["text"] = t
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"сдвинутых слов: знак оставлен {kept}, снят {stripped}, не проверено {unjudged}")


FPS = 12.5  # кадров кодека в секунду


def focus():
    """Frames of every shifted word that kept its mark, for a heavier loss there.

    A shifted syllable is a few frames out of a hundred; averaged over the whole
    phrase its loss barely moves the gradient. Word boundaries come from Whisper on
    the very audio the codes were made from; ±1 frame of margin.
    """
    import re
    from difflib import SequenceMatcher
    se.asr_patch()
    verdict = {r["id"]: r for r in json.load(open(FULL, encoding="utf-8"))}
    by_flat = {p["text"].replace("\u0301", ""): p["id"] for p in json.load(open(PLAN, encoding="utf-8"))[:3000]}
    n_ex = n_words = n_frames = n_all = 0
    with open(os.path.join(se.DS, "espeech_focus.jsonl"), "w", encoding="utf-8") as out:
        for line in open(os.path.join(se.DS, "espeech_clean.jsonl"), encoding="utf-8"):
            ex = json.loads(line)
            ex["focus"] = []
            # знаки после чистки могли сняться, поэтому фраза ищется по тексту без них
            pid = by_flat.get(ex["text"].replace("\u0301", ""))
            r = verdict.get(pid)
            if r is not None:
                marked = [m.group().strip("-") for m in re.finditer(r"[А-Яа-яЁё\u0301\-]+", ex["text"])
                          if sb.norm(m.group().strip("-"))]
                want = [i for i, (t, nrm) in enumerate(zip(r["target"], r["normal"]))
                        if t is not None and t != nrm and sb.eligible(r["words"][i])
                        and i < len(marked) and "\u0301" in marked[i]]
                if want:
                    x, sr = sf.read(os.path.join(se.OUTDIR, "wav", f"{pid:05}.wav"), dtype="float32")
                    heard = sb.transcribe_words(x, sr)
                    a = [sb.norm(w) for w in r["words"]]
                    b = [sb.norm(w["word"]) for w in heard]
                    pos = {}
                    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                        if tag == "equal":
                            pos.update(zip(range(i1, i2), range(j1, j2)))
                    m = len(ex["codes"])
                    for i in want:
                        if i in pos:
                            w = heard[pos[i]]
                            f0, f1 = max(0, int(w["start"] * FPS) - 1), min(m, int(w["end"] * FPS) + 2)
                            if f1 > f0:
                                ex["focus"].append([f0, f1]); n_words += 1; n_frames += f1 - f0
            n_ex += bool(ex["focus"]); n_all += len(ex["codes"])
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"focus: {n_words} слов в {n_ex} фразах, {n_frames} кадров из {n_all} ({n_frames / n_all:.1%})")


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    else:
        {"report": report, "full": full, "clean": clean, "focus": focus}[sys.argv[1]]()
