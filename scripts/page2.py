import json, sys, pathlib, subprocess, html, os
sys.stdout.reconfigure(encoding="utf-8")

OUT = pathlib.Path("round2"); OUT.mkdir(exist_ok=True)
samples = {s["id"]: s for s in json.load(open("samples.json", encoding="utf-8"))}
cer_all = json.load(open("cer.json", encoding="utf-8")) if os.path.exists("cer.json") else []
cer_eng = {(r["name"], r.get("id")): r["cer"] for r in cer_all if r["group"] == "engines"}
final = json.load(open("cer_final.json", encoding="utf-8"))


def enc(src: pathlib.Path) -> str:
    dst = OUT / (src.stem + ".opus")
    if src.exists() and not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                        "-c:a", "libopus", "-ac", "1", "-b:a", "24k", str(dst)], check=True)
    return str(dst).replace("\\", "/")


def badge(c):
    if c is None:
        return ""
    cls = "good" if c < 4 else ("mid" if c < 10 else "bad")
    return f"<span class='cer {cls}'>CER {c}%</span>"


def card(title, path, c, note=""):
    return (f"<div class=card><b>{title}</b>{badge(c)}"
            f"<audio controls preload=none src='{path}'></audio>"
            f"<span class=n>{note}</span></div>")


rows = [
    "<h2>Референсный голос</h2>",
    "<p class=lead>Публичное достояние, LibriVox (Лесков, «Святочные рассказы»). "
    "ESpeech копирует отсюда тембр и артикуляцию — а ритм и интонацию строит сам. "
    "Заменив этот файл, меняешь голос всего курса.</p>",
    "<div class=grid>" + card("исходная запись, 13 с", enc(pathlib.Path("lv/ref_ys.wav")), None, "человек, не синтез") + "</div>",
]

for sid, s in samples.items():
    rows.append(f"<h2>{html.escape(s['label'])}</h2>")
    rows.append(f"<p class=src>{html.escape(s['audio'])}</p>")

    rows.append("<h3>Стало: RL-V2 + человеческий референс</h3><div class=grid>")
    for r in [x for x in final if x["id"] == sid]:
        rows.append(card(f"скорость {r['speed']}", enc(pathlib.Path("final") / r["file"]),
                         r["cer"], f"{r['audio_s']:.0f}с · синтез ×{r['rtf']:.1f}"))
    rows.append("</div>")

    rows.append("<h3>Было: PODCASTER + синтетический референс</h3><div class=grid>")
    p = pathlib.Path("out") / f"{sid}__audio__espeech-podcaster.wav"
    rows.append(card("то, что ты слушал", enc(p), cer_eng.get(("espeech podcaster", sid))))
    rows.append("</div>")

    rows.append("<h3>Для сравнения — самые разборчивые движки</h3><div class=grid>")
    for name, f in (("silero v4 baya", f"{sid}__audio__silero-baya.wav"),
                    ("silero v5 kseniya", f"{sid}__audio__silero5-kseniya.wav"),
                    ("vosk speaker 4", f"{sid}__audio__vosk-s4.wav")):
        pp = pathlib.Path("out") / f
        if pp.exists():
            rows.append(card(name, enc(pp), cer_eng.get((name.replace("v4 ", "").replace("v5 ", "-v5 ")
                                                         .replace("speaker 4", "speaker 4"), sid))))
    rows.append("</div>")

css = """body{font:15px/1.6 system-ui;margin:0 auto;max-width:1000px;padding:24px;background:#0f1115;color:#e6e8ee}
h1{font-size:22px}h2{margin-top:42px;font-size:17px;border-bottom:1px solid #2a2f3a;padding-bottom:8px}
h3{font-size:14px;font-weight:600;color:#9aa4b8;margin:20px 0 8px;text-transform:uppercase;letter-spacing:.4px}
.src{background:#171a21;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#c3c9d6}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:7px}
audio{width:100%;height:34px}.n{color:#7c8496;font-size:12px}
.cer{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;align-self:flex-start}
.cer.good{background:#16351f;color:#6ee7a0}.cer.mid{background:#3a3218;color:#e8c468}.cer.bad{background:#3a1c1c;color:#f08a8a}
.lead{color:#9aa4b8;font-size:13.5px}"""

open("round2.html", "w", encoding="utf-8").write(
    f"<!doctype html><meta charset=utf-8><title>bkdojo — ESpeech, раунд 2</title><style>{css}</style>"
    "<h1>ESpeech после лечения «съеденных букв»</h1>"
    "<p class=lead>Три изменения: модель RL-V2 вместо PODCASTER, живой референс вместо синтетического, "
    "48 шагов диффузии вместо 32. <b>CER</b> — доля символов, теряющихся при обратном распознавании "
    "через Whisper: объективная мера разборчивости.</p>" + "\n".join(rows))
print("ok")
