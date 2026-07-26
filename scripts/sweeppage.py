import json, sys, pathlib, subprocess, html, os
sys.stdout.reconfigure(encoding="utf-8")

PROBE = ("Слушай вопрос: чем опасно объявление внешнего ключа с он делит каскад? "
         "Подумай и ответь вслух. Короткий ответ: удаление одной строки "
         "может рекурсивно удалить тысячи связанных записей.")

res = json.load(open("results_sweep.json", encoding="utf-8"))
cer = {}
if os.path.exists("cer.json"):
    for r in json.load(open("cer.json", encoding="utf-8")):
        if r["group"] == "sweep":
            cer[r["name"]] = r

OUT = pathlib.Path("sweep_opus"); OUT.mkdir(exist_ok=True)
REF_LABEL = {"silero4-short": "референс: Silero v4, 4.7 с (как было)",
             "vosk-clear": "референс: Vosk, 7.4 с (чёткая артикуляция)",
             "silero5": "референс: Silero v5, 6.1 с"}

for r in res:
    src = pathlib.Path("sweep") / r["file"]
    dst = OUT / (src.stem + ".opus")
    if src.exists() and not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                        "-c:a", "libopus", "-ac", "1", "-b:a", "24k", str(dst)], check=True)
    r["opus"] = str(dst).replace("\\", "/")
    key = f"{r['ref']} nfe{r['nfe']} sp{r['speed']}"
    r["cer"] = cer.get(key, {}).get("cer")
    r["hyp"] = cer.get(key, {}).get("hyp", "")

groups = {}
for r in res:
    groups.setdefault(r["ref"], []).append(r)

rows = []
for ref, items in groups.items():
    rows.append(f"<h2>{REF_LABEL.get(ref, ref)}</h2><div class=grid>")
    for r in sorted(items, key=lambda x: (x["nfe"], x["speed"])):
        c = r["cer"]
        badge = ""
        if c is not None:
            cls = "good" if c < 3 else ("mid" if c < 8 else "bad")
            badge = f"<span class='cer {cls}'>CER {c}%</span>"
        hyp = f"<div class=hyp>«{html.escape(r['hyp'])}»</div>" if r["hyp"] else ""
        rows.append(
            f"<div class=card><b>nfe {r['nfe']} · скорость {r['speed']}</b>{badge}"
            f"<audio controls preload=none src='{r['opus']}'></audio>"
            f"<span class=n>{r['audio_s']:.1f}с · синтез ×{r['rtf']:.1f}</span>{hyp}</div>")
    rows.append("</div>")

css = """body{font:15px/1.6 system-ui;margin:0 auto;max-width:1000px;padding:24px;background:#0f1115;color:#e6e8ee}
h1{font-size:22px}h2{margin-top:36px;font-size:16px;border-bottom:1px solid #2a2f3a;padding-bottom:8px}
.src{background:#171a21;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13.5px;color:#c3c9d6}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}
.card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:7px}
audio{width:100%;height:34px}.n{color:#7c8496;font-size:12px}
.cer{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;align-self:flex-start}
.cer.good{background:#16351f;color:#6ee7a0}.cer.mid{background:#3a3218;color:#e8c468}.cer.bad{background:#3a1c1c;color:#f08a8a}
.hyp{font-size:12px;color:#7c8496;font-style:italic;border-top:1px solid #262b36;padding-top:7px}
.lead{color:#9aa4b8;font-size:13.5px}"""

open("sweep.html", "w", encoding="utf-8").write(
    f"<!doctype html><meta charset=utf-8><title>ESpeech — подбор параметров</title><style>{css}</style>"
    "<h1>ESpeech PODCASTER: борьба со «съеданием букв»</h1>"
    "<p class=lead>Один и тот же текст, 12 конфигураций. <b>nfe</b> — число шагов диффузии "
    "(больше = чище согласные, дороже). <b>CER</b> — сколько символов теряется при обратном "
    "распознавании через Whisper: объективная мера того самого «вапас». Курсивом — что реально расслышал Whisper.</p>"
    f"<p class=src>{html.escape(PROBE)}</p>" + "\n".join(rows))
print("ok", len(res))
