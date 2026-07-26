import json, sys, pathlib, glob, subprocess, html
sys.stdout.reconfigure(encoding="utf-8")

res = []
for f in glob.glob("results_*.json"):
    if "sweep" in f:  # different schema, rendered on its own page
        continue
    res += json.load(open(f, encoding="utf-8"))

samples = {s["id"]: s for s in json.load(open("samples.json", encoding="utf-8"))}
MP3 = pathlib.Path("mp3"); MP3.mkdir(exist_ok=True)

# encode everything we have to the shipping format (opus 24k)
for r in res:
    wav = pathlib.Path("out") / r["file"]
    out = MP3 / (wav.stem + ".opus")
    if wav.exists() and not out.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav),
                        "-c:a", "libopus", "-ac", "1", "-b:a", "24k", str(out)], check=True)
    r["mp3"] = str(out).replace("\\", "/")
    r["kb"] = out.stat().st_size / 1024 if out.exists() else 0

ORDER = {"espeech": 0, "vosk": 1, "silero-v5": 2, "silero": 3, "piper": 4}
by = {}
for r in res:
    by.setdefault((r["id"], r["variant"]), []).append(r)

rows = []
for sid, s in samples.items():
    rows.append(f"<h2>{html.escape(s['label'])}</h2>")
    for variant, cap in (("audio", "Подготовленный под аудио"), ("raw", "Сырой текст из JSON (для контраста)")):
        if (sid, variant) not in by:
            continue
        txt = s[variant]
        rows.append(f"<h3>{cap} <span class=n>{len(txt)} симв.</span></h3>")
        rows.append(f"<p class=src>{html.escape(txt)}</p><div class=grid>")
        for r in sorted(by[(sid, variant)], key=lambda x: (ORDER.get(x["engine"], 9), str(x["voice"]))):
            hot = " hot" if r["engine"] in ("espeech", "vosk") else ""
            rows.append(
                f"<div class='card{hot}'><b>{r['engine']} · {str(r['voice']).replace('ru_RU-','')}</b>"
                f"<audio controls preload=none src='{r['mp3']}'></audio>"
                f"<span class=n>{r['audio_s']:.0f}с · {r['kb']:.0f} КБ · синтез ×{r['rtf']:.1f}</span></div>")
        rows.append("</div>")

css = """body{font:15px/1.6 system-ui;margin:0 auto;max-width:1000px;padding:24px;background:#0f1115;color:#e6e8ee}
h1{font-size:22px}h2{margin-top:44px;border-bottom:1px solid #2a2f3a;padding-bottom:8px}
h3{font-size:15px;font-weight:600;color:#9aa4b8;margin:22px 0 8px}
.src{background:#171a21;border-left:3px solid #3b82f6;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13.5px;color:#c3c9d6}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:8px}
.card.hot{border-color:#3f6f4a;background:#141b16}
audio{width:100%;height:34px}.n{color:#7c8496;font-size:12px;font-weight:400}
.lead{color:#9aa4b8;font-size:13.5px}"""

open("compare.html", "w", encoding="utf-8").write(
    f"<!doctype html><meta charset=utf-8><title>bkdojo — сравнение TTS</title><style>{css}</style>"
    "<h1>Сравнение движков озвучки</h1>"
    "<p class=lead>Все локальные, на этой машине. Зелёной рамкой — новые кандидаты (ESpeech, Vosk). "
    "Формат — Opus 24 kbps моно, в котором это поехало бы в dist. "
    "«синтез ×N» — во сколько раз быстрее реального времени.</p>" + "\n".join(rows))
print(f"engines: {sorted({r['engine'] for r in res})}")
print(f"files: {len(res)}")
