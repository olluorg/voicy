import json,sys,pathlib,subprocess,html
sys.stdout.reconfigure(encoding="utf-8")
OUT=pathlib.Path("round3"); OUT.mkdir(exist_ok=True)
def enc(p):
    d=OUT/(pathlib.Path(p).stem+".opus")
    if not d.exists():
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(p),"-c:a","libopus","-ac","1","-b:a","24k",str(d)],check=True)
    return str(d).replace("\\","/")
doc=json.load(open("cues.json",encoding="utf-8"))
cer=json.load(open("cer_cued.json",encoding="utf-8"))
old=json.load(open("samples.json",encoding="utf-8"))[0]["audio"]

STYLE={"narration":("повествование","1.02"),"term":("термин / новый случай","0.88"),
       "key":("ключевая фраза","0.86"),"question":("вопрос","0.92")}
script=[]
for c in doc["cues"]:
    if c["kind"]=="beat":
        script.append(f"<div class=beat>пауза {c['ms']} мс</div>"); continue
    lab,sp=STYLE[c["kind"]]
    script.append(f"<div class='cue {c['kind']}'><span class=tag>{lab} · ×{sp}</span>{html.escape(c['text'])}</div>")

def card(t,f,c,n=""):
    cls="good" if c<4 else ("mid" if c<10 else "bad")
    return (f"<div class=card><b>{t}</b><span class='cer {cls}'>CER {c}%</span>"
            f"<audio controls preload=none src='{enc(f)}'></audio><span class=n>{n}</span></div>")

css="""body{font:15px/1.6 system-ui;margin:0 auto;max-width:900px;padding:24px;background:#0f1115;color:#e6e8ee}
h1{font-size:22px}h2{margin-top:40px;font-size:17px;border-bottom:1px solid #2a2f3a;padding-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:7px}
audio{width:100%;height:34px}.n{color:#7c8496;font-size:12px}
.cer{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;align-self:flex-start}
.cer.good{background:#16351f;color:#6ee7a0}.cer.mid{background:#3a3218;color:#e8c468}.cer.bad{background:#3a1c1c;color:#f08a8a}
.lead{color:#9aa4b8;font-size:13.5px}
.cue{padding:9px 12px;border-radius:6px;margin:5px 0;font-size:13.5px;border-left:3px solid #2a2f3a;background:#161920}
.cue.term{border-left-color:#c98b3a;background:#1d1a15}.cue.key{border-left-color:#4ea36a;background:#151d18}
.cue.question{border-left-color:#3b82f6;background:#151a24}
.tag{display:block;font-size:11px;color:#7c8496;text-transform:uppercase;letter-spacing:.4px;margin-bottom:3px}
.beat{font-size:11.5px;color:#5f6675;padding:2px 12px;font-style:italic}
.bad-old{background:#1d1516;border-left:3px solid #a04a4a;padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#c9b8b8}"""

open("round3.html","w",encoding="utf-8").write(
 f"<!doctype html><meta charset=utf-8><title>bkdojo — ритм и подготовка текста</title><style>{css}</style>"
 "<h1>Ритм и переписанный текст</h1>"
 "<p class=lead>Скрипт перестал быть плоской строкой: каждая реплика несёт свою скорость и паузы вокруг себя. "
 "Плюс текст переписан как объяснение, а не как озвученный листинг.</p>"
 "<h2>Слушать</h2><div class=grid>"
 + card("С ритмом (реплики + паузы)","cued/jc-024__cued.wav",cer["jc-024__cued.wav"]["cer"],"163 с · замедления на терминах")
 + card("Ровным потоком (контроль)","cued/jc-024__flat.wav",cer["jc-024__flat.wav"]["cer"],"141 с · тот же текст, одна скорость")
 + "</div>"
 "<h2>Что было не так с текстом</h2>"
 f"<div class=bad-old>{html.escape(old[:600])}…</div>"
 "<p class=lead>«Лист, вайлдкард экстендс Намбер», «продюсер экстендс, консьюмер супер» — это код, "
 "прочитанный вслух. Ни преподаватель, ни подкастер так не говорит: он произносит смысл, "
 "а нотацию оставляет глазам.</p>"
 "<h2>Партитура: как размечен новый скрипт</h2>"
 "<p class=lead>Цвет — тип реплики, он же задаёт скорость и паузы. Это и есть то, что будет лежать в контенте.</p>"
 + "\n".join(script))
print("ok")
