"""Статическая страница прослушивания для GitHub Pages: article/stress/index.html + opus-файлы.

Те же данные, что у страницы на claude.ai (listen_data.json собран build_listen.py
и paragraph.py), но звук лежит отдельными файлами рядом со статьёй, а не внутри HTML.
"""
import base64, json, os, re

W = "experiments/23-stress-finetune/work"
SITE = "article"
AUDIO = os.path.join(SITE, "assets", "audio", "stress-v3")
os.makedirs(AUDIO, exist_ok=True)
os.makedirs(os.path.join(SITE, "stress"), exist_ok=True)

d = json.load(open(f"{W}/listen_data.json", encoding="utf-8"))


def save(data_uri, name):
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    open(os.path.join(AUDIO, name), "wb").write(raw)
    return f"../assets/audio/stress-v3/{name}"


def slug(w):
    tr = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                  "a b v g d e e zh z i y k l m n o p r s t u f h c ch sh sch _ y _ e yu ya".split()))
    return "".join(tr.get(ch, ch) for ch in w.lower().replace("́", ""))


for c in ("base", "v3-plain", "v3-marked"):
    d["paragraph"]["audio"][c] = save(d["paragraph"]["audio"][c], f"paragraph__{c}.opus")
for i, x in enumerate(d["control"]):
    for m in ("base", "v3"):
        x[m]["audio"] = save(x[m]["audio"], f"control__{i:02}_{slug(x['word'])}_{x['k'] + 1}__{m}.opus")
for i, x in enumerate(d["hard"]):
    for m in ("base", "v3"):
        x[m]["audio"] = save(x[m]["audio"], f"hard__{i:02}_{slug(x['word'])}__{m}.opus")

head = open("experiments/23-stress-finetune/page/listen_head.html", encoding="utf-8").read()
body = open("experiments/23-stress-finetune/page/listen_body.html", encoding="utf-8").read()
# на сайте ссылки ведут на отчёт и ADR, а не на путь в репозитории
body = body.replace(
    '<p class="note">Подробности, методика и все числа: <code>experiments/23-stress-finetune/README.md</code> в репозитории voicy.</p>',
    '<p class="note">Методика и все числа — <a href="https://github.com/olluorg/voicy/blob/master/experiments/23-stress-finetune/README.md">отчёт эксперимента 23</a>, '
    'решение — <a href="../docs/adr/0023-stress-marks-by-lora.md">ADR 0023</a>. '
    'Начало истории с ударениями — <a href="../#udareniya">раунд 13 статьи</a>.</p>')
page = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "\n</head>\n<body>\n"
        + body.replace("__DATA__", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))
        + "\n</body>\n</html>\n")
page = page.replace("body{background:var(--bg);", "html{color-scheme:light;}\nbody{margin:0; background:var(--bg);")
open(os.path.join(SITE, "stress", "index.html"), "w", encoding="utf-8").write(page)
n = len(os.listdir(AUDIO))
print(f"страница: {len(page) // 1024} КБ, звук: {n} файлов, "
      f"{sum(os.path.getsize(os.path.join(AUDIO, f)) for f in os.listdir(AUDIO)) // 1024} КБ")
