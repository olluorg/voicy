"""Every experiment on the site: README → page, its other files beside it, and
an index of all of them.

    python3 scripts/site_experiments.py _site        # needs `markdown`

Built by .github/workflows/pages.yml into _site/experiments/. Only what git
tracks goes out. A relative link in a README is kept when its target is on the
site (an experiment or its files) and otherwise points at the file on GitHub —
ADRs included: Pages would serve their Markdown as raw text, GitHub renders it.
No link on a page is broken.
"""
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPO = "https://github.com/olluorg/voicy"
ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve() / "experiments"

CSS = """
:root{--bg:#f3f5f7;--surface:#fff;--ink:#1c2330;--muted:#5d6878;--line:#d9dfe6;--accent:#3d6a8f;--code:#eef1f4;
  --display:"Alegreya",Georgia,"Times New Roman",serif;--body:"Golos Text",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono",ui-monospace,"SFMono-Regular",Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12161c;--surface:#1a2029;--ink:#e4e9ef;
  --muted:#9aa6b5;--line:#2c3542;--accent:#8fb7d8;--code:#222a35;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12161c;--surface:#1a2029;--ink:#e4e9ef;--muted:#9aa6b5;--line:#2c3542;--accent:#8fb7d8;
  --code:#222a35;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.6 var(--body)}
.wrap{max-width:780px;margin:0 auto;padding-inline:16px;padding-block:24px 64px}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;font:400 13px/1.4 var(--mono);margin-bottom:28px}
a{color:var(--accent)} a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
h1,h2,h3{font-family:var(--display);font-weight:700;line-height:1.2;text-wrap:balance}
h1{font-size:clamp(28px,5vw,38px);margin:0 0 16px} h2{font-size:24px;margin:36px 0 10px} h3{font-size:19px;margin:24px 0 8px}
p,li{max-width:70ch} code{font:0.88em/1.4 var(--mono);background:var(--code);padding:1px 4px;border-radius:3px}
pre{background:var(--code);padding:12px 14px;border-radius:6px;overflow-x:auto} pre code{background:none;padding:0}
.table{overflow-x:auto;margin:12px 0} table{border-collapse:collapse;font-size:14px;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{font:500 12px/1.3 var(--mono);color:var(--muted)} td{font-variant-numeric:tabular-nums}
blockquote{margin:0;border-left:3px solid var(--line);padding-left:14px;color:var(--muted)}
.list{display:grid;gap:10px;padding:0;list-style:none}
.list li{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;max-width:none}
.list .n{font:500 12px/1 var(--mono);color:var(--muted)} .list a{font-weight:600;text-decoration:none}
.list p{margin:6px 0 0;color:var(--muted);font-size:14px}
"""
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Alegreya:wght@700'
         '&family=Golos+Text:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">')


def page(title: str, nav: str, body: str) -> str:
    return (f'<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{html.escape(title)}</title>\n{FONTS}\n<style>{CSS}</style>\n</head>\n<body>\n'
            f'<div class="wrap">\n<nav>{nav}</nav>\n{body}\n</div>\n</body>\n</html>\n')


def tracked() -> list[Path]:
    files = subprocess.run(["git", "ls-files", "experiments"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()
    return [Path(f) for f in files]


def plain(md: str) -> str:
    """A README line as text: no markup, no links."""
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    return re.sub(r"[*_`]", "", md).strip()


def relink(body: str, here: Path, on_site: set[Path]) -> str:
    """href/src relative to the README in the repo → the same thing on the site,
    or the file on GitHub when the site does not have it."""
    def fix(m):
        attr, url = m.group(1), m.group(2)
        if re.match(r"^([a-z]+:|#|//)", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = (here / path).resolve().relative_to(ROOT.resolve()) if path else None
        if target is None:
            return m.group(0)
        site = None
        if target.parts[:1] == ("experiments",) and len(target.parts) >= 2:
            exp = target.parts[1]
            rest = target.parts[2:]
            if not rest or rest == ("README.md",):
                site = f"../{exp}/"
            elif Path(*target.parts) in on_site:
                site = f"../{exp}/{'/'.join(rest)}"
        if site is None:
            kind = "tree" if (ROOT / target).is_dir() else "blob"
            site = f"{REPO}/{kind}/master/{target.as_posix()}"
        return f'{attr}="{site}{"#" + frag if frag else ""}"'
    return re.sub(r'(href|src)="([^"]+)"', fix, body)


def main():
    files = tracked()
    on_site = {f for f in files if f.name != "README.md"}
    readmes = sorted(f for f in files if f.name == "README.md" and len(f.parts) == 3)
    if out.exists():
        shutil.rmtree(out)
    entries = []
    for readme in readmes:
        name = readme.parts[1]
        dst = out / name
        dst.mkdir(parents=True)
        for f in files:
            if f.parts[1] == name and f != readme:
                (dst / Path(*f.parts[2:])).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / f, dst / Path(*f.parts[2:]))
        md = (ROOT / readme).read_text(encoding="utf-8")
        title = next((plain(l[2:]) for l in md.splitlines() if l.startswith("# ")), name)
        question = next((plain(l.split(":**", 1)[1]) for l in md.splitlines() if l.startswith("**Вопрос:**")), "")
        status = next((plain(l.split(":**", 1)[1]) for l in md.splitlines() if l.startswith("**Статус:**")), "")
        body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
        body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
        body = relink(body, (ROOT / readme).parent, on_site)
        # звук статьи, упомянутый в README, на сайте лежит в корне — даём послушать
        body = re.sub(r"<code>article/assets/audio/([^<*]+\.opus)</code>",
                      r'<a href="../../assets/audio/\1"><code>\1</code></a> '
                      r'<audio controls preload="none" src="../../assets/audio/\1"></audio>', body)
        nav = (f'<a href="../">← все эксперименты</a><a href="../../">статья</a>'
               f'<a href="{REPO}/tree/master/experiments/{name}">исходники на GitHub</a>')
        (dst / "index.html").write_text(page(title, nav, body), encoding="utf-8")
        entries.append((name, title, question, status))

    items = "\n".join(
        f'<li><span class="n">{html.escape(n.split("-")[0])}</span> <a href="{n}/">{html.escape(t)}</a>'
        + (f"<p>{html.escape(q)}</p>" if q else "") + (f"<p>Статус: {html.escape(s)}</p>" if s else "") + "</li>"
        for n, t, q, s in entries)
    body = (f"<h1>Эксперименты voicy</h1>\n<p>Что пробовали, чем мерили и к чему пришли — "
            f"{len(entries)} экспериментов по порядку. Решения, которые из них выросли, — в "
            f'<a href="{REPO}/tree/master/docs/adr">ADR</a>.</p>\n<ul class="list">\n{items}\n</ul>')
    nav = f'<a href="../">статья</a><a href="{REPO}/tree/master/experiments">исходники на GitHub</a>'
    (out / "index.html").write_text(page("Эксперименты voicy", nav, body), encoding="utf-8")
    print(f"экспериментов: {len(entries)}, файлов рядом: {len(on_site)}")


if __name__ == "__main__":
    main()
