"""A scene from a sentence, with no human in the loop (experiments/28).

    python3 scripts/sound_scene.py "описание сцены" [OUT_DIR]

    1. plan        a small LLM turns the description into beds and events (sound_plan.py)
    2. candidates  four MOSS takes per element; winds also two synthesized (sound_moss.py, sound_wind.py)
    3. judge       AST + CLAP + Audiobox reject the wrong ones and pick the best (sound_judge.py)
       revise      elements with no acceptable take go back to the planner with what the
                   judge heard; new takes for them only; at most REVISE_ROUNDS times
    4. render      LUFS by role, walls by perspective, events by their real length (sound_render.py)

Each stage is its own process: the planner's LLM and MOSS do not share the card.
A stage whose output exists is skipped, so an interrupted run resumes.
SOUND_GEN picks the generator of takes (moss2 by default; sa3m, sa3s, moss1,
ezaudio, unite): the same plan through each is how models are compared. Output
goes to D: — the WSL disk lives on C:, which has no room for candidates — and
the scene with its plan and reports is copied to ~/voicy-sound-results, since D:
holds only what can be made again.
The planner is Qwen3.5-9B on llama-server (LLAMA_DIR, PLANNER_GGUF), started
for the plan and stopped before MOSS needs the card; PLANNER_URL uses a server
already running instead.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(Path.home() / "src" / "MOSS-TTS" / "moss_soundeffect_v2" / ".venv" / "bin" / "python")
# генератор кандидатов: (python, скрипт); договор один — sound_moss.py OUT_DIR items.json
GENERATORS = {
    "moss2": (PY, "sound_moss.py"),
    "sa3m": (sys.executable, "sound_sa3.py"),
    "sa3s": (sys.executable, "sound_sa3.py"),
    "moss1": (str(Path.home() / "src" / "venvs" / "ezaudio" / "bin" / "python"), "sound_moss1.py"),   # transformers 5
    "ezaudio": (str(Path.home() / "src" / "venvs" / "ezaudio" / "bin" / "python"), "sound_ezaudio.py"),
    "unite": (str(Path.home() / "src" / "venvs" / "unite" / "bin" / "python"), "sound_unite.py"),
}
GEN = os.environ.get("SOUND_GEN", "moss2")
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", Path.home() / "src" / "llama-b11090" / "llama-b11090"))
PLANNER_GGUF = os.environ.get("PLANNER_GGUF", "/mnt/d/ml/gguf/Qwen3.5-9B-Q5_K_M.gguf")
# рантайм CUDA: в архиве llama.cpp его нет, он лежит в библиотеках voicy
CUDA_LIBS = Path.home() / ".cache" / "voicy" / "lib" / "linux-x64-cuda13"
SEEDS = (11, 12, 13, 14)
RETRY_SEEDS = (11, 12, 13, 14, 15, 16, 17, 18)   # элементу на исправлении — вдвое больше попыток
WIND_SEEDS = (21, 22)
REVISE_ROUNDS = int(os.environ.get("REVISE_ROUNDS", "2"))


def run(limit_mb: int, *args, env=None):
    t = time.monotonic()
    subprocess.run([sys.executable, str(HERE / "rss_watch.py"), *map(str, args)], check=True,
                   env={**os.environ, "LIMIT_MB": str(limit_mb), "TORCHDYNAMO_DISABLE": "1", **(env or {})})
    print(f"  {time.monotonic() - t:.0f} с", flush=True)


def planner_server(log: Path):
    """llama-server for the plan only; the caller stops it."""
    import urllib.request
    port = 8091
    env = {**os.environ, "LD_LIBRARY_PATH": f"{LLAMA_DIR}:{CUDA_LIBS}"}
    proc = subprocess.Popen([str(LLAMA_DIR / "llama-server"), "-m", PLANNER_GGUF, "-ngl", "99", "-c", "16384",
                             "--host", "127.0.0.1", "--port", str(port), "--jinja"],
                            env=env, stdout=log.open("wb"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(200):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited, see {log}")
        try:
            with opener.open(url + "/health", timeout=2) as r:
                if b"ok" in r.read():
                    return proc, url
        except OSError:
            pass
        time.sleep(1.5)
    proc.kill()
    raise SystemExit(f"llama-server did not start, see {log}")


def main():
    text = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"/mnt/d/ml/voicy-sound/scene-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "description.txt").write_text(text + "\n")
    plan_path, cand, chosen = out / "plan.json", out / "cand", out / "chosen"

    def with_planner(args, log_name):
        url, server = os.environ.get("PLANNER_URL"), None
        if not url:
            server, url = planner_server(out / "llama-server.log")
        try:
            with (out / log_name).open("a") as log:
                return subprocess.run([PY, str(HERE / "sound_plan.py"), *map(str, args)],
                                      env={**os.environ, "PLANNER_URL": url}, stderr=log).returncode
        finally:
            if server:
                server.terminate()
                server.wait(30)

    def candidates(plan, ids=None, seeds=SEEDS):
        items = []
        for x in plan["beds"] + plan["events"]:
            if ids is not None and x["id"] not in ids:
                continue
            seconds = 30 if x in plan["beds"] else x["seconds"]
            # отпечаток описания в имени: план с тем же id и другим звуком не возьмёт старый кандидат
            tag = hashlib.sha1(f"{x['prompt']}|{x.get('negative', '')}|{seconds}".encode()).hexdigest()[:6]
            items += [{"name": f"{x['id']}__{tag}s{s}", "prompt": x["prompt"], "negative": x.get("negative", ""),
                       "seconds": seconds, "seed": s} for s in seeds]
        (out / "candidates.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
        if any(not (cand / f"{it['name']}.wav").exists() for it in items):
            py, script = GENERATORS[GEN]
            run(11000, py, HERE / script, cand, out / "candidates.json", env={"SOUND_GEN": GEN})
        for b in plan["beds"]:
            if b["kind"] == "wind":
                for s in WIND_SEEDS:
                    f = cand / f"{b['id']}__dsp{s}.wav"
                    if not f.exists():
                        run(4000, PY, HERE / "sound_wind.py", f, 60, s)

    def judge():
        run(9000, PY, HERE / "sound_judge.py", plan_path, cand, chosen, out / "judge.md")
        return json.loads((out / "judge.json").read_text())

    if not plan_path.exists():
        print("1. план", flush=True)
        if with_planner([text, plan_path], "plan.log") != 0:
            raise SystemExit(f"план не построен, см. {out / 'plan.log'}")
    orig_path = out / "plan.orig.json"            # первый план: по нему заморожены ожидания
    if not orig_path.exists():
        backups = sorted(out.glob("plan.r*.json"), key=lambda f: f.stat().st_mtime)
        orig_path.write_text((backups[0] if backups else plan_path).read_text())
    # ожидания — только из первого плана, что бы ни дописали исправления
    frozen = {e["id"]: [c for c in e.get("expect", []) if "music" not in c.lower()]
              for e in json.loads(orig_path.read_text())["events"]}
    plan = json.loads(plan_path.read_text())
    for e in plan["events"]:
        if frozen.get(e["id"]):
            e["expect"] = frozen[e["id"]]
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    plan = json.loads(plan_path.read_text())

    print("2. кандидаты", flush=True)
    candidates(plan)
    print("3. отбор", flush=True)
    verdict = judge()
    for round_ in range(1, REVISE_ROUNDS + 1):
        failed = [k for k, v in verdict.items() if v["all_bad"]]
        # план, сохранённый до новых правил, тоже идёт на исправление
        check = subprocess.run([PY, str(HERE / "sound_plan.py"), "--check", str(plan_path)],
                               capture_output=True, text=True)
        issues = [l for l in check.stdout.splitlines() if l]
        if not failed and not issues:
            break
        print(f"   круг {round_}: не вышло {', '.join(failed) or '—'}; план: {len(issues)} нарушений — назад планировщику",
              flush=True)
        n = len(list(out.glob("plan.r*.json")))      # копии нумеруются подряд, не затирая прошлых прогонов
        prev = out / f"plan.r{n}.json"
        plan_path.rename(prev)
        (out / f"judge.r{n}.md").write_text((out / "judge.md").read_text())
        code = with_planner(["--revise", prev, out / "judge.json", plan_path, orig_path], "revise.log")
        if code != 0:
            prev.rename(plan_path)
            print("   планировщик ничего не исправил", flush=True)
            break
        before = json.loads(prev.read_text())
        old = {x["id"]: x["prompt"] for x in before["beds"] + before["events"]}
        plan = json.loads(plan_path.read_text())
        changed = {x["id"] for x in plan["beds"] + plan["events"] if old.get(x["id"]) != x["prompt"]}
        candidates(plan, set(failed) | changed, RETRY_SEEDS)
        verdict = judge()
    still = [k for k, v in verdict.items() if v["all_bad"]]
    if still:
        print(f"   генератор не справился: {', '.join(still)} — взяты наименее плохие", flush=True)

    print("4. сведение", flush=True)
    run(6000, PY, HERE / "sound_render.py", plan_path, chosen, out / "scene.wav", out / "scene.md")
    # D: может пропасть: там только то, что получится заново; итог — копией в домашний каталог
    keep = Path.home() / "voicy-sound-results" / out.name
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("description.txt", "plan.json", "plan.orig.json", "judge.md", "judge.json", "scene.md", "scene.wav",
              "plan.log", "revise.log"):
        if (out / f).exists():
            (keep / f).write_bytes((out / f).read_bytes())
    print(keep / "scene.wav")


if __name__ == "__main__":
    main()
