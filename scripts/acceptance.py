"""Приёмка живого сервера: один прогон, одна таблица, числа вместо ощущений.

    python scripts/acceptance.py                      # сервер на 127.0.0.1:8080
    python scripts/acceptance.py --url http://хост:8080 --out results/acceptance.json

Набор проверок (`tests/`) отвечает на вопрос «работает ли по договору». Здесь
другой вопрос: «хорошо ли работает у меня» — сколько ждать первого звука
в разговоре, во сколько раз синтез быстрее реального времени, сколько ошибок
в расшифровке, что с памятью после сотни запросов. Это то, что стоит прогнать
перед выпуском и на новой машине, своей или чужой.

Пороги — из ADR и замеров (experiments/20, 22, 24): синтез быстрее реального
времени, CER обратного распознавания ниже 5%, первый звук в потоке — за три
секунды. Не сошлось — приёмка падает и печатает, что именно.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from context_eval import cer  # noqa: E402
from tests.client import Client, read_wav  # noqa: E402

# Десять фраз из замеров разборчивости: знакомый текст сравним с прошлыми числами
from tts_fast_eval import SENTENCES  # noqa: E402

# Без повторов: на повторяющемся тексте Whisper схлопывает второй проход,
# и CER мерил бы эту его особенность, а не сервер.
LONG = " ".join(SENTENCES)
TIMEOUT = 1800  # первый вызов грузит модели, длинный текст синтезируется минуты


class Report:
    """Строки таблицы и итог: что измерено, каким порогом и сошлось ли."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, what: str, value: str, ok: bool | None = None, note: str = "") -> None:
        self.rows.append({"что": what, "значение": value, "ok": ok, "примечание": note})
        mark = " " if ok is None else ("✓" if ok else "✗")
        print(f"{mark} {what:<34} {value:<22} {note}", flush=True)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["ok"] is False]


def vram() -> float | None:
    """Занятая видеопамять всей карты, ГБ: сравнивать имеет смысл только с ней же."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.split("\n")[0]) / 1024
    except Exception:
        return None


def why(r) -> str:
    """Почему ответ не тот: сообщение об ошибке, а если его нет — код и начало тела."""
    body = r.json()
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get("message", "")
    return f"{r.status}: {r.body[:120]!r}"


def recv_json(ws, timeout: float):
    """Следующее событие-JSON; двоичные кадры (звук) пропускаются — их размер
    уже сказан в событии перед ними."""
    while True:
        m = ws.recv(timeout=timeout)
        if not isinstance(m, bytes):
            return json.loads(m)


def speak(c: Client, text: str, **body) -> tuple[bytes, float, float]:
    """Звук, его длительность и сколько ждали."""
    started = time.perf_counter()
    r = c.post("/v1/audio/speech", json_body={"input": text, "response_format": "wav", **body},
               timeout=TIMEOUT)
    spent = time.perf_counter() - started
    assert r.status == 200, why(r)
    return r.body, float(r.headers["x-audio-seconds"]), spent


def hear(c: Client, wav: bytes, **form) -> tuple[str, float]:
    started = time.perf_counter()
    r = c.post("/v1/audio/transcriptions", form={"language": "ru", **form},
               files={"file": ("a.wav", wav, "audio/wav")}, timeout=TIMEOUT)
    assert r.status == 200, why(r)
    return r.json()["text"], time.perf_counter() - started


def warm(c: Client, rep: Report) -> None:
    """Первый вызов каждой модели грузит её с диска — это отдельная величина,
    а не скорость. Греются обе: иначе «распознавание» мерило бы загрузку Whisper."""
    started = time.perf_counter()
    wav, _, _ = speak(c, "Прогрев.")
    tts = time.perf_counter() - started
    started = time.perf_counter()
    hear(c, wav)
    stt = time.perf_counter() - started
    rep.add("прогрев моделей", f"{tts:.0f} + {stt:.0f} с", None, "синтез и распознавание, первый вызов")


def synthesis(c: Client, rep: Report) -> bytes:
    """Скорость на короткой фразе и на длинном тексте."""
    _, seconds, spent = speak(c, SENTENCES[0], seed=1)
    rep.add("синтез, короткая фраза", f"×{seconds / spent:.1f} к реальному", seconds / spent > 1.0,
            f"{seconds:.1f} с звука за {spent:.1f} с")
    wav, seconds, spent = speak(c, LONG, seed=1)
    rep.add("синтез, длинный текст", f"×{seconds / spent:.1f} к реальному", seconds / spent > 1.0,
            f"{seconds:.0f} с звука за {spent:.0f} с")
    return wav


def recognition(c: Client, rep: Report, wav: bytes) -> None:
    """Скорость и ошибки на своём же длинном синтезе."""
    seconds = read_wav(wav)[2]
    text, spent = hear(c, wav)
    rep.add("распознавание", f"×{seconds / spent:.0f} к реальному", seconds / spent > 1.0,
            f"{seconds:.0f} с записи за {spent:.1f} с")
    value = 100 * cer(LONG, text)
    rep.add("CER обратного распознавания", f"{value:.2f}%", value < 5.0, "порог 5%")


def formats(c: Client, rep: Report) -> None:
    """Каждый формат: отдаётся и начинается тем, чем должен."""
    marks = {"wav": b"RIFF", "opus": b"OggS", "mp3": None, "flac": b"fLaC", "aac": None, "pcm": None}
    ok, bad, notes = [], [], []
    for fmt, mark in marks.items():
        r = c.post("/v1/audio/speech", json_body={"input": "Проверка формата.", "response_format": fmt},
                   timeout=TIMEOUT)
        if r.status != 200:
            reason = why(r).splitlines()[0]
            # aac — единственный формат, которому нужен системный ffmpeg
            (notes if "ffmpeg" in reason else bad).append(f"{fmt}: нужен ffmpeg" if "ffmpeg" in reason
                                                          else f"{fmt}: {reason[:60]}")
            continue
        if mark and not r.body.startswith(mark):
            bad.append(f"{fmt}: не та разметка")
            continue
        ok.append(fmt)
    rep.add("форматы", ", ".join(ok), not bad, "; ".join(bad + notes))


def speed_knob(c: Client, rep: Report) -> None:
    """Темп: длительность меняется, разборчивость — нет."""
    base = read_wav(speak(c, SENTENCES[1], seed=7)[0])[2]
    bad = []
    for factor in (0.8, 1.2):
        wav, _, _ = speak(c, SENTENCES[1], seed=7, speed=factor)
        got = read_wav(wav)[2]
        want = base / factor
        if abs(got - want) / want > 0.05:
            bad.append(f"{factor}: {got:.2f} с вместо {want:.2f}")
        value = 100 * cer(SENTENCES[1], hear(c, wav)[0])
        if value > 5.0:
            bad.append(f"{factor}: CER {value:.1f}%")
    rep.add("темп 0.8 и 1.2", "длительность и CER", not bad, "; ".join(bad))


def streaming_speech(c: Client, rep: Report) -> None:
    """Живой разговор со стороны говорящего: когда придёт первый звук."""
    text = "Первое предложение короткое. Второе чуть длиннее, чтобы поток продолжился."
    with c.ws("/v1/audio/speech/stream?sample_rate=16000") as ws:
        assert recv_json(ws, 60)["type"] == "ready"
        started = time.perf_counter()
        for token in text.split(" "):
            ws.send(json.dumps({"type": "text", "text": token + " "}))
        ws.send(json.dumps({"type": "end"}))
        first, audio, seconds = None, 0, 0.0
        while True:
            e = recv_json(ws, 300)
            if e["type"] == "audio":
                first = first or time.perf_counter() - started
                audio += 1
                seconds += e["seconds"]
            elif e["type"] in ("done", "error"):
                break
        total = time.perf_counter() - started
    rep.add("поток: первый звук", f"{first:.1f} с", first is not None and first < 3.0, "порог 3 с")
    rep.add("поток: всего", f"×{seconds / total:.1f} к реальному", seconds / total > 1.0,
            f"{audio} кусков, {seconds:.1f} с звука")


def streaming_listen(c: Client, rep: Report) -> None:
    """Слушающая сторона: когда придёт текст и когда — конец реплики."""
    import numpy as np

    with c.ws("/v1/audio/speech/stream?sample_rate=16000") as ws:
        ws.recv(timeout=60)
        ws.send(json.dumps({"type": "text", "text": "Скажи, какая погода в Москве сегодня."}))
        ws.send(json.dumps({"type": "end"}))
        pcm = b""
        expect_audio = False
        while True:
            m = ws.recv(timeout=300)
            if isinstance(m, bytes):
                pcm += m if expect_audio else b""
                continue
            e = json.loads(m)
            expect_audio = e["type"] == "audio"
            if e["type"] in ("done", "error"):
                break
    speech = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    tail = np.zeros(int(1.5 * 16000), dtype="float32")  # пауза, по которой слышно конец реплики
    audio = np.concatenate([speech, tail])

    with c.ws("/v1/audio/transcriptions/stream?language=ru&sample_rate=16000") as ws:
        ws.recv(timeout=60)
        started = time.perf_counter()
        step = 1600  # 100 мс — как отдаёт микрофон
        events: list[dict] = []
        end_of_audio = None
        for i in range(0, len(audio), step):
            chunk = (audio[i:i + step] * 32767).astype("<i2").tobytes()
            ws.send(chunk)
            time.sleep(step / 16000)  # вживую, а не как получится
        end_of_audio = time.perf_counter() - started
        ws.send(json.dumps({"type": "end"}))
        while True:
            try:
                e = recv_json(ws, 60)
            except Exception:
                break
            events.append({**e, "t": time.perf_counter() - started})
            if e["type"] in ("done", "closed", "error"):
                break
    kinds = {e["type"] for e in events}
    text = next((e for e in events if e["type"] in ("partial", "final")), None)
    turn = next((e for e in events if e["type"] == "turn_end"), None)
    rep.add("слушать: первый текст", f"{text['t']:.1f} с" if text else "нет", text is not None,
            f"звук кончился на {end_of_audio:.1f} с")
    rep.add("слушать: конец реплики", f"{turn['t'] - end_of_audio:+.1f} с" if turn else "нет",
            turn is not None, f"события: {', '.join(sorted(kinds))}; текст: {(turn or {}).get('text', '')[:40]}")


def jobs(c: Client, rep: Report) -> None:
    """Очередь: задание идёт, прогресс растёт, отмена срабатывает."""
    r = c.post("/v1/jobs/speech", json_body={"input": LONG, "response_format": "opus"}, timeout=60)
    assert r.status in (200, 202), why(r)  # задание принято — это 202
    job = r.json()["id"]
    stages, progress = set(), 0.0
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        state = c.get(f"/v1/jobs/{job}", timeout=60).json()
        stages.add(state["progress"].get("stage") or state["state"])
        progress = max(progress, state["progress"].get("produced") or 0)
        if state["state"] != "queued" and progress > 1.0:
            break
        time.sleep(0.5)
    cancelled = c.request("DELETE", f"/v1/jobs/{job}", timeout=60)
    rep.add("задание и отмена", f"прогресс {progress:.0f} с", cancelled.status == 200 and progress > 0,
            f"этапы: {', '.join(sorted(stages))}")


def load(c: Client, rep: Report, times: int) -> None:
    """Сотня коротких запросов подряд: не растёт ли память и не падает ли скорость."""
    before = vram()
    first, spent = None, []
    for i in range(times):
        _, seconds, took = speak(c, SENTENCES[i % len(SENTENCES)], seed=i)
        spent.append(seconds / took)
        first = first or spent[0]
    after = vram()
    slowdown = (first - spent[-1]) / first * 100 if first else 0
    rep.add(f"{times} запросов подряд", f"×{sum(spent) / len(spent):.1f} в среднем", slowdown < 20,
            f"первый ×{first:.1f}, последний ×{spent[-1]:.1f}")
    if before is not None and after is not None:
        rep.add("видеопамять карты", f"{after:.1f} ГБ", after - before < 1.0, f"было {before:.1f} ГБ")


def main() -> int:
    ap = argparse.ArgumentParser(description="приёмка живого сервера voicy")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None, help="если сервер требует ключ")
    ap.add_argument("--repeat", type=int, default=20, help="сколько коротких запросов в нагрузке")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    c = Client(a.url, a.key)
    rep = Report()
    health = c.get("/health")
    if health.status != 200:
        print(f"сервер не отвечает на {a.url}", file=sys.stderr)
        return 2
    h = health.json()
    print(f"сервер: {h.get('server', 'python')}, устройство {h.get('device')}, "
          f"синтез {h['tts']['engine']} ({h['tts'].get('runtime', 'python')}), "
          f"распознавание {h['stt']['engine']} ({h['stt'].get('runtime', 'python')})\n")

    t0 = time.perf_counter()
    warm(c, rep)
    long_wav = synthesis(c, rep)
    recognition(c, rep, long_wav)
    formats(c, rep)
    speed_knob(c, rep)
    streaming_speech(c, rep)
    streaming_listen(c, rep)
    jobs(c, rep)
    load(c, rep, a.repeat)
    print(f"\nвсего {time.perf_counter() - t0:.0f} с")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"health": h, "rows": rep.rows}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"отчёт — {a.out}")
    if rep.failed:
        print(f"не сошлось: {', '.join(r['что'] for r in rep.failed)}", file=sys.stderr)
        return 1
    print("приёмка пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
