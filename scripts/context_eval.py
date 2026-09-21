"""Does a recognition context make engineering terms come out right?

Twenty sentences with terms from the pronunciation dictionary are synthesised by
the running voicy server with `prepare` on — so each term is spoken the way a
developer says it — and recognised back four ways:

  none      — plain Whisper;
  prompt    — a sentence in the style of the domain as `prompt`;
  hotwords  — the built-in `engineering` context (the dictionary's terms);
  both      — the two together.

A term counts only in its canonical spelling: "Kafka", not "Кафка" or "кавка",
because that is what a context is for. CER over the whole sentence (lowercased,
punctuation stripped) checks that nothing else got worse.

The voice is synthetic. It is the one this repository measures everything with,
and the effect of a context is on the decoder, not the voice — but a human
recording would be a fairer judge of the absolute numbers.

    python scripts/context_eval.py <cache dir>  →  results/results_context.json
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SENTENCES = [
    ("Консьюмер читает сообщения из Kafka и фиксирует offset после обработки.", ["Kafka", "offset"]),
    ("В Spring Boot транзакция открывается аннотацией Transactional.", ["Spring Boot", "Transactional"]),
    ("HashMap не потокобезопасен, для этого есть ConcurrentHashMap.", ["HashMap", "ConcurrentHashMap"]),
    ("Сервис отдаёт JSON по REST, а токен передаётся в JWT.", ["JSON", "REST", "JWT"]),
    ("Данные лежат в PostgreSQL, а кэш вынесен в DynamoDB.", ["PostgreSQL", "DynamoDB"]),
    ("Очередь на RabbitMQ, а сообщения с ошибкой уходят в DLQ.", ["RabbitMQ", "DLQ"]),
    ("ArrayList быстрее LinkedList почти во всех сценариях.", ["ArrayList", "LinkedList"]),
    ("CompletableFuture позволяет собрать цепочку асинхронных шагов.", ["CompletableFuture"]),
    ("JVM выделяет память под объекты в куче, а JIT компилирует горячий код.", ["JVM", "JIT"]),
    ("Запрос к базе идёт через JDBC, а маппинг делает JPA.", ["JDBC", "JPA"]),
    ("Если метод equals переопределён, hashCode тоже нужно переопределить.", ["equals", "hashCode"]),
    ("Producer пишет в partition по ключу сообщения.", ["producer", "partition"]),
    ("StringBuilder дешевле конкатенации строк в цикле.", ["StringBuilder"]),
    ("ReentrantLock даёт то же, что synchronized, но с таймаутом.", ["ReentrantLock"]),
    ("CountDownLatch ждёт, пока отработают все потоки.", ["CountDownLatch"]),
    ("Статика раздаётся через CDN, а API закрыт от CSRF.", ["CDN", "API", "CSRF"]),
    ("Кэш хранит запись, пока не истечёт TTL.", ["TTL"]),
    ("CQRS разделяет модели чтения и записи.", ["CQRS"]),
    ("WebFlux построен на реактивных потоках, а не на MVC.", ["WebFlux", "MVC"]),
    ("ThreadLocal хранит значение отдельно для каждого потока.", ["ThreadLocal"]),
]

PROMPT = ("Разговор разработчиков о бэкенде на Java: Spring, Kafka, базы данных, "
          "многопоточность.")


def call(path: str, *, json_body: dict | None = None, form: dict | None = None,
         file: bytes | None = None) -> bytes:
    if json_body is not None:
        req = urllib.request.Request(URL + path, data=json.dumps(json_body).encode(),
                                     headers={"Content-Type": "application/json"})
    else:
        b = "BOUND"
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                        for k, v in form.items() if v is not None)
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
                 ).encode() + file + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(URL + path, data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return OPENER.open(req, timeout=600).read()


def norm(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\w ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(r))


def has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def main() -> None:
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    variants = {"none": {}, "prompt": {"prompt": PROMPT},
                "hotwords": {"context": "engineering"},
                "both": {"context": "engineering", "prompt": PROMPT}}
    rows = []
    for i, (text, terms) in enumerate(SENTENCES):
        wav = cache / f"ctx{i:02d}.wav"
        if not wav.exists():
            wav.write_bytes(call("/v1/audio/speech", json_body={
                "input": text, "response_format": "wav", "prepare": True, "seed": 1}))
        row = {"text": text, "terms": terms}
        for name, extra in variants.items():
            hyp = json.loads(call("/v1/audio/transcriptions", file=wav.read_bytes(),
                                  form={"language": "ru", **extra}))["text"]
            row[name] = {"hyp": hyp, "hits": [t for t in terms if has_term(hyp, t)],
                         "cer": round(100 * cer(text, hyp), 1)}
        rows.append(row)
        print(f"{i:2d} " + "  ".join(f"{n}:{len(row[n]['hits'])}/{len(terms)}" for n in variants),
              flush=True)

    total = sum(len(r["terms"]) for r in rows)
    report = {n: {"terms_pct": round(100 * sum(len(r[n]["hits"]) for r in rows) / total, 1),
                  "cer_pct": round(sum(r[n]["cer"] for r in rows) / len(rows), 1)}
              for n in variants}
    report["terms"] = total
    print(json.dumps(report, ensure_ascii=False, indent=1))
    out = Path(__file__).resolve().parents[1] / "results" / "results_context.json"
    out.write_text(json.dumps({"report": report, "prompt": PROMPT, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
