"""Декодер Qwen3-TTS на разных исполнителях ONNX Runtime: скорость и звук (experiments/27).

Декодирует потоком по 12 кадров, как decode() в rust/core/src/native/qwen.rs.
Коды — из настоящей записи (её прогоняет кодировщик кодека) или случайные.

    python scripts/decoder_ep_eval.py cpu    --wav rust/core/assets/voices/turgenev.wav --out cpu.wav
    python scripts/decoder_ep_eval.py ov-cpu --wav rust/core/assets/voices/turgenev.wav --out ov.wav

Исполнитель задаёт колесо в окружении: `onnxruntime` — `cpu`,
`onnxruntime-openvino` — `ov-cpu` или `ov-gpu` (видеокарта Intel; на Windows
к нему нужен пакет `openvino`), `onnxruntime-directml` — `dml`. Все три ставят
модуль `onnxruntime`, поэтому каждому — своё окружение. На Windows с Arc всё
это ставит и прогоняет `scripts/arc_eval.ps1`.

Если исполнитель не поднялся, ONNX Runtime молча уходит на CPU — тогда скрипт
падает, а не меряет не то.
"""
from __future__ import annotations

import argparse
import os
import time
import wave

import json
import sys

import numpy as np

if sys.platform == "win32":
    # колесо onnxruntime-openvino под Windows берёт библиотеки OpenVINO из пакета openvino
    try:
        import openvino
        libs = os.path.join(os.path.dirname(openvino.__file__), "libs")
        os.add_dll_directory(libs)
        os.environ["PATH"] = libs + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass

import onnxruntime as ort  # noqa: E402

# консоль Windows — cp1251, в ней нет ни «×», ни многого другого
sys.stdout.reconfigure(encoding="utf-8")
# предупреждения о свёртке констант — шум, а PowerShell показывает их как ошибки
ort.set_default_logger_severity(3)

MODELS = os.path.join(os.environ.get("VOICY_CACHE", os.path.expanduser("~/.cache/voicy")),
                      "models", "qwen3-tts-12hz-1.7b-ru-stress-gguf")
Q, CHUNK, LAYERS, RATE = 16, 12, 8, 24000


EP = {"ov": "OpenVINOExecutionProvider", "dml": "DmlExecutionProvider"}


def session(models: str, name: str, provider: str, precision: str | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list = ["CPUExecutionProvider"]
    if provider.startswith("ov-"):
        o = {"device_type": provider[3:].upper()}
        if precision:
            o["precision"] = precision
        providers.insert(0, (EP["ov"], o))
    elif provider == "dml":
        # DirectML не умеет ни шаблонов памяти, ни параллельного исполнения — как в native::session()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers.insert(0, EP["dml"])
    sess = ort.InferenceSession(os.path.join(models, name), opts, providers=providers)
    want = EP.get(provider.split("-")[0])
    if want and sess.get_providers()[0] != want:
        sys.exit(f"{provider}: {want} не поднялся, сессия ушла на {sess.get_providers()[0]}")
    return sess


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())[:, 0]
    return x.astype(np.float32) / 32768


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def decode(sess: ort.InferenceSession, codes: np.ndarray) -> tuple[np.ndarray, list[float]]:
    z = lambda *s: np.zeros(s, np.float16)  # noqa: E731
    st = {"pre_conv_history": z(1, 512, 0), "latent_buffer": z(1, 1024, 0), "conv_history": z(1, 1024, 0)}
    for l in range(LAYERS):
        st[f"past_key_{l}"] = st[f"past_value_{l}"] = z(1, 16, 0, 64)
    audio, per = [], []
    n = (len(codes) + CHUNK - 1) // CHUNK
    for i in range(n):
        last = i == n - 1
        feed = {"audio_codes": codes[i * CHUNK:(i + 1) * CHUNK][None],
                "is_last": np.array([1.0 if last else 0.0], np.float16), **st}
        t = time.perf_counter()
        r = sess.run(None, feed)
        per.append(time.perf_counter() - t)
        wav, valid = r[0].reshape(-1).astype(np.float32), int(np.asarray(r[1]).reshape(-1)[0])
        audio.append(wav if last else wav[:valid])
        st = {"pre_conv_history": r[2], "latent_buffer": r[3], "conv_history": r[4]}
        for l in range(LAYERS):
            st[f"past_key_{l}"], st[f"past_value_{l}"] = r[5 + l], r[5 + LAYERS + l]
    return np.concatenate(audio), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["cpu", "ov-cpu", "ov-gpu"])
    ap.add_argument("--wav", help="запись 24 кГц; без неё — 200 случайных кадров (16 с)")
    ap.add_argument("--out", help="куда записать декодированный звук")
    ap.add_argument("--precision", help="точность OpenVINO: FP32, FP16, ACCURACY")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--models", default=MODELS, help="каталог с файлами декодера и кодировщика")
    ap.add_argument("--json", help="дописать результат строкой JSON в этот файл")
    a = ap.parse_args()

    if a.wav:
        enc = session(a.models, "qwen3_tts_codec_encoder.fp32.onnx", "cpu")
        codes = enc.run(None, {"input_values": read_wav(a.wav)[None]})[0][0]
    else:
        codes = np.random.default_rng(0).integers(0, 2048, size=(200, Q), dtype=np.int64)

    t = time.perf_counter()
    sess = session(a.models, "qwen3_tts_decoder.fp16.onnx", a.provider, a.precision)
    load = time.perf_counter() - t
    t = time.perf_counter()
    decode(sess, codes)
    first = time.perf_counter() - t
    times = []
    for _ in range(a.runs):
        t = time.perf_counter()
        audio, per = decode(sess, codes)
        times.append(time.perf_counter() - t)
    if a.out:
        write_wav(a.out, audio)
    sec = len(audio) / RATE
    print(f"{a.provider}: {sess.get_providers()[0]}, загрузка {load:.1f} с, первый проход {first:.2f} с, "
          f"{sec:.1f} с звука за {' / '.join(f'{x:.2f}' for x in times)} с (×{sec / min(times):.1f}), "
          f"кусок из {CHUNK} кадров — {np.median(per) * 1000:.0f} мс")
    if a.json:
        with open(a.json, "a", encoding="utf-8") as f:
            f.write(json.dumps({"provider": a.provider, "precision": a.precision, "wav": a.wav,
                                "ort": ort.__version__, "load_s": load, "first_s": first, "runs_s": times,
                                "audio_s": sec, "chunk_median_ms": float(np.median(per) * 1000)},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
