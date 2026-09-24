"""Qwen3-TTS → GGUF и ONNX: перевод официальных весов в то, что исполняет сервер.

    python scripts/convert_qwen.py        →  ~/.cache/voicy/models/qwen3-tts-12hz-1.7b-base-gguf
    python scripts/convert_qwen.py --weights <каталог> --name <имя>
        — те же шаги над своими весами той же формы, например со слитым адаптером
          ударений (scripts/stress_ft_merge.py, docs/adr/0023)

Обычно он не нужен: `voicy setup` качает уже переведённые файлы
(sknyazev/qwen3-tts-12hz-1.7b-base-gguf) вместе с остальными моделями.
Этот скрипт — чтобы получить их самому: из официальных весов
Qwen3-TTS-12Hz-1.7B-Base скриптами HaujetZhao/Qwen3-TTS-GGUF (MIT) на
закреплённом коммите: говорящая часть и предсказатель — в GGUF, кодек
и кодировщик голоса — в ONNX, таблицы эмбеддингов — в .npy. Потом говорящая
часть квантуется в q5_k (та же разборчивость и то же сходство голоса, что
у f16, при 2.4 ГБ — experiments/20), предсказатель — в q8_0, декодер кодека
переводится в fp16.

Нужны: окружение сервера (torch, qwen-tts) плюс onnx, onnxscript, gguf; около
20 ГБ места под исходные веса и промежуточные файлы; `voicy setup libs` —
оттуда берётся llama-quantize. Считает на видеокарте, если она есть.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CACHE = Path(os.environ.get("VOICY_CACHE") or Path.home() / ".cache" / "voicy")
PLATFORM = "linux-x64-cuda13"
LIB = CACHE / "lib" / PLATFORM
MODELS = CACHE / "models"

CONVERTER = "https://github.com/HaujetZhao/Qwen3-TTS-GGUF.git"
CONVERTER_COMMIT = "74feb581bc8cefb835fc608107e857036d7580a1"
QWEN = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# что остаётся в кэше: остальное — промежуточные файлы перевода
KEEP = ["tokenizer.json", "qwen3_tts_talker.q5_k.gguf", "qwen3_tts_predictor.q8_0.gguf",
        "qwen3_tts_decoder.fp16.onnx", "qwen3_tts_codec_encoder.fp32.onnx",
        "qwen3_tts_codec_encoder.fp32.onnx.data", "qwen3_tts_speaker_encoder.fp32.onnx",
        "qwen3_tts_speaker_encoder.fp32.onnx.data"]


def say(msg: str) -> None:
    print(f"convert_qwen: {msg}", flush=True)


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def quantize_tool() -> Path:
    """llama-quantize из сборки llama.cpp, которую положил `voicy setup libs`."""
    tool = LIB.parent / "llama-quantize"
    if not tool.exists():
        raise SystemExit(f"нет {tool} — сначала: voicy setup libs")
    return tool


def main() -> None:
    import argparse
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser(description="Qwen3-TTS → GGUF и ONNX")
    ap.add_argument("--weights", type=Path, help="свои веса той же формы вместо официальных")
    ap.add_argument("--name", default="qwen3-tts-12hz-1.7b-base-gguf", help="каталог результата в моделях voicy")
    ap.add_argument("--talkers", default="q5_k", help="варианты говорящей части через запятую: q5_k,q8_0,f16")
    a = ap.parse_args()
    talkers = [t.strip().lower() for t in a.talkers.split(",") if t.strip()]

    out = MODELS / a.name
    if (out / "qwen3_tts_talker.q5_k.gguf").exists():
        say(f"уже переведён — {out}")
        return
    quantize = quantize_tool()
    weights = a.weights.resolve() if a.weights else Path(snapshot_download(QWEN))
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "conv"
        run(["git", "clone", "-q", CONVERTER, str(repo)], Path(d))
        run(["git", "checkout", "-q", CONVERTER_COMMIT], repo)
        # Исходники рассчитаны на ModelScope и Windows: путь к весам и к конвертеру GGUF
        cfg = repo / "export_config.py"
        cfg.write_text(cfg.read_text().replace("model_home / 'Qwen3-TTS-12Hz-1.7B-Base'", f"Path({str(weights)!r})"))
        conv = repo / "33-Convert-Predictor-GGUF.py"
        conv.write_text(conv.read_text().replace('os.path.join(PROJECT_ROOT, "ref", "llama.cpp")',
                                                 'os.path.join(PROJECT_ROOT, "qwen3_tts_gguf", "export")'))
        for step in ("11", "12", "13", "14", "15", "21", "22", "23", "31", "32", "33"):
            script = next(repo.glob(f"{step}-*.py"))
            say(f"шаг {script.name}")
            run([sys.executable, script.name], repo)

        src = repo / "model-base"
        env = {**os.environ, "LD_LIBRARY_PATH": str(LIB)}
        for name, kind in [("talker", t.upper()) for t in talkers if t != "f16"] + [("predictor", "Q8_0")]:
            say(f"квантование: {name} → {kind}")
            subprocess.run([str(quantize), str(src / f"qwen3_tts_{name}.f16.gguf"),
                            str(src / f"qwen3_tts_{name}.{kind.lower()}.gguf"), kind], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        say("декодер кодека — в fp16")
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        m = onnx.load(str(src / "qwen3_tts_decoder.fp32.onnx"))
        m16 = convert_float_to_float16(m, keep_io_types=False, op_block_list=["LayerNormalization", "Softmax", "Range"])
        onnx.save(m16, str(src / "qwen3_tts_decoder.fp16.onnx"))

        out.mkdir(parents=True, exist_ok=True)
        keep = [n for n in KEEP if not n.startswith("qwen3_tts_talker.")]
        keep += [f"qwen3_tts_talker.{t}.gguf" for t in talkers]
        for name in keep:
            shutil.copy(src / name, out / name)
        shutil.copytree(src / "embeddings", out / "embeddings", dirs_exist_ok=True)
    say(f"готово — {out}")


if __name__ == "__main__":
    main()
