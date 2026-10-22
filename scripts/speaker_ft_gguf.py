"""Merged weights → a model directory the Rust server reads (VOICY_TTS_GGUF_DIR).

    python scripts/speaker_ft_gguf.py <merged_weights> <out_dir>

A LoRA changes only talker.model.layers.*, so of everything convert_qwen.py
produces only qwen3_tts_talker.q5_k.gguf differs. This runs just the talker
steps of the converter (HaujetZhao/Qwen3-TTS-GGUF, the commit convert_qwen.py
pins: 21 extract, 22 tokenizer, 23 GGUF f16), quantizes to q5_k with the
llama-quantize of `voicy setup libs`, and takes every other file — predictor,
codec, embeddings, tokenizer, stress.json — from the released stress model.
CPU only, ~4 GB of memory, ~3.5 GB of disk in the converter's directory.

    VOICY_TTS_GGUF_DIR=<out_dir> voicy up --native
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from convert_qwen import CONVERTER, CONVERTER_COMMIT

if sys.platform == "win32":
    CACHE = Path(os.environ.get("VOICY_CACHE") or Path(os.environ["LOCALAPPDATA"]) / "voicy")
    LIB, QUANTIZE = CACHE / "lib" / "windows-x64-cuda13", CACHE / "lib" / "llama-quantize.exe"
else:
    CACHE = Path(os.environ.get("VOICY_CACHE") or Path.home() / ".cache" / "voicy")
    LIB, QUANTIZE = CACHE / "lib" / "linux-x64-cuda13", CACHE / "lib" / "llama-quantize"
LIKE = CACHE / "models" / "qwen3-tts-12hz-1.7b-ru-stress-gguf"
TALKER = "qwen3_tts_talker.q5_k.gguf"


def say(msg):
    print(f"speaker_ft_gguf: {msg}", flush=True)


def main(weights, out):
    weights, out = Path(weights).resolve(), Path(out).resolve()
    conv = Path(os.environ.get("CONV_DIR") or Path(tempfile.gettempdir()) / "qwen3-tts-gguf")
    if not (conv / ".git").exists():
        subprocess.run(["git", "clone", "-q", CONVERTER, str(conv)], check=True)
    subprocess.run(["git", "checkout", "-q", "-f", CONVERTER_COMMIT], cwd=conv, check=True)
    cfg = conv / "export_config.py"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("model_home / 'Qwen3-TTS-12Hz-1.7B-Base'",
                                                           f"Path({str(weights)!r})"), encoding="utf-8")
    shutil.rmtree(conv / "model-base", ignore_errors=True)
    for step in ("21", "22", "23"):
        script = next(conv.glob(f"{step}-*.py"))
        say(f"шаг {script.name}")
        subprocess.run([sys.executable, script.name], cwd=conv, check=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    out.mkdir(parents=True, exist_ok=True)
    for f in LIKE.iterdir():
        if f.name.startswith("qwen3_tts_talker.") or (out / f.name).exists():
            continue
        (shutil.copytree if f.is_dir() else shutil.copy)(f, out / f.name)
    say(f"квантование → {TALKER}")
    env = {**os.environ, "PATH": f"{LIB}{os.pathsep}{os.environ['PATH']}", "LD_LIBRARY_PATH": str(LIB)}
    subprocess.run([str(QUANTIZE), str(conv / "model-base" / "qwen3_tts_talker.f16.gguf"), str(out / TALKER), "Q5_K"],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    say(f"готово — {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
