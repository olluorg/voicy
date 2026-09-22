"""Prepare what the Rust server runs natively: runtime libraries and models.

    python scripts/native_setup.py libs        # llama.cpp, whisper.cpp, ONNX Runtime, CUDA
    python scripts/native_setup.py models      # Qwen3-TTS → GGUF/ONNX, Silero, Smart Turn
    python scripts/native_setup.py all

Everything goes into the cache the server reads (VOICY_CACHE, by default
~/.cache/voicy): lib/<platform>/ and models/. Nothing here is compiled.

  libs    prebuilt llama.cpp (b11090) and whisper.cpp (b5130) from their GitHub
          releases; ONNX Runtime with its CUDA provider and the CUDA and cuDNN
          libraries it needs, taken out of the wheels NVIDIA and Microsoft
          publish on PyPI. Linux x86-64 with CUDA 13 for now — the platform
          this was measured on; the other targets get their own archives.

  models  Qwen3-TTS-12Hz-1.7B-Base converted with the scripts of
          HaujetZhao/Qwen3-TTS-GGUF (MIT) at a pinned commit: the talker and
          the predictor to GGUF, the codec and speaker encoder to ONNX, the
          embedding tables to .npy. This one step needs Python with torch and
          the official weights — the environment the Python server already
          has. Then the talker is quantised to q5_k (experiments/20) and the
          predictor to q8_0. Silero and Smart Turn are the ONNX files the
          Python engines use.

Needs: git, gh or network access to github.com and pypi.org, and for models
the server's .venv (qwen-tts, torch) plus onnx, onnxscript, gguf.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("VOICY_CACHE") or Path.home() / ".cache" / "voicy")
PLATFORM = "linux-x64-cuda13"
LIB = CACHE / "lib" / PLATFORM
MODELS = CACHE / "models"

LLAMA = "b11090"
WHISPER = "b5130"
CONVERTER = "https://github.com/HaujetZhao/Qwen3-TTS-GGUF.git"
CONVERTER_COMMIT = "74feb581bc8cefb835fc608107e857036d7580a1"
QWEN = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Из колёс PyPI: ONNX Runtime с провайдером CUDA и то, что этому провайдеру нужно.
WHEELS = ["onnxruntime-gpu==1.30.0", "nvidia-cudnn-cu13", "nvidia-curand", "nvidia-cufft",
          "nvidia-nvjitlink", "nvidia-cuda-nvrtc"]
WANTED = ("libonnxruntime.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so",
          "libcudnn", "libcurand.so", "libcufft.so", "libnvJitLink.so", "libnvrtc")


def say(msg: str) -> None:
    print(f"native_setup: {msg}", flush=True)


def download(url: str, to: Path) -> Path:
    say(f"скачиваю {url.rsplit('/', 1)[-1]}")
    opener = urllib.request.build_opener()
    with opener.open(url, timeout=600) as r, open(to, "wb") as f:
        shutil.copyfileobj(r, f)
    return to


def release_asset(repo: str, tag: str, name: str, tmp: Path) -> Path:
    return download(f"https://github.com/{repo}/releases/download/{tag}/{name}", tmp / name)


def pypi_wheel(spec: str, dest: Path) -> Path:
    """The manylinux x86-64 wheel of `name[==version]` straight from PyPI's JSON API."""
    import json
    name, _, version = spec.partition("==")
    url = f"https://pypi.org/pypi/{name}/{version + '/' if version else ''}json"
    with urllib.request.urlopen(url, timeout=60) as r:
        files = json.load(r)["urls"]
    ok = [f for f in files if f["filename"].endswith(".whl") and "x86_64" in f["filename"]
          and "manylinux" in f["filename"] and ("cp312" in f["filename"] or "py3-none" in f["filename"])]
    if not ok:
        raise SystemExit(f"no linux x86-64 wheel for {spec}")
    return download(ok[0]["url"], dest / ok[0]["filename"])


def untar_libs(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if ".so" in m.name and (m.isfile() or m.issym()):
                m.name = Path(m.name).name
                t.extract(m, dest, filter="data")


def libs() -> None:
    LIB.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name in (f"llama-{LLAMA}-bin-ubuntu-cuda-13.4-x64.tar.gz",
                     f"cudart-llama-{LLAMA}-bin-ubuntu-cuda-13.4-x64.tar.gz"):
            untar_libs(release_asset("ggml-org/llama.cpp", LLAMA, name, tmp), LIB)
        # libwhisper — из сборки для процессора: CUDA ей даёт ggml llama.cpp (experiments/21)
        whisper = release_asset("ggml-org/whisper.cpp", WHISPER, "whisper-bin-ubuntu-x64.tar.gz", tmp)
        with tarfile.open(whisper) as t:
            for m in t.getmembers():
                if Path(m.name).name.startswith("libwhisper.so"):
                    m.name = Path(m.name).name
                    t.extract(m, LIB, filter="data")
        say("колёса ONNX Runtime, cuDNN и CUDA с PyPI")
        (tmp / "w").mkdir()
        for spec in WHEELS:
            pypi_wheel(spec, tmp / "w")
        for wheel in (tmp / "w").glob("*.whl"):
            with zipfile.ZipFile(wheel) as z:
                for n in z.namelist():
                    base = Path(n).name
                    if ".so" in base and base.startswith(WANTED):
                        (LIB / base).write_bytes(z.read(n))
    so = LIB / "libonnxruntime.so"
    if not so.exists():
        versioned = sorted(LIB.glob("libonnxruntime.so.*"))
        if versioned:
            so.symlink_to(versioned[-1].name)
    say(f"библиотеки — {LIB} ({len(list(LIB.iterdir()))} файлов)")


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def models() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    (MODELS / "vad").mkdir(parents=True, exist_ok=True)
    (MODELS / "turn").mkdir(parents=True, exist_ok=True)
    import faster_whisper
    shutil.copy(Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx", MODELS / "vad")
    shutil.copy(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"), MODELS / "turn")

    out = MODELS / "qwen3-tts-12hz-1.7b-base-gguf"
    if (out / "qwen3_tts_talker.q5_k.gguf").exists():
        say(f"Qwen3-TTS уже сконвертирован — {out}")
        return
    weights = Path(snapshot_download(QWEN))
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
        quantize = LIB.parent / "llama-quantize"
        if not quantize.exists():
            # утилита квантования — из той же сборки llama.cpp
            with tempfile.TemporaryDirectory() as t:
                arc = release_asset("ggml-org/llama.cpp", LLAMA, f"llama-{LLAMA}-bin-ubuntu-cuda-13.4-x64.tar.gz", Path(t))
                with tarfile.open(arc) as tar:
                    m = next(m for m in tar.getmembers() if Path(m.name).name == "llama-quantize")
                    m.name = "llama-quantize"
                    tar.extract(m, LIB.parent, filter="data")
        for name, kind in (("talker", "Q5_K"), ("predictor", "Q8_0")):
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
        keep = ["tokenizer.json", "qwen3_tts_talker.q5_k.gguf", "qwen3_tts_predictor.q8_0.gguf",
                "qwen3_tts_decoder.fp16.onnx", "qwen3_tts_codec_encoder.fp32.onnx",
                "qwen3_tts_codec_encoder.fp32.onnx.data", "qwen3_tts_speaker_encoder.fp32.onnx",
                "qwen3_tts_speaker_encoder.fp32.onnx.data"]
        for name in keep:
            shutil.copy(src / name, out / name)
        shutil.copytree(src / "embeddings", out / "embeddings", dirs_exist_ok=True)
    say(f"модели — {MODELS}")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("libs", "all"):
        libs()
    if what in ("models", "all"):
        models()


if __name__ == "__main__":
    main()
