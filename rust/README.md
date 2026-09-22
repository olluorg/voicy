# voicy на Rust

Тот же сервер, что `server/`, — те же маршруты, тот же набор проверок (`tests/`), —
одним бинарником. Модели исполняются в нём самом, на готовых библиотеках
llama.cpp, whisper.cpp и ONNX Runtime; чего нет в родном виде, идёт через
Python-движки в дочернем процессе. Как и почему — [ADR 0022](../docs/adr/0022-rust-migration-and-hardware.md).

```bash
cargo build --release                         # rust/target/release/voicy
python scripts/native_setup.py all            # библиотеки и модели в ~/.cache/voicy (один раз)
rust/target/release/voicy serve --port 8080
```

## Что где исполняется

| | по умолчанию | как выбрать другое |
|---|---|---|
| синтез | Qwen3-TTS на llama.cpp + ONNX Runtime, talker q5_k | `TTS_GGUF_TALKER=q8_0\|f16`; `TTS_ENGINE=qwen3-tts` — PyTorch в Python |
| распознавание | faster-whisper в Python-процессе | `STT_ENGINE=whisper.cpp` — без Python, но хуже на терминах (experiments/21) |
| конец реплики | Smart Turn на ONNX Runtime | `TURN_ENGINE=…` — движок Python |
| детектор голоса | Silero на ONNX Runtime | `VAD_ENGINE=…` — движок Python |

Родной движок включается, если его библиотеки и модель лежат в кэше; иначе
тот же движок идёт через Python, как в `server/`. Когда все четыре родные,
Python не запускается. `/health` показывает, кто где (`runtime`).

Сравнение с Python-сервером на RTX 3080: синтез ×3.8 против ×1.4 к реальному
времени при той же разборчивости и том же сходстве голоса, видеопамять всей
карты 5.9 ГБ против 9.3 (experiments/20).

## Откуда что берётся

- **Библиотеки** — `~/.cache/voicy/lib/<платформа>/` (или `VOICY_LIB_DIR`):
  llama.cpp b11090, whisper.cpp b5130, ONNX Runtime 1.30 с провайдером CUDA,
  CUDA 13 и cuDNN. Всё — готовые сборки; привязки закреплены за этими версиями
  (раскладка структур сверена с их заголовками).
- **Модели** — `~/.cache/voicy/models/`. Qwen3-TTS переводится из официальных
  весов скриптами HaujetZhao/Qwen3-TTS-GGUF — это единственный шаг, которому
  нужен Python с torch.
- **Голоса, словарь произношений, консоль** — внутри бинарника; рядом
  с репозиторием берутся его файлы, без него голоса живут в `~/.cache/voicy/voices`.

Кэш переносится переменной `VOICY_CACHE`.

## Сейчас только

Linux x86-64 с NVIDIA — где мерилось. Библиотеки под Intel Arc (SYCL, Vulkan),
Apple (Metal) и процессоры x86-64 и ARM64 у llama.cpp и ONNX Runtime есть;
их сборка и замеры — следующий шаг.

## Проверки

```bash
VOICY_IMPL=rust python -m tests                 # свои серверы с учебными движками
python -m tests --url http://localhost:8080     # живой сервер
cargo test --release                            # раскладка структур whisper.cpp
```
