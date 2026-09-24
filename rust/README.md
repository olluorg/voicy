# voicy на Rust

Тот же сервер, что `server/`, — те же маршруты, тот же набор проверок (`tests/`), —
одним бинарником, и в нём же команды CLI. Модели исполняются в самом процессе,
на готовых библиотеках llama.cpp, CTranslate2 и ONNX Runtime; чего нет в родном
виде, идёт через Python-движки в дочернем процессе. Как и почему —
[ADR 0022](../docs/adr/0022-rust-migration-and-hardware.md).

```bash
cargo build --release                         # rust/target/release/voicy
target/release/voicy setup                    # библиотеки и модели в ~/.cache/voicy (один раз)
target/release/voicy up                       # поднять и прогреть
target/release/voicy say "Проверка." out.wav  # тот же бинарник — и клиент
```

`setup` качает готовые библиотеки и модели (7.3 ГБ). Обёртку над C++-интерфейсом
CTranslate2 на Windows собирает `build.rs` — внутрь бинарника, с отложенной
загрузкой `ctranslate2.dll`, так что на машине, где сервер работает, компилятор
не нужен; на Unix её по-прежнему собирает `setup` (нужен g++). Python не нужен:
Qwen3-TTS берётся уже переведённым —
[sknyazev/qwen3-tts-12hz-1.7b-base-gguf](https://huggingface.co/sknyazev/qwen3-tts-12hz-1.7b-base-gguf),
свой репозиторий задаётся через `VOICY_TTS_GGUF_REPO`. Перевести официальные
веса самому — `scripts/convert_qwen.py`, вот там нужен PyTorch.

## Команды

`serve` и `setup` — своё; остальное повторяет Python-CLI, те же флаги и тот же
договор: аудио не пишется в stdout, в stdout идёт результат — путь, расшифровка
или id задания, всё прочее в stderr.

```
say hear job jobs cancel contexts voices add-voice prepare up down status warm
```

`up` поднимает сервер этим же бинарником отдельным процессом: pid и журнал —
в `<кэш>/run/`, останавливает его `down`. Адрес — `--url` или `VOICY_URL`.

## Что где исполняется

| | по умолчанию | как выбрать другое |
|---|---|---|
| синтез | Qwen3-TTS на llama.cpp + ONNX Runtime, talker q5_k | `TTS_GGUF_TALKER=q8_0\|f16`; `TTS_ENGINE=qwen3-tts` — PyTorch в Python |
| распознавание | faster-whisper: CTranslate2 и его логика на Rust (experiments/22) | `STT_ENGINE=whisper.cpp` — хуже на терминах (experiments/21) |
| конец реплики | Smart Turn на ONNX Runtime | `TURN_ENGINE=…` — движок Python |
| детектор голоса | Silero на ONNX Runtime | `VAD_ENGINE=…` — движок Python |

Родной движок включается, если его библиотеки и модель лежат в кэше; иначе
тот же движок идёт через Python, как в `server/`. Когда все четыре родные,
Python не запускается. `/health` показывает, кто где (`runtime`).

Сравнение с Python-сервером на RTX 3080: синтез ×3.8 против ×1.4 к реальному
времени при той же разборчивости и том же сходстве голоса (experiments/20);
распознавание с той же точностью и на 23% быстрее на коротких фразах
(experiments/22); видеопамять всей карты 5.7 ГБ против 9.3, Python не нужен.

## Откуда что берётся

- **Заголовки CTranslate2** — `ct2shim/include/`: те 77 файлов, что нужны
  обёртке, тега v4.8.2, чтобы сборка не зависела от сети. Рядом —
  `ctranslate2-msvc-x64.def`: двадцать символов, которые обёртка вызывает,
  из них делается импортная библиотека (в колесе CTranslate2 её нет).
- **Библиотеки** — `~/.cache/voicy/lib/<платформа>/` (или `VOICY_LIB_DIR`):
  llama.cpp b11090, whisper.cpp b5130, CTranslate2 4.8.2 (из колеса
  faster-whisper) с cuBLAS 12, ONNX Runtime 1.30 с провайдером CUDA, CUDA 13
  и cuDNN. Всё — готовые сборки; привязки закреплены за этими версиями
  (раскладка структур сверена с их заголовками). Собирается одно —
  `ct2shim/`, C-обёртка над C++ API CTranslate2 (`g++`, одна страница кода).
- **Модели** — `~/.cache/voicy/models/`. Whisper — те файлы CTranslate2,
  что качает faster-whisper; Silero — из колеса faster-whisper, файл в файл.
  Qwen3-TTS — GGUF и ONNX, переведённые из официальных весов скриптами
  HaujetZhao/Qwen3-TTS-GGUF (`scripts/convert_qwen.py`; этому шагу нужен Python
  с torch, и он уже сделан — файлы выложены).
- **Голоса, словарь произношений, консоль** — внутри бинарника; рядом
  с репозиторием берутся его файлы, без него голоса живут в `~/.cache/voicy/voices`.

Кэш переносится переменной `VOICY_CACHE`.

## Где проверено

Linux x86-64 и Windows x86-64, оба с NVIDIA. На Windows: синтез 5.04 с звука
за 1.17 с, распознавание верное, 45 из 45 проверок (16 пропущено — форматы,
которым нужен ffmpeg). Четыре особенности Windows, из-за которых это не
работало сразу, описаны в коде рядом с их решениями: экспорт функций из DLL,
имя в `LIBRARY` импортной библиотеки, `localhost` как IPv6 и два OpenMP
в одном процессе.

Библиотеки под Intel Arc (SYCL, Vulkan), Apple (Metal) и процессоры x86-64
и ARM64 у llama.cpp и ONNX Runtime есть; их сборка и замеры — следующий шаг.

## Проверки

```bash
VOICY_IMPL=rust python -m tests                 # свои серверы с учебными движками
python -m tests --url http://localhost:8080     # живой сервер
cargo test --release                            # раскладка структур whisper.cpp
```
