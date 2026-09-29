# voicy-core

Движки voicy как библиотека для Rust: синтез, распознавание, детектор голоса
и конец реплики исполняются в процессе вызывающей программы, без сервера и HTTP.
Это те же движки, что работают в `voicy serve`: сервер — обёртка над этим крейтом.

```toml
[dependencies]
voicy-core = { git = "https://github.com/olluorg/voicy" }  # лучше закрепить tag выпуска
# async-обёртки и скачивание моделей из кода — по желанию:
# voicy-core = { git = "…", features = ["tokio", "download"] }
```

```rust
use voicy_core::{SpeakOptions, Stt, TranscribeOptions, Tts, Voices};

let tts = Tts::load(&Default::default())?;                  // ~4 с, 3.5 ГБ видеопамяти
let voice = Voices::in_cache()?.get("turgenev").unwrap();
let audio = tts.speak("Проверка связи.", &voice, &SpeakOptions::default())?;  // 24 кГц моно
std::fs::write("out.opus", voicy_core::audio::encode(&audio, tts.sample_rate(), "opus")?)?;

let stt = Stt::load(&Default::default())?;                  // 1.5 ГБ видеопамяти
let heard = stt.transcribe_file(&std::fs::read("out.opus")?, &TranscribeOptions::default())?;
println!("{}", heard.text);   // heard.srt(), heard.vtt(), heard.segments
```

Рабочий пример — `examples/roundtrip.rs`:
`cargo run --release -p voicy-core --example roundtrip -- "Текст." out.opus`.

## Что внутри

| | |
|---|---|
| `Tts` | Qwen3-TTS на llama.cpp и ONNX Runtime; словарь произношений, legato и ударения — `SpeakOptions`; темп 0.8–1.2 растяжением готового звука |
| `Stt` | Whisper large-v3-turbo так, как его исполняет faster-whisper; `hotwords`, `prompt`, отметки по словам, перевод |
| `Vad`, `VadStream`, `Turn` | Silero и Smart Turn для живого разговора: вероятность речи по кадрам 32 мс и «договорил ли» |
| `segment::Segmenter` | текст кусками (токены LLM) → куски, которые можно синтезировать сразу: первый звук не ждёт всего ответа |
| `audio` | чтение wav, mp3, flac, ogg, opus, webm, mp4; запись в wav, pcm, opus, mp3, flac; пересчёт частоты |
| `tempo` | темп без смены высоты |
| `textprep` | подготовка текста без синтеза: `prepare`, `legato`, `stress` |
| `voices` | голоса-образцы: два встроенных и каталог с `voices.json`, общий с сервером |
| `setup` | скачать библиотеки и модели в кэш (фича `download`) |

Движков на Python здесь нет — только родные. Нет и очереди, заданий, webhook:
это забота сервера.

## Что нужно на машине

Библиотеки движков (llama.cpp, CTranslate2, ONNX Runtime, CUDA) в программу
не компонуются: они грузятся из кэша при первой загрузке движка, как у сервера.
Программа поэтому запускается и без них. Кэш — `~/.cache/voicy`
(`%LOCALAPPDATA%\voicy`) или `VOICY_CACHE`; из кода — `voicy_core::use_dirs(...)`,
до первой загрузки движка. Наполняет его `voicy setup` (7.8 ГБ) либо сама
программа — `voicy_core::setup::run("all", true)` с фичей `download`.

Для сборки нужен CMake: кодек Opus собирается из исходников. С CMake 4 задайте
`CMAKE_POLICY_VERSION_MINIMUM=3.5` (например, в `[env]` своего
`.cargo/config.toml`): иначе CMake 4 не примет старый `cmake_minimum_required`
в исходниках Opus.

Обёртку над C++-интерфейсом CTranslate2 на Unix собирает `setup`, и ему нужен
`g++`. На Windows её собирает `build.rs` этого крейта (нужна Visual Studio
с C++ — та же, что и для любой сборки Rust под MSVC). Готовая `ct2shim.dll`
едет внутри крейта и кладётся к библиотекам движков при первой загрузке Whisper.

## Как звать

Все вызовы блокирующие: движок держит видеокарту, пока работает. Загруженные
`Tts` и `Stt` — `Send + Sync`, вызовы к одному движку идут по очереди, так что
один `Arc<Tts>` на всю программу — правильный способ. Фича `tokio` добавляет
`load_async`, `speak_async`, `transcribe_async` и `transcribe_file_async`:
то же на пуле блокирующих задач.

Отмена и прогресс — через `speak_with` и `transcribe_with`: колбэк получает,
сколько готово, и возвращает `false`, чтобы остановить работу; тогда результат — `None`.

Настройки по умолчанию от окружения не зависят. `TtsConfig::from_env()`
и `SttConfig::from_env()` читают то же, что сервер: `TTS_GGUF_TALKER`,
`VOICY_TTS_GGUF_DIR`, `FORCE_CPU`, `STT_DEVICE`, `STT_COMPUTE_TYPE`.

Сообщения идут через [`log`](https://docs.rs/log): чего не хватает (нет
RUAccent — синтез без ударений) — `warn`, ход установки — `info`. Без логгера
библиотека молчит. Сами llama.cpp и ggml при загрузке пишут в stderr пару строк
о найденной видеокарте — это их вывод, не крейта.
