# voicy — локальный речевой сервер: синтез (Qwen3-TTS) и распознавание (faster-whisper).
#
# Внутри — тот же бинарник на Rust, что в выпусках, и библиотеки его движков:
# llama.cpp, CTranslate2, ONNX Runtime, CUDA 13 и cuDNN. Python и torch не нужны.
# Хосту нужны драйвер NVIDIA с поддержкой CUDA 13 (580 и новее) и NVIDIA
# Container Toolkit; без проброшенной видеокарты всё работает на процессоре.
#
# Веса моделей в образ НЕ кладутся: около восьми гигабайт, и живут они своей
# жизнью от кода. Первый запуск качает их в то, что смонтировано как /cache, —
# пересборка образа кэш не роняет, перезапуск контейнера не качает заново.

# ------------------------------------------------------------------ сборка
FROM rust:1-bookworm AS build

# cmake, autotools и pkg-config — для кодеков, которые собираются статически;
# g++ — для обёртки над CTranslate2, её собирает voicy setup
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake pkg-config autoconf automake libtool g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml \
    && install -D rust/target/release/voicy /out/voicy

# Библиотеки движков — в образ, а не в том: их версия закреплена за бинарником
# (setup.rs), и с новым образом должны приходить новые
ENV VOICY_CACHE=/tmp/voicy-cache \
    VOICY_LIB_DIR=/opt/voicy/lib
RUN /out/voicy setup libs --yes && rm -rf /tmp/voicy-cache

# ------------------------------------------------------------------ образ
FROM ubuntu:24.04

# ffmpeg — только для aac: остальные форматы бинарник кодирует и читает сам
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/voicy /usr/local/bin/voicy
COPY --from=build /opt/voicy/lib /opt/voicy/lib
# Голоса лежат в образе, но каталог объявлен томом: добавленные через API
# переживают перезапуск, а образ остаётся самодостаточным. Пустую папку хоста
# сервер наполнит этими же голосами сам. Контексты распознавания — так же.
COPY rust/core/assets/voices/ /app/server/voices/

ENV VOICY_CACHE=/cache \
    VOICY_LIB_DIR=/opt/voicy/lib \
    VOICY_VOICES_DIR=/app/server/voices \
    VOICY_CONTEXTS_DIR=/app/server/contexts \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/cache", "/app/server/voices", "/app/server/contexts"]

EXPOSE 8080
# Первый запуск качает модели, и пока качает — сервер ещё не отвечает
HEALTHCHECK --interval=30s --timeout=5s --start-period=60m --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# Модели — в том /cache: setup докачает недостающее (всё на месте — выйдет сразу),
# сервер — процессом №1, чтобы `docker stop` доходил до него
CMD ["sh", "-c", "voicy setup models --yes && exec voicy serve --host 0.0.0.0 --port 8080"]
