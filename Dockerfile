# voicy — локальный речевой сервер: синтез (Qwen3-TTS) и распознавание (faster-whisper).
#
# Веса моделей в образ НЕ кладутся: вместе они весят около пяти гигабайт и живут
# своей жизнью от кода. Они скачиваются при первом обращении в то, что смонтировано
# как /cache — поэтому пересборка образа не роняет кэш, а перезапуск контейнера
# не приводит к повторной загрузке.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch отдельным слоем: он самый большой и меняется реже всего остального
RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY server/ /app/server/
COPY data/pronunciation.json /app/data/pronunciation.json

# Голоса лежат в образе, но каталог объявлен томом: добавленные через API
# переживают перезапуск, а образ остаётся самодостаточным. Контексты
# распознавания — так же.
VOLUME ["/cache", "/app/server/voices", "/app/server/contexts"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app/server"]
