# 01 — базовая линия: Piper и Silero v4

**Вопрос:** хватит ли лёгких CPU-движков.

**Скрипты:** `scripts/synth_piper.py`, `scripts/synth_silero.py`, `scripts/mkref.py`
**Замеры:** `results/results_piper.json`, `results/results_silero.json`
**Аудио:** `article/assets/audio/out__*piper*`, `out__*silero-*`

## Результат

| Движок | Скорость CPU | CER | Вердикт на слух |
|---|---|---|---|
| Piper irina/dmitri/ruslan | ×40 | 1.3–6.3% | «очень плохо» |
| Silero v4 xenia/eugene/baya | ×65 | 0.8–9.1% | «робот» |

Оба отвергнуты. Скорость проблемой не является ни у одного движка.

## Находка

Silero **молча выбрасывает латиницу**: `List<? extends Number> удобно читать`
озвучивается за 1.1 с вместо ожидаемых 3.4 с. Дефект беззвучный — на слух
неотличим от нормально прочитанной фразы. Отсюда требование автоматической
приёмки (ADR 0001).
