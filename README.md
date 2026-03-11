# Emotion AI System MVP

Локальный Python MVP для мультимодального анализа эмоций по видео и аудио. Приложение работает в двух пользовательских режимах:

- поток с камеры;
- загрузка готового видеофайла.

JSON-сценарии и demo-режимы из пользовательского потока удалены.

## Что умеет проект сейчас

- захватывает кадры из камеры или из загруженного видеофайла;
- детектирует лицо и считает базовые facial cues;
- извлекает аудио из видеофайла и строит сегменты речи;
- получает субтитры через `SpeechToTextService` (если доступна Vosk-модель);
- считает голосовые признаки `pitch / energy / tempo` даже без текста;
- объединяет face / voice / text сигналы через `fusion_engine`;
- показывает результат в GUI и экспортирует отчёт в `CSV`, `HTML` и `summary.txt`;
- сохраняет HTML preview с нормальной панелью управления видео.

## Python и платформы

- Python: **3.10+**
- основной target: **Windows / Linux**
- CPU fallback поддерживается
- если установлен Vosk и доступна русская модель, субтитры строятся локально офлайн

## Структура репозитория

```text
emo/
├─ README.md
├─ Architecture.md
├─ requirements.txt
├─ setup.sh
├─ setup.bat
├─ data/
├─ docs/
├─ experiments/
├─ notebooks/
├─ reports/
├─ src/
└─ tests/
```

## Быстрый запуск

### 1. Подготовка окружения

```bash
bash setup.sh
```

или в Windows:

```bat
setup.bat
```

### 2. Запуск GUI

```bash
python src/ui_app.py --gui --results-dir /tmp/emo-results
```

В интерфейсе доступны:

- **Камера** — потоковое видео с live-анализом;
- **Загрузить видео** — анализ видеофайла `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`;
- **Пауза / Заново / Стоп** — базовое управление анализом;
- сохранение результатов в `annotations.csv`, `ui_preview.html`, `summary.txt`.

### 3. Запуск через CLI

Видео:

```bash
python src/ui_app.py --source file --path /absolute/path/to/video.mp4 --results-dir /tmp/emo-results
```

Камера:

```bash
python src/ui_app.py --source camera --results-dir /tmp/emo-camera-results
```

Для ограничения длительности анализа можно добавить `--max-frames 100`.

## Что нужно для субтитров

Субтитры и текстовый анализ работают через `SpeechToTextService`.

1. Установите зависимости из `requirements.txt`.
2. Подготовьте локальную Vosk-модель русского языка, например `vosk-model-small-ru-0.22`.
3. Передайте путь через переменную окружения:

```bash
export EMO_VOSK_MODEL=/absolute/path/to/vosk-model-small-ru-0.22
```

или на Windows:

```bat
set EMO_VOSK_MODEL=C:\path\to\vosk-model-small-ru-0.22
```

Если модель не найдена, приложение всё равно считает голосовые признаки и эмоцию по голосу, но текстовые субтитры останутся пустыми.

## Выходные артефакты

После анализа создаются:

- `annotations.csv` — покадровый экспорт;
- `ui_preview.html` — HTML preview с видеоплеером и timeline;
- `summary.txt` — итоговая текстовая сводка.

## Подробная документация

- `Architecture.md` — краткая схема компонентов;
- `docs/PROJECT_DETAILED_BREAKDOWN.md` — максимально подробный разбор запуска, файлов, логики и формул;
- `docs/ADVANCED_TRAINING_GUIDE.md` — максимально подробный гайд по дополнительному обучению и расширению системы.
