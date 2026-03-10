# Emotion AI System MVP

Локальный Python MVP для мультимодального распознавания эмоций в видеопотоке с приоритетом запуска на Windows + NVIDIA 4060 8 GB VRAM и CPU fallback для других платформ.

## Что уже реализовано

- архитектурный скелет проекта с разбиением на `video_pipeline`, `audio_pipeline`, `fusion`, `inference`, `ui`;
- локальный demo pipeline без обязательных внешних зависимостей;
- pluggable backends для MediaPipe / OpenCV / Vosk / PyTorch, которые можно подключить позднее;
- rule-based baseline для face / voice / text анализа с детекцией `CONFLICT`;
- экспорт результатов в JSON, CSV и HTML preview;
- базовые unit tests и CI workflow.

## Python и платформы

- Python: **3.10+** (проверено на 3.12)
- приоритетный target: **Windows + NVIDIA 4060 8 GB**
- fallback: Linux/macOS CPU
- Mac M4: поддерживается как CPU/ONNX fallback, CoreML описан как следующий шаг

## Структура репозитория

```text
emotion-ai-system/
├─ README.md
├─ Architecture.md
├─ requirements.txt
├─ setup.sh
├─ setup.bat
├─ data/
├─ docs/
├─ experiments/
├─ examples/
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

Или в Windows PowerShell / cmd:

```bat
setup.bat
```

### 2. Графический интерфейс

```bash
python src/ui_app.py --gui --results-dir /tmp/emo-results
```

В GUI доступны:

- запуск анализа эмоций с камеры в реальном времени без подмены synthetic demo-потоком;
- загрузка видеофайла (`.mp4`, `.avi`, `.mov`, `.mkv`) для покадрового анализа с явной ошибкой, если файл не читается;
- открытие JSON demo-сценария как отдельного режима;
- сохранение результатов в `JSON`, `CSV`, `HTML` и `summary.txt`.

### 3. Demo режим с файлом

```bash
python src/ui_app.py --source file --path examples/demo_input.json --results-dir /tmp/emo-results
```

### 4. Demo режим с камерой

```bash
python src/ui_app.py --source camera --results-dir /tmp/emo-camera-results
```

Для ограничения длительности live/demo режима можно добавить `--max-frames 100`.

## Поддерживаемые артефакты

После запуска UI pipeline создаёт:

- `annotations.json` — покадровая разметка;
- `annotations.csv` — плоский экспорт для анализа;
- `ui_preview.html` — pseudo-GUI preview для демонстрации;
- `summary.txt` — человекочитаемый итог.

## Что считается MVP в этом репозитории

Текущая реализация отдаёт **рабочий локальный baseline**, который:

1. читает JSON-файл, видеозапись или поток с камеры;
2. находит/нормализует лицо;
3. оценивает face emotion;
4. обрабатывает speech segment → STT → toxicity + voice emotion;
5. объединяет сигналы через `fusion_engine`;
6. показывает live GUI preview и экспортирует результаты.

Это baseline-реализация для дальнейшего подключения реальных моделей MediaPipe/Vosk/PyTorch/ONNX. В коде предусмотрены адаптеры и точки расширения, чтобы заменить эвристики на обученные модели без смены интерфейсов.

## Ограничения текущего baseline

- точность 95/98% не заявляется для rule-based baseline без дообучения;
- экспорт реального annotated video и live GPU inference требуют установки optional зависимостей;
- training scripts реализованы как scaffold + baseline checkpoint flow, а не как full ResNet/SER training loop.

## Следующий шаг для production-like MVP

- подключить MediaPipe/OpenCV в `face_detector.py`;
- заменить baseline `FaceEmotionInference` на ResNet18 checkpoint;
- заменить `SpeechToTextService` на Vosk/Silero;
- заменить `VoiceEmotionAnalyzer` на SER модель;
- экспортировать модели в ONNX и проверить latency на целевом ноутбуке.
