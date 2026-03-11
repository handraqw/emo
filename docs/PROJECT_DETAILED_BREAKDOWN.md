# Подробный разбор проекта Emotion AI System

Этот документ описывает проект максимально подробно: от запуска приложения до формирования итоговой эмоции, сохранения результатов и точек расширения.

---

## 1. Назначение проекта

Система анализирует один видеопоток и пытается ответить на вопрос:

> какая эмоция наблюдается по лицу, голосу и тексту речи в конкретный момент времени?

Поддерживаются два пользовательских сценария:

1. **Поток с камеры** — видео идёт с камеры, звук берётся с микрофона.
2. **Загрузка видео** — пользователь выбирает файл `.mp4/.avi/.mov/.mkv/.webm`, после чего система читает кадры и извлекает аудио.

После анализа проект выдаёт:

- покадровый CSV;
- HTML preview с плеером;
- текстовую summary-сводку.

---

## 2. Общая схема работы

```text
Пользователь
   │
   ├── выбирает камеру
   │        │
   │        ├── video_capture.iter_video_frames(source="camera")
   │        └── Qt microphone monitor → build_live_speech_segment(...)
   │
   └── загружает видео
            │
            ├── video_capture.iter_video_frames(source="file", path=...)
            └── audio_pipeline.extract_speech_segments(path=...)
                     │
                     ├── ffmpeg / imageio-ffmpeg → WAV
                     ├── сегментация речи
                     └── вычисление voice features

Дальше для каждого кадра:
   │
   ├── FaceDetector
   ├── FaceTracker
   ├── FaceEmotionInference
   ├── SpeechToTextService
   ├── TextToxicityAnalyzer
   ├── VoiceEmotionAnalyzer
   └── FusionEngine
           │
           ├── итоговая эмоция
           ├── timeline
           ├── GUI
           └── export (CSV / HTML / summary)
```

---

## 3. Как запускается программа

Главная точка входа: `src/ui_app.py`

### 3.1. CLI-параметры

Функция `main()` поддерживает:

- `--source file|camera`
- `--path /abs/path/to/video.mp4`
- `--results-dir`
- `--max-frames`
- `--gui`

### 3.2. Два режима запуска

#### Режим GUI

Если передан `--gui` или `--source` не указан, вызывается:

```python
launch_gui(results_dir=args.results_dir)
```

Это создаёт Qt-окно `EmotionWindow`.

#### Режим CLI pipeline

Если пользователь явно указал `--source`, вызывается:

```python
run_pipeline(source=args.source, path=args.path, results_dir=args.results_dir)
```

Этот путь нужен для пакетного анализа без интерактивного GUI.

---

## 4. Что делает `run_pipeline()`

Файл: `src/ui_app.py`

Последовательность такая:

1. создаётся директория результатов;
2. вызывается `analyze_source(...)`;
3. результат сохраняется в `_write_results(...)`.

### 4.1. Что возвращает `analyze_source()`

Функция возвращает `list[dict]`, где каждый элемент — одна запись timeline.

У записи есть поля:

- `timestamp_ms`
- `bbox`
- `face_id`
- `face_emotion`
- `face_confidence`
- `speech_text`
- `audio_level`
- `text_toxicity_label`
- `text_toxicity_score`
- `voice_emotion`
- `voice_confidence`
- `final_emotion`
- `triggered_rules`
- `source`

---

## 5. Видеочасть: как читаются кадры

Файл: `src/video_capture.py`

Главная функция:

```python
iter_video_frames(source: str, path: str | None = None, max_frames: int | None = None)
```

### 5.1. Источники

- `source="camera"` → `cv2.VideoCapture(0)`
- `source="file"` → `cv2.VideoCapture(path)`

### 5.2. Что происходит на каждом кадре

Для каждого реального кадра строится `FramePacket`, в котором есть:

- индекс кадра;
- timestamp в миллисекундах;
- ширина/высота;
- `metadata`.

### 5.3. Что лежит в `metadata`

Функция `_build_visual_metadata(image)` вычисляет:

- `hint_bbox`
- `smile_score`
- `brow_tension`
- `mouth_open_score`
- `eye_open_score`
- `symmetry_score`
- `warmth_score`
- сам `image`

Это нужно, чтобы downstream-модули могли анализировать лицо даже при очень простом baseline.

### 5.4. Важное изменение

JSON-вход больше не поддерживается.  
Если пользователь попытается открыть `.json`, будет выброшен `VideoSourceError`.

---

## 6. Аудиочасть: как извлекается звук

Файл: `src/audio_pipeline.py`

Теперь проект умеет работать с реальной аудиодорожкой видео.

### 6.1. Поиск ffmpeg

Сначала вызывается `_resolve_ffmpeg_binary()`:

1. пробуется системный `ffmpeg`;
2. если его нет — пробуется `imageio_ffmpeg.get_ffmpeg_exe()`.

Это важно, потому что в некоторых средах системного ffmpeg нет вообще.

### 6.2. Извлечение audio track

Функция `_extract_audio_track(video_path)`:

1. создаёт временный `.wav`;
2. запускает ffmpeg:

```text
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -f wav temp.wav
```

3. читает WAV;
4. превращает его в массив PCM samples.

### 6.3. Почему `16 kHz mono`

Это компромисс:

- хватает для речи;
- меньше вычислений;
- удобно для Vosk;
- стабильнее для простых voice features.

---

## 7. Сегментация речи

После извлечения WAV вызывается `_segment_audio(samples, sample_rate)`.

### 7.1. Окно анализа

Аудио делится на окна длиной:

```text
WINDOW_MS = 30 ms
```

То есть для 16 kHz:

```text
window_size = 16000 * 0.03 = 480 samples
```

### 7.2. Энергия окна

Для каждого окна считается RMS:

```text
RMS = sqrt((x1² + x2² + ... + xn²) / n)
```

где `x_i` — нормализованные samples в диапазоне `[-1, 1]`.

### 7.3. Порог речи

В baseline используется:

```text
threshold = max(avg_window_energy * 1.4, 0.03)
```

Идея такая:

- если энергия окна выше порога — считаем, что это участок речи;
- небольшие паузы внутри речи допускаются;
- слишком короткие всплески отбрасываются.

### 7.4. Ограничения

Это не production VAD.  
Это простой energy-based speech segmentation, достаточный для локального baseline.

---

## 8. Как считаются признаки голоса

Тот же `src/audio_pipeline.py`, функция `_estimate_voice_features(samples, sample_rate)`.

Возвращаются три признака:

- `energy`
- `pitch`
- `tempo`

### 8.1. Energy

Снова берётся RMS и приводится к диапазону `[0, 1]`.

```text
energy = clamp(RMS * 3.2, 0, 1)
```

Чем громче голос, тем выше `energy`.

### 8.2. Pitch

В baseline pitch оценивается через zero crossing rate:

```text
estimated_hz = zero_crossings / (2 * duration_seconds)
pitch = clamp((estimated_hz - 80) / 220, 0, 1)
```

Это не точная фундаментальная частота, а грубая proxy-оценка.

### 8.3. Tempo

Tempo оценивается по числу энергетических всплесков в окнах:

```text
tempo = clamp(bursts_per_second / 4.5, 0, 1)
```

Смысл:

- частые энергетические импульсы → более быстрый темп;
- редкие → спокойная речь.

---

## 9. Live audio для камеры

В GUI `EmotionWindow` уже читал уровень микрофона.  
Теперь буфер сырых PCM samples не теряется.

### 9.1. Где это происходит

Файл: `src/ui_app.py`

- `_consume_microphone_level()`:
  - читает байты из `QAudioSource`;
  - преобразует их в `array("h")`;
  - обновляет `self._microphone_buffer`.

- `_apply_live_audio_context(frame)`:
  - на каждом кадре строит `SpeechSegment` через `build_live_speech_segment(...)`;
  - кладёт его в `frame.metadata["speech_segment"]`.

### 9.2. Зачем это нужно

Раньше камера умела показывать только уровень микрофона.  
Теперь этот буфер реально идёт дальше в voice/text pipeline.

---

## 10. Как строятся субтитры

Файл: `src/speech_to_text.py`

Класс: `SpeechToTextService`

### 10.1. Логика работы

1. Если в `SpeechSegment.text` уже есть текст — он возвращается сразу.
2. Иначе, если в `metadata` есть `audio_bytes`, система пытается запустить Vosk.
3. Если модели нет — возвращается пустая строка.

### 10.2. Как ищется модель

Проверяются:

- путь, переданный в конструктор;
- `EMO_VOSK_MODEL`;
- `models/vosk-model-small-ru-0.22`;
- `data/vosk-model-small-ru-0.22`.

### 10.3. Почему это важно

Теперь pipeline не привязан к тестовым строкам в metadata.  
Он умеет брать реальный audio segment и пытаться получить текст.

---

## 11. Анализ текста

Файл: `src/text_toxicity.py`

`TextToxicityAnalyzer` — rule-based классификатор для русского текста.

Он ищет:

- агрессивные слова;
- rude-маркеры;
- саркастические маркеры.

Результат:

```python
{
  "label": "AGGRESSION" | "RUDE" | "SARCASM" | "NEUTRAL",
  "score": float,
  "matches": list[str]
}
```

---

## 12. Анализ эмоции по голосу

Файл: `src/voice_emotion.py`

`VoiceEmotionAnalyzer` получает `SpeechSegment`, вытаскивает:

- `energy`
- `pitch`
- `tempo`
- текст

и на rule-based логике возвращает распределение по эмоциям.

Например:

- высокий `energy` или `pitch` → `AGGRESSION`;
- позитивные слова → `JOY`;
- саркастические маркеры → `SARCASM`;
- умеренно высокий `tempo/energy` → `ANGER` или `IRRITATION`.

---

## 13. Анализ лица

### 13.1. Детекция лица

Файл: `src/face_detector.py`

Алгоритм:

1. если доступен OpenCV cascade и есть реальное изображение — пробуем детектировать лицо;
2. если face detection не сработал и нет metadata-подсказки — возвращаем пустой список;
3. если подсказка есть — можно использовать fallback bbox.

### 13.2. Трекинг

Файл: `src/face_tracker.py`

Задача — удерживать стабильный `face_id`, чтобы не мигали метки между кадрами.

### 13.3. Предобработка

Файл: `src/face_preprocess.py`

Готовит crop лица, который потом читает `FaceEmotionInference`.

### 13.4. Классификация эмоции лица

Файл: `src/face_model/inference_face_emotion.py`

Baseline берёт признаки:

- улыбка;
- напряжение бровей;
- открытость рта;
- открытость глаз;
- симметрия;
- “теплота” изображения.

На их основе строится простое rule-based распределение по эмоциям.

---

## 14. Что происходит, если лицо не найдено

Это важная часть исправления.

Раньше `speech_text` и `voice_emotion` исчезали, если `detections == []`.  
Теперь в `src/ui_app.py`, внутри `_analyze_frame(...)`, создаётся **audio-only record**:

- `face_emotion = "NO_FACE"`
- `bbox = [0, 0, 0, 0]`
- `face_confidence = 0.0`

Но при этом сохраняются:

- `speech_text`
- `audio_level`
- `voice_emotion`
- итог `final_emotion`

Это позволяет:

- не терять субтитры;
- не терять эмоцию по голосу;
- не ломать timeline и экспорт.

---

## 15. Fusion: как получается итоговая эмоция

Файл: `src/fusion_engine.py`

Весовая схема:

```text
face = 0.40
voice = 0.35
text = 0.25
```

### 15.1. Общая формула

Для каждой эмоции:

```text
combined[label] =
    face_probs[label]  * 0.40 +
    voice_probs[label] * 0.35 +
    text_bonus[label]  * 0.25
```

### 15.2. Правило конфликта

Если лицо выглядит позитивно/нейтрально, а текст токсичный:

```text
happy_or_neutral_face_with_toxic_text → CONFLICT
```

Это специально выделенный режим, когда модальности противоречат друг другу.

### 15.3. Правило агрессивного голоса

Если лицо нейтральное, но голос явно агрессивный:

```text
aggressive_voice_overrides_neutral_face
```

Тогда `AGGRESSION` получает дополнительный буст.

---

## 16. Сглаживание по времени

В `src/ui_app.py` есть две стадии smoothing:

### 16.1. Face smoothing

```python
FACE_SMOOTHING_WEIGHT = 0.68
```

Новая оценка лица смешивается с предыдущей:

```text
smoothed = current * 0.68 + previous * 0.32
```

### 16.2. Decision smoothing

```python
DECISION_SMOOTHING_WEIGHT = 0.7
```

Это уменьшает “дребезг” эмоций между соседними кадрами.

---

## 17. Как обновляется GUI

### 17.1. Основные элементы `EmotionWindow`

- кнопка **Камера**
- кнопка **Загрузить видео**
- **Стоп**
- **Пауза**
- **Заново**
- preview-область
- блок субтитров
- audio meter
- правая панель с метриками
- timeline table

### 17.2. Что делает таймер

Qt-таймер раз в ~40 ms:

1. читает следующий кадр;
2. добавляет live-аудио контекст (для камеры);
3. вызывает `analyze_stream_frame(...)`;
4. обновляет preview / subtitle / timeline / side panel.

---

## 18. HTML preview и панель плеера

`_render_html_preview(...)` сохраняет `ui_preview.html`.

Для загруженного видео теперь там есть нормальная панель:

- play / pause;
- restart;
- шаг назад на 5 секунд;
- шаг вперёд на 5 секунд;
- выбор скорости;
- mute;
- fullscreen;
- seek slider;
- timecode.

Это независимый от Qt способ посмотреть результат анализа.

---

## 19. Экспорт результатов

Функция `_write_results(...)` сохраняет:

- `annotations.csv`
- `ui_preview.html`
- `summary.txt`

Пользовательская JSON-поддержка из export flow убрана.

---

## 20. Основные файлы и их роль

### `src/ui_app.py`

Главный orchestration-файл:

- запуск GUI;
- запуск CLI pipeline;
- сглаживание;
- сбор records;
- экспорт;
- HTML preview.

### `src/video_capture.py`

Чтение кадров из камеры/видео и подготовка визуальных metadata.

### `src/audio_pipeline.py`

Извлечение аудио, сегментация речи, voice features, live audio segment.

### `src/speech_to_text.py`

Переход от speech segment к тексту через Vosk.

### `src/voice_emotion.py`

Эвристики эмоции по голосу.

### `src/text_toxicity.py`

Эвристики токсичности/сарказма по тексту.

### `src/fusion_engine.py`

Объединение модальностей в финальное решение.

---

## 21. Ограничения текущей реализации

1. Vosk-модель не входит в репозиторий, её нужно положить отдельно.
2. Voice features — упрощённые proxy-оценки, а не полноценная SER-модель.
3. Face pipeline — baseline, не полноценный deep model.
4. VAD сегментация по энергии шумоуязвима.
5. Для очень сложных видео нужен более продвинутый audio/video sync.

---

## 22. Что улучшать дальше

1. заменить zero-crossing pitch на autocorrelation/YIN;
2. заменить energy-VAD на WebRTC VAD или нейросетевый VAD;
3. добавить полноценную SER-модель;
4. подключить более сильный face detector и face emotion model;
5. добавить явную синхронизацию аудио-сегментов и видеокадров;
6. расширить GUI встроенным scrubbing по timeline анализа.

---

## 23. Где читать про обучение

Для дополнительного обучения и расширения проекта см.:

- `docs/ADVANCED_TRAINING_GUIDE.md`
