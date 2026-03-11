# Architecture

## Цель

Построить локальный мультимодальный pipeline анализа эмоций для одного видеопотока с опорой на:

- видео с камеры;
- загруженный видеофайл.

## Архитектура компонентов

```text
┌─────────────────┐
│ VideoSource     │  camera / uploaded video
└──────┬──────────┘
       │ frames
       v
┌─────────────────┐   optional: OpenCV haar cascade / other detectors
│ FaceDetector    │── bbox + landmarks
└──────┬──────────┘
       │ detections
       v
┌─────────────────┐
│ FaceTracker     │── stable face_id
└──────┬──────────┘
       │ tracked face
       v
┌─────────────────┐
│ FacePreprocess  │── aligned crop metadata
└──────┬──────────┘
       │ crop features
       v
┌─────────────────┐
│ FaceEmotion     │── emotion probabilities
└─────────────────┘

┌─────────────────┐
│ AudioPipeline   │  video audio / live microphone buffer
└──────┬──────────┘
       │ segments + prosody features
       v
┌───────────────┐
│ VoiceEmotion  │
│ prosody rules │
└──────┬────────┘
       │ emotion probabilities
       v
┌─────────────────┐
│ FusionEngine    │── final emotion
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ UI + Logger     │── GUI, HTML preview, CSV, summary
└─────────────────┘
```

## Интерфейсы модулей

- `video_capture.iter_video_frames()` → `FramePacket`
- `audio_pipeline.extract_speech_segments()` → `list[SpeechSegment]`
- `audio_pipeline.build_live_speech_segment()` → live `SpeechSegment`
- `FaceDetector.detect()` → `list[FaceDetection]`
- `FaceTracker.assign_ids()` → `list[FaceDetection]`
- `FaceEmotionInference.predict()` → `dict[str, float]`
- `VoiceEmotionAnalyzer.analyze()` → voice emotion probabilities
- `fuse_signals()` → final decision + triggered rules

## Resource targets

| Component | Target backend | MVP fallback |
|---|---|---|
| Video capture | OpenCV | explicit source error |
| Face detection | OpenCV / MediaPipe | metadata-based bbox fallback only when metadata exists |
| Voice emotion | SER model | prosody heuristics |
| UI | PySide6 | HTML preview |

## Latency budget (MVP target)

| Stage | Budget |
|---|---:|
| Frame capture | 10–20 ms |
| Face detect + preprocess | 25–40 ms |
| Face emotion inference | 15–30 ms |
| Audio extraction / live buffer | 20–60 ms |
| Fusion + UI refresh | 5–10 ms |
| **Total** | **≤ 200 ms target** |

## Дополнительные документы

- `docs/PROJECT_DETAILED_BREAKDOWN.md`
- `docs/ADVANCED_TRAINING_GUIDE.md`
