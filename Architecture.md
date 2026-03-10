# Architecture

## Цель

Построить локальный мультимодальный pipeline распознавания эмоций для одного медиапотока с latency budget до 200 ms на MVP-пути.

## Архитектура компонентов

```text
┌─────────────────┐
│ VideoSource     │  file / camera / RTSP / synthetic JSON
└──────┬──────────┘
       │ frames
       v
┌─────────────────┐   optional: MediaPipe / OpenCV
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
│ FacePreprocess  │── aligned crop metadata (224x224 target)
└──────┬──────────┘
       │ crop features
       v
┌─────────────────┐   optional: ResNet18 / ONNX Runtime CUDA
│ FaceEmotion     │── emotion probabilities
└─────────────────┘

┌─────────────────┐
│ AudioPipeline   │  file audio / microphone / synthetic segments
└──────┬──────────┘
       │ segments
       ├───────────────┐
       v               v
┌──────────────┐   ┌───────────────┐
│ SpeechToText │   │ VoiceEmotion  │
│ Vosk/Silero  │   │ prosody/SER   │
└──────┬───────┘   └──────┬────────┘
       │ text             │ emotion probabilities
       v                  │
┌──────────────┐          │
│ TextToxicity │──────────┘
│ ru toxicity  │
└──────┬───────┘
       │ text label + score
       v
┌─────────────────┐
│ FusionEngine    │── final emotion / CONFLICT
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ UI + Logger     │── console, HTML preview, JSON/CSV
└─────────────────┘
```

## Интерфейсы модулей

- `video_capture.iter_video_frames()` → `FramePacket`
- `face_detector.FaceDetector.detect()` → `list[FaceDetection]`
- `face_tracker.FaceTracker.assign_ids()` → `list[FaceDetection]`
- `face_preprocess.preprocess_face()` → normalized crop metadata
- `FaceEmotionInference.predict()` → `dict[str, float]`
- `extract_speech_segments()` → `list[SpeechSegment]`
- `SpeechToTextService.transcribe()` → transcript text
- `TextToxicityAnalyzer.analyze()` → toxicity score + label
- `VoiceEmotionAnalyzer.analyze()` → voice emotion probabilities
- `fuse_signals()` → final decision + rule trace

## Resource targets

| Component | Target backend | MVP fallback |
|---|---|---|
| Face detection | MediaPipe / OpenCV | metadata-driven synthetic detector |
| Face emotion | PyTorch / ONNX Runtime | rule-based classifier |
| STT | Vosk / Silero | transcript passthrough |
| Voice emotion | SER model | prosody heuristics |
| UI | PySide6 / OpenCV | HTML pseudo-GUI |

## Latency budget (MVP target)

| Stage | Budget |
|---|---:|
| Frame capture | 10–20 ms |
| Face detect + preprocess | 25–40 ms |
| Face emotion inference | 15–30 ms |
| Audio chunk + STT | 60–80 ms |
| Voice/text inference | 20–30 ms |
| Fusion + UI | 5–10 ms |
| **Total** | **≤ 200 ms target** |

## Deployment model

- Primary: Windows laptop, local files + USB camera, CUDA inference.
- Secondary: CPU fallback on Linux/macOS.
- Optional: export to ONNX / CoreML for Mac M4.

## Privacy and ethics

- raw media should remain local by default;
- collected data should be versioned separately and removed after training when policy requires it;
- reports must describe bias, low-light failure modes, accent/noise sensitivity, and false conflict detection risk.
