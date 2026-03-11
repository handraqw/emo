from __future__ import annotations

import json
import os
from pathlib import Path

from utils.schemas import SpeechSegment

try:
    from vosk import KaldiRecognizer, Model
except Exception:  # pragma: no cover - optional runtime dependency
    KaldiRecognizer = None
    Model = None


class SpeechToTextService:
    """Offline-friendly STT adapter with passthrough and optional Vosk transcription."""

    backend_name = "vosk"

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = self._resolve_model_path(model_path)
        self._model = None

    def _resolve_model_path(self, model_path: str | None) -> str | None:
        candidates = [
            model_path,
            os.getenv("EMO_VOSK_MODEL"),
            str(Path("models") / "vosk-model-small-ru-0.22"),
            str(Path("data") / "vosk-model-small-ru-0.22"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(Path(candidate))
        return None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if Model is None or not self._model_path:
            return None
        try:
            self._model = Model(self._model_path)
        except Exception:
            self._model = None
        return self._model

    def _transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int) -> str:
        model = self._load_model()
        if model is None or KaldiRecognizer is None or not audio_bytes:
            return ""
        recognizer = KaldiRecognizer(model, float(sample_rate))
        recognizer.SetWords(True)
        for index in range(0, len(audio_bytes), 4000):
            recognizer.AcceptWaveform(audio_bytes[index : index + 4000])
        try:
            payload = json.loads(recognizer.FinalResult())
        except Exception:
            return ""
        return str(payload.get("text", "")).strip()

    def transcribe(self, segment: SpeechSegment) -> str:
        cached = segment.metadata.get("transcript")
        if isinstance(cached, str):
            return cached.strip()

        if segment.text.strip():
            transcript = segment.text.strip()
            segment.metadata["transcript"] = transcript
            return transcript

        audio_bytes = segment.metadata.get("audio_bytes")
        sample_rate = int(segment.metadata.get("sample_rate", 16_000) or 16_000)
        if isinstance(audio_bytes, bytes):
            transcript = self._transcribe_audio_bytes(audio_bytes, sample_rate)
            segment.metadata["transcript"] = transcript
            return transcript

        return ""
