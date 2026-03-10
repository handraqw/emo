from __future__ import annotations

from utils.schemas import SpeechSegment


class SpeechToTextService:
    """Offline-friendly STT adapter with transcript passthrough fallback."""

    backend_name = "passthrough"

    def transcribe(self, segment: SpeechSegment) -> str:
        return segment.text.strip()
