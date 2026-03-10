from __future__ import annotations

import json
from pathlib import Path

from utils.schemas import FramePacket, SpeechSegment


def extract_speech_segments(source: str, path: str | None = None, frames: list[FramePacket] | None = None) -> list[SpeechSegment]:
    """Extract audio metadata from JSON fixtures or derive it from frame metadata."""
    if path and Path(path).suffix.lower() == ".json" and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        segments = []
        for item in payload.get("speech_segments", []):
            segments.append(
                SpeechSegment(
                    start_ms=int(item.get("start_ms", 0)),
                    end_ms=int(item.get("end_ms", item.get("start_ms", 0) + 1000)),
                    text=item.get("text", ""),
                    metadata={k: v for k, v in item.items() if k not in {"start_ms", "end_ms", "text"}},
                )
            )
        if segments:
            return segments

    if not frames:
        return [SpeechSegment(start_ms=0, end_ms=1200, text="всё спокойно", metadata={"voice_features": {"pitch": 0.3, "energy": 0.2, "tempo": 0.3}})]

    segments = []
    for frame in frames:
        text = frame.metadata.get("speech_text", "")
        if not text:
            continue
        segments.append(
            SpeechSegment(
                start_ms=frame.timestamp_ms,
                end_ms=frame.timestamp_ms + 1000,
                text=text,
                metadata={
                    "voice_features": frame.metadata.get("voice_features", {"pitch": 0.3, "energy": 0.2, "tempo": 0.3}),
                },
            )
        )
    return segments or [SpeechSegment(start_ms=0, end_ms=1200, text="", metadata={})]
