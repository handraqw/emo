from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EMOTION_LABELS = [
    "JOY",
    "ANGER",
    "AGGRESSION",
    "IRRITATION",
    "NEUTRAL",
    "SARCASM",
    "RUDE",
]


@dataclass(slots=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float = 0.9
    landmarks: dict[str, tuple[int, int]] = field(default_factory=dict)
    face_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "landmarks": self.landmarks,
            "face_id": self.face_id,
        }


@dataclass(slots=True)
class FramePacket:
    index: int
    timestamp_ms: int
    width: int = 1280
    height: int = 720
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def covers(self, timestamp_ms: int) -> bool:
        return self.start_ms <= timestamp_ms <= self.end_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
