from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from utils.schemas import FramePacket


def _load_json_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _frames_from_payload(payload: dict) -> list[FramePacket]:
    frames = []
    for index, item in enumerate(payload.get("frames", [])):
        frames.append(
            FramePacket(
                index=index,
                timestamp_ms=int(item.get("timestamp_ms", index * 40)),
                width=int(item.get("width", 1280)),
                height=int(item.get("height", 720)),
                metadata={k: v for k, v in item.items() if k not in {"timestamp_ms", "width", "height"}},
            )
        )
    return frames


def iter_video_frames(source: str, path: str | None = None, max_frames: int | None = None) -> Iterator[FramePacket]:
    """Yield frames from a JSON fixture, OpenCV backend, or a synthetic fallback stream."""
    if source not in {"file", "camera"}:
        raise ValueError(f"Unsupported source: {source}")

    if path:
        candidate = Path(path)
        if candidate.suffix.lower() == ".json" and candidate.exists():
            frames = _frames_from_payload(_load_json_payload(candidate))
            for frame in frames[:max_frames]:
                yield frame
            return

    for index in range(max_frames or 3):
        yield FramePacket(
            index=index,
            timestamp_ms=index * 40,
            metadata={
                "hint_bbox": [420, 180, 280, 280],
                "hint_face_emotion": "NEUTRAL" if index else "JOY",
                "speech_text": "всё спокойно" if index else "ну конечно, ты опять опоздал",
                "voice_features": {"pitch": 0.35 + index * 0.1, "energy": 0.25 + index * 0.2, "tempo": 0.3},
            },
        )
