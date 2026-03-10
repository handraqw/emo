from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from utils.schemas import FramePacket

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy comes with opencv at runtime
    np = None


class VideoSourceError(RuntimeError):
    """Raised when a requested camera or video source cannot be read."""


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


def _build_visual_metadata(image) -> dict:
    height, width = image.shape[:2]
    center_w = max(width // 4, 1)
    center_h = max(height // 3, 1)
    bbox_x = max((width - center_w) // 2, 0)
    bbox_y = max((height - center_h) // 3, 0)

    gray = image.mean(axis=2) if image.ndim == 3 else image.astype("float32")
    upper = gray[: max(height // 2, 1), :]
    lower = gray[height // 2 :, :] if height > 1 else gray
    center = gray[:, width // 4 : max((width * 3) // 4, 1)]
    upper_center = center[: max(center.shape[0] // 2, 1), :]
    lower_center = center[center.shape[0] // 2 :, :] if center.shape[0] > 1 else center
    mean_intensity = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 255.0
    lower_lift = float(lower.mean() - upper.mean()) / 255.0 if upper.size and lower.size else 0.0
    mouth_open = float(lower_center.std()) / 64.0 if lower_center.size else 0.0
    eye_open = float(upper_center.std()) / 64.0 if upper_center.size else 0.0

    left = gray[:, : max(width // 2, 1)]
    right = gray[:, width - left.shape[1] :]
    symmetry = 0.5
    if np is not None and left.size and right.size:
        mirrored_right = np.fliplr(right[:, : left.shape[1]])
        symmetry = 1.0 - float(np.mean(np.abs(left - mirrored_right))) / 255.0

    warmth = 0.5
    if image.ndim == 3:
        warmth = 0.5 + float(image[:, :, 2].mean() - image[:, :, 0].mean()) / 255.0 / 2.0

    return {
        "image": image,
        "hint_bbox": [bbox_x, bbox_y, center_w, center_h],
        "smile_score": round(min(max(0.45 + lower_lift * 2.4 + contrast * 0.45 + (warmth - 0.5) * 0.2, 0.0), 1.0), 3),
        "brow_tension": round(min(max(contrast * 2.1 + abs(lower_lift) * 0.4, 0.0), 1.0), 3),
        "mouth_open_score": round(min(max(mouth_open, 0.0), 1.0), 3),
        "eye_open_score": round(min(max(eye_open, 0.0), 1.0), 3),
        "symmetry_score": round(min(max(symmetry, 0.0), 1.0), 3),
        "warmth_score": round(min(max(warmth, 0.0), 1.0), 3),
    }


def _iter_cv2_frames(path: str | None = None, max_frames: int | None = None) -> Iterator[FramePacket]:
    if cv2 is None:
        return

    capture_target = 0 if path is None else path
    capture = cv2.VideoCapture(capture_target)
    if not capture.isOpened():
        capture.release()
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    try:
        index = 0
        while max_frames is None or index < max_frames:
            ok, image = capture.read()
            if not ok or image is None:
                break
            metadata = _build_visual_metadata(image)
            yield FramePacket(
                index=index,
                timestamp_ms=int((index / fps) * 1000),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                metadata=metadata,
            )
            index += 1
    finally:
        capture.release()


def _iter_synthetic_frames(max_frames: int | None = None) -> Iterator[FramePacket]:
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


def iter_video_frames(source: str, path: str | None = None, max_frames: int | None = None) -> Iterator[FramePacket]:
    """Yield frames from JSON fixtures or real OpenCV-backed camera/video streams."""
    if source not in {"file", "camera"}:
        raise ValueError(f"Unsupported source: {source}")

    if source == "file" and not path:
        raise ValueError("A path is required when source='file'.")

    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input path does not exist: {candidate}")
        if candidate.suffix.lower() == ".json":
            frames = _frames_from_payload(_load_json_payload(candidate))
            for frame in frames[:max_frames]:
                yield frame
            return
        frame_iter = _iter_cv2_frames(path=str(candidate), max_frames=max_frames)
        first_frame = next(frame_iter, None)
        if first_frame is None:
            raise VideoSourceError(f"Unable to read video frames from: {candidate}")
        yield first_frame
        yield from frame_iter
        return

    frame_iter = _iter_cv2_frames(max_frames=max_frames)
    first_frame = next(frame_iter, None)
    if first_frame is None:
        raise VideoSourceError("Unable to open the camera or read frames from it.")
    yield first_frame
    yield from frame_iter
