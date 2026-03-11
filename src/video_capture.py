from __future__ import annotations

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

# Hand-tuned weights for deriving stable facial cues from a coarse face crop.
# The lower-half lift emphasizes smiles, while contrast and warmth add light support
# so the detector remains usable on low-detail webcam frames.
SMILE_BASELINE = 0.45
SMILE_LOWER_LIFT_WEIGHT = 2.4
SMILE_CONTRAST_WEIGHT = 0.45
SMILE_WARMTH_WEIGHT = 0.2
BROW_CONTRAST_WEIGHT = 2.1
BROW_LOWER_LIFT_WEIGHT = 0.4


class VideoSourceError(RuntimeError):
    """Raised when a requested camera or video source cannot be read."""


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
    right = gray[:, max(width - left.shape[1], 0) :]
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
        "smile_score": round(
            min(
                max(
                    SMILE_BASELINE
                    + lower_lift * SMILE_LOWER_LIFT_WEIGHT
                    + contrast * SMILE_CONTRAST_WEIGHT
                    + (warmth - 0.5) * SMILE_WARMTH_WEIGHT,
                    0.0,
                ),
                1.0,
            ),
            3,
        ),
        "brow_tension": round(
            min(max(contrast * BROW_CONTRAST_WEIGHT + abs(lower_lift) * BROW_LOWER_LIFT_WEIGHT, 0.0), 1.0),
            3,
        ),
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
def iter_video_frames(source: str, path: str | None = None, max_frames: int | None = None) -> Iterator[FramePacket]:
    """Yield frames from real OpenCV-backed camera or uploaded video streams."""
    if source not in {"file", "camera"}:
        raise ValueError(f"Unsupported source: {source}")

    if source == "file" and not path:
        raise ValueError("A path is required when source='file'.")

    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input path does not exist: {candidate}")
        if candidate.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise VideoSourceError(
                "Поддерживаются только видеофайлы .mp4, .avi, .mov, .mkv и .webm."
            )
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
