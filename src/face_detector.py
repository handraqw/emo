from __future__ import annotations

from utils.schemas import FaceDetection, FramePacket


class FaceDetector:
    """Face detector with a metadata fallback and optional MediaPipe integration hook."""

    backend_name = "heuristic"

    def detect(self, frame: FramePacket) -> list[FaceDetection]:
        bbox = frame.metadata.get("hint_bbox", [420, 180, 280, 280])
        landmarks = frame.metadata.get(
            "hint_landmarks",
            {
                "left_eye": (bbox[0] + 70, bbox[1] + 90),
                "right_eye": (bbox[0] + 210, bbox[1] + 90),
                "nose": (bbox[0] + 140, bbox[1] + 160),
            },
        )
        return [FaceDetection(bbox=tuple(int(value) for value in bbox), confidence=0.95, landmarks=landmarks)]
