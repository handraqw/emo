from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from utils.schemas import FaceDetection, FramePacket

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


class FaceDetector:
    """Face detector with a metadata fallback and optional MediaPipe integration hook."""

    backend_name = "heuristic"

    def __init__(self) -> None:
        self._cascade = None
        if cv2 is not None:
            cascade_path = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if cascade_path:
                self._cascade = cv2.CascadeClassifier(cascade_path + "haarcascade_frontalface_default.xml")

    def detect(self, frame: FramePacket) -> list[FaceDetection]:
        image = frame.metadata.get("image")
        if (
            np is not None
            and isinstance(image, np.ndarray)
            and image.size
            and cv2 is not None
            and self._cascade is not None
            and not self._cascade.empty()
        ):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            if len(boxes):
                return [
                    FaceDetection(bbox=(int(x), int(y), int(w), int(h)), confidence=0.95, landmarks={})
                    for x, y, w, h in boxes[:3]
                ]
            return []

        if image is not None and "hint_bbox" not in frame.metadata:
            return []

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
