from __future__ import annotations

from utils.schemas import FaceDetection


class FaceTracker:
    """Single-stream tracker assigning deterministic IDs for the MVP."""

    def assign_ids(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        for index, detection in enumerate(detections, start=1):
            detection.face_id = index
        return detections
