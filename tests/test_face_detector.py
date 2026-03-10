from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_detector import FaceDetector
from utils.schemas import FramePacket


class FaceDetectorTests(unittest.TestCase):
    def test_real_frame_without_detected_face_does_not_get_demo_bbox(self) -> None:
        detector = FaceDetector()
        frame = FramePacket(index=0, timestamp_ms=0, metadata={"image": object()})
        with patch("face_detector.cv2", None):
            detections = detector.detect(frame)
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
