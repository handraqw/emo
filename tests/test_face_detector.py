from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_detector import FaceDetector
from utils.schemas import FramePacket


class FaceDetectorTests(unittest.TestCase):
    def test_real_frame_without_detected_face_does_not_get_fallback_bbox(self) -> None:
        detector = FaceDetector()
        frame = FramePacket(index=0, timestamp_ms=0, metadata={"image": mock.Mock(name="real_image_frame")})
        detector._cascade = mock.Mock()
        detector._cascade.empty.return_value = False
        detector._cascade.detectMultiScale.return_value = []
        cv2_stub = mock.Mock()
        cv2_stub.COLOR_BGR2GRAY = object()
        cv2_stub.cvtColor.return_value = mock.Mock(name="gray_frame")
        with patch("face_detector.cv2", cv2_stub):
            detections = detector.detect(frame)
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
