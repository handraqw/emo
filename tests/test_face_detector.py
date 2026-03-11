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

    def test_invalid_image_object_is_ignored_before_opencv_conversion(self) -> None:
        detector = FaceDetector()
        frame = FramePacket(index=0, timestamp_ms=0, metadata={"image": object()})
        detector._cascade = mock.Mock()
        detector._cascade.empty.return_value = False
        cv2_stub = mock.Mock()
        cv2_stub.COLOR_BGR2GRAY = object()
        with patch("face_detector.cv2", cv2_stub):
            detections = detector.detect(frame)
        self.assertEqual(detections, [])
        cv2_stub.cvtColor.assert_not_called()

    def test_numpy_frame_uses_opencv_detection_when_available(self) -> None:
        class FakeArray:
            def __init__(self) -> None:
                self.size = 64 * 64 * 3

        detector = FaceDetector()
        frame = FramePacket(index=0, timestamp_ms=0, metadata={"image": FakeArray()})
        detector._cascade = mock.Mock()
        detector._cascade.empty.return_value = False
        detector._cascade.detectMultiScale.return_value = [(4, 5, 20, 24)]
        cv2_stub = mock.Mock()
        cv2_stub.COLOR_BGR2GRAY = object()
        cv2_stub.cvtColor.return_value = mock.Mock(name="gray_frame")
        with patch("face_detector.cv2", cv2_stub), patch("face_detector.np", mock.Mock(ndarray=FakeArray)):
            detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (4, 5, 20, 24))
        cv2_stub.cvtColor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
