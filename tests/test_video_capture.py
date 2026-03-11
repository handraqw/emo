from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.schemas import FramePacket
from video_capture import VideoSourceError, iter_video_frames


class VideoCaptureTests(unittest.TestCase):
    def test_file_video_path_uses_opencv_iterator_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "sample.mp4"
            video_path.write_bytes(b"stub")
            frame = FramePacket(index=0, timestamp_ms=0, metadata={"hint_bbox": [1, 2, 3, 4]})
            with patch("video_capture._iter_cv2_frames", return_value=iter([frame])) as mocked:
                frames = list(iter_video_frames(source="file", path=str(video_path), max_frames=1))
            self.assertEqual(frames, [frame])
            mocked.assert_called_once_with(path=str(video_path), max_frames=1)

    def test_unreadable_video_file_raises_error_instead_of_fallback_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "broken.mp4"
            video_path.write_bytes(b"not-a-real-video")
            with patch("video_capture._iter_cv2_frames", return_value=iter(())):
                with self.assertRaises(VideoSourceError):
                    list(iter_video_frames(source="file", path=str(video_path), max_frames=1))

    def test_json_file_is_rejected_as_video_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "scenario.json"
            json_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(VideoSourceError):
                list(iter_video_frames(source="file", path=str(json_path), max_frames=1))


if __name__ == "__main__":
    unittest.main()
