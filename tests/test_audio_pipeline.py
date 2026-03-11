from __future__ import annotations

from array import array
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_pipeline import build_live_speech_segment, extract_speech_segments
from utils.schemas import FramePacket, SpeechSegment


class AudioPipelineTests(unittest.TestCase):
    def test_build_live_speech_segment_populates_audio_metadata(self) -> None:
        samples = array("h", [0, 1200, -1200, 2400, -2400] * 4000)
        segment = build_live_speech_segment(samples, timestamp_ms=1500, sample_rate=16_000)
        self.assertEqual(segment.end_ms, 1500)
        self.assertIn("audio_bytes", segment.metadata)
        self.assertIn("voice_features", segment.metadata)
        self.assertGreater(segment.metadata["voice_features"]["energy"], 0.0)

    def test_extract_speech_segments_prefers_embedded_live_segment(self) -> None:
        embedded = SpeechSegment(
            start_ms=100,
            end_ms=800,
            text="тест",
            metadata={"voice_features": {"pitch": 0.4, "energy": 0.5, "tempo": 0.3}},
        )
        frame = FramePacket(index=0, timestamp_ms=120, metadata={"speech_segment": embedded})
        segments = extract_speech_segments(source="camera", frames=[frame])
        self.assertEqual(segments, [embedded])

    def test_extract_speech_segments_uses_audio_track_for_video_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "sample.mp4"
            video_path.write_bytes(b"stub-video")
            expected = [
                SpeechSegment(
                    start_ms=0,
                    end_ms=900,
                    text="",
                    metadata={"voice_features": {"pitch": 0.4, "energy": 0.6, "tempo": 0.3}},
                )
            ]
            with patch("audio_pipeline._extract_video_speech_segments", return_value=expected) as mocked:
                segments = extract_speech_segments(source="file", path=str(video_path), frames=[])
            self.assertEqual(segments, expected)
            mocked.assert_called_once_with(str(video_path))


if __name__ == "__main__":
    unittest.main()
