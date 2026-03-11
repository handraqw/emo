from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speech_to_text import SpeechToTextService
from utils.schemas import SpeechSegment


class SpeechToTextTests(unittest.TestCase):
    def test_returns_existing_transcript_without_backend(self) -> None:
        service = SpeechToTextService()
        segment = SpeechSegment(start_ms=0, end_ms=1000, text="  привет  ", metadata={})
        self.assertEqual(service.transcribe(segment), "привет")

    def test_transcribes_audio_bytes_and_caches_result(self) -> None:
        service = SpeechToTextService()
        segment = SpeechSegment(
            start_ms=0,
            end_ms=1000,
            text="",
            metadata={"audio_bytes": b"\x00\x01" * 100, "sample_rate": 16_000},
        )
        with patch.object(service, "_transcribe_audio_bytes", return_value="распознано") as mocked:
            self.assertEqual(service.transcribe(segment), "распознано")
            self.assertEqual(service.transcribe(segment), "распознано")
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
