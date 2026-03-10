from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.schemas import FramePacket
from ui_app import run_pipeline


class UiAppTests(unittest.TestCase):
    def test_pipeline_exports_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            records = run_pipeline(
                source="file",
                path=str(Path(__file__).resolve().parents[1] / "examples" / "demo_input.json"),
                results_dir=tmp_dir,
            )
            self.assertTrue(records)
            payload = json.loads((Path(tmp_dir) / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["frames"][0]["final_emotion"], "CONFLICT")
            self.assertEqual(payload["frames"][0]["source"], str(Path(__file__).resolve().parents[1] / "examples" / "demo_input.json"))
            self.assertTrue((Path(tmp_dir) / "annotations.csv").exists())
            self.assertTrue((Path(tmp_dir) / "ui_preview.html").exists())
            self.assertTrue((Path(tmp_dir) / "summary.txt").exists())
            summary = (Path(tmp_dir) / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Frames processed: 3", summary)
            self.assertIn("Detections: 3", summary)
            self.assertIn("Source:", summary)

    def test_pipeline_handles_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = Path(tmp_dir) / "empty.json"
            fixture.write_text(json.dumps({"video": "empty.mp4", "frames": [], "speech_segments": []}), encoding="utf-8")
            records = run_pipeline(source="file", path=str(fixture), results_dir=tmp_dir)
            self.assertEqual(records, [])
            payload = json.loads((Path(tmp_dir) / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["frames"], [])
            self.assertIn(
                "No frames or detections",
                (Path(tmp_dir) / "summary.txt").read_text(encoding="utf-8"),
            )

    def test_camera_pipeline_processes_real_stream_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = iter(
                [
                    FramePacket(
                        index=0,
                        timestamp_ms=0,
                        metadata={
                            "hint_bbox": [10, 20, 100, 100],
                            "hint_face_emotion": "JOY",
                            "speech_text": "ну конечно",
                            "voice_features": {"pitch": 0.3, "energy": 0.2, "tempo": 0.3},
                        },
                    ),
                    FramePacket(
                        index=1,
                        timestamp_ms=40,
                        metadata={
                            "hint_bbox": [10, 20, 100, 100],
                            "hint_face_emotion": "NEUTRAL",
                            "speech_text": "всё спокойно",
                            "voice_features": {"pitch": 0.3, "energy": 0.2, "tempo": 0.3},
                        },
                    ),
                ]
            )
            with patch("ui_app.iter_video_frames", return_value=frames):
                records = run_pipeline(source="camera", path=None, results_dir=tmp_dir, max_frames=2)
            self.assertEqual(len(records), 2)
            summary = (Path(tmp_dir) / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Final emotion", summary)
            self.assertIn("Source: camera", summary)
            self.assertTrue((Path(tmp_dir) / "ui_preview.html").exists())


if __name__ == "__main__":
    unittest.main()
