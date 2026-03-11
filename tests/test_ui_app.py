from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.schemas import FramePacket
from ui_app import _render_html_preview, main, run_pipeline


class UiAppTests(unittest.TestCase):
    def test_pipeline_exports_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "sample.mp4"
            video_path.write_bytes(b"stub-video")
            frames = iter(
                [
                    FramePacket(
                        index=0,
                        timestamp_ms=0,
                        metadata={
                            "hint_bbox": [10, 20, 100, 100],
                            "hint_face_emotion": "JOY",
                            "speech_text": "ты совсем тупой",
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
                    FramePacket(
                        index=2,
                        timestamp_ms=80,
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
                records = run_pipeline(source="file", path=str(video_path), results_dir=tmp_dir)
            self.assertTrue(records)
            self.assertTrue((Path(tmp_dir) / "annotations.csv").exists())
            self.assertTrue((Path(tmp_dir) / "ui_preview.html").exists())
            self.assertTrue((Path(tmp_dir) / "summary.txt").exists())
            self.assertFalse((Path(tmp_dir) / "annotations.json").exists())
            self.assertEqual(records[0]["final_emotion"], "CONFLICT")
            self.assertEqual(records[0]["source"], str(video_path))
            self.assertNotIn("speech_text", records[0])
            self.assertGreaterEqual(records[0]["audio_level"], 0.0)
            self.assertLessEqual(records[0]["audio_level"], 1.0)
            summary = (Path(tmp_dir) / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Frames processed: 3", summary)
            self.assertIn("Detections: 3", summary)
            self.assertIn("Source:", summary)
            annotations_csv = (Path(tmp_dir) / "annotations.csv").read_text(encoding="utf-8")
            self.assertNotIn("speech_text", annotations_csv)
            self.assertNotIn("ты совсем тупой", annotations_csv)
            preview_html = (Path(tmp_dir) / "ui_preview.html").read_text(encoding="utf-8")
            self.assertIn("audio-track", preview_html)
            self.assertIn("seek-slider", preview_html)
            self.assertNotIn("subtitle", preview_html)
            self.assertNotIn("ты совсем тупой", preview_html)
            self.assertNotIn("<th>Speech</th>", preview_html)

    def test_pipeline_handles_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "empty.mp4"
            video_path.write_bytes(b"stub-video")
            with patch("ui_app.iter_video_frames", return_value=iter(())):
                records = run_pipeline(source="file", path=str(video_path), results_dir=tmp_dir)
            self.assertEqual(records, [])
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

    def test_html_preview_embeds_video_controls_without_subtitles_for_uploaded_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preview_path = Path(tmp_dir) / "ui_preview.html"
            video_path = Path(tmp_dir) / "sample.mp4"
            video_path.write_bytes(b"stub-video")
            _render_html_preview(
                [
                    {
                        "timestamp_ms": 0,
                        "face_emotion": "JOY",
                        "face_confidence": 0.88,
                        "speech_text": "тестовые субтитры",
                        "audio_level": 0.41,
                        "final_emotion": "JOY",
                    }
                ],
                preview_path,
                source_ref=str(video_path),
            )
            html_payload = preview_path.read_text(encoding="utf-8")
            self.assertIn("<video id='source-video'", html_payload)
            self.assertIn("controls preload='metadata'", html_payload)
            self.assertIn("seek-slider", html_payload)
            self.assertIn("playback-rate", html_payload)
            self.assertIn("fullscreen-toggle", html_payload)
            self.assertNotIn("тестовые субтитры", html_payload)
            self.assertNotIn("<th>Speech</th>", html_payload)

    def test_pipeline_keeps_audio_without_detected_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "speaker.mp4"
            video_path.write_bytes(b"stub-video")
            frames = iter(
                [
                    FramePacket(
                        index=0,
                        timestamp_ms=0,
                        metadata={
                            "image": object(),
                            "speech_text": "ура, получилось",
                            "voice_features": {"pitch": 0.7, "energy": 0.65, "tempo": 0.55},
                        },
                    )
                ]
            )
            with patch("ui_app.iter_video_frames", return_value=frames):
                records = run_pipeline(source="file", path=str(video_path), results_dir=tmp_dir, max_frames=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["face_emotion"], "NO_FACE")
            self.assertGreater(records[0]["audio_level"], 0.0)
            self.assertEqual(records[0]["source"], str(video_path))

    def test_main_rejects_non_windows_runtime(self) -> None:
        for platform_name in ("linux", "darwin"):
            with self.subTest(platform=platform_name):
                with patch("ui_app.sys.platform", platform_name):
                    with self.assertLogs("ui_app", level="ERROR") as captured:
                        self.assertEqual(main(["--source", "camera"]), 1)
                        self.assertIn(
                            "Emotion AI MVP supports only Windows. Please run this application on Windows.",
                            "\n".join(captured.output),
                        )


if __name__ == "__main__":
    unittest.main()
