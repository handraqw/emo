from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
            self.assertTrue((Path(tmp_dir) / "annotations.csv").exists())
            self.assertTrue((Path(tmp_dir) / "ui_preview.html").exists())
            self.assertTrue((Path(tmp_dir) / "summary.txt").exists())

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


if __name__ == "__main__":
    unittest.main()
