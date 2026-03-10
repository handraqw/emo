from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_toxicity import TextToxicityAnalyzer


class TextToxicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = TextToxicityAnalyzer()

    def test_detects_aggressive_russian_text(self) -> None:
        result = self.analyzer.analyze("Ты идиот и заткнись")
        self.assertEqual(result["label"], "AGGRESSION")
        self.assertGreaterEqual(result["score"], 0.9)

    def test_marks_sarcasm_when_marker_present(self) -> None:
        result = self.analyzer.analyze("Ну конечно, очень смешно")
        self.assertEqual(result["label"], "SARCASM")
        self.assertGreaterEqual(result["score"], 0.35)


if __name__ == "__main__":
    unittest.main()
