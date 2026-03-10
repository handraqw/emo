from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_model.inference_face_emotion import FaceEmotionInference


class FaceEmotionInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FaceEmotionInference(
            checkpoint_path=str(
                Path(__file__).resolve().parents[1]
                / "experiments"
                / "example_face_baseline"
                / "checkpoint.json"
            )
        )

    def test_predict_can_surface_sarcasm_from_mixed_smile_and_tension(self) -> None:
        scores = self.model.predict(
            {
                "smile_score": 0.58,
                "brow_tension": 0.52,
                "mouth_open_score": 0.18,
                "eye_open_score": 0.25,
                "symmetry_score": 0.45,
                "warmth_score": 0.48,
            }
        )
        self.assertEqual(max(scores, key=scores.get), "SARCASM")

    def test_predict_can_surface_aggression_for_tense_face(self) -> None:
        scores = self.model.predict(
            {
                "smile_score": 0.15,
                "brow_tension": 0.85,
                "mouth_open_score": 0.55,
                "eye_open_score": 0.2,
                "symmetry_score": 0.72,
                "warmth_score": 0.4,
            }
        )
        self.assertEqual(max(scores, key=scores.get), "AGGRESSION")


if __name__ == "__main__":
    unittest.main()
