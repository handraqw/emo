from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusion_engine import FusionWeights, fuse_signals


class FusionEngineTests(unittest.TestCase):
    def test_face_and_voice_drive_result_without_text_signal(self) -> None:
        decision = fuse_signals(
            face_probs={"JOY": 0.8, "NEUTRAL": 0.1},
            voice_probs={"NEUTRAL": 0.7},
        )
        self.assertEqual(decision.final_emotion, "NEUTRAL")
        self.assertEqual(decision.triggered_rules, [])

    def test_voice_aggression_can_override_neutral_face(self) -> None:
        decision = fuse_signals(
            face_probs={"NEUTRAL": 0.82},
            voice_probs={"AGGRESSION": 0.81},
        )
        self.assertEqual(decision.final_emotion, "AGGRESSION")
        self.assertIn("aggressive_voice_overrides_neutral_face", decision.triggered_rules)

    def test_weights_are_configurable(self) -> None:
        decision = fuse_signals(
            face_probs={"ANGER": 0.7},
            voice_probs={"IRRITATION": 0.7},
            weights=FusionWeights(face=0.1, voice=0.9),
        )
        self.assertEqual(decision.final_emotion, "IRRITATION")


if __name__ == "__main__":
    unittest.main()
