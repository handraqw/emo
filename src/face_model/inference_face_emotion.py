from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.schemas import EMOTION_LABELS

# Blend a small amount of prior knowledge into the fallback rules so unsupported
# classes still retain non-zero probability before the heuristics are applied.
PRIOR_SCALE = 0.12
PRIOR_FLOOR = 0.01


class FaceEmotionInference:
    """Rule-based fallback classifier compatible with future model replacement."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.class_priors = {label: 1 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
        if checkpoint_path and Path(checkpoint_path).exists():
            payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
            self.class_priors.update(payload.get("class_priors", {}))

    def predict(self, crop_metadata: dict) -> dict[str, float]:
        scores = {
            label: round(max(float(self.class_priors.get(label, 0.0)), PRIOR_FLOOR) * PRIOR_SCALE + PRIOR_FLOOR, 4)
            for label in EMOTION_LABELS
        }
        hint = crop_metadata.get("hint_face_emotion")
        if hint in scores:
            scores[hint] += 1.1
        else:
            smile = float(crop_metadata.get("smile_score", 0.0))
            brow = float(crop_metadata.get("brow_tension", 0.0))
            mouth = float(crop_metadata.get("mouth_open_score", 0.0))
            eyes = float(crop_metadata.get("eye_open_score", 0.0))
            symmetry = float(crop_metadata.get("symmetry_score", 0.85))
            warmth = float(crop_metadata.get("warmth_score", 0.5))

            scores["JOY"] += max(0.0, smile - 0.45) * 1.9 + mouth * 0.2 + max(0.0, warmth - 0.5) * 0.4
            scores["SARCASM"] += max(0.0, smile - 0.3) * 0.9 + max(0.0, brow - 0.35) * 0.75 + max(0.0, 0.85 - symmetry) * 0.5
            scores["AGGRESSION"] += max(0.0, brow - 0.65) * 2.2 + max(0.0, mouth - 0.35) * 0.55
            scores["ANGER"] += max(0.0, brow - 0.45) * 1.45 + max(0.0, mouth - 0.25) * 0.25
            scores["IRRITATION"] += max(0.0, brow - 0.25) * 0.9 + max(0.0, 0.45 - smile) * 0.5 + max(0.0, 0.35 - eyes) * 0.3
            scores["RUDE"] += max(0.0, brow - 0.55) * 0.65 + max(0.0, 0.8 - symmetry) * 0.4
            if smile < 0.4 and brow < 0.4:
                scores["NEUTRAL"] += 0.7 + symmetry * 0.1
            elif smile >= 0.55 and brow < 0.45:
                scores["JOY"] += 0.25
            elif brow >= 0.75 and smile < 0.3:
                scores["AGGRESSION"] += 0.3
            else:
                scores["NEUTRAL"] += 0.35
        total = sum(scores.values())
        return {label: round(value / total, 4) for label, value in scores.items()}


def export_onnx_manifest(output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(
            {
                "status": "placeholder",
                "message": "Install PyTorch/ONNX and replace the rule-based classifier with a trained export.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hint-emotion", default="JOY")
    parser.add_argument("--export-onnx", dest="export_onnx", default=None)
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx_manifest(args.export_onnx)
        return 0

    model = FaceEmotionInference(checkpoint_path=args.checkpoint)
    prediction = model.predict({"hint_face_emotion": args.hint_emotion})
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
