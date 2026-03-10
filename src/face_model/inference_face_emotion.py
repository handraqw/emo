from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.schemas import EMOTION_LABELS


class FaceEmotionInference:
    """Rule-based fallback classifier compatible with future model replacement."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.class_priors = {label: 1 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
        if checkpoint_path and Path(checkpoint_path).exists():
            payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
            self.class_priors.update(payload.get("class_priors", {}))

    def predict(self, crop_metadata: dict) -> dict[str, float]:
        scores = {label: 0.01 for label in EMOTION_LABELS}
        hint = crop_metadata.get("hint_face_emotion")
        if hint in scores:
            scores[hint] = 0.85
        else:
            smile = float(crop_metadata.get("smile_score", 0.0))
            brow = float(crop_metadata.get("brow_tension", 0.0))
            if smile >= 0.6:
                scores["JOY"] = 0.8
            elif brow >= 0.75:
                scores["AGGRESSION"] = 0.72
            elif brow >= 0.45:
                scores["ANGER"] = 0.68
            else:
                scores["NEUTRAL"] = 0.7
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
