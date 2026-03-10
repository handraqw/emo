from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from utils.schemas import EMOTION_LABELS


def train_baseline(dataset_path: str, output_path: str) -> dict:
    records = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    labels = [item["label"] for item in records if item.get("label") in EMOTION_LABELS]
    counts = Counter(labels)
    total = sum(counts.values()) or 1
    checkpoint = {
        "model_type": "baseline_rules",
        "classes": EMOTION_LABELS,
        "class_priors": {label: round(counts.get(label, 0) / total, 4) for label in EMOTION_LABELS},
        "samples": total,
    }
    Path(output_path).write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = train_baseline(args.dataset, args.output)
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
