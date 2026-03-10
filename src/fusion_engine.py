from __future__ import annotations

from dataclasses import dataclass, field

from utils.schemas import EMOTION_LABELS


@dataclass(slots=True)
class FusionWeights:
    face: float = 0.4
    voice: float = 0.35
    text: float = 0.25


@dataclass(slots=True)
class FusionDecision:
    final_emotion: str
    combined_scores: dict[str, float]
    triggered_rules: list[str] = field(default_factory=list)


def fuse_signals(
    face_probs: dict[str, float],
    voice_probs: dict[str, float],
    toxicity: dict[str, object],
    weights: FusionWeights | None = None,
) -> FusionDecision:
    weights = weights or FusionWeights()
    combined = {label: 0.0 for label in EMOTION_LABELS}
    rules: list[str] = []

    for label, value in face_probs.items():
        if label in combined:
            combined[label] += value * weights.face
    for label, value in voice_probs.items():
        if label in combined:
            combined[label] += value * weights.voice

    text_label = str(toxicity.get("label", "NEUTRAL"))
    text_score = float(toxicity.get("score", 0.0))
    if text_label in combined:
        combined[text_label] += text_score * weights.text
    else:
        combined["NEUTRAL"] += max(0.0, 1 - text_score) * weights.text

    face_confident_label = max(face_probs, key=face_probs.get)
    face_confident_score = float(face_probs.get(face_confident_label, 0.0))
    voice_aggression = float(voice_probs.get("AGGRESSION", 0.0))
    neutral_face = float(face_probs.get("NEUTRAL", 0.0))

    if face_confident_label in {"JOY", "NEUTRAL", "SARCASM"} and face_confident_score >= 0.6 and text_score >= 0.6:
        rules.append("happy_or_neutral_face_with_toxic_text")
        return FusionDecision("CONFLICT", combined, rules)

    if voice_aggression >= 0.6 and neutral_face >= 0.6:
        rules.append("aggressive_voice_overrides_neutral_face")
        combined["AGGRESSION"] += 0.15

    final_emotion = max(combined, key=combined.get)
    return FusionDecision(final_emotion, {k: round(v, 4) for k, v in combined.items()}, rules)
