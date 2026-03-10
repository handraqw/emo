from __future__ import annotations

from utils.schemas import EMOTION_LABELS, SpeechSegment


class VoiceEmotionAnalyzer:
    """Prosody heuristic baseline for Russian speech emotion analysis."""

    POSITIVE_MARKERS = ("ура", "отлично", "супер", "класс")

    def analyze(self, segment: SpeechSegment) -> dict[str, float]:
        scores = {label: 0.01 for label in EMOTION_LABELS}
        text = segment.text.lower()
        features = segment.metadata.get("voice_features", {})
        energy = float(features.get("energy", 0.2))
        pitch = float(features.get("pitch", 0.2))
        tempo = float(features.get("tempo", 0.2))

        if energy >= 0.75 or pitch >= 0.75 or "!" in text:
            scores["AGGRESSION"] = 0.8
        elif any(marker in text for marker in self.POSITIVE_MARKERS):
            scores["JOY"] = 0.74
        elif any(marker in text for marker in ("ну конечно", "ага", "очень смешно")):
            scores["SARCASM"] = 0.72
        elif energy >= 0.62 or tempo >= 0.62:
            scores["ANGER"] = 0.7
        elif energy >= 0.5 or tempo >= 0.45:
            scores["IRRITATION"] = 0.68
        else:
            scores["NEUTRAL"] = 0.76
        total = sum(scores.values())
        return {label: round(value / total, 4) for label, value in scores.items()}
