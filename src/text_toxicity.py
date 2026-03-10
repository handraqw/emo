from __future__ import annotations

TOXIC_KEYWORDS = {
    "идиот": 0.91,
    "тупой": 0.84,
    "заткнись": 0.96,
    "дурак": 0.76,
    "дура": 0.76,
    "ненавижу": 0.83,
    "бесишь": 0.71,
    "грубость": 0.65,
    "придурок": 0.89,
    "мерзкий": 0.73,
    "отстань": 0.62,
    "достал": 0.68,
}
SARCASTIC_MARKERS = ("ну конечно", "ага", "очень смешно")


class TextToxicityAnalyzer:
    def analyze(self, text: str) -> dict[str, object]:
        lowered = text.lower()
        matches = {word: score for word, score in TOXIC_KEYWORDS.items() if word in lowered}
        toxicity_score = round(max(matches.values(), default=0.0), 2)
        if toxicity_score >= 0.8:
            label = "AGGRESSION"
        elif toxicity_score >= 0.5:
            label = "RUDE"
        elif any(marker in lowered for marker in SARCASTIC_MARKERS):
            label = "SARCASM"
            toxicity_score = max(toxicity_score, 0.35)
        else:
            label = "NEUTRAL"
        return {
            "label": label,
            "score": toxicity_score,
            "matches": sorted(matches),
        }
