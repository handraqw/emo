from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from audio_pipeline import extract_speech_segments
from face_detector import FaceDetector
from face_model.inference_face_emotion import FaceEmotionInference
from face_preprocess import preprocess_face
from face_tracker import FaceTracker
from fusion_engine import FusionWeights, fuse_signals
from speech_to_text import SpeechToTextService
from text_toxicity import TextToxicityAnalyzer
from video_capture import iter_video_frames
from voice_emotion import VoiceEmotionAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _choose_segment(segments, timestamp_ms: int):
    for segment in segments:
        if segment.covers(timestamp_ms):
            return segment
    return segments[-1]


def _render_html_preview(records: list[dict], output_path: Path) -> None:
    latest = records[-1] if records else {}
    rows = "\n".join(
        f"<tr><td>{item['timestamp_ms']}</td><td>{item['face_emotion']}</td><td>{item['speech_text']}</td><td>{item['final_emotion']}</td></tr>"
        for item in records
    )
    html = f"""<!DOCTYPE html>
<html lang='ru'>
<head>
<meta charset='utf-8'>
<title>Emotion AI UI Preview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
.app {{ display: grid; grid-template-columns: 2fr 1fr; min-height: 100vh; }}
.preview {{ padding: 24px; }}
.video {{ height: 420px; border-radius: 16px; background: linear-gradient(135deg, #1d4ed8, #0f766e); position: relative; box-shadow: 0 10px 30px rgba(0,0,0,.35); }}
.bbox {{ position: absolute; left: 32%; top: 18%; width: 26%; height: 42%; border: 4px solid #f8fafc; border-radius: 20px; }}
.badge {{ position: absolute; left: 32%; top: 11%; background: #111827; padding: 10px 14px; border-radius: 999px; }}
.side {{ background: #111827; padding: 24px; }}
.panel {{ background: #1f2937; border-radius: 16px; padding: 16px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
td, th {{ border-bottom: 1px solid #334155; padding: 8px; text-align: left; vertical-align: top; }}
.final {{ font-size: 28px; font-weight: bold; color: #fbbf24; }}
.mono {{ font-family: Consolas, monospace; white-space: pre-line; }}
</style>
</head>
<body>
<div class='app'>
  <div class='preview'>
    <h1>Emotion AI System — MVP Preview</h1>
    <div class='video'>
      <div class='bbox'></div>
      <div class='badge'>Face emotion: {latest.get('face_emotion', 'N/A')} ({latest.get('face_confidence', 0.0)})</div>
    </div>
  </div>
  <div class='side'>
    <div class='panel mono'>Camera 1\n\nFace emotion: {latest.get('face_emotion', 'N/A')} ({latest.get('face_confidence', 0.0)})\nSpeech sentiment: {latest.get('voice_emotion', 'N/A')} ({latest.get('voice_confidence', 0.0)})\nFinal emotion: {latest.get('final_emotion', 'N/A')}</div>
    <div class='panel'>
      <div class='final'>{latest.get('final_emotion', 'N/A')}</div>
      <p>Speech text: {latest.get('speech_text', '')}</p>
      <p>Toxicity: {latest.get('text_toxicity_score', 0.0)}</p>
    </div>
    <div class='panel'>
      <h3>Timeline</h3>
      <table>
        <thead><tr><th>t, ms</th><th>Face</th><th>Speech</th><th>Final</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>
</div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def run_pipeline(source: str, path: str | None, results_dir: str) -> list[dict]:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    frames = list(iter_video_frames(source=source, path=path))
    speech_segments = extract_speech_segments(source=source, path=path, frames=frames)

    detector = FaceDetector()
    tracker = FaceTracker()
    face_model = FaceEmotionInference(checkpoint_path="experiments/example_face_baseline/checkpoint.json")
    stt = SpeechToTextService()
    text_analyzer = TextToxicityAnalyzer()
    voice_analyzer = VoiceEmotionAnalyzer()

    records: list[dict] = []
    for frame in frames:
        detections = tracker.assign_ids(detector.detect(frame))
        segment = _choose_segment(speech_segments, frame.timestamp_ms)
        transcript = stt.transcribe(segment)
        toxicity = text_analyzer.analyze(transcript)
        voice_probs = voice_analyzer.analyze(segment)

        for detection in detections:
            crop = preprocess_face(frame, detection)
            face_probs = face_model.predict(crop)
            face_emotion = max(face_probs, key=face_probs.get)
            decision = fuse_signals(face_probs, voice_probs, toxicity, weights=FusionWeights())
            voice_emotion = max(voice_probs, key=voice_probs.get)
            records.append(
                {
                    "timestamp_ms": frame.timestamp_ms,
                    "bbox": list(detection.bbox),
                    "face_id": detection.face_id,
                    "face_emotion": face_emotion,
                    "face_confidence": round(face_probs[face_emotion], 2),
                    "speech_text": transcript,
                    "text_toxicity_label": toxicity["label"],
                    "text_toxicity_score": toxicity["score"],
                    "voice_emotion": voice_emotion,
                    "voice_confidence": round(voice_probs[voice_emotion], 2),
                    "final_emotion": decision.final_emotion,
                    "triggered_rules": decision.triggered_rules,
                }
            )

    annotations_payload = {"video": path or source, "frames": records}
    (results_path / "annotations.json").write_text(
        json.dumps(annotations_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (results_path / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    _render_html_preview(records, results_path / "ui_preview.html")

    latest = records[-1]
    summary = (
        "Camera 1\n\n"
        f"Face emotion: {latest['face_emotion']} ({latest['face_confidence']})\n"
        f"Speech sentiment: {latest['voice_emotion']} ({latest['voice_confidence']})\n"
        f"Final emotion: {latest['final_emotion']}\n"
    )
    (results_path / "summary.txt").write_text(summary, encoding="utf-8")
    LOGGER.info("Results written to %s", results_path)
    print(summary)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emotion AI MVP UI")
    parser.add_argument("--source", choices=["file", "camera"], required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    run_pipeline(source=args.source, path=args.path, results_dir=args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
