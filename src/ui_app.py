from __future__ import annotations

import argparse
import csv
import html
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
from utils.schemas import SpeechSegment
from video_capture import iter_video_frames
from video_capture import VideoSourceError
from voice_emotion import VoiceEmotionAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _choose_segment(segments, timestamp_ms: int):
    if not segments:
        return SpeechSegment(start_ms=timestamp_ms, end_ms=timestamp_ms, text="", metadata={})
    for segment in segments:
        if segment.covers(timestamp_ms):
            return segment
    return segments[-1]


def _render_html_preview(records: list[dict], output_path: Path, source_ref: str | None) -> None:
    stats = _collect_run_stats(records, source_ref=source_ref)
    latest = records[-1] if records else {}
    rows = "\n".join(
        (
            "<tr>"
            f"<td>{item['timestamp_ms']}</td>"
            f"<td>{html.escape(str(item['face_emotion']))}</td>"
            f"<td>{html.escape(str(item['speech_text']))}</td>"
            f"<td>{html.escape(str(item['final_emotion']))}</td>"
            "</tr>"
        )
        for item in records
    )
    html_content = f"""<!DOCTYPE html>
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
    <h1>Emotion AI System — Analysis Preview</h1>
    <p>Source: {html.escape(stats['source'])}</p>
    <div class='video'>
      <div class='bbox'></div>
      <div class='badge'>Face emotion: {html.escape(str(latest.get('face_emotion', 'N/A')))} ({latest.get('face_confidence', 0.0)})</div>
    </div>
  </div>
  <div class='side'>
    <div class='panel mono'>{html.escape(_render_summary(records, source_ref=source_ref))}</div>
    <div class='panel'>
      <div class='final'>{html.escape(str(latest.get('final_emotion', 'N/A')))}</div>
      <p>Speech text: {html.escape(str(latest.get('speech_text', '')))}</p>
      <p>Toxicity: {latest.get('text_toxicity_score', 0.0)}</p>
      <p>Frames processed: {stats['frames_processed']}</p>
      <p>Detections: {stats['detections']}</p>
      <p>Unique faces: {stats['unique_faces']}</p>
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
    output_path.write_text(html_content, encoding="utf-8")


def _create_runtime() -> dict[str, object]:
    return {
        "detector": FaceDetector(),
        "tracker": FaceTracker(),
        "face_model": FaceEmotionInference(
            checkpoint_path="experiments/example_face_baseline/checkpoint.json"
        ),
        "stt": SpeechToTextService(),
        "text_analyzer": TextToxicityAnalyzer(),
        "voice_analyzer": VoiceEmotionAnalyzer(),
    }


def _analyze_frame(frame, segment: SpeechSegment, runtime: dict[str, object]) -> list[dict]:
    detector: FaceDetector = runtime["detector"]  # type: ignore[assignment]
    tracker: FaceTracker = runtime["tracker"]  # type: ignore[assignment]
    face_model: FaceEmotionInference = runtime["face_model"]  # type: ignore[assignment]
    stt: SpeechToTextService = runtime["stt"]  # type: ignore[assignment]
    text_analyzer: TextToxicityAnalyzer = runtime["text_analyzer"]  # type: ignore[assignment]
    voice_analyzer: VoiceEmotionAnalyzer = runtime["voice_analyzer"]  # type: ignore[assignment]

    detections = tracker.assign_ids(detector.detect(frame))
    transcript = stt.transcribe(segment)
    toxicity = text_analyzer.analyze(transcript)
    voice_probs = voice_analyzer.analyze(segment)

    records: list[dict] = []
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
    return records


def _render_summary(records: list[dict], source_ref: str | None = None) -> str:
    stats = _collect_run_stats(records, source_ref=source_ref)
    if records:
        latest = records[-1]
        return (
            f"Source: {stats['source']}\n"
            f"Frames processed: {stats['frames_processed']}\n"
            f"Detections: {stats['detections']}\n"
            f"Unique faces: {stats['unique_faces']}\n\n"
            f"Face emotion: {latest['face_emotion']} ({latest['face_confidence']})\n"
            f"Speech sentiment: {latest['voice_emotion']} ({latest['voice_confidence']})\n"
            f"Final emotion: {latest['final_emotion']}\n"
        )
    return (
        f"Source: {stats['source']}\n"
        "Frames processed: 0\n"
        "Detections: 0\n"
        "Unique faces: 0\n\n"
        "No frames or detections were available for analysis.\n"
    )


def _collect_run_stats(records: list[dict], source_ref: str | None = None) -> dict[str, object]:
    unique_timestamps = {int(record["timestamp_ms"]) for record in records}
    unique_faces = {
        int(record["face_id"])
        for record in records
        if record.get("face_id") is not None
    }
    source = str(records[-1]["source"]) if records else str(source_ref or "unknown")
    return {
        "source": source,
        "frames_processed": len(unique_timestamps),
        "detections": len(records),
        "unique_faces": len(unique_faces),
    }


def _write_results(records: list[dict], results_path: Path, video_ref: str | None) -> None:
    annotations_payload = {"video": video_ref, "frames": records}
    (results_path / "annotations.json").write_text(
        json.dumps(annotations_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = results_path / "annotations.csv"
    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8")

    _render_html_preview(records, results_path / "ui_preview.html", source_ref=video_ref)
    summary = _render_summary(records, source_ref=video_ref)
    (results_path / "summary.txt").write_text(summary, encoding="utf-8")
    LOGGER.info("Results written to %s", results_path)
    print(summary)


def analyze_source(
    source: str,
    path: str | None = None,
    max_frames: int | None = None,
    on_record=None,
) -> list[dict]:
    frames = list(iter_video_frames(source=source, path=path, max_frames=max_frames))
    speech_segments = extract_speech_segments(source=source, path=path, frames=frames)
    runtime = _create_runtime()

    records: list[dict] = []
    for frame in frames:
        segment = _choose_segment(speech_segments, frame.timestamp_ms)
        frame_records = _analyze_frame(frame, segment, runtime)
        for record in frame_records:
            record["source"] = path or source
        records.extend(frame_records)
        if on_record:
            for record in frame_records:
                on_record(frame, record)
    return records


def analyze_stream_frame(frame, source: str, path: str | None, runtime: dict[str, object]) -> list[dict]:
    segments = extract_speech_segments(source=source, path=path, frames=[frame])
    segment = _choose_segment(segments, frame.timestamp_ms)
    return _analyze_frame(frame, segment, runtime)


def run_pipeline(
    source: str,
    path: str | None,
    results_dir: str,
    max_frames: int | None = None,
) -> list[dict]:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    records = analyze_source(source=source, path=path, max_frames=max_frames)
    _write_results(records, results_path=results_path, video_ref=path or source)
    return records


def launch_gui(results_dir: str = "results") -> int:
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - depends on optional GUI runtime
        LOGGER.error("PySide6 is required for GUI mode: %s", exc)
        return 1

    class EmotionWindow(QMainWindow):  # pragma: no cover - exercised manually
        def __init__(self, output_dir: str) -> None:
            super().__init__()
            self.setWindowTitle("Emotion AI Live Analyzer")
            self.resize(1180, 760)

            self._output_dir = output_dir
            self._source = "file"
            self._path: str | None = None
            self._frame_iter = None
            self._runtime = None
            self._records: list[dict] = []
            self._timer = QTimer(self)
            self._timer.setInterval(40)
            self._timer.timeout.connect(self._process_next_frame)

            root = QWidget(self)
            self.setCentralWidget(root)
            layout = QHBoxLayout(root)

            left = QVBoxLayout()
            layout.addLayout(left, stretch=2)

            controls = QHBoxLayout()
            left.addLayout(controls)

            self.camera_button = QPushButton("Камера")
            self.camera_button.clicked.connect(self._start_camera)
            controls.addWidget(self.camera_button)

            self.video_button = QPushButton("Загрузить видео")
            self.video_button.clicked.connect(self._open_video)
            controls.addWidget(self.video_button)

            self.json_button = QPushButton("Открыть JSON")
            self.json_button.clicked.connect(self._open_json)
            controls.addWidget(self.json_button)

            self.stop_button = QPushButton("Стоп")
            self.stop_button.clicked.connect(self._stop_analysis)
            controls.addWidget(self.stop_button)

            self.preview = QLabel("Выберите камеру или видео для анализа")
            self.preview.setMinimumSize(720, 420)
            self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview.setStyleSheet(
                "background:#111827;color:#e2e8f0;border-radius:16px;padding:12px;font-size:18px;"
            )
            left.addWidget(self.preview)

            self.status = QLabel("Готово")
            left.addWidget(self.status)

            right = QVBoxLayout()
            layout.addLayout(right, stretch=1)

            self.final_label = QLabel("Итоговая эмоция: N/A")
            self.final_label.setStyleSheet("font-size:24px;font-weight:700;color:#fbbf24;")
            right.addWidget(self.final_label)

            self.metrics = QTextEdit()
            self.metrics.setReadOnly(True)
            self.metrics.setMinimumHeight(160)
            right.addWidget(self.metrics)

            self.timeline = QTableWidget(0, 4)
            self.timeline.setHorizontalHeaderLabels(["t, ms", "Лицо", "Речь", "Итог"])
            self.timeline.horizontalHeader().setStretchLastSection(True)
            right.addWidget(self.timeline)

        def closeEvent(self, event) -> None:
            self._stop_analysis()
            super().closeEvent(event)

        def _open_video(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите видео",
                "",
                "Media Files (*.mp4 *.avi *.mov *.mkv *.json);;All Files (*)",
            )
            if path:
                self._start_analysis(source="file", path=path)

        def _open_json(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите JSON сценарий",
                "",
                "JSON Files (*.json);;All Files (*)",
            )
            if path:
                self._start_analysis(source="file", path=path)

        def _start_camera(self) -> None:
            self._start_analysis(source="camera", path=None)

        def _start_analysis(self, source: str, path: str | None) -> None:
            self._stop_analysis(persist=False)
            self._source = source
            self._path = path
            self._runtime = _create_runtime()
            self._records = []
            self.timeline.setRowCount(0)
            self.metrics.clear()
            self.final_label.setText("Итоговая эмоция: анализ...")
            self.status.setText(f"Анализ: {path or source}")
            self._frame_iter = iter_video_frames(source=source, path=path)
            self._timer.start()

        def _stop_analysis(self, persist: bool = True) -> None:
            if self._timer.isActive():
                self._timer.stop()
            if persist and self._records:
                output_path = Path(self._output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                _write_results(self._records, output_path, self._path or self._source)
                self.status.setText(f"Результаты сохранены в {output_path}")

        def _process_next_frame(self) -> None:
            try:
                frame = next(self._frame_iter)
            except VideoSourceError as exc:
                self._stop_analysis(persist=False)
                self.status.setText(str(exc))
                QMessageBox.warning(self, "Источник недоступен", str(exc))
                self.preview.setText(str(exc))
                return
            except StopIteration:
                self._stop_analysis()
                return

            frame_records = analyze_stream_frame(
                frame,
                source=self._source,
                path=self._path,
                runtime=self._runtime,
            )
            if not frame_records:
                self.status.setText("Кадр обработан, лиц не найдено")
                self._update_preview(frame, None)
                return

            for record in frame_records:
                self._records.append(record)
                self._append_timeline(record)
                self._update_side_panel(record)
            self._update_preview(frame, frame_records[-1])

        def _append_timeline(self, record: dict) -> None:
            row_index = self.timeline.rowCount()
            self.timeline.insertRow(row_index)
            values = [
                str(record["timestamp_ms"]),
                str(record["face_emotion"]),
                str(record["speech_text"]),
                str(record["final_emotion"]),
            ]
            for column, value in enumerate(values):
                self.timeline.setItem(row_index, column, QTableWidgetItem(value))
            self.timeline.scrollToBottom()

        def _update_side_panel(self, record: dict) -> None:
            self.final_label.setText(f"Итоговая эмоция: {record['final_emotion']}")
            self.metrics.setPlainText(
                "\n".join(
                    [
                        f"Источник: {self._path or self._source}",
                        f"Face emotion: {record['face_emotion']} ({record['face_confidence']})",
                        f"Speech sentiment: {record['voice_emotion']} ({record['voice_confidence']})",
                        f"Toxicity: {record['text_toxicity_label']} ({record['text_toxicity_score']})",
                        f"Текст: {record['speech_text'] or '—'}",
                    ]
                )
            )
            self.status.setText(f"Последний кадр: {record['timestamp_ms']} ms")

        def _update_preview(self, frame, record: dict | None) -> None:
            image = frame.metadata.get("image")
            if image is not None:
                rgb = image[:, :, ::-1].copy()
                height, width, _ = rgb.shape
                qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
            else:
                pixmap = QPixmap(960, 540)
                pixmap.fill(QColor("#1d4ed8"))
                painter = QPainter(pixmap)
                painter.setPen(QColor("#f8fafc"))
                painter.setFont(QFont("Arial", 22))
                painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Emotion AI Preview")
                painter.end()

            if record:
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor("#fbbf24"), 4))
                x, y, w, h = record["bbox"]
                scale_x = pixmap.width() / max(frame.width, 1)
                scale_y = pixmap.height() / max(frame.height, 1)
                painter.drawRect(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
                painter.fillRect(20, 20, min(420, pixmap.width() - 40), 52, QColor(15, 23, 42, 220))
                painter.setPen(QColor("#e2e8f0"))
                painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
                painter.drawText(
                    32,
                    54,
                    f"{record['final_emotion']} · face {record['face_emotion']} · voice {record['voice_emotion']}",
                )
                painter.end()

            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    app = QApplication.instance() or QApplication([])
    window = EmotionWindow(results_dir)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emotion AI MVP UI")
    parser.add_argument("--source", choices=["file", "camera"], default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gui", action="store_true", help="Launch the Qt graphical interface")
    args = parser.parse_args(argv)

    if args.gui or args.source is None:
        return launch_gui(results_dir=args.results_dir)

    run_pipeline(
        source=args.source,
        path=args.path,
        results_dir=args.results_dir,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
