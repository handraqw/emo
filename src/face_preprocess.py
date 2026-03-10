from __future__ import annotations

from utils.schemas import FaceDetection, FramePacket


def preprocess_face(frame: FramePacket, detection: FaceDetection, target_size: tuple[int, int] = (224, 224)) -> dict:
    x, y, w, h = detection.bbox
    x = max(0, min(x, frame.width - 1))
    y = max(0, min(y, frame.height - 1))
    w = max(1, min(w, frame.width - x))
    h = max(1, min(h, frame.height - y))
    return {
        "frame_index": frame.index,
        "timestamp_ms": frame.timestamp_ms,
        "bbox": [x, y, w, h],
        "target_size": list(target_size),
        "smile_score": float(frame.metadata.get("smile_score", 0.8 if frame.metadata.get("hint_face_emotion") == "JOY" else 0.2)),
        "brow_tension": float(frame.metadata.get("brow_tension", 0.7 if frame.metadata.get("hint_face_emotion") in {"ANGER", "AGGRESSION"} else 0.2)),
        "mouth_open_score": float(frame.metadata.get("mouth_open_score", 0.55 if frame.metadata.get("hint_face_emotion") in {"JOY", "AGGRESSION"} else 0.2)),
        "eye_open_score": float(frame.metadata.get("eye_open_score", 0.45 if frame.metadata.get("hint_face_emotion") == "IRRITATION" else 0.25)),
        "symmetry_score": float(frame.metadata.get("symmetry_score", 0.85)),
        "warmth_score": float(frame.metadata.get("warmth_score", 0.5)),
        "hint_face_emotion": frame.metadata.get("hint_face_emotion"),
    }
