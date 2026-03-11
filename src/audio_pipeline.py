from __future__ import annotations

import math
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path

from utils.schemas import FramePacket, SpeechSegment

try:
    import imageio_ffmpeg as _imageio_ffmpeg_module
except Exception:  # pragma: no cover - optional runtime dependency
    _imageio_ffmpeg_module = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_SAMPLE_RATE = 16_000
WINDOW_MS = 30
MIN_SEGMENT_MS = 350
MAX_SILENCE_MS = 220
MAX_LIVE_BUFFER_SECONDS = 3.0
LOGGER = logging.getLogger(__name__)


def _resolve_ffmpeg_binary() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if _imageio_ffmpeg_module is not None:
        try:
            return str(_imageio_ffmpeg_module.get_ffmpeg_exe())
        except Exception:
            return None
    return None


def _extract_audio_track(video_path: str) -> tuple[array, int]:
    ffmpeg_binary = _resolve_ffmpeg_binary()
    if not ffmpeg_binary:
        return array("h"), DEFAULT_SAMPLE_RATE
    candidate = Path(video_path)
    if not candidate.exists() or not candidate.is_file():
        return array("h"), DEFAULT_SAMPLE_RATE

    file_descriptor, temporary_name = tempfile.mkstemp(suffix=".wav")
    os.close(file_descriptor)
    wav_path = Path(temporary_name)

    try:
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(candidate),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-f",
            "wav",
            str(wav_path),
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            if completed.stderr:
                LOGGER.warning(
                    "ffmpeg audio extraction failed for %s: %s",
                    candidate,
                    completed.stderr.decode("utf-8", errors="ignore").strip(),
                )
            return array("h"), DEFAULT_SAMPLE_RATE
        return _read_wave_samples(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)


def _read_wave_samples(wav_path: Path) -> tuple[array, int]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = int(handle.getframerate() or DEFAULT_SAMPLE_RATE)
        sample_width = handle.getsampwidth()
        channel_count = max(int(handle.getnchannels()), 1)
        pcm_bytes = handle.readframes(handle.getnframes())

    if sample_width != 2:
        return array("h"), sample_rate

    samples = array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % sample_width)])
    if channel_count > 1 and samples:
        mono_samples = array("h")
        for index in range(0, len(samples), channel_count):
            mono_samples.append(int(sum(samples[index : index + channel_count]) / channel_count))
        samples = mono_samples
    return samples, sample_rate


def _normalise_samples(samples: array) -> list[float]:
    return [max(min(sample / 32768.0, 1.0), -1.0) for sample in samples]


def _estimate_voice_features(samples: array, sample_rate: int) -> dict[str, float]:
    if not samples or sample_rate <= 0:
        return {"pitch": 0.2, "energy": 0.0, "tempo": 0.2}

    normalized = _normalise_samples(samples)
    rms = math.sqrt(sum(value * value for value in normalized) / len(normalized))
    energy = min(max(rms * 3.2, 0.0), 1.0)

    zero_crossings = 0
    for previous, current in zip(normalized, normalized[1:]):
        if (previous <= 0 < current) or (previous >= 0 > current):
            zero_crossings += 1
    duration_seconds = max(len(normalized) / sample_rate, 1e-6)
    estimated_hz = zero_crossings / (2.0 * duration_seconds)
    pitch = min(max((estimated_hz - 80.0) / 220.0, 0.0), 1.0)

    window_size = max(int(sample_rate * 0.1), 1)
    window_energies = []
    for index in range(0, len(normalized), window_size):
        window = normalized[index : index + window_size]
        if not window:
            continue
        window_rms = math.sqrt(sum(value * value for value in window) / len(window))
        window_energies.append(window_rms)

    energy_floor = max(sum(window_energies) / len(window_energies) if window_energies else 0.0, 0.02)
    burst_count = sum(
        1
        for previous, current in zip(window_energies, window_energies[1:])
        if current >= energy_floor * 1.35 and current > previous
    )
    bursts_per_second = burst_count / duration_seconds
    tempo = min(max(bursts_per_second / 4.5, 0.0), 1.0)

    return {
        "pitch": round(pitch, 3),
        "energy": round(energy, 3),
        "tempo": round(tempo, 3),
    }


def _segment_audio(samples: array, sample_rate: int) -> list[SpeechSegment]:
    if not samples or sample_rate <= 0:
        return []

    window_size = max(int(sample_rate * (WINDOW_MS / 1000.0)), 1)
    min_segment_windows = max(int(MIN_SEGMENT_MS / WINDOW_MS), 1)
    max_silence_windows = max(int(MAX_SILENCE_MS / WINDOW_MS), 1)

    window_energies = []
    for index in range(0, len(samples), window_size):
        window = samples[index : index + window_size]
        if not window:
            continue
        normalized = _normalise_samples(window)
        rms = math.sqrt(sum(value * value for value in normalized) / len(normalized))
        window_energies.append(rms)

    if not window_energies:
        return []

    average_energy = sum(window_energies) / len(window_energies)
    threshold = max(average_energy * 1.4, 0.03)

    segments: list[tuple[int, int]] = []
    start_window: int | None = None
    silence_windows = 0
    for index, energy in enumerate(window_energies):
        if energy >= threshold:
            if start_window is None:
                start_window = index
            silence_windows = 0
            continue
        if start_window is None:
            continue
        silence_windows += 1
        if silence_windows >= max_silence_windows:
            end_window = max(index - silence_windows + 1, start_window + 1)
            if end_window - start_window >= min_segment_windows:
                segments.append((start_window, end_window))
            start_window = None
            silence_windows = 0

    if start_window is not None:
        end_window = len(window_energies)
        if end_window - start_window >= min_segment_windows:
            segments.append((start_window, end_window))

    result = []
    for start_window, end_window in segments:
        start_sample = start_window * window_size
        end_sample = min(end_window * window_size, len(samples))
        segment_samples = array("h", samples[start_sample:end_sample])
        if not segment_samples:
            continue
        result.append(
            SpeechSegment(
                start_ms=int((start_sample / sample_rate) * 1000),
                end_ms=int((end_sample / sample_rate) * 1000),
                text="",
                metadata={
                    "audio_bytes": segment_samples.tobytes(),
                    "sample_rate": sample_rate,
                    "voice_features": _estimate_voice_features(segment_samples, sample_rate),
                },
            )
        )
    return result


def _extract_video_speech_segments(path: str) -> list[SpeechSegment]:
    samples, sample_rate = _extract_audio_track(path)
    return _segment_audio(samples, sample_rate)


def build_live_speech_segment(samples: array, timestamp_ms: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> SpeechSegment:
    clipped_samples = array("h", samples[-int(sample_rate * MAX_LIVE_BUFFER_SECONDS) :])
    duration_ms = int((len(clipped_samples) / max(sample_rate, 1)) * 1000)
    return SpeechSegment(
        start_ms=max(timestamp_ms - duration_ms, 0),
        end_ms=timestamp_ms,
        text="",
        metadata={
            "audio_bytes": clipped_samples.tobytes(),
            "sample_rate": sample_rate,
            "voice_features": _estimate_voice_features(clipped_samples, sample_rate),
        },
    )


def extract_speech_segments(source: str, path: str | None = None, frames: list[FramePacket] | None = None) -> list[SpeechSegment]:
    """Extract audio-driven speech segments for uploaded videos or live microphone frames."""
    if source == "file" and path and Path(path).suffix.lower() in VIDEO_EXTENSIONS and Path(path).exists():
        segments = _extract_video_speech_segments(path)
        if segments:
            return segments

    if not frames:
        return [SpeechSegment(start_ms=0, end_ms=1200, text="", metadata={})]

    segments = []
    for frame in frames:
        embedded_segment = frame.metadata.get("speech_segment")
        if isinstance(embedded_segment, SpeechSegment):
            segments.append(embedded_segment)
            continue
        has_audio_context = any(
            key in frame.metadata for key in ("voice_features", "audio_bytes", "sample_rate")
        )
        if not has_audio_context:
            continue
        metadata = {}
        if "voice_features" in frame.metadata:
            metadata["voice_features"] = frame.metadata["voice_features"]
        if "audio_bytes" in frame.metadata:
            metadata["audio_bytes"] = frame.metadata["audio_bytes"]
        if "sample_rate" in frame.metadata:
            metadata["sample_rate"] = frame.metadata["sample_rate"]
        segments.append(
            SpeechSegment(
                start_ms=frame.timestamp_ms,
                end_ms=frame.timestamp_ms + 1000,
                text="",
                metadata=metadata or {"voice_features": {"pitch": 0.3, "energy": 0.2, "tempo": 0.3}},
            )
        )
    return segments or [SpeechSegment(start_ms=0, end_ms=1200, text="", metadata={})]
