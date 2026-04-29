from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioFeatureSummary:
    available: bool
    sample_rate: float = 0.0
    duration_s: float = 0.0
    channels: float = 0.0
    frame_count: float = 0.0
    backend: str = ""
    message: str = ""

    def to_features(self) -> dict[str, float | str | bool]:
        return asdict(self)


def extract_audio_features(path: str | Path) -> AudioFeatureSummary:
    """Extract small audio metadata without requiring audio downloads.

    The function tries `soundfile`, then `torchaudio`, then stdlib WAV. It is
    intended for local audio files supplied by the user.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        return AudioFeatureSummary(False, backend="none", message="audio file missing")
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio_path))
        frames = float(getattr(info, "frames", 0) or 0)
        sample_rate = float(getattr(info, "samplerate", 0) or 0)
        channels = float(getattr(info, "channels", 0) or 0)
        return AudioFeatureSummary(
            True,
            sample_rate=sample_rate,
            duration_s=frames / sample_rate if sample_rate > 0 else 0.0,
            channels=channels,
            frame_count=frames,
            backend="soundfile",
            message="ok",
        )
    except Exception:
        pass
    try:
        import torchaudio  # type: ignore

        info = torchaudio.info(str(audio_path))
        sample_rate = float(info.sample_rate)
        frames = float(info.num_frames)
        return AudioFeatureSummary(
            True,
            sample_rate=sample_rate,
            duration_s=frames / sample_rate if sample_rate > 0 else 0.0,
            channels=float(info.num_channels),
            frame_count=frames,
            backend="torchaudio",
            message="ok",
        )
    except Exception:
        pass
    try:
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = float(handle.getframerate())
            frames = float(handle.getnframes())
            channels = float(handle.getnchannels())
        return AudioFeatureSummary(
            True,
            sample_rate=sample_rate,
            duration_s=frames / sample_rate if sample_rate > 0 else 0.0,
            channels=channels,
            frame_count=frames,
            backend="wave",
            message="ok",
        )
    except Exception as exc:
        return AudioFeatureSummary(False, backend="none", message=repr(exc))
