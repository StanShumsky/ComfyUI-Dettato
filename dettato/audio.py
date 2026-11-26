"""Reading audio files: one decode at the file's own rate, mono, then resampled for each model.

The transcriber, the evidence model and the grouper work on 16 kHz audio; the drum model on 44.1 kHz audio and only when the
source itself was recorded at 40 kHz or more.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".opus")


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono at `rate`
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate

    def at(self, rate: int) -> np.ndarray:
        """The audio resampled to ``rate`` (float32 mono)."""
        if rate == self.rate:
            return self.samples
        import soxr

        return soxr.resample(self.samples, self.rate, rate, quality="HQ").astype(np.float32)


def _ffmpeg(path: str) -> Audio:
    rate = _probe_rate(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
                         capture_output=True, check=True).stdout
    return Audio(np.frombuffer(raw, np.float32).copy(), rate)


def _probe_rate(path: str) -> int:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
           "-of", "csv=p=0", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found. Install FFmpeg and ensure its bin directory is on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        message = f"ffprobe could not determine the audio sample rate for {path!r}."
        if detail:
            message += f" {detail}."
        raise RuntimeError(message + " Check the source file and the FFmpeg installation.") from exc
    try:
        rate = int(out.strip().split(",")[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"ffprobe did not report an audio sample rate for {path!r}.") from exc
    if rate <= 0:
        raise RuntimeError(f"ffprobe reported an invalid audio sample rate for {path!r}: {rate}.")
    return rate


def load(path: str) -> Audio:
    """Any audio file -> mono float32 at its own sample rate (soundfile first, ffmpeg for anything it cannot read)."""
    try:
        import soundfile as sf

        x, rate = sf.read(path, dtype="float32", always_2d=True)
        return Audio(x.mean(1).astype(np.float32), int(rate))
    except Exception:  # noqa: BLE001
        return _ffmpeg(path)
