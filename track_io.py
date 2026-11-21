"""Conversions between ComfyUI payloads and Dettato tracks, plus MIDI / JSON writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .dettato.pipeline import Transcription, Track


def audio_to_mono(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO {waveform (B, C, N) or (C, N), sample_rate} -> mono float32 numpy and the rate."""
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    rate = int(audio["sample_rate"])
    if rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {rate}.")
    x = waveform.detach().float()
    if x.dim() == 3:
        x = x[0] if x.shape[0] == 1 else x.mean(0)
    if x.dim() != 2:
        raise ValueError(f"Audio waveform must be (channels, samples) or (1, channels, samples), got {tuple(x.shape)}.")
    if x.shape[-1] == 0:
        raise ValueError("The audio is empty.")
    return x.mean(0).cpu().numpy().astype(np.float32), rate


def make_tracks_payload(tracks: list[Track], seconds: float, source: str, info: dict) -> dict:
    """The DETTATO_TRACKS payload shared by the nodes."""
    return {
        "tracks": tracks,
        "seconds": float(seconds),
        "source": source,
        "info": info,
    }


def build_transcription(payload: dict) -> Transcription:
    return Transcription(
        source=payload.get("source", "audio"),
        seconds=payload.get("seconds", 0.0),
        tracks=payload["tracks"],
        info=dict(payload.get("info", {})),
    )


def summary_json(payload: dict) -> str:
    return json.dumps(build_transcription(payload).summary(), indent=2, sort_keys=True)


def write_midi(payload: dict, path: Path) -> None:
    """Write the transcription as .mid (one General MIDI track per instrument)."""
    build_transcription(payload).save_midi(str(path))


def filter_min_velocity(tracks: list[Track], min_velocity: int) -> list[Track]:
    """Drop notes quieter than min_velocity; tracks that end up empty are removed."""
    out = []
    for track in tracks:
        notes = np.asarray(track.notes, np.float64).reshape(-1, 4)
        if notes.shape[0]:
            notes = notes[notes[:, 3] >= min_velocity]
        if notes.shape[0]:
            out.append(Track(track.name, track.program, notes.astype(np.float64), is_drum=track.is_drum))
    return out
