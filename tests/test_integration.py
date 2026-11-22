"""Integration tests: load the real weights and run the nodes end to end.

Opt in with RUN_DET_WEIGHTS=1; weights download once into DET_MODELS_DIR
(default: a temp dir) on first use.
"""

import json
import os

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_DET_WEIGHTS") != "1",
        reason="set RUN_DET_WEIGHTS=1 to download weights and run integration tests",
    ),
]


def synth_song(t=4.0, rate=48000):
    """A few harmonic-rich notes at 48 kHz (so the drum stage runs too): a C-major arpeggio."""
    freqs = [261.63, 329.63, 392.0, 523.25]  # C4 E4 G4 C5
    n = int(t * rate)
    x = np.zeros(n, np.float32)
    for i, f in enumerate(freqs):
        a, b = int(i * t / len(freqs) * rate), int((i + 1) * t / len(freqs) * rate)
        seg = np.arange(b - a)
        env = np.clip(np.minimum(seg / 100.0, (b - a - seg) / 500.0), 0.0, 1.0)
        tone = sum(np.sin(2 * np.pi * f * h * seg / rate) / h for h in (1, 2, 3, 4))
        x[a:b] += 0.25 * env * tone
    return {"waveform": torch.from_numpy(x)[None, None], "sample_rate": rate}


@pytest.fixture(scope="module")
def wrapper(nodes):
    (w,) = nodes.DettatoLoader.execute("dettato", "auto").args
    for stage in (nodes.DettatoEvidence, nodes.DettatoGrouper, nodes.DettatoDrums):
        (w,) = stage.execute(w).args
    return w


@pytest.fixture(scope="module")
def transcription(nodes, wrapper):
    (payload, text) = nodes.DettatoTranscribe.execute(wrapper, {'audio_0': synth_song()}, 1).args
    return payload, text


def test_transcribe_structure(nodes, transcription):
    payload, text = transcription
    assert payload["seconds"] == pytest.approx(4.0, abs=0.05)
    assert payload["tracks"] and all(hasattr(t, "notes") for t in payload["tracks"])
    n_notes = sum(len(np.asarray(t.notes, np.float64).reshape(-1, 4)) for t in payload["tracks"])
    assert n_notes >= 1, "a 4 s arpeggio should transcribe to at least one note"
    drums = payload["info"]["drums"]
    assert isinstance(drums, dict)  # 48 kHz source: the drum stage must have run
    assert drums["hits"] >= 0
    assert "tuning_cents" in payload["info"]
    data = json.loads(text)
    assert "notes" in data and data["tracks"]


def test_piano_roll_from_real_tracks(nodes, transcription):
    payload = transcription[0]
    (image,) = nodes.DettatoPianoRoll.execute(payload, 0.0, 0, True).args
    assert image.shape[0] == 1 and image.shape[-1] == 3  # (1, H, W, 3)
    assert image.shape[1] > 100 and image.shape[2] > 100
    assert image.shape[-1] == 3


def test_save_midi_roundtrip(nodes, transcription, output_dir):
    payload = transcription[0]
    nodes.DettatoSaveMIDI.execute(payload, "integration/midi")
    from pretty_midi import PrettyMIDI

    path = next((output_dir / "integration").glob("midi_*.mid"))
    pm = PrettyMIDI(str(path))
    assert len(pm.instruments) >= 1
    n_mid = sum(len(i.notes) for i in pm.instruments)
    n_src = sum(len(np.asarray(t.notes, np.float64).reshape(-1, 4)) for t in payload["tracks"])
    assert n_mid == n_src


def test_info_node(nodes, transcription):
    (text,) = nodes.DettatoInfo.execute(transcription[0]).args
    data = json.loads(text)
    assert data["seconds"] == pytest.approx(4.0, abs=0.05)
    assert data["tracks"]
