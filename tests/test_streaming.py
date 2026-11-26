"""Long-audio feature gathering must match the shared pass without allocating its full-song taps."""

import numpy as np
import pytest
import torch

from dettanode.dettato.grouper.core import Grouper
from dettanode.dettato.transcriber.core import Transcriber
from dettanode.dettato.transcriber.model import LaneNet


@pytest.fixture(scope="module")
def transcriber():
    torch.manual_seed(7)
    cfg = {
        "c": 368, "depth": 1, "exp": 2, "kt": 3, "dilations": [1], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    thresholds = {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4, "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}
    return Transcriber(LaneNet(**cfg).eval(), thresholds, "cpu")


def _audio(seconds=12.0):
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
    return (0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)


def test_streaming_features_match_shared_pass(transcriber):
    y = _audio()
    notes = np.array([
        [-0.02, 0.40, 69], [0.20, 1.20, 69], [1.40, 2.00, 72], [2.30, 3.50, 76],
        [4.10, 4.80, 71], [5.20, 6.30, 69], [7.10, 8.00, 74], [9.00, 20.00, 72],
        [11.80, 12.20, 60], [12.10, 12.30, 61],
    ], np.float64)
    _, _, (maps, taps) = transcriber.run(
        y, hop_frac=0.25, batch=2, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"))
    grouper = object.__new__(Grouper)
    expected = grouper.features_from_run(maps, taps, notes, transcriber)
    actual = grouper.features_streaming(y, notes, transcriber, batch=2)
    for a, b in zip(expected, actual):
        assert np.array_equal(a, b)


def test_streaming_features_handle_no_notes(transcriber):
    grouper = object.__new__(Grouper)
    meta, feat, scale = grouper.features_streaming(_audio(1.0), np.empty((0, 3)), transcriber, batch=2)
    assert meta.shape == (0, 12) and meta.dtype == np.float32
    assert feat.shape == (0, 3, 368) and feat.dtype == np.int8
    assert scale.shape == (0, 3) and scale.dtype == np.float16


def test_streaming_features_honor_interrupts(transcriber):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")

    grouper = object.__new__(Grouper)
    with pytest.raises(RuntimeError, match="stop"):
        grouper.features_streaming(_audio(9.0), np.array([[0.2, 1.0, 69]]), transcriber, batch=1, check=check)
    assert calls == 2
