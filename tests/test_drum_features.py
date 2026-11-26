import importlib

import numpy as np
import pytest
import torch

drum_features = importlib.import_module("dettanode.dettato.drums.features")


def reference_logspec(x, n_fft, hop, fb):
    spectrum = torch.stft(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), n_fft,
        hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft),
        center=True, return_complex=True, pad_mode="constant",
    )
    return torch.log(fb @ (spectrum.abs() ** 2) + drum_features.FLOOR).T.numpy()


def signal(seconds, rate):
    t = np.arange(round(seconds * rate), dtype=np.float32) / rate
    rng = np.random.default_rng(123)
    return (0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.mark.parametrize("rate,n_fft,hop,name", [
    (16000, 512, 160, "lo"),
    (44100, 1411, 441, "hi"),
    (16000, 6144, 160, "cq"),
])
def test_chunked_logspec_matches_whole_song(rate, n_fft, hop, name):
    x = signal(11.0, rate)
    fb = drum_features._filters(name)
    reference = reference_logspec(x, n_fft, hop, fb)[:len(x) // hop]
    chunked = drum_features._logspec(x, n_fft, hop, fb)
    assert chunked.shape == reference.shape
    np.testing.assert_allclose(chunked, reference, rtol=0, atol=1e-6)
    different = int(np.count_nonzero(chunked.astype(np.float16) != reference.astype(np.float16)))
    assert different <= max(1, reference.size // 100000)


def test_short_logspec_keeps_original_path():
    x = signal(1.0, 16000)
    fb = drum_features._filters("lo")
    np.testing.assert_array_equal(
        drum_features._logspec(x, 512, 160, fb), reference_logspec(x, 512, 160, fb)
    )


def test_long_drum_features_are_near_bitwise_identical():
    import soxr

    x44 = signal(12.0, drum_features.SR)
    x = (x44 * drum_features.level_gain(x44)).astype(np.float32)
    x16 = soxr.resample(x, drum_features.SR, 16000).astype(np.float32)
    T = len(x16) // 160
    reference = np.concatenate([
        reference_logspec(x16, 512, 160, drum_features._filters("lo"))[:T],
        reference_logspec(x, 1411, 441, drum_features._filters("hi"))[:T],
        reference_logspec(x16, 6144, 160, drum_features._filters("cq"))[:T],
    ], 1).astype(np.float16)
    chunked = drum_features.features(x44)
    assert chunked.shape == (T, 280) and chunked.dtype == np.float16
    np.testing.assert_allclose(chunked, reference, rtol=1e-3, atol=1e-4)
    assert np.count_nonzero(chunked != reference) <= 2


def test_drum_feature_chunks_are_interruptible():
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        drum_features.features(signal(11.0, drum_features.SR), check=check)
    assert calls == 4
