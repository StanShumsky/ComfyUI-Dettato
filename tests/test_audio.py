import subprocess
from types import SimpleNamespace

import pytest

from dettanode.dettato.audio import _ffmpeg, _probe_rate


def test_ffmpeg_rejects_unknown_sample_rate(monkeypatch):
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise subprocess.CalledProcessError(1, ["ffprobe"], stderr="Invalid data found")

    monkeypatch.setattr("dettanode.dettato.audio.subprocess.run", run)

    with pytest.raises(RuntimeError, match="sample rate"):
        _ffmpeg("missing-rate.m4a")
    assert calls == 1


def test_probe_rate_reports_unreadable_source(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffprobe"], stderr="Invalid data found")

    monkeypatch.setattr("dettanode.dettato.audio.subprocess.run", run)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        _probe_rate("broken.m4a")


@pytest.mark.parametrize("output", ["", "N/A", "0"])
def test_probe_rate_rejects_missing_or_invalid_rate(monkeypatch, output):
    monkeypatch.setattr("dettanode.dettato.audio.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=output))

    with pytest.raises(RuntimeError, match="sample rate"):
        _probe_rate("no-audio.m4a")


def test_probe_rate_reports_missing_ffprobe(monkeypatch):
    def run(*args, **kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("dettanode.dettato.audio.subprocess.run", run)

    with pytest.raises(RuntimeError, match="Install FFmpeg"):
        _probe_rate("audio.m4a")
