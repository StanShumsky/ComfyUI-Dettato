import pytest

from dettanode.dettato.transcriber.core import Transcriber


def _transcriber():
    transcriber = Transcriber.__new__(Transcriber)
    transcriber.hop = 0.02
    return transcriber


@pytest.mark.parametrize("frames", [1, 7, 14, 28, 56, 111, 112, 247, 813])
def test_exact_frame_boundaries_do_not_add_a_frame(frames):
    assert _transcriber()._layout(frames * 320, 0.25)[2] == frames


@pytest.mark.parametrize("frames", [1, 7, 28, 111])
def test_partial_final_frame_is_retained(frames):
    assert _transcriber()._layout(frames * 320 + 1, 0.25)[2] == frames + 1
