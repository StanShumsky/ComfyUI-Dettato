import numpy as np

from dettanode.dettato.transcriber.decode import decode


def test_first_frame_refinement_cannot_move_onset_before_recording():
    onset = np.zeros((4, 88), np.float32)
    frame = np.ones((4, 88), np.float32)
    offset = np.zeros((4, 88), np.float32)
    onset[0, 60] = 0.9
    onset[1, 60] = 0.2

    notes = decode(onset, frame, offset, hop=0.02)

    assert len(notes) == 1
    assert notes[0][0] == 0.0


def test_interior_onset_refinement_is_preserved():
    onset = np.zeros((6, 88), np.float32)
    frame = np.ones((6, 88), np.float32)
    offset = np.zeros((6, 88), np.float32)
    onset[1:4, 60] = [0.2, 0.9, 0.8]

    notes = decode(onset, frame, offset, hop=0.02)

    assert len(notes) == 1
    assert np.isclose(notes[0][0], (2.0 + 3.0 / 7.0) * 0.02)
