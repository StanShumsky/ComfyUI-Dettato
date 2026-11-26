"""Evidence repair must keep repaired and inserted notes inside the source recording."""

import numpy as np
import torch

from dettanode.dettato.evidence.model import CAND_DT
from dettanode.dettato.evidence.repair import EvidenceRepairer


def _repairer(monkeypatch, seconds=1.0):
    res = dict(onset=np.array([0.1]), offset=np.array([0.5]), pitch=np.array([60]),
               conf=np.array([0.9]), slot=np.array([1]))
    ev = np.zeros((1, 769), np.float32)
    monkeypatch.setattr("dettanode.dettato.evidence.repair.transcribe_with_evidence", lambda *a, **k: (res, {}, ev))

    class Candidates:
        def candidates(self, batch):
            out = np.zeros(3, CAND_DT)
            out["pitch"] = [60, 62, 64]
            out["on"] = [0.1, 7.9, 0.95]
            out["off"] = [0.5, 8.5, 2.0]
            out["score"] = 0.9
            out["kind"] = [0, 1, 1]
            return [out]

    repairer = EvidenceRepairer.__new__(EvidenceRepairer)
    repairer.model = Candidates()
    repairer.device = torch.device("cpu")
    repairer.keep_th, repairer.insert_th = 0.4, 0.6
    audio = np.zeros(int(seconds * 16000), np.float32)
    return repairer, audio


def test_repairer_drops_insertions_after_recording_end(monkeypatch):
    repairer, audio = _repairer(monkeypatch)
    rep, _, _ = repairer(None, audio)
    assert np.all(rep["onset"] < 1.0)
    assert np.all(rep["offset"] <= 1.0)
    assert not np.any(np.isin(rep["pitch"], 62))


def test_repairer_clamps_insertion_end_to_recording(monkeypatch):
    repairer, audio = _repairer(monkeypatch)
    rep, _, _ = repairer(None, audio)
    row = rep["pitch"] == 64
    assert row.sum() == 1
    assert rep["onset"][row][0] == 0.95
    assert rep["offset"][row][0] == 1.0
