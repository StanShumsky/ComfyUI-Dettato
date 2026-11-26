from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dettanode.dettato.drums import infer as drum_infer
from dettanode.dettato.evidence.model import CAND_DT
from dettanode.dettato.evidence.repair import NOTE_DT, EvidenceRepairer
from dettanode.dettato.grouper.core import Grouper
from dettanode.dettato.transcriber.core import HEADS, Transcriber


class NodeInterrupted(RuntimeError):
    pass


class TinyTranscriberNet(torch.nn.Module):
    hop_out = 0.02
    head = SimpleNamespace(in_channels=4)

    def forward(self, audio, box):
        box = {} if box is None else box
        b, frames = audio.shape[0], audio.shape[-1] // 320
        for name in HEADS:
            box[name] = torch.zeros(b, frames, 88, device=audio.device)
        for name in ("hidden", "stem"):
            box[name] = torch.zeros(b, 4, frames, 88, dtype=torch.float16, device=audio.device)
        return box


def test_transcriber_run_stops_between_window_batches():
    transcriber = Transcriber(TinyTranscriberNet(), {})
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise NodeInterrupted()

    with pytest.raises(NodeInterrupted):
        transcriber.run(np.zeros(24 * 16000, np.float32), batch=2, taps=("hidden",), check=check)
    assert calls == 3


def test_evidence_edits_stop_between_windows():
    class Candidates:
        calls = 0

        def candidates(self, batch):
            self.calls += 1
            out = np.zeros(len(batch["inst"][0]), CAND_DT)
            out["score"] = 1.0
            return [out]

    repairer = EvidenceRepairer.__new__(EvidenceRepairer)
    repairer.device = torch.device("cpu")
    repairer.model = Candidates()
    notes = np.zeros(2, NOTE_DT)
    notes["on"], notes["off"], notes["pitch"] = [0.0, 10.0], [0.5, 10.5], [60, 64]
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NodeInterrupted()

    with pytest.raises(NodeInterrupted):
        repairer.edits(notes, np.zeros((2, 1), np.float32), check=check)
    assert calls == 2 and repairer.model.calls == 1


def test_grouper_predict_stops_between_batches():
    class Predictor(torch.nn.Module):
        calls = 0

        def forward(self, batch):
            self.calls += 1
            b, n = batch["mask"].shape
            return {
                "inst": torch.zeros(b, n, 73, device=batch["mask"].device),
                "emb": torch.zeros(b, n, 4, device=batch["mask"].device),
                "h": torch.zeros(b, n, 2, device=batch["mask"].device),
            }

    grouper = Grouper.__new__(Grouper)
    grouper.device = torch.device("cpu")
    grouper.model = Predictor()
    grouper.window_s, grouper.window_hop, grouper.max_notes, grouper.batch = 8.0, 0.5, 512, 1
    meta = np.zeros((4, 12), np.float32)
    meta[:, 0] = [0.0, 10.0, 20.0, 30.0]
    song = SimpleNamespace(
        meta=meta,
        window=lambda t0, W, sel: {
            "audio": np.zeros(160, np.float32),
            "notes": np.zeros((len(sel), 12), np.float32),
            "feat": np.zeros((len(sel), 3, 368), np.float16),
        },
    )
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NodeInterrupted()

    with pytest.raises(NodeInterrupted):
        grouper.predict(song, check=check)
    assert calls == 2 and grouper.model.calls == 1


def test_drum_transcribe_checks_before_feature_work(monkeypatch):
    drums = drum_infer.DrumTranscriber.__new__(drum_infer.DrumTranscriber)

    def unexpected_features(audio):
        pytest.fail("drum feature extraction started after cancellation")

    monkeypatch.setattr(drum_infer, "features", unexpected_features)

    def check():
        raise NodeInterrupted()

    with pytest.raises(NodeInterrupted):
        drums.transcribe(np.zeros(44100, np.float32), check=check)


def test_transcribe_node_uses_comfyui_interrupt_flag(nodes, monkeypatch):
    from dettanode.dettato.transcriber.core import SR

    transcriber = Transcriber(TinyTranscriberNet(), {
        "on_th": None, "v2_th": None, "off_th": 0.0, "fr_th": 0.0, "fr_low_frames": 1, "on_nms": 0,
    })
    stages = nodes.DettatoStages(transcriber)
    patcher = nodes.comfy.model_patcher.CoreModelPatcher(stages, torch.device("cpu"), torch.device("cpu"))
    wrapper = {
        "transcriber": transcriber, "evidence": None, "grouper": None, "drums": None,
        "stages": stages, "patcher": patcher, "device": torch.device("cpu"),
        "dtype": torch.float32, "name": "tiny",
    }
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise NodeInterrupted()

    monkeypatch.setattr(nodes.mm, "throw_exception_if_processing_interrupted", interrupt)
    audio = {"waveform": torch.zeros(1, 1, 8 * SR), "sample_rate": SR}
    with pytest.raises(NodeInterrupted):
        nodes.DettatoTranscribe.execute(wrapper, {"audio_0": audio}, min_velocity=1, dedup=False)
    assert calls == 5
