"""Unit tests: no weights, no ComfyUI install, no network."""

import json

import numpy as np
import pytest
import torch
from PIL import Image


# --------------------------------------------------------------------- schemas


def test_node_schemas(nodes):
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "DettatoLoader", "DettatoEvidence", "DettatoGrouper", "DettatoDrums",
        "DettatoTranscribe", "DettatoPianoRoll", "DettatoSaveMIDI", "DettatoInfo",
    }
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert isinstance(cls.DESCRIPTION, str) and cls.DESCRIPTION
        assert cls.CATEGORY == "music/Dettato"
        assert callable(getattr(cls, cls.FUNCTION))
        returns = cls.RETURN_TYPES
        assert len(returns) == len(getattr(cls, "RETURN_NAMES", returns))
        required = cls.INPUT_TYPES()["required"]
        assert required
        for spec in required.values():
            assert isinstance(spec, tuple) and len(spec) in (1, 2)
            if isinstance(spec[0], list):
                assert spec[0]
                if len(spec) == 2 and spec[1].get("multi_select"):
                    assert isinstance(spec[1]["default"], list)


def test_loader_choices(nodes, models_dir):
    required = nodes.DettatoLoader.INPUT_TYPES()["required"]
    assert set(required) == {"model", "device"}
    assert required["model"][0][0] == "dettato"
    # a collection folder appears in the dropdown
    (models_dir / "my-pack" / "dettato").mkdir(parents=True)
    (models_dir / "my-pack" / "dettato" / "config.json").write_text("{}")
    (models_dir / "my-pack" / "dettato" / "model.safetensors").write_bytes(b"x")
    assert "my-pack" in nodes.DettatoLoader.INPUT_TYPES()["required"]["model"][0]


# --------------------------------------------------------------------- audio


def test_audio_to_mono(nodes, make_audio):
    mono, rate = nodes.audio_to_mono(make_audio(t=0.5, rate=48000, freq=220.0))
    assert rate == 48000 and len(mono) == 24000 and mono.dtype == np.float32
    stereo = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 16000}
    stereo["waveform"][0, 0] = 1.0
    mono, _ = nodes.audio_to_mono(stereo)
    assert np.allclose(mono, 0.5)
    with pytest.raises(ValueError):
        nodes.audio_to_mono({"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000})
    with pytest.raises(ValueError):
        nodes.audio_to_mono({"waveform": torch.zeros(1, 1, 10), "sample_rate": 0})


# --------------------------------------------------------------------- piano roll


def test_piano_roll_shape_and_panels(nodes, make_tracks):
    img = nodes.render_piano_roll(make_tracks()["tracks"], 8.0)
    from dettanode.piano_roll import N_DRUM_ROWS, N_PITCH_ROWS

    assert img.shape == ((N_PITCH_ROWS + N_DRUM_ROWS + 1) * 2, img.shape[1], 3)
    assert img.min() >= 0 and img.max() <= 1
    without = nodes.render_piano_roll(make_tracks()["tracks"], 8.0, show_drum_panel=False)
    assert without.shape[0] == (N_PITCH_ROWS + 1) * 2


def test_piano_roll_notes_and_ticks(nodes, make_tracks):
    payload = make_tracks(seconds=4.0)
    img = nodes.render_piano_roll(payload["tracks"], 4.0, px_per_ms=2.0)
    assert img.shape[1] == 4096  # 4 s * 2000 px/s, capped
    assert img.max() > 0.5  # note pixels are much brighter than any background colour
    # more than background colours present (notes, ticks, C rows, separator)
    cols = np.unique(img[:, 10:-10, :].reshape(-1, 3), axis=0)
    assert cols.shape[0] > 5


def test_piano_roll_max_seconds(nodes, make_tracks):
    full = nodes.render_piano_roll(make_tracks()["tracks"], 8.0, px_per_ms=0.1)
    short = nodes.render_piano_roll(make_tracks()["tracks"], 8.0, px_per_ms=0.1, max_seconds=2)
    assert short.shape[1] == full.shape[1] // 4


def test_piano_roll_velocity_brightness(nodes, make_track, make_tracks):
    loud = make_tracks(tracks=[make_track("a", 0, ((1.0, 1.5, 60, 127.0)))])
    quiet = make_tracks(tracks=[make_track("a", 0, ((1.0, 1.5, 60, 1.0)))])
    il = nodes.render_piano_roll(loud["tracks"], 4.0, px_per_ms=2.0)
    iq = nodes.render_piano_roll(quiet["tracks"], 4.0, px_per_ms=2.0)
    assert il.max() > iq.max()


# --------------------------------------------------------------------- MIDI writing


def test_save_midi(nodes, output_dir, make_tracks):
    from pretty_midi import PrettyMIDI

    result = nodes.DettatoSaveMIDI().save(make_tracks(), "unit/midi")
    assert result == ()  # no outputs, no preview: the node just writes the file
    files = sorted((output_dir / "unit").glob("midi_*.mid"))
    assert len(files) == 1
    pm = PrettyMIDI(str(files[0]))
    by_name = {i.name: i for i in pm.instruments}
    assert set(by_name) == {"Piano", "Drums"}
    assert by_name["Drums"].is_drum and not by_name["Piano"].is_drum
    assert by_name["Piano"].program == 0
    piano = sorted(by_name["Piano"].notes, key=lambda n: n.start)
    assert len(piano) == 2
    assert abs(piano[0].start - 0.0) < 1e-3 and abs(piano[0].end - 0.25) < 1e-2
    assert piano[1].velocity == 127 and piano[0].velocity == 90
    assert abs(piano[0].pitch - 60) < 1e-6
    # no .json sidecar: the summary travels as info_json
    assert list((output_dir / "unit").glob("midi_*.json")) == []
    # second save increments the counter
    nodes.DettatoSaveMIDI().save(make_tracks(), "unit/midi")
    assert len(list((output_dir / "unit").glob("midi_*.mid"))) == 2


def test_info(nodes, make_tracks):
    (text,) = nodes.DettatoInfo().info(make_tracks())
    data = json.loads(text)
    assert data["source"] == "audio" and data["seconds"] == pytest.approx(8.0)
    assert len(data["tracks"]) == 2 and data["tracks"][1]["is_drum"]


def test_filter_min_velocity(nodes, make_tracks):
    tracks = make_tracks()["tracks"]
    kept = nodes.filter_min_velocity(tracks, 100)
    names = [(t.name, len(t.notes)) for t in kept]
    assert names == [("Piano", 1), ("Drums", 1)]
    assert nodes.filter_min_velocity(tracks, 200) == []


# --------------------------------------------------------------------- weight health & casting


def test_check_weights(nodes):
    good = {"a": torch.randn(64, 64), "b": torch.randint(0, 2, (3,)).to(torch.int64)}
    nodes._check_weights(good, "t")  # no raise
    bad = dict(good, a=torch.full((64, 64), float("nan")))
    with pytest.raises(RuntimeError):
        nodes._check_weights(bad, "t")
    zeros = {f"z{i}": torch.zeros(64, 64) for i in range(4)}
    with pytest.raises(RuntimeError):
        nodes._check_weights(zeros, "t")



# --------------------------------------------------------------------- local collections


def _write_stage(base, stage_dir, config, state):
    d = base / stage_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    from safetensors.torch import save_file

    save_file({k: v.cpu() for k, v in state.items()}, str(d / "model.safetensors"))
    return d


def test_local_collections_discovery(nodes, models_dir):
    assert nodes._local_collections() == []
    (models_dir / "empty-pack").mkdir()
    (models_dir / "half" / "dettato-evidence").mkdir(parents=True)
    (models_dir / "half" / "dettato-evidence" / "config.json").write_text("{}")
    (models_dir / "half" / "dettato-evidence" / "model.safetensors").write_bytes(b"x")
    assert nodes._local_collections() == []  # a collection without the transcriber is not a usable model


def test_small_collection_load_and_run(nodes, models_dir, make_audio):
    """Build tiny random transcriber + evidence stages, save them as a local collection,
    load through the real loader path and run the transcription plumbing offline."""
    from dettanode.dettato.evidence.model import Repairer
    from dettanode.dettato.transcriber.model import LaneNet

    torch.manual_seed(0)
    # c must be 368: the evidence stage reads the transcriber's 368-d hidden state (C_H in
    # dettato/evidence/features.py). depth is tiny, so this stays a small, fast model.
    transcriber_cfg = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    _write_stage(
        models_dir / "small", "dettato",
        {"model": transcriber_cfg, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4, "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        LaneNet(**transcriber_cfg).state_dict(),
    )
    evidence_cfg = {"d": 56, "layers": 1, "heads": 2, "ff_mult": 3, "n_ev": 769}
    _write_stage(
        models_dir / "small", "dettato-evidence",
        {"model": evidence_cfg, "keep_th": 0.4, "insert_th": 0.6, "window_s": 8.0},
        Repairer(**evidence_cfg).state_dict(),
    )

    wrapper = nodes.load_dettato("small", "cpu")
    assert wrapper["evidence"] is None and wrapper["grouper"] is None and wrapper["drums"] is None
    # a stage that is not in the collection cannot be attached
    with pytest.raises(FileNotFoundError, match="lacks stage"):
        nodes.DettatoGrouper().attach(wrapper)
    (wrapper,) = nodes.DettatoEvidence().attach(wrapper)
    assert wrapper["evidence"] is not None and wrapper["grouper"] is None and wrapper["drums"] is None

    payload = nodes.run_transcription(wrapper, make_audio(t=1.0, rate=48000, freq=440.0))
    assert payload["seconds"] == pytest.approx(1.0, abs=0.01)
    assert len(payload["tracks"]) == 1 and payload["tracks"][0].name == "Notes"
    notes = np.asarray(payload["tracks"][0].notes, np.float64).reshape(-1, 4)
    assert notes.shape[1] == 4 and (notes[:, 3] >= 1).all() and (notes[:, 3] <= 127).all()
    assert payload["info"]["notes"]["transcribed"] >= 0
    assert payload["info"]["drums"] == "skipped: drum stage not attached"

    # and the full node path (piano roll + summary + save) works on it
    (roll_image,) = nodes.DettatoPianoRoll().render(payload, 0.0, 0, True)
    assert roll_image.shape[1:] == (roll_image.shape[1], roll_image.shape[2], 3)
    (text,) = nodes.DettatoInfo().info(payload)
    json.loads(text)


def test_local_collection_without_transcriber_error(nodes, models_dir):
    (models_dir / "only-evidence" / "dettato-evidence").mkdir(parents=True)
    (models_dir / "only-evidence" / "dettato-evidence" / "config.json").write_text("{}")
    (models_dir / "only-evidence" / "dettato-evidence" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="lacks the transcriber"):
        nodes.DettatoLoader().load("only-evidence", "cpu")


def test_unknown_model_error(nodes, models_dir):
    with pytest.raises(FileNotFoundError, match="not found"):
        nodes.DettatoLoader().load("nope", "cpu")