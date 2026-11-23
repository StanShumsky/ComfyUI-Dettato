"""Unit tests: no weights, no ComfyUI install, no network."""

import json
import re

import numpy as np
import pytest
import torch
from PIL import Image


# --------------------------------------------------------------------- schemas


def test_node_schemas(nodes):
    """V3: every node's schema (node_id / display name / category / description / io ids) is consistent."""
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    prefix = next(iter(nodes.NODE_CLASS_MAPPINGS))[:7]  # all node ids share the pack prefix
    expected = {prefix + s for s in (
        "Loader", "Evidence", "Grouper", "Drums", "Transcribe", "DedupeMIDI", "QuantizeMIDI",
        "PianoRoll", "SaveMIDI", "Info", "MIDIToText", "ToYuE2ABC", "SaveABC",
    )}
    assert set(nodes.NODE_CLASS_MAPPINGS) == expected
    categories = set()
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        schema = cls.define_schema()
        assert schema.node_id == name
        assert schema.display_name == nodes.NODE_DISPLAY_NAME_MAPPINGS[name]
        categories.add(schema.category)
        assert isinstance(schema.description, str) and schema.description
        assert schema.inputs
        in_ids = [i.id for i in schema.inputs]
        assert len(in_ids) == len(set(in_ids))
        out_ids = [o.id for o in schema.outputs]
        assert len(out_ids) == len(set(out_ids))
    assert len(categories) == 1  # one menu category for the whole pack


def test_v3_entrypoint_lists_all_nodes(nodes):
    import asyncio

    import dettanode

    ext = asyncio.run(dettanode.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    assert [c.__name__ for c in node_list] == list(nodes.NODE_CLASS_MAPPINGS)



def test_loader_choices(nodes, models_dir):
    loader = next(c for k, c in nodes.NODE_CLASS_MAPPINGS.items() if k.endswith("Loader"))

    def model_input():
        schema = loader.define_schema()
        return next(i for i in schema.inputs if i.id == "model")

    assert model_input().options[0] == "dettato"
    assert model_input().default == "dettato"
    device = next(i for i in loader.define_schema().inputs if i.id == "device")
    assert device.options == ["auto", "cuda", "mps", "cpu"]
    precision = next(i for i in loader.define_schema().inputs if i.id == "precision")
    assert precision.options == ["auto", "bf16", "fp16", "fp32"]
    # a collection folder appears in the dropdown
    (models_dir / "my-pack" / "dettato").mkdir(parents=True)
    (models_dir / "my-pack" / "dettato" / "config.json").write_text("{}")
    (models_dir / "my-pack" / "dettato" / "model.safetensors").write_bytes(b"x")
    assert "my-pack" in model_input().options



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

    result = nodes.DettatoSaveMIDI.execute(make_tracks(), "unit/midi")
    assert result.result is None  # no outputs, no preview: the node just writes the file
    files = sorted((output_dir / "unit").glob("midi_*.mid"))
    assert len(files) == 1
    pm = PrettyMIDI(str(files[0]))
    by_name = {i.name: i for i in pm.instruments}
    assert set(by_name) == {"Piano", "Drums"}
    assert by_name["Drums"].is_drum and not by_name["Piano"].is_drum
    assert by_name["Piano"].program == 0
    # GM acoustic kit on the percussion channel, so DAWs play drums (not the piano patch)
    assert by_name["Drums"].program == 37
    import mido
    mf = mido.MidiFile(str(files[0]))
    assert any(any(m.type == "program_change" and m.program == 37 for m in tr) and
               any(m.type == "note_on" and m.channel == 9 for m in tr) for tr in mf.tracks)
    piano = sorted(by_name["Piano"].notes, key=lambda n: n.start)
    assert len(piano) == 2
    assert abs(piano[0].start - 0.0) < 1e-3 and abs(piano[0].end - 0.25) < 1e-2
    assert piano[1].velocity == 127 and piano[0].velocity == 90
    assert abs(piano[0].pitch - 60) < 1e-6
    # no .json sidecar: the summary travels as info_json
    assert list((output_dir / "unit").glob("midi_*.json")) == []
    # second save increments the counter
    nodes.DettatoSaveMIDI.execute(make_tracks(), "unit/midi")
    assert len(list((output_dir / "unit").glob("midi_*.mid"))) == 2


def test_info(nodes, make_tracks):
    (text,) = nodes.DettatoInfo.execute(make_tracks()).args
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

    wrapper = nodes.load_dettato("small", "fp32", "cpu")
    assert wrapper["evidence"] is None and wrapper["grouper"] is None and wrapper["drums"] is None
    # a stage that is not in the collection cannot be attached
    with pytest.raises(FileNotFoundError, match="lacks stage"):
        nodes.DettatoGrouper.execute(wrapper)
    (wrapper,) = nodes.DettatoEvidence.execute(wrapper).args
    assert wrapper["evidence"] is not None and wrapper["grouper"] is None and wrapper["drums"] is None

    payload = nodes.run_transcription(wrapper, make_audio(t=1.0, rate=48000, freq=440.0))
    assert payload["seconds"] == pytest.approx(1.0, abs=0.01)
    assert len(payload["tracks"]) == 1 and payload["tracks"][0].name == "Notes"
    notes = np.asarray(payload["tracks"][0].notes, np.float64).reshape(-1, 4)
    assert notes.shape[1] == 4 and (notes[:, 3] >= 1).all() and (notes[:, 3] <= 127).all()
    assert payload["info"]["notes"]["transcribed"] >= 0
    assert payload["info"]["drums"] == "skipped: drum stage not attached"

    # and the full node path (piano roll + summary + save) works on it
    (roll_image,) = nodes.DettatoPianoRoll.execute(payload, 0.0, 0, True).args
    assert roll_image.shape[1:] == (roll_image.shape[1], roll_image.shape[2], 3)
    (text,) = nodes.DettatoInfo.execute(payload).args
    json.loads(text)


def test_transcriber_weights_are_loaded(nodes, models_dir):
    """The loader must apply the transcriber's state dict (regression: it once ran on a fresh random init)."""
    from dettanode.dettato.transcriber.model import LaneNet
    from safetensors.torch import load_file

    torch.manual_seed(7)
    cfg_model = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    file_sd = LaneNet(**cfg_model).state_dict()
    _write_stage(
        models_dir / "weights-check", "dettato",
        {"model": cfg_model, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4,
                                            "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        file_sd,
    )

    wrapper = nodes.load_dettato("weights-check", "fp32", "cpu")
    loaded = {k: v.cpu() for k, v in wrapper["transcriber"].model.state_dict().items()}
    on_disk = load_file(str(models_dir / "weights-check" / "dettato" / "model.safetensors"))
    assert set(loaded) == set(on_disk)
    mismatched = [k for k in on_disk if not torch.equal(loaded[k], on_disk[k].to(loaded[k].dtype))]
    assert not mismatched, f"{len(mismatched)} tensors differ from the file, e.g. {mismatched[:3]}"


def test_local_collection_without_transcriber_error(nodes, models_dir):
    (models_dir / "only-evidence" / "dettato-evidence").mkdir(parents=True)
    (models_dir / "only-evidence" / "dettato-evidence" / "config.json").write_text("{}")
    (models_dir / "only-evidence" / "dettato-evidence" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="lacks the transcriber"):
        nodes.DettatoLoader.execute("only-evidence", "auto", "cpu")


def test_unknown_model_error(nodes, models_dir):
    with pytest.raises(FileNotFoundError, match="not found"):
        nodes.DettatoLoader.execute("nope", "auto", "cpu")
# --------------------------------------------------------------------- multi-source: flatten, merge, render

from dettanode.track_io import (
    base_track_name,
    build_transcription,
    dedupe_tracks,
    dedup_sources,
    quantize_tracks,
    merge_infos,
    merge_tracks,
)


def test_transcribe_autogrow_schema(nodes):
    schema = nodes.DettatoTranscribe.define_schema()
    assert schema.node_id == "DettatoTranscribe"
    assert [i.id for i in schema.inputs] == ["model", "audios", "min_velocity", "dedup"]
    audios = schema.inputs[1]
    assert isinstance(audios, nodes.io.Autogrow.Input)
    assert audios.template.prefix == "audio_"
    assert audios.template.min == 0
    assert audios.template.max == 9
    assert audios.template.names == [f"audio_{i}" for i in range(9)]
    # the existing consumers accept both payload types
    for cls in (nodes.DettatoPianoRoll, nodes.DettatoSaveMIDI, nodes.DettatoInfo, nodes.DettatoToYuE2ABC):
        schema = cls.define_schema()

        midi = next(i for i in schema.inputs if i.id == "midi")

        assert midi.io_type == "DETTATO_MIDI,DETTATO_TRACKS"


def test_merge_tracks_collision_policy(nodes, make_track):
    t = ((0.0, 0.1, 60, 90.0),)
    merged = merge_tracks([
        ("vocals", [make_track("Piano", 0, t), make_track("Drums", 0, t, is_drum=True)]),
        ("accompaniment", [make_track("Piano", 0, ((0.2, 0.3, 64, 127.0),)), make_track("Bass", 33, t)]),
    ])
    # first occurrence keeps the name; repeats get the source label; order is source order
    assert [x.name for x in merged] == ["Piano", "Drums", "Piano (accompaniment)", "Bass"]
    assert merged[1].is_drum and not merged[2].is_drum
    assert merged[2].notes.shape == (1, 4)  # only the name changed, not the track
    # 'Name (label)' itself taken -> numeric fallback
    m = merge_tracks([
        ("x", [make_track("P", 0, t), make_track("P (y)", 0, t)]),
        ("y", [make_track("P", 0, t)]),
    ])
    assert [x.name for x in m] == ["P", "P (y)", "P (y) 2"]
    # no repeats -> names untouched
    assert [x.name for x in merge_tracks([("a", [make_track("Only", 0, t)])])] == ["Only"]


def test_merge_infos(nodes):
    i1 = {"notes": {"transcribed": 5, "dropped": 1, "inserted": 0, "grouped": 4},
          "tuning_cents": 3.0, "drums": "skipped: drum stage not attached"}
    i2 = {"notes": {"transcribed": 7, "dropped": 0, "inserted": 1, "grouped": 6},
          "tuning_cents": 3.0, "drums": {"hits": 2}}
    merged = merge_infos([("vocals", i1), ("accompaniment", i2)])
    assert merged["notes"] == {"transcribed": 12, "dropped": 1, "inserted": 1, "grouped": 10}
    assert merged["tuning_cents"] == 3.0
    assert merged["drums"] == {"vocals": "skipped: drum stage not attached", "accompaniment": {"hits": 2}}
    assert merged["sources"] == {"vocals": i1, "accompaniment": i2}
    # the single plain 'audio' source keeps the per-source shape exactly (back-compat)
    assert merge_infos([("audio", i1)]) == i1
    # disagreeing tunings -> None
    assert merge_infos([("a", dict(i1)), ("b", dict(i2, tuning_cents=12.0))])["tuning_cents"] is None
    assert merge_infos([("a", dict(i1, tuning_cents=None)), ("b", dict(i2, tuning_cents=None))])["tuning_cents"] is None


def test_make_midi_payload(nodes, make_tracks):
    import io
    from pretty_midi import PrettyMIDI

    p1 = make_tracks(seconds=4.0, info={"notes": {"transcribed": 2, "dropped": 0, "inserted": 0, "grouped": 2},
                                        "tuning_cents": 0.0, "drums": {"hits": 1}})
    p2 = make_tracks(seconds=8.0, info={"notes": {"transcribed": 2, "dropped": 0, "inserted": 0, "grouped": 2},
                                        "tuning_cents": 0.0, "drums": "skipped: unit test"})
    payload = nodes.make_midi_payload([("vocals", p1), ("accompaniment", p2)])
    assert payload["sources"] == ["vocals", "accompaniment"]
    assert payload["seconds"] == 8.0  # the longest source
    assert [t.name for t in payload["tracks"]] == ["Piano", "Drums", "Piano (accompaniment)", "Drums (accompaniment)"]
    assert isinstance(payload["midi"], bytes) and payload["midi"][:4] == b"MThd"
    # the pre-rendered .mid parses back to the same track set
    pm = PrettyMIDI(io.BytesIO(payload["midi"]))
    assert [i.name for i in pm.instruments] == [t.name for t in payload["tracks"]]
    assert sum(len(i.notes) for i in pm.instruments) == sum(len(t.notes) for t in payload["tracks"])
    assert pm.instruments[1].is_drum and not pm.instruments[0].is_drum
    # the generic JSON text is carried in the payload and round-trips
    data = json.loads(payload["midi_text"])
    assert data["seconds"] == 8.0
    assert [tr["name"] for tr in data["tracks"]] == [t.name for t in payload["tracks"]]
    assert sum(len(tr["notes"]) for tr in data["tracks"]) == sum(len(t.notes) for t in payload["tracks"])
    assert data["tracks"][1]["program"] is None and data["tracks"][1]["is_drum"] is True
    # duplicate labels are deduplicated; track names stay unique
    dup = nodes.make_midi_payload([("vocals", p1), ("vocals", p2)])
    assert dup["sources"] == ["vocals", "vocals 2"]
    assert set(dup["info"]["sources"]) == {"vocals", "vocals 2"}
    assert len({t.name for t in dup["tracks"]}) == len(dup["tracks"])
    with pytest.raises(ValueError, match="no audio sources"):
        nodes.make_midi_payload([])


def test_save_midi_payload_bytes_verbatim(nodes, output_dir, make_tracks):
    payload = nodes.make_midi_payload([("audio", make_tracks())])
    assert payload["sources"] == ["audio"]
    assert payload["info"] == make_tracks()["info"]
    nodes.DettatoSaveMIDI.execute(payload, "unit/mid")
    files = sorted((output_dir / "unit").glob("mid_*.mid"))
    assert len(files) == 1
    assert files[0].read_bytes() == payload["midi"]  # the pre-rendered bytes, verbatim
    # a legacy DETTATO_TRACKS payload renders on the fly, to the same bytes the renderer gives
    nodes.DettatoSaveMIDI.execute(make_tracks(), "unit/mid")
    files = sorted((output_dir / "unit").glob("mid_*.mid"))
    assert len(files) == 2
    assert files[1].read_bytes() == nodes.render_midi_bytes(make_tracks()["tracks"], 8.0)


def test_single_audio_payload_backcompat(nodes, make_tracks):
    """One plain AUDIO in: the merged payload is the old tracks payload plus the rendered .mid."""
    p = make_tracks(seconds=8.0)
    m = nodes.make_midi_payload([("audio", p)])
    assert m["sources"] == ["audio"]
    assert m["info"] == p["info"]  # flat, exactly the per-source info
    assert m["seconds"] == p["seconds"]
    assert [t.name for t in m["tracks"]] == [t.name for t in p["tracks"]]
    assert all(mt.notes is ot.notes for mt, ot in zip(m["tracks"], p["tracks"]))
    # the same bytes the old write_midi path produced
    import io
    buf = io.BytesIO()
    build_transcription(p).to_midi().write(buf)
    assert m["midi"] == buf.getvalue()
    # and the summary text is identical to the old payload's
    assert nodes.summary_json(m) == nodes.summary_json(p)


def test_multi_audio_end_to_end(nodes, models_dir, make_audio, output_dir):
    """Two AUDIO sources (one per slot) through the real node path: one
    merged .mid, parsed back."""
    import io
    from pretty_midi import PrettyMIDI
    from dettanode.dettato.transcriber.model import LaneNet

    torch.manual_seed(0)
    transcriber_cfg = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    _write_stage(
        models_dir / "multi-pack", "dettato",
        {"model": transcriber_cfg, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4, "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        LaneNet(**transcriber_cfg).state_dict(),
    )
    wrapper = nodes.load_dettato("multi-pack", "fp32", "cpu")

    a1 = make_audio(t=1.0, rate=44100, freq=440.0)
    a2 = make_audio(t=1.0, rate=44100, freq=220.0)
    out = nodes.DettatoTranscribe.execute(wrapper, {"audio_0": a1, "audio_1": a2}, 1)
    payload, text = out.args
    data = json.loads(text)
    assert payload["sources"] == ["audio_0", "audio_1"]
    assert payload["seconds"] == pytest.approx(1.0, abs=0.02)
    names = [t.name for t in payload["tracks"]]
    assert len(names) == len(set(names))  # unique even if both stems yield the same track name
    # per-source info under 'sources', merged totals on top
    assert set(payload["info"]["sources"]) == {"audio_0", "audio_1"}
    assert payload["info"]["notes"]["grouped"] == sum(len(t.notes) for t in payload["tracks"])
    assert data["source"] == "audio_0, audio_1"
    # the pre-rendered .mid parses back to the same track set and note count
    pm = PrettyMIDI(io.BytesIO(payload["midi"]))
    assert [i.name for i in pm.instruments] == names
    assert sum(len(i.notes) for i in pm.instruments) == sum(len(t.notes) for t in payload["tracks"])
    # Save MIDI writes exactly those bytes
    nodes.DettatoSaveMIDI.execute(payload, "multi/mid")
    written = next((output_dir / "multi").glob("mid_*.mid"))
    assert written.read_bytes() == payload["midi"]

    # a single source labeled audio_0: per-source info kept under "sources", tracks identical to the direct run
    audio = make_audio(t=1.0, rate=48000, freq=440.0)
    out2 = nodes.DettatoTranscribe.execute(wrapper, {"audio_0": audio}, 1)
    payload2, _text2 = out2.args
    direct = nodes.run_transcription(wrapper, audio)
    assert payload2["sources"] == ["audio_0"]
    assert payload2["info"]["sources"]["audio_0"] == direct["info"]
    assert payload2["info"]["drums"] == {"audio_0": "skipped: drum stage not attached"}
    assert [t.name for t in payload2["tracks"]] == [t.name for t in direct["tracks"]]


def test_base_track_name():
    assert base_track_name("Drums") == "Drums"
    assert base_track_name("Drums (audio_1)") == "Drums"
    assert base_track_name("synth lead (audio_4) 2") == "synth lead"
    assert base_track_name("electric bass") == "electric bass"
    assert base_track_name("voice (audio_3)") == "voice"
    assert base_track_name("string ensemble") == "string ensemble"


def test_dedupe_tracks(make_track):
    # two stems heard the same kick 30 ms apart: the louder one stays, the exact
    # duplicate (identical onset) collapses to one
    a = make_track("Drums", 0, ((1.0, 1.05, 36, 60.0), (2.0, 2.1, 38, 120.0)), is_drum=True)
    b = make_track("Drums (audio_1)", 0, ((1.03, 1.08, 36, 110.0), (2.0, 2.1, 38, 120.0)), is_drum=True)
    out, stats = dedupe_tracks([a, b], 50)
    assert len(out) == 1 and out[0].name == "Drums" and out[0].is_drum
    onsets = sorted(float(n[0]) for n in out[0].notes)
    assert onsets == [1.03, 2.0]
    assert [float(n[3]) for n in out[0].notes] == [110.0, 120.0]
    assert stats == {"window_ms": 50, "tracks_before": 2, "tracks_after": 1,
                     "notes_removed": 2, "merged": {"Drums": 2}}
    # a same-pitch repeat at 120 ms is music, not a duplicate
    c = make_track("Drums", 0, ((1.0, 1.05, 36, 100.0), (1.12, 1.17, 36, 100.0)), is_drum=True)
    out2, stats2 = dedupe_tracks([c], 50)
    assert len(out2[0].notes) == 2 and stats2["notes_removed"] == 0 and stats2["merged"] == {}
    # window 0 pools the tracks without dropping anything
    out3, stats3 = dedupe_tracks([a, b], 0)
    assert len(out3) == 1 and len(out3[0].notes) == 4 and stats3["notes_removed"] == 0
    # different instruments never touch each other
    d = make_track("Piano", 0, ((1.0, 1.05, 36, 90.0),))
    out4, _ = dedupe_tracks([a, d], 500)
    assert [t.name for t in out4] == ["Drums", "Piano"] and len(out4[1].notes) == 1


def test_dedupe_ghost_before_real(make_track):
    # a smeared ghost 40 ms early and quieter: the loud real note takes its place
    g = make_track("Drums", 0, ((0.96, 0.99, 36, 40.0),), is_drum=True)
    r = make_track("Drums (audio_1)", 0, ((1.0, 1.05, 36, 120.0),), is_drum=True)
    out, _ = dedupe_tracks([g, r], 50)
    assert len(out) == 1 and len(out[0].notes) == 1
    assert float(out[0].notes[0, 0]) == pytest.approx(1.0)
    assert float(out[0].notes[0, 3]) == 120.0


def test_dedupe_payload_and_node(nodes, make_track):
    import io

    from pretty_midi import PrettyMIDI

    a = make_track("Drums", 0, ((1.0, 1.05, 36, 60.0),), is_drum=True)
    b = make_track("Drums (audio_1)", 0, ((1.03, 1.08, 36, 110.0),), is_drum=True)
    c = make_track("Piano", 0, ((0.0, 0.3, 60, 90.0),))
    payload = {
        "tracks": [c, a, b],
        "seconds": 8.0,
        "sources": ["audio_0", "audio_1"],
        "info": {"notes": {"transcribed": 3, "dropped": 0, "inserted": 0, "grouped": 3},
                 "tuning_cents": 0.0, "drums": "skipped: unit test"},
        "midi": b"MThd",
    }
    out = nodes.DettatoDedupeMIDI.execute(payload, 50)
    p, text = out.args
    assert [t.name for t in p["tracks"]] == ["Piano", "Drums"]
    assert p["sources"] == ["audio_0", "audio_1"]
    assert p["seconds"] == 8.0
    assert p["info"]["dedupe"] == {"window_ms": 50, "tracks_before": 3, "tracks_after": 2,
                                   "notes_removed": 1, "merged": {"Drums": 2}}
    assert p["info"]["notes"] == payload["info"]["notes"]  # the rest of the info is untouched
    # the re-rendered .mid carries the merged set
    pm = PrettyMIDI(io.BytesIO(p["midi"]))
    assert [i.name for i in pm.instruments] == ["Piano", "Drums"]
    assert sum(len(i.notes) for i in pm.instruments) == 2
    assert json.loads(text)["tracks"][1]["name"] == "Drums"

    # a legacy DETTATO_TRACKS payload keeps its shape (no 'midi' bytes, no 'sources')
    legacy = {"tracks": [a, b], "seconds": 8.0, "source": "audio", "info": {}}
    p2 = nodes.DettatoDedupeMIDI.execute(legacy, 50).args[0]
    assert "midi" not in p2 and "sources" not in p2
    assert p2["source"] == "audio" and p2["info"]["dedupe"]["notes_removed"] == 1


def test_quantize_payload_and_node(nodes, make_track):
    import io

    from pretty_midi import PrettyMIDI

    # 120 BPM + 1/16 = a 125 ms grid: 0.120 snaps to 0.125, its length is preserved
    a = make_track("Drums", 0, ((0.120, 0.200, 36, 60.0),), is_drum=True)
    b = make_track("Piano", 0, ((0.250, 0.400, 60, 90.0),))  # already on the grid
    payload = {
        "tracks": [a, b],
        "seconds": 8.0,
        "sources": ["audio_0"],
        "info": {"notes": {"transcribed": 2, "dropped": 0, "inserted": 0, "grouped": 2},
                 "tuning_cents": 0.0, "drums": "skipped: unit test"},
        "midi": b"MThd",
    }
    out = nodes.DettatoQuantizeMIDI.execute(payload, 120.0, "1/16")
    p, text = out.args
    d = np.asarray(p["tracks"][0].notes, np.float64).reshape(-1, 4)
    assert d[0, 0] == pytest.approx(0.125) and d[0, 1] == pytest.approx(0.205)
    pl = np.asarray(p["tracks"][1].notes, np.float64).reshape(-1, 4)
    assert pl[0, 0] == pytest.approx(0.250) and pl[0, 1] == pytest.approx(0.400)
    assert p["sources"] == ["audio_0"] and p["seconds"] == 8.0
    assert p["info"]["quantize"] == {"bpm": 120.0, "steps_per_beat": 4, "grid_ms": 125.0,
                                     "notes_moved": 1, "max_shift_ms": 5.0}
    assert p["info"]["notes"] == payload["info"]["notes"]  # the rest of the info is untouched
    pm = PrettyMIDI(io.BytesIO(p["midi"]))
    assert sum(len(i.notes) for i in pm.instruments) == 2
    assert json.loads(text)["quantize"] == p["info"]["quantize"]

    # a legacy DETTATO_TRACKS payload keeps its shape (no 'midi' bytes, no 'sources')
    legacy = {"tracks": [a], "seconds": 8.0, "source": "audio", "info": {}}
    p2 = nodes.DettatoQuantizeMIDI.execute(legacy, 120.0, "1/16").args[0]
    assert "midi" not in p2 and "sources" not in p2
    assert p2["source"] == "audio" and p2["info"]["quantize"]["notes_moved"] == 1


def test_dedup_sources(nodes, make_track, make_tracks):
    # same instrument in two sources: the one with the most notes is the home stem
    p1 = make_tracks(seconds=2.0, tracks=[make_track("Piano", 0, ((0.0, 1.0, 60, 100.0), (1.0, 2.0, 62, 90.0)))])
    p2 = make_tracks(seconds=3.0, tracks=[make_track("Piano", 0, ((0.0, 0.5, 60, 80.0),))])
    out, stats = dedup_sources([("audio_0", p1), ("audio_1", p2)])
    assert len(out) == 2
    assert [len(t.notes) for t in out[0][1]["tracks"]] == [2]
    assert out[1][1]["tracks"] == []  # the source loses the track, keeps its identity
    assert stats == {"kept": {"Piano": "audio_0"}, "dropped": {"Piano": {"audio_1": 1}}}
    # uncontested names pass through untouched
    p3 = make_tracks(seconds=2.0, tracks=[make_track("Bass", 33, ((0.0, 0.5, 33, 100.0),))])
    out2, stats2 = dedup_sources([("audio_0", p1), ("audio_1", p3)])
    assert stats2 == {"kept": {}, "dropped": {}}
    assert sum(len(t.notes) for _, p in out2 for t in p["tracks"]) == 3
    # the winner is by note count, not by slot order
    out3, stats3 = dedup_sources([("audio_0", p2), ("audio_1", p1)])
    assert stats3["kept"] == {"Piano": "audio_1"}


def test_transcribe_dedup_toggle(nodes, models_dir, make_audio):
    """The toggle on the real node path: off = today's behavior, on = redundant tracks gone."""
    from dettanode.dettato.transcriber.model import LaneNet

    torch.manual_seed(0)
    transcriber_cfg = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    _write_stage(
        models_dir / "dedup-pack", "dettato",
        {"model": transcriber_cfg, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4,
                                                  "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        LaneNet(**transcriber_cfg).state_dict(),
    )
    wrapper = nodes.load_dettato("dedup-pack", "fp32", "cpu")
    a1 = make_audio(t=1.0, rate=44100, freq=440.0)
    a2 = make_audio(t=1.0, rate=44100, freq=220.0)
    p_off, _ = nodes.DettatoTranscribe.execute(wrapper, {"audio_0": a1, "audio_1": a2}, 1).args
    p_on, text = nodes.DettatoTranscribe.execute(wrapper, {"audio_0": a1, "audio_1": a2}, 1, True).args
    assert "dedup" not in p_off["info"]
    n_off = sum(len(t.notes) for t in p_off["tracks"])
    n_on = sum(len(t.notes) for t in p_on["tracks"])
    stats = p_on["info"]["dedup"]
    assert json.loads(text)["dedup"] == stats
    names_off = [t.name for t in p_off["tracks"]]
    names_on = [t.name for t in p_on["tracks"]]
    # kept/dropped agree, and the winner is never one of the losers of its own instrument
    assert set(stats["kept"]) == set(stats["dropped"])
    for name, lost in stats["dropped"].items():
        assert stats["kept"][name] not in lost
    # after dedup each instrument has exactly one track, under its plain name
    seen = []
    for tname in names_off:
        base = re.sub(r" \([^()]*\)$", "", tname)
        if base not in seen:
            seen.append(base)
    assert names_on == seen
    dropped_total = sum(n for lost in stats["dropped"].values() for n in lost.values())
    assert n_off - n_on == dropped_total
    if not stats["dropped"]:
        assert n_off == n_on
    # the re-rendered .mid matches the shrunken track set
    import io
    from pretty_midi import PrettyMIDI
    pm = PrettyMIDI(io.BytesIO(p_on["midi"]))
    assert sum(len(i.notes) for i in pm.instruments) == n_on
    assert [i.name for i in pm.instruments] == names_on


def test_transcribe_dedup_no_cross_stem_duplicates(nodes, models_dir, make_audio):
    """Release invariant: with dedup on, the merge never sees two stems contributing the same
    instrument, so the merged payload contains no merge-renamed 'Name (audio_N)' tracks and no
    two tracks sharing a base name."""
    from dettanode.dettato.transcriber.model import LaneNet

    torch.manual_seed(0)
    transcriber_cfg = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    _write_stage(
        models_dir / "dedup-inv", "dettato",
        {"model": transcriber_cfg, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4,
                                                  "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        LaneNet(**transcriber_cfg).state_dict(),
    )
    wrapper = nodes.load_dettato("dedup-inv", "fp32", "cpu")
    audios = {f"audio_{i}": make_audio(t=1.0, rate=44100, freq=f)
              for i, f in enumerate((440.0, 220.0, 330.0))}
    p, _ = nodes.DettatoTranscribe.execute(wrapper, audios, 1, True).args
    names = [t.name for t in p["tracks"]]
    suffixed = [n for n in names if re.search(r" \([^()]*\)$", n)]
    assert not suffixed, f"merge renamed a cross-stem duplicate: {suffixed}"
    bases = {}
    for n in names:
        bases.setdefault(base_track_name(n), []).append(n)
    assert all(len(v) == 1 for v in bases.values()), bases


def test_quantize_tracks(make_track):
    # 120 BPM + 4 steps/beat = a 125 ms grid; lengths are preserved
    t = make_track("Piano", 0, ((0.120, 0.200, 60, 100.0),
                                (0.130, 0.330, 64, 90.0),
                                (0.250, 0.400, 62, 80.0)))
    out, stats = quantize_tracks([t], 120.0, 4)
    n = out[0].notes
    assert n[0, 0] == pytest.approx(0.125) and n[0, 1] == pytest.approx(0.205)
    assert n[1, 0] == pytest.approx(0.125)  # 0.130 rounds down to the same line
    assert n[2, 0] == pytest.approx(0.250)  # already on the grid, untouched
    assert stats["grid_ms"] == 125.0 and stats["notes_moved"] == 2
    assert stats["max_shift_ms"] == pytest.approx(5.0)
    # finer grid: 1/32 at 120 BPM = 62.5 ms
    out2, stats2 = quantize_tracks([t], 120.0, 8)
    assert stats2["grid_ms"] == 62.5
    assert out2[0].notes[0, 0] == pytest.approx(0.125)
    # lengths survive the shift exactly (a 5 ms note shifted 55 ms back stays 5 ms)
    t2 = make_track("Drums", 0, ((0.180, 0.185, 36, 100.0),), is_drum=True)
    out3, _ = quantize_tracks([t2], 120.0, 4)
    assert out3[0].notes[0, 0] == pytest.approx(0.125)
    assert out3[0].notes[0, 1] - out3[0].notes[0, 0] == pytest.approx(0.005)
    # a 1 ms blip is stretched to the 2 ms minimum, so nothing is zero-length
    t3 = make_track("Drums", 0, ((0.180, 0.181, 36, 100.0),), is_drum=True)
    out4, _ = quantize_tracks([t3], 120.0, 4)
    assert out4[0].notes[0, 0] == pytest.approx(0.125)
    assert out4[0].notes[0, 1] == pytest.approx(0.127)


def test_quantize_midi_node(nodes, models_dir, make_audio):
    """the quantize node: every onset lands on the grid, lengths stay positive, .mid re-rendered."""
    import io
    from pretty_midi import PrettyMIDI
    from dettanode.dettato.transcriber.model import LaneNet

    torch.manual_seed(0)
    transcriber_cfg = {
        "c": 368, "depth": 2, "exp": 2, "kt": 3, "dilations": [1, 2], "mid": 16, "t_stride": 2,
        "attn_at": [], "attn_axes": ["pitch", "time"], "attn_heads": 2,
        "gru_at": [], "gru_hidden": 8, "gru_bf16": True,
        "heads": ["onset", "frame", "offset", "onset2", "frame2", "offset2"],
        "harmonics": [1], "extra_nffts": [], "n_fft": 256, "refine": 1, "n_lanes": 88,
        "pmix": [-2, -1, 1, 2],
    }
    _write_stage(
        models_dir / "quant-pack", "dettato",
        {"model": transcriber_cfg, "thresholds": {"on_th": 0.25, "off_th": 0.3, "fr_th": 0.4,
                                                  "fr_low_frames": 2, "on_nms": 3, "v2_th": 0.3}},
        LaneNet(**transcriber_cfg).state_dict(),
    )
    wrapper = nodes.load_dettato("quant-pack", "fp32", "cpu")
    audio = make_audio(t=1.0, rate=48000, freq=440.0)
    p_off, _ = nodes.DettatoTranscribe.execute(wrapper, {"audio_0": audio}, 1).args
    assert "quantize" not in p_off["info"]  # the transcribe keeps the performance timing
    p_on, text = nodes.DettatoQuantizeMIDI.execute(p_off, 120.0, "1/16").args
    q = p_on["info"]["quantize"]
    assert q["bpm"] == 120.0 and q["steps_per_beat"] == 4 and q["grid_ms"] == 125.0
    assert json.loads(text)["quantize"] == q
    grid = q["grid_ms"] / 1000.0
    n_notes = 0
    for tr in p_on["tracks"]:
        for on, off in np.asarray(tr.notes, np.float64).reshape(-1, 4)[:, :2]:
            assert min(on % grid, grid - on % grid) < 1e-4, f"onset {on} off the {grid} s grid"
            assert off - on >= 0.002 - 1e-9
            n_notes += 1
    pm = PrettyMIDI(io.BytesIO(p_on["midi"]))
    assert sum(len(i.notes) for i in pm.instruments) == n_notes


def test_midi_to_text(nodes, make_track, make_tracks):
    a = make_tracks(seconds=4.0,
                    tracks=[make_track("Piano", 0, ((0.0, 0.25, 60, 90.0),)),
                            make_track("Drums", 0, ((0.1, 0.2, 36, 120.0),), is_drum=True)])
    payload = nodes.make_midi_payload([("audio", a)])
    data = json.loads(payload["midi_text"])
    assert data["seconds"] == 4.0
    assert [tr["name"] for tr in data["tracks"]] == ["Piano", "Drums"]
    assert data["tracks"][0]["program"] == 0 and data["tracks"][0]["is_drum"] is False
    assert data["tracks"][1]["program"] is None and data["tracks"][1]["is_drum"] is True
    assert data["tracks"][0]["notes"] == [[0.0, 0.25, 60, 90.0]]
    # the node outputs the payload's text
    text = nodes.DettatoMIDIToText.execute(payload).args[0]
    assert text == payload["midi_text"]
    # a payload without the text (legacy shape) gets it computed on the fly
    legacy = nodes.make_tracks_payload(a["tracks"], 4.0, "audio", {})
    assert "midi_text" not in legacy
    assert nodes.DettatoMIDIToText.execute(legacy).args[0] == payload["midi_text"]
