"""Converter tests: tracks -> the native two-voice ABC dialect of YuE2/SheetSage2.

No weights, no network. The reference dialect validator (tests/abc_tools.py) is the oracle:
every score the converter emits must parse through it, and its parsed note events are what
the assertions check.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abc_tools  # noqa: E402  (the reference dialect validator, kept in tests/)
from dettanode.abc_score import (ALLOWED_DUR, estimate_bpm, find_key, split_units,
                                 tracks_to_abc)


def parse(abc: str):
    return abc_tools.parse_abc(abc)


def make_track(name, notes, is_drum=False, program=0):
    from dettanode.dettato.pipeline import Track
    return Track(name, program, np.array(notes, np.float64).reshape(-1, 4), is_drum=is_drum)


# --------------------------------------------------------------------- grid helpers


def test_split_units_is_exact_and_allowed():
    for n in list(range(1, 96)) + [100, 144, 200, 317]:
        parts = split_units(n)
        assert all(p in ALLOWED_DUR for p in parts)
        assert sum(parts) == n


def test_find_key_major_and_minor():
    assert find_key(np.array([60, 64, 67, 72, 76, 79, 84, 88, 91], float)) == "C"
    assert find_key(np.array([69, 72, 76, 81, 84, 88, 93, 96, 100], float)) == "Am"


def test_estimate_bpm_on_grid():
    assert estimate_bpm(np.arange(0.0, 8.0, 0.5)) == 120
    assert estimate_bpm(np.arange(0.0, 10.0, 60.0 / 90.0)) == 90
    assert estimate_bpm(np.array([], float)) == 120


# --------------------------------------------------------------------- the serializer


def test_melody_roundtrip_through_validator():
    notes = [(k * 0.5, k * 0.5 + 0.5, 60 + k, 100.0) for k in range(8)]
    abc = tracks_to_abc([make_track("Piano", notes)], 8.0, tempo=120, melody_track="Piano")
    score = parse(abc)
    assert score.bpm == 120
    got = [(int(o), p, int(d)) for o, p, d in score.voices["Vocal"].notes]
    assert got == [(k, 60 + k, 1) for k in range(8)]
    assert score.voices["Ins"].notes == []
    assert score.voices["Vocal"].chords == []


def test_tie_holds_a_note_across_the_barline():
    abc = tracks_to_abc([make_track("Piano", [(0.0, 3.0, 60, 100.0)])], 4.0, tempo=120,
                      melody_track="Piano")
    score = parse(abc)
    notes = score.voices["Vocal"].notes
    assert len(notes) == 1
    o, p, d = notes[0]
    assert (o, p, d) == (Fraction(0), 60, Fraction(6))


def test_full_mode_puts_chords_in_the_vocal_line_only():
    mel = [(0.0, 2.0, 72, 110.0), (2.0, 4.0, 69, 110.0),
           (4.0, 6.0, 74, 110.0), (6.0, 8.0, 77, 110.0)]
    acc = [
        (0.0, 0.5, 48, 90.0), (0.0, 0.5, 55, 90.0), (0.0, 0.5, 60, 90.0),
        (2.0, 2.5, 45, 90.0), (2.0, 2.5, 52, 90.0), (2.0, 2.5, 57, 90.0),
        (4.0, 4.5, 50, 90.0), (4.0, 4.5, 57, 90.0), (4.0, 4.5, 62, 90.0),
        (6.0, 6.5, 53, 90.0), (6.0, 6.5, 60, 90.0), (6.0, 6.5, 65, 90.0),
    ]
    tracks = [make_track("Vocal", mel), make_track("Piano", acc)]
    abc = tracks_to_abc(tracks, 8.0, mode="full", tempo=120, melody_track="Vocal",
                      ins_track="none")  # keep the harmony track out of the Ins voice here
    score = parse(abc)  # the validator fails if a chord leaks into the Ins voice
    chords = [c for _t, c in score.voices["Vocal"].chords]
    assert chords and chords[0] == "C"
    stripped = abc_tools.strip_chords(abc)  # chord removal must keep the melody intact
    assert abc_tools.compare(parse(abc), parse(stripped))["match"]


def test_ins_voice_carries_the_second_track():
    vocal = [(k * 0.5, k * 0.5 + 0.4, 76, 110.0) for k in range(4)]
    flute = [(k * 0.5, k * 0.5 + 0.4, 88, 100.0) for k in range(4)]
    tracks = [make_track("Flute", flute), make_track("Vocal", vocal)]
    abc = tracks_to_abc(tracks, 8.0, tempo=120, melody_track="Vocal", ins_track="Flute")
    score = parse(abc)
    assert len(score.voices["Vocal"].notes) == 4
    assert [p for _o, p, _d in score.voices["Ins"].notes] == [88] * 4


def test_meters_and_grid():
    notes = [(0.0, 0.5, 60, 100.0), (0.5, 1.0, 62, 100.0), (1.0, 1.5, 64, 100.0)]
    score = parse(tracks_to_abc([make_track("Piano", notes)], 3.0, tempo=120,
                            meter="3/4", melody_track="Piano"))
    assert [b[1] for b in score.voices["Vocal"].bars] == [Fraction(3, 1), Fraction(3, 1)]
    score68 = parse(tracks_to_abc(
        [make_track("Piano", notes)], 3.0, tempo=120, meter="6/8", grid="1/32",
        melody_track="Piano"))
    assert [b[1] for b in score68.voices["Vocal"].bars] == [Fraction(3, 1), Fraction(3, 1)]
    assert len(score68.voices["Vocal"].notes) == 3


def test_explicit_key_and_tempo_are_respected():
    notes = [(0.0, 1.0, 69, 100.0), (1.0, 2.0, 72, 100.0)]
    abc = tracks_to_abc([make_track("Piano", notes)], 2.0, tempo=90, key="E",
                      melody_track="Piano")
    score = parse(abc)
    assert score.bpm == 90
    assert "K:E" in abc.splitlines()


def test_auto_picks_the_lead_track_and_meta():
    lead = [(k * 0.5, k * 0.5 + 0.4, 74, 110.0) for k in range(10)]
    bass = [(k * 0.5, k * 0.5 + 0.4, 38, 100.0) for k in range(10)]
    meta = {}
    abc = tracks_to_abc([make_track("Bass Guitar", bass), make_track("Flute", lead)],
                        10.0, tempo=120, out_meta=meta)
    parse(abc)
    assert meta["melody_track"] == "Flute"  # a lead-like track beats a bass, whatever the pitch
    assert meta["ins_track"] is None
    assert meta["bpm"] == 120 and meta["meter"] == "4/4"


def test_top_pitch_wins_over_a_held_note():
    # regression: the reduction used to let a held note swallow every later attack, so a held
    # bass in the same track erased the whole melody
    notes = [(0.0, 4.0, 36, 100.0), (2.0, 2.25, 64, 127.0), (3.0, 3.25, 67, 127.0)]
    abc = tracks_to_abc([make_track("Guitar", notes)], 4.0, tempo=120, melody_track="Guitar",
                      ins_track="none", octave="0")
    score = parse(abc)
    pitches = [p for _o, p, _d in score.voices["Vocal"].notes]
    assert set(pitches) == {36, 64, 67}  # the melody sounds in its gaps; before the fix only 36
    assert pitches[1] == 64 and 67 in pitches[2:]


def test_simultaneous_attack_takes_the_top_note():
    notes = [(0.0, 0.5, 60, 100.0), (0.0, 0.5, 64, 100.0), (0.0, 0.5, 67, 100.0)]
    abc = tracks_to_abc([make_track("Piano", notes)], 2.0, tempo=120, melody_track="Piano",
                      ins_track="none", octave="0")
    score = parse(abc)
    assert [p for _o, p, _d in score.voices["Vocal"].notes] == [67]


def test_voice_track_wins_auto_pick():
    # the grouper names the vocal family 'voice'; auto must prefer it over higher strings/synths
    voice = [(k * 0.5, k * 0.5 + 0.4, 72, 110.0) for k in range(10)]
    violin = [(k * 0.5, k * 0.5 + 0.4, 84, 100.0) for k in range(10)]
    synth = [(k * 0.5, k * 0.5 + 0.4, 80, 100.0) for k in range(10)]
    meta = {}
    abc = tracks_to_abc([make_track("violin", violin), make_track("synth_lead", synth),
                        make_track("voice", voice)], 10.0, tempo=120, out_meta=meta)
    parse(abc)
    assert meta["melody_track"] == "voice"
    assert meta["ins_track"] == "violin"  # tied note density: the higher line wins


def test_ins_auto_picks_busiest_line_above_g3():
    voice = [(k * 0.5, k * 0.5 + 0.4, 72, 110.0) for k in range(8)]
    guitar = [(k * 0.25, k * 0.25 + 0.2, 76, 100.0) for k in range(16)]
    piano = [(k * 1.0, k * 1.0 + 0.5, 60, 100.0) for k in range(4)]
    bass = [(k * 0.5, k * 0.5 + 0.4, 40, 100.0) for k in range(16)]
    meta = {}
    abc = tracks_to_abc([make_track("voice", voice), make_track("piano", piano),
                        make_track("electric_guitar", guitar), make_track("double_bass", bass)],
                       8.0, tempo=120, out_meta=meta)
    parse(abc)
    assert meta["melody_track"] == "voice"
    assert meta["ins_track"] == "electric_guitar"  # busiest, above G3, not a bass


def test_octave_auto_moves_low_lines_into_the_window():
    guitar = [(k * 0.5, k * 0.5 + 0.4, 52, 100.0) for k in range(10)]  # around E3, below C4
    meta = {}
    abc = tracks_to_abc([make_track("electric_guitar", guitar)], 10.0, tempo=120,
                      ins_track="none", out_meta=meta)
    score = parse(abc)
    assert [p for _o, p, _d in score.voices["Vocal"].notes] == [64] * 10  # shifted up one octave
    assert meta["octave"] == {"melody": 1, "ins": 0}


def test_octave_auto_leaves_vocal_range_alone_and_explicit_applies():
    vocal = [(k * 0.5, k * 0.5 + 0.4, 72, 110.0) for k in range(8)]
    meta = {}
    tracks_to_abc([make_track("voice", vocal)], 8.0, tempo=120, ins_track="none", out_meta=meta)
    assert meta["octave"] == {"melody": 0, "ins": 0}
    score = parse(tracks_to_abc([make_track("voice", vocal)], 8.0, tempo=120,
                            ins_track="none", octave="-1"))
    assert [p for _o, p, _d in score.voices["Vocal"].notes] == [60] * 8


def test_errors():
    with pytest.raises(ValueError):
        tracks_to_abc([], 1.0)
    with pytest.raises(ValueError):
        tracks_to_abc([make_track("Drums", [(0.0, 0.1, 36, 1.0)], is_drum=True)], 1.0)
    with pytest.raises(ValueError):
        tracks_to_abc([make_track("Piano", [(0.0, 0.5, 60, 1.0)] * 10)], 1.0, ins_track="Organ")
    with pytest.raises(ValueError):
        tracks_to_abc([make_track("Piano", [(0.0, 0.5, 60, 1.0)] * 10)], 1.0, meter="5/6")


# --------------------------------------------------------------------- the nodes


def test_node_to_yue2_abc(nodes, make_tracks):
    conv = nodes.NODE_CLASS_MAPPINGS[next(c for c in nodes.NODE_CLASS_MAPPINGS
                                          if c.endswith("ToYuE2ABC"))]
    abc, summary = conv().convert(make_tracks(seconds=4.0), "melody", "Piano", "none",
                                  "auto", 120, "4/4", "1/16")
    assert parse(abc).bpm == 120
    meta = json.loads(summary)
    assert meta["melody_track"] == "Piano" and meta["bpm"] == 120


def test_node_save_abc_writes_numbered_files(nodes, output_dir, make_tracks):
    conv = nodes.NODE_CLASS_MAPPINGS[next(c for c in nodes.NODE_CLASS_MAPPINGS
                                          if c.endswith("ToYuE2ABC"))]
    saver = nodes.NODE_CLASS_MAPPINGS[next(c for c in nodes.NODE_CLASS_MAPPINGS
                                           if c.endswith("SaveABC"))]
    prefix = saver.INPUT_TYPES()["required"]["filename_prefix"][1]["default"]
    abc, _ = conv().convert(make_tracks(seconds=4.0), "melody", "Piano", "none",
                            "auto", 120, "4/4", "1/16")
    node = saver()
    for _ in range(2):
        node.save(abc, prefix)
    folder, stem = output_dir / Path(prefix).parent, Path(prefix).name
    files = sorted(p.name for p in folder.glob(stem + "_*.abc"))
    assert files == ["abc_00000.abc", "abc_00001.abc"]
