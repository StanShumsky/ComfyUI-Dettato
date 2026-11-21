"""Dettato tracks -> the native two-voice ABC dialect that YuE2 and SheetSage2 use.

YuE2's Generate Music node takes the score as plain text (the ComfyUI side never parses it),
so this module must emit exactly the bounded dialect described in the YuE2 skill: the fixed
eight-line header, Vocal + Ins monophonic blocks of one to four bars, an L:1/16 or L:1/32
grid, full-measure Z rests, ties for held notes, and quoted native chord symbols in the
Vocal line only. The output is checkable with the reference validator in tests/abc_tools.py.

Grid convention: one quarter note = ``UB`` units (UB = 4 for L:1/16, 8 for L:1/32), so a 4/4
bar is always 16 or 32 units. A time in seconds is a unit index of ``t * bpm / 60 * UB``.
"""

from __future__ import annotations

import numpy as np

from .dettato.pipeline import Track

#: note/rest multipliers the native dialect accepts (in L: units), largest first.
ALLOWED_DUR = (48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)

NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = "CDEFGAB"

#: major and minor keys the validator accepts, by tonic pc (conventional spellings).
_MAJOR = {0: "C", 7: "G", 2: "D", 9: "A", 4: "E", 5: "F", 10: "Bb", 6: "F#", 1: "C#",
          8: "Ab", 3: "Eb", 11: "B"}
_MINOR = {9: "Am", 4: "Em", 11: "Bm", 5: "Fm", 0: "Cm", 7: "Gm", 2: "Dm", 8: "Abm",
          3: "Ebm", 10: "Bbm", 6: "F#m", 1: "C#m"}

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.82, 3.45, 2.4, 2.94, 2.31, 3.06, 3.33])

#: chord quality -> pc offsets from the root; a safe core of the native chord vocabulary.
_QUALITIES = {
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11), "m7b5": (0, 3, 6, 10), "sus4": (0, 5, 7), "dim": (0, 3, 6),
}
_ROOTS = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A",
          10: "Bb", 11: "B"}


def _track_get(track, key, default=None):
    if isinstance(track, dict):
        return track.get(key, default)
    return getattr(track, key, default)


def _notes_of(track) -> np.ndarray:
    """(N, 4) onset s, offset s, pitch, velocity - sorted by onset; empty (0, 4) when none."""
    arr = np.asarray(_track_get(track, "notes"), np.float64)
    if arr.size == 0:
        return np.zeros((0, 4))
    arr = arr.reshape(-1, 4)
    return arr[np.argsort(arr[:, 0])]


def split_units(n: int) -> list[int]:
    """n grid units as a greedy list of allowed multipliers (exact: 1 is allowed)."""
    out = []
    while n > 0:
        for d in ALLOWED_DUR:
            if d <= n:
                out.append(d)
                n -= d
                break
    return out


# --------------------------------------------------------------------- key and tempo

def find_key(pitches: np.ndarray, weights: np.ndarray | None = None) -> str:
    """Krumhansl-Schmuckler key of a pitch collection -> a standard major/minor key name."""
    if pitches.size == 0:
        return "C"
    w = np.ones_like(pitches, float) if weights is None else weights.astype(float)
    profile = np.bincount(np.mod(np.round(pitches).astype(int), 12), weights=w, minlength=12)
    if profile.sum() <= 0:
        return "C"
    profile = profile / profile.sum()
    best, best_name = -2.0, "C"
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        for ks, table in ((_KS_MAJOR, _MAJOR), (_KS_MINOR, _MINOR)):
            score = float(np.corrcoef(rotated, ks)[0, 1]) if np.std(rotated) > 0 else -2.0
            if score > best:
                best, best_name = score, table[tonic]
    return best_name


def estimate_bpm(onsets: np.ndarray, weights: np.ndarray | None = None) -> int:
    """Grid-fit onset autocorrelation -> an integer quarter-note BPM (fallback 120)."""
    if onsets.size == 0:
        return 120
    onsets = np.sort(onsets)
    w = np.ones_like(onsets, float) if weights is None else weights.astype(float)
    span = float(onsets.max() - onsets.min())
    if span < 4.0:
        return 120
    best, best_bpm = -1.0, 120
    for bpm in range(50, 201):
        beat = 60.0 / bpm
        if span / beat < 4:
            continue
        d = np.remainder(onsets, beat)
        dist = np.minimum(d, beat - d)
        score = float(np.sum(w * np.exp(-(dist / 0.045) ** 2))) / np.sqrt(w.size)
        if 60 <= bpm <= 180:
            if score > best:
                best, best_bpm = score, bpm
        elif score > best + 0.35:
            best, best_bpm = score, bpm
    return int(best_bpm)


# --------------------------------------------------------------------- spelling

def _key_letter_pcs(key: str) -> dict[str, int]:
    """The seven diatonic letter pc's of a standard key, starting from its tonic."""
    minor = key.endswith("m")
    tonic = key[:-1] if minor else key
    base_letter = tonic[0]
    base_pc = NATURAL[base_letter] + (1 if "#" in tonic else -1 if "b" in tonic else 0)
    steps = (2, 1, 2, 2, 1, 2, 2) if minor else (2, 2, 1, 2, 2, 2, 1)
    start = LETTERS.index(base_letter)
    letters = LETTERS[start:] + LETTERS[:start]
    pcs, acc = {}, base_pc
    for letter, step in zip(letters, steps):
        pcs[letter] = acc % 12
        acc = (acc + step) % 12
    return pcs


def _spell(midi: int, letter_pcs: dict[str, int]) -> str:
    """A MIDI pitch as ABC letter + octave marks + explicit accidental, spelled against the key."""
    pc = midi % 12
    letter = next((L for L, kpc in letter_pcs.items() if kpc == pc), None)
    if letter is None:
        # rank by distance to the NATURAL letter pitch: a note outside the key is spelled on
        # the letter a step or less away (e.g. C natural in C#m is "=C", not eleven-flat B)
        ranked = sorted(((abs((pc - NATURAL[L] + 6) % 12 - 6), pc - NATURAL[L], L)
                         for L in letter_pcs))
        letter = ranked[0][2]
    # the accidental is written against the natural letter pitch, not the key signature: an
    # explicit ^ / _ / = always overrides the signature, so the sounding pitch is exact
    delta = pc - NATURAL[letter]
    if delta > 0:
        acc_mark = "^" * delta
    elif delta < 0:
        acc_mark = "_" * -delta
    else:
        acc_mark = "=" if letter_pcs[letter] != pc else ""  # cancel a signature alteration
    # the octave mark is not clamped (the dialect allows any number of ' / ,), so even
    # extreme pitches stay inside the dialect's two-mark accidental limit
    base = 60 + NATURAL[letter]
    marks = int(round((midi - base) / 12))
    written = base + 12 * marks
    if abs(midi - written) > 2:  # safety net; standard keys never need it
        marks += -1 if midi < written else 1
        written = base + 12 * marks
    return acc_mark + letter + ("'" * marks if marks > 0 else "," * -marks)


# --------------------------------------------------------------------- melody reduction

def _to_units(notes: np.ndarray, sec_per_unit: float) -> list[tuple[int, int, int, int]]:
    """(N,4) notes -> (start_u, end_u, midi, row) on the unit grid; every note keeps at least one unit."""
    out = []
    for i in range(notes.shape[0]):
        on, off, pitch = notes[i, 0], notes[i, 1], notes[i, 2]
        u0 = int(np.ceil(on / sec_per_unit - 1e-9))
        u1 = int(np.floor(off / sec_per_unit + 1e-9))
        if u1 <= u0:
            u1 = u0 + 1
        out.append((max(0, u0), u1, int(round(pitch)), i))
    return out


def monophonic_line(notes: np.ndarray, sec_per_unit: float, total_units: int):
    """Notes -> held segments (u0, u1, midi, row): at any unit the highest sounding pitch
    sounds, so simultaneous attacks take the top note and a newly struck higher note replaces
    a held lower one (a held low note can never mask the melody). Consecutive units of the
    same sounding note merge into one segment, so ties in the output follow the real note,
    not the grid."""
    segs = [s for s in _to_units(notes, sec_per_unit) if s[1] > 0 and s[0] < total_units]
    if not segs:
        return []
    events: dict[int, list[int]] = {}
    for i, (u0, u1, midi, row) in enumerate(segs):
        events.setdefault(u0, []).append(i)
        events.setdefault(u1, []).append(i)
    active: list[tuple[int, int, int, int]] = []      # (onset_u, midi, row, end_u)
    out: list[tuple[int, int, int, int]] = []
    for u in range(0, total_units):
        for i in events.get(u, []):
            u0, u1, midi, row = segs[i]
            if u == u0:
                active.append((u, midi, row, u1))
            else:
                active = [c for c in active if c[3] != u]
        if not active:
            continue
        active.sort(key=lambda c: c[1], reverse=True)
        _onset_u, midi, row, _ = active[0]
        if out and out[-1][1] == u and out[-1][2] == midi and out[-1][3] == row:
            out[-1] = (out[-1][0], u + 1, midi, row)
        else:
            out.append((u, u + 1, midi, row))
    return out


# --------------------------------------------------------------------- chords (full mode)

def _harmony_columns(tracks, melody, ins, sec_per_unit: float, total_units: int) -> list[np.ndarray]:
    """For each harmony track: per even unit, the top-velocity sounding pitch class (-1: none)."""
    cols = []
    for t in tracks:
        if t is melody or t is ins or _track_get(t, "is_drum"):
            continue
        notes = _notes_of(t)
        if notes.shape[0] == 0:
            continue
        col = np.full(total_units // 2, -1, np.int64)
        a = 0
        for j in range(0, total_units, 2):
            t0, t1 = j * sec_per_unit, (j + 2) * sec_per_unit
            while a < notes.shape[0] and notes[a, 1] <= t0 + 1e-9:
                a += 1
            best_pc, best_v = -1, -1.0
            for k in range(a, notes.shape[0]):
                if notes[k, 0] >= t1 - 1e-9:
                    break
                if notes[k, 3] > best_v:
                    best_v, best_pc = notes[k, 3], int(round(notes[k, 2])) % 12
            col[j // 2] = best_pc
        cols.append(col)
    return cols


def _chord_events(mel_line, harm_cols, total_units: int) -> dict[int, str]:
    """Chord changes at half-beat resolution -> {unit: symbol}, with hysteresis so the
    current chord holds until a different chord is clearly supported."""
    mel_pc = np.full(total_units, -1, np.int64)
    for u0, u1, midi, _r in mel_line:
        mel_pc[u0:u1] = midi % 12
    ev: dict[int, str] = {}
    current = None
    for j in range(0, total_units, 2):
        sounding: dict[int, float] = {}
        pc = int(mel_pc[j])
        if pc >= 0:
            sounding[pc] = 0.6
        for col in harm_cols:
            pc = int(col[j // 2])
            if pc >= 0:
                sounding[pc] = sounding.get(pc, 0.0) + 1.0
        if not sounding:
            continue
        best, best_sym, best_score = None, None, -1e9
        for root in range(12):
            for qual, offs in _QUALITIES.items():
                pcs = {(root + o) % 12 for o in offs}
                score = sum(sounding.get(p, 0.0) for p in pcs) - 0.15 * sum(1.0 for p in pcs if p not in sounding)
                if score > best_score:
                    best_score, best, best_sym = score, root, qual
        if current is not None:
            cur_score = sum(sounding.get(p, 0.0)
                            for p in {(current[0] + o) % 12 for o in _QUALITIES[current[1]]})
            if (best, best_sym) == current or best_score <= cur_score + 0.5:
                continue
        current = (best, best_sym)
        ev[j] = _ROOTS[best] + best_sym
    return ev


# --------------------------------------------------------------------- line assembly

def _voice_bar(runs, bar0: int, units_per_bar: int, chords: dict[int, str] | None, letter_pcs) -> str:
    """One voice, one bar [a, b): rests in gaps, held notes split at bar edges and chord
    changes, ties marking continuations (also across barlines), quoted chords attached at
    their unit even when the voice is resting or the note is tied."""
    a, b = bar0 * units_per_bar, (bar0 + 1) * units_per_bar
    chords = chords or {}
    out: list[str] = []
    pos = a

    def emit_rest(until: int):
        nonlocal pos
        for d in split_units(until - pos):
            sym = chords.get(pos)
            out.append((f'"{sym}"' if sym else "") + "z" + str(d))
            pos += d

    def emit_note(midi: int, u1_full: int, end: int):
        nonlocal pos
        name = _spell(midi, letter_pcs)
        cuts = sorted(u for u in chords if pos < u < end)
        stops = [pos] + cuts + [end]
        for s_, e_ in zip(stops, stops[1:]):
            sym = chords.get(s_)
            pieces = split_units(e_ - s_)
            for k, d in enumerate(pieces):
                tied = (s_ + d) < u1_full
                out.append((f'"{sym}"' if sym else "") + name + str(d) + ("-" if tied else ""))
                pos += d
                sym = None

    for (u0, u1, midi, _r) in runs:
        if u0 >= b:  # runs are sorted; anything from the bar edge on belongs to later bars
            break
        s, e = max(u0, a), min(u1, b)
        if s > pos:
            emit_rest(s)
        if e > pos:
            emit_note(midi, u1, e)
    if pos < b:
        emit_rest(b)
    return "".join(out)


# --------------------------------------------------------------------- the serializer

def tracks_to_abc(
    tracks,
    seconds: float,
    mode: str = "melody",
    melody_track: str = "",
    ins_track: str = "",
    key: str = "auto",
    tempo: int = 0,
    meter: str = "4/4",
    grid: str = "1/16",
    octave: str = "auto",
    out_meta: dict | None = None,
) -> str:
    """Dettato tracks -> a native two-voice ABC score for YuE2.

    ``mode`` 'melody' omits chord symbols (the recommended cover input); 'full' derives them
    from the non-melody tracks. ``melody_track`` empty picks the most lead-like pitched track
    (a voice/vocal track wins, a bass is never picked). ``ins_track`` empty picks the busiest
    remaining line above G3 that is not a bass; 'none' leaves the Ins voice as rests. ``key``
    'auto' runs key-finding on the melody; ``tempo`` 0 estimates the BPM from the onsets (drum
    hits included when present). ``octave`` 'auto'
    moves each line by whole octaves into the C4..A#5 window YuE2 sings best in (the native
    cover scores live there); an integer -2..2 transposes both lines by that many octaves.
    """
    tracks = list(tracks)
    if not tracks:
        raise ValueError("no tracks to convert")
    ub = 8 if grid == "1/32" else 4           # units per quarter note
    n_beats, d_beats = (int(x) for x in meter.split("/"))
    units_per_bar = 4 * n_beats * ub / d_beats
    if abs(units_per_bar - round(units_per_bar)) > 1e-9 or not 4 <= round(units_per_bar) <= 28:
        raise ValueError(f"unsupported meter {meter!r}; use 4/4, 3/4, 6/8 or 7/8")
    units_per_bar = int(round(units_per_bar))
    if tempo > 0:
        bpm = int(tempo)
    else:
        bpm = 120  # refined below once the melody is known

    # -- pick the melody track (explicit name, else the most lead-like pitched track)
    want = (melody_track or "").strip().lower()
    mel = next((t for t in tracks if not _track_get(t, "is_drum")
                and _track_get(t, "name", "").lower() == want), None)
    if mel is None and not want:
        best_score, mel = -1e9, None
        for t in tracks:
            if _track_get(t, "is_drum"):
                continue
            notes = _notes_of(t)
            if notes.shape[0] < 8:
                continue
            name = _track_get(t, "name", "").lower()
            score = float(np.median(notes[:, 2])) / 100.0
            # 'voice' first: that is the grouper's name for the vocal family (GM 52-54), the
            # track a cover is about; 'vocal' kept for names from other sources
            for kw, bonus in (("voice", 3.0), ("vocal", 3.0), ("lead", 2.0), ("soprano", 2.0),
                              ("melody", 2.0), ("flute", 1.0), ("violin", 1.0), ("sax", 1.0),
                              ("trumpet", 1.0)):
                if kw in name:
                    score += bonus
                    break
            if "bass" in name:
                score -= 2.0
            if score > best_score:
                best_score, mel = score, t
    if mel is None:
        raise ValueError(f"no pitched melody track to convert (wanted {want!r})")
    mel_notes = _notes_of(mel)
    if mel_notes.shape[0] == 0:
        raise ValueError("the melody track has no notes")

    # -- optional second (instrumental) voice
    want_ins = (ins_track or "").strip().lower()
    ins = None
    if want_ins and want_ins != "none":
        ins = next((t for t in tracks if t is not mel and not _track_get(t, "is_drum")
                    and _track_get(t, "name", "").lower() == want_ins), None)
        if ins is None:
            raise ValueError(f"ins track {ins_track!r} not found")
    elif not want_ins:  # empty: auto - the busiest remaining line above G3 that is not a bass
        cands = []
        for t in tracks:
            if t is mel or _track_get(t, "is_drum"):
                continue
            notes = _notes_of(t)
            if notes.shape[0] == 0:
                continue
            if "bass" in _track_get(t, "name", "").lower():
                continue
            med = float(np.median(notes[:, 2]))
            if med < 55.0:  # G3: below it a line is more bass than theme
                continue
            cands.append((notes.shape[0] / max(1e-6, seconds), med, t))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
            ins = cands[0][2]
    ins_notes = _notes_of(ins) if ins is not None else None

    # -- tempo (onset grid fit; drum onsets weigh double)
    if tempo <= 0:
        on = list(mel_notes[:, 0])
        wts = [1.0] * len(on)
        for t in tracks:
            if _track_get(t, "is_drum"):
                d = _notes_of(t)
                if d.shape[0]:
                    on += list(d[:, 0])
                    wts += [2.0] * d.shape[0]
        bpm = estimate_bpm(np.asarray(on), np.asarray(wts))

    sec_per_beat = 60.0 / bpm
    sec_per_unit = sec_per_beat / ub
    total_units = max(units_per_bar, int(np.ceil(seconds / sec_per_unit)))
    n_bars = (total_units + units_per_bar - 1) // units_per_bar
    total_units = n_bars * units_per_bar

    if key == "auto":
        w = np.clip(mel_notes[:, 1] - mel_notes[:, 0], 0.02, None) * mel_notes[:, 3]
        key = find_key(mel_notes[:, 2], w)
    letter_pcs = _key_letter_pcs(key)

    mel_line = monophonic_line(mel_notes, sec_per_unit, total_units)
    ins_line = monophonic_line(ins_notes, sec_per_unit, total_units) if ins_notes is not None else []

    # -- register: YuE2 sings the octave it is given, and the native cover scores live around
    # C4..A#5 (see the YuE2 example melody), so move a line by whole octaves into that window.
    # Octave shifts are pitch-class-neutral: the key, chords and rhythm are untouched.
    def _shift(line: list[tuple[int, int, int, int]], amount: int) -> list[tuple[int, int, int, int]]:
        return [(u0, u1, midi + 12 * amount, r) for u0, u1, midi, r in line]

    oct_req = str(octave).strip().lower()
    if oct_req == "auto":
        def _auto(line) -> int:
            if not line:
                return 0
            s, med = 0, float(np.median([seg[2] for seg in line]))
            while med < 60 and s < 2:
                med, s = med + 12, s + 1
            while med > 94 and s > -2:
                med, s = med - 12, s - 1
            return s
        mel_shift, ins_shift = _auto(mel_line), _auto(ins_line)
    else:
        try:
            mel_shift = ins_shift = int(oct_req)
        except ValueError:
            raise ValueError(f"unsupported octave {octave!r}; use 'auto' or an integer -2..2")
        if not -2 <= mel_shift <= 2:
            raise ValueError(f"octave {octave!r} out of range; use 'auto' or an integer -2..2")
    mel_line = _shift(mel_line, mel_shift)
    ins_line = _shift(ins_line, ins_shift)

    chords: dict[int, str] = {}
    if mode == "full":
        chords = _chord_events(mel_line, _harmony_columns(tracks, mel, ins, sec_per_unit, total_units), total_units)

    # -- header (the validator requires exactly this eight-line shape)
    lines = [
        "X:1",
        "T:",
        f"M:{meter}",
        f"L:{grid}",
        f"Q:1/4={bpm}",
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        f"K:{key}",
    ]

    # -- body: groups of up to four bars, a Vocal block then an Ins block
    for g in range(0, n_bars, 4):
        b1 = min(g + 4, n_bars)
        vocal = [_voice_bar(mel_line, bar, units_per_bar, chords if mode == "full" else None, letter_pcs)
                 for bar in range(g, b1)]
        if ins_line:
            ins_txt = [_voice_bar(ins_line, bar, units_per_bar, None, letter_pcs)
                       for bar in range(g, b1)]
        else:
            n = b1 - g
            ins_txt = ["Z" if n == 1 else f"Z{n}"]
        lines.append("V: Vocal")
        lines.append("|".join(vocal) + "|")
        lines.append("V: Ins")
        lines.append("|".join(ins_txt) + "|")
    if out_meta is not None:
        out_meta.update({
            "key": key,
            "bpm": bpm,
            "meter": meter,
            "grid": grid,
            "bars": n_bars,
            "melody_track": _track_get(mel, "name"),
            "melody_notes": int(len(mel_line)),
            "ins_track": _track_get(ins, "name") if ins is not None else None,
            "ins_notes": int(len(ins_line)),
            "chord_changes": int(len(chords)),
            "octave": {"melody": int(mel_shift), "ins": int(ins_shift)},
        })
    return "\n".join(lines) + "\n"
