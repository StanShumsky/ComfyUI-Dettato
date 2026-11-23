"""Conversions between ComfyUI payloads and Dettato tracks, plus multi-source merging,
MIDI rendering and JSON writing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from .dettato.pipeline import Transcription, Track


def audio_to_mono(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO {waveform (B, C, N) or (C, N), sample_rate} -> mono float32 numpy and the rate."""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError(
            "The audio input is not a ComfyUI AUDIO payload (waveform + sample_rate). Connect a "
            "Load Audio node or another node's audio output to the slot."
        )
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    rate = audio.get("sample_rate")
    if rate is None:
        raise ValueError(
            "The connected audio has no sample rate: its file's audio stream does not report one "
            "(some video files do). Re-save the audio (e.g. as .wav) and load that file instead."
        )
    rate = int(rate)
    if rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {rate}.")
    x = waveform.detach().float()
    if x.dim() == 3:
        x = x[0] if x.shape[0] == 1 else x.mean(0)
    if x.dim() != 2:
        raise ValueError(f"Audio waveform must be (channels, samples) or (1, channels, samples), got {tuple(x.shape)}.")
    if x.shape[-1] == 0:
        raise ValueError("The audio is empty.")
    return x.mean(0).cpu().numpy().astype(np.float32), rate


def make_tracks_payload(tracks: list[Track], seconds: float, source: str, info: dict) -> dict:
    """The DETTATO_TRACKS payload shared by the nodes."""
    return {
        "tracks": tracks,
        "seconds": float(seconds),
        "source": source,
        "info": info,
    }


def build_transcription(payload: dict) -> Transcription:
    source = payload.get("source")
    if source is None:
        sources = [str(s) for s in (payload.get("sources") or ["audio"])]
        source = ", ".join(sources) if len(sources) > 1 else sources[0]
    return Transcription(
        source=source,
        seconds=payload.get("seconds", 0.0),
        tracks=payload["tracks"],
        info=dict(payload.get("info", {})),
    )


def summary_json(payload: dict) -> str:
    return json.dumps(build_transcription(payload).summary(), indent=2, sort_keys=True)


def score_text(tracks: list[Track], seconds: float) -> str:
    """The score as generic JSON text: any library that reads JSON can use it, and it can be
    saved with any text node. {seconds, tracks: [{name, program, is_drum,
    notes: [[onset s, offset s, pitch, velocity], ...]}]}; program is null on drum tracks
    (a .mid writer gives them GM program 37)."""
    return json.dumps({
        "seconds": float(seconds),
        "tracks": [{
            "name": tr.name,
            "program": None if tr.is_drum else int(tr.program),
            "is_drum": bool(tr.is_drum),
            "notes": [[float(on), float(off), int(pitch), float(vel)]
                      for on, off, pitch, vel in np.asarray(tr.notes, np.float64).reshape(-1, 4)],
        } for tr in tracks],
    })


def render_midi_bytes(tracks: list[Track], seconds: float) -> bytes:
    """The tracks as .mid bytes (pretty_midi, one GM track per instrument, 120 BPM, times in seconds)."""
    import io

    buf = io.BytesIO()
    Transcription("audio", float(seconds), list(tracks)).to_midi().write(buf)
    return buf.getvalue()


def write_midi(payload: dict, path: Path) -> None:
    """Write the transcription as .mid (one General MIDI track per instrument)."""
    Path(path).write_bytes(render_midi_bytes(payload["tracks"], payload.get("seconds", 0.0)))


def filter_min_velocity(tracks: list[Track], min_velocity: int) -> list[Track]:
    """Drop notes quieter than min_velocity; tracks that end up empty are removed."""
    out = []
    for track in tracks:
        notes = np.asarray(track.notes, np.float64).reshape(-1, 4)
        if notes.shape[0]:
            notes = notes[notes[:, 3] >= min_velocity]
        if notes.shape[0]:
            out.append(Track(track.name, track.program, notes.astype(np.float64), is_drum=track.is_drum))
    return out


# --------------------------------------------------------------------- multi-source


def _dedup_labels(labels: list[str]) -> list[str]:
    """Second and later occurrences of a label get ' 2', ' 3', ... (labels key the info dict)."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            out.append(f"{label} {seen[label]}")
        else:
            seen[label] = 1
            out.append(label)
    return out


def merge_tracks(sources: list[tuple[str, list[Track]]]) -> list[Track]:
    """Concatenate per-source track lists in source order, keeping names unique.

    The first occurrence of a name keeps it; a repeat from a later source is renamed
    'Name (label)', and if that is taken too, 'Name (label) 2', ' 3', ... This is what lets a
    DAW show, e.g., 'Piano' from the vocal stem and 'Piano (accompaniment)' from the rest.
    """
    used: set[str] = set()
    out: list[Track] = []
    for label, tracks in sources:
        for track in tracks:
            name = track.name
            if name in used:
                candidate, n = f"{name} ({label})", 2
                while candidate in used:
                    candidate = f"{name} ({label}) {n}"
                    n += 1
                name = candidate
            used.add(name)
            out.append(Track(name, track.program, track.notes, is_drum=track.is_drum))
    return out


def _agreed_tuning(values: list) -> float | None:
    """The common tuning estimate of the sources, or None when they disagree or have none."""
    vals = [v for v in values if v is not None]
    if vals and all(abs(v - vals[0]) < 1e-6 for v in vals[1:]):
        return vals[0]
    return None


def merge_infos(sources: list[tuple[str, dict]]) -> dict:
    """The per-source info dicts into one.

    A single 'audio' source keeps the per-source shape exactly (one plain input is byte-identical
    to the pre-merge node). Otherwise: merged note totals on top, the common tuning estimate (or
    None), the per-source drum decisions keyed by label, and the per-source dicts under 'sources'.
    """
    if len(sources) == 1 and sources[0][0] == "audio":
        return dict(sources[0][1])
    return {
        "notes": {k: sum(int(i.get("notes", {}).get(k, 0) or 0) for _, i in sources)
                  for k in ("transcribed", "dropped", "inserted", "grouped")},
        "tuning_cents": _agreed_tuning([i.get("tuning_cents") for _, i in sources]),
        "drums": {label: i.get("drums") for label, i in sources},
        "sources": {label: dict(i) for label, i in sources},
    }


def make_midi_payload(results: list[tuple[str, dict]]) -> dict:
    """[(label, per-source payload)] -> the DETTATO_MIDI payload: the merged track set, the
    longest length, the merged info, and the .mid rendered once (so writers just persist bytes)."""
    if not results:
        raise ValueError("Nothing to merge: no audio sources.")
    labels = _dedup_labels([label for label, _ in results])
    tracks = merge_tracks([(lab, payload["tracks"]) for lab, (_, payload) in zip(labels, results)])
    seconds = max(float(payload.get("seconds", 0.0)) for _, payload in results)
    info = merge_infos([(lab, payload.get("info", {})) for lab, (_, payload) in zip(labels, results)])
    return {
        "tracks": tracks,
        "seconds": seconds,
        "sources": labels,
        "info": info,
        "midi": render_midi_bytes(tracks, seconds),
        "midi_text": score_text(tracks, seconds),
    }


# --------------------------------------------------------------------- dedupe

_MERGE_SUFFIX = re.compile(r"\s+\([^()]*\)(\s+\d+)?$")


def base_track_name(name: str) -> str:
    """The instrument name without its merge suffixes: 'Drums (audio_1) 2' -> 'Drums'.

    The grouper's own vocabulary (73 names) contains no parentheses, so a trailing
    ' (label)' / ' (label) N' is always the merge renaming and safe to strip.
    """
    base = name
    while True:
        m = _MERGE_SUFFIX.search(base)
        if not m:
            return base
        base = base[:m.start()]


def _dedupe_notes(notes: np.ndarray, window_s: float) -> tuple[np.ndarray, int]:
    """Drop onset duplicates from one (N, 4) pool: a note is a duplicate when the gap to the
    last kept note of the same pitch is within window_s. The louder note of a duplicate pair
    wins - a quiet ghost can never stand in for a loud real one, and a loud real one replaces
    a quiet ghost that preceded it - equal velocities keep the earlier onset. Returns
    (kept sorted by onset, removed count)."""
    order = np.argsort(notes[:, 0], kind="stable")
    kept: list[int] = []
    last_idx: dict[int, int] = {}
    last_onset: dict[int, float] = {}
    last_vel: dict[int, float] = {}
    for i in order:
        on, vel, pitch = float(notes[i, 0]), float(notes[i, 3]), int(notes[i, 2])
        if pitch in last_onset and on - last_onset[pitch] <= window_s:
            if vel > last_vel[pitch]:
                kept[last_idx[pitch]] = i
                last_idx[pitch] = len(kept) - 1
                last_onset[pitch] = on
                last_vel[pitch] = vel
            # else: drop this quieter note (a duplicate of the louder kept one)
        else:
            last_idx[pitch] = len(kept)
            kept.append(i)
            last_onset[pitch] = on
            last_vel[pitch] = vel
    kept_arr = notes[np.asarray(kept, dtype=int)]
    return kept_arr[np.argsort(kept_arr[:, 0], kind="stable")], len(notes) - len(kept)


def dedupe_tracks(tracks: list[Track], window_ms: int) -> tuple[list[Track], dict]:
    """Consolidate same-named tracks and drop the onset-duplicate notes of a stem separation.

    Tracks whose names differ only by the merge suffix ('Drums', 'Drums (audio_1)',
    'Drums (audio_2) 3') are one instrument split across stems: their notes pool into a
    single track (named after the first member, in MIDI order) and the pool is scanned in
    onset order - a note is a duplicate when the gap to the last kept note of the same pitch
    is within window_ms, the louder note of a duplicate pair winning. 0 pools the tracks
    without dropping anything. Returns the new track list and the stats for info_json.
    """
    groups: dict[str, list[Track]] = {}
    for track in tracks:
        groups.setdefault(base_track_name(track.name), []).append(track)
    out: list[Track] = []
    merged: dict[str, int] = {}
    removed = 0
    for base, members in groups.items():
        first = members[0]
        pool = np.concatenate(
            [np.asarray(t.notes, np.float64).reshape(-1, 4) for t in members], axis=0)
        if window_ms > 0:
            kept, n = _dedupe_notes(pool, window_ms / 1000.0)
        else:
            kept, n = pool, 0
        removed += n
        if len(members) > 1:
            merged[base] = len(members)
        if kept.shape[0]:
            out.append(Track(first.name, first.program, kept, is_drum=first.is_drum))
    return out, {
        "window_ms": int(window_ms),
        "tracks_before": len(tracks),
        "tracks_after": len(out),
        "notes_removed": int(removed),
        "merged": merged,
    }


def dedup_sources(results: list[tuple[str, dict]]) -> tuple[list[tuple[str, dict]], dict]:
    """Pre-merge dedupe for separated stems: when the same instrument was transcribed from
    several sources (one instrument split across the stems), keep only the track with the most
    notes - the stem that carried most of it is its home - and drop the redundant tracks from
    the other sources. No onset analysis: the decision is the per-source note count. A source
    that loses all of its tracks stays in the results, so its info survives in the merged
    summary. Returns the (possibly shrunken) results and the stats for the merged info:
    {'kept': {instrument: home label}, 'dropped': {instrument: {label: note count}}}."""
    owner: dict[str, tuple[int, str]] = {}
    for label, payload in results:
        for track in payload["tracks"]:
            n = len(np.asarray(track.notes, np.float64).reshape(-1, 4))
            cur = owner.get(track.name)
            if cur is None or n > cur[0]:
                owner[track.name] = (n, label)
    out: list[tuple[str, dict]] = []
    dropped: dict[str, dict[str, int]] = {}
    for label, payload in results:
        kept_tracks = []
        for track in payload["tracks"]:
            n = len(np.asarray(track.notes, np.float64).reshape(-1, 4))
            if owner[track.name][1] == label:
                kept_tracks.append(track)
            else:
                dropped.setdefault(track.name, {})[label] = n
        out.append((label, {**payload, "tracks": kept_tracks}))
    return out, {"kept": {name: owner[name][1] for name in dropped}, "dropped": dropped}


def dedupe_payload(payload: dict, window_ms: int) -> dict:
    """A DETTATO_MIDI / DETTATO_TRACKS payload -> the same shape with the deduplicated
    tracks, the re-rendered .mid (merged payloads only) and the dedupe stats in the info."""
    tracks, stats = dedupe_tracks(payload["tracks"], window_ms)
    info = dict(payload.get("info", {}))
    info["dedupe"] = stats
    out = {
        "tracks": tracks,
        "seconds": float(payload.get("seconds", 0.0)),
        "info": info,
    }
    if "sources" in payload:
        out["sources"] = list(payload["sources"])
        out["midi"] = render_midi_bytes(tracks, out["seconds"])
        out["midi_text"] = score_text(tracks, out["seconds"])
    else:
        out["source"] = payload.get("source", "audio")
    return out


def quantize_tracks(tracks: list[Track], bpm: float, steps_per_beat: int) -> tuple[list[Track], dict]:
    """Snap every note onset to a musical grid; the note's length is preserved (the offset
    moves by the same amount as the onset). The grid step is (60 / bpm) / steps_per_beat
    seconds, so 120 BPM + 4 steps/beat is a 16th-note grid at 125 ms. Notes shorter than
    2 ms are stretched to that minimum, so nothing ends up zero-length. Returns the new
    track list and the stats for info_json."""
    grid = (60.0 / float(bpm)) / float(steps_per_beat)
    out = []
    moved = 0
    max_shift = 0.0
    for track in tracks:
        notes = np.asarray(track.notes, np.float64).reshape(-1, 4).copy()
        if notes.shape[0]:
            on = notes[:, 0]
            on_q = np.round(on / grid) * grid
            delta = on_q - on
            moved += int(np.sum(np.abs(delta) > 1e-4))
            max_shift = max(max_shift, float(np.max(np.abs(delta))))
            notes[:, 0] = on_q
            notes[:, 1] = np.maximum(notes[:, 1] + delta, on_q + 0.002)
        out.append(Track(track.name, track.program, notes, is_drum=track.is_drum))
    return out, {
        "bpm": float(bpm),
        "steps_per_beat": int(steps_per_beat),
        "grid_ms": round(grid * 1000.0, 3),
        "notes_moved": int(moved),
        "max_shift_ms": round(max_shift * 1000.0, 3),
    }


#: grid combo value -> notes per beat.
GRID_STEPS = {"1/4": 1, "1/8": 2, "1/16": 4, "1/32": 8}


def quantize_payload(payload: dict, bpm: float, grid: str) -> dict:
    """A DETTATO_MIDI / DETTATO_TRACKS payload -> the same shape with the onsets snapped to
    the tempo x grid, the re-rendered .mid (merged payloads only) and the quantize stats in
    the info."""
    steps = GRID_STEPS.get(grid, 4)
    tracks, stats = quantize_tracks(payload["tracks"], bpm, steps)
    info = dict(payload.get("info", {}))
    info["quantize"] = stats
    out = {
        "tracks": tracks,
        "seconds": float(payload.get("seconds", 0.0)),
        "info": info,
    }
    if "sources" in payload:
        out["sources"] = list(payload["sources"])
        out["midi"] = render_midi_bytes(tracks, out["seconds"])
        out["midi_text"] = score_text(tracks, out["seconds"])
    else:
        out["source"] = payload.get("source", "audio")
    return out
