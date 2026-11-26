"""The whole system: audio file -> instrument-named MIDI tracks + a drum track.

    tuning (estimated first; the audio is resampled onto A440 when it sits clearly off it)
      -> transcriber (notes, two voices per key). One pass feeds both stages on short clips; long clips use bounded note-conditioned
         passes instead of retaining whole-song hidden/stem tensors
      -> evidence repairer (drops wrong notes, fixes onsets / ends, inserts missed notes, reading the transcriber's own evidence)
      -> grouper (assigns every note to an instrument track and names the track)
    drum model (drum-kit and orchestral percussion hits; only for sources recorded at >= 40 kHz)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import audio as A
from .evidence.features import MAX_S_1PASS
from .grouper import retune

MIN_END_S = 0.03            # every pitched note lasts at least 30 ms before grouping
STREAM_AFTER_S = MAX_S_1PASS  # above this length, extra transcriber passes avoid full-song hidden/stem tap buffers


def streaming_pitched(transcriber, repairer, grouper, y, tuning, batch=2, check=None, evidence_done=None):
    """Long-audio pitched stages with note-conditioned tap storage instead of full-song hidden/stem tensors."""
    from .evidence.features import evidence_from_maps_2pass

    maps, _ = transcriber.run(y, hop_frac=0.25, batch=batch, check=check)
    res, ev = evidence_from_maps_2pass(transcriber, y, maps, batch, check=check)
    rep, res, rinfo = repairer.from_evidence(res, ev, check=check, duration=len(y) / 16000)
    if evidence_done is not None:
        evidence_done()
    del maps, ev
    notes = np.stack([rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1)
    streamed = grouper.features_streaming(y, notes, transcriber, batch, check=check)
    groups, ginfo = grouper.group(y, notes, transcriber, tuning=tuning, streamed_features=streamed, check=check)
    return rep, res, rinfo, groups, ginfo


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Track:
    name: str               # instrument name ("Acoustic Grand Piano", ..., "Drums")
    program: int            # General MIDI program 0-127 (ignored for drums)
    notes: np.ndarray       # (N, 4): onset s, offset s, MIDI pitch, velocity 1-127
    is_drum: bool = False


@dataclass
class Transcription:
    source: str
    seconds: float
    tracks: list[Track]
    info: dict = field(default_factory=dict)

    def to_midi(self):
        """-> pretty_midi.PrettyMIDI (120 BPM, times in seconds)."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for t in self.tracks:
            # drum tracks get the acoustic-kit program (37); pretty_midi puts is_drum
            # instruments on the GM percussion channel (10) itself, so DAWs play the kit
            # instead of the piano patch that program 0 would select
            ins = pretty_midi.Instrument(program=37 if t.is_drum else int(t.program),
                                         is_drum=t.is_drum, name=t.name)
            for on, off, pitch, vel in t.notes:
                ins.notes.append(pretty_midi.Note(velocity=int(vel), pitch=int(pitch), start=float(on), end=float(off)))
            pm.instruments.append(ins)
        return pm

    def save_midi(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_midi().write(path)

    def summary(self) -> dict:
        return {"source": self.source, "seconds": round(self.seconds, 2), **self.info,
                "tracks": [{"name": t.name, "program": None if t.is_drum else int(t.program), "is_drum": t.is_drum,
                            "notes": int(len(t.notes))} for t in self.tracks]}


class Dettato:
    """All four models, loaded once.

    >>> model = Dettato.from_pretrained()          # downloads the weights from the Hugging Face Hub on first use
    >>> result = model.transcribe("song.mp3")
    >>> result.save_midi("song.mid")
    """

    def __init__(self, transcriber, repairer, grouper, drums, device):
        self.transcriber, self.repairer, self.grouper, self.drums = transcriber, repairer, grouper, drums
        self.device = device

    @classmethod
    def from_pretrained(cls, device: str | None = None):
        from .drums import DrumTranscriber
        from .evidence import EvidenceRepairer
        from .grouper import Grouper
        from .transcriber import Transcriber

        device = device or default_device()
        return cls(Transcriber.from_pretrained(device), EvidenceRepairer.from_pretrained(device), Grouper.from_pretrained(device),
                   DrumTranscriber.from_pretrained(device), device)

    @torch.no_grad()
    def pitched(self, x, batch: int = 2, check=None):
        """The pitched part of the pipeline on 16 kHz mono audio x: tuning -> transcriber -> evidence repairer -> grouper.
        -> (repaired notes, transcriber notes, repairer info, tracks, grouper info), all times on x's axis."""
        delta, strength, shift, r = retune.ratio(x)
        y = retune.resample(x, r) if shift else x
        tuning = (delta, strength, shift, r)
        if len(y) > STREAM_AFTER_S * 16000:
            rep, res, rinfo, groups, ginfo = streaming_pitched(
                self.transcriber, self.repairer, self.grouper, y, tuning, batch, check=check)
        else:
            # every 2 s: all windows (the repairer: maps + hidden); every 4 s: the windows the grouper reads (maps + hidden + stem)
            maps, bufs, (smaps, sbufs) = self.transcriber.run(
                y, hop_frac=0.25, batch=batch, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"), check=check)
            rep, res, rinfo = self.repairer.from_run(
                self.transcriber, maps, bufs.pop("hidden"), check=check, duration=len(y) / 16000)
            del maps, bufs
            notes = np.stack([rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1)
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            groups, ginfo = self.grouper.group(y, notes, self.transcriber, tuning=tuning, run=(smaps, sbufs), check=check)
        if r != 1.0:                                    # the groups come back on x's axis already; so do these two
            rep, res = ({**d, "onset": d["onset"] * r, "offset": d["offset"] * r} for d in (rep, res))
        return rep, res, rinfo, groups, ginfo

    @torch.no_grad()
    def transcribe(self, source, rate: int | None = None) -> Transcription:
        """source: a path to an audio file, or a mono float32 array with its sample ``rate``."""
        from .drums import MIN_SOURCE_RATE

        if isinstance(source, (str, os.PathLike)):
            name, a = str(source), A.load(str(source))
        else:
            if rate is None:
                raise ValueError("pass the sample rate with an array")
            name, a = "<array>", A.Audio(np.asarray(source, np.float32), int(rate))
        x = a.at(16000)
        _, _, rinfo, groups, ginfo = self.pitched(x)
        tracks = [Track(g["instrument"].replace("_", " "), int(g["program"]), np.asarray(g["notes"], np.float64)) for g in groups]
        info = {"notes": {"transcribed": rinfo["transcribed"], "dropped": rinfo["dropped"], "inserted": rinfo["inserted"],
                          "grouped": int(sum(len(t.notes) for t in tracks))},
                "tuning_cents": ginfo.get("tuning_cents")}
        if a.rate >= MIN_SOURCE_RATE:
            hits = self.drums.transcribe(a.at(44100))
            if hits:
                d = np.array([[h["onset"], h["offset"], h["pitch"], max(1, min(127, int(round(h["velocity"] * 127))))] for h in hits],
                             np.float64)
                tracks.append(Track("Drums", 0, d, is_drum=True))
            info["drums"] = {"hits": len(hits), **getattr(self.drums, "last_info", {})}
        else:
            info["drums"] = f"skipped: source recorded at {a.rate} Hz (the drum model needs >= {MIN_SOURCE_RATE} Hz)"
        return Transcription(name, a.seconds, tracks, info)


_DEFAULT: dict = {}


def transcribe(source, rate: int | None = None, device: str | None = None) -> Transcription:
    """One-call API: loads the models once per device and transcribes ``source`` (path or array + rate)."""
    device = device or default_device()
    if device not in _DEFAULT:
        _DEFAULT[device] = Dettato.from_pretrained(device)
    return _DEFAULT[device].transcribe(source, rate)
