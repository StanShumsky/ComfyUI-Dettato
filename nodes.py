"""ComfyUI nodes for KobiMusic's Dettato multi-instrument audio-to-MIDI transcription.

Dettato is four small models in a chain: a 15.3 M lane network transcribes the notes, a 4.9 M
note-level transformer repairs them with the transcriber's own evidence, a 2.9 M grouper puts
the notes into instrument-named tracks, and a 5.0 M model writes the drums (sources recorded at
40 kHz or more). The weights are standard safetensors for architectures ComfyUI's built-in
loaders do not know, so this pack ships a custom loader: the Loader node loads the transcriber,
and the Evidence / Grouper / Drums nodes attach the optional stages — each downloading its own
weights from Hugging Face on first use (or reading them from a local collection folder).
Everything runs at the released float32 precision. Transcribe takes up to nine audio sources -
any audio: files, stems from a separation pack, a video's audio track - and merges their tracks
into one .mid.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

from .dettato.audio import Audio
from .dettato.drums.infer import DrumTranscriber, MIN_SOURCE_RATE
from .dettato.evidence.model import Repairer
from .dettato.evidence.repair import EvidenceRepairer, conf_vel
from .dettato.grouper import retune
from .dettato.grouper.core import Grouper
from .dettato.grouper.instruments import INSTRUMENTS
from .dettato.grouper.model import NoteNet
from .dettato.pipeline import MIN_END_S, HOST_BUF_AFTER_S, Track
from .dettato.transcriber.core import Transcriber
from .dettato.transcriber.model import LaneNet
from .abc_score import tracks_to_abc
from .piano_roll import render_piano_roll
from .track_io import (
    audio_to_mono,
    dedupe_payload,
    dedup_sources,
    quantize_payload,
    score_text,
    filter_min_velocity,
    make_midi_payload,
    make_tracks_payload,
    render_midi_bytes,
    summary_json,
)

from comfy_api.latest import io

DETTATO_MODEL = io.Custom("DETTATO_MODEL")
DETTATO_MIDI = io.Custom("DETTATO_MIDI")
# The consumers accept both the merged payload and the legacy bare-tracks payload, so graphs
# built against older pack versions keep validating.
DETTATO_PAYLOAD = io.Custom("DETTATO_MIDI,DETTATO_TRACKS")

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path(folder_paths.models_dir) / "dettato"
folder_paths.add_model_folder_path("dettato", str(MODELS_DIR))

#: stage -> (subfolder holding its weights, Hugging Face repo).
STAGE_DIRS = {
    "transcriber": "dettato",
    "evidence": "dettato-evidence",
    "grouper": "dettato-grouper",
    "drums": "dettato-drums",
}
HF_REPOS = {
    "transcriber": "kobimusic/dettato",
    "evidence": "kobimusic/dettato-evidence",
    "grouper": "kobimusic/dettato-grouper",
    "drums": "kobimusic/dettato-drums",
}


# --------------------------------------------------------------------- model folders


def _model_dirs() -> list[Path]:
    """Folders searched for weight collections, in priority order; the first one is where downloads go."""
    dirs: list[Path] = [Path(p) for p in folder_paths.get_folder_paths("dettato")]
    if MODELS_DIR not in dirs:
        dirs.append(MODELS_DIR)
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def _stage_files(root: Path) -> dict[str, Path]:
    """Stage directories (config.json + model.safetensors) present under a collection root."""
    out = {}
    for stage, sub in STAGE_DIRS.items():
        d = root / sub
        if (d / "config.json").is_file() and (d / "model.safetensors").is_file():
            out[stage] = d
    return out


def _local_collections() -> list[str]:
    """Names of folders in the model dirs that hold the transcriber (usable local collections)."""
    names: list[str] = []
    for base in _model_dirs():
        if not base.exists():
            continue
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if sub.name.startswith(".") or ".cache" in sub.parts:
                continue
            if "transcriber" in _stage_files(sub) and sub.name not in names:
                names.append(sub.name)
    return names


def _find_collection(name: str) -> Path | None:
    for base in _model_dirs():
        candidate = base / name
        if candidate.is_dir() and _stage_files(candidate):
            return candidate
    return None


def _purge_stage(stage: str, root: Path) -> None:
    folder = root / STAGE_DIRS[stage]
    if folder.exists():
        LOGGER.warning("Dettato: removing corrupt %s", folder)
        shutil.rmtree(folder, ignore_errors=True)


def _download_stage(stage: str, root: Path, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target = root / STAGE_DIRS[stage]
    if force:
        _purge_stage(stage, root)
    if (target / "model.safetensors").exists() and (target / "config.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", HF_REPOS[stage], target)
    snapshot_download(
        repo_id=HF_REPOS[stage],
        allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE"],
        local_dir=str(target),
    )
    if not (target / "model.safetensors").exists():
        raise FileNotFoundError(f"Download of {HF_REPOS[stage]} did not produce {target / 'model.safetensors'}.")
    return target


def _load_stage_state(stage_dir: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    config = json.loads((stage_dir / "config.json").read_text())
    return config, load_file(str(stage_dir / "model.safetensors"))


def _check_weights(state_dict: dict[str, torch.Tensor], name: str) -> None:
    """Raise if the weights are visibly broken, e.g. a truncated or zero-filled download.

    Such a file still loads with strict=True (all keys and shapes are there) but the model
    predicts garbage, so it is caught here instead. A few all-zero tensors are legitimate
    (fresh initializations), but a zero-filled file has most of its tensors zero.
    """
    checked = 0
    nonfinite, zeros = [], []
    for key, value in state_dict.items():
        if not value.is_floating_point() or value.numel() <= 100:
            continue
        checked += 1
        if not torch.isfinite(value).all():
            nonfinite.append(key)
        elif not value.any():
            zeros.append(key)
    if nonfinite or (checked and len(zeros) > checked / 2):
        bad = (nonfinite + zeros)[:3]
        raise RuntimeError(
            f"{name}: weights look corrupt ({len(nonfinite)} tensors with NaN/inf, "
            f"{len(zeros)} of {checked} tensors all zero, e.g. {bad}). The file is probably an "
            "incomplete download."
        )


def _load_stage_verified(stage: str, root: Path, allow_download: bool) -> tuple[dict, dict[str, torch.Tensor]]:
    """Load one stage from ``root``; with allow_download, fetch it from the Hub and retry once on a bad file."""
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt == 1 and not allow_download:
            break
        target = _download_stage(stage, root, force=attempt == 1)
        try:
            config, state = _load_stage_state(target)
            _check_weights(state, STAGE_DIRS[stage])
            return config, state
        except Exception as error:  # a truncated safetensors raises its own error type
            last_error = error
            if attempt == 0:
                if allow_download:
                    LOGGER.warning("Dettato: %s; re-downloading %s.", error, STAGE_DIRS[stage])
                else:
                    break
    raise RuntimeError(
        f"{STAGE_DIRS[stage]} could not be loaded from {root}: {last_error}. "
        "Delete the folder (and its .cache) and retry, or check free disk space and your connection."
    ) from last_error


# --------------------------------------------------------------------- building the stages


def _resolve_root(name: str) -> Path:
    """The folder holding the model's stage subfolders (the first model dir for 'dettato')."""
    if name == "dettato":
        root = _model_dirs()[0]
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = _find_collection(name)
    if root is None:
        raise FileNotFoundError(
            f"Model {name!r} not found; expected a folder with the transcriber subfolder (dettato/) "
            f"in one of: {[str(d) for d in _model_dirs()]}"
        )
    if "transcriber" not in _stage_files(root):
        raise FileNotFoundError(
            f"Collection {name!r} lacks the transcriber (dettato/); it has: {sorted(_stage_files(root))}."
        )
    return root


def _load_stage(stage: str, name: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """One stage's (config, state dict) from the model's root; 'dettato' downloads it from the Hub."""
    root = _resolve_root(name)
    if name != "dettato" and stage not in _stage_files(root):
        raise FileNotFoundError(
            f"Collection {name!r} lacks stage {STAGE_DIRS[stage]}/; it has: {sorted(_stage_files(root))}."
        )
    return _load_stage_verified(stage, root, allow_download=(name == "dettato"))


def _build_stage(stage: str, config: dict, state: dict[str, torch.Tensor], device: torch.device):
    if stage == "evidence":
        model = Repairer(**config["model"])
        model.load_state_dict(state, strict=True)
        return EvidenceRepairer(model, config["keep_th"], config["insert_th"], device)
    if stage == "grouper":
        if config.get("instruments") != INSTRUMENTS:
            raise RuntimeError(
                f"{STAGE_DIRS[stage]}: the instrument list of the weights does not match this pack; "
                "the weights are for a different dettato release."
            )
        model = NoteNet(**config["model"])
        model.load_state_dict(state, strict=True)
        return Grouper(model, config, device)
    if stage == "drums":
        return DrumTranscriber(config, state, device)
    raise ValueError(f"unknown stage {stage!r}")


def load_dettato(name: str, device: str) -> dict:
    """The model payload: the transcriber (always); the optional stages attach via the stage nodes."""
    if device == "auto":
        torch_device = mm.get_torch_device()
    else:
        torch_device = torch.device(device)
    config, state = _load_stage("transcriber", name)
    model = LaneNet(**config["model"])
    model.load_state_dict(state, strict=True)
    transcriber = Transcriber(model, config["thresholds"], torch_device)
    LOGGER.info("Dettato: loaded the transcriber (%s) in fp32 on %s.", name, torch_device)
    return {
        "transcriber": transcriber,
        "evidence": None,
        "grouper": None,
        "drums": None,
        "device": torch_device,
        "name": name,
    }


# --------------------------------------------------------------------- transcription


def run_transcription(wrapper: dict, audio: dict, progress: ProgressBar | None = None) -> dict:
    """AUDIO payload -> the tracks payload (one source's unit of the merged MIDI payload)."""
    def done() -> None:
        if progress is not None:
            progress.update(1)

    mono, rate = audio_to_mono(audio)
    src = Audio(mono, rate)
    x16 = src.at(16000)
    device = wrapper["device"]
    chunks = 1
    if wrapper["grouper"] is not None:
        chunks += 1
    if wrapper["drums"] is not None and rate >= MIN_SOURCE_RATE:
        chunks += 1
    if progress is None:
        progress = ProgressBar(chunks)

    info: dict = {}
    transcriber = wrapper["transcriber"]
    evidence, grouper = wrapper["evidence"], wrapper["grouper"]
    if evidence is not None and grouper is not None:
        # the upstream path: retune to A440 when clearly off it, then one transcriber pass feeds
        # both the repairer (every window) and the grouper (every other window)
        delta, strength, shift, r = retune.ratio(x16)
        y = retune.resample(x16, r) if shift else x16
        host = len(y) > HOST_BUF_AFTER_S * 16000 and str(device).startswith("cuda")
        maps, bufs, (smaps, sbufs) = transcriber.run(
            y, hop_frac=0.25, batch=2, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"),
            buf_device="cpu" if host else None,
        )
        rep, _, rinfo = evidence.from_run(transcriber, maps, bufs.pop("hidden"))
        del maps, bufs
        notes = np.stack(
            [rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1
        )
        done()
        groups, ginfo = grouper.group(y, notes, transcriber, tuning=(delta, strength, shift, r), run=(smaps, sbufs))
        del smaps, sbufs
        tracks = [
            Track(g["instrument"].replace("_", " "), int(g["program"]), np.asarray(g["notes"], np.float64))
            for g in groups
        ]
        info["tuning_cents"] = ginfo.get("tuning_cents")
        note_stats = {"transcribed": rinfo["transcribed"], "dropped": rinfo["dropped"], "inserted": rinfo["inserted"]}
    else:
        # only one of the two (or neither) attached: the transcriber runs its own pass, and the
        # grouper, when present, estimates the tuning and runs its own pass
        if evidence is not None:
            rep, _, rinfo = evidence(transcriber, x16)
            notes3 = np.stack(
                [rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1
            )
            velocity = conf_vel(rep["score"])
            note_stats = {"transcribed": rinfo["transcribed"], "dropped": rinfo["dropped"], "inserted": rinfo["inserted"]}
        else:
            maps, _ = transcriber.run(x16)
            res = transcriber.notes(maps)
            notes3 = np.stack([res["onset"], res["offset"], res["pitch"].astype(np.float64)], 1)
            velocity = conf_vel(res["conf"])
            note_stats = {"transcribed": int(len(res["onset"])), "dropped": 0, "inserted": 0}
        done()
        if grouper is not None:
            groups, ginfo = grouper.group(x16, notes3, transcriber)
            tracks = [
                Track(g["instrument"].replace("_", " "), int(g["program"]), np.asarray(g["notes"], np.float64))
                for g in groups
            ]
            info["tuning_cents"] = ginfo.get("tuning_cents")
        else:
            tracks = [
                Track(
                    "Notes",
                    0,
                    np.stack(
                        [notes3[:, 0], np.maximum(notes3[:, 1], notes3[:, 0] + MIN_END_S), notes3[:, 2], velocity.astype(np.float64)], 1
                    ),
                )
            ]
            info["tuning_cents"] = None
    note_stats["grouped"] = int(sum(len(t.notes) for t in tracks))
    info["notes"] = note_stats
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    done()

    drums = wrapper["drums"]
    if drums is not None and rate >= MIN_SOURCE_RATE:
        hits = drums.transcribe(src.at(44100))
        if hits:
            drum_notes = np.array(
                [[h["onset"], h["offset"], h["pitch"], max(1, min(127, int(round(h["velocity"] * 127))))] for h in hits],
                np.float64,
            )
            tracks.append(Track("Drums", 0, drum_notes, is_drum=True))
        info["drums"] = {"hits": len(hits), **getattr(drums, "last_info", {})}
    else:
        reason = (
            f"source recorded at {rate} Hz (the drum model needs >= {MIN_SOURCE_RATE} Hz)"
            if drums is not None
            else "drum stage not attached"
        )
        info["drums"] = f"skipped: {reason}"
    done()

    return make_tracks_payload(tracks, src.seconds, "audio", info)


# --------------------------------------------------------------------- nodes


CATEGORY = "music/Dettato"


class DettatoLoader(io.ComfyNode):
    """Load the Dettato transcriber payload (DETTATO_MODEL) that the stage and Transcribe nodes consume."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoLoader",
            display_name="Dettato Loader",
            category=CATEGORY,
            description=(
                "Load KobiMusic's Dettato transcriber: a 15.3 M-parameter lane network that writes the"
                'notes of any recording. Attach the optional stages with the Dettato Evidence / Grouper /'
                'Drums nodes; a stage that is not in the graph is bypassed, and its weights are never'
                "downloaded or loaded. 'dettato' downloads each stage from Hugging Face into models/dettato"
                'on first use; a folder in that directory holding the transcriber subfolder (dettato/, with'
                'its config.json and model.safetensors) is listed and read from disk instead, and the stage'
                'nodes read their stage from the same folder. All weights stay small: the full chain is'
                '28.1 M parameters, 113 MB in fp32, so they run at their released precision.'
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=["dettato"] + _local_collections(),
                    default="dettato",
                    tooltip="The collection to load; 'dettato' auto-downloads each stage into models/dettato "
                            "on first use. A folder in that directory holding the transcriber subfolder is "
                            "listed here and read from disk instead.",
                ),
                io.Combo.Input("device", options=["auto", "cuda", "mps", "cpu"], default="auto"),
            ],
            outputs=[DETTATO_MODEL.Output(id="model", display_name="model")],
        )

    @classmethod
    def execute(cls, model: str, device: str) -> io.NodeOutput:
        return io.NodeOutput(load_dettato(model, device))


class DettatoStageNode(io.ComfyNode):
    """Attach one optional stage to a DETTATO_MODEL payload (the transcriber comes from the Loader).

    The concrete nodes below set STAGE and DESCRIPTION; they are the ones registered, not this base.
    """

    STAGE: str | None = None
    DESCRIPTION: str = ""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=cls.__name__,
            display_name=f"Dettato {cls.STAGE.capitalize()}",
            category=CATEGORY,
            description=cls.DESCRIPTION,
            inputs=[DETTATO_MODEL.Input("model")],
            outputs=[DETTATO_MODEL.Output(id="model", display_name="model")],
        )

    @classmethod
    def execute(cls, model: dict) -> io.NodeOutput:
        if model[cls.STAGE] is not None:
            return io.NodeOutput(model)
        config, state = _load_stage(cls.STAGE, model["name"])
        model[cls.STAGE] = _build_stage(cls.STAGE, config, state, model["device"])
        LOGGER.info("Dettato: attached the %s stage (%s).", cls.STAGE, model["name"])
        return io.NodeOutput(model)


class DettatoEvidence(DettatoStageNode):
    STAGE = "evidence"
    DESCRIPTION = (
        'Attach the evidence-repair stage (a 4.9 M note-level transformer) to the Dettato model:'
        "the transcriber's notes are re-checked against its own frame evidence, dropping false"
        "notes and inserting missed ones. The weights (19 MB) come from the model's collection,"
        "auto-downloaded for 'dettato'. Leave this node out to run on the raw transcriber notes."
    )


class DettatoGrouper(DettatoStageNode):
    STAGE = "grouper"
    DESCRIPTION = (
        'Attach the grouping stage (2.9 M) to the Dettato model: the notes are split into'
        "instrument-named tracks (73 classes, each with a General MIDI program) and the recording's"
        "tuning is estimated. The weights (12 MB) come from the model's collection, auto-downloaded"
        "for 'dettato'. Leave this node out for a single 'Notes' track."
    )


class DettatoDrums(DettatoStageNode):
    STAGE = "drums"
    DESCRIPTION = (
        'Attach the drum stage (5.0 M) to the Dettato model: drum kits and orchestral percussion'
        "become a 'Drums' track, timpani written at their pitch. It needs a source recorded at 40"
        'kHz or more (44.1/48 kHz files qualify) and is skipped otherwise, with the reason in'
        "info_json. The weights (20 MB) come from the model's collection, auto-downloaded for"
        "'dettato'."
    )


class DettatoTranscribe(io.ComfyNode):
    """Up to nine audio sources (autogrowing slots, one per slot) -> one merged MIDI payload."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DettatoTranscribe",
            display_name="Dettato Transcribe to MIDI",
            category="music/Dettato",
            description=(
                "Transcribe one or more audio sources to a single merged .mid. One slot (audio_0) "
                "is shown by default and each used slot reveals the next (audio_1, audio_2, ...), "
                "up to nine total - e.g. one stem per slot from a separation pack, so 'Separate -> "
                "Transcribe -> Save MIDI' is the whole separation-to-MIDI workflow; any audio "
                "works (files, stems, a video's audio track). Every source runs the attached "
                "stages in pipeline "
                "order (evidence -> grouper -> drums); a stage that is not in the graph is "
                "bypassed. Mono is the channel average, at the file's own sample rate; the "
                "transcriber, the repairer and the grouper read 16 kHz internally, and the drum "
                "stage runs only when a source was recorded at 40 kHz or more (44.1 kHz and 48 "
                "kHz files qualify; 8/16/24/32 kHz files and most phone recordings do not). The "
                "notes are performance MIDI: they keep the timing of the recording; the "
                "Dettato Quantize MIDI node snaps onsets to a tempo x grid afterwards when "
                "you want the MIDI to sit on the grid. The sources' tracks merge in slot order; "
                "the first "
                "occurrence of a track name keeps it, and repeats are renamed 'Name (slot)' with "
                "the slot name, so a DAW shows 'Piano' and 'Piano (audio_1)'. Sources of one "
                "recording are length-aligned by the separator, so the merged file is "
                "phase-correct by construction. min_velocity drops quieter notes per source (1 "
                "keeps everything). dedup drops redundant tracks before the merge: when the same "
                "instrument was transcribed from several stems of one mix (separation artifacts), "
                "only the stem with the most notes of it is kept - on for a separated-stem run, "
                "off when the sources are different recordings. Outputs: the merged MIDI payload (tracks, length, source "
                "labels, per-source info, and the .mid rendered once) for the nodes below, and a "
                "JSON summary. A different stage chain per stem means one Transcribe and one Save "
                "MIDI per stem. GPU timing: a 4-minute song takes about 15 s on an RTX 5060; the "
                "CPU works but is roughly 6x slower."
            ),
            inputs=[
                DETTATO_MODEL.Input("model"),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    tooltip="One audio per slot (e.g. one separated stem per slot). audio_0 is shown by "
                            "default; each used slot reveals the next, up to nine total.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"), prefix="audio_", min=0, max=9),
                ),
                io.Int.Input("min_velocity", default=1, min=1, max=127,
                             tooltip="Drop notes quieter than this per source (1 keeps everything)."),
                io.Boolean.Input("dedup", default=False,
                                 tooltip="Drop redundant tracks before the merge: when the same "
                                         "instrument was transcribed from several stems, only the "
                                         "stem with the most notes of it is kept (on for a "
                                         "separated-stem run; off when the sources are different "
                                         "recordings)."),
            ],
            outputs=[DETTATO_MIDI.Output(display_name="midi"), io.String.Output(display_name="info_json")],
        )

    @classmethod
    def execute(cls, model: dict, audios: dict | None, min_velocity: int = 1,
                dedup: bool = False) -> io.NodeOutput:
        """All connected audio slots (audio_0, audio_1, ...) -> the merged MIDI payload."""
        audios = {name: aud for name, aud in (audios or {}).items() if aud is not None}
        if not audios:
            raise ValueError("Connect at least one audio source.")
        sources = [(name, audios[name]) for name in
                   sorted(audios, key=lambda n: int(n.rsplit("_", 1)[-1]))]
        if model["drums"] is None:
            eligible = [name for name, aud in sources if int(aud["sample_rate"]) >= MIN_SOURCE_RATE]
            if eligible:
                LOGGER.warning(
                    "Dettato: no Drums stage in the model chain - %s record(s) at %d+ Hz would "
                    "have produced a 'Drums' track. Attach the Dettato Drums node to the model "
                    "to get drums.",
                    ", ".join(eligible), MIN_SOURCE_RATE,
                )
        chunks = 0
        for _, aud in sources:
            rate = int(aud["sample_rate"])
            chunks += 1
            if model["grouper"] is not None:
                chunks += 1
            if model["drums"] is not None and rate >= MIN_SOURCE_RATE:
                chunks += 1
        progress = ProgressBar(chunks)
        results = []
        for label, aud in sources:
            payload = run_transcription(model, aud, progress)
            payload = make_tracks_payload(
                filter_min_velocity(payload["tracks"], min_velocity), payload["seconds"], label, payload["info"]
            )
            results.append((label, payload))
        mm.soft_empty_cache()
        if dedup:
            results, stats = dedup_sources(results)
        else:
            stats = None
        payload = make_midi_payload(results)
        if stats is not None:
            payload["info"]["dedup"] = stats
        return io.NodeOutput(payload, summary_json(payload))



class DettatoDedupeMIDI(io.ComfyNode):
    """Merge same-named tracks and drop the onset-duplicate notes of a stem separation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoDedupeMIDI",
            display_name="Dettato Dedupe MIDI",
            category=CATEGORY,
            description=(
                "Cleans the duplication artifacts of a stem separation: when the same sound is "
                "transcribed from several stems, the merge shows it once per stem ('Drums' plus "
                "'Drums (audio_1)' ... 'Drums (audio_5)') with small onset shifts, so the drums "
                "- and any other instrument split across stems - sound doubled. This node pools "
                "the tracks of one instrument (the names differ only by the ' (slot)' merge "
                "suffix) into a single track and scans the pool in onset order: a note is a "
                "duplicate when the gap to the last kept note of the same pitch is within "
                "window_ms, and the louder note of a duplicate pair wins (a quiet ghost never "
                "replaces a loud real one, and a loud real one replaces a quiet ghost that "
                "preceded it). 50 ms is the safe default: on a 6-stem run it removed all the "
                "tight drum duplicates while the main drum track kept 100% of its own fast "
                "repeats; raise it to reach smeared copies shifted by 100-300 ms, knowing that "
                "fast same-pitch repeats inside that gap will be collapsed. 0 pools the tracks "
                "without dropping anything. Put it between Transcribe and Save MIDI; its "
                "outputs feed the same consumers as Transcribe's."
            ),
            inputs=[
                DETTATO_PAYLOAD.Input("midi"),
                io.Int.Input("window_ms", default=50, min=0, max=500,
                             tooltip="Onset window in ms for same-pitch duplicates "
                                     "(0 = pool the tracks, drop nothing)."),
            ],
            outputs=[DETTATO_MIDI.Output(display_name="midi"), io.String.Output(display_name="info_json")],
        )

    @classmethod
    def execute(cls, midi: dict, window_ms: int) -> io.NodeOutput:
        payload = dedupe_payload(midi, window_ms)
        return io.NodeOutput(payload, summary_json(payload))


class DettatoQuantizeMIDI(io.ComfyNode):
    """Snap every note onset of a MIDI payload to a tempo x grid."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoQuantizeMIDI",
            display_name="Dettato Quantize MIDI",
            category=CATEGORY,
            description=(
                "Snaps every note onset to a musical grid defined by the tempo and the grid "
                "division (1/4, 1/8, 1/16 or 1/32 notes): a note closer to a grid line than to "
                "the one before or after it moves to that line, and the note's length is "
                "preserved - the offset moves by the same amount as the onset, so nothing ends "
                "up shorter or longer (notes under 2 ms are stretched to that minimum, so "
                "nothing is zero-length). It works on any DETTATO_MIDI payload - the Transcribe "
                "output or the output of another MIDI node like Dedupe - so the grid follows "
                "the whole chain of edits, and it writes its stats (grid, notes moved, max "
                "shift) into info.quantize. max_shift_ms is a quick tell for a wrong tempo: if "
                "notes moved more than half a grid step, the tempo or the division is off. Put "
                "it between the last MIDI node and Save MIDI; its outputs feed the same "
                "consumers."
            ),
            inputs=[
                DETTATO_PAYLOAD.Input("midi"),
                io.Float.Input("tempo", default=120.0, min=20.0, max=300.0, step=1.0,
                               tooltip="Grid tempo in BPM."),
                io.Combo.Input("grid", options=["1/4", "1/8", "1/16", "1/32"], default="1/16",
                               tooltip="Grid division (notes per beat)."),
            ],
            outputs=[DETTATO_MIDI.Output(display_name="midi"), io.String.Output(display_name="info_json")],
        )

    @classmethod
    def execute(cls, midi: dict, tempo: float = 120.0, grid: str = "1/16") -> io.NodeOutput:
        payload = quantize_payload(midi, tempo, grid)
        return io.NodeOutput(payload, summary_json(payload))


class DettatoPianoRoll(io.ComfyNode):
    """The piano roll image of a transcription payload."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoPianoRoll",
            display_name="Dettato Piano Roll",
            category=CATEGORY,
            description=(
                'The piano roll of the transcription: the upper panel is the 88-key range (G7..A0, C rows'
                'shaded), the lower panel the GM drum keys 27-87; tracks are coloured in track order, drums'
                'in red, note brightness follows velocity, and a line marks every second.'
                'px_per_ms is the horizontal scale (0 = fit the song into about 1400 px); max_seconds caps'
                'the visible length (0 = the whole song). Feed the result to a Preview Image node.'
            ),
            inputs=[
                DETTATO_PAYLOAD.Input("midi"),
                io.Float.Input("px_per_ms", default=0.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("max_seconds", default=0, min=0, max=36000),
                io.Boolean.Input("show_drum_panel", default=True),
            ],
            outputs=[io.Image.Output(id="piano_roll", display_name="piano_roll")],
        )

    @classmethod
    def execute(cls, midi: dict, px_per_ms: float, max_seconds: int,
                show_drum_panel: bool) -> io.NodeOutput:
        roll = render_piano_roll(midi["tracks"], midi["seconds"], px_per_ms, max_seconds, show_drum_panel)
        return io.NodeOutput(torch.from_numpy(roll)[None].contiguous())


class DettatoSaveMIDI(io.ComfyNode):
    """Write the merged MIDI payload to ComfyUI's output folder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoSaveMIDI",
            display_name="Dettato Save MIDI",
            category=CATEGORY,
            description=(
                "Write the .mid to ComfyUI's output folder as <prefix>_XXXXX.mid - one General MIDI track"
                'per instrument, named after it, 120 BPM with times in seconds, numbered so runs never'
                "overwrite each other. The MIDI payload (the output of the Transcribe node) is written"
                'verbatim: its .mid was rendered once at the merge. The JSON summary (track names and note'
                'counts, the tuning estimate, the drum-model decision) is the info_json output of the'
                'Transcribe node; save it with a text node when needed.'
            ),
            is_output_node=True,
            inputs=[
                DETTATO_PAYLOAD.Input("midi"),
                io.String.Input("filename_prefix", default="dettato/midi"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, midi: dict, filename_prefix: str) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.mid")))
        path = Path(folder) / f"{filename}_{counter:05d}.mid"
        data = midi.get("midi")
        if data is None:
            # a legacy DETTATO_TRACKS payload (bare tracks, no rendered bytes): render it now
            data = render_midi_bytes(midi["tracks"], midi.get("seconds", 0.0))
        path.write_bytes(data)
        n_notes = sum(len(np.asarray(t.notes, np.float64).reshape(-1, 4)) for t in midi["tracks"])
        LOGGER.info("Dettato: wrote %s (%d tracks, %d notes).", path, len(midi["tracks"]), n_notes)
        # an output node returns no values (None would break ComfyUI's merge_result_data)
        return io.NodeOutput()


class DettatoInfo(io.ComfyNode):
    """The transcription summary as a JSON string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoInfo",
            display_name="Dettato Info",
            category=CATEGORY,
            description=(
                'The transcription summary as JSON: source, length in seconds, per-track name / General'
                'MIDI program / is_drum / note count, the note statistics (transcribed, dropped by the'
                'repairer, inserted by the repairer, grouped), the tuning estimate in cents and the drum'
                "stage's decision (hits and router value, or why it was skipped). Use it to branch or to"
                "save the run's metadata with another text node."
            ),
            inputs=[DETTATO_PAYLOAD.Input("midi")],
            outputs=[io.String.Output(id="info_json", display_name="info_json")],
        )

    @classmethod
    def execute(cls, midi: dict) -> io.NodeOutput:
        return io.NodeOutput(summary_json(midi))


class DettatoMIDIToText(io.ComfyNode):
    """The transcription as generic JSON text for other libraries and text nodes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoMIDIToText",
            display_name="Dettato MIDI to Text",
            category=CATEGORY,
            description=(
                "The transcription as generic JSON text: {seconds, tracks: [{name, program, is_drum, "
                "notes: [[onset s, offset s, pitch, velocity], ...]]}}. Any library that reads JSON can "
                "use it, and it can be saved with any text node (e.g. the core Save Text) or fed to "
                "another pack's text input - unlike the .mid, it needs no MIDI parser. program is null "
                "on drum tracks. Takes the midi output of Transcribe (or Dedupe MIDI); the same text is "
                "carried inside the payload."
            ),
            inputs=[DETTATO_PAYLOAD.Input("midi")],
            outputs=[io.String.Output(id="midi_text", display_name="midi_text")],
        )

    @classmethod
    def execute(cls, midi: dict) -> io.NodeOutput:
        text = midi.get("midi_text")
        if text is None:
            text = score_text(midi["tracks"], midi.get("seconds", 0.0))
        return io.NodeOutput(text)


class DettatoToYuE2ABC(io.ComfyNode):
    """Convert the transcription to YuE2's native two-voice ABC score."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoToYuE2ABC",
            display_name="Dettato To YuE2 ABC",
            category=CATEGORY,
            description=(
                "Convert the transcription to the native two-voice ABC score that ComfyUI's built-in YuE2"
                "Generate Music node renders: wire the abc_score output to that node's abc input and YuE2"
                "regenerates the song from your transcribed melody. mode: 'melody' (the default and the"
                'recommended cover path) leaves the score chord-free, so YuE2 keeps your melody and writes'
                "its own accompaniment; 'full' derives chord symbols from the non-melody tracks."
                'melody_track: the exact name of the track to use as the melody (empty = pick the most'
                'lead-like pitched track automatically - a voice/vocal track wins, a bass track is never'
                'picked); ins_track: a second pitched track for the Ins voice (empty = the busiest'
                "remaining line above G3 that is not a bass; 'none' = the Ins voice is rests). key: 'auto'"
                "runs key-finding on the melody, or a standard key such as 'Am' or 'Db'; tempo: 0 fits a"
                "grid to the onsets (drum hits weighted double), or a fixed BPM. octave: 'auto' moves a"
                'line by whole octaves into the C4..A#5 window when its middle pitch lies outside it,'
                'because YuE2 sings the octave it is given; an integer transposes both lines. The output is'
                'the exact dialect YuE2 and SheetSage2 use, so you can inspect it, correct a few notes in'
                "any text editor and feed the edited text to YuE2. See the README's YuE2 section."
            ),
            inputs=[
                DETTATO_PAYLOAD.Input("midi"),
                io.Combo.Input("mode", options=["melody", "full"], default="melody"),
                io.String.Input("melody_track", default=""),
                io.String.Input("ins_track", default=""),
                io.String.Input("key", default="auto"),
                io.Int.Input("tempo", default=0, min=0, max=500),
                io.Combo.Input("meter", options=["4/4", "3/4", "6/8", "7/8"], default="4/4"),
                io.Combo.Input("grid", options=["1/16", "1/32"], default="1/16"),
                io.Combo.Input("octave", options=["auto", "-2", "-1", "0", "1", "2"], default="auto"),
            ],
            outputs=[
                io.String.Output(id="abc_score", display_name="abc_score"),
                io.String.Output(id="abc_summary", display_name="abc_summary"),
            ],
        )

    @classmethod
    def execute(cls, midi: dict, mode: str, melody_track: str, ins_track: str, key: str,
                tempo: int, meter: str, grid: str, octave: str) -> io.NodeOutput:
        meta = {}
        abc = tracks_to_abc(
            midi["tracks"], midi["seconds"], mode=mode, melody_track=melody_track,
            ins_track=ins_track, key=key, tempo=tempo, meter=meter, grid=grid, octave=octave,
            out_meta=meta,
        )
        return io.NodeOutput(abc, json.dumps(meta, indent=2, sort_keys=True))


class DettatoSaveABC(io.ComfyNode):
    """Write an ABC score to ComfyUI's output folder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DettatoSaveABC",
            display_name="Dettato Save ABC",
            category=CATEGORY,
            description=(
                "Write an ABC score to ComfyUI's output folder as <prefix>_XXXXX.abc (e.g. the abc_score"
                'output of the To YuE2 ABC node), numbered so runs never overwrite each other. Use it to'
                "keep the run's symbolic record, or to open the score in an editor, correct a few notes and"
                'feed the edited text to YuE2.'
            ),
            is_output_node=True,
            inputs=[
                io.String.Input("abc", default="", multiline=True),
                io.String.Input("filename_prefix", default="dettato/abc"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, abc: str, filename_prefix: str) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.abc")))
        path = Path(folder) / f"{filename}_{counter:05d}.abc"
        path.write_text(abc, encoding="utf-8")
        LOGGER.info("Dettato: wrote %s (%d chars).", path, len(abc))
        return io.NodeOutput()


#: id -> class registry. The ComfyUI loader does not read this dict: the pack registers through
#: the V3 entrypoint in __init__.py (comfy_entrypoint -> get_node_list). The mapping stays as the
#: single reference list and the test seam; it must match get_node_list.
NODE_CLASS_MAPPINGS = {
    "DettatoLoader": DettatoLoader,
    "DettatoEvidence": DettatoEvidence,
    "DettatoGrouper": DettatoGrouper,
    "DettatoDrums": DettatoDrums,
    "DettatoTranscribe": DettatoTranscribe,
    "DettatoDedupeMIDI": DettatoDedupeMIDI,
    "DettatoQuantizeMIDI": DettatoQuantizeMIDI,
    "DettatoPianoRoll": DettatoPianoRoll,
    "DettatoSaveMIDI": DettatoSaveMIDI,
    "DettatoInfo": DettatoInfo,
    "DettatoMIDIToText": DettatoMIDIToText,
    "DettatoToYuE2ABC": DettatoToYuE2ABC,
    "DettatoSaveABC": DettatoSaveABC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DettatoLoader": "Dettato Loader",
    "DettatoEvidence": "Dettato Evidence",
    "DettatoGrouper": "Dettato Grouper",
    "DettatoDrums": "Dettato Drums",
    "DettatoTranscribe": "Dettato Transcribe to MIDI",
    "DettatoDedupeMIDI": "Dettato Dedupe MIDI",
    "DettatoQuantizeMIDI": "Dettato Quantize MIDI",
    "DettatoPianoRoll": "Dettato Piano Roll",
    "DettatoSaveMIDI": "Dettato Save MIDI",
    "DettatoInfo": "Dettato Info",
    "DettatoMIDIToText": "Dettato MIDI to Text",
    "DettatoToYuE2ABC": "Dettato To YuE2 ABC",
    "DettatoSaveABC": "Dettato Save ABC",
}
