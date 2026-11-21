"""ComfyUI nodes for KobiMusic's Dettato multi-instrument audio-to-MIDI transcription.

Dettato is four small models in a chain: a 15.3 M lane network transcribes the notes, a 4.9 M
note-level transformer repairs them with the transcriber's own evidence, a 2.9 M grouper puts
the notes into instrument-named tracks, and a 5.0 M model writes the drums (sources recorded at
40 kHz or more). The weights are standard safetensors for architectures ComfyUI's built-in
loaders do not know, so this pack ships a custom loader: the Loader node loads the transcriber,
and the Evidence / Grouper / Drums nodes attach the optional stages — each downloading its own
weights from Hugging Face on first use (or reading them from a local collection folder).
Everything runs at the released float32 precision.
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
from .track_io import audio_to_mono, filter_min_velocity, make_tracks_payload, summary_json, write_midi

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
    transcriber = Transcriber(LaneNet(**config["model"]), config["thresholds"], torch_device)
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
    """AUDIO payload -> DETTATO_TRACKS payload (transcription without velocity filtering)."""
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


class DettatoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        choices = ["dettato"] + _local_collections()
        return {
            "required": {
                "model": (choices, {"default": "dettato"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DETTATO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "Load KobiMusic's Dettato transcriber: a 15.3 M-parameter lane network that writes the "
        "notes of any recording. Attach the optional stages with the Dettato Evidence / Grouper / "
        "Drums nodes; a stage that is not in the graph is bypassed, and its weights are never "
        "downloaded or loaded. 'dettato' downloads each stage from Hugging Face into "
        "models/dettato on first use; a folder in that directory holding the transcriber "
        "subfolder (dettato/, with its config.json and model.safetensors) is listed and read from "
        "disk instead, and the stage nodes read their stage from the same folder. All weights "
        "stay small: the full chain is 28.1 M parameters, 113 MB in fp32, so they run at their "
        "released precision."
    )

    def load(self, model: str, device: str):
        return (load_dettato(model, device),)


class DettatoStageNode:
    """Attach one optional stage to a DETTATO_MODEL payload (the transcriber comes from the Loader)."""

    STAGE: str

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("DETTATO_MODEL",)}}

    RETURN_TYPES = ("DETTATO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    CATEGORY = "music/Dettato"

    def attach(self, model: dict) -> tuple[dict,]:
        if model[self.STAGE] is not None:
            return (model,)
        config, state = _load_stage(self.STAGE, model["name"])
        model[self.STAGE] = _build_stage(self.STAGE, config, state, model["device"])
        LOGGER.info("Dettato: attached the %s stage (%s).", self.STAGE, model["name"])
        return (model,)


class DettatoEvidence(DettatoStageNode):
    STAGE = "evidence"
    DESCRIPTION = (
        "Attach the evidence-repair stage (a 4.9 M note-level transformer) to the Dettato model: "
        "the transcriber's notes are re-checked against its own frame evidence, dropping false "
        "notes and inserting missed ones. The weights (19 MB) come from the model's collection, "
        "auto-downloaded for 'dettato'. Leave this node out to run on the raw transcriber notes."
    )


class DettatoGrouper(DettatoStageNode):
    STAGE = "grouper"
    DESCRIPTION = (
        "Attach the grouping stage (2.9 M) to the Dettato model: the notes are split into "
        "instrument-named tracks (73 classes, each with a General MIDI program) and the "
        "recording's tuning is estimated. The weights (12 MB) come from the model's collection, "
        "auto-downloaded for 'dettato'. Leave this node out for a single 'Notes' track."
    )


class DettatoDrums(DettatoStageNode):
    STAGE = "drums"
    DESCRIPTION = (
        "Attach the drum stage (5.0 M) to the Dettato model: drum kits and orchestral percussion "
        "become a 'Drums' track, timpani written at their pitch. It needs a source recorded at "
        "40 kHz or more (44.1/48 kHz files qualify) and is skipped otherwise, with the reason in "
        "info_json. The weights (20 MB) come from the model's collection, auto-downloaded for "
        "'dettato'."
    )


class DettatoTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DETTATO_MODEL",),
                "audio": ("AUDIO",),
                "min_velocity": ("INT", {"default": 1, "min": 1, "max": 127}),
            }
        }

    RETURN_TYPES = ("DETTATO_TRACKS", "STRING")
    RETURN_NAMES = ("tracks", "info_json")
    FUNCTION = "transcribe"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "Transcribe the audio to MIDI tracks, running the attached stages in pipeline order "
        "(evidence → grouper → drums); a stage that is not in the graph is bypassed. Mono is the "
        "channel average, at the file's own "
        "sample rate; the transcriber, the repairer and the grouper read 16 kHz internally, and "
        "the drum stage runs only when the source was recorded at 40 kHz or more (44.1 kHz and "
        "48 kHz files qualify; 8/16/24/32 kHz files and most phone recordings do not). The notes are "
        "performance MIDI: they keep the timing of the recording and are "
        "not quantized to a beat. min_velocity drops quieter notes (1 keeps everything). "
        "Outputs: tracks for the Dettato Piano Roll and the MIDI writers below, and a JSON summary "
        "(track names and note counts, the tuning estimate, which drum model ran). GPU timing: a "
        "4-minute song takes about 15 s on an RTX 5060; the CPU works but is roughly 6x slower."
    )

    def transcribe(self, model: dict, audio: dict, min_velocity: int):
        mono, rate = audio_to_mono(audio)
        chunks = 1
        if model["grouper"] is not None:
            chunks += 1
        if model["drums"] is not None and rate >= MIN_SOURCE_RATE:
            chunks += 1
        progress = ProgressBar(chunks)
        payload = run_transcription(model, audio, progress)
        payload = make_tracks_payload(
            filter_min_velocity(payload["tracks"], min_velocity), payload["seconds"], payload["source"], payload["info"]
        )
        mm.soft_empty_cache()
        return (payload, summary_json(payload))


class DettatoPianoRoll:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("DETTATO_TRACKS",),
                "px_per_ms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_seconds": ("INT", {"default": 0, "min": 0, "max": 36000}),
                "show_drum_panel": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("piano_roll",)
    FUNCTION = "render"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "The piano roll of the transcription: the upper panel is the 88-key range (G7..A0, C rows "
        "shaded), the lower panel the GM drum keys 27-87; tracks are coloured in track order, "
        "drums in red, note brightness follows velocity, and a line marks every second. "
        "px_per_ms is the horizontal scale (0 = fit the song into about 1400 px); max_seconds "
        "caps the visible length (0 = the whole song). Feed the result to a Preview Image node."
    )

    def render(self, tracks: dict, px_per_ms: float, max_seconds: int, show_drum_panel: bool):
        roll = render_piano_roll(tracks["tracks"], tracks["seconds"], px_per_ms, max_seconds, show_drum_panel)
        return (torch.from_numpy(roll)[None].contiguous(),)


class DettatoSaveMIDI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("DETTATO_TRACKS",),
                "filename_prefix": ("STRING", {"default": "dettato/midi"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "Write the transcription to ComfyUI's output folder as .mid (one General MIDI track per "
        "instrument, named after it, 120 BPM with times in seconds). The JSON summary (track "
        "names and note counts, the tuning estimate, the drum-model decision) is the info_json "
        "output of the Transcribe node; save it with a text node when needed."
    )

    def save(self, tracks: dict, filename_prefix: str):
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.mid")))
        path = Path(folder) / f"{filename}_{counter:05d}.mid"
        write_midi(tracks, path)
        n_notes = sum(len(np.asarray(t.notes, np.float64).reshape(-1, 4)) for t in tracks["tracks"])
        LOGGER.info("Dettato: wrote %s (%d tracks, %d notes).", path, len(tracks["tracks"]), n_notes)
        # an output node must return a tuple; () = no outputs (None breaks ComfyUI's merge_result_data)
        return ()



class DettatoInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("DETTATO_TRACKS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_json",)
    FUNCTION = "info"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "The transcription summary as JSON: source, length in seconds, per-track name / General "
        "MIDI program / is_drum / note count, the note statistics (transcribed, dropped by the "
        "repairer, inserted by the repairer, grouped), the tuning estimate in cents and the drum "
        "stage's decision (hits and router value, or why it was skipped). Use it to branch or to "
        "save the run's metadata with another text node."
    )

    def info(self, tracks: dict):
        return (summary_json(tracks),)


class DettatoToYuE2ABC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("DETTATO_TRACKS",),
                "mode": (["melody", "full"], {"default": "melody"}),
                "melody_track": ("STRING", {"default": ""}),
                "ins_track": ("STRING", {"default": ""}),
                "key": ("STRING", {"default": "auto"}),
                "tempo": ("INT", {"default": 0, "min": 0, "max": 500}),
                "meter": (["4/4", "3/4", "6/8", "7/8"], {"default": "4/4"}),
                "grid": (["1/16", "1/32"], {"default": "1/16"}),
                "octave": (["auto", "-2", "-1", "0", "1", "2"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc_score", "abc_summary")
    FUNCTION = "convert"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "Convert the transcription to the native two-voice ABC score that ComfyUI's built-in "
        "YuE2 Generate Music node renders: wire the abc_score output to that node's abc input and "
        "YuE2 regenerates the song from your transcribed melody. mode: 'melody' (the default and "
        "the recommended cover path) leaves the score chord-free, so YuE2 keeps your melody and "
        "writes its own accompaniment; 'full' derives chord symbols from the non-melody tracks. "
        "melody_track: the exact name of the track to use as the melody (empty = pick the most "
        "lead-like pitched track automatically - a voice/vocal track wins, a bass track is never "
        "picked); ins_track: a second pitched track for the Ins voice (empty = the busiest "
        "remaining line above G3 that is not a bass; 'none' = the Ins voice is rests). key: 'auto' "
        "runs key-finding on the melody, or a standard key such as 'Am' or 'Db'; tempo: 0 fits a "
        "grid to the onsets (drum hits weighted double), or a fixed BPM. octave: 'auto' moves a "
        "line by whole octaves into the C4..A#5 window when its middle pitch lies outside it, "
        "because YuE2 sings the octave it is given; an integer transposes both lines. The output "
        "is the exact dialect YuE2 and SheetSage2 use, so you can inspect it, correct a few notes "
        "in any text editor and feed the edited text to YuE2. See the README's YuE2 section."
    )

    def convert(self, tracks: dict, mode: str, melody_track: str, ins_track: str, key: str,
                tempo: int, meter: str, grid: str, octave: str = "auto"):
        meta = {}
        abc = tracks_to_abc(
            tracks["tracks"], tracks["seconds"], mode=mode, melody_track=melody_track,
            ins_track=ins_track, key=key, tempo=tempo, meter=meter, grid=grid, octave=octave,
            out_meta=meta,
        )
        return (abc, json.dumps(meta, indent=2, sort_keys=True))


class DettatoSaveABC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"default": "", "multiline": True}),
                "filename_prefix": ("STRING", {"default": "dettato/abc"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "music/Dettato"
    DESCRIPTION = (
        "Write an ABC score to ComfyUI's output folder as <prefix>_XXXXX.abc (e.g. the "
        "abc_score output of the To YuE2 ABC node), numbered so runs never overwrite each other. "
        "Use it to keep the run's symbolic record, or to open the score in an editor, correct a "
        "few notes and feed the edited text to YuE2."
    )

    def save(self, abc: str, filename_prefix: str):
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.abc")))
        path = Path(folder) / f"{filename}_{counter:05d}.abc"
        path.write_text(abc, encoding="utf-8")
        LOGGER.info("Dettato: wrote %s (%d chars).", path, len(abc))
        return ()


NODE_CLASS_MAPPINGS = {
    "DettatoLoader": DettatoLoader,
    "DettatoEvidence": DettatoEvidence,
    "DettatoGrouper": DettatoGrouper,
    "DettatoDrums": DettatoDrums,
    "DettatoTranscribe": DettatoTranscribe,
    "DettatoPianoRoll": DettatoPianoRoll,
    "DettatoSaveMIDI": DettatoSaveMIDI,
    "DettatoInfo": DettatoInfo,
    "DettatoToYuE2ABC": DettatoToYuE2ABC,
    "DettatoSaveABC": DettatoSaveABC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DettatoLoader": "Dettato Loader",
    "DettatoEvidence": "Dettato Evidence",
    "DettatoGrouper": "Dettato Grouper",
    "DettatoDrums": "Dettato Drums",
    "DettatoTranscribe": "Dettato Transcribe to MIDI",
    "DettatoPianoRoll": "Dettato Piano Roll",
    "DettatoSaveMIDI": "Dettato Save MIDI",
    "DettatoInfo": "Dettato Info",
    "DettatoToYuE2ABC": "Dettato To YuE2 ABC",
    "DettatoSaveABC": "Dettato Save ABC",
}
