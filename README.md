# ComfyUI-Dettato

ComfyUI nodes for [KobiMusic's dettato](https://github.com/kobimusic/dettato) — a multi-instrument
audio-to-MIDI transcriber. Give it any recording (a band, an orchestra, a jazz trio, a pop song
with vocals) and it writes a MIDI file with one track per instrument, each named after its
instrument (73 classes, each with a General MIDI program), plus a drum track that covers drum
kits and orchestral percussion, with timpani written at their pitch.

It outperforms the next-best open transcriber (MuScriptor, 1.3 B parameters) at about 1/65th
the size.

The whole chain is small — 28.1 M parameters, 113 MB in fp32 — and runs at the model's released
float32 precision, with no precision setting to configure.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-Dettato
pip install -r ComfyUI-Dettato/requirements.txt
```

Restart ComfyUI. Each stage's weights download automatically from Hugging Face into
`ComfyUI/models/dettato/` on first use of that stage (transcriber 61 MB, evidence 19 MB,
grouper 12 MB, drums 20 MB). You can also drop your own copies into that folder: any subfolder
holding the transcriber folder (`dettato/`, with its `config.json` and `model.safetensors`) is
listed in the loader, and the stage nodes read their stage from the same folder. Corrupt or
truncated downloads are detected and re-fetched once automatically.

## Nodes

All nodes are in the `music/Dettato` category.

### Dettato Loader
Loads the transcriber — the first stage, always on.
- `model`: `dettato` (auto-download) or a local collection folder in `models/dettato` — a folder
  holding stage subfolders (`dettato/`, `dettato-evidence/`, `dettato-grouper/`,
  `dettato-drums/`, each with its `config.json` and `model.safetensors`); it must at least hold
  the transcriber.
- `device`: `auto` (ComfyUI's device), `cuda`, `mps`, `cpu`.

### Dettato Evidence
Attaches the evidence-repair stage: the transcriber's notes are re-checked against its own frame
evidence — false notes dropped, missed ones inserted. Without it the raw transcriber notes are
used. Its weights (19 MB) download on first use, or come from the model's collection folder.

### Dettato Grouper
Attaches the grouping stage: the notes are split into instrument-named tracks (73 classes, each
with a General MIDI program) and the recording's tuning is estimated. Without it the output is
one `Notes` track. Its weights (12 MB) download on first use, or come from the collection.

### Dettato Drums
Attaches the drum stage: drum kits and orchestral percussion become a `Drums` track, timpani
written at their pitch. Without it there is no drum track. Its weights (20 MB) download on first
use, or come from the collection.

Chain the stage nodes after the Loader — `Loader → Evidence → Grouper → Drums → Transcribe`.
The Transcribe node runs the attached stages in pipeline order (evidence → grouper → drums),
whatever the chain's shape; a stage that is not in the graph is bypassed, and its weights are
never downloaded or loaded.

### Dettato Transcribe to MIDI
Takes ComfyUI's `AUDIO` type (from **Load Audio**, or the audio track of a loaded video) and runs
the attached stages.
- Mono is the channel average, at the file's own sample rate. The transcriber, repairer and
  grouper read 16 kHz internally; the drum stage needs a source recorded at **40 kHz or more**
  (44.1 kHz and 48 kHz files qualify) and is skipped otherwise — the reason is in `info_json`.
- `min_velocity`: drop notes quieter than this (1 keeps everything).
- Outputs:
  - `tracks` (DETTATO_TRACKS): the instrument tracks, for the Piano Roll and the MIDI writers below.
  - `info_json` (STRING): track names and note counts, note statistics (transcribed / dropped /
    inserted / grouped), the tuning estimate in cents and the drum stage's decision.

### Dettato Piano Roll
The piano roll of `tracks`: the upper panel is the 88-key range
(G7..A0, C rows shaded), the lower panel the GM drum keys 27–87; tracks are coloured in track
order, drums in red, note brightness follows velocity, a line marks every second.
`px_per_ms` is the horizontal scale (`0` = fit the song into about 1400 px), `max_seconds` caps
the visible length (`0` = whole song), and `show_drum_panel` hides the drum panel.

### Dettato Save MIDI
Writes the transcription to ComfyUI's output folder as `<prefix>_XXXXX.mid` — one General MIDI
track per instrument, named after it, 120 BPM with times in seconds. The summary (track
names and note counts, tuning estimate, drum-stage decision) is the `info_json` output of the
Transcribe node — save it with a text node when you need the JSON on disk.

### Dettato Info
The transcription summary as a JSON string (the same content as the transcribe node's
`info_json`), for branching or saving run metadata with another text node.

## Example workflows

Drag into ComfyUI (or use Workflow → Open):

- [example_workflows/audio2midi.json](example_workflows/audio2midi.json) — Load Audio → Loader →
  Evidence → Grouper → Drums → Transcribe → Piano Roll → Preview Image, plus Save MIDI. Keep
  ComfyUI up to date for the built-in audio nodes.

## Notes

- The output is **performance MIDI**: notes keep the timing of the recording and are not
  quantized to a beat. If you want something to play from, like sheet music, quantize afterwards
  (e.g. in your DAW or with a beat-detection pass).
- Instrument names are only roughly right: on real jazz about a quarter of a saxophone's or lead
guitar's notes can land in the piano track, and synth-heavy music is named less reliably.
- The weakest step is the grouping: expect occasional track mix-ups on dense recordings.
- The tuning estimate comes from the grouper stage; without the Dettato Grouper node in the
  graph, `tuning_cents` is null.
- With the evidence and grouper stages both attached, the transcriber runs once — its single
  pass feeds both stages — on the audio retuned to A440 when it sits clearly off it. Above
  about 4 minutes on a CUDA GPU the pass's intermediate buffers move to host memory, so GPU
  memory stays bounded.
- Speed: on an RTX 5060 a 4-minute song takes about 15 s and a 6-minute one about 25 s, both
  peaking under 4 GB. The CPU works but is slow (a 15 s clip took 87 s including model load).

## License

- The node code in this repository is Apache-2.0 (see `LICENSE`).
- The bundled [dettato/](dettato/) code is KobiMusic's, redistributed under its Apache-2.0 license
  (see `dettato/LICENSE`).
- **The model weights are released under Apache-2.0, so commercial use is fine** (see the model
  cards for the licences of the training data; some of the grouper's and drum model's training
datasets carry non-commercial terms).

## Credits

Nathan Mills / KobiMusic — [dettato](https://github.com/kobimusic/dettato),
[model collection](https://huggingface.co/collections/kobimusic/dettato),
[introducing dettato](https://kobi.music/research/dettato).