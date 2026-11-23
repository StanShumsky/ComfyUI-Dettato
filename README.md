# ComfyUI-Dettato

ComfyUI nodes for [KobiMusic's dettato](https://github.com/kobimusic/dettato) - a multi-instrument
audio-to-MIDI transcriber. Give it a recording - a band, an orchestra, a jazz trio, a pop song
with vocals - and it writes a MIDI file with one track per instrument. Tracks are named after
73 instrument classes, each with a General MIDI program, plus a `Drums` track covering drum
kits and orchestral percussion, with timpani written at their pitch.

The whole chain is small: 28.1 M parameters - 113 MB in float32 (about half that in
bf16/fp16), and it runs at whatever precision you pick (the default follows
ComfyUI's session dtype, with fp32 as the released option). For comparison, the
next-best open transcriber (MuScriptor) has 1.3 B parameters, and dettato
outperforms it at about 1/65th the size.
(MuScriptor) has 1.3 B parameters, and dettato outperforms it at about 1/65th the size.

The pack also drives ComfyUI's built-in YuE2 song generator: the **Dettato To YuE2 ABC** node
turns a transcription into the ABC score YuE2 renders, so a recording can be re-sung as a
complete new song with vocals and accompaniment (see [Covers with YuE2](#covers-with-yue2)).

## Install

**With ComfyUI-Manager** (the easy way): *Manager - Install Custom Nodes* - search for
**ComfyUI-Dettato** - Install. The registry name is `shumsky/ComfyUI-Dettato`; Manager installs
the requirements as well.

**Manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-Dettato
pip install -r ComfyUI-Dettato/requirements.txt
```

Restart ComfyUI.

Model weights download from Hugging Face on first use, into `ComfyUI/models/dettato/`
(transcriber 61 MB, evidence 19 MB, grouper 12 MB, drums 20 MB) - and only for the stages you
actually attach. You can also bring your own copies: any folder inside `models/dettato` that
holds the `dettato/` subfolder (`config.json` + `model.safetensors`) is offered in the Loader,
and the stage nodes read their stage from the same folder. A corrupt or truncated download is
detected and re-fetched once automatically.

## Quick start

```
Load Audio
  -> Dettato Loader -> Dettato Evidence -> Dettato Grouper -> Dettato Drums
  -> Dettato Transcribe to MIDI -> Dettato Save MIDI
```

- The middle stage nodes (Evidence, Grouper, Drums) are optional: attach the ones you want and
  leave the rest out (see [How the chain works](#how-the-chain-works)).
- On the first run the transcriber's weights (61 MB) download; each attached stage downloads
  its own weights the first time its node runs.
- The result is `<prefix>_XXXXX.mid` in `ComfyUI/output`: one General MIDI track per
  instrument, named after it. The Transcribe node's `info_json` output is the run summary
  (track names, note counts, tuning estimate).
- To look at the result, wire the `midi` output to **Dettato Piano Roll** and then to a
  **Preview Image** node.

On an RTX 5060 a 4-minute song takes about 15 seconds; the rest of the practical notes are
under [Good to know](#good-to-know).

## How the chain works

Dettato is four small models in a chain. Each stage is its own node; a stage that is not in
the graph is bypassed, and its weights are never downloaded.

| Stage | Node | What it does | Weights |
|---|---|---|---|
| Transcriber (always on) | Dettato Loader | Transcribes the notes | 61 MB |
| Evidence | Dettato Evidence | Re-checks the notes against the transcriber's own frame evidence - drops false notes, inserts missed ones | 19 MB |
| Grouper | Dettato Grouper | Splits the notes into instrument-named tracks, estimates the tuning | 12 MB |
| Drums | Dettato Drums | Writes the `Drums` track (drum kits, orchestral percussion) | 20 MB |

Chain the stages after the Loader: `Loader -> Evidence -> Grouper -> Drums -> Transcribe`.
The Transcribe node runs the attached stages in pipeline order (evidence - grouper - drums),
whatever the chain's shape.

## Nodes

All nodes are in the `music/Dettato` category.

| Node | Role |
|---|---|
| Dettato Loader | Loads the transcriber (always on) |
| Dettato Evidence / Grouper / Drums | The optional chain stages |
| Dettato Transcribe to MIDI | The main node: audio in, one merged MIDI out |
| Dettato Dedupe MIDI | Merges the duplicate tracks a stem separation produces |
| Dettato Quantize MIDI | Snaps note onsets to a tempo x grid |
| Dettato Piano Roll | Draws the transcription as an image |
| Dettato Save MIDI | Writes the `.mid` to the output folder |
| Dettato Info / MIDI to Text | The run summary as JSON / the score as plain JSON text |
| Dettato To YuE2 ABC / Save ABC | The ABC score for YuE2, and its file writer |

### Dettato Loader

Loads the transcriber - the first stage, always on.

- `model`: `dettato` (auto-download) or a local collection folder - a folder in
  `models/dettato` that holds stage subfolders (`dettato/`, `dettato-evidence/`,
  `dettato-grouper/`, `dettato-drums/`, each with `config.json` + `model.safetensors`). It
  must at least hold the transcriber.
- `precision`: `auto` (ComfyUI's session dtype - fp16 on CUDA, fp32 on CPU), `bf16`,
  `fp16` (half the weight memory), `fp32` (the released precision).
- `device`: `auto` (ComfyUI's device policy for auxiliary models - honors `--gpu-only`,
  falls back to the CPU on machines without usable VRAM), `cuda`, `mps`, `cpu`.


The weights ride in RAM and occupy the GPU only while a run is in progress -
ComfyUI's model management evicts them when other nodes need the VRAM, so the
chain composes with sampler and separation models. A transcription that runs
out of memory is retried once at half the window batch before raising an
actionable error.
### Dettato Evidence

Attaches the evidence stage: the transcriber's notes are re-checked against its own frame
evidence - false notes are dropped, missed notes inserted. Without it the raw notes are used.

### Dettato Grouper

Attaches the grouping stage: the notes are split into instrument-named tracks (73 classes,
each with a General MIDI program) and the recording's tuning is estimated. Without it the
output is one `Notes` track.

### Dettato Drums

Attaches the drum stage: drum kits and orchestral percussion become a `Drums` track, with
timpani written at their pitch. Without it there is no drum track.

### Dettato Transcribe to MIDI

The main node. It takes up to nine audio sources, runs the attached stages on each, and merges
the results into one MIDI.

The audio inputs grow as you use them: `audio_0` is shown first, and each connected slot
reveals the next one (`audio_1`, `audio_2`, ...), up to nine. Any ComfyUI `AUDIO` works -
**Load Audio**, a video's audio track, or one stem at a time from **Roformer To Audio** (the
usual separation-to-MIDI path; see [Working with ComfyUI-Roformer](#working-with-comfyui-roformer)).

- `min_velocity`: drops notes quieter than this, per source (1 keeps everything).
- `dedup`: drops redundant tracks before the merge. When the same instrument was transcribed
  from several stems of one mix, only the stem with the most notes of it is kept. On for a
  separated-stem run, off when the sources are different recordings.
- The audio is read as mono (the channel average) at the file's own sample rate. The
  transcriber, repairer and grouper read 16 kHz internally. The drum stage needs a source
  recorded at **40 kHz or more** (44.1 kHz and 48 kHz files qualify); for slower sources it
  is skipped, and the reason is in `info_json`.
- The sources' tracks merge in slot order: the first occurrence of a track name keeps it, and
  a repeat from a later slot is renamed `Name (slot)` - a DAW shows `Piano` and
  `Piano (audio_1)`, `Drums (audio_2)`, and so on. Stems of one recording are length-aligned
  by the separator, so the merged file is phase-correct.
- `midi` (DETTATO_MIDI): the merged tracks, the length, the source labels, the per-source
  info, and the `.mid` rendered once - the writers just persist it.
- `info_json` (STRING): the run summary - note statistics (transcribed / dropped / inserted /
  grouped), the common tuning estimate in cents, the per-source drum decisions and the track
  list.

### Dettato Dedupe MIDI

Fixes the duplication a stem separation leaves behind. When the same sound is transcribed
from several stems, the merge shows it once per stem - `Drums`, `Drums (audio_1)`,
`Drums (audio_2)` - with small onset shifts, so the drums (or any instrument split across
stems) sound doubled. This node pools the tracks of one instrument into a single track and
drops the duplicate notes: scanning in onset order, a note is a duplicate when the gap to the
last kept note of the same pitch is within `window_ms`, and the louder note of a duplicate
pair wins.

- `window_ms` (default 50): the onset window for same-pitch duplicates. 50 ms is the safe
  default - on a 6-stem run it removed all the tight drum duplicates while the main drum
  track kept 100% of its own fast repeats. Raise it to reach smeared copies shifted by
  100-300 ms, knowing that fast same-pitch repeats inside that gap will be collapsed. 0 pools
  the tracks without dropping anything.

Put it between Transcribe and Save MIDI; its outputs feed the same consumers.

### Dettato Quantize MIDI

Snaps every note onset to a musical grid: `tempo` (BPM) x `grid` division (`1/4`, `1/8`,
`1/16`, `1/32`) - 120 x `1/16` is a 125 ms 16th-note grid. A note closer to a grid line moves
to that line, and its length is preserved (the offset moves with the onset; notes under 2 ms
are stretched to that minimum, so nothing is zero-length).

It takes the `midi` output of Transcribe (or any other MIDI node) and writes the same payload,
so it slots in anywhere before Save MIDI. The stats land in `info.quantize`; if
`max_shift_ms` exceeds half a grid step, the tempo or the division is probably wrong.

### Dettato Piano Roll

Draws the transcription: the upper panel is the 88-key range (G7..A0, C rows shaded), the
lower panel the GM drum keys 27-87. Tracks are coloured in track order, drums in red, note
brightness follows velocity, and a line marks every second. Feed the result to a
**Preview Image** node.

- `px_per_ms`: the horizontal scale (0 fits the song into about 1400 px).
- `max_seconds`: caps the visible length (0 = the whole song).
- `show_drum_panel`: hides the drum panel.

### Dettato Save MIDI

Writes the `.mid` to the output folder as `<prefix>_XXXXX.mid` - one General MIDI track per
instrument, named after it, 120 BPM with times in seconds. It writes the `.mid` the
Transcribe node already rendered, so it just persists bytes.

### Dettato Info

The run summary as a JSON string - the same content as the Transcribe node's `info_json` -
for branching, or for saving the run metadata with another text node.

### Dettato MIDI to Text

The score as **generic JSON text**: `{seconds, tracks: [{name, program, is_drum, notes:
[[onset, offset, pitch, velocity], ...]]}`. Any library that reads JSON can use it, with no
MIDI parser. Save it with any text node (e.g. the core **Save Text**) or feed it to another
pack's text input. On drum tracks `program` is `null`.

### Dettato To YuE2 ABC

Turns the transcription into the native two-voice ABC score that ComfyUI's built-in YuE2
Generate Music node renders (ComfyUI v0.36.0 or later). Wire the `abc_score` output to that
node's `abc` input and YuE2 regenerates the song from the transcribed melody.

- `mode`: `melody` (default, the cover path) leaves the score chord-free, so YuE2 keeps the
  melody and writes its own accompaniment; `full` derives chord symbols from the non-melody
  tracks.
- `melody_track`: exact name of the track to use as the melody. Empty picks the most
  lead-like pitched track automatically - a voice/vocal track wins, a bass is never picked.
- `ins_track`: a second pitched track for the Ins voice. Empty picks the busiest remaining
  line above G3 that is not a bass; `none` makes the Ins voice rests.
- `key`: `auto` runs key-finding on the melody, or a standard key such as `Am` or `Db`.
- `tempo`: `0` fits a grid to the onsets (drum hits weighted double), or a fixed BPM.
- `octave`: `auto` moves a line by whole octaves into the C4-A#5 window when its middle pitch
  lies outside it - YuE2 sings the octave it is given, and the native cover scores live
  there. An integer `±1..2` transposes both lines. Shifts are pitch-class-neutral, so the key,
  chords and rhythm are untouched.
- `meter` (`4/4`, `3/4`, `6/8`, `7/8`) and `grid` (`1/16`, `1/32`) set the bar shape and the
  note resolution.
- `abc_score` (STRING): the score text, editable in any text editor.
- `abc_summary` (STRING): JSON with the resolved key and BPM, the chosen tracks, the bar
  count, the number of chord changes and the octave shifts applied.

### Dettato Save ABC

Writes an ABC score (e.g. the `abc_score` output above) to the output folder as
`<prefix>_XXXXX.abc`, numbered per run. Keep it as the symbolic record of the run, or open
the score, correct a few notes by hand, and feed the edited text to YuE2.

## Working with ComfyUI-Roformer

[ComfyUI-Roformer](https://github.com/StanShumsky/ComfyUI-Roformer) separates a recording into
named stems (vocals, accompaniment, bass, drums, ...); Dettato transcribes them to MIDI. Pick
each stem with **Roformer To Audio** and wire one into a Transcribe slot - up to nine stems
land in one file, phase-aligned by construction:

```
Load Audio -> Roformer Loader -> Roformer Separate
  -> Roformer To Audio (one per stem you want: vocals, accompaniment, drums, bass, ...)
  -> Dettato chain: Loader -> Evidence -> Grouper -> Drums
  -> Dettato Transcribe (one stem per audio slot, up to nine)
  -> Dettato Save MIDI
```

In this graph the Transcribe `dedup` toggle is on: when the same instrument was transcribed
from several stems, only the stem with the most notes of it survives the merge, so no
duplicated `Drums (audio_N)` tracks remain. For a different stage chain per stem, or one file
per stem, run separate Transcribe and Save MIDI nodes.

## Covers with YuE2

YuE2 is a full-song generator built into ComfyUI (v0.36.0 or later): it turns a style prompt,
lyrics and an optional symbolic score into a complete song with vocals and accompaniment.
**Dettato To YuE2 ABC** emits the native two-voice ABC dialect YuE2 uses (fixed eight-line
header, Vocal and Ins blocks of up to four bars, ties across barlines, full-measure `Z`
rests) - so the score can be inspected, corrected by hand and re-checked before generating:

```
Load Audio
  -> Dettato chain: Loader -> Evidence -> Grouper -> Drums -> Transcribe
  -> Dettato To YuE2 ABC (mode: melody)
  -> YuE2GenerateMusic (abc input; plus style and lyrics)
  -> Empty YuE2 Latent Audio (seconds) -> ... -> YuE2 VAE Decode -> Save Audio
```

In `melody` mode (the default) the score is chord-free: YuE2 keeps the transcribed melody and
composes the arrangement, harmony and vocals around it - the cover path its own docs
recommend for audio covers. Use `full` to seed the harmony from the transcription as well.

Because the transcription is performance MIDI, the score keeps the recording's timing; set
`key` and `tempo` by hand when you want a strict grid. The melody voice is the most lead-like
track (a vocal track wins), the Ins voice is the busiest remaining line above G3, and
`octave` keeps both lines in the register YuE2 sings best in. The YuE2 nodes are built in -
nothing extra to install, just keep ComfyUI up to date.

## Example workflows

Drag into ComfyUI (or use Workflow - Open):

- [audio2midi.json](example_workflows/audio2midi.json) - the plain path: Load Audio - full
  Dettato chain - Transcribe - Piano Roll + Save MIDI.
- [audio2midi_roformer.json](example_workflows/audio2midi_roformer.json) - the separation
  path: Load Audio - Roformer Separate - six x Roformer To Audio - Transcribe (dedup on) -
  Piano Roll + Save MIDI.

Both need a current ComfyUI for the built-in audio nodes. For a YuE2 cover, extend the graph:
Transcribe - Dettato To YuE2 ABC - the `abc` input of a built-in YuE2GenerateMusic node (see
[Covers with YuE2](#covers-with-yue2)).

## Good to know

- The output is **performance MIDI**: notes keep the timing of the recording and are not
  quantized to a beat. The **Dettato Quantize MIDI** node snaps them to a `tempo` x `grid`
  grid (note lengths are preserved) when you want the MIDI to sit on the grid, e.g. before
  writing sheet music.
- Instrument names are only roughly right: on real jazz about a quarter of a saxophone's or
  lead guitar's notes can land in the piano track, and synth-heavy music is named less
  reliably.
- The weakest step is the grouping: expect occasional track mix-ups on dense recordings.
- The tuning estimate comes from the grouper stage; without it in the graph, `tuning_cents`
  is null.
- In 0.3.0 the Transcribe node's output changed from `tracks` (DETTATO_TRACKS) to `midi`
  (DETTATO_MIDI). In a saved workflow, re-plug that one output - every node it feeds accepts
  both types, and legacy payloads still work everywhere.
- With the evidence and grouper stages both attached, the transcriber runs once: its single
  pass feeds both stages, on the audio retuned to A440 when it sits clearly off. Above about
  4 minutes on a CUDA GPU the pass's intermediate buffers move to host memory, so GPU memory
  stays bounded.
- Speed: on an RTX 5060 a 4-minute song takes about 15 s and a 6-minute one about 25 s, both
  peaking under 4 GB. The CPU works but is slow - a 15 s clip took 87 s, including model load.

## License

- The node code in this repository is Apache-2.0 (see `LICENSE`).
- The bundled [dettato/](dettato/) code is KobiMusic's, redistributed under its Apache-2.0
  license (see `dettato/LICENSE`).
- **The model weights are released under Apache-2.0, so commercial use is fine** (see the
  model cards for the licences of the training data; some of the grouper's and drum model's
  training datasets carry non-commercial terms).

## Credits

Nathan Mills / KobiMusic - [dettato](https://github.com/kobimusic/dettato),
[model collection](https://huggingface.co/collections/kobimusic/dettato),
[introducing dettato](https://kobi.music/research/dettato).
