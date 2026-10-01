# How to Analyse Compumedics and Other Cut EEGLAB Recordings

This guide shows you how to load an EEGLAB `.set` file that comes from the
Compumedics export pipeline (or any file that had data cut out before you
received it), stage it correctly, run detection, and compute sleep cycles.

For why such files need special handling, read
[About cut recordings and time bases](../explanation/cut-recordings-and-time-bases.md).

!!! warning "Results from earlier releases may be misaligned"
    Before 4.5.0, TurtleWave imported `etc.stages` as if it fitted a cut
    signal, so every stage after the first cut was on the wrong data. If you
    staged or detected on a cut file with an earlier release, re-annotate and
    re-detect. The explanation page lists the affected datasets.

## Load a top-level `.set` file

**Problem:** `wonambi.Dataset(path)` fails with `object 'EEG' doesn't exist` on a
Compumedics `.set` file.

**Solution:** Open the file with `open_dataset`. It reads both EEGLAB layouts
(a single `EEG` variable, or the fields stored as top-level variables) and
every format Wonambi supports.

```python
from turtlewave_hdEEG import open_dataset

dataset = open_dataset("sub-02dg_ses-1_task-psg_run-1_desc-clean_eeg.set")

print(dataset.header["chan_type"][:3])        # e.g. ['EEG', 'EEG', 'EEG']
print(dataset.header["reference"])            # {'ref': 'average', 'n_good': 220} or None
print(dataset.header["interp_channels"][:5])  # interpolated channel names, [] if none listed
```

In the GUI, choose the file in the Setup tab and click **Load Data**. A file
TurtleWave cannot read produces a short message naming what is missing from
the file (the sampling rate and channel list) and what was found at its top
level. If the `.set` points to a separate `.fdt` file that is not beside it,
the message names both files; copy the `.fdt` next to the `.set` and load
again.

Duplicate channel labels are renamed so every channel is unique (for example
the second `ECG` becomes `ECG_2`).

## Read the Dataset Information panel

After **Load Data**, the Setup tab's Dataset Information panel reports, in this
order:

- `File`, `Recording start` and `Recording end`. The end is the clock time of
  the last sample in the original recording, so it includes the removed data.
- `Signal duration`, the length of the signal in the file.
- `Removed data`, for example `107.7 min at 189 boundary events (original
  recording 429.5 min)`. `none` means the file is uncut.
- `Sampling rate` and `Channels`, for example `277 (257 EEG, 20 other)`. For a
  file without channel types the line reads `channel types not stated in the
  file; all listed as EEG`.
- `Reference`, as stored in the file and never changed by TurtleWave, for
  example `average of 220 of 257 EEG channels`.
- `Interpolated channels`, for example `37 (AF3, F3, F1, ... and 27 more)`, or
  `none stated in the file`.
- `EEG channels` and `Other channels` previews.

A line that cannot be read says `could not be read (see log)`; the other lines
still appear.

## List only EEG channels, and show the others when you need them

**Problem:** The channel lists are full of EOG, ECG, EMG and respiratory
channels that the sleep-event detectors are not built for.

**Solution:** Nothing to do. When the file types its channels, the
Spindle, Slow Wave, K-Complex and PAC tabs list only channels typed EEG, in
file order. "Add All >>" adds only the listed channels.

To list the others, tick **Show non-EEG channels (N)** on any detection tab.
The box is shared: ticking it on one tab ticks it on the other three. It is
unticked again each time you load a file, and hidden when the file has no
non-EEG channels.

A non-EEG channel that is already in the Selected list stays selected when you
untick the box. A run that includes one logs
`Note: 2 non-EEG channels selected: ECG, EMGChin.` The mastoid and reference
channels (`M1`, `M2`, `REF`) stay listed if the file types them as EEG.

## Recognise interpolated channels

**Problem:** You want to know which channels the cleaning pipeline
reconstructed from their neighbours before you trust them.

**Solution:** Look for the marks, or read the log.

- In the GUI's channel lists, an interpolated channel is shown in italics with
  the tooltip "Interpolated channel (reconstructed from neighbours by the
  cleaning pipeline)". The channel name itself is unchanged.
- In the review GUI, the Filters dock list appends ` ~` to the name (`Cz ~`).
- Selecting an interpolated channel for a run logs
  `Note: 1 interpolated channel selected: Cz (reconstructed from neighbours by the cleaning pipeline).`
- `detect_spindles`, `detect_slow_waves`, `detect_kcomplexes` and
  `analyze_pac` log one `WARNING` at the start naming the selected channels
  that are interpolated, and record them in the
  `detection_runs.interpolated_channels` column (a JSON list) of the database.

Interpolated channels are flagged, never blocked. Decide per analysis whether
to keep them. To check in code:

```python
from turtlewave_hdEEG.utils import interpolated_channels

print(interpolated_channels(dataset.header, ["Cz", "Fz", "Pz"]))  # e.g. ['Cz']
```

A reconstructed channel carries the signal of its neighbours, so a spindle or
slow-wave density on it is not an independent measurement.

## Stage a cut recording

**Problem:** You need sleep stages that fit the cut signal.

**Solution:** Run the usual annotation step. `add_stages_from_header` detects
the boundary events, decides how to read the hypnogram, and writes the stages.

```python
import logging

from turtlewave_hdEEG import XLAnnotations, open_dataset

logging.basicConfig(level=logging.INFO)

dataset = open_dataset("sub-02dg_ses-1_task-psg_run-1_desc-clean_eeg.set")
annotations = XLAnnotations(dataset, "wonambi/sub-02dg_annotations.xml", rater_name="Anon")
annotations.add_stages_from_header()
```

Read the `turtlewave_hdEEG.annotation` log lines. One says which source was
used and how well the file's stage events agreed with it:

```text
Staging source: etc.stages through the boundary time map (...) -> 755 exact epochs over 19308 s; stage events agree on 646 of 646 through the time map and ...
```

The possible sources are:

| Source in the log | Meaning |
|---|---|
| `etc.stages (header, as stored)` | Uncut file, or `etc.stages` already fits the cut signal. A 30 s grid, as in earlier releases. |
| `etc.stages through the boundary time map` | Cut file with a full-night `etc.stages`. Exact epochs. |
| `stage events` | No usable `etc.stages`; the stage events in the file are used. Exact epochs, with `Undefined` in gaps. |

Then check the warnings:

- **Stage events disagree** (`N of M stage events (x %) disagree ...`): more
  than 1 % of the file's stage events contradict the chosen reading. Check the
  boundary events before trusting the staging. Above 20 % the reading is
  rejected and the stage events are used instead.
- **No stage events to tell the readings apart**: the file has no stage events,
  and `etc.stages` would fit both the full night and the cut signal. It is
  read as the full night. Check the staging if `etc.stages` was rescored after
  the cut.
- **Staging not imported ... ERROR**: nothing aligned with the signal, and no
  stage events to fall back on. Nothing is written.

### The timeline sidecar

When the staging uses the time map or the stage events, TurtleWave writes
`<annotation xml stem>_timeline.json` beside the annotation XML. It holds the
full-night hypnogram, the boundary table and a copy of the exact epochs.
Sleep-cycle steps read it because they have no dataset. Keep it with the XML,
and copy both together.

If you re-annotate with a source that needs no sidecar, TurtleWave warns that
an older sidecar beside the XML is stale; delete it.

## Run detection

**Problem:** You want spindles, slow waves, K-complexes or PAC on a cut file.

**Solution:** Run them exactly as on any other file. Stage selection, the
per-event stage (`epoch_stage`) and the artefact-free density denominators all
follow the exact epochs.

```python
from turtlewave_hdEEG import ParalEvents

events = ParalEvents(dataset=dataset, annotations=annotations)
events.detect_spindles(
    method="Moelle2011",
    chan=["Cz", "Fz", "Pz"],
    frequency=(11, 16),
    stage=["NREM2", "NREM3"],
    db_path="wonambi/neural_events.db",
    subject="sub-02dg",
)
```

Expect the `WARNING` for any interpolated channel among `chan`. Filters
run across splice joins, so an event straddling a join can be an artefact of
the join; TurtleWave does not remove such events in this release.

## Compute sleep cycles on the full night

**Problem:** Cycle lengths and stage minutes must come from the whole night,
not from the cut fragment.

**Solution:** Use `timeline='auto'` (the default).

```python
from turtlewave_hdEEG import CustomAnnotations, finalize_cycles_and_durations

annot = CustomAnnotations("wonambi/sub-02dg_annotations.xml")
cycles_by_method = finalize_cycles_and_durations(
    annot,
    "wonambi/neural_events.db",
    subject="sub-02dg",
    timeline="auto",
)
```

With a sidecar beside the XML, cycles and stage durations are computed on the
full-night hypnogram, and the cycle boundaries are then converted to the cut
file's time so events and XML markers line up with the signal. With
`plot=True` the figure shows the full night. Without a sidecar and with uniform
30 s epochs, nothing changes from earlier releases; without a sidecar and with
variable-length epochs, the call raises `ValueError`. A file staged from its
stage events has no full-night hypnogram, so no cycles are computed; see
the failure table below.
See [How to Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md#cycles-on-a-cut-recording)
for the options, the failure modes and the new database columns.

## Troubleshooting

### The wrong staging source was used

**Problem:** The log names a source you did not expect.

**Solution:** Read the two scores in the log line (how many stage events agree
through the time map and as stored). The better-agreeing reading wins. If the
boundary events are wrong, the time map is wrong; ask whoever cut the file.

### The Dataset Information panel shows `Removed data: none` on a file you know was cut

**Problem:** The file has no `boundary` events, for example because the cut
was made by another tool.

**Solution:** TurtleWave can only see cuts that EEGLAB recorded as `boundary`
events. Without them the file is treated as uncut, and a
`stage events disagree` warning may be the only sign.

## See also

- [About cut recordings and time bases](../explanation/cut-recordings-and-time-bases.md)
- [How to Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md)
- [Reference: timeline](../reference/api/timeline.md) and [Reference: eeglab_io](../reference/api/eeglab_io.md)
