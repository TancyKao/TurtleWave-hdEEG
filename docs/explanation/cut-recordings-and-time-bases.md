# About Cut Recordings and Time Bases

This page explains why a recording that had data cut out of it cannot use its
stored hypnogram as-is, and how TurtleWave 4.5.0 handles that. For the steps,
see [How to analyse Compumedics and other cut EEGLAB recordings](../how-to/analyse-compumedics-recordings.md).

## Why `etc.stages` no longer fits a cut file

A sleep scorer stages the whole night, and the Compumedics and similar export
pipelines store that full-night hypnogram in the EEGLAB structure as
`etc.stages`, one stage code per 30-second epoch counted from the start of the
recording.

Later steps then clean the signal. EEGLAB's `pop_select` removes stretches of
data (movement, noise, bad channels' bad moments) and splices the remaining
samples together. At each splice it inserts a `boundary` event whose duration
is the number of samples removed there. The stage events in the file (the
`wake`, `n1`, `n2`, `n3`, `rem` markers) are ordinary events, so EEGLAB shifts
them with the data. `etc.stages` is not an event. Nothing shifts it, and
neither EEG_Processor nor neuvo_cleanline updates it afterwards.

The result is a file whose signal is shorter than the night, whose stage
events sit at the right place on the cut signal, and whose `etc.stages` still
describes the full night. Importing `etc.stages` as if it fitted the signal
puts every stage after the first cut on the wrong data, by the total length
removed so far.

## The two time bases

Two clocks are in play, and TurtleWave keeps both explicit.

- The **cut time base** is the time of the signal as stored. Sample 0 is
  second 0, and the splices are invisible. Everything that reads the signal
  (detectors, the review GUI, Wonambi's `fetch`) lives here.
- The **original time base** is the time of the recording before anything was
  removed. The scorer's hypnogram, and so `etc.stages`, lives here.

A time in one base converts to the other by adding or subtracting the data
removed before it.

## The boundary time map

`turtlewave_hdEEG.timeline.RecordingTimeline` reads the `boundary` events and
builds the conversion. For each splice it records where it falls on the cut
signal and how many seconds were removed there. EEGLAB puts a boundary
half-way between two samples, so a splice sits at `(latency - 0.5) / s_freq`
on the cut signal.

`cut_to_original` adds the removal of every splice at or before the time.
`original_to_cut` subtracts it, and a time that was itself removed clamps to
the splice where it vanished. From the map, `stage_intervals_cut` cuts each
original epoch at the splices and returns the surviving pieces on the cut time
base, each tagged with the index of the original epoch it came from.

The map is only as good as the boundary events. A cut made without recording a
`boundary` event is invisible to it.

## How the reading of `etc.stages` is chosen

When a file has boundary events that removed data, `etc.stages` can be one of
two things: the full night (read through the time map), or already rewritten
on the cut time base (read as stored). Neither known cleaning pipeline
rewrites it, so the full-night reading is the default, but TurtleWave checks
rather than assumes.

The check uses the stage events, which are correct on the cut signal. For each
reading, it takes every stage event, finds the `etc.stages` epoch that reading
puts there, and counts the events whose code differs or whose onset is more
than half a second off the epoch grid. The reading with the smaller share of
disagreeing events wins; a tie goes to the time map. A reading is a candidate
only if it is plausible: the time map when `etc.stages` ends where the night
ends (to within one sample per splice), or earlier if the stage events back it;
the as-stored reading when `etc.stages` is no longer than the signal plus one
epoch.

The winner is then held to its own score. More than 1 % disagreement logs a
warning to check the boundary events. More than 20 % rejects it, and the stage
events themselves become the source. If neither reading is possible and there
are no stage events, nothing is imported and an error is logged, because
importing anyway would misalign every stage after the first cut.

Without stage events, there is no evidence to weigh. The time map wins when
the lengths fit, and a warning is logged when both readings fit and at least
half an epoch was removed.

A file with no stage events that was rescored on the cut signal after a
cut of under about a minute still reads as a full night, with a warning.
Nothing can detect that case without stage events.

## Exact epochs

The obvious way to stage a cut file keeps a 30-second grid on the cut signal
and labels each grid epoch with the stage under it. It is wrong in a
measurable way. After the first cut whose length is not a multiple of 30 s,
every grid epoch is out of phase with the scorer's epochs and straddles two of
them. Measured on one night, that mislabels 3.6 to 4.2 % of N2 time and about
17 % of N1 and Wake time. Densities in N2 and N3 are only mildly biased by
this, but N1 and Wake are not usable.

So TurtleWave writes **exact epochs**: each surviving piece of each original
epoch becomes its own epoch, carrying that original epoch's stage. Most are
30 s long. The ones next to a splice are shorter. Wonambi's own
`import_staging` can only write a fixed grid, so
`XLAnnotations._write_exact_epochs` writes the `<epoch>` elements directly, in
the layout `import_staging` produces.

### Whole-second rounding

Wonambi's annotation file stores epoch starts and ends as integers, and
`set_stage_for_epoch` matches an epoch by `int()` of its start. Exact epochs
therefore sit on whole seconds. Every edge goes through one rule, round half
up (`round_edge`), so two pieces that share an edge before rounding share it
after, and the epochs stay contiguous. The final edge is `int(n_samples /
s_freq)`, the `last_second` Wonambi records.

This is why 1-second epochs exist: a piece between 0.5 and 1.5 s long rounds
to 1 s, and 1 s is the shortest epoch. A piece that rounds to zero length (a
sliver under about 0.5 s) is dropped, not merged, and its time goes to
whichever neighbour the rounding gives it. That relabels at most 0.5 s per
splice edge. On the first test night, 865 raw pieces gave 110 slivers dropped
and 755 exact epochs (569 of 30 s, 31 of 1 s, 26 of 2 to 5 s).

The epochs must tile the signal with no gaps, because Wonambi looks events up
by bisecting on epoch starts. A gap becomes an `Undefined` epoch.

## The sidecar and its validation

The cycle steps have no dataset, only the annotation XML. The XML on its own
holds the cut-time epochs but not the full-night hypnogram. So when the
staging uses the time map or the stage events, TurtleWave writes
`<annotation xml stem>_timeline.json` beside the XML. It holds the full-night
stages, the boundary table, `last_second`, and `cut_epochs`, a copy of the
epochs written to the XML.

A sidecar is only worth using if it still describes its XML. `load_sidecar_for`
checks that the file exists, that it names this annotation file, that its schema
is 2 or later, and that its `cut_epochs` and `last_second` equal the XML's
epochs. Any difference raises `SidecarMismatchError`, listing the first three
differing epochs and telling you to re-run the annotation step. It fails
loudly on purpose: a sidecar that outlived a rescoring would silently give
cycles from the wrong hypnogram.

## Full-night cycles, converted to cut time

Sleep cycles are defined on the whole night: a cycle is NREM then REM, and
whether a wake bout ends an NREM period depends on the scorer's epochs, not on
what survived cleaning. Running the detector on the cut fragment would
shorten and merge cycles. So with a sidecar, `detect_cycles` and the stage
durations run on the full-night hypnogram at 30 s, and the `*_min` durations
are full-night minutes.

The seconds columns are then converted to the cut time base so that they line
up with the signal, the XML cycle markers and the events: each of
`nrem_start_sec`, `nrem_end_sec` and `rem_end_sec` is converted with
`original_to_cut` and snapped to the nearest exact-epoch edge. Snapping matters
because Wonambi's `get_epochs(time=...)` needs a bound to contain whole
epochs. The original values are kept in `*_orig` columns and `time_base`
records `'cut'`. XML cycle markers and `events.cycle` use the converted
seconds. A cycle whose converted span is empty because the cut removed nearly
all of it gets no XML marker, and a warning says so.

One consequence: on a cut file, `cycle_dur_min` is the full-night duration and
no longer equals the difference of the converted seconds columns. Both are
right, in different time bases.

## Per-cycle coverage

A detector only sees the cut, artefact-free part of a cycle. A cycle that lost
most of its N2 to cleaning has a density computed on little data, and the
density does not say so. `analysed_time_cycles` stores, per cycle and stage,
the full-night seconds, the seconds removed by the cut, the seconds masked by
artefact rejection, the seconds actually analysed, their ratio (`coverage`),
and whether that is under a floor (`coverage_floor_min`, 5 minutes by
default). Treat a low-coverage cycle with caution before pooling it.

The rows are per single stage (`NREM1`, `NREM2`, `NREM3` and `REM` unless you
name others), and `coverage` is capped at 1 because exact epochs are rounded to
whole seconds and can sum to a few seconds more than the original. `coverage`
is NULL when the cycle has no full-night time in that stage. The table only
has rows when the sidecar holds a full-night hypnogram.

## Recordings staged from stage events

When the staging came from the stage events (see above), the sidecar's
`fullnight_stages` is empty: the scorer's full-night hypnogram is not
available. Sleep cycles are then not computed, because cycle detection counts
30 s epochs of the whole night and the cut file's variable-length epochs are
not that. One `ERROR` is logged, `sleep_cycles`, `events.cycle` and
`analysed_time_cycles` are left empty, and any earlier rows for the subject are
cleared.

Stage durations are still written. They are summed from the cut file's epoch
durations and stored with `time_base='cut'`, so they cover the surviving data
only, not the night. Their `epoch_length` column holds the **median** epoch
duration of the file. It describes the file; it is not a length to multiply an
epoch count by. The same holds for any variable-epoch file: the minutes columns
are sums of real durations, and `epoch_length` is only a description. A
full-night or uniform-epoch row stores 30.

The stage events give each epoch an onset but no end, so each one is taken to
end at the next stage event, 30 s after its onset, or at the next splice,
whichever is first. The splice rule matters because the data after a splice
comes from a later part of the night. If its own stage event was removed with
the cut, nothing says what stage it is, and it is left `Undefined` instead of
inheriting the stage before the splice. The staging log line reports how many
seconds were left Undefined this way.

## What was wrong before 4.5.0

Before 4.5.0, `etc.stages` was imported as stored whenever its length was
close to the signal's, and otherwise as a 30 s grid. On a cut file, staging
after the first cut was misaligned, so every stage-dependent result was
computed on a mix of stages: stage selection for detection, the per-event
stage, stage durations and sleep cycles. Respiratory-event rejection also
matched fewer labels than it should have.

Datasets known to be affected:

- Emotion, subjects 16js and 18sb.
- MCI, the `clean_rebuilt` files.
- LocalSleep excerpts.

Re-annotate and re-detect these before analysing them. Do not pool new
results with older rows in the same database: the stage labels, and for
Massimini-type detectors and K-complexes the event counts, differ. Uncut files
are unchanged byte for byte.

## Why not edit cut-file annotations in Wonambi's scoring GUI

Wonambi's own scoring GUI assumes fixed-length epochs. Its epoch navigation,
and its rescoring, step by a fixed epoch length, so the 1 to 5 s epochs of a
cut file do not fit it, and a save from there can change the epoch list. A
rescored XML no longer matches its sidecar, and TurtleWave then refuses it,
by design. Fix staging on a cut file by re-running the annotation step, not by
editing the XML.

## See also

- [How to analyse Compumedics and other cut EEGLAB recordings](../how-to/analyse-compumedics-recordings.md)
- [How to Finalize Sleep Cycles & Stage Durations](../how-to/detect-sleep-cycles.md)
- [Reference: timeline](../reference/api/timeline.md)
