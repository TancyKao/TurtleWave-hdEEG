# How to Finalize Sleep Cycles & Stage Durations

This guide shows you how to run the **post-detection finalize step**: detect
NREM-REM sleep cycles from a scored hypnogram and populate `neural_events.db`
with per-cycle boundaries, per-stage sleep durations, and cycle-tagged events.

This is not a peer detector alongside spindles, slow waves, or K-complexes —
it doesn't detect anything from the raw EEG signal. Run it **once, after**
event detection has already populated `neural_events.db` (or as a standalone
backfill against a database you detected into previously); it reads only the
scored hypnogram and the events already in the database.

## When to use this

**Problem:** You have detected events (spindles, slow waves, K-complexes)
across a night, but `events.cycle` is empty, there's no per-cycle NREM/REM
breakdown, and no per-stage minute totals — every downstream analysis that
needs "which cycle did this spindle fall in" or "how many minutes of N3" has
nothing to query.

**Solution:** Call `finalize_cycles_and_durations`. A single call detects
sleep cycles under both definitions, writes them to the `sleep_cycles` table,
computes per-stage minutes into `stage_durations`, tags every event in
`events.cycle`, and writes cycle markers back into the annotation XML so the
review GUI shows cycle bands. All writes are idempotent, so re-running is
always safe.

## The two cycle definitions

Two NREM-REM cycle definitions are supported via `method`:

- **`'2022'`** (default `tag_method`) — NREM-based. A cycle is one contiguous
  NREM period plus the inter-NREM (REM) segment that follows it. Short
  awakenings are absorbed into NREM and too-short NREM runs are dropped.
  Always yields cycles even when REM scoring is sparse.
- **`'1979'`** — Feinberg/Floyd-Feinberg. As above, but a cycle only closes
  when a qualifying REM period follows the NREM block (the first cycle needs
  REM of at least one epoch, later cycles at least `rem_min` epochs). NREM
  periods not followed by qualifying REM are merged into the next cycle.

By default `finalize_cycles_and_durations` detects and stores **both**
definitions side by side in `sleep_cycles` (keyed by `(subject, method)`), but
only `tag_method` (`'2022'` by default) owns `events.cycle` and the XML cycle
markers, since a single event can only carry one cycle number.

## Run it after detection (fresh database)

Mirror
[`examples/hdEEG_cycle_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_cycle_detector.py):

```python
import os
from turtlewave_hdEEG import CustomAnnotations, finalize_cycles_and_durations

root_dir = "data/sub-001js/ses-1/"
annot_file = os.path.join(root_dir, "wonambi", "sub-001js_annotations.xml")
db_path = os.path.join(root_dir, "wonambi", "neural_events.db")
subject = "sub-001js"

annot = CustomAnnotations(annot_file)

# Detects cycles for both '2022' and '1979', writes stage durations, tags
# events.cycle with the '2022' definition, and writes '2022' XML markers.
cycles_by_method = finalize_cycles_and_durations(
    annot,
    db_path,
    subject=subject,
    # Defaults; override if needed:
    # methods=('2022', '1979'),  # cycle definitions to store
    # tag_method='2022',         # which definition owns events.cycle + XML
    # wake_thresh=10,            # max Wake epochs absorbed into NREM
    # nrem_min=30,               # min NREM epochs to count as an NREM period
    # rem_min=10,                # min REM epochs to close a cycle (1979 only)
)

for method, cycles in cycles_by_method.items():
    print(f"Method '{method}': {len(cycles)} cycle(s)")
    for c in cycles:
        print(f"  cycle {c['cycle_number']}: "
              f"NREM {c['nrem_dur_min']:.1f} min, "
              f"REM {c['rem_dur_min']:.1f} min, "
              f"total {c['cycle_dur_min']:.1f} min "
              f"(start {c['nrem_start_sec']:.0f}s)")
```

`finalize_cycles_and_durations` returns `{method: [cycle dicts]}` for every
method requested. Each cycle dict carries `cycle_number`, `method`,
epoch/second boundaries (`nrem_start_epoch`, `nrem_end_epoch`,
`rem_start_epoch`, `rem_end_epoch`, `nrem_start_sec`, `nrem_end_sec`,
`rem_end_sec`), and durations `nrem_dur_min`, `nrem_n23_dur_min` (N2+N3
minutes only), `rem_dur_min`, `cycle_dur_min`.

## Backfill an existing database (batch)

If you already have a fleet of `neural_events.db` files with events detected
but no cycle/duration data, run the same finalize call over every subject.
Mirror
[`examples/backfill_cycles.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/backfill_cycles.py):

```python
import glob
import os

from turtlewave_hdEEG import CustomAnnotations, finalize_cycles_and_durations

ROOT = "/path/to/subjects_root"

for name in sorted(os.listdir(ROOT)):
    subj_dir = os.path.join(ROOT, name)
    wonambi_dir = os.path.join(subj_dir, "wonambi")
    db_path = os.path.join(wonambi_dir, "neural_events.db")
    if not os.path.isfile(db_path):
        continue  # not a subject folder with a detection DB

    xmls = sorted(glob.glob(os.path.join(wonambi_dir, "sub-*.xml")))
    if not xmls:
        print(f"[{name}] no annotation XML found, skipping")
        continue

    annot = CustomAnnotations(xmls[0])
    cycles_by_method = finalize_cycles_and_durations(
        annot, db_path, subject=name)
    summary = ", ".join(f"{m}={len(c)} cyc" for m, c in cycles_by_method.items())
    print(f"[{name}] {summary}")
```

`examples/backfill_cycles.py` is a hardened, ready-to-run version of this loop
(subject-id derivation, per-subject try/except so one bad folder doesn't abort
the batch, and a PASS/FAIL summary):

```bash
python examples/backfill_cycles.py
```

Edit its `ROOT` (and optional `SUBJECTS` allowlist) constants at the top of
the file before running.

## What lands in the database

- **`sleep_cycles`** — one row per `(subject, method, cycle_number)`: NREM/REM
  start/end times and `nrem_dur_min`, `nrem_n23_dur_min`, `rem_dur_min`,
  `cycle_dur_min`. Re-running replaces the existing rows for that
  `(subject, method)` pair, so it stays idempotent.
- **`stage_durations`** — one row per subject: minutes in Wake / N1 / N2 / N3
  / REM / artefact, reconciled to the full hypnogram span. Written even when
  no cycles are detected (an all-Wake or unscorable night still has stage
  durations).
- **`events.cycle`** — every event in the `events` table gets tagged with its
  cycle number under `tag_method` (`'2022'` by default). Because tagging
  rewrites event rows by time window regardless of method ("last run wins"),
  `finalize_cycles_and_durations` always runs `tag_method` last among
  `methods` so its numbering is the one that survives.
- **Annotation XML** — cycle markers for `tag_method` only, so
  `Annotations.get_cycles()` and the review GUI show cycle bands without a
  numbering conflict between the two definitions.

## Plot the hypnogram with cycle bands

The finalize call never plots by default. To get the PNG (blue NREM / red REM
bars over the hypnogram, one row per method), either pass `plot=True` /
`plot_path=...` to `finalize_cycles_and_durations`, or call
`plot_from_annotations` directly once you have `cycles_by_method`:

```python
from turtlewave_hdEEG import plot_from_annotations

out_png = os.path.join(root_dir, "wonambi",
                       f"{subject}_hypnogram_cycles_2022_vs_1979.png")
plot_from_annotations(annot, cycles_by_method, out_png, subject=subject)
```

## Lower-level building blocks

`finalize_cycles_and_durations` is a convenience wrapper around two lower
level pieces, useful if you need finer control:

- `detect_cycles(hypnogram, epoch_length=30, wake_thresh=10, nrem_min=30, method='2022', rem_min=10, epoch_starts=None)`
  — the pure hypnogram-in, cycle-list-out detector. No dataset, annotations,
  or database required; works on any numeric per-epoch stage sequence
  (Wake=0, NREM1/2/3=1/2/3, REM=4, artefact/undefined=-1).
- `ParalCycles(dataset=None, annotations=annot, subject=subject).run(db_path, method='2022', write_xml=True, subject=None, epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10)`
  — detects cycles for a single method and persists them (cycles, stage
  durations, event tagging, optional XML markers) in one call. Use this
  directly if you only want one cycle definition rather than both.

## Common Issues

### No Cycles Detected

If `cycles_by_method[method]` comes back empty:

- **Check the hypnogram**: An empty or all-Wake hypnogram legitimately
  returns no cycles — `stage_durations` is still written.
- **Check `nrem_min`**: NREM runs shorter than `nrem_min` epochs (default 30,
  i.e. 15 minutes at 30 s epochs) are dropped as too short to count as an
  NREM period.
- **Under `'1979'` only**: A trailing NREM period with no qualifying REM after
  it still becomes a final cycle, so an empty result under `'1979'` but not
  `'2022'` usually means the NREM/REM structure itself doesn't meet the
  stricter 1979 rule — inspect the hypnogram directly.

### `events.cycle` Doesn't Match the XML Markers

This happens only if `tag_method` is not included in `methods` — the XML gate
never fires but `events.cycle` is still overwritten by the last method run.
Keep `tag_method` within `methods` (the default already satisfies this).

## Next Steps

After finalizing cycles and stage durations, you might want to:

- Query `sleep_cycles` / `stage_durations` directly, or join `events` on
  `cycle` for per-cycle event density
- Review events with cycle context in the QC dashboard — [Review EEG Events](review-eeg-events.md)
- Reference: [Sleep Cycle Processor](../reference/api/cycleprocessor.md),
  [Sleep Cycle Plotting](../reference/api/cycleplot.md)
