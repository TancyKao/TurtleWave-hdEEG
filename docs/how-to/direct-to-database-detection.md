# Write Detection Results Directly to the Database

This guide shows you how to run spindle, slow-wave or K-complex detection with
the direct-to-database write path, instead of the legacy JSON → CSV → import
pipeline.

## When to use this

**Problem:** The legacy pipeline (JSON per channel, then a CSV export step,
then a CSV → SQLite import step) is three passes over the same data, and a
crash mid-run leaves you guessing which channels finished.

**Solution:** Pass `write_db=True` (and a few related keyword arguments) to
`detect_spindles` / `detect_slow_waves` / `detect_kcomplexes`. Each channel's
events are written straight into `neural_events.db` in one transaction, along
with a `detection_runs` provenance row recording the method, citation, full
parameter dict, reference/polarity, artefact-rejection settings and library
versions used.

## Enable it from a detection script

The direct-write path is opt-in and keyword-only, so existing calls are
unaffected unless you pass these arguments explicitly:

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

data = WonambiDataset("sub-001_eeg.set")
annot = CustomAnnotations("sub-001_eeg.xml")
event_processor = ParalEvents(dataset=data, annotations=annot)

spindles = event_processor.detect_spindles(
    method="Moelle2011",
    chan=["E110", "E111", "E112"],
    frequency=(11, 13),
    duration=(0.5, 3),
    stage=["NREM2", "NREM3"],
    reject_artifacts=True,
    reject_arousals=False,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir="wonambi/spindle_results",   # still written for provenance/debugging
    write_db=True,
    db_path="wonambi/neural_events.db",
    resume=False,
)
```

`ParalSWA.detect_slow_waves` and `ParalKC.detect_kcomplexes` accept the same
`write_db` / `db_path` / `resume` / `run_params` / `replace_channels` keywords.

!!! tip
    `examples/hdEEG_spindle_detector.py` and `examples/hdEEG_sw_detector.py`
    already wire this up behind `--write-db` / `--resume` CLI flags (parsed
    with `argparse.parse_known_args`, so the rest of the script's hard-coded
    parameters are unaffected). Run e.g.:
    ```bash
    python examples/hdEEG_spindle_detector.py --write-db --resume
    ```

## Resume an interrupted run

**Problem:** A batch job died partway through a subject with 128 channels and
you don't want to re-detect the 90 that already succeeded.

**Solution:** Pass `resume=True` alongside `write_db=True`. A channel is
skipped only when a `processing_status` row already recorded `success = 1` for
the **exact same scope** — event type, method, frequency band and stage set.
A channel that previously failed (`success = 0`) is retried, not skipped.

```python
event_processor.detect_spindles(
    method="Moelle2011",
    chan=all_channels,
    frequency=(11, 13),
    stage=["NREM2", "NREM3"],
    write_db=True,
    db_path="wonambi/neural_events.db",
    resume=True,
)
```

## Run it on the NCI Gadi cluster

The `_GADI.py` driver scripts mirror the same flags:

```bash
python hdEEG_spindle_detector_GADI.py \
    --root /scratch/xx99/subjects --subject sub-001 \
    --method Moelle2011 --stages NREM2,NREM3 --freq 9.0,12.0 \
    --write-db --resume
```

See `examples/NCI_commands/hdEEG_spindle_detector_GADI.py` and
`examples/NCI_commands/hdEEG_sw_detector_GADI.py`.

## What lands in the database

When `write_db=True`, each event row in `events` additionally carries:

- **Detector-own morphology** — `det_trough`, `det_peak`, `det_ptp`,
  `det_trough_time`, `det_peak_time`: the values the detector itself decided
  on (trough/peak/peak-to-peak amplitude and their timestamps). Spindle events
  are oscillatory rather than trough-based, so these are typically `NULL` for
  spindles and populated for slow waves / K-complexes.
- **Re-measured spectral/RMS columns** — `rms`, `power`, `peak_power_freq`,
  `energy`, `peak_energy_freq`, plus `min_amp` / `max_amp` / `peak2peak_amp`:
  computed once per channel over all of that channel's in-memory event windows
  (`compute_batched_params`), rather than one raw-file re-read per event. These
  are the same quantities the legacy CSV exporter's `event_params` re-read
  produced, so column semantics are unchanged.
- **`run_id`** — links the event back to the `detection_runs` row for this
  invocation.

A `detection_runs` row is written per invocation with the method, a literature
citation (resolved automatically for the built-in methods — Ferrarelli2007,
Moelle2011, Nir2011, Wamsley2012, Martin2013, Ray2015, Lacourse2018,
Massimini2004, Ngo2015, Staresina2015, etc.), the full parameter dict, the
reference channel(s), polarity, requested stages, artefact/arousal rejection
flags, and `turtlewave_hdEEG` / `wonambi` / `numpy` versions plus the git SHA.

Each row's `uuid` is a deterministic `uuid5` of its detection scope (event
type, channel, start time, method, band, stage), so re-detecting an unchanged
channel with `write_db=True` is a true row-level no-op under
`INSERT OR REPLACE` — it never duplicates rows.

## Pull a CSV back out of the database

**Problem:** Your downstream stats pipeline expects a flat CSV, but you wrote
straight to the database.

**Solution:** Call `export_events_to_csv` for the scope you need. It writes
the SAME column layout as the legacy JSON → CSV exporters (plus the additional
`det_*` columns at the end), so the file round-trips back through
`import_parameters_csv_to_database` unchanged.

```python
from turtlewave_hdEEG import export_events_to_csv

csv_path = export_events_to_csv(
    db_path="wonambi/neural_events.db",
    event_type="spindle",
    method="Moelle2011",
    frequency=(11, 13),
    stage=["NREM2", "NREM3"],
    output_dir="wonambi/spindle_results",
)
print(f"Wrote {csv_path}")
```

`csv_file=None` (the default) builds the standard filename with
`default_csv_path` — the same
`{event_type}_{method}_{freq_lo}-{freq_hi}Hz_{stages_joined}` convention used
everywhere else in the pipeline. If the scope genuinely has no matching
events, `export_events_to_csv` returns `None` and writes nothing; if the
`stage` filter excludes every row even though the same type/method/band DOES
have events under other stages, it raises `ValueError` instead of silently
writing an empty file, so a stage-token typo is caught immediately.

## See also

- [Re-run detection on reviewer-selected channels](rerun-detection-on-channels.md)
  — the direct-write path's `replace_channels` argument is what makes a scoped
  re-detection possible.
- [Explanation: Event density is artefact-free](../explanation/overview.md#event-density-is-artefact-free)
- [Reference: dbwrite module](../reference/api/dbwrite.md)
