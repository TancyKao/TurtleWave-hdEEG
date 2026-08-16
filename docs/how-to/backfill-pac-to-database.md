# Back-fill Existing PAC Results into the Database

This guide shows you how to load phase-amplitude coupling (PAC) results that
were already written to CSV by `ParalPAC.analyze_pac` /
`export_pac_parameters_to_csv` into the `pac_coupling` table of
`neural_events.db`, without re-running the analysis.

## When to use this

**Problem:** You ran PAC analysis before the `pac_coupling` table existed (or
with `write_db=False`), and now want those results queryable alongside
spindle/slow-wave events in the same database.

**Solution:** Walk the subject's PAC results tree and import every
per-channel CSV with `examples/backfill_pac_to_db.py`.

```bash
python examples/backfill_pac_to_db.py \
    --root /path/to/SUBJECT/wonambi/pac_results \
    --db   /path/to/SUBJECT/wonambi/neural_events.db
```

By default (`--subject` omitted, or `folder`) the subject id is resolved by
the shared [`derive_subject`](../reference/api/utils.md) with `root_dir=--root`
— the basename of `--root` itself, prefixed with `sub-` if it doesn't already
have one. Because `--root` in the example above points at the `pac_results`
leaf directory rather than the subject folder, that default resolves to
`sub-pac_results`, which is almost never what you want here — pass `--subject`
explicitly whenever `--root` isn't itself the subject directory:

```bash
python examples/backfill_pac_to_db.py \
    --root .../pac_results --db .../neural_events.db --subject SUB001
```

## Ordering matters if your CSVs predate the preferred-phase fix

If the PAC CSVs were produced before the preferred-phase 180-degree fix, run
the historical migration **first**:

```bash
python examples/fix_pac_preferred_phase.py /path/to/pac_results
```

then back-fill. `backfill_pac_to_db.py` stores the preferred-phase value
exactly as it appears in the CSV — it does not correct polarity itself.

## What gets imported, and what doesn't

The back-fill walks `--root` and, in every directory, imports the
per-channel `*_pac_parameters.csv` files (`*_pac_parameters.csv`, not
`pac_summary_*`). `pac_summary_*` files are skipped whenever per-channel CSVs
are present in the same directory — the per-channel files are the source of
truth. A directory with only `pac_summary_*` files (no per-channel CSVs) is
logged and skipped, not imported.

For each per-channel CSV, the importer:

- Infers `method`, `stage`, `event_type` and the phase/amplitude frequency
  bounds from the results directory layout
  (`<method_dir>/<stage_str>/<channel>_<event>_pha-..-..Hz_amp-..-..Hz_pac_parameters.csv`)
  rather than a lossy filename split.
- Recovers `n_events` from the sibling `*_mean_amps.npy` file's shape. **A row
  whose event count can't be recovered is rejected, not stored as `0`/`NULL`**
  — it's counted under `n_events_missing` and logged as needing a re-run.
- Writes one idempotent row per `(subject, channel, event_type, method, stage,
  phase_freq_lower, phase_freq_upper, amp_freq_lower, amp_freq_upper)` —
  re-running the back-fill on the same tree reports `added=0` for rows already
  stored.

## Check the summary

```text
PAC back-fill summary:
  files: 42
  added: 40
  updated: 0
  skipped: 0
  n_events_missing: 2
```

The script exits non-zero if any row was rejected for a missing event count,
so a batch harness can flag files needing attention. Investigate any
`n_events_missing` rows — they need their sibling `*_mean_amps.npy` restored,
or a re-run of that channel's PAC analysis.

`backfill_pac_directory` (the function underneath the script) also returns a
`failed` count of CSVs whose import raised an exception. A per-file failure
is caught and logged so one bad CSV doesn't abort the whole walk, but it no
longer disappears into a printed traceback — it is tallied and the walk logs
a warning if `failed` is non-zero. `examples/backfill_pac_to_db.py`'s summary
does not currently print this field; call `backfill_pac_directory` directly
(see below) if you need to check it programmatically.

## Programmatic equivalent

The same walk is available directly from `ParalPAC` if you want it inside
your own script (e.g. right after a batch of `analyze_pac` calls that used
`write_db=False`):

```python
from turtlewave_hdEEG import ParalPAC

pac = ParalPAC(dataset=dataset, annotations=annot, rootpath=root_dir)
totals = pac.backfill_pac_directory(root_dir, db_path, subject_from='folder')
```

Or import a single CSV with `import_pac_csv_to_database`, overriding any
context that can't be inferred from the path (e.g. an ambiguous channel
label):

```python
pac.import_pac_csv_to_database(
    csv_path="pac_results/Wamsley2012/NREM2NREM3/E101_slow_wave_pha-0.5-1.25Hz_amp-11-16Hz_pac_parameters.csv",
    db_path="neural_events.db",
    subject="SUB001",
)
```

## See also

- [Reference: PAC Processor](../reference/api/pacprocessor.md)
- [Direct-to-database detection](direct-to-database-detection.md) — writing
  PAC results straight to the database during analysis instead of
  back-filling afterwards (`analyze_pac(..., write_db=True, subject=...)`).
