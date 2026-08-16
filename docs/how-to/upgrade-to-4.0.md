# How to Upgrade to turtlewave-hdEEG 4.0

This guide gets existing 3.x work running under 4.0. It is ordered by how much
old output each change invalidates, worst first.

## Before you start

- You have a 3.x project with a `neural_events.db`, Wonambi annotation XML
  files, and possibly PAC or density CSVs from earlier runs.
- No code changes are required to upgrade (see the last section) — this guide
  is about which *outputs* to regenerate, not which calls to rewrite.

## Step 1 — Back up your database and annotations

`neural_events.db` is migrated automatically the first time any 4.0 code
writes to it. The migration that widens `processing_status` (see
[Database schema changes](#step-4-review-the-database-schema-changes) below)
creates a new table, copies rows across, drops the old table and renames the
new one into place — it is not reversible without a backup. This runs inside
`ensure_direct_write_schema` in `turtlewave_hdEEG/dbwrite.py`.

Before running any 4.0 detection, export, or GUI session against an existing
project:

```bash
cp neural_events.db neural_events.db.bak-v3.3.0
cp -r wonambi/*.xml /path/to/backup/
```

## Step 2 — Regenerate event densities

Density (events per minute) now divides by the artefact-free analysed time the
detector actually pooled, not by the total scored time of a stage. Every
density CSV produced before 4.0.0 under-estimates density and is not
comparable with new output. Whole-night density additionally now excludes
Wake unless Wake was itself a detection stage. See
[Event Density is Artefact-Free](../explanation/overview.md#event-density-is-artefact-free)
for why.

Regenerate from the existing detection JSON using the same exporters you
already call, for example:

```python
density2CSV = event_processor.export_spindle_density_to_csv(
    json_input=json_dir,
    csv_file=os.path.join(json_dir, f'spindle_density_{method}_{freq_range}_{stages_str}.csv'),
    stage=test_stages,
    file_pattern=file_pattern,
)
```

or the slow-wave equivalent, `export_slow_wave_density_to_csv`, with the same
arguments. No re-detection is needed — only the density export step, driven
off the JSON you already have. Do not mix a pre-4.0 density CSV with a
post-4.0 one in the same analysis.

## Step 3 — Regenerate PAC preferred phase

`preferred_phase_rad` / `preferred_phase_deg` were reported 180 degrees off in
every version before 4.0.0, in both `ParalPAC.analyze_pac` and
`ParalPAC.compare_conditions`. The bin-centre vector used to compute the
preferred phase spanned `[0, 2π)` while the amplitude binning itself spans
`[-π, π)`, so up-state and down-state coupling conclusions are inverted in any
PAC output computed before this release.

Modulation index, mean vector length, rho and the Rayleigh statistics are
**unaffected** — you do not need to re-run PAC detection, only correct the
preferred-phase columns already on disk.

Run the migration script against your existing PAC results tree:

```bash
python examples/fix_pac_preferred_phase.py /path/to/pac_results --dry-run
python examples/fix_pac_preferred_phase.py /path/to/pac_results
```

It walks every `*_pac_parameters.csv` and `pac_summary_*.csv` under the given
root, backs up each original to `<name>.bak`, and rotates only the affected
columns (`preferred_phase_deg`, `PP_degrees`, `Condition1_PP_deg`,
`Condition2_PP_deg`, `preferred_phase_rad`, `PP_rad`, `Condition1_PP_rad`,
`Condition2_PP_rad`) by 180 degrees / π radians. A file that already has a
`.bak` sibling is skipped, so re-running the script cannot double-correct.
The `*_mean_amps.npy` files are untouched.

If you are about to back-fill old PAC CSVs into the database with
`examples/backfill_pac_to_db.py`, run this fix first — see
[Ordering matters if your CSVs predate the preferred-phase fix](backfill-pac-to-database.md#ordering-matters-if-your-csvs-predate-the-preferred-phase-fix).

## Step 4 — Review the database schema changes

All changes to `neural_events.db` are additive except one primary-key
widening:

- `processing_status`'s primary key widens from `(channel, event_type)` to
  `(channel, event_type, method, freq_lower, freq_upper, stage)`. Any external
  query that assumed one status row per `(channel, event_type)` will now see
  several rows for a channel detected with more than one method, band, or
  stage set.
- `events.cycle` changes from always `NULL` to a 1-based sleep-cycle index,
  populated when cycles have been detected.
- New tables: `sleep_cycles`, `stage_durations`, `detection_runs`,
  `rerun_log`, `pac_coupling`.
- New `events` columns: `rms`, `power`, `peak_power_freq`, `energy`,
  `peak_energy_freq`.

The migration is idempotent and runs automatically; you don't need to touch
it, but update any hand-written SQL that groups or joins on
`processing_status` by `(channel, event_type)` alone.

## Step 5 — Adjust to the review GUI workflow change

`eeg_review_gui` dropped the per-event Events tab: there is no more per-event
accept/reject, stratified sampling, or Compare-methods view. Two tabs remain —
**Channels (QC)** is the landing surface, **Epochs** is the drill-down. `F`
flags the selected channel for re-detect, and **Export QC report…** replaces
the old **Export Reviewed Events…** menu action. If you had a review workflow
built around per-event verdicts, it now operates at channel granularity
instead.

See the [EEG Review GUI Tutorial](../tutorials/eeg-review-gui-tutorial.md) and
[EEG Review GUI Architecture](../explanation/eeg-review-gui-architecture.md)
for the current workflow — parts of both predate this redesign and are being
updated separately.

## Step 6 — Re-run spindle detection that used `polar='opposite'`

`polar='normal'` is unaffected at every version, and every shipped example
script defaults to it — so this section only concerns you if you explicitly
passed `polar='opposite'`. If you didn't, skip ahead.

**Spindle detection with `polar='opposite'` never produced any output, at any
released version.** Wonambi's `Data` has no `.copy()` method, so the old
inversion code raised `AttributeError` on every channel. That error was
caught by the per-channel try/except in `eventprocessor.py` (~line 540) and
logged as `Failed to process channel {ch}`, so the run completed but silently
returned an empty result set rather than failing loudly. 4.0.0 makes
`polar='opposite'` spindle detection work for the first time — if you ever
ran it, there is nothing to correct in the old output, because there was no
output. Re-run it.

**Slow waves and K-complexes with `polar='opposite'` were correct at v3.3.0
and earlier, in every configuration, and need no re-run.** Three separate
inversions were in play at that version: the processor negated the shared
segment array in place (`swprocessor.py:309-310`),
`ImprovedDetectSlowWave.__call__` negated that same array back, and Wonambi's
own slow-wave detectors negated their private copy
(`wonambi/detect/slowwave.py:192`, `:256`, `:322`). The first two cancelled
on the shared array, leaving Wonambi's as the only one that reached the
detection, which is exactly one inversion — the correct result. Because the
shared array was restored to its original state after every method,
multi-method runs were correct too.

This was verified by replaying the v3.3.0 processor loop against the v3.3.0
detector and comparing event start times element-wise against true-inverted
ground truth (232 and 352 events on the two Massimini variants, with no
alternation across methods). 4.0.0 produces bit-identical numbers, so old and
new slow-wave output are directly comparable.

If you ran from a git checkout of this branch rather than a released
version, note that the in-progress branch state briefly broke `polar='opposite'`
slow-wave and K-complex detection entirely before landing on the fix
described above — this only matters if you detected from that checkout
directly, not from a PyPI release.

**Known limitations, unchanged in 4.0.0 (not fixed in this release):**

- The spindle CIRUS method returns before the spindle inversion block runs
  (`extensions.py:645-648`), so a CIRUS run with `polar='opposite'` silently
  detects on uninverted signal. CIRUS has never applied the polarity
  inversion; this is not a regression, but CIRUS still needs `polar='normal'`
  for sensible results.
- `detrend=True` is silently ignored in slow-wave and K-complex detection.
  Both call sites (`swprocessor.py:476`, `kcomplexprocessor.py:362`) pass the
  string `'detrend'` as the `operator` argument to `wonambi.trans.math`,
  which expects a callable and raises `AttributeError` when it isn't one; the
  broad `except Exception` around the call logs an error and proceeds on
  un-detrended data. Anyone who believed they detrended their slow waves or
  K-complexes did not — at any version, including 4.0.0.

## Step 7 — Exporters and importers now raise instead of failing silently

This step needs your attention only if a script wraps these calls and
inspects their return value for an error, instead of letting exceptions
propagate. **No new arguments are involved — this is a behaviour change to
existing calls.**

- `export_spindle_parameters_to_csv`, `export_slow_wave_parameters_to_csv`
  and `export_kc_parameters_to_csv` (and their density-export equivalents)
  now default to `strict=True`: when `file_pattern` matches zero JSON files,
  they raise `FileNotFoundError` instead of writing a one-line placeholder
  CSV. A zero-match export is almost always a filename round-trip bug (the
  band or method token in the pattern doesn't match what the detector wrote),
  and the placeholder CSV made that indistinguishable from a genuinely empty
  run downstream. Pass `strict=False` to restore the old placeholder-CSV
  behaviour if you rely on it.
- `import_parameters_csv_to_database` (on `ParalEvents`, `ParalSWA`,
  `ParalKC`) now raises (`FileNotFoundError`, `RuntimeError`, or whatever the
  underlying failure was) instead of returning
  `{"error": ..., "added": 0}`. It also gained `event_type=`, `method=` and
  `force=` keywords — `force=True` is required to re-import a CSV over rows
  already written by the `write_db=True` path (otherwise it raises
  `RuntimeError`, protecting those rows' `run_id` provenance link).
- `ParalPAC.store_pac_to_database` and `import_pac_csv_to_database` now
  re-raise on failure instead of logging a traceback and returning as if
  nothing had gone wrong; `backfill_pac_directory` catches per-file so one
  bad CSV doesn't abort a whole walk, and reports the count in a new
  `failed` key.

See
[Write Detection Results Directly to the Database](direct-to-database-detection.md#pull-a-csv-back-out-of-the-database)
and
[About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md)
for the reasoning.

## Step 8 — No further code changes required

All imports and function signatures are unchanged. Every new parameter across
the detectors and exporters is keyword-only with a default, so existing calls
that don't rely on the old silent-failure behaviour corrected in Step 7 behave
exactly as before. `turtlewave_hdEEG/__init__.py` gained exports and lost
none. Console script names (`turtlewave_gui`, `eeg_review_gui`) are unchanged.
If none of steps 1–7 apply to your project, you can stop here.

## New in 4.0 worth adopting

A few 4.0 capabilities are opt-in and don't require any of the steps above:

- **Sleep cycles and stage durations** — `finalize_cycles_and_durations`
  populates `sleep_cycles` and `stage_durations` and backfills `events.cycle`.
- **Direct-to-database detection** — pass `write_db=True` to skip the
  JSON → CSV → import round-trip. (Since 4.2 this is the default behaviour,
  not just an opt-in — see [Upgrade to 4.2](upgrade-to-4.2.md).) See
  [Write Detection Results Directly to the Database](direct-to-database-detection.md).
- **Re-run detection on reviewer-selected channels** — re-detect only the
  channels flagged in the review GUI, with a scoped replace. See
  [Re-run Detection on Reviewer-Selected Channels](rerun-detection-on-channels.md).

## Related

- [Upgrade to 4.2](upgrade-to-4.2.md) — the next boundary: direct-to-database
  detection becomes the *default*, not just an opt-in.
- [Event Density is Artefact-Free](../explanation/overview.md#event-density-is-artefact-free)
- [About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md)
- [Back-fill PAC Results into the Database](backfill-pac-to-database.md)
- [Write Detection Results Directly to the Database](direct-to-database-detection.md)
- [Re-run Detection on Reviewer-Selected Channels](rerun-detection-on-channels.md)
- [Reference: Direct-to-Database Write](../reference/api/dbwrite.md)
- [Reference: Utilities](../reference/api/utils.md)
