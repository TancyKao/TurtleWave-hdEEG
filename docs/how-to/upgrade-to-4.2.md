# How to Upgrade to turtlewave-hdEEG 4.2

4.2 makes `neural_events.db` the default output of detection and PAC
analysis, replacing the per-channel JSON → CSV → import pipeline as the
common path. This guide covers what changes for existing scripts, PBS jobs
and GUI usage.

If you're also crossing the 4.0 boundary, read
[Upgrade to 4.0](upgrade-to-4.0.md) first — the density-denominator, PAC
preferred-phase, and exception-instead-of-error-dict changes there are
independent of, and precede, this one.

## Before you start

```bash
cp neural_events.db neural_events.db.bak-pre-4.2
```

The schema change is additive (new `analysed_time` table; no columns dropped
or retyped), but back up before running any 4.2 code against a database you
care about, same as any other version.

!!! info "Your first 4.2 run against an existing database claims it — check the log"
    The first 4.2+ detection or PAC run against a database that predates 4.2
    (or that only ever saw `--legacy-json` writes) logs a **WARNING** naming
    the database, its event count, and the subject it is claiming the
    database for — see
    [One database, one subject](#one-database-one-subject-what-the-guard-does).
    No flag or manual step is needed; just make sure the subject that first
    run uses is the right one, since every later run checks against it.

## What changed

- `write_db` moves from defaulting to `False` to defaulting to `None`
  (**AUTO**) on `detect_spindles`, `detect_slow_waves`, `detect_kcomplexes`
  and `analyze_pac`. AUTO resolves a database path (explicit `db_path` →
  sibling of the output/`json_dir` → `./neural_events.db`) and writes there.
  `write_db=False` still works exactly as before — nothing about the legacy
  path changed.
- Per-channel JSON is no longer written on the default path. The
  `create_empty_json` parameter is removed from all three detectors — there
  is nothing for it to gate now that the default path writes no JSON at all.
- The three GUI detection paths and the `_GADI.py` cluster drivers lost their
  export → import → density-CSV block entirely on the default path.
- `--write-db` is gone from the CLI drivers as a meaningful flag — it's
  accepted for backward compatibility so an existing PBS script doesn't fail
  to parse, and it now names the default rather than opting into anything.
  On `examples/hdEEG_spindle_detector.py`, `hdEEG_sw_detector.py` and
  `hdEEG_kcomplex_detector.py` it's a pure no-op (registered with
  `help=SUPPRESS`; nothing reads it, so passing it prints nothing). Only the
  two `_GADI.py` cluster drivers detect it and log a warning that it does
  nothing. **`--legacy-json` is the new opt-out**, on all five drivers.
- New `dbwrite.resolve_db_target` **raises** when a database write is
  requested (explicitly or via AUTO) and no path can be resolved, instead of
  silently downgrading to no write — the same class of fix as 4.0.1's
  frequency-token bug, applied to the write path itself. `analyze_pac` also
  resolves its `db_path` through this same helper (unconditionally, not only
  when writing — it's also how PAC locates the events it reads), so it can no
  longer end up pointed at a different database than the rest of the
  pipeline.
- `analyze_pac` gained `write_csv=None` (AUTO: the per-channel
  `*_pac_parameters.csv` is written only when the run is on the legacy
  `write_db=False` path). The sibling `*_mean_amps.npy` — the per-event
  modulogram matrix, not reconstructable from `pac_coupling` — is **always**
  written regardless of `write_csv`.
- New `turtlewave_hdEEG.density` module: `event_density(db_path, ...)`
  derives density on read from `events` and a new `analysed_time` table
  (the artefact-free denominator), instead of a CSV. The three
  `export_*_density_to_csv` methods and the CSV import route
  (`import_parameters_csv_to_database`, `import_pac_csv_to_database`) are now
  deprecated (`DeprecationWarning` **and** `logger.warning`; removal is 5.0).
- New `dbwrite.assert_single_subject` refuses a direct-write call whose
  `subject` doesn't match a subject already present in the target database.
  A database with events but no subject recorded anywhere (any pre-4.2
  database) is **claimed** for the current subject instead of refused — a
  loud `logger.warning`, no flag, no manual step — after which a *different*
  subject is refused. See
  [One database, one subject](#one-database-one-subject-what-the-guard-does)
  below.

## One database, one subject: what the guard does

One `neural_events.db` per subject has always been the deployment contract —
`events` has no subject column, and an event's id is a `uuid5` of
`(event_type, channel, start_time, method, band, stage)` alone, so two
subjects sharing a database collide silently: identical channel labels at
identical times produce identical ids, the second subject's `INSERT OR
REPLACE` overwrites the first subject's rows, and a scoped re-detection's
`DELETE` removes the other subject's channel outright. `verify_channel_coverage`
then reports full coverage over the wrong data.

4.2 adds a guard for this, `dbwrite.assert_single_subject`, called by every
direct-write call before it writes a single row. It normalises both sides to
the canonical `sub-` form (see
[Subject ids are normalised](#subject-ids-are-normalised) below) and then:

1. **If the database already names a *different* subject** — in
   `analysed_time`, `detection_runs`, `pac_coupling`, `sleep_cycles` or
   `stage_durations` — it raises `ValueError` unconditionally. This is the
   real protection: the database provably belongs to someone else, and
   nothing overrides it.
2. **If the database has rows in `events` but names no subject anywhere** —
   every pre-4.2 database, and any database that has only ever seen
   `write_db=False` / `--legacy-json` writes — it does **not** raise. It
   *claims* the database for the current subject: it stamps that subject onto
   every `detection_runs` row that currently has a `NULL` subject, and logs a
   **WARNING** naming the database path, the event count, and the subject
   being claimed. From that point on, a *different* subject hitting the same
   database falls into case 1 above and is refused.

No flag, no manual step, no failed first run. The only thing to actually do
here is **read that warning the first time it appears** and confirm the
subject named in it is correct — claiming is a one-time, unverified
assertion (the guard has no way to independently confirm which recording a
pre-4.2 database belongs to), and an incorrect claim silently mislabels every
existing `detection_runs` row rather than raising. If you're not sure which
subject an old database belongs to, check what's actually in it before your
first 4.2 run touches it:

```bash
sqlite3 neural_events.db "SELECT event_type, method, COUNT(*), COUNT(DISTINCT channel) FROM events GROUP BY 1, 2;"
```

and cross-check the count/channels against whatever you already know about
the recording (its directory name, its annotation XML) — then pass that
subject explicitly (`subject=`/`--subject`) rather than relying on whatever a
driver script derives by default.

`analyze_pac` calls a different check: `pac_coupling` already carries its own
`subject` column and natural key, so the single-database ambiguity this guard
exists for doesn't arise there.

### Subject ids are normalised

`subject` is normalised to the canonical `sub-` form (`normalize_subject`) on
every read and write path now — the three detectors, `analyze_pac`,
`event_density`, and `ParalCycles.store_cycles_to_database` /
`store_stage_durations` all agree that `'10sd'` and `'sub-10sd'` name the
same recording. You don't need to standardise `--subject` values across
scripts; passing the bare folder name works exactly the same as passing the
`sub-` form.

## Does your existing script need changes?

**No, if** you already pass `write_db=True` explicitly (the 4.0/4.1 opt-in
direct-write path) — that call shape is unchanged, and AUTO simply makes it
the default for everyone else too.

**No, if** you're happy for new runs to go straight into the database and
your downstream code already reads `neural_events.db` rather than a CSV.
This is most people: the outputs just get simpler.

**Yes, if** any of the following are true:

- **You parse a per-channel JSON file, or a CSV built from one, that
  detection used to write.** Nothing writes it anymore unless you pass
  `write_db=False` (or `--legacy-json`) on the run that produces it. See
  [Where did my CSV go?](read-database-with-pandas-and-r.md#where-did-my-csv-go).
- **You call `detect_*` with `create_empty_json=...`.** The parameter is
  removed; drop it. There is no replacement to configure — resume behaviour
  is covered by `processing_status`, not a sentinel JSON file.
- **Any script passes `--write-db`.** It still runs — the flag is accepted
  and is now a no-op — but update the script when convenient; it no longer
  does anything. Only the `_GADI.py` cluster drivers (the ones PBS jobs
  actually invoke) log a warning about it; the three plain
  `examples/hdEEG_*_detector.py` scripts accept and silently ignore it.
- **You compute density from `stage_durations` or a CSV you built yourself.**
  Switch to `turtlewave_hdEEG.density.event_density`, which reads the
  artefact-free denominator from `analysed_time`. A database written on the
  legacy `write_db=False` path has no `analysed_time` table —
  `event_density` raises rather than silently substituting
  `stage_durations`, which would reintroduce the artefact-scaled bias 4.0
  fixed. Either re-run detection with the default (AUTO) path, or keep using
  the JSON-backed `export_*_density_to_csv` exporters against that legacy
  JSON directory (deprecated but functional).
- **You call `import_parameters_csv_to_database` or
  `import_pac_csv_to_database` in a script that checks the return value for
  an error key**, or ignores `DeprecationWarning`. These still work — they're
  the recovery path for historical JSON/CSV output and for
  `examples/backfill_pac_to_db.py`/`backfill_cycles.py`/`csv_to_db_import.py`
  — but budget for their removal in 5.0.

## Regenerate outputs you rely on

Detection results themselves are unaffected — 4.2 does not change what gets
detected, only where results land. You do **not** need to re-run detection
just to upgrade. You do need to decide, per script, whether to:

- let new runs go to the database (do nothing — this is the default), or
- keep producing JSON/CSV for a downstream tool that isn't ready to read
  SQLite yet (`write_db=False` / `--legacy-json`).

If a script currently reads a CSV that a prior *4.0/4.1* run produced, that
CSV is still valid — nothing about 4.2 invalidates existing output, it only
changes what *new* runs produce by default.

## GUI users

The Spindle/Slow Wave/K-complex/PAC detection tabs in `turtlewave_gui` write
straight into `neural_events.db` — there is no GUI toggle to opt back into
per-channel JSON. If you need the legacy files, run the corresponding
`examples/hdEEG_*_detector.py --legacy-json` script instead of the GUI tab.
`eeg_review_gui` is unaffected — it already reads from `neural_events.db`.

Pointing the GUI at an existing pre-4.2 database follows the same automatic
claim described in
[One database, one subject](#one-database-one-subject-what-the-guard-does).
The GUI has no subject field to set — it derives the subject the same way the
driver scripts do (a `sub-XXXX` token in the annotation filename, else the
output directory's basename), so check the run log for the claim warning on
your first run and confirm that derived subject is the one you meant before
running anything else against that database.

## Related

- [Where did my CSV go?](read-database-with-pandas-and-r.md#where-did-my-csv-go)
- [Write Detection Results Directly to the Database](direct-to-database-detection.md)
- [Read the database with pandas and R](read-database-with-pandas-and-r.md)
- [Reference: density module](../reference/api/density.md)
- [Reference: dbwrite module](../reference/api/dbwrite.md)
- [Upgrade to 4.0](upgrade-to-4.0.md)
