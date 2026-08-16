# How to Upgrade to turtlewave-hdEEG 4.3

4.3 changes what `events.stage` means — from the event's own scored epoch to
the detection run's stage *set* — and makes a detection run fill sleep cycles
and stage durations on its own. This guide covers what changes for existing
scripts, PBS jobs and GUI usage, and what to do about a database that
predates the change.

If you're also crossing the 4.2 boundary, read
[Upgrade to 4.2](upgrade-to-4.2.md) first — the direct-write-by-default
change is independent of, and precedes, this one.

## Before you start

```bash
cp neural_events.db neural_events.db.bak-pre-4.3
```

The schema change is additive (new `events.epoch_stage` column, new `db_meta`
table, new `v_event_density` view; no columns dropped or retyped), but back up
before running 4.3 code against a database you care about, same as any other
version. If you go on to run `examples/migrate_stage_to_joint.py --apply`, it
takes its own backup automatically before writing — see
[Migrate a database to the joint stage token](migrate-stage-to-joint.md).

## What changed

- **`events.stage` is now the run's joint stage token, not the event's own
  epoch.** A run over `stage=['NREM2', 'NREM3']` stores the joined token
  `'NREM2NREM3'` on every event it writes — from all three detectors
  (spindles, slow waves, K-complexes), on both the direct-write and legacy
  JSON paths. A single-stage run stores `'NREM2'`. **Before 4.3, every event
  type stored per-epoch stage on the direct-write path** — spindles and
  K-complexes were no different from slow waves there, and a database written
  by 4.2 (the direct-write default) has per-epoch stages on all three,
  spindles included. The joined form only ever appeared through the legacy
  JSON → CSV → import route, and only for spindles and K-complexes on that
  route — slow waves stayed per-epoch there too. See
  [What `events.stage` means](direct-to-database-detection.md#what-eventsstage-means)
  for the full explanation, including the rule that **a strict subset of a
  stored joint token returns no row, not a wrong number.**
- **New additive `events.epoch_stage` column** holds each event's own scored
  epoch (`'NREM2'` or `'NREM3'`), computed for free at detection time. Use it
  if you need the N2-vs-N3 split that `events.stage` no longer carries — a
  plain `GROUP BY channel, epoch_stage` recovers it.
- **New stage-token vocabulary in `dbwrite`** — `join_stage_token`,
  `stage_components`, `stage_tokens_covering`, `resolve_stage_tokens`,
  `pooled_denominator`. `events.stage` itself and every reader of it
  (`export_events_to_csv`, `count_db_events`, `pacprocessor`'s DB queries, the
  CSV density exporters, `default_csv_path`) now go through
  `join_stage_token`/`resolve_stage_tokens` instead of a raw `''.join(stage)`.
  A few raw joins remain **outside** that path, deliberately or not yet
  converted: the CSV-import stage normalizers in `eventprocessor.py` /
  `swprocessor.py` (matching the historical CSV column format on purpose, not
  a stage-identity computation); and, still open, the PAC output *directory*
  naming in `pacprocessor.py` and the stage filter in
  `examples/hdEEG_pac_detector.py`, which are order-dependent and not
  currently canonicalised the way `pac_coupling.stage` itself now is — check
  the CHANGELOG for the current status of that gap before relying on PAC
  output directory names matching a differently-ordered stage list. See the
  [dbwrite reference](../reference/api/dbwrite.md#stage-tokens).
- **New `db_meta` table with a `stage_format` marker.** A database written by
  4.2 or earlier (or one with rows but no marker at all) is unmarked or
  `'per_epoch'`; a 4.3+ database is `'joint'`. `dbwrite.assert_stage_format_compatible`
  is called before every direct-write detection and **raises** in two
  situations: the marker-based one just described, and a **new, independent**
  one — re-detecting an already-joint scope under a *different* stage set
  than it was written with (widening or narrowing which stages you detect
  over, on channels you've already run). The second one fires even on a
  database that is fully migrated, since the marker alone can't see it — see
  [A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events)
  below for why and how to clear each.
- **New `v_event_density` SQL view**, so R and plain SQL can read density
  without Python. Created automatically by `ensure_direct_write_schema`, so
  every current database has it. It differs from `density.event_density` in
  three ways: no honest zeros (a channel that ran and fired nothing has no
  row), no per-identity stage scope (nothing is filtered by which run
  searched what), and it under-reports density (returns `NULL` instead of a
  number) for a stage label outside the known vocabulary even when
  `analysed_time` holds a matching row — see the
  [density reference](../reference/api/density.md#v_event_density-density-in-plain-sql-for-r-and-non-python-callers).
- **Cycles and stage durations now populate automatically on a detection
  run.** `detect_spindles`, `detect_slow_waves` and `detect_kcomplexes` each
  call `finalize_cycles_and_durations` themselves after a run — both
  `'2022'` and `'1979'` are stored in `sleep_cycles`, `'2022'` owns
  `events.cycle`. **A detection run never writes the annotation XML** —
  `write_xml=False, plot=False` are passed unconditionally and are not
  caller-controllable from `detect_*`. See
  [Cycles and stage durations populate automatically](direct-to-database-detection.md#cycles-and-stage-durations-populate-automatically).
  You can still call
  [`finalize_cycles_and_durations`](detect-sleep-cycles.md) yourself for a
  pre-4.3 database, a failed automatic back-fill, or finer control over
  `wake_thresh`/`nrem_min`/`rem_min`.
- **`finalize_cycles_and_durations` now raises `ValueError`** when
  `tag_method` isn't one of `methods`, instead of silently writing no XML
  markers and letting the last-run method own `events.cycle` with nothing
  recording the disagreement. See
  [`ValueError: tag_method is not one of methods`](detect-sleep-cycles.md#valueerror-tag_method-is-not-one-of-methods).
- **New `examples/migrate_stage_to_joint.py`** collapses a pre-4.3 database's
  per-epoch stages to the run's joint token and stamps
  `db_meta.stage_format = 'joint'`, so it can accept a re-detection without
  raising. Dry run by default; `--apply` to write. See
  [Migrate a database to the joint stage token](migrate-stage-to-joint.md).

## Does your existing script need changes?

**No, if** you only ever read `events.stage` to filter a query by the exact
stage set your run requested (`WHERE stage = 'NREM2NREM3'`, or
`stage IN (...)` built from `dbwrite.resolve_stage_tokens`) — that continues
to work the same way it always has, and now agrees across all three event
types.

**No, if** you don't re-detect into existing databases — a script that
always writes into a fresh `neural_events.db`, or that always passes
`replace_channels=<channels this run writes>`, never hits the
`assert_stage_format_compatible` refusal.

**Yes, if** any of the following are true:

- **You compute an N2-vs-N3 (or any per-component) split by grouping on
  `events.stage`.** That value is now the run's joint token, not the
  per-epoch stage. Switch the `GROUP BY` to the new `events.epoch_stage`
  column instead.
- **You re-detect a scope against an existing database that was last
  written by 4.2 or earlier code.** The first such run raises
  `ValueError` naming `examples/migrate_stage_to_joint.py`. Either run that
  script against the database first, or re-run detection into a fresh
  database / with `replace_channels` set to the channels you're
  re-detecting. See
  [A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events).
- **You re-detect the same channels over a DIFFERENT stage set than you
  detected them with before** — widen an N2-only run to N2+N3, or narrow the
  other way. This raises even on a database that is already fully migrated
  to the joint format (a **new** check, independent of the marker), naming
  the stored token(s), the token the new run would write, and the affected
  channels. This is an ordinary, expected thing to do, not a sign of a
  broken database — pass `replace_channels=<those channels>` to proceed; see
  the same section as above.
- **You call `finalize_cycles_and_durations` with a `tag_method` that isn't
  in `methods`**, expecting the old (buggy) silent behaviour. It now raises;
  fix the call (put `tag_method` in `methods`, or pass `tag_method=None` if
  you meant to tag nothing).
- **You call `density.event_density(..., combine_stages=True)` against a
  normal joint-token run** (one stage token, already the union you asked
  for). This is now a correct no-op and is logged as such — before 4.3 it
  logged in a way that read like a failure even though nothing was wrong.
- **You've stopped calling `finalize_cycles_and_durations` explicitly and
  expect `events.cycle` / `sleep_cycles` to stay empty until you do** — as
  of 4.3 a normal detection run fills them itself. This is additive
  behaviour, not a breaking one, but it means a script that used to check
  "cycles are empty" as a signal of "cycle finalize hasn't run yet" needs a
  different signal now.

## Regenerate outputs you rely on

Detection results themselves are unaffected in what gets detected — 4.3 does
not change spindle/slow-wave/K-complex thresholds or morphology. What
changes is `events.stage`'s meaning on any *new* run, and whether cycles get
filled automatically. You do not need to re-run detection just to upgrade a
database that only ever gets read, not written to, going forward — the
pre-4.3 data is still valid and queryable, you only need
`examples/migrate_stage_to_joint.py` (or `assert_stage_format_compatible`'s
refusal) once you try to write into it again with 4.3+ code.

## Migrate vs. re-run: which one you need

This only applies to **check 1** (a pre-4.3, per-epoch database). Migrating
does rewrite `events.stage` itself — that's exactly what check 2 compares —
but only to the token the scope's *own recorded* stage set collapses to; it
does not change what any channel was actually searched over. So migrating
lets a **same**-stage-set re-run through cleanly (the token it wrote now
matches what that re-run will write), but it does nothing for **check 2** (a
*different* stage set re-detected onto the same channels afterwards) — that
still raises, because the newly-written token still won't match. `replace_channels`
is the only way past check 2; see
[A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events).
For check 1, migrate and re-run trade off differently:

| | Migrate (`migrate_stage_to_joint.py`) | Re-run detection |
|---|---|---|
| Keeps existing event `uuid`/`run_id` provenance | Yes, unchanged | No — new uuids, new `run_id` |
| Needs the scoring file | Only for the optional density/cycle back-fills | Yes, always |
| Re-computes `analysed_time` (density denominator) | Only if you pass `--backfill-analysed-time` (default on when empty) | Yes, automatically |
| Re-computes `sleep_cycles`/`stage_durations`/`events.cycle` | Only if you pass `--backfill-cycles` (default on when empty) | Yes, automatically |
| Cost | One `UPDATE` per channel group, in one transaction, over existing rows | Full re-detection cost |
| Reversible | Only by restoring the pre-migration backup | Only by restoring whatever backup you took before re-running |

If a scope was never fully detected, or you'd rather not trust old detector
output, re-run. If the events are correct and you only need `db_meta` and the
stage column brought current, migrate.

## GUI users

The Spindle/Slow Wave/K-complex detection tabs in `turtlewave_gui` inherit
this change automatically — a run against a pre-4.3 database that already
holds rows for the requested scope shows the same refusal error the
`detect_*` functions raise, naming `examples/migrate_stage_to_joint.py`.
Run the migration script from a terminal (there's no GUI button for it yet)
before re-running that scope from the GUI.

`eeg_review_gui` **was** affected, and the fix already shipped in this
release: before it, the QC dashboard's density read `events.stage` as if it
were a single scored stage, so a joint token like `'NREM2NREM3'` matched no
`analysed_time` row and QC density silently went to zero. The fix is
`qc_density_stage_scope`, which splits a joint token back into its component
stages (via `dbwrite.split_stage_token`, dropping Wake) before computing
density, the same way `events.epoch_stage` does for a `GROUP BY`. No action
is needed from you — this is fixed code you're already running as of 4.3, not
something to migrate or configure.

## Related

- [Write Detection Results Directly to the Database](direct-to-database-detection.md) —
  the full explanation of what `events.stage` means and the refusal mechanics.
- [Migrate a database to the joint stage token](migrate-stage-to-joint.md)
- [Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md)
- [Read the database with pandas and R](read-database-with-pandas-and-r.md) —
  `v_event_density` for SQL/R.
- [Reference: dbwrite module](../reference/api/dbwrite.md)
- [Reference: density module](../reference/api/density.md)
- [Upgrade to 4.2](upgrade-to-4.2.md)
