# Direct-to-Database Write API Reference

The `dbwrite` module holds the shared primitives that `ParalEvents`, `ParalSWA`,
`ParalKC` and `ParalPAC` use to write results straight into `neural_events.db`
— the default path since 4.2 (`write_db=None`). It also provides the on-demand
DB → CSV export used to pull a flat file back out of the database, and the
`analysed_time` schema/readers that back
[`turtlewave_hdEEG.density`](density.md).

See [Direct-to-database detection](../../how-to/direct-to-database-detection.md)
for a task-oriented walkthrough, and
[About naming, subject identity & provenance conventions](../../explanation/naming-and-identity-conventions.md)
for the reasoning behind `fmt_freq_token` and the `method_db`/`method_str`
split.

A few of these primitives are worth knowing about even outside a detection call:

- **`resolve_db_target(db_path, output_dir, logger)`** is the single source of
  truth for "which database file does this run write to" — explicit `db_path`
  wins, then a database beside `output_dir`/`json_dir`, then
  `./neural_events.db`. Every detector and `analyze_pac` call this rather than
  resolving a path locally; an unresolvable target raises instead of silently
  downgrading to no write at all.
- **`recording_root_from_db(db_path)`** derives the recording's root directory
  from a database path (stripping a trailing `wonambi` component), used as the
  `root_dir` hint for `derive_subject`.
- **`ensure_analysed_time_schema` / `record_analysed_time` / `store_analysed_time`
  / `read_analysed_time`** create and populate the `analysed_time` table — the
  artefact-free density denominator every direct-write detection run stores
  automatically, and the only thing
  [`density.event_density`](density.md) reads besides `events` itself.
  `store_analysed_time` swallows its own failure by default (a detector run
  that already succeeded must not be lost to a denominator problem) — pass
  `strict=True` to raise instead, for a caller whose whole job *is* that
  write, such as a back-fill script. Additive; the default is unchanged, so
  the three detectors' own behaviour doesn't change.
- **`fmt_freq_token(lo, hi)`** is the single source of truth for the
  `{freq_lo}-{freq_hi}Hz` filename/pattern token — use it on both sides of any
  round-trip that names or re-finds a detector's output.
- **`verify_channel_coverage`** answers "did every requested channel actually
  make it into the database" for a completed run; both `_GADI.py` batch
  drivers call it and exit non-zero on a gap. Its return dict's `failed` and
  `events_only` keys distinguish an in-scope crash from a channel only
  vouched for by (stage-unscoped) event rows, and `scoped_status=False` flags
  a database that predates the per-scope `processing_status` schema, where the
  check falls back to a weaker, event-type-only comparison.
- **`guard_run_id`** is what a CSV import (`import_parameters_csv_to_database`
  and friends) consults before writing: it refuses to blank the `run_id`
  provenance link on rows the direct-write path already wrote, unless the
  caller passes `force=True`.
- **`open_write_connection` preserves an existing database's journal mode; it
  only imposes `WAL` on a database it creates.** Converting a database with
  `set_journal_mode` therefore sticks — a later detection run or review-GUI
  connection will not silently flip it back. See
  [Database concurrency and journalling](../../explanation/database-concurrency-and-journalling.md#why-open_write_connection-preserves-rather-than-imposes)
  for the rationale.
- **`TURTLEWAVE_SQLITE_JOURNAL`** is an explicit override in either
  direction — it forces the named mode on every database the process opens,
  including ones it creates, and beats the preserve rule above. Its main
  remaining job is forcing `DELETE` on a *brand-new* `neural_events.db`
  created directly on a mapped/network drive or synced cloud folder, since
  there's nothing yet on disk to preserve. See
  [Run with the database on a network drive](../../how-to/run-with-database-on-a-network-drive.md)
  and `set_journal_mode` for converting an existing database permanently.
- **`VALID_JOURNAL_MODES`** is the public tuple of journal modes SQLite
  accepts (`DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF`) — both
  `TURTLEWAVE_SQLITE_JOURNAL` and the `journal`/`mode` arguments of
  `open_write_connection`/`set_journal_mode` are validated against it, and
  `examples/set_db_journal_mode.py` uses it directly as its `--mode`
  argparse `choices`, so a typo is rejected at the CLI before any database is
  touched.

### Stage tokens

Since 4.3, `events.stage` stores a run's canonical **joint** stage token
(`'NREM2NREM3'` for a run over N2+N3) rather than each event's own epoch —
see
[What `events.stage` means](../../how-to/direct-to-database-detection.md#what-eventsstage-means).
This vocabulary is what every reader and writer uses instead of a raw
`''.join(stage)`:

- **`join_stage_token(stages)`** — the one write-side function. Dedupes and
  sorts into canonical order (NREM1, NREM2, NREM3, REM, Wake, then anything
  else alphabetically) before joining, so a caller passing
  `['NREM3', 'NREM2']` writes the same token as one passing
  `['NREM2', 'NREM3']`. Round-trips with `split_stage_token`.
- **`split_stage_token(stage)`** / **`stage_components(stage)`** — the
  read-side decomposition. `split_stage_token` raises on an unfamiliar label;
  `stage_components` is the forgiving counterpart used by every reader (an
  unsplittable token, e.g. `'Undefined'`, is returned as a one-item list
  rather than failing the whole query).
- **`stage_tokens_covering(tokens, requested)`** — the single read-side
  primitive: which of the stored tokens fall inside a requested stage set.
  This is what keeps a reader correct against **both** a pre-4.3 per-epoch
  database (one-component tokens) and a joint one at once. A request for
  `['NREM2']` alone does **not** cover a stored `'NREM2NREM3'` row — a joint
  row cannot be attributed to one of its components, so that is a missing
  answer, not a wrong one.
- **`resolve_stage_tokens(conn, requested, where, params)`** — the
  query-building counterpart: reads `SELECT DISTINCT stage` (or another
  column) under an optional `WHERE`, then calls `stage_tokens_covering`. What
  callers put in `stage IN (...)`; an empty return means "no stored token is
  inside the requested set" and must be rendered as a matched-nothing filter,
  never as "no filter."
- **`pooled_denominator(token, denom)`** — sums `analysed_time`'s
  per-single-stage `analysed_seconds` / `artefact_seconds_excluded` across a
  token's components. All-or-nothing: `NaN`, never a partial sum, if any
  component is missing from `denom`. What
  [`density.event_density`](density.md) and `v_event_density` (below) both
  use.
- **`db_meta` / `stage_format(conn)` / `assert_stage_format_compatible(conn,
  event_type, methods, freq_lower, freq_upper, *, stage_token, channels=None,
  replace_channels=None, db_path=None, logger=None)`** — the `db_meta`
  key/value table (created by `ensure_direct_write_schema`) carries a
  `stage_format` marker: `'joint'` for a 4.3+ database, `'per_epoch'` or
  absent for one written by 4.2 and earlier. `assert_stage_format_compatible`
  is called by every direct-write detection before its first write, and
  **raises `ValueError` in two independent situations**, both because
  `event_uuid5` hashes the stage and the stage is also the last column of the
  `event_chan_time` UNIQUE constraint — so a re-detection whose rows would
  carry a different `stage` than what's stored produces both a new primary
  key and a new unique key, and `INSERT OR REPLACE` appends a duplicate set
  instead of replacing:
  1. the database is unmarked or `'per_epoch'` and already holds rows for the
     scope about to be written (a pre-4.3 database) — fixed by
     `examples/migrate_stage_to_joint.py`;
  2. the scope already holds rows under a stage token **different** from
     `stage_token` — the run's own canonical token
     (`join_stage_token(stage)`) — even on a database that is fully
     `'joint'`, since both runs are "joint," just over different stage sets.
     `stage_token` is a **required keyword-only argument** precisely so this
     check can't be silently skipped by a caller that omits it. Fixed by
     re-detecting with `replace_channels=<the affected channels>` (named in
     the error message itself), not by the migration script.
  See
  [A database refuses a re-detection that would duplicate events](../../how-to/direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events)
  for both error messages and the escape hatch for each.
- **`ensure_density_view(conn)`** creates the `v_event_density` SQL view
  (part of `ensure_direct_write_schema`), so R, `sqlite3` and any BI tool can
  read density without Python. See
  [density reference: `v_event_density`](density.md#v_event_density-density-in-plain-sql-for-r-and-non-python-callers).
- **`ensure_cycles_populated(conn, annotations, subject, ...)`** /
  **`tag_run_cycles(conn, subject, run_id, method)`** — called automatically
  by all three detectors so `sleep_cycles`, `stage_durations` and
  `events.cycle` populate on a normal detection run, with no separate
  `finalize_cycles_and_durations` call needed. Both pass
  `write_xml=False, plot=False` unconditionally to the cycle finalizer — a
  detection run never modifies the annotation XML. See
  [Cycles and stage durations populate automatically](../../how-to/direct-to-database-detection.md#cycles-and-stage-durations-populate-automatically).

::: turtlewave_hdEEG.dbwrite
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
