# Direct-to-Database Write API Reference

The `dbwrite` module holds the shared primitives that `ParalEvents`, `ParalSWA`
and `ParalKC` use to write detected events straight into `neural_events.db`
(the `write_db=True` path), bypassing the JSON → CSV → import round-trip. It
also provides the on-demand DB → CSV export used to pull a flat file back out
of the database.

See [Direct-to-database detection](../../how-to/direct-to-database-detection.md)
for a task-oriented walkthrough, and
[About naming, subject identity & provenance conventions](../../explanation/naming-and-identity-conventions.md)
for the reasoning behind `fmt_freq_token` and the `method_db`/`method_str`
split.

A few of these primitives are worth knowing about even outside the direct-write
path:

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

::: turtlewave_hdEEG.dbwrite
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
