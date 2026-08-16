# Journal-Mode CLI Reference

The `cli` module backs the `turtlewave_set_journal_mode` console script — the
repair for a `neural_events.db` stuck in SQLite's WAL journal mode on a mapped
network drive, an SMB/NFS share, or a Dropbox-/OneDrive-synced folder. It
exists as a console script (rather than only an `examples/` script) because
`examples/` is not part of the installed wheel, so a `pip install
turtlewave-hdEEG` user has no other way to reach it.

See [Run with the database on a network drive](../../how-to/run-with-database-on-a-network-drive.md)
for the task-oriented walkthrough, and
[Database concurrency and journalling](../../explanation/database-concurrency-and-journalling.md)
for why journal mode matters and why it is safe to leave a database's mode
alone.

## Launch command

```bash
turtlewave_set_journal_mode /path/to/wonambi/neural_events.db
```

The positional argument is either a single `.db` file (any name), or a
directory that is searched recursively for files named `neural_events.db`
specifically — other `*.db` files under the tree are left alone. `--glob
PATTERN` matches several databases by an explicit pattern instead (quote it
so the shell does not expand it, and use it to reach a non-standard
filename).
`--mode` selects the target journal mode and defaults to `DELETE`, the
network-safe choice; valid values are `DELETE`, `TRUNCATE`, `PERSIST`,
`MEMORY`, `WAL`, `OFF`.

```bash
# Every neural_events.db under a subject tree
turtlewave_set_journal_mode /data/participants

# An explicit pattern
turtlewave_set_journal_mode --glob "K:/study/*/wonambi/neural_events.db"

# Back to WAL, on local disk only
turtlewave_set_journal_mode --mode WAL ./neural_events.db
```

Close every GUI and script using the target database(s) first. Converting
takes an exclusive lock; another open connection makes the change a
silent no-op, which `set_journal_mode` turns into an error and this CLI
reports per database as `[FAIL]` rather than pretending it worked.

Exit status is `0` when every database converted, `1` if any failed or
nothing matched, and `2` for a usage error (unknown flag, invalid `--mode`,
or giving both a path and `--glob`) — suitable for scripting in a batch
pipeline.

!!! note "The example script still works"
    `examples/set_db_journal_mode.py` (only present in a repo checkout, not
    in the installed wheel) is now a thin wrapper that calls the same
    `main()` this console script calls. If you're already used to running it
    from a checkout, there is no need to switch — both invoke identical code.

::: turtlewave_hdEEG.cli
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
