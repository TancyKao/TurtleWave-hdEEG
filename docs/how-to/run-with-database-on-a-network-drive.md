# How to Run With `neural_events.db` on a Network Drive

This guide shows you how to fix `disk I/O error` when a detection or backfill
script writes to `neural_events.db` on a mapped network drive, a Dropbox- or
OneDrive-synced folder, or any other non-local filesystem.

## When to use this

**Problem:** A script that writes to the database — `backfill_cycles.py`, a
detection run with `write_db=True`, `turtlewave_set_journal_mode` itself —
fails partway through with a traceback like this:

```text
Traceback (most recent call last):
  ...
  File "...\turtlewave_hdEEG\cycleprocessor.py", line ..., in store_cycles_to_database
    conn.commit()
sqlite3.OperationalError: disk I/O error
```

The database path is on a mapped drive letter (`K:\...`) or a synced cloud
folder, and the failure is intermittent — the same script may run cleanly on
a different night, or fail on a different subject each time.

**Solution:** the database is stuck in SQLite's **WAL** (write-ahead logging)
journal mode. WAL requires a memory-mapped shared-memory sidecar file, which
does not work over SMB/CIFS network shares or synced cloud folders — SQLite
surfaces this as `SQLITE_IOERR`, i.e. `disk I/O error`. Since turtlewave-hdEEG
4.0.2, a database this package *creates* defaults to `DELETE` journal mode,
which needs no sidecar and works everywhere — so a brand-new
`neural_events.db` on a share is safe out of the box. What still needs fixing
is a database that predates 4.0.2, or one an explicit
`TURTLEWAVE_SQLITE_JOURNAL=WAL` created: convert it to `DELETE` mode once with
`set_journal_mode` — the conversion is permanent, so this is normally a
one-time fix per database, not something you repeat on every run.

## Fix: convert the database once

Close every GUI and script that has the database open first. An open review
GUI or detection run holds the database, and `set_journal_mode` (and the
`turtlewave_set_journal_mode` command below, which calls it) will not pretend
that worked: it checks the mode actually changed and raises `RuntimeError` if
not, which the command reports as `[FAIL]` on that database's line rather
than aborting the batch.

Since 4.0.2, `turtlewave_set_journal_mode` is installed as a console script,
so it's available from any `pip install turtlewave-hdEEG` without a repo
checkout:

```bash
turtlewave_set_journal_mode /path/to/wonambi/neural_events.db --mode DELETE
```

Or convert every subject in a tree at once — either point it at the root
directory (it searches recursively for files named `neural_events.db`
specifically; use `--glob` below for any other filename), or give it an
explicit glob:

```bash
turtlewave_set_journal_mode /path/to/ROOT --mode DELETE
# or
turtlewave_set_journal_mode --glob "/path/to/ROOT/*/wonambi/neural_events.db" --mode DELETE
```

Each database is reported on its own line and one failure does not abort the
batch, so a single locked or unreachable file doesn't stop the rest from
converting. Exit status is `0` when every database converted, `1` if any
failed or nothing matched. See the [CLI reference](../reference/api/cli.md)
for the full command surface.

!!! note "Working from a repo checkout instead"
    `examples/set_db_journal_mode.py` still works and calls the identical
    `main()` function under the hood — use it interchangeably with the
    console script if you're already running from a checkout:

    ```bash
    python examples/set_db_journal_mode.py /path/to/wonambi/neural_events.db --mode DELETE
    ```

If neither the console script nor the checkout is available (e.g. a bare
environment with only the library, or a version predating 4.0.2), the raw
equivalent is below. Unlike `set_journal_mode`, nothing here checks whether
the mode actually changed and raises on your behalf — that check is exactly
what makes `set_journal_mode` safer to use, so prefer it when you can.
Without that check, a database another process still holds is a **silent
no-op**: the pragma reports the *current* mode (still `wal`) rather than
erroring, so the printed value is your only signal something is still open —
which is why the snippet below insists you inspect it before trusting the
conversion:

```python
import sqlite3

db_path = "/path/to/wonambi/neural_events.db"
conn = sqlite3.connect(db_path, timeout=60)
conn.execute("PRAGMA busy_timeout=60000")
busy, log, checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
if busy:
    raise SystemExit(
        "checkpoint blocked — something else has the database open. "
        "Close every GUI/process using it and retry; do NOT copy the .db "
        "file alone right now, it would miss commits still in -wal.")
mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
print(mode)   # MUST print 'delete' — anything else means another process
              # still holds the database and this was a silent no-op;
              # close it and retry
conn.close()
```

!!! note "The conversion sticks"
    Once a database is converted, every later plain connection — a
    detection run, a review GUI, `backfill_cycles.py` — preserves that
    choice instead of imposing WAL back onto it. You do not need to set any
    environment variable to keep a converted database off WAL. Re-run
    `turtlewave_set_journal_mode --glob` once per subject tree and you're
    done for every database it touched.

    The one case that still overrides a converted database on purpose is an
    *explicit* request: `TURTLEWAVE_SQLITE_JOURNAL=WAL`, or code that passes
    `journal='WAL'` directly, converts a `DELETE`-mode database back to WAL.
    That's intentional — an explicit request always wins — so don't set
    `TURTLEWAVE_SQLITE_JOURNAL=WAL` on a share unless you mean it.

## New databases: already safe by default since 4.0.2

Since 4.0.2, a **brand-new** `neural_events.db` — a first detection run for a
new subject, say — is created in `DELETE` mode automatically, whether it
lands on local disk, a mapped drive, or a synced cloud folder. There is
nothing to configure for this case; the fix in the previous section is only
for a database that already exists in WAL, whether it predates 4.0.2 or was
created with `TURTLEWAVE_SQLITE_JOURNAL=WAL` set.

The environment variable still has two real uses:

- **Opting a fresh database into `WAL`** on fast local disk, for its
  concurrency benefits (see
  [Database concurrency and journalling](../explanation/database-concurrency-and-journalling.md)).
  Set `TURTLEWAVE_SQLITE_JOURNAL=WAL` before running the script; nothing
  about it is network-safe, so don't set it for anything that writes to a
  share or a synced folder.
- **Forcing `DELETE`** for a colleague, a shared script, or a cluster node
  still on a pre-4.0.2 install, where a new database would otherwise still
  default to WAL. Set `TURTLEWAVE_SQLITE_JOURNAL=DELETE` before running the
  script, so every database the process *creates* (not just opens) comes out
  in `DELETE` mode:

=== "Windows (cmd)"

    ```bat
    set TURTLEWAVE_SQLITE_JOURNAL=DELETE
    python examples\backfill_cycles.py
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:TURTLEWAVE_SQLITE_JOURNAL = "DELETE"
    python examples\backfill_cycles.py
    ```

=== "macOS / Linux"

    ```bash
    export TURTLEWAVE_SQLITE_JOURNAL=DELETE
    python examples/backfill_cycles.py
    ```

!!! warning "PowerShell's `$env:` doesn't survive closing the terminal"
    `$env:TURTLEWAVE_SQLITE_JOURNAL = "DELETE"` only sets the variable for
    the current PowerShell session — it dies with the shell, so a new
    terminal (or a scheduled task, or a different PBS job) won't see it. For
    a value that needs to persist across sessions, set it with `setx`
    instead and open a new terminal afterwards for it to take effect:

    ```bat
    setx TURTLEWAVE_SQLITE_JOURNAL "DELETE"
    ```

Valid values are `DELETE`, `TRUNCATE`, `PERSIST`, `MEMORY`, `WAL`, `OFF`
(case-insensitive). An unrecognised value raises `ValueError` naming the
variable, rather than silently falling back to a default. Precedence is: an
explicit `journal=`/`mode=` argument passed in code, then
`TURTLEWAVE_SQLITE_JOURNAL`, then the `DELETE` default for a database the
call creates (or the existing database's own mode, preserved, for one that
already exists). Once a run has created (or converted) a database in the
mode you want, later runs don't need the variable either — the preserve rule
above keeps it there.

## Copy the database to local disk instead

If the share keeps failing even in `DELETE` mode (see the limit below), or
you just want a faster working copy, copy the database to local disk, work
against the local copy, then copy the result back.

!!! danger "Never copy a WAL database without its sidecars"
    While a database is in WAL mode, recently committed data can live only in
    `neural_events.db-wal`, not in `neural_events.db` itself. Copying
    `neural_events.db` alone while a `-wal` file sits next to it **silently
    discards every commit still in the WAL** — no error, no warning, just
    quietly missing rows in the copy.

    Do one of the following before copying:

    - Checkpoint first, so everything is flushed into the main file:
      `PRAGMA wal_checkpoint(TRUNCATE)` (this is what
      `turtlewave_set_journal_mode` and `set_journal_mode` already do for
      you).
    - Or copy the whole set together — `neural_events.db`,
      `neural_events.db-wal`, and `neural_events.db-shm` — never the `.db`
      file on its own.

!!! warning "Applies to synced cloud folders too, and close every GUI first"
    This is not only a mapped-drive problem. A `neural_events.db` living in a
    Dropbox- or OneDrive-synced folder hits the same WAL/shared-memory
    limitation, because the sync client sees the same non-local filesystem
    semantics a network share does.

    Also close every GUI and script before converting or copying. An open
    review GUI holds its own connection to the database. Converting the
    journal mode while that connection is live is safe either way:
    `set_journal_mode` (and `turtlewave_set_journal_mode`) checks the mode
    actually changed and raises `RuntimeError` — reported as `[FAIL]` by the
    command — rather than pretending it worked. A plain file copy has no such
    check: copying while a GUI holds the file open is a silent no-op on the
    copy's correctness, with no error and no warning.

## The limit of this fix

On a share that is already failing, the conversion itself is a write, and it
can fail with the very same `disk I/O error` it's meant to fix. If that
happens, or if the database is already in `DELETE` mode and the
`disk I/O error` persists, the share is not failing because of WAL's
shared-memory requirement — it's failing on plain byte-range file locking,
which no journal mode setting can work around. Copy-to-local (above) is then
the only supported workflow: do all detection and backfill work against a
local copy of the database, and only copy the finished file back to the
network location afterwards.

The durable fix is to keep `neural_events.db` on local disk in the first
place, and treat the network/cloud-synced copy as a destination you copy
*finished* files to, not a location scripts write into directly.

## See also

- [Explanation: Database concurrency and journalling](../explanation/database-concurrency-and-journalling.md)
  — why WAL exists, why `DELETE` is the safe default for a new database, why
  journal mode is persistent, and why a plain connection preserves an
  existing database's mode instead of imposing one.
- [Reference: `turtlewave_hdEEG.dbwrite`](../reference/api/dbwrite.md) —
  `open_write_connection`, `set_journal_mode`, `TURTLEWAVE_SQLITE_JOURNAL`,
  `DEFAULT_NEW_DB_JOURNAL_MODE`.
- [Reference: Journal-Mode CLI](../reference/api/cli.md) — the
  `turtlewave_set_journal_mode` console script's full command surface.
- [Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md) — the
  workflow that first surfaced this failure.
