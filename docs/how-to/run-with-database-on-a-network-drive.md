# How to Run With `neural_events.db` on a Network Drive

This guide shows you how to fix `disk I/O error` when a detection or backfill
script writes to `neural_events.db` on a mapped network drive, a Dropbox- or
OneDrive-synced folder, or any other non-local filesystem.

## When to use this

**Problem:** A script that writes to the database — `backfill_cycles.py`, a
detection run with `write_db=True`, `set_db_journal_mode.py` — fails partway
through with a traceback like this:

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

**Solution:** `neural_events.db` is created in SQLite's **WAL** (write-ahead
logging) journal mode. WAL requires a memory-mapped shared-memory sidecar
file, which does not work over SMB/CIFS network shares or synced cloud
folders — SQLite surfaces this as `SQLITE_IOERR`, i.e. `disk I/O error`.
Convert the database to `DELETE` mode once with `set_journal_mode` — the
conversion is permanent, so this is normally a one-time fix per database, not
something you repeat on every run.

## Fix: convert the database once

Close every GUI and script that has the database open first. An open review
GUI or detection run holds the database, and `set_journal_mode` (and
`set_db_journal_mode.py`, which calls it) will not pretend that worked: it
checks the mode actually changed and raises `RuntimeError` if not, which the
script reports as `[FAIL]` on that database's line rather than aborting the
batch.

```bash
python examples/set_db_journal_mode.py /path/to/wonambi/neural_events.db --mode DELETE
```

Or convert every subject in a tree at once:

```bash
python examples/set_db_journal_mode.py --glob "/path/to/ROOT/*/wonambi/neural_events.db" --mode DELETE
```

Each database is reported on its own line and one failure does not abort the
batch, so a single locked or unreachable file doesn't stop the rest from
converting.

If you don't have the script available (e.g. an older install), the raw
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
    environment variable to keep a converted database off WAL; that was true
    of an earlier version of this fix, but it no longer applies. Re-run
    `set_db_journal_mode.py --glob` once per subject tree and you're done for
    every database it touched.

    The one case that still overrides a converted database on purpose is an
    *explicit* request: `TURTLEWAVE_SQLITE_JOURNAL=WAL`, or code that passes
    `journal='WAL'` directly, converts a `DELETE`-mode database back to WAL.
    That's intentional — an explicit request always wins — so don't set
    `TURTLEWAVE_SQLITE_JOURNAL=WAL` on a share unless you mean it.

## New databases on a share: force the mode with the environment variable

The conversion above fixes an *existing* database. If a script is going to
**create** a brand-new `neural_events.db` directly on the share — a first
detection run for a new subject, say — it will still default to WAL, because
there's nothing yet to preserve. For that case, set
`TURTLEWAVE_SQLITE_JOURNAL=DELETE` before running the script, so every
database the process creates (not just opens) comes out in `DELETE` mode from
the start:

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

Valid values are `DELETE`, `TRUNCATE`, `PERSIST`, `MEMORY`, `WAL`, `OFF`
(case-insensitive). `DELETE` is the network-safe choice — it is SQLite's
classic rollback journal and needs no shared-memory file. An unrecognised
value raises `ValueError` naming the variable, rather than silently falling
back to `WAL`. Once that first run has created the database in `DELETE` mode,
later runs don't need the variable either — the preserve rule above keeps it
there.

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
      `set_db_journal_mode.py` and `set_journal_mode` already do for you).
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
    `set_journal_mode` (and `set_db_journal_mode.py`) checks the mode
    actually changed and raises `RuntimeError` — reported as `[FAIL]` by the
    script — rather than pretending it worked. A plain file copy has no such
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
  — why WAL was chosen, why it's persistent, and why a plain connection now
  preserves an existing database's mode instead of imposing one.
- [Reference: `turtlewave_hdEEG.dbwrite`](../reference/api/dbwrite.md) —
  `open_write_connection`, `set_journal_mode`, `TURTLEWAVE_SQLITE_JOURNAL`.
- [Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md) — the
  workflow that first surfaced this failure.
