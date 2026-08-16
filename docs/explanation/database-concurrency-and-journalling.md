# Database Concurrency and Journalling

This page explains what SQLite's WAL (write-ahead logging) journal mode
actually does, why it turns out to be incompatible with network and
cloud-synced filesystems, why `neural_events.db` is created in the safer
`DELETE` mode by default since 4.0.2, and why the library goes out of its way
to leave an existing database's mode alone rather than imposing one on every
connection. For the task of getting a stuck database working again, see
[How to run with the database on a network drive](../how-to/run-with-database-on-a-network-drive.md).

## The concurrency shape this package writes for

`neural_events.db` is written and read by more than one process at once, but
never by more than one *writer* at once, and the two access patterns are
deliberately different:

- **Batch detection (`qsub`/PBS on the NCI Gadi cluster, or a local loop over
  subjects)** — one process per subject writes spindles, slow waves,
  K-complexes, PAC results, and cycle/duration data into that subject's own
  database. Subjects never share a database file, so there is never more than
  one writer per file from this path.
- **The review GUI** — one or more instances can be open reading the same
  database a batch job is currently writing to, so a researcher can inspect
  already-detected events on a subject while other subjects are still being
  processed.

The requirement this shape imposes is: **one writer, arbitrarily many
concurrent readers, no reader ever blocked by the writer and vice versa.**

## Rollback journal vs. write-ahead log

SQLite offers two fundamentally different ways to keep a database consistent
across a crash while a transaction is in progress.

**A rollback journal** (`DELETE`, `TRUNCATE`, `PERSIST` modes) works by
copying the *original* content of every page about to be modified into a
separate journal file before writing the change into the database file
itself. Every reader has to see either the fully-committed old state or the
fully-committed new state, so a reader holds a lock that keeps a writer from
committing while the reader is active, and a writer holds a lock that keeps
any reader from starting until the writer commits. Readers and the one
active writer serialize against each other.

**A write-ahead log** (`WAL` mode) inverts this: instead of writing the old
page content elsewhere and modifying the database file in place, new page
content is *appended* to a separate `-wal` file, and the main database file
is left untouched until a checkpoint later folds the WAL back into it.
Readers that started before a commit keep reading the pre-commit database
snapshot from the WAL/database as it stood at the time; the writer appends
new frames without ever touching a page a reader might be looking at. That is
what lets a writer proceed while readers are active, and readers proceed
while a writer is active, without either blocking the other — precisely the
property the batch-writer / concurrent-GUI-reader shape above needs. Without
it, a review GUI open on a subject whose detection job is still writing would
routinely hit `database is locked`.

## Why journal mode is persistent, and therefore infectious

Journal mode in SQLite is not a per-connection setting the way `busy_timeout`
or `synchronous` are — it is stored in the database file's own header (byte
18/19) and, for WAL specifically, in flags SQLite reads before it will even
open the file. Once a database is created or converted to WAL, **every later
connection to that file inherits WAL**, whether or not that connection ever
asked for it.

This is why a bare `sqlite3.connect(db_path)` call — in a one-off script, a
notebook, or any code path that never mentions journal mode at all — still
writes through whatever mode the file was created in, WAL included, and still
needs the shared-memory sidecar if that mode is WAL. The property was set
once, at database creation, by whichever code path happened to create the
file first, and it silently governs every connection after that. Nobody
chose WAL for that particular connection; the file remembers a choice made
somewhere else, possibly weeks earlier and by a different process entirely.
This project has hit the failure mode of that infectiousness directly: an
earlier version of `cycleprocessor.py`'s cycle-backfill path opened several
un-pragma'd `sqlite3.connect()` calls per subject, each of which silently
inherited WAL from the database's creation-time choice and needed the
shared-memory sidecar it could not get on a network share.

## Why a new database defaults to `DELETE`, not `WAL`

Before 4.0.2, a database this package *created* was set to `WAL`
unconditionally, on the reasoning that every new database benefits from the
concurrency properties above. That reasoning turned out to be wrong for the
common case: a database born in `WAL` on an SMB share, a mapped network
drive, or a Dropbox-/OneDrive-synced folder is broken from its very first
write — not merely slow, but unusable, surfacing as `disk I/O error` or, if
the sync client manages to partially reconcile the `-wal`/`-shm` sidecars,
`database disk image is malformed`. A pipeline whose whole point is to run
unattended across many subjects, some of whose output paths a researcher
chose on a shared drive without necessarily knowing SQLite's filesystem
requirements, cannot default to a mode that fails silently-until-it-doesn't
on that layout.

So since 4.0.2, `open_write_connection`'s default for a database it *creates*
is `DELETE` (`dbwrite.DEFAULT_NEW_DB_JOURNAL_MODE`), not `WAL`. `DELETE`'s
rollback journal is an ordinary file with ordinary whole-file locking
semantics, which every filesystem this package has to run on — local disk,
SMB, NFS, a sync-client folder — supports. The cost is the one named at the
top of this page: under `DELETE`, a reader and the one writer briefly
serialize against each other instead of proceeding independently, so a
review GUI open on a subject whose detection job is mid-write blocks for up
to the 60 s busy timeout rather than reading immediately. That trade is
deliberately taken as the default, because a slower read that eventually
succeeds is a wildly better failure mode than a write that corrupts the
database. `WAL` remains available as an explicit opt-in
(`TURTLEWAVE_SQLITE_JOURNAL=WAL`) for a lab that keeps `neural_events.db` on
local disk and wants the non-blocking concurrency back.

## Why `open_write_connection` preserves, rather than imposes

The same infectiousness that makes a journal mode spread to every later
connection also means a library function that sets one *unconditionally* —
even with good intentions, such as "always use `WAL` for the concurrency
benefits above" — overrides a choice the user may have made deliberately.
Concretely: if `open_write_connection` issued `PRAGMA journal_mode=WAL` on
every call regardless of the database's current state, then converting a
database to `DELETE` with `set_journal_mode` (the fix for a network-drive
database) would be undone by the very next detection run or review-GUI
connection against that file, silently, because both routes end up calling
`open_write_connection`.

So the function's rule is: **a database it creates gets `DELETE`
(`DEFAULT_NEW_DB_JOURNAL_MODE`); a database that already exists keeps
whatever mode it is already in**, unless the caller passes an explicit
`journal=` argument or sets `TURTLEWAVE_SQLITE_JOURNAL` — an explicit
request is still an override in either direction, including converting a
`DELETE`-mode database back to `WAL`, or converting a database still in
`WAL` from a pre-4.0.2 run. Precedence is a strict chain: the explicit
`journal=` argument beats `TURTLEWAVE_SQLITE_JOURNAL`, which beats the
`DELETE` default (for a database being created) or the preserve rule (for
one that already exists). The one edge case worth naming: a zero-byte
placeholder file (from `touch`, an interrupted copy, or a failed create)
counts as "existing" for this rule, since the check runs before
`sqlite3.connect()` would otherwise create the file itself, and SQLite
reports an unformatted file's mode as `delete` — so an empty placeholder
comes out in `DELETE` either way, which was already the safer of the two
outcomes even before the default changed.

The practical consequence is that `set_journal_mode` is a true one-time fix:
convert a database once, and every ordinary connection after that —
detection run, backfill script, review GUI — leaves it converted. Nothing
changes for a database created since 4.0.2 on any filesystem, because it was
already `DELETE` and preserving `DELETE` is still `DELETE`; only a database
that predates 4.0.2, or one somebody deliberately created with
`TURTLEWAVE_SQLITE_JOURNAL=WAL`, needs the conversion at all.

## Why WAL cannot work on SMB/NFS

WAL's readers-don't-block-writer property depends on a `-shm` file that every
connected process memory-maps and uses as shared state to coordinate which
WAL frames each reader has already seen. Memory-mapping a file requires the
underlying filesystem to support the same page consistently across
processes, including reliable byte-range locking on that shared region.
Network filesystems — SMB/CIFS mapped drives, NFS shares, and by extension
the local sync-client folders Dropbox and OneDrive present as ordinary local
directories but ultimately reconcile over the network — do not reliably
provide this. SQLite's own documentation states plainly that WAL does not
work over a network filesystem for this reason. In practice, on SMB, the
failure is not a clean "unsupported" error at connection time; it surfaces
intermittently as `SQLITE_IOERR` (`disk I/O error`) on whichever write
happens to hit the inconsistency, which is what makes it read like a flaky
one-off rather than a structural incompatibility.

`DELETE` mode has no such requirement — the rollback journal is an ordinary
file with ordinary whole-file locking semantics, which network filesystems
support. That is the entire reason `DELETE` is the default for a database
this package creates (above), and the reason `set_journal_mode` and
`TURTLEWAVE_SQLITE_JOURNAL=DELETE` exist for converting a database that is
*already* stuck in `WAL`: not to add a feature, but to get back to the mode
that doesn't need shared memory.

## The trade-off not taken: `locking_mode=EXCLUSIVE`

An alternative fix considered and rejected: SQLite's `locking_mode=EXCLUSIVE`
pragma takes and holds a single exclusive lock on the whole database file for
the connection's lifetime, which sidesteps a lot of the shared-memory
coordination WAL otherwise needs and can be more forgiving on flaky
filesystems. It was deliberately **not** adopted here, because it inverts the
concurrency model this package actually needs: an exclusive lock held by a
batch writer would lock every review GUI out of that subject's database for
the duration of the run, which is precisely the "researcher can review
already-detected subjects while others are still processing" workflow this
package is built to support. `DELETE` journal mode keeps the readers/writer
serialization semantics of a rollback journal (a writer and a reader still
briefly contend, but neither is locked out for an entire run) without
requiring shared memory — the right trade for a network-drive workaround,
where WAL's exclusive-lock alternative would trade one outage for another.

## Further reading

- [How to run with the database on a network drive](../how-to/run-with-database-on-a-network-drive.md)
  — the task-oriented fix.
- [Reference: `turtlewave_hdEEG.dbwrite`](../reference/api/dbwrite.md) —
  `open_write_connection`, `set_journal_mode`, `TURTLEWAVE_SQLITE_JOURNAL`,
  `DEFAULT_NEW_DB_JOURNAL_MODE`.
- [Reference: Journal-Mode CLI](../reference/api/cli.md) — the
  `turtlewave_set_journal_mode` console script.
- SQLite's own documentation on
  [Write-Ahead Logging](https://www.sqlite.org/wal.html) covers the
  network-filesystem limitation directly.
