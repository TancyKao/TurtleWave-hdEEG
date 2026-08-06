# Database Concurrency and Journalling

This page explains why `neural_events.db` is created in SQLite's WAL
(write-ahead logging) journal mode, what that mode actually does, why it
turns out to be incompatible with network and cloud-synced filesystems, and
why the library goes out of its way to leave an existing database's mode
alone rather than imposing WAL on every connection. For the task of getting a
stuck database working again, see
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

## Why `open_write_connection` preserves, rather than imposes

The same infectiousness that makes WAL spread to every later connection also
means a library function that sets a journal mode *unconditionally* — even
with good intentions, such as "always use WAL for the concurrency benefits
above" — overrides a choice the user may have made deliberately. Concretely:
if `open_write_connection` issued `PRAGMA journal_mode=WAL` on every call
regardless of the database's current state, then converting a database to
`DELETE` with `set_journal_mode` (the fix for a network-drive database) would
be undone by the very next detection run or review-GUI connection against
that file, silently, because both routes end up calling
`open_write_connection`.

So the function's rule is: **a database it creates gets `WAL`; a database
that already exists keeps whatever mode it is already in**, unless the
caller passes an explicit `journal=` argument or sets
`TURTLEWAVE_SQLITE_JOURNAL` — an explicit request is still an override in
either direction, including converting a `DELETE`-mode database back to
`WAL`. The one edge case worth naming: a zero-byte placeholder file (from
`touch`, an interrupted copy, or a failed create) counts as "existing" for
this rule, since the check runs before `sqlite3.connect()` would otherwise
create the file itself, and SQLite reports an unformatted file's mode as
`delete` — so an empty placeholder comes out in `DELETE`, not `WAL`. That is
the safer of the two possible failure directions, so it was left as-is
rather than special-cased.

The practical consequence is that `set_journal_mode` is a true one-time fix:
convert a database once, and every ordinary connection after that —
detection run, backfill script, review GUI — leaves it converted. Nothing
changes for a database on local disk, because every database this pipeline
creates is already `WAL` and preserving `WAL` is still `WAL`; only a database
somebody has deliberately converted behaves differently, which is the point.

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
support. That is the entire reason `TURTLEWAVE_SQLITE_JOURNAL=DELETE` and
`set_journal_mode` exist: not to add a feature, but to opt back into the mode
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
  `open_write_connection`, `set_journal_mode`, `TURTLEWAVE_SQLITE_JOURNAL`.
- SQLite's own documentation on
  [Write-Ahead Logging](https://www.sqlite.org/wal.html) covers the
  network-filesystem limitation directly.
