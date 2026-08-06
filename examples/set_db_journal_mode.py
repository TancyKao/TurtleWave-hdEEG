"""Convert ``neural_events.db`` files to a network-safe SQLite journal mode.

Use this when writing to a database fails with ``sqlite3.OperationalError:
disk I/O error`` and the database lives on a mapped network drive, an SMB/NFS
share, or a Dropbox/OneDrive-synced folder.

TurtleWave's databases are created in SQLite **WAL** (write-ahead logging) mode,
which needs an mmapped shared-memory (``-shm``) file. Network filesystems
generally cannot provide one, and SQLite reports the failure as ``SQLITE_IOERR``
= "disk I/O error". Journal mode is a persistent on-disk property, so a database
created by an earlier run stays in WAL for every later connection until it is
explicitly converted -- which is what this script does.

Close every review GUI and any other process using the databases first. Leaving
WAL takes an exclusive lock; while another connection holds the file the change
is a silent no-op, and this script reports that as a FAIL rather than pretending
it worked.

Usage
-----
One database::

    python examples/set_db_journal_mode.py /path/to/wonambi/neural_events.db

A whole subject tree::

    python examples/set_db_journal_mode.py --glob "K:/.../participants/*/wonambi/neural_events.db"

Quote the pattern so the shell does not expand it. ``--mode`` defaults to
``DELETE``; pass ``WAL`` to convert back on local disk.

After converting, also set the environment variable so newly created databases
do not go back into WAL::

    export TURTLEWAVE_SQLITE_JOURNAL=DELETE     # macOS / Linux
    set TURTLEWAVE_SQLITE_JOURNAL=DELETE        # Windows cmd
    $env:TURTLEWAVE_SQLITE_JOURNAL="DELETE"     # Windows PowerShell

.. danger::

    Never copy a WAL database without its ``-wal``/``-shm`` sidecars. Copying
    ``neural_events.db`` alone while a ``-wal`` file exists silently discards
    every commit still held in the WAL. Convert first (this script checkpoints),
    or copy ``neural_events.db*`` as a set.

If the share is failing so badly that this script cannot write either, copy the
database (with its sidecars) to local disk, convert it there, and copy it back.
"""

import argparse
import glob as globmod
import os
import sqlite3
import sys

from turtlewave_hdEEG.dbwrite import VALID_JOURNAL_MODES, set_journal_mode


def current_mode(db_path):
    """Read a database's journal mode without changing it.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    str
        Journal mode as SQLite reports it (lower-case), or ``'?'`` if it could
        not be read (so a reporting failure never aborts the conversion).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        try:
            return str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        finally:
            conn.close()
    except Exception:
        return '?'


def main():
    parser = argparse.ArgumentParser(
        description="Set the SQLite journal mode of one or more "
                    "neural_events.db files (fixes 'disk I/O error' on "
                    "network drives).")
    parser.add_argument(
        'db_path', nargs='?', default=None,
        help="Path to a single database file.")
    parser.add_argument(
        '--glob', dest='pattern', default=None,
        help="Glob matching several databases, e.g. "
             "'ROOT/*/wonambi/neural_events.db'. Quote it.")
    # Validated up front against the library's own list, so a typo fails once
    # here rather than once per database in the loop below.
    parser.add_argument(
        '--mode', default='DELETE', type=str.upper,
        choices=list(VALID_JOURNAL_MODES),
        help="Target journal mode (default: DELETE, the network-safe choice).")
    args = parser.parse_args()

    if bool(args.db_path) == bool(args.pattern):
        parser.error("give exactly one of a db_path or --glob PATTERN")

    if args.db_path:
        paths = [args.db_path]
    else:
        paths = sorted(globmod.glob(args.pattern))
        if not paths:
            print(f"No databases matched: {args.pattern}")
            return 1

    print(f"Setting journal_mode={args.mode.upper()} on {len(paths)} "
          f"database(s)\n")

    n_pass = 0
    n_fail = 0
    for path in paths:
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"no such file: {path}")
            before = current_mode(path)
            after = set_journal_mode(path, mode=args.mode)
            print(f"[PASS] {path}\n       {before} -> {after}")
            n_pass += 1
        except Exception as exc:  # one bad database must not abort the batch
            print(f"[FAIL] {path}\n       {type(exc).__name__}: {exc}")
            n_fail += 1

    print("\n" + "=" * 60)
    print(f"Done. {n_pass} passed, {n_fail} failed, {len(paths)} total.")
    if n_fail:
        print("Close every GUI/process using the failed databases and retry. "
              "If the write itself fails, copy the database and its "
              "-wal/-shm sidecars to local disk, convert there, copy back.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
