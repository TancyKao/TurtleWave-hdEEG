"""Convert ``neural_events.db`` files to a network-safe SQLite journal mode.

Thin wrapper around the installed console script. Since 4.0.2 the same tool is
available without a checkout::

    turtlewave_set_journal_mode /path/to/wonambi/neural_events.db

This file stays so the workflow in older notes and docs keeps working; both
paths run the identical :func:`turtlewave_hdEEG.cli.main`.

Use it when writing to a database fails with ``sqlite3.OperationalError: disk
I/O error`` and the database lives on a mapped network drive, an SMB/NFS share,
or a Dropbox/OneDrive-synced folder. Those filesystems cannot provide the
memory-mapped ``-shm`` file that SQLite's **WAL** (write-ahead logging) mode
needs, and SQLite reports the failure as ``SQLITE_IOERR`` = "disk I/O error".

From 4.0.2 a database TurtleWave *creates* is in ``DELETE`` mode, so this is
only needed for a database created by an earlier version, or one created with
``TURTLEWAVE_SQLITE_JOURNAL=WAL``. Journal mode is a persistent on-disk
property: such a database stays in WAL for every later connection until it is
explicitly converted, which is what this does.

Close every review GUI and any other process using the databases first. Leaving
WAL takes an exclusive lock; while another connection holds the file the change
is a silent no-op, and this reports that as a FAIL rather than pretending it
worked.

Usage
-----
One database::

    python examples/set_db_journal_mode.py /path/to/wonambi/neural_events.db

A whole subject tree, by directory or by pattern::

    python examples/set_db_journal_mode.py /data/participants
    python examples/set_db_journal_mode.py --glob "K:/.../participants/*/wonambi/neural_events.db"

Quote the pattern so the shell does not expand it. ``--mode`` defaults to
``DELETE``; pass ``WAL`` to convert back on local disk.

.. danger::

    Never copy a WAL database without its ``-wal``/``-shm`` sidecars. Copying
    ``neural_events.db`` alone while a ``-wal`` file exists silently discards
    every commit still held in the WAL. Convert first (this checkpoints), or
    copy ``neural_events.db*`` as a set.

If the share is failing so badly that this cannot write either, copy the
database (with its sidecars) to local disk, convert it there, and copy it back.
"""

import sys

from turtlewave_hdEEG.cli import main

if __name__ == "__main__":
    sys.exit(main())
