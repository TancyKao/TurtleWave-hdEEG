"""Command-line entry points shipped with the installed package.

Currently one: ``turtlewave_set_journal_mode``, which converts existing
``neural_events.db`` files to another SQLite journal mode. It exists as a
console script because the equivalent example script
(``examples/set_db_journal_mode.py``) is not part of the wheel, so a user who
installed from PyPI has no way to reach it.

This module deliberately imports nothing from :mod:`frontend` and no Qt: the
repair it performs is most often needed on a headless cluster node or from a
plain terminal on a machine where the GUIs will not even start.
"""

import argparse
import glob as globmod
import os
import sqlite3
import sys

from .dbwrite import VALID_JOURNAL_MODES, set_journal_mode

__all__ = ['DB_FILENAME', 'current_mode', 'find_databases', 'main']

# The project's canonical database filename. Directory mode matches this and
# nothing else, so a recursive search never converts (or fails on) a file this
# tool does not own. --glob is the escape hatch for a non-standard name.
DB_FILENAME = 'neural_events.db'


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
        not be read, so a reporting failure never aborts the conversion this
        is only annotating.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        try:
            return str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        finally:
            conn.close()
    except Exception:
        return '?'


def find_databases(target=None, pattern=None):
    """Resolve the CLI's target into a sorted list of database paths.

    Directory mode searches for this project's canonical database filename,
    :data:`DB_FILENAME` (``neural_events.db``), **not** every ``*.db`` beneath
    the tree. Converting the journal mode of an arbitrary ``.db`` is a
    persistent change to a file this tool does not own: a broad ``*.db`` sweep
    silently re-modes a third-party database that happens to live under the
    same study folder and reports it ``[PASS]``, and a stray non-SQLite
    ``.db`` file makes the whole batch exit non-zero even though every real
    database converted. ``--glob`` remains the escape hatch for a
    non-standard name, where the user has named the target explicitly.

    Parameters
    ----------
    target : str or None, optional
        A single database file -- any filename, since naming a file is
        explicit -- or a directory to search recursively for
        :data:`DB_FILENAME`. A directory is how the subject-tree case
        (``ROOT/<subject>/wonambi/neural_events.db``) is reached without
        having to quote a shell glob, which is awkward in PowerShell.
    pattern : str or None, optional
        Glob matching several databases, e.g.
        ``'ROOT/*/wonambi/neural_events.db'``. Expanded here rather than by the
        shell, so the caller must quote it. Not filtered by name: an explicit
        pattern is the user saying exactly which files they mean.

    Returns
    -------
    list of str
        Sorted, de-duplicated paths. A `target` naming a single file is
        returned as-is without an existence check, so a typo is reported per
        database by the caller rather than silently yielding an empty list.

    Raises
    ------
    ValueError
        If neither or both of `target` and `pattern` were given. Exactly one
        is required; accepting both would leave the precedence between them
        undefined.
    """
    if bool(target) == bool(pattern):
        raise ValueError(
            "give exactly one of a path (file or directory) or --glob PATTERN")
    if pattern:
        return sorted(set(globmod.glob(pattern, recursive=True)))
    if os.path.isdir(target):
        # Canonical filename only -- see the docstring: a '*.db' sweep would
        # re-mode databases this tool does not own and fail the batch on
        # unrelated files.
        return sorted(set(globmod.glob(
            os.path.join(target, '**', DB_FILENAME), recursive=True)))
    return [target]


def main(argv=None):
    """Convert one or more databases to a SQLite journal mode. Console script.

    Installed as ``turtlewave_set_journal_mode``. The repair for a
    ``neural_events.db`` that fails with ``disk I/O error`` on a mapped network
    drive, an SMB/NFS share or a Dropbox/OneDrive-synced folder: those
    filesystems cannot provide the memory-mapped ``-shm`` file that WAL
    journalling needs. Journal mode is a persistent on-disk property, so a
    database created by an older TurtleWave (or with
    ``TURTLEWAVE_SQLITE_JOURNAL=WAL``) stays in WAL until it is converted once,
    here.

    Close every review GUI and any other process using the databases first.
    Leaving WAL takes an exclusive lock; while another connection holds the
    file the change is a silent no-op, which
    :func:`~turtlewave_hdEEG.dbwrite.set_journal_mode` turns into an error and
    this reports as ``[FAIL]`` rather than pretending it worked.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector, excluding the program name. ``None`` (the default)
        reads :data:`sys.argv`. Passing a list is what makes this testable.

    Returns
    -------
    int
        Process exit status: ``0`` when every database converted, ``1`` if any
        failed or nothing matched. Suitable for ``sys.exit(main())``. A usage
        error -- an unknown flag, a bad ``--mode``, or giving both a path and
        ``--glob`` -- never reaches this return: :mod:`argparse` prints the
        usage message and exits ``2`` directly.

    Notes
    -----
    One failing database never aborts the batch: each is reported ``[PASS]``
    or ``[FAIL]`` with its before/after mode and the run continues, because a
    single locked subject should not block the other 40.

    A directory argument matches ``neural_events.db`` only, not every ``*.db``
    beneath the tree (:func:`find_databases`); use ``--glob`` for a
    non-standard filename.

    Converting is not destructive, but a WAL database can hold committed rows
    in its ``-wal`` sidecar;
    :func:`~turtlewave_hdEEG.dbwrite.set_journal_mode` checkpoints before
    converting and warns if the checkpoint was blocked. Never copy a WAL
    database without its ``-wal``/``-shm`` sidecars.

    Examples
    --------
    One database::

        turtlewave_set_journal_mode /path/to/wonambi/neural_events.db

    Every ``neural_events.db`` under a subject tree::

        turtlewave_set_journal_mode /data/participants

    An explicit pattern (quote it so the shell does not expand it)::

        turtlewave_set_journal_mode --glob "K:/study/*/wonambi/neural_events.db"

    Back to WAL, on local disk only::

        turtlewave_set_journal_mode --mode WAL ./neural_events.db
    """
    parser = argparse.ArgumentParser(
        prog='turtlewave_set_journal_mode',
        description="Set the SQLite journal mode of one or more "
                    "neural_events.db files (fixes 'disk I/O error' on "
                    "network and cloud-synced drives). Close every GUI first.")
    parser.add_argument(
        'path', nargs='?', default=None,
        help=f"A single database file (any name), or a directory to search "
             f"recursively for {DB_FILENAME} files. Other *.db files under a "
             f"directory are ignored -- use --glob to reach them.")
    parser.add_argument(
        '--glob', dest='pattern', default=None,
        help="Glob matching several databases by an explicit pattern, e.g. "
             "'ROOT/*/wonambi/neural_events.db'. Quote it. Use this for a "
             "database whose filename is not the canonical one.")
    # Validated up front against the library's own list, so a typo fails once
    # here rather than once per database in the loop below.
    parser.add_argument(
        '--mode', default='DELETE', type=str.upper,
        choices=list(VALID_JOURNAL_MODES),
        help="Target journal mode (default: DELETE, the network-safe choice).")
    args = parser.parse_args(argv)

    try:
        paths = find_databases(args.path, args.pattern)
    except ValueError as exc:
        parser.error(str(exc))     # exits 2

    if not paths:
        where = args.pattern if args.pattern else args.path
        print(f"No databases matched: {where}")
        if not args.pattern and os.path.isdir(args.path):
            # The narrowed directory search is the likely surprise here: say so
            # rather than leave the user thinking the tree is empty.
            print(f"A directory is searched for {DB_FILENAME} only. For a "
                  f"database with a different filename, name the file "
                  f"directly or use --glob \"{args.path}/**/*.db\".")
        return 1

    print(f"Setting journal_mode={args.mode} on {len(paths)} database(s)\n")

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


if __name__ == '__main__':
    sys.exit(main())
