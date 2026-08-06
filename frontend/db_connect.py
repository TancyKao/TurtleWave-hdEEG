#!/usr/bin/env python3
"""One way for every GUI in this package to open ``neural_events.db``.

Deliberately depends on nothing but the standard library. Importing this module
must never require PyQt5, pyqtgraph or ``turtlewave_hdEEG``, because the GUI
modules that use it must themselves stay importable in a headless or partial
install.
"""

import sqlite3

# Optional: the library is not guaranteed to be importable from a GUI-only or
# partially installed environment, and a failure here must not stop this module
# (or any GUI importing it) from loading. Same pattern as the GUI imports in
# ``frontend/__init__.py`` and ``turtlewave_hdEEG/__init__.py``.
try:
    from turtlewave_hdEEG.dbwrite import open_write_connection
except ImportError:
    open_write_connection = None

#: Busy-wait applied to every connection opened here, in seconds. Python's
#: ``sqlite3.connect`` default is 5 s; this matches the 60 s
#: ``turtlewave_hdEEG.dbwrite.open_write_connection`` gives the writer side.
BUSY_TIMEOUT_SECONDS = 60.0


def connect_events_db(db_path, write=False):
    """Open ``neural_events.db`` with a long busy timeout and no journal pragma.

    Every GUI in this package connects through here so that three properties
    hold in one place rather than in four copies.

    **No journal mode is imposed.** Journal mode is a persistent *on-disk*
    property, so the ``PRAGMA journal_mode=WAL`` the review GUIs used to issue
    flipped any database the user had converted to ``DELETE`` -- the only mode
    that works on a mapped network drive -- straight back to WAL, re-breaking
    the batch pipeline the moment they opened a GUI.
    :func:`turtlewave_hdEEG.dbwrite.open_write_connection` preserves an existing
    database's mode and imposes one only when ``journal=`` or
    ``TURTLEWAVE_SQLITE_JOURNAL`` says to, so a converted database stays
    converted, a WAL database stays WAL, and the environment variable still
    overrides either way.

    **No ``mmap_size`` either.** Memory mapping is precisely what fails on
    SMB/NFS shares, which is the failure this whole change exists to fix.

    **A 60 s busy timeout on every path**, up from Python's 5 s default. Five
    seconds was survivable only because these databases were all in WAL, where a
    reader does not block on a writer at all; on a database converted to
    ``DELETE`` a GUI query contends with a running detection job, and outwaiting
    it is the difference between a working GUI and ``database is locked``. The
    ``timeout`` keyword sets exactly the ``busy_timeout`` pragma, so it is not
    set again separately.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database, normally ``neural_events.db``.
    write : bool, optional
        ``True`` when the caller will write (e.g. ``CREATE INDEX``). Those route
        through :func:`turtlewave_hdEEG.dbwrite.open_write_connection` when the
        library is available, so writers share one definition of how this
        pipeline opens a database for writing. Read-only callers connect
        directly: there is nothing the library helper adds for a reader, and it
        keeps the read path working where the library is not installed.

    Returns
    -------
    sqlite3.Connection
        Open connection with ``busy_timeout=60000`` ms.

    Raises
    ------
    ValueError
        Propagated from ``open_write_connection`` when
        ``TURTLEWAVE_SQLITE_JOURNAL`` names a mode SQLite does not recognise.
        Deliberately not caught: the value came from the user, and the message
        names the variable and lists the valid modes.

    Notes
    -----
    A non-existent `db_path` is created by either path, as ``sqlite3.connect``
    always has been. Only the writer path chooses a journal mode for a database
    it creates (WAL, per ``open_write_connection``); the reader path leaves a
    newly created file in SQLite's own ``DELETE`` default.
    """
    if write and open_write_connection is not None:
        return open_write_connection(db_path)
    return sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS)
