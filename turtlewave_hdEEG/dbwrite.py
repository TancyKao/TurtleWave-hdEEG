"""Direct-to-database event write path (P2, stage 1).

Shared primitives used by :class:`~turtlewave_hdEEG.eventprocessor.ParalEvents`,
:class:`~turtlewave_hdEEG.swprocessor.ParalSWA` and
:class:`~turtlewave_hdEEG.kcomplexprocessor.ParalKC` to write detected events
straight into ``neural_events.db``, bypassing the JSON -> CSV -> import
round-trip.

The design goals of this path are:

* **Deterministic identity.** Each event row's ``uuid`` is a :func:`uuid.uuid5`
  of its detection scope (:func:`event_uuid5`), so re-detecting an unchanged
  channel is a true row-level no-op under ``INSERT OR REPLACE``.
* **One batched morphology re-measurement per channel.** The five spectral/RMS
  columns and the three re-measured amplitude columns are computed with a
  single :func:`wonambi.trans.analyze.event_params` call over in-memory event
  windows (:func:`compute_batched_params`), rather than one raw-file re-read per
  event.
* **Provenance.** Every invocation writes one row into ``detection_runs``
  (:func:`record_run`) capturing method, citation, the full parameter dict,
  reference/polarity, artifact rejection settings and library versions + git
  SHA. ``events.run_id`` links each row back to its run.
* **Resumability.** ``processing_status`` is keyed by the full detection scope
  so a resume skips only channels already completed for the *same* method,
  band and stage set (:func:`resume_skip_channels`).

All schema changes are additive and idempotent (:func:`ensure_direct_write_schema`);
existing databases upgrade in place without touching existing rows.

Notes
-----
This module never writes on import and never mutates a database unless a
processor is explicitly invoked with ``write_db=True``. The legacy
JSON + CSV pipeline is unaffected.
"""

import os
import csv
import uuid
import logging
import sqlite3
import subprocess
import datetime
from collections import namedtuple
from weakref import WeakKeyDictionary

import numpy as np

from wonambi.trans import select
from wonambi.trans.analyze import event_params


# Fixed namespace so uuid5 identities are stable across processes and releases.
# Do NOT change this value: doing so re-keys every event written by this path.
NAMESPACE = uuid.UUID('b9d0f1a2-3c4d-5e6f-8a9b-0c1d2e3f4a5b')


# Detector-own morphology columns added to ``events``. These hold the values the
# detector itself decided on (trough/peak/peak-to-peak), as distinct from the
# re-measured ``min_amp``/``max_amp``/``peak2peak_amp`` columns which are
# computed post-hoc by ``event_params``.
_DET_MORPH_COLUMNS = (
    ('det_trough', 'REAL'),       # detector's own negative-peak amplitude (uV)
    ('det_peak', 'REAL'),         # detector's own positive-peak amplitude (uV)
    # uV from 4.3 on; Wonambi's sample COUNT before that. db_meta.det_ptp_units
    # (PTP_UNITS_KEY) records which, because the two ranges overlap.
    ('det_ptp', 'REAL'),          # detector's own peak-to-peak amplitude (uV)
    ('det_trough_time', 'REAL'),  # time of detector trough (s from rec start)
    ('det_peak_time', 'REAL'),    # time of detector peak (s from rec start)
)

# ``run_id`` foreign-key-ish link from an event to its detection_runs row.
_RUN_ID_COLUMN = ('run_id', 'TEXT')

# Per-event scored epoch stage, additive and nullable. ``events.stage`` holds
# the RUN's canonical stage token ('NREM2NREM3'); this holds the stage of the
# single scored epoch the event actually fell in ('NREM3'). It is in no key and
# no index, so adding it re-keys nothing -- its only job is to keep an
# N2-vs-N3 split recoverable with a plain GROUP BY, without depending on the
# annotation XML still existing and being unedited.
_EPOCH_STAGE_COLUMN = ('epoch_stage', 'TEXT')

# Column order used by the direct-write INSERT. Kept in one place so the SQL and
# the value tuple never drift.
EVENT_INSERT_COLUMNS = (
    'uuid', 'event_type', 'channel',
    'start_time', 'end_time', 'duration', 'start_time_hms',
    'stage', 'cycle', 'method',
    'freq_band', 'freq_lower', 'freq_upper',
    'min_amp', 'max_amp', 'peak2peak_amp',
    'rms', 'power', 'peak_power_freq', 'energy', 'peak_energy_freq',
    'det_trough', 'det_peak', 'det_ptp', 'det_trough_time', 'det_peak_time',
    'processing_timestamp', 'n_fft_sec', 'run_id',
)


def _fmt_num(value):
    """Format a numeric scope component stably for the uuid5 key.

    Parameters
    ----------
    value : float or int or None
        Value to format.

    Returns
    -------
    str
        Fixed 6-decimal representation, or ``'None'`` for ``None`` so that a
        missing band bound still hashes deterministically.
    """
    if value is None:
        return 'None'
    return format(float(value), '.6f')


def event_uuid5(event_type, channel, start_time, method, freq_lo, freq_hi, stage):
    """Deterministic per-event UUID from its detection scope.

    Two runs that detect the *same* event (same type, channel, start time,
    method, band and stage) produce the *same* UUID, so ``INSERT OR REPLACE``
    is idempotent and a re-run is a net-zero row change.

    Parameters
    ----------
    event_type : str
        Event type (e.g. ``'spindle'``, ``'slow_wave'``, ``'k_complex'``).
    channel : str
        Channel label.
    start_time : float
        Event start time in seconds from recording start.
    method : str
        Detection method name.
    freq_lo, freq_hi : float or None
        Lower/upper detection band bounds in Hz.
    stage : str or None
        Stage scope label of the event, as stored in ``events.stage``. From
        4.3 this is the run's canonical joint token
        (:func:`join_stage_token`, e.g. ``'NREM2NREM3'``); earlier releases
        passed the event's own epoch stage.

    Returns
    -------
    str
        String form of the uuid5.

    Warnings
    --------
    Because the stage is part of the key, re-detecting a scope whose rows were
    written under the *other* stage convention yields a different uuid AND a
    different stage, so ``INSERT OR REPLACE`` appends instead of replacing.
    :func:`assert_stage_format_compatible` is the guard that catches this
    before the first write; do not bypass it.
    """
    key = "|".join([
        str(event_type), str(channel), _fmt_num(start_time), str(method),
        _fmt_num(freq_lo), _fmt_num(freq_hi), str(stage),
    ])
    return str(uuid.uuid5(NAMESPACE, key))


# Literature citations per detection method, for the detection_runs provenance
# row. Keyed by the method token as passed to the detector.
_CITATIONS = {
    # Spindles
    'Ferrarelli2007': 'Ferrarelli et al. 2007, Am J Psychiatry 164(3):483-492',
    'Moelle2011': 'Moelle et al. 2011, Sleep 34(10):1411-1421',
    'Nir2011': 'Nir et al. 2011, Neuron 70(1):153-169',
    'Wamsley2012': 'Wamsley et al. 2012, Biol Psychiatry 71(2):154-161',
    'Martin2013': 'Martin et al. 2013, Neurobiol Aging 34(2):468-476',
    'Ray2015': 'Ray et al. 2015, Front Hum Neurosci 9:16',
    'Lacourse2018': 'Lacourse et al. 2018, J Neurosci Methods 316:3-11 (A7)',
    'FASST': 'FASST toolbox',
    'FASST2': 'FASST toolbox',
    'concordia': 'Concordia CIRUS method',
    'UCSD': 'UCSD method',
    # Slow waves / slow oscillations
    'Massimini2004': 'Massimini et al. 2004, J Neurosci 24(31):6862-6870',
    'AASM/Massimini2004': 'AASM 2007 / Massimini et al. 2004',
    'Ngo2015': 'Ngo et al. 2015, J Neurosci 35(17):6630-6638',
    'Staresina2015': 'Staresina et al. 2015, Nat Neurosci 18(11):1679-1686',
}


def method_citation(method):
    """Return a literature citation for a detection method token.

    Parameters
    ----------
    method : str
        Method name (possibly a ``'_'``-joined multi-method string).

    Returns
    -------
    str
        Semicolon-joined citations for the constituent methods, or a generic
        placeholder when unknown.
    """
    parts = str(method).split('_')
    cites = []
    # Recombine escaped slash-methods (e.g. 'AASM_Massimini2004') when possible.
    for token in [str(method)] + parts:
        if token in _CITATIONS and _CITATIONS[token] not in cites:
            cites.append(_CITATIONS[token])
    if not cites:
        return f"Unknown method '{method}' (no citation on record)"
    return '; '.join(cites)


def git_sha(cwd=None):
    """Return the current git commit SHA, or ``None`` outside a checkout.

    Parameters
    ----------
    cwd : str or None
        Directory to resolve the SHA in. Defaults to this module's directory.

    Returns
    -------
    str or None
        40-character SHA, or ``None`` when git is unavailable or the path is
        not a git working tree.
    """
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd,
            stderr=subprocess.DEVNULL)
        return out.decode('utf-8', 'replace').strip() or None
    except Exception:
        return None


def provenance():
    """Collect library versions for a ``detection_runs`` row.

    Returns
    -------
    dict
        Keys ``turtlewave_version``, ``wonambi_version``, ``numpy_version``.
        The turtlewave version is read from the package ``__version__`` (never a
        hard-coded literal).
    """
    versions = {'turtlewave_version': None, 'wonambi_version': None,
                'numpy_version': None}
    try:
        import turtlewave_hdEEG
        versions['turtlewave_version'] = getattr(
            turtlewave_hdEEG, '__version__', None)
    except Exception:
        pass
    try:
        import wonambi
        versions['wonambi_version'] = getattr(wonambi, '__version__', None)
    except Exception:
        pass
    try:
        import numpy
        versions['numpy_version'] = numpy.__version__
    except Exception:
        pass
    return versions


# Environment variable overriding the SQLite journal mode for every database
# this process opens. Exists because WAL cannot work on a network filesystem.
_JOURNAL_ENV = 'TURTLEWAVE_SQLITE_JOURNAL'

# Journal modes SQLite accepts. A value outside this set is a user typo and is
# rejected loudly rather than silently leaving the database in the wrong mode.
# Public so CLIs (turtlewave_set_journal_mode) can validate up front against the
# same list instead of keeping their own copy.
VALID_JOURNAL_MODES = ('DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF')

# Journal mode given to a database this package CREATES when neither the caller
# nor the environment asked for one. DELETE, not WAL: WAL needs a memory-mapped
# -shm sidecar that SMB/NFS/mapped drives and cloud-sync clients (Dropbox,
# OneDrive) cannot provide, and a database born in WAL on such a share is broken
# from its first write ('disk I/O error', or a sync client corrupting the
# sidecars into 'database disk image is malformed'). DELETE works everywhere at
# the cost of readers and the writer blocking each other; that is the right
# trade for a pipeline that runs one writer per subject. Set
# TURTLEWAVE_SQLITE_JOURNAL=WAL to get WAL back on fast local disk.
DEFAULT_NEW_DB_JOURNAL_MODE = 'DELETE'


def _is_blank(value):
    """Return True for ``None`` or a string that is empty/whitespace-only.

    Parameters
    ----------
    value : str or None
        Candidate journal-mode value from an argument or the environment.

    Returns
    -------
    bool
        ``True`` when the value expresses no preference at all.
    """
    return value is None or not str(value).strip()


def _resolve_journal_mode(requested=None):
    """Resolve an explicitly *requested* journal mode, or ``None`` if unstated.

    Precedence is the explicit argument, then the ``TURTLEWAVE_SQLITE_JOURNAL``
    environment variable. This function itself applies no default: ``None``
    means "the caller expressed no preference", which lets
    :func:`open_write_connection` preserve whatever mode an existing database is
    already in instead of imposing one on it, and fall back to
    :data:`DEFAULT_NEW_DB_JOURNAL_MODE` only for a database it creates.

    An empty or whitespace-only value counts as *unset*, at either level. This
    matters because blanking a variable rather than unsetting it is ordinary
    POSIX practice -- a PBS/bash job template that does
    ``export TURTLEWAVE_SQLITE_JOURNAL="$SOMETHING_UNSET"`` exports an empty
    string, and treating that as a mode would make every database open in the
    job fail with a ``ValueError`` about an unrecognised mode ``''``.

    Parameters
    ----------
    requested : str or None, optional
        Explicit mode. When ``None`` (or blank) the environment variable is
        consulted.

    Returns
    -------
    str or None
        Upper-case journal mode (one of :data:`VALID_JOURNAL_MODES`), or
        ``None`` when neither the argument nor the environment variable named
        one (including when either is set but blank).

    Raises
    ------
    ValueError
        If a non-blank value *was* given but is not a recognised SQLite journal
        mode. This is deliberately fatal: a typo that fell through to a default
        would leave the user in WAL believing they had left it.
    """
    mode = requested
    if _is_blank(mode):
        mode = os.environ.get(_JOURNAL_ENV)
    if _is_blank(mode):
        # Nothing (or nothing but whitespace) requested at either level.
        return None
    mode = str(mode).strip().upper()
    if mode not in VALID_JOURNAL_MODES:
        raise ValueError(
            f"Unrecognised SQLite journal mode {mode!r}. Valid modes are "
            f"{', '.join(VALID_JOURNAL_MODES)}. Check the value of the "
            f"{_JOURNAL_ENV} environment variable (or the 'journal' argument).")
    return mode


def _explain_io_error(exc, db_path):
    """Attach the network-drive diagnosis to a SQLite ``disk I/O error``.

    ``SQLITE_IOERR`` reaches the user as a bare
    ``sqlite3.OperationalError: disk I/O error`` that names neither the file nor
    the cause, and the overwhelmingly common cause in this pipeline is a
    WAL-mode database on a network filesystem. This rewrites the message to say
    so and to point at the fix.

    Deliberately fail-open: anything that is not the I/O error is returned
    unchanged, so a lock, a missing table or a corrupt file is never
    mis-diagnosed as a network problem.

    Parameters
    ----------
    exc : sqlite3.OperationalError
        The original exception.
    db_path : str
        Database the operation was against, named in the new message.

    Returns
    -------
    Exception
        Either ``exc`` unchanged, or a new ``sqlite3.OperationalError`` (same
        class, so existing ``except`` clauses still match) chained to ``exc``.
    """
    if 'disk i/o error' not in str(exc).lower():
        return exc
    explained = sqlite3.OperationalError(
        f"disk I/O error on {db_path}. The usual cause is a database on a "
        f"network/mapped drive or a synced folder (Dropbox, OneDrive) while in "
        f"WAL journal mode: WAL needs a shared-memory (-shm) file that such "
        f"filesystems cannot provide. Fix: close every GUI, then convert the "
        f"database once with "
        f"turtlewave_hdEEG.set_journal_mode(r'{db_path}') -- or run "
        f"turtlewave_set_journal_mode \"{db_path}\" from the command line "
        f"(it also takes a directory or --glob to convert a whole tree). If "
        f"that write also "
        f"fails, copy the database WITH its -wal/-shm sidecars to local disk, "
        f"convert it there and copy it back. See "
        f"docs/how-to/run-with-database-on-a-network-drive.md.")
    # Chained by hand: 'raise X from Y' is only valid on a raise statement, and
    # this helper returns the exception for the caller to raise.
    explained.__cause__ = exc
    return explained


def open_write_connection(db_path, journal=None, logger=None):
    """Open a connection that preserves an existing database's journal mode.

    A database this call *creates* is set to
    :data:`DEFAULT_NEW_DB_JOURNAL_MODE` (``'DELETE'``, the only mode that works
    on a network or cloud-synced drive); an existing database keeps whatever
    mode it is already in, unless ``journal`` or
    ``TURTLEWAVE_SQLITE_JOURNAL`` explicitly names one. Every connection also
    gets a 60 s busy timeout, which is what lets a per-subject writer coexist
    with concurrent readers (e.g. the review GUI) without ``database is
    locked`` errors; it does not make concurrent *writers* safe -- the
    pipeline runs one writer per subject.

    See ``docs/explanation/database-concurrency-and-journalling.md`` for why a
    new database is created in DELETE rather than WAL, why journal mode is a
    persistent on-disk property, and why that makes an unconditional default
    dangerous; see ``docs/how-to/run-with-database-on-a-network-drive.md`` for
    the task-oriented fix.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    journal : str or None, optional
        Journal mode to impose. ``None`` (the default) falls back to the
        ``TURTLEWAVE_SQLITE_JOURNAL`` environment variable and, failing that, to
        preserving an existing database's mode /
        :data:`DEFAULT_NEW_DB_JOURNAL_MODE` for a new one.
    logger : logging.Logger or None, optional
        Logger for the mode-in-force INFO line and the not-applied warning.
        Defaults to this module's logger.

    Returns
    -------
    sqlite3.Connection
        Open connection with a 60 s busy timeout and, when SQLite accepted
        it, the requested journal mode.

    Raises
    ------
    ValueError
        If ``journal`` or ``TURTLEWAVE_SQLITE_JOURNAL`` names a mode SQLite
        does not recognise.

    Notes
    -----
    The contract a caller needs:

    * An existing database's mode is read and left alone unless ``journal``
      or ``TURTLEWAVE_SQLITE_JOURNAL`` names one explicitly -- an explicit
      request always overrides, in either direction (including converting a
      ``DELETE``-mode database back to ``WAL``). To convert a database once
      and have it stick, use :func:`set_journal_mode` instead of relying on
      an env var on every run.
    * A database this call creates gets :data:`DEFAULT_NEW_DB_JOURNAL_MODE`
      (``'DELETE'``) when nothing else is requested, so a database created
      straight onto a share or a synced folder is usable from its first write.
      ``TURTLEWAVE_SQLITE_JOURNAL=WAL`` opts back into WAL on local disk.
    * Existence is checked with :func:`os.path.exists` *before* connecting
      (connecting would otherwise create the file). A zero-byte placeholder
      file therefore counts as existing and is left in SQLite's reported
      ``delete`` mode.
    * The 60 s ``timeout`` passed to :func:`sqlite3.connect` is the busy
      timeout, already in force before any pragma runs -- leaving WAL takes
      an exclusive lock and would otherwise fail instantly with
      ``database is locked``.
    * When a mode *is* imposed and SQLite does not honour it, this **warns,
      not raises** -- so a detection run survives another process
      legitimately holding the database. :func:`set_journal_mode` is the
      version of this check that raises, for callers whose only job is the
      conversion.
    * A ``disk I/O error`` raised while opening is re-raised with the
      network-filesystem diagnosis and the fix attached; see
      :func:`_explain_io_error`. ``WAL`` needs a memory-mapped shared-memory
      file that SMB/NFS/mapped-drive/cloud-synced filesystems generally
      cannot provide, which is what SQLite reports as this error.
    * ``OFF`` disables the rollback journal entirely -- a crash or power
      loss mid-transaction leaves the database corrupt, with no atomicity.
      Do not use it on data you cannot regenerate.
    """
    mode = _resolve_journal_mode(journal)
    log = logger if logger is not None else logging.getLogger(__name__)
    # Tested before connect(): connecting creates the file.
    existed = os.path.exists(db_path)
    conn = None
    try:
        # timeout= IS the busy timeout: sqlite3.connect passes it to
        # sqlite3_busy_timeout, so it is already in force before any pragma
        # below. That ordering matters because leaving WAL takes an exclusive
        # lock and would otherwise fail instantly with 'database is locked'.
        conn = sqlite3.connect(db_path, timeout=60.0)

        if mode is None:
            if existed:
                # No mode requested and the database is not ours to re-mode.
                # Read (never set) the current mode and report it, so a run
                # against a DELETE-mode database is visible in the log rather
                # than silent.
                current = str(conn.execute(
                    'PRAGMA journal_mode').fetchone()[0]).lower()
                log.info(
                    f"SQLite journal_mode={current} in force for {db_path} "
                    f"(existing database; mode preserved, not overridden). "
                    f"Set {_JOURNAL_ENV} or pass journal= to impose one.")
                return conn
            # This call is creating the database, so the choice is ours, and it
            # is DELETE: a database born in WAL on a share or a synced folder is
            # unusable (see DEFAULT_NEW_DB_JOURNAL_MODE). Logged, because it is
            # the one place this package decides a persistent on-disk property.
            mode = DEFAULT_NEW_DB_JOURNAL_MODE
            log.info(
                f"Creating {db_path} with SQLite journal_mode={mode.lower()} "
                f"(default for a new database; network- and sync-safe). Set "
                f"{_JOURNAL_ENV}=WAL for WAL on local disk.")

        actual = conn.execute(f'PRAGMA journal_mode={mode}').fetchone()[0]
        if str(actual).upper() != mode:
            log.warning(
                f"Requested SQLite journal_mode={mode} for {db_path} but the "
                f"database is in {str(actual).upper()}. Another connection is "
                f"probably holding it. Set {_JOURNAL_ENV} and/or run "
                f"set_journal_mode() with every other process closed.")
        return conn
    except sqlite3.OperationalError as e:
        if conn is not None:
            conn.close()
        raise _explain_io_error(e, db_path)


def set_journal_mode(db_path, mode='DELETE', logger=None):
    """Convert an existing database to a different journal mode, permanently.

    The repair for a ``neural_events.db`` stuck in WAL on a network drive: WAL
    is a persistent on-disk property, so a database created by an earlier run
    stays in WAL for every later connection until it is explicitly converted.
    The conversion sticks: :func:`open_write_connection` preserves an existing
    database's mode, so a later detection run will not silently undo this.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    mode : str, optional
        Target journal mode (default ``'DELETE'``, the network-safe choice).
        Passing ``None`` falls back to ``TURTLEWAVE_SQLITE_JOURNAL`` and raises
        if that is unset too -- unlike :func:`open_write_connection` there is
        nothing sensible to preserve here.
    logger : logging.Logger or None, optional
        Logger for the blocked-checkpoint warning. Defaults to this module's
        logger.

    Returns
    -------
    str
        The journal mode SQLite reports after the change, lower-case as SQLite
        spells it (e.g. ``'delete'``).

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised SQLite journal mode, or resolves to
        nothing at all.
    RuntimeError
        If the mode did not change. Leaving WAL requires an exclusive lock, so
        another open connection blocks the conversion. SQLite signals this two
        different ways -- ``SQLITE_BUSY`` ("database is locked"), or returning
        the *current* mode, i.e. output shaped like success for what was a
        no-op. Both become this one error, because converting is this
        function's only job. Close every review GUI and other process first.
        A ``sqlite3.OperationalError`` that is *not* a lock (notably
        ``disk I/O error``) keeps its class and propagates, so the
        network-filesystem failure is never mis-reported as a lock; only its
        message is rewritten, with the diagnosis and fix, by
        :func:`_explain_io_error`. That applies to the pre-conversion
        checkpoint as much as to the conversion itself -- on a failing share
        the checkpoint is the first write and so usually fails first.

    Notes
    -----
    Checkpointing is not optional here. When the database is in WAL, committed
    data can live in the ``-wal`` sidecar rather than the ``.db`` file, so this
    runs ``PRAGMA wal_checkpoint(TRUNCATE)`` first and checks the returned
    ``(busy, log, checkpointed)`` row. A non-zero ``busy`` means the checkpoint
    was blocked, the ``-wal`` file still holds committed transactions, and a
    later copy of ``neural_events.db`` alone would silently discard them; that
    case is warned about explicitly.

    On a share that is *already* failing, this conversion is itself a write and
    may fail with the same ``disk I/O error``. The reliable recipe is then:
    copy the database to local disk **together with its ``-wal``/``-shm``
    sidecars** (or checkpoint it first), convert it locally, and copy it back.
    Never copy a WAL database without its sidecars.
    """
    mode = _resolve_journal_mode(mode)
    if mode is None:
        # Only reachable via an explicit mode=None with no environment
        # variable. Unlike open_write_connection there is no sane "preserve"
        # fallback here: converting is the entire job.
        raise ValueError(
            f"set_journal_mode needs a target mode: pass mode= (one of "
            f"{', '.join(VALID_JOURNAL_MODES)}) or set {_JOURNAL_ENV}.")
    log = logger if logger is not None else logging.getLogger(__name__)
    # timeout= is the busy timeout; no separate PRAGMA busy_timeout needed.
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        # The mode probe and the checkpoint both run BEFORE the journal_mode
        # pragma below, and on a failing share the checkpoint is the first
        # *write* attempted -- so it, not the conversion, is where 'disk I/O
        # error' usually surfaces. Translate it here too, or the user gets the
        # bare error with none of the network-drive guidance. _explain_io_error
        # fails open, so a lock here still propagates unchanged as before.
        try:
            before = str(conn.execute(
                'PRAGMA journal_mode').fetchone()[0]).lower()
            if before == 'wal':
                row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
                if row is not None and row[0]:
                    log.warning(
                        f"wal_checkpoint(TRUNCATE) was blocked on {db_path} "
                        f"(busy={row[0]}): committed data may still sit in the "
                        f"-wal sidecar. Copying the .db file alone would lose "
                        f"it.")
        except sqlite3.OperationalError as e:
            raise _explain_io_error(e, db_path)
        try:
            after = str(conn.execute(
                f'PRAGMA journal_mode={mode}').fetchone()[0]).lower()
        except sqlite3.OperationalError as e:
            # A held lock is the expected "another process has it open" case and
            # becomes a RuntimeError with the same actionable message as the
            # silent-no-op case. Anything else -- notably 'disk I/O error', the
            # network-share failure this whole feature exists for -- must
            # propagate unchanged rather than be mis-diagnosed as a lock.
            msg = str(e).lower()
            if 'lock' not in msg and 'busy' not in msg:
                # Still an OperationalError, but with the network-drive
                # diagnosis attached when it is the disk I/O error.
                raise _explain_io_error(e, db_path)
            raise RuntimeError(
                f"Failed to set journal_mode={mode} on {db_path}: {e}. "
                f"Another connection is holding the database. Close every "
                f"GUI/process using it and retry.") from e
        if after != mode.lower():
            raise RuntimeError(
                f"Failed to set journal_mode={mode} on {db_path}: still in "
                f"{after!r}. Another connection is holding the database "
                f"(SQLite reports the current mode instead of erroring). "
                f"Close every GUI/process using it and retry.")
        return after
    finally:
        conn.close()


#: Default basename of the store of record. One database per recording/subject.
DEFAULT_DB_NAME = 'neural_events.db'


def resolve_db_target(db_path=None, output_dir=None, logger=None):
    """Resolve the SQLite file a detection run writes its events to.

    Single source of truth for "which database?", shared by every detector so
    the spindle, slow-wave, K-complex and PAC paths cannot drift apart. The
    resolution order is:

    1. an explicit ``db_path`` -- a file path is used as given; a path that is
       an existing *directory* becomes ``<db_path>/neural_events.db``;
    2. ``output_dir`` -- ``<output_dir>/neural_events.db`` when that file
       already exists (never create a second database beside one that is
       already there), otherwise the sibling
       ``<parent of output_dir>/neural_events.db``, which is the deployment
       shape in use (``.../<subject>/wonambi/neural_events.db`` beside
       ``.../<subject>/wonambi/<results dir>/``);
    3. ``./neural_events.db`` in the current working directory.

    Unlike the code this replaces, an unresolvable target **raises**. The old
    behaviour -- log an error, set ``write_db = False`` and carry on -- turned
    a demanded database write into a silently discarded run, the same class of
    data loss as a ``file_pattern`` that matches nothing.

    Parameters
    ----------
    db_path : str or None, optional
        Explicit database file, or a directory to place ``neural_events.db``
        in. Default ``None``.
    output_dir : str or None, optional
        Results directory of the run, used to locate the database when
        ``db_path`` is not given. Default ``None``.
    logger : logging.Logger or None, optional
        Logger for the one-line resolution message. Default ``None``.

    Returns
    -------
    str
        Absolute path of the database file to write. Its parent directory
        exists on return.

    Raises
    ------
    ValueError
        If ``db_path`` is given but blank, if the resolved path is an existing
        directory, or if the parent directory does not exist and cannot be
        created. Every one of these means "a database write was asked for and
        no database can be written", which must never be downgraded to a
        no-op.

    Examples
    --------
    >>> resolve_db_target(db_path='/data/sub-01/wonambi/neural_events.db')
    '/data/sub-01/wonambi/neural_events.db'
    """
    source = None
    resolved = None

    if db_path is not None:
        if not str(db_path).strip():
            raise ValueError(
                "A database write was requested but db_path is an empty "
                "string. Pass a database file, a directory to create "
                f"{DEFAULT_DB_NAME} in, or None to resolve one automatically.")
        candidate = os.path.abspath(os.path.expanduser(str(db_path).strip()))
        if os.path.isdir(candidate):
            resolved = os.path.join(candidate, DEFAULT_DB_NAME)
            source = 'explicit db_path (directory)'
        else:
            resolved = candidate
            source = 'explicit db_path'
    elif output_dir is not None and str(output_dir).strip():
        out = os.path.abspath(os.path.expanduser(str(output_dir).strip()))
        inside = os.path.join(out, DEFAULT_DB_NAME)
        resolved = os.path.join(os.path.dirname(out) or out, DEFAULT_DB_NAME)
        source = 'sibling of the output directory'
        # The rule is fixed, never "whichever file happens to exist": two
        # invocations of one command must resolve to one database. When a
        # second candidate exists inside the output directory the situation is
        # genuinely ambiguous, so refuse rather than pick.
        if os.path.isfile(inside) and os.path.abspath(inside) != resolved:
            raise ValueError(
                f"Two candidate databases for output_dir {out!r}: "
                f"{inside!r} (inside it) and {resolved!r} (its sibling, which "
                f"is the rule). Refusing to guess which one this run belongs "
                f"to -- pass db_path explicitly.")
    else:
        resolved = os.path.abspath(os.path.join(os.getcwd(), DEFAULT_DB_NAME))
        source = 'current working directory'

    if os.path.isdir(resolved):
        raise ValueError(
            f"A database write was requested but the resolved target "
            f"{resolved!r} is a directory, not a file ({source}). Pass an "
            f"explicit db_path.")

    parent = os.path.dirname(resolved)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"A database write was requested but its directory {parent!r} "
                f"does not exist and could not be created ({e}). Pass an "
                f"explicit db_path to an existing directory, or pass "
                f"write_db=False to run without a database.") from e

    if logger is not None:
        logger.info(f"Database target resolved to {resolved} ({source})")
    return resolved


#: Subject-bearing tables consulted by :func:`assert_single_subject`.
#: ``events`` is deliberately absent -- it has no subject column, which is
#: precisely why the guard is needed.
_SUBJECT_TABLES = ('analysed_time', 'detection_runs', 'pac_coupling',
                   'sleep_cycles', 'stage_durations')


def subjects_in_database(conn):
    """Return every subject id already present in a database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.

    Returns
    -------
    set of str
        Subject ids found across ``analysed_time``, ``detection_runs``,
        ``pac_coupling``, ``sleep_cycles`` and ``stage_durations``, each put
        into the canonical ``sub-`` form by
        :func:`turtlewave_hdEEG.utils.normalize_subject`. NULL and empty ids
        are ignored (rows written before the column existed).

    Notes
    -----
    Normalising here is what makes the comparison in
    :func:`assert_single_subject` an identity test rather than a string test.
    ``ParalCycles`` historically stored whatever it was handed, and the cycle
    how-to tells users to pass the bare folder name, so one recording can
    genuinely carry ``'10sd'`` in ``stage_durations`` and ``'sub-10sd'`` in
    ``analysed_time``. Those are one subject, and comparing the raw strings
    would refuse the recording's own next run.
    """
    from .utils import normalize_subject
    found = set()
    for table in _SUBJECT_TABLES:
        cols = _table_columns(conn, table)
        if not cols or 'subject' not in cols:
            continue
        for (subj,) in conn.execute(
                f"SELECT DISTINCT subject FROM {table} "
                f"WHERE subject IS NOT NULL AND subject != ''"):
            canonical = normalize_subject(str(subj))
            if canonical:
                found.add(canonical)
    return found


def assert_single_subject(conn, subject, db_path=None, logger=None):
    """Refuse to write a second recording's events into another's database.

    One database per recording is the deployment shape, and the schema
    depends on it: ``events`` has **no subject column**, and
    :func:`event_uuid5` keys a row on
    ``(event_type, channel, start_time, method, band, stage)`` only. Two
    subjects sharing a database therefore collide -- identical channel labels
    at identical times produce identical uuids, so the second subject's
    ``INSERT OR REPLACE`` overwrites the first's rows, and a scoped
    re-detection ``DELETE`` (which is also subject-blind) removes the other
    subject's channel outright. ``verify_channel_coverage`` then reports full
    coverage over the wrong data. None of that raises, and none of it is
    recoverable.

    This is the one cheap check that catches it: if the database already
    carries a *different* subject, stop before the first write.

    **Databases written before 4.2 carry no subject at all** -- there was no
    ``analysed_time`` table and ``detection_runs`` had no ``subject`` column.
    Such a database is *unattributed*: it cannot be proved to belong to this
    recording. It is claimed rather than refused -- the subject is stamped on
    and a WARNING names the database, its event count and the subject being
    claimed. Refusing it would cost a manual step on every irreplaceable
    database that already exists while buying very little, because the
    overwrite risk lives in the *second* recording written to a database, and
    once stamped that second recording is refused.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    subject : str
        Subject this run belongs to, already normalised
        (:func:`turtlewave_hdEEG.utils.normalize_subject`).
    db_path : str or None, optional
        Path used in the message. Default ``None``.
    logger : logging.Logger or None, optional
        Logger for the claim warning and the confirmation line. Default
        ``None``.

    Returns
    -------
    set of str
        The subjects already present (empty for a fresh or unattributed
        database).

    Raises
    ------
    ValueError
        If the database already holds rows for a subject other than
        ``subject``. Both sides of that comparison are normalised, so one
        recording spelled two ways is one subject, not two.
    """
    from .utils import normalize_subject
    subject = normalize_subject(str(subject))
    existing = subjects_in_database(conn)
    others = {s for s in existing if s != subject}
    if others:
        raise ValueError(
            f"{db_path or 'This database'} already holds data for subject(s) "
            f"{sorted(others)}, and this run is subject '{subject}'. One "
            f"database per recording is required: the events table has no "
            f"subject column and event ids are keyed on "
            f"(event_type, channel, start_time, method, band, stage), so a "
            f"second subject's identical channel labels would overwrite the "
            f"first's rows and a scoped re-run would delete them. Point "
            f"db_path at this recording's own neural_events.db.")

    if not existing:
        # No subject named anywhere. If the database already holds events it
        # was written before 4.2 (or by the CSV importers): claim it for this
        # recording and say so loudly, so that from here on a *different*
        # recording is refused.
        n_events = 0
        if _table_columns(conn, 'events'):
            n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if n_events:
            stamped = 0
            if _table_columns(conn, 'detection_runs'):
                cur = conn.execute(
                    "UPDATE detection_runs SET subject = ? "
                    "WHERE subject IS NULL", (subject,))
                stamped = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                conn.commit()
            if logger is not None:
                if stamped:
                    logger.warning(
                        "%s holds %d event(s) but named no subject (written "
                        "before 4.2). Claiming it for '%s' and stamping %d "
                        "existing detection_runs row(s). From now on a "
                        "different subject is refused. If those events belong "
                        "to another recording, stop and point db_path at this "
                        "recording's own neural_events.db.",
                        db_path or 'This database', n_events, subject, stamped)
                else:
                    # Be honest: nothing was stamped. This run's own
                    # detection_runs row (written moments later by record_run)
                    # is what will attribute the database.
                    logger.warning(
                        "%s holds %d event(s) but named no subject (written "
                        "before 4.2). Claiming it for '%s'. No existing "
                        "detection_runs row could be stamped -- there are "
                        "none with a NULL subject -- so the attribution comes "
                        "from this run's own provenance row. From now on a "
                        "different subject is refused. If those events belong "
                        "to another recording, stop and point db_path at this "
                        "recording's own neural_events.db.",
                        db_path or 'This database', n_events, subject)

    if logger is not None and existing:
        logger.debug(f"Database subject check passed: only '{subject}' present")
    return existing


def recording_root_from_db(db_path):
    """Recording root directory implied by a database path.

    ``neural_events.db`` lives in the recording's ``wonambi`` working
    directory, so the directory that *names* the recording is one level up in
    that layout. Used only as the ``root_dir`` hint for
    :func:`turtlewave_hdEEG.utils.derive_subject`; the subject id it produces
    keys ``analysed_time``, never ``events``.

    Parameters
    ----------
    db_path : str or None
        Path to the database file.

    Returns
    -------
    str or None
        The directory holding the database, with a trailing ``wonambi``
        component stripped, or ``None`` when ``db_path`` is ``None``.

    Examples
    --------
    >>> recording_root_from_db('/data/10sd/wonambi/neural_events.db')
    '/data/10sd'
    >>> recording_root_from_db('/data/10sd/neural_events.db')
    '/data/10sd'
    """
    if not db_path:
        return None
    parent = os.path.dirname(os.path.abspath(str(db_path)))
    if os.path.basename(parent).lower() == 'wonambi':
        return os.path.dirname(parent)
    return parent


def _table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def verify_channel_coverage(db_path, event_type, method, requested_channels,
                            freq_lower, freq_upper, stage_key, logger=None):
    """Check that every requested channel is accounted for in the database.

    A detection run is complete only if each requested channel either has
    events stored for the run's scope, or carries a ``processing_status`` row
    for that same scope recording ``success = 1``. Counting events alone would
    flag a genuinely event-free channel as a failure; counting status alone
    would miss a channel whose events never reached ``events``. The union of
    the two is the honest check, and it is what lets a batch driver exit
    non-zero instead of printing "All done" over an empty database.

    Two properties are load-bearing, and getting either wrong reproduces the
    silent-success bug this function exists to catch:

    * **``success = 1``, not ``processed = 1``.** ``upsert_processing_status``
      hardcodes ``processed = 1`` and encodes the outcome in ``success``, so
      :func:`record_channel_failure` writes ``processed = 1, success = 0``.
      Filtering on ``processed`` would count a channel that crashed
      mid-detection as covered.
    * **The full scope, not just the event type.** Without the
      method/band/stage columns, a status row left by an earlier run of a
      different method or band masks a channel that never ran in the current
      one.

    The ``events`` half is deliberately NOT scoped by stage. From 4.3 on
    ``events.stage`` holds the run's joined stage token and filtering on it
    would work -- but a database may still hold rows written per-epoch by an
    earlier release, and filtering those out would report a fully-covered
    channel as missing. Two things limit the resulting blind spot. A channel with an in-scope ``success = 0`` row is excluded
    whatever events exist for it, which closes the case where a re-run over a
    narrower stage set crashes on a channel an earlier run had populated. And
    channels vouched for by events alone are returned in ``events_only`` so a
    caller can report how much of its "complete" rests on the weaker
    evidence. A channel killed outright (no status row written at all) whose
    events predate the run is the residual case ``events_only`` exists to
    surface.

    The unscoped fallback fires only on ``sqlite3.OperationalError``, i.e. a
    ``processing_status`` table not yet widened to the per-scope primary key
    by :func:`ensure_direct_write_schema` — the same condition and the same
    handling as :func:`resume_skip_channels`. It still requires
    ``success = 1``.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    event_type : str
        Event type of the run (``'spindle'``, ``'slow_wave'``, ...).
    method : str
        Detection method as stored in ``events.method`` (the unescaped string,
        e.g. ``'AASM/Massimini2004'``).
    requested_channels : list of str
        Channels the run was asked to process.
    freq_lower, freq_upper : float
        Band bounds of the run. Required: they are part of the scope.
    stage_key : str
        Joined stage set of the run (e.g. ``'NREM2NREM3'``). Required: it is
        part of the scope.
    logger : logging.Logger or None, optional
        Logger used to report when the unscoped fallback was taken.

    Returns
    -------
    dict
        ``{'requested': int, 'with_events': int, 'covered': int,
        'missing': list of str, 'complete': bool, 'scoped_status': bool,
        'failed': list of str, 'events_only': list of str}``.
        ``scoped_status`` is False when the unscoped fallback was used;
        ``failed`` lists channels with an in-scope failure; ``events_only``
        lists channels credited by event rows alone, whose evidence cannot be
        stage-scoped. A caller reporting success should report
        ``events_only`` too.
    """
    requested = [str(c) for c in (requested_channels or [])]
    result = {'requested': len(requested), 'with_events': 0, 'covered': 0,
              'missing': list(requested), 'complete': False,
              'scoped_status': True}

    if not os.path.exists(db_path):
        return result

    # Read-only, but with the writers' 60 s busy timeout: under DELETE journal
    # mode a writer blocks readers, so Python's 5 s default would fail this
    # coverage check whenever a detection run is mid-write.
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        # NOTE: the events half is not scoped by stage. events.stage holds the
        # run's joined token ('NREM2NREM3') from 4.3 on, but rows written by
        # an earlier release hold the per-epoch stage ('NREM2') and filtering
        # on either spelling alone would reject the other's valid rows. Two
        # mitigations below: an in-scope FAILURE always wins over event
        # evidence, and channels credited by events alone are counted and
        # reported.
        with_events = {str(r[0]) for r in conn.execute('''
            SELECT DISTINCT channel FROM events
            WHERE event_type = ? AND method = ?
              AND freq_lower = ? AND freq_upper = ?
            ''', (event_type, method, float(freq_lower),
                  float(freq_upper))).fetchall()}

        try:
            succeeded = {str(r[0]) for r in conn.execute('''
                SELECT channel FROM processing_status
                WHERE event_type = ? AND method = ? AND freq_lower = ?
                  AND freq_upper = ? AND stage = ? AND success = 1
                ''', (event_type, method, float(freq_lower), float(freq_upper),
                      stage_key)).fetchall()}
            # A channel that FAILED in exactly this scope is never covered,
            # whatever events an earlier run over a different stage set left
            # behind. This is what closes the stage-scope leak: run A over
            # NREM2+NREM3 leaves Cz events; run B over NREM3 only crashes on
            # Cz; run B must not be credited by run A's rows.
            failed = {str(r[0]) for r in conn.execute('''
                SELECT channel FROM processing_status
                WHERE event_type = ? AND method = ? AND freq_lower = ?
                  AND freq_upper = ? AND stage = ? AND success = 0
                ''', (event_type, method, float(freq_lower), float(freq_upper),
                      stage_key)).fetchall()}
        except sqlite3.OperationalError:
            # processing_status not yet migrated to the wide schema; the scope
            # columns do not exist. Same fallback as resume_skip_channels.
            result['scoped_status'] = False
            if logger is not None:
                logger.warning(
                    "processing_status has no per-scope columns (unmigrated "
                    "database); falling back to an event_type-only status "
                    "check. A status row from a different method or band "
                    "cannot be distinguished.")
            succeeded = {str(r[0]) for r in conn.execute(
                "SELECT channel FROM processing_status "
                "WHERE event_type = ? AND success = 1",
                (event_type,)).fetchall()}
            failed = {str(r[0]) for r in conn.execute(
                "SELECT channel FROM processing_status "
                "WHERE event_type = ? AND success = 0",
                (event_type,)).fetchall()} - succeeded
    finally:
        conn.close()

    covered = (with_events | succeeded) - failed
    missing = [c for c in requested if c not in covered]
    # Channels vouched for by events alone, with no status row for this exact
    # scope. Their events may predate this run (they cannot be stage-scoped),
    # so this is the weaker half of the evidence and the caller should say so.
    events_only = [c for c in requested
                   if c in covered and c not in succeeded]
    result.update({
        'with_events': len(with_events & set(requested)),
        'covered': len(requested) - len(missing),
        'missing': missing,
        'complete': not missing,
        'failed': sorted(failed & set(requested)),
        'events_only': events_only,
    })
    return result


def guard_run_id(conn, event_type, method, freq_lower=None, freq_upper=None,
                 force=False, logger=None):
    """Refuse a CSV import that would blank direct-written provenance.

    The CSV importers write with ``INSERT OR REPLACE`` keyed on a
    deterministic event UUID, and their column list has no ``run_id``. So
    re-importing a CSV over rows that the direct-to-database path wrote
    replaces those rows with ``run_id = NULL``, severing them from their
    ``detection_runs`` provenance without any error. This guard detects that
    situation before the write loop starts.

    A database with no ``run_id`` column at all (one built entirely by the
    legacy CSV path) has nothing to protect: the function returns 0 rather
    than raising an ``OperationalError`` on the missing column.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the target database.
    event_type : str
        Event type being imported (``'spindle'``, ``'slow_wave'``, ...).
    method : str
        Detection method being imported.
    freq_lower, freq_upper : float or None, optional
        Band bounds narrowing the scope. Omitted from the check when None.
    force : bool, optional
        Proceed (with a warning) instead of raising.
    logger : logging.Logger or None, optional
        Logger for the warning emitted when ``force`` is True.

    Returns
    -------
    int
        Number of direct-written rows found in the scope (0 when the column
        is absent or nothing matches).

    Raises
    ------
    RuntimeError
        If direct-written rows exist in the scope and ``force`` is False.
    """
    if 'run_id' not in _table_columns(conn, 'events'):
        return 0

    sql = ("SELECT COUNT(*) FROM events "
           "WHERE run_id IS NOT NULL AND event_type = ? AND method = ?")
    params = [event_type, method]
    if freq_lower is not None:
        sql += " AND freq_lower = ?"
        params.append(float(freq_lower))
    if freq_upper is not None:
        sql += " AND freq_upper = ?"
        params.append(float(freq_upper))

    n_rows = conn.execute(sql, params).fetchone()[0]
    if not n_rows:
        return 0

    band = ''
    if freq_lower is not None and freq_upper is not None:
        band = f" {fmt_freq_token(freq_lower, freq_upper)}"
    scope = f"event_type={event_type!r}, method={method!r}{band}"

    if force:
        if logger is not None:
            logger.warning(
                f"force=True: importing over {n_rows} direct-written row(s) "
                f"({scope}); their run_id provenance will be cleared.")
        return n_rows

    raise RuntimeError(
        f"Refusing to import: {n_rows} row(s) in this scope ({scope}) were "
        f"written by the direct-to-database path and carry a run_id linking "
        f"them to detection_runs. A CSV import is INSERT OR REPLACE and would "
        f"blank that run_id. Re-run detection with write_db=True to update "
        f"them, or pass force=True to overwrite and lose the provenance link.")


def ensure_pac_schema(conn):
    """Create the ``pac_coupling`` table and its indexes if absent.

    Purely additive: it never touches ``events``, ``processing_status``,
    ``sleep_cycles`` or ``stage_durations``. The primary key is the natural
    key of a PAC result (subject, channel, event type, method, stage, and the
    phase/amplitude frequency bounds), so a re-run with identical parameters
    replaces its own row rather than inserting a duplicate.

    This lives in ``dbwrite`` rather than ``pacprocessor`` so that
    :func:`ensure_direct_write_schema` can create the table on every detection
    run. A reader (review GUI, analysis notebook) then never hits
    ``OperationalError: no such table: pac_coupling`` on a database where PAC
    has not been run yet — it sees an empty table, which is the truth.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. The caller owns the connection lifecycle;
        this function commits but does not close.

    Returns
    -------
    None
    """
    conn.execute('''
    CREATE TABLE IF NOT EXISTS pac_coupling (
        -- Natural key
        subject TEXT NOT NULL,
        channel TEXT NOT NULL,
        event_type TEXT NOT NULL,       -- 'slow_wave', 'spindle', 'sw_spindle'
        method TEXT NOT NULL,           -- detector / pairing method token
        stage TEXT NOT NULL,            -- combined stage string (e.g. 'NREM2NREM3')
        phase_freq_lower REAL NOT NULL,
        phase_freq_upper REAL NOT NULL,
        amp_freq_lower REAL NOT NULL,
        amp_freq_upper REAL NOT NULL,

        -- Coupling metrics (NaN stored as NULL)
        mi_raw REAL,
        mi_norm REAL,
        median_mi_pval REAL,
        preferred_phase_rad REAL,
        preferred_phase_deg REAL,
        mean_vector_length REAL,
        rho REAL,
        rayleigh_z REAL,
        rayleigh_p REAL,

        -- Number of artefact-free segments/events actually coupled.
        -- NOT NULL by design: a row with an unrecoverable event count is
        -- rejected upstream and flagged for re-run, never stored as 0/NULL.
        n_events INTEGER NOT NULL,

        -- Provenance
        idpac TEXT,                     -- str(tuple) of (method, surrogate, correction)
        ref_chan TEXT,
        invert INTEGER,                 -- 0/1 polarity flag actually used
        turtlewave_version TEXT,
        processing_timestamp TEXT,
        source_path TEXT,

        PRIMARY KEY (subject, channel, event_type, method, stage,
                     phase_freq_lower, phase_freq_upper,
                     amp_freq_lower, amp_freq_upper)
    )''')

    conn.execute('CREATE INDEX IF NOT EXISTS idx_pac_subject ON pac_coupling(subject)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_pac_channel ON pac_coupling(channel)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_pac_method_stage ON pac_coupling(method, stage)')
    conn.commit()


def ensure_analysed_time_schema(conn):
    """Create the ``analysed_time`` table if absent.

    ``analysed_time`` holds the **density denominator**: the artefact-free
    in-stage seconds actually fed to the detector, per sleep stage. It is the
    one quantity a density cannot be derived from ``events`` alone, so it is
    stored once at detection time and every density is computed on read from
    it (see :mod:`turtlewave_hdEEG.density`).

    It is keyed on ``(subject, stage, reject_artifacts, reject_arousals)``
    because the rejection settings *define* the denominator: a run that kept
    arousal epochs analysed more seconds than one that dropped them, and the
    two must never be mixed. ``stage_durations`` is deliberately NOT a
    fallback -- that table holds raw hypnogram time with no artefact
    subtraction, and dividing an artefact-free numerator by it re-introduces
    the artefact-scaled under-estimation of density that 4.0 removed.

    Purely additive; it touches no existing table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. Commits, does not close.

    Returns
    -------
    None
    """
    conn.execute('''
    CREATE TABLE IF NOT EXISTS analysed_time (
        subject TEXT NOT NULL,
        stage TEXT NOT NULL,              -- single scored stage, e.g. 'NREM2'
        reject_artifacts INTEGER NOT NULL,
        reject_arousals INTEGER NOT NULL,

        analysed_seconds REAL NOT NULL,   -- artefact-free in-stage seconds
        artefact_seconds_excluded REAL,   -- in-stage seconds removed
        epoch_length REAL,                -- nominal scoring epoch (s)
        source TEXT,                      -- 'detection' | 'backfill' | ...
        annotation_file TEXT,
        turtlewave_version TEXT,
        processing_timestamp TEXT,

        PRIMARY KEY (subject, stage, reject_artifacts, reject_arousals)
    )''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_analysed_time_subject '
                 'ON analysed_time(subject)')
    conn.commit()


def record_analysed_time(conn, subject, stage, analysed_seconds,
                         artefact_seconds_excluded=None, reject_artifacts=True,
                         reject_arousals=True, epoch_length=30,
                         source='detection', annotation_file=None):
    """Insert-or-replace one ``analysed_time`` row (one stage).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    subject : str
        Subject identifier, as resolved by
        :func:`turtlewave_hdEEG.utils.derive_subject`.
    stage : str
        A single scored stage label (e.g. ``'NREM2'``). Never a joined set --
        a combined denominator is the sum of its stage rows, computed on read.
    analysed_seconds : float
        Artefact-free in-stage seconds fed to the detector.
    artefact_seconds_excluded : float or None, optional
        In-stage seconds removed by artefact/arousal rejection. Default
        ``None``.
    reject_artifacts, reject_arousals : bool, optional
        The rejection settings this denominator was computed under. Part of
        the primary key. Default ``True``.
    epoch_length : float, optional
        Nominal scoring epoch length in seconds. Default ``30``.
    source : str, optional
        Provenance tag: ``'detection'`` when written by a detection run.
        Default ``'detection'``.
    annotation_file : str or None, optional
        Scoring file the denominator was computed from. Default ``None``.

    Returns
    -------
    None

    Notes
    -----
    ``subject`` is normalised through
    :func:`turtlewave_hdEEG.utils.normalize_subject` here as well as at
    resolution time, so a caller reaching this function directly cannot key
    one recording under two spellings.
    """
    from .utils import normalize_subject
    conn.execute('''
    INSERT OR REPLACE INTO analysed_time
        (subject, stage, reject_artifacts, reject_arousals, analysed_seconds,
         artefact_seconds_excluded, epoch_length, source, annotation_file,
         turtlewave_version, processing_timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(normalize_subject(subject)), str(stage),
        1 if reject_artifacts else 0, 1 if reject_arousals else 0,
        float(analysed_seconds),
        None if artefact_seconds_excluded is None else float(artefact_seconds_excluded),
        None if epoch_length is None else float(epoch_length),
        source, annotation_file,
        provenance()['turtlewave_version'],
        datetime.datetime.now().isoformat(),
    ))


def store_analysed_time(conn, subject, annotations, dataset, stages,
                        reject_artifacts, reject_arousals, epoch_length=30,
                        source='detection', annotation_file=None, logger=None,
                        strict=False):
    """Compute and store the density denominators for a detection run.

    Wraps :func:`turtlewave_hdEEG.utils.build_density_denominators` -- the same
    artefact-free-time computation the (now deprecated) CSV density exporters
    used -- and writes one ``analysed_time`` row per requested stage, so
    density can be derived from the database alone afterwards.

    Failures are logged and swallowed **by default**: a denominator that
    cannot be computed must not lose a detection run that already succeeded,
    and the consequence stays visible because
    :func:`density.event_density` refuses to invent a denominator and reports
    the missing rows.

    That contract is right for a detector and wrong for a caller whose whole
    job is this write -- a back-fill has nothing else to lose and needs to
    know. Such a caller passes ``strict=True`` and gets the exception. The
    default is unchanged, so the three detectors keep behaving exactly as
    before.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    subject : str
        Subject identifier keying the rows.
    annotations : instance of Annotations
        Scoring handed to the detector.
    dataset : instance of Dataset
        Used only for ``header['s_freq']``.
    stages : list of str or None
        The stages the run detected on. ``None`` or empty stores nothing (an
        all-stage run has no defined per-stage denominator here).
    reject_artifacts, reject_arousals : bool
        The run's rejection settings. Stored as part of the key.
    epoch_length : float, optional
        Nominal scoring epoch length in seconds. Default ``30``.
    source : str, optional
        Provenance tag. Default ``'detection'``.
    annotation_file : str or None, optional
        Scoring file path for provenance. Default ``None``.
    logger : logging.Logger or None, optional
        Logger for the per-stage summary. Default ``None``.
    strict : bool, optional
        Re-raise instead of swallowing; treat an empty ``stages``, and a
        denominator that comes out as zero seconds for every requested stage,
        as errors rather than as a warning and a stored row. For callers whose
        only purpose is this write (the migration back-fill). Default
        ``False``.

    Returns
    -------
    dict
        ``{stage: analysed_seconds}`` for the rows written (empty on failure
        when ``strict`` is False).

    Raises
    ------
    Exception
        Only when ``strict`` is True: whatever the denominator computation
        raised, or ``ValueError`` for an empty ``stages`` or an all-zero
        denominator.
    """
    if not stages:
        msg = ("No stage list for this run, so no density denominator was "
               "stored. Density for this scope will be unavailable until a "
               "stage-scoped run or a back-fill writes analysed_time.")
        if strict:
            raise ValueError(msg)
        if logger is not None:
            logger.warning(msg)
        return {}

    from .utils import build_density_denominators

    written = {}
    ensure_analysed_time_schema(conn)
    # All-or-nothing: a partial denominator is worse than none, because a
    # stage whose row was written before the failure looks complete on read.
    # A plain `return {}` would leave those rows pending in the connection's
    # open transaction, and the next write_channel_events commit would commit
    # them. The savepoint makes the rollback real.
    conn.execute("SAVEPOINT tw_analysed_time")
    try:
        dd = build_density_denominators(
            annotations, dataset,
            reject_artifacts=reject_artifacts, reject_arousals=reject_arousals,
            stage_list=list(stages), stages_present=(), logger=logger)
        for stg in sorted({str(s) for s in stages}):
            clean_sec, artefact_sec = dd.analysed_seconds(stg)
            record_analysed_time(
                conn, subject, stg, clean_sec,
                artefact_seconds_excluded=artefact_sec,
                reject_artifacts=reject_artifacts,
                reject_arousals=reject_arousals,
                epoch_length=epoch_length, source=source,
                annotation_file=annotation_file)
            written[stg] = clean_sec
    except Exception as e:
        try:
            conn.execute("ROLLBACK TO SAVEPOINT tw_analysed_time")
            conn.execute("RELEASE SAVEPOINT tw_analysed_time")
        except Exception:
            pass
        if logger is not None:
            logger.error(
                f"Could not store the density denominator (analysed_time) for "
                f"subject '{subject}': {e}. No partial rows were kept; "
                f"detection results are unaffected, but density will be "
                f"unavailable for this scope until it is back-filled.",
                exc_info=True)
        if strict:
            raise
        return {}

    # A denominator of zero seconds across EVERY requested stage means the
    # scoring produced nothing usable -- an unreadable or epoch-less
    # annotation file yields this rather than an exception, because
    # build_density_denominators is deliberately tolerant. For a detector that
    # is survivable (the row is written, and density reports a zero
    # denominator as NaN with a warning rather than as a density of 0). For a
    # caller whose only job is this write it is a failure with a
    # success-shaped return, so strict must not let it pass.
    if strict and not sum(written.values()):
        conn.execute("ROLLBACK TO SAVEPOINT tw_analysed_time")
        conn.execute("RELEASE SAVEPOINT tw_analysed_time")
        raise ValueError(
            f"The density denominator for subject '{subject}' came out as "
            f"0 seconds for every requested stage ({sorted(written)}). The "
            f"scoring could not be read, contains no scored epochs, or has "
            f"none of these stages -- build_density_denominators reports that "
            f"as zero time rather than raising. No row was kept: a "
            f"zero-second denominator is not a denominator, and storing it "
            f"would make every density in this scope NaN with no explanation "
            f"of why. Check the annotation file.")

    conn.execute("RELEASE SAVEPOINT tw_analysed_time")
    conn.commit()
    if logger is not None:
        summary = ", ".join(f"{k}={v / 60.0:.2f} min"
                            for k, v in sorted(written.items()))
        logger.info(
            f"Stored density denominators for subject '{subject}' "
            f"(artefact-free analysed time): {summary}")
    return written


def subject_has_cycles(conn, subject, methods=('2022', '1979')):
    """Report whether a subject's cycles and stage durations are already stored.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    subject : str
        Subject id, in any spelling (normalised here).
    methods : sequence of str, optional
        Cycle definitions that must ALL be present for the answer to be True.
        Default ``('2022', '1979')``.

    Returns
    -------
    bool
        True when ``sleep_cycles`` holds at least one row for this subject
        under every method in ``methods`` **and** ``stage_durations`` holds
        its row. Anything less is False, so a partially-populated database is
        completed rather than left half-done.

    Notes
    -----
    Matches every stored spelling of the subject id, not just the canonical
    one: ``ParalCycles`` historically stored whatever it was handed, so one
    recording can carry ``'10sd'`` in one table and ``'sub-10sd'`` in another.
    """
    from .utils import normalize_subject
    subject = normalize_subject(str(subject))

    def _matches(table):
        cols = _table_columns(conn, table)
        if not cols or 'subject' not in cols:
            return []
        return [str(r[0]) for r in conn.execute(
            f"SELECT DISTINCT subject FROM {table} "
            f"WHERE subject IS NOT NULL AND subject != ''")
            if normalize_subject(str(r[0])) == subject]

    cyc_spellings = _matches('sleep_cycles')
    if not cyc_spellings:
        return False
    placeholders = ",".join("?" * len(cyc_spellings))
    stored_methods = {str(r[0]) for r in conn.execute(
        f"SELECT DISTINCT method FROM sleep_cycles "
        f"WHERE subject IN ({placeholders})", cyc_spellings)}
    if not set(str(m) for m in methods).issubset(stored_methods):
        return False
    return bool(_matches('stage_durations'))


def ensure_cycles_populated(conn, annotations, subject, db_path=None,
                            methods=('2022', '1979'), tag_method='2022',
                            epoch_length=30, force=False, logger=None):
    """Fill ``sleep_cycles`` and ``stage_durations`` during a detection run.

    Before this existed both tables were created by every run and filled by
    none: only the standalone cycle script ever populated them, so a database
    produced entirely through the GUI had two empty tables and
    ``events.cycle`` NULL on every row.

    Two properties are load-bearing:

    * **The annotation XML is never written.** ``write_xml=False`` and
      ``plot=False`` are passed unconditionally and are not exposed as
      arguments. A detection run reads the scoring; it must not modify the
      file a human rater owns, and a marker write is a silent modification of
      the input.
    * **The caller's connection is reused.** Opening a second write connection
      while a detector holds one is a writer-vs-writer collision under DELETE
      journal mode -- the network-drive failure mode 4.0.2 was cut for.

    ``events.cycle`` is deliberately NOT tagged here: this runs before the
    channel loop, when the run's rows do not exist yet, and a table-wide
    tagging pass would renumber an earlier run's events. Tagging is
    :func:`tag_run_cycles`, called after the loop with the run's own id.

    Every failure is caught and logged: a completed detection must never be
    lost to a cycle-detection problem, and the two tables are recoverable
    afterwards with ``examples/`` cycle backfill.

    Parameters
    ----------
    conn : sqlite3.Connection
        The detector's open write connection.
    annotations : object
        Annotation wrapper exposing ``get_hypnogram()`` and ``epochs``
        (:class:`turtlewave_hdEEG.annotation.CustomAnnotations`). A plain
        Wonambi ``Annotations`` has neither; that is detected and skipped with
        a warning rather than raising.
    subject : str
        Subject id keying the two tables.
    db_path : str or None, optional
        Path of the database ``conn`` is on, for messages. Default ``None``.
    methods : sequence of str, optional
        Cycle definitions to store. Default ``('2022', '1979')`` -- both are
        stored, keyed by ``(subject, method)``, so neither definition is lost.
    tag_method : str, optional
        The definition that will own ``events.cycle`` (used by
        :func:`tag_run_cycles`). Default ``'2022'``.
    epoch_length : float, optional
        Scoring epoch length in seconds. Default ``30``.
    force : bool, optional
        Recompute even when the subject already has cycles stored. Default
        ``False`` (a no-op on the second and later detectors of a run).
    logger : logging.Logger or None, optional
        Logger for the summary and any failure. Default ``None``.

    Returns
    -------
    dict or None
        ``{method: [cycle dicts]}`` when cycles were computed, ``{}`` when the
        subject already had them (no-op), or ``None`` when the step was
        skipped or failed.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    if annotations is None:
        log.warning(
            "No annotations available, so sleep cycles and stage durations "
            "were not stored. events.cycle will stay NULL.")
        return None
    if not hasattr(annotations, 'get_hypnogram'):
        log.warning(
            "The annotation object (%s) has no get_hypnogram(), so sleep "
            "cycles and stage durations were not stored. Pass a "
            "turtlewave_hdEEG.CustomAnnotations to have them filled "
            "automatically.", type(annotations).__name__)
        return None

    try:
        if not force and subject_has_cycles(conn, subject, methods=methods):
            log.debug(
                "Sleep cycles and stage durations are already stored for "
                "'%s'; not recomputing.", subject)
            return {}

        # Lazy import: keeps dbwrite free of an import edge on cycleprocessor
        # (which imports dbwrite) at module load.
        from .cycleprocessor import finalize_cycles_and_durations
        cycles = finalize_cycles_and_durations(
            annotations, db_path, subject=subject, methods=tuple(methods),
            tag_method=tag_method,
            # Not arguments: a detection run must never touch the rater's XML
            # and must never block on plotting.
            write_xml=False, plot=False,
            epoch_length=epoch_length, conn=conn,
            tag_events=False, log_level=log.level or logging.INFO)
        log.info(
            "Stored sleep cycles for subject '%s': %s (annotation XML NOT "
            "modified). events.cycle is tagged after detection, from '%s'.",
            subject,
            ", ".join(f"{m}={len(c)}" for m, c in cycles.items()) or 'none',
            tag_method)
        return cycles
    except Exception as e:
        log.error(
            "Could not store sleep cycles / stage durations for subject "
            "'%s': %s. Detection results are unaffected; back-fill them with "
            "turtlewave_hdEEG.finalize_cycles_and_durations.", subject, e,
            exc_info=True)
        return None


def tag_run_cycles(conn, subject, run_id=None, method='2022', logger=None):
    """Write ``events.cycle`` for one detection run from the stored cycles.

    Reads ``sleep_cycles`` rather than re-detecting, so the numbering in
    ``events.cycle`` is by construction the same one the table holds, and the
    hypnogram is read once per run instead of once per detector.

    Scoped to ``run_id`` by default so a detection annotates the rows it just
    wrote and leaves every other run's ``cycle`` alone.

    Parameters
    ----------
    conn : sqlite3.Connection
        The detector's open write connection.
    subject : str
        Subject whose cycles to apply, in any spelling.
    run_id : str or None, optional
        Restrict the update to this run's rows. ``None`` tags every row (the
        backfill case). Default ``None``.
    method : str, optional
        Which stored cycle definition owns ``events.cycle``. Default
        ``'2022'``.
    logger : logging.Logger or None, optional
        Logger for the count and any failure. Default ``None``.

    Returns
    -------
    int
        Number of event rows tagged (0 when no cycles are stored, or on
        failure -- which is logged, never raised, so a completed detection is
        never lost to a tagging problem).
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    try:
        from .utils import normalize_subject
        from .cycleprocessor import ParalCycles

        canonical = normalize_subject(str(subject))
        spellings = [str(r[0]) for r in conn.execute(
            "SELECT DISTINCT subject FROM sleep_cycles "
            "WHERE subject IS NOT NULL AND subject != ''")
            if normalize_subject(str(r[0])) == canonical]
        if not spellings:
            log.warning(
                "No stored sleep cycles for subject '%s', so events.cycle was "
                "left NULL for this run.", subject)
            return 0
        placeholders = ",".join("?" * len(spellings))
        rows = conn.execute(
            f"SELECT cycle_number, nrem_start, rem_end FROM sleep_cycles "
            f"WHERE subject IN ({placeholders}) AND method = ? "
            f"ORDER BY cycle_number", (*spellings, str(method))).fetchall()
        if not rows:
            log.warning(
                "No '%s' sleep cycles stored for subject '%s', so "
                "events.cycle was left NULL for this run.", method, subject)
            return 0
        # Only the three fields tag_events_with_cycles reads.
        cycles = [{'cycle_number': r[0], 'nrem_start_sec': r[1],
                   'rem_end_sec': r[2]} for r in rows]
        pc = ParalCycles(log_level=log.level or logging.INFO)
        return pc.tag_events_with_cycles(cycles, conn=conn, run_id=run_id)
    except Exception as e:
        log.error(
            "Could not tag events with cycle numbers for subject '%s': %s. "
            "Detection results are unaffected; events.cycle stays NULL and "
            "can be back-filled.", subject, e, exc_info=True)
        return 0


def read_analysed_time(db_path, subject=None, reject_artifacts=True,
                       reject_arousals=True):
    """Read stored density denominators.

    Parameters
    ----------
    db_path : str
        Path to ``neural_events.db``.
    subject : str or None, optional
        Restrict to one subject. Normalised through
        :func:`turtlewave_hdEEG.utils.normalize_subject`, so a bare ``'10sd'``
        finds the rows detection stored as ``'sub-10sd'``. ``None`` (default)
        returns every subject.
    reject_artifacts, reject_arousals : bool, optional
        The rejection settings whose denominator is wanted; these are part of
        the key, so asking for the wrong pair returns nothing rather than a
        mismatched number. Default ``True``.

    Returns
    -------
    dict
        ``{(subject, stage): {'analysed_seconds': float,
        'artefact_seconds_excluded': float or None, 'source': str}}``.
        Empty when the table is absent or holds no matching row.
    """
    out = {}
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='analysed_time'")
        if cur.fetchone() is None:
            return out
        sql = ("SELECT subject, stage, analysed_seconds, "
               "artefact_seconds_excluded, source FROM analysed_time "
               "WHERE reject_artifacts = ? AND reject_arousals = ?")
        params = [1 if reject_artifacts else 0, 1 if reject_arousals else 0]
        if subject is not None:
            from .utils import normalize_subject
            sql += " AND subject = ?"
            params.append(str(normalize_subject(subject)))
        for row in conn.execute(sql, params):
            out[(row[0], row[1])] = {
                'analysed_seconds': row[2],
                'artefact_seconds_excluded': row[3],
                'source': row[4],
            }
    finally:
        conn.close()
    return out


def ensure_direct_write_schema(conn, logger=None):
    """Additively migrate a database for the direct-write path.

    Idempotent and safe on an already-current database. Every step below is
    guarded, and none touches existing rows or unrelated tables:

    1. Add detector-own morphology columns (``det_trough`` etc.), ``run_id``
       and ``epoch_stage`` to ``events`` (only the absent ones, via
       ``PRAGMA table_info``), and create ``idx_events_run`` on ``run_id`` so
       cycle tagging visits one run's rows rather than every run's events in
       the same time span.
    2. Create the ``detection_runs`` provenance table if missing.
    3. Widen the ``processing_status`` primary key from
       ``(channel, event_type)`` to
       ``(channel, event_type, method, freq_lower, freq_upper, stage)`` via a
       create-new / copy / drop / rename migration, run once (detected by the
       absence of the ``method`` column). Legacy rows copy across with the new
       scope columns defaulted to ``''`` / ``0`` so the coarse CSV-import
       markers remain idempotent.
    4. Create the ``pac_coupling`` table via :func:`ensure_pac_schema`, so a
       reader never hits a missing table on a database where PAC has not run.
    5. Create the ``analysed_time`` table via
       :func:`ensure_analysed_time_schema`, which holds the density
       denominator (artefact-free analysed seconds per stage).
    6. Create ``db_meta`` and the ``v_event_density`` view, and stamp
       ``stage_format='joint'`` only on a database holding no events — an
       existing one keeps its unmarked state, which is the only evidence that
       it predates 4.3 and must be migrated before it is re-detected into.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. The caller need not have created the base
        ``events`` table; this only augments it when present.
    logger : logging.Logger or None
        Optional logger for migration messages.
    """
    cur = conn.cursor()

    # (1) events: det_* morphology + run_id + epoch_stage ------------------
    existing = _table_columns(conn, 'events')
    if existing:  # events table exists (fresh DBs get it via initialize_*)
        added = []
        for col, col_type in _DET_MORPH_COLUMNS + (_RUN_ID_COLUMN,
                                                   _EPOCH_STAGE_COLUMN):
            if col not in existing:
                cur.execute(f"ALTER TABLE events ADD COLUMN {col} {col_type}")
                added.append(col)
        if added and logger is not None:
            logger.info(f"Migrated events table: added columns {added}")

        # (1a) events.run_id index. cycleprocessor.tag_run_cycles updates
        # `start_time BETWEEN ... AND run_id = ?` once per cycle; without this
        # index the only usable index is idx_timerange(start_time, end_time),
        # so run_id is a residual and every OTHER run's events in the cycle's
        # time span are read and discarded. Purely a lookup path -- it changes
        # which rows are visited, never which rows match.
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)')

    # (2) detection_runs provenance table ---------------------------------
    conn.execute('''
    CREATE TABLE IF NOT EXISTS detection_runs (
        run_id TEXT PRIMARY KEY,
        subject TEXT,              -- recording this run belongs to
        event_type TEXT,
        method TEXT,
        citation TEXT,
        params_json TEXT,          -- full threshold/band/duration dict
        ref_chan TEXT,
        polar TEXT,
        stages TEXT,
        reject_artifacts INTEGER,
        reject_arousals INTEGER,
        turtlewave_version TEXT,
        wonambi_version TEXT,
        numpy_version TEXT,
        git_sha TEXT,
        timestamp TEXT
    )''')

    # (2a) detection_runs.subject: added so a run is attributable to a
    # recording. events has no subject column, so this (with analysed_time) is
    # what assert_single_subject reads to refuse a second recording's writes.
    dr_cols = _table_columns(conn, 'detection_runs')
    if dr_cols and 'subject' not in dr_cols:
        cur.execute("ALTER TABLE detection_runs ADD COLUMN subject TEXT")
        if logger is not None:
            logger.info("Migrated detection_runs table: added column subject")

    # (2b) rerun_log: one row per scoped channel re-detection (P3) ---------
    conn.execute('''
    CREATE TABLE IF NOT EXISTS rerun_log (
        rerun_id TEXT PRIMARY KEY,
        run_id TEXT,               -- detection_runs row of THIS re-run
        event_type TEXT,
        method TEXT,
        freq_lower REAL,
        freq_upper REAL,
        stages TEXT,
        replace_channels TEXT,     -- JSON list: channels handed to the re-run
        redetected_channels TEXT,  -- JSON list: actually re-detected
        dropped_channels TEXT,     -- JSON list: forced-drop by the clean gate
        sidecar_path TEXT,         -- reviewer artefact sidecar XML
        backup_path TEXT,          -- qc_backup snapshot dir (the rollback)
        requested_by TEXT,
        timestamp TEXT
    )''')

    # (3) widen processing_status PK --------------------------------------
    ps_cols = _table_columns(conn, 'processing_status')
    if ps_cols and 'method' not in ps_cols:
        if logger is not None:
            logger.info(
                "Migrating processing_status to per-scope primary key "
                "(channel, event_type, method, freq_lower, freq_upper, stage)")
        conn.execute('''
        CREATE TABLE processing_status_new (
            channel TEXT NOT NULL,
            event_type TEXT NOT NULL,
            method TEXT NOT NULL DEFAULT '',
            freq_lower REAL NOT NULL DEFAULT 0,
            freq_upper REAL NOT NULL DEFAULT 0,
            stage TEXT NOT NULL DEFAULT '',
            json_file TEXT,
            processed BOOLEAN DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            last_attempt_time TEXT,
            success BOOLEAN DEFAULT 0,
            error_message TEXT,
            PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage)
        )''')
        conn.execute('''
        INSERT INTO processing_status_new
            (channel, event_type, json_file, processed, attempts,
             last_attempt_time, success, error_message)
        SELECT channel, event_type, json_file, processed, attempts,
               last_attempt_time, success, error_message
        FROM processing_status''')
        conn.execute('DROP TABLE processing_status')
        conn.execute('ALTER TABLE processing_status_new RENAME TO processing_status')

    # (4) pac_coupling: created eagerly so a reader never faces a missing
    # table on a database where PAC has not (yet) been run.
    ensure_pac_schema(conn)

    # (5) analysed_time: the density denominator. Created eagerly for the same
    # reason -- density.event_density must be able to tell "no denominator
    # stored" from "table does not exist".
    ensure_analysed_time_schema(conn)

    # (6) db_meta + the stage_format marker -------------------------------
    # Seeded 'joint' ONLY for a database with no events yet. An existing
    # database full of per-epoch rows must stay UNMARKED: the absence of the
    # marker is the evidence assert_stage_format_compatible reads to refuse a
    # duplicating re-detection, and stamping it here would destroy that
    # evidence on the very databases the guard exists to protect.
    ensure_db_meta_schema(conn)
    n_events = 0
    if _table_columns(conn, 'events'):
        try:
            n_events = conn.execute(
                "SELECT COUNT(*) FROM events").fetchone()[0]
        except sqlite3.OperationalError:
            n_events = 0

    # (7) det_ptp units ---------------------------------------------------
    # Same rule and the same reason as stage_format: seed the marker ONLY on
    # a database with no events. An existing database's slow-wave and
    # K-complex rows hold Wonambi's sample count, and the microvolt values
    # this release writes overlap that range numerically (102-113 samples vs
    # 125-171 uV on the same recordings), so stamping 'microvolts' over
    # pre-4.3 rows would assert something false about them and destroy the
    # only evidence that they are a different quantity. An unmarked database
    # that already holds events stays unmarked and is reported once.
    if ptp_units(conn) is None:
        if not n_events:
            set_db_meta(conn, PTP_UNITS_KEY, PTP_UNITS_MICROVOLTS)
        elif logger is not None:
            logger.info(
                "This database holds %d event row(s) and carries no "
                "db_meta.%s marker, so its slow-wave and K-complex det_ptp "
                "values are Wonambi's SAMPLE COUNT, not microvolts. Rows "
                "written from 4.3 on are microvolts. Do not pool det_ptp "
                "across the two; peak2peak_amp is microvolts throughout and "
                "is unaffected.", n_events, PTP_UNITS_KEY)

    if stage_format(conn) is None:
        if not n_events:
            set_db_meta(conn, STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT)
        elif logger is not None:
            logger.info(
                "This database holds %d event row(s) and carries no "
                "db_meta.%s marker, so it is treated as pre-4.3 (per-epoch "
                "events.stage). Detection into a scope that already has rows "
                "will be refused rather than silently duplicated; see "
                "examples/migrate_stage_to_joint.py.", n_events,
                STAGE_FORMAT_KEY)

    # (8) v_event_density: density in plain SQL, for R and sqlite3 callers.
    ensure_density_view(conn, logger=logger)

    conn.commit()


# SQL fragment counting how many stage components a token spans. Built from
# string lengths because SQLite has no split(): each removal of a known label
# shortens the token by that label's length, so the number of removals is the
# length difference divided by the label length. Written once here and reused
# in both halves of the view so the "all components present" test cannot drift
# from the pooling it guards.
#
# Order matters exactly as it does in split_stage_token: the NREM labels are
# removed FIRST, because 'NREM1' contains 'REM' and a naive REM test would
# count NREM epochs as REM. `_SQL_STAGE_REST` is the token with every NREM
# label already removed, which is what makes the REM test safe.
_SQL_STAGE_REST = ("replace(replace(replace({col}, 'NREM1', ''), "
                   "'NREM2', ''), 'NREM3', '')")

_SQL_STAGE_N_COMPONENTS = (
    "((length({col}) - length(" + _SQL_STAGE_REST + ")) / 5"
    " + (length(" + _SQL_STAGE_REST + ") - length(replace("
    + _SQL_STAGE_REST + ", 'REM', ''))) / 3"
    " + (length(" + _SQL_STAGE_REST + ") - length(replace("
    + _SQL_STAGE_REST + ", 'Wake', ''))) / 4)")


def ensure_density_view(conn, logger=None):
    """(Re)create the ``v_event_density`` SQL view.

    Density without Python: joins ``events`` to ``analysed_time`` and does the
    joint-token component pooling in SQL, so R, ``sqlite3`` and any BI tool can
    read the same numbers :func:`turtlewave_hdEEG.density.event_density`
    computes.

    A **view**, never a table: it recomputes at query time, so a scoped channel
    re-detection (which deletes and re-inserts that channel's rows) can never
    leave it holding pre-correction numbers. A materialised density table would
    go stale exactly when the pipeline corrects itself.

    Dropped and recreated on every call rather than
    ``CREATE VIEW IF NOT EXISTS``, so a database opened by a newer release
    always carries that release's definition instead of a stale one.

    Columns
    -------
    ``subject``, ``channel``, ``event_type``, ``method``, ``stage`` (the
    stored token), ``freq_lower``, ``freq_upper``, ``n_events``,
    ``analysed_minutes``, ``density_per_min``, ``artefact_minutes_excluded``,
    ``mean_duration_sec``, ``reject_artifacts``, ``reject_arousals``,
    ``denominator_complete``.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. Commits, does not close.
    logger : logging.Logger or None, optional
        Logger for a failure to create the view. Default ``None``.

    Returns
    -------
    bool
        True when the view exists on return.

    Notes
    -----
    Two behaviours differ from :func:`turtlewave_hdEEG.density.event_density`
    and are deliberate, because SQL cannot express them without inventing rows:

    * **No honest zeros.** A channel that ran and detected nothing has no row
      in ``events`` and therefore none here. The Python API cross-joins
      ``processing_status`` to add it. A montage summary taken from this view
      alone is computed over the channels that fired.
    * **No per-identity stage scope.** Rows appear for whatever
      ``(event_type, method, stage)`` combinations exist in ``events``;
      nothing is fabricated, but nothing is filtered by which run searched
      what either.

    ``density_per_min`` is NULL, never 0, when a component of the stage token
    has no ``analysed_time`` row (``denominator_complete = 0``). Rows with a
    NULL ``stage`` are excluded: they have no denominator at all.
    """
    n_comp = _SQL_STAGE_N_COMPONENTS.format(col='e.stage')
    try:
        conn.execute("DROP VIEW IF EXISTS v_event_density")
        conn.execute(f'''
        CREATE VIEW v_event_density AS
        SELECT
            d.subject                                   AS subject,
            e.channel                                   AS channel,
            e.event_type                                AS event_type,
            e.method                                    AS method,
            e.stage                                     AS stage,
            e.freq_lower                                AS freq_lower,
            e.freq_upper                                AS freq_upper,
            COUNT(*)                                    AS n_events,
            -- NULL, not a partial sum, when a component of the token has no
            -- analysed_time row: an incomplete denominator divided into a
            -- complete count inflates density silently.
            CASE WHEN d.n_components = d.n_expected
                 THEN d.analysed_seconds / 60.0
                 ELSE NULL END                          AS analysed_minutes,
            CASE WHEN d.n_components = d.n_expected AND d.analysed_seconds > 0
                 THEN COUNT(*) / (d.analysed_seconds / 60.0)
                 ELSE NULL END                          AS density_per_min,
            CASE WHEN d.n_components = d.n_expected
                 THEN d.artefact_seconds / 60.0
                 ELSE NULL END                          AS artefact_minutes_excluded,
            AVG(e.duration)                             AS mean_duration_sec,
            d.reject_artifacts                          AS reject_artifacts,
            d.reject_arousals                           AS reject_arousals,
            CASE WHEN d.n_components = d.n_expected THEN 1 ELSE 0 END
                                                        AS denominator_complete
        FROM events e
        JOIN (
            SELECT
                t.stage       AS token,
                a.subject     AS subject,
                a.reject_artifacts AS reject_artifacts,
                a.reject_arousals  AS reject_arousals,
                SUM(a.analysed_seconds) AS analysed_seconds,
                SUM(COALESCE(a.artefact_seconds_excluded, 0)) AS artefact_seconds,
                COUNT(*)      AS n_components,
                t.n_expected  AS n_expected
            FROM (
                SELECT DISTINCT e.stage AS stage,
                       {n_comp} AS n_expected
                FROM events e WHERE e.stage IS NOT NULL
            ) t
            JOIN analysed_time a
              ON (CASE a.stage
                    WHEN 'REM' THEN instr(
                        replace(replace(replace(t.stage, 'NREM1', ''),
                                'NREM2', ''), 'NREM3', ''), 'REM')
                    ELSE instr(t.stage, a.stage)
                  END) > 0
            GROUP BY t.stage, a.subject, a.reject_artifacts, a.reject_arousals
        ) d ON d.token = e.stage
        WHERE e.stage IS NOT NULL
        GROUP BY d.subject, e.channel, e.event_type, e.method, e.stage,
                 e.freq_lower, e.freq_upper, d.reject_artifacts,
                 d.reject_arousals
        ''')
        conn.commit()
        return True
    except sqlite3.Error as e:
        # A view is a convenience: never lose a detection run over one.
        if logger is not None:
            logger.warning(
                "Could not create the v_event_density SQL view (%s). Density "
                "from Python (turtlewave_hdEEG.density.event_density) is "
                "unaffected.", e)
        return False


def record_run(conn, run_id, event_type, method, citation, params_json,
               ref_chan, polar, stages, reject_artifacts, reject_arousals,
               subject=None):
    """Write one ``detection_runs`` provenance row for an invocation.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    run_id : str
        Unique id for this invocation (a fresh ``uuid4`` per call).
    event_type : str
        Event type detected.
    method : str
        Method name(s), joined for multi-method runs.
    citation : str
        Literature citation for the method.
    params_json : str
        JSON string of the full parameter dict (thresholds/band/durations).
    ref_chan : str
        Reference channel(s), serialized.
    polar : str
        Polarity flag (``'normal'`` / ``'opposite'``).
    stages : str
        Requested stage set, serialized.
    reject_artifacts, reject_arousals : bool
        Artifact/arousal rejection settings.
    subject : str or None, optional
        Recording this run belongs to. Stored so a run is attributable and so
        :func:`assert_single_subject` can see it even on a run that stored no
        ``analysed_time`` row. Default ``None``.
    """
    prov = provenance()
    conn.execute('''
    INSERT OR REPLACE INTO detection_runs
        (run_id, subject, event_type, method, citation, params_json, ref_chan,
         polar, stages, reject_artifacts, reject_arousals, turtlewave_version,
         wonambi_version, numpy_version, git_sha, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, (None if subject is None else str(subject)),
        event_type, method, citation, params_json,
        str(ref_chan), str(polar), str(stages),
        1 if reject_artifacts else 0, 1 if reject_arousals else 0,
        prov['turtlewave_version'], prov['wonambi_version'],
        prov['numpy_version'], git_sha(), datetime.datetime.now().isoformat(),
    ))
    conn.commit()


def record_rerun(conn, rerun_id, run_id, event_type, method, freq_lower,
                 freq_upper, stages, replace_channels, redetected_channels,
                 dropped_channels, sidecar_path, backup_path, requested_by):
    """Write one ``rerun_log`` provenance row for a scoped channel re-detection.

    Makes a scoped re-run self-describing: which channels were requested, which
    were actually re-detected vs forced-drop by the clean gate, and the sidecar
    and ``qc_backup`` snapshot paths that constitute the rollback.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    rerun_id : str
        Unique id for this re-run invocation (a fresh ``uuid4``).
    run_id : str
        The ``detection_runs`` row id written for this re-run's detection.
    event_type, method : str
        Detection scope.
    freq_lower, freq_upper : float
        Band bounds.
    stages : str
        Requested stage set, serialized.
    replace_channels : sequence of str
        Channels handed to the re-run as the replace scope.
    redetected_channels : sequence of str
        Channels actually re-detected (passed the clean gate).
    dropped_channels : sequence of str
        Channels the clean gate forced to drop (not re-detected).
    sidecar_path : str or None
        Path to the reviewer artefact sidecar XML.
    backup_path : str or None
        Path to the ``qc_backup`` snapshot directory (the rollback point).
    requested_by : str or None
        Reviewer name, for provenance.
    """
    import json as _json
    conn.execute('''
    INSERT OR REPLACE INTO rerun_log
        (rerun_id, run_id, event_type, method, freq_lower, freq_upper, stages,
         replace_channels, redetected_channels, dropped_channels,
         sidecar_path, backup_path, requested_by, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        rerun_id, run_id, event_type, method,
        float(freq_lower) if freq_lower is not None else None,
        float(freq_upper) if freq_upper is not None else None,
        str(stages),
        _json.dumps(sorted(str(c) for c in (replace_channels or []))),
        _json.dumps(sorted(str(c) for c in (redetected_channels or []))),
        _json.dumps(sorted(str(c) for c in (dropped_channels or []))),
        sidecar_path, backup_path, requested_by,
        datetime.datetime.now().isoformat(),
    ))
    conn.commit()


def recover_run_scope(db_path, event_type, method, freq_lower=None,
                      freq_upper=None):
    """Read the reference/polarity/cat of the most recent matching run.

    Used by the re-run guard to reuse the ORIGINAL run's invariant parameters
    (``ref_chan``, ``polar``, ``cat``) rather than a driver/GUI default -- a
    wrong ``polar`` inverts trough polarity (the same failure axis as the PAC
    180 degree bug), and a different ``ref_chan`` makes every amplitude
    threshold incomparable. Returns ``None`` when no matching run is on record,
    so the caller can REFUSE to re-detect rather than guess.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    event_type : str
        Event type of the scope to recover (e.g. ``'spindle'``).
    method : str
        Method string to match against ``detection_runs.method``.
    freq_lower, freq_upper : float or None, optional
        Band bounds. When both are given the params dict is checked for a
        matching band before accepting the row, so a same-method run at a
        different band is not mistaken for this scope. When ``None`` the band is
        not used to disambiguate.

    Returns
    -------
    dict or None
        ``{'ref_chan', 'polar', 'cat', 'cat_recorded', 'reject_artifacts',
        'reject_arousals', 'stages', 'run_id', 'params'}`` from the most recent
        matching ``detection_runs`` row, or ``None`` if no such row exists.
        ``params`` is the full recorded parameter dict (thresholds/band/
        durations) so a re-run can reuse the original detector thresholds, not
        just the invariants.

        ``ref_chan`` and ``polar`` are returned as real Python objects: preferred
        from the typed ``params_json`` when present (P3+ runs), otherwise parsed
        back from the ``detection_runs`` column, which ``record_run`` stored as
        ``str(...)``. The column is parsed with :func:`ast.literal_eval` so
        ``"['M1', 'M2']"`` becomes the list ``['M1', 'M2']`` and ``'[]'`` becomes
        ``[]`` (NOT the repr strings) and ``polar`` ``'None'`` maps to ``None``.
        This matters because the primary re-run target -- databases written by the
        committed P2 path -- has NO ``ref_chan``/``cat`` in ``params_json`` and
        would otherwise recover a string that ``read_data`` cannot use.

        ``cat`` is read only from ``params_json`` (it was never stored in a
        column). ``cat_recorded`` is ``True`` only when the ``cat`` key was
        actually present, so the caller can tell a genuine ``cat=None`` (no
        concatenation) apart from a pre-P3 run that simply never recorded it and
        REFUSE in the latter case (production runs use ``cat=(1, 1, 1, 0)``, so a
        silently-assumed ``None`` would pool differently and shift thresholds).
    """
    import ast as _ast
    import json as _json

    def _parse_ref_col(col):
        """Parse a ``str(ref_chan)`` column back to a Python object."""
        if col is None:
            return None
        s = str(col)
        try:
            return _ast.literal_eval(s)  # "['M1','M2']"->list, '[]'->[], 'None'->None
        except (ValueError, SyntaxError):
            return col  # a bare channel label stored as a plain string

    def _parse_polar_col(col):
        """Parse a ``str(polar)`` column: 'None'->None, else the string."""
        if col is None:
            return None
        s = str(col)
        return None if s == 'None' else s

    # Read-only; 60 s busy timeout for the same reason as the writers (a
    # DELETE-mode database lets a writer block this read).
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        try:
            cur = conn.execute('''
            SELECT run_id, params_json, ref_chan, polar, stages,
                   reject_artifacts, reject_arousals
            FROM detection_runs
            WHERE event_type = ? AND method = ?
            ORDER BY timestamp DESC
            ''', (str(event_type), str(method)))
        except sqlite3.OperationalError:
            return None
        for row in cur.fetchall():
            run_id, params_json, ref_chan, polar, stages, rj_a, rj_r = row
            params = {}
            if params_json:
                try:
                    params = _json.loads(params_json)
                except Exception:
                    params = {}
            if freq_lower is not None and freq_upper is not None:
                band = params.get('frequency')
                if band is not None and len(band) == 2:
                    if (abs(float(band[0]) - float(freq_lower)) > 1e-6 or
                            abs(float(band[1]) - float(freq_upper)) > 1e-6):
                        continue  # same method, different band -- keep looking
            return {
                'run_id': run_id,
                # Prefer the typed params_json value; else parse the stringified
                # column back to a real object (P2 runs lack it in params_json).
                'ref_chan': (params['ref_chan'] if 'ref_chan' in params
                             else _parse_ref_col(ref_chan)),
                'polar': (params['polar'] if 'polar' in params
                          else _parse_polar_col(polar)),
                'cat': params.get('cat'),
                'cat_recorded': 'cat' in params,
                'reject_artifacts': bool(rj_a),
                'reject_arousals': bool(rj_r),
                'stages': stages,
                'params': params,
            }
        return None
    finally:
        conn.close()


def resume_skip_channels(conn, event_type, method, freq_lower, freq_upper,
                         stage_key):
    """Return channels already completed for this exact detection scope.

    A channel is skippable on resume only when a ``processing_status`` row
    matches the full scope (type, method, band, stage set) *and* recorded
    ``success = 1``. Failed channels (``success = 0``) are re-run.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection.
    event_type : str
        Event type.
    method : str
        Method string of the current run.
    freq_lower, freq_upper : float
        Band bounds of the current run.
    stage_key : str
        Joined stage set of the current run (e.g. ``'NREM2NREM3'``).

    Returns
    -------
    set of str
        Channels to skip.
    """
    try:
        cur = conn.execute('''
        SELECT channel FROM processing_status
        WHERE event_type = ? AND method = ? AND freq_lower = ?
          AND freq_upper = ? AND stage = ? AND success = 1
        ''', (event_type, method, freq_lower, freq_upper, stage_key))
        return {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        # Table not yet migrated to the wide schema (no rows to skip).
        return set()


def _get(ev, *keys):
    """Return the first present, non-``None`` value among ``keys``, else ``None``.

    Deliberately maps a missing morphology field to ``None`` (SQL ``NULL``),
    never ``0.0`` -- the known 0.0-vs-NaN trap where an absent trough becomes a
    real zero-amplitude row.
    """
    for k in keys:
        if k in ev and ev[k] is not None:
            try:
                return float(ev[k])
            except (TypeError, ValueError):
                return None
    return None


def event_det_morphology(ev):
    """Extract the detector's own morphology from a raw event dict.

    Slow-wave and K-complex events carry ``trough_val`` / ``peak_val`` / ``ptp``
    directly. Spindle events generally do not (they are oscillatory, not
    trough-based); absent fields resolve to ``None`` (NULL), not ``0.0``.

    Parameters
    ----------
    ev : dict
        Raw event dict from the Wonambi detector.

    Returns
    -------
    dict
        Keys ``det_trough``, ``det_peak``, ``det_ptp``, ``det_trough_time``,
        ``det_peak_time`` (each ``float`` or ``None``).
    """
    return {
        'det_trough': _get(ev, 'trough_val'),
        'det_peak': _get(ev, 'peak_val'),
        'det_ptp': _get(ev, 'ptp', 'ptp_det'),
        'det_trough_time': _get(ev, 'trough_time'),
        'det_peak_time': _get(ev, 'peak_time'),
    }


def sec_to_hms(start_sec, recording_start_time=None):
    """Format an event start time as ``HH:MM:SS.mmm``.

    Parameters
    ----------
    start_sec : float
        Start time in seconds from recording start.
    recording_start_time : datetime.datetime or None
        Absolute recording start. When ``None`` (the common case, since the
        real pipeline bypasses ``LargeDataset``), a relative wall-clock is used.

    Returns
    -------
    str
        Formatted time string.
    """
    try:
        start_sec = float(start_sec)
    except (TypeError, ValueError):
        return ''
    if recording_start_time is not None:
        try:
            evt = recording_start_time + datetime.timedelta(seconds=start_sec)
            return evt.strftime('%H:%M:%S.%f')[:-3]
        except Exception:
            pass
    hours = int(start_sec // 3600)
    minutes = int((start_sec % 3600) // 60)
    sec = start_sec % 60
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def compute_batched_params(param_segments, frequency, s_freq, n_fft_sec,
                           logger=None):
    """Re-measure amplitude/spectral parameters for a channel in one call.

    Runs :func:`wonambi.trans.analyze.event_params` once over all of a channel's
    in-memory event windows, instead of re-reading the raw file per event. The
    windows are slices of the data the detector actually saw (post
    re-reference / detrend), so with a non-empty reference or detrend enabled
    these values can differ slightly from the legacy per-event raw re-read --
    that difference is intentional (the in-memory values reflect the detector's
    input).

    Parameters
    ----------
    param_segments : list of dict
        Wonambi-style segment dicts (each with a single-channel ``'data'``
        Data object), one per event, in event order.
    frequency : tuple
        Detection band ``(lo, hi)`` used for band power.
    s_freq : float
        Sampling frequency (Hz), for the FFT length.
    n_fft_sec : float or None
        FFT window in seconds; ``None`` disables an explicit ``n_fft``.
    logger : logging.Logger or None
        Optional logger.

    Returns
    -------
    list of dict
        One dict per input segment (aligned by order) with keys ``min_amp``,
        ``max_amp``, ``peak2peak_amp``, ``rms``, ``power``, ``peak_power_freq``,
        ``energy``, ``peak_energy_freq``. Values are ``None`` for windows that
        were empty or failed measurement.
    """
    empty = {'min_amp': None, 'max_amp': None, 'peak2peak_amp': None,
             'rms': None, 'power': None, 'peak_power_freq': None,
             'energy': None, 'peak_energy_freq': None}
    results = [dict(empty) for _ in param_segments]
    if not param_segments:
        return results

    # Drop windows with no samples so a single bad slice can't fail the batch.
    non_empty = []
    for idx, seg in enumerate(param_segments):
        try:
            n = seg['data'].axis['time'][0].size
        except Exception:
            n = 0
        if n > 0:
            non_empty.append((idx, seg))

    if not non_empty:
        return results

    n_fft = int(n_fft_sec * s_freq) if (n_fft_sec and s_freq) else None
    try:
        params = event_params(
            [seg for _, seg in non_empty], 'all',
            band=frequency, n_fft=n_fft)
    except Exception as e:
        if logger is not None:
            logger.warning(f"Batched event_params failed for channel: {e}")
        return results

    if not params:
        return results

    for (orig_idx, _), p in zip(non_empty, params):
        try:
            chan = p['data'].axis['chan'][0][0]
            results[orig_idx] = {
                'min_amp': float(p['minamp'](chan=chan)[0]),
                'max_amp': float(p['maxamp'](chan=chan)[0]),
                'peak2peak_amp': float(p['ptp'](chan=chan)[0]),
                'rms': float(p['rms'](chan=chan)[0]),
                'power': float(p['power'][chan]),
                'peak_power_freq': float(p['peakpf'][chan]),
                'energy': float(p['energy'][chan]),
                'peak_energy_freq': float(p['peakef'][chan]),
            }
        except Exception as e:
            if logger is not None:
                logger.warning(f"Could not read batched params for one event: {e}")
    return results


# Per-``Data``-object memo of "is this trial's time axis strictly increasing?".
#
# ``fast_time_slice`` needs that verdict on EVERY event, but establishing it
# costs O(n_samples); on a whole-night concatenated segment (~2 x 10^6 samples)
# paying it per event would reintroduce a large part of the cost the index-based
# slice exists to remove. The verdict is therefore computed once per (Data,
# time-array) pair and memoised here.
#
# Keys are weak, so caching never keeps a night-length segment alive. The value
# holds a strong reference to the time array it was computed from and the fast
# path re-validates with ``is``, so replacing ``data.axis['time'][trial]`` (as
# ``select``/``math`` do, by building a new Data) invalidates the entry rather
# than silently reusing a verdict about a different array.
_MONOTONIC_TIME_CACHE = WeakKeyDictionary()


def _time_axis_is_strictly_increasing(data, trial=0):
    """Whether a trial's time axis increases strictly, memoised per object.

    Strict increase (no duplicated timestamps) is what makes an index range from
    :func:`numpy.searchsorted` equivalent to Wonambi's value-based selection: a
    repeated timestamp would make ``_get_indices`` resolve every copy to its
    FIRST occurrence, which a contiguous slice does not reproduce.

    Parameters
    ----------
    data : wonambi Data
        Segment whose time axis is tested.
    trial : int, optional
        Trial index. Default 0.

    Returns
    -------
    bool
        True when the trial's time axis is non-empty and strictly increasing.
    """
    try:
        time_values = data.axis['time'][trial]
    except Exception:
        return False
    if not isinstance(time_values, np.ndarray) or time_values.ndim != 1:
        return False

    try:
        cached = _MONOTONIC_TIME_CACHE.get(data)
    except TypeError:  # unhashable / non-weakref-able Data subclass
        cached = None
        cacheable = False
    else:
        cacheable = True
    if cached is not None:
        cached_trial, cached_values, cached_ok = cached
        if cached_trial == trial and cached_values is time_values:
            return cached_ok

    ok = bool(time_values.size > 0 and np.all(np.diff(time_values) > 0))
    if cacheable:
        try:
            _MONOTONIC_TIME_CACHE[data] = (trial, time_values, ok)
        except TypeError:
            pass
    return ok


def fast_time_slice(data, t0, t1):
    """Index-based equivalent of ``select(data, time=(t0, t1))``.

    Wonambi's :func:`wonambi.trans.select.select` resolves a time range by
    building the boolean mask ``(t0 <= values) & (values < t1)`` and then handing
    the selected VALUES to ``Data.__call__``, which calls
    ``wonambi.datatype._get_indices`` -- a Python loop that rescans the whole
    trial's time axis once per requested timestamp ("probably not very fast, but
    it's pretty robust"). On a whole-night concatenated segment that is
    ``O(window_samples x night_samples)`` per event, which dominated the
    direct-to-database detection path.

    Because the time axis within a trial is monotonic, the same window is found
    with two binary searches and taken as a contiguous slice:
    ``O(log night + window)``.

    Boundary convention
    -------------------
    HALF-OPEN, ``[t0, t1)`` -- start inclusive, end exclusive -- reproducing
    ``wonambi/trans/select.py`` exactly. ``searchsorted(..., side='left')`` gives
    the first index with ``value >= t0`` (the first sample the mask keeps) and
    the first index with ``value >= t1`` (one past the last sample it keeps).
    An off-by-one here shifts every measured window, so it is pinned by
    ``test_fast_time_slice_boundary_is_half_open``.

    Why the two agree exactly
    -------------------------
    ``searchsorted`` and Wonambi's mask perform the SAME float64 comparison on
    the SAME values, so on a sorted axis they cannot disagree -- the equality is
    structural, not a numerical coincidence. Every condition below exists to
    establish one of the two premises (sortedness; a common float64 comparison
    domain), and the function declines rather than approximating when it cannot:

    * **strictly increasing** -- duplicated timestamps make
      ``_get_indices`` resolve every copy to its FIRST occurrence, which a
      contiguous slice does not reproduce (on ``t = [1, 2, 2, 3, 4]`` the two
      genuinely return different data).
    * **float64 time axis** -- with a float32 axis NumPy's value-based casting
      evaluates ``t0 <= values`` in float32 while ``searchsorted`` compares in
      float64, and the two disagree by one sample on roughly half of all
      windows. No shipped reader produces one (``wonambi/dataset.py:409`` and
      ``turtlewave_hdEEG/dataset.py:642`` both build float64), so this is a
      guard, not a live case.
    * **finite bounds** -- ``searchsorted`` sorts NaN LAST, so ``t1 = nan``
      would return the whole remainder of the night where the mask
      ``(values < nan)`` selects nothing. That is the one input where an
      unguarded index slice yields a plausible-looking but physically wrong
      measurement window instead of an empty one. Infinities happen to agree,
      but are declined with NaN for one simple predicate.

    Parameters
    ----------
    data : wonambi Data
        Segment to slice. Only single-trial segments with a strictly increasing
        float64 time axis take the fast path; anything else returns ``None`` so
        the caller can fall back to :func:`select`.
    t0, t1 : float
        Window bounds in seconds from recording start. Must both be finite;
        non-finite bounds are declined.

    Returns
    -------
    wonambi Data or None
        A new Data of the same class holding the sliced window, byte-identical
        to ``select(data, time=(t0, t1))``; or ``None`` when the fast path does
        not apply (multi-trial, no time axis, non-monotonic or duplicated
        timestamps, a non-float64 time axis, a non-finite bound, or a data array
        whose shape disagrees with its axes).
    """
    # Non-finite bounds: see "finite bounds" above. Checked FIRST because it is
    # the only condition whose failure would otherwise produce a WRONG window
    # rather than an exception or a mere inefficiency. A non-numeric bound
    # (None, as in Wonambi's open-ended ``time=(None, t1)`` idiom) is declined
    # here too rather than crashing in searchsorted.
    try:
        if not (np.isfinite(t0) and np.isfinite(t1)):
            return None
    except (TypeError, ValueError):
        return None

    # Multi-trial is not handled: ``select`` slices every trial and each trial
    # carries its own time axis. Segments from wonambi's ``fetch(...).read_data``
    # are always single-trial (one ChanTime per segment dict, whatever ``cat``
    # is -- ``cat`` changes the NUMBER of segment dicts, not the trials inside
    # one), so this is a safety net rather than an expected case.
    try:
        if data.number_of('trial') != 1:
            return None
    except Exception:
        return None
    if 'time' not in data.axis:
        return None

    time_values = data.axis['time'][0]
    # float64 only: see "float64 time axis" above. Checked BEFORE the O(n)
    # monotonicity scan so a declined axis costs nothing.
    if (not isinstance(time_values, np.ndarray)
            or time_values.dtype != np.float64):
        return None
    if not _time_axis_is_strictly_increasing(data, 0):
        return None

    array = data.data[0]
    if not isinstance(array, np.ndarray):
        return None
    time_pos = data.index_of('time')
    # An axis/data shape mismatch (e.g. the channel-concatenated segment that
    # ``read_data(concat_chan=True)`` ravels, whose time axis no longer
    # describes the data) must not be sliced by index.
    if array.ndim != len(data.axis) or array.shape[time_pos] != time_values.size:
        return None

    lo = int(np.searchsorted(time_values, t0, side='left'))
    hi = int(np.searchsorted(time_values, t1, side='left'))
    if hi < lo:  # t1 < t0: the mask would be empty, so is the slice
        hi = lo

    # Rebuild the container the way ``select`` does: same class, same s_freq /
    # start_time / attr, one trial, non-selected axes passed through by
    # reference and the time axis replaced by a fresh (copied) array.
    output = data._copy(axis=False)
    for axis_name in data.axis:
        output.axis[axis_name] = np.empty(1, dtype='O')
        output.axis[axis_name][0] = data.axis[axis_name][0]
    output.axis['time'][0] = time_values[lo:hi].copy()
    output.data = np.empty(1, dtype='O')
    index = [slice(None)] * array.ndim
    index[time_pos] = slice(lo, hi)
    output.data[0] = array[tuple(index)].copy()
    return output


def make_param_segment(data, start_time, end_time, event_type, stage,
                       chan, buffer=0.1):
    """Slice an in-memory Data window for one event, for batched measurement.

    Uses :func:`fast_time_slice` (two binary searches + a contiguous slice) and
    falls back to Wonambi's :func:`select` when the fast path does not apply.
    The two produce byte-identical windows; see ``fast_time_slice`` for the
    conditions and the half-open ``[t0, t1)`` boundary convention.

    Parameters
    ----------
    data : wonambi Data
        The (single- or multi-channel) segment the detector ran on.
    start_time, end_time : float
        Event bounds in seconds from recording start.
    event_type : str
        Placed in the segment ``'name'``.
    stage : str or None
        Resolved single stage (metadata only).
    chan : str
        Channel label (metadata; the actual label is taken from the sliced
        data when reading parameters).
    buffer : float
        Symmetric flank in seconds added before/after the event, matching the
        legacy exporter's 100 ms buffer so measured values stay comparable.

    Returns
    -------
    dict or None
        A Wonambi-style segment dict, or ``None`` if slicing failed.
    """
    try:
        t0 = max(0.0, float(start_time) - buffer)
        t1 = float(end_time) + buffer
        sub = fast_time_slice(data, t0, t1)
        if sub is None:
            sub = select(data, time=(t0, t1))
        return {'data': sub, 'name': event_type, 'start': float(start_time),
                'end': float(end_time), 'n_stitch': 0, 'stage': stage,
                'cycle': None, 'chan': chan}
    except Exception:
        return None


def upsert_processing_status(conn, event_type, channel, method, freq_lower,
                             freq_upper, stage_key, success, error_message=None,
                             json_file=None):
    """Insert-or-replace a per-scope ``processing_status`` row.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection.
    event_type, channel, method : str
        Scope identity.
    freq_lower, freq_upper : float
        Band bounds of the run.
    stage_key : str
        Joined stage set of the run.
    success : bool
        Whether the channel completed. ``error_message`` is set to
        ``'No events detected'`` by the caller for a successful-but-empty
        channel to preserve the empty-vs-failed distinction.
    error_message : str or None
        Failure detail (truncated to 500 chars).
    json_file : str or None
        Associated JSON path, if any.
    """
    conn.execute('''
    INSERT OR REPLACE INTO processing_status
        (channel, event_type, method, freq_lower, freq_upper, stage,
         json_file, processed, attempts, last_attempt_time, success,
         error_message)
    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, datetime('now'), ?, ?)
    ''', (
        channel, event_type, method, freq_lower, freq_upper, stage_key,
        json_file, 1 if success else 0,
        (error_message[:500] if error_message else None),
    ))


def write_channel_events(conn, run_id, event_type, channel, method,
                         freq_lower, freq_upper, stage_key, events, batched,
                         recording_start_time, n_fft_sec, logger=None,
                         replace=False, replace_methods=None):
    """Write one channel's events + status in a single transaction.

    Opens an explicit transaction, ``INSERT OR REPLACE`` s every event row
    (deterministic uuid5, detector-own ``det_*`` morphology, batched re-measured
    and spectral columns) and upserts the channel's ``processing_status`` to
    ``success = 1``, then commits. An empty channel still commits a status row
    with ``error_message = 'No events detected'`` to preserve the
    empty-vs-failed distinction. On any error the transaction is rolled back and
    the exception re-raised for the caller to record as a failure.

    Scoped channel re-detection (P3)
    --------------------------------
    When ``replace`` is True, a scoped ``DELETE`` runs FIRST, inside the same
    transaction as the inserts, so this channel's stale rows for the run scope
    are cleared before the fresh set is written. This is required because a clean
    re-run (with more artefact epochs excluded) generally yields *fewer* events
    than the original: a blind ``INSERT OR REPLACE`` would leave the surplus
    original rows behind. The delete is scoped to
    ``(event_type, channel, freq_lower, freq_upper, method IN replace_methods)``
    and is deliberately NOT scoped by stage -- a re-run may re-resolve an event's
    epoch stage, so a stage-scoped delete could orphan a row that moved stages.
    ``freq_lower``/``freq_upper`` are matched with ``IS`` (NULL-safe) so a NULL
    band still deletes. Only channels the caller flags are ever touched; every
    other channel's rows are untouched. Because delete + insert share one
    ``BEGIN``/``commit``, a concurrent reader never sees the channel with zero
    rows.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    run_id : str
        Detection run id linking these rows to ``detection_runs``.
    event_type, channel, method : str
        Run scope identity. ``method`` is the run's (possibly joined) method-set
        used for ``processing_status``; it is the fallback for an event's
        ``method`` column only when the event dict omits its own.
    freq_lower, freq_upper : float
        Band bounds.
    stage_key : str
        Joined stage set of the run (used for ``processing_status``).
    events : list of dict
        Normalized event dicts with keys ``uuid``, ``start_time``, ``end_time``,
        ``duration``, ``stage`` (the run's canonical joint stage token, the
        same value hashed into the event's uuid5 -- see
        :func:`join_stage_token`), the optional ``epoch_stage`` (the event's
        own scored epoch stage, stored in the additive ``epoch_stage`` column
        when the database has it), ``method`` (the per-event detecting method,
        stored in the ``method`` column so it agrees with the event's uuid5 and
        the events UNIQUE constraint), plus ``det_trough`` / ``det_peak`` /
        ``det_ptp`` / ``det_trough_time`` / ``det_peak_time``.
    batched : list of dict
        Output of :func:`compute_batched_params`, aligned to ``events``.
    recording_start_time : datetime.datetime or None
        For ``start_time_hms`` formatting.
    n_fft_sec : float or None
        Stored in the ``n_fft_sec`` column for provenance.
    logger : logging.Logger or None
        Optional logger.
    replace : bool, optional
        When True, delete this channel's existing rows for the run scope before
        inserting (scoped channel re-detection). Default False, which is the P2
        append/upsert behaviour (no delete). See the class-level note above.
    replace_methods : sequence of str or None, optional
        The constituent per-event methods to delete when ``replace`` is True.
        Pass the run's method *list* (e.g. ``['Wamsley2012']`` or
        ``['AASM/Massimini2004']``), NOT the ``'_'``-joined ``method`` string,
        because events store their per-event method and a joined string would
        never match them (leaving stale rows). Defaults to ``[method]`` when
        ``None``. Ignored unless ``replace`` is True.

    Returns
    -------
    int
        Number of event rows written.
    """
    now = datetime.datetime.now().isoformat()
    # Same helper the detectors use for the JSON filename token, so this is
    # the last place the two spellings could drift apart.
    freq_band = fmt_freq_token(freq_lower, freq_upper)
    # epoch_stage is presence-gated rather than assumed: it is additive and a
    # database whose schema was never ensured (a direct call on a legacy file)
    # must still take the write, minus the column it does not have.
    insert_columns = list(EVENT_INSERT_COLUMNS)
    has_epoch_stage = _EPOCH_STAGE_COLUMN[0] in _table_columns(conn, 'events')
    if has_epoch_stage:
        insert_columns.append(_EPOCH_STAGE_COLUMN[0])
    placeholders = ', '.join(['?'] * len(insert_columns))
    sql = (f"INSERT OR REPLACE INTO events ({', '.join(insert_columns)}) "
           f"VALUES ({placeholders})")

    conn.execute('BEGIN')
    try:
        if replace:
            # Scoped DELETE-then-INSERT for channel re-detection. Delete by the
            # constituent per-event methods so a multi-method run clears every
            # method's rows (the joined method_str never matches a stored
            # per-event method). NOT stage-scoped; freq matched NULL-safe.
            del_methods = [str(m) for m in (replace_methods or [method])]
            m_placeholders = ', '.join(['?'] * len(del_methods))
            del_sql = (
                f"DELETE FROM events WHERE event_type = ? AND channel = ? "
                f"AND freq_lower IS ? AND freq_upper IS ? "
                f"AND method IN ({m_placeholders})")
            cur = conn.execute(
                del_sql,
                [event_type, channel, freq_lower, freq_upper] + del_methods)
            if logger is not None:
                logger.info(
                    f"Scoped replace: deleted {cur.rowcount} existing "
                    f"{event_type} rows for channel {channel} "
                    f"(methods={del_methods}, band={freq_lower}-{freq_upper}Hz) "
                    f"before re-insert")

        for i, ev in enumerate(events):
            b = batched[i] if i < len(batched) else {}
            # Store the PER-EVENT detecting method (the same value hashed into
            # the event's uuid5), NOT the run's joined method-set. Otherwise two
            # methods detecting an event at the same start_time/stage in a
            # channel get distinct uuid5s but share method_str, colliding on the
            # events UNIQUE constraint and silently dropping one row.
            ev_method = ev.get('method', method)
            row = (
                ev['uuid'], event_type, channel,
                ev['start_time'], ev['end_time'], ev['duration'],
                sec_to_hms(ev['start_time'], recording_start_time),
                ev['stage'], None, ev_method,
                freq_band, freq_lower, freq_upper,
                b.get('min_amp'), b.get('max_amp'), b.get('peak2peak_amp'),
                b.get('rms'), b.get('power'), b.get('peak_power_freq'),
                b.get('energy'), b.get('peak_energy_freq'),
                ev.get('det_trough'), ev.get('det_peak'), ev.get('det_ptp'),
                ev.get('det_trough_time'), ev.get('det_peak_time'),
                now, n_fft_sec, run_id,
            )
            if has_epoch_stage:
                # The event's OWN scored epoch stage, kept beside the run's
                # joint token in `stage`. None when no scored epoch contains
                # the event (the detectors count and report those).
                row = row + (ev.get('epoch_stage'),)
            conn.execute(sql, row)

        upsert_processing_status(
            conn, event_type, channel, method, freq_lower, freq_upper,
            stage_key, success=True,
            error_message=None if events else 'No events detected')
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(events)


def record_channel_failure(conn, event_type, channel, method, freq_lower,
                           freq_upper, stage_key, error):
    """Roll back and record a failed channel in ``processing_status``.

    Called from a processor's per-channel ``except`` block when ``write_db`` is
    on, in place of writing an error-sentinel JSON.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection.
    event_type, channel, method : str
        Scope identity.
    freq_lower, freq_upper : float
        Band bounds.
    stage_key : str
        Joined stage set of the run.
    error : str
        Failure detail.
    """
    try:
        conn.rollback()
    except Exception:
        pass
    try:
        upsert_processing_status(
            conn, event_type, channel, method, freq_lower, freq_upper,
            stage_key, success=False, error_message=str(error))
        conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DB -> CSV export (P2, stage 2)
# ---------------------------------------------------------------------------

# CSV filename prefix per event type. These MATCH the prefixes the example
# driver scripts pass to ``export_*_parameters_to_csv`` / ``import_parameters_
# csv_to_database`` so a DB-exported CSV re-imports through the *existing*
# importer without a naming mismatch. The importer parses ``method`` from
# ``filename.split('_')[2]``, so the prefix must occupy exactly ``parts[0:2]``.
_CSV_PREFIX = {
    'spindle': 'spindle_parameters',
    'slow_wave': 'sw_parameters',
    'k_complex': 'kc_parameters',
}

# Single source of truth for the DB -> CSV column layout. Header order, the
# SELECT column list and per-row emission are ALL driven from this one list so
# they can never drift out of alignment. Each entry is
# ``(csv_header, source)`` where ``source`` is one of:
#
# * ``('seq',)``                 -- synthesised 1-based segment index.
# * ``('const', value)``         -- a literal constant (e.g. Stitches = 0).
# * ``('db', db_column, gated)`` -- value from the ``events`` row; ``gated``
#   columns are only selected when present in the table (older DBs may predate
#   ``det_*``) and export '' when absent. NULL always exports '' (blank cell ->
#   pandas NaN -> importer NULL).
#
# The first block (through 'UUID') reproduces the legacy JSON->CSV header
# byte-for-byte EXCEPT for the added 'Method' column, which carries the truthful
# ``events.method`` (incl. slash-methods) so a re-import never has to fall back
# to the lossy ``filename.split('_')[2]`` parse. The importer reads 'Method' by
# name; legacy JSON-exported CSVs (which lack it) are unaffected. The trailing
# ``det_*`` columns are additive detector-own morphology, ignored by the
# importer.
_EXPORT_COLUMNS = [
    ('Segment index', ('seq',)),
    ('Start time', ('db', 'start_time', False)),
    ('Start time (HH:MM:SS)', ('db', 'start_time_hms', False)),
    ('End time', ('db', 'end_time', False)),
    ('Stitches', ('const', 0)),
    ('Stage', ('db', 'stage', False)),
    ('Cycle', ('db', 'cycle', False)),
    ('Event type', ('db', 'event_type', False)),
    ('Method', ('db', 'method', False)),
    ('Channel', ('db', 'channel', False)),
    ('Duration (s)', ('db', 'duration', False)),
    ('Min. amplitude (uV)', ('db', 'min_amp', False)),
    ('Max. amplitude (uV)', ('db', 'max_amp', False)),
    ('Peak-to-peak amplitude (uV)', ('db', 'peak2peak_amp', False)),
    ('RMS (uV)', ('db', 'rms', False)),
    ('Power (uV^2)', ('db', 'power', False)),
    ('Peak power frequency (Hz)', ('db', 'peak_power_freq', False)),
    ('Energy (uV^2s)', ('db', 'energy', False)),
    ('Peak energy frequency (Hz)', ('db', 'peak_energy_freq', False)),
    ('UUID', ('db', 'uuid', False)),
    ('det_trough (uV)', ('db', 'det_trough', True)),
    ('det_peak (uV)', ('db', 'det_peak', True)),
    ('det_ptp (uV)', ('db', 'det_ptp', True)),
    ('det_trough_time (s)', ('db', 'det_trough_time', True)),
    ('det_peak_time (s)', ('db', 'det_peak_time', True)),
]

# Canonical sleep-stage vocabulary, matching the rest of the codebase
# (eventprocessor / swprocessor density counts). Longest-first so a greedy split
# of a joined scope token is unambiguous (all NREM* start with 'N', 'REM' with
# 'R', 'Wake' with 'W' -- no stage is a prefix of another under this order).
_STAGE_VOCAB = ['NREM1', 'NREM2', 'NREM3', 'Wake', 'REM']

#: Canonical ORDER of the stage vocabulary, used by :func:`join_stage_token`.
#: Distinct from :data:`_STAGE_VOCAB`, whose order exists only to make the
#: greedy decomposition in :func:`split_stage_token` unambiguous. This one is
#: the order a joined token is written in, sleep-depth then REM then Wake, so
#: that one stage SET has exactly one spelling.
_STAGE_CANONICAL_ORDER = ['NREM1', 'NREM2', 'NREM3', 'REM', 'Wake']


def split_stage_token(stage):
    """Normalise a stage scope argument into a list of constituent stages.

    Accepts the three forms the pipeline uses for a run's stage set:

    * a list/tuple (returned as a list of strings),
    * a single stage string (e.g. ``'NREM2'``),
    * a joined scope token (e.g. ``'NREM2NREM3'``, the ``''.join(stages)`` form
      used in filenames and ``processing_status``), which is greedily split
      against :data:`_STAGE_VOCAB`.

    Parameters
    ----------
    stage : list or tuple or str or None
        Stage set in any of the accepted forms. ``None`` returns ``None``.

    Returns
    -------
    list of str or None
        Constituent stage labels, or ``None`` when ``stage`` is ``None``.

    Raises
    ------
    ValueError
        When a string cannot be fully decomposed into known stages (so a
        malformed token fails loudly rather than silently matching nothing).
    """
    if stage is None:
        return None
    if isinstance(stage, (list, tuple)):
        return [str(s) for s in stage]
    s = str(stage)
    if s in _STAGE_VOCAB:
        return [s]
    # Greedy longest-first decomposition of a joined token.
    remaining = s
    out = []
    while remaining:
        for tok in _STAGE_VOCAB:
            if remaining.startswith(tok):
                out.append(tok)
                remaining = remaining[len(tok):]
                break
        else:
            raise ValueError(
                f"Cannot split stage token {s!r} into known stages "
                f"{_STAGE_VOCAB}; pass an explicit list of stages instead.")
    return out


def stage_components(stage):
    """Split a stage token, treating an unrecognised token as one component.

    The forgiving counterpart of :func:`split_stage_token`, for the read side.
    A reader that raises on an unfamiliar label (``'Undefined'``,
    ``'Movement'``, or anything a future scoring vocabulary adds) turns a
    curiosity in one row into a failed query over the whole table, so an
    unsplittable token is returned whole and compared as an opaque unit.

    Parameters
    ----------
    stage : list or tuple or str or None
        Stage set in any of the forms :func:`split_stage_token` accepts.

    Returns
    -------
    list of str
        Constituent stage labels. ``[]`` for ``None``; ``[stage]`` when the
        string cannot be decomposed into the known vocabulary.

    Examples
    --------
    >>> stage_components('NREM2NREM3')
    ['NREM2', 'NREM3']
    >>> stage_components('Undefined')
    ['Undefined']
    """
    if stage is None:
        return []
    try:
        return list(split_stage_token(stage) or [])
    except ValueError:
        return [str(stage)]


def join_stage_token(stages):
    """Join a stage set into its single canonical token.

    The one spelling of a run's stage scope, used for ``events.stage``, the
    :func:`event_uuid5` stage argument, ``processing_status.stage`` and the
    filename token. Replaces every raw ``''.join(stage)`` in the codebase.

    **Order is load-bearing.** A caller passing ``['NREM3', 'NREM2']`` under a
    raw join writes ``'NREM3NREM2'``, which never equality-matches the
    ``'NREM2NREM3'`` a differently-ordered caller wrote -- the same scope is
    then two tokens, the table fragments, and every reader that filters on one
    silently misses the other. Sorting into a fixed order removes that failure
    mode entirely: one stage SET has exactly one token.

    Round-trips with :func:`split_stage_token`:
    ``split_stage_token(join_stage_token(s))`` is ``s`` deduplicated and
    reordered.

    Parameters
    ----------
    stages : list or tuple or set or str or None
        Stage set in any form the pipeline uses: a list/tuple/set of labels, a
        single label, or an already-joined token (which is re-split and
        re-joined, so passing one through is idempotent).

    Returns
    -------
    str
        The canonical joined token, e.g. ``'NREM2NREM3'``. ``''`` for ``None``
        or an empty set -- callers that need a scope label for an all-stage run
        supply their own (``'all'``).

    Notes
    -----
    Labels outside the known vocabulary (``'Undefined'``, ``'Movement'``, ...)
    are not an error: they are kept and appended after the known stages in
    alphabetical order, so the token stays deterministic and order-insensitive.
    Such a token is not splittable by :func:`split_stage_token`, exactly as
    before this function existed.

    Examples
    --------
    >>> join_stage_token(['NREM3', 'NREM2'])
    'NREM2NREM3'
    >>> join_stage_token(['NREM2', 'NREM3']) == join_stage_token(['NREM3', 'NREM2'])
    True
    >>> join_stage_token('NREM2NREM3')
    'NREM2NREM3'
    >>> join_stage_token(None)
    ''
    """
    if stages is None:
        return ''
    if isinstance(stages, (list, tuple, set, frozenset)):
        parts = []
        for s in stages:
            parts.extend(stage_components(s))
    else:
        parts = stage_components(stages)

    seen = set()
    unique = []
    for p in (str(x) for x in parts):
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    known = [s for s in _STAGE_CANONICAL_ORDER if s in seen]
    unknown = sorted(p for p in unique if p not in _STAGE_CANONICAL_ORDER)
    return "".join(known + unknown)


def stage_tokens_covering(tokens, requested):
    """Select the stored stage tokens that fall inside a requested stage set.

    The single read-side primitive of the joint-token scheme, and what keeps
    every reader working against **both** database shapes at once:

    * a database written per-epoch (before 4.3) stores one-component tokens
      (``'NREM2'``, ``'NREM3'``), and a request for ``['NREM2', 'NREM3']``
      covers both;
    * a database written jointly stores ``'NREM2NREM3'``, and the same request
      covers that one token.

    A request for ``['NREM2']`` alone covers the first database's ``'NREM2'``
    rows but **not** the second's ``'NREM2NREM3'`` rows -- correctly, because a
    joint row cannot be attributed to one of its components. That is a missing
    answer, not a wrong one, and callers should report it as such rather than
    dividing a joint count by an N2-only denominator.

    Parameters
    ----------
    tokens : iterable of str or None
        Stage tokens present in the data (typically
        ``SELECT DISTINCT stage FROM events``). ``None`` entries are dropped;
        the caller handles NULL-stage rows separately.
    requested : list or tuple or str or None
        The stage set asked for, in any accepted form. ``None`` means "no
        restriction" and returns every token unchanged.

    Returns
    -------
    list of str
        The covered tokens, de-duplicated, in the order first seen.

    Examples
    --------
    >>> stage_tokens_covering(['NREM2', 'NREM3', 'REM'], ['NREM2', 'NREM3'])
    ['NREM2', 'NREM3']
    >>> stage_tokens_covering(['NREM2NREM3'], ['NREM2', 'NREM3'])
    ['NREM2NREM3']
    >>> stage_tokens_covering(['NREM2NREM3'], ['NREM2'])
    []
    """
    present = []
    seen = set()
    for tok in (tokens or []):
        if tok is None:
            continue
        tok = str(tok)
        if tok and tok not in seen:
            seen.add(tok)
            present.append(tok)
    if requested is None:
        return present
    wanted = set(stage_components(requested) if not isinstance(
        requested, (list, tuple, set, frozenset))
        else [c for s in requested for c in stage_components(s)])
    if not wanted:
        return present
    return [tok for tok in present
            if set(stage_components(tok)).issubset(wanted)]


def resolve_stage_tokens(conn, requested, where=None, params=None,
                         table='events', column='stage'):
    """Resolve a requested stage set to the tokens actually stored.

    The query-building half of :func:`stage_tokens_covering`: reads the
    distinct stage tokens a scope holds and returns the ones a caller should
    put in ``stage IN (...)``. Use it in place of interpolating the requested
    stage list directly -- that list matches a per-epoch database and misses
    every joint-token row, which is a silent zero-result, not an error.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    requested : list or tuple or str or None
        Stage set asked for, in any accepted form. ``None`` returns every
        stored token (no restriction).
    where : list of str or None, optional
        Additional SQL predicates (``'event_type = ?'`` ...) narrowing which
        rows' tokens are considered. Default ``None``.
    params : list or None, optional
        Bound parameters for ``where``. Default ``None``.
    table, column : str, optional
        Table and column to read tokens from. Defaults ``'events'`` /
        ``'stage'``. **Interpolated into the SQL, so never pass user input.**

    Returns
    -------
    list of str
        Tokens to filter on. An **empty list means "no stored token is inside
        the requested set"**, which callers must render as a matched-nothing
        filter (``1 = 0``), never as "no filter" -- dropping the predicate
        would return the whole table.

    Examples
    --------
    >>> resolve_stage_tokens(conn, ['NREM2', 'NREM3'],      # doctest: +SKIP
    ...                      where=['event_type = ?'], params=['spindle'])
    ['NREM2NREM3']
    """
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table}{clause}",
            list(params or [])).fetchall()
    except sqlite3.OperationalError:
        return [] if requested is None else stage_tokens_covering([], requested)
    return stage_tokens_covering([r[0] for r in rows], requested)


#: Result of :func:`pooled_denominator`.
#:
#: ``analysed_seconds`` and ``artefact_seconds_excluded`` are ``nan`` when
#: ``missing`` is non-empty, so an incomplete denominator can never be divided
#: into a count by accident.
PooledDenominator = namedtuple(
    'PooledDenominator',
    'analysed_seconds artefact_seconds_excluded source missing components')


def pooled_denominator(token, denom):
    """Sum the per-stage density denominators a stage token spans.

    ``analysed_time`` is keyed per **single** scored stage -- artefact-free
    analysed time is a physical quantity of a stage, whereas a joint token is a
    label for a run's scope. So a joint token's denominator is assembled on
    read, here, by summing its components.

    All-or-nothing by design: if any component has no stored row the pooled
    value is ``nan``, never a partial sum. A partial denominator is worse than
    none, because dividing a full joint count by part of the time inflates
    density silently.

    Parameters
    ----------
    token : str
        Stage token as stored in ``events.stage`` -- one component
        (``'NREM2'``) or several (``'NREM2NREM3'``).
    denom : dict
        ``{stage: {'analysed_seconds': float,
        'artefact_seconds_excluded': float or None, 'source': str}}``, as read
        from ``analysed_time`` for one subject and one pair of rejection
        settings.

    Returns
    -------
    PooledDenominator
        ``(analysed_seconds, artefact_seconds_excluded, source, missing,
        components)``. ``source`` is the ``'+'``-joined distinct sources of the
        contributing rows, or ``'missing'`` when incomplete. ``missing`` lists
        the components with no stored row.

    Examples
    --------
    >>> d = {'NREM2': {'analysed_seconds': 1800.0,
    ...                'artefact_seconds_excluded': 60.0, 'source': 'detection'},
    ...      'NREM3': {'analysed_seconds': 1200.0,
    ...                'artefact_seconds_excluded': 0.0, 'source': 'detection'}}
    >>> pooled_denominator('NREM2NREM3', d).analysed_seconds
    3000.0
    >>> pooled_denominator('NREM2REM', d).missing
    ['REM']
    """
    components = stage_components(token)
    missing = [c for c in components if c not in (denom or {})]
    if not components or missing:
        return PooledDenominator(float('nan'), float('nan'), 'missing',
                                 missing, components)
    seconds = sum(float(denom[c]['analysed_seconds']) for c in components)
    # An artefact total is NaN, not 0, when any component never recorded one:
    # "no artefact time was excluded" and "nobody wrote it down" are different
    # claims and only the first is a zero.
    raw_artefact = [denom[c].get('artefact_seconds_excluded')
                    for c in components]
    artefact = (float('nan') if any(a is None for a in raw_artefact)
                else sum(float(a) for a in raw_artefact))
    source = "+".join(sorted({str(denom[c].get('source')) for c in components}))
    return PooledDenominator(seconds, artefact, source, [], components)


# ---------------------------------------------------------------------------
# db_meta: schema-shape markers
# ---------------------------------------------------------------------------

#: ``db_meta`` key recording how ``events.stage`` is populated in a database.
#: ``'joint'`` = the run's canonical stage token (4.3+); absent or
#: ``'per_epoch'`` = each event's own epoch stage (4.2 and earlier).
STAGE_FORMAT_KEY = 'stage_format'

#: Value written by :func:`ensure_direct_write_schema` for databases this
#: release's detectors populate.
STAGE_FORMAT_JOINT = 'joint'

#: Value a migration stamps (or a reader infers) for pre-4.3 databases.
STAGE_FORMAT_PER_EPOCH = 'per_epoch'

#: ``db_meta`` key recording what ``events.det_ptp`` holds. Before 4.3 every
#: slow-wave method stored Wonambi's ``abs(ev[3] - ev[1])`` on sample INDICES
#: -- a sampling-rate-dependent count, in a column named for microvolts. From
#: 4.3 it is the real peak-to-peak amplitude, ``det_peak - det_trough``.
#:
#: The two ranges OVERLAP (observed: sample counts 102-113 against microvolt
#: values 125-171 on the same recordings), so nothing in the numbers
#: themselves tells a query which it is holding. This marker is that
#: evidence, in the same spirit as :data:`STAGE_FORMAT_KEY`.
PTP_UNITS_KEY = 'det_ptp_units'

#: Value written for databases this release's detectors populate.
PTP_UNITS_MICROVOLTS = 'microvolts'

#: Value a reader infers for a pre-4.3 database (absent marker + rows).
PTP_UNITS_SAMPLES = 'samples'


def ensure_db_meta_schema(conn):
    """Create the ``db_meta`` key/value table if absent.

    Holds markers describing the *shape* of a database rather than its data --
    currently only :data:`STAGE_FORMAT_KEY`. Purely additive.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. Commits, does not close.

    Returns
    -------
    None
    """
    conn.execute('''
    CREATE TABLE IF NOT EXISTS db_meta (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated TEXT
    )''')
    conn.commit()


def get_db_meta(conn, key, default=None):
    """Read one ``db_meta`` value.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    key : str
        Marker name.
    default : object, optional
        Returned when the table or the key is absent. Default ``None``.

    Returns
    -------
    str or object
        The stored value, else ``default``.
    """
    try:
        row = conn.execute("SELECT value FROM db_meta WHERE key = ?",
                           (str(key),)).fetchone()
    except sqlite3.OperationalError:
        return default
    return default if row is None or row[0] is None else str(row[0])


def set_db_meta(conn, key, value):
    """Write one ``db_meta`` value (insert or replace).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (``db_meta`` already created).
    key : str
        Marker name.
    value : str
        Marker value.

    Returns
    -------
    None
    """
    conn.execute(
        "INSERT OR REPLACE INTO db_meta (key, value, updated) VALUES (?, ?, ?)",
        (str(key), str(value), datetime.datetime.now().isoformat()))
    conn.commit()


def stage_format(conn):
    """Report how ``events.stage`` is populated in this database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.

    Returns
    -------
    str or None
        :data:`STAGE_FORMAT_JOINT`, :data:`STAGE_FORMAT_PER_EPOCH`, or ``None``
        when the database carries no marker. ``None`` is *not* the same as
        ``'per_epoch'``: it means the shape was never recorded, which is true
        of every database written before 4.3 **and** of one whose events came
        entirely from the CSV importers.
    """
    return get_db_meta(conn, STAGE_FORMAT_KEY, None)


def ptp_units(conn):
    """Report what ``events.det_ptp`` holds in this database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.

    Returns
    -------
    str or None
        :data:`PTP_UNITS_MICROVOLTS`, or ``None`` when the database carries
        no marker. ``None`` means the units were never recorded, which is
        true of every database written before 4.3: those hold Wonambi's
        sample count (:data:`PTP_UNITS_SAMPLES`) for slow waves and
        K-complexes. A database can also be MIXED -- rows written before and
        after 4.3 under one marker-less history -- which is exactly what the
        marker exists to make visible.

    Notes
    -----
    ``peak2peak_amp`` is unaffected and has always been microvolts: it is
    re-measured from the signal by ``compute_batched_params`` rather than
    taken from the detector.
    """
    return get_db_meta(conn, PTP_UNITS_KEY, None)


def assert_stage_format_compatible(conn, event_type, methods, freq_lower,
                                   freq_upper, *, stage_token,
                                   channels=None, replace_channels=None,
                                   db_path=None, logger=None):
    """Refuse a run that would append a duplicate set instead of replacing.

    :func:`event_uuid5` hashes the stage, and the stage is also the last
    component of the ``event_chan_time`` UNIQUE constraint. So re-detecting a
    scope whose stored rows carry a **different** stage value produces a new
    primary key AND a new unique key: ``INSERT OR REPLACE`` inserts, and the
    run appends a complete duplicate set beside the old one. Every count and
    every density in that scope doubles, and nothing raises.

    Two different situations produce that, and both are checked here:

    1. **A pre-4.3 (per-epoch) database.** Its rows carry ``'NREM2'`` and
       ``'NREM3'``; a 4.3 run over both writes ``'NREM2NREM3'``. Fixed by
       ``examples/migrate_stage_to_joint.py``.
    2. **A joint database re-detected under a DIFFERENT stage set.** Run 1 over
       ``['NREM2']`` stores ``'NREM2'``; run 2 over ``['NREM2', 'NREM3']``
       stores ``'NREM2NREM3'`` and every event of run 1 survives as a
       duplicate. The ``stage_format`` marker cannot see this one -- both runs
       are joint -- so the check is on the token itself, not on the marker.

    A NULL stage counts as "different": such a row was keyed on ``'None'`` and
    would be duplicated in exactly the same way.

    Loud by design, matching :func:`resolve_db_target`: a run that would
    silently double a database is stopped before its first write.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (schema already ensured).
    event_type : str
        Event type about to be written (``'spindle'``, ...).
    methods : str or sequence of str
        The run's constituent per-event method(s), as they are stored in
        ``events.method`` -- **not** the ``'_'``-joined run label, which never
        matches a stored row.
    freq_lower, freq_upper : float or None
        Band bounds of the run. Matched NULL-safe.
    stage_token : str, keyword-only, REQUIRED
        The canonical stage token this run will write
        (:func:`join_stage_token`), or ``'all'`` for an unscoped run.
        Keyword-only and without a default **on purpose**: it is the whole of
        check (2), and a caller that omitted it would keep only the
        marker-based check (1) and lose duplicate protection against a
        same-format, different-stage-set re-run -- silently, since the
        remaining check still runs and still passes. A missing or ``None``
        value therefore fails loudly (``TypeError`` / ``ValueError``) rather
        than degrading the guard.
    channels : sequence of str or None, optional
        Channels this run will write. ``None`` (default) checks every channel
        in the scope, which is the conservative reading.
    replace_channels : sequence of str or None, optional
        Channels whose existing rows this run deletes first (scoped
        re-detection). Their rows cannot duplicate, so they are excluded from
        the check. Default ``None``.
    db_path : str or None, optional
        Path named in the error message. Default ``None``.
    logger : logging.Logger or None, optional
        Logger for the one-line all-clear. Default ``None``.

    Returns
    -------
    int
        Number of at-risk rows found (always 0 when this returns).

    Raises
    ------
    TypeError
        If ``stage_token`` is not passed at all.
    ValueError
        If ``stage_token`` is ``None``, or if the scope about to be written
        already holds rows under a different stage token, or if the database
        is unmarked / marked ``'per_epoch'`` and the scope already holds rows.
    """
    if stage_token is None:
        raise ValueError(
            "assert_stage_format_compatible needs the stage token this run "
            "will write (dbwrite.join_stage_token(stage), or 'all' for an "
            "unscoped run). Passing None would skip the check that a "
            "re-detection under a DIFFERENT stage set appends a duplicate "
            "set instead of replacing, and that failure is silent.")
    if not _table_columns(conn, 'events'):
        return 0

    def _as_list(value):
        """One-or-many to a list, without iterating a bare string's letters."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(v) for v in value]
        return [str(value)]

    method_list = [m for m in _as_list(methods) if m.strip()]
    if not method_list:
        # An empty list renders `method IN ()`, which SQLite accepts and
        # evaluates as always-false: the guard would find nothing, return 0
        # and let the write proceed unchecked. Exactly the silent downgrade
        # stage_token is keyword-required to prevent, so it fails the same way.
        raise ValueError(
            f"assert_stage_format_compatible needs at least one non-blank "
            f"method (got {methods!r}). An empty method list would render "
            f"'method IN ()', which SQLite treats as always-false, so the "
            f"guard would match no rows and silently allow a write that "
            f"appends a duplicate set. Pass the run's constituent per-event "
            f"method(s), as stored in events.method.")
    at_risk = set()
    if channels is not None:
        at_risk = set(_as_list(channels)) - set(_as_list(replace_channels))
        if not at_risk:
            return 0

    where = ["event_type = ?",
             "method IN (%s)" % ",".join("?" * len(method_list)),
             "freq_lower IS ?", "freq_upper IS ?"]
    params = [str(event_type)] + method_list + [
        None if freq_lower is None else float(freq_lower),
        None if freq_upper is None else float(freq_upper)]
    if at_risk:
        chans = sorted(at_risk)
        where.append("channel IN (%s)" % ",".join("?" * len(chans)))
        params.extend(chans)
    elif replace_channels:
        reps = sorted(set(_as_list(replace_channels)))
        where.append("channel NOT IN (%s)" % ",".join("?" * len(reps)))
        params.extend(reps)
    clause = " AND ".join(where)

    scope = (f"{event_type} / method(s) {method_list} / "
             f"{fmt_freq_token(freq_lower, freq_upper)}")
    how_to_proceed = (
        "To proceed: re-detect with replace_channels=<the channels above> so "
        "the stale rows are DELETEd in the same transaction; or delete that "
        "scope's rows first; or write this run to a different database. "
        "Keeping both is not an option -- they overlap, so every count and "
        "density over this scope would be inflated.")

    # --- check 2: a different stage token in the same scope ---------------
    # Unconditional, and independent of the marker: both sides can be 'joint',
    # so the marker cannot see this case. stage_token is keyword-required
    # precisely so this check cannot be skipped by omission.
    tok_where = clause + " AND (stage IS NULL OR stage != ?)"
    tok_params = params + [str(stage_token)]
    n_rows, n_chans = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT channel) FROM events WHERE "
        + tok_where, tok_params).fetchone()
    n_rows = int(n_rows or 0)
    if n_rows:
        raw = [r[0] for r in conn.execute(
            "SELECT DISTINCT stage FROM events WHERE " + tok_where,
            tok_params)]
        has_null = any(r is None for r in raw)
        named = sorted(str(r) for r in raw if r is not None)
        stored = (named + ['NULL (no stage at all)']) if has_null else named
        chans_hit = [str(r[0]) for r in conn.execute(
            "SELECT DISTINCT channel FROM events WHERE " + tok_where
            + " ORDER BY channel LIMIT 10", tok_params)]
        # The advice differs by what is actually stored, so that it is
        # applicable rather than merely reassuring. A NULL stage is not a
        # stale per-epoch label the migration can collapse by inference: the
        # row records no stage at all.
        advice = []
        if named:
            advice.append(
                "The named token(s) are a different stage SET than this run's "
                "-- either a pre-4.3 per-epoch stage, or an earlier run over "
                "a different stage selection. "
                "examples/migrate_stage_to_joint.py rewrites a pre-4.3 "
                "per-epoch stage to the run's token; for an earlier run over "
                "a genuinely different stage set, no migration can merge the "
                "two and you must choose one.")
        if has_null:
            advice.append(
                "The NULL-stage row(s) carry no stage at all -- a 4.2 event "
                "whose scored epoch could not be resolved, or a CSV import "
                "with no Stage column. Nothing can infer their stage set from "
                "the row itself. examples/migrate_stage_to_joint.py labels "
                "them with the run's own recorded stage token (which is what "
                "events.stage means from 4.3: the run's scope, with the "
                "per-epoch uncertainty kept in events.epoch_stage) when that "
                "token is on record for their channel; otherwise re-detect "
                "those channels with replace_channels, or delete them.")
        raise ValueError(
            f"{db_path or 'This database'} already holds {n_rows} "
            f"{scope} row(s) across {int(n_chans or 0)} channel(s) under "
            f"stage token(s) {stored}, and this run would write "
            f"'{stage_token}'. The stage is part of both the event uuid5 "
            f"and the event_chan_time UNIQUE constraint, so those rows "
            f"would NOT be replaced: INSERT OR REPLACE would APPEND A "
            f"COMPLETE DUPLICATE SET beside them. Channels affected "
            f"(first 10): {chans_hit}. {how_to_proceed} "
            + " ".join(advice))

    # --- check 1: a pre-4.3 database, even when the tokens agree ----------
    fmt = stage_format(conn)
    if fmt == STAGE_FORMAT_JOINT:
        return 0

    n_rows, n_chans = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT channel) FROM events WHERE " + clause,
        params).fetchone()
    n_rows = int(n_rows or 0)
    if not n_rows:
        if logger is not None:
            logger.debug(
                "Stage-format check passed: this database carries no "
                "'%s' marker but holds no %s rows, so a joint-token write "
                "cannot duplicate anything.", STAGE_FORMAT_KEY, scope)
        return 0

    tokens = sorted({str(r[0]) for r in conn.execute(
        "SELECT DISTINCT stage FROM events WHERE " + clause, params)
        if r[0] is not None})
    raise ValueError(
        f"{db_path or 'This database'} was written before the joint stage "
        f"token (db_meta.{STAGE_FORMAT_KEY} is {fmt!r}) and already holds "
        f"{n_rows} {scope} row(s) across {int(n_chans or 0)} channel(s), "
        f"stored under stage token(s) {tokens}. Those rows carry uuids "
        f"computed under the old convention, so a re-detection cannot be "
        f"proven to replace rather than duplicate them. Convert the database "
        f"first with examples/migrate_stage_to_joint.py -- it rewrites the "
        f"stored stages to the run's token where needed and stamps "
        f"db_meta.{STAGE_FORMAT_KEY}='{STAGE_FORMAT_JOINT}', and is a "
        f"near-no-op on a database whose rows already carry that token. "
        f"{how_to_proceed}")


def _fmt_freq_component(value):
    """Format one band bound for a CSV filename (integral floats drop ``.0``).

    Parameters
    ----------
    value : float or int
        Band bound in Hz.

    Returns
    -------
    str
        ``'11'`` for ``11.0``, ``'0.5'`` for ``0.5``.
    """
    v = float(value)
    return str(int(v)) if v.is_integer() else repr(v)


def fmt_freq_token(lo, hi):
    """Format a detection band as the ``{freq_lo}-{freq_hi}Hz`` filename token.

    This is the single source of truth for the frequency component of the
    project naming convention
    ``{event_type}_{method}_{freq_lo}-{freq_hi}Hz_{stages_joined}``. It must
    be used on *both* sides of the filename round-trip: where a detector
    writes its per-channel JSON, and where a caller rebuilds the
    ``file_pattern`` to find those files again.

    It is deliberately the plain historical expression ``f"{lo}-{hi}Hz"`` and
    applies no normalisation, so existing result directories keep matching.
    The bug it fixes was never the format but the *divergence*: a driver that
    re-derived the token with a different formatter (``f"{lo:.1f}"``, which
    turns a 1.25 Hz bound into ``1.2``) matched zero files and produced an
    empty run that still reported success. Feeding one function the same
    ``frequency`` tuple the detector used removes that failure mode whatever
    the format is.

    Parameters
    ----------
    lo : float or int
        Lower band bound in Hz.
    hi : float or int
        Upper band bound in Hz.

    Returns
    -------
    str
        The band token, e.g. ``'0.5-1.25Hz'``, ``'11-16Hz'``, ``'9.0-12.0Hz'``.

    Examples
    --------
    >>> fmt_freq_token(0.5, 1.25)
    '0.5-1.25Hz'
    >>> fmt_freq_token(11, 16)
    '11-16Hz'
    >>> fmt_freq_token(9.0, 12.0)
    '9.0-12.0Hz'
    """
    return f"{lo}-{hi}Hz"


def default_csv_path(output_dir, event_type, method, frequency, stage):
    """Build the standard parameter-CSV path for a detection scope.

    Reproduces the exact filename the example driver scripts pass to the legacy
    exporter, so the file re-imports through
    ``import_parameters_csv_to_database`` unchanged:
    ``{prefix}_{method}_{lo}-{hi}Hz_{stages_joined}.csv``.

    Parameters
    ----------
    output_dir : str
        Directory to place the CSV in.
    event_type : str
        Event type (``'spindle'`` / ``'slow_wave'`` / ``'k_complex'``); selects
        the filename prefix.
    method : str
        Detection method (``'/'`` is replaced with ``'_'`` for filesystem
        safety, matching the driver scripts).
    frequency : tuple of float
        Detection band ``(lo, hi)`` in Hz.
    stage : list of str or str
        Stage set of the run; joined with no separator (``['NREM2','NREM3']``
        -> ``'NREM2NREM3'``).

    Returns
    -------
    str
        Absolute or relative path following the naming convention.
    """
    prefix = _CSV_PREFIX.get(str(event_type), f"{event_type}_parameters")
    method_token = str(method).replace('/', '_')
    lo, hi = frequency
    # NOTE: this deliberately keeps _fmt_freq_component's normalisation
    # (9.0 -> '9'), which DIVERGES from fmt_freq_token ('9.0-12.0Hz'). The
    # divergence predates the Stage A work and is left alone here because
    # changing it would rename CSVs users already have on disk. Closing it is
    # scheduled for Stage B, when the file round-trip goes away entirely.
    freq_token = f"{_fmt_freq_component(lo)}-{_fmt_freq_component(hi)}Hz"
    # Normalise any accepted stage form (list, single, or joined token) to the
    # canonical joined token, so the filename is identical whichever form the
    # caller passes AND identical to the token the same run stored in
    # events.stage -- one function, so the filename and the database cannot
    # drift into two spellings of one scope.
    stage_token = join_stage_token(stage)
    fname = f"{prefix}_{method_token}_{freq_token}_{stage_token}.csv"
    return os.path.join(output_dir, fname)


def export_events_to_csv(db_path, event_type, method, frequency, stage,
                         csv_file=None, output_dir=None, append=False,
                         logger=None):
    """Export events for one detection scope from the DB to a parameter CSV.

    On-demand DB -> CSV export for the direct-write path (``write_db=True``).
    SELECTs the rows for one run scope (``event_type`` / ``method`` / band /
    ``stage`` set) and writes them in the SAME shape and column names the legacy
    JSON -> CSV exporters produce, so downstream stats keep a flat-file option
    and the file round-trips back through
    ``import_parameters_csv_to_database`` with ``added = 0`` when the DB was
    populated by the direct path.

    The legacy amplitude/spectral columns (``Min. amplitude (uV)`` ...
    ``Peak energy frequency (Hz)``) carry the **re-measured** values
    (``events.min_amp`` etc., computed once per channel by
    :func:`compute_batched_params`) -- the same quantity the legacy exporter's
    ``event_params`` re-read produced, so column semantics are unchanged. The
    detector's OWN morphology is exported additionally in the trailing
    ``det_*`` columns; those are ignored by the importer and never overwrite the
    legacy columns.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database populated by the direct-write path.
    event_type : str
        Event type to export (``'spindle'`` / ``'slow_wave'`` / ``'k_complex'``).
    method : str
        Detection method of the run scope. Matched against ``events.method``
        (the per-event method stored by the direct path). ``'/'`` is preserved
        for the DB match but replaced with ``'_'`` in the default filename.
    frequency : tuple of float or None
        Band ``(lo, hi)`` in Hz used to filter and to name the file. When
        ``None`` the band filter is skipped and the band is inferred from the
        matched rows for the filename.
    stage : list of str or str or None
        Stage set of the run, used both to filter and to name the file.
        Accepts a list (``['NREM2','NREM3']``), a single stage (``'NREM2'``)
        or the joined scope token (``'NREM2NREM3'``); all three are equivalent.
        The filter is resolved through :func:`resolve_stage_tokens`, so it
        matches a joint-token database and a per-epoch one alike. A request
        for a strict subset of a stored joint token matches nothing and
        raises, rather than exporting a slice that does not exist. When
        ``None`` no stage filter is applied.
    csv_file : str or None
        Explicit output path. When ``None`` a path is built with
        :func:`default_csv_path` under ``output_dir``.
    output_dir : str or None
        Directory for the default filename when ``csv_file`` is ``None``.
        Defaults to the directory of ``db_path``.
    append : bool
        When ``True`` and ``csv_file`` already exists, append data rows without
        rewriting the header (for accumulating multiple scopes into one file).
        Default ``False`` (overwrite).
    logger : logging.Logger or None
        Optional logger for progress/warnings.

    Returns
    -------
    str or None
        Path to the written CSV, or ``None`` when the scope is genuinely empty
        (no events of this type/method/band exist at all). No file is written in
        that case.

    Raises
    ------
    ValueError
        When the stage filter excludes every row but the same type/method/band
        DOES have events under other stages -- i.e. a stage-token mismatch,
        surfaced loudly instead of silently writing nothing. Also propagates
        :func:`split_stage_token`'s error for a malformed stage token.

    Notes
    -----
    This is an ADDITION to, not a replacement for, the JSON-based
    ``export_*_parameters_to_csv`` methods, which remain the path for
    ``write_db=False`` runs. A density export from the DB is intentionally out
    of scope here; density stays on its dedicated path.

    Header order, the SELECT column list and per-row values are all driven from
    the single :data:`_EXPORT_COLUMNS` layout, so they cannot drift apart.
    """
    stage_list = split_stage_token(stage)  # may raise on a malformed token

    # Read-only, 60 s busy timeout. This is the reader most likely to meet a
    # writer: it runs straight after detection, and under DELETE journal mode
    # a writer blocks readers (under WAL it would not).
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        present = _table_columns(conn, 'events')
        if not present:
            if logger is not None:
                logger.warning(f"No events table in {db_path}; nothing to export")
            return None

        # Resolve which DB columns to SELECT from the single-source layout,
        # dropping presence-gated columns an older DB predates.
        db_columns = []
        for _header, source in _EXPORT_COLUMNS:
            if source[0] != 'db':
                continue
            db_col, gated = source[1], source[2]
            if (not gated) or (db_col in present):
                if db_col not in db_columns:
                    db_columns.append(db_col)

        where = ["event_type = ?", "method = ?"]
        params = [str(event_type), str(method)]
        if frequency is not None:
            where.append("freq_lower = ? AND freq_upper = ?")
            params.extend([float(frequency[0]), float(frequency[1])])
        if stage_list is not None:
            # Resolve to the tokens actually stored, so this works against a
            # joint-token database ('NREM2NREM3') and a per-epoch one
            # ('NREM2','NREM3') alike. Interpolating the requested stage list
            # directly matched only the latter and returned nothing -- silently
            # -- against everything written from 4.3 on.
            stage_tokens = resolve_stage_tokens(
                conn, stage_list, where=list(where), params=list(params))
            if stage_tokens:
                placeholders = ", ".join(["?"] * len(stage_tokens))
                where.append(f"stage IN ({placeholders})")
                params.extend(stage_tokens)
            else:
                # No stored token lies inside the requested set. This must
                # match nothing -- dropping the predicate would export the
                # whole scope under a filename claiming a narrower one.
                where.append("1 = 0")

        sql = (f"SELECT {', '.join(db_columns)} FROM events "
               f"WHERE {' AND '.join(where)} ORDER BY channel, start_time")
        col_index = {name: i for i, name in enumerate(db_columns)}
        rows = conn.execute(sql, params).fetchall()

        # Distinguish a genuinely-empty scope (return None) from a stage-token
        # mismatch that would otherwise write nothing silently (raise).
        if not rows and stage_list is not None:
            noscope_where = ["event_type = ?", "method = ?"]
            noscope_params = [str(event_type), str(method)]
            if frequency is not None:
                noscope_where.append("freq_lower = ? AND freq_upper = ?")
                noscope_params.extend([float(frequency[0]), float(frequency[1])])
            n_without_stage = conn.execute(
                f"SELECT COUNT(*) FROM events WHERE {' AND '.join(noscope_where)}",
                noscope_params).fetchone()[0]
            if n_without_stage > 0:
                avail = [r[0] for r in conn.execute(
                    f"SELECT DISTINCT stage FROM events "
                    f"WHERE {' AND '.join(noscope_where)}", noscope_params)]
                raise ValueError(
                    f"Stage filter {stage_list} matched 0 of {n_without_stage} "
                    f"{event_type}/{method} events; the stage token(s) stored "
                    f"in this scope are {sorted(str(s) for s in avail)}. A "
                    f"stored token is only exported when ALL of its stages "
                    f"were asked for, because a joint token like 'NREM2NREM3' "
                    f"labels a run that searched both stages as one segment "
                    f"and its events cannot be attributed to one of them. Ask "
                    f"for the full set (a list is accepted).")
    finally:
        conn.close()

    if not rows:
        if logger is not None:
            logger.info(
                f"No {event_type} rows for method={method}, freq={frequency}, "
                f"stage={stage} in {db_path}; scope is empty, no CSV written")
        return None

    # Resolve the output path.
    if csv_file is None:
        out_dir = output_dir or os.path.dirname(os.path.abspath(db_path))
        freq_for_name = frequency
        if freq_for_name is None:
            # Infer band from the first row for the filename.
            fl = rows[0][col_index['freq_lower']] if 'freq_lower' in col_index else None
            fu = rows[0][col_index['freq_upper']] if 'freq_upper' in col_index else None
            freq_for_name = (fl if fl is not None else 0,
                             fu if fu is not None else 0)
        csv_file = default_csv_path(out_dir, event_type, method,
                                    freq_for_name, stage_list)

    csv_dir = os.path.dirname(csv_file)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    write_header = not (append and os.path.exists(csv_file))
    mode = 'a' if (append and os.path.exists(csv_file)) else 'w'

    def _value(source, row, seq):
        """Resolve one column's value from its single-source spec."""
        kind = source[0]
        if kind == 'seq':
            return seq
        if kind == 'const':
            return source[1]
        # ('db', db_col, gated): '' for absent (old DB) or NULL.
        db_col = source[1]
        if db_col not in col_index:
            return ''
        val = row[col_index[db_col]]
        return '' if val is None else val

    header = [h for h, _ in _EXPORT_COLUMNS]
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            # Single provenance prefix line, mirroring the legacy 'Wonambi vX'
            # line. Skipped by the importer (which locates the header by the
            # 'Start time' cell) and must not itself contain 'Start time'.
            prov = provenance()
            writer.writerow([
                f"turtlewave_hdEEG DB export v{prov['turtlewave_version']}"])
            writer.writerow(header)
        for i, row in enumerate(rows, start=1):
            writer.writerow([_value(source, row, i)
                             for _header, source in _EXPORT_COLUMNS])

    if logger is not None:
        logger.info(f"Exported {len(rows)} {event_type} rows to {csv_file}")
    return csv_file
