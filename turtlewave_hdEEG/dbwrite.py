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
    ('det_ptp', 'REAL'),          # detector's own peak-to-peak amplitude (uV)
    ('det_trough_time', 'REAL'),  # time of detector trough (s from rec start)
    ('det_peak_time', 'REAL'),    # time of detector peak (s from rec start)
)

# ``run_id`` foreign-key-ish link from an event to its detection_runs row.
_RUN_ID_COLUMN = ('run_id', 'TEXT')

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
        Single resolved sleep stage of the event.

    Returns
    -------
    str
        String form of the uuid5.
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
# rejected loudly rather than silently leaving the database in WAL. Public so
# CLIs (examples/set_db_journal_mode.py) can validate up front against the same
# list instead of keeping their own copy.
VALID_JOURNAL_MODES = ('DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF')


def _resolve_journal_mode(requested=None):
    """Resolve an explicitly *requested* journal mode, or ``None`` if unstated.

    Precedence is the explicit argument, then the ``TURTLEWAVE_SQLITE_JOURNAL``
    environment variable. There is deliberately no default: ``None`` means "the
    caller expressed no preference", which lets
    :func:`open_write_connection` preserve whatever mode an existing database is
    already in instead of imposing one on it.

    Parameters
    ----------
    requested : str or None, optional
        Explicit mode. When ``None`` the environment variable is consulted.

    Returns
    -------
    str or None
        Upper-case journal mode (one of :data:`VALID_JOURNAL_MODES`), or
        ``None`` when neither the argument nor the environment variable was set.

    Raises
    ------
    ValueError
        If a value *was* given but is not a recognised SQLite journal mode.
        This is deliberately fatal: a typo that fell through to a default would
        leave the user in WAL believing they had left it.
    """
    mode = requested
    if mode is None:
        mode = os.environ.get(_JOURNAL_ENV)
    if mode is None:
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
        f"examples/set_db_journal_mode.py over the tree. If that write also "
        f"fails, copy the database WITH its -wal/-shm sidecars to local disk, "
        f"convert it there and copy it back. See "
        f"docs/how-to/run-with-database-on-a-network-drive.md.")
    # Chained by hand: 'raise X from Y' is only valid on a raise statement, and
    # this helper returns the exception for the caller to raise.
    explained.__cause__ = exc
    return explained


def open_write_connection(db_path, journal=None, logger=None):
    """Open a connection that preserves an existing database's journal mode.

    Only a database this call *creates* is set to ``WAL``; an existing
    database keeps whatever mode it is already in, unless ``journal`` or
    ``TURTLEWAVE_SQLITE_JOURNAL`` explicitly names one. Every connection also
    gets a 60 s busy timeout, which is what lets a per-subject writer coexist
    with concurrent readers (e.g. the review GUI) without ``database is
    locked`` errors; it does not make concurrent *writers* safe -- the
    pipeline runs one writer per subject.

    See ``docs/explanation/database-concurrency-and-journalling.md`` for why
    WAL was chosen, why journal mode is a persistent on-disk property, and
    why that makes an unconditional default dangerous; see
    ``docs/how-to/run-with-database-on-a-network-drive.md`` for the
    task-oriented fix.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    journal : str or None, optional
        Journal mode to impose. ``None`` (the default) falls back to the
        ``TURTLEWAVE_SQLITE_JOURNAL`` environment variable and, failing that, to
        preserving an existing database's mode / ``'WAL'`` for a new one.
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
    * A database this call creates gets ``'WAL'`` when nothing else is
      requested.
    * Existence is checked with :func:`os.path.exists` *before* connecting
      (connecting would otherwise create the file). A zero-byte placeholder
      file therefore counts as existing and is left in SQLite's reported
      ``delete`` mode rather than promoted to ``WAL``.
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
            # This call is creating the database, so the choice is ours: WAL.
            mode = 'WAL'

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
        ``disk I/O error``) propagates unchanged, so the network-filesystem
        failure is never mis-reported as a lock.

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
        before = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        if before == 'wal':
            row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if row is not None and row[0]:
                log.warning(
                    f"wal_checkpoint(TRUNCATE) was blocked on {db_path} "
                    f"(busy={row[0]}): committed data may still sit in the "
                    f"-wal sidecar. Copying the .db file alone would lose it.")
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

    The ``events`` half cannot be scoped by stage: ``events.stage`` records
    each event's own epoch stage, not the run's joined stage key, so
    filtering on it would discard valid rows. Two things limit the resulting
    blind spot. A channel with an in-scope ``success = 0`` row is excluded
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
        # NOTE: the events half cannot be scoped by stage. events.stage holds
        # the per-epoch stage of each event ('NREM2'), not the run's joined
        # stage key ('NREM2NREM3'), so filtering on it would reject valid
        # rows. Two mitigations below: an in-scope FAILURE always wins over
        # event evidence, and channels credited by events alone are counted
        # and reported.
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


def ensure_direct_write_schema(conn, logger=None):
    """Additively migrate a database for the direct-write path.

    Idempotent and safe on an already-current database. Performs three guarded
    migrations, none of which touch existing rows or unrelated tables:

    1. Add detector-own morphology columns (``det_trough`` etc.) and ``run_id``
       to ``events`` (only the absent ones, via ``PRAGMA table_info``).
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

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection. The caller need not have created the base
        ``events`` table; this only augments it when present.
    logger : logging.Logger or None
        Optional logger for migration messages.
    """
    cur = conn.cursor()

    # (1) events: det_* morphology + run_id -------------------------------
    existing = _table_columns(conn, 'events')
    if existing:  # events table exists (fresh DBs get it via initialize_*)
        added = []
        for col, col_type in _DET_MORPH_COLUMNS + (_RUN_ID_COLUMN,):
            if col not in existing:
                cur.execute(f"ALTER TABLE events ADD COLUMN {col} {col_type}")
                added.append(col)
        if added and logger is not None:
            logger.info(f"Migrated events table: added columns {added}")

    # (2) detection_runs provenance table ---------------------------------
    conn.execute('''
    CREATE TABLE IF NOT EXISTS detection_runs (
        run_id TEXT PRIMARY KEY,
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

    conn.commit()


def record_run(conn, run_id, event_type, method, citation, params_json,
               ref_chan, polar, stages, reject_artifacts, reject_arousals):
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
    """
    prov = provenance()
    conn.execute('''
    INSERT OR REPLACE INTO detection_runs
        (run_id, event_type, method, citation, params_json, ref_chan, polar,
         stages, reject_artifacts, reject_arousals, turtlewave_version,
         wonambi_version, numpy_version, git_sha, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, event_type, method, citation, params_json,
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


def make_param_segment(data, start_time, end_time, event_type, stage,
                       chan, buffer=0.1):
    """Slice an in-memory Data window for one event, for batched measurement.

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
        ``duration``, ``stage`` (single resolved stage), ``method`` (the
        per-event detecting method, stored in the ``method`` column so it agrees
        with the event's uuid5 and the events UNIQUE constraint), plus
        ``det_trough`` / ``det_peak`` / ``det_ptp`` / ``det_trough_time`` /
        ``det_peak_time``.
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
    placeholders = ', '.join(['?'] * len(EVENT_INSERT_COLUMNS))
    sql = (f"INSERT OR REPLACE INTO events ({', '.join(EVENT_INSERT_COLUMNS)}) "
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
    # caller passes.
    stages = split_stage_token(stage)
    stage_token = "".join(stages) if stages else ''
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
        Stage set of the run. Filters ``events.stage IN (...)`` (each row holds
        its single resolved stage) and names the file. Accepts a list
        (``['NREM2','NREM3']``), a single stage (``'NREM2'``) or the joined
        scope token (``'NREM2NREM3'``) used in filenames -- the latter is split
        via :func:`split_stage_token`. When ``None`` no stage filter is applied.
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
            placeholders = ", ".join(["?"] * len(stage_list))
            where.append(f"stage IN ({placeholders})")
            params.extend(stage_list)

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
                    f"{event_type}/{method} events; available stages in this "
                    f"scope are {sorted(str(s) for s in avail)}. Pass a stage "
                    f"set that intersects them (a list is accepted).")
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
