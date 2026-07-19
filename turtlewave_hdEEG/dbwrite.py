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
import uuid
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


def open_write_connection(db_path):
    """Open a single-writer SQLite connection tuned for the qsub-per-subject model.

    WAL journalling plus a long busy timeout let a per-subject writer coexist
    with concurrent readers without ``database is locked`` errors. This does not
    make concurrent *writers* safe; the pipeline runs one writer per subject.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Open connection with ``journal_mode=WAL`` and ``busy_timeout=60000`` ms.
    """
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=60000')
    return conn


def _table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


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
                         recording_start_time, n_fft_sec, logger=None):
    """Write one channel's events + status in a single transaction.

    Opens an explicit transaction, ``INSERT OR REPLACE`` s every event row
    (deterministic uuid5, detector-own ``det_*`` morphology, batched re-measured
    and spectral columns) and upserts the channel's ``processing_status`` to
    ``success = 1``, then commits. An empty channel still commits a status row
    with ``error_message = 'No events detected'`` to preserve the
    empty-vs-failed distinction. On any error the transaction is rolled back and
    the exception re-raised for the caller to record as a failure.

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

    Returns
    -------
    int
        Number of event rows written.
    """
    now = datetime.datetime.now().isoformat()
    freq_band = f"{freq_lower}-{freq_upper}Hz"
    placeholders = ', '.join(['?'] * len(EVENT_INSERT_COLUMNS))
    sql = (f"INSERT OR REPLACE INTO events ({', '.join(EVENT_INSERT_COLUMNS)}) "
           f"VALUES ({placeholders})")

    conn.execute('BEGIN')
    try:
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
