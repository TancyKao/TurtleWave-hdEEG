"""
cycleprocessor.py

Sleep-cycle detection for TurtleWave-hdEEG.

Ports the rule-based NREM-REM cycle detector from the MATLAB
``SpectraDynamic_Analysis`` toolbox (``cal_SleepCycle_HdEEG.m`` and its
Feinberg-1979 variant) to Python, and connects it to TurtleWave's existing
persistence layers:

* the ``events.cycle`` column of ``neural_events.db`` (previously always empty),
* a new ``sleep_cycles`` table holding per-cycle boundaries and durations,
* cycle markers written back into the Wonambi annotation XML so the review GUI
  and ``Annotations.get_cycles()`` can see them.

The detector works purely on the hypnogram (per-epoch sleep stages); no spectral
data is required. Two cycle definitions are supported via ``method``:

``'2022'``
    NREM-based. A cycle is one contiguous NREM period plus the inter-NREM
    (REM) segment that follows it. Short awakenings are absorbed into NREM and
    too-short NREM runs are dropped. Always yields cycles even when REM scoring
    is sparse.
``'1979'``
    Feinberg/Floyd-Feinberg. As above, but a cycle only closes when a
    qualifying REM period follows the NREM block (the first cycle needs REM of
    at least one epoch, later cycles at least ``rem_min`` epochs). NREM periods
    not followed by qualifying REM are merged into the next cycle.
"""

import logging
import os

import numpy as np

# Database writes go through dbwrite.open_write_connection so this module picks
# up the journal-mode override and the busy timeout (it used to call
# sqlite3.connect directly, with neither).
from . import dbwrite


# Numeric hypnogram codes as produced by ``XLAnnotations.get_hypnogram()``:
# Wake=0, NREM1/2/3=1/2/3, REM=4, artefact/movement/undefined=-1.
_NREM_STAGES = (1, 2, 3)
_REM_STAGE = 4

# Coarse scores used by the detection rule (mirrors the MATLAB re-mapping).
_WAKE = 0
_NREM = 2
_ABSORBED_WAKE = 99   # short wake bout absorbed into surrounding NREM
_REM = 4


def _bool_runs(mask):
    """Return ``(start, end_inclusive)`` index pairs for each run of True.

    This is the numpy stand-in for the MATLAB ``bwconncomp`` / ``RunLength``
    calls: it labels maximal contiguous blocks of a boolean array.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # Pad with False on both ends so edge runs are detected by the diff.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_cycles(hypnogram, epoch_length=30, wake_thresh=10, nrem_min=30,
                  method='2022', rem_min=10, epoch_starts=None):
    """Detect NREM-REM sleep cycles from a per-epoch hypnogram.

    Parameters
    ----------
    hypnogram : sequence of int
        Numeric per-epoch stage codes as returned by
        ``CustomAnnotations.get_hypnogram()`` (Wake=0, NREM1/2/3=1/2/3, REM=4,
        artefact/undefined=-1).
    epoch_length : float, optional
        Epoch duration in seconds (default 30).
    wake_thresh : int, optional
        Maximum length, in epochs, of a Wake bout that is absorbed into the
        surrounding NREM instead of breaking the cycle (default 10).
    nrem_min : int, optional
        Minimum length, in epochs, of an NREM run for it to count as an NREM
        period (default 30).
    method : {'2022', '1979'}, optional
        Cycle definition. ``'2022'`` is NREM-based; ``'1979'`` additionally
        requires a qualifying REM period to close each cycle.
    rem_min : int, optional
        Minimum REM run length, in epochs, required to close a cycle under the
        ``'1979'`` method (the first cycle needs only one REM epoch). Ignored
        for ``'2022'`` (default 10).
    epoch_starts : sequence of float, optional
        Start time in seconds of each epoch, same length as ``hypnogram``. If
        omitted, epoch ``i`` is assumed to start at ``i * epoch_length``.

    Returns
    -------
    list of dict
        One dict per cycle, in chronological order, with keys:
        ``cycle_number`` (1-based), ``method``, ``nrem_start_epoch``,
        ``nrem_end_epoch``, ``rem_start_epoch``, ``rem_end_epoch`` (all
        inclusive epoch indices; ``rem_*`` is the inter-NREM segment, which may
        be empty -> ``rem_start_epoch > rem_end_epoch``), ``nrem_start_sec``,
        ``nrem_end_sec``, ``rem_end_sec`` (cycle end in seconds), and the
        durations ``nrem_dur_min`` (full period, N1+N2+N3+absorbed wake),
        ``nrem_n23_dur_min`` (N2+N3 only, within the same period),
        ``rem_dur_min``, ``cycle_dur_min``.

    Notes
    -----
    Artefact/undefined epochs (-1) are treated as Wake for the absorb/break
    logic, matching how the MATLAB detector handles gaps. An empty or all-Wake
    hypnogram returns ``[]``.
    """
    hyp = np.asarray(list(hypnogram), dtype=float)
    n = hyp.size
    if n == 0:
        return []

    if epoch_starts is not None:
        starts = np.asarray(list(epoch_starts), dtype=float)
        if starts.size != n:
            raise ValueError("epoch_starts must match hypnogram length")
    else:
        starts = np.arange(n, dtype=float) * epoch_length

    def epoch_start_sec(i):
        return float(starts[i])

    def epoch_end_sec(i):
        # Use the next epoch's start when available so gaps are respected;
        # fall back to start + epoch_length for the final epoch.
        if i + 1 < n:
            return float(starts[i + 1])
        return float(starts[i] + epoch_length)

    # Step 1: coarse re-map. NREM -> 2, REM -> 4, everything else (Wake and
    # artefact/undefined) -> 0.
    score = np.full(n, _WAKE, dtype=int)
    score[np.isin(hyp, _NREM_STAGES)] = _NREM
    score[hyp == _REM_STAGE] = _REM

    # Step 2: absorb short Wake bouts into the surrounding NREM.
    for s, e in _bool_runs(score == _WAKE):
        if (e - s + 1) <= wake_thresh:
            score[s:e + 1] = _ABSORBED_WAKE

    # Step 3-4: contiguous NREM (real + absorbed wake), dropping short runs.
    nrem_mask = (score == _NREM) | (score == _ABSORBED_WAKE)
    nrem_periods = []
    for s, e in _bool_runs(nrem_mask):
        if (e - s + 1) <= nrem_min:
            continue
        # Step 5: trim leading/trailing absorbed-wake so the NREM period starts
        # and ends on real NREM.
        real = np.where(score[s:e + 1] == _NREM)[0]
        if real.size == 0:
            continue
        nrem_periods.append((s + int(real[0]), s + int(real[-1])))

    if not nrem_periods:
        return []

    # Pair each NREM period with the inter-NREM segment that follows it (up to
    # the next NREM period, or the end of the recording for the last one).
    raw_cycles = []
    for idx, (ns, ne) in enumerate(nrem_periods):
        seg_start = ne + 1
        seg_end = (nrem_periods[idx + 1][0] - 1
                   if idx + 1 < len(nrem_periods) else n - 1)
        raw_cycles.append({'nrem': (ns, ne), 'seg': (seg_start, seg_end)})

    if method == '2022':
        grouped = [[rc] for rc in raw_cycles]
    elif method == '1979':
        # Merge NREM periods until one is followed by a qualifying REM period.
        grouped = []
        pending = []
        for rc in raw_cycles:
            pending.append(rc)
            seg_s, seg_e = rc['seg']
            rem_runs = ([r for r in _bool_runs(score[seg_s:seg_e + 1] == _REM)]
                        if seg_e >= seg_s else [])
            longest_rem = max((e - s + 1 for s, e in rem_runs), default=0)
            need = 1 if not grouped else rem_min
            if longest_rem >= need:
                grouped.append(pending)
                pending = []
        if pending:  # trailing NREM with no qualifying REM -> final cycle
            grouped.append(pending)
    else:
        raise ValueError(f"Unknown method {method!r}; use '2022' or '1979'")

    cycles = []
    for cyc_num, group in enumerate(grouped, start=1):
        ns = group[0]['nrem'][0]
        ne = group[-1]['nrem'][1]
        seg_start, seg_end = group[-1]['seg']
        has_seg = seg_end >= seg_start

        nrem_dur_min = (ne - ns + 1) * epoch_length / 60.0
        # N2+N3 minutes within the period (excludes N1 and absorbed wake). The
        # period boundaries stay defined by N1+N2+N3, matching the MATLAB
        # detector; this is the separate "consolidated NREM" metric (the
        # MATLAB avgN2N3Minute).
        n23_epochs = int(np.count_nonzero(
            (hyp[ns:ne + 1] == 2) | (hyp[ns:ne + 1] == 3)))
        nrem_n23_dur_min = n23_epochs * epoch_length / 60.0
        rem_dur_min = ((seg_end - seg_start + 1) * epoch_length / 60.0
                       if has_seg else 0.0)
        cycle_end_epoch = seg_end if has_seg else ne

        cycles.append({
            'cycle_number': cyc_num,
            'method': method,
            'nrem_start_epoch': ns,
            'nrem_end_epoch': ne,
            'rem_start_epoch': seg_start if has_seg else ne + 1,
            'rem_end_epoch': seg_end,
            'nrem_start_sec': epoch_start_sec(ns),
            'nrem_end_sec': epoch_end_sec(ne),
            'rem_end_sec': epoch_end_sec(cycle_end_epoch),
            'nrem_dur_min': round(nrem_dur_min, 3),
            'nrem_n23_dur_min': round(nrem_n23_dur_min, 3),
            'rem_dur_min': round(rem_dur_min, 3),
            'cycle_dur_min': round(nrem_dur_min + rem_dur_min, 3),
        })

    return cycles


def compute_stage_durations(hypnogram, epoch_length=30):
    """Sum per-epoch sleep-stage minutes from a hypnogram.

    Counts how many epochs fall in each numeric stage code and converts to
    minutes. Mirrors the free-function pattern of :func:`detect_cycles` so it is
    unit-testable without a database.

    Parameters
    ----------
    hypnogram : sequence of int
        Numeric per-epoch stage codes as returned by
        ``CustomAnnotations.get_hypnogram()`` (Wake=0, NREM1/2/3=1/2/3, REM=4,
        artefact/undefined=-1).
    epoch_length : float, optional
        Epoch duration in seconds (default 30).

    Returns
    -------
    dict
        Duration summary with keys ``epoch_length`` (seconds), ``wake_min``,
        ``n1_min``, ``n2_min``, ``n3_min``, ``rem_min``, ``artefact_min`` (all
        non-sleep-stage epochs; see Notes), and ``total_min``. ``total_min`` is
        ``n_epochs * epoch_length / 60`` (the hypnogram span), so the stage
        parts reconcile exactly by construction: ``wake + n1 + n2 + n3 + rem +
        artefact == total``.

    Notes
    -----
    ``artefact_min`` is computed as the remainder ``total - (wake + n1 + n2 +
    n3 + rem)`` rather than by counting a single code. In normal use these are
    the -1 artefact/undefined epochs (``get_hypnogram`` only emits
    ``{0, 1, 2, 3, 4, -1}``), but folding *any* code outside ``{0, 1, 2, 3, 4}``
    into the remainder guarantees the reconciliation invariant can never be
    silently broken by an unexpected code. An empty hypnogram returns all-zero
    durations.
    """
    hyp = np.asarray(list(hypnogram), dtype=float)
    n = hyp.size
    per_epoch_min = epoch_length / 60.0

    def stage_min(code):
        return float(np.count_nonzero(hyp == code)) * per_epoch_min

    wake_min = stage_min(0)
    n1_min = stage_min(1)
    n2_min = stage_min(2)
    n3_min = stage_min(3)
    rem_min = stage_min(_REM_STAGE)
    total_min = float(n) * per_epoch_min
    # Fold every non-sleep-stage epoch (typically code -1, but also any
    # unexpected code) into the remainder so the parts always sum to total.
    artefact_min = total_min - (wake_min + n1_min + n2_min + n3_min + rem_min)

    return {
        'epoch_length': float(epoch_length),
        'wake_min': wake_min,
        'n1_min': n1_min,
        'n2_min': n2_min,
        'n3_min': n3_min,
        'rem_min': rem_min,
        'artefact_min': artefact_min,
        'total_min': total_min,
    }


def _require_existing_db(db_path):
    """Refuse to run a backfill against a database that does not exist.

    Cycle detection and stage-duration accounting are *post-detection* steps:
    they annotate an existing ``neural_events.db``, they never originate one.
    Without this check a mistyped path is silently created as an empty database
    (``sqlite3.connect`` creates the file), the run then dies on
    ``no such table: main.events``, and a stray file is left behind -- on a
    network share, in whatever journal mode the creating call chose.

    Parameters
    ----------
    db_path : str
        Path to the ``neural_events.db`` SQLite database.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``db_path``.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No database at {db_path}. Sleep-cycle backfill annotates an "
            f"existing neural_events.db and never creates one -- run event "
            f"detection first, or correct the path.")


class ParalCycles:
    """Detect sleep cycles and persist them to the DB and annotation XML.

    Follows the ``Paral*`` convention used by the other processors: it takes a
    ``dataset``/``annotations`` pair, owns a ``logging.Logger``, and exposes a
    single :meth:`run` entry point that both backfills existing databases and
    tags new detection runs.
    """

    def __init__(self, dataset=None, annotations=None, subject=None,
                 log_level=logging.INFO, log_file=None):
        """Initialize the ParalCycles object.

        Parameters
        ----------
        dataset : Dataset, optional
            Dataset object (kept for API symmetry; not required for detection).
        annotations : CustomAnnotations
            Annotations wrapper providing ``get_hypnogram`` / ``epochs`` and,
            for marker writing, the Wonambi cycle-marker methods (delegated).
        subject : str, optional
            Subject identifier stored in the ``sleep_cycles`` table.
        log_level : int
            Logging level (e.g. ``logging.INFO``).
        log_file : str or None
            Path to a log file. If None, logs to console only.
        """
        self.dataset = dataset
        self.annotations = annotations
        self.subject = subject
        self.logger = self._setup_logger(log_level, log_file)

    def _setup_logger(self, log_level, log_file=None):
        """Set up a dedicated logger for this processor."""
        logger = logging.getLogger('turtlewave_hdEEG.cycleprocessor')
        logger.setLevel(log_level)

        # Process-wide singleton: clear stale handlers so batch loops don't
        # duplicate lines or leak file handles.
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _epoch_starts(self):
        """Return per-epoch start seconds, or None if unavailable."""
        try:
            epochs = self.annotations.epochs
        except Exception as e:
            self.logger.warning(f"Could not read epochs for timing: {e}")
            return None
        if not epochs:
            return None
        try:
            return [float(ep['start']) for ep in epochs]
        except (KeyError, TypeError, ValueError):
            return None

    def detect(self, method='2022', epoch_length=30, wake_thresh=10,
               nrem_min=30, rem_min=10, hypnogram=None):
        """Run cycle detection on the annotation hypnogram.

        Parameters
        ----------
        hypnogram : sequence of int, optional
            Pre-read hypnogram to reuse. If ``None`` (default), it is read from
            ``self.annotations``. Allows :meth:`run` to read the hypnogram once
            and share it with stage-duration computation.

        Returns
        -------
        list of dict
            Cycle dicts as produced by :func:`detect_cycles`.
        """
        if self.annotations is None:
            raise ValueError("annotations are required for cycle detection")

        if hypnogram is None:
            hypnogram = self.annotations.get_hypnogram()
        if not hypnogram:
            self.logger.warning("Empty hypnogram; no cycles detected.")
            return []

        epoch_starts = self._epoch_starts()
        cycles = detect_cycles(
            hypnogram, epoch_length=epoch_length, wake_thresh=wake_thresh,
            nrem_min=nrem_min, method=method, rem_min=rem_min,
            epoch_starts=epoch_starts)
        self.logger.info(
            f"Detected {len(cycles)} cycle(s) using method '{method}'.")
        return cycles

    def write_cycle_markers(self, cycles):
        """Write cycle boundaries into the Wonambi annotation XML.

        Uses the underlying Wonambi ``clear_cycles`` / ``set_cycle_mrkr`` (via
        ``CustomAnnotations`` delegation). Markers must land exactly on existing
        epoch starts, so boundaries are taken from the epoch grid. Failures are
        logged and swallowed so DB persistence is never blocked by XML issues.
        """
        if not cycles:
            return False
        epoch_starts = self._epoch_starts()
        if epoch_starts is None:
            self.logger.warning(
                "No epoch grid available; skipping XML cycle markers.")
            return False
        n = len(epoch_starts)

        try:
            self.annotations.clear_cycles()
        except Exception as e:
            self.logger.warning(f"clear_cycles failed: {e}")

        written = 0
        for cyc in cycles:
            start_i = cyc['nrem_start_epoch']
            # End marker: start of the epoch after the cycle when it exists,
            # so the boundary matches the next cycle's start; otherwise the
            # last epoch's start.
            end_i = cyc['rem_end_epoch']
            if end_i < cyc['rem_start_epoch']:      # empty REM segment
                end_i = cyc['nrem_end_epoch']
            marker_end_i = end_i + 1 if end_i + 1 < n else end_i
            try:
                self.annotations.set_cycle_mrkr(
                    int(round(epoch_starts[start_i])))
                self.annotations.set_cycle_mrkr(
                    int(round(epoch_starts[marker_end_i])), end=True)
                written += 1
            except Exception as e:
                self.logger.warning(
                    f"Could not mark cycle {cyc['cycle_number']}: {e}")
        self.logger.info(f"Wrote markers for {written} cycle(s) to XML.")
        return written > 0

    @staticmethod
    def _ensure_sleep_cycles_table(conn):
        """Create the sleep_cycles table + events cycle index if missing.

        Also migrates a sleep_cycles table created by an earlier version by
        adding any missing columns, so backfilling an existing DB never fails
        on an outdated schema.
        """
        conn.execute('''
        CREATE TABLE IF NOT EXISTS sleep_cycles (
            subject TEXT,
            method TEXT,
            cycle_number INTEGER,
            nrem_start REAL,
            nrem_end REAL,
            rem_start REAL,
            rem_end REAL,
            nrem_dur_min REAL,
            nrem_n23_dur_min REAL,
            rem_dur_min REAL,
            cycle_dur_min REAL,
            PRIMARY KEY (subject, method, cycle_number)
        )''')
        # Additive migration for tables made before nrem_n23_dur_min existed.
        existing = {r[1] for r in conn.execute(
            'PRAGMA table_info(sleep_cycles)').fetchall()}
        for col in ('nrem_n23_dur_min',):
            if col not in existing:
                conn.execute(
                    f'ALTER TABLE sleep_cycles ADD COLUMN {col} REAL')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_cycle ON events(cycle)')

    def store_cycles_to_database(self, cycles, db_path, subject=None,
                                 method=None, conn=None):
        """Insert detected cycles into the ``sleep_cycles`` table.

        Existing rows for the same ``(subject, method)`` are replaced so reruns
        stay idempotent.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        subject : str, optional
            Subject identifier. Falls back to ``self.subject`` (then ``''``).
        method : str, optional
            Cycle definition, used only when ``cycles`` is empty.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it and this method neither opens nor closes a connection,
            which is what lets a whole subject share one connection. When
            ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.

        Returns
        -------
        int
            Number of cycles written.
        """
        subject = subject if subject is not None else (self.subject or '')
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            method_vals = {c['method'] for c in cycles} or {method}
            for m in method_vals:
                conn.execute(
                    'DELETE FROM sleep_cycles WHERE subject=? AND method=?',
                    (subject, m))
            conn.executemany('''
                INSERT INTO sleep_cycles
                    (subject, method, cycle_number, nrem_start, nrem_end,
                     rem_start, rem_end, nrem_dur_min, nrem_n23_dur_min,
                     rem_dur_min, cycle_dur_min)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (subject, c['method'], c['cycle_number'],
                 c['nrem_start_sec'], c['nrem_end_sec'],
                 c['nrem_end_sec'], c['rem_end_sec'],
                 c['nrem_dur_min'], c['nrem_n23_dur_min'],
                 c['rem_dur_min'], c['cycle_dur_min'])
                for c in cycles])
            conn.commit()
            self.logger.info(
                f"Stored {len(cycles)} cycle(s) for subject "
                f"'{subject}' in {db_path}.")
        finally:
            if own:
                conn.close()
        return len(cycles)

    @staticmethod
    def _ensure_stage_durations_table(conn):
        """Create the ``stage_durations`` table if it does not exist.

        One row per subject holding per-stage minutes derived from the
        hypnogram, so downstream analysis can rely on ``neural_events.db`` alone
        instead of separate stage-summary CSVs.
        """
        conn.execute('''
        CREATE TABLE IF NOT EXISTS stage_durations (
            subject TEXT,
            epoch_length REAL,
            wake_min REAL,
            n1_min REAL,
            n2_min REAL,
            n3_min REAL,
            rem_min REAL,
            artefact_min REAL,
            total_min REAL,
            PRIMARY KEY (subject)
        )''')

    def store_stage_durations(self, stage_durations, db_path, subject=None,
                              conn=None):
        """Upsert per-stage sleep durations into the ``stage_durations`` table.

        The existing row for ``subject`` is deleted then re-inserted so reruns
        stay idempotent (matching :meth:`store_cycles_to_database`).

        Parameters
        ----------
        stage_durations : dict
            Duration summary as returned by :func:`compute_stage_durations`.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        subject : str, optional
            Subject identifier. Falls back to ``self.subject`` (then ``''``)
            exactly like the cycle-storage methods.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it. When ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.

        Returns
        -------
        int
            Number of rows written (always 1).
        """
        subject = subject if subject is not None else (self.subject or '')
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_stage_durations_table(conn)
            conn.execute(
                'DELETE FROM stage_durations WHERE subject=?', (subject,))
            conn.execute('''
                INSERT INTO stage_durations
                    (subject, epoch_length, wake_min, n1_min, n2_min, n3_min,
                     rem_min, artefact_min, total_min)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subject,
                stage_durations['epoch_length'],
                stage_durations['wake_min'],
                stage_durations['n1_min'],
                stage_durations['n2_min'],
                stage_durations['n3_min'],
                stage_durations['rem_min'],
                stage_durations['artefact_min'],
                stage_durations['total_min']))
            conn.commit()
            self.logger.info(
                f"Stored stage durations for subject '{subject}' in {db_path} "
                f"(total {stage_durations['total_min']:.1f} min).")
        finally:
            if own:
                conn.close()
        return 1

    def tag_events_with_cycles(self, cycles, db_path, conn=None):
        """Assign a cycle number to each event in the ``events`` table.

        Each event is tagged by testing its ``start_time`` against the cycle
        spans. Events outside every cycle keep ``cycle=NULL``. The full cycle
        span (NREM start .. next cycle start, or recording end) is used so
        events in the inter-NREM segment are tagged too.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it. When ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.

        Returns
        -------
        int
            Number of event rows tagged.
        """
        if not cycles:
            return 0
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            total = 0
            for idx, cyc in enumerate(cycles):
                lo = cyc['nrem_start_sec']
                if idx + 1 < len(cycles):
                    hi = cycles[idx + 1]['nrem_start_sec']
                    where = 'start_time >= ? AND start_time < ?'
                else:
                    hi = cyc['rem_end_sec']
                    where = 'start_time >= ? AND start_time <= ?'
                cur = conn.execute(
                    f'UPDATE events SET cycle=? WHERE {where}',
                    (str(cyc['cycle_number']), lo, hi))
                total += cur.rowcount
            conn.commit()
            self.logger.info(f"Tagged {total} event(s) with a cycle number.")
        finally:
            if own:
                conn.close()
        return total

    def run(self, db_path, method='2022', write_xml=True, subject=None,
            epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
            conn=None):
        """Detect cycles, then persist to XML and the database.

        This single entry point serves both backfilling an existing
        ``neural_events.db`` and tagging a freshly detected one. Per-stage sleep
        durations are always written (via :meth:`store_stage_durations`), even
        when no cycles are detected, since an all-wake or unscorable night still
        has stage durations.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract), passed straight through to the
            three storage methods so one whole run shares one connection. When
            supplied the caller owns closing it; when ``None`` this method
            opens one connection for the three writes and closes it on exit.

        Returns
        -------
        list of dict
            The detected cycles (also stored in the DB).

        Raises
        ------
        FileNotFoundError
            If ``db_path`` does not exist and no ``conn`` was supplied. This is
            a post-detection step; it annotates an existing database and never
            creates one.
        """
        if self.annotations is None:
            raise ValueError("annotations are required for cycle detection")

        # Read the hypnogram once and reuse it for both cycle detection and
        # stage-duration accounting.
        hypnogram = self.annotations.get_hypnogram()

        cycles = self.detect(
            method=method, epoch_length=epoch_length, wake_thresh=wake_thresh,
            nrem_min=nrem_min, rem_min=rem_min, hypnogram=hypnogram)

        own = conn is None
        if own:
            # Backfill onto an existing database only; never create one.
            # Skipped when the caller supplied a connection: they already
            # opened the database, so it exists by construction.
            _require_existing_db(db_path)
            conn = dbwrite.open_write_connection(db_path)
        try:
            if cycles:
                if write_xml:
                    try:
                        self.write_cycle_markers(cycles)
                    except Exception as e:
                        self.logger.warning(
                            f"Cycle-marker writing skipped: {e}")
                self.store_cycles_to_database(cycles, db_path, subject=subject,
                                              method=method, conn=conn)
                self.tag_events_with_cycles(cycles, db_path, conn=conn)
            else:
                self.logger.info(
                    "No cycles detected; writing stage durations only.")

            # Stage durations are written regardless of the cycle count, so
            # long as the hypnogram itself is non-empty.
            if hypnogram:
                stage_durations = compute_stage_durations(
                    hypnogram, epoch_length=epoch_length)
                self.store_stage_durations(
                    stage_durations, db_path, subject=subject, conn=conn)
            else:
                self.logger.warning(
                    "Empty hypnogram; no stage durations stored.")
        finally:
            if own:
                conn.close()

        return cycles


def finalize_cycles_and_durations(
        annotations, db_path, subject=None,
        methods=('2022', '1979'), tag_method='2022',
        write_xml=True, plot=False, plot_path=None,
        epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
        log_level=logging.INFO):
    """Populate ``neural_events.db`` with sleep cycles + stage durations.

    The explicit post-detection finalize step. Run it once after event
    detection: it detects sleep cycles for every method in ``methods``, stores
    them in ``sleep_cycles`` (both definitions coexist, keyed by
    ``(subject, method)``), writes per-stage durations to ``stage_durations``,
    tags ``events.cycle`` and (optionally) writes cycle markers into the
    annotation XML, and can emit the hypnogram/cycle PNG. Every write is
    idempotent, so re-running is safe.

    Because :meth:`ParalCycles.tag_events_with_cycles` rewrites *all* event rows
    by time window regardless of method ("last run wins"), ``methods`` is
    reordered so ``tag_method`` runs LAST. That makes ``tag_method``'s cycle
    numbering the one that survives in ``events.cycle`` and — when
    ``write_xml`` is True — in the XML markers, deterministically.

    Parameters
    ----------
    annotations : CustomAnnotations
        Annotation wrapper exposing ``get_hypnogram`` / ``epochs``.
    db_path : str
        Path to the ``neural_events.db`` SQLite database.
    subject : str, optional
        Subject identifier stored in ``sleep_cycles`` / ``stage_durations``.
    methods : sequence of str, optional
        Cycle definitions to detect and store (default ``('2022', '1979')``).
    tag_method : str, optional
        The method whose cycle numbering owns ``events.cycle`` and the XML
        markers. Forced to run last among ``methods`` (default ``'2022'``). If
        it is not present in ``methods`` it is not run for tagging, and the
        last method in ``methods`` wins instead.
    write_xml : bool, optional
        Write cycle markers to the annotation XML for ``tag_method`` only
        (default True).
    plot : bool, optional
        If True, also write the hypnogram/cycle PNG (default False; plotting is
        normally a separate call).
    plot_path : str, optional
        Destination PNG when ``plot`` is True. Defaults to a file next to
        ``db_path`` named
        ``{subject or 'hypnogram'}_hypnogram_cycles_{methods joined by _vs_}.png``.
    epoch_length : float, optional
        Epoch duration in seconds (default 30).
    wake_thresh : int, optional
        Max Wake epochs absorbed into surrounding NREM (default 10).
    nrem_min : int, optional
        Min NREM epochs to count as an NREM period (default 30).
    rem_min : int, optional
        Min REM epochs to close a cycle under method ``'1979'`` (default 10).
    log_level : int, optional
        Logging level for the internal ``ParalCycles`` (default
        ``logging.INFO``).

    Returns
    -------
    dict
        ``{method: [cycle dicts]}`` for every method in ``methods``, in the
        original ``methods`` order (not the reordered execution order).

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist. This is the post-detection finalize
        step: it annotates an existing ``neural_events.db`` and never creates
        one, so a missing file is a wrong path rather than a database to
        create. Checked before connecting, since connecting would create it.

    Notes
    -----
    Only ``tag_method`` writes XML markers, so the XML never ends up with two
    conflicting cycle numberings. All methods are stored in ``sleep_cycles``.

    If ``tag_method`` is NOT one of ``methods``, the ``write_xml and
    m == tag_method`` gate never fires, so NO XML cycle markers are written at
    all — yet ``events.cycle`` is still overwritten by the last method in the
    run order ("last run wins"). The XML markers (unchanged from a prior run, or
    absent) can then disagree with ``events.cycle``. This is a misconfiguration
    path; keep ``tag_method`` within ``methods`` (the default already does) so
    the XML markers and ``events.cycle`` stay consistent.
    """
    pc = ParalCycles(annotations=annotations, subject=subject,
                     log_level=log_level)

    methods = list(methods)
    # Reorder so tag_method runs LAST (tagging is "last run wins").
    if tag_method in methods:
        run_order = [m for m in methods if m != tag_method] + [tag_method]
    else:
        run_order = list(methods)

    # One connection for the whole subject: every method's cycle storage, event
    # tagging and stage-duration write share it. Previously each of the three
    # storage methods opened and closed its own untimed connection per method
    # (six connect/close cycles for the default two methods), and on a WAL
    # database each close deletes and each connect recreates the -wal/-shm
    # sidecars -- the operation that fails on a network share.
    cycles_by_method = {}
    # Fail fast before connecting: connecting would create the file. This is a
    # backfill onto an existing neural_events.db, so a missing one is a wrong
    # path, not a database to create.
    _require_existing_db(db_path)
    conn = dbwrite.open_write_connection(db_path, logger=pc.logger)
    try:
        for m in run_order:
            cycles = pc.run(
                db_path, method=m, write_xml=(write_xml and m == tag_method),
                subject=subject, epoch_length=epoch_length,
                wake_thresh=wake_thresh, nrem_min=nrem_min, rem_min=rem_min,
                conn=conn)
            cycles_by_method[m] = cycles
    finally:
        conn.close()

    # Return in the caller's requested method order for stable plotting.
    cycles_by_method = {m: cycles_by_method[m] for m in methods}

    if plot:
        if plot_path is None:
            import os
            db_dir = os.path.dirname(os.path.abspath(db_path))
            stem = subject if subject else 'hypnogram'
            fname = (f"{stem}_hypnogram_cycles_"
                     f"{'_vs_'.join(str(m) for m in methods)}.png")
            plot_path = os.path.join(db_dir, fname)
        # Imported lazily so cycleprocessor stays matplotlib-free at module
        # load (headless-safe library import).
        from .cycleplot import plot_from_annotations
        plot_from_annotations(annotations, cycles_by_method, plot_path,
                              epoch_length=epoch_length, subject=subject)
        pc.logger.info("Cycle plot written to %s", plot_path)

    return cycles_by_method
