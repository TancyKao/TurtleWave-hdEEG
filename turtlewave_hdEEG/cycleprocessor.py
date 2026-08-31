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
from .utils import normalize_subject

#: Fallback for module-level helpers called without a processor logger.
LOGGER = logging.getLogger('turtlewave_hdEEG.cycleprocessor')


def _subject_spellings(conn, table, subject, logger=None):
    """Every stored spelling of one recording's id in ``table``.

    The idempotency delete in the cycle writers is keyed on ``subject``. Now
    that the writers normalise before inserting, a row written earlier under
    the bare folder name (which the cycle how-to tells users to pass) is not
    matched by a delete on the canonical id, so the insert adds a *second* row
    instead of replacing the first. ``stage_durations`` has
    ``PRIMARY KEY (subject)`` -- one row per recording is its whole contract --
    and the duplicate doubles any total computed from it.

    Matching every spelling that normalises to the same canonical id makes the
    delete do what it always claimed to.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    table : str
        Table holding a ``subject`` column.
    subject : str
        Canonical (already normalised) subject id.
    logger : logging.Logger or None, optional
        Logger for the stale-spelling notice. Default ``None``.

    Returns
    -------
    list of str
        Stored spellings equivalent to ``subject``, canonical first. On a
        failed lookup this degrades to ``[subject]`` -- the pre-fix
        single-spelling delete -- and says so at WARNING, because that
        degradation silently re-introduces the duplicate row this function
        exists to prevent.
    """
    try:
        stored = [r[0] for r in conn.execute(
            f"SELECT DISTINCT subject FROM {table} "
            f"WHERE subject IS NOT NULL AND subject != ''")]
    except Exception as e:
        (logger or LOGGER).warning(
            "Could not read the stored subject spellings from %s (%s), so the "
            "idempotency delete falls back to the canonical id '%s' alone. If "
            "this recording has rows under an older spelling of its id they "
            "will NOT be replaced, and the insert that follows adds a "
            "duplicate instead. Check the table and re-run.", table, e, subject)
        return [subject]
    equivalent = [s for s in stored
                  if str(s) != subject and normalize_subject(str(s)) == subject]
    if equivalent and logger is not None:
        logger.warning(
            "%s holds this recording under %d older spelling(s) of its "
            "subject id (%s); they are being replaced by '%s' so the "
            "recording keeps one row per key instead of gaining a duplicate.",
            table, len(equivalent), ", ".join(repr(s) for s in equivalent),
            subject)
    return [subject] + equivalent


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
        """Replace the cycle markers in the Wonambi annotation XML.

        Uses the underlying Wonambi ``clear_cycles`` / ``set_cycle_mrkr`` (via
        ``CustomAnnotations`` delegation). Markers must land exactly on existing
        epoch starts, so boundaries are taken from the epoch grid. Failures are
        logged and swallowed so DB persistence is never blocked by XML issues.

        The clear always runs, including when ``cycles`` is empty: this is a
        *replacement*, not an append. A re-run that finds no cycles (a raised
        ``nrem_min``, an all-Wake night, rescored epochs) must leave the XML
        agreeing with ``sleep_cycles`` and ``events.cycle``, and keeping the
        previous run's markers would leave the annotation file describing
        cycles that no longer exist anywhere else.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. An empty list
            clears the markers and writes none.

        Returns
        -------
        bool
            True when at least one cycle's markers were written. False when
            ``cycles`` was empty (the markers were still cleared) or the epoch
            grid needed to place them was unavailable.
        """
        # Clear before anything can bail out, so an empty cycle list -- or a
        # missing epoch grid -- still leaves the XML free of the previous run's
        # markers rather than silently keeping them.
        try:
            self.annotations.clear_cycles()
        except Exception as e:
            self.logger.warning(f"clear_cycles failed: {e}")

        if not cycles:
            self.logger.info(
                "No cycles to mark; cleared any existing XML cycle markers.")
            return False
        epoch_starts = self._epoch_starts()
        if epoch_starts is None:
            self.logger.warning(
                "No epoch grid available; XML cycle markers were cleared but "
                "none were written.")
            return False
        n = len(epoch_starts)

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
        """Replace this subject's rows in the ``sleep_cycles`` table.

        Existing rows for the same ``(subject, method)`` are deleted, then the
        new cycles are inserted, so reruns stay idempotent.

        The delete runs even when ``cycles`` is empty. That is what makes a
        re-run finding no cycles (a raised ``nrem_min``, an all-Wake night,
        rescored epochs) *replace* the previous run rather than leave its rows
        behind: :meth:`tag_events_with_cycles` clears ``events.cycle`` for the
        same case, so skipping the delete here would leave the table claiming
        cycles that no event is tagged with and no XML marker records.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. An empty list
            deletes the subject's rows for ``method`` and inserts nothing.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        subject : str, optional
            Subject identifier. Falls back to ``self.subject`` (then ``''``).
        method : str, optional
            Cycle definition. Read from the cycles themselves when there are
            any; when ``cycles`` is empty it is the only thing naming the rows
            to delete, so passing ``None`` there deletes nothing and is warned
            about.
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
        # One canonical spelling, matching analysed_time / pac_coupling and the
        # detectors. The cycle how-to tells users to pass the bare folder name,
        # so without this a recording carries '10sd' here and 'sub-10sd' there
        # -- two subjects to SQL, and the detectors' single-subject guard then
        # refuses the recording's own next run.
        subject = normalize_subject(
            subject if subject is not None else (self.subject or ''))
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            method_vals = {c['method'] for c in cycles} or {method}
            if None in method_vals:
                # Only reachable with an empty cycles list and method=None:
                # nothing names the rows to replace, so DELETE ... method=NULL
                # matches nothing and the previous run's rows would survive.
                self.logger.warning(
                    "store_cycles_to_database was called with no cycles and "
                    "no method for subject '%s', so no existing sleep_cycles "
                    "rows could be identified for replacement and any "
                    "previous run's rows were left in place. Pass method= to "
                    "make the replacement complete.", subject)
                method_vals = {m for m in method_vals if m is not None}
            # Delete every stored spelling of this recording's id, not just the
            # canonical one, or a row written under the bare folder name
            # survives and the insert adds a duplicate cycle.
            spellings = _subject_spellings(conn, 'sleep_cycles', subject,
                                           self.logger)
            placeholders = ",".join("?" * len(spellings))
            deleted = 0
            for m in method_vals:
                deleted += conn.execute(
                    f'DELETE FROM sleep_cycles WHERE subject IN ({placeholders}) '
                    f'AND method=?', (*spellings, m)).rowcount
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
            if cycles:
                self.logger.info(
                    f"Stored {len(cycles)} cycle(s) for subject "
                    f"'{subject}' in {db_path} (replacing {deleted} "
                    f"previously stored row(s)).")
            else:
                self.logger.info(
                    "No cycles to store for subject '%s' in %s; removed %d "
                    "previously stored row(s) for method(s) %s so the table "
                    "describes this run.", subject, db_path, deleted,
                    sorted(str(m) for m in method_vals) or 'none')
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
        # Same canonical spelling as write_cycles_to_database; see there.
        subject = normalize_subject(
            subject if subject is not None else (self.subject or ''))
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_stage_durations_table(conn)
            # Same as write_cycles_to_database: match every stored spelling of
            # this recording's id. stage_durations is PRIMARY KEY (subject), so
            # a missed old-spelling row is a second row for one recording and
            # doubles any SUM over the table.
            spellings = _subject_spellings(conn, 'stage_durations', subject,
                                           self.logger)
            conn.execute(
                'DELETE FROM stage_durations WHERE subject IN (%s)'
                % ",".join("?" * len(spellings)), spellings)
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

    def tag_events_with_cycles(self, cycles, db_path=None, conn=None,
                               run_id=None):
        """Replace the ``cycle`` column of the ``events`` table in one scope.

        The ``cycle`` column is first cleared across the whole scope, then each
        event is tagged by testing its ``start_time`` against the cycle spans;
        both happen in one transaction. Clearing first is what makes a re-run
        *replace* rather than merge: the per-cycle ``UPDATE``s only touch rows
        that fall inside a new span, so without the clear an event tagged by a
        previous run whose spans have since moved (a different ``wake_thresh``,
        rescored epochs) would keep its old, now-wrong cycle number. Events
        outside every cycle end up ``cycle=NULL``. The full cycle span (NREM
        start .. next cycle start, or recording end) is used so events in the
        inter-NREM segment are tagged too.

        Passing an empty ``cycles`` list is therefore *not* a no-op: it clears
        the scope (every ``cycle`` value in it becomes NULL) and tags nothing,
        which is the right result when a re-run detects no cycles at all and
        the previous run's tags would otherwise survive. The return value is
        still an ``int`` -- 0, the number of rows *tagged* -- so a caller
        cannot tell a clear from a no-op by the return; the cleared count is
        logged at INFO.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. Only
            ``cycle_number``, ``nrem_start_sec`` and ``rem_end_sec`` are read,
            so a caller can rebuild them from stored ``sleep_cycles`` rows
            instead of re-detecting (see
            :func:`turtlewave_hdEEG.dbwrite.tag_run_cycles`).
        db_path : str, optional
            Path to the ``neural_events.db`` SQLite database. Only used when
            ``conn`` is ``None``; ignored otherwise.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it. When ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.
        run_id : str, optional
            When given, only rows carrying this ``events.run_id`` are tagged.
            A detection run passes its own id so it annotates the rows it just
            wrote and leaves every other run's ``cycle`` value alone --
            without it, tagging is a table-wide ``UPDATE`` and one detector's
            finalize step silently renumbers every other detector's events
            (harmlessly when the cycles agree, wrongly when they were computed
            from different scoring). ``None`` (the backfill case) tags the
            whole table, which is what a backfill is for.

        Returns
        -------
        int
            Number of event rows tagged. Zero when ``cycles`` is empty, even
            though rows may have been cleared in that call.
        """
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            scoped = run_id is not None
            # Clear the scope before re-tagging, in the same transaction as the
            # per-cycle updates below, so the column is never left holding a
            # mix of this run's numbering and a previous run's. Restricted to
            # rows that actually carry a tag, which costs nothing extra and
            # makes rowcount an honest count of previously tagged rows.
            clear_sql = 'UPDATE events SET cycle=NULL WHERE cycle IS NOT NULL'
            clear_params = []
            if scoped:
                clear_sql += ' AND run_id = ?'
                clear_params.append(str(run_id))
            cleared = conn.execute(clear_sql, clear_params).rowcount
            total = 0
            for idx, cyc in enumerate(cycles):
                lo = cyc['nrem_start_sec']
                if idx + 1 < len(cycles):
                    hi = cycles[idx + 1]['nrem_start_sec']
                    where = 'start_time >= ? AND start_time < ?'
                else:
                    hi = cyc['rem_end_sec']
                    where = 'start_time >= ? AND start_time <= ?'
                params = [str(cyc['cycle_number']), lo, hi]
                if scoped:
                    where += ' AND run_id = ?'
                    params.append(str(run_id))
                cur = conn.execute(
                    f'UPDATE events SET cycle=? WHERE {where}', params)
                total += cur.rowcount
            conn.commit()
            self.logger.info(
                "Cleared %d previous cycle tag(s), then tagged %d event(s) "
                "with a cycle number%s.", cleared, total,
                f" (run_id={run_id})" if scoped else " (whole table)")
        finally:
            if own:
                conn.close()
        return total

    def run(self, db_path, method='2022', write_xml=True, subject=None,
            epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
            conn=None, run_id=None, tag_events=True):
        """Detect cycles, then persist to XML and the database.

        This single entry point serves both backfilling an existing
        ``neural_events.db`` and tagging a freshly detected one. Per-stage sleep
        durations are always written (via :meth:`store_stage_durations`), even
        when no cycles are detected, since an all-wake or unscorable night still
        has stage durations.

        Every store is a *replacement* and all of them run whether or not this
        run detected any cycles: the ``sleep_cycles`` rows for
        ``(subject, method)``, the XML cycle markers (when ``write_xml``) and
        ``events.cycle`` (when ``tag_events``) are cleared before the new
        values, if any, go in. A re-run that finds no cycles therefore leaves
        all three empty and agreeing, instead of an emptied ``events.cycle``
        beside a stale ``sleep_cycles`` table and stale XML markers.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract), passed straight through to the
            three storage methods so one whole run shares one connection. When
            supplied the caller owns closing it; when ``None`` this method
            opens one connection for the three writes and closes it on exit.
        run_id : str, optional
            Passed to :meth:`tag_events_with_cycles` so a detection run tags
            only the rows it wrote. Default ``None`` (tag every row).
        tag_events : bool, optional
            When False, cycles and stage durations are stored but
            ``events.cycle`` is left alone. When True the column is rewritten
            for the scope even if no cycles were detected -- tagging clears
            before it writes, so a previous run's tags never survive. Used by
            :func:`turtlewave_hdEEG.dbwrite.ensure_cycles_populated`, which
            runs BEFORE a detection's channel loop -- there are no rows to tag
            yet, and tagging then would rewrite an earlier run's. Default
            ``True``.

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
        ValueError
            If ``self.annotations`` is None, or if its hypnogram is
            unscorable: empty, or every epoch ``-1`` (an epoch grid with no
            scoring saved -- ``get_hypnogram`` maps Undefined/Unknown/
            Artefact/Movement to ``-1``). Either shape is a wrong or unscored
            annotation file rather than a night without cycles, so it fails
            loudly and writes nothing -- silently continuing would clear every
            existing ``events.cycle`` tag, store a 100%-artefact
            ``stage_durations`` row, and report success with zero cycles. A
            scored night with no cycles (all Wake) is not refused.
        """
        if self.annotations is None:
            raise ValueError("annotations are required for cycle detection")

        # Read the hypnogram once and reuse it for both cycle detection and
        # stage-duration accounting.
        hypnogram = self.annotations.get_hypnogram()
        # Refuse an unscorable hypnogram rather than proceed. Two shapes mean
        # "the wrong or an unscored annotation file", not "a night without
        # cycles": no epochs at all, and epochs that are all -1. The second is
        # the epoched-but-unscored case -- get_hypnogram maps Undefined,
        # Unknown, Artefact and Movement (and anything unrecognised) to -1, so
        # a file with an epoch grid and no scoring returns [-1] * n_epochs.
        # Proceeding on either would clear every existing events.cycle tag
        # (tagging clears before it writes) and, for the all -1 case, also
        # store a stage_durations row reading 100% artefact, while reporting
        # success with zero cycles. Raising leaves the database untouched and
        # makes the caller's per-subject handler count it as a failure.
        # A genuinely scored night that happens to contain no cycles -- all
        # Wake, say -- contains 0s, passes this guard, and goes on to clear
        # its stale tags, which is the correct outcome there.
        if not hypnogram:
            raise ValueError(
                "the annotation file has an empty hypnogram, so no cycles or "
                "stage durations can be computed; nothing was written and "
                "any existing events.cycle tags were left alone. Check that "
                "this is the right XML and that it has been scored.")
        if all(stage == -1 for stage in hypnogram):
            raise ValueError(
                f"none of the {len(hypnogram)} epochs in the annotation file "
                f"carries a sleep stage (every epoch reads as "
                f"Undefined/Unknown/Artefact/Movement), so no cycles or stage "
                f"durations can be computed; nothing was written and any "
                f"existing events.cycle tags were left alone. Check that this "
                f"is the right XML and that its scoring has been saved.")

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
            # Both writes are REPLACEMENTS and both run unconditionally, so
            # sleep_cycles, events.cycle and the XML markers always describe
            # the same run. Gating them on `cycles` was the defect fixed here:
            # a re-run detecting none (nrem_min raised, an all-Wake night,
            # rescored epochs) cleared every events.cycle tag while leaving the
            # previous run's sleep_cycles rows and XML markers in place --
            # three stores disagreeing, with nothing recording which was
            # current.
            if write_xml:
                try:
                    self.write_cycle_markers(cycles)
                except Exception as e:
                    self.logger.warning(
                        f"Cycle-marker writing skipped: {e}")
            self.store_cycles_to_database(cycles, db_path, subject=subject,
                                          method=method, conn=conn)
            if not cycles:
                self.logger.info(
                    "No cycles detected for method '%s'; any previously "
                    "stored cycles were removed and stage durations were "
                    "written.", method)

            if tag_events:
                # Called even with no cycles: tagging clears the scope first,
                # so this is also what removes tags left behind by a previous
                # run whose thresholds did find cycles.
                self.tag_events_with_cycles(cycles, db_path, conn=conn,
                                            run_id=run_id)

            # Stage durations are written regardless of the cycle count; an
            # empty hypnogram was already refused above.
            stage_durations = compute_stage_durations(
                hypnogram, epoch_length=epoch_length)
            self.store_stage_durations(
                stage_durations, db_path, subject=subject, conn=conn)
        finally:
            if own:
                conn.close()

        return cycles


def finalize_cycles_and_durations(
        annotations, db_path, subject=None,
        methods=('2022', '1979'), tag_method='2022',
        write_xml=True, plot=False, plot_path=None,
        epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
        log_level=logging.INFO, conn=None, run_id=None, tag_events=True):
    """Populate ``neural_events.db`` with sleep cycles + stage durations.

    The explicit post-detection finalize step. Run it once after event
    detection: it detects sleep cycles for every method in ``methods``, stores
    them in ``sleep_cycles`` (both definitions coexist, keyed by
    ``(subject, method)``), writes per-stage durations to ``stage_durations``,
    tags ``events.cycle`` and (optionally) writes cycle markers into the
    annotation XML, and can emit the hypnogram/cycle PNG. Every write is a
    replacement of this subject's previous values rather than an addition to
    them -- including when a run detects no cycles, which empties all three
    stores instead of leaving the earlier run's rows and markers behind -- so
    re-running is safe and always leaves the three agreeing.

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
    tag_method : str or None, optional
        The method whose cycle numbering owns ``events.cycle`` and the XML
        markers. Forced to run last among ``methods`` (default ``'2022'``).
        **Must be one of ``methods``**, or ``None`` to tag nothing and write
        no markers -- a ``tag_method`` outside ``methods`` raises (see
        ``Raises``).
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
    conn : sqlite3.Connection, optional
        An already-open write connection **on ``db_path``**. Pass it when a
        detector is holding one: opening a second write connection while the
        first is open is a writer-vs-writer collision under DELETE journal
        mode, which is exactly the network-drive failure 4.0.2 was cut for.
        When supplied the caller owns closing it, and the
        database-exists check is skipped (the caller has it open, so it
        exists). Default ``None`` (open and close one here).
    run_id : str, optional
        Passed through to the event tagging so a detection run tags only its
        own rows. Default ``None`` (tag every row -- the backfill case).
    tag_events : bool, optional
        When False, cycles and stage durations are stored but ``events.cycle``
        is not touched. Default ``True``.

    Returns
    -------
    dict
        ``{method: [cycle dicts]}`` for every method in ``methods``, in the
        original ``methods`` order (not the reordered execution order).

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist and no ``conn`` was supplied. This is
        the post-detection finalize step: it annotates an existing
        ``neural_events.db`` and never creates one, so a missing file is a
        wrong path rather than a database to create. Checked before
        connecting, since connecting would create it.
    ValueError
        If ``tag_method`` is neither ``None`` nor one of ``methods``. That
        combination used to be silently self-contradictory: the
        ``m == tag_method`` gate never fires, so NO XML markers are written at
        all, while ``events.cycle`` is still overwritten by whichever method
        happens to run last. The database and the XML then disagree about the
        cycle numbering with nothing recording which is which, and the mistake
        is a single misspelled argument. Pass ``tag_method=None`` if tagging
        nothing is what you meant.

        Also propagated from :meth:`ParalCycles.run` when ``annotations``
        yields an unscorable hypnogram -- empty, or every epoch ``-1`` (an
        epoch grid whose scoring was never saved). Nothing is written and no
        existing ``events.cycle`` tag is cleared, so a batch caller can count
        the subject as failed and move on. A scored night with no cycles (all
        Wake) is not refused: it stores its stage durations and clears its
        stale tags.

    Notes
    -----
    Only ``tag_method`` writes XML markers, so the XML never ends up with two
    conflicting cycle numberings. All methods are stored in ``sleep_cycles``.
    """
    pc = ParalCycles(annotations=annotations, subject=subject,
                     log_level=log_level)

    methods = list(methods)
    if tag_method is not None and tag_method not in methods:
        raise ValueError(
            f"tag_method={tag_method!r} is not one of methods={methods}. It "
            f"names the cycle definition that owns events.cycle and the XML "
            f"markers, so a value outside the list writes NO markers at all "
            f"while events.cycle silently takes whichever method ran last -- "
            f"a database and an annotation file that disagree, from one "
            f"misspelled argument. Use one of {methods}, or tag_method=None "
            f"to store both definitions and tag nothing.")

    # Reorder so tag_method runs LAST (tagging is "last run wins" whenever
    # more than one method tags). With tag_method=None nothing tags.
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
    own_conn = conn is None
    if own_conn:
        # Fail fast before connecting: connecting would create the file. This
        # is a backfill onto an existing neural_events.db, so a missing one is
        # a wrong path, not a database to create. Skipped when the caller
        # supplied a connection -- they have the database open already.
        _require_existing_db(db_path)
        conn = dbwrite.open_write_connection(db_path, logger=pc.logger)
    try:
        for m in run_order:
            cycles = pc.run(
                db_path, method=m, write_xml=(write_xml and m == tag_method),
                subject=subject, epoch_length=epoch_length,
                wake_thresh=wake_thresh, nrem_min=nrem_min, rem_min=rem_min,
                conn=conn, run_id=run_id,
                tag_events=(tag_events and m == tag_method))
            cycles_by_method[m] = cycles
    finally:
        if own_conn:
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
