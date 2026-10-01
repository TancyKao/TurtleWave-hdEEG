#!/usr/bin/env python3
"""
TurtleWave Event Review GUI - Modern 3-Panel Design
Optimized for high-density EEG event review with virtualized table and timeline
"""

import sys
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QFileDialog,
                            QGroupBox, QCheckBox, QComboBox, QSlider,
                            QProgressBar, QTextEdit, QSplitter, QTableView,
                            QHeaderView, QAbstractItemView, QTreeWidget,
                            QTreeWidgetItem, QLineEdit, QMenuBar, QMenu,
                            QAction, QStatusBar, QToolBar, QShortcut,
                            QDockWidget, QStackedWidget)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal

import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush

try:
    from turtlewave_hdEEG import LargeDataset, CustomAnnotations
    from scipy import signal
    from frontend.data_manager import DataManager
    from frontend.waveform_loader import WaveformBackgroundLoader, WaveformCache
    import mne
except ImportError as e:
    print(f"Import warning: {e}")
    mne = None

try:
    from frontend.db_connect import connect_events_db
except ImportError:  # run as a script: frontend/ is on sys.path, not its parent
    from db_connect import connect_events_db

# Non-EEG rule, default review channels and load-failure wording, shared with
# turtlewave_gui and waveform_loader. Qt-free.
try:
    from frontend.channel_types import (ChannelTypeSummary,
                                        default_review_channels,
                                        load_failure_message,
                                        INTERPOLATED_MARK,
                                        INTERPOLATED_TOOLTIP,
                                        REVIEW_GUI_ERROR_WHERE)
except ImportError:  # run as a script
    from channel_types import (ChannelTypeSummary, default_review_channels,
                               load_failure_message, INTERPOLATED_MARK,
                               INTERPOLATED_TOOLTIP, REVIEW_GUI_ERROR_WHERE)

# Per-event review text, population checks and panel rows. Qt-free.
try:
    from frontend import event_review as _er
except ImportError:  # run as a script
    import event_review as _er
CHECK_COLUMNS = _er.CHECK_COLUMNS
CHECK_N = _er.CHECK_N
CHECK_MIN_N = _er.CHECK_MIN_N
TOO_FEW_TIP = _er.TOO_FEW_TIP
THRESHOLD_NOT_RECORDED = _er.THRESHOLD_NOT_RECORDED
fmt_check = _er.fmt_check
method_has_ratio = _er.method_has_ratio
no_ratio_note = _er.no_ratio_note

try:
    from frontend import sample_review as _sr
    from frontend.review_sample_widgets import (SampleBar, DrawSampleDialog,
                                                PrecisionReportDialog)
except ImportError:  # run as a script
    import sample_review as _sr
    from review_sample_widgets import (SampleBar, DrawSampleDialog,
                                       PrecisionReportDialog)

try:
    from frontend.channel_types import (neighbour_channels, physio_channels)
except ImportError:  # run as a script
    from channel_types import neighbour_channels, physio_channels

try:
    from turtlewave_hdEEG.utils import region_from_label
except ImportError:
    def region_from_label(label):
        """Library absent: no 10-20 / 10-5 mapping, every label is 'other'."""
        return 'other'

logger = logging.getLogger('frontend.eeg_review_gui')


# ============================================================================
# Per-event review decisions
#
# Stored by the library in ``event_reviews`` (keyed by uuid and reviewer), so a
# re-detection that rewrites ``events`` keeps them. Looked up lazily: the GUI
# still opens a database with a library that predates the table, and then saves
# no decisions rather than inventing a schema of its own.
# ============================================================================

#: Fallback vocabularies, used only when the library does not provide them.
#: Must equal ``turtlewave_hdEEG.dbwrite.REVIEW_DECISIONS`` / ``REVIEW_REASONS``
#: exactly (the library's CHECK constraint refuses anything else);
#: tests/test_review_gui_event_decisions.py asserts it.
REVIEW_DECISIONS = ('accept', 'reject', 'unsure')
REVIEW_REASONS = ('artefact', 'eye-movement', 'not-in-raw', 'filter-ringing',
                  'off-band', 'too-short', 'arousal', 'single-channel',
                  'not-isolated', 'wrong-morphology', 'other')


def _review_backend():
    """``turtlewave_hdEEG.dbwrite`` when it can store event reviews, else None.

    Only ``ensure_event_reviews_schema`` and ``store_event_review`` are
    required; the vocabularies fall back to the copies above.
    """
    try:
        from turtlewave_hdEEG import dbwrite as dw
    except ImportError:
        return None
    if all(hasattr(dw, n) for n in ('ensure_event_reviews_schema',
                                    'store_event_review')):
        return dw
    return None


def _sqlite_columns(conn, table):
    """Column names of ``table`` on ``conn``; empty when it does not exist."""
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def review_vocabulary():
    """``(REVIEW_DECISIONS, REVIEW_REASONS)``: ``dbwrite``'s constants when
    the library is importable, else the local copies above."""
    try:
        from turtlewave_hdEEG import dbwrite as dw
    except ImportError:
        dw = None
    return (tuple(getattr(dw, 'REVIEW_DECISIONS', REVIEW_DECISIONS)),
            tuple(getattr(dw, 'REVIEW_REASONS', REVIEW_REASONS)))


# ============================================================================
# Excluded event types
#
# The detection run's exclusion set decides how much time the detector actually
# searched, and therefore the density denominator this dashboard divides by. It
# is read back from the run, never assumed, and it is always displayed next to
# the number it defines.
# ============================================================================

#: The detector defaults, used only as the stated fallback when the database
#: cannot say what a run excluded. Taken from the library when it is importable
#: so this file cannot claim a default the detectors do not apply.
try:
    from turtlewave_hdEEG.utils import DEFAULT_REJECT_TYPES as _DEFAULT_REJECTS
    DEFAULT_REJECT_TYPES = tuple(_DEFAULT_REJECTS)
except ImportError:
    DEFAULT_REJECT_TYPES = ('Artefact', 'Arousal', 'Move')

#: Fixed order for every set this GUI shows or returns: the defaults, then the
#: opt-in types. NOT alphabetical - a set must read the same here as it does in
#: the detection GUI, and comparisons are made on sets, never on this order.
REJECT_TYPE_ORDER = tuple(DEFAULT_REJECT_TYPES) + tuple(
    t for t in ('Artefact', 'Arousal', 'Move', 'Resp', 'Snore')
    if t not in DEFAULT_REJECT_TYPES)

#: What every run before 4.4.0 excluded. Those runs had two booleans and no way
#: to name anything else, so a file that records no set ANYWHERE is by
#: definition pre-4.4.0 and this is what its runs did. Deliberately the same
#: value ``turtlewave_gui._choose_db_scope`` assumes in the same condition: two
#: different guesses for one condition would be indefensible.
PRE_4_4_REJECT_TYPES = ('Artefact', 'Arousal')

#: Stored token -> the word a sleep researcher uses. Duplicated from
#: ``frontend.turtlewave_gui`` on purpose: importing the detection GUI here
#: would pull a second QMainWindow into every review session for five strings.
REJECT_TYPE_LABELS = {
    'Artefact': 'Artefact',
    'Arousal': 'Arousal',
    'Move': 'Movement',
    'Resp': 'Respiratory',
    'Snore': 'Snoring',
}

def order_reject_types(reject_types):
    """Put a reject set into :data:`REJECT_TYPE_ORDER`, de-duplicated.

    Parameters
    ----------
    reject_types : iterable of str or None
        Stored tokens in any order.

    Returns
    -------
    list of str
        The same set, canonically ordered; unknown types keep their relative
        order and follow the known ones.
    """
    if not reject_types:
        return []
    if isinstance(reject_types, str):
        reject_types = [reject_types]
    seen = []
    for t in reject_types:
        t = str(t).strip()
        if t and t not in seen:
            seen.append(t)
    known = [t for t in REJECT_TYPE_ORDER if t in seen]
    return known + [t for t in seen if t not in known]


def reject_types_display(reject_types):
    """Render a reject set in researcher-facing words, canonically ordered.

    Parameters
    ----------
    reject_types : iterable of str or None
        Stored tokens.

    Returns
    -------
    str
        e.g. ``'Artefact, Arousal, Movement'``, or ``'nothing'`` for an empty
        set.
    """
    tokens = order_reject_types(reject_types)
    if not tokens:
        return "nothing"
    return ", ".join(REJECT_TYPE_LABELS.get(t, t) for t in tokens)


# ============================================================================
# Logging for the density helpers
# ============================================================================

class _RepeatSuppressingFilter(logging.Filter):
    """Let each distinct message through once per context, then drop repeats.

    The QC dashboard rebuilds its density denominators on every refresh, and
    ``turtlewave_hdEEG.utils.compute_analysed_seconds`` warns about an
    inconsistent annotation each time it does. The warning is about the file,
    not about the refresh, so repeating it once per stage per refresh buries
    the rest of the log without adding information.

    Suppression happens here, in the GUI, and never in the library: that
    warning reports a data-integrity condition and the library deliberately
    refuses to be silenced by a ``logger=None`` argument. What this does is
    accept the record once and stop the identical restatement.

    The context is the annotation file the message is about, so opening a
    different file reports its own condition rather than inheriting the last
    file's silence.
    """

    def __init__(self):
        super().__init__()
        self.context = None
        self._seen = set()

    def filter(self, record):
        key = (self.context, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


#: Logger the review GUI hands to the density helpers, so their warnings pass
#: through the repeat filter above instead of going straight to the library's
#: module logger.
_density_logger = logging.getLogger('frontend.eeg_review_gui.density')
_density_repeat_filter = _RepeatSuppressingFilter()
_density_logger.addFilter(_density_repeat_filter)


# ============================================================================
# EventDatabase Class (from eeg_eventview.py)
# ============================================================================

class EventDatabase:
    """Enhanced database handler with automatic optimization"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        # write=True: this class creates the review/QC tables, adds
        # columns and saves review decisions.
        self.conn = connect_events_db(db_path, write=True)

        # Auto-optimize on connection
        self._auto_optimize()
        self.create_review_tables()
        self.create_qc_tables()
        
        # Import DataManager for advanced caching
        try:
            self.data_manager = DataManager(db_path, None)
        except:
            self.data_manager = None
            print("DataManager not available, using basic caching")
    
    def _auto_optimize(self):
        """Automatically apply performance optimizations"""
        cursor = self.conn.cursor()
        
        # Performance PRAGMAs. journal_mode is deliberately absent -- it is
        # decided once, in frontend/db_connect.py. mmap_size is absent too: memory
        # mapping is exactly what fails on SMB/NFS shares.
        # synchronous=NORMAL is only corruption-safe under WAL; every other
        # journal mode (DELETE included) needs FULL, SQLite's safe default.
        try:
            journal_mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
        except (sqlite3.Error, TypeError):
            journal_mode = ""
        synchronous = "NORMAL" if str(journal_mode).lower() == "wal" else "FULL"

        optimizations = [
            f"PRAGMA synchronous={synchronous}",
            "PRAGMA cache_size=-64000",
            "PRAGMA temp_store=MEMORY",
        ]
        
        for pragma in optimizations:
            try:
                cursor.execute(pragma)
            except sqlite3.Error as e:
                print(f"Warning: Could not apply {pragma}: {e}")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_channel_starttime ON events(channel, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_stage ON events(stage)",
            "CREATE INDEX IF NOT EXISTS idx_eventtype_channel ON events(event_type, channel)",
            "CREATE INDEX IF NOT EXISTS idx_method ON events(method)",
            "CREATE INDEX IF NOT EXISTS idx_freq_band ON events(freq_lower, freq_upper)",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error:
                pass
        
        self.conn.commit()
    
    @property
    def has_review_backend(self):
        """True when the library can store per-event decisions."""
        return _review_backend() is not None

    def create_review_tables(self):
        """Ensure the ``event_reviews`` table, and nothing else.

        Decisions live in their own table keyed by ``(uuid, reviewer)``, so a
        re-detection that replaces ``events`` rows cannot wipe them. The
        ``events`` table is never altered here. Only the review-table step is
        run: the full ``ensure_direct_write_schema`` would overwrite
        ``db_meta.turtlewave_version`` from a GUI session. Without a library
        that provides the table this is a no-op and decisions are not saved.
        """
        dw = _review_backend()
        if dw is None:
            return
        try:
            dw.ensure_event_reviews_schema(self.conn)
            self.conn.commit()
        except Exception as err:   # never block opening the database
            logger.warning(f"Could not create the event_reviews table "
                           f"({err}); review decisions will not be saved.")

    def _table_columns(self, table):
        """Column names of ``table``; empty when it does not exist."""
        return _sqlite_columns(self.conn, table)

    # ------------------------------------------------------------------
    # QC-by-outlier-triage state (GUI-side only; the detection-output
    # `events` schema is never modified — same posture as review columns).
    # ------------------------------------------------------------------
    def create_qc_tables(self):
        """Create GUI-side QC state tables. Idempotent.

        - channel_qc: per-channel keep/drop verdict (drop => omitted from the
          re-detect channels.csv).
        - qc_artefact_intervals: WHOLE-MONTAGE artefact time windows the
          reviewer confirmed as global. `evidence_channel` is provenance only;
          the emitted Wonambi event is chan='(all)'.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channel_qc (
                channel TEXT,
                event_type TEXT,
                verdict TEXT,
                reviewer TEXT,
                qc_timestamp TEXT,
                PRIMARY KEY (channel, event_type)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qc_artefact_intervals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                evidence_channel TEXT,
                reviewer TEXT,
                qc_timestamp TEXT,
                exported INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def get_channel_qc_aggregates(self, event_types=None):
        """Per-(channel, event_type) aggregates over the existing events table.

        Cheap index-backed GROUP BY for counts/means/extrema. Percentiles and
        the robust-outlier flag are computed in pandas by ``compute_channel_qc``
        from the montage-wide pull (SQLite has no percentile function).
        Amplitudes are Wonambi µV params computed on the detection-band signal,
        so callers must keep comparisons within a single event_type.
        """
        query = '''
            SELECT channel,
                   event_type,
                   COUNT(*)                       AS n,
                   AVG(max_amp)                   AS mean_amp,
                   AVG(peak2peak_amp)             AS mean_p2p,
                   MAX(peak2peak_amp)             AS max_p2p,
                   MIN(start_time)                AS first_start,
                   MAX(start_time)                AS last_start
            FROM events
            WHERE 1=1
        '''
        params = []
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        query += " GROUP BY channel, event_type ORDER BY channel"
        return pd.read_sql_query(query, self.conn, params=params)

    def set_channel_verdict(self, channel, event_type, verdict, reviewer=""):
        """Persist a keep/drop verdict for a (channel, event_type)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO channel_qc
                (channel, event_type, verdict, reviewer, qc_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (channel, event_type, verdict, reviewer, datetime.now().isoformat()))
        self.conn.commit()

    def get_channel_verdicts(self):
        """Return {(channel, event_type): verdict} for resume across sessions."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT channel, event_type, verdict FROM channel_qc")
        except sqlite3.OperationalError:
            return {}
        return {(ch, et): v for ch, et, v in cursor.fetchall()}

    def add_qc_artefact_interval(self, start_time, end_time,
                                 evidence_channel="", reviewer=""):
        """Record a confirmed WHOLE-MONTAGE artefact window."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO qc_artefact_intervals
                (start_time, end_time, evidence_channel, reviewer, qc_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (float(start_time), float(end_time), evidence_channel,
              reviewer, datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid

    def get_qc_artefact_intervals(self, unexported_only=False):
        """Return recorded whole-montage artefact windows as a DataFrame."""
        query = "SELECT * FROM qc_artefact_intervals"
        if unexported_only:
            query += " WHERE exported = 0"
        query += " ORDER BY start_time"
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception:
            return pd.DataFrame()

    def remove_qc_artefact_interval(self, interval_id):
        """Unmark a previously recorded artefact window."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM qc_artefact_intervals WHERE id = ?",
                       (interval_id,))
        self.conn.commit()

    def mark_artefact_intervals_exported(self, ids):
        """Flag intervals as written into a re-run package."""
        if not ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join(['?' for _ in ids])
        cursor.execute(
            f"UPDATE qc_artefact_intervals SET exported = 1 WHERE id IN ({placeholders})",
            list(ids))
        self.conn.commit()

    def get_run_rejections(self, event_type=None, methods=None):
        """The event types the detection run excluded from its search.

        These are not a display preference: they decide how much time the
        detector actually searched, and therefore the density denominator. The
        QC dashboard used to assume both were on, which silently inflates
        density for any run that had one unticked - arousal time gets
        subtracted from the denominator although the detector searched it.

        Read from ``detection_runs``, which the direct-write path populates,
        taking the MOST RECENT matching run rather than requiring every stored
        run to agree. Requiring agreement sounds safer and is not: a database
        accumulates runs over months, so one old experimental run with arousal
        rejection off would make this return ``None`` forever afterwards, and
        the caller's fallback to the detector defaults reintroduces exactly the
        density inflation this method exists to prevent. The newest run is the
        one whose events the dashboard is showing.

        The caller scopes this to the event type and method combo currently on
        screen, which narrows it further: the ordering only has to break ties
        between runs that are already the right type and method.

        ``timestamp`` is ``datetime.now().isoformat()``, so a lexicographic
        ``ORDER BY`` is chronological. Rows with no timestamp (written before
        the column was populated) sort last under ``DESC`` in SQLite, so a
        dated run always beats an undated one.

        Parameters
        ----------
        event_type : str or None, optional
            Restrict to one event type. Default ``None`` (all).
        methods : list of str or None, optional
            Restrict to these methods, UNESCAPED as stored. Default ``None``.

        Returns
        -------
        list of str or None
            The excluded event types of the most recent matching run, in
            :data:`REJECT_TYPE_ORDER`, e.g. ``['Artefact', 'Arousal', 'Move']``.
            An empty list is a real answer ("that run excluded nothing"), not a
            missing one. ``None`` only when nothing matches, when nothing about
            the exclusion set is recorded, or when ``detection_runs`` is absent
            (a database written before the direct-write path existed).

            Read from the ``reject_types`` column where it exists. When it is
            absent or NULL but the two boolean columns are populated, the set is
            reconstructed from them - exact, not a guess, because a run written
            before 4.4.0 had those two booleans and no way to exclude anything
            else.
        """
        columns = set(_sqlite_columns(self.conn, 'detection_runs'))
        if not columns:
            return None
        cursor = self.conn.cursor()
        has_types = 'reject_types' in columns

        where = []
        if has_types:
            # A row is informative if it carries the set OR the old booleans.
            where.append("(reject_types IS NOT NULL OR (reject_artifacts IS "
                         "NOT NULL AND reject_arousals IS NOT NULL))")
        else:
            where.append("reject_artifacts IS NOT NULL")
            where.append("reject_arousals IS NOT NULL")
        params = []
        if event_type is not None:
            where.append("event_type = ?")
            params.append(str(event_type))
        if methods:
            where.append("method IN (%s)" % ",".join("?" * len(methods)))
            params += [str(m) for m in methods]
        clause = " WHERE " + " AND ".join(where)
        select = ("reject_types, reject_artifacts, reject_arousals"
                  if has_types else
                  "NULL, reject_artifacts, reject_arousals")
        try:
            cursor.execute(
                f"SELECT {select} "
                f"FROM detection_runs{clause} "
                "ORDER BY timestamp DESC LIMIT 1", params)
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        rj_types, rj_a, rj_r = row
        if rj_types is not None:
            return order_reject_types(
                [t for t in str(rj_types).split(',') if t])
        if rj_a is None and rj_r is None:
            return None
        return order_reject_types(
            (['Artefact'] if rj_a else []) + (['Arousal'] if rj_r else []))

    def get_run_info(self, run_id):
        """One ``detection_runs`` row as a dict, with its parameters parsed.

        Parameters
        ----------
        run_id : str
            The run an event row names in its ``run_id`` column. Two runs can
            cover the same scope, so look up by the row's own run, never by
            method alone.

        Returns
        -------
        dict
            Every stored column (``method``, ``stages``, ``event_type``, ...)
            plus ``params``: ``params_json`` decoded to a dict (``{}`` when it
            is absent or not valid JSON). Empty when ``run_id`` is missing,
            unknown, or the database has no ``detection_runs`` table.
        """
        if run_id is None or (isinstance(run_id, float) and np.isnan(run_id)):
            return {}
        if not self._table_columns('detection_runs'):
            return {}
        try:
            cur = self.conn.execute(
                "SELECT * FROM detection_runs WHERE run_id = ?", (str(run_id),))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        except sqlite3.Error:
            return {}
        if row is None:
            return {}
        info = dict(zip(names, row))
        try:
            params = json.loads(info.get('params_json') or '{}')
        except (TypeError, ValueError):
            params = {}
        info['params'] = params if isinstance(params, dict) else {}
        return info

    def get_events(self, event_type=None, channels=None, stages=None,
                   reviewed_only=False, unreviewed_only=False, confidence_threshold=0.0,
                   methods=None, freq_band=None, columns=None):
        """Get events with comprehensive filtering including method and freq_band.

        ``columns`` (list of column names) emits a lean ``SELECT col, ...``
        instead of ``SELECT *`` — used by the QC/Epochs path, which only needs
        a handful of columns out of 23. WHERE clauses can still reference
        non-selected columns. Default None keeps ``SELECT *`` for the Events
        tab, which needs the display columns. Requested columns the table
        does not have (``run_id`` / ``epoch_stage`` on a database written
        before the direct-write path) are left out rather than failing.

        ``reviewed_only`` / ``unreviewed_only`` test for any row in
        ``event_reviews`` (any reviewer). Without that table nothing is
        reviewed.
        """
        if columns:
            have = set(self._table_columns('events'))
            columns = [c for c in columns if c in have] if have else columns
        sel = "*" if not columns else ", ".join(columns)
        query = f"SELECT {sel} FROM events WHERE 1=1"
        params = []
        
        # Filter by event type
        if event_type:
            if isinstance(event_type, list):
                placeholders = ','.join(['?' for _ in event_type])
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_type)
            else:
                query += " AND event_type = ?"
                params.append(event_type)
        
        # Filter by channels
        if channels:
            placeholders = ','.join(['?' for _ in channels])
            query += f" AND channel IN ({placeholders})"
            params.extend(channels)
        
        # Filter by stages
        if stages:
            stage_conditions = []
            for stage in stages:
                stage_conditions.append("stage LIKE ?")
                params.append(f"%{stage}%")
            query += f" AND ({' OR '.join(stage_conditions)})"
        
        # Filter by method
        if methods:
            if isinstance(methods, list):
                placeholders = ','.join(['?' for _ in methods])
                query += f" AND method IN ({placeholders})"
                params.extend(methods)
            else:
                query += " AND method = ?"
                params.append(methods)
        
        # Filter by frequency band
        if freq_band:
            if isinstance(freq_band, tuple) and len(freq_band) == 2:
                # freq_band is (lower, upper) tuple
                # Show events where freq_lower and freq_upper EXACTLY match the filter band
                # For 9-12 Hz filter: show events with freq_lower=9.0 AND freq_upper=12.0
                # For 12-15 Hz filter: show events with freq_lower=12.0 AND freq_upper=15.0
                query += " AND freq_lower = ? AND freq_upper = ?"
                params.extend([freq_band[0], freq_band[1]])
            elif isinstance(freq_band, list):
                # Multiple freq bands as list of tuples
                freq_conditions = []
                for fb in freq_band:
                    if isinstance(fb, tuple) and len(fb) == 2:
                        freq_conditions.append("(freq_lower = ? AND freq_upper = ?)")
                        params.extend([fb[0], fb[1]])
                if freq_conditions:
                    query += f" AND ({' OR '.join(freq_conditions)})"
        
        # Filter by review status (event_reviews, keyed by uuid)
        if reviewed_only or unreviewed_only:
            has_reviews = bool(self._table_columns('event_reviews'))
            if reviewed_only:
                query += (" AND EXISTS (SELECT 1 FROM event_reviews r "
                          "WHERE r.uuid = events.uuid)"
                          if has_reviews else " AND 0")
            elif has_reviews:
                query += (" AND NOT EXISTS (SELECT 1 FROM event_reviews r "
                          "WHERE r.uuid = events.uuid)")
        
        # Confidence threshold
        if confidence_threshold > 0:
            query += " AND (confidence_score >= ? OR confidence_score IS NULL)"
            params.append(confidence_threshold)
        
        query += " ORDER BY channel, start_time"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def add_review(self, uuid, decision, reviewer="", comments="", reason=None,
                   reviewed_at=None):
        """Store one reviewer's decision on one event in ``event_reviews``.

        A wrapper over ``turtlewave_hdEEG.dbwrite.store_event_review``; a
        second call by the same reviewer replaces their earlier decision, a
        different reviewer adds a row beside it.

        Parameters
        ----------
        uuid : str
            The event's ``events.uuid``.
        decision : {'accept', 'reject', 'unsure'}
        reviewer : str, optional
        comments : str, optional
            Free text, stored as the review's ``comment``.
        reason : str or None, optional
            One of ``REVIEW_REASONS``.

        Returns
        -------
        bool
            True when stored; False when the library has no review table
            (nothing is written).
        """
        dw = _review_backend()
        if dw is None:
            return False
        dw.store_event_review(self.conn, str(uuid), decision,
                              reviewer or '', reason=reason,
                              comment=comments or None,
                              reviewed_at=reviewed_at)
        self.conn.commit()
        return True

    def get_reviews_for(self, uuids, reviewer=None):
        """Decisions on the given events.

        Parameters
        ----------
        uuids : iterable of str
        reviewer : str or None, optional
            Only this reviewer's decisions. Default ``None``: any reviewer,
            the most recent decision winning when several reviewed one event.

        Returns
        -------
        dict
            ``{uuid: (decision, reason, reviewer)}`` for the reviewed ones
            only; empty without an ``event_reviews`` table.
        """
        ids = [str(u) for u in uuids if u is not None and u == u]
        if not ids or not self._table_columns('event_reviews'):
            return {}
        out = {}
        for k in range(0, len(ids), 900):     # SQLite variable limit
            chunk = ids[k:k + 900]
            sql = ("SELECT uuid, decision, reason, reviewer FROM event_reviews "
                   f"WHERE uuid IN ({','.join('?' * len(chunk))})")
            params = list(chunk)
            if reviewer is not None:
                sql += " AND reviewer = ?"
                params.append(str(reviewer))
            # oldest first, so the newest decision is the one kept;
            # reviewed_at is to the second, so rowid (write order) breaks ties
            sql += " ORDER BY reviewed_at, rowid"
            try:
                for u, dec, rsn, who in self.conn.execute(sql, params):
                    out[u] = (dec, rsn, who)
            except sqlite3.Error as err:
                logger.warning(f"Could not read event_reviews ({err}).")
                return {}
        return out

    def get_review(self, uuid, reviewer):
        """One reviewer's stored row on one event, or ``None``.

        Returns
        -------
        dict or None
            ``decision, reason, comment, reviewed_at`` (and every other
            stored column) for ``(uuid, reviewer)``.
        """
        if not self._table_columns('event_reviews'):
            return None
        try:
            cur = self.conn.execute(
                "SELECT * FROM event_reviews WHERE uuid = ? AND reviewer = ?",
                (str(uuid), str(reviewer).strip()))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        except sqlite3.Error:
            return None
        return dict(zip(names, row)) if row else None

    def get_other_reviews(self, uuid, reviewer):
        """Rows of every OTHER reviewer on one event, oldest first."""
        if not self._table_columns('event_reviews'):
            return []
        try:
            cur = self.conn.execute(
                "SELECT * FROM event_reviews WHERE uuid = ? AND reviewer <> ? "
                "ORDER BY reviewed_at, rowid",
                (str(uuid), str(reviewer or '').strip()))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]
        except sqlite3.Error:
            return []

    def restore_review(self, uuid, reviewer, before):
        """Put one reviewer's row back as it was (undo / Clear's undo).

        ``before`` is a :meth:`get_review` dict, or ``None`` to remove the
        row. Returns True when the database changed.
        """
        dw = _review_backend()
        if dw is None:
            return False
        if before is None:
            if not hasattr(dw, 'delete_event_review'):
                return False
            ok = dw.delete_event_review(self.conn, str(uuid), reviewer)
            self.conn.commit()
            return bool(ok)
        dw.store_event_review(
            self.conn, str(uuid), before['decision'], reviewer,
            reason=before.get('reason'), comment=before.get('comment'),
            reviewed_at=before.get('reviewed_at'))
        self.conn.commit()
        return True

    def get_event(self, uuid):
        """The full ``events`` row of one event, as a dict (``{}`` if gone)."""
        try:
            cur = self.conn.execute("SELECT * FROM events WHERE uuid = ?",
                                    (str(uuid),))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        except sqlite3.Error:
            return {}
        return dict(zip(names, row)) if row else {}

    def ptp_units_microvolts(self):
        """True when ``db_meta.det_ptp_units`` says ``events.det_ptp`` is µV."""
        if not self._table_columns('db_meta'):
            return False
        try:
            row = self.conn.execute(
                "SELECT value FROM db_meta WHERE key = 'det_ptp_units'"
            ).fetchone()
        except sqlite3.Error:
            return False
        return bool(row) and str(row[0]).lower() == 'microvolts'

    def get_thresholds(self, run_id, channel=None, method=None, at_time=None):
        """``read_detection_thresholds`` of the row's own run; an empty
        frame without the library or the table."""
        try:
            from turtlewave_hdEEG.dbwrite import read_detection_thresholds
        except ImportError:
            return pd.DataFrame()
        if run_id is None:
            return pd.DataFrame()
        try:
            return read_detection_thresholds(self.conn, run_id, channel=channel,
                                             method=method, at_time=at_time)
        except Exception as err:
            logger.warning(f"Could not read detection thresholds ({err}).")
            return pd.DataFrame()

    def neighbour_events(self, channels, event_type, t0, t1, methods=None,
                         freq_band=None):
        """``channel, start_time, end_time`` of same-type events on
        ``channels`` overlapping ``[t0, t1]``."""
        if not channels:
            return pd.DataFrame(columns=['channel', 'start_time', 'end_time'])
        q = (f"SELECT channel, start_time, end_time FROM events "
             f"WHERE event_type = ? AND channel IN "
             f"({','.join('?' * len(channels))}) "
             f"AND start_time < ? AND end_time > ?")
        p = [str(event_type)] + [str(c) for c in channels] + [float(t1),
                                                             float(t0)]
        if methods:
            q += f" AND method IN ({','.join('?' * len(methods))})"
            p += list(methods)
        if freq_band:
            q += " AND freq_lower = ? AND freq_upper = ?"
            p += [float(freq_band[0]), float(freq_band[1])]
        try:
            return pd.read_sql_query(q, self.conn, params=p)
        except Exception:
            return pd.DataFrame(columns=['channel', 'start_time', 'end_time'])

    def get_review_stats(self):
        """Counts over ``event_reviews`` for events still in ``events``.

        Returns
        -------
        dict
            ``total`` events, ``reviewed`` events (any reviewer), and
            ``<decision>_count`` = events with at least one such decision.
        """
        cursor = self.conn.cursor()
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['total'] = cursor.fetchone()[0]
        stats['reviewed'] = 0
        if not self._table_columns('event_reviews'):
            return stats
        cursor.execute("""
            SELECT COUNT(DISTINCT r.uuid)
            FROM event_reviews r JOIN events e ON e.uuid = r.uuid
        """)
        stats['reviewed'] = cursor.fetchone()[0]
        cursor.execute("""
            SELECT r.decision, COUNT(DISTINCT r.uuid)
            FROM event_reviews r JOIN events e ON e.uuid = r.uuid
            GROUP BY r.decision
        """)
        for decision, count in cursor.fetchall():
            if decision:
                stats[f'{decision}_count'] = count
        return stats
    
    def get_unique_methods(self, event_type=None):
        """Get unique detection methods from database"""
        cursor = self.conn.cursor()
        if event_type:
            if isinstance(event_type, list):
                placeholders = ','.join(['?' for _ in event_type])
                query = f"SELECT DISTINCT method FROM events WHERE event_type IN ({placeholders}) AND method IS NOT NULL ORDER BY method"
                cursor.execute(query, event_type)
            else:
                cursor.execute("SELECT DISTINCT method FROM events WHERE event_type = ? AND method IS NOT NULL ORDER BY method", (event_type,))
        else:
            cursor.execute("SELECT DISTINCT method FROM events WHERE method IS NOT NULL ORDER BY method")
        return [row[0] for row in cursor.fetchall()]
    
    def get_unique_freq_bands(self, event_type=None):
        """Get unique frequency bands from database as (lower, upper) tuples"""
        cursor = self.conn.cursor()
        if event_type:
            if isinstance(event_type, list):
                placeholders = ','.join(['?' for _ in event_type])
                query = f"SELECT DISTINCT freq_lower, freq_upper FROM events WHERE event_type IN ({placeholders}) AND freq_lower IS NOT NULL AND freq_upper IS NOT NULL ORDER BY freq_lower, freq_upper"
                cursor.execute(query, event_type)
            else:
                cursor.execute("SELECT DISTINCT freq_lower, freq_upper FROM events WHERE event_type = ? AND freq_lower IS NOT NULL AND freq_upper IS NOT NULL ORDER BY freq_lower, freq_upper", (event_type,))
        else:
            cursor.execute("SELECT DISTINCT freq_lower, freq_upper FROM events WHERE freq_lower IS NOT NULL AND freq_upper IS NOT NULL ORDER BY freq_lower, freq_upper")
        return [(row[0], row[1]) for row in cursor.fetchall()]
    
    def export_reviewed_events(self, output_path):
        """Write reviewed events to CSV, one row per (event, reviewer).

        Every ``events`` column, then ``reviewer``, ``review_decision``,
        ``review_reason``, ``review_comment``, ``reviewed_at``. Returns the
        number of rows written (0, with a header-only file, without an
        ``event_reviews`` table).
        """
        if not self._table_columns('event_reviews'):
            df = pd.read_sql_query("SELECT * FROM events WHERE 0", self.conn)
            df.to_csv(output_path, index=False)
            return 0
        query = """
            SELECT e.*, r.reviewer AS reviewer, r.decision AS review_decision,
                   r.reason AS review_reason, r.comment AS review_comment,
                   r.reviewed_at AS reviewed_at
            FROM events e JOIN event_reviews r ON r.uuid = e.uuid
            ORDER BY e.channel, e.start_time, r.reviewer
        """
        df = pd.read_sql_query(query, self.conn)
        # an old database still carries the GUI's former review columns on
        # events; the event_reviews values above are the ones that count
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        df.to_csv(output_path, index=False)
        return len(df)


# ============================================================================
# QC-by-outlier-triage metrics
# ============================================================================

def _region_for(channel):
    """Coarse scalp region from a channel label, used only when real electrode
    coordinates are unavailable. EGI-style ``E<idx>`` labels use an INDEX-based
    guess (EGI numbering spirals around the head, so these buckets are
    approximate and do NOT track true scalp position; that is what
    ``_region_from_xy`` is for). Every other label goes through the 10-20 /
    10-5 rule in ``turtlewave_hdEEG.utils.region_from_label``."""
    s = str(channel)
    if not s.startswith('E') or not s[1:].isdigit():
        return region_from_label(s)
    i = int(s[1:])
    if i <= 15 or 17 <= i <= 25 or 30 <= i <= 60:
        return 'frontal'
    if i >= 240:
        return 'neck'
    if i >= 220:
        return 'cheek'
    if 61 <= i <= 110:
        return 'central'
    if 111 <= i <= 160:
        return 'temporal'
    if 161 <= i <= 200:
        return 'parietal'
    return 'occipital'


def _region_from_xy(x, y):
    """Scalp region from the topo 2-D projection (nose-up: +y front, −y back,
    ±x right/left, radius 0 = vertex). Same coordinates the topography paints,
    so a channel's region matches where its dot sits on the map.

    Thresholds (validated against real EGI-256 coords): vertex/near-centre →
    ``central``; far-lateral (|x| large) → ``temporal``; strongly anterior →
    ``frontal``; strongly posterior → ``occipital``; the mild-posterior /
    central belt → ``parietal``.
    """
    r = (x * x + y * y) ** 0.5
    if r < 0.20:
        return 'central'
    if abs(x) > 0.45:
        return 'temporal'
    if y > 0.30:
        return 'frontal'
    if y < -0.35:
        return 'occipital'
    return 'parietal'


def _region_for_channel(channel, coords=None):
    """Region for one channel: coordinate-based when ``coords`` (a
    ``{label: (x, y)}`` map) carries this channel, else the index fallback."""
    if coords:
        xy = coords.get(str(channel))
        if xy is not None:
            try:
                return _region_from_xy(float(xy[0]), float(xy[1]))
            except (TypeError, ValueError):
                pass
    return _region_for(channel)


def compute_channel_qc(events_df, scored_minutes=None, artefact_intervals=None,
                        hard_z=3.5, soft_z=2.0, dead_frac=0.15, coords=None):
    """Per-channel QC metrics + tri-state outlier flag for ONE event type.

    Parameters
    ----------
    events_df : pandas.DataFrame
        Events for a single event_type. Uses columns: channel, start_time,
        end_time, max_amp, peak2peak_amp (falls back to max_amp-min_amp).
    scored_minutes : float or None
        Total minutes in the scored sleep stages for the recording, used as a
        single shared density denominator (a within-subject *relative* QC
        metric, comparable across channels). None => density NaN (greyed).
    artefact_intervals : list[tuple[float, float]] or None
        Whole-montage artefact windows for the %-in-global-artefact column.
    hard_z, soft_z, dead_frac : float
        Tunable thresholds (View -> Outlier threshold...).
    coords : dict or None
        ``{channel_label: (x, y)}`` topo projection. When given, the ``region``
        column is derived from real scalp position (so it agrees with the
        topography); otherwise it falls back to the EGI index heuristic.

    Returns
    -------
    pandas.DataFrame
        One row per channel: n, density, mean_amp, p95_amp, mean_p2p, max_p2p,
        pct_in_artefact, flag ('hard'|'soft'|'dead'|''), flag_reasons.

    Notes
    -----
    Amplitudes are Wonambi µV parameters on the detection-band-filtered signal,
    comparable only WITHIN one event type — hence the single-type slice. The
    flag is a "look here" heuristic, not a verdict: real slow-wave amplitude is
    frontally dominant, so expect physiological topography to flag; the topo
    card is the disambiguator.
    """
    import numpy as np
    import pandas as pd

    cols = ['channel', 'region', 'n', 'density', 'mean_amp', 'p95_amp',
            'mean_p2p', 'max_p2p', 'pct_in_artefact', 'flag', 'flag_reasons',
            'z_mean_amp', 'z_p95_amp', 'z_max_p2p', 'outlier_score']
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame(columns=cols)

    df = events_df.copy()
    # p2p: prefer the stored Wonambi column; fall back to max-min if absent.
    if 'peak2peak_amp' not in df.columns or df['peak2peak_amp'].isna().all():
        if {'max_amp', 'min_amp'}.issubset(df.columns):
            df['peak2peak_amp'] = df['max_amp'] - df['min_amp']
        else:
            df['peak2peak_amp'] = np.nan
    if 'max_amp' not in df.columns:
        df['max_amp'] = np.nan

    grp = df.groupby('channel', sort=True)
    agg = grp.agg(
        n=('channel', 'size'),
        mean_amp=('max_amp', 'mean'),
        p95_amp=('max_amp',
                 lambda s: np.nanpercentile(s, 95) if s.notna().any() else np.nan),
        mean_p2p=('peak2peak_amp', 'mean'),
        max_p2p=('peak2peak_amp', 'max'),
    ).reset_index()

    if scored_minutes and scored_minutes > 0:
        agg['density'] = agg['n'] / float(scored_minutes)
    else:
        agg['density'] = np.nan

    if artefact_intervals:
        iv = np.asarray([(float(a), float(b)) for a, b in artefact_intervals],
                        dtype=float)
        s = df['start_time'].to_numpy(dtype=float)
        e = (df['end_time'] if 'end_time' in df.columns
             else df['start_time']).to_numpy(dtype=float)
        overlap = ((s[:, None] < iv[:, 1][None, :]) &
                   (e[:, None] > iv[:, 0][None, :])).any(axis=1)
        pct = (pd.DataFrame({'channel': df['channel'].to_numpy(), 'ov': overlap})
               .groupby('channel')['ov'].mean().mul(100.0))
        agg['pct_in_artefact'] = agg['channel'].map(pct).fillna(0.0)
    else:
        agg['pct_in_artefact'] = 0.0

    def _robust_z(series):
        x = series.to_numpy(dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if not np.isfinite(mad) or mad == 0:
            return np.zeros(len(x))
        return np.abs(x - med) / (1.4826 * mad)

    flag = np.array([''] * len(agg), dtype=object)
    reasons = [[] for _ in range(len(agg))]
    zmap = {'mean_amp': np.zeros(len(agg)),
            'p95_amp': np.zeros(len(agg)),
            'max_p2p': np.zeros(len(agg))}
    if len(agg) >= 3:
        for metric in ('mean_amp', 'p95_amp', 'max_p2p'):
            z = _robust_z(agg[metric])
            zmap[metric] = z
            for i, zi in enumerate(z):
                if zi > hard_z:
                    flag[i] = 'hard'
                    reasons[i].append(f"{metric} z={zi:.1f}")
                elif zi > soft_z:
                    if flag[i] != 'hard':
                        flag[i] = 'soft'
                    reasons[i].append(f"{metric} z={zi:.1f}")
        n_med = np.nanmedian(agg['n'].to_numpy(dtype=float))
        if np.isfinite(n_med) and n_med > 0:
            for i, nv in enumerate(agg['n'].to_numpy()):
                if nv < dead_frac * n_med:
                    flag[i] = 'dead'
                    reasons[i] = [f"n={int(nv)} < {dead_frac:.0%} of median {n_med:.0f}"]
    agg['flag'] = flag
    agg['flag_reasons'] = ['; '.join(r) for r in reasons]
    agg['z_mean_amp'] = zmap['mean_amp']
    agg['z_p95_amp'] = zmap['p95_amp']
    agg['z_max_p2p'] = zmap['max_p2p']
    agg['outlier_score'] = np.maximum.reduce(
        [zmap['mean_amp'], zmap['p95_amp'], zmap['max_p2p']])
    # Region: coordinate-based when a montage is loaded (matches the topo),
    # else the EGI index-bucket fallback.
    agg['region'] = agg['channel'].map(
        lambda ch: _region_for_channel(ch, coords))

    return agg[cols]


# ============================================================================
# Scored-epoch table (variable-length epochs)
# ============================================================================

#: Epoch length of the synthetic grid used when no staging is loaded, and of
#: the grid built from a bare stage list (the pre-4.5 ``hypno=`` argument).
#: Never used when an annotation file supplies its own epochs.
DEFAULT_EPOCH_S = 30.0


class EpochTable:
    """Scored epochs as parallel ``starts`` / ``ends`` / ``stages``.

    Cut recordings carry epochs from 1 s to 30 s long, so no staging quantity
    in the review GUI may be derived as ``index * 30``; everything goes
    through :meth:`index_at` and :meth:`span`.

    Parameters
    ----------
    intervals : iterable of (float, float, str)
        ``(start_s, end_s, stage)`` per epoch; sorted by start here. Pieces
        with ``end <= start`` are dropped.
    """

    def __init__(self, intervals):
        rows = sorted(((float(a), float(b), '' if s is None else str(s))
                       for a, b, s in intervals if float(b) > float(a)),
                      key=lambda r: r[0])
        self.starts = np.array([r[0] for r in rows], dtype=float)
        self.ends = np.array([r[1] for r in rows], dtype=float)
        self.stages = [r[2] for r in rows]

    @classmethod
    def grid(cls, trec, epoch_s=DEFAULT_EPOCH_S, stages=None):
        """Fixed-length grid: one epoch per stage in ``stages`` when given,
        else ``ceil(trec / epoch_s)`` unstaged epochs (at least one)."""
        epoch_s = float(epoch_s)
        if stages:
            n = len(stages)
        else:
            n = max(1, int(np.ceil(float(trec or 0.0) / epoch_s)))
            stages = [''] * n
        return cls((i * epoch_s, (i + 1) * epoch_s, stages[i])
                   for i in range(n))

    @classmethod
    def from_annotations(cls, annotations):
        """Table from a loaded annotation object, ``None`` when it has no
        epochs. Prefers ``get_stage_intervals()``; falls back to the Wonambi
        epoch dicts (``'start'``, ``'end'``, ``'stage'``)."""
        if annotations is None:
            return None
        intervals = None
        getter = getattr(annotations, 'get_stage_intervals', None)
        if callable(getter):
            try:
                intervals = list(getter())
            except Exception as err:
                logger.debug(f"get_stage_intervals failed ({err}); "
                             f"reading the epochs directly")
                intervals = None
        if intervals is None:
            try:
                intervals = [(ep['start'], ep['end'], ep.get('stage'))
                             for ep in (annotations.epochs or [])]
            except Exception:
                intervals = None
        if intervals is None:
            # An object that only lists stage codes carries no times; the
            # only reading is the legacy fixed grid.
            try:
                stages = list(annotations.get_stages() or [])
            except Exception:
                return None
            return cls.grid(None, stages=stages) if stages else None
        table = cls(intervals)
        return table if len(table) else None

    def __len__(self):
        return int(self.starts.size)

    @property
    def end(self):
        return float(self.ends[-1]) if len(self) else 0.0

    @property
    def durations(self):
        return self.ends - self.starts

    def is_uniform(self, tol=1e-6):
        d = self.durations
        return bool(d.size == 0 or np.all(np.abs(d - d[0]) <= tol))

    def index_at(self, t):
        """Epoch whose start is the last one ``<= t``, clamped to the table
        (before the first epoch -> 0, past the last -> the last)."""
        n = len(self)
        if n == 0:
            return 0
        try:
            t = float(t)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(t):
            return 0
        i = int(np.searchsorted(self.starts, t, side='right')) - 1
        return max(0, min(i, n - 1))

    def indices_at(self, ts):
        """Vectorised epoch id per time: ``-1`` where ``t`` lies outside every
        ``[start, end)`` (before the first epoch, after the last, in a gap)."""
        ts = np.asarray(ts, dtype=float)
        if len(self) == 0:
            return np.full(ts.shape, -1, dtype=int)
        i = np.searchsorted(self.starts, ts, side='right') - 1
        ok = (i >= 0)
        ic = np.clip(i, 0, len(self) - 1)
        ok &= ts < self.ends[ic]
        return np.where(ok, ic, -1).astype(int)

    def span(self, i):
        """``(start_s, end_s)`` of epoch ``i`` (clamped to the table)."""
        if len(self) == 0:
            return 0.0, DEFAULT_EPOCH_S
        i = max(0, min(int(i), len(self) - 1))
        return float(self.starts[i]), float(self.ends[i])

    def stage(self, i):
        return self.stages[i] if 0 <= int(i) < len(self) else ''

    def stage_at(self, t):
        i = self.indices_at([t])[0]
        return self.stages[i] if i >= 0 else ''

    def snap(self, t0, t1):
        """Widen ``[t0, t1]`` outward to epoch edges."""
        a, b = sorted((float(t0), float(t1)))
        return self.span(self.index_at(a))[0], self.span(self.index_at(b))[1]

    def count_in(self, t0, t1, tol=1e-6):
        """Number of whole epochs inside ``[t0, t1]``."""
        a, b = sorted((float(t0), float(t1)))
        return int(np.count_nonzero((self.starts >= a - tol)
                                    & (self.ends <= b + tol)))


def _epoch_len_text(seconds):
    """Epoch length for the Epochs header: ``'1 s'``, ``'2.5 s'``,
    ``'5 min 12 s'`` (60 s and over), ``'2 min'``."""
    s = float(seconds)
    if s < 60:
        return f"{s:g} s"
    m, r = divmod(int(round(s)), 60)
    return f"{m} min {r} s" if r else f"{m} min"


def interpolated_defaults_note(selected, interp):
    """Status line when default channels include interpolated ones, else
    ``None``: ``'Showing Cz, Fz, Pz. Cz is interpolated (...)'``."""
    interp = set(interp or ())
    hit = [ch for ch in selected if ch in interp]
    if not hit:
        return None
    if len(hit) == 1:
        who = f"{hit[0]} is"
    else:
        who = f"{', '.join(hit[:-1])} and {hit[-1]} are"
    return (f"Showing {', '.join(selected)}. {who} interpolated "
            f"(reconstructed from neighbours by the cleaning pipeline).")


def annotation_recording_seconds(annotations):
    """Recording length in seconds from an annotation object, ``None`` when it
    cannot say. Uses ``recording_seconds()`` when the library provides it,
    else Wonambi's ``last_second``, else the end of the last epoch."""
    if annotations is None:
        return None
    fn = getattr(annotations, 'recording_seconds', None)
    if callable(fn):
        try:
            v = fn()
            if v:
                return float(v)
        except Exception:
            pass
    try:
        v = getattr(annotations, 'last_second', None)
        if v:
            return float(v)
    except Exception:
        pass
    table = EpochTable.from_annotations(annotations)
    return table.end if table is not None else None


def _as_epoch_table(epochs=None, hypno=None, trec=None):
    """Normalise the ways callers describe epochs to one :class:`EpochTable`:
    an ``EpochTable``, a list of ``(start, end, stage)``, a bare stage list
    (``hypno``, one 30 s epoch each, the pre-4.5 contract) or nothing (a
    synthetic 30 s grid over ``trec``)."""
    if isinstance(epochs, EpochTable):
        return epochs
    if epochs:
        return EpochTable(epochs)
    if hypno:
        return EpochTable.grid(None, stages=[str(s) for s in hypno])
    return EpochTable.grid(trec)


# ============================================================================
# Timeline Overview Widget
# ============================================================================

class TimelineWidget(PlotWidget):
    """Slim hypnogram strip for the Spot-check Events tab. Color-codes each
    scored epoch (its true [start, end] width) by stage via STAGE_COLOR (gray/magenta/blue/teal/green for
    Wake/REM/N1/N2/N3). Click anywhere to emit the row index of the event in
    current_events whose start_time is closest to the click. The currently-
    selected event is shown as a white vertical line with an accent-blue
    ▼ arrow above it (set_current_event_marker)."""

    event_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        _theme_plot(self)
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False); self.hideButtons()
        self.hideAxis('left')
        self.setMaximumHeight(80); self.setMinimumHeight(60)
        self._events_df = None
        self._trec = 0.0
        # persistent marker overlays — re-added after every clear()
        self._cursor = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen('w', width=1.4))
        self._cursor.setZValue(20); self._cursor.hide(); self.addItem(self._cursor)
        self._arrow = pg.ArrowItem(angle=-90, headLen=10, tipAngle=30,
                                   pen=pg.mkPen(THEME['accent'], width=1),
                                   brush=pg.mkBrush(THEME['accent']))
        self._arrow.setZValue(21); self._arrow.hide(); self.addItem(self._arrow)
        self.scene().sigMouseClicked.connect(self._on_click)

    def plot_timeline(self, events_df, current_index=-1, annotations=None,
                      recording_start_time=None):
        """Render hypnogram + current-event marker. Signature preserved for
        call sites (events_df + annotations + indices); the older event-
        markers path is intentionally dropped (perf + visual simplification)."""
        self._events_df = events_df
        hyp = None; trec = 0.0
        try:
            hyp = EpochTable.from_annotations(annotations)
            if hyp is not None:
                trec = max(hyp.end,
                           annotation_recording_seconds(annotations) or 0.0)
        except Exception:
            hyp = None
        if not trec and events_df is not None and len(events_df):
            try:
                trec = float(pd.to_numeric(events_df['end_time'],
                                           errors='coerce').max() or 0.0)
            except Exception:
                trec = 0.0
        self._trec = trec
        self.clear()
        # re-add persistent overlays (clear drops them)
        self.addItem(self._cursor); self.addItem(self._arrow)
        if hyp and trec:
            _draw_hypnogram(self, hyp, trec)
        if (events_df is not None and len(events_df)
                and 0 <= current_index < len(events_df)):
            try:
                t = float(events_df.iloc[current_index]['start_time'])
                self.set_current_event_marker(t)
            except Exception:
                pass

    def set_current_event_marker(self, t):
        """Place / move the white cursor + ▼ arrow at recording time t.
        Pass None to hide both."""
        if t is None or not np.isfinite(t):
            self._cursor.hide(); self._arrow.hide(); return
        self._cursor.setValue(float(t)); self._cursor.show()
        self._arrow.setPos(float(t), 4.4)
        self._arrow.show()

    def _on_click(self, ev):
        try:
            if ev.button() != Qt.LeftButton: return
            cx = float(self.getPlotItem().vb.mapSceneToView(ev.scenePos()).x())
        except Exception:
            return
        df = self._events_df
        if df is None or len(df) == 0: return
        st = pd.to_numeric(df['start_time'], errors='coerce')
        try:
            i = (st - cx).abs().idxmin()
            pos = int(df.index.get_loc(i))
        except Exception:
            return
        self.event_clicked.emit(pos)


# ============================================================================
# EEG Detail Plot Widget
# ============================================================================

class EEGDetailWidget(PlotWidget):
    """EEG detail plot with real-time filtering using PyQtGraph"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_event = None
        self.waveform_data = None
        self.sampling_rate = 500
        self.filter_enabled = False
        self.filter_settings = {'low': 0.5, 'high': 30}
        self.window_duration = 30.0  # Default 30-second window
        self.recording_start_time = None  # Store recording start time for HMS display
        
        # Lock the view: scroll-zoom / right-click menu / auto-range buttons
        # all break the canonical 30-s window. The window-duration spinbox is
        # the only legitimate way to change span. (setBackground/showGrid
        # dropped — _theme_plot is applied to this widget at construction
        # and owns the chrome.)
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseTracking(False)
        # Per-trace TextItem labels (see plot_event) show channel names, so
        # the left axis's numeric tick indices ("1 / 2 / 3") were redundant.
        self.hideAxis('left')
        self.setLabel('bottom', 'Time (s)', **{'font-size': '10pt', 'font-weight': 'bold'})

        # Disable auto-range for better control
        self.enableAutoRange(False, False)

        # Store plot items
        self.channel_curves = []
        self.channel_labels = []
        self.event_items = []
        # Channels last rendered by plot_event — toggle_filter replots with
        # the SAME list (no hardcoded fallback that masked row-change bugs).
        self._last_channels = []
    
    def plot_event(self, event_row, waveform_data, channels, context_seconds=None):
        """Plot EEG waveform for current event with configurable window duration"""
        self.current_event = event_row
        self.waveform_data = waveform_data
        
        # Clear previous plot
        self.clear()
        self.channel_curves = []
        self.channel_labels = []
        self.event_items = []
        
        if waveform_data is None:
            text = pg.TextItem('No waveform data available', anchor=(0.5, 0.5))
            self.addItem(text)
            return
        
        try:
            # Get sampling rate
            if hasattr(waveform_data, 'axis') and 's_freq' in waveform_data.axis:
                self.sampling_rate = waveform_data.axis['s_freq']
            
            # Use configurable window duration (default 30s), centered on event
            event_center = (event_row['start_time'] + event_row['end_time']) / 2
            half_window = self.window_duration / 2
            start_time = event_center - half_window
            end_time = event_center + half_window
            n_samples = waveform_data.data[0].shape[1]
            time_axis = np.linspace(start_time, end_time, n_samples)
            
            # Adaptive channel spacing based on number of channels
            num_channels = len(channels)
            if num_channels <= 3:
                y_spacing = 150  # Wide spacing for few channels
            elif num_channels <= 6:
                y_spacing = 100  # Medium spacing
            elif num_channels <= 10:
                y_spacing = 75   # Tighter spacing
            else:
                y_spacing = max(50, 300 / num_channels)  # Adaptive, minimum 50µV
            
            y_offset = 0
            
            channel_labels = waveform_data.axis['chan'][0]
            target_channel = event_row['channel']
            
            # Plot each channel
            for ch in channels:
                if ch in channel_labels:
                    ch_idx = np.where(channel_labels == ch)[0][0]
                    signal_data = waveform_data.data[0][ch_idx, :]
                    
                    # Apply filter if enabled
                    if self.filter_enabled:
                        signal_data = self.apply_filter(signal_data)
                    
                    # Highlight target channel with better visual distinction
                    is_target = ch == target_channel
                    color = (211, 47, 47) if is_target else (66, 66, 66)  # Red for target, dark gray for others
                    linewidth = 2.5 if is_target else 1.2
                    alpha = 255 if is_target else 153  # 1.0 vs 0.6
                    
                    # Plot channel trace
                    pen = mkPen(color=(*color, alpha), width=linewidth)
                    curve = self.plot(time_axis, signal_data + y_offset, pen=pen)
                    self.channel_curves.append(curve)
                    
                    # Channel label with background for better readability
                    # Position label at the baseline (y_offset) where the trace is centered
                    label_x = start_time - (end_time - start_time) * 0.02
                    bg_color = (255, 255, 255) if not is_target else (255, 235, 238)
                    border_color = color
                    
                    label = pg.TextItem(
                        ch,
                        anchor=(1, 0.5),  # Right-aligned, vertically centered
                        color=border_color,
                        fill=mkBrush(*bg_color, 230),
                        border=mkPen(*border_color, width=1.5 if is_target else 1)
                    )
                    # Position label at the same y_offset as the trace baseline
                    label.setPos(label_x, y_offset)
                    self.addItem(label)
                    self.channel_labels.append(label)
                    
                    # Add subtle horizontal reference line at baseline
                    baseline = pg.InfiniteLine(
                        pos=y_offset,
                        angle=0,
                        pen=mkPen((128, 128, 128), width=0.3, style=QtCore.Qt.DashLine)
                    )
                    self.addItem(baseline)
                    self.event_items.append(baseline)
                    
                    y_offset += y_spacing
            
            # Add vertical dashed lines every 30 seconds
            first_mark = int(start_time / 30) * 30
            if first_mark < start_time:
                first_mark += 30
            
            current_mark = first_mark
            while current_mark <= end_time:
                vline = pg.InfiniteLine(
                    pos=current_mark,
                    angle=90,
                    pen=mkPen((128, 128, 128), width=1, style=QtCore.Qt.DashLine)
                )
                self.addItem(vline)
                self.event_items.append(vline)
                current_mark += 30
            
            # Highlight event boundaries with improved visual design
            event_height = len(channels) * y_spacing
            
            # Event region (filled rectangle)
            event_region = pg.LinearRegionItem(
                values=[event_row['start_time'], event_row['end_time']],
                orientation='vertical',
                brush=mkBrush(255, 205, 210, 64),  # #FFCDD2 with alpha
                movable=False
            )
            # Remove default lines from LinearRegionItem
            event_region.lines[0].setPen(mkPen(None))
            event_region.lines[1].setPen(mkPen(None))
            self.addItem(event_region)
            self.event_items.append(event_region)
            
            # Vertical lines at event boundaries - solid lines
            start_line = pg.InfiniteLine(
                pos=event_row['start_time'],
                angle=90,
                pen=mkPen((211, 47, 47), width=2.5)
            )
            end_line = pg.InfiniteLine(
                pos=event_row['end_time'],
                angle=90,
                pen=mkPen((211, 47, 47), width=2.5)
            )
            self.addItem(start_line)
            self.addItem(end_line)
            self.event_items.extend([start_line, end_line])
            
            # Add event duration annotation at top
            mid_time = (event_row['start_time'] + event_row['end_time']) / 2
            duration_label = pg.TextItem(
                f"{event_row['duration']:.2f}s",
                anchor=(0.5, 1),
                color=(211, 47, 47),
                fill=mkBrush(255, 255, 255, 230),
                border=mkPen((211, 47, 47), width=1)
            )
            duration_label.setPos(mid_time, event_height - y_spacing/4)
            self.addItem(duration_label)
            self.event_items.append(duration_label)
            
            # Set axis ranges
            self.setXRange(start_time, end_time, padding=0)
            
            y_min = -y_spacing/2
            y_max = event_height - y_spacing/2
            if abs(y_max - y_min) < 1:  # If too close, expand range
                y_max = y_min + 100
            self.setYRange(y_min, y_max, padding=0)
            
            # Configure X-axis to show time in seconds (not HMS)
            x_axis = self.getAxis('bottom')
            x_axis.enableAutoSIPrefix(False)
            
            # Use simple time in seconds display
            window_span = end_time - start_time
            
            # Determine appropriate tick interval based on window size
            if window_span <= 10:
                tick_interval = 1.0
            elif window_span <= 30:
                tick_interval = 2.0
            elif window_span <= 60:
                tick_interval = 5.0
            else:
                tick_interval = 10.0
            
            # Generate tick labels at regular intervals
            num_ticks = int(window_span / tick_interval) + 1
            tick_labels = []
            for i in range(num_ticks):
                tick_pos = start_time + (i * tick_interval)
                if tick_pos <= end_time:
                    tick_labels.append((tick_pos, f"{tick_pos:.1f}"))
            
            if tick_labels:
                x_axis.setTicks([tick_labels])

            # Fixed scale bar at bottom-right: 50 µV vertical + 1 s horizontal.
            # Mouse interaction is disabled, so plot coords stay put — no
            # drift on zoom (which can't happen anyway).
            sb_x_right = end_time - (end_time - start_time) * 0.02   # 2% in
            sb_x_left = sb_x_right - 1.0                              # 1 s wide
            sb_y_bot = -y_spacing * 0.6                               # below first trace
            sb_y_top = sb_y_bot + 50.0                                # 50 µV
            sb_pen = pg.mkPen(THEME['text_2'], width=1.5)
            self.addItem(pg.PlotCurveItem(
                x=[sb_x_right, sb_x_right], y=[sb_y_bot, sb_y_top], pen=sb_pen))
            self.addItem(pg.PlotCurveItem(
                x=[sb_x_left, sb_x_right], y=[sb_y_bot, sb_y_bot], pen=sb_pen))
            t_uv = pg.TextItem('50 µV', anchor=(0, 0.5), color=THEME['text_2'])
            t_uv.setPos(sb_x_right + (end_time - start_time) * 0.005,
                         (sb_y_top + sb_y_bot) / 2.0)
            self.addItem(t_uv)
            t_s = pg.TextItem('1 s', anchor=(0.5, 0), color=THEME['text_2'])
            t_s.setPos((sb_x_left + sb_x_right) / 2.0, sb_y_bot - 4.0)
            self.addItem(t_s)
            self.event_items.extend([t_uv, t_s])

            # Remember channels for toggle_filter: replot must reuse them.
            self._last_channels = list(channels)
            
        except Exception as e:
            print(f"Error plotting event: {e}")
            import traceback
            traceback.print_exc()
    
    def set_window_duration(self, duration):
        """Set the window duration for event display"""
        self.window_duration = duration
        # Redraw current event if available
        if self.current_event is not None and self.waveform_data is not None:
            # Get current channels from the plot
            channels = [label.toPlainText() for label in self.channel_labels]
            if channels:
                self.plot_event(self.current_event, self.waveform_data, channels)
    
    def apply_filter(self, data):
        """Apply bandpass filter with proper handling for slow waves"""
        try:
            nyquist = self.sampling_rate / 2
            low = self.filter_settings['low'] / nyquist
            high = self.filter_settings['high'] / nyquist
            
            # Ensure normalized frequencies are within valid range
            low = max(0.001, min(low, 0.999))
            high = max(0.001, min(high, 0.999))
            
            if low >= high:
                return data
            
            # Use lower order filter (2nd order) for better slow wave preservation
            # Higher order filters can cause more phase distortion at low frequencies
            b, a = signal.butter(2, [low, high], btype='band')
            
            # Use filtfilt for zero-phase filtering (preserves waveform shape)
            filtered_data = signal.filtfilt(b, a, data)
            
            return filtered_data
        except Exception as e:
            print(f"Filter error: {e}")
            return data
    
    def toggle_filter(self, enabled):
        """Toggle the bandpass filter on/off — NEVER gates the trace render
        itself (the trace's existence is owned by update_eeg_plot /
        update_event_display). Reuses the last-rendered channel list so the
        button is a pure filter toggle, not a hidden re-render with a
        hardcoded fallback that masked row-change bugs."""
        self.filter_enabled = enabled
        if (self.current_event is not None and self.waveform_data is not None
                and self._last_channels):
            self.plot_event(self.current_event, self.waveform_data,
                             self._last_channels)


# ============================================================================
# Main GUI Window
# ============================================================================

# ============================================================================
# QC-by-outlier-triage widgets (Channels / Epochs surfaces)
# ============================================================================

_FLAG_BG = {
    'hard': QtGui.QColor(74, 35, 38),
    'soft': QtGui.QColor(72, 57, 28),
    'dead': QtGui.QColor(40, 44, 52),
}
_QC_COLS = [
    ('channel', 'Channel'), ('state', 'State'), ('region', 'Region'),
    ('n', 'n'), ('mean_amp', 'mean µV*'), ('p95_amp', 'p95 µV*'),
    ('mean_p2p', 'mean p2p*'), ('max_p2p', 'max p2p*'),
    ('flag', 'flag'),
    # population checks (4.6 per-event figures); see frontend/event_review.py
    ('pct_off_band', 'off-band %'), ('pct_low_prom', 'low prom. %'),
    ('pct_dur_floor', 'at floor %'), ('med_amp_ratio', 'amp/bg ×'),
    ('med_thresh_ratio', 'amp/thr ×'), ('checks_flag', 'checks'),
]
#: Column index of each key in the QC table (for hiding / header tooltips).
_QC_COL_INDEX = {k: i for i, (k, _) in enumerate(_QC_COLS)}

# Selection-state glyph vocabulary, shared by the State column + the tray.
_STATE_ART_COLOR = '#f85149'   # ⚑ channel artefact (verdict drop/channel_artefact)
_STATE_RD_COLOR = '#5a8fce'    # ↻ queued for re-detection
_STATE_ART_GLYPH = '⚑ artefact'
_STATE_RD_GLYPH = '↻ re-detect'
# Note: density / pct_in_artefact / verdict are still computed and used —
# verdict still shades rows (BackgroundRole) and drives Drop/Keep; density
# still appears in the Epochs-tab title — they're just hidden from the table.
# amp columns whose cell is heat-shaded by the matching robust z-score
_HEAT_Z = {'mean_amp': 'z_mean_amp', 'p95_amp': 'z_p95_amp',
           'max_p2p': 'z_max_p2p'}


def _heat_bg(z):
    """Cell tint growing with robust |z| (amber → red), like the mockup."""
    try:
        z = float(z)
    except Exception:
        return None
    if not np.isfinite(z) or z <= 0.5:
        return None
    t = max(0.0, min(1.0, (z - 0.5) / (6.0 - 0.5)))
    a = 0.10 + t * 0.55
    if z > 3.5:
        base = (248, 81, 73)
    elif z > 2.0:
        base = (210, 153, 34)
    else:
        base = (88, 116, 160)
        a *= 0.5
    bg = (11, 14, 19)  # window bg, for flat blend onto opaque cell
    return QtGui.QColor(*[int(bg[i] + (base[i] - bg[i]) * a) for i in range(3)])


_STATUS_TEXT = {'': 'untriaged', 'keep': 'kept', 'drop': 'channel artefact',
                'channel_artefact': 'channel artefact'}


def _hms(seconds):
    try:
        s = int(round(float(seconds)))
    except Exception:
        return "—"
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _h_label(text):
    """Small uppercase section header used in the docks."""
    q = QLabel(text)
    q.setStyleSheet("color:#6b7585;font-size:10px;font-weight:600;"
                    "letter-spacing:0.05em;margin-top:6px;")
    return q


def _short_stage(s):
    """Normalise a Wonambi/EEGLAB sleep-stage label to one of
    N1/N2/N3/REM/W. Returns '—' when the stage is missing or unrecognised."""
    if s is None:
        return '—'
    m = {'Wake': 'W', 'W': 'W', 'REM': 'REM',
         'NREM1': 'N1', 'N1': 'N1', 'Stage1': 'N1',
         'NREM2': 'N2', 'N2': 'N2', 'Stage2': 'N2',
         'NREM3': 'N3', 'N3': 'N3', 'Stage3': 'N3'}
    return m.get(str(s).strip(), '—')


# Human-readable topo metric labels (title + combo), keyed by df column.
TOPO_METRIC_LABEL = {'density': 'density (ev/min)',
                     'mean_amp': 'mean amp (µV)',
                     'max_p2p': 'max p2p (µV)'}
TOPO_METRIC_LABEL.update({c: v[1] for c, v in CHECK_COLUMNS.items()})

# Shared "impossible physiological scale" red used by both worst lists.
IMPOSSIBLE_AMP_COLOR = '#f85149'


def _amp_cell(amp):
    """Return ``(display_text, is_impossible)`` for a µV amplitude — the single
    scale rule shared by the global worst-events list and the per-channel
    worst-epochs list. >1000 µV renders as ``kµV`` with a trailing ⚠
    (physiologically impossible peak-to-peak → almost certainly artefact)."""
    if amp > 1000:
        return f"{amp / 1000:.1f} kµV ⚠", True
    return f"{int(round(amp))} µV", False


def _artefact_tooltip(amp):
    """Tooltip for an impossible-scale (>1000 µV) amplitude row."""
    return (f"{amp:.0f} µV peak-to-peak — exceeds physiological scale "
            f"(>1000 µV); almost certainly artefact.")


def _eeglab_polar_to_xy(chanlocs):
    """Convert EEGLAB polar chanlocs to 2-D topoplot coordinates.

    ``chanlocs`` is a list of ``{'label', 'theta', 'radius'}`` dicts (theta in
    degrees, radius normalised — EEGLAB convention). Returns ``{label: (x, y)}``
    using the nose-up topoplot projection (x = r·sin θ, y = r·cos θ).
    """
    coords = {}
    for ch in chanlocs or []:
        try:
            th = np.deg2rad(float(ch['theta']))
            rd = float(ch['radius'])
            coords[str(ch['label'])] = (rd * np.sin(th), rd * np.cos(th))
        except (KeyError, TypeError, ValueError):
            continue
    return coords


class ChannelQCModel(QAbstractTableModel):
    """Per-channel QC table for ONE event type. Numeric sort via UserRole.

    (* amplitude columns are Wonambi µV on the detection-band signal —
    comparable only within one event type.)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame(columns=[c[0] for c in _QC_COLS])
        self._records = []   # list[dict] — O(1) cell access (no pandas .iloc)
        self._redetect = set()   # channels queued for re-detection
        self._event_type = ''    # active event type (for State tooltip)
        # population-check context: recorded?, method, ratio allowed,
        # header tooltips (set by ChannelQCWidget.set_checks_context)
        self._checks = {'recorded': False, 'method': '', 'ratio': True,
                        'tips': {}, 'pending': False}

    def set_data(self, qc_df, verdicts=None, redetect=None, event_type=''):
        self.beginResetModel()
        self._redetect = {str(c) for c in (redetect or ())}
        self._event_type = str(event_type or '')
        df = qc_df.copy() if qc_df is not None else pd.DataFrame()
        if 'verdict' not in df.columns:
            df['verdict'] = ''
        if verdicts:
            df['verdict'] = df['channel'].map(verdicts).fillna(df['verdict'])
        self.df = df.reset_index(drop=True)
        # Back the model with a plain records list: the QTableView + sort proxy
        # call data() thousands of times per reset, and pandas .iloc-per-call
        # was ~340 ms for 257 rows. dict access is O(1).
        self._records = self.df.to_dict('records')
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent=QModelIndex()):
        return len(_QC_COLS)

    def channel_at(self, row):
        if 0 <= row < len(self._records):
            return str(self._records[row].get('channel', ''))
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._records)):
            return QVariant()
        rec = self._records[index.row()]
        key = _QC_COLS[index.column()][0]
        val = rec.get(key, '')
        if key == 'state':
            # Selection state from active-event-type verdict + re-detect set.
            art = str(rec.get('verdict', '')) in ('drop', 'channel_artefact')
            rd = str(rec.get('channel', '')) in self._redetect
            if role == Qt.DisplayRole:
                parts = []
                if art:
                    parts.append(_STATE_ART_GLYPH)
                if rd:
                    parts.append(_STATE_RD_GLYPH)
                return '  '.join(parts)
            if role == Qt.UserRole:               # both=3, art=2, rd=1, none=0
                return (2 if art else 0) + (1 if rd else 0)
            if role == Qt.ForegroundRole:
                return QtGui.QColor(_STATE_ART_COLOR if art else _STATE_RD_COLOR)
            if role == Qt.ToolTipRole:
                tips = []
                if art:
                    tips.append(
                        f"Marked as channel artefact for {self._event_type}. "
                        "Excluded from export. Unmark from the button or the "
                        "Selection tray.")
                if rd:
                    tips.append(
                        "Queued for re-detection. Build the re-detect request "
                        "to export the channel list.")
                return '\n'.join(tips)
            # BackgroundRole falls through to the shared verdict/heat wash.
        if key in CHECK_COLUMNS or key == 'checks_flag':
            out = self._check_cell(rec, key, role)
            if out is not None:
                return out
            if role != Qt.BackgroundRole:
                return QVariant()
        if role == Qt.DisplayRole:
            if key == 'verdict':
                return _STATUS_TEXT.get(str(val or ''), str(val))
            if key == 'flag':
                fl = str(val or '')
                if fl in ('hard', 'soft'):
                    return f"{fl} z={float(rec.get('outlier_score', 0)):.1f}"
                return fl or 'ok'
            if key in ('density', 'mean_amp', 'p95_amp', 'mean_p2p',
                       'max_p2p', 'pct_in_artefact'):
                try:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return '—'
                    return f"{float(val):.2f}"
                except Exception:
                    return '—'
            return '' if val is None else str(val)
        if role == Qt.UserRole:  # numeric sort key
            if key == 'flag':
                try:
                    return float(rec.get('outlier_score', 0))
                except Exception:
                    return 0.0
            try:
                return float(val)
            except Exception:
                return str(val)
        if role == Qt.BackgroundRole:
            if key in _HEAT_Z:
                c = _heat_bg(rec.get(_HEAT_Z[key]))
                if c is not None:
                    return c
            if str(rec.get('verdict', '')) in ('drop', 'channel_artefact'):
                # muted RED wash — distinct from dead's neutral (40,44,52)
                return QtGui.QColor(48, 34, 36)
            flag = str(rec.get('flag', ''))
            if key in ('flag', 'verdict') and flag in _FLAG_BG:
                return _FLAG_BG[flag]
        if role == Qt.ForegroundRole and key == 'flag':
            flag = str(rec.get('flag', ''))
            return {'hard': QtGui.QColor(248, 81, 73),
                    'soft': QtGui.QColor(210, 153, 34),
                    'dead': QtGui.QColor(155, 166, 181)}.get(
                flag, QtGui.QColor(63, 185, 80))
        if role == Qt.ToolTipRole and key == 'flag':
            return str(rec.get('flag_reasons', ''))
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _QC_COLS[section][1]
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            tip = self._checks['tips'].get(_QC_COLS[section][0])
            if tip:
                return tip
        return QVariant()

    def _check_cell(self, rec, key, role):
        """Display / sort / tooltip / colour of a population-check cell."""
        ck = self._checks
        if key == 'checks_flag':
            fl = str(rec.get('checks_flag', '') or '')
            if role == Qt.DisplayRole:
                return fl.upper()
            if role == Qt.UserRole:
                return {'hard': 2, 'soft': 1}.get(fl, 0)
            if role == Qt.ForegroundRole and fl in ('hard', 'soft'):
                return (QtGui.QColor(248, 81, 73) if fl == 'hard'
                        else QtGui.QColor(210, 153, 34))
            if role == Qt.BackgroundRole and fl in _FLAG_BG:
                return _FLAG_BG[fl]
            if role == Qt.ToolTipRole:
                return str(rec.get('checks_reasons', '') or '') or None
            return None
        val = rec.get(key)
        missing = _er._finite(val) is None
        if role == Qt.DisplayRole:
            if ck.get('pending') and not ck.get('recorded'):
                return '…'
            return fmt_check(key, val)
        if role == Qt.UserRole:
            return -1.0 if missing else float(val)
        if role == Qt.ToolTipRole:
            if not ck.get('recorded'):
                if ck.get('figures_off'):
                    return ('Not computed for this run: event figures were '
                            'switched off when it was detected.')
                if key == 'med_thresh_ratio' and method_has_ratio(ck['method']):
                    return THRESHOLD_NOT_RECORDED + '.'
                return ('Not recorded for this run (detected with 4.5 or '
                        'earlier).')
            if key == 'med_thresh_ratio' and not ck.get('ratio'):
                return no_ratio_note(ck['method'])
            n = rec.get(CHECK_N[key])
            if missing and n is not None and n < CHECK_MIN_N:
                return TOO_FEW_TIP.format(n=int(n))
            if key == 'pct_off_band' and rec.get('n_no_peak') is not None:
                return (f"Events with no spectral peak are not counted: "
                        f"{int(rec.get('n_no_peak') or 0)} on this channel.")
            return None
        if role == Qt.BackgroundRole:
            return _heat_bg(rec.get('z_' + key)) if not missing else None
        return None


class EventDecisionPanel(QWidget):
    """EVENT + DECISION block of the right dock (UX spec sections 4-5).

    Renders what the main window computes: the rows from
    :func:`frontend.event_review.build_event_rows`, the current decision, the
    progress line and the hint. It holds no decision logic; its signals go to
    ``EventReviewGUI``, which writes, arms and undoes.

    For tests: :meth:`row_keys` lists the visible rows in order and
    :meth:`row_text` returns a row's value and sub-lines as plain text.
    """

    decisionClicked = pyqtSignal(str)   # 'accept' | 'reject' | 'unsure'
    reasonChosen = pyqtSignal(str)      # token; '' = No reason (mouse pick)
    commentSubmitted = pyqtSignal()     # Enter in the comment field
    commentEscape = pyqtSignal()        # Esc in the comment field
    clearClicked = pyqtSignal()
    prevClicked = pyqtSignal()
    nextClicked = pyqtSignal()
    autoAdvanceToggled = pyqtSignal(bool)
    openReportClicked = pyqtSignal()
    revisitClicked = pyqtSignal()

    EMPTY_TEXT = ('Click an event band, or press ] for the next unreviewed '
                  'event.')
    EMPTY_SAMPLE_TEXT = 'Press ] for the next sample event.'
    HINT = 'A accept · R reject · U unsure · ] next · Ctrl+Z undo'
    HINT_SAMPLE = ('A accept · R reject · U unsure · ] next in sample · '
                   '} next on channel · Ctrl+Z undo')

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(_h_label("EVENT"))
        self.empty_lbl = QLabel(self.EMPTY_TEXT)
        self.empty_lbl.setWordWrap(True)
        self.empty_lbl.setStyleSheet("color:#888888;font-size:11px;")
        lay.addWidget(self.empty_lbl)
        self.note_lbl = QLabel("")
        self.note_lbl.setWordWrap(True)
        self.note_lbl.setStyleSheet("color:#888888;font-size:11px;")
        self.note_lbl.setVisible(False)
        lay.addWidget(self.note_lbl)
        self._rows_box = QWidget()
        self._grid = QtWidgets.QGridLayout(self._rows_box)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(1)
        lay.addWidget(self._rows_box)
        self._rows = []           # list of dict rows as given
        self._row_widgets = {}    # key -> (label, value, [sub labels])

        lay.addWidget(_h_label("DECISION"))
        brow = QHBoxLayout()
        self.btn = {}
        for dec, text in (('accept', 'Accept  A'), ('reject', 'Reject  R'),
                          ('unsure', 'Unsure  U')):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, d=dec: self.decisionClicked.emit(d))
            brow.addWidget(b)
            self.btn[dec] = b
        lay.addLayout(brow)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Reason"))
        self.reason_combo = QComboBox()
        self.reason_combo.setFocusPolicy(Qt.ClickFocus)
        self.reason_combo.addItem('No reason', '')
        for key, token, label, _short, tip in _er.REASONS:
            self.reason_combo.addItem(f"{key}  {label}" if key else label,
                                      token)
            if tip:
                self.reason_combo.setItemData(self.reason_combo.count() - 1,
                                              tip, Qt.ToolTipRole)
        self.reason_combo.activated.connect(
            lambda i: self.reasonChosen.emit(
                str(self.reason_combo.itemData(i) or '')))
        rrow.addWidget(self.reason_combo, 1)
        lay.addLayout(rrow)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Comment"))
        self.comment = QLineEdit()
        self.comment.setMaxLength(500)
        self.comment.setPlaceholderText(
            'Comment (optional) — C to type, Enter to save')
        self.comment.returnPressed.connect(self.commentSubmitted.emit)
        self.comment.installEventFilter(self)
        crow.addWidget(self.comment, 1)
        lay.addLayout(crow)
        cur = QHBoxLayout()
        cur.addWidget(QLabel("Current"))
        self.current_lbl = QLabel("Not reviewed")
        self.current_lbl.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:12px;")
        cur.addWidget(self.current_lbl, 1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFlat(True)
        self.clear_btn.setFocusPolicy(Qt.NoFocus)
        self.clear_btn.clicked.connect(self.clearClicked.emit)
        cur.addWidget(self.clear_btn)
        lay.addLayout(cur)
        self.current_sub = QLabel("")
        self.current_sub.setWordWrap(True)
        self.current_sub.setStyleSheet("color:#888888;font-size:11px;")
        lay.addWidget(self.current_sub)
        nrow = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev  [")
        self.next_btn = QPushButton("Next  ] ▶")
        for b, sig, tip in ((self.prev_btn, self.prevClicked,
                             'Previous unreviewed event on this channel'),
                            (self.next_btn, self.nextClicked,
                             'Next unreviewed event on this channel')):
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(sig.emit)
            nrow.addWidget(b)
        lay.addLayout(nrow)
        self.auto_chk = QCheckBox("Go to next unreviewed after deciding")
        self.auto_chk.setFocusPolicy(Qt.NoFocus)
        self.auto_chk.setChecked(True)
        self.auto_chk.toggled.connect(self.autoAdvanceToggled.emit)
        lay.addWidget(self.auto_chk)
        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet("font-size:11px;")
        lay.addWidget(self.progress_lbl)
        self.hint_lbl = QLabel(self.HINT)
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setStyleSheet("color:#888888;font-size:11px;")
        lay.addWidget(self.hint_lbl)
        # end of the review sample (replaces the hint line)
        self.end_box = QWidget()
        eb = QVBoxLayout(self.end_box)
        eb.setContentsMargins(0, 0, 0, 0)
        self.end_lbl = QLabel("")
        self.end_lbl.setWordWrap(True)
        eb.addWidget(self.end_lbl)
        er_row = QHBoxLayout()
        self.btn_open_report = QPushButton("Open precision report")
        self.btn_open_report.setFocusPolicy(Qt.NoFocus)
        self.btn_open_report.clicked.connect(self.openReportClicked.emit)
        self.btn_revisit = QPushButton("")
        self.btn_revisit.setFocusPolicy(Qt.NoFocus)
        self.btn_revisit.clicked.connect(self.revisitClicked.emit)
        er_row.addWidget(self.btn_open_report)
        er_row.addWidget(self.btn_revisit)
        eb.addLayout(er_row)
        self.end_box.setVisible(False)
        lay.addWidget(self.end_box)
        self._sample_mode = False
        self.set_empty(self.EMPTY_TEXT)

    # ---- comment field keys --------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.comment and ev.type() == QtCore.QEvent.KeyPress \
                and ev.key() == Qt.Key_Escape:
            self.commentEscape.emit()
            return True
        return super().eventFilter(obj, ev)

    # ---- rows ------------------------------------------------------------
    _LEVEL_STYLE = {'warn': "color:#e0a334;font-weight:600;",
                    'bad': "color:#e0533f;font-weight:600;",
                    'muted': "color:#888888;"}

    def _clear_grid(self):
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._row_widgets = {}

    def set_rows(self, rows):
        """Show ``rows`` (``build_event_rows`` dicts) in order."""
        self._clear_grid()
        self._rows = list(rows or [])
        r = 0
        for row in self._rows:
            lab = QLabel(row['label'])
            lab.setStyleSheet("color:#b8b8b8;font-size:11px;")
            val = QLabel(str(row['value']))
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setWordWrap(True)
            val.setStyleSheet(
                "font-family:'IBM Plex Mono',monospace;font-size:12px;"
                + self._LEVEL_STYLE.get(row.get('level'), "color:#e5e5e5;"))
            if row.get('tooltip'):
                val.setToolTip(row['tooltip'])
                lab.setToolTip(row['tooltip'])
            self._grid.addWidget(lab, r, 0, Qt.AlignTop)
            self._grid.addWidget(val, r, 1)
            subs = []
            for text in row.get('sub', []):
                r += 1
                sl = QLabel(text)
                sl.setWordWrap(True)
                sl.setAlignment(Qt.AlignRight)
                warn_words = ('low prominence', 'barely', 'at the floor',
                              'outside run limits', 'at the ceiling', 'fails')
                hot = any(w in text for w in warn_words)
                sl.setStyleSheet("font-size:11px;" + (
                    "color:#e0a334;" if hot else "color:#888888;"))
                self._grid.addWidget(sl, r, 1)
                subs.append(sl)
            self._row_widgets[row['key']] = (lab, val, subs)
            r += 1
        self._rows_box.setVisible(bool(self._rows))
        self.empty_lbl.setVisible(not self._rows)

    def row_keys(self):
        return [row['key'] for row in self._rows]

    def row_text(self, key):
        """Value and sub-lines of one row as plain text (``None`` if absent)."""
        for row in self._rows:
            if row['key'] == key:
                return '\n'.join([str(row['value'])] + list(row['sub']))
        return None

    def row(self, key):
        return next((r for r in self._rows if r['key'] == key), None)

    # ---- states ----------------------------------------------------------
    def set_empty(self, message):
        self.set_rows([])
        if getattr(self, '_sample_mode', False) and \
                message == self.EMPTY_TEXT:
            message = self.EMPTY_SAMPLE_TEXT
        self.empty_lbl.setText(message)
        self.note_lbl.setVisible(False)
        self.set_current('Not reviewed', [])
        self.set_armed(None)
        self.set_controls_enabled(False)

    def set_note(self, text):
        self.note_lbl.setText(text or '')
        self.note_lbl.setVisible(bool(text))

    def set_controls_enabled(self, on):
        for b in list(self.btn.values()) + [self.clear_btn]:
            b.setEnabled(bool(on))
        self.reason_combo.setEnabled(bool(on))
        self.comment.setEnabled(bool(on))

    def set_current(self, text, subs, tooltip=''):
        self.current_lbl.setText(text)
        self.current_lbl.setToolTip(tooltip or '')
        self.current_sub.setText('\n'.join(subs or []))
        self.current_sub.setVisible(bool(subs))

    def set_decision(self, decision):
        for d, b in self.btn.items():
            b.setChecked(d == decision)

    def set_armed(self, decision, hint=None):
        """Show an armed Reject/Unsure (or none) and its hint."""
        for d, b in self.btn.items():
            b.setStyleSheet("border:2px solid #5a8fce;" if d == decision
                            else "")
        self.hint_lbl.setText(hint or (self.HINT_SAMPLE
                                       if getattr(self, '_sample_mode', False)
                                       else self.HINT))

    def set_reason(self, token):
        i = self.reason_combo.findData(token or '')
        self.reason_combo.setCurrentIndex(max(0, i))

    def set_progress(self, text):
        self.progress_lbl.setText(text or '')

    def set_sample_mode(self, on):
        """Sample-mode labels: checkbox, hint, Prev/Next tooltips."""
        self._sample_mode = bool(on)
        self.auto_chk.setText("Go to next sample event after deciding" if on
                              else "Go to next unreviewed after deciding")
        self.hint_lbl.setText(self.HINT_SAMPLE if on else self.HINT)
        tip = ("Previous / next undecided sample event" if on
               else "Previous / next unreviewed event on this channel")
        self.prev_btn.setToolTip(tip)
        self.next_btn.setToolTip(tip)
        if not self._rows:
            self.empty_lbl.setText(self.EMPTY_SAMPLE_TEXT if on
                                   else self.EMPTY_TEXT)
        if not on:
            self.set_end(None)

    def set_end(self, text, n_unsure=0):
        """Show the end-of-sample block (``text``) or hide it (``None``)."""
        on = bool(text)
        self.end_box.setVisible(on)
        self.hint_lbl.setVisible(not on)
        if on:
            self.end_lbl.setText(text)
            self.btn_revisit.setText(f"Revisit the {n_unsure} unsure")
            self.btn_revisit.setVisible(n_unsure > 0)


class ChannelDetailDock(QWidget):
    """Right-dock: a "worst epochs" list and a topography card (empty-state
    default; scipy griddata when channel coords are available).
    Clicking a row in the worst-epochs list emits gotoEpochRequested(idx),
    which the main window wires to switch to the Epochs tab and call
    EpochsPanel._goto_epoch."""

    loadMontageRequested = pyqtSignal()
    gotoEpochRequested = pyqtSignal(int)          # epoch idx on current channel
    gotoChannelEpochRequested = pyqtSignal(str, float)  # channel, event start_t
    channelPicked = pyqtSignal(str)             # topo electrode clicked -> select
    unmarkArtefactRequested = pyqtSignal(int)   # interval id (× button)
    checkLinkActivated = pyqtSignal(str, str)   # channel, check column

    def __init__(self, parent=None):
        super().__init__(parent)
        self._coords = None  # {channel: (x, y)}
        self.topo_metric = 'density'
        self._event_type = 'slow_wave'
        self._colorbar = None
        lay = QVBoxLayout(self)

        # --- topography card -------------------------------------------
        # Live scalp interpolation of the active QC metric. Coords come from
        # the loaded EEGLAB .set (preferred) or a label,x,y CSV fallback; the
        # "Load montage…" button appears ONLY when no coords are available.
        topo_row = QHBoxLayout()
        topo_row.addWidget(QLabel("Metric:"))
        self.topo_combo = QComboBox()
        for col, label in (('density', 'density (ev/min)'),
                           ('mean_amp', 'mean amp (µV)'),
                           ('max_p2p', 'max p2p (µV)')):
            self.topo_combo.addItem(label, col)
        self.topo_combo.currentIndexChanged.connect(self._on_metric)
        topo_row.addWidget(self.topo_combo, 1)
        self.load_montage_btn = QPushButton("Load montage…")
        self.load_montage_btn.clicked.connect(self.loadMontageRequested.emit)
        topo_row.addWidget(self.load_montage_btn)
        lay.addLayout(topo_row)

        # What the density in this dock is divided by. Always visible: a
        # density is not interpretable without the definition of its
        # denominator, and the two contributions to that denominator (the
        # detection run's exclusion set, and any marks the reviewer has added
        # this session) must never be conflated.
        self.mask_caption = QLabel("")
        self.mask_caption.setWordWrap(True)
        self.mask_caption.setStyleSheet("color:#6b7585;font-size:11px;")
        self.mask_caption.setToolTip(
            "Time marked with these events was not searched for events and is "
            "not counted in the denominator. Read from the detection run that "
            "produced the events in view.")
        lay.addWidget(self.mask_caption)

        self.topo = pg.PlotWidget(title="Topography")
        _theme_plot(self.topo)
        self.topo.setMaximumHeight(210)
        self.topo.hideAxis('bottom')
        self.topo.hideAxis('left')
        lay.addWidget(self.topo)
        # population-check caption (not recorded / which run)
        self.checks_caption = QLabel("")
        self.checks_caption.setWordWrap(True)
        self.checks_caption.setStyleSheet("color:#6b7585;font-size:11px;")
        lay.addWidget(self.checks_caption)
        self._checks_recorded = None

        # --- global worst events (ALL channels) ------------------------
        # Read-only ranking of the most extreme events for the current event
        # type across the whole montage. Refreshes on event-type / filter
        # change (NOT on channel selection). Click a row -> jump to that
        # channel + epoch. Populated by the main window via set_global_worst.
        self.global_worst_hdr = _h_label("WORST EVENTS — ALL CHANNELS")
        lay.addWidget(self.global_worst_hdr)
        self.global_worst_list = QtWidgets.QListWidget()
        self.global_worst_list.setMaximumHeight(250)
        self.global_worst_list.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:11px;")
        self.global_worst_list.itemClicked.connect(self._on_global_worst_click)
        lay.addWidget(self.global_worst_list)

        # --- strong divider: everything below is scoped to ONE channel ---
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setStyleSheet("color:#3a4250;background:#3a4250;"
                              "min-height:2px;max-height:2px;margin:8px 0;")
        lay.addWidget(divider)

        # --- selected-channel header -----------------------------------
        lay.addWidget(_h_label("SELECTED CHANNEL"))
        self.title = QLabel("—")
        self.title.setStyleSheet("font:600 15px 'IBM Plex Mono',monospace;"
                                 "color:#e7eef7;")
        lay.addWidget(self.title)
        self.subtitle = QLabel("")
        self.subtitle.setStyleSheet("color:#6b7585;font-size:11px;")
        lay.addWidget(self.subtitle)
        # population-check line: flagged phrases are links into the Epochs
        # tab with a matching event filter
        self.checks_line = QLabel("")
        self.checks_line.setWordWrap(True)
        self.checks_line.setTextFormat(Qt.RichText)
        self.checks_line.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.checks_line.setStyleSheet("font-size:12px;")
        self.checks_line.linkActivated.connect(self._on_check_link)
        self.checks_line.linkHovered.connect(self._on_check_hover)
        lay.addWidget(self.checks_line)
        self.checks_hint = QLabel(_er.DOCK_HINT)
        self.checks_hint.setWordWrap(True)
        self.checks_hint.setStyleSheet("color:#888888;font-size:11px;")
        self.checks_hint.setVisible(False)
        lay.addWidget(self.checks_hint)
        self._check_items = []
        self._check_channel = None

        # --- the selected event and its decision ------------------------
        self.event_panel = EventDecisionPanel()
        lay.addWidget(self.event_panel)

        # --- per-channel worst epochs ----------------------------------
        # Top-12 epochs sorted by (n_outliers desc, max_amp desc), filtered
        # to n_outliers > 0. Click a row -> gotoEpochRequested(idx); main
        # window switches to Epochs tab and pages there.
        self.worst_hdr = _h_label("WORST EPOCHS ON —")
        lay.addWidget(self.worst_hdr)
        self.worst_list = QtWidgets.QListWidget()
        self.worst_list.setMaximumHeight(200)
        self.worst_list.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:11px;")
        self.worst_list.itemClicked.connect(self._on_worst_click)
        lay.addWidget(self.worst_list)

        # --- robust z readout ------------------------------------------
        lay.addWidget(_h_label("ROBUST Z-SCORES"))
        zgrid = QtWidgets.QGridLayout()
        self._z_labels = {}
        for i, (k, lbl) in enumerate((('z_mean_amp', 'mean amp'),
                                      ('z_p95_amp', 'p95 amp'),
                                      ('z_max_p2p', 'max p2p'))):
            name = QLabel(lbl)
            name.setStyleSheet("color:#9ba6b5;")
            val = QLabel("—")
            val.setAlignment(Qt.AlignRight)
            val.setStyleSheet("font-family:'IBM Plex Mono',monospace;")
            zgrid.addWidget(name, i, 0)
            zgrid.addWidget(val, i, 1)
            self._z_labels[k] = val
        lay.addLayout(zgrid)

        # --- marked artefact (compact, channel-scoped) -----------------
        # Replaces the standalone EpochsPanel "MARKED ARTEFACT RANGES"
        # panel. Single-line rows: HH:MM:SS–HH:MM:SS + ep count + × button.
        self.marked_hdr = _h_label("MARKED ARTEFACT (0)")
        lay.addWidget(self.marked_hdr)
        self._marked_box = QWidget()
        self._marked_layout = QVBoxLayout(self._marked_box)
        self._marked_layout.setContentsMargins(0, 0, 0, 0)
        self._marked_layout.setSpacing(2)
        lay.addWidget(self._marked_box)

        self._qc_df = None
        self._epochs = None   # EpochTable of the loaded annotations, or None
        self._render_topo_empty()
        lay.addStretch()

    def set_epoch_table(self, epochs):
        """Scored-epoch table used to turn marked times into epoch ids and
        epoch counts; ``None`` falls back to a synthetic 30 s grid."""
        self._epochs = epochs if isinstance(epochs, EpochTable) or not epochs \
            else EpochTable(epochs)

    def _table_for(self, t):
        """The loaded epoch table, or a synthetic 30 s grid reaching ``t``."""
        if self._epochs is not None and len(self._epochs):
            return self._epochs
        return EpochTable.grid(float(t) + DEFAULT_EPOCH_S)

    def set_marked(self, marked, channel=None, total=None):
        """Rebuild the compact channel-scoped marked-artefact rows.
        Time label jumps the Epochs window; × emits unmarkArtefactRequested.
        Header shows '(N on {ch} · {total} total)' so a reviewer doesn't
        think marks on other channels vanished."""
        while self._marked_layout.count():
            it = self._marked_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        marked = list(marked or [])
        if channel is not None and total is not None:
            self.marked_hdr.setText(
                f"MARKED ARTEFACT ({len(marked)} on {channel} · {total} total)")
        else:
            self.marked_hdr.setText(f"MARKED ARTEFACT ({len(marked)})")
        for m in marked:
            t0, t1 = float(m['start_time']), float(m['end_time'])
            mid = int(m['id'])
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            dur = t1 - t0
            table = self._table_for(t1)
            n_ep = table.count_in(t0, t1)
            if n_ep >= 1:
                dtxt = f"{n_ep} ep"
            elif dur >= 1:
                dtxt = f"{dur:.1f}s sub"
            else:
                dtxt = f"{int(dur * 1000)}ms sub"
            lbl = QPushButton(f"{_hms(t0)}–{_hms(t1)}  ({dtxt})")
            lbl.setFlat(True)
            lbl.setStyleSheet(
                "text-align:left;color:#d6dee8;"
                "font-family:'IBM Plex Mono',monospace;font-size:11px;")
            lbl.clicked.connect(
                lambda _=False, i=table.index_at(t0):
                    self.gotoEpochRequested.emit(int(i)))
            x = QPushButton("×")
            x.setMaximumWidth(24)
            x.setStyleSheet("color:#f85149;font-weight:600;")
            x.clicked.connect(
                lambda _=False, i=mid: self.unmarkArtefactRequested.emit(i))
            rl.addWidget(lbl, 1)
            rl.addWidget(x)
            self._marked_layout.addWidget(row)

    def set_denominator_mask(self, types, source='from the detection run',
                             pending_marks=0):
        """Say which exclusion set the density on screen was computed under.

        Parameters
        ----------
        types : iterable of str
            The excluded event types, as stored tokens.
        source : str, optional
            Where the set came from: ``'from the detection run'`` when it was
            read back, ``'assumed (not recorded)'`` when it could not be. The
            assumed case renders in the warning colour, because an assumed
            denominator can be wrong in either direction. Default
            ``'from the detection run'``.
        pending_marks : int, optional
            Artefact intervals the reviewer has marked this session. Appended
            separately rather than folded into the set, so the reviewer's own
            contribution to the denominator is never mistaken for the run's.
            Default ``0``.
        """
        assumed = 'assumed' in str(source)
        text = (f"density excludes {reject_types_display(types)} "
                f"\u00b7 {source}")
        if pending_marks:
            text += (f" \u00b7 + {int(pending_marks)} mark"
                     f"{'' if int(pending_marks) == 1 else 's'} you added")
        self.mask_caption.setText(text)
        self.mask_caption.setStyleSheet(
            "color:#d29922;font-size:11px;" if assumed
            else "color:#6b7585;font-size:11px;")

    def set_coords(self, coords):
        self._coords = coords or None
        self.update_topo(self._qc_df)

    # ---- population checks --------------------------------------------
    def set_checks_context(self, recorded, event_type='spindle', ratio=True,
                           method='', caption=''):
        """Topography combo items and caption for the population checks.

        The five items are listed for every event type except
        ``low prominence`` (spindles only); on a run without stored figures
        they are disabled with the suffix `` — not recorded for this run``.
        """
        keep = self.topo_combo.currentData()
        self.topo_combo.blockSignals(True)
        while self.topo_combo.count() > 3:
            self.topo_combo.removeItem(3)
        model = self.topo_combo.model()
        for col, (_hdr, label) in CHECK_COLUMNS.items():
            if col == 'pct_low_prom' and event_type != 'spindle':
                continue
            text = label
            enabled = bool(recorded)
            if not recorded:
                text += ' — not recorded for this run'
            elif col == 'med_thresh_ratio' and not ratio:
                text += f' — no ratio for {method}'
                enabled = False
            self.topo_combo.addItem(text, col)
            item = model.item(self.topo_combo.count() - 1)
            if item is not None:
                item.setEnabled(enabled)
        idx = self.topo_combo.findData(keep)
        if idx < 0 or not (model.item(idx) and model.item(idx).isEnabled()):
            idx = 0
        self.topo_combo.setCurrentIndex(idx)
        self.topo_combo.blockSignals(False)
        self.topo_metric = self.topo_combo.currentData() or 'density'
        self._checks_recorded = bool(recorded)
        self.checks_caption.setText(caption or '')

    def set_check_line(self, channel, items, recorded=True):
        """The SELECTED CHANNEL population-check line (spec section 1)."""
        self._check_items = list(items or [])
        self._check_channel = channel
        if channel is None:
            self.checks_line.setText('')
            self.checks_hint.setVisible(False)
            return
        if not recorded or not items:
            text = _er.dock_check_line(channel, items, recorded)
            self.checks_line.setText(
                f"<span style='color:#888888'>{text}</span>")
            self.checks_hint.setVisible(False)
            return
        parts = []
        for d in self._check_items:
            col = '#e0533f' if d['severity'] == 'hard' else '#e0a334'
            parts.append(f"<a href='{d['key']}' style='color:{col};"
                         f"font-weight:600;text-decoration:none'>"
                         f"{d['text']}</a>")
        self.checks_line.setText(
            f"<span style='color:#e5e5e5'>{channel}: </span>"
            + " <span style='color:#888888'>·</span> ".join(parts))
        self.checks_hint.setVisible(True)

    def check_line_text(self):
        """The dock line as plain text (what a reader sees)."""
        return QtGui.QTextDocumentFragment.fromHtml(
            self.checks_line.text()).toPlainText()

    def _on_check_link(self, key):
        if self._check_channel:
            self.checkLinkActivated.emit(str(self._check_channel), str(key))

    def _on_check_hover(self, key):
        tip = next((d['tooltip'] for d in self._check_items
                    if d['key'] == key), '')
        self.checks_line.setToolTip(tip)

    def set_event_type(self, event_type):
        """Set the event type used in the topo title + global-worst header."""
        self._event_type = str(event_type or 'slow_wave')
        self.global_worst_hdr.setText(
            f"WORST EVENTS — ALL CHANNELS · {self._event_type}")

    def _on_metric(self, _idx=0):
        self.topo_metric = self.topo_combo.currentData() or 'density'
        self.update_topo(self._qc_df)

    def _clear_colorbar(self):
        # ColorBarItem is inserted into the plotItem layout (not the scene),
        # so PlotWidget.clear() does not remove it — track + drop it here.
        if self._colorbar is not None:
            try:
                self.topo.plotItem.layout.removeItem(self._colorbar)
                self._colorbar.close()
            except Exception:
                pass
            self._colorbar = None

    def _render_topo_empty(self):
        self._clear_colorbar()
        self.topo.clear()
        self.topo.setTitle("Topography")
        self.load_montage_btn.setVisible(True)   # button only in fallback
        txt = pg.TextItem(
            "No channel coordinates in this recording.\n"
            "Load an EEGLAB .set or montage file to enable topography.",
            color=(120, 120, 120), anchor=(0.5, 0.5))
        self.topo.addItem(txt)
        txt.setPos(0.5, 0.5)
        self.topo.setXRange(0, 1); self.topo.setYRange(0, 1)

    def update_channel(self, channel, df_slice, qc_row=None):
        self.title.setText(str(channel) if channel else "—")
        self.worst_hdr.setText(
            f"WORST EPOCHS ON {channel}" if channel else "WORST EPOCHS ON —")
        self.worst_list.clear()
        if qc_row is not None:
            self.subtitle.setText(
                f"{qc_row.get('region', '')} · "
                f"flag {qc_row.get('flag', '') or 'ok'} · "
                f"n={int(qc_row.get('n', 0))}")
            for k, val in self._z_labels.items():
                z = qc_row.get(k)
                try:
                    z = float(z)
                    val.setText(f"{z:.2f}")
                    val.setStyleSheet(
                        "font-family:'IBM Plex Mono',monospace;color:" +
                        ("#f85149" if abs(z) > 2 else "#d6dee8"))
                except Exception:
                    val.setText("—")
        else:
            self.subtitle.setText("")
            for val in self._z_labels.values():
                val.setText("—")
        if df_slice is None or len(df_slice) == 0:
            return
        # qc_row optionally carries _epochs (the scored-epoch table) +
        # _event_type so the worst-list can key rows by epoch id, stage-tag
        # them and pick the right amplitude column without widening
        # update_channel's signature. A bare '_hypno' stage list (older
        # callers) still works on the 30 s grid.
        epochs = self._epochs
        hyp = None
        evt = None
        if qc_row is not None:
            try:
                epochs = qc_row.get('_epochs', epochs)
                hyp = qc_row.get('_hypno')
                evt = qc_row.get('_event_type')
            except Exception:
                pass
        amp_col = AMP_COL.get(str(evt), 'max_amp') if evt else 'max_amp'
        agg = _compute_epoch_outliers(df_slice, hypno=hyp, amp_col=amp_col,
                                      epochs=epochs)
        agg = agg[agg['n_outliers'] > 0]
        agg = agg.sort_values(['n_outliers', 'max_amp'],
                              ascending=[False, False]).head(12)
        for _, r in agg.iterrows():
            idx = int(r['idx'])
            stage = (str(r['stage']) or '—')[:5]
            amp = float(r['max_amp'])   # already the AMP_COL max for this epoch
            amp_txt, impossible = _amp_cell(amp)
            txt = (f"ep {idx + 1:03d}  {stage:<5}  "
                   f"{int(r['n_outliers']):>2}×  {amp_txt}")
            it = QtWidgets.QListWidgetItem(txt)
            it.setData(Qt.UserRole, idx)
            if impossible:
                it.setForeground(QtGui.QColor(IMPOSSIBLE_AMP_COLOR))
                it.setToolTip(_artefact_tooltip(amp))
            self.worst_list.addItem(it)

    def _on_worst_click(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self.gotoEpochRequested.emit(int(idx))

    # ---- global worst-events list (all channels) ----------------------
    def set_global_worst(self, rows, event_type=None):
        """Populate the read-only 'worst events across all channels' list.

        ``rows`` is a list of dicts with keys ``channel``, ``start_time``,
        ``stage``, ``amp`` (already sorted by amp desc and capped by the
        caller). Each item stores ``(channel, start_time)`` on ``Qt.UserRole``;
        clicking jumps to that channel + epoch. Impossible-scale events
        (>1000 µV) render red with a ⚠ and an artefact tooltip.
        """
        if event_type is not None:
            self.set_event_type(event_type)
        self.global_worst_list.clear()
        et = self._event_type
        if not rows:
            it = QtWidgets.QListWidgetItem(
                f"No {et} events in this subject. "
                f"Switch event type or run detection.")
            it.setForeground(QtGui.QColor('#6b7585'))
            it.setFlags(Qt.NoItemFlags)
            self.global_worst_list.addItem(it)
            return
        for r in rows:
            ch = str(r['channel'])
            amp = float(r['amp'])
            stage = _short_stage(r.get('stage'))
            amp_txt, impossible = _amp_cell(amp)
            txt = f"{ch:<6} {_hms(r['start_time'])} {stage:<3} {amp_txt}"
            it = QtWidgets.QListWidgetItem(txt)
            it.setData(Qt.UserRole, (ch, float(r['start_time'])))
            if impossible:
                it.setForeground(QtGui.QColor(IMPOSSIBLE_AMP_COLOR))
                it.setToolTip(_artefact_tooltip(amp))
            self.global_worst_list.addItem(it)

    def _on_global_worst_click(self, item):
        data = item.data(Qt.UserRole)
        if data:
            ch, t0 = data
            self.gotoChannelEpochRequested.emit(str(ch), float(t0))

    def _on_topo_click(self, _scatter, points):
        """Topo electrode clicked -> select that channel (no drill, no
        recompute). Empty-space clicks don't fire sigClicked; guard anyway."""
        if not len(points):
            return
        data = points[0].data()          # (channel, metric_value)
        if data:
            self.channelPicked.emit(str(data[0]))

    def update_topo(self, qc_df):
        self._qc_df = qc_df
        if self._coords is None or qc_df is None or len(qc_df) == 0:
            self._render_topo_empty()
            return
        try:
            from scipy.interpolate import griddata
        except Exception:
            self._render_topo_empty()
            return
        metric = self.topo_metric
        pts, vals, chans = [], [], []
        for _, r in qc_df.iterrows():
            ch = str(r['channel'])
            if ch in self._coords and pd.notna(r.get(metric)):
                pts.append(self._coords[ch]); vals.append(float(r[metric]))
                chans.append(ch)
        # Interpolate only channels present in BOTH coords and the QC frame.
        if len(pts) < 4:
            self._render_topo_empty()
            return
        pts = np.asarray(pts); vals = np.asarray(vals)
        xi = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 80)
        yi = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 80)
        gx, gy = np.meshgrid(xi, yi)
        gz = griddata(pts, vals, (gx, gy), method='cubic')
        # Robust colour limits (2nd–98th percentile). Real QC metrics carry a
        # heavy artefact tail — a handful of impossible-scale channels (e.g.
        # max_p2p up to ~11000 µV) push a raw min/max scale so that the whole
        # physiological population (median ~250 µV) collapses into the bottom
        # ~1% of the colormap and the scalp renders near-black. Percentile
        # limits keep the physiological contrast visible; the >98th-pct
        # artefact channels simply saturate at the top colour. Falls back to
        # raw min/max when the distribution is degenerate (all-equal / <2 pts).
        lo, hi = (float(x) for x in np.nanpercentile(vals, [2, 98]))
        if not (hi > lo):
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        vmin, vmax = lo, hi
        cmap = pg.colormap.get('viridis')
        self._clear_colorbar()
        self.topo.clear()
        self.load_montage_btn.setVisible(False)  # map live -> hide the button
        label = TOPO_METRIC_LABEL.get(metric, metric)
        self.topo.setTitle(f"Topography · {label} ({self._event_type})")
        img = pg.ImageItem(gz.T)
        img.setLookupTable(cmap.getLookupTable())
        if vmax > vmin:
            img.setLevels((vmin, vmax))
        self.topo.addItem(img)
        # Per-electrode scatter: each spot carries its channel label (data=)
        # so hover shows the label + metric value and a click selects it.
        # density is sub-1 ev/min -> 2 decimals; µV metrics -> integer.
        _is_uv = metric in ('mean_amp', 'p95_amp', 'max_p2p')
        unit = 'µV' if _is_uv else 'ev/min'
        vfmt = '.0f' if _is_uv else '.2f'

        def _tip(x, y, data, _vl=label, _u=unit, _f=vfmt, _m=metric):
            ch, v = data
            if _m in CHECK_COLUMNS:
                return f"{ch}\n{_vl.split(' (')[0]} {fmt_check(_m, v)}"
            return f"{ch}\n{_vl.split(' (')[0]} {v:{_f}} {_u}"

        sp = pg.ScatterPlotItem(
            x=(pts[:, 0] - pts[:, 0].min()) / max(np.ptp(pts[:, 0]), 1e-9) * 80,
            y=(pts[:, 1] - pts[:, 1].min()) / max(np.ptp(pts[:, 1]), 1e-9) * 80,
            size=5, brush=(20, 20, 20),
            data=list(zip(chans, (float(v) for v in vals))),
            hoverable=True, hoverSize=9,
            hoverPen=pg.mkPen('#e7eef7', width=1.5), tip=_tip)
        sp.sigClicked.connect(self._on_topo_click)
        self.topo.addItem(sp)
        # Colorbar / legend with numeric min/max endpoints.
        try:
            self._colorbar = pg.ColorBarItem(
                values=(vmin, vmax), colorMap=cmap, width=12,
                interactive=False)
            self._colorbar.setImageItem(img, insert_in=self.topo.plotItem)
        except Exception:
            self._colorbar = None


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps its items onto new rows (Qt's standard
    flow-layout pattern). Used for the Selection-tray chip rows."""

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only):
        x, y, line_h = rect.x(), rect.y(), 0
        sp = self.spacing()
        for it in self._items:
            w, h = it.sizeHint().width(), it.sizeHint().height()
            nx = x + w + sp
            if nx - sp > rect.right() and line_h > 0:
                x, y = rect.x(), y + line_h + sp
                nx, line_h = x + w + sp, 0
            if not test_only:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), it.sizeHint()))
            x, line_h = nx, max(line_h, h)
        return y + line_h - rect.y()


class ChannelQCWidget(QWidget):
    """Tab 0 — primary QC surface: sortable per-channel table + verdict
    actions + a Selection tray for artefact / re-detect channels."""

    channelSelected = pyqtSignal(str)
    requestDrill = pyqtSignal(str)
    verdictChanged = pyqtSignal()
    addToRedetect = pyqtSignal(str)
    requestQueueAllHard = pyqtSignal()
    requestBuildRedetect = pyqtSignal()
    loadMontageRequested = pyqtSignal()
    unmarkArtefact = pyqtSignal(str)       # tray chip ✕ → clear artefact verdict
    removeFromRedetect = pyqtSignal(str)   # tray chip ✕ → un-queue channel

    def __init__(self, parent=None):
        super().__init__(parent)
        ll = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Event type:"))
        self.evt_combo = QComboBox()
        self.evt_combo.addItems(['slow_wave', 'spindle', 'k_complex', 'pac'])
        bar.addWidget(self.evt_combo)
        bar.addWidget(QLabel("Outlier:"))
        self.flag_combo = QComboBox()
        self.flag_combo.addItems(['any', 'hard', 'soft', 'dead', 'ok',
                                  'checks: hard', 'checks: soft'])
        bar.addWidget(self.flag_combo)
        bar.addStretch()
        self.counts_lbl = QLabel("")
        self.counts_lbl.setTextFormat(Qt.RichText)
        bar.addWidget(self.counts_lbl)
        ll.addLayout(bar)

        self.model = ChannelQCModel()
        self.proxy = QtCore.QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().currentRowChanged.connect(self._on_row)
        ll.addWidget(self.table)

        btns = QHBoxLayout()
        self._sel_lbl = QLabel("On selected (—):")
        self._sel_lbl.setStyleSheet("color:#6b7585;")
        btns.addWidget(self._sel_lbl)
        self.btn_drill = QPushButton("Drill into epochs ▸")
        self.btn_drill.clicked.connect(self._drill)
        self.btn_mark = QPushButton("Mark channel artefact")
        self.btn_mark.setObjectName("danger")
        self.btn_mark.clicked.connect(self._toggle_artefact)
        self.btn_redetect = QPushButton("Add to re-detect queue")
        self.btn_redetect.clicked.connect(self._redetect)
        for b in (self.btn_drill, self.btn_mark, self.btn_redetect):
            btns.addWidget(b)
        btns.addStretch()
        self.btn_queue_hard = QPushButton("Queue all HARD")
        self.btn_queue_hard.clicked.connect(
            lambda: self.requestQueueAllHard.emit())
        btns.addWidget(self.btn_queue_hard)
        self.btn_build = QPushButton("Build re-detect request…")
        self.btn_build.setObjectName("primary")
        self.btn_build.clicked.connect(
            lambda: self.requestBuildRedetect.emit())
        btns.addWidget(self.btn_build)
        ll.addLayout(btns)

        # --- Selection tray (collapsible): artefact + re-detect chips ------
        self._tray_open = True
        tray_hdr = QHBoxLayout()
        tlbl = QLabel("Selection")
        tlbl.setStyleSheet("color:#9ba6b5;font-weight:600;")
        tray_hdr.addWidget(tlbl)
        tray_hdr.addStretch()
        self.tray_toggle = QPushButton("▾")
        self.tray_toggle.setFlat(True)
        self.tray_toggle.setMaximumWidth(28)
        self.tray_toggle.setCursor(Qt.PointingHandCursor)
        self.tray_toggle.clicked.connect(self._toggle_tray)
        tray_hdr.addWidget(self.tray_toggle)
        ll.addLayout(tray_hdr)

        self.tray_body = QWidget()
        tb = QVBoxLayout(self.tray_body)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(2)
        self.art_hdr = _h_label("CHANNEL ARTEFACTS (0)")
        tb.addWidget(self.art_hdr)
        self.art_scroll, self.art_flow = self._make_chip_area()
        tb.addWidget(self.art_scroll)
        self.rd_hdr = _h_label("RE-DETECT QUEUE (0)")
        tb.addWidget(self.rd_hdr)
        self.rd_scroll, self.rd_flow = self._make_chip_area()
        tb.addWidget(self.rd_scroll)
        ll.addWidget(self.tray_body)

        self._qc_full = None      # full computed df for current event type
        self._events_slice = None  # montage-wide events df (current event type)
        self._verdicts = {}
        self._redetect_ref = set()  # window's queue (for button label state)
        self.flag_combo.currentTextChanged.connect(self._apply_flag_filter)

    def current_event_type(self):
        return self.evt_combo.currentText()

    def set_checks_context(self, recorded, method='', ratio=True,
                           event_type='spindle', tips=None, pending=False,
                           figures_off=False):
        """Population-check state for the table: whether the run stored
        figures, its method (for the amp/thr tooltip), header tooltips, and
        whether the low-prominence column applies (spindles only)."""
        self.model._checks = {'recorded': bool(recorded),
                              'method': str(method or ''),
                              'ratio': bool(ratio), 'tips': dict(tips or {}),
                              'pending': bool(pending),
                              'figures_off': bool(figures_off)}
        self.table.setColumnHidden(_QC_COL_INDEX['pct_low_prom'],
                                   str(event_type) != 'spindle')
        self.model.headerDataChanged.emit(Qt.Horizontal, 0,
                                          len(_QC_COLS) - 1)
        if self.model.rowCount():
            self.model.dataChanged.emit(
                self.model.index(0, 0),
                self.model.index(self.model.rowCount() - 1,
                                 len(_QC_COLS) - 1))

    def set_data(self, qc_df, events_df, verdicts, redetect_set=None):
        self._qc_full = qc_df
        self._events_slice = events_df
        self._verdicts = verdicts or {}
        if redetect_set is not None:
            self._redetect_ref = redetect_set
        # _apply_flag_filter calls model.set_data with the (optionally
        # flag-filtered) cached df — so the direct call here was a redundant
        # first full model reset (each reset re-sorts the proxy + view).
        self._apply_flag_filter(self.flag_combo.currentText())
        self._update_counts()
        if self.table.model().rowCount() and not self.table.currentIndex().isValid():
            self.table.selectRow(0)
        self._update_action_state()
        self._rebuild_tray()

    def _update_counts(self):
        df = self._qc_full
        if df is None or len(df) == 0:
            self.counts_lbl.setText("")
            return
        c = {k: int((df['flag'] == v).sum()) for k, v in
             (('HARD', 'hard'), ('SOFT', 'soft'), ('DEAD', 'dead'))}
        c['OK'] = int((df['flag'] == '').sum())
        cmap = {'HARD': '#f85149', 'SOFT': '#d29922',
                'DEAD': '#9ba6b5', 'OK': '#3fb950'}
        self.counts_lbl.setText("&nbsp;&nbsp;".join(
            f"<span style='color:{cmap[k]}'>{k}</span> {c[k]}"
            for k in ('HARD', 'SOFT', 'DEAD', 'OK')) +
            f"&nbsp;&nbsp;<span style='color:#6b7585'>"
            f"{len(df)} ch</span>")

    def _update_action_state(self):
        ch = self._current_channel()
        self._sel_lbl.setText(f"On selected ({ch or '—'}):")
        en = ch is not None
        for b in (self.btn_drill, self.btn_mark, self.btn_redetect):
            b.setEnabled(en)
        if ch is not None:
            v = self._verdicts.get(ch, '')
            self.btn_mark.setText(
                "Unmark channel artefact" if v in ('drop', 'channel_artefact')
                else "Mark channel artefact")
            self.btn_redetect.setText(
                "Remove from re-detect queue" if ch in self._redetect_ref
                else "Add to re-detect queue")
        n_hard = 0 if self._qc_full is None or len(self._qc_full) == 0 \
            else int((self._qc_full['flag'] == 'hard').sum())
        self.btn_queue_hard.setText(f"Queue all HARD ({n_hard})")
        self.btn_queue_hard.setEnabled(n_hard > 0)
        self.btn_build.setText(
            f"Build re-detect request… ({len(self._redetect_ref)})")
        self.btn_build.setEnabled(len(self._redetect_ref) > 0)

    def _apply_flag_filter(self, mode):
        # filter the source model in-place by rebuilding from cached df
        if self._qc_full is None:
            return
        df = self._qc_full
        if mode == 'ok':
            df = df[df['flag'] == '']
        elif mode in ('hard', 'soft', 'dead'):
            df = df[df['flag'] == mode]
        elif mode.startswith('checks: '):
            want = mode.split(': ', 1)[1]
            df = (df[df['checks_flag'] == want] if 'checks_flag' in df.columns
                  else df.iloc[0:0])
        self.model.set_data(df, self._verdicts, self._redetect_ref,
                            self.current_event_type())

    def _current_channel(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        src = self.proxy.mapToSource(idx)
        return self.model.channel_at(src.row())

    def _on_row(self, *_):
        ch = self._current_channel()
        self._update_action_state()
        if ch is None:
            return
        self.channelSelected.emit(ch)

    def _toggle_artefact(self):
        ch = self._current_channel()
        if not ch:
            return
        cur = self._verdicts.get(ch, '')
        # Unmark returns to '' (untriaged), NOT 'keep' — a '' channel is still
        # included in export; only drop/channel_artefact are excluded.
        self._verdict('' if cur in ('drop', 'channel_artefact') else 'drop')

    def events_slice_for(self, ch):
        if self._events_slice is not None and len(self._events_slice):
            return self._events_slice[self._events_slice['channel'] == ch]
        return None

    def _verdict(self, v):
        ch = self._current_channel()
        if ch:
            self._verdicts[ch] = v
            self.verdictChangedTo = (ch, v)
            self.verdictChanged.emit()

    def _drill(self):
        ch = self._current_channel()
        if ch:
            self.requestDrill.emit(ch)

    def select_channel(self, ch):
        """Select the row for ``ch`` in the QC table (used when a global
        worst-events click jumps to another channel). No-op if not present."""
        for prow in range(self.proxy.rowCount()):
            src = self.proxy.mapToSource(self.proxy.index(prow, 0))
            if self.model.channel_at(src.row()) == str(ch):
                self.table.selectRow(prow)
                return True
        return False

    def _redetect(self):
        ch = self._current_channel()
        if ch:
            self.addToRedetect.emit(ch)

    # ---- Selection tray ----------------------------------------------
    def _make_chip_area(self):
        """A scrollable (~2 rows) wrapping chip area. Returns (scroll, flow)."""
        wrap = QWidget()
        flow = FlowLayout(wrap, margin=2, spacing=6)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumHeight(64)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll, flow

    def _toggle_tray(self):
        self._tray_open = not self._tray_open
        self.tray_body.setVisible(self._tray_open)
        self.tray_toggle.setText("▾" if self._tray_open else "▸")

    def _clear_flow(self, flow):
        while flow.count():
            it = flow.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _empty_label(self, text):
        q = QLabel(text)
        q.setStyleSheet("color:#6b7585;font-style:italic;font-size:11px;")
        return q

    def _make_chip(self, ch, color, on_remove):
        """A [ label ✕ ] chip: body click selects the channel, ✕ removes it."""
        chip = QWidget()
        chip.setStyleSheet(
            f"background:#161b22;border:1px solid {color};border-radius:9px;")
        h = QHBoxLayout(chip)
        h.setContentsMargins(7, 1, 3, 1)
        h.setSpacing(2)
        body = QPushButton(str(ch))
        body.setFlat(True)
        body.setCursor(Qt.PointingHandCursor)
        body.setStyleSheet(
            f"border:0;background:transparent;color:{color};"
            "font-family:'IBM Plex Mono',monospace;font-size:11px;")
        body.clicked.connect(lambda _=False, c=str(ch): self.select_channel(c))
        x = QPushButton("✕")
        x.setFlat(True)
        x.setMaximumWidth(16)
        x.setCursor(Qt.PointingHandCursor)
        x.setStyleSheet(f"border:0;background:transparent;color:{color};"
                        "font-size:10px;")
        x.clicked.connect(lambda _=False: on_remove())
        h.addWidget(body)
        h.addWidget(x)
        return chip

    def _rebuild_tray(self):
        """Rebuild both chip sections from the current verdicts + queue.
        Artefact chips reflect the ACTIVE event type; re-detect is channel-level.
        """
        self._clear_flow(self.art_flow)
        self._clear_flow(self.rd_flow)
        evt = self.current_event_type()
        art = sorted(ch for ch, v in self._verdicts.items()
                     if v in ('drop', 'channel_artefact'))
        rd = sorted(str(c) for c in self._redetect_ref)
        self.art_hdr.setText(f"CHANNEL ARTEFACTS ({len(art)})")
        self.rd_hdr.setText(f"RE-DETECT QUEUE ({len(rd)})")
        if art:
            for ch in art:
                self.art_flow.addWidget(self._make_chip(
                    ch, _STATE_ART_COLOR,
                    lambda c=ch: self.unmarkArtefact.emit(str(c))))
        else:
            self.art_flow.addWidget(self._empty_label(
                f"No channels marked as artefact for {evt}."))
        if rd:
            for ch in rd:
                self.rd_flow.addWidget(self._make_chip(
                    ch, _STATE_RD_COLOR,
                    lambda c=ch: self.removeFromRedetect.emit(str(c))))
        else:
            self.rd_flow.addWidget(self._empty_label("Re-detect queue empty."))


DEFAULT_BAND = {'slow_wave': (0.5, 2.0), 'k_complex': (0.5, 1.5),
                'spindle': (11.0, 16.0), 'pac': (0.5, 2.0)}

# Fixed filtered-trace ±y-range (µV) per event_type, so the reviewer's eye
# doesn't recalibrate on every epoch click. raw_plot stays autoscaled.
FILT_YRANGE = {'spindle': 20, 'slow_wave': 80, 'k_complex': 100, 'pac': 80}

# Mode-1 (Compare methods) ribbon colour palette — cycles if >4 methods.
METHOD_COLORS = ['#5a8fce', '#e0a334', '#69b35d', '#a371f7']

# ± window for cross-method agreement (seconds), per event_type.
AGREEMENT_THRESH = {'spindle': 0.5, 'slow_wave': 1.0,
                    'k_complex': 0.3, 'pac': 1.0}


HYPNO_RANK = {'Wake': 4, 'W': 4, 'REM': 3,
              'N1': 2, 'NREM1': 2, 'Stage1': 2,
              'N2': 1, 'NREM2': 1, 'Stage2': 1,
              'N3': 0, 'NREM3': 0, 'Stage3': 0}


def _draw_hypnogram(pw, hypno, trec=None):
    """Render a stage-rank-line hypnogram, color-coded by stage via
    STAGE_COLOR. One horizontal segment per scored epoch spanning its true
    ``[start, end]`` (1-30 s on cut recordings); segments from the same stage
    are joined as a polyline with NaN gaps so they render as one
    PlotDataItem per stage. Y: stage rank (Wake=4 top, N3=0 bottom).
    X: recording time (seconds). ``hypno`` is an :class:`EpochTable`; a bare
    stage list is still accepted and spread evenly over ``trec`` (the old
    contract). Caller must clear() and re-add any persistent overlays after
    this returns."""
    if hypno is None or len(hypno) == 0:
        return
    if isinstance(hypno, EpochTable):
        table = hypno
    else:
        n = len(hypno)
        ep = float(trec) / n if trec else DEFAULT_EPOCH_S
        table = EpochTable.grid(None, epoch_s=ep, stages=[str(s) for s in hypno])
    stages = np.asarray(table.stages, dtype=object)
    for stage_key in dict.fromkeys(table.stages):
        m = stages == stage_key
        k = int(m.sum())
        col = STAGE_COLOR.get(stage_key, '#888888')
        y = HYPNO_RANK.get(stage_key, 2)
        xs = np.column_stack([table.starts[m], table.ends[m],
                              np.full(k, np.nan)]).ravel()
        ys = np.column_stack([np.full(k, y, dtype=float),
                              np.full(k, y, dtype=float),
                              np.full(k, np.nan)]).ravel()
        pw.plot(xs, ys, pen=pg.mkPen(col, width=2), connect='finite')
    pw.setYRange(-0.5, 4.5, padding=0)
    pw.setXRange(0, float(max(trec or 0.0, table.end)), padding=0)
    pw.hideAxis('left')


def _agreement_clusters(events_df, thresh_s):
    """Cluster events by MIDPOINT proximity (≤ thresh_s) regardless of method.
    Returns (clusters, dfsorted) where clusters is a list of
    (list[idx_in_dfsorted], frozenset(methods))."""
    if events_df is None or len(events_df) == 0:
        return [], pd.DataFrame()
    df = events_df.copy()
    df['_mid'] = (pd.to_numeric(df['start_time'], errors='coerce')
                  + pd.to_numeric(df['end_time'], errors='coerce')) / 2.0
    df = df.dropna(subset=['_mid']).sort_values('_mid').reset_index(drop=True)
    clusters, cur, methods, prev = [], [], set(), None
    for i, row in df.iterrows():
        m = row['_mid']
        if not cur or (m - prev <= thresh_s):
            cur.append(i); methods.add(str(row['method']))
        else:
            clusters.append((cur, frozenset(methods)))
            cur, methods = [i], {str(row['method'])}
        prev = m
    if cur:
        clusters.append((cur, frozenset(methods)))
    return clusters, df

# Lean column set for the QC/Epochs path — avoids marshalling all 23 event
# columns × ~370k rows when only these are consumed: compute_channel_qc
# (channel, start_time, end_time, max_amp, min_amp, peak2peak_amp), the Epochs
# drill band (freq_lower/freq_upper) + amp, and the global worst-events list.
QC_EVENT_COLS = ['channel', 'start_time', 'end_time', 'stage',
                 'min_amp', 'max_amp', 'peak2peak_amp',
                 'freq_lower', 'freq_upper']

# The drilled channel's slice adds the columns that identify the event a
# reviewer selects and the run whose parameters judge it. Fetched for one
# channel at drill time (~1.5k rows, a few ms) instead of montage-wide, where
# they cost ~+300 ms and ~+100 MB per refresh on a 372k-event subject.
# get_events drops any of these an older database lacks.
QC_DRILL_COLS = QC_EVENT_COLS + ['uuid', 'duration', 'run_id', 'method',
                                 'epoch_stage',
                                 # 4.6 figures behind the Epochs check filter
                                 'in_band', 'low_prominence', 'near_bound',
                                 'amp_ratio', 'thresh_ratio']


def _band_for(df_slice, event_type):
    """Source the filtered-trace passband from the EVENT ROWS themselves
    (freq_lower / freq_upper), not the filter dropdown — the dropdown is a
    UI proxy for what's already stamped on each event. Returns
    (lo, hi, label). Falls back to DEFAULT_BAND when the rows carry no band
    or disagree on it.
    """
    lo_d, hi_d = DEFAULT_BAND.get(event_type, (0.5, 2.0))
    if (df_slice is None or len(df_slice) == 0
            or 'freq_lower' not in df_slice.columns
            or 'freq_upper' not in df_slice.columns):
        return lo_d, hi_d, f"default for {event_type}"
    pairs = (df_slice[['freq_lower', 'freq_upper']]
             .apply(pd.to_numeric, errors='coerce')
             .dropna().drop_duplicates())
    if len(pairs) == 1:
        return float(pairs.iloc[0, 0]), float(pairs.iloc[0, 1]), "from events"
    if len(pairs) > 1:
        return (lo_d, hi_d,
                f"default for {event_type} · events span {len(pairs)} bands")
    return lo_d, hi_d, f"default for {event_type}"


def _bandpass(data, sfreq, lo, hi):
    """Zero-phase 2nd-order Butterworth band-pass (slow-wave safe)."""
    try:
        ny = sfreq / 2.0
        lo_n = max(1e-3, min(lo / ny, 0.999))
        hi_n = max(1e-3, min(hi / ny, 0.999))
        if lo_n >= hi_n:
            return data
        b, a = signal.butter(2, [lo_n, hi_n], btype='band')
        return signal.filtfilt(b, a, data)
    except Exception:
        return data


# ---------------------------------------------------------------------------
# Event-type-aware amplitude column for the outlier rule.
# (a)-style: rule is defined once here; strip, ticker, worst-list all read it.
# Flip to per-call override later by passing amp_col= into the helper.
# ---------------------------------------------------------------------------
AMP_COL = {
    'slow_wave': 'peak2peak_amp',
    'k_complex': 'peak2peak_amp',
    'spindle':   'max_amp',
    'pac':       'max_amp',
}


def _mad_threshold(amp):
    """Robust outlier cutoff: median + 3.5 * 1.4826 * MAD. Returns (thr, n).

    The 1.4826 scale makes the MAD a consistent estimator of sigma for
    normal data, so 3.5 here is ~3.5 sigma but resists the very artefacts
    we're flagging — a handful of 10 mV spikes no longer inflate the cutoff
    the way mean+sd did. Single source of truth for the rule (strip,
    ticker, worst-list, cached EpochsPanel threshold all route through it).
    """
    amp = pd.to_numeric(amp, errors='coerce').dropna()
    n = int(len(amp))
    if n == 0:
        return float('inf'), 0
    med = float(amp.median())
    mad = float((amp - med).abs().median())
    thr = med + 3.5 * 1.4826 * mad if mad > 0 else float('inf')
    return thr, n


def _compute_epoch_outliers(df_slice, hypno=None, epoch_len=DEFAULT_EPOCH_S,
                            amp_col='max_amp', epochs=None):
    """Return DataFrame[idx, t0, n_events, n_outliers, max_amp, stage].

    Outlier rule: amp > median + 3.5*1.4826*MAD (see _mad_threshold),
    computed once over the whole (channel, event_type) df_slice. ``max_amp``
    in the returned frame is the chosen amp_col's max within the epoch —
    kept under that column name so callers don't need to know which metric
    was used. Cells with n_events==0 are omitted (strip handles as gaps).

    ``idx`` is the epoch id in ``epochs`` (an :class:`EpochTable` or a list
    of ``(start, end, stage)``), found by a sorted search on epoch starts, so
    variable-length epochs are keyed correctly. An event whose start lies in
    no epoch is left out. Without ``epochs`` the old fixed grid of
    ``epoch_len`` seconds is used, staged from ``hypno`` when given.
    """
    cols = ['idx', 't0', 'n_events', 'n_outliers', 'max_amp', 'stage']
    if df_slice is None or len(df_slice) == 0:
        return pd.DataFrame(columns=cols)
    if amp_col not in df_slice.columns:
        amp_col = 'max_amp' if 'max_amp' in df_slice.columns else amp_col
    df = df_slice.copy()
    df['_st'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['_amp'] = pd.to_numeric(df.get(amp_col), errors='coerce')
    df = df.dropna(subset=['_st', '_amp'])
    if df.empty:
        return pd.DataFrame(columns=cols)
    thr, _ = _mad_threshold(df['_amp'])
    df['_is_out'] = df['_amp'] > thr
    if isinstance(epochs, EpochTable) or epochs:
        table = _as_epoch_table(epochs)
    else:
        # legacy fixed grid: wide enough for every event and every stage
        n = max(int(np.floor(df['_st'].max() / float(epoch_len))) + 1,
                len(hypno) if hypno else 0, 1)
        stages = ([str(hypno[i]) if hypno and i < len(hypno) else ''
                   for i in range(n)])
        table = EpochTable.grid(None, epoch_s=epoch_len, stages=stages)
    df['ep_idx'] = table.indices_at(df['_st'].to_numpy(dtype=float))
    df = df[df['ep_idx'] >= 0]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby('ep_idx', sort=True).agg(
        n_events=('_st', 'size'),
        n_outliers=('_is_out', 'sum'),
        max_amp=('_amp', 'max'),
    ).reset_index().rename(columns={'ep_idx': 'idx'})
    g['t0'] = table.starts[g['idx'].to_numpy()]
    g['stage'] = [table.stages[i] for i in g['idx']]
    return g[cols]


#: Band / ticker colour of a reviewed event, by decision. Reject is magenta,
#: not red, so it never reads as the outlier flag.
DECISION_COLOR = {'accept': '#69b35d', 'reject': '#c45bb0',
                  'unsure': '#e0a334'}


def _event_frame(df_slice, amp_thr, amp_col):
    """Per-drill event frame for bands, picking and unreviewed paging.

    Columns ``_start``, ``_end``, ``_is_out`` (amp above ``amp_thr``) and
    ``uuid`` (str, or None when the database has none), sorted by start time
    with a 0..n-1 index, so position in the frame is start-time order. Built
    once per drill; ``_end`` falls back to start + duration, then start +
    0.5 s.
    """
    cols = ['_start', '_end', '_is_out', 'uuid']
    if df_slice is None or len(df_slice) == 0 \
            or 'start_time' not in df_slice.columns:
        return pd.DataFrame(columns=cols)
    st = pd.to_numeric(df_slice['start_time'], errors='coerce')
    if 'end_time' in df_slice.columns:
        en = pd.to_numeric(df_slice['end_time'], errors='coerce')
    elif 'duration' in df_slice.columns:
        en = st + pd.to_numeric(df_slice['duration'], errors='coerce')
    else:
        en = st + 0.5
    en = en.fillna(st + 0.5)
    amp = (pd.to_numeric(df_slice[amp_col], errors='coerce')
           if amp_col in df_slice.columns
           else pd.Series(np.nan, index=df_slice.index))
    uu = (df_slice['uuid'].map(lambda u: None if u is None or u != u
                               else str(u))
          if 'uuid' in df_slice.columns
          else pd.Series(None, index=df_slice.index, dtype=object))
    out = pd.DataFrame({'_start': st, '_end': en,
                        '_is_out': (amp > amp_thr).fillna(False),
                        'uuid': uu.astype(object)})
    out = out.dropna(subset=['_start'])
    return out.sort_values('_start', kind='mergesort').reset_index(drop=True)


class _EpochStripViewBox(pg.ViewBox):
    """ViewBox that captures Shift+drag and emits a snapped epoch range.
    Plain click is left to scene().sigMouseClicked on the parent plot.

    ``snap(t0, t1) -> (e0, e1)`` widens the dragged range to epoch edges;
    the owner passes its epoch table's snap so variable-length epochs snap
    to their own edges. Without one a synthetic 30 s grid is used."""

    sigShiftDrag = pyqtSignal(float, float, bool)  # t0, t1, is_finished

    def __init__(self, *a, snap=None, **kw):
        super().__init__(*a, **kw)
        self._snap = snap

    def _snapped(self, t0, t1):
        if callable(self._snap):
            return self._snap(t0, t1)
        grid = EpochTable.grid(max(t0, t1) + DEFAULT_EPOCH_S)
        return grid.snap(t0, t1)

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            ev.accept()
            p0 = self.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.mapSceneToView(ev.scenePos())
            t0, t1 = sorted([float(p0.x()), float(p1.x())])
            e0, e1 = self._snapped(t0, t1)
            self.sigShiftDrag.emit(float(e0), float(e1), ev.isFinish())
            return
        super().mouseDragEvent(ev, axis=axis)


def _review_settings():
    """The review GUI's ``QSettings`` (org ``turtlewave``, app
    ``eeg_review_gui``, the names ``main()`` pins)."""
    return QtCore.QSettings('turtlewave', 'eeg_review_gui')


def _setting_bool(key, default):
    v = _review_settings().value(key, default)
    if isinstance(v, str):
        return v.lower() in ('1', 'true', 'yes')
    return bool(v)


def _highpass(data, sfreq, lo):
    try:
        b, a = signal.butter(2, max(1e-3, min(lo / (sfreq / 2.0), 0.999)),
                             btype='high')
        return signal.filtfilt(b, a, data)
    except Exception:
        return data


class _Collapsible(QWidget):
    """Header button ``▾ TITLE`` that shows / hides a body; open state is
    persisted under ``settings_key``."""

    toggled = pyqtSignal(bool)

    def __init__(self, title, settings_key, parent=None):
        super().__init__(parent)
        self._title = title
        self._key = settings_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.head = QPushButton()
        self.head.setFlat(True)
        self.head.setFocusPolicy(Qt.NoFocus)
        self.head.setStyleSheet("text-align:left;color:#888888;font-size:10px;"
                                "font-weight:600;letter-spacing:0.05em;")
        self.head.clicked.connect(lambda: self.set_open(not self._open))
        lay.addWidget(self.head)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        self.body_lay.setSpacing(1)
        lay.addWidget(self.body)
        self._open = _setting_bool(settings_key, True)
        self._apply()

    def is_open(self):
        return self._open

    def set_title(self, title):
        self._title = title
        self._apply()

    def set_open(self, on):
        self._open = bool(on)
        _review_settings().setValue(self._key, self._open)
        self._apply()
        self.toggled.emit(self._open)

    def _apply(self):
        self.head.setText(("▾ " if self._open else "▸ ") + self._title)
        self.body.setVisible(self._open)


class NeighboursPlot(EEGDetailWidget):
    """``EEGDetailWidget`` drawing the selected event on its neighbours:
    target on top in the accent colour, one shared fixed scale, the event
    span shaded and the neighbours' own same-type events shaded per row."""

    def plot_neighbours(self, ts, data, labels, event_span, scale,
                        neighbour_spans=None, detected=None):
        """``data`` rows follow ``labels``; row 0 is the target."""
        self.clear()
        self.channel_curves, self.channel_labels, self.event_items = [], [], []
        ts = np.asarray(ts, dtype=float)
        n = len(labels)
        step = 2.0 * scale
        x0, x1 = float(ts[0]), float(ts[-1])
        for i, lab in enumerate(labels):
            y = -i * step
            pen = (pg.mkPen(THEME['accent'], width=1.5) if i == 0
                   else pg.mkPen(THEME['text_2'], width=1))
            row = np.clip(np.asarray(data[i], dtype=float), -scale, scale)
            self.channel_curves.append(self.plot(ts, row + y, pen=pen))
            for s, e in (neighbour_spans or {}).get(lab, []):
                r = QtWidgets.QGraphicsRectItem(max(s, x0), y - scale,
                                                min(e, x1) - max(s, x0), step)
                r.setBrush(pg.mkBrush(136, 136, 136, 40))
                r.setPen(pg.mkPen(None))
                self.addItem(r)
                self.event_items.append(r)
            text = lab + (' · detected' if detected and lab in detected
                          else '')
            t = pg.TextItem(text, anchor=(0, 0.5),
                            color=THEME['accent'] if i == 0 else THEME['text_2'])
            t.setPos(x0, y + 0.6 * scale)
            self.addItem(t, ignoreBounds=True)
            self.channel_labels.append(t)
        reg = pg.LinearRegionItem(values=list(event_span), movable=False,
                                  brush=pg.mkBrush(90, 143, 206, 30),
                                  pen=pg.mkPen(None))
        self.addItem(reg)
        self.event_items.append(reg)
        self.setXRange(x0, x1, padding=0)
        self.setYRange(-(n - 1) * step - scale, scale, padding=0)
        self._last_channels = list(labels)

    def row_labels(self):
        return [t.toPlainText() for t in self.channel_labels]


class NeighboursGroup(_Collapsible):
    """``▾ NEIGHBOURS`` under the filtered trace (spec section 8). Reads
    nothing while collapsed; one multi-channel read per selection."""

    def __init__(self, panel):
        super().__init__('NEIGHBOURS', 'review/neighbours_open', panel)
        self.panel = panel
        self.header = QLabel("")
        self.header.setStyleSheet("color:#888888;font-size:11px;")
        self.body_lay.addWidget(self.header)
        self.filter_chk = QCheckBox("Band-filter (run band)")
        self.filter_chk.setFocusPolicy(Qt.NoFocus)
        self.filter_chk.setChecked(
            _setting_bool('review/neighbours_filtered', False))
        self.filter_chk.toggled.connect(self._on_filter)
        self.body_lay.addWidget(self.filter_chk)
        self.plot = NeighboursPlot()
        _theme_plot(self.plot)
        self.plot.hideAxis('left')
        self.plot.setMinimumHeight(160)
        self.body_lay.addWidget(self.plot)
        self.empty = QLabel("Select an event to see it on neighbouring "
                            "channels.")
        self.empty.setStyleSheet("color:#888888;font-size:11px;")
        self.body_lay.addWidget(self.empty)
        self.n_reads = 0
        self._args = None
        self.toggled.connect(lambda on: self._render() if on else None)
        self._show_empty(self.empty.text())

    def _on_filter(self, on):
        _review_settings().setValue('review/neighbours_filtered', bool(on))
        self._render()

    def _show_empty(self, text):
        self.empty.setText(text)
        self.empty.setVisible(True)
        self.plot.setVisible(False)

    def set_event(self, target, row, event_type, band):
        self._args = (target, row, event_type, band)
        lo, hi = band
        self.filter_chk.setText(f"Band-filter ({lo:g}–{hi:g} Hz, run band)")
        if self.is_open():
            self._render()

    def _render(self):
        p = self.panel
        if self._args is None or self._args[1] is None:
            self.header.setText('')
            self._show_empty("Select an event to see it on neighbouring "
                             "channels.")
            return
        target, row, event_type, band = self._args
        if not callable(p.read_channels) or not callable(p.neighbour_provider):
            self._show_empty("Load the EEG file to see neighbouring channels.")
            return
        win = 4.0 if event_type == 'spindle' else 6.0
        s, e = float(row['start_time']), float(row['end_time'])
        mid = (s + e) / 2.0
        t0, t1 = mid - win / 2.0, mid + win / 2.0
        chans, header, labels = p.neighbour_provider(target, win)
        self.header.setText(header)
        self.n_reads += 1
        out = p.read_channels([target] + list(chans), t0, t1)
        if not out:
            self._show_empty("Load the EEG file to see neighbouring channels.")
            return
        ts, data, sf, got = out
        keep = [i for i, c in enumerate(got)]
        data = np.asarray(data, dtype=float)
        if self.filter_chk.isChecked():
            data = np.vstack([_bandpass(r, sf, band[0], band[1]) for r in data])
        data = data - np.nanmedian(data, axis=1, keepdims=True)
        lab_of = dict(zip([target] + list(chans), [labels[0]] + labels[1:]))
        row_labels = [lab_of.get(c, c) for c in got]
        spans, detected = {}, set()
        if callable(p.neighbour_events) and len(chans):
            evs = p.neighbour_events(list(chans), t0, t1)
            for _, r in (evs if evs is not None else pd.DataFrame()).iterrows():
                lab = lab_of.get(str(r['channel']), str(r['channel']))
                spans.setdefault(lab, []).append(
                    (float(r['start_time']), float(r['end_time'])))
                if float(r['start_time']) < e and float(r['end_time']) > s:
                    detected.add(lab)
        scale = 50.0 if event_type == 'spindle' else 150.0
        self.empty.setVisible(False)
        self.plot.setVisible(True)
        self.plot.plot_neighbours(ts, data[keep], row_labels, (s, e), scale,
                                  spans, detected)


#: Physiology rows: kind -> (row label, filter, fixed half-range or None).
PHYSIO_ROWS = {'eog': ('EOG', (0.3, 15.0), 150.0),
               'emg': ('Chin EMG', (10.0, None), 40.0),
               'ecg': ('ECG', None, None)}


class PhysioStrip(_Collapsible):
    """``▾ PHYSIOLOGY (n)`` under Neighbours (spec section 9): typed EOG,
    chin EMG and ECG, X-linked to the raw trace, read once per epoch."""

    def __init__(self, panel):
        super().__init__('PHYSIOLOGY (0)', 'review/physio_open', panel)
        self.panel = panel
        self.rows = []          # (kind, channel, PlotWidget)
        self.legend = QLabel("")
        self.legend.setStyleSheet("color:#888888;font-size:11px;")
        self.body_lay.addWidget(self.legend)
        self.n_reads = 0
        self._window = None
        self.toggled.connect(lambda on: self._render() if on else None)
        self.setVisible(False)

    def set_channels(self, phys):
        """Build the rows from ``physio_channels()``; hidden when empty."""
        for _k, _c, w in self.rows:
            self.body_lay.removeWidget(w)
            w.deleteLater()
        self.rows = []
        phys = phys or {}
        order = [('eog', c) for c in phys.get('eog', [])]
        if phys.get('emg'):
            order.append(('emg', phys['emg']))
        if phys.get('ecg'):
            order.append(('ecg', phys['ecg']))
        for kind, ch in order:
            w = pg.PlotWidget()
            _theme_plot(w)
            w.setFixedHeight(36)
            w.setMouseEnabled(False, False)
            w.setMenuEnabled(False)
            w.hideButtons()
            w.hideAxis('bottom')
            w.getAxis('left').setStyle(showValues=False)
            w.getAxis('left').setWidth(self.panel.AXIS_W)
            w.setXLink(self.panel.raw_plot)
            self.body_lay.insertWidget(self.body_lay.count() - 1, w)
            self.rows.append((kind, ch, w))
        kinds = []
        if phys.get('eog'):
            kinds.append('EOG 0.3–15 Hz')
        if phys.get('emg'):
            kinds.append('chin EMG above 10 Hz')
        if phys.get('ecg'):
            kinds.append('ECG unfiltered')
        if kinds and (phys.get('eog') or phys.get('emg')):
            kinds.append('fixed scales except ECG' if phys.get('ecg')
                         else 'fixed scales')
        self.legend.setText(' · '.join(kinds))
        self.set_title(f"PHYSIOLOGY ({len(self.rows)})")
        self.setVisible(bool(self.rows))

    def row_titles(self):
        return [f"{PHYSIO_ROWS[k][0]} · {c}" for k, c, _w in self.rows]

    def show_window(self, t0, t1):
        self._window = (t0, t1)
        if self.is_open():
            self._render()

    def _render(self):
        if not self.rows or self._window is None \
                or not callable(self.panel.read_channels):
            return
        t0, t1 = self._window
        self.n_reads += 1
        out = self.panel.read_channels([c for _k, c, _w in self.rows], t0, t1)
        for k, c, w in self.rows:
            w.clear()
        if not out:
            return
        ts, data, sf, got = out
        idx = {c: i for i, c in enumerate(got)}
        for kind, ch, w in self.rows:
            label, filt, half = PHYSIO_ROWS[kind]
            title = f"{label} · {ch}"
            if ch not in idx:
                continue
            x = np.asarray(data[idx[ch]], dtype=float)
            if filt and filt[1]:
                x = _bandpass(x, sf, filt[0], filt[1])
            elif filt:
                x = _highpass(x, sf, filt[0])
            w.plot(ts, x, pen=pg.mkPen(THEME['text_2'], width=1))
            if half:
                w.setYRange(-half, half, padding=0)
                scale = f"±{half:g} µV"
            else:
                w.enableAutoRange('y', True)
                scale = 'auto'
            t = pg.TextItem(title, anchor=(0, 0), color=THEME['text_2'])
            t.setPos(t0, half if half else float(np.nanmax(x)) if x.size else 0)
            w.addItem(t, ignoreBounds=True)
            r = pg.TextItem(scale, anchor=(1, 0), color=THEME['text_3'])
            r.setPos(t1, half if half else float(np.nanmax(x)) if x.size else 0)
            w.addItem(r, ignoreBounds=True)


class EpochsPanel(QWidget):
    """Tab 2 — paged scored-epoch viewer with per-channel artefact triage.

    Epochs come from the annotation file's own epoch table (``epochs=`` in
    :meth:`set_channel`), so a cut recording pages through its 1-30 s epochs
    exactly as scored; :meth:`index_at` and :meth:`span` are the only way
    times and epoch ids are converted. With no annotation file a synthetic
    30 s grid over the recording is used.

    `self.plot` is the EPOCH STRIP (one bar per epoch: grey = regular events,
    red stacked on top = outliers under the rule amp > mean+3.5*sd over the
    drilled channel × event_type). Click an epoch bar to jump; Shift+drag
    selects an epoch-aligned range that can be marked as artefact via the
    "Mark N epochs as artefact" button.

    The main view is the current epoch's window: raw + band-filtered traces
    (mouse pan/zoom disabled, X-locked to the epoch). A thin event ticker
    above the raw trace marks regular (grey) and outlier (red) events in
    the active epoch. Sub-epoch artefacts can also be marked by brushing
    the blue region on the trace and clicking *Mark as artefact*.

    Sidecar XML contract unchanged: per-channel artefacts are written
    under rater ``review-qc`` in a separate file; the scorer XML is never
    modified and a ``*.xml.bak`` is saved on first write.

    The hypnogram PlotWidget (`self.hypno`) is kept as an attribute for
    headless-gate compatibility but is hidden in the UI (replaced by the
    strip's epoch context).

    Every event in the window is a band on both traces (grey, red outlier,
    decision colour once reviewed). Clicking the raw trace or the ticker
    selects the event there (:attr:`eventSelected`); selection never moves
    the artefact brush.

    Keys: Left/Right step ±1 epoch (via button shortcuts); P/N jump to
    previous/next epoch with outliers; Esc clears any strip-range selection;
    A/R/U request accept/reject/unsure on the selected event
    (:attr:`decisionRequested`); ]/[ select the next/previous unreviewed
    event.
    """

    dropChannelRequested = pyqtSignal(str)
    globalArtefactConfirmed = pyqtSignal(float, float, str)
    markArtefactRequested = pyqtSignal(str, float, float)   # ch, t0, t1
    unmarkArtefactRequested = pyqtSignal(int)                # interval id
    requestChannel = pyqtSignal(str)                         # re-drill ch
    eventSelected = pyqtSignal(str)                          # event uuid
    decisionRequested = pyqtSignal(str)        # 'accept' | 'reject' | 'unsure'
    selectionCleared = pyqtSignal()            # paging / Esc dropped it
    checkFilterCleared = pyqtSignal()          # chip ✕ / Esc
    epochChanged = pyqtSignal(int)

    #: A click this close (s) to an event that does not contain it selects it.
    PICK_TOLERANCE_S = 0.3
    _DECISION_KEYS = {Qt.Key_A: 'accept', Qt.Key_R: 'reject',
                      Qt.Key_U: 'unsure'}

    # Kept for callers that read it: the synthetic-grid length only. Staging
    # quantities come from the epoch table (index_at / span), never from this.
    EPOCH_LEN = DEFAULT_EPOCH_S
    AXIS_W = 58       # shared left-axis column width (px) for ticker/raw/filt

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel = None
        self._all_events = None
        self._df = None
        self._event_type = 'slow_wave'
        self._trec = 1.0
        self._hypno = None
        self._epochs = EpochTable.grid(self._trec)   # replaced per drill
        self._epoch = 0          # current epoch index
        self._marked = []
        self._agg = None         # _compute_epoch_outliers result, per drill
        self._n_max = 1          # max(n_events) — strip y normalisation
        self._amp_col = AMP_COL.get('slow_wave', 'max_amp')
        self._amp_thr = float('inf')  # cached outlier threshold per drill
        self._amp_n = 0               # n events behind the threshold
        self._band = DEFAULT_BAND.get('slow_wave', (0.5, 2.0))
        self._band_label = ''         # band source, shown on filt header
        # per-event selection + decisions (see _event_frame / set_reviews)
        self._ev = _event_frame(None, float('inf'), 'max_amp')
        self._reviews = {}            # uuid -> (decision, reason, reviewer)
        self._selected_uuid = None
        self._event_items = []        # bands on raw/filt, redrawn in place
        self._ticker_items = []
        self._check_filter = None     # {'col', 'uuids'} from a dock link
        self._last_click = None       # (plot, scene x) for overlap cycling
        self.last_nav = None          # 'moved' | 'wrapped' | 'none'
        self._raw_ytop = 1.0          # glyph row on the raw trace
        # injected by the main window:
        self.read_window = None   # (ch, t0, t1) -> (t, data, sfreq) | None
        # (channels, t0, t1) -> (t, data[n, t], sfreq, labels) | None
        self.read_channels = None
        # target channel -> (neighbour channels, header text, labels)
        self.neighbour_provider = None
        # (channels, t0, t1) -> DataFrame[channel, start_time, end_time]
        self.neighbour_events = None
        # () -> {'eog': [...], 'emg': ch, 'ecg': ch}
        self.physio_provider = None
        lay = QVBoxLayout(self)

        # --- top strip --------------------------------------------------
        top = QHBoxLayout()
        top.addWidget(_h_label("DRILL: CHANNEL"))
        self.chan_combo = QComboBox()
        self.chan_combo.setMinimumWidth(90)
        self.chan_combo.currentTextChanged.connect(self._on_chan_combo)
        top.addWidget(self.chan_combo)
        self.title = QLabel("Drill into a channel from the Channels tab")
        self.title.setStyleSheet("color:#9ba6b5;")
        top.addWidget(self.title)
        # removable check-filter chip (Channels-tab dock link)
        self.chip = QWidget()
        self.chip.setStyleSheet(
            "background:#2c4666;border:1px solid #5a8fce;border-radius:9px;")
        ch_l = QHBoxLayout(self.chip)
        ch_l.setContentsMargins(8, 1, 3, 1)
        self.chip_lbl = QLabel("")
        self.chip_lbl.setStyleSheet("border:0;color:#e5e5e5;font-size:11px;")
        ch_l.addWidget(self.chip_lbl)
        self.chip_x = QPushButton("✕")
        self.chip_x.setFlat(True)
        self.chip_x.setMaximumWidth(18)
        self.chip_x.setFocusPolicy(Qt.NoFocus)
        self.chip_x.setStyleSheet("border:0;color:#e5e5e5;")
        self.chip_x.clicked.connect(lambda: self.clear_check_filter())
        ch_l.addWidget(self.chip_x)
        self.chip.setVisible(False)
        top.addWidget(self.chip)
        top.addStretch()
        # (Overview Y selector removed — strip is event-count based.)
        lay.addLayout(top)

        # --- main split: viewer (left) | marked-ranges (right) ---------
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        lc = QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)

        # Hypnogram PlotWidget: kept as an attribute for headless-gate
        # compatibility, but HIDDEN per user spec — the epoch strip below
        # carries epoch context now.
        self.hypno = pg.PlotWidget()
        _theme_plot(self.hypno)
        self.hypno.setMouseEnabled(False, False)
        self.hypno.setMenuEnabled(False)
        self.hypno.hideButtons()
        self.hypno.hideAxis('left')
        self._hypno_marker = pg.LinearRegionItem(
            brush=(88, 166, 255, 70), movable=False)
        self._hypno_marker.setZValue(10)
        self.hypno.setVisible(False)
        self.hypno.setMaximumHeight(0)
        lc.addWidget(self.hypno)

        # ---- EPOCH STRIP (self.plot, custom viewbox for Shift+drag) ---
        self._strip_vb = _EpochStripViewBox(snap=self.snap)
        self.plot = pg.PlotWidget(viewBox=self._strip_vb)
        _theme_plot(self.plot)
        self.plot.setMaximumHeight(110)
        self.plot.setMouseEnabled(False, False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setLabel('bottom', 'recording time (s)')
        # left label one size smaller than default (#5)
        self.plot.setLabel('left', 'events/epoch', **{'font-size': '9pt'})
        self.plot.getAxis('left').setStyle(showValues=False)
        self.plot.setYRange(0, 1.15, padding=0)
        # persistent overlays on the strip
        self._ov_marker = pg.LinearRegionItem(
            brush=(88, 166, 255, 70), movable=False)
        self._ov_marker.setZValue(10)
        self._strip_range = pg.LinearRegionItem(
            values=[0, 0], brush=(167, 113, 247, 50),
            pen=pg.mkPen('#a371f7', width=1, style=Qt.DashLine),
            movable=False)
        self._strip_range.setZValue(15)
        self._strip_range.hide()
        self._ep_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('w', width=1.4))
        self._ep_cursor.setZValue(20)
        self.plot.scene().sigMouseClicked.connect(self._on_overview_click)
        self._strip_vb.sigShiftDrag.connect(self._on_shift_drag)
        # strip header exposing the active outlier rule + threshold
        self.strip_hdr = QLabel("Outlier rule: —")
        self.strip_hdr.setStyleSheet("color:#9ba6b5;font-size:11px;")
        lc.addWidget(self.strip_hdr)
        lc.addWidget(self.plot)

        # --- epoch navigation bar (Prev | Prev-outlier | label |
        #                            Next-outlier | Next) -------------
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setShortcut("Left")
        self.prev_btn.clicked.connect(self._prev)
        self.prev_out_btn = QPushButton("◀◀ Prev outlier")
        self.prev_out_btn.clicked.connect(self._prev_outlier)
        self.epoch_lbl = QLabel("Epoch —")
        self.epoch_lbl.setAlignment(Qt.AlignCenter)
        self.epoch_lbl.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;color:#d6dee8;")
        self.next_out_btn = QPushButton("Next outlier ▶▶")
        self.next_out_btn.clicked.connect(self._next_outlier)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setShortcut("Right")
        self.next_btn.clicked.connect(self._next)
        for w in (self.prev_btn, self.prev_out_btn):
            nav.addWidget(w)
        nav.addWidget(self.epoch_lbl, 1)
        for w in (self.next_out_btn, self.next_btn):
            nav.addWidget(w)
        lc.addLayout(nav)

        # Strip-range commit bar: "Mark N epochs as artefact" + Clear
        rangebar = QHBoxLayout()
        rb_hint = QLabel("Shift+drag the strip to select epochs · "
                         "Esc clears the selection")
        rb_hint.setStyleSheet("color:#6b7585;font-size:11px;")
        rangebar.addWidget(rb_hint)
        rangebar.addStretch()
        self.mark_n_btn = QPushButton("Mark 0 epochs as artefact")
        self.mark_n_btn.setEnabled(False)
        self.mark_n_btn.clicked.connect(self._mark_strip_range)
        rangebar.addWidget(self.mark_n_btn)
        self.clear_range_btn = QPushButton("Clear range")
        self.clear_range_btn.clicked.connect(self._clear_strip_range)
        rangebar.addWidget(self.clear_range_btn)
        lc.addLayout(rangebar)

        # ---- EVENT TICKER (X-linked to raw_plot, ~22 px tall) ---------
        self.ticker = pg.PlotWidget()
        _theme_plot(self.ticker)
        self.ticker.setMaximumHeight(22)
        self.ticker.setMouseEnabled(False, False)
        self.ticker.setMenuEnabled(False)
        self.ticker.hideButtons()
        # Keep a left axis (do NOT hideAxis — that strips its width) but make
        # it invisible. Its width is matched to raw_plot's in _sync_ticker_axis
        # so ticker bars share raw_plot's data area (events near t0 stay in).
        self.ticker.getAxis('left').setStyle(showValues=False, tickLength=0)
        self.ticker.hideAxis('bottom')
        self.ticker.setYRange(0, 16, padding=0)
        lc.addWidget(self.ticker)

        # --- 30 s raw + filtered trace (fixed window, no zoom) ---------
        self.raw_plot = pg.PlotWidget()
        _theme_plot(self.raw_plot)
        self.raw_plot.setMouseEnabled(False, False)
        self.raw_plot.setMenuEnabled(False)
        self.raw_plot.hideButtons()
        self.raw_plot.setLabel('left', 'raw µV')
        self.raw_plot.setLabel('bottom', 'time (s)')
        self.filt_plot = pg.PlotWidget()
        _theme_plot(self.filt_plot)
        self.filt_plot.setMouseEnabled(False, False)
        self.filt_plot.setMenuEnabled(False)
        self.filt_plot.hideButtons()
        self.filt_plot.setLabel('left', 'filtered µV')
        self.filt_plot.setXLink(self.raw_plot)
        self.ticker.setXLink(self.raw_plot)
        self._sync_axis_widths()   # shared left-axis column (re-asserted per epoch)
        # click an event band (raw trace) or a ticker bar to select it
        self.raw_plot.scene().sigMouseClicked.connect(
            lambda ev: self._on_trace_click(ev, self.raw_plot))
        self.ticker.scene().sigMouseClicked.connect(
            lambda ev: self._on_trace_click(ev, self.ticker))
        self.filt_plot.scene().sigMouseClicked.connect(
            lambda ev: self._on_trace_click(ev, self.filt_plot))
        # brush region — lives on raw_plot, draggable even with viewbox
        # mouse pan/zoom disabled (LinearRegionItem handles its own mouse).
        # Outline-only until dragged: transparent fill by default, fill
        # restored on a genuine user drag (persists once selection committed).
        self.region = pg.LinearRegionItem(brush=(0, 0, 0, 0))
        self.region.setZValue(10)
        self._region_programmatic = False
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.trace_note = QLabel(
            "drag the blue brush on the trace to set an artefact range · "
            "load an EEG file to enable the signal trace")
        self.trace_note.setStyleSheet("color:#6b7585;font-size:11px;")
        # header for the filtered trace: passband + its source
        self.filt_hdr = QLabel("filtered —")
        self.filt_hdr.setStyleSheet("color:#9ba6b5;font-size:11px;")
        lc.addWidget(self.raw_plot, 2)
        lc.addWidget(self.filt_hdr)
        lc.addWidget(self.filt_plot, 2)
        lc.addWidget(self.trace_note)
        # neighbouring channels and EOG / EMG / ECG for the selected event
        self.neighbours = NeighboursGroup(self)
        lc.addWidget(self.neighbours)
        self.physio = PhysioStrip(self)
        lc.addWidget(self.physio)

        # --- action bar ------------------------------------------------
        bar = QHBoxLayout()
        self.sel_lbl = QLabel("brush a range on the trace, then Mark")
        self.sel_lbl.setStyleSheet("color:#6b7585;font-size:11px;")
        bar.addWidget(self.sel_lbl)
        bar.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._reset_region)
        bar.addWidget(self.clear_btn)
        self.mark_btn = QPushButton("Mark as artefact (writes XML)")
        self.mark_btn.setObjectName("primary")
        self.mark_btn.clicked.connect(self._mark)
        bar.addWidget(self.mark_btn)
        lc.addLayout(bar)

        # channel-level actions (preserved)
        crow = QHBoxLayout()
        clbl = QLabel("Channel-level:")
        clbl.setStyleSheet("color:#6b7585;")
        crow.addWidget(clbl)
        self.drop_btn = QPushButton("Drop channel")
        self.drop_btn.setObjectName("danger")
        self.drop_btn.clicked.connect(self._drop)
        crow.addWidget(self.drop_btn)
        # Hidden: _add_global_artefact writes only a channel-scoped DB row
        # (no sidecar XML), so it has NO re-detect effect despite the dialog's
        # whole-montage claim. Misleading half-stub — hide until a real
        # global-artefact mechanism (distinct rater + sidecar) is built.
        self.global_btn = QPushButton("Inspect selection as GLOBAL artefact…")
        self.global_btn.clicked.connect(self._inspect_global)
        self.global_btn.setVisible(False)
        crow.addWidget(self.global_btn)
        crow.addStretch()
        lc.addLayout(crow)
        split.addWidget(left)

        # Standalone "MARKED ARTEFACT RANGES" panel removed — marked ranges
        # now show as purple-dashed overlays on the strip + a compact list
        # in ChannelDetailDock. ranges_list kept as a HIDDEN attribute (still
        # populated by _set_ranges) for headless-gate compatibility (P4
        # asserts ranges_list.count()).
        self.ranges_list = QtWidgets.QListWidget()
        self.ranges_list.setParent(self)
        self.ranges_list.setVisible(False)
        split.setSizes([1100])
        lay.addWidget(split)

        # PageUp/PageDown also page (Left/Right covered by button shortcuts)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---- epoch table --------------------------------------------------
    def index_at(self, t):
        """Epoch id containing recording time ``t`` (clamped)."""
        return self._epochs.index_at(t)

    def span(self, i):
        """``(start_s, end_s)`` of epoch ``i`` (clamped)."""
        return self._epochs.span(i)

    def snap(self, t0, t1):
        """Widen ``[t0, t1]`` outward to epoch edges."""
        return self._epochs.snap(t0, t1)

    # ---- population ---------------------------------------------------
    def set_channel(self, channel, df_slice, all_events, event_type=None,
                     hypno=None, trec=None, marked=None, epochs=None,
                     reviews=None):
        """Drill into ``channel``.

        ``epochs`` is the scored-epoch table (an :class:`EpochTable` or a
        list of ``(start_s, end_s, stage)``); pass it for any annotation
        file. ``hypno`` (a bare stage list, one 30 s epoch each) is kept for
        older callers; with neither, a synthetic 30 s grid over ``trec`` is
        used. ``reviews`` is ``{uuid: (decision, reason, reviewer)}`` for
        this channel's events (see :meth:`set_reviews`). Any selected event
        is cleared.
        """
        self._reviews = dict(reviews or {})
        self._selected_uuid = None
        self._last_click = None
        self.clear_check_filter(emit=False)
        self._channel = channel
        self._all_events = all_events
        self._df = df_slice
        if event_type:
            self._event_type = event_type
        if trec:
            self._trec = float(trec)
        self._epochs = _as_epoch_table(epochs, hypno, self._trec)
        self._hypno = (list(self._epochs.stages)
                       if (epochs is not None or hypno) else None)
        # event-type-aware amp column, then per-drill outlier aggregation
        self._amp_col = AMP_COL.get(str(self._event_type), 'max_amp')
        if self._amp_col not in (df_slice.columns
                                  if df_slice is not None else []):
            self._amp_col = 'max_amp'
        self._agg = _compute_epoch_outliers(
            df_slice, epochs=self._epochs, amp_col=self._amp_col)
        self._n_max = int(self._agg['n_events'].max()) \
            if len(self._agg) else 1
        # cache the robust outlier threshold once per drill (median + MAD;
        # used by the trace overlay + ticker for the active epoch)
        amp = (df_slice[self._amp_col]
               if df_slice is not None and self._amp_col in df_slice.columns
               else pd.Series(dtype=float))
        self._amp_thr, self._amp_n = _mad_threshold(amp)
        self._ev = _event_frame(df_slice, self._amp_thr, self._amp_col)
        # filtered-trace passband, sourced from the event rows themselves
        lo, hi, src = _band_for(df_slice, self._event_type)
        self._band = (lo, hi)
        self._band_label = src
        self.filt_hdr.setText(f"filtered {lo:g}–{hi:g} Hz ({src})")
        # expose the active rule + threshold in the strip header
        if np.isfinite(self._amp_thr):
            self.strip_hdr.setText(
                f"Outlier rule: amp > {self._amp_thr:.1f} µV "
                f"(median + 3.5·MAD, n={self._amp_n})")
        else:
            self.strip_hdr.setText(
                f"Outlier rule: n/a — insufficient spread (n={self._amp_n})")

        n = 0 if df_slice is None else len(df_slice)
        dens = ''
        if n and trec:
            dens = f" · density {n / (trec / 60.0):.2f} ev/min"
        self.title.setText(f"<b>{channel}</b> · {self._event_type} · "
                           f"n={n}{dens}")
        # channel dropdown (montage-wide)
        chans = []
        if all_events is not None and len(all_events):
            chans = sorted(all_events['channel'].astype(str).unique())
        if channel and channel not in chans:
            chans = [channel] + chans
        self.chan_combo.blockSignals(True)
        self.chan_combo.clear()
        self.chan_combo.addItems(chans or ([channel] if channel else []))
        if channel:
            self.chan_combo.setCurrentText(channel)
        self.chan_combo.blockSignals(False)
        self._set_hypno(self._hypno)
        self._set_ranges(marked or [])
        self._render_strip()
        self._clear_strip_range()
        # pick a sensible starting epoch: first epoch with outliers (the
        # one the reviewer probably wants to look at), else first epoch
        # with any event, else 0.
        start_ep = 0
        out_ix = self._outlier_epoch_indices()
        if out_ix:
            start_ep = int(out_ix[0])
        elif df_slice is not None and len(df_slice):
            try:
                st = pd.to_numeric(df_slice['start_time'],
                                    errors='coerce').dropna()
                if len(st):
                    start_ep = self.index_at(st.min())
            except Exception:
                pass
        self._goto_epoch(start_ep)

    def _on_chan_combo(self, ch):
        if ch and ch != self._channel:
            # re-drill through the window (keeps DB/slice logic in one place)
            self.requestChannel.emit(ch)

    def _set_hypno(self, hypno):
        """Hidden stage line over the epoch table, one ``[start, end]`` pair
        per epoch (drawn only when the drill carries stages)."""
        self.hypno.clear()
        # marker survives clear() by re-adding (LinearRegionItem, not data item)
        self.hypno.addItem(self._hypno_marker)
        if not hypno:
            return
        tb = self._epochs
        ys = np.array([HYPNO_RANK.get(str(s), 2) for s in tb.stages],
                      dtype=float)
        xs = np.column_stack([tb.starts, tb.ends]).ravel()
        self.hypno.plot(xs, np.repeat(ys, 2),
                        pen=pg.mkPen((120, 160, 200), width=2),
                        connect='pairs')
        self.hypno.setXLink(self.plot)

    def _set_ranges(self, marked):
        self._marked = list(marked)
        self.ranges_list.clear()
        for m in self._marked:
            t0, t1 = float(m['start_time']), float(m['end_time'])
            it = QtWidgets.QListWidgetItem(
                f"{_hms(t0)} – {_hms(t1)}  "
                f"({self._epochs.count_in(t0, t1)} ep)")
            it.setData(Qt.UserRole, int(m['id']))
            it.setData(Qt.UserRole + 1, float(t0))   # for jump-to
            self.ranges_list.addItem(it)
        # overlays inside the current 30 s window are redrawn by _goto_epoch

    # ---- plotting -----------------------------------------------------
    def _render_strip(self):
        """Epoch strip: ONE BarGraphItem per layer, vectorised.

        Bottom layer (grey)   = (n_events - n_outliers) / max(n_events) * H.
        Top layer    (red)    = n_outliers / max(n_events) * H.
        Marked-artefact bands overlay as dashed purple LinearRegionItems.
        Selected-epoch InfiniteLine + Shift-drag LinearRegionItem are
        persistent items re-added after self.plot.clear().
        """
        self.plot.clear()
        # re-add persistent items (clear() drops them)
        self.plot.addItem(self._ov_marker)
        self.plot.addItem(self._strip_range)
        self.plot.addItem(self._ep_cursor)
        for it in getattr(self, '_sample_marks', []):
            self.plot.addItem(it)
        self.plot.setXRange(0, max(self._trec, self._epochs.end), padding=0)
        if self._agg is None or len(self._agg) == 0:
            return
        idx = self._agg['idx'].to_numpy(dtype=int)
        n_ev = self._agg['n_events'].to_numpy(dtype=float)
        n_out = self._agg['n_outliers'].to_numpy(dtype=float)
        denom = max(1.0, float(self._n_max))
        H = 1.0
        h_reg = (n_ev - n_out) / denom * H
        h_out = n_out / denom * H
        # each bar spans its own epoch (1-30 s on cut recordings)
        t0s = self._epochs.starts[idx]
        t1s = self._epochs.ends[idx]
        centres = (t0s + t1s) / 2.0
        widths = (t1s - t0s) * 0.95
        # bottom (grey) layer — one item, vectorised
        self.plot.addItem(pg.BarGraphItem(
            x=centres, width=widths,
            y0=0, height=h_reg, brush=THEME['text_3'], pen=None))
        # top (red) layer stacked on top
        self.plot.addItem(pg.BarGraphItem(
            x=centres, width=widths,
            y0=h_reg, height=h_out, brush=THEME['bad'], pen=None))
        # marked-artefact bands (dashed purple, transparent fill)
        edge = pg.mkPen('#a371f7', width=1, style=Qt.DashLine)
        for m in self._marked:
            self.plot.addItem(pg.LinearRegionItem(
                values=[float(m['start_time']), float(m['end_time'])],
                brush=(0, 0, 0, 0), pen=edge, movable=False))

    def _draw_window_overlays(self):
        """Red strips for marked artefact ranges intersecting the current
        epoch window. Called only from _goto_epoch (right after raw/filt
        clears), so we never accumulate stale overlays."""
        t0, t1 = self.span(self._epoch)
        for m in self._marked:
            a, b = float(m['start_time']), float(m['end_time'])
            if b < t0 or a > t1:
                continue
            for p in (self.raw_plot, self.filt_plot):
                it = pg.LinearRegionItem(
                    values=[max(a, t0), min(b, t1)],
                    brush=(248, 81, 73, 50), movable=False)
                it.setZValue(5)
                p.addItem(it)

    def _epoch_stage(self):
        if not self._hypno:
            return ''
        return self._epochs.stage(self._epoch)

    def _n_epochs(self):
        return max(1, len(self._epochs))

    @staticmethod
    def _default_region(t0, t1):
        """Brush placed at the epoch centre: 4 s wide on a 30 s epoch (as
        before), shrunk to 30 % of shorter epochs so it stays inside."""
        mid = (t0 + t1) / 2.0
        half = min(2.0, 0.15 * (t1 - t0))
        return [mid - half, mid + half]

    def _goto_epoch(self, i):
        n = self._n_epochs()
        i = max(0, min(int(i), n - 1))
        self._epoch = i
        t0, t1 = self.span(i)
        # paging keeps the selection only when the event starts here
        dropped = False
        if self._selected_uuid is not None:
            hit = self._ev[self._ev['uuid'] == self._selected_uuid]
            s0 = float(hit['_start'].iloc[0]) if len(hit) else None
            if s0 is None or not (t0 <= s0 < t1):
                self._selected_uuid = None
                dropped = True
        dur = t1 - t0
        stage = self._epoch_stage() or '—'
        n_ev, n_out = self._epoch_counts(i)
        # name the length only when it is not the standard 30 s
        dtxt = ('' if abs(dur - DEFAULT_EPOCH_S) < 1e-6
                else f" ({_epoch_len_text(dur)})")
        self.epoch_lbl.setText(
            f"Epoch {i + 1}/{n} · {_hms(t0)}–{_hms(t1)}{dtxt} · {stage} · "
            f"{n_ev} events ({n_out} outlier{'s' if n_out != 1 else ''})")
        # move overview + hypno current-epoch markers + strip cursor
        self._ov_marker.setRegion([t0, t1])
        self._hypno_marker.setRegion([t0, t1])
        self._ep_cursor.setValue((t0 + t1) / 2.0)
        # render the epoch's raw + filtered window + ticker
        self.raw_plot.clear()
        self.filt_plot.clear()
        self.ticker.clear()
        self._event_items = []
        self._ticker_items = []
        # re-add brush region (outline-only) centred in window; guard the
        # programmatic setRegion so it stays unfilled until the user drags.
        self._region_programmatic = True
        self.region.setRegion(self._default_region(t0, t1))
        self._region_programmatic = False
        self._region_fill(False)
        self.raw_plot.addItem(self.region)
        # lock to exactly the epoch
        for p in (self.raw_plot, self.filt_plot):
            p.setXRange(t0, t1, padding=0)
            p.enableAutoRange('x', False)
        self.ticker.setXRange(t0, t1, padding=0)
        # fixed filt y-range per event type (raw stays autoscaled) so the
        # reviewer's eye doesn't recalibrate on every click.
        yr = FILT_YRANGE.get(self._event_type, 50)
        self.filt_plot.setYRange(-yr, yr, padding=0)
        self.filt_plot.enableAutoRange('y', False)
        # re-assert the shared left-axis column width (holds once the panel
        # is visible) → ticker bars align with the trace data area.
        self._sync_axis_widths()
        out = None
        if callable(self.read_window) and self._channel is not None:
            try:
                out = self.read_window(self._channel, t0, t1)
            except Exception:
                out = None
        if out:
            ts, data, sfreq = out
            data = np.asarray(data, dtype=float).ravel()
            ts = np.asarray(ts, dtype=float).ravel()
            if data.size and ts.size == data.size:
                self.trace_note.setText(
                    f"signal trace · {self._channel} · "
                    f"epoch {i + 1}/{n} ({_hms(t0)}–{_hms(t1)})")
                self.raw_plot.plot(
                    ts, data, pen=pg.mkPen((155, 166, 181), width=1))
                fin = data[np.isfinite(data)]
                self._raw_ytop = float(fin.max()) if fin.size else 1.0
                lo, hi = self._band   # sourced from event rows in set_channel
                self.filt_plot.plot(
                    ts, _bandpass(data, sfreq, lo, hi),
                    pen=pg.mkPen(QtGui.QColor(
                        EVT_COLOR.get(self._event_type, '#5fd3a4')),
                        width=1))
            else:
                self.trace_note.setText(
                    f"signal trace · {self._channel} · "
                    f"no data in epoch {i + 1}/{n}")
        else:
            self.trace_note.setText(
                "load an EEG file to enable the signal trace")
            self._raw_ytop = 1.0
        self._draw_window_overlays()
        self._draw_trace_events(t0, t1)
        self._draw_ticker(t0, t1)
        self.physio.show_window(t0, t1)
        if dropped:
            self._refresh_neighbours()
            self.selectionCleared.emit()
        self.epochChanged.emit(i)

    # ---- epoch-event utilities ----------------------------------------
    def _epoch_counts(self, i):
        if self._agg is None or len(self._agg) == 0:
            return 0, 0
        row = self._agg[self._agg['idx'] == i]
        if len(row) == 0:
            return 0, 0
        r = row.iloc[0]
        return int(r['n_events']), int(r['n_outliers'])

    def _epoch_events(self, t0, t1):
        """Return events in [t0,t1) with _start/_end/_is_out columns.
        Uses the per-drill cached threshold self._amp_thr (computed once
        in set_channel) — no per-epoch recomputation."""
        if self._df is None or len(self._df) == 0:
            return pd.DataFrame()
        col = self._amp_col if self._amp_col in self._df.columns else 'max_amp'
        st = pd.to_numeric(self._df['start_time'], errors='coerce')
        amp = pd.to_numeric(self._df.get(col), errors='coerce')
        mask = st.notna() & (st >= t0) & (st < t1)
        sub = self._df.loc[mask].copy()
        sub['_start'] = st[mask]
        sub['_amp'] = amp[mask]
        sub['_is_out'] = sub['_amp'] > self._amp_thr
        if 'end_time' in sub.columns:
            sub['_end'] = pd.to_numeric(sub['end_time'], errors='coerce')
        elif 'duration' in sub.columns:
            sub['_end'] = sub['_start'] + pd.to_numeric(
                sub['duration'], errors='coerce').fillna(0.5)
        else:
            sub['_end'] = sub['_start'] + 0.5
        return sub

    def _window_events(self, t0, t1):
        """Rows of the per-drill event frame that START in ``[t0, t1)``
        (spec: an event that runs past the epoch is drawn clipped)."""
        ev = self._ev
        if ev is None or ev.empty:
            return ev
        return ev[(ev['_start'] >= t0) & (ev['_start'] < t1)]

    def _decision_of(self, uuid):
        dec = self._reviews.get(uuid) if uuid is not None else None
        return dec[0] if dec else None

    def _band_style(self, uuid, is_out, filtered_out=False):
        """``(fill rgba, edge pen, z, glyph, glyph colour)`` of one band
        (UX spec section 3). A decision replaces the outlier style; the
        selected event's edge is replaced by a 2 px accent line on top."""
        dec = self._decision_of(uuid)
        c3, bad, ok, warn = (QtGui.QColor(THEME[k]) for k in
                             ('text_3', 'bad', 'ok', 'warn'))

        def rgb(c):
            return (c.red(), c.green(), c.blue())
        glyph, gcol = None, None
        if dec == 'accept':
            fill, a = rgb(ok), 40
            pen, z = pg.mkPen(*rgb(ok), width=1), 4
            glyph, gcol = '✓', THEME['ok']
        elif dec == 'reject':
            fill, a = rgb(c3), 20
            pen, z = pg.mkPen(*rgb(bad), width=1, style=Qt.DashLine), 4
            glyph, gcol = '✗', THEME['bad']
        elif dec == 'unsure':
            fill, a = rgb(warn), 40
            pen, z = pg.mkPen(*rgb(warn), width=1, style=Qt.DotLine), 4
            glyph, gcol = '?', THEME['warn']
        elif is_out:
            fill, a = rgb(bad), 46
            pen, z = pg.mkPen(*rgb(bad), 140), 4
        else:
            fill, a = rgb(c3), 30
            pen, z = pg.mkPen(None), 3
        if filtered_out:
            a = a // 2
        if uuid is not None and uuid == self._selected_uuid:
            pen, z = pg.mkPen(THEME['accent'], width=2), 8
        return (*fill, a), pen, z, glyph, gcol

    def _passes_filter(self, uuid):
        return (self._check_filter is None
                or uuid in self._check_filter['uuids'])

    def _draw_trace_events(self, t0, t1):
        """One band per event starting in the window, on the raw and the
        filtered trace, clipped at the epoch end, styled by
        :meth:`_band_style`; decided events carry a glyph on the raw trace.
        Replaces the previous bands in place, so selecting or deciding
        redraws only these items, never the traces or the artefact brush."""
        for p, it in self._event_items:
            p.removeItem(it)
        self._event_items = []
        sub = self._window_events(t0, t1)
        if sub is None or sub.empty:
            return
        ytop = self._raw_ytop
        for s, e, out, u in zip(sub['_start'].to_numpy(dtype=float),
                                sub['_end'].to_numpy(dtype=float),
                                sub['_is_out'].to_numpy(dtype=bool),
                                sub['uuid'].tolist()):
            fill, pen, z, glyph, gcol = self._band_style(
                u, out, not self._passes_filter(u))
            e = min(e, t1)
            for p in (self.raw_plot, self.filt_plot):
                it = pg.LinearRegionItem(values=[s, e], brush=fill, pen=pen,
                                         movable=False)
                it.setZValue(z)
                p.addItem(it)
                self._event_items.append((p, it))
            if glyph:
                txt = pg.TextItem(glyph, color=gcol, anchor=(0, 0))
                f = txt.textItem.font()
                f.setPixelSize(10)
                txt.setFont(f)
                txt.setPos(s, ytop)
                txt.setZValue(z + 1)
                txt.uuid = u
                self.raw_plot.addItem(txt, ignoreBounds=True)
                self._event_items.append((self.raw_plot, txt))

    def band_items(self, plot=None):
        """The current band items (``LinearRegionItem``), for tests."""
        return [it for p, it in self._event_items
                if isinstance(it, pg.LinearRegionItem)
                and (plot is None or p is plot)]

    def glyph_items(self):
        return [it for p, it in self._event_items
                if isinstance(it, pg.TextItem)]

    def _draw_ticker(self, t0, t1):
        # Pure visual indicator, same events as the bands: grey = regular,
        # red (taller) = outlier, green / grey outline / amber = accepted /
        # rejected / unsure, accent outline = selected.
        for it in self._ticker_items:
            self.ticker.removeItem(it)
        self._ticker_items = []
        sub = self._window_events(t0, t1)
        if sub is None or sub.empty:
            return
        uu = sub['uuid']
        dec = uu.map(self._decision_of)
        reviewed = dec.notna()
        grey = THEME['text_3']
        layers = [(sub[~sub['_is_out'] & ~reviewed], 0.4, 9, grey, None),
                  (sub[sub['_is_out'] & ~reviewed], 0.6, 14, THEME['bad'], None),
                  (sub[dec == 'accept'], 0.4, 9, THEME['ok'], None),
                  (sub[dec == 'reject'], 0.4, 9, None, pg.mkPen(grey, width=1)),
                  (sub[dec == 'unsure'], 0.4, 9, THEME['warn'], None)]
        # Visible widths: 0.4 s reg / 0.6 s outlier — narrower than the
        # smallest spindle (~0.5 s) but reliably visible at 1920 px.
        for rows, w, h, brush, pen in layers:
            if len(rows):
                it = pg.BarGraphItem(x=rows['_start'].to_numpy(dtype=float),
                                     width=w, y0=0, height=h,
                                     brush=brush, pen=pen)
                self.ticker.addItem(it)
                self._ticker_items.append(it)
        sel = sub[uu == self._selected_uuid] if self._selected_uuid else sub[:0]
        if len(sel):
            it = pg.BarGraphItem(x=sel['_start'].to_numpy(dtype=float),
                                 width=0.6, y0=0, height=16, brush=None,
                                 pen=pg.mkPen(THEME['accent'], width=1.5))
            it.setZValue(5)
            self.ticker.addItem(it)
            self._ticker_items.append(it)

    # ---- event selection + decisions ----------------------------------
    def set_reviews(self, reviews):
        """Replace the decision map ``{uuid: (decision, reason, reviewer)}``
        (the current reviewer's decisions only) and restyle the window."""
        self._reviews = dict(reviews or {})
        self._redraw_event_layers()

    def _redraw_event_layers(self):
        t0, t1 = self.span(self._epoch)
        self._draw_trace_events(t0, t1)
        self._draw_ticker(t0, t1)

    def selected_event(self):
        """The selected event's row of the drilled slice, or ``None``."""
        if self._selected_uuid is None or self._df is None \
                or 'uuid' not in self._df.columns:
            return None
        hit = self._df[self._df['uuid'] == self._selected_uuid]
        return hit.iloc[0] if len(hit) else None

    def _events_at(self, x):
        """uuids of events in the window that contain ``x``, nearest centre
        first; else the single nearest within :attr:`PICK_TOLERANCE_S`."""
        t0, t1 = self.span(self._epoch)
        sub = self._window_events(t0, t1)
        if sub is None or sub.empty:
            return []
        sub = sub[sub['uuid'].notna()]
        if sub.empty:
            return []
        s = sub['_start'].to_numpy(dtype=float)
        e = np.minimum(sub['_end'].to_numpy(dtype=float), t1)
        inside = (s <= x) & (x <= e)
        if inside.any():
            ix = np.flatnonzero(inside)
            ix = ix[np.argsort(np.abs((s[ix] + e[ix]) / 2.0 - x),
                               kind='mergesort')]
            return [str(sub['uuid'].iloc[i]) for i in ix]
        gap = np.minimum(np.abs(s - x), np.abs(e - x))
        best = int(np.argmin(gap))
        if gap[best] <= self.PICK_TOLERANCE_S + 1e-9:
            return [str(sub['uuid'].iloc[best])]
        return []

    def _event_at(self, x):
        """uuid of the event at time ``x`` (see :meth:`_events_at`)."""
        hits = self._events_at(x)
        return hits[0] if hits else None

    def _on_trace_click(self, ev, plot):
        """Left click on the raw or filtered trace, or the ticker, selects
        an event; a second click within 2 px cycles overlapping events; a
        click near nothing keeps the current selection."""
        try:
            if ev.button() != Qt.LeftButton:
                return
            vb = plot.getPlotItem().vb
            pos = ev.scenePos()
            if not vb.sceneBoundingRect().contains(pos):
                return
            x = float(vb.mapSceneToView(pos).x())
            px = float(pos.x())
        except Exception:
            return
        hits = self._events_at(x)
        if hits:
            last = self._last_click
            if (last is not None and last[0] is plot
                    and abs(last[1] - px) <= 2.0 and len(hits) > 1
                    and self._selected_uuid in hits):
                uuid = hits[(hits.index(self._selected_uuid) + 1) % len(hits)]
            else:
                uuid = hits[0]
            self._last_click = (plot, px)
            self.select_event(uuid)
        self.setFocus()      # so the review keys reach this panel

    def select_event(self, uuid, emit=True):
        """Select the event ``uuid`` of the drilled channel.

        Pages to the epoch the event starts in only when it is not the
        current one, so selecting inside the window never moves the
        artefact brush. Emits :attr:`eventSelected` unless ``emit`` is
        False. Returns True when the event exists.
        """
        ev = self._ev
        if uuid is None or ev is None or ev.empty:
            return False
        hit = ev[ev['uuid'] == str(uuid)]
        if hit.empty:
            return False
        self._selected_uuid = str(uuid)
        s = float(hit['_start'].iloc[0])
        t0, t1 = self.span(self._epoch)
        if t0 <= s < t1:
            self._redraw_event_layers()
        else:
            self._goto_epoch(self.index_at(s))
        if emit:
            self.eventSelected.emit(self._selected_uuid)
        self._refresh_neighbours()
        return True

    def clear_selection(self, emit=True):
        """Drop the selection (restyles the window only)."""
        if self._selected_uuid is None:
            return False
        self._selected_uuid = None
        self._redraw_event_layers()
        self._refresh_neighbours()
        if emit:
            self.selectionCleared.emit()
        return True

    def _anchor_pos(self):
        """Index in ``_ev`` of the selected event when it starts in the
        current epoch, else ``None``."""
        if self._selected_uuid is None:
            return None
        ev = self._ev
        t0, t1 = self.span(self._epoch)
        hit = ev.index[(ev['uuid'] == self._selected_uuid)
                       & (ev['_start'] >= t0) & (ev['_start'] < t1)]
        return int(hit[0]) if len(hit) else None

    def _step(self, ok, step, wrap=False):
        """Select the next (``step`` > 0) / previous event among rows where
        ``ok`` is True, from the selected event when it is on screen, else
        from the current epoch's start. ``wrap`` goes round once. Sets
        :attr:`last_nav` to ``'moved'``, ``'wrapped'`` or ``'none'``."""
        ev = self._ev
        self.last_nav = 'none'
        if ev is None or ev.empty:
            return None
        t0, _t1 = self.span(self._epoch)
        pos = self._anchor_pos()
        if pos is not None:
            after = ev.index > pos if step > 0 else ev.index < pos
        else:
            after = (ev['_start'] >= t0) if step > 0 else (ev['_start'] < t0)
        cand = ev.index[ok & after]
        if len(cand):
            self.last_nav = 'moved'
        elif wrap:
            cand = ev.index[ok]
            if pos is not None:
                cand = cand[cand != pos]
            if len(cand):
                self.last_nav = 'wrapped'
        if not len(cand):
            return None
        u = str(ev.at[cand[0] if step > 0 else cand[-1], 'uuid'])
        self.select_event(u)
        return u

    def _unreviewed_mask(self):
        ev = self._ev
        return ev['uuid'].notna() & ~ev['uuid'].isin(list(self._reviews))

    def next_unreviewed(self, wrap=True):
        """Select the next unreviewed event (``]``), wrapping once."""
        return self._step(self._unreviewed_mask(), +1, wrap)

    def prev_unreviewed(self, wrap=True):
        """Select the previous unreviewed event (``[``), wrapping once."""
        return self._step(self._unreviewed_mask(), -1, wrap)

    def _filter_mask(self):
        ev = self._ev
        ok = ev['uuid'].notna()
        if self._check_filter is not None:
            ok &= ev['uuid'].isin(list(self._check_filter['uuids']))
        return ok

    def next_event(self):
        """Next event of any status (``}``); only filtered events while a
        check filter is on."""
        return self._step(self._filter_mask(), +1)

    def prev_event(self):
        """Previous event of any status (``{``)."""
        return self._step(self._filter_mask(), -1)

    def n_unreviewed(self):
        ev = self._ev
        return 0 if ev is None or ev.empty else int(self._unreviewed_mask().sum())

    # ---- check filter (Channels-tab link) -----------------------------
    def set_check_filter(self, col, chip_text, uuids):
        """Show only events failing one population check: the others are
        drawn at half fill and skipped by ``}`` / ``{``."""
        self._check_filter = {'col': col, 'uuids': set(uuids)}
        self.chip_lbl.setText(chip_text[:-2] if chip_text.endswith(' ✕')
                              else chip_text)
        self.chip.setVisible(True)
        self._redraw_event_layers()

    def clear_check_filter(self, emit=True):
        if self._check_filter is None:
            return False
        self._check_filter = None
        self.chip.setVisible(False)
        self._redraw_event_layers()
        if emit:
            self.checkFilterCleared.emit()
        return True

    def chip_text(self):
        """The chip as shown, with its ✕ (empty when no filter is on)."""
        return (self.chip_lbl.text() + ' ✕') if self.chip.isVisible() else ''

    def set_sample_marks(self, times):
        """2 px accent ticks on the epoch strip at the sample events of the
        drilled channel (empty list: none)."""
        for it in getattr(self, '_sample_marks', []):
            self.plot.removeItem(it)
        self._sample_marks = []
        for t in times or []:
            it = pg.InfiniteLine(pos=float(t), angle=90, movable=False,
                                 pen=pg.mkPen(THEME['accent'], width=2))
            it.setZValue(18)
            self.plot.addItem(it)
            self._sample_marks.append(it)

    def _refresh_neighbours(self):
        """Hand the selection to the Neighbours group (it reads only when
        open)."""
        if hasattr(self, 'neighbours'):
            self.neighbours.set_event(self._channel, self.selected_event(),
                                      self._event_type, self._band)

    def escape_step(self):
        """Esc after any armed decision / comment step: clear the strip
        range, else remove the check filter, else clear the selection.
        Returns what was done (or ``None``)."""
        if self._strip_range.isVisible():
            self._clear_strip_range()
            return 'strip'
        if self.clear_check_filter():
            return 'filter'
        if self.clear_selection():
            return 'selection'
        return None

    def _sync_axis_widths(self):
        """Pin a common left-axis column width on ticker + raw + filt so their
        data areas start at the same x (events near t0 stay inside the data
        area). Fixed rather than matched to autoscaled raw — reading an
        autoscaled axis width is one paint behind, which aligned fragilely."""
        for ax in (self.raw_plot.getAxis('left'),
                   self.filt_plot.getAxis('left'),
                   self.ticker.getAxis('left')):
            ax.setWidth(self.AXIS_W)

    # ---- navigation ---------------------------------------------------
    def _prev(self):
        self._goto_epoch(self._epoch - 1)

    def _next(self):
        self._goto_epoch(self._epoch + 1)

    def _outlier_epoch_indices(self):
        if self._agg is None or len(self._agg) == 0:
            return []
        return self._agg.loc[self._agg['n_outliers'] > 0,
                              'idx'].astype(int).tolist()

    def _prev_outlier(self):
        ix = [j for j in self._outlier_epoch_indices() if j < self._epoch]
        if ix:
            self._goto_epoch(max(ix))

    def _next_outlier(self):
        ix = [j for j in self._outlier_epoch_indices() if j > self._epoch]
        if ix:
            self._goto_epoch(min(ix))

    def keyPressEvent(self, ev):
        # Used when the panel is shown on its own; inside the main window the
        # review keys are window shortcuts on the Epochs tab and reach the
        # same methods. Left/Right are button shortcuts; P/N hop outliers.
        k = ev.key()
        plain = not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier
                                       | Qt.MetaModifier))
        if k == Qt.Key_P:
            self._prev_outlier(); return
        if k == Qt.Key_N:
            self._next_outlier(); return
        if k == Qt.Key_Escape:
            self.escape_step(); return
        if plain and k in self._DECISION_KEYS:
            if self._selected_uuid is not None:
                self.decisionRequested.emit(self._DECISION_KEYS[k])
            return
        nav = {Qt.Key_BracketRight: self.next_unreviewed,
               Qt.Key_BracketLeft: self.prev_unreviewed,
               Qt.Key_BraceRight: self.next_event,
               Qt.Key_BraceLeft: self.prev_event}
        if k in nav:
            nav[k](); return
        super().keyPressEvent(ev)

    # ---- strip shift-drag range ---------------------------------------
    def _on_shift_drag(self, t0, t1, finished):
        self._strip_range.setRegion([t0, t1])
        self._strip_range.show()
        n = self._epochs.count_in(t0, t1)
        self.mark_n_btn.setEnabled(n > 0 and self._channel is not None)
        self.mark_n_btn.setText(
            f"Mark {n} epoch{'' if n == 1 else 's'} as artefact")

    def _clear_strip_range(self):
        self._strip_range.setRegion([0, 0])
        self._strip_range.hide()
        self.mark_n_btn.setEnabled(False)
        self.mark_n_btn.setText("Mark 0 epochs as artefact")

    def _mark_strip_range(self):
        if self._channel is None:
            return
        t0, t1 = self._strip_range.getRegion()
        s, e = float(min(t0, t1)), float(max(t0, t1))
        if self._epochs.count_in(s, e) < 1:
            return
        self.markArtefactRequested.emit(self._channel, s, e)
        self._clear_strip_range()

    def _on_overview_click(self, ev):
        try:
            if ev.button() != Qt.LeftButton:
                return
            vb = self.plot.getPlotItem().vb
            p = vb.mapSceneToView(ev.scenePos())
            i = self.index_at(p.x())
        except Exception:
            return
        self._goto_epoch(i)

    def _on_range_jump(self, item):
        try:
            t0 = float(item.data(Qt.UserRole + 1))
        except Exception:
            return
        self._goto_epoch(self.index_at(t0))

    def _region_fill(self, on):
        self.region.setBrush(pg.mkBrush(88, 166, 255, 40) if on
                             else pg.mkBrush(0, 0, 0, 0))

    def _on_region_changed(self):
        # programmatic setRegion (epoch change / Clear) stays outline-only;
        # a genuine user drag restores the fill.
        if not self._region_programmatic:
            self._region_fill(True)

    def _reset_region(self):
        t0, t1 = self.span(self._epoch)
        self._region_programmatic = True
        self.region.setRegion(self._default_region(t0, t1))
        self._region_programmatic = False
        self._region_fill(False)

    # ---- actions ------------------------------------------------------
    def _sel(self):
        a, b = self.region.getRegion()
        return float(min(a, b)), float(max(a, b))

    def _mark(self):
        if self._channel is None:
            return
        s, e = self._sel()
        if e - s < 0.5:
            QtWidgets.QMessageBox.information(
                self, "Mark artefact",
                "Brush a wider range on the trace first.")
            return
        self.markArtefactRequested.emit(self._channel, s, e)

    def _unmark(self):
        it = self.ranges_list.currentItem()
        if it is not None:
            self.unmarkArtefactRequested.emit(int(it.data(Qt.UserRole)))

    def _drop(self):
        if self._channel:
            self.dropChannelRequested.emit(self._channel)

    def _inspect_global(self):
        if self._channel is None:
            return
        s, e = self._sel()
        corro = 0
        if self._all_events is not None and len(self._all_events):
            win = self._all_events[
                (self._all_events['start_time'] < e) &
                (self._all_events['start_time'] > s)]
            corro = win['channel'].nunique()
        msg = (f"Window {s:.1f}–{e:.1f}s.\n\n"
               f"{corro} channels have events in this window.\n\n"
               "Marking this as a GLOBAL artefact removes this time from ALL "
               "channels at re-detection. Only confirm if the contamination "
               "is genuinely whole-head (movement/electrical), NOT a "
               "single-channel problem (use 'Drop channel' for that).\n\n"
               "Confirm global artefact?")
        if QtWidgets.QMessageBox.question(
                self, "Confirm GLOBAL artefact", msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
            self.globalArtefactConfirmed.emit(s, e, self._channel)


# ============================================================================
# Shared theme / colour constants + global filter dock
# ============================================================================

EVT_COLOR = {
    'slow_wave': '#5fd3a4', 'spindle': '#f0b056',
    'k_complex': '#d680e0', 'pac': '#a78bfa',
}
STAGE_COLOR = {
    'Wake': '#5d6776', 'W': '#5d6776',
    'N1': '#58a6ff', 'NREM1': '#58a6ff', 'Stage1': '#58a6ff',
    'N2': '#5fd3a4', 'NREM2': '#5fd3a4', 'Stage2': '#5fd3a4',
    'N3': '#3fb950', 'NREM3': '#3fb950', 'Stage3': '#3fb950',
    'REM': '#d680e0',
}

# Best-effort dark theme (structure+behaviour fidelity; not pixel-exact).
# ---------------------------------------------------------------------------
# THEME — single source of truth for chrome + plot colors. Chrome is neutral
# mid-grey (PyQt5 Fusion-ish); plot interiors are pure black so EEG traces and
# red outlier overlays read cleanly. Data colors stay unchanged.
# ---------------------------------------------------------------------------
THEME = {
    # chrome
    'bg_window':     '#3a3a3a',
    'bg_titlebar':   '#2d2d2d',
    'bg_menubar':    '#353535',
    'bg_toolbar':    '#3a3a3a',
    'bg_panel':      '#424242',   # left + right docks
    'bg_sub':        '#4a4a4a',   # sub-panels, chips
    'bg_row':        '#404040',
    'bg_row_alt':    '#454545',
    'bg_row_hover':  '#525252',
    'bg_row_sel':    '#2f4d77',
    'border':        '#1f1f1f',
    'border_strong': '#5a5a5a',
    'text':          '#e5e5e5',
    'text_2':        '#b8b8b8',
    'text_3':        '#888888',
    'accent':        '#5a8fce',
    'accent_soft':   '#2c4666',
    # data
    'ok':            '#69b35d',
    'warn':          '#e0a334',
    'bad':           '#e0533f',
    'dead':          '#888888',
    # plot interiors — kept black regardless of chrome theme
    'plot_bg':       '#0a0a0a',
    'plot_axis':     '#888888',
    'plot_grid':     '#1f1f1f',
}


def _theme_plot(pw):
    """Apply plot-interior theme (pure black bg + grey axes) to a PlotWidget.
    PyQtGraph ignores QSS for its plot scene — this must be called per-widget,
    AFTER the PlotWidget is constructed."""
    pw.setBackground(THEME['plot_bg'])
    for axis in ('left', 'bottom'):
        ax = pw.getPlotItem().getAxis(axis)
        ax.setPen(THEME['plot_axis'])
        ax.setTextPen(THEME['plot_axis'])


# QApplication stylesheet — applied at app boot (see app.setStyleSheet near
# main()). Symbol kept as DARK_QSS so the apply site doesn't move.
DARK_QSS = f"""
QWidget {{
    background: {THEME['bg_window']};
    color: {THEME['text']};
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
}}
QMainWindow {{ background: {THEME['bg_window']}; }}
QMenuBar {{
    background: {THEME['bg_menubar']};
    border-bottom: 1px solid {THEME['border']};
}}
QMenuBar::item:selected {{ background: #4a4a4a; }}
QStatusBar {{
    background: {THEME['bg_titlebar']};
    border-top: 1px solid {THEME['border']};
    color: {THEME['text_2']};
}}
QDockWidget {{
    background: {THEME['bg_panel']};
    color: {THEME['text']};
}}
QDockWidget::title {{
    background: {THEME['bg_titlebar']};
    padding: 6px 10px;
    font-size: 10.5px;
    text-transform: uppercase;
    color: {THEME['text_3']};
    border-bottom: 1px solid {THEME['border']};
}}
QTabWidget::pane {{
    background: {THEME['bg_window']};
    border: 1px solid {THEME['border']};
}}
QTabBar::tab {{
    background: {THEME['bg_titlebar']};
    color: {THEME['text_2']};
    padding: 6px 14px;
    border: 1px solid transparent;
}}
QTabBar::tab:selected {{
    background: {THEME['bg_window']};
    color: {THEME['text']};
    border-color: {THEME['border']};
}}
QTableView, QListWidget {{
    background: {THEME['bg_window']};
    alternate-background-color: {THEME['bg_row_alt']};
    color: {THEME['text']};
    gridline-color: {THEME['border']};
    selection-background-color: {THEME['bg_row_sel']};
    selection-color: {THEME['text']};
}}
QHeaderView::section {{
    background: {THEME['bg_titlebar']};
    color: {THEME['text_2']};
    padding: 6px 8px;
    border: 0;
    border-right: 1px solid {THEME['border']};
    border-bottom: 1px solid {THEME['border']};
    font-weight: 500;
}}
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #545454, stop:1 #3e3e3e);
    color: {THEME['text']};
    border: 1px solid {THEME['border_strong']};
    border-radius: 3px;
    padding: 4px 10px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #5e5e5e, stop:1 #484848);
}}
QPushButton:disabled {{ color: {THEME['text_3']}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #3d6fa8, stop:1 #2a5489);
    color: white;
    border-color: #4477b3;
}}
QComboBox, QLineEdit, QSpinBox {{
    background: #2a2a2a;
    color: {THEME['text']};
    border: 1px solid {THEME['border_strong']};
    border-radius: 2px;
    padding: 2px 8px;
    min-height: 18px;
}}
QComboBox QAbstractItemView {{
    background: {THEME['bg_sub']};
    selection-background-color: {THEME['bg_row_sel']};
    border: 1px solid {THEME['border']};
}}
QCheckBox {{ color: {THEME['text']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    background: #2a2a2a;
    border: 1px solid {THEME['border_strong']};
    border-radius: 2px;
}}
QCheckBox::indicator:checked {{
    background: {THEME['accent']};
    border-color: {THEME['accent']};
}}
QScrollBar:vertical {{
    background: {THEME['bg_panel']};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: #5a5a5a;
    min-height: 24px;
    border-radius: 3px;
}}
QGroupBox {{
    background: {THEME['bg_panel']};
    border: 1px solid {THEME['border']};
    margin-top: 8px;
    padding-top: 8px;
}}
QGroupBox::title {{
    color: {THEME['text_3']};
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
"""


def _swatch(color_hex, d=11):
    """Small colour chip QLabel for the event-type rows."""
    lbl = QLabel()
    lbl.setFixedSize(d, d)
    lbl.setStyleSheet(
        f"background:{color_hex};border-radius:2px;")
    return lbl


class FilterDock(QDockWidget):
    """Global left dock — event-type / method / frequency / channel filters
    that apply across both tabs.

    Owns the canonical filter widgets. The main window aliases the child
    widgets onto itself (``self.spindle_check`` etc.) so the QC/populate
    methods keep working verbatim.
    """

    def __init__(self, parent=None):
        super().__init__("Filters", parent)
        self.setObjectName("FilterDock")
        self.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        def _h(text):
            q = QLabel(text)
            q.setStyleSheet("color:#6b7585;font-size:10px;"
                            "font-weight:600;letter-spacing:0.05em;")
            return q

        # --- event type -------------------------------------------------
        lay.addWidget(_h("EVENT TYPE"))
        self.evt_checks = {}
        self._count_labels = {}
        for key, label in (('slow_wave', 'Slow wave'), ('spindle', 'Spindle'),
                           ('k_complex', 'K-complex'), ('pac', 'PAC')):
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setChecked(key != 'pac')
            row.addWidget(_swatch(EVT_COLOR[key]))
            row.addWidget(cb, 1)
            cnt = QLabel("—")
            cnt.setStyleSheet("color:#6b7585;font-family:monospace;")
            row.addWidget(cnt)
            lay.addLayout(row)
            self.evt_checks[key] = cb
            self._count_labels[key] = cnt
        # back-compat aliases used by existing apply_filters/populate code
        self.spindle_check = self.evt_checks['spindle']
        self.slowwave_check = self.evt_checks['slow_wave']
        self.kcomplex_check = self.evt_checks['k_complex']
        self.pac_check = self.evt_checks['pac']

        # --- method -----------------------------------------------------
        lay.addWidget(_h("METHOD"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("All Methods")
        lay.addWidget(self.method_combo)

        # --- frequency band --------------------------------------------
        lay.addWidget(_h("FREQUENCY BAND"))
        self.freq_band_combo = QComboBox()
        self.freq_band_combo.addItem("All Frequencies")
        lay.addWidget(self.freq_band_combo)

        # --- channels ---------------------------------------------------
        lay.addWidget(_h("CHANNELS"))
        self.channel_search = QLineEdit()
        self.channel_search.setPlaceholderText("search…  e.g. E33 or Cz")
        self.channel_search.textChanged.connect(self._filter_channel_list)
        lay.addWidget(self.channel_search)
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMaximumHeight(220)
        lay.addWidget(self.channel_list)
        cbtns = QHBoxLayout()
        self.sel_all_btn = QPushButton("All")
        self.sel_none_btn = QPushButton("None")
        cbtns.addWidget(self.sel_all_btn)
        cbtns.addWidget(self.sel_none_btn)
        lay.addLayout(cbtns)

        note = QLabel("Filters apply globally to both tabs.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7585;font-size:10px;")
        lay.addWidget(note)
        lay.addStretch()
        self.setWidget(body)

    # ---- helpers ------------------------------------------------------
    def _filter_channel_list(self, text):
        t = (text or "").lower()
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def set_event_counts(self, counts):
        """counts: {event_type: int}. Greys 0 / missing."""
        for k, lbl in self._count_labels.items():
            v = counts.get(k)
            lbl.setText("—" if not v else f"{int(v):,}")

    def populate_channels(self, channels, checked=None):
        """Fill the channel list; names in ``checked`` start ticked."""
        checked = set(checked or [])
        self.channel_list.clear()
        for ch in channels:
            it = QtWidgets.QListWidgetItem(str(ch))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if str(ch) in checked else Qt.Unchecked)
            it.setData(Qt.UserRole, str(ch))
            self.channel_list.addItem(it)

    def decorate_channels(self, artefact_set=None, redetect_set=None,
                          interp_set=None):
        """Append ⚑ (channel-artefact verdict) / ↻ (re-detect queued) /
        `` ~`` (interpolated by the cleaning pipeline, with a tooltip).

        ``interp_set`` is remembered, so later calls that pass only the two
        review sets keep the interpolated marks.
        """
        artefact_set = artefact_set or set()
        redetect_set = redetect_set or set()
        if interp_set is not None:
            self._interp_set = set(str(c) for c in interp_set)
        interp = getattr(self, '_interp_set', set())
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            base = str(it.data(Qt.UserRole))
            tag = ""
            if base in interp:
                tag += INTERPOLATED_MARK
            if base in artefact_set:
                tag += " ⚑"
            if base in redetect_set:
                tag += " ↻"
            it.setText(base + tag)
            it.setToolTip(INTERPOLATED_TOOLTIP if base in interp else "")


# Wake-like labels are never a detection stage, so they are dropped from the
# event-derived scope before it becomes the density time base.
QC_WAKE_STAGES = {'Wake', 'W', 'Undefined', 'Unknown', 'None', ''}


def qc_density_stage_scope(event_stages, wake_stages=None):
    """Stage labels behind a set of ``events.stage`` values, Wake dropped.

    ``events.stage`` holds the *run's* stage scope, which since the joint-token
    change is a single joined label such as ``'NREM2NREM3'`` rather than one
    row per epoch stage. That token matches no scored epoch, so feeding it
    straight to :func:`turtlewave_hdEEG.utils.build_density_denominators` finds
    zero analysed seconds and the QC dashboard's density silently blanks or
    goes wrong. Splitting it back into components restores real stage labels.

    Module-level (not a method) so the derivation can be tested without a Qt
    application, and lazily importing the library so this module stays
    importable headless.

    Parameters
    ----------
    event_stages : iterable of str or None
        Distinct ``events.stage`` values of the loaded events. Each may be a
        joint token (``'NREM2NREM3'``), a single stage (``'NREM2'``), or a
        legacy short label the library's vocabulary does not know (``'N2'``),
        which is passed through unsplit rather than discarded.
    wake_stages : set of str or None
        Labels to drop. Defaults to :data:`QC_WAKE_STAGES`.

    Returns
    -------
    list of str or None
        Sorted, de-duplicated non-Wake stage labels, or ``None`` when
        ``event_stages`` is ``None`` or nothing survives - both of which mean
        "no event-derived scope", the caller's signal to fall back.
    """
    if event_stages is None:
        return None
    drop = QC_WAKE_STAGES if wake_stages is None else wake_stages
    try:
        from turtlewave_hdEEG.dbwrite import split_stage_token
    except Exception:
        split_stage_token = None
    scope = set()
    for value in event_stages:
        text = str(value)
        parts = None
        if split_stage_token is not None:
            try:
                parts = split_stage_token(text)
            except Exception:
                # Unknown vocabulary (legacy 'N2', 'Stage2', a site-specific
                # label): keep the value whole. Dropping it would shrink the
                # denominator; guessing at it would corrupt it.
                parts = None
        if not parts:
            parts = [text]
        for part in parts:
            label = str(part)
            if label not in drop:
                scope.add(label)
    return sorted(scope) or None


#: Confirmation text of Review > Show other reviewers. Nothing records that
#: a decision was made with others visible (event_reviews has no ``blind``
#: column yet), so the text says so rather than promising it.
SHOW_OTHERS_WARNING = (
    "Decisions you make while other reviewers' choices are visible are not "
    "independent and should not be used for inter-rater agreement. This is "
    "not recorded automatically.")


class _ReportSource:
    """What :class:`PrecisionReportDialog` reads, bound to the window's
    database and loaded sample."""

    def __init__(self, win):
        self.win = win
        s = win._sample
        self.design = s['design']
        self.n_total = len(s['rows'])
        self.db_path = win.db.db_path
        self.db_name = os.path.basename(win.db.db_path)
        self.current_reviewer = win.reviewer_name
        ids = []
        try:
            ids = json.loads(self.design.get('run_ids') or '[]')
        except (TypeError, ValueError):
            pass
        run = win._run_info(ids[0]) if ids else {}
        self.run_text = _er.run_label(run, ids[0] if ids else None) + (
            f" (+{len(ids) - 1} run{'s' if len(ids) > 2 else ''})"
            if len(ids) > 1 else '')

    def labels(self):
        return _sr.reviewer_labels(self.win.db.conn, self.design['sample_id'])

    def frame(self, reviewer):
        return _sr.precision_frame(self.win.db.conn,
                                   self.design['sample_id'], reviewer)

    def event_line(self, uuid):
        r = self.win._sample['rows'].get(uuid, {})
        return f"{r.get('channel')} {_er.fmt_hms1(r.get('start_time'))}"

    def status(self, text):
        self.win.status_bar.showMessage(text)


class _FiguresNotComputable(Exception):
    """The figures cannot be computed faithfully; the message says why."""


def _rereference_like_wonambi(data, labels, ts, sfreq, target, ref):
    """``target`` re-referenced to ``ref`` through Wonambi's own
    ``montage(ref_chan=...)`` (the call the detectors' ``read_data(chan,
    ref_chan)`` makes), then ``nan_to_num`` as Wonambi's ``_create_data``
    does. Without Wonambi, :func:`event_review.rereference` (same formula)."""
    if not ref:
        return np.nan_to_num(np.asarray(data, dtype=float)[
            list(labels).index(target)])
    try:
        from wonambi.datatype import ChanTime
        from wonambi.trans import montage
    except ImportError:
        return _er.rereference(data, labels, target, ref)
    ct = ChanTime()
    ct.s_freq = float(sfreq)
    ct.axis['chan'] = np.empty(1, dtype='O')
    ct.axis['chan'][0] = np.asarray(list(labels), dtype='U')
    ct.axis['time'] = np.empty(1, dtype='O')
    ct.axis['time'][0] = np.asarray(ts, dtype=float)
    ct.data = np.empty(1, dtype='O')
    ct.data[0] = np.asarray(data, dtype=float)
    m = montage(ct, ref_chan=[str(r) for r in ref])
    return np.nan_to_num(np.asarray(m(chan=target, trial=0), dtype=float))


class _PopulationWorker(QtCore.QThread):
    """Reads the population checks off the GUI thread, on its own
    read-only connection (``event_review.load_population``)."""

    done = pyqtSignal(object, object)   # key, result dict

    def __init__(self, key, db_path, event_type, methods, freq_band,
                 parent=None):
        super().__init__(parent)
        self.key = key
        self._args = (db_path, event_type, methods, freq_band)

    def run(self):
        self.done.emit(self.key, _er.load_population(*self._args))


class EventReviewGUI(QMainWindow):
    """Main event review GUI with 3-panel design"""

    @property
    def db(self):
        """The open :class:`EventDatabase` (or ``None``)."""
        return self.__dict__.get('_db')

    @db.setter
    def db(self, value):
        # Every per-database cache goes with the database: a reopened file
        # gets a fresh connection whose PRAGMA data_version restarts at 1, so
        # a stale population or figure entry could otherwise match again.
        self.__dict__['_db'] = value
        for name in ('_pop_cache', '_fig_cache', '_run_cache'):
            self.__dict__[name] = {}
        self.__dict__['_pop_view'] = None
        self.__dict__['_undo_stack'] = []
    
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1800, 1000)

        # Data
        self.db = None
        self.eeg_data = None
        self.annotations = None
        self.reviewer_name = ""   # provenance field; intentionally unset
        self.selected_event_uuid = None   # event picked on the Epochs trace
        # decision state (spec section 5) and session-only review settings
        self._armed = None                # 'reject' | 'unsure' while armed
        self._pending_other = False       # 'other' chosen, comment awaited
        self._last_reject_reason = None   # this session's last reject reason
        self._undo_stack = []
        self._show_others = False         # never persisted
        self._show_others_confirmed = False
        self._auto_advance = _setting_bool('review/auto_advance', True)
        self._pop_worker = None
        self._pop_view = None
        # review sample (spec section 2)
        self._sample = None               # loaded sample of the scope in view
        self._sample_active = False
        self._revisit = None              # unsure uuids while revisiting
        self._sample_cursor = None        # last sample event visited
        self._sample_times = []           # this session's decision times
        self._report = None
        self.recording_start_time = None

        # Chrome / QC state
        self.subject = "—"
        self.annot_file_path = None
        self.eeg_file_path = None
        self._redetect_queue = set()
        self._qc_thresholds = dict(hard_z=3.5, soft_z=2.0, dead_frac=0.15)
        self._setWindowTitleFromSubject()
        
        # Waveform caching
        self.waveform_cache = {}
        self.cache_lock = QtCore.QMutex()
        self.background_loader = None
        self.is_closing = False
        
        # UI state
        # Waveform channels. Filled from default_review_channels() when an EEG
        # file (or, failing that, a database) loads, but only while the user
        # has not changed the selection: their own choice is never replaced.
        self.selected_channels = []
        self._channels_user_set = False
        self.selected_event_types = ['spindle', 'slow_wave', 'k_complex']
        
        # Debounce timer for channel selection
        self.channel_filter_timer = QtCore.QTimer()
        self.channel_filter_timer.setSingleShot(True)
        self.channel_filter_timer.timeout.connect(self.apply_channel_filter)
        
        # Setup UI
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_ui()
        self.setup_status_bar()
        self.setup_keyboard_shortcuts()
        # Empty channel list plus a hint until a database or EEG file loads.
        self.load_channels()

    # ------------------------------------------------------------------
    # Chrome helpers (title / toolbar pills / subject)
    # ------------------------------------------------------------------
    def _setWindowTitleFromSubject(self):
        self.setWindowTitle(
            f"TurtleWave hdEEG · Event Review · {self.subject}")

    def _derive_subject(self):
        """Best-effort subject id from the loaded artefacts."""
        for p in (self.annot_file_path, getattr(self, 'eeg_file_path', None),
                  getattr(self.db, 'db_path', None) if self.db else None):
            if p:
                stem = os.path.splitext(os.path.basename(p))[0]
                for suf in ('_annotations', '_eeg', '_events',
                            'neural_events'):
                    stem = stem.replace(suf, '')
                stem = stem.strip('_- ')
                if stem:
                    return stem
        return "—"

    def _set_led(self, pill, ok):
        color = '#3fb950' if ok else '#6b7585'
        pill.setStyleSheet(
            f"QLabel{{padding:2px 8px;border:1px solid #262d39;"
            f"border-radius:3px;background:#131821;color:#d6dee8;}}")
        pill.setText(pill.property('label') +
                     ('  ●' if ok else '  ○'))
        pill._dot = color

    def _make_pill(self, label):
        q = QLabel()
        q.setProperty('label', label)
        q.setTextFormat(Qt.PlainText)
        self._set_led(q, False)
        return q

    def _fmt_hms(self, seconds):
        try:
            s = int(seconds)
        except Exception:
            return "—"
        return f"{s // 3600}h {s % 3600 // 60}m"

    def setup_menu_bar(self):
        """Menubar: File / Edit / View / Analysis / Export / Help."""
        mb = self.menuBar()

        m_file = mb.addMenu('&File')
        for text, slot in (
                ('Open Database…', self.open_database),
                ('Open EEG File…', self.open_eeg_file),
                ('Open Annotation File…', self.open_annotation_file)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            m_file.addAction(a)
        m_file.addSeparator()
        a = QAction('Exit', self)
        a.triggered.connect(self.close)
        m_file.addAction(a)

        m_edit = mb.addMenu('&Edit')
        a = QAction('Flag selected channel for re-detect (F)', self)
        a.triggered.connect(self._flag_selected_qc_row)
        m_edit.addAction(a)

        m_review = mb.addMenu('&Review')
        a = QAction('Reviewer name…', self)
        a.triggered.connect(self._prompt_reviewer_name)
        m_review.addAction(a)
        self.act_reviewer_name = a
        self.m_review = m_review
        for text, slot, attr in (
                ('Draw review sample…', '_open_draw_dialog', 'act_draw_sample'),
                ('Resume review sample', '_start_sample', 'act_resume_sample'),
                ('Exit review sample', '_exit_sample', 'act_exit_sample')):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, n=slot: getattr(self, n)())
            m_review.addAction(a)
            setattr(self, attr, a)
        m_review.aboutToShow.connect(self._update_review_menu)
        m_review.addSeparator()
        a = QAction('Show other reviewers', self, checkable=True)
        a.setChecked(False)   # off at every launch, never persisted
        a.toggled.connect(self._on_show_others)
        m_review.addAction(a)
        self.act_show_others = a
        m_review.addSeparator()
        a = QAction('Precision report…', self)
        a.triggered.connect(lambda: self._open_report())
        m_review.addAction(a)
        self.act_precision_report = a

        m_view = mb.addMenu('&View')
        a = QAction('Outlier threshold…', self)
        a.triggered.connect(self.open_outlier_threshold_dialog)
        m_view.addAction(a)
        a = QAction('Filters dock', self, checkable=True, checked=True)
        a.triggered.connect(
            lambda v: self.filter_dock.setVisible(v))
        m_view.addAction(a)
        a = QAction('Topography & detail dock', self,
                    checkable=True, checked=True)
        a.triggered.connect(
            lambda v: self.detail_dock.setVisible(v))
        m_view.addAction(a)

        m_an = mb.addMenu('&Analysis')
        a = QAction('Refresh QC dashboard', self)
        a.triggered.connect(self.refresh_qc_dashboard)
        m_an.addAction(a)
        a = QAction('Build re-detect request…', self)
        a.triggered.connect(self.open_redetect_modal)
        m_an.addAction(a)

        m_exp = mb.addMenu('E&xport')
        for text, slot in (
                ('Export QC report…', self.export_qc_summary),
                (None, None),
                ('Export Re-run Package…', self.export_rerun_package),
                ('Export Figure…', self.export_figure)):
            if text is None:
                m_exp.addSeparator()
                continue
            a = QAction(text, self)
            a.triggered.connect(slot)
            m_exp.addAction(a)

        m_help = mb.addMenu('&Help')
        a = QAction('Design notes', self)
        a.triggered.connect(self.open_design_notes)
        m_help.addAction(a)
        a = QAction('About', self)
        a.triggered.connect(lambda: QtWidgets.QMessageBox.about(
            self, "About",
            "TurtleWave hdEEG · Event Review\nQC-by-outlier-triage GUI"))
        m_help.addAction(a)

    def setup_toolbar(self):
        """Toolbar: connection LEDs · recording duration/TST · detector ·
        keyboard-shortcut legend."""
        tb = QToolBar()
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.led_db = self._make_pill('DB')
        self.led_xml = self._make_pill('XML')
        self.led_eeg = self._make_pill('EEG')
        for w in (self.led_db, self.led_xml, self.led_eeg):
            tb.addWidget(w)
        tb.addSeparator()

        self.lbl_duration = QLabel("rec —  ·  TST —")
        self.lbl_duration.setStyleSheet("color:#9ba6b5;")
        tb.addWidget(self.lbl_duration)
        tb.addSeparator()
        self.lbl_detector = QLabel("detector: —")
        self.lbl_detector.setStyleSheet("color:#9ba6b5;")
        tb.addWidget(self.lbl_detector)

        spacer = QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)

        legend = QLabel("F flag channel for re-detect · "
                        "click a worst-events row to drill")
        legend.setStyleSheet("color:#6b7585;font-size:11px;")
        tb.addWidget(legend)

    def _refresh_toolbar_state(self):
        self._set_led(self.led_db, self.db is not None)
        self._set_led(self.led_xml, self.annotations is not None)
        n_eeg = 0
        try:
            if self.eeg_data is not None and hasattr(self.eeg_data, 'channels'):
                n_eeg = len(self.eeg_data.channels)
            elif self.eeg_data is not None and hasattr(self.eeg_data,
                                                       'ch_names'):
                n_eeg = len(self.eeg_data.ch_names)
        except Exception:
            n_eeg = 0
        self._set_led(self.led_eeg, self.eeg_data is not None)
        if n_eeg:
            self.led_eeg.setText(f"EEG  {n_eeg} ch")
        tst = self._scored_minutes()
        try:
            rec = annotation_recording_seconds(self.annotations)
        except Exception:
            rec = None
        self.lbl_duration.setText(
            f"rec {self._fmt_hms(rec) if rec else '—'}  ·  "
            f"TST {self._fmt_hms(tst * 60) if tst else '—'}")
    
    def setup_ui(self):
        """Setup UI: two tabs — the per-channel QC dashboard (landing) and the
        per-channel Epochs drill. The right dock carries live topography, the
        global worst-events list, and the selected-channel detail."""
        # --- QC reframe: two tabs (Channels QC + Epochs) ---
        self.tabs = QtWidgets.QTabWidget()
        self.qc_widget = ChannelQCWidget()
        self.epochs_panel = EpochsPanel()
        self.tabs.addTab(self.qc_widget, "1 · Channels (QC)")
        self.tabs.addTab(self.epochs_panel, "2 · Epochs")
        self.setCentralWidget(self.tabs)

        # --- global LEFT dock: filters across both tabs ---
        self.filter_dock = FilterDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filter_dock)
        fd = self.filter_dock
        # alias child widgets so existing filter/apply methods work verbatim
        self.spindle_check = fd.spindle_check
        self.slowwave_check = fd.slowwave_check
        self.kcomplex_check = fd.kcomplex_check
        self.pac_check = fd.pac_check
        self.method_combo = fd.method_combo
        self.freq_band_combo = fd.freq_band_combo
        self.channel_list = fd.channel_list
        for cb in (fd.spindle_check, fd.slowwave_check, fd.kcomplex_check):
            cb.stateChanged.connect(self.update_event_type_filter)
        fd.pac_check.stateChanged.connect(
            lambda *_: self.refresh_qc_dashboard())
        fd.method_combo.currentIndexChanged.connect(self.update_method_filter)
        fd.freq_band_combo.currentIndexChanged.connect(
            self.update_freq_band_filter)
        fd.channel_list.itemChanged.connect(self.on_channel_changed)
        fd.sel_all_btn.clicked.connect(self.select_all_channels)
        fd.sel_none_btn.clicked.connect(self.deselect_all_channels)

        # --- global RIGHT dock: topography & selected-channel detail ---
        self.detail_dock = QDockWidget("Topography & detail", self)
        self.detail_dock.setObjectName("DetailDock")
        self.detail_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                     QDockWidget.DockWidgetFloatable)
        self.detail_dock_w = ChannelDetailDock()
        self.detail_dock_w.loadMontageRequested.connect(self.on_load_montage)
        # Per-channel worst-epochs row click -> switch to Epochs tab + jump
        self.detail_dock_w.gotoEpochRequested.connect(
            self._on_worst_epoch_goto)
        # Global worst-events row click -> switch channel AND epoch
        self.detail_dock_w.gotoChannelEpochRequested.connect(
            self._on_global_worst_goto)
        # Topo electrode click -> select that channel (no drill / no tab switch)
        self.detail_dock_w.channelPicked.connect(self._on_topo_channel_picked)
        # × in the dock's compact marked-artefact list -> unmark
        self.detail_dock_w.unmarkArtefactRequested.connect(
            self._unmark_artefact)
        self.detail_dock_w.checkLinkActivated.connect(self._on_check_link)
        dock_scroll = QtWidgets.QScrollArea()
        dock_scroll.setWidgetResizable(True)
        dock_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        dock_scroll.setWidget(self.detail_dock_w)
        self.detail_dock.setWidget(dock_scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.detail_dock)
        evp = self.detail_dock_w.event_panel
        evp.decisionClicked.connect(self._key_decide)
        evp.reasonChosen.connect(self._on_reason_picked)
        evp.commentSubmitted.connect(self._on_comment_submitted)
        evp.commentEscape.connect(self._on_comment_escape)
        evp.clearClicked.connect(self._clear_decision)
        evp.prevClicked.connect(lambda: self._nav_unreviewed(-1))
        evp.nextClicked.connect(lambda: self._nav_unreviewed(+1))
        evp.auto_chk.setChecked(self._auto_advance)
        evp.autoAdvanceToggled.connect(self._on_auto_advance)

        self.qc_widget.channelSelected.connect(self.on_qc_channel_selected)
        self.qc_widget.requestDrill.connect(self.on_qc_drill)
        self.qc_widget.verdictChanged.connect(self.on_qc_verdict_changed)
        self.qc_widget.addToRedetect.connect(self.on_qc_add_redetect)
        self.qc_widget.unmarkArtefact.connect(self._qc_unmark_artefact)
        self.qc_widget.removeFromRedetect.connect(self._qc_remove_redetect)
        self.qc_widget.requestQueueAllHard.connect(self.on_qc_queue_all_hard)
        self.qc_widget.requestBuildRedetect.connect(self.open_redetect_modal)
        self.qc_widget.loadMontageRequested.connect(self.on_load_montage)
        self.qc_widget.evt_combo.currentTextChanged.connect(
            lambda *_: self._refresh_all())
        self.epochs_panel.dropChannelRequested.connect(self._drop_channel)
        self.epochs_panel.globalArtefactConfirmed.connect(self._add_global_artefact)
        self.epochs_panel.markArtefactRequested.connect(
            self._mark_channel_artefact)
        self.epochs_panel.unmarkArtefactRequested.connect(
            self._unmark_artefact)
        self.epochs_panel.requestChannel.connect(self.on_qc_drill)
        self.epochs_panel.eventSelected.connect(self._on_event_selected)
        self.epochs_panel.decisionRequested.connect(
            self._on_decision_requested)
        self.sample_bar = SampleBar()
        self.epochs_panel.layout().insertWidget(0, self.sample_bar)
        self.sample_bar.drawClicked.connect(self._open_draw_dialog)
        self.sample_bar.resumeClicked.connect(self._start_sample)
        self.sample_bar.exitClicked.connect(self._exit_sample)
        self.sample_bar.reportClicked.connect(self._open_report)
        self.sample_bar.showOthersToggled.connect(self._on_bar_show_others)
        evp.openReportClicked.connect(self._open_report)
        evp.revisitClicked.connect(self._revisit_unsure)
        self.epochs_panel.selectionCleared.connect(self._on_selection_cleared)
        self.epochs_panel.checkFilterCleared.connect(
            self._on_check_filter_cleared)
        self.epochs_panel.read_window = self._read_eeg_window
        self.epochs_panel.read_channels = self._read_eeg_channels
        self.epochs_panel.neighbour_provider = self._neighbour_provider
        self.epochs_panel.neighbour_events = self._neighbour_events
        self._review_qc_sidecar = None

        # Always land on the Channels (QC) dashboard — it is the triage entry
        # point; the last-used tab is deliberately NOT restored.
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # QC dashboard wiring
    # ------------------------------------------------------------------

    _SCORED_STAGES = {'NREM1', 'NREM2', 'NREM3', 'REM',
                      'N1', 'N2', 'N3', 'Stage1', 'Stage2', 'Stage3'}

    # Fallback stage set for the QC density denominator (N2 + N3 + REM per the
    # redesign spec) used only when the run's scope can't be inferred from the
    # loaded events; Wake and N1 are excluded. Label variants are matched by
    # exact stage label in the annotation, so both Wonambi ('NREM2') and short
    # ('N2') forms are listed.
    _QC_DENSITY_STAGES = {'NREM2', 'NREM3', 'REM',
                          'N2', 'N3', 'Stage2', 'Stage3'}

    # Wake-like labels are never a detection stage, so they are dropped from the
    # event-derived scope before it becomes the density time base. Aliased to
    # the module-level set so the Qt-free helper and the widget cannot drift.
    _WAKE_STAGES = QC_WAKE_STAGES

    def _scored_minutes(self):
        """Total minutes in scored sleep stages (total sleep time, TST): the
        sum of the scored epochs' own durations, so 1 s epochs on a cut
        recording count 1 s. None when annotations are absent. Used for the
        toolbar TST readout — NOT the density denominator (see
        :meth:`_qc_density_minutes`)."""
        table = self._epoch_table()
        if table is None:
            return None
        scored = np.array([s in self._SCORED_STAGES for s in table.stages],
                          dtype=bool)
        secs = float(table.durations[scored].sum()) if scored.any() else 0.0
        return secs / 60.0 if secs else None

    def _qc_density_minutes(self, extra_intervals=None, event_stages=None):
        """Artefact-free analysed minutes over the detection run's stage scope —
        the density denominator for the QC dashboard.

        :func:`turtlewave_hdEEG.density.event_density` is now the single
        definition of density, and its denominator is the ``analysed_time`` row
        the detection run stored — the time the detector actually searched. This
        method cannot simply call it, because the QC dashboard has to answer a
        question the stored row cannot: what the denominator becomes once the
        reviewer marks an epoch as artefact. A stored number is fixed; marking an
        epoch has to shrink it on the next refresh or the dashboard's whole
        interaction is inert.

        So it recomputes, but from the same function the library stored its row
        with (:func:`turtlewave_hdEEG.utils.build_density_denominators`, whose
        ``whole_night_analysed_min`` is the per-stage seconds summed over the
        scope — exactly what ``event_density(combine_stages=True)`` pools), and
        it is anchored to the library in two ways:

        * **The run's own exclusion set is read from the database**
          (``detection_runs.reject_types``) instead of being assumed. Assuming
          is a real bias in both directions: a run that did not exclude arousals
          searched the arousal time, so subtracting it here shrinks the
          denominator and inflates every density on the dashboard, and a run
          that excluded more than assumed is deflated the same way. The library
          never has this problem, because the exclusion set is part of the
          stored row's key — asking with the wrong set returns nothing rather
          than a mismatched number.
        * **With no live marks, the result is checked against the stored row**
          and a disagreement is reported. Agreement is the expected case; a
          difference means the scoring loaded for review is not the scoring
          detection ran on, which silently changes every density in the table.

        The stage scope is the distinct stages of the loaded events, with joint
        run tokens split into their components and Wake dropped (see
        :func:`qc_density_stage_scope`), which is the same scope
        ``event_density`` derives from the rows it counts. When no event stages
        are available it falls back to the fixed N2+N3+REM set present in the
        annotation, so density greys out gracefully rather than erroring.

        Channel-global by design: the numerator is per-channel event counts,
        the denominator is shared.

        Parameters
        ----------
        extra_intervals : list of (float, float) or None
            The reviewer's live artefact spans in seconds (from
            ``qc_artefact_intervals``). ``None`` reproduces the annotation-only
            denominator, which is the case checked against the library.
        event_stages : iterable of str or None
            ``events.stage`` values of the loaded events; their distinct
            non-Wake *components* define the density time base (the run's
            scope). ``None``/empty falls back to the fixed N2+N3+REM set.

        Returns
        -------
        float or None
            Artefact-free minutes over the run's stage scope, or ``None`` when
            annotations are absent, none of the scope stages are scored, or the
            computation fails while the reviewer has marks pending (density
            greyed). That last case deliberately does NOT fall back to the
            stored denominator: it predates the marks, so it would report a
            density that ignores them.
        """
        # Forget the previous refresh's answer before anything can return
        # early. _qc_reject_types_in_use paints the dock caption from these,
        # and a set left over from another event type or filter would be
        # displayed under the confident "from the detection run" wording.
        self._qc_reject_types = None
        self._qc_reject_source = 'assumed (not recorded)'

        if self.annotations is None:
            return None
        # Density time base = the stages the run actually detected on, inferred
        # from the loaded events (joint tokens split back into components, Wake
        # dropped), matching the scope event_density derives from the rows it
        # counts. build_density_denominators keys on scored epoch labels, so it
        # must be given components: a joint 'NREM2NREM3' matches no epoch and
        # would produce an empty denominator without a word said.
        stage_list = qc_density_stage_scope(event_stages,
                                            wake_stages=self._WAKE_STAGES)
        if stage_list is None:
            # Fallback: no event scope available -> fixed N2+N3+REM present.
            try:
                stages_present = {str(s) for s in self.annotations.get_stages()}
            except Exception:
                return None
            stage_list = sorted(s for s in stages_present
                                if s in self._QC_DENSITY_STAGES)
        if not stage_list:
            return None

        reject_types = self._qc_run_rejections()

        # Lazy import keeps GUI imports headless-safe; utils is Qt-free and the
        # single source of truth for the artefact-free subtraction.
        from turtlewave_hdEEG.utils import build_density_denominators
        # Route the helpers' warnings through the repeat filter rather than
        # leaving them on the library's module logger. The denominators are
        # rebuilt on every refresh, so an annotation-inconsistency warning -
        # which is about the file, not about this refresh - would otherwise be
        # restated once per stage every time the dashboard redraws. Passing
        # None would not silence it either (the library refuses to be silenced
        # on a data-integrity condition, correctly); it would only send it
        # somewhere this GUI cannot de-duplicate.
        _density_repeat_filter.context = getattr(self, 'annot_file_path', None)
        try:
            dens = build_density_denominators(
                self.annotations, self.eeg_data,
                # reject_types= ONLY: the two deprecated booleans cannot name
                # a set that includes Move, and passing both is how the two
                # silently disagree.
                reject_types=list(reject_types),
                stage_list=stage_list, stages_present=stage_list,
                logger=_density_logger,
                extra_artefact_intervals=extra_intervals)
        except Exception:
            # Nothing computable locally.
            #
            # With no marks pending, the stored denominator IS the answer - the
            # same number this computation would have produced - so use it
            # rather than greying density out.
            #
            # With marks pending it is not an answer at all. The stored row was
            # written at detection time and knows nothing of the reviewer's
            # marks, so serving it here would show a density that silently
            # ignores their edits: they mark an epoch, the number does not
            # move, and nothing says why. Subtracting the marks from it is not
            # available either - working out how much of a mark falls in-stage
            # and is not already excluded by an annotation artefact is exactly
            # the computation that just failed, and approximating it would put
            # a wrong number in the same place. So density is withheld and the
            # reason is stated.
            if extra_intervals:
                _density_repeat_filter.context = getattr(
                    self, 'annot_file_path', None)
                _density_logger.warning(
                    "Density is unavailable: the artefact-free denominator "
                    "could not be computed from the scoring, and there are "
                    "%d reviewer artefact mark(s) pending. The denominator "
                    "stored by the detection run is not used here because it "
                    "predates those marks and would show a density that "
                    "ignores them. The density column stays blank until the "
                    "scoring loads cleanly.", len(extra_intervals))
                return None
            return self._qc_stored_density_minutes(stage_list, reject_types)

        minutes = dens.whole_night_analysed_min or None
        if not extra_intervals:
            self._qc_check_against_stored(minutes, stage_list, reject_types)
        return minutes

    def _qc_run_rejections(self):
        """The exclusion set of the runs in view, or the detector defaults.

        Also records where the answer came from in
        ``self._qc_reject_source`` (``'from the detection run'``,
        ``'from another run in this file'`` or ``'assumed (not recorded)'``) so
        the dashboard caption can say whether the mask it names was read from
        the run on screen, read from a different run, or assumed.

        Returns
        -------
        list of str
            The excluded event types, canonically ordered. Falls back to
            :data:`PRE_4_4_REJECT_TYPES` only when nothing in the file records
            a set at all, and says so once via the repeat-filtered logger
            rather than on every dashboard refresh.

        Notes
        -----
        The lookup WIDENS before it guesses. A scoped miss is usually a method
        filter narrower than the runs on record, and in that case the file
        still holds the answer, so guessing would discard a recorded fact. The
        order is: the current scope, then this event type, then any run in the
        file. Only when nothing anywhere records a set does the fallback apply,
        and it is then :data:`PRE_4_4_REJECT_TYPES` - the same value
        ``turtlewave_gui._choose_db_scope`` assumes, because a file that
        records no set anywhere is by definition pre-4.4.0. Two different
        guesses for one condition would be indefensible.
        """
        fallback = list(PRE_4_4_REJECT_TYPES)
        if self.db is None:
            self._set_qc_rejections(fallback, 'assumed (not recorded)')
            return list(fallback)
        evt = None
        try:
            evt = self.qc_widget.current_event_type()
        except Exception:
            pass
        methods, _ = self._current_method_freq()

        # Widen, then guess. Each query is one SELECT ... LIMIT 1 on a tiny
        # table, so trying three costs nothing. Duplicates are skipped: with no
        # method filter the first two queries are the same query.
        attempts = []
        for kwargs in ({'event_type': evt, 'methods': methods},
                       {'event_type': evt},
                       {}):
            probe = {k: v for k, v in kwargs.items() if v}
            if probe not in [a[0] for a in attempts]:
                attempts.append((probe, kwargs))

        for index, (_probe, kwargs) in enumerate(attempts):
            try:
                found = self.db.get_run_rejections(**kwargs)
            except Exception:
                found = None
            if found is None:
                continue
            types = order_reject_types(found)
            if index == 0:
                self._set_qc_rejections(types, 'from the detection run')
                return list(types)
            # A reading, not a guess, so NOT the warning colour - but say which
            # run it was read from, because it may not be the run on screen.
            self._set_qc_rejections(types, 'from another run in this file')
            _density_repeat_filter.context = getattr(
                self, 'annot_file_path', None)
            _density_logger.warning(
                "No exclusion set is recorded for the runs matching the "
                "current method filter, so the denominator uses the set "
                "recorded for the most recent run in this file (%s). If the "
                "events on screen came from a run with a different set, the "
                "densities shown are not that run's.",
                reject_types_display(types))
            return list(types)

        # Nothing in the file records a set anywhere, which makes it pre-4.4.0.
        self._set_qc_rejections(fallback, 'assumed (not recorded)')
        _density_repeat_filter.context = getattr(self, 'annot_file_path', None)
        # Direction of the bias, stated the way round it actually goes:
        # excluding MORE time makes the denominator SMALLER, so a run that
        # excluded more than this assumption has a higher density than shown.
        _density_logger.warning(
            "The detection run's exclusion set could not be read from the "
            "database for this scope, so the searched-time denominator assumes "
            "%s, which is what every run stored without a recorded set did. If "
            "the run excluded more than that, the densities shown are lower "
            "than the run's own; if it excluded less, they are higher.",
            reject_types_display(fallback))
        return list(fallback)

    def _set_qc_rejections(self, types, source):
        """Record the resolved set and where it came from, for the dock caption.

        Parameters
        ----------
        types : list of str
            The exclusion set the denominator will use.
        source : str
            ``'from the detection run'``, ``'from another run in this file'`` or
            ``'assumed (not recorded)'``. Only the last renders in the warning
            colour (see :meth:`ChannelDetailDock.set_denominator_mask`).
        """
        self._qc_reject_types = list(types)
        self._qc_reject_source = source

    def _qc_reject_types_in_use(self):
        """The exclusion set the density currently on screen was computed under.

        Returns
        -------
        list of str
            The set :meth:`_qc_run_rejections` last resolved, resolving it now
            if the density path has not run yet (which happens when there are
            no annotations loaded to compute a denominator from).
        """
        types = getattr(self, '_qc_reject_types', None)
        if types is None:
            return self._qc_run_rejections()
        return list(types)

    def _qc_stored_density_minutes(self, stage_list, reject_types):
        """The library's own denominator: analysed_time summed over the scope.

        Parameters
        ----------
        stage_list : list of str
            Stages in scope.
        reject_types : list of str
            The run's exclusion set. It selects which stored rows are read, so
            passing the run's own set is what keeps this denominator on the
            same time base as the events counted against it.

        Returns
        -------
        float or None
            Minutes, or ``None`` when any stage in scope has no stored row -
            deliberately all-or-nothing, because a partial sum is a denominator
            covering less time than the numerator's events span.
        """
        if self.db is None or not getattr(self.db, 'db_path', None):
            return None
        try:
            from turtlewave_hdEEG.dbwrite import read_analysed_time
            stored = read_analysed_time(self.db.db_path,
                                        reject_types=list(reject_types))
        except Exception:
            return None
        if not stored:
            return None
        by_stage = {}
        for (_subject, stage), row in stored.items():
            by_stage.setdefault(str(stage), []).append(row['analysed_seconds'])
        total = 0.0
        for stage in stage_list:
            values = by_stage.get(str(stage))
            if not values:
                return None
            # More than one subject in one database is not the shape this GUI
            # reviews; taking the max would invent time, so refuse instead.
            if len(set(values)) > 1:
                return None
            total += float(values[0])
        return (total / 60.0) or None

    def _qc_check_against_stored(self, minutes, stage_list, reject_types):
        """Warn when the recomputed denominator disagrees with the stored one.

        With no live artefact marks the two are the same quantity computed by
        the same function, so they should agree to within rounding. A real
        difference means the scoring open for review is not the scoring
        detection ran on, and every density in the table is then computed
        against a different amount of time than the exported density is.
        """
        stored = self._qc_stored_density_minutes(stage_list, reject_types)
        if stored is None or minutes is None:
            return
        if abs(stored - minutes) <= max(0.05, 0.001 * stored):
            return
        _density_repeat_filter.context = getattr(self, 'annot_file_path', None)
        _density_logger.warning(
            "Density denominator mismatch: this dashboard computes %.2f "
            "analysed minutes over %s from the scoring loaded for review, but "
            "the detection run stored %.2f minutes for the same stages and "
            "exclusion set. The densities shown will not match the "
            "exported ones. The usual cause is a different or edited scoring "
            "file; the stored value is the one the detector actually used.",
            minutes, "+".join(stage_list), stored)

    def _current_method_freq(self):
        """Read the method + frequency-band filter combos. Returns
        (methods_list|None, (lo, hi)|None). Single source of truth for the
        QC table and the Epochs drill so they can't drift apart. Prefers the
        exact (lo, hi) stored on the combo item over re-parsing the rounded
        display text."""
        methods = None
        if getattr(self, 'method_combo', None) is not None \
                and self.method_combo.currentIndex() > 0:
            methods = [self.method_combo.currentText()]
        freq_band = None
        if getattr(self, 'freq_band_combo', None) is not None \
                and self.freq_band_combo.currentIndex() > 0:
            data = self.freq_band_combo.currentData()   # exact (lo, hi) stored
            if data is not None:
                freq_band = (float(data[0]), float(data[1]))
            else:                                        # text-parse fallback
                try:
                    lo, hi = (self.freq_band_combo.currentText()
                              .replace(' Hz', '').split('-'))
                    freq_band = (float(lo), float(hi))
                except Exception:
                    pass
        return methods, freq_band

    def _refresh_all(self):
        """One refresh for any filter change: QC table, the right dock, and
        the active Epochs drill — all from current filters. Re-drills with
        switch_tab=False so focus stays put."""
        if self.db is None:
            return
        self.refresh_qc_dashboard()                     # QC table (filter-aware)
        selch = getattr(self, '_qc_selected_channel', None)
        if selch:
            self.on_qc_channel_selected(selch)          # right dock
        drillch = self.epochs_panel._channel
        if drillch:
            self.on_qc_drill(drillch, switch_tab=False)  # Epochs trace + band

    # TODO(perf, deferred 2026-05-25): cached per-(channel, method, freq_band)
    # QC summary table for a guaranteed <300 ms refresh on dense subjects.
    # v1 ships at ~854 ms _refresh_all (down from 2294 ms); good enough until a
    # reviewer flags the lag OR subjects scale meaningfully past ~372k events.
    # When implementing the cache:
    #   - invalidate on every write that changes the inputs: artefact
    #     mark/unmark and re-detect (events + qc_artefact_intervals).
    #   - preserve p95_amp (percentile — feeds the hard/soft flag AND is a
    #     displayed column) and pct_in_artefact (per-event interval overlap);
    #     neither does per-channel in one SQLite GROUP BY, so they need a
    #     window/secondary query or must stay in pandas.
    #   - benchmark with _scratch/mci042_filter_refresh_check.py against
    #     MCI042 (372k spindles) as the before/after baseline.
    def refresh_qc_dashboard(self):
        """Recompute the per-channel QC table for the active event type."""
        if self.db is None:
            return
        evt = self.qc_widget.current_event_type()
        methods, freq_band = self._current_method_freq()
        try:
            df = self.db.get_events(event_type=evt, methods=methods,
                                    freq_band=freq_band, columns=QC_EVENT_COLS)
        except Exception as e:
            self.status_bar.showMessage(f"QC load failed: {e}")
            return
        all_verdicts = self.db.get_channel_verdicts()   # queried once per refresh
        verdicts = {ch: v for (ch, et), v in all_verdicts.items() if et == evt}
        intervals = self.db.get_qc_artefact_intervals()
        ivs = list(zip(intervals['start_time'], intervals['end_time'])) \
            if len(intervals) else None
        # Density denominator = artefact-free analysed minutes over the run's
        # stage scope (inferred from the loaded events, matching the export),
        # subtracting the annotation artefacts AND the live marks in `ivs`, so
        # it matches the library export and shrinks when a mark is added.
        # Computed once per refresh (per-stage cache inside the helper), so the
        # 256-channel table costs one build, not one per row.
        evt_stages = (df['stage'].dropna().unique()
                      if 'stage' in df.columns and len(df) else None)
        density_min = self._qc_density_minutes(extra_intervals=ivs,
                                               event_stages=evt_stages)
        qc = compute_channel_qc(df, scored_minutes=density_min,
                                artefact_intervals=ivs,
                                coords=self.detail_dock_w._coords,
                                **self._qc_thresholds)
        # population checks: cached per run, else read in the background
        qc = self._apply_population(qc, evt, methods, freq_band)
        self._qc_events_df = df
        self._qc_df = qc
        self.qc_widget.set_data(qc, df, verdicts, self._redetect_queue)
        self.detail_dock_w.set_event_type(evt)
        # Name the denominator next to the number it defines. _qc_density_minutes
        # has just resolved the set (and whether it was read or assumed), so
        # this reports what was actually used, not a second lookup that could
        # disagree with it.
        self.detail_dock_w.set_denominator_mask(
            self._qc_reject_types_in_use(),
            getattr(self, '_qc_reject_source', 'assumed (not recorded)'),
            pending_marks=len(ivs or []))
        self.detail_dock_w.update_topo(qc)
        self.detail_dock_w.set_global_worst(
            self._global_worst_rows(df, evt), event_type=evt)
        # filter-dock decorations: ⚑ channel-artefact verdicts, ↻ queued
        artefact_set = {c for (c, e2), v in all_verdicts.items()
                        if v in ('drop', 'channel_artefact')}
        self.filter_dock.decorate_channels(artefact_set, self._redetect_queue,
                                           interp_set=self._eeg_channel_info()[2])
        self._refresh_event_counts()
        self._refresh_status_segments()
        self._refresh_toolbar_state()
        n_hard = int((qc['flag'] == 'hard').sum()) if len(qc) else 0
        n_soft = int((qc['flag'] == 'soft').sum()) if len(qc) else 0
        n_marked = len({str(r.get('evidence_channel'))
                        for _, r in intervals.iterrows()}) if len(intervals) else 0
        self.status_bar.showMessage(
            f"{len(qc)} channels · {n_hard} hard · {n_soft} soft · "
            f"{n_marked} marked artefact · {evt}")

    def _refresh_event_counts(self):
        """Per-event-type counts for the filter-dock swatches (one GROUP BY
        round-trip instead of four COUNT queries)."""
        if self.db is None:
            return
        counts = {et: 0 for et in ('slow_wave', 'spindle', 'k_complex', 'pac')}
        try:
            cur = self.db.conn.cursor()
            for et, c in cur.execute(
                    "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"):
                if et in counts:
                    counts[et] = c
        except Exception:
            pass
        self.filter_dock.set_event_counts(counts)

    def on_qc_channel_selected(self, ch):
        self._qc_selected_channel = ch
        sl = self.qc_widget.events_slice_for(ch)
        qc_row = None
        df = getattr(self, '_qc_df', None)
        if df is not None and len(df):
            hit = df[df['channel'] == ch]
            if len(hit):
                qc_row = hit.iloc[0]
        # Carry _hypno + _event_type through qc_row so ChannelDetailDock can
        # stage-tag worst-epoch rows and pick the right amplitude column
        # without widening update_channel's signature.
        if qc_row is not None:
            try:
                qc_row = dict(qc_row)
                qc_row['_epochs'] = self._epoch_table()
                qc_row['_event_type'] = self.qc_widget.current_event_type()
            except Exception:
                pass
        self.detail_dock_w.set_epoch_table(self._epoch_table())
        self.detail_dock_w.update_channel(ch, sl, qc_row)
        self._update_check_line(ch)
        self.detail_dock_w.set_marked(self._marked_for(ch),
                                      channel=ch, total=self._total_marked())

    def _on_worst_epoch_goto(self, idx):
        """Per-channel worst-epochs list -> switch to Epochs tab, jump the
        window (stays on the currently-drilled channel)."""
        try:
            self.tabs.setCurrentIndex(1)
            self.epochs_panel._goto_epoch(int(idx))
        except Exception:
            pass

    def _global_worst_rows(self, df, evt, limit=50):
        """Top-``limit`` most extreme events across ALL channels for the
        current event type, sorted by amplitude descending. Amplitude column
        follows AMP_COL (peak2peak for slow_wave/k_complex, max_amp for
        spindle/pac). Returns a list of dicts (channel, start_time, stage,
        amp) for ChannelDetailDock.set_global_worst."""
        if df is None or len(df) == 0:
            return []
        amp_col = AMP_COL.get(evt, 'max_amp')
        if amp_col not in df.columns:
            amp_col = 'max_amp' if 'max_amp' in df.columns else None
        if amp_col is None:
            return []
        d = pd.DataFrame({
            'channel': df['channel'].astype(str),
            'start_time': pd.to_numeric(df['start_time'], errors='coerce'),
            'amp': pd.to_numeric(df[amp_col], errors='coerce'),
            'stage': (df['stage'].astype(str) if 'stage' in df.columns
                      else ''),
        }).dropna(subset=['start_time', 'amp'])
        if d.empty:
            return []
        d = d.sort_values('amp', ascending=False).head(int(limit))
        hyp = self._epoch_table()
        rows = []
        for _, r in d.iterrows():
            stage = str(r['stage'])
            if (not stage or stage.lower() in ('', 'nan', 'none')) and hyp:
                stage = hyp.stage_at(r['start_time'])
            rows.append(dict(channel=str(r['channel']),
                             start_time=float(r['start_time']),
                             stage=stage, amp=float(r['amp'])))
        return rows

    def _on_global_worst_goto(self, ch, t0):
        """Global worst-events click: switch CHANNEL *and* epoch. Loads the
        channel into the Epochs panel, pages to the event's epoch, selects
        that channel's QC-table row, and shows the Epochs tab."""
        try:
            self.on_qc_drill(str(ch), switch_tab=False)
            self.epochs_panel._goto_epoch(
                self.epochs_panel.index_at(float(t0)))
            self.qc_widget.select_channel(str(ch))
            self.tabs.setCurrentIndex(1)
        except Exception:
            pass

    def _on_topo_channel_picked(self, ch):
        """Topo electrode click: SELECT that channel in the QC table (fires
        channelSelected -> on_qc_channel_selected, populating the per-channel
        detail). No drill, no tab switch, no topo recompute."""
        try:
            self.qc_widget.select_channel(str(ch))
        except Exception:
            pass

    def _drill_slice(self, ch, df=None):
        """Events of channel ``ch`` for the Epochs panel, with
        :data:`QC_DRILL_COLS`, under the dashboard's current event type,
        method and band filters. Falls back to the montage-wide QC frame
        (no uuid, so no selection) when there is no database or the read
        fails."""
        if self.db is not None:
            try:
                methods, freq_band = self._current_method_freq()
                return self.db.get_events(
                    event_type=self.qc_widget.current_event_type(),
                    channels=[str(ch)], methods=methods,
                    freq_band=freq_band, columns=QC_DRILL_COLS)
            except Exception as err:
                logger.warning(f"Could not read the events of {ch} ({err}); "
                               f"event selection is unavailable.")
        if df is not None and len(df):
            return df[df['channel'] == ch]
        return None

    def on_qc_drill(self, ch, switch_tab=True):
        df = getattr(self, '_qc_events_df', None)
        sl = self._drill_slice(ch, df)
        evt = self.qc_widget.current_event_type()
        self.selected_event_uuid = None
        self.epochs_panel.set_channel(
            ch, sl, df, event_type=evt,
            epochs=self._epoch_table(), trec=self._recording_seconds(),
            marked=self._marked_for(ch), reviews=self._reviews_for_slice(sl))
        if switch_tab:
            self.tabs.setCurrentIndex(1)
        # keys (]/[, A/R/U) reach the panel without a click on the trace
        self.epochs_panel.setFocus(Qt.OtherFocusReason)
        self._update_sample_marks()
        if not self._sample_active:
            self._refresh_sample_bar()

    # ------------------------------------------------------------------
    # Per-event selection, figures and decisions (UX spec sections 3-7)
    # ------------------------------------------------------------------
    UNDO_DEPTH = 500

    def _reviews_for_slice(self, sl):
        """The CURRENT reviewer's ``{uuid: (decision, reason, reviewer)}`` on
        a drilled slice (blind: other reviewers never tint bands or count as
        reviewed). Empty while no reviewer name is set."""
        if self.db is None or sl is None or not len(sl) \
                or 'uuid' not in sl.columns or not self.reviewer_name:
            return {}
        try:
            return self.db.get_reviews_for(sl['uuid'].tolist(),
                                           reviewer=self.reviewer_name)
        except Exception as err:
            logger.warning(f"Could not read event reviews ({err}).")
            return {}

    def _run_info(self, run_id):
        """``get_run_info`` cached per run for the session."""
        if self.db is None or run_id is None or run_id != run_id:
            return {}
        cache = self.__dict__.setdefault('_run_cache', {})
        key = (self.db.db_path, str(run_id))
        if key not in cache:
            cache[key] = self.db.get_run_info(str(run_id))
        return cache[key]

    def _selected_row(self):
        """``(row dict, run dict)`` of the selected event, or ``(None, {})``."""
        uuid = self.selected_event_uuid
        if uuid is None or self.db is None:
            return None, {}
        row = self.db.get_event(uuid)
        if not row:
            return None, {}
        return row, self._run_info(row.get('run_id'))

    def _event_status_line(self, row, prefix='Selected'):
        evt = self.epochs_panel._event_type
        word = _er.EVENT_SINGULAR.get(evt, evt)
        rv = self.epochs_panel._reviews.get(self.selected_event_uuid)
        stage = row.get('epoch_stage') or '—'
        lead = ''
        if self._sample_active and self._sample is not None and \
                self.selected_event_uuid in self._sample['rows']:
            lead = (f"Sample event "
                    f"{self._sample['pos'][self.selected_event_uuid] + 1} of "
                    f"{len(self._sample['order'])} · ")
        return lead + (f"{prefix} {word} on {row.get('channel')} at "
                f"{_er.fmt_hms1(row.get('start_time'))} · {stage} · "
                f"{_er.decision_word(*(rv[:2] if rv else (None, None)))} — "
                f"A accept · R reject · U unsure")

    def _on_event_selected(self, uuid):
        """Selection from the Epochs tab: fill the Event panel, the Current
        line and the comment; name the event in the status bar."""
        cancelled = self._disarm(cancel_message=True)
        self.selected_event_uuid = uuid
        row, run = self._selected_row()
        if row is None:
            self.detail_dock_w.event_panel.set_empty(
                EventDecisionPanel.EMPTY_TEXT)
            return
        self._render_event_panel(row, run)
        if not cancelled:
            self.status_bar.showMessage(self._event_status_line(row))

    def _on_selection_cleared(self):
        cancelled = self._disarm(cancel_message=True)
        self.selected_event_uuid = None
        self.detail_dock_w.event_panel.set_empty(EventDecisionPanel.EMPTY_TEXT)
        if not cancelled:
            self._refresh_progress()

    def _figures_for(self, row, run):
        """``(figures dict, state, note)`` for one event (build_event_rows).

        Stored figures when the row has any; ``unavailable`` when the run
        computed figures but this row has none at all (the channel's figures
        pass failed); otherwise computed on selection from the raw signal,
        or ``missing`` / ``unavailable`` without the EEG file.
        """
        evt = self.epochs_panel._event_type
        if _er.has_stored_figures(row, evt):
            return row, 'stored', None
        if _er.run_has_figures(run):
            return {}, 'unavailable', _er.FIGURES_FAILED_FIG
        cache = self.__dict__.setdefault('_fig_cache', {})
        uuid = str(row.get('uuid'))
        if uuid in cache:
            figs, note = cache[uuid]
            return (figs, 'computed', None) if figs else (
                {}, 'unavailable', note or _er.FIGURES_FAILED_FIG)
        if self.eeg_data is None:
            if _er.run_figures_switched_off(run):
                return {}, 'unavailable', _er.FIGURES_OFF_FIG
            return {}, 'missing', None
        return {}, 'computing', None

    def _render_event_panel(self, row, run):
        """Build and show the Event panel rows for ``row``."""
        panel = self.detail_dock_w.event_panel
        ep = self.epochs_panel
        evt = ep._event_type
        figs, state, note = self._figures_for(row, run)
        thr = self.db.get_thresholds(row.get('run_id'), row.get('channel'),
                                     row.get('method'), row.get('start_time')) \
            if self.db is not None else None
        interp = str(row.get('channel')) in self._eeg_channel_info()[2]
        rows = _er.build_event_rows(
            row, event_type=evt, run=run, run_id=row.get('run_id'),
            thresholds=thr, figures=figs, figure_state=state,
            interpolated=interp, outlier_thr=ep._amp_thr, amp_col=ep._amp_col,
            ptp_units_uv=self.db.ptp_units_microvolts() if self.db else False,
            figure_note=note)
        srow = self._sample_row_for(str(row.get('uuid')))
        if srow is not None:
            rows = [srow] + rows
        panel.set_rows(rows)
        panel.set_note(_er.LOAD_EEG_NOTE if state == 'missing' else '')
        panel.set_controls_enabled(True)
        self._refresh_current_line(row)
        self._refresh_progress()
        if state == 'computing':
            uuid = str(row.get('uuid'))
            QtCore.QTimer.singleShot(
                0, lambda u=uuid, r=row, rn=run: self._compute_figures_later(
                    u, r, rn))

    def _compute_figures_later(self, uuid, row, run):
        """Compute a pre-4.6 event's figures from the raw signal (with the
        run's ``ref_chan``), cache them, and re-render if still selected."""
        note = None
        try:
            figs = self._compute_figures(row, run)
        except _FiguresNotComputable as err:
            figs, note = None, str(err)
        except Exception as err:
            logger.warning(f"Could not compute the event figures ({err}).")
            figs, note = None, f"figures could not be computed ({err})"
        cache = self.__dict__.setdefault('_fig_cache', {})
        cache[uuid] = (figs or {}, note)
        if self.selected_event_uuid == uuid:
            r2, run2 = self._selected_row()
            if r2 is not None:
                self._render_event_panel(r2, run2)

    def _compute_figures(self, row, run):
        """``event_metrics.event_figures`` on a ±(S + P) read."""
        from turtlewave_hdEEG import event_metrics as em
        evt = self.epochs_panel._event_type
        geo = em.GEOMETRY.get(evt, em.GEOMETRY['spindle'])
        start, end = float(row['start_time']), float(row['end_time'])
        pad = geo['S'] + geo['P']
        ch = str(row['channel'])
        # The detector's own reference, target included when it is one of
        # the reference channels (an average reference), as Wonambi's
        # read_data(chan, ref_chan) -> montage(ref_chan=...) does.
        ref = _er.run_ref_chan(run)
        names = set(self._eeg_channel_info()[0])
        missing = [r for r in ref if r not in names]
        if missing:
            logger.warning(
                f"Run {row.get('run_id')} was detected re-referenced to "
                f"{', '.join(ref)}; {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not in this "
                f"recording, so the figures for {ch} are not computed (the "
                f"stored reference is not what the detector saw).")
            raise _FiguresNotComputable(
                f"not computed: reference channel{'s' if len(missing) > 1 else ''}"
                f" {', '.join(missing)} not in this recording")
        out = self._read_eeg_channels(list(dict.fromkeys([ch] + ref)),
                                      start - pad, end + pad)
        if not out:
            return {}
        ts, data, sf, got = out
        x = _rereference_like_wonambi(data, got, ts, sf, ch, ref)
        note = (f"re-referenced to {len(ref)} channel(s)" if ref
                else 'as stored')
        df = self.epochs_panel._df
        others = []
        if df is not None and len(df) and 'run_id' in df.columns:
            same = df[(df['run_id'] == row.get('run_id'))
                      & (df['uuid'] != row.get('uuid'))]
            others = list(zip(pd.to_numeric(same['start_time']),
                              pd.to_numeric(same['end_time'])))
        table = self._epoch_table()
        stage_epochs = None
        if table is not None and len(table):
            stage_epochs = [(float(a), float(b), s) for a, b, s in
                            zip(table.starts, table.ends, table.stages)
                            if b > start - pad and a < end + pad]
        reject = []
        if self.annotations is not None:
            params = (run or {}).get('params') or {}
            for t in params.get('reject_types') or ['Artefact', 'Arousal']:
                try:
                    for e in self.annotations.get_events(name=t) or []:
                        reject.append((float(e['start']), float(e['end'])))
                except Exception:
                    continue
        th = self.db.get_thresholds(row.get('run_id'), ch, row.get('method'),
                                    start) if self.db is not None else None
        thd = ({str(r['name']): float(r['value']) for _, r in th.iterrows()}
               if th is not None and len(th) else None)
        fig = em.event_figures(
            x, ts, sf, start, end,
            (float(row['freq_lower']), float(row['freq_upper'])),
            others=others, stage_epochs=stage_epochs,
            run_stages=_er.run_stages(run) or None, thresholds=thd,
            event_type=evt, ref_note=note, method=row.get('method'),
            event_values={k: row.get(k) for k in
                          ('peak_val_det', 'det_trough', 'det_ptp',
                           'det_zero_time')},
            duration_bounds=_er.run_duration_bounds(run, row.get('method')),
            reject_intervals=reject or None)
        return fig.as_dict()

    def _refresh_current_line(self, row=None):
        """``Current`` line + sub-lines (other reviewers, comment)."""
        panel = self.detail_dock_w.event_panel
        uuid = self.selected_event_uuid
        if uuid is None or self.db is None:
            panel.set_current('Not reviewed', [])
            return
        mine = (self.db.get_review(uuid, self.reviewer_name)
                if self.reviewer_name else None)
        subs, tip = [], ''
        if mine:
            dec = mine['decision']
            text = (f"{_er.DECISION_TITLE.get(dec, dec)} by "
                    f"{self.reviewer_name} · {_er.fmt_when(mine.get('reviewed_at'))}")
            if mine.get('reason'):
                text += f" · {_er.REASON_SHORT.get(mine['reason'], mine['reason'])}"
            if mine.get('comment'):
                c = str(mine['comment'])
                tip = c
                subs.append(c if len(c) <= 80 else c[:79] + '…')
            panel.set_decision(dec)
            if not panel.comment.hasFocus():
                panel.comment.setText(str(mine.get('comment') or ''))
            panel.set_reason(mine.get('reason') or '')
        else:
            text = 'Not reviewed'
            panel.set_decision(None)
            if not panel.comment.hasFocus():
                panel.comment.setText('')
            panel.set_reason('')
        others = self.db.get_other_reviews(uuid, self.reviewer_name or '')
        if others:
            if self._show_others:
                for o in others:
                    subs.append(
                        f"{o['reviewer']}: "
                        f"{_er.decision_word(o['decision'], o.get('reason'))}"
                        f" · {_er.fmt_when(o.get('reviewed_at'), today=datetime.min.date())}")
            else:
                subs.append(f"Also decided by {len(others)} other "
                            f"reviewer(s); hidden so your decisions stay "
                            f"independent.")
        panel.set_current(text, subs, tip)

    def _refresh_progress(self):
        ep = self.epochs_panel
        ev = ep._ev
        panel = self.detail_dock_w.event_panel
        if ev is None or ev.empty or ep._channel is None:
            panel.set_progress('')
            return
        if self._sample_active and self._sample is not None:
            pr = self._sample_progress()
            if pr is not None:
                panel.set_progress(
                    f"Progress  {pr['n_reviewed']} of {pr['n_total']} in "
                    f"sample · {pr['n_accept']} accepted · {pr['n_reject']} "
                    f"rejected · {pr['n_unsure']} unsure")
                return
        n_all = int(ev['uuid'].notna().sum()) or len(ev)
        decs = [v[0] for u, v in ep._reviews.items()]
        counts = {d: decs.count(d) for d in ('accept', 'reject', 'unsure')}
        panel.set_progress(
            f"Progress  {len(decs)} reviewed / {n_all} on {ep._channel} · "
            f"{counts['accept']} accepted · {counts['reject']} rejected · "
            f"{counts['unsure']} unsure")

    # ---- keys --------------------------------------------------------
    def _setup_review_shortcuts(self):
        """Window shortcuts for the review keys, live only on the Epochs tab
        and silent while a text field (the comment field included) has
        focus."""
        binds = [('A', lambda: self._key_decide('accept')),
                 ('R', lambda: self._key_decide('reject')),
                 ('U', lambda: self._key_decide('unsure')),
                 ('C', self._key_comment),
                 (']', lambda: self._nav_unreviewed(+1)),
                 ('[', lambda: self._nav_unreviewed(-1)),
                 ('}', lambda: self._nav_any(+1)),
                 ('{', lambda: self._nav_any(-1)),
                 (QtGui.QKeySequence.Undo, self._undo),
                 ('Escape', self._key_escape),
                 ('Return', self._key_enter), ('Enter', self._key_enter)]
        binds += [(str(d), lambda d=str(d): self._key_digit(d))
                  for d in range(1, 10)]
        self._review_shortcuts = []
        for key, slot in binds:
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)
            self._review_shortcuts.append(sc)
        self.tabs.currentChanged.connect(self._update_review_shortcuts)
        QApplication.instance().focusChanged.connect(
            lambda *_: self._update_review_shortcuts())
        self._update_review_shortcuts()

    def _update_review_shortcuts(self, *_):
        fw = QApplication.focusWidget()
        typing = isinstance(fw, (QLineEdit, QtWidgets.QTextEdit,
                                 QtWidgets.QPlainTextEdit,
                                 QtWidgets.QAbstractSpinBox))
        on = self.tabs.currentIndex() == 1 and not typing
        for sc in getattr(self, '_review_shortcuts', []):
            sc.setEnabled(on)

    def review_shortcuts_enabled(self):
        return any(sc.isEnabled() for sc in getattr(self, '_review_shortcuts', []))

    def _key_comment(self):
        if self.selected_event_uuid is not None:
            c = self.detail_dock_w.event_panel.comment
            c.setFocus(Qt.ShortcutFocusReason)
            c.selectAll()

    def _key_escape(self):
        if self._disarm(cancel_message=True):
            return
        self.epochs_panel.escape_step()

    def _on_comment_escape(self):
        if self._disarm(cancel_message=True):
            return
        self.epochs_panel.setFocus(Qt.OtherFocusReason)

    def _hint_for_arm(self):
        if self._armed == 'reject':
            if self._last_reject_reason:
                lbl = _er.REASON_SHORT.get(self._last_reject_reason,
                                           self._last_reject_reason)
                return f"Choose a reason: 1–9, or Enter for “{lbl}”"
            return 'Choose a reason: 1–9'
        return 'Optional reason: 1–9, or Enter to save without one'

    def _key_decide(self, decision):
        """A writes at once; R / U arm and wait for a reason (spec 5)."""
        if self.selected_event_uuid is None:
            return
        panel = self.detail_dock_w.event_panel
        if decision == 'accept':
            self._disarm()
            self._write_decision('accept', None)
            return
        self._armed = decision
        self._pending_other = False
        if decision == 'reject':
            panel.set_reason(self._last_reject_reason or '')
        else:
            panel.set_reason('')
        panel.set_armed(decision, self._hint_for_arm())

    _on_decision_requested = _key_decide   # EpochsPanel.decisionRequested

    def _key_digit(self, digit):
        if self._armed is None:
            return
        self._choose_reason(_er.REASON_BY_DIGIT.get(str(digit)))

    def _on_reason_picked(self, token):
        """Mouse pick in the reason combo: writes while armed."""
        if self._armed is None:
            return
        self._choose_reason(token or None)

    def _choose_reason(self, token):
        panel = self.detail_dock_w.event_panel
        if self._armed == 'reject' and not token:
            panel.set_armed('reject', 'Choose a reason: 1–9')
            return
        panel.set_reason(token or '')
        if token == 'other' and not panel.comment.text().strip():
            self._pending_other = True
            panel.set_armed(self._armed, 'Describe the reason, then Enter')
            panel.comment.setFocus(Qt.OtherFocusReason)
            return
        self._write_decision(self._armed, token)

    def _key_enter(self):
        """Enter: confirm an armed decision with the preselected reason."""
        panel = self.detail_dock_w.event_panel
        if self._armed is None:
            return
        if self._pending_other:
            if not panel.comment.text().strip():
                return
            self._write_decision(self._armed, 'other')
            return
        if self._armed == 'reject':
            if not self._last_reject_reason:
                return
            self._choose_reason(self._last_reject_reason)
            return
        token = str(panel.reason_combo.currentData() or '') or None
        self._choose_reason(token)

    def _on_comment_submitted(self):
        """Enter in the comment field: complete an armed decision, or save
        the comment on this reviewer's existing decision."""
        if self._armed is not None:
            self._key_enter()
            if self._armed is None:
                self.epochs_panel.setFocus(Qt.OtherFocusReason)
            return
        uuid = self.selected_event_uuid
        if uuid is None or self.db is None or not self.reviewer_name:
            return
        mine = self.db.get_review(uuid, self.reviewer_name)
        if mine:
            self._write_decision(mine['decision'], mine.get('reason'),
                                 advance=False)
        self.epochs_panel.setFocus(Qt.OtherFocusReason)

    def _disarm(self, cancel_message=False):
        """Drop an armed Reject / Unsure. Returns True when one was armed."""
        armed = getattr(self, '_armed', None)
        self._armed = None
        self._pending_other = False
        self.detail_dock_w.event_panel.set_armed(None)
        if armed and cancel_message:
            self.status_bar.showMessage(
                'Reject cancelled — no reason chosen.' if armed == 'reject'
                else 'Unsure cancelled — nothing saved.')
        return bool(armed)

    # ---- reviewer ------------------------------------------------------
    REVIEWER_PROMPT = ('Your name or initials. It is saved with every accept '
                       '/ reject decision, so two reviewers\' decisions on the '
                       'same recording can be compared.')

    def _ask_reviewer_name(self, prefill):
        """The reviewer dialog; returns ``(text, ok)``. Separate so tests can
        replace it."""
        return QtWidgets.QInputDialog.getText(
            self, 'Reviewer name', self.REVIEWER_PROMPT, QLineEdit.Normal,
            prefill)

    def _ensure_reviewer_name(self):
        """Ask once per session; never invent a name."""
        if self.reviewer_name:
            return True
        prefill = str(_review_settings().value('review/reviewer_name', '') or '')
        name, ok = self._ask_reviewer_name(prefill)
        name = str(name or '').strip()[:40]
        if not ok or not name:
            self.status_bar.showMessage(
                'Decision not saved — a reviewer name is needed.')
            return False
        self.set_reviewer_name(name)
        return True

    def set_reviewer_name(self, name):
        """Set the name decisions are saved under and the bands, Current
        line and progress are scoped to; saved to ``QSettings``. Returns the
        name set (whitespace stripped, at most 40 characters)."""
        old = self.reviewer_name
        self.reviewer_name = str(name or '').strip()[:40]
        if self.reviewer_name:
            _review_settings().setValue('review/reviewer_name',
                                        self.reviewer_name)
        self.seg_reviewer.setText(
            f"Reviewer: {self.reviewer_name}" if self.reviewer_name
            else "Reviewer: not set")
        if old != self.reviewer_name:
            self._undo_stack = []
            ep = self.epochs_panel
            if ep._channel is not None:
                ep.set_reviews(self._reviews_for_slice(ep._df))
                if self.selected_event_uuid is not None:
                    self._refresh_current_line()
                self._refresh_progress()
            # revisit list, end message and sample position belong to the
            # previous reviewer: the next one starts at their first
            # undecided event
            self._revisit = None
            self._sample_cursor = None
            self._sample_times = []
            self.detail_dock_w.event_panel.set_end(None)
            self._refresh_sample_bar()
            if self._sample_active:
                self._refresh_progress()
            if old:
                self.status_bar.showMessage(
                    "Changing the name shows that reviewer's decisions and "
                    "sample progress instead.")
            else:
                self.status_bar.showMessage(
                    f"Reviewer: {self.reviewer_name}" if self.reviewer_name
                    else "Reviewer: not set")
        return self.reviewer_name

    def _prompt_reviewer_name(self):
        """Review ▸ Reviewer name… and the status-bar segment."""
        prefill = self.reviewer_name or str(
            _review_settings().value('review/reviewer_name', '') or '')
        name, ok = self._ask_reviewer_name(prefill)
        name = str(name or '').strip()[:40]
        if ok and name:
            self.set_reviewer_name(name)

    def _confirm_show_others(self):
        """Ask once per session before showing other reviewers."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Show other reviewers' decisions?")
        box.setText("Show other reviewers' decisions?")
        box.setInformativeText(SHOW_OTHERS_WARNING)
        show = box.addButton('Show them', QtWidgets.QMessageBox.AcceptRole)
        box.addButton('Cancel', QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        return box.clickedButton() is show

    def _on_show_others(self, checked):
        if checked and not self._show_others_confirmed:
            if not self._confirm_show_others():
                self.act_show_others.blockSignals(True)
                self.act_show_others.setChecked(False)
                self.act_show_others.blockSignals(False)
                return
            self._show_others_confirmed = True
        self._show_others = bool(checked)
        bar = getattr(self, 'sample_bar', None)
        if bar is not None and bar.others_chk.isChecked() != bool(checked):
            bar.others_chk.blockSignals(True)
            bar.others_chk.setChecked(bool(checked))
            bar.others_chk.blockSignals(False)
        if self.selected_event_uuid is not None:
            self._refresh_current_line()

    # ---- writing and undo ------------------------------------------------
    def _write_decision(self, decision, reason, advance=True):
        """Store ``decision`` for the selected event; push undo; advance."""
        uuid = self.selected_event_uuid
        if uuid is None:
            return False
        if self.db is None or not self.db.has_review_backend:
            self._disarm()
            self.status_bar.showMessage(
                "Decision not saved: this TurtleWave library cannot store "
                "event reviews.")
            return False
        if not self._ensure_reviewer_name():
            self._disarm()
            return False
        panel = self.detail_dock_w.event_panel
        comment = panel.comment.text().strip() or None
        if reason == 'other' and not comment:
            self._pending_other = True
            panel.set_armed(decision, 'Describe the reason, then Enter')
            return False
        who = self.reviewer_name
        before = self.db.get_review(uuid, who)
        try:
            self.db.add_review(uuid, decision, reviewer=who,
                               comments=comment or '', reason=reason)
        except Exception as err:
            self._disarm()
            self.status_bar.showMessage(f"Decision not saved: {err}")
            return False
        row, _run = self._selected_row()
        row = row or {}
        self._undo_stack.append({
            'uuid': uuid, 'reviewer': who, 'before': before,
            'action': decision, 'channel': row.get('channel'),
            'start': row.get('start_time'), 'event_type':
                self.epochs_panel._event_type})
        del self._undo_stack[:-self.UNDO_DEPTH]
        if decision == 'reject':
            self._last_reject_reason = reason
        self._disarm()
        ep = self.epochs_panel
        reviews = dict(ep._reviews)
        reviews[uuid] = (decision, reason, who)
        ep.set_reviews(reviews)
        self._refresh_current_line()
        self._refresh_progress()
        # status: changed / new, next unreviewed or channel complete
        chg = before and (before['decision'], before.get('reason')) != (
            decision, reason)
        evt = ep._event_type
        if chg:
            msg = (f"Changed from {_er.decision_word(before['decision'], before.get('reason'))}"
                   f" to {_er.decision_word(decision, reason)} · Ctrl+Z to undo")
        else:
            word = _er.EVENT_SINGULAR.get(evt, evt)
            msg = (f"{_er.DECISION_TITLE[decision]} {word} on "
                   f"{row.get('channel')} at {_er.fmt_hms1(row.get('start_time'))}")
            if reason:
                msg += f" ({_er.REASON_SHORT.get(reason, reason)})"
            nxt = self._peek_next_unreviewed(uuid)
            if nxt is None:
                msg += (f" · all {int(ep._ev['uuid'].notna().sum())} on "
                        f"{ep._channel} reviewed")
            else:
                msg += f" · next unreviewed {_er.fmt_hms1(nxt)}"
            msg += ' · Ctrl+Z to undo'
        if self._sample_active and self._sample is not None:
            self._refresh_progress()
            self._after_sample_write(
                uuid, decision, reason, row,
                (before['decision'], before.get('reason')) if chg else None,
                advance)
            return True
        if advance and self._auto_advance and not chg:
            if ep.n_unreviewed():
                ep.next_unreviewed()
        self.status_bar.showMessage(msg)
        return True

    def _peek_next_unreviewed(self, uuid):
        """Start time of the next unreviewed event after ``uuid`` (wrapping),
        or ``None`` when every event on the channel is reviewed."""
        ev = self.epochs_panel._ev
        ok = ev['uuid'].notna() & ~ev['uuid'].isin(
            list(self.epochs_panel._reviews))
        if not ok.any():
            return None
        pos = ev.index[ev['uuid'] == uuid]
        after = ev.index[ok & (ev.index > (pos[0] if len(pos) else -1))]
        cand = after if len(after) else ev.index[ok]
        return float(ev.at[cand[0], '_start'])

    def _undo(self):
        """Ctrl+Z: restore this session's last decision write."""
        self._disarm()
        if not self._undo_stack:
            self.status_bar.showMessage('Nothing to undo.')
            return
        ent = self._undo_stack.pop()
        try:
            self.db.restore_review(ent['uuid'], ent['reviewer'], ent['before'])
        except Exception as err:
            self.status_bar.showMessage(f"Undo failed: {err}")
            return
        ep = self.epochs_panel
        if ent['channel'] is not None and ent['channel'] != ep._channel:
            self.on_qc_drill(str(ent['channel']), switch_tab=False)
        else:
            reviews = dict(ep._reviews)
            b = ent['before']
            if b is None:
                reviews.pop(ent['uuid'], None)
            else:
                reviews[ent['uuid']] = (b['decision'], b.get('reason'),
                                        ent['reviewer'])
            ep.set_reviews(reviews)
        ep.select_event(ent['uuid'])
        b = ent['before']
        now = ('not reviewed' if b is None
               else _er.decision_word(b['decision'], b.get('reason')))
        self._refresh_sample_bar()
        self._refresh_report()
        self.status_bar.showMessage(
            f"Undid {ent['action']} of {ent['channel']} "
            f"{_er.fmt_hms1(ent['start'])} — now {now}")

    def _clear_decision(self):
        """Clear: delete this reviewer's decision on the selected event."""
        uuid = self.selected_event_uuid
        if uuid is None or self.db is None or not self.reviewer_name:
            return
        before = self.db.get_review(uuid, self.reviewer_name)
        if before is None:
            self.status_bar.showMessage('Nothing to clear.')
            return
        self.db.restore_review(uuid, self.reviewer_name, None)
        row, _run = self._selected_row()
        row = row or {}
        self._undo_stack.append({
            'uuid': uuid, 'reviewer': self.reviewer_name, 'before': before,
            'action': 'clear', 'channel': row.get('channel'),
            'start': row.get('start_time'),
            'event_type': self.epochs_panel._event_type})
        ep = self.epochs_panel
        reviews = dict(ep._reviews)
        reviews.pop(uuid, None)
        ep.set_reviews(reviews)
        self._refresh_current_line()
        self._refresh_progress()
        self.status_bar.showMessage(
            f"Cleared your decision on {row.get('channel')} "
            f"{_er.fmt_hms1(row.get('start_time'))} · Ctrl+Z to undo")

    # ------------------------------------------------------------------
    # Review sample (UX spec section 2; library review_sampling)
    # ------------------------------------------------------------------
    def _current_scope(self):
        """Detection scope of the events in view: the drilled channel's run,
        else the dashboard's population run, else the newest run matching
        the dashboard filters."""
        if self.db is None:
            return None
        evt = self.qc_widget.current_event_type()
        run_id = None
        df = self.epochs_panel._df
        if df is not None and len(df) and 'run_id' in df.columns:
            vc = df['run_id'].dropna().value_counts()
            run_id = vc.index[0] if len(vc) else None
        if run_id is None and self._pop_view and self._pop_view[1]:
            run_id = self._pop_view[1]['res'].get('run_id')
        if run_id is None:
            methods, band = self._current_method_freq()
            q = "SELECT run_id FROM events WHERE event_type = ? AND run_id IS NOT NULL"
            p = [evt]
            if methods:
                q += f" AND method IN ({','.join('?' * len(methods))})"
                p += list(methods)
            if band:
                q += " AND freq_lower = ? AND freq_upper = ?"
                p += [float(band[0]), float(band[1])]
            try:
                hit = self.db.conn.execute(q + " LIMIT 1", p).fetchone()
            except Exception:
                hit = None
            run_id = hit[0] if hit else None
        try:
            return _sr.scope_for_run(self.db.conn, run_id, evt)
        except Exception:
            return None

    def _sample_scope_tag(self, scope):
        """`` · Moelle2011 9–12 Hz`` when two runs share the dashboard view."""
        view = self._pop_view[1] if self._pop_view else None
        if view and len(view['res'].get('runs_in_view') or []) > 1 and scope:
            return (f" · {scope['method']} {float(scope['freq_lower']):g}–"
                    f"{float(scope['freq_upper']):g} Hz")
        return ''

    def _load_sample(self, design):
        conn = self.db.conn
        rows = _sr.sample_rows(conn, design['sample_id'])
        order = _sr.presentation_order(conn, design['sample_id'])
        order += [u for u in sorted(rows) if u not in set(order)]
        self._sample = {'id': design['sample_id'], 'design': design,
                        'rows': rows, 'order': order,
                        'pos': {u: i for i, u in enumerate(order)}}
        return self._sample

    def _sample_progress(self):
        s = self._sample
        if s is None or not self.reviewer_name:
            return None
        from turtlewave_hdEEG.review_sampling import sample_progress
        return sample_progress(self.db.conn, s['id'],
                               reviewer=self.reviewer_name)

    def _refresh_sample_bar(self):
        bar = self.sample_bar
        if self.db is None:
            bar.set_state('none', _sr.NO_SAMPLE_TEXT)
            return
        scope = self._current_scope()
        tag = self._sample_scope_tag(scope)
        if self._sample_active and self._sample is not None:
            pr = self._sample_progress()
            n, tot = (pr['n_reviewed'], pr['n_total']) if pr else (0, 0)
            u = pr['n_unsure'] if pr else 0
            if self._revisit is not None and self._revisit:
                bar.set_state('revisit', _sr.bar_text(
                    'revisit', n_revisit=len(self._revisit), scope_tag=tag))
            elif pr and not pr['next_uuids']:
                bar.set_state('done', _sr.bar_text(
                    'done', n=n, total=tot, unsure=u, scope_tag=tag))
            else:
                eta = _sr.session_eta(self._sample_times, tot - n)
                bar.set_state('active', _sr.bar_text(
                    'active', n=n, total=tot, unsure=u, eta_s=eta,
                    scope_tag=tag))
            return
        designs = _sr.samples_for_scope(self.db.conn, scope) if scope else []
        if not designs:
            bar.set_state('none', _sr.bar_text('none', scope_tag=tag))
            self._sample = None
            return
        # keep the sample chosen this session (drawn or reopened) while the
        # scope is the same; only a different scope loads its newest design
        if self._sample is None or self._sample.get('scope') != scope:
            self._load_sample(designs[0])
            self._sample['scope'] = scope
        pr = self._sample_progress()
        bar.set_state('idle', _sr.bar_text(
            'idle', design=self._sample['design'],
            total=len(self._sample['rows']),
            n=pr['n_reviewed'] if pr else 0, reviewer=self.reviewer_name,
            scope_tag=tag))

    def _set_panel_sample_mode(self, on):
        evp = self.detail_dock_w.event_panel
        evp.set_sample_mode(on)
        self._update_sample_marks()

    def _update_sample_marks(self):
        ep = self.epochs_panel
        if not self._sample_active or self._sample is None:
            ep.set_sample_marks([])
            return
        ep.set_sample_marks([float(r['start_time'])
                             for r in self._sample['rows'].values()
                             if r['channel'] == ep._channel])

    def _start_sample(self):
        """Resume (or start) the sample of the events in view."""
        if self.db is None:
            return
        if not self._ensure_reviewer_name():
            return
        if self._sample is None:
            self._refresh_sample_bar()
        if self._sample is None:
            self.status_bar.showMessage('No review sample for this run yet.')
            return
        filtered = self.epochs_panel.clear_check_filter(emit=False)
        self._sample_active = True
        self._revisit = None
        self._set_panel_sample_mode(True)
        self.tabs.setCurrentIndex(1)
        self._refresh_sample_bar()
        self._sample_nav(+1, quiet_wrap=True, include_cursor=True)
        if filtered:
            self.status_bar.showMessage(_sr.FILTER_OFF)

    def _exit_sample(self):
        self._sample_active = False
        self._revisit = None
        self._set_panel_sample_mode(False)
        self.detail_dock_w.event_panel.set_end(None)
        self._refresh_sample_bar()
        self._refresh_progress()
        self.status_bar.showMessage('Left the review sample.')

    def _sample_candidates(self):
        """Undecided (or, when revisiting, unsure) sample uuids for the
        current reviewer, as a set."""
        if self._revisit is not None:
            return set(self._revisit)
        pr = self._sample_progress()
        return set(pr['next_uuids']) if pr else set()

    def _sample_nav(self, step, quiet_wrap=False, include_cursor=False):
        """``]`` / ``[`` in sample mode: next / previous undecided sample
        event in the library's presentation order, any channel.

        The search starts AT the cursor when the cursor event is still
        undecided and is not the selection (after ``}`` browsing, or with
        ``include_cursor`` on Resume), so the reviewer comes back to it;
        otherwise just past it (``]`` on an undecided event skips it).
        """
        s = self._sample
        if s is None:
            return None
        todo = self._sample_candidates()
        if not todo:
            self._show_sample_end()
            return None
        order = s['order']
        # The sample position is the cursor, moved only by ] / [ and by a
        # sample-mode advance; } / {, clicks and the report never move it,
        # so browsing to a later sample event cannot make ] skip the ones
        # in between.
        cur = s['pos'].get(self._sample_cursor, -1)
        back_to_cursor = (cur >= 0 and self._sample_cursor in todo and (
            include_cursor or self.selected_event_uuid != self._sample_cursor))
        if step > 0:
            start = cur if back_to_cursor else cur + 1
            after = [u for u in order[max(start, 0):] if u in todo]
            pick = after[0] if after else None
            if pick is None:
                pick = next(u for u in order if u in todo)
                if not quiet_wrap:
                    self.status_bar.showMessage(
                        'Back to the first undecided sample event.')
                    return self._goto_sample_event(pick, status=False)
        else:
            end = cur + 1 if back_to_cursor else max(cur, 0)
            before = [u for u in order[:end] if u in todo]
            pick = before[-1] if before else None
            if pick is None:
                return None
        return self._goto_sample_event(pick)

    def _goto_sample_event(self, uuid, status=True, move_cursor=True):
        """Drill the event's channel when needed, page and select it.
        ``move_cursor`` is False for jumps that are not sample navigation
        (the report's disagreement list)."""
        row = self._sample['rows'].get(uuid) if self._sample else None
        if row is None:
            return None
        ch = str(row['channel'])
        ep = self.epochs_panel
        moved = ch != ep._channel
        if moved:
            self.on_qc_drill(ch, switch_tab=False)
            self.qc_widget.select_channel(ch)
            self._update_sample_marks()
        if move_cursor:
            self._sample_cursor = uuid
        if not ep.select_event(uuid):
            self.status_bar.showMessage(
                f"Sample event on {ch} at {_er.fmt_hms1(row['start_time'])} "
                f"is not in the events in view; clear the method or band "
                f"filter to reach it.")
            return None
        if status:
            i = self._sample['pos'][uuid] + 1
            self.status_bar.showMessage(
                f"Sample event {i} of {len(self._sample['order'])}"
                + (f" · moved to {ch}." if moved else ''))
        return uuid

    def _show_sample_end(self):
        pr = self._sample_progress()
        if pr is None:
            return
        u = pr['n_unsure']
        text = _sr.end_text(pr['n_total'], self.reviewer_name, u)
        self.detail_dock_w.event_panel.set_end(text, u)
        self._revisit = None
        self._refresh_sample_bar()
        self.status_bar.showMessage(
            f"Review sample complete: {pr['n_reviewed']} of {pr['n_total']} "
            f"decided by {self.reviewer_name}.")

    def _revisit_unsure(self):
        """Make ``]`` / ``[`` step through this reviewer's unsure sample
        events."""
        s = self._sample
        if s is None or not self.reviewer_name:
            return
        mine = _sr.reviewer_labels(self.db.conn, s['id']).get(
            self.reviewer_name, {})
        self._revisit = [u for u in s['order']
                         if mine.get(u, (None,))[0] == 'unsure']
        self.detail_dock_w.event_panel.set_end(None)
        self._refresh_sample_bar()
        if self._revisit:
            self._goto_sample_event(self._revisit[0])

    def _after_sample_write(self, uuid, decision, reason, row, chg, advance):
        """Status, auto-advance and progress after a write in sample mode."""
        import time as _time
        s = self._sample
        if uuid not in s['rows']:
            self.status_bar.showMessage(_sr.OUTSIDE_SAMPLE)
            self._refresh_sample_bar()
            return
        self._sample_times.append(_time.time())
        if self._revisit is not None and decision in ('accept', 'reject'):
            self._revisit = [u for u in self._revisit if u != uuid]
        self._refresh_sample_bar()
        self._refresh_report()
        todo = self._sample_candidates()
        evt = self.epochs_panel._event_type
        word = _er.EVENT_SINGULAR.get(evt, evt)
        if chg:
            head = (f"Changed from {_er.decision_word(*chg)} to "
                    f"{_er.decision_word(decision, reason)}")
        else:
            head = (f"{_er.DECISION_TITLE[decision]} {word} on "
                    f"{row.get('channel')} at "
                    f"{_er.fmt_hms1(row.get('start_time'))}"
                    + (f" ({_er.REASON_SHORT.get(reason, reason)})"
                       if reason else ''))
        if not todo:
            if self._revisit is not None:
                self._revisit = None
                self._refresh_sample_bar()
            self._show_sample_end()
            return
        order = s['order']
        cur = s['pos'].get(self._sample_cursor, -1)
        nxt = next((u for u in order[max(cur, 0):] if u in todo),
                   next(u for u in order if u in todo))
        nrow = s['rows'][nxt]
        msg = (f"{head} · next in sample: {nrow['channel']} at "
               f"{_er.fmt_hms1(nrow['start_time'])} · Ctrl+Z to undo")
        if advance and self._auto_advance and not chg:
            self._goto_sample_event(nxt, status=False)
        self.status_bar.showMessage(msg)

    def _sample_row_for(self, uuid):
        """The Event panel's ``sample`` row, shown only after the current
        reviewer has decided this sample event (the stratum and flags are
        never shown before a decision; ``is_shared`` never)."""
        if not self._sample_active or self._sample is None:
            return None
        srow = self._sample['rows'].get(uuid)
        mine = self.epochs_panel._reviews.get(uuid)
        # accept / reject only: an unsure event is revisited later and must
        # not be primed by its stratum or flags
        if srow is None or not mine or mine[0] not in ('accept', 'reject'):
            return None
        return {'key': 'sample', 'label': 'Sample',
                'value': _sr.stratum_text(srow),
                'sub': [f"event {self._sample['pos'][uuid] + 1} of "
                        f"{len(self._sample['order'])}"],
                'tooltip': _sr.weight_tooltip(srow.get('weight')),
                'level': None}

    # ---- draw dialog and report ------------------------------------------
    def _update_review_menu(self):
        has = self._sample is not None
        self.act_resume_sample.setEnabled(has and not self._sample_active)
        self.act_exit_sample.setEnabled(self._sample_active)
        self.act_precision_report.setEnabled(has)

    def _exec_dialog(self, dlg):
        """``dlg.exec_()``; separate so tests can drive the dialog."""
        return dlg.exec_()

    def _open_draw_dialog(self):
        if self.db is None:
            return
        if not self._ensure_reviewer_name():
            return
        scope = self._current_scope()
        if scope is None:
            self.status_bar.showMessage(
                'Drill into a channel of one detection run first.')
            return
        from turtlewave_hdEEG import review_sampling as _rsm
        _rsm.ensure_review_sampling_schema(self.db.conn)
        pop, err = None, None
        try:
            pop = _sr.population_for(self.db.conn, scope)
        except ValueError as e:
            err = str(e)
        designs = _sr.samples_for_scope(self.db.conn, scope)
        note = None
        if designs:
            d0 = dict(designs[0])
            d0['_n_rows'] = len(_sr.sample_rows(self.db.conn, d0['sample_id']))
            counts = {rv: len(lab) for rv, lab in _sr.reviewer_labels(
                self.db.conn, d0['sample_id']).items()}
            note = _sr.existing_sample_note(d0, counts)
        run = self._run_info(self._pop_view[1]['res'].get('run_id')) \
            if self._pop_view and self._pop_view[1] else {}
        ev = _sr.EVENT_PLURAL.get(scope['event_type'], scope['event_type'])
        line = (f"{ev} · {scope['method']} {float(scope['freq_lower']):g}–"
                f"{float(scope['freq_upper']):g} Hz · "
                f"{_er.run_label(run, run.get('run_id'))}")
        dlg = DrawSampleDialog(pop['subject'] if pop else self.subject, line,
                               pop, existing_note=note, parent=self,
                               error=err)
        self._draw_dialog = dlg
        if self._exec_dialog(dlg) != QtWidgets.QDialog.Accepted or pop is None:
            return
        size, seed = dlg.size_spin.value(), dlg.seed_spin.value()
        before = {d['sample_id'] for d in designs}
        try:
            sid = _rsm.draw_review_sample(self.db.conn, scope=scope,
                                          n_total=size, seed=seed)
        except ValueError as e:
            self.status_bar.showMessage(f"Sample not drawn: {e}")
            return
        design = next(d for d in _sr.samples_for_scope(self.db.conn, scope)
                      if d['sample_id'] == sid)
        self._load_sample(design)
        self._sample['scope'] = scope
        n = len(self._sample['rows'])
        groups = len({(r['region'], r['stage'])
                      for r in self._sample['rows'].values()})
        self._start_sample()
        self.status_bar.showMessage(
            f"Drew {n} events across {groups} region × stage groups "
            f"(seed {seed})." if sid not in before else
            f"This size and seed give the sample drawn on "
            f"{str(design['drawn_at'])[:10]}; reopened it.")

    def _open_report(self):
        if self.db is None:
            return
        if self._sample is None:
            self._refresh_sample_bar()
        if self._sample is None:
            self.status_bar.showMessage('No review sample for this run yet.')
            return
        src = _ReportSource(self)
        dlg = PrecisionReportDialog(src, self)
        dlg.eventRequested.connect(self._on_report_event)
        self._report = dlg
        dlg.show()
        return dlg

    def _refresh_report(self):
        dlg = getattr(self, '_report', None)
        if dlg is not None and dlg.isVisible():
            dlg.refresh()

    def _on_report_event(self, uuid):
        if self._sample and uuid in self._sample['rows']:
            self.tabs.setCurrentIndex(1)
            self._goto_sample_event(uuid, move_cursor=False)

    def _on_bar_show_others(self, on):
        if self.act_show_others.isChecked() != bool(on):
            self.act_show_others.setChecked(bool(on))
        if self.sample_bar.others_chk.isChecked() != \
                self.act_show_others.isChecked():
            self.sample_bar.others_chk.blockSignals(True)
            self.sample_bar.others_chk.setChecked(
                self.act_show_others.isChecked())
            self.sample_bar.others_chk.blockSignals(False)

    # ---- navigation --------------------------------------------------------
    def _nav_unreviewed(self, step):
        self._disarm(cancel_message=False)
        if self._sample_active:
            self._sample_nav(step)
            return
        ep = self.epochs_panel
        u = ep.next_unreviewed() if step > 0 else ep.prev_unreviewed()
        if ep.last_nav == 'wrapped':
            self.status_bar.showMessage('Wrapped to the start of the night.')
        elif u is None and ep._channel is not None and not ep._ev.empty:
            evt = ep._event_type
            self.status_bar.showMessage(
                f"All {int(ep._ev['uuid'].notna().sum())} "
                f"{_er.EVENT_PLURAL.get(evt, evt)} on {ep._channel} reviewed "
                f"by {self.reviewer_name or 'you'}. Pick the next channel in "
                f"the Channels tab.")

    def _nav_any(self, step):
        self._disarm(cancel_message=False)
        ep = self.epochs_panel
        if step > 0:
            ep.next_event()
        else:
            ep.prev_event()

    def _on_auto_advance(self, on):
        self._auto_advance = bool(on)
        _review_settings().setValue('review/auto_advance', bool(on))

    # ---- population checks ----------------------------------------------
    def _population_key(self, evt, methods, freq_band):
        """Cache key: ``(path, evt, methods, band, stamp)``.

        ``stamp`` is SQLite's ``PRAGMA data_version`` on the GUI's connection,
        which changes when ANOTHER connection commits (a re-detection or a
        backfill) and not on this GUI's own review / artefact writes. File
        mtime and size would change on every decision and throw the cache
        away mid-review.
        """
        path = getattr(self.db, 'db_path', None)
        conn = getattr(self.db, 'conn', None)
        try:
            stamp = conn.execute("PRAGMA data_version").fetchone()[0]
        except Exception:
            stamp = None
        return (path, str(evt), tuple(methods or ()),
                tuple(freq_band) if freq_band else None, id(conn), stamp)

    def _apply_population(self, qc, evt, methods, freq_band):
        """Merge cached population checks into ``qc`` (in place copy), or
        start the background read. Returns the merged frame."""
        key = self._population_key(evt, methods, freq_band)
        cache = self.__dict__.setdefault('_pop_cache', {})
        res = cache.get(key)
        if res is None:
            self._start_population(key, evt, methods, freq_band)
            self._pop_view = (key, None)
            self.qc_widget.set_checks_context(False, event_type=evt,
                                              pending=True)
            self.detail_dock_w.set_checks_context(
                True, event_type=evt, caption='Reading event checks…')
            return qc
        return self._merge_population(qc, evt, key, res)

    def _merge_population(self, qc, evt, key, res):
        run = res.get('run') or {}
        method = str(run.get('method') or '')
        ratio = method_has_ratio(method)
        pop, med = _er.population_flags(
            res['pop'], hard_z=self._qc_thresholds['hard_z'],
            soft_z=self._qc_thresholds['soft_z'], event_type=evt,
            ratio_allowed=ratio)
        self._pop_view = (key, {'res': res, 'medians': med, 'pop': pop})
        out = qc.drop(columns=[c for c in pop.columns
                               if c != 'channel' and c in qc.columns])
        out = out.merge(pop, on='channel', how='left') if len(pop) else out
        if 'checks_flag' not in out.columns:
            out['checks_flag'] = ''
        out['checks_flag'] = out['checks_flag'].fillna('')
        params = run.get('params') or {}
        band = params.get('frequency')
        bounds = _er.run_duration_bounds(run, method)
        tips = _er.header_tooltips(
            tuple(band) if band else None,
            bounds[0] if bounds else None,
            int(pop['n_no_peak'].sum()) if len(pop) else None)
        recorded = bool(res.get('recorded'))
        self.qc_widget.set_checks_context(recorded, method, ratio, evt, tips,
                                          figures_off=bool(res.get('figures_off')))
        caption = ''
        if not recorded:
            caption = (_er.FIGURES_OFF_RUN if res.get('figures_off')
                       else _er.NOT_RECORDED_RUN)
        elif len(res.get('runs_in_view') or []) > 1:
            k = len(res['runs_in_view']) - 1
            caption = (f"Event checks from {_er.run_label(run, res['run_id'])}"
                       f"; {k} other run{'s' if k > 1 else ''} in view "
                       f"{'are' if k > 1 else 'is'} not included.")
        self.detail_dock_w.set_checks_context(recorded, evt, ratio, method,
                                              caption)
        return out

    def _start_population(self, key, evt, methods, freq_band):
        if self._pop_worker is not None and self._pop_worker.key == key:
            return
        path = getattr(self.db, 'db_path', None)
        if not path:
            return
        w = _PopulationWorker(key, path, evt, methods, freq_band, self)
        w.done.connect(self._on_population_ready)
        self._pop_worker = w
        w.start()

    def _on_population_ready(self, key, res):
        conn = getattr(self.db, 'conn', None)
        if key[4] != id(conn):        # a database opened since the read began
            if self._pop_worker is not None and self._pop_worker.key == key:
                self._pop_worker = None
            return
        self.__dict__.setdefault('_pop_cache', {})[key] = res
        if self._pop_worker is not None and self._pop_worker.key == key:
            self._pop_worker = None
        if res.get('error'):
            logger.warning(f"Event checks could not be read: {res['error']}")
        view = getattr(self, '_pop_view', None)
        if view is None or view[0] != key or getattr(self, '_qc_df', None) is None:
            return
        qc = self._merge_population(self._qc_df, key[1], key, res)
        self._qc_df = qc
        self.qc_widget.set_data(qc, self._qc_events_df,
                                self.qc_widget._verdicts, self._redetect_queue)
        self.detail_dock_w.update_topo(qc)
        ch = getattr(self, '_qc_selected_channel', None)
        if ch:
            self._update_check_line(ch)

    def wait_population(self, timeout_s=60.0):
        """Block until the background population read finishes (tests)."""
        import time
        t_end = time.time() + timeout_s
        while self._pop_worker is not None and time.time() < t_end:
            self._pop_worker.wait(50)
            QApplication.processEvents()
        QApplication.processEvents()
        return self._pop_worker is None

    def _update_check_line(self, ch):
        view = getattr(self, '_pop_view', None)
        qc = getattr(self, '_qc_df', None)
        if view is None or view[1] is None or qc is None:
            self.detail_dock_w.set_check_line(None, [])
            return
        recorded = bool(view[1]['res'].get('recorded'))
        hit = qc[qc['channel'] == ch]
        items = (_er.dock_check_items(hit.iloc[0].to_dict(), view[0][1],
                                      view[1]['medians'])
                 if recorded and len(hit) else [])
        self.detail_dock_w.set_check_line(ch, items, recorded)

    def _on_check_link(self, ch, col):
        """Dock phrase link: drill the channel and filter its events to
        those failing that check (spec section 1)."""
        self.on_qc_drill(ch, switch_tab=True)
        ep = self.epochs_panel
        df = ep._df
        if df is None or not len(df) or 'uuid' not in df.columns:
            return
        view = getattr(self, '_pop_view', None)
        med = (view[1]['medians'].get(col) if view and view[1] else None)
        if col in _er.RATIO_COLUMNS and (med is None or med != med):
            return
        mask = _er.failing_mask(df, col, med)
        uuids = df.loc[mask.fillna(False), 'uuid'].astype(str).tolist()
        evt = ep._event_type
        ep.set_check_filter(col, _er.filter_chip_text(
            col, evt, len(uuids), int(df['uuid'].notna().sum()), med), uuids)

    def _on_check_filter_cleared(self):
        self.status_bar.showMessage('Event filter removed.')

    # ---- data readers for the Epochs tab ----------------------------------
    def _read_eeg_channels(self, channels, t0, t1):
        """One multi-channel read: ``(times, data[n, t], sfreq, labels)``
        for the channels the file has, or ``None``."""
        if self.eeg_data is None or not channels:
            return None
        t0 = max(0.0, float(t0))
        t1 = float(t1)
        names = self._eeg_channel_info()[0]
        chans = [c for c in dict.fromkeys(channels) if c in names]
        if not chans:
            return None
        try:
            if hasattr(self.eeg_data, 'read_data'):
                wf = self.eeg_data.read_data(chan=chans, begtime=t0,
                                             endtime=t1)
                arr = np.asarray(wf.data[0], dtype=float)
                if arr.ndim == 1:
                    arr = arr[None, :]
                sf = float(getattr(wf, 's_freq', 0) or 0) or 500.0
                try:
                    ts = np.asarray(wf.axis['time'][0], dtype=float)
                    if ts.size != arr.shape[1]:
                        raise ValueError
                except Exception:
                    ts = t0 + np.arange(arr.shape[1]) / sf
                try:
                    got = [str(c) for c in wf.axis['chan'][0]]
                except Exception:
                    got = chans
                return ts, arr, sf, got
            if hasattr(self.eeg_data, 'get_data'):
                sf = float(self.eeg_data.info['sfreq'])
                allc = list(self.eeg_data.ch_names)
                i0, i1 = int(t0 * sf), int(t1 * sf)
                arr = np.asarray(self.eeg_data.get_data(
                    picks=[allc.index(c) for c in chans], start=i0, stop=i1),
                    dtype=float)
                return t0 + np.arange(arr.shape[1]) / sf, arr, sf, chans
        except Exception as err:
            logger.warning(f"Could not read {len(chans)} channel(s) ({err}).")
            return None
        return None

    def _neighbour_provider(self, target, window_s):
        """``(channels, header, labels)`` for the Neighbours group."""
        names, types, interp = self._eeg_channel_info()
        eeg = ChannelTypeSummary(names, types).eeg
        coords = self.detail_dock_w._coords or {}
        chans, source, region = neighbour_channels(
            target, eeg, coords, region_of=lambda c: _region_for_channel(c),
            selected=self.selected_channels, k=6)
        labels = [('~' + c) if c in interp else c for c in [target] + chans]
        header = _er.neighbour_header(labels[0], chans, source, window_s,
                                      region)
        return chans, header, labels

    def _neighbour_events(self, chans, t0, t1):
        if self.db is None:
            return None
        methods, freq_band = self._current_method_freq()
        return self.db.neighbour_events(chans, self.epochs_panel._event_type,
                                        t0, t1, methods, freq_band)

    def _refresh_physio_channels(self):
        names, types, _interp = self._eeg_channel_info()
        self.epochs_panel.physio.set_channels(physio_channels(names, types))

    def _recording_seconds(self):
        """Recording length: the annotation file's (``recording_seconds()``,
        i.e. Wonambi's ``last_second``), else the last event end, else 8 h."""
        try:
            rec = annotation_recording_seconds(self.annotations)
            if rec:
                return float(rec)
        except Exception:
            pass
        df = getattr(self, '_qc_events_df', None)
        if df is not None and len(df) and 'end_time' in df.columns:
            try:
                return float(pd.to_numeric(df['end_time'],
                                           errors='coerce').max())
            except Exception:
                pass
        return 8 * 3600.0

    def _epoch_table(self):
        """:class:`EpochTable` of the loaded annotations (true per-epoch
        start/end, 1-30 s on cut recordings), or ``None`` without staging.
        Cached per annotation object; ``open_annotation_file`` replaces the
        object, which invalidates it."""
        ann = self.annotations
        if ann is None:
            return None
        cached = getattr(self, '_epoch_table_cache', None)
        if cached is not None and cached[0] is ann:
            return cached[1]
        try:
            table = EpochTable.from_annotations(ann)
        except Exception as err:
            logger.warning(f"Could not read the scored epochs from the "
                           f"annotation file ({err}); staging is not shown.")
            table = None
        self._epoch_table_cache = (ann, table)
        return table

    def _marked_for(self, ch):
        """Channel-scoped artefact intervals (evidence_channel == ch)."""
        if self.db is None:
            return []
        try:
            iv = self.db.get_qc_artefact_intervals()
        except Exception:
            return []
        if iv is None or len(iv) == 0:
            return []
        sub = iv[iv['evidence_channel'].astype(str) == str(ch)]
        return sub.to_dict('records')

    def _total_marked(self):
        """Total qc_artefact_intervals across all channels (dock header)."""
        if self.db is None:
            return 0
        try:
            iv = self.db.get_qc_artefact_intervals()
            return 0 if iv is None else int(len(iv))
        except Exception:
            return 0

    def _read_eeg_window(self, channel, t0, t1):
        """Synchronous ±window single-channel read for the scrub trace.
        Returns (times, data, sfreq) or None (no EEG / unreadable)."""
        if self.eeg_data is None or channel is None:
            return None
        t0 = max(0.0, float(t0))
        t1 = float(t1)
        try:
            if hasattr(self.eeg_data, 'read_data'):
                wf = self.eeg_data.read_data(chan=[channel],
                                             begtime=t0, endtime=t1)
                arr = np.asarray(wf.data[0], dtype=float)
                while arr.ndim > 1:
                    arr = arr[0]
                sf = float(getattr(wf, 's_freq', 0) or 0) or 500.0
                try:
                    ts = np.asarray(wf.axis['time'][0], dtype=float)
                    if ts.size != arr.size:
                        raise ValueError
                except Exception:
                    ts = t0 + np.arange(arr.size) / sf
                return ts, arr, sf
            if hasattr(self.eeg_data, 'get_data'):
                sf = float(self.eeg_data.info['sfreq'])
                names = list(self.eeg_data.ch_names)
                if channel not in names:
                    return None
                i0, i1 = int(t0 * sf), int(t1 * sf)
                data = self.eeg_data.get_data(
                    picks=[names.index(channel)], start=i0, stop=i1)
                arr = np.asarray(data, dtype=float).ravel()
                ts = t0 + np.arange(arr.size) / sf
                return ts, arr, sf
        except Exception:
            return None
        return None

    def _mark_channel_artefact(self, ch, s, e):
        if self.db is None:
            return
        self.db.add_qc_artefact_interval(s, e, ch, self.reviewer_name)
        ok = self._write_review_qc_sidecar()
        self.status_bar.showMessage(
            f"{ch}: artefact {_hms(s)}–{_hms(e)} marked"
            + (f" · sidecar {os.path.basename(self._review_qc_sidecar)}"
               if ok else " · (DB only — no XML loaded)"))
        self.refresh_qc_dashboard()
        self.on_qc_drill(ch, switch_tab=False)
        self.detail_dock_w.set_marked(self._marked_for(ch),
                                      channel=ch, total=self._total_marked())

    def _unmark_artefact(self, interval_id):
        if self.db is None:
            return
        self.db.remove_qc_artefact_interval(int(interval_id))
        self._write_review_qc_sidecar()
        self.refresh_qc_dashboard()
        if getattr(self, '_qc_selected_channel', None):
            self.on_qc_drill(self._qc_selected_channel, switch_tab=False)
            self.detail_dock_w.set_marked(
                self._marked_for(self._qc_selected_channel),
                channel=self._qc_selected_channel, total=self._total_marked())
        self.status_bar.showMessage(f"Artefact range {interval_id} unmarked")

    def _write_review_qc_sidecar(self):
        """Write all qc_artefact_intervals into a SIDECAR Wonambi XML whose
        rater is literally 'review-qc'; back the original up to *.xml.bak on
        first write. The loaded sleep-scorer XML is never modified."""
        if not getattr(self, 'annot_file_path', None) or \
                not os.path.exists(self.annot_file_path):
            return False
        import shutil
        src = self.annot_file_path
        stem, _ext = os.path.splitext(src)
        sidecar = stem + "_review-qc.xml"
        bak = src + ".bak"
        try:
            if not os.path.exists(bak):
                shutil.copy2(src, bak)
            from wonambi.attr import Annotations as WAnn
            shutil.copy2(src, sidecar)
            ann = WAnn(sidecar)
            try:
                if getattr(ann, 'rater', None) is None and ann.raters:
                    ann.get_rater(ann.raters[0])
                ann.add_rater('review-qc')
            except Exception:
                pass
            try:
                ann.add_event_type('Artefact')
            except Exception:
                pass
            iv = self.db.get_qc_artefact_intervals()
            for _, r in iv.iterrows():
                chan = str(r.get('evidence_channel') or '(all)')
                try:
                    ann.add_event('Artefact',
                                  (float(r['start_time']),
                                   float(r['end_time'])), chan=chan)
                except Exception:
                    pass
            ann.save() if hasattr(ann, 'save') else ann.export(sidecar)
            self._review_qc_sidecar = sidecar
            return True
        except Exception as ex:
            self.status_bar.showMessage(f"Sidecar write failed: {ex}")
            return False





    def on_qc_verdict_changed(self):
        ch, v = getattr(self.qc_widget, 'verdictChangedTo', (None, None))
        if ch is None:
            return
        evt = self.qc_widget.current_event_type()
        self.db.set_channel_verdict(ch, evt, v, self.reviewer_name)
        self.status_bar.showMessage(
            f"{ch}: {'artefact cleared' if v == '' else v} ({evt})")
        self.refresh_qc_dashboard()

    def _qc_unmark_artefact(self, ch):
        """Tray chip ✕ (or button): clear the channel-artefact verdict back to
        '' (untriaged). No sidecar XML is written for channel marks, so there
        is nothing in XML to reverse. Single refresh updates State + tray +
        counts + filter-dock glyphs."""
        evt = self.qc_widget.current_event_type()
        self.db.set_channel_verdict(str(ch), evt, '', self.reviewer_name)
        self.status_bar.showMessage(f"{ch}: artefact cleared ({evt})")
        self.refresh_qc_dashboard()

    def _qc_remove_redetect(self, ch):
        """Tray chip ✕: drop the channel from the re-detect queue."""
        self._redetect_queue.discard(str(ch))
        self.status_bar.showMessage(
            f"Re-detect queue: {len(self._redetect_queue)} channel(s)")
        self.refresh_qc_dashboard()

    def on_qc_add_redetect(self, ch):
        if ch in self._redetect_queue:
            self._redetect_queue.discard(ch)
        else:
            self._redetect_queue.add(ch)
        self.status_bar.showMessage(
            f"Re-detect queue: {len(self._redetect_queue)} channel(s)")
        self.refresh_qc_dashboard()

    def on_qc_queue_all_hard(self):
        df = getattr(self, '_qc_df', None)
        if df is None or len(df) == 0:
            return
        for ch in df.loc[df['flag'] == 'hard', 'channel']:
            self._redetect_queue.add(str(ch))
        self.status_bar.showMessage(
            f"Re-detect queue: {len(self._redetect_queue)} channel(s)")
        self.refresh_qc_dashboard()

    def _drop_channel(self, ch):
        evt = self.qc_widget.current_event_type()
        self.db.set_channel_verdict(ch, evt, 'drop', self.reviewer_name)
        self.status_bar.showMessage(f"{ch} dropped ({evt})")
        self.refresh_qc_dashboard()

    def _add_global_artefact(self, s, e, ch):
        self.db.add_qc_artefact_interval(s, e, ch, self.reviewer_name)
        self.status_bar.showMessage(
            f"Global artefact {s:.1f}-{e:.1f}s recorded (whole-montage)")
        self.refresh_qc_dashboard()


    def _coords_from_set(self):
        """Read EEGLAB chanlocs from the loaded .set and convert the polar
        theta/radius to 2-D topoplot coordinates. Returns {label:(x,y)} (empty
        when there is no .set or it carries no chanlocs)."""
        path = getattr(self, 'eeg_file_path', None)
        if not path or not str(path).lower().endswith('.set'):
            return {}
        try:
            from turtlewave_hdEEG.dataset import read_eeglab_chanlocs
            chanlocs = read_eeglab_chanlocs(path)
        except Exception:
            return {}
        return _eeglab_polar_to_xy(chanlocs)

    def _autoload_set_coords(self):
        """Auto-attempt topo coords from the loaded .set (no dialog). Silent
        when the recording has no chanlocs — the empty-state card + the
        Load montage… button remain in place."""
        coords = self._coords_from_set()
        if coords:
            self.detail_dock_w.set_coords(coords)
            # Region column is coordinate-based once a montage is loaded.
            if self.db is not None:
                self.refresh_qc_dashboard()
            self.status_bar.showMessage(
                f"Topography live: {len(coords)} channel coordinates from .set")

    def on_load_montage(self):
        """Populate topo coords. Prefers chanlocs from the loaded EEGLAB .set;
        falls back to a 'label,x,y' CSV."""
        coords = self._coords_from_set()
        source = "EEGLAB .set chanlocs"
        if not coords:
            fp, _ = QFileDialog.getOpenFileName(
                self, "Load montage CSV (label,x,y)", "",
                "CSV (*.csv);;All Files (*)")
            if not fp:
                return
            try:
                m = pd.read_csv(fp)
                cols = [c.lower() for c in m.columns]
                m.columns = cols
                coords = {str(r[cols[0]]): (float(r['x']), float(r['y']))
                          for _, r in m.iterrows()}
                source = os.path.basename(fp)
            except Exception as ex:
                QtWidgets.QMessageBox.warning(self, "Montage", f"Could not parse: {ex}")
                return
        if not coords:
            QtWidgets.QMessageBox.information(
                self, "Montage", "No channel coordinates found in the .set.")
            return
        self.detail_dock_w.set_coords(coords)
        # Region column is coordinate-based once a montage is loaded.
        if self.db is not None:
            self.refresh_qc_dashboard()
        self.status_bar.showMessage(
            f"Coordinates loaded from {source}: {len(coords)} channels")

    # ------------------------------------------------------------------
    # Dialogs / modals
    # ------------------------------------------------------------------
    def open_outlier_threshold_dialog(self):
        """Tune hard_z / soft_z / dead_frac (View → Outlier threshold…)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Outlier threshold")
        form = QtWidgets.QFormLayout(dlg)
        hz = QtWidgets.QDoubleSpinBox()
        hz.setRange(1.0, 12.0); hz.setSingleStep(0.5)
        hz.setValue(self._qc_thresholds['hard_z'])
        sz = QtWidgets.QDoubleSpinBox()
        sz.setRange(0.5, 10.0); sz.setSingleStep(0.5)
        sz.setValue(self._qc_thresholds['soft_z'])
        dz = QtWidgets.QDoubleSpinBox()
        dz.setRange(0.0, 1.0); dz.setSingleStep(0.05)
        dz.setValue(self._qc_thresholds['dead_frac'])
        form.addRow("hard |z| >", hz)
        form.addRow("soft |z| >", sz)
        form.addRow("dead: n < frac·median", dz)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._qc_thresholds = dict(
                hard_z=hz.value(), soft_z=sz.value(), dead_frac=dz.value())
            self.refresh_qc_dashboard()

    def open_design_notes(self):
        """Help → Design notes: the redesign thesis (read-only)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Design notes · eeg_review_gui redesign")
        dlg.resize(640, 520)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(
            "<h2>Design notes — QC-by-outlier-triage</h2>"
            "<p><b>Thesis.</b> The review GUI's job is to spot outlier / "
            "impossible-physiology events and exclude bad epochs / channels, "
            "then re-detect — not to accept/reject individual events. Two "
            "surfaces, plus a right dock with live topography and a global "
            "worst-events ranking.</p>"
            "<ul>"
            "<li><b>Channels (QC)</b> — sortable per-channel aggregates + "
            "robust-outlier flags; the landing triage surface.</li>"
            "<li><b>Epochs</b> — per-channel amplitude strip + raw/filtered "
            "trace; brush a time range to mark an artefact (written to a "
            "sidecar XML under rater <code>review-qc</code>; original backed "
            "up to <code>*.xml.bak</code>).</li>"
            "</ul>"
            "<p>The right dock carries a live scalp topography of the active "
            "QC metric (from the EEGLAB <code>.set</code> chanlocs) and a "
            "read-only worst-events list across all channels — click a row to "
            "drill into that channel and epoch. Re-detection is a "
            "one-directional JSON hand-off to turtlewave_gui — this GUI never "
            "runs detection.</p>")
        lay.addWidget(te)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        bb.clicked.connect(lambda *_: dlg.accept())
        lay.addWidget(bb)
        dlg.exec_()

    def _dropped_channels(self, verdicts=None):
        """Channels excluded from analysis entirely (drop / channel_artefact
        verdict, any event type)."""
        if verdicts is None:
            verdicts = self.db.get_channel_verdicts() if self.db else {}
        return {str(c) for (c, _e), v in verdicts.items()
                if v in ('drop', 'channel_artefact')}

    def _redetect_queue_channels(self, verdicts=None):
        """The reviewer-selected re-detect queue as a sorted, deduped list,
        MINUS any dropped channel. A dropped channel is excluded from analysis
        entirely, so it must never land in the re-detect list even if it was
        also queued — the P3 driver's --channels consumes exactly this list to
        replace only these channels' events."""
        dropped = self._dropped_channels(verdicts)
        return sorted(set(map(str, self._redetect_queue)) - dropped)

    def _build_redetect_request(self):
        """Assemble the schema-v1 re-detect request dict."""
        verdicts = self.db.get_channel_verdicts() if self.db else {}
        # Queue-minus-dropped: what the driver actually re-detects.
        redetect = self._redetect_queue_channels(verdicts)
        # exclude_channels stays queue ∪ drops (dropped channels ARE excluded).
        excl = sorted(set(map(str, self._redetect_queue)) |
                      self._dropped_channels(verdicts))
        epochs = []
        if self.db is not None:
            try:
                iv = self.db.get_qc_artefact_intervals()
                for _, r in iv.iterrows():
                    epochs.append(dict(
                        channel=str(r.get('evidence_channel') or '(all)'),
                        t0=round(float(r['start_time']), 1),
                        t1=round(float(r['end_time']), 1)))
            except Exception:
                pass
        evt = self.qc_widget.current_event_type()
        fb = (0.5, 2.0)
        if evt == 'spindle':
            fb = (11.0, 16.0)
        elif evt == 'k_complex':
            fb = (0.5, 1.5)
        return {
            'schema_version': 1,
            'subject': self.subject,
            'source_db': getattr(self.db, 'db_path', None),
            'source_xml': getattr(self, 'annot_file_path', None),
            'eeg_file': getattr(self, 'eeg_file_path', None),
            'method': (self.method_combo.currentText()
                       if self.method_combo.currentIndex() > 0
                       else 'Wamsley2012'),
            'freq_band': list(fb),
            'stages': ['NREM2', 'NREM3'],
            'event_types': [evt],
            'exclude_channels': excl,
            'redetect_channels': redetect,
            'exclude_epochs': epochs,
            'requested_at': datetime.now().isoformat(timespec='seconds'),
            'requested_by': self.reviewer_name,
        }

    def open_redetect_modal(self):
        """Full JSON-preview modal: Save writes redetect_request.json beside
        the XML for turtlewave_gui to pick up (one-directional hand-off; this
        GUI never runs detection)."""
        if self.db is None:
            QtWidgets.QMessageBox.warning(
                self, "Re-detect", "Load a database first.")
            return
        import json
        req = self._build_redetect_request()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Build re-detect request")
        dlg.resize(640, 560)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "This GUI does not run detection. It writes "
            "<b>redetect_request.json</b> beside the XML; turtlewave_gui "
            "picks it up."))
        lay.addWidget(_h_label(
            f"CHANNELS TO EXCLUDE / RE-DETECT ({len(req['exclude_channels'])})"))
        chips = QLabel(", ".join(req['exclude_channels']) or "(none)")
        chips.setWordWrap(True)
        chips.setStyleSheet("font-family:'IBM Plex Mono',monospace;"
                            "color:#d6dee8;")
        lay.addWidget(chips)
        lay.addWidget(_h_label(
            f"SELECTED FOR RE-DETECT ({len(req['redetect_channels'])})"))
        rd_chips = QLabel(", ".join(req['redetect_channels'])
                          or "(nothing queued)")
        rd_chips.setWordWrap(True)
        rd_chips.setStyleSheet("font-family:'IBM Plex Mono',monospace;"
                               "color:#d6dee8;")
        lay.addWidget(rd_chips)
        lay.addWidget(_h_label(
            f"ARTEFACT EPOCH RANGES ({len(req['exclude_epochs'])})"))
        lay.addWidget(_h_label("JSON PREVIEW"))
        te = QTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("font-family:'IBM Plex Mono',monospace;"
                         "font-size:11px;")
        te.setPlainText(json.dumps(req, indent=2))
        lay.addWidget(te)
        bb = QtWidgets.QDialogButtonBox()
        cancel = bb.addButton("Cancel", QtWidgets.QDialogButtonBox.RejectRole)
        save = bb.addButton("Save request",
                            QtWidgets.QDialogButtonBox.AcceptRole)
        save.setObjectName("primary")
        cancel.clicked.connect(dlg.reject)
        save.clicked.connect(
            lambda: (self._write_redetect_json(req), dlg.accept()))
        lay.addWidget(bb)
        dlg.exec_()

    def _write_redetect_json(self, req):
        """Write redetect_request.json beside the XML (or the DB). Purely a
        one-directional hand-off — turtlewave_gui picks the file up; this GUI
        never launches it."""
        import json
        base = (os.path.dirname(self.annot_file_path)
                if getattr(self, 'annot_file_path', None)
                else (os.path.dirname(self.db.db_path)
                      if self.db else os.getcwd()))
        out = os.path.join(base, "redetect_request.json")
        try:
            with open(out, 'w', encoding='utf-8') as fh:
                json.dump(req, fh, indent=2)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Re-detect", str(ex))
            return
        self.status_bar.showMessage(f"Re-detect request written: {out}")

    
    
    def setup_status_bar(self):
        """Segmented status bar: subject · hard · marked · ranges · queue ·
        build re-detect."""
        self.status_bar = self.statusBar()
        # first permanent segment: who decisions are saved under
        self.seg_reviewer = QPushButton("Reviewer: not set")
        self.seg_reviewer.setFlat(True)
        self.seg_reviewer.setFocusPolicy(Qt.NoFocus)
        self.seg_reviewer.setStyleSheet("padding:0 8px;")
        self.seg_reviewer.clicked.connect(self._prompt_reviewer_name)
        self.status_bar.addPermanentWidget(self.seg_reviewer)

        def _seg(text):
            q = QLabel(text)
            q.setStyleSheet("padding:0 8px;border-left:1px solid #262d39;")
            self.status_bar.addPermanentWidget(q)
            return q

        self.seg_subject = _seg("—")
        self.seg_hard = _seg("0 hard outliers")
        self.seg_marked = _seg("0 channels marked artefact")
        self.seg_ranges = _seg("0 artefact ranges")
        self.seg_queue = _seg("re-detect queue: 0")
        self.btn_build_redetect = QPushButton("Build re-detect request…")
        self.btn_build_redetect.setEnabled(False)
        self.btn_build_redetect.clicked.connect(self.open_redetect_modal)
        self.status_bar.addPermanentWidget(self.btn_build_redetect)
        # (reviewer name is provenance-only — still written to the DB, never
        # surfaced in the UI.)
        # legacy sinks (open_database / review_event still call .setText on
        # these; keep them off the bar so the text is harmlessly absorbed)
        self.db_size_label = QLabel()
        self.last_saved_label = QLabel()
        self.status_bar.showMessage("Ready — Open database to begin")

    def _refresh_status_segments(self):
        self.seg_subject.setText(self.subject)
        evt = self.qc_widget.current_event_type()
        df = getattr(self, '_qc_df', None)
        hard = 0 if df is None or len(df) == 0 else int((df['flag'] == 'hard').sum())
        self.seg_hard.setText(f"{hard} hard outliers")
        marked = 0
        if self.db is not None:
            marked = len({c for (c, e2), v in
                          self.db.get_channel_verdicts().items()
                          if v in ('drop', 'channel_artefact')})
        self.seg_marked.setText(
            f"{marked} channel{'' if marked == 1 else 's'} marked artefact")
        nranges = 0
        if self.db is not None:
            try:
                nranges = len(self.db.get_qc_artefact_intervals())
            except Exception:
                nranges = 0
        self.seg_ranges.setText(
            f"{nranges} artefact range{'' if nranges == 1 else 's'}")
        nq = len(self._redetect_queue)
        self.seg_queue.setText(f"re-detect queue: {nq}")
        self.btn_build_redetect.setEnabled(nq > 0 or nranges > 0)

    def setup_keyboard_shortcuts(self):
        """F flags the selected QC channel for re-detect; the review keys
        are window shortcuts live on the Epochs tab only."""
        QShortcut(QtGui.QKeySequence('F'), self, self._flag_selected_qc_row)
        self._setup_review_shortcuts()

    def _flag_selected_qc_row(self):
        """Toggle the currently-selected QC channel in the re-detect queue
        (the ↻ flag). No-op when no channel is selected."""
        ch = self.qc_widget._current_channel()
        if ch:
            self.on_qc_add_redetect(ch)

    # ========================================================================
    # Data Loading
    # ========================================================================
    
    def open_database(self):
        """Open database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Events Database", "",
            "Database Files (*.db *.sqlite);;All Files (*)"
        )
        
        if file_path:
            try:
                self.db = EventDatabase(file_path)
                self._undo_stack = []
                self.status_bar.showMessage("Database loaded successfully - Select channels to load events")
                
                # Get database size
                db_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                self.db_size_label.setText(f"DB: {db_size_mb:.1f} MB")

                # Populate filter options + global channel list. With no EEG
                # file loaded, the database's channels decide the defaults.
                self.populate_filter_options()
                if self.eeg_data is None:
                    self._apply_default_channels(self._all_db_channels())
                self.load_channels()

                # QC reframe: land on the per-channel dashboard
                self.refresh_qc_dashboard()
                self._refresh_chrome()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load database: {str(e)}")

    def _refresh_chrome(self):
        """Re-derive subject and repaint title / toolbar / status segments."""
        self.subject = self._derive_subject()
        self._setWindowTitleFromSubject()
        self._refresh_toolbar_state()
        self._refresh_status_segments()
    
    def open_eeg_file(self):
        """Ask for an EEG file and load it (see ``load_eeg_file``)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EEG File", "",
            "EEG Files (*.set *.edf *.bdf *.fif);;All Files (*)"
        )
        if file_path:
            self.load_eeg_file(file_path)

    def _read_with_mne(self, file_path):
        """MNE reader for ``file_path`` by extension."""
        if mne is None:
            raise RuntimeError("MNE is not installed")
        readers = {'.set': mne.io.read_raw_eeglab, '.edf': mne.io.read_raw_edf,
                   '.bdf': mne.io.read_raw_bdf, '.fif': mne.io.read_raw_fif}
        reader = readers.get(os.path.splitext(file_path)[1].lower())
        if reader is None:
            raise RuntimeError("MNE has no reader for this file type")
        return reader(file_path, preload=False)

    def load_eeg_file(self, file_path):
        """Load an EEG file: TurtleWave's reader first, then MNE.

        When both fail the dialog explains TurtleWave's reason (not MNE's),
        in the wording of ``channel_types.load_failure_message``; the full
        errors go to the log.
        """
        import traceback
        self.status_bar.showMessage("Loading EEG file...")
        try:
            self.eeg_data = LargeDataset(file_path, create_memmap=False)
            self.eeg_file_path = file_path
            self.status_bar.showMessage(f"EEG file loaded: {os.path.basename(file_path)}")
        except Exception as tw_error:
            reason = str(tw_error).strip().splitlines()[0] if str(tw_error).strip() \
                else type(tw_error).__name__
            logger.warning(f"TurtleWave could not open this file ({reason}); "
                           f"trying MNE instead.")
            logger.debug(traceback.format_exc())
            try:
                self.eeg_data = self._read_with_mne(file_path)
                self.eeg_file_path = file_path
                self.status_bar.showMessage(
                    f"EEG file loaded (MNE): {os.path.basename(file_path)}")
            except Exception as mne_error:
                logger.error(f"MNE could not open the file either: {mne_error}")
                traceback.print_exception(type(tw_error), tw_error,
                                          tw_error.__traceback__)
                self.status_bar.showMessage("Error")
                QtWidgets.QMessageBox.critical(
                    self, "Error", load_failure_message(
                        file_path, tw_error, where=REVIEW_GUI_ERROR_WHERE))
                return

        try:
            names, types, interp = self._eeg_channel_info()
            defaults_applied = self._apply_default_channels(names, types)
            self.load_channels()
            if defaults_applied:
                note = interpolated_defaults_note(self.selected_channels,
                                                  interp)
                if note:
                    self.status_bar.showMessage(note)
                    logger.info(note)
            self._refresh_physio_channels()
            # Start background waveform loader
            self.start_background_loader()
            self._refresh_chrome()
            # Auto-attempt live topo coords from the EEGLAB .set chanlocs
            self._autoload_set_coords()
        except Exception as e:
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"The recording opened, but its channels could not be set "
                f"up: {first}. The full error is {REVIEW_GUI_ERROR_WHERE}.")
            traceback.print_exc()

    def _eeg_channel_info(self):
        """``(channels, chan_type, interpolated)`` of the loaded EEG file;
        ``([], None, set())`` when none is loaded. MNE's lower-case types go
        through the same non-EEG rule. ``interpolated`` is the set of channel
        names in ``header['interp_channels']`` (empty when the file names
        none, and always for MNE reads)."""
        data = self.eeg_data
        if data is None:
            return [], None, set()
        if hasattr(data, 'channels'):
            header = getattr(data, 'header', None) or {}
            if not hasattr(header, 'get'):
                header = {}
            summary = ChannelTypeSummary([], None,
                                         header.get('interp_channels'))
            return (list(data.channels), header.get('chan_type'),
                    set(summary.interpolated))
        if hasattr(data, 'ch_names'):
            try:
                types = list(data.get_channel_types())
            except Exception:
                types = None
            return list(data.ch_names), types, set()
        return [], None, set()

    def _apply_default_channels(self, channels, chan_type=None):
        """Replace the waveform channels with ``default_review_channels``,
        unless the user has chosen their own and every one of them exists in
        the newly loaded ``channels``. A choice made on a different montage
        is dropped (reading it would give blank waveforms).

        Returns ``True`` when the defaults were applied, ``False`` when the
        user's own selection was kept."""
        names = set(str(c) for c in channels)
        if self._channels_user_set:
            missing = [ch for ch in self.selected_channels if ch not in names]
            if self.selected_channels and not missing:
                logger.info(f"Keeping your channel selection "
                            f"({', '.join(self.selected_channels)}): all "
                            f"present in the new file.")
                return False
            self._channels_user_set = False
            self.selected_channels = default_review_channels(channels, chan_type)
            logger.info(f"Channel selection reset to defaults "
                        f"({', '.join(self.selected_channels) or 'none'}): "
                        f"{', '.join(missing) or 'the selection was empty'} "
                        f"not in the new file.")
            return True
        self.selected_channels = default_review_channels(channels, chan_type)
        return True

    def open_annotation_file(self):
        """Open annotation file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", "",
            "XML Files (*.xml);;All Files (*)"
        )
        
        if file_path:
            try:
                self.annotations = CustomAnnotations(file_path)
                self.annot_file_path = file_path
                
                # Extract recording start time
                if hasattr(self.annotations, 'wonb_annot') and hasattr(self.annotations.wonb_annot, 'start_time'):
                    self.recording_start_time = self.annotations.wonb_annot.start_time

                self.status_bar.showMessage(f"Annotations loaded: {os.path.basename(file_path)}")

                self._refresh_chrome()
                if self.db is not None:
                    self.refresh_qc_dashboard()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load annotations: {str(e)}")
    
    def start_background_loader(self):
        """Start background thread for loading waveforms"""
        if self.background_loader is None:
            from frontend.waveform_loader import WaveformBackgroundLoader
            self.background_loader = WaveformBackgroundLoader(self)
            self.background_loader.waveform_loaded.connect(self.on_waveform_loaded)
            self.background_loader.start()
            print("Background waveform loader started")
    
    def on_waveform_loaded(self, event_uuid, waveform_data):
        """Cache a background-loaded waveform (Epochs drill reads synchronously
        via _read_eeg_window; this just keeps the shared cache warm)."""
        self.cache_lock.lock()
        self.waveform_cache[event_uuid] = waveform_data
        self.cache_lock.unlock()


    
    def load_channels(self):
        """Populate the global filter-dock channel list: the database's
        channels, else the EEG file's EEG channels, else nothing (with a
        status-bar hint)."""
        try:
            channels = self._all_db_channels() if self.db else []
            names, types, interp = self._eeg_channel_info()
            if not channels and self.eeg_data is not None:
                channels = ChannelTypeSummary(names, types).eeg
            self.channel_list.blockSignals(True)
            self.filter_dock.populate_channels(channels,
                                               checked=self.selected_channels)
            # " ~" + tooltip on channels the cleaning pipeline interpolated
            self.filter_dock.decorate_channels(interp_set=interp)
            self.channel_list.blockSignals(False)
            if channels:
                self.status_bar.showMessage(f"Loaded {len(channels)} channels")
            else:
                self.status_bar.showMessage(
                    "No channels yet - open an event database or an EEG file.")
        except Exception as e:
            print(f"Error loading channels: {e}")
            import traceback
            traceback.print_exc()
    
    
    # ========================================================================
    # Navigation
    # ========================================================================
    
    
    


    
    
    
    
    
    
    # ========================================================================
    # Review Actions
    # ========================================================================
    
    
    
    # ========================================================================
    # UI Callbacks
    # ========================================================================
    
    def on_channel_changed(self, item=None):
        """Handle channel-list check change — debounced to avoid freezing."""
        self._channels_user_set = True
        self.selected_channels = []
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            if it.checkState() == Qt.Checked:
                self.selected_channels.append(str(it.data(Qt.UserRole)))
        # Debounce rapid multi-clicks
        self.channel_filter_timer.stop()
        self.channel_filter_timer.start(500)

    def apply_channel_filter(self):
        """Drop the cached waveforms after a channel-list change (debounced).
        The QC dashboard aggregates over all channels, so no reload here."""
        self.cache_lock.lock()
        self.waveform_cache.clear()
        self.cache_lock.unlock()

    def _set_all_channels(self, state):
        self.channel_list.blockSignals(True)
        for i in range(self.channel_list.count()):
            self.channel_list.item(i).setCheckState(state)
        self.channel_list.blockSignals(False)
        self.on_channel_changed()

    def select_all_channels(self):
        """Select all channels"""
        self._set_all_channels(Qt.Checked)

    def deselect_all_channels(self):
        """Deselect all channels"""
        self._set_all_channels(Qt.Unchecked)
    
    def update_event_type_filter(self):
        """Update event type filter"""
        self.selected_event_types = []
        if self.spindle_check.isChecked():
            self.selected_event_types.append('spindle')
        if self.slowwave_check.isChecked():
            self.selected_event_types.append('slow_wave')
        if self.kcomplex_check.isChecked():
            self.selected_event_types.append('k_complex')
        
        # Update method and freq_band filters based on selected event types
        self.populate_filter_options()

        # Refresh the QC dashboard for the new filter selection
        self.refresh_qc_dashboard()
    
    def update_method_filter(self):
        """Method filter changed — refresh Events, QC, dock, and drill."""
        self._refresh_all()

    def update_freq_band_filter(self):
        """Frequency-band filter changed — refresh all surfaces."""
        self._refresh_all()
    
    def populate_filter_options(self):
        """Populate method and frequency band filter options based on current event types"""
        if not self.db:
            return
        
        try:
            # Get selected event types
            event_types = []
            if self.spindle_check.isChecked():
                event_types.append('spindle')
            if self.slowwave_check.isChecked():
                event_types.append('slow_wave')
            if self.kcomplex_check.isChecked():
                event_types.append('k_complex')

            if not event_types:
                return

            # Block combo signals during repopulation — clear()/addItem()/
            # setCurrentIndex() each emit currentIndexChanged, which would
            # storm _refresh_all (measured 4× per checkbox toggle). Those
            # re-fires are spurious here since we drive refresh_qc_dashboard
            # explicitly after repopulating.
            self.method_combo.blockSignals(True)
            self.freq_band_combo.blockSignals(True)
            # Populate method filter
            methods = self.db.get_unique_methods(event_types)
            current_method = self.method_combo.currentText()
            self.method_combo.clear()
            self.method_combo.addItem("All Methods")
            self.method_combo.addItems(methods)
            
            # Restore previous selection if still available
            index = self.method_combo.findText(current_method)
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
            
            # Populate frequency band filter
            freq_bands = self.db.get_unique_freq_bands(event_types)
            current_freq = self.freq_band_combo.currentText()
            self.freq_band_combo.clear()
            self.freq_band_combo.addItem("All Frequencies")
            for lower, upper in freq_bands:
                # Display frequency band with 2 decimal places to show exact values
                # For slow waves: display "0.50-1.25 Hz" (actual database values)
                display_text = f"{lower:.2f}-{upper:.2f} Hz"
                self.freq_band_combo.addItem(display_text)
                # Store the actual frequency values as item data for precise filtering
                self.freq_band_combo.setItemData(self.freq_band_combo.count() - 1, (lower, upper))
            
            # Restore previous selection if still available
            index = self.freq_band_combo.findText(current_freq)
            if index >= 0:
                self.freq_band_combo.setCurrentIndex(index)
                
        except Exception as e:
            print(f"Error populating filter options: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.method_combo.blockSignals(False)
            self.freq_band_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # QC re-run package + figure / summary exports
    # ------------------------------------------------------------------
    def _all_db_channels(self):
        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT DISTINCT channel FROM events WHERE channel IS NOT NULL")
            return sorted(r[0] for r in cur.fetchall())
        except Exception:
            return []

    def export_rerun_package(self):
        """Snapshot originals, then write channels.csv + a sidecar XML the
        existing local detector scripts consume via --annot/--channels.

        Artefacts are appended under the rater the detector auto-selects
        (raters[0]) inside the SIDECAR copy — never the original (R4)."""
        if not self.db:
            QtWidgets.QMessageBox.warning(self, "Warning", "No database loaded")
            return
        if not getattr(self, 'annot_file_path', None) or \
                not os.path.exists(self.annot_file_path):
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Load the base annotation XML first (File → Open Annotation File).")
            return
        root_dir = QFileDialog.getExistingDirectory(
            self, "Select subject root dir (expects ./wonambi/, ./channels.csv)")
        if not root_dir:
            return

        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")

        # dropped channels (any event type) -> excluded from channels.csv
        verdicts = self.db.get_channel_verdicts()
        dropped = sorted({ch for (ch, _et), v in verdicts.items() if v == 'drop'})
        kept = [c for c in self._all_db_channels() if c not in dropped]
        # re-detect queue MINUS any dropped/channel_artefact channel — a dropped
        # channel is excluded from analysis, never re-detected (matches the JSON
        # field so the two hand-off artefacts can't disagree).
        redetect = self._redetect_queue_channels(verdicts)
        intervals = self.db.get_qc_artefact_intervals(unexported_only=True)

        # ---- snapshot originals BEFORE anything can overwrite them --------
        backup = os.path.join(root_dir, "qc_backup", ts)
        snapped = []
        try:
            os.makedirs(backup, exist_ok=True)
            won = os.path.join(root_dir, "wonambi")
            if os.path.isdir(won):
                for name in os.listdir(won):
                    src = os.path.join(won, name)
                    if name.endswith("_results") and os.path.isdir(src):
                        shutil.copytree(src, os.path.join(backup, name),
                                        dirs_exist_ok=True); snapped.append(name)
                    elif name.endswith(".csv"):
                        shutil.copy2(src, os.path.join(backup, name)); snapped.append(name)
            if os.path.exists(self.db.db_path):
                shutil.copy2(self.db.db_path,
                             os.path.join(backup, os.path.basename(self.db.db_path)))
                snapped.append(os.path.basename(self.db.db_path))
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, "Snapshot failed",
                f"Aborting — originals NOT snapshotted: {ex}")
            return

        # ---- build sidecar XML (copy + artefacts under detector rater) ---
        sidecar = os.path.join(backup, "rerun_sidecar.xml")
        try:
            from wonambi.attr import Annotations as WAnn
            shutil.copy2(self.annot_file_path, sidecar)
            ann = WAnn(sidecar)
            if getattr(ann, 'rater', None) is None and ann.raters:
                ann.get_rater(ann.raters[0])
            try:
                ann.add_event_type('Artefact')
            except Exception:
                pass  # type may already exist
            n_iv = 0
            for _, r in intervals.iterrows():
                ann.add_event('Artefact',
                              (float(r['start_time']), float(r['end_time'])),
                              chan='(all)')
                n_iv += 1
            ann.save() if hasattr(ann, 'save') else ann.export(sidecar)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, "Sidecar failed",
                f"Originals are snapshotted in {backup}. Sidecar error: {ex}")
            return

        # ---- channels.csv (no header, one per row) ------------------------
        chan_csv = os.path.join(backup, "channels.csv")
        try:
            import csv as _csv
            with open(chan_csv, 'w', newline='', encoding='utf-8') as fh:
                w = _csv.writer(fh)
                for c in kept:
                    w.writerow([c])
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "channels.csv failed", str(ex))
            return

        # ---- redetect_channels.csv (only the reviewer-selected re-detect
        # queue; the P3 re-run driver's --channels points straight at this).
        # Skip the file entirely when nothing is queued.
        redetect_csv = None
        if redetect:
            redetect_csv = os.path.join(backup, "redetect_channels.csv")
            try:
                with open(redetect_csv, 'w', newline='', encoding='utf-8') as fh:
                    w = _csv.writer(fh)
                    for c in redetect:
                        w.writerow([c])
            except Exception as ex:
                QtWidgets.QMessageBox.critical(
                    self, "redetect_channels.csv failed", str(ex))
                return

        # ---- confirm + record ---------------------------------------------
        # Point the re-run driver's --channels at the re-detect list when the
        # reviewer queued any; otherwise fall back to the kept channels.csv.
        driver_channels = redetect_csv or chan_csv
        cmd = (f"python examples/hdEEG_sw_detector.py "
               f"--annot {sidecar} --channels {driver_channels}")
        redetect_line = (
            f"redetect_channels.csv: {len(redetect)} channel(s) queued for "
            f"re-detect\n"
            if redetect else "redetect_channels.csv: none queued (not written)\n")
        msg = (f"Snapshot: {backup}\n  ({', '.join(snapped) or 'nothing found to snapshot'})\n\n"
               f"channels.csv: {len(kept)} kept, {len(dropped)} dropped\n"
               f"{redetect_line}"
               f"Sidecar artefacts appended (whole-montage): {n_iv}\n\n"
               f"Re-running detection OVERWRITES wonambi/*_results + the DB — "
               f"the snapshot above is your rollback.\n\n"
               f"Run e.g.:\n  {cmd}\n\nFiles written. OK.")
        if len(intervals):
            self.db.mark_artefact_intervals_exported(list(intervals['id']))
        QtWidgets.QMessageBox.information(self, "Re-run package ready", msg)
        self.status_bar.showMessage(f"Re-run package written to {backup}")

    def export_figure(self):
        """Export the active tab's plot to PNG (pg.exporters; no new dep)."""
        try:
            import pyqtgraph.exporters as pe
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Export figure", str(ex))
            return
        idx = self.tabs.currentIndex()
        target = (self.epochs_panel.plot if idx == 1
                  else self.detail_dock_w.topo)
        fp, _ = QFileDialog.getSaveFileName(self, "Export figure", "",
                                            "PNG (*.png)")
        if not fp:
            return
        try:
            pe.ImageExporter(target.plotItem).export(fp)
            self.status_bar.showMessage(f"Figure written: {fp}")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Export figure", str(ex))

    def export_qc_summary(self):
        """One-page Markdown QC report: the per-channel QC table, flagged
        channels, and marked artefact ranges for the current event type."""
        if not self.db:
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Export QC report", "",
                                            "Markdown (*.md)")
        if not fp:
            return
        df = getattr(self.qc_widget, '_qc_full', None)
        verdicts = self.db.get_channel_verdicts()
        iv = self.db.get_qc_artefact_intervals()
        try:
            with open(fp, 'w', encoding='utf-8') as fh:
                fh.write(f"# QC report — {self.qc_widget.current_event_type()}\n\n")
                fh.write(f"- Channels: {0 if df is None else len(df)}\n")
                fh.write(f"- Dropped: {sorted({c for (c,_),v in verdicts.items() if v=='drop'})}\n")
                fh.write(f"- Global artefact windows: {len(iv)}\n\n")
                if df is not None and len(df):
                    flg = df[df['flag'] != '']
                    fh.write(f"## Flagged ({len(flg)})\n\n")
                    fh.write("| channel | flag | n | density | max_p2p | reasons |\n")
                    fh.write("|---|---|---|---|---|---|\n")
                    for _, r in flg.iterrows():
                        fh.write(f"| {r['channel']} | {r['flag']} | {int(r['n'])} | "
                                 f"{r['density']:.2f} | {r['max_p2p']:.1f} | "
                                 f"{r['flag_reasons']} |\n")
            self.status_bar.showMessage(f"QC summary written: {fp}")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Export QC summary", str(ex))

    def closeEvent(self, event):
        """Handle application close event"""
        self.is_closing = True
        
        # Stop background loader thread
        if self.background_loader is not None:
            self.background_loader.stop()
            self.background_loader = None
        
        # Close database connection
        if self.db is not None:
            try:
                self.db.conn.close()
            except:
                pass
        
        event.accept()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Pin QSettings org/app once so persisted keys land consistently across
    # launches and platforms.
    if not QtCore.QCoreApplication.organizationName():
        QtCore.QCoreApplication.setOrganizationName("turtlewave")
    if not QtCore.QCoreApplication.applicationName():
        QtCore.QCoreApplication.setApplicationName("eeg_review_gui")
    app.setStyleSheet(DARK_QSS)

    window = EventReviewGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
