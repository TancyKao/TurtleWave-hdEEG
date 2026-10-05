#!/usr/bin/env python3
"""
TurtleWave Event Review GUI - Modern 3-Panel Design
Optimized for high-density EEG event review with virtualized table and timeline

Importing this module sets the environment variable
``TURTLEWAVE_QUIET_WONAMBI=1`` when it is not already set (``setdefault``),
before the first ``turtlewave_hdEEG`` import, so the GUI does not print
Wonambi's two harmless DeprecationWarnings (the fooof notice and a NumPy
scalar conversion). Export ``TURTLEWAVE_QUIET_WONAMBI=0`` to see them.
"""

import re
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

# Hide Wonambi's two harmless DeprecationWarnings (the fooof notice and a
# NumPy scalar conversion) in the GUI. The library reads this variable before
# its first Wonambi import, so it must be set before the import below; a call
# after it would be too late for the fooof notice. setdefault: export
# TURTLEWAVE_QUIET_WONAMBI=0 to see the warnings.
os.environ.setdefault('TURTLEWAVE_QUIET_WONAMBI', '1')

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
                                                PrecisionReportDialog,
                                                PrecisionRuleDialog)
except ImportError:  # run as a script
    import sample_review as _sr
    from review_sample_widgets import (SampleBar, DrawSampleDialog,
                                       PrecisionReportDialog,
                                       PrecisionRuleDialog)

try:
    from frontend.channel_types import (neighbour_channels, physio_channels,
                                        channel_units)
except ImportError:  # run as a script
    from channel_types import (neighbour_channels, physio_channels,
                               channel_units)

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

        - channel_qc: per-channel verdict (``'drop'`` = excluded: left out of
          the re-run's ``channels.csv``, of review samples and of the
          montage flag statistics; a stored ``'channel_artefact'`` reads the
          same) and ``redetect`` (1 = queued for re-detection; the queue is
          per channel, so a channel is queued when any of its rows says so).
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
                redetect INTEGER DEFAULT 0,
                PRIMARY KEY (channel, event_type)
            )
        ''')
        # additive migration for tables created before the queue was stored
        if 'redetect' not in {r[1] for r in cursor.execute(
                "PRAGMA table_info(channel_qc)")}:
            cursor.execute(
                "ALTER TABLE channel_qc ADD COLUMN redetect INTEGER DEFAULT 0")
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
        """Persist a verdict for a (channel, event_type): ``'drop'`` =
        excluded, ``''`` = not excluded. The row's ``redetect`` mark is
        kept."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        # UPDATE then INSERT (no UPSERT: SQLite older than 3.24 lacks it)
        cursor.execute(
            "UPDATE channel_qc SET verdict = ?, reviewer = ?, "
            "qc_timestamp = ? WHERE channel = ? AND event_type = ?",
            (verdict, reviewer, now, channel, event_type))
        if cursor.rowcount == 0:
            cursor.execute(
                "INSERT INTO channel_qc (channel, event_type, verdict, "
                "reviewer, qc_timestamp, redetect) VALUES (?, ?, ?, ?, ?, 0)",
                (channel, event_type, verdict, reviewer, now))
        self.conn.commit()

    def set_channel_redetect(self, channel, on, event_type=""):
        """Add a channel to the re-detect queue or take it off (the queue is
        per channel, stored on ``channel_qc.redetect``).

        Parameters
        ----------
        channel : str
            Channel label.
        on : bool
            True queues the channel; False clears the mark on every row of
            the channel.
        event_type : str, optional
            Event type of the row that carries the mark when the channel has
            no row yet. Default ``""``.
        """
        cursor = self.conn.cursor()
        channel = str(channel)
        if not on:
            cursor.execute(
                "UPDATE channel_qc SET redetect = 0 WHERE channel = ?",
                (channel,))
        else:
            cursor.execute(
                "UPDATE channel_qc SET redetect = 1 WHERE channel = ? AND "
                "event_type = ?", (channel, str(event_type)))
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO channel_qc (channel, event_type, verdict, "
                    "reviewer, qc_timestamp, redetect) VALUES (?, ?, '', '', "
                    "?, 1)", (channel, str(event_type),
                              datetime.now().isoformat()))
        self.conn.commit()

    def get_redetect_queue(self):
        """Channels queued for re-detection (a set of labels)."""
        try:
            cur = self.conn.execute(
                "SELECT DISTINCT channel FROM channel_qc WHERE redetect = 1")
        except sqlite3.OperationalError:
            return set()
        return {str(r[0]) for r in cur.fetchall()}

    def recording_subjects(self):
        """The subject ids this database records (``detection_runs.subject``,
        else ``events.subject``), as a sorted list; empty when none."""
        out = set()
        for table in ('detection_runs', 'events'):
            try:
                cols = {r[1] for r in self.conn.execute(
                    f"PRAGMA table_info({table})")}
                if 'subject' not in cols:
                    continue
                out |= {str(r[0]) for r in self.conn.execute(
                    f"SELECT DISTINCT subject FROM {table} WHERE subject IS "
                    f"NOT NULL AND subject != ''")}
            except sqlite3.Error:
                continue
            if out:
                break
        return sorted(out)

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
    """Region for one channel.

    A 10-20 / 10-5 label (``Fz``, ``F1h``, ``PPO2h``) is read from the label
    (``region_from_label``), the same rule the review sample uses, so the
    table, the topography and the sample strata agree. The coordinate rule
    (``_region_from_xy``, cut-offs tuned on EGI nets) is used only for EGI
    ``E<n>`` labels, or when the label gives ``'other'`` and ``coords`` (a
    ``{label: (x, y)}`` map) carries the channel. Without coordinates an
    EGI label falls back to the index guess.
    """
    s = str(channel)
    egi = s.startswith('E') and s[1:].isdigit()
    if not egi:
        region = region_from_label(s)
        if region != 'other':
            return region
    xy = coords.get(s) if coords else None
    if xy is not None:
        try:
            return _region_from_xy(float(xy[0]), float(xy[1]))
        except (TypeError, ValueError):
            pass
    return _region_for(channel)


#: amp-flag measures and the word the flag cell uses for each (R4.6)
AMP_TRIGGERS = (('mean_amp', 'mean'), ('p95_amp', '95th pct'),
                ('max_p2p', 'largest event'))


def compute_channel_qc(events_df, scored_minutes=None, artefact_intervals=None,
                        hard_z=3.5, soft_z=2.0, dead_frac=0.15, coords=None,
                        excluded=None):
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
        column of EGI ``E<n>`` labels (and labels the 10-20 / 10-5 rule
        cannot place) is derived from scalp position; see
        ``_region_for_channel``.
    excluded : set of str or None
        Excluded channels (``channel_qc.verdict`` 'drop'). They are left out
        of the montage median and MAD and of the dead-channel median, and
        are not flagged themselves (``flag`` ``''``); their z values are
        still given against the other channels.

    Returns
    -------
    pandas.DataFrame
        One row per channel: n, density, mean_amp, p95_amp, mean_p2p, max_p2p,
        pct_in_artefact, flag ('hard'|'soft'|'dead'|''), flag_reasons,
        ``excluded`` (bool), ``flag_trigger`` (the measure with the largest absolute z among those over
        the limit: 'mean_amp', 'p95_amp' or 'max_p2p'; ``''`` when not amp
        flagged), ``sz_mean_amp`` / ``sz_p95_amp`` / ``sz_max_p2p`` (signed
        robust z of each measure) and ``amp_z`` (the signed z of the
        trigger; for an unflagged channel the signed z with the largest
        absolute value of the three).

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
            'z_mean_amp', 'z_p95_amp', 'z_max_p2p', 'outlier_score', 'amp_z',
            'flag_trigger', 'sz_mean_amp', 'sz_p95_amp', 'sz_max_p2p',
            'excluded']
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

    excluded = {str(c) for c in (excluded or ())}
    inc = ~agg['channel'].astype(str).isin(excluded).to_numpy()

    def _signed_z(series):
        """Signed robust z of every channel against the median and MAD of
        the INCLUDED channels (zeros when the MAD is 0 or not finite)."""
        x = series.to_numpy(dtype=float)
        ref = x[inc]
        if not np.isfinite(ref).any():
            return np.zeros(len(x))
        med = np.nanmedian(ref)
        mad = np.nanmedian(np.abs(ref - med))
        if not np.isfinite(mad) or mad == 0:
            return np.zeros(len(x))
        return np.nan_to_num((x - med) / (1.4826 * mad))

    metrics = [m for m, _ in AMP_TRIGGERS]
    flag = np.array([''] * len(agg), dtype=object)
    trigger = np.array([''] * len(agg), dtype=object)
    reasons = [[] for _ in range(len(agg))]
    sz = {m: np.zeros(len(agg)) for m in metrics}
    if int(inc.sum()) >= 3:
        for metric in metrics:
            sz[metric] = _signed_z(agg[metric])
        for i in range(len(agg)):
            if not inc[i]:
                continue                # an excluded channel is not judged
            over = [(abs(sz[m][i]), m) for m in metrics
                    if abs(sz[m][i]) > soft_z]
            if not over:
                continue
            flag[i] = 'hard' if max(over)[0] > hard_z else 'soft'
            trigger[i] = max(over, key=lambda t: t[0])[1]
            reasons[i] = [f"{m} z={z:.1f}" for z, m in over]
        n_all = agg['n'].to_numpy(dtype=float)
        n_med = np.nanmedian(n_all[inc])
        if np.isfinite(n_med) and n_med > 0:
            for i, nv in enumerate(n_all):
                if inc[i] and nv < dead_frac * n_med:
                    flag[i] = 'dead'
                    trigger[i] = ''
                    reasons[i] = [f"n={int(nv)} < {dead_frac:.0%} of median {n_med:.0f}"]
    agg['flag'] = flag
    agg['excluded'] = ~inc
    agg['flag_trigger'] = trigger
    agg['flag_reasons'] = ['; '.join(r) for r in reasons]
    for m in metrics:
        agg['z_' + m] = np.abs(sz[m])
        agg['sz_' + m] = sz[m]
    stack = np.vstack([sz[m] for m in metrics])          # 3 x channels
    agg['outlier_score'] = np.abs(stack).max(axis=0)
    # Amp z: the trigger's signed z; unflagged: the largest |z| of the three
    pick = np.abs(stack).argmax(axis=0)
    for i, t in enumerate(trigger):
        if t:
            pick[i] = metrics.index(t)
    agg['amp_z'] = stack[pick, np.arange(len(agg))] if len(agg) else []
    # Region: from the 10-20 / 10-5 label; coordinates only for EGI labels
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


def _span_text(t0, t1):
    """Length of an excluded range for a list row, always as a duration
    (a free-brushed range need not line up with epochs): ``'350 ms'``,
    ``'5.4 s'``, ``'60 s'``, and ``'2 min 05 s'`` from 120 s up."""
    dur = abs(float(t1) - float(t0))
    ms = int(round(dur * 1000))
    if ms < 1000:
        return f"{ms} ms"
    if round(dur, 1) < 120:
        return f"{round(dur, 1):g} s"
    m, sec = divmod(int(round(dur)), 60)
    return f"{m} min {sec:02d} s"


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
_QC_COLS = [   # UX spec revision 5, R5.0: eight columns, amplitude QC only
    ('channel', 'Channel'), ('region', 'Region'), ('n', 'Events'),
    ('density', 'Density /min'), ('mean_amp', 'Mean amp µV'),
    ('amp_z', 'Amp z'), ('flag', 'Amp flag'), ('verdict', 'Status'),
]
_AMP_FLAG_TEXT = {'': '✓ OK', 'hard': '× HARD', 'soft': '▲ SOFT',
                  'dead': 'DEAD'}
SAMPLE_HIDDEN_TIP = 'Hidden while you review the sample.'
#: ``channel_qc.verdict`` values that mean "excluded" ('drop' is what the
#: Exclude channel toggle writes; 'channel_artefact' is from older versions)
EXCLUDED_VERDICTS = ('drop', 'channel_artefact')
QUEUED_TIP = ('Queued for re-detection. Use File ▸ Export re-run package… '
              'to re-run these channels.')
EXCLUDE_TIP = ('Leaves this channel out of review samples, the re-run export, '
               'the flag statistics and the topography. Event density and '
               'exported events are unchanged in 4.6.0. Click again to '
               'include it.')
EXCLUDE_TIME_TIP = ('Excludes this time from analysis for every channel. It '
                    'takes effect when you export a re-run package (File ▸ '
                    'Export re-run package…) and re-detect with it. Events '
                    'already detected are not changed.')
#: the review-qc sidecar's role, said wherever it is mentioned
REVIEW_QC_RECORD = ('The review-qc XML beside the annotation file is a record '
                    'of this review; not read by detection.')
EXCLUDE_TIME_HINT = ('Brush a range on the trace to exclude it, or click a '
                     'hatched range to remove it.')
CLEAR_RANGE_TIP = 'Clear the unsaved range (Esc).'
REMOVE_EXCLUSION_TIP = ('Stop excluding this time. It is deleted from this '
                        'review (and its review-qc record) and counts as '
                        'analysed time again here; export a new re-run '
                        'package to apply this at re-detection.')
EXCLUDED_PURPLE = '#a371f7'


SAMPLE_BANNER_TEXT = ('Review sample in progress: channel checks are hidden. '
                      'Exit sample to see them.')
SAMPLE_STAGE_TIP = ('Stages apply to the channel checks, which are hidden '
                    'while you review the sample.')
INTERP_LEGEND = '~ = interpolated'
INTERP_LEGEND_TIP = ('Interpolated channel: its signal was rebuilt from '
                     'neighbouring channels, not recorded.')
REDETECT_ADD_TIP = 'Add this channel to the re-detect queue (F).'
REDETECT_REMOVE_TIP = 'Remove this channel from the re-detect queue (F).'
QUEUE_HARD_TIP = ('Add every channel with a hard amp flag to the re-detect '
                  'queue.')
RERUN_STATUS_TIP = ('Writes the queued channels (redetect_channels.csv) for '
                    'examples/rerun_detection.py --channels.')
RERUN_EMPTY_TEXT = ('Nothing is queued for re-detection. Select a channel and '
                    'use "Add to re-detect queue".')
AMP_Z_HEADER_TIP = ('Robust z of the amplitude measure furthest from the '
                    'other channels: the mean, the 95th percentile or the '
                    'largest event. Hover a cell for all three.')
MEAN_AMP_HEADER_TIP = ("Mean amplitude of this channel's events "
                       "(detection-band signal).")



def fit_checked_width(button, texts=None):
    """Give a checkable button room for its text at weight 600 (the checked
    style), so the label is not clipped when it turns bold. ``texts`` are
    all the labels the button can show (default: its current text). The
    width is the button's own size hint plus what the bold text adds, so it
    follows the stylesheet's font and padding."""
    button.ensurePolished()
    font = QtGui.QFont(button.font())
    normal = QtGui.QFontMetrics(font)
    font.setWeight(QtGui.QFont.DemiBold)
    bold = QtGui.QFontMetrics(font)
    chrome = button.sizeHint().width() - normal.horizontalAdvance(
        button.text())
    widest = max(bold.horizontalAdvance(t) for t in (texts or [button.text()]))
    button.setMinimumWidth(chrome + widest + 4)


def exclude_tip(event_type=None):
    """The Exclude channel tooltip; with an event type it ends ``Applies to
    spindles only.`` (the exclusion is stored per event type)."""
    if not event_type:
        return EXCLUDE_TIP
    events = _er.EVENT_PLURAL.get(str(event_type), str(event_type))
    return f"{EXCLUDE_TIP} Applies to {events} only."


def set_exclude_button(button, excluded, event_type=None):
    """One Exclude channel toggle (R4.0): ``Exclude channel`` in the danger
    style for an included channel, ``Include channel`` shown checked for an
    excluded one; the tooltip names the event type it applies to."""
    button.setToolTip(exclude_tip(event_type))
    button.setText('Include channel' if excluded else 'Exclude channel')
    button.setChecked(bool(excluded))
    name = '' if excluded else 'danger'
    if button.objectName() != name:
        button.setObjectName(name)
        button.style().unpolish(button)
        button.style().polish(button)


class _SepWrapLabel(QLabel):
    """A word-wrapped label that breaks only at its `` · `` separators:
    spaces inside a part are shown as no-break spaces. ``text()`` returns
    the text as it was set."""

    def setText(self, text):
        self._plain = str(text or '')
        parts = [p.replace(' ', '\u00a0') for p in self._plain.split(' · ')]
        super().setText('\u00a0· '.join(parts))

    def text(self):
        return getattr(self, '_plain', super().text())


class _LinkLabel(QLabel):
    """A rich-text label (with links) whose ``text()`` is the plain text,
    as it reads on screen; :meth:`html` gives the markup."""

    def setText(self, text):
        import html as _html
        self._html = str(text or '')
        self._plain = _html.unescape(re.sub(r'<[^>]+>', '', self._html))
        super().setText(self._html)

    def text(self):
        return getattr(self, '_plain', super().text())

    def html(self):
        return getattr(self, '_html', '')


class _ElidedLabel(QLabel):
    """A one-line label that paints its text elided with ``…`` when the
    widget is too narrow; ``text()`` keeps the whole line."""

    def minimumSizeHint(self):
        return QtCore.QSize(40, super().minimumSizeHint().height())

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        fm = self.fontMetrics()
        p.drawText(self.rect(), int(Qt.AlignLeft | Qt.AlignVCenter),
                   fm.elidedText(self.text(), Qt.ElideRight, self.width()))
        p.end()
#: Column index of each key in the QC table (for hiding / header tooltips).
_QC_COL_INDEX = {k: i for i, (k, _) in enumerate(_QC_COLS)}

_STATE_RD_COLOR = '#5a8fce'    # ↻ queued for re-detection
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
    (peak-to-peak beyond physiological scale)."""
    if amp > 1000:
        return f"{amp / 1000:.1f} kµV ⚠", True
    return f"{int(round(amp))} µV", False


def _artefact_tooltip(amp):
    """Tooltip for an impossible-scale (>1000 µV) amplitude row."""
    return (f"{amp:.0f} µV peak-to-peak — exceeds physiological scale "
            f"(>1000 µV).")


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
    """Per-channel QC table for ONE event type (UX spec revision 3,
    section 1). Sort keys on ``Qt.UserRole``; ``None`` means missing and
    always sorts last (see :class:`_QCSortProxy`).

    (Amplitude columns are Wonambi µV on the detection-band signal —
    comparable only within one event type.)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame(columns=[c[0] for c in _QC_COLS])
        self._records = []   # list[dict] — O(1) cell access (no pandas .iloc)
        self._redetect = set()   # channels queued for re-detection
        self._event_type = ''
        # population-check context (ChannelQCWidget.set_checks_context)
        self._checks = {'recorded': False, 'method': '', 'ratio': True,
                        'tips': {}, 'pending': False, 'medians': {},
                        'bounds': None}

    def set_data(self, qc_df, verdicts=None, redetect=None, event_type=''):
        self.beginResetModel()
        self._redetect = {str(c) for c in (redetect or ())}
        self._event_type = str(event_type or '')
        df = qc_df.copy() if qc_df is not None else pd.DataFrame()
        if 'verdict' not in df.columns:
            df['verdict'] = ''
        if verdicts and len(df):
            df['verdict'] = df['channel'].map(verdicts).fillna(df['verdict'])
        df['redetect'] = (df['channel'].astype(str).isin(self._redetect)
                          if len(df) else False)
        self.df = df.reset_index(drop=True)
        # A plain records list: the view and the sort proxy call data()
        # thousands of times per reset; dict access is O(1).
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

    def record(self, row):
        return self._records[row] if 0 <= row < len(self._records) else None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._records)):
            return QVariant()
        rec = self._records[index.row()]
        key = _QC_COLS[index.column()][0]
        dropped = str(rec.get('verdict', '')) in EXCLUDED_VERDICTS
        if dropped and key == 'flag':
            # an excluded channel is not judged (R4.4)
            if role == Qt.DisplayRole:
                return '—'
            if role == Qt.ForegroundRole:
                return QtGui.QColor(THEME['text_3'])
            if role == Qt.BackgroundRole:
                return QtGui.QColor(48, 34, 36)
            return -1 if role == Qt.UserRole else QVariant()
        val = rec.get(key, '')
        queued = str(rec.get('channel', '')) in self._redetect
        if key == 'verdict' and role == Qt.ToolTipRole and queued:
            return QUEUED_TIP
        if key == 'verdict' and role == _STATUS_HTML_ROLE:
            head = (f"<span style='color:{THEME['bad']}'>× excluded</span>"
                    if dropped else 'kept')
            return head + (f" · <span style='color:{_STATE_RD_COLOR}'>"
                           f"↻ re-detect</span>" if queued else '')
        if role == Qt.DisplayRole:
            if key == 'verdict':
                return ('× excluded' if dropped else 'kept') + (
                    ' · ↻ re-detect' if queued else '')
            if key == 'flag':
                text = _AMP_FLAG_TEXT.get(str(val or ''), '✓ OK')
                word = dict(AMP_TRIGGERS).get(
                    str(rec.get('flag_trigger') or ''))
                if word and str(val) in ('hard', 'soft'):
                    text += f" · {word}"
                return text
            if key == 'n':
                try:
                    return f"{int(val):,}"
                except (TypeError, ValueError):
                    return '—'
            if key in ('density', 'amp_z'):
                f = _er._finite(val)
                return '—' if f is None else f"{f:.1f}".replace('-', '−')
            if key == 'mean_amp':
                f = _er._finite(val)
                return '—' if f is None else f"{f:.2f}"
            if key == 'region':
                return str(val or '').capitalize()
            return '' if val is None else str(val)
        if role == Qt.UserRole:  # sort key
            if key == 'flag':
                return {'hard': 3, 'soft': 2, 'dead': 1}.get(
                    str(val or ''), 0) * 1000 + float(
                        _er._finite(rec.get('outlier_score')) or 0)
            if key == 'verdict':
                return 1 if dropped else 0
            if key in ('channel', 'region'):
                return str(val or '')
            if key == 'amp_z':              # Amp z sorts by |z| (R4.6)
                f = _er._finite(val)
                return None if f is None else abs(f)
            return _er._finite(val)
        if role == Qt.BackgroundRole:
            if key in _HEAT_Z:
                c = _heat_bg(rec.get(_HEAT_Z[key]))
                if c is not None:
                    return c
            if dropped:
                return QtGui.QColor(48, 34, 36)
            flag = str(rec.get('flag', ''))
            if key == 'flag' and flag in _FLAG_BG:
                return _FLAG_BG[flag]
        if role == Qt.ForegroundRole:
            if key == 'flag':
                return {'hard': QtGui.QColor(THEME['bad']),
                        'soft': QtGui.QColor(THEME['warn']),
                        'dead': QtGui.QColor(155, 166, 181)}.get(
                    str(val or ''), QtGui.QColor(THEME['ok']))
            if key == 'verdict' and dropped:
                return QtGui.QColor(THEME['bad'])
            if dropped:                     # the row keeps its values, dimmed
                return QtGui.QColor(THEME['text_3'])
        if role == Qt.ToolTipRole and key in ('flag', 'amp_z'):
            zs = [_er._finite(rec.get(k)) for k in
                  ('sz_mean_amp', 'sz_p95_amp', 'sz_max_p2p')]
            if all(z is not None for z in zs):
                return (f"mean z {zs[0]:.1f} · 95th pct z {zs[1]:.1f} · "
                        f"largest event z {zs[2]:.1f}")
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return QVariant()
        if role == Qt.DisplayRole:
            return _QC_COLS[section][1]
        if role == Qt.ToolTipRole:
            key = _QC_COLS[section][0]
            tip = {'amp_z': AMP_Z_HEADER_TIP,
                   'mean_amp': MEAN_AMP_HEADER_TIP,
                   'flag': getattr(self, 'amp_flag_tip', None)
                   or _er.amp_flag_tooltip(3.5, 2.0)}.get(key)
            if tip:
                return tip
        return QVariant()

#: Role carrying the Status cell as HTML (two colours in one cell).
_STATUS_HTML_ROLE = Qt.UserRole + 7


class _StatusDelegate(QtWidgets.QStyledItemDelegate):
    """Paints the Status cell from :data:`_STATUS_HTML_ROLE`, so
    ``× excluded`` (``bad``) and ``↻ re-detect`` (re-detect blue) keep their
    own colours."""

    def paint(self, painter, option, index):
        html = index.data(_STATUS_HTML_ROLE)
        if not html:
            return super().paint(painter, option, index)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ''
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, opt, painter,
                          opt.widget)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(opt.font)
        doc.setHtml(f"<span style='color:{THEME['text']}'>{html}</span>")
        painter.save()
        painter.translate(opt.rect.left() + 4, opt.rect.top() + max(
            0, (opt.rect.height() - doc.size().height()) / 2))
        doc.drawContents(painter)
        painter.restore()


class _QCSortProxy(QtCore.QSortFilterProxyModel):
    """Sorts on ``Qt.UserRole`` with missing values (``None``) last in both
    directions, and filters rows by the Show combo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keep = None          # set of channels, or None for all

    def filterAcceptsRow(self, row, parent):
        if self.keep is None:
            return True
        return self.sourceModel().channel_at(row) in self.keep

    def lessThan(self, left, right):
        a = self.sourceModel().data(left, Qt.UserRole)
        b = self.sourceModel().data(right, Qt.UserRole)
        a = None if a is None or (isinstance(a, QVariant) and a.isNull()) else a
        b = None if b is None or (isinstance(b, QVariant) and b.isNull()) else b
        desc = self.sortOrder() == Qt.DescendingOrder
        if a is None or b is None:
            if a is None and b is None:
                return False
            # missing last: ascending -> missing is "greater"; descending ->
            # missing is "smaller"
            return (b is None) != desc
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


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
    helpRequested = pyqtSignal()        # "What do these mean?"

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
        # header row: EVENT i OF n IN EPOCH ............ What do these mean?
        hrow = QHBoxLayout()
        self.header_lbl = _h_label("EVENT")
        hrow.addWidget(self.header_lbl)
        hrow.addStretch()
        self.help_btn = QPushButton(_er.HELP_LINK_TEXT)
        self.help_btn.setFlat(True)
        self.help_btn.setFocusPolicy(Qt.NoFocus)
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setStyleSheet(
            f"border:0;background:transparent;color:{THEME['accent']};"
            "text-decoration:underline;font-size:11px;padding:0;")
        self.help_btn.clicked.connect(lambda: self.helpRequested.emit())
        hrow.addWidget(self.help_btn)
        lay.addLayout(hrow)
        # header line: time · channel · stage · method band (R4.8)
        self.event_line = _SepWrapLabel("")
        self.event_line.setWordWrap(True)
        self.event_line.setStyleSheet(
            f"color:{THEME['text']};font-size:12px;"
            "font-family:'IBM Plex Mono',monospace;")
        self.event_line.setVisible(False)
        lay.addWidget(self.event_line)
        # sample line, directly under the header line (sample events only)
        self.sample_lbl = QLabel("")
        self.sample_lbl.setWordWrap(True)
        self.sample_lbl.setStyleSheet(
            f"color:{THEME['text_2']};font-size:11px;")
        self.sample_lbl.setVisible(False)
        lay.addWidget(self.sample_lbl)
        self.hidden_lbl = QLabel(_er.HIDDEN_LABELS_NOTE)
        self.hidden_lbl.setWordWrap(True)
        self.hidden_lbl.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        self.hidden_lbl.setVisible(False)
        lay.addWidget(self.hidden_lbl)
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
        for dec, text in (('accept', '✓ Accept  A'), ('reject', '× Reject  R'),
                          ('unsure', '? Unsure  U')):
            b = QPushButton(text)
            b.setCheckable(True)        # checked = this reviewer's saved one
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, d=dec: self._on_decision_click(d))
            fit_checked_width(b)
            brow.addWidget(b)
            self.btn[dec] = b
        lay.addLayout(brow)
        # reason grid (revision 3): per event type, digits 1-8 and 0
        self.grid_hdr = QLabel("Reason (required for Reject)")
        self.grid_hdr.setAlignment(Qt.AlignRight)
        self.grid_hdr.setStyleSheet(f"color:{THEME['text_2']};font-size:11px;")
        lay.addWidget(self.grid_hdr)
        self.grid_box = QWidget()
        self.grid_box.setObjectName('reasonGrid')
        self.grid_lay = QtWidgets.QGridLayout(self.grid_box)
        self.grid_lay.setContentsMargins(2, 2, 2, 2)
        self.grid_lay.setSpacing(2)
        lay.addWidget(self.grid_box)
        self.reason_buttons = {}
        self._grid_type = None
        self.set_grid('spindle')
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Comment"))
        self.comment = QLineEdit()
        self.comment.setMaxLength(500)
        self.comment.setPlaceholderText('Comment (C) — required for "other"')
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
        self.clear_btn = QPushButton("Clear decision")
        self.clear_btn.setToolTip(
            'Delete your decision on this event (Ctrl+Z brings it back).')
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
        self.progress_lbl.setWordWrap(True)
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

    def _clear_grid(self):
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._row_widgets = {}

    def _value_html(self, row):
        """The value as rich text: the number plain, the reading word in
        the ``warn`` / ``bad`` colour at weight 600 (R4.8). A muted
        (compact-state) row and a row without a level are plain."""
        import html
        level = row.get('level')
        word = row.get('word') or ''
        number = row.get('number', row.get('value', ''))
        if level not in ('warn', 'bad') or not word:
            return None
        colour = THEME['warn'] if level == 'warn' else THEME['bad']
        span = (f"<span style='color:{colour};font-weight:600'>"
                f"{html.escape(word)}</span>")
        num = html.escape(str(number))
        return (span + num) if row.get('word_first') else (num + span)

    def set_event_line(self, text, tooltip=''):
        """The header line under ``EVENT …`` (empty hides it)."""
        self.event_line.setText(text or '')
        self.event_line.setToolTip(tooltip or '')
        self.event_line.setVisible(bool(text))

    def set_sample_line(self, text, tooltip=''):
        """The sample line under the header line (empty hides it)."""
        self.sample_lbl.setText(text or '')
        self.sample_lbl.setToolTip(tooltip or '')
        self.sample_lbl.setVisible(bool(text))

    def set_rows(self, rows):
        """Show ``rows`` (``build_event_rows`` dicts) in order."""
        self._clear_grid()
        self._rows = list(rows or [])
        r = 0
        for row in self._rows:
            lab = QLabel(row['label'])
            lab.setStyleSheet("color:#b8b8b8;font-size:11px;")
            rich = self._value_html(row)
            val = QLabel(rich if rich is not None else str(row['value']))
            val.setTextFormat(Qt.RichText if rich is not None
                              else Qt.PlainText)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setWordWrap(True)
            val.setStyleSheet(
                "font-family:'IBM Plex Mono',monospace;font-size:12px;"
                + ("color:#888888;" if row.get('level') == 'muted'
                   else "color:#e5e5e5;"))
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
                sl.setStyleSheet("font-size:11px;color:#888888;")
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
        self.set_decision(None)
        self.set_armed(None)
        self.set_controls_enabled(False)
        self.set_header('EVENT')
        self.set_event_line('')
        self.set_sample_line('')
        self.set_hidden(False)

    def set_header(self, text):
        self.header_lbl.setText(text)

    def set_hidden(self, on):
        """Show the live-sample-review line ``Labels hidden until …``."""
        self.hidden_lbl.setVisible(bool(on))

    def set_grid(self, event_type):
        """Rebuild the reason grid for ``event_type`` (spindle grid, or the
        slow-wave / K-complex grid)."""
        kind = 'spindle' if str(event_type) == 'spindle' else 'slow'
        if kind == self._grid_type:
            return
        self._grid_type = kind
        while self.grid_lay.count():
            it = self.grid_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self.reason_buttons = {}
        for i, (d, tok, label, tip) in enumerate(_er.reason_grid(
                'spindle' if kind == 'spindle' else 'slow_wave')):
            b = QPushButton(f"{d}  {label}")
            b.setCheckable(True)
            b.setFlat(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(tip)
            b.setStyleSheet("text-align:left;padding:1px 4px;")
            b.clicked.connect(lambda _=False, t=tok: self.reasonChosen.emit(t))
            self.grid_lay.addWidget(b, i // 2, i % 2)
            self.reason_buttons[tok] = b

    def grid_texts(self):
        """Button texts in grid order (row-major)."""
        out = []
        for i in range(self.grid_lay.count()):
            w = self.grid_lay.itemAt(i).widget()
            if w is not None:
                out.append(w.text())
        return out

    def current_reason(self):
        return next((t for t, b in self.reason_buttons.items()
                     if b.isChecked()), '')

    def set_note(self, text):
        self.note_lbl.setText(text or '')
        self.note_lbl.setVisible(bool(text))

    def set_controls_enabled(self, on):
        for b in list(self.btn.values()) + [self.clear_btn]:
            b.setEnabled(bool(on))
        for b in getattr(self, 'reason_buttons', {}).values():
            b.setEnabled(bool(on))
        self.comment.setEnabled(bool(on))

    def set_current(self, text, subs, tooltip=''):
        self.current_lbl.setText(text)
        self.current_lbl.setToolTip(tooltip or '')
        self.current_sub.setText('\n'.join(subs or []))
        self.current_sub.setVisible(bool(subs))

    def set_decision(self, decision):
        """Check the button of this reviewer's SAVED decision only (R5.3);
        ``None``: none checked. An armed decision is shown by
        :meth:`set_armed` (border only), never by the checked state."""
        self._saved_decision = decision
        for d, b in self.btn.items():
            b.setChecked(d == decision)

    def _on_decision_click(self, decision):
        # a click toggles a checkable button by itself; put the saved state
        # back before the window decides what the click means
        self.set_decision(getattr(self, '_saved_decision', None))
        self.decisionClicked.emit(decision)

    #: an ARMED button (waiting for a reason, nothing written yet) has the
    #: 1 px accent border and NO fill; a CHECKED one (the saved decision) is
    #: filled and bold (``QPushButton:checked`` in ``DARK_QSS``), so the two
    #: read differently (spec R4.2)
    ARMED_QSS = "QPushButton { border: 1px solid #5a8fce; }"

    def set_armed(self, decision, hint=None):
        """Show an armed Reject/Unsure (or none) and its hint: the button
        gets a dashed accent border, the reason grid a 1 px accent border."""
        for d, b in self.btn.items():
            b.setStyleSheet(self.ARMED_QSS if d == decision else "")
        self.grid_box.setStyleSheet(
            f"#reasonGrid{{border:1px solid {THEME['accent']};}}"
            if decision else "")
        self.hint_lbl.setText(hint or (self.HINT_SAMPLE
                                       if getattr(self, '_sample_mode', False)
                                       else self.HINT))

    def set_reason(self, token):
        for t, b in self.reason_buttons.items():
            b.setChecked(t == token)

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


class _ClickLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, ev):
        self.clicked.emit()
        super().mousePressEvent(ev)


class FlaggedChannelList(QWidget):
    """``CHECKS — FLAGGED CHANNELS`` under the topography (UX spec revision
    3, section 1): facts only, one row per checks-flagged channel, hard
    first; clicking a row selects the channel and shows ``Open in Epochs``
    and ``Exclude channel`` on that row only."""

    channelPicked = pyqtSignal(str)
    openInEpochs = pyqtSignal(str)
    dropChannel = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        head = QHBoxLayout()
        self.header = _h_label("CHECKS — FLAGGED CHANNELS")
        head.addWidget(self.header)
        head.addStretch()
        self.counts = QLabel("")
        self.counts.setStyleSheet(f"color:{THEME['text_2']};font-size:11px;")
        head.addWidget(self.counts)
        lay.addLayout(head)
        self.empty = QLabel("")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(f"color:{THEME['text_3']};font-size:11px;")
        lay.addWidget(self.empty)
        self.box = QWidget()
        self.box_lay = QVBoxLayout(self.box)
        self.box_lay.setContentsMargins(0, 0, 0, 0)
        self.box_lay.setSpacing(1)
        lay.addWidget(self.box)
        self.rows = []
        self.expanded = None

    def set_rows(self, rows, n_hard, n_soft, empty_text=''):
        """``rows``: dicts ``channel, flag, facts (list), tooltip,
        dropped``, already in display order."""
        while self.box_lay.count():
            it = self.box_lay.takeAt(0)
            w = it.widget()
            if w is not None:          # gone at once, not at the next loop
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.rows = []
        self.expanded = None
        self.counts.setText(f"{n_hard} HARD · {n_soft} SOFT")
        self.empty.setText(empty_text)
        self.empty.setVisible(not rows)
        for r in rows:
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(1)
            line = QHBoxLayout()
            hard = r['flag'] == 'hard'
            badge = QLabel('× HARD' if hard else '▲ SOFT')
            badge.setStyleSheet(
                f"color:{THEME['bad'] if hard else THEME['warn']};"
                "font-weight:600;font-size:11px;"
                "font-family:'IBM Plex Mono',monospace;")
            name = r['channel'] + (' · excluded' if r.get('dropped') else '')
            text = ' · '.join([name] + list(r['facts']))
            lbl = _ClickLabel(text)
            lbl.setWordWrap(True)
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setToolTip(r.get('tooltip') or '')
            lbl.setStyleSheet("font-size:11px;" + (
                f"color:{THEME['text_3']};" if r.get('dropped') else ''))
            line.addWidget(badge)
            line.addWidget(lbl, 1)
            wl.addLayout(line)
            btns = QWidget()
            bl = QHBoxLayout(btns)
            bl.setContentsMargins(56, 0, 0, 2)
            b_open = QPushButton("Open in Epochs")
            b_open.setObjectName("primary")
            b_open.clicked.connect(
                lambda _=False, c=r['channel']: self.openInEpochs.emit(c))
            bl.addWidget(b_open)
            b_drop = None
            if not r.get('dropped'):
                b_drop = QPushButton("Exclude channel")
                b_drop.setObjectName("danger")
                b_drop.setToolTip(exclude_tip(
                    getattr(self, 'event_type', None)))
                b_drop.clicked.connect(
                    lambda _=False, c=r['channel']: self.dropChannel.emit(c))
                bl.addWidget(b_drop)
            bl.addStretch()
            btns.setVisible(False)
            wl.addWidget(btns)
            lbl.clicked.connect(lambda c=r['channel']: self.click_row(c))
            self.box_lay.addWidget(w)
            self.rows.append({'channel': r['channel'], 'text': text,
                              'label': lbl, 'badge': badge.text(),
                              'buttons': btns, 'open': b_open,
                              'drop': b_drop})

    def click_row(self, channel):
        """Select ``channel`` and show its two buttons (others hide)."""
        for r in self.rows:
            r['buttons'].setVisible(r['channel'] == channel)
        self.expanded = channel
        self.channelPicked.emit(channel)

    def row(self, channel):
        return next((r for r in self.rows if r['channel'] == channel), None)

    def texts(self):
        return [r['text'] for r in self.rows]


class ChannelDetailDock(QWidget):
    """Right-dock: a "worst epochs" list and a topography card (empty-state
    default; scipy griddata when channel coords are available).
    Clicking a row in the worst-epochs list emits gotoEpochRequested(idx),
    which the main window wires to switch to the Epochs tab and call
    EpochsPanel._goto_epoch."""

    EXCLUDED_LEGEND = '○ excluded channel (not used for the map)'

    loadMontageRequested = pyqtSignal()
    gotoEpochRequested = pyqtSignal(int)          # epoch idx on current channel
    gotoChannelEpochRequested = pyqtSignal(str, float)  # channel, event start_t
    channelPicked = pyqtSignal(str)             # topo electrode clicked -> select
    unmarkArtefactRequested = pyqtSignal(int)   # interval id (× button)
    exclusionClicked = pyqtSignal(int)          # interval id (row label)
    openInEpochs = pyqtSignal(str)              # flagged list row button
    dropChannelRequested = pyqtSignal(str)      # flagged list row button

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
        topo_row.addWidget(QLabel("Topo:"))
        self.topo_combo = QComboBox()
        for col, label in (('density', 'Event density'),
                           ('mean_amp', 'Mean amp (µV)'),
                           ('max_p2p', 'Max p2p (µV)')):
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
        # caption of the check metric on the topography (two lines)
        self.topo_caption = QLabel("")
        self.topo_caption.setWordWrap(True)
        self.topo_caption.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        lay.addWidget(self.topo_caption)
        # shown only when at least one channel is excluded (R4.4)
        self.topo_excluded_lbl = QLabel(self.EXCLUDED_LEGEND)
        self.topo_excluded_lbl.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        self.topo_excluded_lbl.setVisible(False)
        lay.addWidget(self.topo_excluded_lbl)
        self.excluded_spots = None      # hollow markers of excluded channels
        self.topo_input_channels = []   # channels the map is interpolated from
        # population-check caption (not recorded / which run)
        self.checks_caption = QLabel("")
        self.checks_caption.setWordWrap(True)
        self.checks_caption.setStyleSheet("color:#6b7585;font-size:11px;")
        lay.addWidget(self.checks_caption)
        self._checks_recorded = None
        self.flagged = FlaggedChannelList()
        self.flagged.channelPicked.connect(self.channelPicked.emit)
        self.flagged.openInEpochs.connect(self.openInEpochs.emit)
        self.flagged.dropChannel.connect(self.dropChannelRequested.emit)
        lay.addWidget(self.flagged)
        # ring state for check metrics: {ch: (flag, z)}, dropped channels
        self._check_flags = {}
        self._dropped = set()
        self._caption_ctx = None
        self.ring_items, self.ring_labels = [], []

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
        self.global_worst_hidden = self._hidden_line()
        lay.addWidget(self.global_worst_hidden)

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
        # facts line: name, region, events; amp and checks flags; off-band
        self.facts_line = QLabel("")
        self.facts_line.setWordWrap(True)
        self.facts_line.setStyleSheet("font-size:11px;")
        lay.addWidget(self.facts_line)

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
        self.worst_hidden = self._hidden_line()
        lay.addWidget(self.worst_hidden)

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
        self.marked_hdr = _h_label("EXCLUDED TIME (0)")
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
                f"EXCLUDED TIME ({len(marked)} from {channel} · {total} total)")
        else:
            self.marked_hdr.setText(f"EXCLUDED TIME ({len(marked)})")
        for m in marked:
            t0, t1 = float(m['start_time']), float(m['end_time'])
            mid = int(m['id'])
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            dtxt = _span_text(t0, t1)
            lbl = QPushButton(f"{_hms(t0)}–{_hms(t1)}  ({dtxt})")
            lbl.setFlat(True)
            lbl.setStyleSheet(
                "text-align:left;color:#d6dee8;"
                "font-family:'IBM Plex Mono',monospace;font-size:11px;")
            lbl.clicked.connect(
                lambda _=False, i=mid: self.exclusionClicked.emit(int(i)))
            x = QPushButton("×")
            x.setFixedWidth(32)
            x.setToolTip('Remove exclusion')
            x.setStyleSheet("color:#f85149;font-weight:600;font-size:15px;"
                            "padding:0 6px;")
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
            text += (f" \u00b7 + {int(pending_marks)} excluded range"
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

        Items that do not apply are absent (``Low-prominence share`` for
        slow waves and K-complexes, ``Amp vs threshold`` for a method with
        no ratio); on a run without stored figures the check items are
        disabled with the suffix `` — not recorded for this run``.
        """
        keep = self.topo_combo.currentData()
        self.topo_combo.blockSignals(True)
        while self.topo_combo.count() > 3:
            self.topo_combo.removeItem(3)
        model = self.topo_combo.model()
        for col, (_hdr, label) in CHECK_COLUMNS.items():
            if col == 'pct_low_prom' and event_type != 'spindle':
                continue
            if col == 'med_thresh_ratio' and recorded and not ratio:
                continue
            text = label + ('' if recorded
                            else ' — not recorded for this run')
            self.topo_combo.addItem(text, col)
            item = model.item(self.topo_combo.count() - 1)
            if item is not None:
                item.setEnabled(bool(recorded))
        idx = self.topo_combo.findData(keep)
        if idx < 0 or not (model.item(idx) and model.item(idx).isEnabled()):
            idx = 0
        self.topo_combo.setCurrentIndex(idx)
        self.topo_combo.blockSignals(False)
        self.topo_metric = self.topo_combo.currentData() or 'density'
        self._checks_recorded = bool(recorded)
        self.checks_caption.setText(caption or '')
        self._apply_sample_lock()

    def set_check_state(self, flags, dropped=(), caption_ctx=None):
        """Rings and labels for the check metrics: ``flags`` is
        ``{channel: (flag, z)}`` of checks-flagged channels, ``dropped`` the
        dropped channels (no ring), ``caption_ctx`` ``(event_type,
        stage_text, band, min_dur)`` for the caption."""
        self._check_flags = dict(flags or {})
        self._dropped = set(dropped or ())
        self._caption_ctx = caption_ctx

    SAMPLE_HIDDEN = 'Channel checks are hidden while you review the sample.'

    @staticmethod
    def _hidden_line():
        q = QLabel(SAMPLE_HIDDEN_TIP)
        q.setStyleSheet(f"color:{THEME['text_3']};font-size:11px;")
        q.setVisible(False)
        return q

    def set_sample_mode(self, on):
        """Live sample review: channel-level flags stay out of the dock (one
        facts line, no flagged list, no rings, check metrics disabled in the
        Topo combo), and the two amplitude rankings (worst events, worst
        epochs) are emptied and replaced by one line each, because they
        would show which events are amplitude outliers. The caller redraws;
        Exit refills the rankings from what was last given."""
        self._sample_mode = bool(on)
        self._apply_sample_lock()
        for lst, line in ((self.global_worst_list, self.global_worst_hidden),
                          (self.worst_list, self.worst_hidden)):
            lst.setVisible(not on)
            line.setVisible(bool(on))
        if on:
            self.global_worst_list.clear()
            self.worst_list.clear()
        else:
            if getattr(self, '_last_global', None) is not None:
                self.set_global_worst(*self._last_global)
            if getattr(self, '_last_channel', None) is not None:
                self.update_channel(*self._last_channel)

    def _apply_sample_lock(self):
        """Disable (with tooltip) or restore the Topo combo's check items."""
        on = getattr(self, '_sample_mode', False)
        model = self.topo_combo.model()
        for i in range(3, self.topo_combo.count()):
            item = model.item(i)
            if item is None:
                continue
            item.setEnabled(bool(self._checks_recorded) and not on)
            self.topo_combo.setItemData(
                i, SAMPLE_HIDDEN_TIP if on else None, Qt.ToolTipRole)
        if on and self.topo_combo.currentIndex() >= 3:
            # remember the check metric so Exit sample puts it back
            self._metric_before_sample = self.topo_combo.currentIndex()
            self.topo_combo.setCurrentIndex(0)
        elif not on:
            back = getattr(self, '_metric_before_sample', None)
            self._metric_before_sample = None
            if (back is not None and back < self.topo_combo.count()
                    and self.topo_combo.model().item(back) is not None
                    and self.topo_combo.model().item(back).isEnabled()):
                self.topo_combo.setCurrentIndex(back)

    def set_facts(self, channel, qc_row, event_type):
        """The SELECTED CHANNEL facts line (two lines, facts only; the first
        line only while the sample is reviewed)."""
        if not channel or qc_row is None:
            self.facts_line.setText('')
            return
        ev = _er.EVENT_PLURAL.get(event_type, event_type)
        marks = {'hard': '× hard', 'soft': '▲ soft'}
        amp = marks.get(str(qc_row.get('flag') or ''), '✓ ok')
        chk = marks.get(str(qc_row.get('checks_flag') or ''), '✓ ok')
        try:
            n = int(qc_row.get('n') or 0)
        except (TypeError, ValueError):
            n = 0
        ob = fmt_check('pct_off_band', qc_row.get('pct_off_band'))
        first = (f"{channel} · {str(qc_row.get('region') or '').capitalize()}"
                 f" · {n:,} {ev}")
        if getattr(self, '_sample_mode', False):
            self.facts_line.setText(first)
            return
        self.facts_line.setText(
            f"{first}\namp {amp} · checks {chk} · off-band {ob}")

    def set_event_type(self, event_type):
        """Set the event type used in the topo title + global-worst header."""
        self._event_type = str(event_type or 'slow_wave')
        self.flagged.event_type = self._event_type
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
        self.ring_items, self.ring_labels = [], []
        self.topo_caption.setText('')
        if hasattr(self, 'topo_excluded_lbl'):
            self.topo_excluded_lbl.setVisible(False)
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
            for val in self._z_labels.values():
                val.setText("—")
        self._last_channel = (channel, df_slice, qc_row)
        if getattr(self, '_sample_mode', False):
            return                  # ranking hidden while the sample is reviewed
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
        self._last_global = (rows, None)
        self.global_worst_list.clear()
        if getattr(self, '_sample_mode', False):
            return                  # hidden while the sample is reviewed
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

    MAX_RING_LABELS = 12

    def _draw_rings(self, metric, pts, chans):
        """Rings (hard solid 2 px, soft dashed 1.5 px, 1.8× the dot) and up
        to 12 name labels on check metrics; none on density / amplitude."""
        self.ring_items, self.ring_labels = [], []
        ctx = self._caption_ctx
        if metric not in CHECK_COLUMNS:
            self.topo_caption.setText('')
            return
        x = (pts[:, 0] - pts[:, 0].min()) / max(np.ptp(pts[:, 0]), 1e-9) * 80
        y = (pts[:, 1] - pts[:, 1].min()) / max(np.ptp(pts[:, 1]), 1e-9) * 80
        pos = {c: (float(a), float(b)) for c, a, b in zip(chans, x, y)}
        ringed = [(c, f, z) for c, (f, z) in self._check_flags.items()
                  if c in pos and c not in self._dropped]
        if getattr(self, '_sample_mode', False):
            ringed = []          # no channel flags during sample review
        for c, f, _z in ringed:
            pen = (pg.mkPen(THEME['text'], width=2) if f == 'hard' else
                   pg.mkPen(THEME['warn'], width=1.5, style=Qt.DashLine))
            ring = pg.ScatterPlotItem(x=[pos[c][0]], y=[pos[c][1]], size=9,
                                      symbol='o', brush=None, pen=pen,
                                      data=[(c, 0.0)])
            ring.flag = f
            ring.channel = c
            ring.sigClicked.connect(self._on_topo_click)
            self.topo.addItem(ring)
            self.ring_items.append(ring)
        labelled = sorted(ringed, key=lambda t: -float(t[2] or 0))
        for c, _f, _z in labelled[:self.MAX_RING_LABELS]:
            t = pg.TextItem(c, color=THEME['text'], anchor=(0, 1))
            f = t.textItem.font()
            f.setPixelSize(10)
            t.setFont(f)
            t.setPos(pos[c][0] + 1.5, pos[c][1] + 1.5)
            self.topo.addItem(t)
            self.ring_labels.append(t)
        cap = ''
        if ctx is not None:
            et, stages_text, band, min_dur = ctx
            cap = _er.topo_caption(metric, et, stages_text, band, min_dur)
        extra = len(ringed) - self.MAX_RING_LABELS
        if extra > 0:
            cap += f" · {extra} more flagged channels ringed, not labelled"
        self.topo_caption.setText(cap)

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
        pts, vals, chans, hollow = [], [], [], []
        for _, r in qc_df.iterrows():
            ch = str(r['channel'])
            if ch not in self._coords:
                continue
            if bool(r.get('excluded', False)):
                hollow.append(ch)       # drawn hollow, never interpolated
            elif pd.notna(r.get(metric)):
                pts.append(self._coords[ch]); vals.append(float(r[metric]))
                chans.append(ch)
        self.excluded_spots = None
        self.topo_input_channels = list(chans)
        # Interpolate only channels present in BOTH coords and the QC frame,
        # and not excluded.
        if len(pts) < 4:
            self._render_topo_empty()
            return
        self.topo_excluded_lbl.setVisible(bool(hollow))
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
        if hollow:
            hp = np.asarray([self._coords[c] for c in hollow], dtype=float)
            self.excluded_spots = pg.ScatterPlotItem(
                x=(hp[:, 0] - pts[:, 0].min()) / max(np.ptp(pts[:, 0]), 1e-9) * 80,
                y=(hp[:, 1] - pts[:, 1].min()) / max(np.ptp(pts[:, 1]), 1e-9) * 80,
                size=6, brush=None, pen=pg.mkPen(THEME['text_3'], width=1),
                data=[(c, float('nan')) for c in hollow], hoverable=True,
                tip=lambda x, y, data: f"{data[0]}\nexcluded")
            self.excluded_spots.sigClicked.connect(self._on_topo_click)
            self.topo.addItem(self.excluded_spots)
        self._draw_rings(metric, pts, chans)
        # Colorbar / legend with numeric min/max endpoints.
        try:
            self._colorbar = pg.ColorBarItem(
                values=(vmin, vmax), colorMap=cmap, width=12,
                interactive=False)
            self._colorbar.setImageItem(img, insert_in=self.topo.plotItem)
        except Exception:
            self._colorbar = None


class WrapRowLayout(QtWidgets.QLayout):
    """One row of items that wraps onto further rows when the widget is too
    narrow, so the row's minimum width is its widest single item.

    The last ``tail`` items are pushed to the right edge of the line they
    end up on when that is the last line (the place a stretch would give
    them in a plain row). Hidden widgets take no space.
    """

    def __init__(self, parent=None, spacing=8, tail=0):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        self._items = []
        self._tail = int(tail)

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
        return self._do_layout(QtCore.QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, True)

    def sizeHint(self):
        """Everything on one line."""
        shown = [it for it in self._items if not it.isEmpty()]
        w = sum(it.sizeHint().width() for it in shown) \
            + self.spacing() * max(0, len(shown) - 1)
        h = max([it.sizeHint().height() for it in shown] or [0])
        return QtCore.QSize(w, h)

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            if not it.isEmpty():
                size = size.expandedTo(it.minimumSize())
        return size

    def _do_layout(self, rect, apply):
        sp = self.spacing()
        lines, line, x = [], [], rect.x()
        for i, it in enumerate(self._items):
            if it.isEmpty():
                continue
            w = max(min(it.sizeHint().width(), rect.width()),
                    it.minimumSize().width())
            if line and x + w > rect.right() + 1:
                lines.append(line)
                line, x = [], rect.x()
            line.append((i, it, x, w))
            x += w + sp
        if line:
            lines.append(line)
        y = rect.y()
        first_tail = len(self._items) - self._tail
        for n, ln in enumerate(lines):
            h = max(it.sizeHint().height() for _i, it, _x, _w in ln)
            shift = 0
            if n == len(lines) - 1 and self._tail:
                end = ln[-1][2] + ln[-1][3]
                shift = max(0, rect.right() + 1 - end)
            if apply:
                for i, it, x0, w in ln:
                    dx = shift if i >= first_tail else 0
                    hh = it.sizeHint().height()
                    it.setGeometry(QtCore.QRect(x0 + dx, y + (h - hh) // 2,
                                                w, hh))
            y += h + sp
        return max(0, y - sp - rect.y())


class _FrozenColumnTable(QTableView):
    """A table whose first column (Channel) stays at the left edge while the
    other columns scroll horizontally (R5.1). The frozen column is a second
    view over the same model and selection model, laid over the first
    column, so its row height, selection and sort are the table's own."""

    MIN_FROZEN_PX = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frozen = QTableView(self)
        self.frozen.setFocusPolicy(Qt.NoFocus)
        self.frozen.verticalHeader().hide()
        self.frozen.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.frozen.setSelectionMode(QAbstractItemView.SingleSelection)
        self.frozen.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Fixed)
        self.frozen.horizontalHeader().setSortIndicatorShown(True)
        self.frozen.setStyleSheet(
            f"QTableView{{border:0;border-right:1px solid "
            f"{THEME['border_strong']};}}")
        self.viewport().stackUnder(self.frozen)
        self.horizontalHeader().sectionResized.connect(self._on_resized)
        self.verticalScrollBar().valueChanged.connect(
            self.frozen.verticalScrollBar().setValue)
        self.frozen.verticalScrollBar().valueChanged.connect(
            self.verticalScrollBar().setValue)
        self.horizontalHeader().sortIndicatorChanged.connect(
            lambda c, o: self.frozen.horizontalHeader().setSortIndicator(
                c if c == 0 else -1, o))
        # a click on the frozen header sorts the table by Channel
        self.frozen.horizontalHeader().sectionClicked.connect(
            self._on_frozen_header)
        # The frozen view takes no focus, so a click in it gives click focus
        # to this table (its nearest focusable ancestor): arrow keys and F
        # then act on the clicked channel (tested)

    def setModel(self, model):
        super().setModel(model)
        self.frozen.setModel(model)
        self.frozen.setSelectionModel(self.selectionModel())
        for c in range(1, model.columnCount()):
            self.frozen.setColumnHidden(c, True)
        model.modelReset.connect(self.fit_frozen)
        model.layoutChanged.connect(self._sync_rows)
        self.fit_frozen()
        self.frozen.show()

    def setAlternatingRowColors(self, on):
        super().setAlternatingRowColors(on)
        self.frozen.setAlternatingRowColors(on)

    def _on_frozen_header(self, col):
        hdr = self.horizontalHeader()
        order = (Qt.DescendingOrder if hdr.sortIndicatorSection() == 0
                 and hdr.sortIndicatorOrder() == Qt.AscendingOrder
                 else Qt.AscendingOrder)
        self.sortByColumn(0, order)

    def fit_frozen(self):
        """Channel column width: the longest label, at least 80 px."""
        self.resizeColumnToContents(0)
        self.setColumnWidth(0, max(self.MIN_FROZEN_PX, self.columnWidth(0)))
        self._sync_rows()

    def _sync_rows(self, *_):
        self.frozen.verticalHeader().setDefaultSectionSize(
            self.verticalHeader().defaultSectionSize())
        self._update_frozen()

    def _on_resized(self, col, _old, new):
        if col == 0:
            self.frozen.setColumnWidth(0, new)
            self._update_frozen()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_frozen()

    def _update_frozen(self):
        self.frozen.setColumnWidth(0, self.columnWidth(0))
        fw = self.frameWidth()
        self.frozen.setGeometry(
            fw, fw, self.columnWidth(0),
            self.viewport().height() + self.horizontalHeader().height())

    def scrollTo(self, index, hint=QAbstractItemView.EnsureVisible):
        """Keep the frozen column out of horizontal scrolling."""
        if index.column() > 0:
            super().scrollTo(index, hint)


class _ShrinkCombo(QComboBox):
    """A combo box as wide as its longest item when there is room, that may
    shrink to :attr:`MIN_PX` (the popup list keeps its full width)."""

    MIN_PX = 140

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.view().setTextElideMode(Qt.ElideNone)
        self.view().setMinimumWidth(240)

    def minimumSizeHint(self):
        return QtCore.QSize(self.MIN_PX, super().minimumSizeHint().height())


def _row_group(*widgets):
    """Widgets that stay together when a :class:`WrapRowLayout` wraps."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w)
    return box


class ChannelQCWidget(QWidget):
    """Tab 0 — primary QC surface (UX spec revision 3, section 1): control
    row (event type, Stage, Show, Sort, header counts), the channel table, a
    bottom action bar for the selected channel (R5.1: eight columns, no
    footer; the rules are in the Amp flag and flagged-list tooltips)."""

    channelSelected = pyqtSignal(str)
    requestDrill = pyqtSignal(str)          # Open in Epochs
    requestExclude = pyqtSignal(str, bool)  # Exclude channel toggle
    addToRedetect = pyqtSignal(str)
    requestQueueAllHard = pyqtSignal()
    loadMontageRequested = pyqtSignal()
    stageChanged = pyqtSignal(str)         # stage text or COMBINED
    exitSampleRequested = pyqtSignal()     # banner button
    exportRerunRequested = pyqtSignal()    # queue hint link
    checksLinkActivated = pyqtSignal()     # "{c} checks flagged" link

    COMBINED = '__all__'
    COLUMN_HEADER = 'Column header'
    SORT_SETTING = 'review/channels_sort'
    STATUS_COL_PX = 190

    def _fit_status_column(self):
        """Status keeps at least :attr:`STATUS_COL_PX` and takes the width
        the other columns leave free."""
        hdr = self.table.horizontalHeader()
        last = _QC_COL_INDEX['verdict']
        others = hdr.length() - hdr.sectionSize(last)
        spare = self.table.viewport().width() - others
        hdr.blockSignals(True)
        self.table.setColumnWidth(last, max(self.STATUS_COL_PX, spare))
        hdr.blockSignals(False)
        hdr.viewport().update()

    def __init__(self, parent=None):
        super().__init__(parent)
        ll = QVBoxLayout(self)
        # sample-mode banner (R4.1): only while a review sample is active
        self.sample_banner = QtWidgets.QFrame()
        self.sample_banner.setObjectName('sampleBanner')
        self.sample_banner.setStyleSheet(
            f"#sampleBanner{{background:{THEME['accent_soft']};"
            f"border-left:3px solid {THEME['accent']};}}")
        bl = QHBoxLayout(self.sample_banner)
        bl.setContentsMargins(10, 4, 6, 4)
        self.banner_lbl = QLabel(SAMPLE_BANNER_TEXT)
        self.banner_lbl.setStyleSheet(
            f"color:{THEME['text']};font-size:12px;background:transparent;")
        bl.addWidget(self.banner_lbl)
        bl.addStretch()
        self.banner_exit_btn = QPushButton("Exit sample")
        self.banner_exit_btn.clicked.connect(
            lambda: self.exitSampleRequested.emit())
        bl.addWidget(self.banner_exit_btn)
        self.sample_banner.setVisible(False)
        ll.addWidget(self.sample_banner)
        # control row: wraps when the tab is narrow, so it never pushes the
        # right dock out of the window
        self.control_row = QWidget()
        bar = WrapRowLayout(self.control_row)
        self.evt_combo = QComboBox()
        self.evt_combo.addItems(['slow_wave', 'spindle', 'k_complex', 'pac'])
        bar.addWidget(_row_group(QLabel("Event type"), self.evt_combo))
        self.stage_box = QWidget()
        self.stage_lay = QHBoxLayout(self.stage_box)
        self.stage_lay.setContentsMargins(0, 0, 0, 0)
        self.stage_lay.setSpacing(0)
        self.stage_group = QtWidgets.QButtonGroup(self)
        self.stage_group.setExclusive(True)
        self.stage_group.buttonClicked.connect(self._on_stage_clicked)
        self.stage_box.setToolTip(_er.STAGE_TOOLTIP)
        self.stage_buttons = {}
        bar.addWidget(_row_group(QLabel("Stage"), self.stage_box))
        self.show_combo = _ShrinkCombo()
        for item in _er.SHOW_ITEMS:
            self.show_combo.addItem(f"{item} (0)", item)
        bar.addWidget(_row_group(QLabel("Show"), self.show_combo))
        self.sort_combo = _ShrinkCombo()
        for label, _k, _d in _er.SORT_ITEMS:
            self.sort_combo.addItem(label)
        bar.addWidget(_row_group(QLabel("Sort"), self.sort_combo))
        ll.addWidget(self.control_row)
        # header count line: its own row under the control row, left-aligned;
        # "{c} checks flagged" links to the dock's flagged-channel list
        self.counts_lbl = _LinkLabel("")
        self.counts_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.counts_lbl.setTextFormat(Qt.RichText)
        self.counts_lbl.linkActivated.connect(
            lambda *_: self.checksLinkActivated.emit())
        self.counts_lbl.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:12px;"
            f"color:{THEME['text_2']};")
        ll.addWidget(self.counts_lbl)

        self.model = ChannelQCModel()
        self.proxy = _QCSortProxy()
        self.proxy.setSourceModel(self.model)
        self.table = _FrozenColumnTable()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        # Status (the last column) fits "× excluded · ↻ re-detect" and takes
        # any spare width: see _fit_status_column
        self.table.setColumnWidth(_QC_COL_INDEX['verdict'],
                                  self.STATUS_COL_PX)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().currentRowChanged.connect(self._on_row)
        self._status_delegate = _StatusDelegate(self.table)
        self.table.setItemDelegateForColumn(_QC_COL_INDEX['verdict'],
                                            self._status_delegate)
        self.table.horizontalHeader().sortIndicatorChanged.connect(
            self._on_header_sort)
        ll.addWidget(self.table)
        self.empty_lbl = QLabel("", self.table.viewport())
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setStyleSheet(f"color:{THEME['text_3']};")
        self.empty_lbl.setVisible(False)

        # --- bottom action bar (selected channel) ----------------------
        self._sel_lbl = QLabel("—")
        self._sel_lbl.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-weight:600;")
        self.btn_open = QPushButton("Open in Epochs")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self._drill)
        self.btn_drill = self.btn_open           # older name
        # one toggle: checked = the selected channel is excluded
        self.btn_exclude = QPushButton("Exclude channel")
        self.btn_exclude.setCheckable(True)
        self.btn_exclude.setToolTip(EXCLUDE_TIP)
        self.btn_exclude.clicked.connect(self._exclude_clicked)
        set_exclude_button(self.btn_exclude, False)
        fit_checked_width(self.btn_exclude,
                          ['Exclude channel', 'Include channel'])
        self.btn_redetect = QPushButton("Add to re-detect queue")
        self.btn_redetect.setCheckable(True)    # checked = queued
        self.btn_redetect.clicked.connect(self._redetect)
        fit_checked_width(self.btn_redetect, ['Add to re-detect queue',
                                              'Remove from re-detect queue'])
        self.btn_redetect.setToolTip(REDETECT_ADD_TIP)
        # queue hint: a flat link, only while something is queued (R4.5)
        self.queue_link = QPushButton("")
        self.queue_link.setFlat(True)
        self.queue_link.setCursor(Qt.PointingHandCursor)
        self.queue_link.setStyleSheet(
            f"border:0;background:transparent;color:{THEME['accent']};"
            "text-decoration:underline;")
        self.queue_link.clicked.connect(
            lambda: self.exportRerunRequested.emit())
        self.queue_link.setVisible(False)
        self.btn_queue_hard = QPushButton("Queue all HARD")
        self.btn_queue_hard.setToolTip(QUEUE_HARD_TIP)
        self.btn_queue_hard.clicked.connect(
            lambda: self.requestQueueAllHard.emit())
        # R4.5 order: name · Open · Exclude · queue toggle · (space) ·
        # queue link · Queue all HARD. Wraps when the tab is narrow; the
        # last two sit at the right edge of their line.
        self.action_row = QWidget()
        btns = WrapRowLayout(self.action_row, tail=2)
        btns.addWidget(_row_group(self._sel_lbl, self.btn_open,
                                  self.btn_exclude))
        btns.addWidget(self.btn_redetect)
        btns.addWidget(self.queue_link)
        btns.addWidget(self.btn_queue_hard)
        ll.addWidget(self.action_row)

        self._qc_full = None      # full computed df for current event type
        self._events_slice = None  # montage-wide events df (current event type)
        self._verdicts = {}
        self._redetect_ref = set()  # window's queue (for button label state)
        self._recorded = True
        self._sorting_from_combo = False
        self.show_combo.currentIndexChanged.connect(
            lambda *_: self._apply_show_filter())
        # the Sort the user chose last time; one on a column that no longer
        # exists opens as the default (R5.1)
        self.sort_combo.setCurrentText(_er.sort_label_or_default(str(
            _review_settings().value(self.SORT_SETTING, _er.DEFAULT_SORT)
            or _er.DEFAULT_SORT)))
        self.sort_combo.currentIndexChanged.connect(self._on_sort_combo)
        self._on_sort_combo()

    def current_event_type(self):
        return self.evt_combo.currentText()

    # ---- stage toggle ------------------------------------------------
    def set_stages(self, stages, current=None):
        """One checkable button per run stage plus the combined button
        (``NREM2 + NREM3``); a one-stage run shows one checked, disabled
        button. ``current`` is a stage or :attr:`COMBINED` (default)."""
        for b in list(self.stage_group.buttons()):
            self.stage_group.removeButton(b)
            self.stage_lay.removeWidget(b)
            b.deleteLater()
        self.stage_buttons = {}
        stages = list(stages or [])
        if len(stages) == 1:
            items = [(stages[0], self.COMBINED)]
        else:
            items = [(st, st) for st in stages]
            if stages:
                items.append((' + '.join(stages), self.COMBINED))
        for text, key in items:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setProperty('stage_key', key)
            fit_checked_width(b)
            self.stage_group.addButton(b)
            self.stage_lay.addWidget(b)
            self.stage_buttons[key] = b
        key = current if current in self.stage_buttons else self.COMBINED
        if key in self.stage_buttons:
            self.stage_buttons[key].setChecked(True)
        self._apply_stage_lock()

    def current_stage(self):
        b = self.stage_group.checkedButton()
        return b.property('stage_key') if b is not None else self.COMBINED

    def stage_text(self):
        b = self.stage_group.checkedButton()
        return b.text() if b is not None else ''

    def _on_stage_clicked(self, b):
        self.stageChanged.emit(str(b.property('stage_key')))

    # ---- check context -------------------------------------------------
    def set_checks_context(self, recorded, method='', ratio=True,
                           event_type='spindle', tips=None, pending=False,
                           figures_off=False, medians=None, bounds=None):
        """Whether the run stored event figures (the header line's checks
        count). The table itself has no check columns (R5.1)."""
        self._recorded = bool(recorded)
        self.model._checks = {'recorded': bool(recorded),
                              'method': str(method or ''),
                              'ratio': bool(ratio), 'pending': bool(pending),
                              'figures_off': bool(figures_off),
                              'medians': dict(medians or {}),
                              'bounds': bounds, 'tips': {}}
        self._update_counts()

    def set_flag_limits(self, hard_z, soft_z):
        """The live z limits in the ``Amp flag`` header tooltip."""
        self.model.amp_flag_tip = _er.amp_flag_tooltip(hard_z, soft_z)
        self.model.headerDataChanged.emit(Qt.Horizontal, 0,
                                          len(_QC_COLS) - 1)

    def set_data(self, qc_df, events_df, verdicts, redetect_set=None):
        keep = self._current_channel()     # a refresh keeps the selection
        self._qc_full = qc_df
        self._events_slice = events_df
        self._verdicts = verdicts or {}
        if redetect_set is not None:
            self._redetect_ref = redetect_set
        self.model.set_data(qc_df, self._verdicts, self._redetect_ref,
                            self.current_event_type())
        self._apply_show_filter()
        self._update_counts()
        if keep is not None and not self.table.currentIndex().isValid():
            self.select_channel(keep)
        if self.table.model().rowCount() and not self.table.currentIndex().isValid():
            self.table.selectRow(0)
        self._update_action_state()

    def _frame(self):
        """The model's frame (verdicts merged)."""
        return self.model.df

    def set_sample_mode(self, on, channel_order=None):
        """Live sample review: the banner shows, the Stage toggle is
        disabled and the header count reads ``checks hidden``; Exit sample
        brings them back. ``channel_order`` is accepted for older callers
        and not used."""
        on = bool(on)
        self._sample_mode = on
        self.sample_banner.setVisible(on)
        self._apply_stage_lock()
        self._update_counts()
        self._apply_show_filter()

    def _apply_stage_lock(self):
        """Stage buttons: disabled with the R4.0 tooltip in sample mode (the
        checked one still shows as checked); a one-stage run's single button
        stays disabled either way."""
        sample = getattr(self, '_sample_mode', False)
        single = len(self.stage_buttons) == 1
        for b in self.stage_group.buttons():
            b.setEnabled(not sample and not single)
            b.setToolTip(SAMPLE_STAGE_TIP if sample else '')
        self.stage_box.setToolTip(SAMPLE_STAGE_TIP if sample
                                  else _er.STAGE_TOOLTIP)

    def _update_counts(self):
        df = self._frame()
        sample = getattr(self, '_sample_mode', False)
        if df is None or len(df) == 0:
            self.counts_lbl.setText("")
            self.counts_lbl.setToolTip('')
        else:
            line = _er.header_count_line(df, self._recorded, sample=sample)
            head, sep, tail = line.partition(' · ')
            if head.endswith(' checks flagged'):
                # the checks count is a link to the dock's list (R5.1)
                self.counts_lbl.setText(
                    f"<a href='checks' style='color:{THEME['accent']}'>"
                    f"{head}</a>{sep}{tail}")
                self.counts_lbl.setToolTip(_er.CHECKS_LINK_TIP)
            else:
                self.counts_lbl.setText(line)
                self.counts_lbl.setToolTip('')
        # Show combo counts (over all channels)
        self.show_combo.blockSignals(True)
        for i, item in enumerate(_er.SHOW_ITEMS):
            n = int(_er.show_mask(df, item, sample=sample).sum()) \
                if df is not None and len(df) else 0
            self.show_combo.setItemText(i, f"{item} ({n})")
        self.show_combo.blockSignals(False)

    def _update_action_state(self):
        ch = self._current_channel()
        self._sel_lbl.setText(ch or '—')
        en = ch is not None
        for b in (self.btn_open, self.btn_exclude, self.btn_redetect):
            b.setEnabled(en)
        set_exclude_button(self.btn_exclude, en and self._verdicts.get(
            ch, '') in EXCLUDED_VERDICTS, self.current_event_type())
        queued = en and ch in self._redetect_ref
        self.btn_redetect.setChecked(bool(queued))
        self.btn_redetect.setText("Remove from re-detect queue" if queued
                                  else "Add to re-detect queue")
        self.btn_redetect.setToolTip(REDETECT_REMOVE_TIP if queued
                                     else REDETECT_ADD_TIP)
        nq = len(self._redetect_ref)
        self.queue_link.setText(f"{nq} queued · Export re-run package…")
        self.queue_link.setVisible(nq > 0)
        n_hard = 0 if self._qc_full is None or len(self._qc_full) == 0 \
            else int((self._qc_full['flag'] == 'hard').sum())
        self.btn_queue_hard.setText(f"Queue all HARD ({n_hard})")
        self.btn_queue_hard.setEnabled(n_hard > 0)

    # ---- Show and Sort ---------------------------------------------------
    def show_item(self):
        return self.show_combo.currentData() or 'All channels'

    def _apply_show_filter(self, *_):
        df = self._frame()
        item = self.show_item()
        if item == 'All channels' or df is None or not len(df):
            self.proxy.keep = None
        else:
            self.proxy.keep = set(df.loc[_er.show_mask(
                df, item, sample=getattr(self, '_sample_mode', False)).values,
                'channel'].astype(str))
        self.proxy.invalidateFilter()
        empty = self.proxy.rowCount() == 0 and df is not None and len(df) > 0
        self.empty_lbl.setText(f'No channels match "{item}".')
        self.empty_lbl.setVisible(empty)
        if empty:
            self.empty_lbl.resize(self.table.viewport().size())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.empty_lbl.resize(self.table.viewport().size())
        self._fit_status_column()

    def _sort_spec(self, label):
        for lab, key, desc in _er.SORT_ITEMS:
            if lab == label:
                return key, desc
        return None

    def _on_sort_combo(self, *_):
        label = self.sort_combo.currentText()
        if label == self.COLUMN_HEADER:
            return
        spec = self._sort_spec(label)
        if spec is None:
            return
        key, desc = spec
        _review_settings().setValue(self.SORT_SETTING, label)
        i = self.sort_combo.findText(self.COLUMN_HEADER)
        if i >= 0:
            self.sort_combo.blockSignals(True)
            self.sort_combo.removeItem(i)
            self.sort_combo.blockSignals(False)
        self._sorting_from_combo = True
        self.table.sortByColumn(_QC_COL_INDEX[key], Qt.DescendingOrder
                                if desc else Qt.AscendingOrder)
        self._sorting_from_combo = False

    def _on_header_sort(self, col, order):
        if self._sorting_from_combo or col < 0 or col >= len(_QC_COLS):
            return                  # no sorted column (e.g. channel order)
        key = _QC_COLS[col][0]
        desc = order == Qt.DescendingOrder
        match = next((lab for lab, k, d in _er.SORT_ITEMS
                      if k == key and d == desc), None)
        self.sort_combo.blockSignals(True)
        if match is not None:
            i = self.sort_combo.findText(self.COLUMN_HEADER)
            if i >= 0:
                self.sort_combo.removeItem(i)
            self.sort_combo.setCurrentText(match)
        else:
            if self.sort_combo.findText(self.COLUMN_HEADER) < 0:
                self.sort_combo.addItem(self.COLUMN_HEADER)
            self.sort_combo.setCurrentText(self.COLUMN_HEADER)
        self.sort_combo.blockSignals(False)

    def visible_channels(self):
        """Channels in table order (Show filter and sort applied)."""
        out = []
        for prow in range(self.proxy.rowCount()):
            src = self.proxy.mapToSource(self.proxy.index(prow, 0))
            out.append(self.model.channel_at(src.row()))
        return out

    # ---- selection and actions --------------------------------------------
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

    def _exclude_clicked(self, checked):
        ch = self._current_channel()
        if ch:
            self.requestExclude.emit(ch, bool(checked))

    def events_slice_for(self, ch):
        if self._events_slice is not None and len(self._events_slice):
            return self._events_slice[self._events_slice['channel'] == ch]
        return None

    def _drill(self):
        ch = self._current_channel()
        if ch:
            self.requestDrill.emit(ch)

    def select_channel(self, ch):
        """Select the row for ``ch`` in the QC table. No-op if not shown."""
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


class _BrushViewBox(pg.ViewBox):
    """The raw trace's view box: a left-button drag (no modifier) draws the
    exclusion brush; ``sigBrush(t0, t1, finished)``. Clicks are left to the
    scene's ``sigMouseClicked`` (event and exclusion selection)."""

    sigBrush = pyqtSignal(float, float, bool)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.LeftButton or ev.modifiers() != Qt.NoModifier:
            ev.ignore()
            return
        ev.accept()
        p0 = self.mapSceneToView(ev.buttonDownScenePos())
        p1 = self.mapSceneToView(ev.scenePos())
        t0, t1 = sorted([float(p0.x()), float(p1.x())])
        self.sigBrush.emit(t0, t1, bool(ev.isFinish()))


def _review_settings():
    """The review GUI's ``QSettings`` (org ``turtlewave``, app
    ``eeg_review_gui``, the names ``main()`` pins), in
    ``QSettings.defaultFormat()``: native in the GUI, and whatever a test
    sets (``QSettings(org, app)`` would ignore the default format and always
    write the user's real preferences)."""
    return QtCore.QSettings(QtCore.QSettings.defaultFormat(),
                            QtCore.QSettings.UserScope, 'turtlewave',
                            'eeg_review_gui')


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
        # closed on first use (R4.11); after that the saved state
        self._open = _setting_bool(settings_key, False)
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
    """``EEGDetailWidget`` drawing the selected event on its neighbours
    (spec R4.10): target on top in the accent colour, one shared scale, the
    selected event as two thin accent lines through every row, and each
    neighbour's own same-type events as a 3 px bar under its trace. No
    shaded band or block covers the signal."""

    BAR_PX = 3

    def plot_neighbours(self, ts, data, labels, event_span, scale,
                        neighbour_spans=None, detected=None):
        """``data`` rows follow ``labels``; row 0 is the target. ``scale`` is
        the shared half-range: a sample beyond it is clipped at its row
        edge."""
        self.clear()
        self.channel_curves, self.channel_labels, self.event_items = [], [], []
        self.sel_lines, self.bar_items = [], []
        ts = np.asarray(ts, dtype=float)
        n = len(labels)
        step = 2.0 * scale
        x0, x1 = float(ts[0]), float(ts[-1])
        bar_pen = pg.mkPen(THEME['text_2'], width=self.BAR_PX)
        bar_pen.setCapStyle(Qt.FlatCap)
        for i, lab in enumerate(labels):
            y = -i * step
            pen = (pg.mkPen(THEME['accent'], width=1.5) if i == 0
                   else pg.mkPen(THEME['text_2'], width=1))
            row = np.clip(np.asarray(data[i], dtype=float), -scale, scale)
            self.channel_curves.append(self.plot(ts, row + y, pen=pen))
            # the neighbour's own detections: a bar at the row's lower edge
            # (the target row has none; its event is the two lines)
            for s, e in ((neighbour_spans or {}).get(lab, []) if i else []):
                a, b = max(float(s), x0), min(float(e), x1)
                if b <= a:
                    continue
                yb = y - 0.94 * scale
                bar = pg.PlotCurveItem(x=[a, b], y=[yb, yb], pen=bar_pen)
                self.addItem(bar)
                self.bar_items.append({'label': lab, 'span': (a, b),
                                       'item': bar})
                self.event_items.append(bar)
            text = lab + (' · detected' if detected and lab in detected
                          else '')
            t = pg.TextItem(text, anchor=(0, 0.5),
                            color=THEME['accent'] if i == 0 else THEME['text_2'])
            t.setPos(x0, y + 0.6 * scale)
            self.addItem(t, ignoreBounds=True)
            self.channel_labels.append(t)
        for x in event_span:
            ln = pg.InfiniteLine(pos=float(x), angle=90, movable=False,
                                 pen=pg.mkPen(THEME['accent'], width=1))
            self.addItem(ln, ignoreBounds=True)
            self.sel_lines.append(ln)
        self.setXRange(x0, x1, padding=0)
        self.setYRange(-(n - 1) * step - scale, scale, padding=0)
        self._last_channels = list(labels)

    def row_labels(self):
        return [t.toPlainText() for t in self.channel_labels]

    def bars_for(self, label):
        """Bar spans drawn under the row whose label starts with ``label``."""
        return [b['span'] for b in getattr(self, 'bar_items', [])
                if b['label'] == label]


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
        self.legend = QLabel("")            # R4.0 legend, live half-range
        self.legend.setStyleSheet(f"color:{THEME['text_3']};font-size:11px;")
        self.legend.setWordWrap(True)
        self.legend.setVisible(False)
        self.body_lay.addWidget(self.legend)
        self.half_range = None              # applied shared half-range, µV
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
        self.legend.setVisible(False)
        self.half_range = None

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
        # one scale for every row, from the data in the window (R4.10)
        scale = _er.neighbour_half_range(
            data[keep], event_type, self.filter_chk.isChecked())
        self.half_range = scale
        self.legend.setText(_er.neighbours_legend(scale))
        self.empty.setVisible(False)
        self.plot.setVisible(True)
        self.legend.setVisible(True)
        self.plot.plot_neighbours(ts, data[keep], row_labels, (s, e), scale,
                                  spans, detected)


#: Physiology rows: kind -> (row label, filter). Every row is scaled from
#: its own data in the epoch (spec R4.9); there are no fixed scales.
PHYSIO_ROWS = {'eog': ('EOG', (0.3, 15.0)),
               'emg': ('Chin EMG', (10.0, None)),
               'ecg': ('ECG', None)}


class PhysioStrip(_Collapsible):
    """``▾ PHYSIOLOGY (n)`` under Neighbours (spec section 9, R4.9): typed
    EOG, chin EMG and ECG, X-linked to the raw trace, read once per epoch,
    each row scaled to its own signal. The scale label names a unit only
    when the file states one (``panel.channel_unit``)."""

    ROW_PX = 44

    def __init__(self, panel):
        super().__init__('PHYSIOLOGY (0)', 'review/physio_open', panel)
        self.panel = panel
        self.rows = []          # (kind, channel, PlotWidget)
        self.legend = QLabel("")
        self.legend.setStyleSheet("color:#888888;font-size:11px;")
        self.body_lay.addWidget(self.legend)
        self.n_reads = 0
        self._window = None
        self.row_scales = {}    # channel -> (centre, half-range, label text)
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
            w.setFixedHeight(self.ROW_PX)
            w.setMouseEnabled(False, False)
            w.setMenuEnabled(False)
            w.hideButtons()
            w.hideAxis('bottom')
            w.getAxis('left').setStyle(showValues=False)
            w.getAxis('left').setWidth(self.panel.AXIS_W)
            w.setXLink(self.panel.raw_plot)
            self.body_lay.insertWidget(self.body_lay.count() - 1, w)
            self.rows.append((kind, ch, w))
        self.row_scales = {}
        self._update_legend()
        self.set_title(f"PHYSIOLOGY ({len(self.rows)})")
        self.setVisible(bool(self.rows))

    def _unit(self, channel):
        fn = getattr(self.panel, 'channel_unit', None)
        return fn(channel) if callable(fn) else None

    def _update_legend(self):
        kinds = [k for k, _c, _w in self.rows]
        no_unit = {k for k, c, _w in self.rows if not self._unit(c)}
        self.legend.setText(_er.physio_legend(kinds, no_unit)
                            if self.rows else '')

    def row_titles(self):
        return [f"{PHYSIO_ROWS[k][0]} · {c}" for k, c, _w in self.rows]

    def set_selection(self, span):
        """Mark the selected event on every row with exactly two 1 px
        accent lines at its start and end; no box, fill or text covers the
        signal. ``None`` removes them. Never reads data."""
        for w, ln in getattr(self, '_sel_lines', []):
            w.removeItem(ln)
        self._sel_lines = []
        self._sel_span = span
        if span is None:
            return
        c = QtGui.QColor(THEME['accent'])
        pen = pg.mkPen(c.red(), c.green(), c.blue(), 160, width=1)
        for _k, _c, w in self.rows:
            for x in span:
                ln = pg.InfiniteLine(pos=float(x), angle=90, movable=False,
                                     pen=pen)
                w.addItem(ln)
                self._sel_lines.append((w, ln))

    def selection_lines(self, row_index):
        """The selection lines on one row (for tests)."""
        w = self.rows[row_index][2]
        return [ln for ww, ln in getattr(self, '_sel_lines', []) if ww is w]

    def show_window(self, t0, t1):
        self._window = (t0, t1)
        if self.is_open():
            self._render()

    def _render(self):
        """Read and draw the rows, then put the selection lines back (the
        read clears every row)."""
        self._render_rows()
        self.set_selection(getattr(self, '_sel_span', None))

    def _render_rows(self):
        if not self.rows or self._window is None \
                or not callable(self.panel.read_channels):
            return
        t0, t1 = self._window
        self.n_reads += 1
        out = self.panel.read_channels([c for _k, c, _w in self.rows], t0, t1)
        for k, c, w in self.rows:
            w.clear()
        self._sel_lines = []
        if not out:
            return
        ts, data, sf, got = out
        idx = {c: i for i, c in enumerate(got)}
        self.row_scales = {}
        self.row_text_items = {}       # channel -> (title item, scale item)
        for kind, ch, w in self.rows:
            label, filt = PHYSIO_ROWS[kind]
            title = f"{label} · {ch}"
            if ch not in idx:
                continue
            x = np.asarray(data[idx[ch]], dtype=float)
            if filt and filt[1]:
                x = _bandpass(x, sf, filt[0], filt[1])
            elif filt:
                x = _highpass(x, sf, filt[0])
            w.plot(ts, x, pen=pg.mkPen(THEME['text_2'], width=1))
            # the row's own scale in this epoch; a unit only if the file
            # states one (never a claimed µV)
            centre, half = _er.physio_range(x)
            w.setYRange(centre - half, centre + half, padding=0)
            scale = _er.physio_scale_label(half, self._unit(ch))
            self.row_scales[ch] = (centre, half, scale)
            # dark backing so the trace never runs through the text
            t = pg.TextItem(title, anchor=(0, 0), color=THEME['text_2'],
                            fill=pg.mkBrush(10, 10, 10, 200))
            t.setPos(t0, centre + half)
            t.setZValue(20)
            w.addItem(t, ignoreBounds=True)
            r = pg.TextItem(scale, anchor=(1, 0), color=THEME['text_2'],
                            fill=pg.mkBrush(10, 10, 10, 200))
            r.setPos(t1, centre + half)
            r.setZValue(20)
            w.addItem(r, ignoreBounds=True)
            self.row_text_items[ch] = (t, r)
        self._update_legend()


class EpochsPanel(QWidget):
    """Tab 2 — paged scored-epoch viewer with per-channel artefact triage.

    Epochs come from the annotation file's own epoch table (``epochs=`` in
    :meth:`set_channel`), so a cut recording pages through its 1-30 s epochs
    exactly as scored; :meth:`index_at` and :meth:`span` are the only way
    times and epoch ids are converted. With no annotation file a synthetic
    30 s grid over the recording is used.

    `self.plot` is the EPOCH STRIP (one bar per epoch: grey = regular events,
    red stacked on top = outliers under the rule amp > mean+3.5*sd over the
    drilled channel × event_type). Click an epoch bar to jump.

    The main view is the current epoch's window: raw + band-filtered traces
    (mouse pan/zoom disabled, X-locked to the epoch). A thin event ticker
    above the raw trace marks regular (grey) and outlier (red) events in
    the active epoch. Time is excluded by dragging a range on the raw trace
    and clicking *Exclude time range…*; a saved (hatched) range is selected
    by a click and removed with *Remove exclusion* (R5.2).

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

    excludeChannelRequested = pyqtSignal(str, bool)   # Exclude channel toggle
    statusMessage = pyqtSignal(str)                   # for the status bar
    globalArtefactConfirmed = pyqtSignal(float, float, str)
    markArtefactRequested = pyqtSignal(str, float, float)   # ch, t0, t1
    unmarkArtefactRequested = pyqtSignal(int)                # interval id
    requestChannel = pyqtSignal(str)                         # re-drill ch
    eventSelected = pyqtSignal(str)                          # event uuid
    decisionRequested = pyqtSignal(str)        # 'accept' | 'reject' | 'unsure'
    selectionCleared = pyqtSignal()            # paging / Esc dropped it
    reviewKey = pyqtSignal(str)                # '0'-'9' or 'Enter' (focus path)
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
        self._status_allowed = None   # Show events filter (None = all)
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
        # channel -> the unit the file states ('µV', 'mV', ...) or None
        self.channel_unit = None
        # uuid -> True while an outlier mark must stay hidden (an undecided
        # sample event in live sample review)
        self.outlier_hidden = None
        self._trace = None            # (ts, raw, filtered) of the epoch
        self._clip_items = []         # (plot, item) clip marks and notes
        self._clip_notes = {}         # 'raw' / 'filt' -> note text or None
        self.raw_half_range = None    # applied half-ranges, µV
        self.filt_half_range = None
        self._filt_ytop = 1.0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

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

        # centre column (R4.11): a vertical splitter. Top pane = strip,
        # navigation, raw and filtered traces, action rows. Bottom pane = a
        # scroll area holding Neighbours and Physiology, so opening a group
        # grows or scrolls the bottom pane and never squeezes the traces.
        left = QSplitter(Qt.Vertical)
        left.setChildrenCollapsible(False)
        self.v_split = left
        top_pane = QWidget()
        lc = QVBoxLayout(top_pane)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(3)        # compact: the pane's minimum height matters

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

        # ---- EPOCH STRIP (self.plot): a click pages; no range tool ------
        self.plot = pg.PlotWidget()
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
        self._ep_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('w', width=1.4))
        self._ep_cursor.setZValue(20)
        self.plot.scene().sigMouseClicked.connect(self._on_overview_click)
        # strip header exposing the active outlier rule + threshold
        self.strip_hdr = QLabel("Outlier rule: —")
        self.strip_hdr.setStyleSheet("color:#9ba6b5;font-size:11px;")
        lc.addWidget(self.strip_hdr)
        lc.addWidget(self.plot)
        self.strip_legend = QLabel(_er.STRIP_LEGEND)
        self.strip_legend.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        lc.addWidget(self.strip_legend)

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

        # ---- Show events (R5.4): the current reviewer's decisions ------
        srow = QHBoxLayout()
        show_lbl = QLabel("Show events:")
        show_lbl.setToolTip(_er.SHOW_EVENTS_TIP)
        srow.addWidget(show_lbl)
        self.show_checks = {}
        for label, _dec in _er.SHOW_EVENTS_ITEMS:
            cb = QCheckBox(label)
            cb.setChecked(True)            # all ticked at launch, not saved
            cb.setFocusPolicy(Qt.NoFocus)
            cb.toggled.connect(lambda *_: self._on_show_events())
            srow.addWidget(cb)
            self.show_checks[label] = cb
        self.shown_chip = QWidget()
        self.shown_chip.setStyleSheet(
            f"background:{THEME['accent_soft']};border:1px solid "
            f"{THEME['accent']};border-radius:9px;")
        sc_l = QHBoxLayout(self.shown_chip)
        sc_l.setContentsMargins(8, 1, 3, 1)
        self.shown_chip_lbl = QLabel("")
        self.shown_chip_lbl.setStyleSheet(
            f"border:0;color:{THEME['text']};font-size:11px;")
        sc_l.addWidget(self.shown_chip_lbl)
        self.shown_chip_x = QPushButton("✕")
        self.shown_chip_x.setFlat(True)
        self.shown_chip_x.setMaximumWidth(18)
        self.shown_chip_x.setFocusPolicy(Qt.NoFocus)
        self.shown_chip_x.setToolTip('Show all events again.')
        self.shown_chip_x.setStyleSheet(
            f"border:0;color:{THEME['text']};")
        self.shown_chip_x.clicked.connect(self.show_all_events)
        sc_l.addWidget(self.shown_chip_x)
        self.shown_chip.setVisible(False)
        srow.addWidget(self.shown_chip)
        srow.addStretch()
        self.prev_shown_btn = QPushButton("◀ previous shown")
        self.prev_shown_btn.setToolTip(
            'Previous shown event on this channel ({)')
        self.prev_shown_btn.clicked.connect(self.prev_event)
        self.next_shown_btn = QPushButton("next shown ▶")
        self.next_shown_btn.setToolTip('Next shown event on this channel (})')
        self.next_shown_btn.clicked.connect(self.next_event)
        for b in (self.prev_shown_btn, self.next_shown_btn):
            b.setFocusPolicy(Qt.NoFocus)
            b.setVisible(False)
            srow.addWidget(b)
        lc.addLayout(srow)
        self.shown_ticks = None          # strip ticks under shown epochs
        self.shown_tick_epochs = []

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
        self._raw_vb = _BrushViewBox()
        self._raw_vb.sigBrush.connect(self._on_brush_drag)
        self.raw_plot = pg.PlotWidget(viewBox=self._raw_vb)
        _theme_plot(self.raw_plot)
        self.raw_plot.setMouseEnabled(False, False)
        self.raw_plot.setMenuEnabled(False)
        self.raw_plot.hideButtons()
        self.raw_plot.setLabel('left', 'raw µV')
        self.raw_plot.setLabel('bottom', 'time (s)')
        self.raw_plot.setMinimumHeight(self.RAW_MIN_PX)
        self.filt_plot = pg.PlotWidget()
        _theme_plot(self.filt_plot)
        self.filt_plot.setMouseEnabled(False, False)
        self.filt_plot.setMenuEnabled(False)
        self.filt_plot.hideButtons()
        self.filt_plot.setLabel('left', 'filtered µV')
        self.filt_plot.setMinimumHeight(self.FILT_MIN_PX)
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
        # The one selection tool (R5.2): the unsaved brush, drawn by a drag
        # on the raw trace; its edges can then be dragged. Mirrored on the
        # filtered trace. Hidden while there is no brush.
        acc = QtGui.QColor(THEME['accent'])
        brush_fill = pg.mkBrush(acc.red(), acc.green(), acc.blue(), 50)
        brush_pen = pg.mkPen(THEME['accent'], width=1)
        self.region = pg.LinearRegionItem(brush=brush_fill, pen=brush_pen)
        self.region.setZValue(10)
        self.region_filt = pg.LinearRegionItem(brush=brush_fill,
                                               pen=brush_pen, movable=False)
        self.region_filt.setZValue(10)
        self.region_label = pg.TextItem('not saved', color=THEME['accent'],
                                        anchor=(0, 0))
        _f = self.region_label.textItem.font()
        _f.setPixelSize(10)
        self.region_label.setFont(_f)
        self.region_label.setZValue(11)
        self._brush = None             # (t0, t1) of the unsaved brush
        self._exclude_allowed, self._exclude_block_tip = True, ''
        self._sel_excl = None          # id of the selected excluded range
        self._excl_items = {}          # id -> [(plot, region item)]
        self._region_programmatic = False
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.trace_note = QLabel(
            "drag the blue brush on the trace to choose a time range to "
            "exclude · load an EEG file to enable the signal trace")
        self.trace_note.setStyleSheet("color:#6b7585;font-size:11px;")
        # header for the filtered trace: passband + its source
        self.filt_hdr = QLabel("filtered —")
        self.filt_hdr.setStyleSheet("color:#9ba6b5;font-size:11px;")
        lc.addWidget(self.raw_plot, 2)
        # trace header: the filtered band, and Full range (off by default,
        # not saved, reset when the epoch changes)
        fh = QHBoxLayout()
        fh.addWidget(self.filt_hdr)
        fh.addStretch()
        self.full_range_chk = QCheckBox("Full range")
        self.full_range_chk.setFocusPolicy(Qt.NoFocus)
        self.full_range_chk.setToolTip(
            "Show the whole signal range of this epoch instead of the "
            "range that keeps ordinary activity readable.")
        self.full_range_chk.toggled.connect(self._on_full_range)
        fh.addWidget(self.full_range_chk)
        lc.addLayout(fh)
        lc.addWidget(self.filt_plot, 2)
        lc.addWidget(self.trace_note)
        # neighbouring channels and EOG / EMG / ECG for the selected event:
        # in the bottom pane's scroll area
        self.neighbours = NeighboursGroup(self)
        self.physio = PhysioStrip(self)
        groups = QWidget()
        gl = QVBoxLayout(groups)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(2)
        gl.addWidget(self.neighbours)
        gl.addWidget(self.physio)
        gl.addStretch()
        self.groups_scroll = QtWidgets.QScrollArea()
        self.groups_scroll.setWidgetResizable(True)
        self.groups_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.groups_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.groups_scroll.setWidget(groups)
        # just the two header rows when both groups are closed
        self.groups_scroll.setMinimumHeight(40)
        self._groups_box = groups

        # --- trace action row: hint · Clear range · primary (R5.2) -----
        bar = QHBoxLayout()
        self.sel_lbl = QLabel(EXCLUDE_TIME_HINT)
        self.sel_lbl.setStyleSheet("color:#6b7585;font-size:11px;")
        bar.addWidget(self.sel_lbl)
        bar.addStretch()
        self.clear_range_btn = QPushButton("Clear range")
        self.clear_range_btn.setToolTip(CLEAR_RANGE_TIP)
        self.clear_range_btn.clicked.connect(self.clear_range)
        bar.addWidget(self.clear_range_btn)
        self.mark_btn = QPushButton("Exclude time range…")
        self.mark_btn.clicked.connect(self._primary_action)
        fit_checked_width(self.mark_btn, ['Exclude time range…',
                                          'Remove exclusion'])
        bar.addWidget(self.mark_btn)
        lc.addLayout(bar)
        self._update_range_controls()

        # channel-level actions (preserved)
        crow = QHBoxLayout()
        clbl = QLabel("Channel-level:")
        clbl.setStyleSheet("color:#6b7585;")
        crow.addWidget(clbl)
        # the same Exclude channel toggle as the Channels tab's bottom bar
        self.exclude_btn = QPushButton("Exclude channel")
        self.exclude_btn.setCheckable(True)
        self.exclude_btn.setToolTip(EXCLUDE_TIP)
        self.exclude_btn.clicked.connect(self._exclude_clicked)
        set_exclude_button(self.exclude_btn, False)
        fit_checked_width(self.exclude_btn,
                          ['Exclude channel', 'Include channel'])
        crow.addWidget(self.exclude_btn)
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
        left.addWidget(top_pane)
        left.addWidget(self.groups_scroll)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 0)
        split.addWidget(left)
        # the saved splitter position, else the bottom pane fitted to the
        # groups' two header rows; a group toggle refits it
        state = _review_settings().value('review/epochs_splitter')
        try:
            restored = state is not None and bool(left.restoreState(state))
        except TypeError:           # a stored value of another type
            restored = False
        if not restored:
            QtCore.QTimer.singleShot(0, self._fit_groups_pane)
        left.splitterMoved.connect(lambda *_: _review_settings().setValue(
            'review/epochs_splitter', left.saveState()))
        self.neighbours.toggled.connect(lambda *_: self._fit_groups_pane())
        self.physio.toggled.connect(lambda *_: self._fit_groups_pane())

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

    RAW_MIN_PX, FILT_MIN_PX = 160, 110

    def _fit_groups_pane(self):
        """Size the bottom pane to what the two groups need, without taking
        the top pane below its minimum (the raw and filtered traces keep
        their minimum heights; the rest scrolls)."""
        sp = self.v_split
        total = sum(sp.sizes())
        if total <= 0:
            return
        self._groups_box.layout().activate()
        want = self._groups_box.sizeHint().height() + 4
        top_min = sp.widget(0).minimumSizeHint().height()
        bottom = max(24, min(want, total - top_min))
        sp.setSizes([total - bottom, bottom])
        _review_settings().setValue('review/epochs_splitter', sp.saveState())

    # ---- trace y-range (R4.11) -----------------------------------------
    def _on_full_range(self, _on):
        self._apply_trace_ranges()
        self._redraw_event_layers()

    def clip_note(self, which='raw'):
        """The clip note shown on the raw (``'raw'``) or filtered
        (``'filt'``) trace, or ``None``."""
        return self._clip_notes.get(which)

    def _apply_trace_ranges(self):
        """Draw the epoch's raw and filtered curves in a y-range that one
        large event cannot flatten: centre = median, half-range = the larger
        of the floor (50 / 10 µV) and 8 robust SDs, no larger than the
        largest sample, rounded up to a 1-2-5 number. Samples beyond it are
        drawn clipped at the plot edge, with a 2 px line along that edge and
        a note; ``Full range`` shows everything."""
        for p, it in self._clip_items:
            p.removeItem(it)
        self._clip_items = []
        self._clip_notes = {'raw': None, 'filt': None}
        if self._trace is None:
            self.raw_half_range = self.filt_half_range = None
            return
        ts, raw, filt = self._trace
        full = self.full_range_chk.isChecked()
        warn = pg.mkPen(THEME['warn'], width=2)
        for key, plot, x, floor, pen in (
                ('raw', self.raw_plot, raw, _er.RAW_FLOOR,
                 pg.mkPen((155, 166, 181), width=1)),
                ('filt', self.filt_plot, filt, _er.FILTERED_FLOOR,
                 pg.mkPen(QtGui.QColor(EVT_COLOR.get(self._event_type,
                                                     '#5fd3a4')), width=1))):
            centre, half, largest = _er.trace_range(x, floor)
            if full:
                half = _er.round_up_125(max(largest, half))
            lo, hi = centre - half, centre + half
            curve = getattr(self, f'_{key}_curve', None)
            y = np.clip(x, lo, hi)
            if curve is None or curve.scene() is None:
                curve = plot.plot(ts, y, pen=pen)
                setattr(self, f'_{key}_curve', curve)
            else:
                curve.setData(ts, y)
            plot.setYRange(lo, hi, padding=0)
            plot.enableAutoRange('y', False)
            if key == 'raw':
                self.raw_half_range, self._raw_ytop = half, hi
            else:
                self.filt_half_range, self._filt_ytop = half, hi
            if largest > half * (1 + 1e-9):
                for edge, mask in ((hi, x > hi), (lo, x < lo)):
                    if mask.any():
                        mark = pg.PlotCurveItem(
                            ts, np.where(mask, edge, np.nan), pen=warn,
                            connect='finite')
                        mark.setZValue(6)
                        plot.addItem(mark, ignoreBounds=True)
                        self._clip_items.append((plot, mark))
                text = _er.clip_note(half, largest)
                note = pg.TextItem(text, color=THEME['warn'], anchor=(1, 0))
                f = note.textItem.font()
                f.setPixelSize(10)
                note.setFont(f)
                note.setPos(float(ts[-1]), hi)
                note.setZValue(7)
                plot.addItem(note, ignoreBounds=True)
                self._clip_items.append((plot, note))
                self._clip_notes[key] = text

    # ---- epoch table --------------------------------------------------
    def index_at(self, t):
        """Epoch id containing recording time ``t`` (clamped)."""
        return self._epochs.index_at(t)

    def span(self, i):
        """``(start_s, end_s)`` of epoch ``i`` (clamped)."""
        return self._epochs.span(i)

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

        self._set_title(trec)
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
        self._update_shown()
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

    def _set_title(self, trec):
        """``<b>Cz</b> · spindle · n=9 · density 0.42 ev/min`` (no density
        without a recording length)."""
        df_slice = self._df
        n = 0 if df_slice is None else len(df_slice)
        dens = ''
        if n and trec:
            dens = f" · density {n / (trec / 60.0):.2f} ev/min"
        self.title.setText(f"<b>{self._channel}</b> · {self._event_type} · "
                           f"n={n}{dens}")

    def set_epoch_table(self, epochs, trec=None):
        """Replace the epoch table of the open drill without re-drilling: an
        annotation file opened (or unloaded) after the channel was drilled.
        ``epochs`` is an :class:`EpochTable` or ``None`` (a synthetic 30 s
        grid over ``trec``, unstaged). Keeps the channel, its events, the
        reviews and the selected event; recomputes the per-epoch outliers,
        redraws the hypnogram, strip and header, and pages to the epoch that
        holds the selected event's start, else the current window start."""
        if trec:
            self._trec = float(trec)
        if self._channel is None:
            self._epochs = _as_epoch_table(epochs, None, self._trec)
            self._hypno = list(self._epochs.stages) if epochs else None
            return
        t_now = self.span(self._epoch)[0]
        if self._selected_uuid is not None and self._ev is not None:
            hit = self._ev[self._ev['uuid'] == self._selected_uuid]
            if len(hit):
                t_now = float(hit['_start'].iloc[0])
        self._epochs = _as_epoch_table(epochs, None, self._trec)
        self._hypno = list(self._epochs.stages) if epochs else None
        self._agg = _compute_epoch_outliers(
            self._df, epochs=self._epochs, amp_col=self._amp_col)
        self._n_max = int(self._agg['n_events'].max()) \
            if len(self._agg) else 1
        self._set_title(self._trec if trec else None)
        self._set_hypno(self._hypno)
        self._set_ranges(getattr(self, '_marked', []) or [])
        self._render_strip()
        self._update_shown()
        self._goto_epoch(self._epochs.index_at(t_now))

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
                f"({_span_text(t0, t1)})")
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
        self.plot.addItem(self._ep_cursor)
        for it in getattr(self, '_sample_marks', []):
            self.plot.addItem(it)
        self.plot.setXRange(0, max(self._trec, self._epochs.end), padding=0)
        if self._agg is None or len(self._agg) == 0:
            return
        idx = self._agg['idx'].to_numpy(dtype=int)
        n_ev = self._agg['n_events'].to_numpy(dtype=float)
        n_out = self._agg['n_outliers'].to_numpy(dtype=float)
        if getattr(self, '_hide_outliers', False):
            n_out = np.zeros_like(n_ev)     # sample review: one neutral layer
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
        # top (red) layer stacked on top; not drawn in sample review
        if not getattr(self, '_hide_outliers', False):
            self.plot.addItem(pg.BarGraphItem(
                x=centres, width=widths,
                y0=h_reg, height=h_out, brush=THEME['bad'], pen=None))
        # Show events: a 2 px tick along the bottom under each epoch that
        # holds at least one shown event on the drilled channel (R5.4)
        self.shown_ticks, self.shown_tick_epochs = None, []
        if self._status_allowed is not None and self._ev is not None \
                and not self._ev.empty:
            st = self._ev.loc[self._shown_mask(), '_start'].to_numpy(
                dtype=float)
            eps = sorted({int(self.index_at(t)) for t in st})
            self.shown_tick_epochs = eps
            if eps:
                xs = np.ravel([[self._epochs.starts[e], self._epochs.ends[e]]
                               for e in eps])
                ys = np.full(len(xs), 0.015)
                self.shown_ticks = pg.PlotCurveItem(
                    xs, ys, connect='pairs',
                    pen=pg.mkPen(THEME['text'], width=3))
                self.shown_ticks.setZValue(12)
                self.plot.addItem(self.shown_ticks)
        # marked-artefact bands (dashed purple, transparent fill)
        edge = pg.mkPen('#a371f7', width=1, style=Qt.DashLine)
        for m in self._marked:
            self.plot.addItem(pg.LinearRegionItem(
                values=[float(m['start_time']), float(m['end_time'])],
                brush=(167, 113, 247, 60), pen=edge, movable=False))

    def _draw_window_overlays(self):
        """Excluded time in the current window (R5.2): purple diagonal
        hatch with a dashed edge on the raw and filtered traces, labelled
        ``excluded`` on the raw trace; the selected one has a 2 px solid
        edge. Event bands sit on top. Called from _goto_epoch right after
        the plots are cleared, and after a selection change."""
        for items in self._excl_items.values():
            for p, it in items:
                p.removeItem(it)
        self._excl_items = {}
        t0, t1 = self.span(self._epoch)
        c = QtGui.QColor(EXCLUDED_PURPLE)
        for m in self._marked:
            a, b = float(m['start_time']), float(m['end_time'])
            if b <= t0 or a >= t1:
                continue
            mid = int(m['id'])
            sel = mid == self._sel_excl
            items = []
            for p in (self.raw_plot, self.filt_plot):
                fill = QtGui.QBrush(QtGui.QColor(c.red(), c.green(),
                                                 c.blue(), 90),
                                    Qt.BDiagPattern)
                pen = (pg.mkPen(EXCLUDED_PURPLE, width=2) if sel else
                       pg.mkPen(EXCLUDED_PURPLE, width=1,
                                style=Qt.DashLine))
                it = pg.LinearRegionItem(values=[max(a, t0), min(b, t1)],
                                         brush=fill, pen=pen, movable=False)
                it.setZValue(2)
                it.exclusion_id = mid
                p.addItem(it, ignoreBounds=True)
                items.append((p, it))
            lab = pg.TextItem('excluded', color=EXCLUDED_PURPLE, anchor=(0, 0))
            f = lab.textItem.font()
            f.setPixelSize(10)
            lab.setFont(f)
            lab.setPos(max(a, t0), self._raw_ytop)
            lab.setZValue(3)
            self.raw_plot.addItem(lab, ignoreBounds=True)
            items.append((self.raw_plot, lab))
            self._excl_items[mid] = items

    def exclusion_items(self, plot=None):
        """``{id: [LinearRegionItem, ...]}`` drawn in the window (tests)."""
        return {k: [it for p, it in v if isinstance(it, pg.LinearRegionItem)
                    and (plot is None or p is plot)]
                for k, v in self._excl_items.items()}

    def set_exclusions(self, rows):
        """Replace the excluded ranges (all of the reviewer's, every
        channel) and redraw the strip and window without paging."""
        keep = self._sel_excl
        self._set_ranges(rows or [])
        if keep is not None and keep not in {int(m['id'])
                                             for m in self._marked}:
            self._sel_excl = None
        self._render_strip()
        self._draw_window_overlays()
        self._update_range_controls()

    def _epoch_stage(self):
        if not self._hypno:
            return ''
        return self._epochs.stage(self._epoch)

    def _n_epochs(self):
        return max(1, len(self._epochs))

    def _goto_epoch(self, i):
        n = self._n_epochs()
        i = max(0, min(int(i), n - 1))
        if i != self._epoch and self.full_range_chk.isChecked():
            self.full_range_chk.blockSignals(True)   # reset per epoch
            self.full_range_chk.setChecked(False)
            self.full_range_chk.blockSignals(False)
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
        self._update_epoch_label()
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
        self._clip_items = []
        self._raw_curve = self._filt_curve = None
        self._trace = None
        # a new window starts with no brush and no selected exclusion (the
        # plots were just cleared, so the brush items are re-added hidden)
        self._brush = None
        self._sel_excl = None
        for p, it in ((self.raw_plot, self.region),
                      (self.filt_plot, self.region_filt),
                      (self.raw_plot, self.region_label)):
            it.hide()
            p.addItem(it, ignoreBounds=True)
        self._update_range_controls()
        # lock to exactly the epoch
        for p in (self.raw_plot, self.filt_plot):
            p.setXRange(t0, t1, padding=0)
            p.enableAutoRange('x', False)
        self.ticker.setXRange(t0, t1, padding=0)
        # y-ranges come from the epoch's data (_apply_trace_ranges); with no
        # signal the filtered trace keeps its nominal range
        yr = FILT_YRANGE.get(self._event_type, 50)
        self.filt_plot.setYRange(-yr, yr, padding=0)
        self.filt_plot.enableAutoRange('y', False)
        self._filt_ytop = float(yr)
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
                lo, hi = self._band   # sourced from event rows in set_channel
                self._trace = (ts, data, np.asarray(
                    _bandpass(data, sfreq, lo, hi), dtype=float))
            else:
                self.trace_note.setText(
                    f"signal trace · {self._channel} · "
                    f"no data in epoch {i + 1}/{n}")
        else:
            self.trace_note.setText(
                "load an EEG file to enable the signal trace")
            self._raw_ytop = 1.0
        self._apply_trace_ranges()
        self._draw_window_overlays()
        self._draw_trace_events(t0, t1)
        self._draw_ticker(t0, t1)
        self.physio.show_window(t0, t1)
        self.physio.set_selection(self._selected_span())
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
        """Check filter (Channels-tab link) and Show events filter (the
        current reviewer's decisions; an event decided only by another
        reviewer is unreviewed)."""
        if self._check_filter is not None and \
                uuid not in self._check_filter['uuids']:
            return False
        if self._status_allowed is not None:
            st = self._decision_of(uuid) or 'unreviewed'
            if st not in self._status_allowed:
                return False
        return True

    def set_status_filter(self, allowed):
        """Show events: ``allowed`` subset of ``{'unreviewed', 'accept',
        'reject', 'unsure'}``, or ``None`` for no filter. Dims the others and
        drops them from ``}`` / ``{`` (and the shown buttons); ``]`` / ``[``,
        sample navigation and the EVENT n OF N header ignore it. The
        checkboxes follow."""
        full = {'unreviewed', 'accept', 'reject', 'unsure'}
        self._status_allowed = None if allowed is None or set(allowed) >= \
            full else set(allowed)
        want = full if self._status_allowed is None else self._status_allowed
        for label, dec in _er.SHOW_EVENTS_ITEMS:
            cb = self.show_checks[label]
            cb.blockSignals(True)
            cb.setChecked(dec in want)
            cb.blockSignals(False)
        self._redraw_event_layers()
        self._update_shown()
        self._render_strip()

    def _on_show_events(self):
        self.set_status_filter({dec for label, dec in _er.SHOW_EVENTS_ITEMS
                                if self.show_checks[label].isChecked()})

    def show_all_events(self):
        """The chip's ✕: tick all four."""
        self.set_status_filter(None)

    def shown_what(self):
        """The ticked names joined by ', ' (``accepted, unsure``)."""
        return ', '.join(label for label, _d in _er.SHOW_EVENTS_ITEMS
                         if self.show_checks[label].isChecked())

    def _shown_mask(self):
        """Events on the channel that pass Show events (all of them while
        nothing is filtered); the check filter is not part of it."""
        ev = self._ev
        ok = ev['uuid'].notna()
        if self._status_allowed is not None:
            st = ev['uuid'].map(lambda u: self._decision_of(u) or 'unreviewed')
            ok &= st.isin(list(self._status_allowed))
        return ok

    def _update_shown(self):
        """Chip, shown buttons and legend while Show events filters."""
        on = self._status_allowed is not None and self._channel is not None
        n = total = 0
        if on and self._ev is not None and not self._ev.empty:
            total = int(self._ev['uuid'].notna().sum())
            n = int(self._shown_mask().sum())
        self.shown_chip_lbl.setText(_er.shown_chip_text(
            self.shown_what(), n, total, self._channel or '—')[:-2])
        self.shown_chip.setVisible(on)
        for b in (self.prev_shown_btn, self.next_shown_btn):
            b.setVisible(on)
            b.setEnabled(on and n > 0)
        self._update_legend()

    def shown_chip_text(self):
        """The chip as the spec writes it (with ``✕``), or '' when hidden."""
        if not self.shown_chip.isVisibleTo(self):
            return ''
        return self.shown_chip_lbl.text() + ' ✕'

    def _update_legend(self):
        sample = getattr(self, '_hide_outliers', False)
        text = (_er.STRIP_LEGEND_NO_OUTLIERS + _er.STRIP_LEGEND_SAMPLE
                if sample else _er.STRIP_LEGEND)
        if self._status_allowed is not None:
            text += _er.STRIP_LEGEND_SHOWN
        self.strip_legend.setText(text)

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
            # live sample review: an undecided sample event is a regular
            # band, with no outlier tint or label
            out = bool(out) and not self._outlier_hidden(u)
            fill, pen, z, glyph, gcol = self._band_style(
                u, out, not self._passes_filter(u))
            e = min(e, t1)
            for p in (self.raw_plot, self.filt_plot):
                it = pg.LinearRegionItem(values=[s, e], brush=fill, pen=pen,
                                         movable=False)
                it.setZValue(z)
                p.addItem(it)
                self._event_items.append((p, it))
            if out:
                # amplitude outliers keep a text label on both traces
                ftop = float(self._filt_ytop)
                for p, ytop_p in ((self.raw_plot, ytop), (self.filt_plot, ftop)):
                    lab = pg.TextItem('outlier', color=THEME['bad'],
                                      anchor=(1, 0))
                    fo = lab.textItem.font()
                    fo.setPixelSize(10)
                    lab.setFont(fo)
                    lab.setPos(e, ytop_p)
                    lab.setZValue(z + 1)
                    p.addItem(lab, ignoreBounds=True)
                    self._event_items.append((p, lab))
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

    def _outlier_hidden(self, uuid):
        fn = self.outlier_hidden
        try:
            return bool(fn(uuid)) if callable(fn) else False
        except Exception:
            return False

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
        # an undecided sample event is a regular bar (no outlier height/red)
        is_out = sub['_is_out'].astype(bool) & ~uu.map(self._outlier_hidden)
        layers = [(sub[~is_out & ~reviewed], 0.4, 9, grey, None),
                  (sub[is_out & ~reviewed], 0.6, 14, THEME['bad'], None),
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
        if self._status_allowed is not None:
            self._update_shown()        # counts and ticks follow decisions
            self._render_strip()

    def _redraw_event_layers(self):
        t0, t1 = self.span(self._epoch)
        self._draw_trace_events(t0, t1)
        self._draw_ticker(t0, t1)
        self.physio.set_selection(self._selected_span())

    def _selected_span(self):
        """``(start, end)`` of the selected event, or ``None``."""
        if self._selected_uuid is None or self._ev is None or self._ev.empty:
            return None
        hit = self._ev[self._ev['uuid'] == self._selected_uuid]
        if hit.empty:
            return None
        return float(hit['_start'].iloc[0]), float(hit['_end'].iloc[0])

    def epoch_events_order(self):
        """``(i, n)``: the selected event's 1-based position among the
        events starting in the current epoch (all of them, whatever the
        filters), or ``(None, n)``."""
        t0, t1 = self.span(self._epoch)
        sub = self._window_events(t0, t1)
        n = 0 if sub is None else len(sub)
        if self._selected_uuid is None or not n:
            return None, n
        uu = list(sub['uuid'])
        return ((uu.index(self._selected_uuid) + 1)
                if self._selected_uuid in uu else None), n

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
        elif plot is not self.ticker:
            # a click inside excluded time (away from any event) selects it
            mid = self._exclusion_at(x)
            if mid is not None:
                self.select_exclusion(mid, page=False)
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
        if self._status_allowed is not None:
            st = ev['uuid'].map(lambda u: self._decision_of(u) or 'unreviewed')
            ok &= st.isin(list(self._status_allowed))
        return ok

    def next_event(self):
        """Next shown event (``}`` and ``next shown ▶``): events dimmed by
        Show events or a check filter are skipped."""
        return self._shown_step(+1)

    def prev_event(self):
        """Previous shown event (``{`` and ``◀ previous shown``)."""
        return self._shown_step(-1)

    def _shown_step(self, step):
        mask = self._filter_mask()
        if self._status_allowed is not None and self._ev is not None \
                and not self._ev.empty and not bool(mask.any()):
            self.last_nav = 'none'
            self.statusMessage.emit(
                f"No {self.shown_what()} events on {self._channel}.")
            return None
        return self._step(mask, step)

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

    def set_sample_legend(self, on):
        """Sample mode on the panel: the strip legend gains the blue ticks,
        and every amplitude-outlier indication is hidden (strip colour and
        legend, the epoch label's outlier count, the outlier-rule line, the
        Prev / Next outlier buttons and N / P), so an undecided sample
        event's outlier status cannot be read off any of them. Exit sample
        brings them back."""
        on = bool(on)
        self._hide_outliers = on
        self._update_legend()
        self.strip_hdr.setVisible(not on)
        for b in (self.prev_out_btn, self.next_out_btn):
            b.setEnabled(not on)
            b.setToolTip(SAMPLE_HIDDEN_TIP if on else '')
        if self._channel is not None:
            self._render_strip()
            self._update_epoch_label()

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
        """Esc: clear the unsaved brush or the selected excluded range,
        else remove the check filter, else clear the selection. Returns
        what was done (or ``None``)."""
        if self.clear_range():
            return 'range'
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

    def _update_epoch_label(self):
        """``Epoch 3/755 · 00:01:00–00:01:30 · NREM2 · 5 events (1
        outlier)``; the outlier count is left out in sample review."""
        i, n = self._epoch, self._n_epochs()
        t0, t1 = self.span(i)
        dur = t1 - t0
        stage = self._epoch_stage() or '—'
        n_ev, n_out = self._epoch_counts(i)
        # name the length only when it is not the standard 30 s
        dtxt = ('' if abs(dur - DEFAULT_EPOCH_S) < 1e-6
                else f" ({_epoch_len_text(dur)})")
        outs = ('' if getattr(self, '_hide_outliers', False)
                else f" ({n_out} outlier{'s' if n_out != 1 else ''})")
        self.epoch_lbl.setText(
            f"Epoch {i + 1}/{n} · {_hms(t0)}–{_hms(t1)}{dtxt} · {stage} · "
            f"{n_ev} events{outs}")

    def _outlier_epoch_indices(self):
        if getattr(self, '_hide_outliers', False):
            return []           # N / P and the buttons do nothing
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
        if plain and Qt.Key_0 <= k <= Qt.Key_9:
            self.reviewKey.emit(chr(k)); return
        if plain and k in (Qt.Key_Return, Qt.Key_Enter):
            self.reviewKey.emit('Enter'); return
        nav = {Qt.Key_BracketRight: self.next_unreviewed,
               Qt.Key_BracketLeft: self.prev_unreviewed,
               Qt.Key_BraceRight: self.next_event,
               Qt.Key_BraceLeft: self.prev_event}
        if k in nav:
            nav[k](); return
        super().keyPressEvent(ev)

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

    # ---- the brush and excluded ranges (R5.2) ----------------------------
    def _on_brush_drag(self, t0, t1, finished):
        """A drag on the raw trace draws the unsaved brush."""
        if self._channel is None:
            return
        w0, w1 = self.span(self._epoch)
        self.set_brush(max(t0, w0), min(t1, w1))

    def set_brush(self, t0, t1):
        """Show the unsaved brush over ``[t0, t1]`` (clears a selected
        excluded range)."""
        a, b = sorted((float(t0), float(t1)))
        if b - a <= 0:
            return
        if self._sel_excl is not None:
            self._sel_excl = None
            self._draw_window_overlays()
        self._brush = (a, b)
        self._region_programmatic = True
        self.region.setRegion([a, b])
        self._region_programmatic = False
        self._show_brush()

    def _show_brush(self):
        a, b = self._brush
        self.region_filt.setRegion([a, b])
        self.region_label.setPos(a, self._raw_ytop)
        for it in (self.region, self.region_filt, self.region_label):
            it.show()
        self._update_range_controls()

    def _on_region_changed(self):
        # the user drags an edge of the brush
        if self._region_programmatic or self._brush is None:
            return
        a, b = self.region.getRegion()
        self._brush = (float(min(a, b)), float(max(a, b)))
        self._show_brush()

    def has_brush(self):
        return self._brush is not None

    def clear_range(self):
        """``Clear range`` / Esc: remove the unsaved brush, or deselect the
        selected excluded range. Returns True when something was cleared."""
        if self._brush is not None:
            self._brush = None
            for it in (self.region, self.region_filt, self.region_label):
                it.hide()
            self._update_range_controls()
            return True
        if self._sel_excl is not None:
            self._sel_excl = None
            self._draw_window_overlays()
            self._update_range_controls()
            return True
        return False

    def _exclusion(self, mid):
        return next((m for m in self._marked if int(m['id']) == int(mid)),
                    None)

    def _exclusion_at(self, x):
        t0, t1 = self.span(self._epoch)
        for m in self._marked:
            a, b = float(m['start_time']), float(m['end_time'])
            if a <= x <= b and b > t0 and a < t1:
                return int(m['id'])
        return None

    def select_exclusion(self, mid, page=True):
        """Select an excluded range (clears the brush); pages to it when it
        is not in the window and ``page`` is True."""
        m = self._exclusion(mid)
        if m is None:
            return False
        a = float(m['start_time'])
        t0, t1 = self.span(self._epoch)
        if page and not (a < t1 and float(m['end_time']) > t0):
            self._goto_epoch(self.index_at(a))
        if self._brush is not None:
            self._brush = None
            for it in (self.region, self.region_filt, self.region_label):
                it.hide()
        self._sel_excl = int(mid)
        self._draw_window_overlays()
        self._update_range_controls()
        return True

    def selected_exclusion(self):
        return self._sel_excl

    def set_exclude_allowed(self, ok, tip=''):
        """Whether a new exclusion may be saved (the matching annotation
        file is loaded); ``tip`` says why not."""
        self._exclude_allowed, self._exclude_block_tip = bool(ok), tip or ''
        self._update_range_controls()

    def _update_range_controls(self):
        """Hint, ``Clear range`` and the primary button for the current
        state: no range, an unsaved brush, or a selected excluded range."""
        btn = self.mark_btn
        if self._sel_excl is not None and self._exclusion(self._sel_excl):
            m = self._exclusion(self._sel_excl)
            a, b = float(m['start_time']), float(m['end_time'])
            rater = str(m.get('reviewer') or '') or 'an unnamed reviewer'
            date = str(m.get('qc_timestamp') or '')[:10] or 'an unknown date'
            self.sel_lbl.setText(
                f"Excluded {_hms(a)}–{_hms(b)} ({b - a:.1f} s), saved by "
                f"{rater} on {date}.")
            btn.setText('Remove exclusion')
            btn.setToolTip(REMOVE_EXCLUSION_TIP)
            name = 'danger'
            btn.setEnabled(True)
        else:
            if self._brush is not None:
                a, b = self._brush
                self.sel_lbl.setText(
                    f"Unsaved range {_hms(a)}–{_hms(b)} ({b - a:.1f} s).")
            else:
                self.sel_lbl.setText(EXCLUDE_TIME_HINT)
            btn.setText('Exclude time range…')
            allowed = getattr(self, '_exclude_allowed', True)
            btn.setToolTip(EXCLUDE_TIME_TIP if allowed
                           else self._exclude_block_tip)
            name = 'primary'
            btn.setEnabled(self._brush is not None
                           and self._channel is not None and allowed)
        if btn.objectName() != name:
            btn.setObjectName(name)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.clear_range_btn.setEnabled(self._brush is not None
                                        or self._sel_excl is not None)

    # ---- actions ------------------------------------------------------
    def _sel(self):
        a, b = self._brush if self._brush is not None else (0.0, 0.0)
        return float(a), float(b)

    def _primary_action(self):
        """``Exclude time range…`` saves the brush; ``Remove exclusion``
        removes the selected excluded range."""
        if self._sel_excl is not None:
            self.unmarkArtefactRequested.emit(int(self._sel_excl))
            return
        self._mark()

    def _mark(self):
        if self._channel is None or self._brush is None or \
                not getattr(self, '_exclude_allowed', True):
            return
        s, e = self._sel()
        if e - s < 0.5:
            QtWidgets.QMessageBox.information(
                self, "Exclude time range",
                "Brush a wider range on the trace first.")
            return
        self.markArtefactRequested.emit(self._channel, s, e)

    def _unmark(self):
        it = self.ranges_list.currentItem()
        if it is not None:
            self.unmarkArtefactRequested.emit(int(it.data(Qt.UserRole)))

    def _exclude_clicked(self, checked):
        if self._channel:
            self.excludeChannelRequested.emit(self._channel, bool(checked))
        else:
            set_exclude_button(self.exclude_btn, False)

    def set_channel_excluded(self, excluded):
        """Show the drilled channel as excluded (``Include channel``,
        checked) or included (``Exclude channel``)."""
        set_exclude_button(self.exclude_btn, bool(excluded),
                           self._event_type)

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
               "single-channel problem (use 'Exclude channel' for that).\n\n"
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


def qss_url(path):
    """``url("...")`` for a file path in a Qt style sheet: forward slashes,
    the path in double quotes, backslashes and double quotes escaped, so a
    folder name with spaces or parentheses does not break the rule."""
    text = str(path).replace(os.sep, '/').replace('\\', '/')
    return 'url("' + text.replace('"', '\\"') + '")'


def checkbox_mark_qss(folder=None):
    """Stylesheet rules that put a tick (``✓``) on a checked checkbox and a
    dash (``–``) on a partly checked one (spec R4.2), so the state does not
    rely on the fill colour alone.

    Qt style sheets take the mark only as an image file, so two small PNGs
    are painted once into a temporary folder (removed at exit). Needs a
    ``QApplication``; returns ``''`` if the images cannot be written.
    """
    import atexit
    import shutil
    import tempfile
    default = folder is None        # only the default call is cached
    cached = getattr(checkbox_mark_qss, '_qss', None)
    if cached is not None and default:
        return cached
    qss = ''
    try:
        if folder is None:
            folder = tempfile.mkdtemp(prefix='tw_review_marks_')
            atexit.register(shutil.rmtree, folder, ignore_errors=True)
        paths = {}
        for name, mark in (('checked', '✓'), ('partial', '–')):
            pm = QtGui.QPixmap(26, 26)          # 2x for sharp scaling
            pm.fill(Qt.transparent)
            p = QtGui.QPainter(pm)
            font = p.font()
            font.setPixelSize(22)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QtGui.QColor(THEME['text']))
            p.drawText(pm.rect(), int(Qt.AlignCenter), mark)
            p.end()
            path = os.path.join(folder, name + '.png')
            if not pm.save(path, 'PNG'):
                raise OSError(path)
            paths[name] = qss_url(path)
        qss = (f"\nQCheckBox::indicator:checked "
               f"{{ image: {paths['checked']}; }}"
               f"\nQCheckBox::indicator:indeterminate "
               f"{{ image: {paths['partial']}; }}\n")
    except Exception:
        qss = ''
    if default:
        checkbox_mark_qss._qss = qss
    return qss

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
QPushButton#danger {{
    color: #f0a59a;
    border-color: {THEME['bad']};
}}
QPushButton#danger:disabled {{
    color: {THEME['text_3']};
    border-color: {THEME['border_strong']};
}}
/* Every checkable button (spec R4.2): checked = soft accent fill, 1 px
   accent border, weight 600, so the state reads without colour. Placed
   after #primary / #danger so the checked look wins. */
QPushButton:checked, QToolButton:checked {{
    background: {THEME['accent_soft']};
    border: 1px solid {THEME['accent']};
    color: {THEME['text']};
    font-weight: 600;
}}
QPushButton:checked:hover, QToolButton:checked:hover {{ background: #36557d; }}
QPushButton:checked:disabled, QToolButton:checked:disabled {{
    border: 1px solid {THEME['text_3']};
    color: {THEME['text_2']};
    font-weight: 600;
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
QCheckBox::indicator:checked, QCheckBox::indicator:indeterminate {{
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
    that apply across both tabs. (The decision filter is the Epochs tab's
    Show events row, R5.4.)

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

        note = QLabel("Filters apply globally to both tabs.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7585;font-size:10px;")
        lay.addWidget(note)

        # --- channels ---------------------------------------------------
        lay.addWidget(_h("CHANNELS"))
        self.channel_search = QLineEdit()
        self.channel_search.setPlaceholderText("search…  e.g. E33 or Cz")
        self.channel_search.textChanged.connect(self._filter_channel_list)
        lay.addWidget(self.channel_search)
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMaximumHeight(220)
        lay.addWidget(self.channel_list)
        # "~ = interpolated": only when a listed channel is interpolated
        self.interp_legend = QLabel(INTERP_LEGEND)
        self.interp_legend.setToolTip(INTERP_LEGEND_TIP)
        self.interp_legend.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        self.interp_legend.setVisible(False)
        lay.addWidget(self.interp_legend)
        cbtns = QHBoxLayout()
        self.sel_all_btn = QPushButton("All")
        self.sel_none_btn = QPushButton("None")
        cbtns.addWidget(self.sel_all_btn)
        cbtns.addWidget(self.sel_none_btn)
        lay.addLayout(cbtns)
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
        self.interp_legend.setVisible(any(
            str(self.channel_list.item(i).data(Qt.UserRole)) in interp
            for i in range(self.channel_list.count())))


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


class CheatSheetDialog(QtWidgets.QDialog):
    """The ``?`` keys sheet: frameless, closed by ``?``, Esc or a click
    outside (spec section 15)."""

    def __init__(self, text, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        self.label = QLabel(text)
        self.label.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:12px;")
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)
        lay.addWidget(self.label)
        self.setStyleSheet(f"QDialog{{background:{THEME['bg_panel']};"
                           f"border:1px solid {THEME['border_strong']};}}")

    def text(self):
        return self.label.text()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Question, Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(ev)


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

    def event_info(self, uuid):
        """``(channel, start_time, stage)`` of a sample event."""
        r = self.win._sample['rows'].get(uuid, {})
        return r.get('channel'), r.get('start_time'), r.get('stage')

    def comments(self, reviewer):
        """``{uuid: comment}`` of one reviewer's decisions on the sample."""
        uu = list(self.win._sample['rows'])
        if not uu or not reviewer:
            return {}
        out = {}
        try:
            for i in range(0, len(uu), 500):
                part = uu[i:i + 500]
                cur = self.win.db.conn.execute(
                    f"SELECT uuid, comment FROM event_reviews WHERE reviewer "
                    f"= ? AND uuid IN ({','.join('?' * len(part))})",
                    [reviewer] + part)
                out.update({str(u): c for u, c in cur.fetchall() if c})
        except Exception:
            return {}
        return out

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
        self._preselect = None            # reason preselected by a grid click
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
        a = QAction('Export re-run package…', self)
        a.setStatusTip(RERUN_STATUS_TIP)
        a.triggered.connect(self.export_rerun_package)
        m_file.addAction(a)
        self.act_export_rerun = a
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
        a = QAction('Precision rule…', self)
        a.triggered.connect(lambda: self._open_precision_rule())
        m_review.addAction(a)
        self.act_precision_rule = a

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

        m_exp = mb.addMenu('E&xport')
        for text, slot in (
                ('Export QC report…', self.export_qc_summary),
                (None, None),
                ('Export re-run package…', self.export_rerun_package),
                ('Export Figure…', self.export_figure)):
            if text is None:
                m_exp.addSeparator()
                continue
            a = QAction(text, self)
            if slot == self.export_rerun_package:
                a.setStatusTip(RERUN_STATUS_TIP)
                self.act_export_rerun2 = a
            a.triggered.connect(slot)
            m_exp.addAction(a)

        m_help = mb.addMenu('&Help')
        a = QAction('Keyboard shortcuts…', self)
        a.triggered.connect(self.open_cheat_sheet)
        m_help.addAction(a)
        a = QAction(_er.HELP_TITLE, self)
        a.triggered.connect(self.open_event_help)
        m_help.addAction(a)
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

        # key hints for the tab in view (spec section 15)
        self.key_hint_lbl = QLabel(_er.KEY_HINTS['channels'])
        self.key_hint_lbl.setStyleSheet(
            f"color:{THEME['text_3']};font-size:11px;")
        tb.addWidget(self.key_hint_lbl)

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
        self.detail_dock.setMinimumWidth(300)
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
        self.detail_dock_w.exclusionClicked.connect(self._on_exclusion_row)
        self.detail_dock_w.openInEpochs.connect(self._open_in_epochs)
        self.detail_dock_w.dropChannelRequested.connect(
            lambda ch: self._set_channel_excluded(ch, True))
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
        evp.helpRequested.connect(self.open_event_help)

        self.qc_widget.channelSelected.connect(self.on_qc_channel_selected)
        self.qc_widget.requestDrill.connect(self._open_in_epochs)
        self.qc_widget.requestExclude.connect(self._set_channel_excluded)
        self.qc_widget.exitSampleRequested.connect(self._exit_sample)
        self.qc_widget.checksLinkActivated.connect(self._go_to_checks_list)
        self.qc_widget.exportRerunRequested.connect(self.export_rerun_package)
        self.qc_widget.stageChanged.connect(self._on_check_stage)
        self.qc_widget.addToRedetect.connect(self.on_qc_add_redetect)
        self.qc_widget.requestQueueAllHard.connect(self.on_qc_queue_all_hard)
        self.qc_widget.loadMontageRequested.connect(self.on_load_montage)
        self.qc_widget.evt_combo.currentTextChanged.connect(
            lambda *_: self._refresh_all())
        self.epochs_panel.excludeChannelRequested.connect(
            self._set_channel_excluded)
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
        self.epochs_panel.reviewKey.connect(
            lambda k: self._key_enter() if k == 'Enter' else self._key_digit(k))
        self.epochs_panel.epochChanged.connect(
            lambda *_: self._refresh_position_segment())
        self.epochs_panel.statusMessage.connect(
            lambda text: self.status_bar.showMessage(text))
        self.epochs_panel.checkFilterCleared.connect(
            self._on_check_filter_cleared)
        self.epochs_panel.read_window = self._read_eeg_window
        self.epochs_panel.read_channels = self._read_eeg_channels
        self.epochs_panel.neighbour_provider = self._neighbour_provider
        self.epochs_panel.neighbour_events = self._neighbour_events
        self.epochs_panel.channel_unit = self._channel_unit
        self.epochs_panel.outlier_hidden = self._outlier_marks_hidden
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
                    "%d excluded time range(s) pending. The denominator "
                    "stored by the detection run is not used here because it "
                    "predates those ranges and would show a density that "
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
        excluded = {str(c) for c, v in verdicts.items()
                    if v in EXCLUDED_VERDICTS}
        self._load_redetect_queue()
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
        self._qc_density_min = density_min      # what the table used
        qc = compute_channel_qc(df, scored_minutes=density_min,
                                artefact_intervals=ivs,
                                coords=self.detail_dock_w._coords,
                                excluded=excluded,
                                **self._qc_thresholds)
        # population checks: cached per run, else read in the background
        qc = self._apply_population(qc, evt, methods, freq_band)
        self._qc_events_df = df
        self._qc_df = qc
        self.qc_widget.set_data(qc, df, verdicts, self._redetect_queue)
        self.epochs_panel.set_channel_excluded(
            str(self.epochs_panel._channel) in excluded)
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
                        if v in EXCLUDED_VERDICTS}
        self.filter_dock.decorate_channels(artefact_set, self._redetect_queue,
                                           interp_set=self._eeg_channel_info()[2])
        self._refresh_event_counts()
        self._refresh_status_segments()
        self._refresh_toolbar_state()
        self._apply_recording_gates()
        n_hard = int((qc['flag'] == 'hard').sum()) if len(qc) else 0
        n_soft = int((qc['flag'] == 'soft').sum()) if len(qc) else 0
        self.status_bar.showMessage(
            f"{len(qc)} channels · {n_hard} hard · {n_soft} soft · "
            f"{len(excluded)} excluded · {evt}")

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
            marked=self._all_exclusions(), reviews=self._reviews_for_slice(sl))
        self.epochs_panel.set_channel_excluded(str(ch) in self._excluded_now(evt))
        if switch_tab:
            self.tabs.setCurrentIndex(1)
        # keys (]/[, A/R/U) reach the panel without a click on the trace
        self.epochs_panel.setFocus(Qt.OtherFocusReason)
        self.detail_dock_w.event_panel.set_grid(evt)
        self._update_sample_marks()
        if not self._sample_active:
            self._refresh_sample_bar()
        self._refresh_position_segment()

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

    def _rerender_selected(self):
        """Rebuild the Event panel for the selection (after a write or a
        Clear, so hidden labels appear or hide again at once)."""
        row, run = self._selected_row()
        if row is not None:
            self._render_event_panel(row, run)
        else:
            self._refresh_current_line()

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
            figure_note=note, outlier_n=ep._amp_n)
        uuid = str(row.get('uuid'))
        hidden = self._labels_hidden(uuid)
        if hidden:
            rows = _er.hide_flag_words(rows)
        panel.set_grid(evt)
        i, n = ep.epoch_events_order()
        panel.set_header(f"EVENT {i} OF {n} IN EPOCH" if i else 'EVENT')
        panel.set_event_line(
            _er.event_header_line(row, interp),
            _er.event_header_tooltip(row, run, row.get('run_id')))
        panel.set_sample_line(*self._sample_line_for(uuid))
        panel.set_hidden(hidden)
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
            panel.set_decision(None)
            return
        mine = (self.db.get_review(uuid, self.reviewer_name)
                if self.reviewer_name else None)
        subs, tip = [], ''
        if mine:
            dec = mine['decision']
            text = (f"{_er.DECISION_TITLE.get(dec, dec)} by "
                    f"{self.reviewer_name} · {_er.fmt_when(mine.get('reviewed_at'))}")
            if mine.get('reason'):
                text += f" · {_er.REASON_LABEL.get(mine['reason'], mine['reason'])}"
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
                  for d in range(0, 10)]
        self._review_shortcuts = []
        for key, slot in binds:
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)
            self._review_shortcuts.append(sc)
        self._help_shortcut = QShortcut(QtGui.QKeySequence('?'), self)
        self._help_shortcut.setContext(Qt.WindowShortcut)
        self._help_shortcut.activated.connect(self.open_cheat_sheet)
        self.tabs.currentChanged.connect(self._update_review_shortcuts)
        self.tabs.currentChanged.connect(
            lambda *_: self._refresh_position_segment())
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
        sc = getattr(self, '_help_shortcut', None)
        if sc is not None:
            sc.setEnabled(not typing)       # ? works on both tabs

    def open_cheat_sheet(self):
        """``?`` / Help ▸ Keyboard shortcuts…: open (or close) the sheet
        with the drilled event type's reason grid."""
        dlg = getattr(self, '_cheat_sheet', None)
        if dlg is not None and dlg.isVisible():
            dlg.close()
            return dlg
        ep = self.epochs_panel
        evt = ep._event_type if ep._channel is not None else 'spindle'
        undo = QtGui.QKeySequence(QtGui.QKeySequence.Undo).toString(
            QtGui.QKeySequence.NativeText) or 'Ctrl+Z'
        dlg = CheatSheetDialog(_er.cheat_sheet_text(evt, undo), self)
        dlg.adjustSize()
        g = self.frameGeometry()
        dlg.move(g.center().x() - dlg.width() // 2,
                 g.center().y() - dlg.height() // 2)
        self._cheat_sheet = dlg
        dlg.show()
        return dlg

    def open_event_help(self):
        """``What do these mean?`` / Help ▸ What the event figures mean: a
        small non-modal dialog. Its text is built into the GUI, so it needs
        no network and no docs folder; the link below it opens the published
        how-to in the browser."""
        dlg = getattr(self, '_event_help', None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(_er.HELP_TITLE)
            dlg.setModal(False)
            lay = QVBoxLayout(dlg)
            body = QLabel(_er.HELP_BODY)
            body.setWordWrap(True)
            body.setTextFormat(Qt.PlainText)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            body.setMinimumWidth(460)
            lay.addWidget(body)
            link = QLabel(f"<a href='{_er.HELP_GUIDE_URL}' "
                          f"style='color:{THEME['accent']}'>"
                          f"{_er.HELP_GUIDE_TEXT}</a>")
            link.setTextFormat(Qt.RichText)
            link.setOpenExternalLinks(True)
            lay.addWidget(link)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            bb.rejected.connect(dlg.close)
            lay.addWidget(bb)
            dlg.body_lbl, dlg.link_lbl = body, link
            self._event_help = dlg
        dlg.show()
        dlg.raise_()
        return dlg

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
                return (f"Choose a reason: 1–8, 0 for other, or Enter for "
                        f"“{lbl}”")
            return 'Choose a reason: 1–8, 0 for other'
        return 'Optional reason: 1–8, 0 for other, or Enter to save without one'

    def _key_decide(self, decision):
        """A writes at once; R / U arm and wait for a reason (spec 5)."""
        if self.selected_event_uuid is None:
            return
        panel = self.detail_dock_w.event_panel
        if decision == 'accept':
            self._disarm()
            self._write_decision('accept', None)
            return
        self._preselect = None
        self._armed = decision
        self._pending_other = False
        if decision == 'reject':
            panel.set_reason(self._last_reject_reason or '')
        else:
            panel.set_reason('')
        panel.set_armed(decision, self._hint_for_arm())

    _on_decision_requested = _key_decide   # EpochsPanel.decisionRequested

    def _key_digit(self, digit):
        """A digit picks a reason from the event type's grid while Reject or
        Unsure is armed; otherwise (and ``9`` always) it is ignored."""
        if self._armed is None:
            return
        token = _er.digit_reason(self.epochs_panel._event_type, digit)
        if token is None:
            return
        self._choose_reason(token)

    def _on_reason_picked(self, token):
        """A click on a grid button. With nothing armed it ARMS Reject with
        that reason preselected (a second click on the same button, or
        Enter, writes; another reason switches the preselection; Esc,
        another event or paging cancels). While Reject / Unsure is armed by
        its key, a click writes with that reason."""
        if self.selected_event_uuid is None or not token:
            return
        panel = self.detail_dock_w.event_panel
        pre = getattr(self, '_preselect', None)
        if self._armed is None or (self._armed == 'reject' and pre
                                   and pre != token):
            self._armed = 'reject'
            self._pending_other = False
            self._preselect = token
            panel.set_reason(token)
            lbl = _er.REASON_LABEL.get(token, token)
            panel.set_armed('reject', f"Click {lbl} again or press Enter to "
                                      f"reject ({lbl}).")
            return
        self._choose_reason(token)

    def _choose_reason(self, token):
        panel = self.detail_dock_w.event_panel
        if self._armed == 'reject' and not token:
            panel.set_armed('reject', 'Choose a reason: 1–8, 0 for other')
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
        if self._armed == 'reject' and getattr(self, '_preselect', None):
            self._choose_reason(self._preselect)
            return
        if self._armed == 'reject':
            if not self._last_reject_reason:
                return
            self._choose_reason(self._last_reject_reason)
            return
        token = panel.current_reason() or None
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
        if getattr(self, '_preselect', None):
            self._preselect = None
            self.detail_dock_w.event_panel.set_reason('')
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
        self._refresh_position_segment()
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
        self._rerender_selected()
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
        self._rerender_selected()
        self._refresh_progress()
        self.status_bar.showMessage(
            f"Deleted your decision on {row.get('channel')} "
            f"{_er.fmt_hms1(row.get('start_time'))} · Ctrl+Z brings it back")

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
        self.epochs_panel.set_sample_legend(on)
        # channel flags out of the dock while the sample is reviewed
        dock = self.detail_dock_w
        dock.set_sample_mode(on)
        self.qc_widget.set_sample_mode(on)
        qc = getattr(self, '_qc_df', None)
        if qc is not None:
            self._refresh_flagged(qc, self.qc_widget.current_event_type())
            dock.update_topo(qc)
        ch = getattr(self, '_qc_selected_channel', None)
        if ch:
            self._update_check_line(ch)
        self._update_sample_marks()
        if self.epochs_panel._channel is not None:
            # outlier marks of undecided sample events appear / disappear
            self.epochs_panel._redraw_event_layers()
        self._refresh_position_segment()

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

    def _labels_hidden(self, uuid):
        """Live sample review: flag words stay hidden on a sample event until
        this reviewer's decision is accept or reject (unsure keeps them
        hidden, so a revisit is not primed)."""
        if not self._sample_active or self._sample is None or \
                uuid not in self._sample['rows']:
            return False
        mine = self.epochs_panel._reviews.get(uuid)
        return not mine or mine[0] not in ('accept', 'reject')

    def _outlier_marks_hidden(self, uuid):
        """Live sample review: no event shows an outlier tint or label on
        the traces or the ticker, sample event or not, until this reviewer
        has accepted or rejected that event (then its own band may show
        it). Outside sample mode nothing is hidden."""
        if not self._sample_active:
            return False
        mine = self.epochs_panel._reviews.get(uuid)
        return not mine or mine[0] not in ('accept', 'reject')

    def _sample_line_for(self, uuid):
        """``(text, tooltip)`` of the Event panel's sample line (sample mode,
        sample events; ``('', '')`` otherwise).

        ``Sample event i of N · region · stage``. The `` · flagged: …`` /
        `` · not flagged`` part and the sampling-weight tooltip appear only
        after this reviewer's accept or reject: within a cell flagged events
        carry a lower weight, so the weight would give the flag away.
        ``is_shared`` never.
        """
        if not self._sample_active or self._sample is None:
            return '', ''
        srow = self._sample['rows'].get(uuid)
        if srow is None:
            return '', ''
        hidden = self._labels_hidden(uuid)
        text = (f"Sample event {self._sample['pos'][uuid] + 1} of "
                f"{len(self._sample['order'])} · "
                f"{_sr.stratum_text(srow, hide_flags=hidden)}")
        return text, ('' if hidden
                      else _sr.weight_tooltip(srow.get('weight')))

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
        conn = self.db.conn

        err, subject, prepared = None, self.subject, None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_bar.showMessage('Reading events…')
        QApplication.processEvents()
        try:
            # read the scope once when the library can hand the population
            # to each preview; otherwise each preview reads it again
            prepared = _sr.prepare(conn, scope)
            if prepared is not None:
                subject = _sr.prepared_subject(prepared) or subject
            else:
                subject = _sr.preview(conn, scope, 120, 1)[2].get(
                    'subject') or subject
        except ValueError as e:
            err = str(e)
        finally:
            QApplication.restoreOverrideCursor()
            self.status_bar.clearMessage()

        def preview_fn(size, seed, _sc=scope, _p=prepared):
            return _sr.preview(conn, _sc, size, seed, prepared=_p)
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
        dlg = DrawSampleDialog(subject, line, None if err else preview_fn,
                               existing_note=note, parent=self, error=err)
        self._draw_dialog = dlg
        if self._exec_dialog(dlg) != QtWidgets.QDialog.Accepted or err:
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

    def _open_precision_rule(self):
        """Review ▸ Precision rule…: the pooling rule, then redraw an open
        report with it."""
        dlg = PrecisionRuleDialog(self)
        self._rule_dialog = dlg
        if self._exec_dialog(dlg) == QtWidgets.QDialog.Accepted:
            rep = getattr(self, '_report', None)
            if rep is not None and rep.isVisible():
                rep._render()
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
        """Merge cached population checks into ``qc`` (a copy), or start the
        background read. Returns the merged frame. ``qc`` is kept as the
        base, so a Stage toggle re-merges without re-reading events."""
        self._qc_base = qc
        key = self._population_key(evt, methods, freq_band)
        cache = self.__dict__.setdefault('_pop_cache', {})
        res = cache.get(key)
        if res is None:
            self._start_population(key, evt, methods, freq_band)
            self._pop_view = (key, None)
            self.qc_widget.set_checks_context(
                False, event_type=evt, pending=True)
            self._apply_rule_tooltips()
            self.detail_dock_w.set_checks_context(
                True, event_type=evt, caption='Reading event checks…')
            self.detail_dock_w.set_check_state({}, (), None)
            self.detail_dock_w.flagged.set_rows([], 0, 0,
                                                'Reading event checks…')
            return qc
        return self._merge_population(qc, evt, key, res)

    def _apply_rule_tooltips(self):
        """The two rule tooltips with the live z limits: ``Amp flag`` header
        (amplitude rule) and ``CHECKS — FLAGGED CHANNELS`` (checks rule)."""
        hz, sz = self._qc_thresholds['hard_z'], self._qc_thresholds['soft_z']
        self.qc_widget.set_flag_limits(hz, sz)
        self.detail_dock_w.flagged.header.setToolTip(
            _er.checks_rule_tooltip(hz, sz))

    def _go_to_checks_list(self):
        """The header line's ``{c} checks flagged`` link: scroll the right
        dock to ``CHECKS — FLAGGED CHANNELS`` and flash its header in the
        accent colour for 1 s."""
        self.detail_dock.show()
        hdr = self.detail_dock_w.flagged.header
        scroll = self.detail_dock.widget()
        if isinstance(scroll, QtWidgets.QScrollArea):
            scroll.ensureWidgetVisible(hdr, 0, 40)
        base = hdr.styleSheet()
        hdr.setStyleSheet(base + f"color:{THEME['accent']};")
        QtCore.QTimer.singleShot(1000, lambda: hdr.setStyleSheet(base))

    def _check_stage_setting(self, evt):
        return f'review/check_stage/{evt}'

    def _merge_population(self, qc, evt, key, res):
        run = res.get('run') or {}
        method = str(run.get('method') or '')
        ratio = method_has_ratio(method)
        stages = list(res.get('stages') or [])
        combined = ChannelQCWidget.COMBINED
        stored = str(_review_settings().value(self._check_stage_setting(evt),
                                              combined) or combined)
        stage = stored if stored in stages and len(stages) > 1 else combined
        src = (res.get('by_stage', {}).get(stage) if stage != combined
               else res.get('pooled'))
        if src is None:
            src = res.get('pop')
        pop, med = _er.population_flags(
            src, hard_z=self._qc_thresholds['hard_z'],
            soft_z=self._qc_thresholds['soft_z'], event_type=evt,
            ratio_allowed=ratio, excluded=self._excluded_now(evt))
        # the check frame's own event count must not replace the table's
        # Events column (which follows the Filters dock, not the Stage toggle)
        pop_m = pop.rename(columns={'n': 'n_checks'})
        out = qc.drop(columns=[c for c in pop_m.columns
                               if c != 'channel' and c in qc.columns])
        out = out.merge(pop_m, on='channel', how='left') if len(pop_m) \
            else out
        if 'checks_flag' not in out.columns:
            out['checks_flag'] = ''
        out['checks_flag'] = out['checks_flag'].fillna('')
        params = run.get('params') or {}
        band = params.get('frequency')
        band = tuple(band) if band else None
        bounds = _er.run_duration_bounds(run, method)
        recorded = bool(res.get('recorded'))
        if [b.text() for b in self.qc_widget.stage_group.buttons()] != (
                stages if len(stages) == 1 else
                stages + ([' + '.join(stages)] if stages else [])):
            self.qc_widget.set_stages(stages, stage)
        stage_text = self.qc_widget.stage_text() or ' + '.join(stages)
        self.qc_widget.set_checks_context(
            recorded, method, ratio, evt,
            figures_off=bool(res.get('figures_off')), medians=med,
            bounds=bounds)
        self._apply_rule_tooltips()
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
        self._pop_view = (key, {'res': res, 'medians': med, 'pop': pop,
                                'stage': stage, 'stage_text': stage_text,
                                'band': band, 'bounds': bounds,
                                'ratio': ratio, 'method': method})
        self._refresh_flagged(out, evt)
        return out

    def _refresh_flagged(self, qc, evt):
        """Flagged-channel list, rings and the facts line from ``qc``."""
        view = self._pop_view[1] if self._pop_view else None
        dock = self.detail_dock_w
        if view is None:
            return
        recorded = bool(view['res'].get('recorded'))
        try:
            verdicts = {c: v for (c, et), v in
                        self.db.get_channel_verdicts().items() if et == evt}
        except Exception:
            verdicts = dict(self.qc_widget._verdicts)
        dropped = {c for c, v in verdicts.items()
                   if v in EXCLUDED_VERDICTS}
        bounds = view['bounds']
        min_dur = bounds[0] if bounds else None
        rows, flags = [], {}
        if recorded and len(qc) and 'checks_flag' in qc.columns:
            hit = qc[qc['checks_flag'].isin(['hard', 'soft'])]
            recs = hit.to_dict('records')
            recs.sort(key=lambda r: (r['checks_flag'] != 'hard',
                                     -float(r.get('checks_z') or 0)))
            for r in recs:
                ch = str(r['channel'])
                flags[ch] = (r['checks_flag'], float(r.get('checks_z') or 0))
                rows.append({'channel': ch, 'flag': r['checks_flag'],
                             'facts': _er.flagged_facts(
                                 r, evt, view['medians'], min_dur),
                             'tooltip': _er.flagged_tooltip(r),
                             'dropped': ch in dropped})
        n_hard = sum(1 for r in rows if r['flag'] == 'hard')
        n_soft = len(rows) - n_hard
        if not recorded:
            empty = _er.NOT_RECORDED_LIST
        else:
            empty = (f"No channel is flagged by the checks for "
                     f"{view['stage_text']}.")
        if self._sample_active:
            dock.flagged.set_rows([], '—', '—', dock.SAMPLE_HIDDEN)
            dock.flagged.counts.setText('—')
        else:
            dock.flagged.set_rows(rows, n_hard, n_soft, empty)
        dock.set_check_state(flags, dropped, (evt, view['stage_text'],
                                              view['band'], min_dur))

    def _on_check_stage(self, stage_key):
        """Stage toggle: persist per event type and re-merge from the base
        frame (no event read; density and amplitude columns unchanged)."""
        evt = self.qc_widget.current_event_type()
        _review_settings().setValue(self._check_stage_setting(evt), stage_key)
        base = getattr(self, '_qc_base', None)
        view = self._pop_view
        if base is None or view is None or view[1] is None:
            return
        qc = self._merge_population(base, evt, view[0], view[1]['res'])
        self._qc_df = qc
        self.qc_widget.set_data(qc, self._qc_events_df,
                                self.qc_widget._verdicts, self._redetect_queue)
        self.detail_dock_w.update_topo(qc)
        ch = getattr(self, '_qc_selected_channel', None)
        if ch:
            self._update_check_line(ch)

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
        base = getattr(self, '_qc_base', None)
        if view is None or view[0] != key or base is None:
            return
        qc = self._merge_population(base, key[1], key, res)
        self._qc_df = qc
        self.qc_widget.set_data(qc, self._qc_events_df,
                                self.qc_widget._verdicts, self._redetect_queue)
        self.detail_dock_w.update_topo(qc)
        self._refresh_status_segments()
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
        """The SELECTED CHANNEL facts line (facts only, no sentence)."""
        qc = getattr(self, '_qc_df', None)
        row = None
        if qc is not None and len(qc):
            hit = qc[qc['channel'] == ch]
            row = hit.iloc[0].to_dict() if len(hit) else None
        self.detail_dock_w.set_facts(ch, row,
                                     self.qc_widget.current_event_type())

    def _open_in_epochs(self, ch):
        """Open in Epochs (bottom bar or flagged list): drill the channel and,
        when it is checks-flagged, filter to its largest-z flagged column."""
        self.qc_widget.select_channel(ch)
        qc = getattr(self, '_qc_df', None)
        col = None
        if qc is not None and len(qc) and 'checks_flag' in qc.columns:
            hit = qc[qc['channel'] == ch]
            if len(hit):
                col = _er.top_flag_column(hit.iloc[0].to_dict())
        if col is None:
            self.on_qc_drill(ch, switch_tab=True)
            return
        self._on_check_link(ch, col)

    def _on_check_link(self, ch, col):
        """Dock phrase link: drill the channel and filter its events to
        those failing that check (spec section 1). Never during live sample
        review: the filter would show which undecided events fail a check."""
        self.on_qc_drill(ch, switch_tab=True)
        if self._sample_active:
            self.epochs_panel.clear_check_filter(emit=False)
            self.status_bar.showMessage(
                'Check filter not applied while you review the sample.')
            return
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
        # rank (1 = nearest) for the position rule only; no distances
        labels = _er.neighbour_labels(target, chans, source, interp)
        header = _er.neighbour_header(('~' + target) if target in interp
                                      else target, chans, source, window_s,
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

    def _refresh_epoch_table(self):
        """Give an open Epochs drill the epoch table of the annotation file
        now loaded (or the unstaged grid when none is), without re-drilling;
        an annotation opened after the drill otherwise leaves the grid."""
        ep = self.epochs_panel
        if ep._channel is None:
            return
        ep.set_epoch_table(self._epoch_table(), self._recording_seconds())

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

    def _all_exclusions(self):
        """Every excluded time range the reviewer saved (they apply to all
        channels), as row dicts."""
        if self.db is None:
            return []
        try:
            iv = self.db.get_qc_artefact_intervals()
        except Exception:
            return []
        return [] if iv is None or not len(iv) else iv.to_dict('records')

    def _refresh_exclusions(self):
        """Redraw the excluded ranges on the Epochs tab (no paging) and the
        dock's list after a change."""
        self.epochs_panel.set_exclusions(self._all_exclusions())
        ch = (self.epochs_panel._channel
              or getattr(self, '_qc_selected_channel', None))
        if ch:
            self.detail_dock_w.set_marked(self._marked_for(ch), channel=ch,
                                          total=self._total_marked())

    def _on_exclusion_row(self, interval_id):
        """A row of the dock's EXCLUDED TIME list: select that range on the
        Epochs tab and page to it."""
        if self.epochs_panel._channel is None:
            return
        self.tabs.setCurrentIndex(1)
        self.epochs_panel.select_exclusion(int(interval_id))

    def _mark_channel_artefact(self, ch, s, e):
        if self.db is None:
            return
        self.db.add_qc_artefact_interval(s, e, ch, self.reviewer_name)
        ok = self._write_review_qc_sidecar()
        self.refresh_qc_dashboard()
        self.epochs_panel.clear_range()
        self._refresh_exclusions()
        # after the refresh, whose own summary would replace it
        self.status_bar.showMessage(
            f"Excluded {_hms(s)}–{_hms(e)} for every channel. It takes "
            f"effect when you export a re-run package (File ▸ Export re-run "
            f"package…) and re-detect with it.")

    def _unmark_artefact(self, interval_id):
        """``Remove exclusion`` (or the dock list's ×): delete the range's
        row, rewrite the sidecar without it, refresh the density
        denominator, redraw (R5.2). No confirmation: brushing the same range
        again restores it."""
        if self.db is None:
            return
        row = next((r for r in self._all_exclusions()
                    if int(r['id']) == int(interval_id)), None)
        self.db.remove_qc_artefact_interval(int(interval_id))
        ok = self._write_review_qc_sidecar()
        self.refresh_qc_dashboard()
        if self.epochs_panel.selected_exclusion() == int(interval_id):
            self.epochs_panel.clear_range()
        self._refresh_exclusions()
        rng = (f" {_hms(row['start_time'])}–{_hms(row['end_time'])}"
               if row is not None else '')
        msg = (f"Removed exclusion{rng}. Export a new re-run package to "
               f"apply this at re-detection.")
        if not ok:
            # the database row is gone, but the review-qc record file could
            # not be rewritten: say so
            msg += (" The review-qc record was not updated because no "
                    "annotation file is loaded.")
        self.status_bar.showMessage(msg)

    def _rewrite_known_sidecar(self):
        """Rewrite the review-qc rater of the sidecar this session wrote
        (``self._review_qc_sidecar``) from ``qc_artefact_intervals``.
        False when there is none on disk or the rewrite fails."""
        side = getattr(self, '_review_qc_sidecar', None)
        if not side or not os.path.exists(side) or self.db is None:
            return False
        try:
            from wonambi.attr import Annotations as WAnn
            ann = WAnn(side)
            try:
                ann.remove_rater('review-qc')
            except Exception:
                pass
            ann.add_rater('review-qc')
            try:
                ann.add_event_type('Artefact')
            except Exception:
                pass
            for _, r in self.db.get_qc_artefact_intervals().iterrows():
                ann.add_event('Artefact', (float(r['start_time']),
                                           float(r['end_time'])),
                              chan='(all)')
            ann.save()
            return True
        except Exception as ex:
            self.status_bar.showMessage(f"Sidecar write failed: {ex}")
            return False

    def _write_review_qc_sidecar(self):
        """Write all qc_artefact_intervals into a SIDECAR Wonambi XML whose
        rater is literally 'review-qc'; back the original up to *.xml.bak on
        first write. The loaded sleep-scorer XML is never modified.

        The sidecar is a record of this review; nothing reads the
        ``review-qc`` rater at detection. Excluded time reaches detection
        only through ``Export re-run package…``, which writes it under the
        scorer rater the detector reads. Events are whole-montage
        (``chan='(all)'``); ``evidence_channel`` is provenance only."""
        if not getattr(self, 'annot_file_path', None) or \
                not os.path.exists(self.annot_file_path):
            # no annotation file now, but this session wrote a sidecar for
            # the recording: rewrite it in place from the database rows, so
            # it never keeps a range the review no longer has
            return self._rewrite_known_sidecar()
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
            # whole-montage, as detection and the density denominator treat
            # every Artefact; evidence_channel is provenance only
            iv = self.db.get_qc_artefact_intervals()
            for _, r in iv.iterrows():
                try:
                    ann.add_event('Artefact',
                                  (float(r['start_time']),
                                   float(r['end_time'])), chan='(all)')
                except Exception:
                    pass
            ann.save() if hasattr(ann, 'save') else ann.export(sidecar)
            self._review_qc_sidecar = sidecar
            return True
        except Exception as ex:
            self.status_bar.showMessage(f"Sidecar write failed: {ex}")
            return False





    def _excluded_now(self, evt=None):
        """Channels excluded for the event type in view (``verdict`` 'drop',
        or 'channel_artefact' from an older version)."""
        if self.db is None:
            return set()
        evt = evt or self.qc_widget.current_event_type()
        return {str(c) for (c, et), v in self.db.get_channel_verdicts().items()
                if et == evt and v in EXCLUDED_VERDICTS}

    def _set_channel_excluded(self, ch, on):
        """The one Exclude channel action (Channels bottom bar, flagged
        list, Epochs tab): writes ``verdict='drop'`` or ``''``. An excluded
        channel leaves the montage flag statistics, the topography, review
        samples and the re-run export; its events stay in the database."""
        if self.db is None or not ch:
            return
        evt = self.qc_widget.current_event_type()
        self.db.set_channel_verdict(str(ch), evt, 'drop' if on else '',
                                    self.reviewer_name)
        ev = _er.EVENT_PLURAL.get(evt, evt)
        self.refresh_qc_dashboard()
        self.status_bar.showMessage(
            f"Excluded {ch} from {ev}: left out of review samples, the "
            f"re-run export, the flag statistics and the topography." if on
            else f"Included {ch} again.")

    def _load_redetect_queue(self):
        """The re-detect queue as stored in ``channel_qc`` (kept across
        sessions)."""
        self._redetect_queue = (self.db.get_redetect_queue()
                                if self.db is not None else set())
        return self._redetect_queue

    def _queue_message(self):
        self.status_bar.showMessage(
            f"Re-detect queue: {len(self._redetect_queue)} channel(s)")

    def on_qc_add_redetect(self, ch):
        """Add the channel to the stored re-detect queue, or take it off."""
        if self.db is None or not ch:
            return
        ch = str(ch)
        self.db.set_channel_redetect(
            ch, ch not in self._load_redetect_queue(),
            self.qc_widget.current_event_type())
        self.refresh_qc_dashboard()
        self._queue_message()

    def on_qc_queue_all_hard(self):
        df = getattr(self, '_qc_df', None)
        if df is None or len(df) == 0 or self.db is None:
            return
        evt = self.qc_widget.current_event_type()
        for ch in df.loc[df['flag'] == 'hard', 'channel']:
            self.db.set_channel_redetect(str(ch), True, evt)
        self.refresh_qc_dashboard()
        self._queue_message()

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
            "trace; brush a time range and use \"Exclude time range…\" to "
            "leave it out of analysis for every channel. It takes effect "
            "when you export a re-run package (File ▸ Export re-run "
            "package…) and re-detect with it. A review-qc sidecar XML "
            "(rater <code>review-qc</code>) keeps a record of this review; "
            "it is not read by detection.</li>"
            "</ul>"
            "<p>The right dock carries a live scalp topography of the active "
            "QC metric (from the EEGLAB <code>.set</code> chanlocs) and a "
            "read-only worst-events list across all channels — click a row to "
            "drill into that channel and epoch. Re-detection: queue "
            "channels, then File ▸ Export re-run package… writes the channel "
            "list that examples/rerun_detection.py reads. This GUI never "
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
                if v in EXCLUDED_VERDICTS}

    def _redetect_queue_channels(self, verdicts=None):
        """The reviewer-selected re-detect queue as a sorted, deduped list,
        MINUS any dropped channel. A dropped channel is excluded from analysis
        entirely, so it must never land in the re-detect list even if it was
        also queued — the P3 driver's --channels consumes exactly this list to
        replace only these channels' events."""
        dropped = self._dropped_channels(verdicts)
        return sorted(set(map(str, self._redetect_queue)) - dropped)

    def setup_status_bar(self):
        """Segmented status bar: reviewer · position · save line."""
        self.status_bar = self.statusBar()
        # first permanent segment: who decisions are saved under
        self.seg_reviewer = QPushButton("Reviewer: not set")
        self.seg_reviewer.setFlat(True)
        self.seg_reviewer.setFocusPolicy(Qt.NoFocus)
        self.seg_reviewer.setStyleSheet("padding:0 8px;")
        self.seg_reviewer.clicked.connect(self._prompt_reviewer_name)
        self.status_bar.addPermanentWidget(self.seg_reviewer)

        def _seg(text, on_bar=True):
            q = QLabel(text)
            q.setStyleSheet("padding:0 8px;border-left:1px solid #262d39;")
            if on_bar:
                self.status_bar.addPermanentWidget(q)
            return q

        # spec section 15: reviewer · position · save line
        self.seg_position = _seg("—")
        self.seg_save = _seg(_er.SAVE_LINE_NO_NAME)
        # earlier segments kept as off-bar sinks (still filled; the Channels
        # tab shows the same facts in its header and bottom bar)
        self.seg_subject = _seg("—", False)
        self.seg_hard = _seg("0 hard outliers", False)
        self.seg_marked = _seg("0 channels excluded", False)
        self.seg_ranges = _seg("0 excluded time ranges", False)
        self.seg_queue = _seg("re-detect queue: 0", False)
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
                          if v in EXCLUDED_VERDICTS})
        self.seg_marked.setText(
            f"{marked} channel{'' if marked == 1 else 's'} excluded")
        nranges = 0
        if self.db is not None:
            try:
                nranges = len(self.db.get_qc_artefact_intervals())
            except Exception:
                nranges = 0
        self.seg_ranges.setText(
            f"{nranges} excluded time range{'' if nranges == 1 else 's'}")
        nq = len(self._redetect_queue)
        self.seg_queue.setText(f"re-detect queue: {nq}")
        self._refresh_position_segment()

    def _refresh_position_segment(self):
        """Status-bar position and save segments, and the top-bar hints."""
        if not hasattr(self, 'seg_position'):
            return
        ep = self.epochs_panel
        on_epochs = self.tabs.currentIndex() == 1
        if hasattr(self, 'lbl_detector'):
            text, tip = _er.detector_label(
                self._pop_view[1] if self._pop_view else None)
            self.lbl_detector.setText('detector: ' + text)
            self.lbl_detector.setToolTip(tip)
        if on_epochs and ep._channel is not None:
            self.seg_position.setText(
                f"{ep._channel} · epoch {ep._epoch + 1} / {ep._n_epochs()}")
        else:
            evt = self.qc_widget.current_event_type()
            qc = getattr(self, '_qc_df', None)
            view = self._pop_view[1] if self._pop_view else None
            method = (view or {}).get('method') or '—'
            band = (view or {}).get('band')
            btxt = f"{band[0]:g}–{band[1]:g} Hz" if band else '—'
            n = 0 if qc is None else len(qc)
            h = s_ = d = 0
            if qc is not None and len(qc):
                if 'checks_flag' in qc.columns:
                    h = int((qc['checks_flag'] == 'hard').sum())
                    s_ = int((qc['checks_flag'] == 'soft').sum())
                try:
                    d = sum(1 for (c, et), v in
                            self.db.get_channel_verdicts().items()
                            if et == evt and v in EXCLUDED_VERDICTS)
                except Exception:
                    d = 0
            ev = _er.EVENT_PLURAL.get(evt, evt)
            # live sample review: the number of check-flagged channels stays
            # out of sight here too
            checks = ('checks hidden' if self._sample_active
                      else f"{h} hard / {s_} soft checks")
            self.seg_position.setText(
                f"{ev[:1].upper()}{ev[1:]} · {method} · {btxt} · {n} channels "
                f"· {checks} · {d} channel(s) excluded")
        if self.db is None or not self.db.has_review_backend:
            self.seg_save.setText(_er.SAVE_LINE_NO_STORE if self.db is not None
                                  else _er.SAVE_LINE_NO_NAME)
        elif not self.reviewer_name:
            self.seg_save.setText(_er.SAVE_LINE_NO_NAME)
        else:
            self.seg_save.setText(_er.SAVE_LINE.format(
                db=os.path.basename(self.db.db_path)))
        self.key_hint_lbl.setText(_er.KEY_HINTS[
            'channels' if not on_epochs else
            'sample' if self._sample_active else 'epochs'])

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

                # files loaded for another recording go (R5 follow-up)
                self._review_qc_sidecar = None
                self._check_recording_files()
                # QC reframe: land on the per-channel dashboard
                self.refresh_qc_dashboard()
                self._refresh_chrome()
                self._show_unload_message()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load database: {str(e)}")

    def _recording_name(self):
        subs = self.db.recording_subjects() if self.db is not None else []
        if subs:
            return ', '.join(subs)
        return (os.path.basename(self.db.db_path) if self.db is not None
                else 'this database')

    def _check_recording_files(self):
        """Unload an annotation or EEG file that provably belongs to another
        recording than the open database's (its recorded subject id is not in
        the file name), and say so. A database with no subject recorded
        cannot be checked: the file stays, with a status note. Returns True
        when something was unloaded."""
        if self.db is None:
            return False
        subs = self.db.recording_subjects()
        msgs, notes = [], []
        ap = getattr(self, 'annot_file_path', None)
        ep_path = getattr(self, 'eeg_file_path', None)
        # no subject recorded: nothing to compare; keep the file, say so
        for p in (ap, ep_path):
            if p and _er.file_belongs(p, subs) is None:
                notes.append(f"Cannot check that "
                             f"{os.path.splitext(os.path.basename(p))[0]} "
                             f"belongs to this database (no subject "
                             f"recorded). Check it is the right recording.")
        if ap and _er.file_belongs(ap, subs) is False:
            self.annotations = None
            self.annot_file_path = None
            self._review_qc_sidecar = None
            self._refresh_epoch_table()
            msgs.append(f"Annotation file unloaded: it belongs to "
                        f"{os.path.splitext(os.path.basename(ap))[0]}. Load "
                        f"the annotation for {self._recording_name()}.")
        if ep_path and _er.file_belongs(ep_path, subs) is False:
            self.eeg_data = None
            self.eeg_file_path = None
            self._unit_cache = None
            try:
                self._refresh_physio_channels()
            except Exception:
                pass
            msgs.append(f"EEG file unloaded: it belongs to "
                        f"{os.path.splitext(os.path.basename(ep_path))[0]}. "
                        f"Load the EEG file for {self._recording_name()}.")
        if msgs or notes:
            self._unload_msg = ' '.join(msgs + notes)
            self.status_bar.showMessage(self._unload_msg)
            self._apply_recording_gates()
        return bool(msgs)

    def _show_unload_message(self):
        """Put the unload message back after a refresh replaced it."""
        msg = getattr(self, '_unload_msg', None)
        self._unload_msg = None
        if msg:
            self.status_bar.showMessage(msg)

    def annotation_ready(self):
        """A loaded annotation file that is not provably another
        recording's (a database with no subject recorded cannot be checked;
        the file is then trusted, with a status note on loading)."""
        ap = getattr(self, 'annot_file_path', None)
        if not ap or not os.path.exists(ap) or self.db is None:
            return False
        return _er.file_belongs(ap, self.db.recording_subjects()) is not False

    def _apply_recording_gates(self):
        """Exclude time range… and Export re-run package… need the matching
        annotation file (the package is built from it)."""
        ok = self.annotation_ready()
        tip = ('' if ok else f"Load the annotation file for "
               f"{self._recording_name() if self.db is not None else 'this recording'} "
               f"first (File ▸ Open Annotation File…).")
        self.epochs_panel.set_exclude_allowed(ok, tip)
        for a in (getattr(self, 'act_export_rerun', None),
                  getattr(self, 'act_export_rerun2', None)):
            if a is not None:
                a.setEnabled(ok)
                a.setToolTip(tip or RERUN_STATUS_TIP)
        link = self.qc_widget.queue_link
        link.setEnabled(ok)
        link.setToolTip(tip)

    def _refresh_chrome(self):
        """Re-derive subject and repaint title / toolbar / status segments."""
        self.subject = self._derive_subject()
        self._apply_recording_gates()
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
        # an EEG file of another recording than the open database goes again
        if self._check_recording_files():
            self.load_channels()
            self._refresh_chrome()
            self._show_unload_message()
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

    def _channel_unit(self, channel):
        """The unit the loaded file states for ``channel`` (``'µV'``,
        ``'mV'``, ...), or ``None`` when it states none. Read once per file
        (header, EDF physical dimension, or a BIDS ``*_channels.tsv`` beside
        the file)."""
        data = self.eeg_data
        key = (id(data), getattr(self, 'eeg_file_path', None))
        cache = getattr(self, '_unit_cache', None)
        if cache is None or cache[0] != key:
            header = getattr(data, 'header', None) if data is not None else None
            cache = (key, channel_units(header, key[1]))
            self._unit_cache = cache
        return cache[1].get(str(channel))

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

                self._review_qc_sidecar = None
                self.status_bar.showMessage(f"Annotations loaded: {os.path.basename(file_path)}")
                self._check_recording_files()

                self._refresh_chrome()
                if self.db is not None:
                    self.refresh_qc_dashboard()
                # a drill opened before this file still shows the grid
                self._refresh_epoch_table()
                self._show_unload_message()

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

    def _rerun_channel_lists(self):
        """``(kept, excluded, redetect)`` channel lists of the re-run
        package: excluded channels (any event type; 'drop' or an older
        'channel_artefact') are left out of ``channels.csv``; ``redetect`` is
        the stored queue minus the excluded channels
        (``redetect_channels.csv``)."""
        verdicts = self.db.get_channel_verdicts()
        dropped = sorted(self._dropped_channels(verdicts))
        kept = [c for c in self._all_db_channels() if c not in dropped]
        self._load_redetect_queue()
        return kept, dropped, self._redetect_queue_channels(verdicts)

    def _queued_but_excluded(self):
        """Queued channels left out of ``redetect_channels.csv`` because
        they are excluded (for any event type)."""
        return sorted(set(map(str, self._redetect_queue))
                      & self._dropped_channels())

    def export_rerun_package(self):
        """Snapshot originals, then write channels.csv + a sidecar XML the
        existing local detector scripts consume via --annot/--channels.

        Artefacts are appended under the rater the detector auto-selects
        (raters[0]) inside the SIDECAR copy — never the original (R4)."""
        if not self.db:
            QtWidgets.QMessageBox.warning(self, "Warning", "No database loaded")
            return
        kept, dropped, redetect = self._rerun_channel_lists()
        if getattr(self, 'annot_file_path', None) and \
                not self.annotation_ready():
            self.status_bar.showMessage(
                f"Load the annotation file for {self._recording_name()} "
                f"first (File ▸ Open Annotation File…).")
            return
        if not redetect and not dropped and not len(
                self.db.get_qc_artefact_intervals()):
            # nothing queued, excluded, or excluded in time: no package
            self.status_bar.showMessage(RERUN_EMPTY_TEXT)
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

        # every current exclusion goes into every package (a package replaces
        # the previous one); 'exported' only counts what is new in this one
        intervals = self.db.get_qc_artefact_intervals()
        n_new = int((intervals['exported'].fillna(0).astype(int) == 0).sum()) \
            if len(intervals) and 'exported' in intervals.columns \
            else len(intervals)

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
        evt = self.qc_widget.current_event_type()
        view = self._pop_view[1] if self._pop_view else None
        run = ((view or {}).get('res') or {}).get('run') or {}
        cmd = _er.rerun_command(
            evt, sidecar, chan_csv, redetect_csv,
            eeg=getattr(self, 'eeg_file_path', None), db=self.db.db_path,
            method=(view or {}).get('method') or None,
            band=(view or {}).get('band'),
            stages=_er.run_stages(run) or None)
        self._last_rerun_command = cmd
        redetect_line = (
            f"redetect_channels.csv: {len(redetect)} channel(s) queued for "
            f"re-detect\n"
            if redetect else "redetect_channels.csv: none queued (not written)\n")
        msg = (f"Snapshot: {backup}\n  ({', '.join(snapped) or 'nothing found to snapshot'})\n\n"
               + _er.rerun_summary(len(kept), dropped,
                                   self._queued_but_excluded()) +
               f"{redetect_line}"
               f"Excluded time ranges appended (all channels): {n_iv}"
               f" ({n_new} new since the last package)\n\n"
               f"Re-running detection OVERWRITES wonambi/*_results + the DB — "
               f"the snapshot above is your rollback.\n\n"
               + ((("Re-detects the queued channels only:" if redetect else
                    "Nothing is queued, so every kept channel "
                    "(channels.csv) is re-detected:")
                   + f"\n  {cmd}\n\n") if cmd else
                  f"There is no command-line re-run for {evt}; re-run it "
                  f"from turtlewave_gui with rerun_sidecar.xml.\n\n")
               + "Files written. OK.")
        if len(intervals):
            self.db.mark_artefact_intervals_exported(list(intervals['id']))
        QtWidgets.QMessageBox.information(self, "Re-run package ready", msg)
        self.status_bar.showMessage(
            f"Re-run package written to {backup}: {len(redetect)} "
            f"channel(s) queued." if redetect else RERUN_EMPTY_TEXT)

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
                fh.write("- " + _er.excluded_any_type_line(
                    self._dropped_channels(verdicts)) + "\n")
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
    app.setStyleSheet(DARK_QSS + checkbox_mark_qss())

    window = EventReviewGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
