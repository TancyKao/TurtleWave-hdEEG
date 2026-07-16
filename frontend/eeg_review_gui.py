#!/usr/bin/env python3
"""
TurtleWave Event Review GUI - Modern 3-Panel Design
Optimized for high-density EEG event review with virtualized table and timeline
"""

import sys
import sqlite3
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


# ============================================================================
# EventDatabase Class (from eeg_eventview.py)
# ============================================================================

class EventDatabase:
    """Enhanced database handler with automatic optimization"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
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
        
        # Performance PRAGMAs
        optimizations = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA mmap_size=268435456",
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
            "CREATE INDEX IF NOT EXISTS idx_reviewed ON events(reviewed)",
            "CREATE INDEX IF NOT EXISTS idx_eventtype_channel ON events(event_type, channel)",
            "CREATE INDEX IF NOT EXISTS idx_review_decision ON events(review_decision)",
            "CREATE INDEX IF NOT EXISTS idx_method ON events(method)",
            "CREATE INDEX IF NOT EXISTS idx_freq_band ON events(freq_lower, freq_upper)",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error:
                pass
        
        self.conn.commit()
    
    def create_review_tables(self):
        """Create additional columns for review functionality"""
        cursor = self.conn.cursor()
        
        # Only keep essential review columns
        new_columns = [
            ('reviewed', 'INTEGER DEFAULT 0'),
            ('review_decision', 'TEXT'),
            ('reviewer', 'TEXT'),
            ('review_timestamp', 'TEXT'),
            ('review_comments', 'TEXT'),
        ]
        
        for col_name, col_def in new_columns:
            try:
                cursor.execute(f'ALTER TABLE events ADD COLUMN {col_name} {col_def}')
            except sqlite3.OperationalError:
                pass  # Column already exists

        self.conn.commit()

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

    def get_events(self, event_type=None, channels=None, stages=None,
                   reviewed_only=False, unreviewed_only=False, confidence_threshold=0.0,
                   methods=None, freq_band=None, columns=None):
        """Get events with comprehensive filtering including method and freq_band.

        ``columns`` (list of column names) emits a lean ``SELECT col, ...``
        instead of ``SELECT *`` — used by the QC/Epochs path, which only needs
        a handful of columns out of 23. WHERE clauses can still reference
        non-selected columns. Default None keeps ``SELECT *`` for the Events
        tab, which needs the display columns.
        """
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
        
        # Filter by review status
        if reviewed_only:
            query += " AND reviewed = 1"
        elif unreviewed_only:
            query += " AND (reviewed = 0 OR reviewed IS NULL)"
        
        # Confidence threshold
        if confidence_threshold > 0:
            query += " AND (confidence_score >= ? OR confidence_score IS NULL)"
            params.append(confidence_threshold)
        
        query += " ORDER BY channel, start_time"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def add_review(self, uuid, decision, reviewer="", comments=""):
        """Add review decision for an event"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE events 
            SET reviewed = 1, review_decision = ?, review_comments = ?, 
                reviewer = ?, review_timestamp = ?
            WHERE uuid = ?
        ''', (decision, comments, reviewer, timestamp, uuid))
        self.conn.commit()
    
    def get_review_stats(self):
        """Get comprehensive review statistics"""
        cursor = self.conn.cursor()
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['total'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE reviewed = 1")
        stats['reviewed'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT review_decision, COUNT(*) 
            FROM events 
            WHERE reviewed = 1 
            GROUP BY review_decision
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
        """Export all reviewed events to CSV"""
        query = "SELECT * FROM events WHERE reviewed = 1 ORDER BY channel, start_time"
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_path, index=False)
        return len(df)


# ============================================================================
# QC-by-outlier-triage metrics
# ============================================================================

def _region_for(channel):
    """Coarse scalp region from an EGI-style ``E<idx>`` label. INDEX-based
    fallback used only when real electrode coordinates are unavailable — EGI
    numbering spirals around the head, so these buckets are approximate and do
    NOT track true scalp position (that's what ``_region_from_xy`` is for).
    Non-E labels → 'other'."""
    s = str(channel)
    if not s.startswith('E') or not s[1:].isdigit():
        return 'other'
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
# Timeline Overview Widget
# ============================================================================

class TimelineWidget(PlotWidget):
    """Slim hypnogram strip for the Spot-check Events tab. Color-codes each
    30-s epoch by stage via STAGE_COLOR (gray/magenta/blue/teal/green for
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
            if annotations is not None and hasattr(annotations, 'get_stages'):
                hyp = annotations.get_stages()
                if hyp: trec = len(hyp) * 30.0
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
    ('channel', 'Channel'), ('region', 'Region'), ('n', 'n'),
    ('mean_amp', 'mean µV*'), ('p95_amp', 'p95 µV*'),
    ('mean_p2p', 'mean p2p*'), ('max_p2p', 'max p2p*'),
    ('flag', 'flag'),
]
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

    def set_data(self, qc_df, verdicts=None):
        self.beginResetModel()
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
                return QtGui.QColor(40, 44, 52)
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
        return QVariant()


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
        self.topo = pg.PlotWidget(title="Topography")
        _theme_plot(self.topo)
        self.topo.setMaximumHeight(210)
        self.topo.hideAxis('bottom')
        self.topo.hideAxis('left')
        lay.addWidget(self.topo)

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
        self._render_topo_empty()
        lay.addStretch()

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
            if dur >= 30:
                dtxt = f"{int(round(dur / 30))} ep"
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
                lambda _=False, t=t0: self.gotoEpochRequested.emit(
                    int(t // 30)))
            x = QPushButton("×")
            x.setMaximumWidth(24)
            x.setStyleSheet("color:#f85149;font-weight:600;")
            x.clicked.connect(
                lambda _=False, i=mid: self.unmarkArtefactRequested.emit(i))
            rl.addWidget(lbl, 1)
            rl.addWidget(x)
            self._marked_layout.addWidget(row)

    def set_coords(self, coords):
        self._coords = coords or None
        self.update_topo(self._qc_df)

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
        # qc_row optionally carries _hypno + _event_type so the worst-list
        # can stage-tag rows and pick the right amplitude column without
        # widening update_channel's signature.
        hyp = None
        evt = None
        if qc_row is not None:
            try:
                hyp = qc_row.get('_hypno')
                evt = qc_row.get('_event_type')
            except Exception:
                pass
        amp_col = AMP_COL.get(str(evt), 'max_amp') if evt else 'max_amp'
        agg = _compute_epoch_outliers(df_slice, hypno=hyp, amp_col=amp_col)
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

        def _tip(x, y, data, _vl=label, _u=unit, _f=vfmt):
            ch, v = data
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


class ChannelQCWidget(QWidget):
    """Tab 0 — primary QC surface: sortable per-channel table + verdict
    actions + the selected-channel detail dock."""

    channelSelected = pyqtSignal(str)
    requestDrill = pyqtSignal(str)
    verdictChanged = pyqtSignal()
    addToRedetect = pyqtSignal(str)
    requestQueueAllHard = pyqtSignal()
    requestBuildRedetect = pyqtSignal()
    loadMontageRequested = pyqtSignal()

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
        self.flag_combo.addItems(['any', 'hard', 'soft', 'dead', 'ok'])
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

        self._qc_full = None      # full computed df for current event type
        self._events_slice = None  # montage-wide events df (current event type)
        self._verdicts = {}
        self._redetect_ref = set()  # window's queue (for button label state)
        self.flag_combo.currentTextChanged.connect(self._apply_flag_filter)

    def current_event_type(self):
        return self.evt_combo.currentText()

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
                "Unmark channel" if v in ('drop', 'channel_artefact')
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
        self.model.set_data(df, self._verdicts)

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
        self._verdict('keep' if cur in ('drop', 'channel_artefact')
                      else 'drop')

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


def _draw_hypnogram(pw, hypno, trec):
    """Render a stage-rank-line hypnogram, color-coded by stage via
    STAGE_COLOR. One short horizontal segment per 30-s epoch; segments from
    the same stage are joined as a polyline with NaN gaps so they render as
    one PlotDataItem per stage. Y: stage rank (Wake=4 top, N3=0 bottom).
    X: recording time (seconds). Caller must clear() and re-add any
    persistent overlays after this returns."""
    if not hypno:
        return
    rank = {'Wake': 4, 'W': 4, 'REM': 3,
            'N1': 2, 'NREM1': 2, 'Stage1': 2,
            'N2': 1, 'NREM2': 1, 'Stage2': 1,
            'N3': 0, 'NREM3': 0, 'Stage3': 0}
    n = len(hypno)
    ep = float(trec) / max(n, 1)
    by_stage = {}
    for i, s in enumerate(hypno):
        by_stage.setdefault(str(s), []).append(i)
    for stage_key, idxs in by_stage.items():
        col = STAGE_COLOR.get(stage_key, '#888888')
        y = rank.get(stage_key, 2)
        xs, ys = [], []
        for i in idxs:
            xs += [i * ep, (i + 1) * ep, float('nan')]
            ys += [y, y, float('nan')]
        pw.plot(xs, ys, pen=pg.mkPen(col, width=2), connect='finite')
    pw.setYRange(-0.5, 4.5, padding=0)
    pw.setXRange(0, float(trec), padding=0)
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


def _compute_epoch_outliers(df_slice, hypno=None, epoch_len=30.0,
                            amp_col='max_amp'):
    """Return DataFrame[idx, t0, n_events, n_outliers, max_amp, stage].

    Outlier rule: amp > median + 3.5*1.4826*MAD (see _mad_threshold),
    computed once over the whole (channel, event_type) df_slice. ``max_amp``
    in the returned frame is the chosen amp_col's max within the epoch —
    kept under that column name so callers don't need to know which metric
    was used. Cells with n_events==0 are omitted (strip handles as gaps).
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
    df['ep_idx'] = (df['_st'] // epoch_len).astype(int)
    g = df.groupby('ep_idx', sort=True).agg(
        n_events=('_st', 'size'),
        n_outliers=('_is_out', 'sum'),
        max_amp=('_amp', 'max'),
    ).reset_index().rename(columns={'ep_idx': 'idx'})
    g['t0'] = g['idx'] * epoch_len
    if hypno:
        g['stage'] = g['idx'].map(
            lambda i: str(hypno[i]) if 0 <= i < len(hypno) else '')
    else:
        g['stage'] = ''
    return g[cols]


class _EpochStripViewBox(pg.ViewBox):
    """ViewBox that captures Shift+drag and emits a snapped epoch range.
    Plain click is left to scene().sigMouseClicked on the parent plot."""

    sigShiftDrag = pyqtSignal(float, float, bool)  # t0, t1, is_finished

    def __init__(self, *a, epoch_len=30.0, **kw):
        super().__init__(*a, **kw)
        self._epoch_len = float(epoch_len)

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            ev.accept()
            p0 = self.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.mapSceneToView(ev.scenePos())
            t0, t1 = sorted([float(p0.x()), float(p1.x())])
            e0 = (int(t0 // self._epoch_len)) * self._epoch_len
            e1 = (int(t1 // self._epoch_len) + 1) * self._epoch_len
            self.sigShiftDrag.emit(e0, e1, ev.isFinish())
            return
        super().mouseDragEvent(ev, axis=axis)


class EpochsPanel(QWidget):
    """Tab 2 — paged 30-second epoch viewer with per-channel artefact triage.

    `self.plot` is the EPOCH STRIP (one bar per epoch: grey = regular events,
    red stacked on top = outliers under the rule amp > mean+3.5*sd over the
    drilled channel × event_type). Click an epoch bar to jump; Shift+drag
    selects an epoch-aligned range that can be marked as artefact via the
    "Mark N epochs as artefact" button.

    The main view is a fixed 30 s window: raw + band-filtered traces
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

    Keys: Left/Right step ±1 epoch (via button shortcuts); P/N jump to
    previous/next epoch with outliers; Esc clears any strip-range selection.
    """

    dropChannelRequested = pyqtSignal(str)
    globalArtefactConfirmed = pyqtSignal(float, float, str)
    markArtefactRequested = pyqtSignal(str, float, float)   # ch, t0, t1
    unmarkArtefactRequested = pyqtSignal(int)                # interval id
    requestChannel = pyqtSignal(str)                         # re-drill ch

    EPOCH_LEN = 30.0  # seconds — fixed window
    AXIS_W = 58       # shared left-axis column width (px) for ticker/raw/filt

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel = None
        self._all_events = None
        self._df = None
        self._event_type = 'slow_wave'
        self._trec = 1.0
        self._hypno = None
        self._epoch = 0          # current epoch index
        self._marked = []
        self._agg = None         # _compute_epoch_outliers result, per drill
        self._n_max = 1          # max(n_events) — strip y normalisation
        self._amp_col = AMP_COL.get('slow_wave', 'max_amp')
        self._amp_thr = float('inf')  # cached outlier threshold per drill
        self._amp_n = 0               # n events behind the threshold
        self._band = DEFAULT_BAND.get('slow_wave', (0.5, 2.0))
        self._band_label = ''         # band source, shown on filt header
        # injected by the main window:
        self.read_window = None   # (ch, t0, t1) -> (t, data, sfreq) | None
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
        self._strip_vb = _EpochStripViewBox(epoch_len=self.EPOCH_LEN)
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

    # ---- population ---------------------------------------------------
    def set_channel(self, channel, df_slice, all_events, event_type=None,
                     hypno=None, trec=None, marked=None):
        self._channel = channel
        self._all_events = all_events
        self._df = df_slice
        if event_type:
            self._event_type = event_type
        if trec:
            self._trec = float(trec)
        self._hypno = list(hypno) if hypno else None
        # event-type-aware amp column, then per-drill outlier aggregation
        self._amp_col = AMP_COL.get(str(self._event_type), 'max_amp')
        if self._amp_col not in (df_slice.columns
                                  if df_slice is not None else []):
            self._amp_col = 'max_amp'
        self._agg = _compute_epoch_outliers(
            df_slice, hypno=self._hypno, epoch_len=self.EPOCH_LEN,
            amp_col=self._amp_col)
        self._n_max = int(self._agg['n_events'].max()) \
            if len(self._agg) else 1
        # cache the robust outlier threshold once per drill (median + MAD;
        # used by the trace overlay + ticker for the active epoch)
        amp = (df_slice[self._amp_col]
               if df_slice is not None and self._amp_col in df_slice.columns
               else pd.Series(dtype=float))
        self._amp_thr, self._amp_n = _mad_threshold(amp)
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
                    start_ep = int(st.min() // self.EPOCH_LEN)
            except Exception:
                pass
        self._goto_epoch(start_ep)

    def _on_chan_combo(self, ch):
        if ch and ch != self._channel:
            # re-drill through the window (keeps DB/slice logic in one place)
            self.requestChannel.emit(ch)

    def _set_hypno(self, hypno):
        self.hypno.clear()
        # marker survives clear() by re-adding (LinearRegionItem, not data item)
        self.hypno.addItem(self._hypno_marker)
        if not hypno:
            return
        ymap = {'Wake': 4, 'W': 4, 'REM': 3, 'N1': 2, 'NREM1': 2,
                'Stage1': 2, 'N2': 1, 'NREM2': 1, 'Stage2': 1,
                'N3': 0, 'NREM3': 0, 'Stage3': 0}
        n = len(hypno)
        ep = self._trec / max(n, 1)
        xs, ys = [], []
        for i, s in enumerate(hypno):
            y = ymap.get(str(s), 2)
            xs += [i * ep, (i + 1) * ep]
            ys += [y, y]
        self.hypno.plot(xs, ys, pen=pg.mkPen((120, 160, 200), width=2),
                        connect='pairs')
        self.hypno.setXLink(self.plot)

    def _set_ranges(self, marked):
        self._marked = list(marked)
        self.ranges_list.clear()
        for m in self._marked:
            t0, t1 = float(m['start_time']), float(m['end_time'])
            it = QtWidgets.QListWidgetItem(
                f"{_hms(t0)} – {_hms(t1)}  ({int(round((t1 - t0) / 30))} ep)")
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
        self.plot.setXRange(0, max(self._trec, self.EPOCH_LEN), padding=0)
        if self._agg is None or len(self._agg) == 0:
            return
        idx = self._agg['idx'].to_numpy()
        n_ev = self._agg['n_events'].to_numpy(dtype=float)
        n_out = self._agg['n_outliers'].to_numpy(dtype=float)
        denom = max(1.0, float(self._n_max))
        H = 1.0
        h_reg = (n_ev - n_out) / denom * H
        h_out = n_out / denom * H
        centres = idx * self.EPOCH_LEN + self.EPOCH_LEN / 2.0
        # bottom (grey) layer — one item, vectorised
        self.plot.addItem(pg.BarGraphItem(
            x=centres, width=self.EPOCH_LEN * 0.95,
            y0=0, height=h_reg, brush=THEME['text_3'], pen=None))
        # top (red) layer stacked on top
        self.plot.addItem(pg.BarGraphItem(
            x=centres, width=self.EPOCH_LEN * 0.95,
            y0=h_reg, height=h_out, brush=THEME['bad'], pen=None))
        # marked-artefact bands (dashed purple, transparent fill)
        edge = pg.mkPen('#a371f7', width=1, style=Qt.DashLine)
        for m in self._marked:
            self.plot.addItem(pg.LinearRegionItem(
                values=[float(m['start_time']), float(m['end_time'])],
                brush=(0, 0, 0, 0), pen=edge, movable=False))

    def _draw_window_overlays(self):
        """Red strips for marked artefact ranges intersecting the current
        30 s window. Called only from _goto_epoch (right after raw/filt
        clears), so we never accumulate stale overlays."""
        t0 = self._epoch * self.EPOCH_LEN
        t1 = t0 + self.EPOCH_LEN
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
        i = self._epoch
        return str(self._hypno[i]) if 0 <= i < len(self._hypno) else ''

    def _n_epochs(self):
        return max(1, int(np.ceil(self._trec / self.EPOCH_LEN)))

    def _goto_epoch(self, i):
        n = self._n_epochs()
        i = max(0, min(int(i), n - 1))
        self._epoch = i
        t0 = i * self.EPOCH_LEN
        t1 = t0 + self.EPOCH_LEN
        stage = self._epoch_stage() or '—'
        n_ev, n_out = self._epoch_counts(i)
        self.epoch_lbl.setText(
            f"Epoch {i + 1}/{n} · {_hms(t0)}–{_hms(t1)} · {stage} · "
            f"{n_ev} events ({n_out} outlier{'s' if n_out != 1 else ''})")
        # move overview + hypno current-epoch markers + strip cursor
        self._ov_marker.setRegion([t0, t1])
        self._hypno_marker.setRegion([t0, t1])
        self._ep_cursor.setValue(t0 + self.EPOCH_LEN / 2.0)
        # render 30 s raw + filtered window + ticker
        self.raw_plot.clear()
        self.filt_plot.clear()
        self.ticker.clear()
        # re-add brush region (outline-only) centred in window; guard the
        # programmatic setRegion so it stays unfilled until the user drags.
        self._region_programmatic = True
        self.region.setRegion([t0 + 13.0, t0 + 17.0])
        self._region_programmatic = False
        self._region_fill(False)
        self.raw_plot.addItem(self.region)
        # lock to exactly 30 s
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
        self._draw_window_overlays()
        self._draw_trace_outliers(t0, t1)
        self._draw_ticker(t0, t1)

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

    def _draw_trace_outliers(self, t0, t1):
        """Red translucent overlay on raw + filt for each outlier event in
        the active epoch. raw/filt are cleared by _goto_epoch, so just add."""
        sub = self._epoch_events(t0, t1)
        if sub.empty:
            return
        for _, r in sub[sub['_is_out']].iterrows():
            s, e = float(r['_start']), float(r['_end'])
            for p in (self.raw_plot, self.filt_plot):
                p.addItem(pg.LinearRegionItem(
                    values=[s, e],
                    brush=(224, 83, 63, 46),
                    pen=pg.mkPen(224, 83, 63, 140),
                    movable=False))

    def _draw_ticker(self, t0, t1):
        # Pure visual indicator: grey bars = regular events, red (taller) =
        # outliers. No text — counts live in the epoch header line; the
        # outlier rule lives in the strip header at the top of the tab.
        sub = self._epoch_events(t0, t1)
        if sub.empty:
            return
        reg = sub[~sub['_is_out']]
        out = sub[sub['_is_out']]
        # Visible widths: 0.4 s reg / 0.6 s outlier — narrower than the
        # smallest spindle (~0.5 s) but reliably visible at 1920 px.
        if len(reg):
            x = reg['_start'].to_numpy(dtype=float)
            self.ticker.addItem(pg.BarGraphItem(
                x=x, width=0.4, y0=0, height=9,
                brush=THEME['text_3'], pen=None))
        if len(out):
            x = out['_start'].to_numpy(dtype=float)
            self.ticker.addItem(pg.BarGraphItem(
                x=x, width=0.6, y0=0, height=14,
                brush=THEME['bad'], pen=None))

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
        # Left/Right are wired as button shortcuts; P/N for outlier hopping;
        # Esc clears any strip-range selection.
        k = ev.key()
        if k == Qt.Key_P:
            self._prev_outlier(); return
        if k == Qt.Key_N:
            self._next_outlier(); return
        if k == Qt.Key_Escape:
            self._clear_strip_range(); return
        super().keyPressEvent(ev)

    # ---- strip shift-drag range ---------------------------------------
    def _on_shift_drag(self, t0, t1, finished):
        self._strip_range.setRegion([t0, t1])
        self._strip_range.show()
        n = max(0, int(round((t1 - t0) / self.EPOCH_LEN)))
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
        if e - s < self.EPOCH_LEN / 2:
            return
        self.markArtefactRequested.emit(self._channel, s, e)
        self._clear_strip_range()

    def _on_overview_click(self, ev):
        try:
            if ev.button() != Qt.LeftButton:
                return
            vb = self.plot.getPlotItem().vb
            p = vb.mapSceneToView(ev.scenePos())
            i = int(p.x() // self.EPOCH_LEN)
        except Exception:
            return
        self._goto_epoch(i)

    def _on_range_jump(self, item):
        try:
            t0 = float(item.data(Qt.UserRole + 1))
        except Exception:
            return
        self._goto_epoch(int(t0 // self.EPOCH_LEN))

    def _region_fill(self, on):
        self.region.setBrush(pg.mkBrush(88, 166, 255, 40) if on
                             else pg.mkBrush(0, 0, 0, 0))

    def _on_region_changed(self):
        # programmatic setRegion (epoch change / Clear) stays outline-only;
        # a genuine user drag restores the fill.
        if not self._region_programmatic:
            self._region_fill(True)

    def _reset_region(self):
        t0 = self._epoch * self.EPOCH_LEN
        self._region_programmatic = True
        self.region.setRegion([t0 + 13.0, t0 + 17.0])
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

    def populate_channels(self, channels):
        self.channel_list.clear()
        for ch in channels:
            it = QtWidgets.QListWidgetItem(str(ch))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, str(ch))
            self.channel_list.addItem(it)

    def decorate_channels(self, artefact_set=None, redetect_set=None):
        """Append ⚑ (channel-artefact verdict) / ↻ (re-detect queued)."""
        artefact_set = artefact_set or set()
        redetect_set = redetect_set or set()
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            base = str(it.data(Qt.UserRole))
            tag = ""
            if base in artefact_set:
                tag += " ⚑"
            if base in redetect_set:
                tag += " ↻"
            it.setText(base + tag)


class EventReviewGUI(QMainWindow):
    """Main event review GUI with 3-panel design"""
    
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1800, 1000)

        # Data
        self.db = None
        self.eeg_data = None
        self.annotations = None
        self.reviewer_name = "Reviewer1"
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
        self.selected_channels = ['E112', 'E118', 'Cz']
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
        rec = None
        try:
            if self.annotations is not None:
                stages = self.annotations.get_stages()
                if stages:
                    rec = len(stages) * 30.0
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
        self.detail_dock.setWidget(self.detail_dock_w)
        self.addDockWidget(Qt.RightDockWidgetArea, self.detail_dock)

        self.qc_widget.channelSelected.connect(self.on_qc_channel_selected)
        self.qc_widget.requestDrill.connect(self.on_qc_drill)
        self.qc_widget.verdictChanged.connect(self.on_qc_verdict_changed)
        self.qc_widget.addToRedetect.connect(self.on_qc_add_redetect)
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
        self.epochs_panel.read_window = self._read_eeg_window
        self._review_qc_sidecar = None

        # Always land on the Channels (QC) dashboard — it is the triage entry
        # point; the last-used tab is deliberately NOT restored.
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # QC dashboard wiring
    # ------------------------------------------------------------------

    _SCORED_STAGES = {'NREM1', 'NREM2', 'NREM3', 'REM',
                      'N1', 'N2', 'N3', 'Stage1', 'Stage2', 'Stage3'}

    def _scored_minutes(self):
        """Total minutes in scored sleep stages (shared density denominator).
        None when annotations are absent (density is then greyed)."""
        if self.annotations is None:
            return None
        try:
            stages = self.annotations.get_stages()
        except Exception:
            return None
        if not stages:
            return None
        n = sum(1 for s in stages if str(s) in self._SCORED_STAGES)
        return (n * 30.0) / 60.0 if n else None

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
        qc = compute_channel_qc(df, scored_minutes=self._scored_minutes(),
                                artefact_intervals=ivs,
                                coords=self.detail_dock_w._coords,
                                **self._qc_thresholds)
        self._qc_events_df = df
        self._qc_df = qc
        self.qc_widget.set_data(qc, df, verdicts, self._redetect_queue)
        self.detail_dock_w.set_event_type(evt)
        self.detail_dock_w.update_topo(qc)
        self.detail_dock_w.set_global_worst(
            self._global_worst_rows(df, evt), event_type=evt)
        # filter-dock decorations: ⚑ channel-artefact verdicts, ↻ queued
        artefact_set = {c for (c, e2), v in all_verdicts.items()
                        if v in ('drop', 'channel_artefact')}
        self.filter_dock.decorate_channels(artefact_set, self._redetect_queue)
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
                qc_row['_hypno'] = self._hypnogram()
                qc_row['_event_type'] = self.qc_widget.current_event_type()
            except Exception:
                pass
        self.detail_dock_w.update_channel(ch, sl, qc_row)
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
        hyp = self._hypnogram()
        rows = []
        for _, r in d.iterrows():
            stage = str(r['stage'])
            if (not stage or stage.lower() in ('', 'nan', 'none')) and hyp:
                i = int(r['start_time'] // 30)
                stage = hyp[i] if 0 <= i < len(hyp) else ''
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
            self.epochs_panel._goto_epoch(int(float(t0) // 30))
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

    def on_qc_drill(self, ch, switch_tab=True):
        df = getattr(self, '_qc_events_df', None)
        sl = df[df['channel'] == ch] if df is not None and len(df) else None
        evt = self.qc_widget.current_event_type()
        self.epochs_panel.set_channel(
            ch, sl, df, event_type=evt,
            hypno=self._hypnogram(), trec=self._recording_seconds(),
            marked=self._marked_for(ch))
        if switch_tab:
            self.tabs.setCurrentIndex(1)






    def _recording_seconds(self):
        try:
            if self.annotations is not None:
                stages = self.annotations.get_stages()
                if stages:
                    return len(stages) * 30.0
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

    def _hypnogram(self):
        try:
            if self.annotations is not None:
                st = self.annotations.get_stages()
                if st:
                    return [str(s) for s in st]
        except Exception:
            pass
        return None

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
        self.status_bar.showMessage(f"{ch} → {v} ({evt})")
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

    def _build_redetect_request(self):
        """Assemble the schema-v1 re-detect request dict."""
        verdicts = self.db.get_channel_verdicts() if self.db else {}
        excl = sorted(set(map(str, self._redetect_queue)) |
                      {str(c) for (c, _e), v in verdicts.items()
                       if v in ('drop', 'channel_artefact')})
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
            'exclude_epochs': epochs,
            'requested_at': datetime.now().isoformat(timespec='seconds'),
            'requested_by': self.reviewer_name,
        }

    def open_redetect_modal(self):
        """Full JSON-preview modal: write redetect_request.json beside the
        XML, optionally open turtlewave_gui (one-directional hand-off; this
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
        save = bb.addButton("Save request only",
                            QtWidgets.QDialogButtonBox.AcceptRole)
        save_open = bb.addButton("Save & open in turtlewave_gui",
                                 QtWidgets.QDialogButtonBox.AcceptRole)
        save_open.setObjectName("primary")
        cancel.clicked.connect(dlg.reject)
        save.clicked.connect(
            lambda: (self._write_redetect_json(req, False), dlg.accept()))
        save_open.clicked.connect(
            lambda: (self._write_redetect_json(req, True), dlg.accept()))
        lay.addWidget(bb)
        dlg.exec_()

    def _write_redetect_json(self, req, open_gui):
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
        if open_gui:
            try:
                from PyQt5.QtGui import QDesktopServices
                from PyQt5.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(out))
            except Exception:
                pass
    
    
    
    def setup_status_bar(self):
        """Segmented status bar: subject · hard · marked · ranges · queue ·
        build re-detect · reviewer."""
        self.status_bar = self.statusBar()

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
        self.seg_reviewer = _seg(f"reviewer: {self.reviewer_name}")
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
        self.seg_reviewer.setText(f"reviewer: {self.reviewer_name}")

    def setup_keyboard_shortcuts(self):
        """F flags the selected QC channel for re-detect."""
        QShortcut(QtGui.QKeySequence('F'), self, self._flag_selected_qc_row)

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
                self.status_bar.showMessage("Database loaded successfully - Select channels to load events")
                
                # Get database size
                db_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                self.db_size_label.setText(f"DB: {db_size_mb:.1f} MB")

                # Populate filter options + global channel list
                self.populate_filter_options()
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
        """Open EEG file using MNE or TurtleWave"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EEG File", "",
            "EEG Files (*.set *.edf *.bdf *.fif);;All Files (*)"
        )
        
        if file_path:
            try:
                self.status_bar.showMessage("Loading EEG file...")
                
                # Try TurtleWave LargeDataset first
                try:
                    self.eeg_data = LargeDataset(file_path, create_memmap=False)
                    self.eeg_file_path = file_path
                    self.status_bar.showMessage(f"EEG file loaded: {os.path.basename(file_path)}")
                except:
                    # Fallback to MNE
                    if mne:
                        if file_path.endswith('.set'):
                            self.eeg_data = mne.io.read_raw_eeglab(file_path, preload=False)
                        elif file_path.endswith('.edf'):
                            self.eeg_data = mne.io.read_raw_edf(file_path, preload=False)
                        elif file_path.endswith('.bdf'):
                            self.eeg_data = mne.io.read_raw_bdf(file_path, preload=False)
                        elif file_path.endswith('.fif'):
                            self.eeg_data = mne.io.read_raw_fif(file_path, preload=False)
                        
                        self.eeg_file_path = file_path
                        self.status_bar.showMessage(f"EEG file loaded (MNE): {os.path.basename(file_path)}")
                    else:
                        raise Exception("MNE not available and TurtleWave failed")
                
                # Start background waveform loader
                self.start_background_loader()
                self._refresh_chrome()
                # Auto-attempt live topo coords from the EEGLAB .set chanlocs
                self._autoload_set_coords()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load EEG file: {str(e)}")
                import traceback
                traceback.print_exc()
    
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
        """Populate the global filter-dock channel list (DB channels if a
        database is loaded, else a 1..256 placeholder)."""
        try:
            channels = self._all_db_channels() if self.db else []
            if not channels:
                channels = [f"E{i}" for i in range(1, 257)]
            self.channel_list.blockSignals(True)
            self.filter_dock.populate_channels(channels)
            self.channel_list.blockSignals(False)
            self.status_bar.showMessage(f"Loaded {len(channels)} channels")
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

        # ---- confirm + record ---------------------------------------------
        cmd = (f"python examples/hdEEG_sw_detector.py "
               f"--annot {sidecar} --channels {chan_csv}")
        msg = (f"Snapshot: {backup}\n  ({', '.join(snapped) or 'nothing found to snapshot'})\n\n"
               f"channels.csv: {len(kept)} kept, {len(dropped)} dropped\n"
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
