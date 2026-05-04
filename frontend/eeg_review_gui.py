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
                            QAction, QStatusBar, QToolBar, QShortcut)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal

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
    
    def get_events(self, event_type=None, channels=None, stages=None,
                   reviewed_only=False, unreviewed_only=False, confidence_threshold=0.0,
                   methods=None, freq_band=None):
        """Get events with comprehensive filtering including method and freq_band"""
        query = "SELECT * FROM events WHERE 1=1"
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
# Virtualized Table Model
# ============================================================================

class EventTableModel(QAbstractTableModel):
    """Virtualized table model for efficient display of large event lists"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.events = pd.DataFrame()
        self.headers = ['Time (HMS)', 'Chan', 'Type', 'Method', 'Freq(Hz)', 'Dur', 'Min(µV)', 'Max(µV)', 'Status']
        self.current_row = -1
        self.recording_start_time = None
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.events)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.events)):
            return QVariant()
        
        row = self.events.iloc[index.row()]
        col = index.column()
        
        if role == Qt.DisplayRole:
            if col == 0:  # Time (HMS)
                # Display absolute clock time if recording_start_time is available
                if self.recording_start_time is not None:
                    from datetime import datetime, timedelta
                    try:
                        # Parse start time
                        if isinstance(self.recording_start_time, str):
                            start_dt = datetime.fromisoformat(self.recording_start_time)
                        else:
                            start_dt = self.recording_start_time
                        
                        # Add relative seconds to get absolute time
                        relative_seconds = row.get('start_time', 0)
                        absolute_time = start_dt + timedelta(seconds=relative_seconds)
                        return absolute_time.strftime('%H:%M:%S')
                    except:
                        pass
                
                # Fallback: Display start_time_hms if available, otherwise convert from seconds
                if 'start_time_hms' in row and pd.notna(row['start_time_hms']):
                    return str(row['start_time_hms'])
                else:
                    # Convert seconds to HH:MM:SS (relative time)
                    total_seconds = int(row.get('start_time', 0))
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            elif col == 1:  # Channel
                return str(row['channel'])
            elif col == 2:  # Type
                event_type = str(row['event_type']).lower()
                abbrev = {'spindle': 'Spin', 'slow_wave': 'SW', 'k_complex': 'KC'}
                return abbrev.get(event_type, event_type[:3].upper())
            elif col == 3:  # Method
                method = row.get('method', '')
                return str(method) if pd.notna(method) else 'N/A'
            elif col == 4:  # Frequency Band
                freq_lower = row.get('freq_lower', None)
                freq_upper = row.get('freq_upper', None)
                if pd.notna(freq_lower) and pd.notna(freq_upper):
                    # Display with 2 decimal places to show exact values (e.g., 0.50-1.25 not 0.5-1.2)
                    return f"{freq_lower:.2f}-{freq_upper:.2f}"
                return 'N/A'
            elif col == 5:  # Duration
                return f"{row['duration']:.2f}s"
            elif col == 6:  # Min amplitude
                min_amp = row.get('min_amp', 0)
                return f"{min_amp:.1f}" if pd.notna(min_amp) else 'N/A'
            elif col == 7:  # Max amplitude
                max_amp = row.get('max_amp', 0)
                return f"{max_amp:.1f}" if pd.notna(max_amp) else 'N/A'
            elif col == 8:  # Status
                reviewed = row.get('reviewed', 0)
                return '●' if reviewed else '○'
        
        elif role == Qt.BackgroundRole:
            if index.row() == self.current_row:
                return QtGui.QColor(200, 230, 255)  # Highlight current
            reviewed = row.get('reviewed', 0)
            if reviewed:
                return QtGui.QColor(230, 255, 230)  # Green for reviewed
        
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        
        return QVariant()
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return QVariant()
    
    def set_events(self, events_df, recording_start_time=None):
        """Update events dataframe"""
        self.beginResetModel()
        self.events = events_df
        self.current_row = -1
        self.recording_start_time = recording_start_time
        self.endResetModel()
    
    def set_current_row(self, row):
        """Highlight current row"""
        old_row = self.current_row
        self.current_row = row
        
        if old_row >= 0:
            self.dataChanged.emit(self.index(old_row, 0), self.index(old_row, self.columnCount() - 1))
        if row >= 0:
            self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))


# ============================================================================
# Timeline Overview Widget
# ============================================================================

class TimelineWidget(PlotWidget):
    """Timeline overview showing event markers across entire session using PyQtGraph"""
    
    event_clicked = pyqtSignal(int)  # Emits event index when clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.events_df = pd.DataFrame()
        self.current_index = -1
        self.recording_start_time = None
        
        # Event type colors (RGB tuples for PyQtGraph)
        self.event_colors = {
            'spindle': (0, 0, 255),      # blue
            'slow_wave': (255, 255, 0),  # yellow
            'k_complex': (255, 128, 0),  # orange
            'artifact': (255, 0, 0)      # red
        }
        
        # Configure plot
        self.setBackground('w')
        self.showGrid(x=True, y=False, alpha=0.3)
        self.setLabel('bottom', 'Time (HH:MM)')
        self.setLabel('left', 'Sleep Stage')
        
        # Set fixed height for timeline
        self.setMaximumHeight(150)
        self.setMinimumHeight(120)
        
        # Enable mouse interaction
        self.scene().sigMouseClicked.connect(self.on_click)
        
        # Store plot items for legend
        self.legend = self.addLegend(offset=(10, 10))
        self.plot_items = {}
    
    def plot_timeline(self, events_df, current_index=-1, annotations=None, recording_start_time=None):
        """Plot timeline with ONLY sleep hypnogram (no event markers for performance)"""
        self.events_df = events_df
        self.current_index = current_index
        self.recording_start_time = recording_start_time
        
        self.clear()
        self.legend.clear()
        self.plot_items = {}
        
        if not annotations:
            text = pg.TextItem('Load annotation file to see hypnogram', anchor=(0.5, 0.5))
            self.addItem(text)
            return
        
        # Extract recording start time from annotations if available
        if annotations and recording_start_time is None:
            if hasattr(annotations, 'wonb_annot') and hasattr(annotations.wonb_annot, 'start_time'):
                self.recording_start_time = annotations.wonb_annot.start_time
        
        # Plot ONLY sleep hypnogram (no event markers - too slow)
        if annotations:
            self.plot_hypnogram(annotations, self.recording_start_time)
    
    def plot_hypnogram(self, annotations, recording_start_time=None):
        """Plot sleep hypnogram with proper time axis and auto-scaling"""
        try:
            from datetime import datetime, timedelta
            
            stages = annotations.get_stages()
            if not stages:
                return
            
            # Stage mapping - Wake at top, REM, then N1, N2, N3 at bottom
            stage_map = {'Wake': 0, 'REM': 1, 'NREM1': 2, 'NREM2': 3, 'NREM3': 4}
            stage_colors = {
                'Wake': (255, 255, 0, 80),      # yellow with alpha
                'NREM1': (173, 216, 230, 80),   # lightblue
                'NREM2': (0, 0, 255, 80),       # blue
                'NREM3': (0, 0, 139, 80),       # darkblue
                'REM': (255, 0, 0, 80)          # red
            }
            
            # Get recording start time from annotations
            if recording_start_time is None and hasattr(annotations, 'wonb_annot'):
                if hasattr(annotations.wonb_annot, 'start_time'):
                    recording_start_time = annotations.wonb_annot.start_time
            
            epoch_duration = 30  # seconds
            times = []
            stage_values = []
            unique_stages = set()
            
            for i, stage in enumerate(stages):
                epoch_time_seconds = i * epoch_duration
                times.append(epoch_time_seconds)
                stage_val = stage_map.get(stage, 0)
                stage_values.append(stage_val)
                unique_stages.add(stage_val)
            
            # Plot as step function
            if times:
                # Add colored regions first (background)
                for i in range(len(times) - 1):
                    stage = stages[i]
                    color = stage_colors.get(stage, (128, 128, 128, 80))
                    
                    # Create filled region
                    region = pg.LinearRegionItem(
                        values=[times[i], times[i+1]],
                        orientation='vertical',
                        brush=mkBrush(*color),
                        movable=False
                    )
                    # Hide the region boundary lines
                    region.lines[0].setPen(mkPen(None))
                    region.lines[1].setPen(mkPen(None))
                    self.addItem(region)
                
                # Main hypnogram line on top
                step_curve = pg.PlotCurveItem(
                    x=times,
                    y=stage_values,
                    stepMode='right',
                    pen=mkPen('k', width=2)
                )
                self.addItem(step_curve)
                
                # Auto-scale Y-axis based on actual stages present
                if unique_stages:
                    min_stage = min(unique_stages)
                    max_stage = max(unique_stages)
                    # Add padding
                    y_padding = 0.5
                    self.setYRange(min_stage - y_padding, max_stage + y_padding)
                else:
                    # Fallback to full range
                    self.setYRange(-0.5, 4.5)
                
                # Set y-axis ticks (only show stages that are present)
                y_ticks = [(v, k) for k, v in stage_map.items() if v in unique_stages]
                y_ticks.sort(key=lambda x: x[0])  # Sort by value
                self.getAxis('left').setTicks([y_ticks])
                
                # Auto-scale X-axis to fit all data
                if times:
                    self.setXRange(times[0], times[-1], padding=0.02)
                
                # Format x-axis with custom time labels using absolute clock time
                axis = self.getAxis('bottom')
                
                # Create custom axis item with time formatting
                class TimeAxisItem(pg.AxisItem):
                    def __init__(self, start_time, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.start_time = start_time
                    
                    def tickStrings(self, values, scale, spacing):
                        """Format ticks as HH:MM using absolute clock time"""
                        strings = []
                        for x in values:
                            if self.start_time is not None:
                                try:
                                    if isinstance(self.start_time, str):
                                        start_dt = datetime.fromisoformat(self.start_time)
                                    else:
                                        start_dt = self.start_time
                                    absolute_time = start_dt + timedelta(seconds=int(x))
                                    strings.append(absolute_time.strftime('%H:%M'))
                                except:
                                    total_seconds = int(x)
                                    hours = total_seconds // 3600
                                    minutes = (total_seconds % 3600) // 60
                                    strings.append(f"{hours:02d}:{minutes:02d}")
                            else:
                                total_seconds = int(x)
                                hours = total_seconds // 3600
                                minutes = (total_seconds % 3600) // 60
                                strings.append(f"{hours:02d}:{minutes:02d}")
                        return strings
                
                # Replace bottom axis with custom time axis
                self.plotItem.setAxisItems({'bottom': TimeAxisItem(recording_start_time, orientation='bottom')})
                
        except Exception as e:
            print(f"Error plotting hypnogram: {e}")
            import traceback
            traceback.print_exc()
    
    def on_click(self, event):
        """Handle mouse click to jump to event"""
        if self.events_df.empty:
            return
        
        # Get click position in data coordinates
        pos = self.plotItem.vb.mapSceneToView(event.scenePos())
        click_time = pos.x()
        
        # Find nearest event
        idx = (self.events_df['start_time'] - click_time).abs().idxmin()
        self.event_clicked.emit(idx)


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
        
        # Configure plot
        self.setBackground('w')
        self.showGrid(x=True, y=False, alpha=0.2)
        self.setLabel('bottom', 'Time (s)', **{'font-size': '10pt', 'font-weight': 'bold'})
        
        # Disable auto-range for better control
        self.enableAutoRange(False, False)
        
        # Store plot items
        self.channel_curves = []
        self.channel_labels = []
        self.event_items = []
    
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
            
            # Hide y-axis ticks
            self.getAxis('left').setTicks([])
            
            # Add vertical µV scale bar (50 µV)
            scale_bar_height = 50  # µV
            scale_bar_x = end_time - (end_time - start_time) * 0.05  # 5% from right edge
            scale_bar_y = y_max - y_spacing * 0.5  # Near top
            
            # Draw scale bar (vertical line)
            scale_bar = pg.PlotCurveItem(
                x=[scale_bar_x, scale_bar_x],
                y=[scale_bar_y, scale_bar_y + scale_bar_height],
                pen=mkPen('k', width=3)
            )
            self.addItem(scale_bar)
            self.event_items.append(scale_bar)
            
            # Add horizontal caps
            cap_width = (end_time - start_time) * 0.01
            top_cap = pg.PlotCurveItem(
                x=[scale_bar_x - cap_width/2, scale_bar_x + cap_width/2],
                y=[scale_bar_y + scale_bar_height, scale_bar_y + scale_bar_height],
                pen=mkPen('k', width=3)
            )
            bottom_cap = pg.PlotCurveItem(
                x=[scale_bar_x - cap_width/2, scale_bar_x + cap_width/2],
                y=[scale_bar_y, scale_bar_y],
                pen=mkPen('k', width=3)
            )
            self.addItem(top_cap)
            self.addItem(bottom_cap)
            self.event_items.extend([top_cap, bottom_cap])
            
            # Add label
            scale_label = pg.TextItem(
                '50 µV',
                anchor=(0, 0.5),
                color=(0, 0, 0),
                fill=mkBrush(255, 255, 255, 230),
                border=mkPen((0, 0, 0), width=1)
            )
            scale_label.setPos(scale_bar_x + cap_width, scale_bar_y + scale_bar_height/2)
            self.addItem(scale_label)
            self.event_items.append(scale_label)
            
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
        """Toggle filter on/off"""
        self.filter_enabled = enabled
        if self.current_event is not None and self.waveform_data is not None:
            # Replot with current settings
            channels = ['E112', 'E118', 'Cz']  # Default channels
            self.plot_event(self.current_event, self.waveform_data, channels)


# ============================================================================
# Main GUI Window
# ============================================================================

class EventReviewGUI(QMainWindow):
    """Main event review GUI with 3-panel design"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleWave Event Review")
        self.setGeometry(100, 100, 1800, 1000)
        
        # Data
        self.db = None
        self.eeg_data = None
        self.annotations = None
        self.current_events = pd.DataFrame()
        self.current_event_index = 0
        self.reviewer_name = "Reviewer1"
        self.recording_start_time = None
        
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
    
    def setup_menu_bar(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_db_action = QAction('Open Database...', self)
        open_db_action.triggered.connect(self.open_database)
        file_menu.addAction(open_db_action)
        
        open_eeg_action = QAction('Open EEG File...', self)
        open_eeg_action.triggered.connect(self.open_eeg_file)
        file_menu.addAction(open_eeg_action)
        
        open_annot_action = QAction('Open Annotation File...', self)
        open_annot_action.triggered.connect(self.open_annotation_file)
        file_menu.addAction(open_annot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Review menu
        review_menu = menubar.addMenu('Review')
        
        accept_action = QAction('Accept Event (A)', self)
        accept_action.triggered.connect(lambda: self.review_event('accept'))
        review_menu.addAction(accept_action)
        
        reject_action = QAction('Reject Event (R)', self)
        reject_action.triggered.connect(lambda: self.review_event('reject'))
        review_menu.addAction(reject_action)
        
        # Export menu
        export_menu = menubar.addMenu('Export')
        
        export_action = QAction('Export Reviewed Events...', self)
        export_action.triggered.connect(self.export_results)
        export_menu.addAction(export_action)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Navigation
        prev_action = QAction('◀ Prev', self)
        prev_action.triggered.connect(self.previous_event)
        toolbar.addAction(prev_action)
        
        next_action = QAction('Next ▶', self)
        next_action.triggered.connect(self.next_event)
        toolbar.addAction(next_action)
        
        toolbar.addSeparator()
        
        # Review
        accept_action = QAction('✓ Accept', self)
        accept_action.triggered.connect(lambda: self.review_event('accept'))
        toolbar.addAction(accept_action)
        
        reject_action = QAction('✗ Reject', self)
        reject_action.triggered.connect(lambda: self.review_event('reject'))
        toolbar.addAction(reject_action)
    
    def setup_ui(self):
        """Setup main UI with 3-panel design"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for 3 panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Channel selector
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Middle panel: Timeline + Event list + Detail plot
        middle_panel = self.create_middle_panel()
        splitter.addWidget(middle_panel)
        
        # Right panel: Navigation + Review
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes - make timeline shorter, detail plot wider
        splitter.setSizes([300, 1300, 200])
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self):
        """Create left panel with channel selector"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("Channel Selector")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Channel tree
        self.channel_tree = QTreeWidget()
        self.channel_tree.setHeaderLabels(["Channels"])
        self.channel_tree.itemChanged.connect(self.on_channel_changed)
        layout.addWidget(self.channel_tree)
        
        # Quick select buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self.select_all_channels)
        btn_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("None")
        select_none_btn.clicked.connect(self.deselect_all_channels)
        btn_layout.addWidget(select_none_btn)
        
        layout.addLayout(btn_layout)
        
        # Event type filter
        layout.addWidget(QLabel("Show Only:"))
        
        self.spindle_check = QCheckBox("Spindles")
        self.spindle_check.setChecked(True)
        self.spindle_check.stateChanged.connect(self.update_event_type_filter)
        layout.addWidget(self.spindle_check)

        self.slowwave_check = QCheckBox("Slow Waves")
        self.slowwave_check.setChecked(True)
        self.slowwave_check.stateChanged.connect(self.update_event_type_filter)
        layout.addWidget(self.slowwave_check)

        self.kcomplex_check = QCheckBox("K-Complexes")
        self.kcomplex_check.setChecked(True)
        self.kcomplex_check.stateChanged.connect(self.update_event_type_filter)
        layout.addWidget(self.kcomplex_check)
        
        # Method filter
        layout.addWidget(QLabel("Method Filter:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("All Methods")
        self.method_combo.currentIndexChanged.connect(self.update_method_filter)
        layout.addWidget(self.method_combo)
        
        # Frequency band filter
        layout.addWidget(QLabel("Freq Band Filter:"))
        self.freq_band_combo = QComboBox()
        self.freq_band_combo.addItem("All Frequencies")
        self.freq_band_combo.currentIndexChanged.connect(self.update_freq_band_filter)
        layout.addWidget(self.freq_band_combo)
        
        # Load channels button
        load_btn = QPushButton("Load Channels")
        load_btn.clicked.connect(self.load_channels)
        layout.addWidget(load_btn)
        
        layout.addStretch()
        
        return panel
    
    def create_middle_panel(self):
        """Create middle panel with timeline, event list, and detail plot"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Timeline overview
        timeline_group = QGroupBox("Timeline Overview (entire session)")
        timeline_layout = QVBoxLayout()
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.event_clicked.connect(self.jump_to_event)
        timeline_layout.addWidget(self.timeline_widget)
        timeline_group.setLayout(timeline_layout)
        layout.addWidget(timeline_group)
        
        # Event list (virtualized table)
        event_list_group = QGroupBox("Event List (Virtualized Table)")
        event_list_layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Status:"))
        
        self.pending_check = QCheckBox("Pending")
        self.pending_check.setChecked(True)
        filter_layout.addWidget(self.pending_check)
        
        self.accepted_check = QCheckBox("Accepted")
        filter_layout.addWidget(self.accepted_check)
        
        self.rejected_check = QCheckBox("Rejected")
        filter_layout.addWidget(self.rejected_check)
        
        filter_layout.addStretch()
        
        apply_filter_btn = QPushButton("Apply")
        apply_filter_btn.clicked.connect(self.apply_filters)
        filter_layout.addWidget(apply_filter_btn)
        
        event_list_layout.addLayout(filter_layout)
        
        # Table view
        self.event_table = QTableView()
        self.event_table_model = EventTableModel()
        self.event_table.setModel(self.event_table_model)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.selectionModel().currentRowChanged.connect(self.on_table_row_changed)
        self.event_table.doubleClicked.connect(self.on_table_double_click)
        event_list_layout.addWidget(self.event_table)
        
        # Event count label
        self.event_count_label = QLabel("Showing 0 of 0 events")
        event_list_layout.addWidget(self.event_count_label)
        
        event_list_group.setLayout(event_list_layout)
        layout.addWidget(event_list_group)
        
        # Current event detail
        detail_group = QGroupBox("Current Event Detail")
        detail_layout = QVBoxLayout()
        
        # Window duration control
        window_control_layout = QHBoxLayout()
        window_control_layout.addWidget(QLabel("Window Duration (s):"))
        self.window_duration_spin = QtWidgets.QSpinBox()
        self.window_duration_spin.setRange(5, 120)
        self.window_duration_spin.setValue(30)
        self.window_duration_spin.setSuffix(" s")
        self.window_duration_spin.valueChanged.connect(self.update_window_duration)
        window_control_layout.addWidget(self.window_duration_spin)
        window_control_layout.addStretch()
        detail_layout.addLayout(window_control_layout)
        
        # Filter toggle
        filter_control_layout = QHBoxLayout()
        self.filter_toggle = QCheckBox("Apply Filter")
        self.filter_toggle.stateChanged.connect(self.toggle_eeg_filter)
        filter_control_layout.addWidget(self.filter_toggle)
        
        filter_control_layout.addWidget(QLabel("HP:"))
        self.hp_spin = QtWidgets.QDoubleSpinBox()
        self.hp_spin.setRange(0.1, 100)
        self.hp_spin.setValue(0.5)
        self.hp_spin.setSuffix(" Hz")
        self.hp_spin.valueChanged.connect(self.update_filter_params)
        filter_control_layout.addWidget(self.hp_spin)
        
        filter_control_layout.addWidget(QLabel("LP:"))
        self.lp_spin = QtWidgets.QDoubleSpinBox()
        self.lp_spin.setRange(0.1, 100)
        self.lp_spin.setValue(30)
        self.lp_spin.setSuffix(" Hz")
        self.lp_spin.valueChanged.connect(self.update_filter_params)
        filter_control_layout.addWidget(self.lp_spin)
        
        filter_control_layout.addStretch()
        detail_layout.addLayout(filter_control_layout)
        
        # EEG plot
        self.eeg_widget = EEGDetailWidget()
        detail_layout.addWidget(self.eeg_widget)
        
        # Event info
        self.event_info_label = QLabel("Event Info: No event selected")
        detail_layout.addWidget(self.event_info_label)
        
        detail_group.setLayout(detail_layout)
        layout.addWidget(detail_group)
        
        # Set proportions - make timeline smaller, event list larger, detail plot larger
        layout.setStretch(0, 0)  # Timeline (fixed height ~120-150px)
        layout.setStretch(1, 2)  # Event list (larger)
        layout.setStretch(2, 3)  # Detail plot (larger)
        
        return panel
    
    def create_right_panel(self):
        """Create right panel with navigation and review"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        
        self.nav_label = QLabel("Event 0 / 0")
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        nav_layout.addWidget(self.nav_label)
        
        self.progress_bar = QProgressBar()
        nav_layout.addWidget(self.progress_bar)
        
        # Navigation buttons
        btn_layout = QVBoxLayout()
        
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self.previous_event)
        btn_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_event)
        btn_layout.addWidget(next_btn)
        
        prev_chan_btn = QPushButton("⬆ Prev Chan")
        prev_chan_btn.clicked.connect(self.previous_channel)
        btn_layout.addWidget(prev_chan_btn)
        
        next_chan_btn = QPushButton("⬇ Next Chan")
        next_chan_btn.clicked.connect(self.next_channel)
        btn_layout.addWidget(next_chan_btn)
        
        nav_layout.addLayout(btn_layout)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Review actions (simplified - only reject and flag)
        review_group = QGroupBox("Review Actions")
        review_layout = QVBoxLayout()
        
        info_label = QLabel("All events are auto-accepted.\nOnly reject or flag if needed.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        review_layout.addWidget(info_label)
        
        reject_btn = QPushButton("✗ Reject Event")
        reject_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        reject_btn.clicked.connect(lambda: self.review_event('reject'))
        review_layout.addWidget(reject_btn)
        
        flag_btn = QPushButton("⚠ Flag for Review")
        flag_btn.setStyleSheet("background-color: #FF9800; color: white;")
        flag_btn.clicked.connect(lambda: self.review_event('flag'))
        review_layout.addWidget(flag_btn)
        
        review_layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlaceholderText("Optional notes...")
        review_layout.addWidget(self.notes_edit)
        
        review_group.setLayout(review_layout)
        layout.addWidget(review_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Load data to see statistics")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Open database to begin")
        
        # Add permanent widgets
        self.db_size_label = QLabel("DB: 0 MB")
        self.status_bar.addPermanentWidget(self.db_size_label)
        
        self.last_saved_label = QLabel("Last saved: Never")
        self.status_bar.addPermanentWidget(self.last_saved_label)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        QShortcut(QtGui.QKeySequence('R'), self, lambda: self.review_event('reject'))
        QShortcut(QtGui.QKeySequence('F'), self, lambda: self.review_event('flag'))
        QShortcut(QtGui.QKeySequence(Qt.Key_Right), self, self.next_event)
        QShortcut(QtGui.QKeySequence(Qt.Key_Left), self, self.previous_event)
        QShortcut(QtGui.QKeySequence(Qt.Key_Up), self, self.previous_channel)
        QShortcut(QtGui.QKeySequence(Qt.Key_Down), self, self.next_channel)
    
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
                
                # DO NOT load any events by default
                # User must select channels first
                # self.apply_filters()
                
                # Populate filter options after database is loaded
                self.populate_filter_options()
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load database: {str(e)}")
    
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
                    # Pass recording start time to EEG widget for HMS display
                    if hasattr(self, 'eeg_widget'):
                        self.eeg_widget.recording_start_time = self.recording_start_time
                
                self.status_bar.showMessage(f"Annotations loaded: {os.path.basename(file_path)}")
                
                # ALWAYS update timeline with hypnogram when annotations loaded
                # Timeline is static and only needs to be drawn once
                self.timeline_widget.plot_timeline(
                    self.current_events,
                    self.current_event_index,
                    self.annotations,
                    self.recording_start_time
                )
                
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
        """Handle waveform loaded in background"""
        self.cache_lock.lock()
        self.waveform_cache[event_uuid] = waveform_data
        self.cache_lock.unlock()
        
        # If this is the current event, update display
        if not self.current_events.empty and self.current_event_index < len(self.current_events):
            current_event = self.current_events.iloc[self.current_event_index]
            if current_event['uuid'] == event_uuid:
                self.update_eeg_plot()
    
    def update_eeg_plot(self):
        """Update EEG plot with cached waveform"""
        if self.current_events.empty or self.current_event_index >= len(self.current_events):
            return
        
        # Check if EEG data is loaded
        if self.eeg_data is None:
            # Show message in plot area
            self.eeg_widget.clear()
            text = pg.TextItem('Load EEG file to view waveforms', anchor=(0.5, 0.5))
            self.eeg_widget.addItem(text)
            return
        
        event_row = self.current_events.iloc[self.current_event_index]
        event_uuid = event_row['uuid']
        
        # Get waveform from cache
        self.cache_lock.lock()
        waveform_data = self.waveform_cache.get(event_uuid)
        self.cache_lock.unlock()
        
        if waveform_data:
            self.eeg_widget.plot_event(event_row, waveform_data, self.selected_channels)
        else:
            # Load waveform immediately if not in cache
            try:
                # Get time window
                window_duration = self.window_duration_spin.value()
                event_center = (event_row['start_time'] + event_row['end_time']) / 2
                start_time = event_center - window_duration / 2
                end_time = event_center + window_duration / 2
                
                # Load waveform data
                waveform_data = self.eeg_data.read_data(chan=None, begtime=start_time, endtime=end_time)
                
                # Cache it
                self.cache_lock.lock()
                self.waveform_cache[event_uuid] = waveform_data
                self.cache_lock.unlock()
                
                # Plot it
                self.eeg_widget.plot_event(event_row, waveform_data, self.selected_channels)
                
            except Exception as e:
                print(f"Error loading waveform: {e}")
                import traceback
                traceback.print_exc()
                
                # Still queue for background loading as fallback
                if self.background_loader:
                    self.background_loader.queue_event(event_row)
    
    def load_channels(self):
        """Load available channels from EEG data"""
        try:
            # Temporarily disconnect the itemChanged signal to avoid triggering apply_filters
            self.channel_tree.itemChanged.disconnect(self.on_channel_changed)
            
            # Placeholder - would load from EEG file
            channels = [f"E{i}" for i in range(1, 257)]
            
            self.channel_tree.clear()
            
            # Create groups of 32
            for i in range(0, len(channels), 32):
                group_name = f"E{i+1}-E{min(i+32, len(channels))}"
                group_item = QTreeWidgetItem(self.channel_tree, [group_name])
                group_item.setFlags(group_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                
                for ch in channels[i:i+32]:
                    ch_item = QTreeWidgetItem(group_item, [ch])
                    ch_item.setFlags(ch_item.flags() | Qt.ItemIsUserCheckable)
                    ch_item.setCheckState(0, Qt.Unchecked)
            
            # Reconnect the signal
            self.channel_tree.itemChanged.connect(self.on_channel_changed)
            
            self.status_bar.showMessage(f"Loaded {len(channels)} channels")
            
        except Exception as e:
            print(f"Error loading channels: {e}")
            import traceback
            traceback.print_exc()
            # Make sure to reconnect signal even if error occurs
            try:
                self.channel_tree.itemChanged.connect(self.on_channel_changed)
            except:
                pass
    
    def apply_filters(self):
        """Apply current filters and load events - OPTIMIZED to skip timeline redraw"""
        if not self.db:
            return
        
        try:
            # Get event types
            event_types = []
            if self.spindle_check.isChecked():
                event_types.append('spindle')
            if self.slowwave_check.isChecked():
                event_types.append('slow_wave')
            if self.kcomplex_check.isChecked():
                event_types.append('k_complex')
            
            # Get review status
            reviewed_only = self.accepted_check.isChecked() or self.rejected_check.isChecked()
            unreviewed_only = self.pending_check.isChecked() and not reviewed_only
            
            # Get method filter
            selected_methods = None
            if hasattr(self, 'method_combo') and self.method_combo.currentIndex() > 0:
                selected_methods = [self.method_combo.currentText()]
            
            # Get frequency band filter
            selected_freq_band = None
            if hasattr(self, 'freq_band_combo') and self.freq_band_combo.currentIndex() > 0:
                freq_text = self.freq_band_combo.currentText()
                # Parse "X.X-Y.Y Hz" format
                try:
                    freq_parts = freq_text.replace(' Hz', '').split('-')
                    if len(freq_parts) == 2:
                        selected_freq_band = (float(freq_parts[0]), float(freq_parts[1]))
                except:
                    pass
            
            # Load events
            self.current_events = self.db.get_events(
                event_type=event_types if event_types else None,
                channels=self.selected_channels if self.selected_channels else None,
                reviewed_only=reviewed_only,
                unreviewed_only=unreviewed_only,
                confidence_threshold=0.0,
                methods=selected_methods,
                freq_band=selected_freq_band
            )
            
            # Update table
            self.event_table_model.set_events(self.current_events, self.recording_start_time)
            self.event_count_label.setText(f"Showing {len(self.current_events)} events")
            
            # Timeline is now static (hypnogram only), no need to redraw on filter changes
            # This dramatically improves performance
            
            # Show first event AND trigger waveform load
            if not self.current_events.empty:
                self.current_event_index = 0
                # Select first row in table to trigger display
                self.event_table.selectRow(0)
                self.update_event_display()
            else:
                # Clear display if no events
                self.eeg_widget.clear()
            
            # Update statistics
            self.update_statistics()
            
        except Exception as e:
            print(f"Error applying filters: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Navigation
    # ========================================================================
    
    def on_table_row_changed(self, current, previous):
        """Handle table row selection"""
        if current.isValid():
            self.current_event_index = current.row()
            self.update_event_display()
    
    def on_table_double_click(self, index):
        """Handle table double-click"""
        if index.isValid():
            self.current_event_index = index.row()
            self.update_event_display()
    
    def jump_to_event(self, index):
        """Jump to specific event"""
        if 0 <= index < len(self.current_events):
            self.current_event_index = index
            self.event_table.selectRow(index)
            self.update_event_display()
    
    def previous_event(self):
        """Navigate to previous event"""
        if self.current_event_index > 0:
            self.jump_to_event(self.current_event_index - 1)
    
    def next_event(self):
        """Navigate to next event"""
        if self.current_event_index < len(self.current_events) - 1:
            self.jump_to_event(self.current_event_index + 1)
    
    def previous_channel(self):
        """Navigate to previous channel's event"""
        if self.current_events.empty:
            return
        
        current_channel = self.current_events.iloc[self.current_event_index]['channel']
        channels = self.current_events['channel'].unique()
        current_ch_idx = list(channels).index(current_channel)
        
        if current_ch_idx > 0:
            prev_channel = channels[current_ch_idx - 1]
            # Find first event in previous channel
            idx = self.current_events[self.current_events['channel'] == prev_channel].index[0]
            self.jump_to_event(idx)
    
    def next_channel(self):
        """Navigate to next channel's event"""
        if self.current_events.empty:
            return
        
        current_channel = self.current_events.iloc[self.current_event_index]['channel']
        channels = self.current_events['channel'].unique()
        current_ch_idx = list(channels).index(current_channel)
        
        if current_ch_idx < len(channels) - 1:
            next_channel = channels[current_ch_idx + 1]
            # Find first event in next channel
            idx = self.current_events[self.current_events['channel'] == next_channel].index[0]
            self.jump_to_event(idx)
    
    def update_event_display(self):
        """Update display for current event - optimized to NOT redraw timeline every time"""
        if self.current_events.empty or self.current_event_index >= len(self.current_events):
            return
        
        event_row = self.current_events.iloc[self.current_event_index]
        
        # Update table highlight
        self.event_table_model.set_current_row(self.current_event_index)
        
        # OPTIMIZATION: Only update timeline current event marker, NOT full redraw
        # Full timeline redraw is VERY slow (960 epochs = 960 LinearRegionItems)
        # We only need to update the current event indicator line
        if hasattr(self, 'timeline_widget') and not self.current_events.empty:
            # Just update the current event marker position
            # The timeline hypnogram stays the same, only the red line moves
            pass  # Timeline will be updated only when needed (filter changes, etc.)
        
        # Update navigation label
        self.nav_label.setText(f"Event {self.current_event_index + 1} / {len(self.current_events)}")
        self.progress_bar.setMaximum(len(self.current_events))
        self.progress_bar.setValue(self.current_event_index + 1)
        
        # Update event info with HMS time
        channel = event_row.get('channel', 'N/A')
        start_time = event_row.get('start_time', 0)
        duration = event_row.get('duration', 0)
        
        # Convert start_time to HMS if recording_start_time is available
        if self.recording_start_time:
            from datetime import datetime, timedelta
            try:
                # Parse recording start time
                if isinstance(self.recording_start_time, str):
                    start_dt = datetime.fromisoformat(self.recording_start_time)
                else:
                    start_dt = self.recording_start_time
                
                # Calculate absolute time
                event_dt = start_dt + timedelta(seconds=start_time)
                time_str = event_dt.strftime('%H:%M:%S')
            except:
                # Fallback to seconds if conversion fails
                time_str = f"{start_time:.2f}s"
        else:
            time_str = f"{start_time:.2f}s"
        
        self.event_info_label.setText(
            f"Event Info: Channel {channel}, {time_str}, Duration {duration:.3f}s"
        )
        
        # Load and display waveform
        self.update_eeg_plot()
        
        # Prefetch surrounding events
        if self.background_loader and not self.current_events.empty:
            self.background_loader.queue_events_around(
                self.current_events,
                self.current_event_index,
                num_before=5,
                num_after=5
            )
    
    # ========================================================================
    # Review Actions
    # ========================================================================
    
    def review_event(self, decision):
        """Review current event"""
        if self.current_events.empty or self.current_event_index >= len(self.current_events):
            return
        
        event_row = self.current_events.iloc[self.current_event_index]
        uuid = event_row['uuid']
        comments = self.notes_edit.toPlainText()
        
        try:
            self.db.add_review(uuid, decision, self.reviewer_name, comments)
            
            # Update event in dataframe
            self.current_events.at[self.current_event_index, 'reviewed'] = 1
            self.current_events.at[self.current_event_index, 'review_decision'] = decision
            
            # Clear notes
            self.notes_edit.clear()
            
            # Move to next unreviewed
            self.find_next_unreviewed()
            
            # Update statistics
            self.update_statistics()
            
            # Update last saved time
            self.last_saved_label.setText(f"Last saved: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save review: {str(e)}")
    
    def find_next_unreviewed(self):
        """Find and jump to next unreviewed event"""
        # Look forward
        for i in range(self.current_event_index + 1, len(self.current_events)):
            if not self.current_events.iloc[i].get('reviewed', 0):
                self.jump_to_event(i)
                return
        
        # Look backward
        for i in range(0, self.current_event_index):
            if not self.current_events.iloc[i].get('reviewed', 0):
                self.jump_to_event(i)
                return
        
        # All reviewed, just go to next
        if self.current_event_index < len(self.current_events) - 1:
            self.next_event()
    
    # ========================================================================
    # UI Callbacks
    # ========================================================================
    
    def on_channel_changed(self, item, column):
        """Handle channel selection change - debounced to prevent freezing"""
        # Update selected channels list
        self.selected_channels = []
        
        for i in range(self.channel_tree.topLevelItemCount()):
            group_item = self.channel_tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                ch_item = group_item.child(j)
                if ch_item.checkState(0) == Qt.Checked:
                    self.selected_channels.append(ch_item.text(0))
        
        # Debounce: Wait 500ms after last channel change before filtering
        # This prevents freezing when clicking multiple channels rapidly
        self.channel_filter_timer.stop()
        self.channel_filter_timer.start(500)  # 500ms delay
    
    def apply_channel_filter(self):
        """Apply channel filter after debounce delay"""
        # Clear waveform cache since we're changing channels
        self.cache_lock.lock()
        self.waveform_cache.clear()
        self.cache_lock.unlock()
        
        # Reload events from database with selected channels
        self.apply_filters()
    
    def select_all_channels(self):
        """Select all channels"""
        for i in range(self.channel_tree.topLevelItemCount()):
            group_item = self.channel_tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                group_item.child(j).setCheckState(0, Qt.Checked)
    
    def deselect_all_channels(self):
        """Deselect all channels"""
        for i in range(self.channel_tree.topLevelItemCount()):
            group_item = self.channel_tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                group_item.child(j).setCheckState(0, Qt.Unchecked)
    
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
        
        # Reload events with new event type filter
        self.apply_filters()
    
    def update_method_filter(self):
        """Update method filter and reload events"""
        self.apply_filters()
    
    def update_freq_band_filter(self):
        """Update frequency band filter and reload events"""
        self.apply_filters()
    
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
    
    def update_window_duration(self, value):
        """Update the window duration for event detail display"""
        if hasattr(self, 'eeg_widget'):
            self.eeg_widget.set_window_duration(float(value))
    
    def toggle_eeg_filter(self, state):
        """Toggle EEG filter"""
        enabled = state == Qt.Checked
        self.eeg_widget.filter_settings = {
            'low': self.hp_spin.value(),
            'high': self.lp_spin.value()
        }
        self.eeg_widget.toggle_filter(enabled)
    
    def update_filter_params(self):
        """Update filter parameters when HP/LP values change"""
        if hasattr(self, 'eeg_widget') and self.filter_toggle.isChecked():
            self.eeg_widget.filter_settings = {
                'low': self.hp_spin.value(),
                'high': self.lp_spin.value()
            }
            # Refresh the current event display with new filter settings
            if not self.current_events.empty and self.current_event_index < len(self.current_events):
                self.update_event_display()
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.db:
            return
        
        try:
            stats = self.db.get_review_stats()
            
            total = stats.get('total', 0)
            reviewed = stats.get('reviewed', 0)
            accepted = stats.get('accept_count', 0)
            rejected = stats.get('reject_count', 0)
            
            percent = (reviewed / total * 100) if total > 0 else 0
            
            stats_text = f"""
{percent:.0f}% reviewed

Total: {total}
Reviewed: {reviewed}
Accepted: {accepted}
Rejected: {rejected}
            """
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def export_results(self):
        """Export reviewed events"""
        if not self.db:
            QtWidgets.QMessageBox.warning(self, "Warning", "No database loaded")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Reviewed Events", "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                count = self.db.export_reviewed_events(file_path)
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", 
                    f"Exported {count} reviewed events to {file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", 
                    f"Failed to export: {str(e)}"
                )
   
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
    
    window = EventReviewGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
