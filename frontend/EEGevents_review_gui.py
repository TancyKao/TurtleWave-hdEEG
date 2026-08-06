#!/usr/bin/env python3
"""
TurtleWave EEG Events Review GUI - Optimized Layout
Left Panel: Detection settings and channel selection
Center: Hypnogram + Dual signal views (Filtered + Broadband)
Right Panel: Event list with confidence indicators
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QGroupBox, QCheckBox, QComboBox, QSlider, 
                            QProgressBar, QTextEdit, QSplitter, QListWidget,
                            QListWidgetItem, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QScrollArea, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont

import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush

try:
    from turtlewave_hdEEG import LargeDataset, CustomAnnotations
    from scipy import signal as scipy_signal
    from frontend.data_manager import DataManager
except ImportError as e:
    print(f"Import warning: {e}")

try:
    from frontend.db_connect import connect_events_db
except ImportError:  # run as a script: frontend/ is on sys.path, not its parent
    from db_connect import connect_events_db


# ============================================================================
# EventDatabase Class
# ============================================================================

class EventDatabase:
    """Enhanced database handler with automatic optimization"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        # write=True: this class creates the review/QC tables, adds
        # columns and saves review decisions.
        self.conn = connect_events_db(db_path, write=True)
        self._auto_optimize()
        self.create_review_tables()
    
    def _auto_optimize(self):
        """Apply performance optimizations"""
        cursor = self.conn.cursor()
        # journal_mode is deliberately absent -- it is decided once, in
        # frontend/db_connect.py.
        optimizations = [
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",
            "PRAGMA temp_store=MEMORY",
        ]
        for pragma in optimizations:
            try:
                cursor.execute(pragma)
            except sqlite3.Error:
                pass
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_channel_starttime ON events(channel, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_reviewed ON events(reviewed)",
            "CREATE INDEX IF NOT EXISTS idx_eventtype ON events(event_type)",
        ]
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error:
                pass
        self.conn.commit()
    
    def create_review_tables(self):
        """Create review columns"""
        cursor = self.conn.cursor()
        new_columns = [
            ('reviewed', 'INTEGER DEFAULT 0'),
            ('review_decision', 'TEXT'),
            ('reviewer', 'TEXT'),
            ('review_timestamp', 'TEXT'),
        ]
        for col_name, col_def in new_columns:
            try:
                cursor.execute(f'ALTER TABLE events ADD COLUMN {col_name} {col_def}')
            except sqlite3.OperationalError:
                pass
        self.conn.commit()
    
    def get_events(self, event_type=None, channels=None, reviewed_only=False, unreviewed_only=False):
        """Get events with filtering"""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            if isinstance(event_type, list):
                placeholders = ','.join(['?' for _ in event_type])
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_type)
            else:
                query += " AND event_type = ?"
                params.append(event_type)
        
        if channels:
            placeholders = ','.join(['?' for _ in channels])
            query += f" AND channel IN ({placeholders})"
            params.extend(channels)
        
        if reviewed_only:
            query += " AND reviewed = 1"
        elif unreviewed_only:
            query += " AND (reviewed = 0 OR reviewed IS NULL)"
        
        query += " ORDER BY start_time, channel"
        return pd.read_sql_query(query, self.conn, params=params)
    
    def add_review(self, uuid, decision, reviewer=""):
        """Add review decision"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            UPDATE events 
            SET reviewed = 1, review_decision = ?, reviewer = ?, review_timestamp = ?
            WHERE uuid = ?
        ''', (decision, reviewer, timestamp, uuid))
        self.conn.commit()
    
    def get_review_stats(self):
        """Get review statistics"""
        cursor = self.conn.cursor()
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['total'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE reviewed = 1")
        stats['reviewed'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE review_decision = 'accept'")
        stats['accepted'] = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE review_decision = 'reject'")
        stats['rejected'] = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        return stats


# ============================================================================
# Hypnogram Widget
# ============================================================================

class HypnogramWidget(PlotWidget):
    """Compact hypnogram display"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.setFixedHeight(120)
        self.setLabel('left', 'Stage')
        self.showGrid(x=True, y=False, alpha=0.3)
        self.current_time_line = None
        
    def plot_hypnogram(self, annotations, current_time=None):
        """Plot sleep stages"""
        self.clear()
        
        if not annotations:
            return
        
        try:
            stages = annotations.get_stages()
            if not stages:
                return
            
            # Stage mapping: Wake at top, REM, N1, N2, N3 at bottom
            stage_map = {'Wake': 0, 'REM': 1, 'NREM1': 2, 'NREM2': 3, 'NREM3': 4}
            stage_colors = {
                'Wake': (255, 255, 0),    # Yellow
                'REM': (255, 0, 0),        # Red
                'NREM1': (173, 216, 230),  # Light blue
                'NREM2': (0, 0, 255),      # Blue
                'NREM3': (0, 0, 139)       # Dark blue
            }
            
            # Convert stages to values
            epoch_duration = 30  # seconds
            times = []
            stage_values = []
            
            for i, stage in enumerate(stages):
                time_sec = i * epoch_duration
                times.append(time_sec)
                stage_values.append(stage_map.get(stage, 0))
            
            # Add final time point for step plot
            if times:
                times.append(times[-1] + epoch_duration)
            
            # Plot as step function
            if times and stage_values:
                self.plot(times, stage_values, stepMode=True, pen=mkPen('k', width=2))
                
                # Add colored regions
                for i in range(len(times) - 1):
                    stage = stages[i]
                    color = stage_colors.get(stage, (128, 128, 128))
                    region = pg.LinearRegionItem(
                        values=[times[i], times[i+1]],
                        brush=mkBrush(*color, 100),
                        movable=False
                    )
                    region.lines[0].setPen(mkPen(None))
                    region.lines[1].setPen(mkPen(None))
                    self.addItem(region)
                
                # Set axis
                self.setYRange(-0.5, 4.5)
                y_axis = self.getAxis('left')
                y_axis.setTicks([[(v, k) for k, v in stage_map.items()]])
                
                # Current time marker
                if current_time is not None:
                    self.current_time_line = pg.InfiniteLine(
                        pos=current_time,
                        angle=90,
                        pen=mkPen('r', width=2)
                    )
                    self.addItem(self.current_time_line)
        
        except Exception as e:
            print(f"Error plotting hypnogram: {e}")


# ============================================================================
# Dual Signal View Widget
# ============================================================================

class DualSignalWidget(QWidget):
    """Dual signal view: Filtered + Broadband"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Filtered signal panel
        self.filtered_widget = SignalPlotWidget("Filtered Signal (11-16 Hz Spindle Band)", "±50µV | Fixed Scale")
        self.filtered_widget.set_filter_range(11, 16)
        self.filtered_widget.set_y_scale(50)
        layout.addWidget(self.filtered_widget)
        
        # Broadband signal panel
        self.broadband_widget = SignalPlotWidget("Broadband Signal (0.3-35 Hz)", "±100µV | Fixed Scale")
        self.broadband_widget.set_filter_range(0.3, 35)
        self.broadband_widget.set_y_scale(100)
        layout.addWidget(self.broadband_widget)
        
        # Equal sizing
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
    
    def plot_event(self, event_row, waveform_data, channels):
        """Plot event in both views"""
        self.filtered_widget.plot_event(event_row, waveform_data, channels)
        self.broadband_widget.plot_event(event_row, waveform_data, channels)


class SignalPlotWidget(QWidget):
    """Single signal plot with header"""
    
    def __init__(self, title, scale_info, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #E8F4FD; border: 1px solid #BEE5EB; padding: 3px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 2, 5, 2)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #004085;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        scale_label = QLabel(scale_info)
        scale_label.setStyleSheet("color: #666;")
        header_layout.addWidget(scale_label)
        
        layout.addWidget(header)
        
        # Plot widget
        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)  # Enable both X and Y grid for amplitude reference
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.getAxis('left').setTicks([])
        layout.addWidget(self.plot_widget)
        
        # Settings
        self.filter_low = 0.5
        self.filter_high = 30
        self.y_scale = 100
        self.sampling_rate = 500
        
    def set_filter_range(self, low, high):
        """Set filter frequency range"""
        self.filter_low = low
        self.filter_high = high
    
    def set_y_scale(self, scale):
        """Set Y-axis scale in µV"""
        self.y_scale = scale
    
    def plot_event(self, event_row, waveform_data, channels):
        """Plot EEG waveform"""
        self.plot_widget.clear()
        
        if waveform_data is None:
            return
        
        try:
            # Get sampling rate
            if hasattr(waveform_data, 'axis') and 's_freq' in waveform_data.axis:
                self.sampling_rate = waveform_data.axis['s_freq']
            
            # Time window (30 seconds centered on event)
            event_center = (event_row['start_time'] + event_row['end_time']) / 2
            start_time = event_center - 15
            end_time = event_center + 15
            
            n_samples = waveform_data.data[0].shape[1]
            time_axis = np.linspace(start_time, end_time, n_samples)
            
            # Channel spacing
            y_spacing = self.y_scale
            y_offset = 0
            
            channel_labels = waveform_data.axis['chan'][0]
            target_channel = event_row['channel']
            
            # Plot each channel
            for ch in channels:
                if ch in channel_labels:
                    ch_idx = np.where(channel_labels == ch)[0][0]
                    signal_data = waveform_data.data[0][ch_idx, :]
                    
                    # Apply filter
                    signal_data = self.apply_filter(signal_data)
                    
                    # Highlight target channel
                    is_target = ch == target_channel
                    color = (211, 47, 47) if is_target else (66, 66, 66)
                    linewidth = 2.5 if is_target else 1.2
                    
                    pen = mkPen(color=color, width=linewidth)
                    self.plot_widget.plot(time_axis, signal_data + y_offset, pen=pen)
                    
                    # Channel label
                    label = pg.TextItem(ch, anchor=(1, 0.5), color=color)
                    label.setPos(start_time - 0.5, y_offset)
                    self.plot_widget.addItem(label)
                    
                    y_offset += y_spacing
            
            # Event boundaries
            event_region = pg.LinearRegionItem(
                values=[event_row['start_time'], event_row['end_time']],
                brush=mkBrush(255, 205, 210, 64),
                movable=False
            )
            event_region.lines[0].setPen(mkPen(None))
            event_region.lines[1].setPen(mkPen(None))
            self.plot_widget.addItem(event_region)
            
            # Vertical lines at boundaries
            start_line = pg.InfiniteLine(pos=event_row['start_time'], angle=90, pen=mkPen((211, 47, 47), width=2))
            end_line = pg.InfiniteLine(pos=event_row['end_time'], angle=90, pen=mkPen((211, 47, 47), width=2))
            self.plot_widget.addItem(start_line)
            self.plot_widget.addItem(end_line)
            
            # Set ranges
            self.plot_widget.setXRange(start_time, end_time, padding=0)
            self.plot_widget.setYRange(-y_spacing/2, len(channels) * y_spacing - y_spacing/2, padding=0)
        
        except Exception as e:
            print(f"Error plotting signal: {e}")
    
    def apply_filter(self, data):
        """Apply bandpass filter"""
        try:
            nyquist = self.sampling_rate / 2
            low = self.filter_low / nyquist
            high = self.filter_high / nyquist
            low = max(0.001, min(low, 0.999))
            high = max(0.001, min(high, 0.999))
            
            if low >= high:
                return data
            
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, data)
            return filtered
        except:
            return data


# ============================================================================
# Event List Widget
# ============================================================================

class EventListWidget(QListWidget):
    """Event list with confidence indicators"""
    
    event_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.events_df = pd.DataFrame()
        self.itemClicked.connect(self.on_item_clicked)
        
    def set_events(self, events_df):
        """Populate event list"""
        self.clear()
        self.events_df = events_df
        
        for idx, row in events_df.iterrows():
            # Format time
            time_str = self.format_time(row['start_time'])
            
            # Get confidence (use power as proxy)
            confidence = row.get('power', 0)
            max_power = events_df['power'].max() if 'power' in events_df.columns else 1
            confidence_pct = int((confidence / max_power * 100)) if max_power > 0 else 0
            
            # Format text
            channel = row['channel']
            text = f"{time_str}\n{row['event_type']} | {channel} | {confidence_pct}%"
            
            # Create item
            item = QListWidgetItem(text)
            
            # Color based on confidence
            if confidence_pct >= 90:
                item.setBackground(QColor(200, 255, 200))  # Light green
            elif confidence_pct >= 70:
                item.setBackground(QColor(255, 255, 200))  # Light yellow
            else:
                item.setBackground(QColor(255, 230, 230))  # Light red
            
            # Check mark if reviewed
            if row.get('reviewed', 0):
                decision = row.get('review_decision', '')
                if decision == 'accept':
                    item.setText(text + " ✓")
                elif decision == 'reject':
                    item.setText(text + " ✗")
            
            self.addItem(item)
    
    def format_time(self, seconds):
        """Format time as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def on_item_clicked(self, item):
        """Handle item click"""
        row = self.row(item)
        self.event_selected.emit(row)


# ============================================================================
# Main GUI
# ============================================================================

class EEGEventsReviewGUI(QMainWindow):
    """Main GUI with optimized layout"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Review System - Sleep Spindle Detection")
        self.setGeometry(100, 100, 1600, 900)
        
        # Data
        self.db = None
        self.eeg_data = None
        self.annotations = None
        self.current_events = pd.DataFrame()
        self.current_event_index = 0
        self.selected_channels = []
        self.waveform_cache = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
    
    def setup_ui(self):
        """Setup main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for 2 panels (left + center)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (with collapsible event list)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Set sizes: Left=280, Center=expand
        splitter.setSizes([280, 1200])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self):
        """Create left panel with settings and collapsible event list"""
        panel = QWidget()
        panel.setMaximumWidth(280)
        
        # Create scroll area for left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)
        
        # Event Detection Settings
        settings_group = QGroupBox("Event Detection Settings")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        settings_group.setCheckable(True)
        settings_group.setChecked(True)
        settings_layout = QVBoxLayout()
        
        # Spindle Range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Spindle Range:"))
        self.spindle_min = QSpinBox()
        self.spindle_min.setRange(1, 30)
        self.spindle_min.setValue(11)
        self.spindle_min.setSuffix(" Hz")
        range_layout.addWidget(self.spindle_min)
        range_layout.addWidget(QLabel("-"))
        self.spindle_max = QSpinBox()
        self.spindle_max.setRange(1, 30)
        self.spindle_max.setValue(16)
        self.spindle_max.setSuffix(" Hz")
        range_layout.addWidget(self.spindle_max)
        settings_layout.addLayout(range_layout)
        
        # Min Duration
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Min Duration:"))
        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0.1, 5.0)
        self.min_duration.setValue(0.5)
        self.min_duration.setSingleStep(0.1)
        self.min_duration.setSuffix(" s")
        dur_layout.addWidget(self.min_duration)
        settings_layout.addLayout(dur_layout)
        
        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 10.0)
        self.threshold.setValue(2.5)
        self.threshold.setSingleStep(0.1)
        self.threshold.setSuffix(" σ")
        thresh_layout.addWidget(self.threshold)
        settings_layout.addLayout(thresh_layout)
        
        # Window
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.window = QSpinBox()
        self.window.setRange(5, 120)
        self.window.setValue(30)
        self.window.setSuffix(" s")
        window_layout.addWidget(self.window)
        settings_layout.addLayout(window_layout)
        
        # Fixed Scale
        self.fixed_scale_check = QCheckBox("Fixed Scale")
        self.fixed_scale_check.setChecked(True)
        settings_layout.addWidget(self.fixed_scale_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Channel Selection
        channel_group = QGroupBox("Channel Selection")
        channel_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        channel_layout = QVBoxLayout()
        
        # Channel list
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.MultiSelection)
        channel_layout.addWidget(self.channel_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self.select_all_channels)
        btn_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("None")
        select_none_btn.clicked.connect(self.deselect_all_channels)
        btn_layout.addWidget(select_none_btn)
        channel_layout.addLayout(btn_layout)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Detected Events (collapsible with event list)
        events_group = QGroupBox("Detected Events")
        events_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        events_group.setCheckable(True)
        events_group.setChecked(True)
        events_layout = QVBoxLayout()
        
        # Summary
        self.events_summary = QLabel("Total: 0\nReviewed: 0\nPending: 0")
        events_layout.addWidget(self.events_summary)
        
        # Event list
        self.event_list_widget = EventListWidget()
        self.event_list_widget.event_selected.connect(self.on_event_selected)
        self.event_list_widget.setMaximumHeight(300)
        events_layout.addWidget(self.event_list_widget)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self.previous_event)
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_event)
        nav_layout.addWidget(next_btn)
        events_layout.addLayout(nav_layout)
        
        # Review buttons
        accept_btn = QPushButton("✓ Accept (A)")
        accept_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        accept_btn.clicked.connect(lambda: self.review_event('accept'))
        events_layout.addWidget(accept_btn)
        
        reject_btn = QPushButton("✗ Reject (R)")
        reject_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        reject_btn.clicked.connect(lambda: self.review_event('reject'))
        events_layout.addWidget(reject_btn)
        
        events_group.setLayout(events_layout)
        layout.addWidget(events_group)
        
        # Load button
        load_btn = QPushButton("Load Data")
        load_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        load_btn.clicked.connect(self.load_data)
        layout.addWidget(load_btn)
        
        layout.addStretch()
        
        scroll.setWidget(scroll_content)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        return panel
    
    def create_center_panel(self):
        """Create center panel with hypnogram and dual signals"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with current info
        header = QFrame()
        header.setStyleSheet("background-color: #E8F4FD; border: 1px solid #BEE5EB; padding: 5px;")
        header_layout = QHBoxLayout(header)
        
        title_label = QLabel("Sleep Hypnogram - Night Recording")
        title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.current_info_label = QLabel("Current: N2 Sleep | 02:34:46")
        header_layout.addWidget(self.current_info_label)
        
        layout.addWidget(header)
        
        # Hypnogram
        self.hypnogram_widget = HypnogramWidget()
        layout.addWidget(self.hypnogram_widget)
        
        # Dual signal views
        self.dual_signal_widget = DualSignalWidget()
        layout.addWidget(self.dual_signal_widget)
        
        # Set proportions
        layout.setStretch(0, 0)  # Header
        layout.setStretch(1, 0)  # Hypnogram (fixed height)
        layout.setStretch(2, 1)  # Dual signals (expand)
        
        return panel
    
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Load data to begin review")
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    
    def load_data(self):
        """Load database, EEG, and annotations"""
        try:
            # Open database
            db_path, _ = QFileDialog.getOpenFileName(
                self, "Select Events Database", "",
                "Database Files (*.db *.sqlite);;All Files (*)"
            )
            if not db_path:
                return
            
            self.db = EventDatabase(db_path)
            self.status_bar.showMessage("Database loaded successfully")
            
            # Open EEG file
            eeg_path, _ = QFileDialog.getOpenFileName(
                self, "Select EEG File", "",
                "EEG Files (*.set *.edf *.bdf *.fif);;All Files (*)"
            )
            if not eeg_path:
                return
            
            # Try TurtleWave LargeDataset first
            try:
                self.eeg_data = LargeDataset(eeg_path, create_memmap=False)
                self.status_bar.showMessage(f"EEG file loaded: {os.path.basename(eeg_path)}")
            except:
                # Fallback to MNE if available
                try:
                    import mne
                    if eeg_path.endswith('.set'):
                        self.eeg_data = mne.io.read_raw_eeglab(eeg_path, preload=False)
                    elif eeg_path.endswith('.edf'):
                        self.eeg_data = mne.io.read_raw_edf(eeg_path, preload=False)
                    elif eeg_path.endswith('.bdf'):
                        self.eeg_data = mne.io.read_raw_bdf(eeg_path, preload=False)
                    elif eeg_path.endswith('.fif'):
                        self.eeg_data = mne.io.read_raw_fif(eeg_path, preload=False)
                    self.status_bar.showMessage(f"EEG file loaded (MNE): {os.path.basename(eeg_path)}")
                except:
                    raise Exception("Failed to load EEG file with both TurtleWave and MNE")
            
            # Populate channels
            self.populate_channels()
            
            # Open annotations
            annot_path, _ = QFileDialog.getOpenFileName(
                self, "Select Annotation File", "",
                "XML Files (*.xml);;All Files (*)"
            )
            if not annot_path:
                return
            
            self.annotations = CustomAnnotations(annot_path)
            
            # Extract recording start time
            if hasattr(self.annotations, 'wonb_annot') and hasattr(self.annotations.wonb_annot, 'start_time'):
                self.recording_start_time = self.annotations.wonb_annot.start_time
            
            self.status_bar.showMessage(f"Annotations loaded: {os.path.basename(annot_path)}")
            
            # Plot hypnogram
            self.hypnogram_widget.plot_hypnogram(self.annotations)
            
            # Load events
            self.load_events()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def populate_channels(self):
        """Populate channel list"""
        self.channel_list.clear()
        
        if hasattr(self.eeg_data, 'channels'):
            channels = self.eeg_data.channels
        else:
            channels = ['C3', 'C4', 'Cz', 'F3', 'F4']
        
        for ch in channels:
            self.channel_list.addItem(ch)
        
        # Select first few by default
        for i in range(min(3, len(channels))):
            self.channel_list.item(i).setSelected(True)
        
        self.update_selected_channels()
    
    def select_all_channels(self):
        """Select all channels"""
        self.channel_list.selectAll()
        self.update_selected_channels()
    
    def deselect_all_channels(self):
        """Deselect all channels"""
        self.channel_list.clearSelection()
        self.update_selected_channels()
    
    def update_selected_channels(self):
        """Update selected channels list"""
        self.selected_channels = [item.text() for item in self.channel_list.selectedItems()]
    
    def load_events(self):
        """Load events from database"""
        if not self.db:
            return
        
        try:
            # Get events
            self.current_events = self.db.get_events(event_type='spindle')
            
            # Update event list
            self.event_list_widget.set_events(self.current_events)
            
            # Update summary
            total = len(self.current_events)
            reviewed = len(self.current_events[self.current_events['reviewed'] == 1])
            pending = total - reviewed
            
            self.events_summary.setText(f"Total: {total}\nReviewed: {reviewed}\nPending: {pending}")
            
            # Show first event
            if not self.current_events.empty:
                self.current_event_index = 0
                self.update_event_display()
            
        except Exception as e:
            print(f"Error loading events: {e}")
    
    # ========================================================================
    # Event Navigation
    # ========================================================================
    
    def on_event_selected(self, index):
        """Handle event selection from list"""
        self.current_event_index = index
        self.update_event_display()
    
    def previous_event(self):
        """Navigate to previous event"""
        if self.current_event_index > 0:
            self.current_event_index -= 1
            self.event_list_widget.setCurrentRow(self.current_event_index)
            self.update_event_display()
    
    def next_event(self):
        """Navigate to next event"""
        if self.current_event_index < len(self.current_events) - 1:
            self.current_event_index += 1
            self.event_list_widget.setCurrentRow(self.current_event_index)
            self.update_event_display()
    
    def update_event_display(self):
        """Update display for current event"""
        if self.current_events.empty or self.current_event_index >= len(self.current_events):
            return
        
        event_row = self.current_events.iloc[self.current_event_index]
        
        # Update current info
        time_str = self.event_list_widget.format_time(event_row['start_time'])
        stage = event_row.get('stage', 'N/A')
        self.current_info_label.setText(f"Current: {stage} | {time_str}")
        
        # Update hypnogram marker
        if self.annotations:
            self.hypnogram_widget.plot_hypnogram(self.annotations, event_row['start_time'])
        
        # Load waveform
        self.load_and_display_waveform(event_row)
    
    def load_and_display_waveform(self, event_row):
        """Load and display waveform for event"""
        try:
            # Get time window
            event_center = (event_row['start_time'] + event_row['end_time']) / 2
            start_time = event_center - 15
            end_time = event_center + 15
            
            # Load waveform
            waveform_data = self.eeg_data.read_data(chan=None, begtime=start_time, endtime=end_time)
            
            # Plot in dual views
            if self.selected_channels:
                self.dual_signal_widget.plot_event(event_row, waveform_data, self.selected_channels)
        
        except Exception as e:
            print(f"Error loading waveform: {e}")
    
    # ========================================================================
    # Review Actions
    # ========================================================================
    
    def review_event(self, decision):
        """Review current event"""
        if self.current_events.empty or self.current_event_index >= len(self.current_events):
            return
        
        event_row = self.current_events.iloc[self.current_event_index]
        uuid = event_row['uuid']
        
        # Save review
        self.db.add_review(uuid, decision, "Reviewer1")
        
        # Update display
        self.current_events.at[self.current_event_index, 'reviewed'] = 1
        self.current_events.at[self.current_event_index, 'review_decision'] = decision
        
        # Refresh event list
        self.event_list_widget.set_events(self.current_events)
        self.event_list_widget.setCurrentRow(self.current_event_index)
        
        # Update summary
        self.load_events()
        
        # Auto-advance to next
        self.next_event()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = EEGEventsReviewGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
