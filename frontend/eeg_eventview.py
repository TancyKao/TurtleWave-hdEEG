#!/usr/bin/env python3

import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QGroupBox, QCheckBox, QListWidget, QComboBox, 
                            QSlider, QProgressBar, QTextEdit, QTabWidget,
                            QSplitter, QFrame, QSpinBox, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QAbstractItemView,
                            QScrollArea, QDoubleSpinBox)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

try:
    from turtlewave_hdEEG import LargeDataset, CustomAnnotations
    from scipy import signal
except ImportError as e:
    print(f"Import warning: {e}")

class EventDatabase:
    """Enhanced database handler for event review"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_review_tables()
    
    def create_review_tables(self):
        """Create additional tables for review functionality"""
        cursor = self.conn.cursor()
        
        # Add review columns to existing table if they don't exist
        try:
            cursor.execute('ALTER TABLE spindle_parameters ADD COLUMN reviewed INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute('ALTER TABLE spindle_parameters ADD COLUMN review_decision TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE spindle_parameters ADD COLUMN review_comments TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE spindle_parameters ADD COLUMN reviewer TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE spindle_parameters ADD COLUMN review_timestamp TEXT')
        except sqlite3.OperationalError:
            pass
        
        self.conn.commit()
    
    def get_events(self, event_type='spindle', channels=None, stages=None, reviewed_only=False, unreviewed_only=False):
        """Get events with comprehensive filtering"""
        query = "SELECT * FROM spindle_parameters WHERE 1=1"
        params = []
        
        if channels:
            placeholders = ','.join(['?' for _ in channels])
            query += f" AND channel IN ({placeholders})"
            params.extend(channels)
        
        if stages:
            stage_conditions = []
            for stage in stages:
                stage_conditions.append("stage LIKE ?")
                params.append(f"%{stage}%")
            query += f" AND ({' OR '.join(stage_conditions)})"
        
        if reviewed_only:
            query += " AND reviewed = 1"
        elif unreviewed_only:
            query += " AND (reviewed = 0 OR reviewed IS NULL)"
        
        query += " ORDER BY start_time"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_event_by_uuid(self, uuid):
        """Get single event by UUID"""
        return pd.read_sql_query("SELECT * FROM spindle_parameters WHERE uuid = ?", 
                               self.conn, params=[uuid])
    
    def add_review(self, uuid, decision, reviewer="", comments=""):
        """Add review decision for an event"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE spindle_parameters 
            SET reviewed = 1, review_decision = ?, review_comments = ?, 
                reviewer = ?, review_timestamp = ?
            WHERE uuid = ?
        ''', (decision, comments, reviewer, timestamp, uuid))
        self.conn.commit()
    
    def get_review_stats(self):
        """Get comprehensive review statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total events
        cursor.execute("SELECT COUNT(*) FROM spindle_parameters")
        stats['total'] = cursor.fetchone()[0]
        
        # Reviewed events
        cursor.execute("SELECT COUNT(*) FROM spindle_parameters WHERE reviewed = 1")
        stats['reviewed'] = cursor.fetchone()[0]
        
        # By decision
        cursor.execute("""
            SELECT review_decision, COUNT(*) 
            FROM spindle_parameters 
            WHERE reviewed = 1 
            GROUP BY review_decision
        """)
        for decision, count in cursor.fetchall():
            if decision:
                stats[f'{decision}_count'] = count
        
        # By channel
        cursor.execute("""
            SELECT channel, 
                   COUNT(*) as total,
                   SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) as reviewed_count,
                   SUM(CASE WHEN review_decision = 'accept' THEN 1 ELSE 0 END) as accepted,
                   SUM(CASE WHEN review_decision = 'reject' THEN 1 ELSE 0 END) as rejected
            FROM spindle_parameters 
            GROUP BY channel 
            ORDER BY channel
        """)
        stats['by_channel'] = [
            {
                'channel': row[0], 
                'total': row[1], 
                'reviewed': row[2], 
                'accepted': row[3], 
                'rejected': row[4]
            } 
            for row in cursor.fetchall()
        ]
        
        return stats
    
    def export_reviewed_events(self, output_path):
        """Export all reviewed events to CSV"""
        query = """
            SELECT * FROM spindle_parameters 
            WHERE reviewed = 1 
            ORDER BY channel, start_time
        """
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_path, index=False)
        return len(df)

class EEGVisualizationWidget(FigureCanvas):
    """Enhanced EEG visualization with multiple channel support"""
    
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create subplots
        self.raw_ax = self.fig.add_subplot(211)
        self.filtered_ax = self.fig.add_subplot(212)
        
        self.fig.tight_layout(pad=3.0)
        
        # State variables
        self.current_event = None
        self.eeg_data = None
        self.annotations = None
        self.visible_annotations = set()
        self.sampling_rate = 500
        
        # Color scheme for different event types
        self.event_colors = {
            'spindle': 'red',
            'slow_wave': 'blue',
            'arousal': 'orange',
            'artifact': 'gray',
            'movement': 'purple',
            'respiratory': 'green'
        }
    

    def set_annotation_visibility(self, annotation_types):
        """Set which annotation types should be visible"""
        self.visible_annotations = set(annotation_types)
        if self.current_event is not None:
            self.refresh_plot()
    

    def set_filter_settings(self, filter_settings):
        """Set filter parameters"""
        self.filter_settings = filter_settings


    def plot_event(self, event_data, eeg_data, annotations, channels, window_duration=10.0):
        """Plot EEG data around an event with annotations overlay"""
        self.current_event = event_data
        self.eeg_data = eeg_data
        self.annotations = annotations
        
        # Clear previous plots
        self.raw_ax.clear()
        self.filtered_ax.clear()
        
        # Get event details
        start_time = float(event_data['start_time'])
        end_time = float(event_data['end_time'])
        target_channel = event_data['channel']
        
        # Define window
        window_start = max(0, start_time - window_duration/2)
        window_end = end_time + window_duration/2
        
        # Get sampling rate
        try:
            self.sampling_rate = eeg_data.header['s_freq']
        except:
            self.sampling_rate = 500
        
        # Read data for the window
        try:
            data = eeg_data.read_data(chan=None, begtime=window_start, 
                                    endtime=window_end)
            
            # Time axis
            n_samples = data.data[0].shape[1] if hasattr(data, 'data') else 0
            if n_samples == 0:
                raise ValueError("No data samples were read")
            
            time_axis = np.linspace(window_start, window_end, n_samples)
            
            # Plot parameters
            y_spacing = 100  # µV
            y_offset = 0
            

            channel_labels = data.axis['chan'][0]
            # Plot each channel
            for i, ch in enumerate(channels):
                if ch in channel_labels:
                    #ch_idx = channel_labels.index(ch)
                    ch_idx = np.where(channel_labels == ch)[0][0]
                    raw_signal = data.data[0][ch_idx, :]
                    
                    # Highlight target channel
                    if ch == target_channel:
                        color = 'red'
                        linewidth = 2
                        alpha = 1.0
                    else:
                        color = 'black'
                        linewidth = 1
                        alpha = 0.7
                    
                    # Raw signal
                    self.raw_ax.plot(time_axis, raw_signal + y_offset, 
                                   color=color, linewidth=linewidth, alpha=alpha)
                    
                    # Filtered signal 
                    filtered_signal = self.apply_bandpass_filter(
                        raw_signal, 
                        getattr(self, 'filter_settings', {}).get('low_freq', 9), 
                        getattr(self, 'filter_settings', {}).get('high_freq', 15)
                    )

                    self.filtered_ax.plot(time_axis, filtered_signal + y_offset, 
                                        color=color, linewidth=linewidth, alpha=alpha)
                    
                    # Channel labels
                    self.raw_ax.text(window_start - 0.1, y_offset, ch, 
                                   verticalalignment='center', fontweight='bold' if ch == target_channel else 'normal')
                    self.filtered_ax.text(window_start - 0.1, y_offset, ch, 
                                        verticalalignment='center', fontweight='bold' if ch == target_channel else 'normal')
                    
                    y_offset += y_spacing
            
            # Highlight the main event
            event_height = len(channels) * y_spacing
            event_patch_raw = patches.Rectangle(
                (start_time, -y_spacing/2), 
                end_time - start_time, 
                event_height,
                facecolor='red', alpha=0.2, edgecolor='red', linewidth=2
            )
            event_patch_filt = patches.Rectangle(
                (start_time, -y_spacing/2), 
                end_time - start_time, 
                event_height,
                facecolor='red', alpha=0.2, edgecolor='red', linewidth=2
            )
            self.raw_ax.add_patch(event_patch_raw)
            self.filtered_ax.add_patch(event_patch_filt)
            
            # Add other annotations if available
            if annotations and self.visible_annotations:
                self.add_annotation_overlays(window_start, window_end, event_height)
            
            # Formatting
            self.raw_ax.set_title(f'Event: {event_data.get("uuid", "Unknown")} | '
                                f'Channel: {target_channel} | '
                                f'Time: {start_time:.1f}-{end_time:.1f}s | '
                                f'Duration: {end_time-start_time:.2f}s')
            self.filtered_ax.set_title('Filtered EEG (9-15 Hz)')
            self.filtered_ax.set_xlabel('Time (s)')
            
            # Set limits
            self.raw_ax.set_xlim(window_start, window_end)
            self.filtered_ax.set_xlim(window_start, window_end)
            self.raw_ax.set_ylim(-y_spacing/2, event_height - y_spacing/2)
            self.filtered_ax.set_ylim(-y_spacing/2, event_height - y_spacing/2)
            
            # Remove y-ticks
            self.raw_ax.set_yticks([])
            self.filtered_ax.set_yticks([])
            
            # Grid
            self.raw_ax.grid(True, alpha=0.3)
            self.filtered_ax.grid(True, alpha=0.3)
            
            self.draw()
            
        except Exception as e:
            print(f"Error plotting event: {e}")
            import traceback
            traceback.print_exc()
            # Show error message on plot
            self.raw_ax.text(0.5, 0.5, f"Error loading data: {str(e)}", 
                           transform=self.raw_ax.transAxes, 
                           ha='center', va='center', fontsize=12, color='red')
            self.draw()
    
    def add_annotation_overlays(self, window_start, window_end, plot_height):
        """Add annotation overlays for selected event types"""
        try:
            # Get all annotation events in the window
            for event_type in self.visible_annotations:
                color = self.event_colors.get(event_type.lower(), 'gray')
                
                # Get events of this type from annotations
                events = self.annotations.get_events(event_type)
                if events:
                    for event in events:
                        event_start = event['start']
                        event_end = event['end']
                        
                        # Check if event overlaps with current window
                        if event_end >= window_start and event_start <= window_end:
                            # Clip to window
                            clip_start = max(event_start, window_start)
                            clip_end = min(event_end, window_end)
                            
                            # Add overlay patches
                            overlay_raw = patches.Rectangle(
                                (clip_start, -10), 
                                clip_end - clip_start, 
                                plot_height + 20,
                                facecolor=color, alpha=0.1, 
                                edgecolor=color, linewidth=1, linestyle='--'
                            )
                            overlay_filt = patches.Rectangle(
                                (clip_start, -10), 
                                clip_end - clip_start, 
                                plot_height + 20,
                                facecolor=color, alpha=0.1, 
                                edgecolor=color, linewidth=1, linestyle='--'
                            )
                            
                            self.raw_ax.add_patch(overlay_raw)
                            self.filtered_ax.add_patch(overlay_filt)
                            
                            # Add label
                            self.raw_ax.text(clip_start, plot_height + 10, event_type, 
                                           fontsize=8, color=color, rotation=0)
        
        except Exception as e:
            print(f"Error adding annotation overlays: {e}")
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter to signal"""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            return data  # Return original if invalid range
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered
        except:
            return data  # Return original if filtering fails
    
    def refresh_plot(self):
        """Refresh the current plot with updated annotation visibility"""
        if self.current_event is not None and self.eeg_data is not None:
            # Re-plot with current settings
            channels = ['E1', 'E2', 'E3', 'E4', 'E5']  # Default channels
            self.plot_event(self.current_event, self.eeg_data, self.annotations, channels)

class HypnogramWidget(FigureCanvas):
    """Sleep hypnogram widget with current time marker"""
    
    def __init__(self, parent=None, width=12, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        
        self.current_time = 0
        
        # Event display properties
        self.event_colors = {
            'Artefact': 'gray',
            'Arousal': 'orange',
            'Resp': 'blue',
            'Move': 'green', 
            'Snore': 'purple'
        }



    def add_event_markers(self, annotations, event_types, times_hours,recording_start_seconds=0):
        """Add event markers to the hypnogram"""
        # Define y-positions for different event types
        event_positions = {
            'Artefact': 4.6,
            'Arousal': 4.8,
            'Resp': 5.0,
            'Move': 5.2,
            'Snore': 5.4
        }
        
        max_y_pos = 4.0  # Default max y value (REM stage)
        day_seconds = 24 * 60 * 60  # Seconds in a day

        # Plot each event type
        for event_type in event_types:
            # Get events of this type
            try:
                
                events = annotations.get_events(event_type)
                if not events:
                    continue
                
                color = self.event_colors.get(event_type, 'black')
                y_pos = event_positions.get(event_type, 5.6)
                
                max_y_pos = max(max_y_pos, y_pos + 0.2)  # Update max y value
                
                # Find max time for label positioning
                max_time = max(times_hours) if times_hours else 24
                
                # Add a label for this event type at the right side of the plot
                self.ax.text(max_time * 1.01, y_pos, event_type, 
                            fontsize=8, color=color, verticalalignment='center')
                            
                for event in events:
                    try:
                        # Get absolute time in seconds
                        event_start_seconds = event['start']
                        event_end_seconds = event['end']

                        # Handle wrapping around midnight
                        if event_start_seconds >= day_seconds:
                            event_start_seconds = event_start_seconds % day_seconds
                        if event_end_seconds >= day_seconds:
                            event_end_seconds = event_end_seconds % day_seconds
                        
                        # Convert to hours
                        event_start = event_start_seconds / 3600
                        event_end = event_end_seconds / 3600

                        
                        
                        # Check if event spans midnight
                        if event_end < event_start:
                            # Split into two segments: from start to midnight, and from midnight to end
                            # First segment
                            self.ax.plot([event_start, 24], [y_pos, y_pos], 
                                    linewidth=3, color=color, alpha=0.7)
                            # Second segment
                            self.ax.plot([0, event_end], [y_pos, y_pos], 
                                    linewidth=3, color=color, alpha=0.7)
                        else:
                            # Normal case (no midnight crossing)
                            # Create a horizontal line at the y-position
                            self.ax.plot([event_start, event_end], [y_pos, y_pos], 
                                    linewidth=3, color=color, alpha=0.7)
                        
                        # Add a vertical line at the start of the event
                        self.ax.axvline(event_start, ymin=0, ymax=0.95, 
                                    color=color, linestyle='-', alpha=0.2)
                    except Exception as e:
                        print(f"Error plotting event {event_type}: {e}")
                        continue
            except Exception as e:
                print(f"Error getting events for type {event_type}: {e}")
                continue
        
        # Adjust y-limits to accommodate event markers
        self.ax.set_ylim(-0.5, max_y_pos)
                    
 

    def plot_hypnogram(self, annotations, current_time=0, event_types=None):
        """Plot hypnogram from annotations with optional events overlay"""
        self.ax.clear()
        
        try:
            # Get sleep stages
            stages = annotations.get_stages()
            
            if not stages:
                self.ax.text(0.5, 0.5, 'No sleep stages available', 
                        transform=self.ax.transAxes, ha='center', va='center')
                self.draw()
                return
            
            # Convert stages to numerical values
            stage_map = {'Wake': 0, 'NREM1': 1, 'NREM2': 2, 'NREM3': 3, 'REM': 4}
            stage_colors = {
                'Wake': 'yellow', 'NREM1': 'lightblue', 'NREM2': 'blue', 
                'NREM3': 'darkblue', 'REM': 'red'
            }

            # Get recording start time
            recording_start_time = None
            recording_start_seconds = 0  # Default value
            if hasattr(annotations, 'wonb_annot') and hasattr(annotations.wonb_annot, 'start_time'):
                try:
                    # Get start time as a datetime object
                    recording_start_time = annotations.wonb_annot.start_time
                    if isinstance(recording_start_time, datetime):
                        # Get seconds since midnight
                        midnight = recording_start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                        recording_start_seconds = (recording_start_time - midnight).total_seconds()
                        print(f"Recording start time: {recording_start_time}, seconds since midnight: {recording_start_seconds}")
                except Exception as e:
                    print(f"Error processing start time: {e}")
                    recording_start_seconds = 0
            
            # Get stage sequence (assuming 30-second epochs)
            epoch_duration = 30  # seconds
            times = []
            stage_values = []
            day_seconds = 24 * 60 * 60  # Seconds in a day
            
            # Safety check for stages
            if not stages or len(stages) == 0:
                self.ax.text(0.5, 0.5, 'No sleep stages available', 
                        transform=self.ax.transAxes, ha='center', va='center')
                self.draw()
                return
                
            for i, stage in enumerate(stages):
                # Calculate time in seconds since midnight
                epoch_time_seconds = recording_start_seconds + (i * epoch_duration)
                
                # Handle time wrapping around midnight
                if epoch_time_seconds >= day_seconds:
                    epoch_time_seconds = epoch_time_seconds % day_seconds
                
                times.append(epoch_time_seconds)
                stage_values.append(stage_map.get(stage, 0))
            
            # Convert times to hours for display
            times_hours = [t / 3600 for t in times]
            
            # Convert current_time to hours since midnight
            if current_time is not None:
                # Calculate absolute time in seconds
                absolute_time_seconds = recording_start_seconds + current_time
                # Handle wrapping around midnight
                if absolute_time_seconds >= day_seconds:
                    absolute_time_seconds = absolute_time_seconds % day_seconds
                current_time_hours = absolute_time_seconds / 3600
            else:
                current_time_hours = 0
            
            # Plot as step function
            if times_hours and len(times_hours) > 0:
                # Safety check - make sure we have at least one time point
                self.ax.step(times_hours, stage_values, where='post', linewidth=2, color='black')
                
                # Add colored regions - only if we have at least 2 time points
                if len(times_hours) >= 2:
                    for i in range(len(times_hours) - 1):  # Ensure we don't go out of bounds
                        if i < len(stages):  # Make sure we have a corresponding stage
                            stage = stages[i]
                            color = stage_colors.get(stage, 'gray')
                            
                            # Handle midnight transition
                            if times_hours[i+1] < times_hours[i]:  # Time wrapped around midnight
                                # Add span until midnight (24 hours)
                                self.ax.axvspan(times_hours[i], 24, alpha=0.3, color=color)
                                # Add span from midnight to next point
                                self.ax.axvspan(0, times_hours[i+1], alpha=0.3, color=color)
                            else:
                                self.ax.axvspan(times_hours[i], times_hours[i+1], alpha=0.3, color=color)
                
                # Add events if event types are provided
                if event_types:
                    self.add_event_markers(annotations, event_types, times_hours, recording_start_seconds)

                # Current time marker
                self.ax.axvline(current_time_hours, color='red', linestyle='--', linewidth=2, label='Current Event')
                
                # Formatting
                self.ax.set_ylabel('Sleep Stage')
                self.ax.set_xlabel('Time (hours)')
                self.ax.set_yticks(list(stage_map.values()))
                self.ax.set_yticklabels(list(stage_map.keys()))
                self.ax.grid(True, alpha=0.3)
                
                # Set x-axis limits
                # Check if times wrap around midnight
                has_midnight_wrap = False
                for i in range(len(times_hours)-1):
                    if times_hours[i+1] < times_hours[i]:
                        has_midnight_wrap = True
                        break
                        
                if has_midnight_wrap:
                    self.ax.set_xlim(0, 24)  # Show full 24 hour range
                else:
                    # Set limits to show just the time range with some padding
                    min_time = max(0, min(times_hours) - 0.5) if times_hours else 0
                    max_time = min(24, max(times_hours) + 0.5) if times_hours else 24
                    self.ax.set_xlim(min_time, max_time)
                
                # Format x-axis with hours
                from matplotlib.ticker import FuncFormatter
                
                def format_hours(x, pos):
                    hours = int(x)
                    minutes = int((x - hours) * 60)
                    return f"{hours:02d}:{minutes:02d}"
                
                self.ax.xaxis.set_major_formatter(FuncFormatter(format_hours))
                
                # Legend
                self.ax.legend(loc='upper right')
            else:
                # Handle case with no time points
                self.ax.text(0.5, 0.5, 'No sleep stage time points available', 
                        transform=self.ax.transAxes, ha='center', va='center')
        
        except Exception as e:
            print(f"Error plotting hypnogram: {e}")
            import traceback
            traceback.print_exc()

            self.ax.text(0.5, 0.5, f'Error: {str(e)}', 
                    transform=self.ax.transAxes, ha='center', va='center')
        
        self.draw()


class EventReviewInterface(QMainWindow):
    """Main event review interface"""
    
    def __init__(self, eeg_data=None, annotations=None, eeg_file_path="", annot_file_path=""):
        super().__init__()
        self.setWindowTitle("TurtleWave Event Review")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Store pre-loaded data
        self.eeg_data = eeg_data
        self.annotations = annotations
        self.eeg_file_path = eeg_file_path
        self.annot_file_path = annot_file_path
        
        # Initialize other components
        self.db = None
        self.current_events = pd.DataFrame()
        self.current_event_index = 0
        self.reviewer_name = "Reviewer1"
        
        # UI state
        self.selected_channels = []
        self.annotation_visibility = set()
        self.time_window = 10.0  # Default 10-second window
        self.filter_settings = {
            'low_freq': 0.5,
            'high_freq': 4.0,  # Default to delta band
            'order': 4
        }
        
        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
        
        # If data was pre-loaded, initialize interface
        if self.eeg_data and self.annotations:
            self.initialize_with_data()

        # Default selected channels
        self.selected_channels = ['E21', 'E36', 'E224', 'Cz', 'E59', 'E183','E87','E153','E101',
                            'E116','E126','E150']
        self.channel_checkboxes = {}

        # Add debug flag for development
        self.debug = True
        
        if self.debug:
            print("Debug mode enabled")
            print(f"Python version: {sys.version}")
            try:
                import wonambi
                print(f"Wonambi version: {wonambi.__version__}")
            except:
                print("Wonambi version: unknown")


    
    def initialize_with_data(self):
        """Initialize interface with pre-loaded data"""
        try:
            # Pre-fill paths in UI
            self.eeg_file_edit.setText(self.eeg_file_path)
            self.eeg_file_edit.setReadOnly(True)
            self.browse_eeg_btn.setEnabled(False)
            
            self.annot_file_edit.setText(self.annot_file_path)
            self.annot_file_edit.setReadOnly(True)
            self.browse_annot_btn.setEnabled(False)
            
            # Populate available channels from dataset
            if self.eeg_data:
                self.populate_available_channels()
            
            # Populate annotation types from annotations
            if self.annotations:
                self.populate_annotation_types()
                self.populate_sleep_stages()
                
                # Plot hypnogram
                self.hypnogram_widget.plot_hypnogram(self.annotations)
            
            # Enable UI elements
            self.update_ui_state(True)
            
            self.status_bar.showMessage("Data pre-loaded successfully")
            
        except Exception as e:
            print(f"Error initializing with pre-loaded data: {e}")
            self.status_bar.showMessage("Error initializing with data")

    def setup_ui(self):
        """Setup main UI components"""
        main_layout = QHBoxLayout()
        
        # Create left sidebar
        left_sidebar = self.create_left_sidebar()
        
        # Create right area with visualizations
        right_area = self.create_right_area()
        
        # Create splitter for resizable panels
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_sidebar)
        splitter.addWidget(right_area)
        splitter.setSizes([300, 1300])  # Initial sizes
        
        main_layout.addWidget(splitter)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        


    def create_channel_selection_widget(self):
        """Create an improved channel selection widget for high-density EEG"""
        channel_widget = QWidget()
        layout = QVBoxLayout(channel_widget)
        
        # Add search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.channel_search = QtWidgets.QLineEdit()
        self.channel_search.setPlaceholderText("Type to filter channels...")
        self.channel_search.textChanged.connect(self.filter_channels)
        search_layout.addWidget(self.channel_search)
        layout.addLayout(search_layout)
        
        # Quick select buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self.select_all_channels)
        btn_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("None")
        select_none_btn.clicked.connect(self.deselect_all_channels)
        btn_layout.addWidget(select_none_btn)
        
        # Range selection
        range_btn = QPushButton("Range...")
        range_btn.clicked.connect(self.select_channel_range)
        btn_layout.addWidget(range_btn)
        
        # Add to layout
        layout.addLayout(btn_layout)
        
        # Create tree widget for channels
        self.channel_tree = QtWidgets.QTreeWidget()
        self.channel_tree.setHeaderLabels(["Channels"])
        self.channel_tree.setAlternatingRowColors(True)
        self.channel_tree.itemChanged.connect(self.on_channel_tree_changed)
        layout.addWidget(self.channel_tree)
        
        return channel_widget

    def populate_channel_tree(self):
        """Populate the channel tree with grouped channels"""
        # Clear existing items
        self.channel_tree.clear()
        
        # Get channels from EEG data
        if hasattr(self.eeg_data, 'channels'):
            channels = self.eeg_data.channels
        elif hasattr(self.eeg_data, 'list_of_labels'):
            channels = self.eeg_data.list_of_labels
        else:
            try:
                channels = self.eeg_data.header.get('chan_name', [])
            except:
                channels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'Cz']
        
        # Convert to strings
        channels = [str(ch) for ch in channels]
        
        # Group channels
        e_channels = [ch for ch in channels if ch.startswith('E') and ch[1:].isdigit()]
        reference_channels = [ch for ch in channels if ch in ['Cz', 'Pz', 'Fz', 'Oz', 'Fpz', 'M1', 'M2']]
        #physio_channels = [ch for ch in channels if not ch.startswith('E') and ch not in reference_channels]
        physio_channels = [ch for ch in channels if ch in ['ECG', 'EMG', 'EOG', 'Respiration', 'Temperature', 'GSR']]
        # Create channel groups dictionary
        self.channel_checkboxes = {}
        
        # Create E-channel groups (in batches of 32)
        if e_channels:
            # Sort E-channels by number
            e_channels.sort(key=lambda x: int(x[1:]))
            
            # Create groups of 32 channels
            for i in range(0, len(e_channels), 32):
                end_idx = min(i + 31, len(e_channels) - 1)
                group_name = f"E{i+1}-E{int(e_channels[end_idx][1:])}"
                
                group_item = QtWidgets.QTreeWidgetItem(self.channel_tree, [group_name])
                group_item.setFlags(group_item.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)
                
                # Add channels to this group
                for ch in e_channels[i:i+32]:
                    ch_item = QtWidgets.QTreeWidgetItem(group_item, [ch])
                    ch_item.setFlags(ch_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    ch_item.setCheckState(0, QtCore.Qt.Unchecked)
                    self.channel_checkboxes[ch] = ch_item
        
        # Reference channels group
        if reference_channels:
            ref_group = QtWidgets.QTreeWidgetItem(self.channel_tree, ["Reference"])
            ref_group.setFlags(ref_group.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)
            
            for ch in sorted(reference_channels):
                ch_item = QtWidgets.QTreeWidgetItem(ref_group, [ch])
                ch_item.setFlags(ch_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                ch_item.setCheckState(0, QtCore.Qt.Unchecked)
                self.channel_checkboxes[ch] = ch_item
        
        # Physiological channels group
        if physio_channels:
            physio_group = QtWidgets.QTreeWidgetItem(self.channel_tree, ["Physiological"])
            physio_group.setFlags(physio_group.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)
            
            for ch in sorted(physio_channels):
                ch_item = QtWidgets.QTreeWidgetItem(physio_group, [ch])
                ch_item.setFlags(ch_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                ch_item.setCheckState(0, QtCore.Qt.Unchecked)
                self.channel_checkboxes[ch] = ch_item
        
        # Expand first group by default
        if self.channel_tree.topLevelItemCount() > 0:
            self.channel_tree.topLevelItem(0).setExpanded(True)
        
        # Select default channels (first few E-channels + Cz if available)
        self.select_default_channels()

    def select_all_channels(self):
        """Select all channels"""
        for ch_item in self.channel_checkboxes.values():
            ch_item.setCheckState(0, QtCore.Qt.Checked)
        self.update_selected_channels_from_tree()

    def deselect_all_channels(self):
        """Deselect all channels"""
        for ch_item in self.channel_checkboxes.values():
            ch_item.setCheckState(0, QtCore.Qt.Unchecked)
        self.update_selected_channels_from_tree()

    def update_selected_channels_from_tree(self):
        """Update selected channels list from tree selection"""
        selected = []
        
        # Get all checked channels
        for ch, item in self.channel_checkboxes.items():
            if item.checkState(0) == QtCore.Qt.Checked:
                selected.append(ch)
        
        self.selected_channels = selected
        
        # Refresh current event display if available
        if hasattr(self, 'current_events') and len(self.current_events) > 0 and self.current_event_index < len(self.current_events):
            self.update_event_display()



    def select_default_channels(self):
        """Select a reasonable default set of channels"""
        # Get all channels
        all_channels = list(self.channel_checkboxes.keys())
        
        # Find key channels to select by default
        default_channels = []
        
        # Add Cz if available
        if 'Cz' in all_channels:
            default_channels.append('Cz')
        
        # Add first 6 E-channels
        e_channels = [ch for ch in all_channels if ch.startswith('E') and ch[1:].isdigit()]
        e_channels.sort(key=lambda x: int(x[1:]))
        default_channels.extend(e_channels[:6])
        
        # Select the channels
        for ch, item in self.channel_checkboxes.items():
            if ch in default_channels:
                item.setCheckState(0, QtCore.Qt.Checked)
            else:
                item.setCheckState(0, QtCore.Qt.Unchecked)
        
        self.update_selected_channels_from_tree()

    def filter_channels(self, text):
        """Filter the channel tree by search text"""
        # Show all items if search is empty
        if not text:
            for i in range(self.channel_tree.topLevelItemCount()):
                group_item = self.channel_tree.topLevelItem(i)
                group_item.setHidden(False)
                for j in range(group_item.childCount()):
                    group_item.child(j).setHidden(False)
            return
        
        # Hide/show based on search text
        text = text.lower()
        for i in range(self.channel_tree.topLevelItemCount()):
            group_item = self.channel_tree.topLevelItem(i)
            all_hidden = True
            
            # Check each channel in this group
            for j in range(group_item.childCount()):
                ch_item = group_item.child(j)
                if text in ch_item.text(0).lower():
                    ch_item.setHidden(False)
                    all_hidden = False
                else:
                    ch_item.setHidden(True)
            
            # Hide group if all children are hidden
            group_item.setHidden(all_hidden)

    def select_channel_range(self):
        """Open dialog to select a range of E-channels"""
        # Get list of E-channels
        e_channels = [ch for ch in self.channel_checkboxes.keys() 
                    if ch.startswith('E') and ch[1:].isdigit()]
        
        if not e_channels:
            return
        
        # Find min and max channel numbers
        min_num = min(int(ch[1:]) for ch in e_channels)
        max_num = max(int(ch[1:]) for ch in e_channels)
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Channel Range")
        layout = QVBoxLayout(dialog)
        
        # Range selectors
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("From:"))
        start_spin = QSpinBox()
        start_spin.setRange(min_num, max_num)
        start_spin.setValue(min_num)
        range_layout.addWidget(start_spin)
        
        range_layout.addWidget(QLabel("To:"))
        end_spin = QSpinBox()
        end_spin.setRange(min_num, max_num)
        end_spin.setValue(min(min_num + 7, max_num))
        range_layout.addWidget(end_spin)
        
        layout.addLayout(range_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            start = start_spin.value()
            end = end_spin.value()
            
            # Uncheck all first
            self.deselect_all_channels()
            
            # Check channels in range
            for i in range(start, end + 1):
                channel = f"E{i}"
                if channel in self.channel_checkboxes:
                    self.channel_checkboxes[channel].setCheckState(0, QtCore.Qt.Checked)
            
            self.update_selected_channels_from_tree()

    def on_channel_tree_changed(self, item, column):
        """Handle channel tree item check state changes"""
        # Only process actual checkbox changes
        if column != 0 or item.childCount() > 0:
            return
            
        # Update selected channels list
        self.update_selected_channels_from_tree()

    def update_selected_channels_from_tree(self):
        """Update selected channels list from tree selection"""
        selected = []
        
        # Get all checked channels
        for ch, item in self.channel_checkboxes.items():
            if item.checkState(0) == QtCore.Qt.Checked:
                selected.append(ch)
        
        self.selected_channels = selected
        
        # Refresh current event display if available
        if hasattr(self, 'current_events') and len(self.current_events) > 0 and self.current_event_index < len(self.current_events):
            self.update_event_display()



    def print_debug_info(self):
        """Print debug information about loaded data"""
        if not self.debug:
            return
            
        print("\n=== DEBUG INFO ===")
        if hasattr(self, 'eeg_data'):
            print(f"EEG Data type: {type(self.eeg_data)}")
            if hasattr(self.eeg_data, 'channels'):
                print(f"Channels: {len(self.eeg_data.channels)} available")
                print(f"First 5 channels: {self.eeg_data.channels[:5]}")
            print(f"Sampling rate: {getattr(self.eeg_data, 'sampling_rate', 'unknown')}")
        
        if hasattr(self, 'annotations'):
            print(f"Annotations type: {type(self.annotations)}")
            if hasattr(self.annotations, 'raters'):
                print(f"Available raters: {self.annotations.raters}")
            if hasattr(self.annotations, 'epochs') and self.annotations.epochs:
                print(f"Number of epochs: {len(self.annotations.epochs)}")
                print(f"First epoch: {self.annotations.epochs[0]}")
        
        if hasattr(self, 'current_events') and len(self.current_events) > 0:
            print(f"Number of events: {len(self.current_events)}")
            if len(self.current_events) > 0:
                print(f"First event columns: {list(self.current_events.columns)}")
                print(f"First event data: {self.current_events.iloc[0].to_dict()}")
        
        print("=== END DEBUG ===\n")







    # def populate_default_channels(self):
    #     """Populate default channels at startup"""
    #     if not hasattr(self, 'channel_checkboxes') or not self.channel_checkboxes:
    #         self.channel_checkboxes = {}
    #         default_channels = ['E21', 'E36', 'E224', 'Cz', 'E59', 'E183','E87','E153','E101',
    #                         'E116','E126','E150']
            
    #         # Add a header
    #         self.channel_layout.addWidget(QLabel("EEG:"))
            
    #         # Add default channel checkboxes
    #         for ch in default_channels:
    #             checkbox = QCheckBox(ch)
    #             checkbox.setChecked(True)
    #             checkbox.stateChanged.connect(self.update_selected_channels)
    #             self.channel_checkboxes[ch] = checkbox
    #             self.channel_layout.addWidget(checkbox)
            
    #         self.selected_channels = default_channels

    def populate_available_channels(self):
        """Populate channel selection from loaded EEG data"""
        try:
            # Clear existing channel checkboxes
            if hasattr(self, 'channel_checkboxes'):
                for checkbox in self.channel_checkboxes.values():
                    if checkbox is not None:
                        checkbox.deleteLater()
            
            self.channel_checkboxes = {}
            
            # Get available channels from dataset
            if hasattr(self.eeg_data, 'channels'):
                channels = self.eeg_data.channels
            else:
                # Fallback
                channels = ['E21', 'E36', 'E224', 'Cz', 'E59', 'E183','E87','E153','E101',
                            'E116','E126','E150'] # Fz, F3, F4, Cz, C3, C4, P3, 'Pz', P4, 01,Oz,O2
            
            # Add a label first (in case it was removed)
            if self.channel_layout.count() == 0:
                self.channel_layout.addWidget(QLabel("Display Channels:"))
            
 

            # Group channels by type
            eeg_channels = [ch for ch in channels if not (ch.startswith('EOG') or ch.startswith('EMG'))]
            eog_channels = [ch for ch in channels if ch.startswith('EOG')]
            emg_channels = [ch for ch in channels if ch.startswith('EMG')]
            
            # Add EEG channels
            if eeg_channels:
                label = QLabel("EEG:")
                self.channel_layout.addWidget(label)
                for ch in eeg_channels:
                    checkbox = QCheckBox(ch)
                    checkbox.setChecked(ch in ['E21', 'E36', 'E224', 'Cz', 'E59', 'E183','E87','E153','E101',
                            'E116','E126','E150'])
                    checkbox.stateChanged.connect(self.update_selected_channels)
                    self.channel_checkboxes[ch] = checkbox
                    self.channel_layout.addWidget(checkbox)
            
            # Add EOG channels
            if eog_channels:
                self.channel_layout.addWidget(QLabel("EOG:"))
                for ch in eog_channels:
                    checkbox = QCheckBox(ch)
                    checkbox.setChecked(False)
                    checkbox.stateChanged.connect(self.update_selected_channels)
                    self.channel_checkboxes[ch] = checkbox
                    self.channel_layout.addWidget(checkbox)
            
            # Add EMG channels
            if emg_channels:
                self.channel_layout.addWidget(QLabel("EMG:"))
                for ch in emg_channels:
                    checkbox = QCheckBox(ch)
                    checkbox.setChecked(False)
                    checkbox.stateChanged.connect(self.update_selected_channels)
                    self.channel_checkboxes[ch] = checkbox
                    self.channel_layout.addWidget(checkbox)
            
            # Update selected channels list
            self.update_selected_channels()
            
        except Exception as e:
            print(f"Error populating channels: {e}")


    def populate_annotation_types(self):
        """Populate annotation types from loaded annotation file"""
        try:
            # Clear existing checkboxes
            for widget in self.find_children_by_type(self.event_types_layout, QCheckBox):
                if widget != self.all_events_check:  # Keep the "All" checkbox
                    widget.deleteLater()
            
            self.annotation_checkboxes = {}
            
            # Get available annotation types from annotations
            if self.annotations:
                 annotation_types = self.annotations.wonb_annot.event_types
            else:
                # Fallback
                annotation_types = ['Arousal', 'Artifact', 'Movement', 'Respiratory']
            
            # Add checkboxes for each type
            for ann_type in annotation_types:
                checkbox = QCheckBox(ann_type)
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self.update_annotation_visibility)
                self.annotation_checkboxes[ann_type] = checkbox
                self.event_types_layout.addWidget(checkbox)
            
            # Update visibility
            self.update_annotation_visibility()
            
        except Exception as e:
            print(f"Error populating annotation types: {e}")
    
    def populate_sleep_stages(self):
        """Populate sleep stage filters from annotations"""
        try:
            # Create sleep stage checkboxes if they don't exist
            if not hasattr(self, 'stage_checkboxes'):
                self.stage_checkboxes = {}
                self.stage_layout = QVBoxLayout()
                
                # Add to filter group
                self.filter_layout.addLayout(self.stage_layout)
            
            # Clear existing stage checkboxes
            for i in reversed(range(self.stage_layout.count())): 
                widget = self.stage_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # Add header
            self.stage_layout.addWidget(QLabel("Sleep Stages:"))
            
            # Standard sleep stages
            stages = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']
            
            # Add checkboxes
            for stage in stages:
                checkbox = QCheckBox(stage)
                checkbox.setChecked(stage in ['NREM2', 'NREM3'])  # Default to N2/N3 
                self.stage_checkboxes[stage] = checkbox
                self.stage_layout.addWidget(checkbox)
            
        except Exception as e:
            print(f"Error populating sleep stages: {e}")
    
    def find_children_by_type(self, layout, widget_type):
        """Helper to find all children of a specific type in a layout"""
        widgets = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), widget_type):
                widgets.append(item.widget())
            elif item.layout():
                widgets.extend(self.find_children_by_type(item.layout(), widget_type))
        return widgets
    

    def create_left_sidebar(self):
        """Create left sidebar with controls"""
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ccc;")
        
        layout = QVBoxLayout(sidebar)
        
        # File loading section
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout()
        
        # Database file
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("Events Database:"))
        self.db_file_edit = QtWidgets.QLineEdit()
        db_layout.addWidget(self.db_file_edit)
        self.browse_db_btn = QPushButton("Browse")
        self.browse_db_btn.clicked.connect(self.browse_db_file)
        db_layout.addWidget(self.browse_db_btn)
        file_layout.addLayout(db_layout)
        
        # EEG file
        eeg_layout = QHBoxLayout()
        eeg_layout.addWidget(QLabel("EEG Data:"))
        self.eeg_file_edit = QtWidgets.QLineEdit()
        eeg_layout.addWidget(self.eeg_file_edit)
        self.browse_eeg_btn = QPushButton("Browse")
        self.browse_eeg_btn.clicked.connect(self.browse_eeg_file)
        eeg_layout.addWidget(self.browse_eeg_btn)
        file_layout.addLayout(eeg_layout)
        
        # Annotation file
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotations:"))
        self.annot_file_edit = QtWidgets.QLineEdit()
        annot_layout.addWidget(self.annot_file_edit)
        self.browse_annot_btn = QPushButton("Browse")
        self.browse_annot_btn.clicked.connect(self.browse_annot_file)
        annot_layout.addWidget(self.browse_annot_btn)
        file_layout.addLayout(annot_layout)
        
        # Load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        self.load_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        file_layout.addWidget(self.load_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Event Types (Wonambi-style)
        event_types_group = QGroupBox("Event Types")
        self.event_types_layout = QVBoxLayout()
        
        # All events checkbox
        self.all_events_check = QCheckBox("All event types")
        self.all_events_check.setChecked(True)
        self.all_events_check.stateChanged.connect(self.toggle_all_annotations)
        self.event_types_layout.addWidget(self.all_events_check)
        
        event_types_group.setLayout(self.event_types_layout)
        layout.addWidget(event_types_group)
        
        # Filter controls - use tabs for better space efficiency
        filter_group = QGroupBox("Review Filters")
        filter_layout = QVBoxLayout()
        
        # Create tabbed widget
        filter_tabs = QTabWidget()
        filter_tabs.setTabPosition(QTabWidget.North)
        
        # Channel tab - use our new tree widget
        channel_tab = self.create_channel_selection_widget()
        filter_tabs.addTab(channel_tab, "Channels")
        
        # Review status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        
        # Review status filter
        status_layout.addWidget(QLabel("Review Status:"))
        
        self.show_all_radio = QtWidgets.QRadioButton("Show All")
        self.show_unreviewed_radio = QtWidgets.QRadioButton("Unreviewed Only")
        self.show_reviewed_radio = QtWidgets.QRadioButton("Reviewed Only")
        
        self.show_unreviewed_radio.setChecked(True)  # Default
        
        status_layout.addWidget(self.show_all_radio)
        status_layout.addWidget(self.show_unreviewed_radio)
        status_layout.addWidget(self.show_reviewed_radio)
        status_layout.addStretch()
        
        filter_tabs.addTab(status_tab, "Status")
        
        # Sleep stages tab
        stage_tab = QWidget()
        stage_layout = QVBoxLayout(stage_tab)
        stage_layout.addWidget(QLabel("Sleep Stages:"))
        
        # Add stage checkboxes
        self.stage_checkboxes = {}
        for stage in ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']:
            checkbox = QCheckBox(stage)
            checkbox.setChecked(stage in ['NREM2', 'NREM3'])
            self.stage_checkboxes[stage] = checkbox
            stage_layout.addWidget(checkbox)
        
        stage_layout.addStretch()
        filter_tabs.addTab(stage_tab, "Stages")
        
        # Add tabs to layout
        filter_layout.addWidget(filter_tabs)
        
        # Apply filters button
        self.apply_filters_btn = QPushButton("Apply Filters")
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        filter_layout.addWidget(self.apply_filters_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        
        # Progress display
        self.progress_label = QLabel("0/0")
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        nav_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        nav_layout.addWidget(self.progress_bar)
        
        self.progress_percent_label = QLabel("0% Complete")
        self.progress_percent_label.setAlignment(QtCore.Qt.AlignCenter)
        nav_layout.addWidget(self.progress_percent_label)
        
        # Navigation buttons
        nav_buttons_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_event)
        nav_buttons_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_event)
        nav_buttons_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(nav_buttons_layout)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Review actions
        review_group = QGroupBox("Review Actions")
        review_layout = QVBoxLayout()
        
        # Review buttons
        button_layout = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.accept_btn.clicked.connect(lambda: self.review_current_event('accept'))
        button_layout.addWidget(self.accept_btn)
        
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.reject_btn.clicked.connect(lambda: self.review_current_event('reject'))
        button_layout.addWidget(self.reject_btn)
        
        review_layout.addLayout(button_layout)
        
        # Comments
        review_layout.addWidget(QLabel("Comments:"))
        self.comments_edit = QTextEdit()
        self.comments_edit.setMaximumHeight(60)
        self.comments_edit.setPlaceholderText("Optional comments...")
        review_layout.addWidget(self.comments_edit)
        
        review_group.setLayout(review_layout)
        layout.addWidget(review_group)
        
        # Statistics display
        stats_group = QGroupBox("Review Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Load data to see statistics")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        # Export button
        self.export_btn = QPushButton("Export Review Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        stats_layout.addWidget(self.export_btn)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return sidebar    

    
    def create_right_area(self):
        """Create right area with visualizations"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Top: Current event info and confidence threshold
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #e8f4fd; border: 1px solid #bee5eb; padding: 5px;")
        info_layout = QHBoxLayout(info_frame)
        
        self.current_event_label = QLabel("No event selected")
        self.current_event_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.current_event_label.setWordWrap(True)
        info_layout.addWidget(self.current_event_label, 1)  # Give it stretch
        
        # Add confidence threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(QtCore.Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        self.confidence_slider.setMaximumWidth(150)
        threshold_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        threshold_layout.addWidget(self.confidence_label)
        
        # Add spindle count display
        self.threshold_count_label = QLabel("Displaying 89 spindles above threshold")
        threshold_layout.addWidget(self.threshold_count_label)
        
        info_layout.addLayout(threshold_layout)
        
        # Reviewer name
        info_layout.addWidget(QLabel("Reviewer:"))
        self.reviewer_edit = QtWidgets.QLineEdit(self.reviewer_name)
        self.reviewer_edit.setMaximumWidth(100)
        self.reviewer_edit.textChanged.connect(lambda text: setattr(self, 'reviewer_name', text))
        info_layout.addWidget(self.reviewer_edit)
        
        layout.addWidget(info_frame)
        
        # Middle: Hypnogram
        self.hypnogram_widget = HypnogramWidget()
        layout.addWidget(self.hypnogram_widget)
        
        # Bottom: EEG visualization tabs
        viz_tabs = QTabWidget()
        
        # EEG Views tab with filter controls
        eeg_tab = self.setup_eeg_tab()
        viz_tabs.addTab(eeg_tab, "EEG Views")
        
        # Event details tab
        details_widget = self.create_event_details_widget()
        viz_tabs.addTab(details_widget, "Event Details")
        
        # Placeholder tabs for future implementation
        time_freq_tab = QWidget()
        time_freq_tab.setLayout(QVBoxLayout())
        time_freq_tab.layout().addWidget(QLabel("Time-Frequency analysis will be available in a future update"))
        viz_tabs.addTab(time_freq_tab, "Time-Frequency")
        
        topo_tab = QWidget()
        topo_tab.setLayout(QVBoxLayout())
        topo_tab.layout().addWidget(QLabel("Topography visualization will be available in a future update"))
        viz_tabs.addTab(topo_tab, "Topography")
        
        layout.addWidget(viz_tabs)
        
        # Set proportions
        layout.setStretch(0, 0)  # Info bar
        layout.setStretch(1, 0)  # Hypnogram
        layout.setStretch(2, 1)  # Main visualization
        
        return right_widget
    
    def setup_eeg_tab(self):
        """Create EEG tab with filter controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Filter controls
        filter_group = QGroupBox("Filter Settings")
        filter_layout = QHBoxLayout()
        
        # Low cutoff frequency
        filter_layout.addWidget(QLabel("Low Cutoff (Hz):"))
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 30)
        self.low_freq_spin.setValue(0.5)
        self.low_freq_spin.setSingleStep(0.1)
        filter_layout.addWidget(self.low_freq_spin)
        
        # High cutoff frequency
        filter_layout.addWidget(QLabel("High Cutoff (Hz):"))
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(1, 50)
        self.high_freq_spin.setValue(4.0)
        self.high_freq_spin.setSingleStep(0.5)
        filter_layout.addWidget(self.high_freq_spin)
        
        # Preset buttons for common frequency bands
        filter_layout.addWidget(QLabel("Presets:"))
        
        # Add preset buttons for common bands
        preset_layout = QHBoxLayout()
        presets = [
            ("Delta", 0.5, 4.0),
            ("Theta", 4.0, 8.0),
            ("Alpha", 8.0, 12.0),
            ("Sigma", 12.0, 16.0),
            ("Beta", 16.0, 30.0)
        ]
        
        for name, low, high in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, l=low, h=high: self.set_filter_preset(l, h))
            preset_layout.addWidget(btn)
        
        filter_layout.addLayout(preset_layout)
        
        # Apply filter button
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self.update_filter_settings)
        filter_layout.addWidget(apply_filter_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Time window control
        window_group = QGroupBox("Time Window")
        window_layout = QHBoxLayout()
        
        window_layout.addWidget(QLabel("Window Size (s):"))
        self.time_window_spin = QDoubleSpinBox()
        self.time_window_spin.setRange(2, 60)
        self.time_window_spin.setValue(10.0)
        self.time_window_spin.setSingleStep(1.0)
        self.time_window_spin.valueChanged.connect(self.update_time_window)
        window_layout.addWidget(self.time_window_spin)
        
        window_layout.addStretch()
        
        # Add preset buttons for window sizes
        window_presets = [
            ("5s", 5),
            ("10s", 10),
            ("15s", 15),
            ("30s", 30),
            ("60s", 60)
        ]
        
        for name, size in window_presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, s=size: self.time_window_spin.setValue(s))
            window_layout.addWidget(btn)
        
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # EEG visualization
        self.eeg_widget = EEGVisualizationWidget()
        layout.addWidget(self.eeg_widget)
        
        return tab
        
    def update_filter_settings(self):
        """Update filter settings and refresh display"""
        self.filter_settings = {
            'low_freq': self.low_freq_spin.value(),
            'high_freq': self.high_freq_spin.value(),
            'order': 4  # Fixed order for simplicity
        }
        
        self.eeg_widget.set_filter_settings(self.filter_settings)

        # Update EEG widget
        if hasattr(self.eeg_widget, 'filter_settings'):
            self.eeg_widget.filter_settings = self.filter_settings
        
        

        # Refresh current event display
        self.update_event_display()
    
    def set_filter_preset(self, low, high):
        """Set filter to a preset frequency band"""
        self.low_freq_spin.setValue(low)
        self.high_freq_spin.setValue(high)
        self.update_filter_settings()
    
    def update_time_window(self):
        """Update time window size and refresh display"""
        self.time_window = self.time_window_spin.value()
        
        # Update EEG widget
        if hasattr(self.eeg_widget, 'time_window'):
            self.eeg_widget.time_window = self.time_window
        
        # Refresh current event display
        self.update_event_display()
    
    def update_confidence_threshold(self):
        """Update confidence threshold value"""
        value = self.confidence_slider.value() / 100.0
        self.confidence_threshold = value
        self.confidence_label.setText(f"{value:.2f}")
        
        # Update count display if events are loaded
        if len(self.current_events) > 0:
            # Count events above threshold (using power as proxy for confidence)
            above_threshold = len(self.current_events[
                self.current_events['power'] >= self.current_events['power'].max() * value
            ])
            
            self.threshold_count_label.setText(f"Displaying {above_threshold} spindles above threshold")
        
        # Apply filters with new threshold
        self.apply_filters()


    
    def create_event_details_widget(self):
        """Create widget showing detailed event parameters"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Event parameters table
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        self.details_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.details_table)
        
        return widget
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Load data to begin review")
        
        # Add progress indicator
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(200)
        self.status_progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.status_progress)
    
    def browse_db_file(self):
        """Browse for database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Events Database", "", 
            "Database Files (*.db *.sqlite);;All Files (*)"
        )
        if file_path:
            self.db_file_edit.setText(file_path)
    
    def browse_eeg_file(self):
        """Browse for EEG file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EEG File", "", 
            "EEG Files (*.set *.edf *.bdf);;All Files (*)"
        )
        if file_path:
            self.eeg_file_edit.setText(file_path)
    
    def browse_annot_file(self):
        """Browse for annotation file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", "", 
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.annot_file_edit.setText(file_path)
    
    def load_data(self):
        """Load all required data files"""
        try:
            self.status_bar.showMessage("Loading data...")
            self.status_progress.setVisible(True)
            self.status_progress.setRange(0, 0)  # Indeterminate
            
            # Load database
            db_path = self.db_file_edit.text()
            if not db_path or not os.path.exists(db_path):
                raise Exception("Please select a valid database file")
            
            self.db = EventDatabase(db_path)
            
            # Load EEG data
            eeg_path = self.eeg_file_edit.text()
            if not eeg_path or not os.path.exists(eeg_path):
                raise Exception("Please select a valid EEG file")
            
            self.eeg_data = LargeDataset(eeg_path, create_memmap=False)
            print(f"Loaded EEG data from {eeg_path}")
            print(f"Available channels: {self.eeg_data.channels[:10]}...")


            self.populate_channel_tree()  # Populate channels in the sidebar
            # Load annotations
            annot_path = self.annot_file_edit.text()
            if not annot_path or not os.path.exists(annot_path):
                raise Exception("Please select a valid annotation file")
            
            self.annotations = CustomAnnotations(annot_path)
            print(f"Loaded annotations from {annot_path}")

            self.populate_annotation_types() # Populate annotation types

            # Load initial events
            self.apply_filters()
            
            # Update UI state
            self.update_ui_state(True)
            
            # Update statistics
            self.update_statistics()
            
            # Plot hypnogram
            if self.annotations:
                self.hypnogram_widget.plot_hypnogram(self.annotations)
            
            self.status_bar.showMessage("Data loaded successfully")
            self.status_progress.setVisible(False)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.status_bar.showMessage("Error loading data")
            self.status_progress.setVisible(False)
            import traceback
            traceback.print_exc()
        self.print_debug_info()



    def apply_filters(self):
        """Apply current filter settings and load events"""
        if not self.db:
            return
        
        try:
            # Determine review status filter
            reviewed_only = self.show_reviewed_radio.isChecked()
            unreviewed_only = self.show_unreviewed_radio.isChecked()
            
            # Get events from database
            self.current_events = self.db.get_events(
                reviewed_only=reviewed_only,
                unreviewed_only=unreviewed_only
            )
            

            if len(self.current_events) > 0 and hasattr(self, 'confidence_threshold'):
                # Use power as proxy for confidence
                power_threshold = self.current_events['power'].max() * self.confidence_threshold
                self.current_events = self.current_events[self.current_events['power'] >= power_threshold]

            if len(self.current_events) > 0:
                self.current_event_index = 0
                self.update_event_display()
            else:
                self.current_event_label.setText("No events match current filters")
                self.update_progress_display()
            
            self.update_statistics()
            
        except Exception as e:
            print(f"Error applying filters: {e}")
    


    def toggle_all_annotations(self, state):
        """Toggle all annotation checkboxes"""
        checked = state == QtCore.Qt.Checked
        for checkbox in self.annotation_checkboxes.values():
            checkbox.setChecked(checked)
        self.update_annotation_visibility()
    
    def update_annotation_visibility(self):
        """Update which annotations are visible in EEG plot"""
        visible_types = []
        for ann_type, checkbox in self.annotation_checkboxes.items():
            if checkbox.isChecked():
                visible_types.append(ann_type)
        
        self.annotation_visibility = set(visible_types)
        
        # Update EEG plot if available
        if hasattr(self, 'eeg_widget'):
            self.eeg_widget.set_annotation_visibility(visible_types)
    
    def update_selected_channels(self):
        """Update selected channels for display"""
        selected = []
        for ch, checkbox in self.channel_checkboxes.items():
            if checkbox.isChecked():
                selected.append(ch)
        
        self.selected_channels = selected
        
        # Refresh current event display if available
        if len(self.current_events) > 0 and self.current_event_index < len(self.current_events):
            self.update_event_display()
    
    def previous_event(self):
        """Navigate to previous event"""
        if len(self.current_events) > 0 and self.current_event_index > 0:
            self.current_event_index -= 1
            self.update_event_display()
    
    def next_event(self):
        """Navigate to next event"""
        if (len(self.current_events) > 0 and 
            self.current_event_index < len(self.current_events) - 1):
            self.current_event_index += 1
            self.update_event_display()
    
    def update_event_display(self):
        """Update display for current event"""
        if len(self.current_events) == 0:
            return
        
        current_event = self.current_events.iloc[self.current_event_index]
        
        # Update event info
        uuid = current_event.get('uuid', 'Unknown')
        channel = current_event.get('channel', 'Unknown')
        start_time = current_event.get('start_time', 0)
        end_time = current_event.get('end_time', 0)
        duration = end_time - start_time if end_time and start_time else 0
        stage = current_event.get('stage', 'Unknown')
        
        info_text = (f"Event: {uuid} | Channel: {channel} | "
                    f"Time: {start_time:.1f}-{end_time:.1f}s | "
                    f"Duration: {duration:.2f}s | Stage: {stage}")
        
        # Add review status
        reviewed = current_event.get('reviewed', 0)
        if reviewed:
            decision = current_event.get('review_decision', 'Unknown')
            info_text += f" | Status: {decision.upper()}"
        else:
            info_text += " | Status: UNREVIEWED"
        
        self.current_event_label.setText(info_text)
        
        # Update progress
        self.update_progress_display()
        
        # Update EEG plot
        if self.eeg_data and self.annotations:
            channels_to_plot = self.selected_channels
            self.eeg_widget.plot_event(
                current_event, self.eeg_data, self.annotations, 
                channels_to_plot, window_duration=self.time_window
            )
        
        # Update hypnogram with current time
        if self.annotations:
            visible_event_types = list(self.annotation_visibility)
            self.hypnogram_widget.plot_hypnogram(
                        self.annotations,  current_time=start_time,
                        event_types=visible_event_types
                    )

        # Update event details
        self.update_event_details(current_event)
        

        # Clear comments
        self.comments_edit.clear()
        
        # Load existing review if present
        if reviewed:
            comments = current_event.get('review_comments', '')
            if comments:
                self.comments_edit.setText(comments)
    
    def update_progress_display(self):
        """Update progress indicators"""
        if len(self.current_events) == 0:
            self.progress_label.setText("0/0")
            self.progress_bar.setValue(0)
            self.progress_percent_label.setText("0% Complete")
            return
        
        current = self.current_event_index + 1
        total = len(self.current_events)
        percent = (current / total) * 100
        
        self.progress_label.setText(f"{current}/{total}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_percent_label.setText(f"{percent:.1f}% Complete")
    
    def update_event_details(self, event):
        """Update event details table"""
        # Clear existing rows
        self.details_table.setRowCount(0)
        
        # Add event parameters
        parameters = [
            ('UUID', event.get('uuid', '')),
            ('Channel', event.get('channel', '')),
            ('Start Time', f"{event.get('start_time', 0):.3f} s"),
            ('End Time', f"{event.get('end_time', 0):.3f} s"),
            ('Duration', f"{event.get('duration', 0):.3f} s"),
            ('Stage', event.get('stage', '')),
            ('Method', event.get('method', '')),
            ('Min Amplitude', f"{event.get('min_amp', 0):.1f} µV"),
            ('Max Amplitude', f"{event.get('max_amp', 0):.1f} µV"),
            ('Peak-to-Peak', f"{event.get('peak2peak_amp', 0):.1f} µV"),
            ('RMS', f"{event.get('rms', 0):.1f} µV"),
            ('Power', f"{event.get('power', 0):.1f} µV²"),
            ('Peak Power Freq', f"{event.get('peak_power_freq', 0):.1f} Hz"),
            ('Energy', f"{event.get('energy', 0):.1f} µV²s"),
        ]
        
        # Add review information if available
        if event.get('reviewed', 0):
            parameters.extend([
                ('Review Decision', event.get('review_decision', '')),
                ('Reviewer', event.get('reviewer', '')),
                ('Review Time', event.get('review_timestamp', '')),
                ('Comments', event.get('review_comments', '')),
            ])
        
        # Populate table
        self.details_table.setRowCount(len(parameters))
        for i, (param, value) in enumerate(parameters):
            self.details_table.setItem(i, 0, QTableWidgetItem(param))
            self.details_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # Resize columns
        self.details_table.resizeColumnsToContents()
    
    def review_current_event(self, decision):
        """Review current event with given decision"""
        if len(self.current_events) == 0 or self.current_event_index >= len(self.current_events):
            return
        
        current_event = self.current_events.iloc[self.current_event_index]
        uuid = current_event['uuid']
        comments = self.comments_edit.toPlainText()
        
        # Save review to database
        self.db.add_review(uuid, decision, self.reviewer_name, comments)
        
        # Update statistics
        self.update_statistics()
        
        # Auto-advance to next unreviewed event
        self.find_next_unreviewed_event()
    
    def find_next_unreviewed_event(self):
        """Find and navigate to next unreviewed event"""
        if len(self.current_events) == 0:
            return
        
        # Look for next unreviewed event starting from current position
        for i in range(self.current_event_index + 1, len(self.current_events)):
            event = self.current_events.iloc[i]
            if not event.get('reviewed', 0):
                self.current_event_index = i
                self.update_event_display()
                return
        
        # If no unreviewed events after current, look from beginning
        for i in range(0, self.current_event_index):
            event = self.current_events.iloc[i]
            if not event.get('reviewed', 0):
                self.current_event_index = i
                self.update_event_display()
                return
        
        # If all events are reviewed, just go to next event
        if self.current_event_index < len(self.current_events) - 1:
            self.current_event_index += 1
            self.update_event_display()
    
    def update_statistics(self):
        """Update review statistics display"""
        if not self.db:
            return
        
        try:
            stats = self.db.get_review_stats()
            
            total = stats.get('total', 0)
            reviewed = stats.get('reviewed', 0)
            accepted = stats.get('accept_count', 0)
            rejected = stats.get('reject_count', 0)
            
            def safe_percentage(numerator, denominator):
                return (numerator / denominator * 100) if denominator > 0 else 0
            
            stats_text = f"""
    Review Progress:
    • Total Events: {total}
    • Reviewed: {reviewed} ({safe_percentage(reviewed, total):.1f}%)
    • Accepted: {accepted}
    • Rejected: {rejected}

    By Channel:
    """
            
            # Add channel statistics
            channel_stats = stats.get('by_channel', [])
            for ch_stat in channel_stats[:10]:  # Show first 10 channels
                ch = ch_stat['channel']
                ch_total = ch_stat['total']
                ch_reviewed = ch_stat['reviewed']
                ch_accepted = ch_stat['accepted']
                ch_rejected = ch_stat['rejected']
                
                stats_text += f"• {ch}: {ch_reviewed}/{ch_total} ({ch_accepted}✓, {ch_rejected}✗)\n"
            
            if len(channel_stats) > 10:
                stats_text += f"... and {len(channel_stats) - 10} more channels"
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def export_results(self):
        """Export review results to CSV"""
        if not self.db:
            QtWidgets.QMessageBox.warning(self, "Warning", "No database loaded")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Review Results", "", 
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
                    f"Failed to export results: {str(e)}"
                )
    
    def update_ui_state(self, data_loaded):
        """Update UI state based on whether data is loaded"""
        # Enable/disable navigation and review controls
        self.prev_btn.setEnabled(data_loaded)
        self.next_btn.setEnabled(data_loaded)
        self.accept_btn.setEnabled(data_loaded)
        self.reject_btn.setEnabled(data_loaded)
        self.apply_filters_btn.setEnabled(data_loaded)
        self.export_btn.setEnabled(data_loaded)

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = EventReviewInterface()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()