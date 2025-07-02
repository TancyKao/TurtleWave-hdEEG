# Main tab integration
import os
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QTabWidget, QSplitter, QGroupBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox)


from turtlewave_hdEEG import LargeDataset, XLAnnotations, ParalEvents, ParalSWA, CustomAnnotations
from .eeg_visualizer import EEGVisualizerWidget
from .hypnogram_visualizer import HypnogramWidget
from .event_navigator import EventNavigator
from .result_manager import ResultManager

class EventReviewTab(QWidget):
    """Event Review Tab for TurtleWave GUI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent  # Parent is the main TurtleWave GUI
        
        # Components
        self.eeg_visualizer = None
        self.hypnogram_widget = None
        self.event_navigator = None
        self.result_manager = None
        
        # Data
        self.dataset = None
        self.annotations = None
        self.events_df = None
        self.current_event_idx = 0
        self.filtered_events = []
        self.filter_settings = {
            'confidence_threshold': 0.5,
            'filter_low': 0.5,
            'filter_high': 45.0,
            'selected_stages': ['N1', 'N2', 'N3', 'REM', 'W']
        }
        self.review_results = {}  # {event_id: {'decision': 'accept'|'reject', 'comment': '...'}}
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        
        # Top section - Controls
        self.controls_group = QGroupBox("Review Controls")
        controls_layout = QHBoxLayout(self.controls_group)
        
        # Event type selection
        type_layout = QVBoxLayout()
        type_layout.addWidget(QLabel("Event Type:"))
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(["Spindles", "Slow Waves"])
        self.event_type_combo.currentTextChanged.connect(self.on_event_type_changed)
        type_layout.addWidget(self.event_type_combo)
        controls_layout.addLayout(type_layout)
        
        # Confidence threshold
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(QLabel("0.5", alignment=QtCore.Qt.AlignCenter))
        controls_layout.addLayout(conf_layout)
        
        # Filter settings
        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("Filter Settings:"))
        filter_controls = QHBoxLayout()
        filter_controls.addWidget(QLabel("Low:"))
        self.filter_low = QDoubleSpinBox()
        self.filter_low.setRange(0.1, 49.9)
        self.filter_low.setValue(0.5)
        self.filter_low.setSingleStep(0.1)
        filter_controls.addWidget(self.filter_low)
        
        filter_controls.addWidget(QLabel("High:"))
        self.filter_high = QDoubleSpinBox()
        self.filter_high.setRange(0.2, 50.0)
        self.filter_high.setValue(45.0)
        self.filter_high.setSingleStep(0.1)
        filter_controls.addWidget(self.filter_high)
        
        self.apply_filter_btn = QPushButton("Apply")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        filter_controls.addWidget(self.apply_filter_btn)
        
        filter_layout.addLayout(filter_controls)
        controls_layout.addLayout(filter_layout)
        
        # Navigation controls
        nav_layout = QVBoxLayout()
        nav_layout.addWidget(QLabel("Navigation:"))
        nav_buttons = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.go_to_previous)
        nav_buttons.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.go_to_next)
        nav_buttons.addWidget(self.next_btn)
        
        nav_layout.addLayout(nav_buttons)
        controls_layout.addLayout(nav_layout)
        
        # Progress
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("0/0")
        progress_layout.addWidget(self.progress_label, alignment=QtCore.Qt.AlignCenter)
        controls_layout.addLayout(progress_layout)
        
        main_layout.addWidget(self.controls_group)
        
        # Middle section - Visualization
        self.viz_tabs = QTabWidget()
        
        # EEG View tab
        self.eeg_tab = QWidget()
        eeg_layout = QVBoxLayout(self.eeg_tab)
        
        # Will be initialized when data is loaded
        self.eeg_visualizer = EEGVisualizerWidget()
        
        # Hypnogram view (will be added above EEG visualizer when data is loaded)
        self.hypnogram_widget = HypnogramWidget()
        
        eeg_layout.addWidget(self.hypnogram_widget)
        eeg_layout.addWidget(self.eeg_visualizer)
        
        self.viz_tabs.addTab(self.eeg_tab, "EEG View")
        
        # Add placeholder for other tabs
        self.viz_tabs.addTab(QWidget(), "Time-Frequency")
        self.viz_tabs.addTab(QWidget(), "Topography")
        
        main_layout.addWidget(self.viz_tabs, 1)  # 1 = stretch factor
        
        # Bottom section - Review actions
        action_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("Accept Event")
        self.accept_btn.setStyleSheet("background-color: #00cc66; color: white; font-weight: bold;")
        self.accept_btn.clicked.connect(lambda: self.mark_event('accept'))
        action_layout.addWidget(self.accept_btn)
        
        self.reject_btn = QPushButton("Reject Event")
        self.reject_btn.setStyleSheet("background-color: #ff4b4b; color: white; font-weight: bold;")
        self.reject_btn.clicked.connect(lambda: self.mark_event('reject'))
        action_layout.addWidget(self.reject_btn)
        
        self.comment_input = QtWidgets.QLineEdit()
        self.comment_input.setPlaceholderText("Optional comment...")
        action_layout.addWidget(self.comment_input)
        
        main_layout.addLayout(action_layout)
        
        # Initialize components in disabled state until data is loaded
        self.set_controls_enabled(False)
    
    def setup_event_navigator(self):
        """Initialize the event navigator"""
        self.event_navigator = EventNavigator(self.events_df, self.filter_settings)
        self.update_navigator_ui()
    
    def set_controls_enabled(self, enabled):
        """Enable or disable controls based on data availability"""
        self.event_type_combo.setEnabled(enabled)
        self.conf_slider.setEnabled(enabled)
        self.filter_low.setEnabled(enabled)
        self.filter_high.setEnabled(enabled)
        self.apply_filter_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.accept_btn.setEnabled(enabled)
        self.reject_btn.setEnabled(enabled)
        self.comment_input.setEnabled(enabled)
    
    def on_event_type_changed(self, event_type):
        """Handle event type selection change"""
        self.load_events(event_type)
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        self.filter_settings['confidence_threshold'] = value / 100.0
        # Update the displayed value
        self.conf_slider.parent().layout().itemAt(2).widget().setText(f"{self.filter_settings['confidence_threshold']:.2f}")
        self.apply_filters()
    
    def apply_filter(self):
        """Apply EEG filter settings"""
        self.filter_settings['filter_low'] = self.filter_low.value()
        self.filter_settings['filter_high'] = self.filter_high.value()
        
        # If we have a dataset, apply the filter to the data
        if self.dataset and hasattr(self.eeg_visualizer, 'update_filter'):
            self.eeg_visualizer.update_filter(
                self.filter_settings['filter_low'],
                self.filter_settings['filter_high']
            )
    
    def apply_filters(self):
        """Apply all filters to the events dataframe"""
        if self.event_navigator:
            self.event_navigator.apply_filters(self.filter_settings)
            self.filtered_events = self.event_navigator.get_filtered_events()
            self.update_navigator_ui()
            
            # Reset to first event if list changed
            if len(self.filtered_events) > 0:
                self.current_event_idx = 0
                self.display_current_event()
    
    def update_navigator_ui(self):
        """Update UI elements related to navigation"""
        if not self.event_navigator:
            self.progress_bar.setValue(0)
            self.progress_label.setText("0/0")
            return
            
        total = len(self.filtered_events)
        reviewed = sum(1 for event_id in self.filtered_events if event_id in self.review_results)
        
        # Update progress bar
        if total > 0:
            self.progress_bar.setValue(int(reviewed / total * 100))
        else:
            self.progress_bar.setValue(0)
            
        # Update progress label
        self.progress_label.setText(f"{reviewed}/{total}")
        
        # Enable/disable navigation buttons
        self.prev_btn.setEnabled(self.current_event_idx > 0)
        self.next_btn.setEnabled(self.current_event_idx < total - 1)
    
    def go_to_previous(self):
        """Go to previous event"""
        if self.current_event_idx > 0:
            self.current_event_idx -= 1
            self.display_current_event()
            self.update_navigator_ui()
    
    def go_to_next(self):
        """Go to next event"""
        if self.current_event_idx < len(self.filtered_events) - 1:
            self.current_event_idx += 1
            self.display_current_event()
            self.update_navigator_ui()
    
    def mark_event(self, decision):
        """Mark current event as accepted or rejected"""
        if not self.filtered_events or self.current_event_idx >= len(self.filtered_events):
            return
            
        event_id = self.filtered_events[self.current_event_idx]
        comment = self.comment_input.text()
        
        self.review_results[event_id] = {
            'decision': decision,
            'comment': comment,
            'timestamp': pd.Timestamp.now()
        }
        
        # Clear comment field
        self.comment_input.clear()
        
        # Update UI
        self.update_navigator_ui()
        
        # Auto-advance to next event
        self.go_to_next()
    
    def display_current_event(self):
        """Display the current event in the visualizer"""
        if not self.filtered_events or self.current_event_idx >= len(self.filtered_events):
            return
            
        event_id = self.filtered_events[self.current_event_idx]
        event = self.events_df.loc[event_id]
        
        # Update the EEG visualizer with the current event
        start_time = event['start_time']
        end_time = event['end_time']
        channel = event['channel']
        
        # Add 1 second before and after for context
        context_start = max(0, start_time - 1)
        context_end = end_time + 1
        
        # Update visualizers
        if hasattr(self.eeg_visualizer, 'display_event'):
            self.eeg_visualizer.display_event(
                self.dataset,
                channel,
                context_start,
                context_end,
                start_time,
                end_time
            )
        
        if hasattr(self.hypnogram_widget, 'display_current_time'):
            self.hypnogram_widget.display_current_time(start_time)
        
        # Update event info in UI
        event_type = self.event_type_combo.currentText()
        self.viz_tabs.setTabText(0, f"{event_type[:-1]} #{event_id} - Ch: {channel}")
    
    def load_events(self, event_type):
        """Load events based on selected type"""
        if not self.parent or not hasattr(self.parent, 'output_dir'):
            QMessageBox.warning(self, "Warning", "No output directory defined")
            return
            
        # Determine the directory based on event type
        if event_type == "Spindles":
            results_dir = os.path.join(self.parent.output_dir, "wonambi", "spindle_results")
        else:  # Slow Waves
            results_dir = os.path.join(self.parent.output_dir, "wonambi", "sw_results")
        
        if not os.path.exists(results_dir):
            QMessageBox.warning(self, "Warning", f"No {event_type.lower()} results found")
            return
        
        # Find parameter files
        param_files = [f for f in os.listdir(results_dir) if f.startswith(f"{'spindle' if event_type == 'Spindles' else 'sw'}_parameters_")]
        
        if not param_files:
            QMessageBox.warning(self, "Warning", f"No {event_type.lower()} parameter files found")
            return
        
        # Use the first parameter file
        param_file = os.path.join(results_dir, param_files[0])
        
        try:
            # Load events dataframe
            self.events_df = pd.read_csv(param_file)
            
            # Check if we have required columns
            required_cols = ['start_time', 'end_time', 'channel']
            missing_cols = [col for col in required_cols if col not in self.events_df.columns]
            if missing_cols:
                QMessageBox.warning(self, "Warning", f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Add confidence column if not present
            if 'confidence' not in self.events_df.columns:
                self.events_df['confidence'] = 1.0
            
            # Setup navigator
            self.setup_event_navigator()
            
            # Enable controls
            self.set_controls_enabled(True)
            
            # Display first event
            if len(self.filtered_events) > 0:
                self.current_event_idx = 0
                self.display_current_event()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load events: {str(e)}")
    
    def load_data(self, dataset, annotations=None):
        """Load dataset and annotations for review"""
        self.dataset = dataset  # Use the dataset passed from the main application
        
        self.annotations = annotations
        
        # Initialize visualizers with dataset
        if hasattr(self.eeg_visualizer, 'set_dataset'):
            self.eeg_visualizer.set_dataset(dataset)
        
        if hasattr(self.hypnogram_widget, 'set_dataset'):
            self.hypnogram_widget.set_dataset(dataset, annotations)
        
        # Load default event type
        self.load_events(self.event_type_combo.currentText())
    
    def save_results(self):
        """Save review results to CSV"""
        if not self.review_results:
            QMessageBox.information(self, "Information", "No review results to save.")
            return
            
        event_type = self.event_type_combo.currentText().lower()
        output_dir = os.path.join(self.parent.output_dir, "wonambi", f"{event_type[:-1]}_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Create results dataframe
            results = []
            for event_id, result in self.review_results.items():
                if event_id in self.events_df.index:
                    event = self.events_df.loc[event_id]
                    results.append({
                        'event_id': event_id,
                        'start_time': event['start_time'],
                        'end_time': event['end_time'],
                        'channel': event['channel'],
                        'confidence': event.get('confidence', 1.0),
                        'decision': result['decision'],
                        'comment': result.get('comment', ''),
                        'review_time': result.get('timestamp', pd.Timestamp.now())
                    })
            
            if not results:
                QMessageBox.information(self, "Information", "No valid results to save.")
                return
                
            results_df = pd.DataFrame(results)
            
            # Save to CSV
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type[:-1]}_review_results_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            results_df.to_csv(filepath, index=False)
            QMessageBox.information(self, "Success", f"Results saved to {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")