#!/usr/bin/env python3

"""
TurtleWave hdEEG GUI
A graphical user interface for the TurtleWave hdEEG package, combining annotation
and spindle detection functionalities in a user-friendly interface.
"""

import os
import sys
import threading
import json
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, 
                            QGroupBox, QCheckBox, QListWidget, QListWidgetItem, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QFrame,
                            QSplitter, QAbstractItemView, QProgressBar)
import logging

# Try importing the required packages
try:
    from turtlewave_hdEEG import LargeDataset, XLAnnotations, ParalEvents, ParalSWA, CustomAnnotations
    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave, ImprovedDetectSpindle
    
    #from wonambi.dataset import Dataset as WonambiDataset
except ImportError as e:
    print(f"Error importing TurtleWave hdEEG package: {e}")


class LoggingOutput(QtCore.QObject):
    """Class to capture and redirect logging to the GUI"""
    text_written = QtCore.pyqtSignal(str)
    
    def write(self, text):
        if text.strip():  # Only emit if there's actual text
            self.text_written.emit(text.rstrip())
    
    def flush(self):
        pass

class GUILogHandler(logging.Handler):
    """Custom log handler to redirect logs to the GUI"""
    
    def __init__(self, signal_fn):
        """Initialize with a function to emit log messages to"""
        super().__init__()
        self.signal_fn = signal_fn
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """Emit a log record to the GUI"""
        log_message = self.format(record)
        # Use the signal function to write to the GUI log
        self.signal_fn(log_message)

class TurtleWaveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup window properties
        self.setWindowTitle("TurtleWave hdEEG - Sleep Event Detection and Coupling Suite")
        self.setGeometry(100, 100, 1200, 800)
        

        # Variables
        self.data_file_path = ""
        self.output_dir = ""
        self.annot_file_path = ""
        self.spindle_method = "Moelle2011"
        self.min_freq = 9.0
        self.max_freq = 12.0
        self.min_duration = 0.5
        self.max_duration = 3.0
        self.selected_channels = []
        self.available_channels = []
        self.dataset = None
        self.annotations = None
        
        # slow wave variables
        self.sw_method = "Massimini2004"
        self.sw_min_freq = 0.1
        self.sw_max_freq = 4.0
        self.sw_min_duration = 0.3
        self.sw_max_duration = 1.5
        self.sw_neg_peak_thresh = -80.0
        self.sw_p2p_thresh = 140.0
        self.sw_invert = False


        # Initialize log text area to avoid reference before assignment
        self.log_text = None

        # Setup UI
        self.setup_ui()
        
        # Redirect stdout for logging
        self.log_output = LoggingOutput()
        self.log_output.text_written.connect(self.write_log)
        sys.stdout = self.log_output
    
    def setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.setup_tab = QWidget()
        self.annotation_tab = QWidget()
        self.spindle_tab = QWidget()
        self.pac_tab = QWidget()  # Add PAC tab
        self.log_tab = QWidget()
        self.sw_tab = QWidget()  
        #self.review_tab = EventReviewTab(self) <================

        # Add tabs to widget
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.annotation_tab, "Annotation")
        self.tabs.addTab(self.spindle_tab, "Spindle Detection")
        self.tabs.addTab(self.sw_tab, "Slow Wave Detection")
        self.tabs.addTab(self.pac_tab, "PAC Analysis") 
        #self.tabs.addTab(self.review_tab, "Event Review") <================
        self.tabs.addTab(self.log_tab, "Log")
        # Connect tab change signal
        self.tabs.currentChanged.connect(self.handle_tab_change)
        
        # Setup tab contents
        self.setup_setup_tab()
        self.setup_annotation_tab()
        self.setup_spindle_tab()
        self.setup_sw_tab()  
        self.setup_pac_tab()  # Add setup for PAC tab
        self.setup_log_tab()
        
        # Add the tabs to the main layout
        self.main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Progress bar in status bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        
        # Disable tabs that require data to be loaded
        self.tabs.setTabEnabled(1, False)  # Annotation tab
        self.tabs.setTabEnabled(2, False)  # Spindle tab
        self.tabs.setTabEnabled(3, False)  # Slow Wave tab
        self.tabs.setTabEnabled(4, False)  # PAC tab
        #self.tabs.setTabEnabled(self.tabs.indexOf(self.review_tab), False)<=================

    def handle_tab_change(self, index):
        """Handle tab changes"""
        # If switching to PAC tab, make sure methods are populated
        if index == 4:  # PAC tab index
            self.populate_detection_methods()

    def setup_setup_tab(self):
        # Main layout
        layout = QVBoxLayout(self.setup_tab)
        

        # File selection group
        file_group = QGroupBox("Data Selection")
        file_layout = QVBoxLayout()
        
        # EEG data file
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("EEG Data File:"))
        self.data_file_edit = QLineEdit()
        data_layout.addWidget(self.data_file_edit)
        self.browse_data_btn = QPushButton("Browse...")
        self.browse_data_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.browse_data_btn)
        file_layout.addLayout(data_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        output_layout.addWidget(self.output_dir_edit)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_output_btn)
        file_layout.addLayout(output_layout)
        
        # Annotation file
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation File (Optional):"))
        self.annot_file_edit = QLineEdit()
        annot_layout.addWidget(self.annot_file_edit)
        self.browse_annot_btn = QPushButton("Browse...")
        self.browse_annot_btn.clicked.connect(self.browse_annot_file)
        annot_layout.addWidget(self.browse_annot_btn)
        file_layout.addLayout(annot_layout)
        
        # Load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data_thread)
        self.load_btn.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(self.load_btn)
        

        # Event Review button
        self.review_btn = QPushButton("Launch Event Review")
        self.review_btn.clicked.connect(self.launch_event_review)
        self.review_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        self.review_btn.setEnabled(False)  # Enable after data is loaded
        
        # Add this line to temporarily disable the button completely:
        self.review_btn.setVisible(False)  # Hide the button temporarily

        file_layout.addWidget(self.review_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Dataset information group
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setText("No dataset loaded. Please select an EEG data file and click 'Load Data'.")
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group, 1)  # 1 means this will stretch
    
    def setup_annotation_tab(self):
        # Main layout
        layout = QVBoxLayout(self.annotation_tab)
        
        # Top area - split into left and right
        top_layout = QHBoxLayout()
        
        # Left side - options
        options_group = QGroupBox("Annotation Options")
        options_layout = QVBoxLayout()
        
        # Checkboxes for annotation types
        self.artifact_check = QCheckBox("Process Artifacts")
        self.artifact_check.setChecked(True)
        options_layout.addWidget(self.artifact_check)
        
        self.arousal_check = QCheckBox("Process Arousals")
        self.arousal_check.setChecked(True)
        options_layout.addWidget(self.arousal_check)
        
        self.stage_check = QCheckBox("Process Sleep Stages")
        self.stage_check.setChecked(True)
        options_layout.addWidget(self.stage_check)
        
        options_layout.addStretch(1)
        options_group.setLayout(options_layout)
        top_layout.addWidget(options_group)
        
        # Right side - actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.generate_annot_btn = QPushButton("Generate Annotations")
        self.generate_annot_btn.clicked.connect(self.process_annotations_thread)
        actions_layout.addWidget(self.generate_annot_btn)
        
        self.view_annot_btn = QPushButton("View Annotation File")
        self.view_annot_btn.clicked.connect(self.view_annotation_file)
        self.view_annot_btn.setEnabled(False)
        actions_layout.addWidget(self.view_annot_btn)
        
        info_label = QLabel("Note: Annotation generation may take some time\nfor large datasets.")
        actions_layout.addWidget(info_label)
        
        actions_layout.addStretch(1)
        actions_group.setLayout(actions_layout)
        top_layout.addWidget(actions_group)
        
        layout.addLayout(top_layout)
    
    def clear_layout(self, layout):
        """Remove all widgets and layouts from the given layout"""
        if layout is None:
            return
            
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)

    def setup_sw_tab(self):
        """Setup the slow wave detection tab"""
        # Main layout
        layout = QVBoxLayout(self.sw_tab)
        
        # Top section split into two columns
        top_splitter = QSplitter(QtCore.Qt.Horizontal)
        
        # Left column - parameters
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Parameters group
        params_group = QGroupBox("Slow Wave Detection Parameters")
        params_form = QVBoxLayout()
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Detection Method:"))
        self.sw_method_combo = QComboBox()
        self.sw_method_combo.addItems(["Massimini2004", "AASM/Massimini2004", "Ngo2015", "Staresina2015"])
        method_layout.addWidget(self.sw_method_combo)
        params_form.addLayout(method_layout)
        
        # Add a container for method-specific parameters ===
        self.method_params_container = QGroupBox("Method-Specific Parameters")
        self.method_params_layout = QVBoxLayout(self.method_params_container)
        params_form.addWidget(self.method_params_container)
        
        # Connect method change to parameter update
        self.sw_method_combo.currentTextChanged.connect(self.update_sw_params_for_method)
        
        
        
        # Options
        self.reject_artifacts_check = QCheckBox("Reject Artifacts")
        self.reject_artifacts_check.setChecked(True)
        params_form.addWidget(self.reject_artifacts_check)
        
        self.reject_arousals_check = QCheckBox("Reject Arousals")
        self.reject_arousals_check.setChecked(True)
        params_form.addWidget(self.reject_arousals_check)
        
        params_group.setLayout(params_form)
        params_layout.addWidget(params_group)
        
        # Stage selection group
        stages_group = QGroupBox("Sleep Stage Selection")
        stages_layout = QHBoxLayout()
        
        self.sw_stage_checks = {}
        stages = ["NREM1", "NREM2", "NREM3", "REM", "Wake"]
        default_selected = ["NREM2", "NREM3"]
        
        for stage in stages:
            check = QCheckBox(stage)
            check.setChecked(stage in default_selected)
            stages_layout.addWidget(check)
            self.sw_stage_checks[stage] = check
        
        stages_group.setLayout(stages_layout)
        params_layout.addWidget(stages_group)
        
        params_layout.addStretch(1)
        top_splitter.addWidget(params_widget)
        
        # Right column - channel selection
        channels_widget = QWidget()
        channels_layout = QVBoxLayout(channels_widget)
        
        channels_group = QGroupBox("Channel Selection")
        channels_content = QHBoxLayout()
        
        # Available channels
        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Available Channels:"))
        self.sw_available_list = QListWidget()
        self.sw_available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        avail_layout.addWidget(self.sw_available_list)
        channels_content.addLayout(avail_layout)
        
        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.addStretch(1)
        
        self.sw_add_btn = QPushButton("Add >")  # Changed from self.add_btn
        self.sw_add_btn.clicked.connect(self.add_sw_channels)  # Need a new method
        btn_layout.addWidget(self.sw_add_btn)
        
        self.sw_remove_btn = QPushButton("< Remove")  # Changed from self.remove_btn
        self.sw_remove_btn.clicked.connect(self.remove_sw_channels)  # Need a new method
        btn_layout.addWidget(self.sw_remove_btn)
        
        self.sw_add_all_btn = QPushButton("Add All >>")  # Changed from self.add_all_btn
        self.sw_add_all_btn.clicked.connect(self.add_all_sw_channels)  # Need a new method
        btn_layout.addWidget(self.sw_add_all_btn)
        
        self.sw_remove_all_btn = QPushButton("<< Remove All")  # Changed from self.remove_all_btn
        self.sw_remove_all_btn.clicked.connect(self.remove_all_sw_channels)  # Need a new method
        btn_layout.addWidget(self.sw_remove_all_btn)
        
        btn_layout.addStretch(1)
        channels_content.addLayout(btn_layout)
        
        # Selected channels
        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected Channels:"))
        self.sw_selected_list = QListWidget()  # Changed from self.selected_list
        self.sw_selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(self.sw_selected_list)
        channels_content.addLayout(sel_layout)
        
        
        channels_group.setLayout(channels_content)
        channels_layout.addWidget(channels_group)
        
        top_splitter.addWidget(channels_widget)
        
        # Add the splitter to the main layout
        layout.addWidget(top_splitter)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.detect_sw_btn = QPushButton("Detect Slow Waves")
        self.detect_sw_btn.clicked.connect(self.detect_sw_thread)
        self.detect_sw_btn.setStyleSheet("font-weight: bold;")
        action_layout.addWidget(self.detect_sw_btn)
        
        self.view_sw_results_btn = QPushButton("View Results")
        self.view_sw_results_btn.clicked.connect(self.view_sw_results)
        self.view_sw_results_btn.setEnabled(False)
        action_layout.addWidget(self.view_sw_results_btn)
        
        layout.addLayout(action_layout)
        
        # Initialize method-specific parameters for the default method
        self.update_sw_params_for_method(self.sw_method_combo.currentText())    


    def update_sw_params_for_method(self, method_name):
        """Update slow wave detection parameters based on selected method"""
        # Clear previous parameter widgets
        self.clear_layout(self.method_params_layout)
        
        # Import the detector class to access parameters
        try:
            
            # Create a temporary detector with the selected method to access its parameters
            detector = ImprovedDetectSlowWave(method=method_name)
            
            # Display info about the method
            method_descriptions = {
                "Massimini2004": "Detects slow waves via bandpass filtering (0.1 to 4 Hz) and amplitude/duration thresholds, marking negative half-waves and peaks.",
                "AASM/Massimini2004": "Adapts the Massimini method to use AASM-recommended amplitude and duration thresholds for standardized detection.",
                "Ngo2015": "Uses adaptive thresholds based on EEG variance for individualized, real-time slow wave detection, ideal for closed-loop stimulation.",
                "Staresina2015": "Targets very low frequencies (<1.25 Hz) with strict criteria to isolate slow oscillations in the EEG."
            }
            
            # Add description label
            if method_name in method_descriptions:
                desc_label = QLabel(method_descriptions[method_name])
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: #333; font-style: italic; background-color: #f0f4f7; padding: 8px; border-radius: 4px;")
                self.method_params_layout.addWidget(desc_label)
                self.method_params_layout.addSpacing(10)
            
            # Create header
            info_label = QLabel(f"<b>Parameters for {method_name}:</b>")
            info_label.setAlignment(QtCore.Qt.AlignCenter)
            self.method_params_layout.addWidget(info_label)
            
            # Initialize dict to store UI elements
            self.sw_param_widgets = {}
            
            if method_name in ["Massimini2004", "AASM/Massimini2004"]:
                # === Filter settings ===
                filter_group = QGroupBox("Filter Settings")
                filter_layout = QVBoxLayout()
                
                # Filter order
                order_layout = QHBoxLayout()
                order_layout.addWidget(QLabel("Filter Order:"))
                order_spin = QSpinBox()
                order_spin.setRange(1, 10)
                order_spin.setValue(detector.det_filt.get('order', 2))
                order_layout.addWidget(order_spin)
                filter_layout.addLayout(order_layout)
                
                # Frequency range
                freq_layout = QHBoxLayout()
                freq_layout.addWidget(QLabel("Frequency Range (Hz):"))
                
                freq_range = detector.det_filt.get('freq', (0.1, 4.0))
                
                freq_layout.addWidget(QLabel("Min:"))
                min_freq_spin = QDoubleSpinBox()
                min_freq_spin.setRange(0.01, 10.0)
                min_freq_spin.setSingleStep(0.1)
                min_freq_spin.setValue(freq_range[0])
                freq_layout.addWidget(min_freq_spin)
                
                freq_layout.addWidget(QLabel("Max:"))
                max_freq_spin = QDoubleSpinBox()
                max_freq_spin.setRange(0.1, 20.0)
                max_freq_spin.setSingleStep(0.1)
                max_freq_spin.setValue(freq_range[1])
                freq_layout.addWidget(max_freq_spin)
                
                filter_layout.addLayout(freq_layout)
                filter_group.setLayout(filter_layout)
                self.method_params_layout.addWidget(filter_group)
                
                # Store filter widgets
                self.sw_param_widgets["filter"] = {
                    "order": order_spin,
                    "min_freq": min_freq_spin,
                    "max_freq": max_freq_spin
                }
                
                # MODIFIED: Get trough duration from detector
                trough_group = QGroupBox("Trough Duration (Negative Half-Wave)")
                trough_layout = QHBoxLayout()
                
                trough_duration = detector.trough_duration
                
                trough_layout.addWidget(QLabel("Min (s):"))
                min_trough_spin = QDoubleSpinBox()
                min_trough_spin.setRange(0.01, 5.0)
                min_trough_spin.setSingleStep(0.05)
                min_trough_spin.setValue(trough_duration[0])
                trough_layout.addWidget(min_trough_spin)
                
                trough_layout.addWidget(QLabel("Max (s):"))
                max_trough_spin = QDoubleSpinBox()
                max_trough_spin.setRange(0.1, 10.0)
                max_trough_spin.setSingleStep(0.1)
                max_trough_spin.setValue(trough_duration[1])
                trough_layout.addWidget(max_trough_spin)
                
                trough_group.setLayout(trough_layout)
                self.method_params_layout.addWidget(trough_group)
                
                # Store trough duration widgets
                self.sw_param_widgets["trough_duration"] = {
                    "min": min_trough_spin,
                    "max": max_trough_spin
                }
                
                # MODIFIED: Get amplitude thresholds from detector
                threshold_group = QGroupBox("Amplitude Thresholds")
                threshold_layout = QVBoxLayout()
                
                # Negative peak threshold
                neg_layout = QHBoxLayout()
                neg_layout.addWidget(QLabel("Negative Peak Threshold (μV):"))
                neg_peak_spin = QDoubleSpinBox()
                neg_peak_spin.setRange(-200, 0)
                neg_peak_spin.setValue(detector.max_trough_amp)
                neg_layout.addWidget(neg_peak_spin)
                threshold_layout.addLayout(neg_layout)
                
                # Peak-to-peak threshold
                p2p_layout = QHBoxLayout()
                p2p_layout.addWidget(QLabel("Peak-to-Peak Threshold (μV):"))
                p2p_spin = QDoubleSpinBox()
                p2p_spin.setRange(0, 300)
                p2p_spin.setValue(detector.min_ptp)
                p2p_layout.addWidget(p2p_spin)
                threshold_layout.addLayout(p2p_layout)
                
                threshold_group.setLayout(threshold_layout)
                self.method_params_layout.addWidget(threshold_group)
                
                # Store threshold widgets
                self.sw_param_widgets["max_trough_amp"] = neg_peak_spin
                self.sw_param_widgets["min_ptp"] = p2p_spin
                    
            # MODIFIED: Different parameters for Ngo2015 and Staresina2015
            elif method_name in ["Ngo2015", "Staresina2015"]:
                # MODIFIED: Get lowpass filter settings from detector
                filter_group = QGroupBox("Lowpass Filter")
                filter_layout = QVBoxLayout()
                
                # Filter order
                order_layout = QHBoxLayout()
                order_layout.addWidget(QLabel("Filter Order:"))
                order_spin = QSpinBox()
                order_spin.setRange(1, 10)
                order_spin.setValue(detector.lowpass.get('order', 2))
                order_layout.addWidget(order_spin)
                filter_layout.addLayout(order_layout)
                
                # Cutoff frequency
                freq_layout = QHBoxLayout()
                freq_layout.addWidget(QLabel("Cutoff Frequency (Hz):"))
                freq_spin = QDoubleSpinBox()
                freq_spin.setRange(0.1, 20.0)
                freq_spin.setSingleStep(0.1)
                freq_spin.setValue(detector.lowpass.get('freq', 3.5))
                freq_layout.addWidget(freq_spin)
                filter_layout.addLayout(freq_layout)
                
                filter_group.setLayout(filter_layout)
                self.method_params_layout.addWidget(filter_group)
                
                # Store filter widgets
                self.sw_param_widgets["lowpass"] = {
                    "order": order_spin,
                    "freq": freq_spin
                }
                
                # MODIFIED: Get duration from detector
                dur_group = QGroupBox("Slow Wave Duration")
                dur_layout = QHBoxLayout()
                
                dur_layout.addWidget(QLabel("Min (s):"))
                min_dur_spin = QDoubleSpinBox()
                min_dur_spin.setRange(0.01, 5.0)
                min_dur_spin.setSingleStep(0.05)
                min_dur_spin.setValue(detector.min_dur)
                dur_layout.addWidget(min_dur_spin)
                
                dur_layout.addWidget(QLabel("Max (s):"))
                max_dur_spin = QDoubleSpinBox()
                max_dur_spin.setRange(0.1, 10.0)
                max_dur_spin.setSingleStep(0.1)
                max_dur_spin.setValue(detector.max_dur)
                dur_layout.addWidget(max_dur_spin)
                
                dur_group.setLayout(dur_layout)
                self.method_params_layout.addWidget(dur_group)
                
                # Store duration widgets
                self.sw_param_widgets["duration"] = {
                    "min": min_dur_spin,
                    "max": max_dur_spin
                }
                
                # MODIFIED: Add calculated frequency range display based on duration
                freq_group = QGroupBox("Calculated Frequency Range")
                freq_layout = QVBoxLayout()
                
                # Calculate frequency range based on duration
                min_freq = 1.0 / detector.max_dur
                max_freq = 1.0 / detector.min_dur
                
                info_text = QLabel(f"Based on duration: {min_freq:.2f} - {max_freq:.2f} Hz")
                info_text.setAlignment(QtCore.Qt.AlignCenter)
                freq_layout.addWidget(info_text)
                
                # Setup connections to update frequency range when duration changes
                def update_freq_range():
                    try:
                        min_dur = min_dur_spin.value()
                        max_dur = max_dur_spin.value()
                        if min_dur > 0 and max_dur > 0:
                            min_freq = 1.0 / max_dur
                            max_freq = 1.0 / min_dur
                            info_text.setText(f"Based on duration: {min_freq:.2f} - {max_freq:.2f} Hz")
                        else:
                            info_text.setText("Error: Duration values must be greater than zero")
                    except ZeroDivisionError:
                        info_text.setText("Error: Duration values cannot be zero")
                
                min_dur_spin.valueChanged.connect(update_freq_range)
                max_dur_spin.valueChanged.connect(update_freq_range)
                
                freq_group.setLayout(freq_layout)
                self.method_params_layout.addWidget(freq_group)
                
                # MODIFIED: Method-specific thresholds
                if method_name == "Ngo2015":
                    # MODIFIED: Get adaptive thresholds from detector
                    thresh_group = QGroupBox("Adaptive Thresholds")
                    thresh_layout = QVBoxLayout()
                    
                    # Peak threshold
                    peak_layout = QHBoxLayout()
                    peak_layout.addWidget(QLabel("Peak Threshold (σ):"))
                    peak_spin = QDoubleSpinBox()
                    peak_spin.setRange(0, 10)
                    peak_spin.setSingleStep(0.05)
                    peak_spin.setValue(detector.peak_thresh)
                    peak_spin.setToolTip("Threshold in standard deviations (σ) above mean")
                    peak_layout.addWidget(peak_spin)
                    thresh_layout.addLayout(peak_layout)
                    
                    # Peak-to-peak threshold
                    ptp_layout = QHBoxLayout()
                    ptp_layout.addWidget(QLabel("Peak-to-Peak Threshold (σ):"))
                    ptp_spin = QDoubleSpinBox()
                    ptp_spin.setRange(0, 10)
                    ptp_spin.setSingleStep(0.05)
                    ptp_spin.setValue(detector.ptp_thresh)
                    ptp_spin.setToolTip("Threshold in standard deviations (σ) above mean")
                    ptp_layout.addWidget(ptp_spin)
                    thresh_layout.addLayout(ptp_layout)
                    
                    thresh_group.setLayout(thresh_layout)
                    self.method_params_layout.addWidget(thresh_group)
                    
                    # Store threshold widgets
                    self.sw_param_widgets["peak_thresh"] = peak_spin
                    self.sw_param_widgets["ptp_thresh"] = ptp_spin
                    
                elif method_name == "Staresina2015":
                    # MODIFIED: Get p2p threshold from detector
                    ptp_group = QGroupBox("Amplitude Threshold")
                    ptp_layout = QHBoxLayout()
                    ptp_layout.addWidget(QLabel("Peak-to-Peak Threshold (μV):"))
                    ptp_spin = QDoubleSpinBox()
                    ptp_spin.setRange(0, 1000)
                    ptp_spin.setValue(detector.ptp_thresh)
                    ptp_layout.addWidget(ptp_spin)
                    ptp_group.setLayout(ptp_layout)
                    self.method_params_layout.addWidget(ptp_group)
                    
                    # Store ptp threshold widget
                    self.sw_param_widgets["ptp_thresh"] = ptp_spin
            
            else:
                # If method not recognized
                error_label = QLabel(f"Error: Parameters for method '{method_name}' not available.")
                error_label.setStyleSheet("color: red;")
                self.method_params_layout.addWidget(error_label)
            
            # Add common options
            options_group = QGroupBox("Signal Processing Options")
            options_layout = QHBoxLayout()
            
            
            invert_check = QCheckBox("Invert Signal")
            invert_check.setChecked(False)  # Default is normal polarity
            options_layout.addWidget(invert_check)
            
            options_group.setLayout(options_layout)
            self.method_params_layout.addWidget(options_group)
            
            # Store option widgets
            self.sw_param_widgets["invert"] = invert_check
            
            # Add a spacer at the end
            self.method_params_layout.addStretch(1)
            
        except Exception as e:
            # If we can't import or access the detector
            self.write_log(f"Error loading parameters from ImprovedDetectSlowWave: {str(e)}")
            import traceback
            traceback.print_exc()
            
            error_label = QLabel(f"Error loading parameters: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.method_params_layout.addWidget(error_label)
            
            # Log the change
            self.write_log(f"Updated parameters for {method_name} slow wave detection method")

    def detect_sw_thread(self):
        """Start slow wave detection in a separate thread"""
        if not self.dataset:
            QMessageBox.critical(self, "Error", "No dataset loaded. Please load a dataset first.")
            return

        
        # Check if annotation file exists
        if not os.path.isfile(self.annot_file_path):
            response = QMessageBox.question(
                self, "Annotation File Missing", 
                "No annotation file found. Would you like to generate annotations first?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                self.tabs.setCurrentIndex(1)  # Switch to annotation tab
                return
            else:
                return


        # Get the selected method
        self.sw_method = self.sw_method_combo.currentText()
        
        # Check if channels are selected
        if not self.selected_channels:
            QMessageBox.critical(self, "Error", "No channels selected. Please select at least one channel.")
            return
        
        # Get selected sleep stages
        self.selected_stages = [stage for stage, check in self.sw_stage_checks.items() if check.isChecked()]
        if not self.selected_stages:
            QMessageBox.critical(self, "Error", "No sleep stages selected. Please select at least one stage.")
            return
        


        # ===  Get method-specific parameters depending on the selected method ===
        try:
            # Common parameters for all methods
            polar = 'opposite' if self.sw_param_widgets.get("invert", QCheckBox()).isChecked() else 'normal'
            
            # ===  Method-specific parameters ===
            if self.sw_method in ["Massimini2004", "AASM/Massimini2004"]:
                # Get filter parameters
                filter_widgets = self.sw_param_widgets["filter"]
                frequency = (filter_widgets["min_freq"].value(), filter_widgets["max_freq"].value())
                
                # For Massimini methods, use trough_duration instead of min_dur/max_dur 
                trough_widgets = self.sw_param_widgets["trough_duration"]
                trough_duration = (trough_widgets["min"].value(), trough_widgets["max"].value())
                
                # Get amplitude thresholds
                neg_peak_thresh = self.sw_param_widgets["max_trough_amp"].value()
                p2p_thresh = self.sw_param_widgets["min_ptp"].value()
 
                # These methods don't use min_dur/max_dur
                min_dur = None
                max_dur = None
                
                
            elif self.sw_method in ["Ngo2015", "Staresina2015"]:
                # Get duration range from specific widgets
                dur_widgets = self.sw_param_widgets["duration"]
                min_dur = dur_widgets["min"].value()
                max_dur = dur_widgets["max"].value()
                
                frequency = (1.0/max_dur, 1.0/min_dur) if min_dur > 0 and max_dur > 0 else (0.5, 1.25)
                # These methods don't use trough_duration
                trough_duration = None
                
                if self.sw_method == "Ngo2015":
                   # Get thresholds - these are in sigma units for adaptive thresholds
                    peak_thresh_sigma = self.sw_param_widgets["peak_thresh"].value()
                    ptp_thresh_sigma = self.sw_param_widgets["ptp_thresh"].value()

                    
                    # These will be overridden by sigma thresholds in the detector
                    neg_peak_thresh = -80.0  # Default value
                    p2p_thresh = 140.0      # Default value
                
                    
                else:  # Staresina2015
                    # Get threshold in μV
                    neg_peak_thresh = -75.0  # Default value 
                    p2p_thresh = self.sw_param_widgets["ptp_thresh"].value()
                    peak_thresh_sigma = None
                    ptp_thresh_sigma = None
            
            else:
                # Default fallback if method is not recognized
                self.write_log(f"Warning: Unknown method '{self.sw_method}', using default parameters")
                frequency = (0.1, 4.0)
                
                # Default to using trough_duration 
                trough_duration = (0.3, 1.0)
                min_dur = None
                max_dur = None
                
                neg_peak_thresh = -80.0
                p2p_thresh = 140.0
                
            # Log the processed parameters
            self.write_log(f"Using method: {self.sw_method}")
            self.write_log(f"Frequency range: {frequency} Hz")
            
            #  Log the appropriate duration parameter based on method 
            if self.sw_method in ["Massimini2004", "AASM/Massimini2004"]:
                self.write_log(f"Trough duration: {trough_duration[0]:.2f}-{trough_duration[1]:.2f} s")
            else:
                self.write_log(f"Duration range: {min_dur:.2f}-{max_dur:.2f} s")
                
            self.write_log(f"Negative peak threshold: {neg_peak_thresh} μV")
            self.write_log(f"Peak-to-peak threshold: {p2p_thresh} μV")

            if self.sw_method == "Ngo2015":
                self.write_log(f"Adaptive peak threshold: {peak_thresh_sigma} σ")
                self.write_log(f"Adaptive peak-to-peak threshold: {ptp_thresh_sigma} σ")

            self.write_log(f"Signal polarity: {polar}")
            
        except Exception as e:
            self.write_log(f"Error processing parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error processing parameters: {str(e)}")
            return
        
        # Store parameters for the detection thread, including method-specific duration params
        self.sw_detection_params = {
            'method': self.sw_method,
            'chan': self.selected_channels,
            'frequency': frequency,
            'neg_peak_thresh': neg_peak_thresh,
            'p2p_thresh': p2p_thresh,
            'polar': polar,
            'reject_artifacts': self.reject_artifacts_check.isChecked(),
            'reject_arousals': self.reject_arousals_check.isChecked(),
            'stage': self.selected_stages
        }
        
        # Add the appropriate duration parameter based on the method 
        if self.sw_method in ["Massimini2004", "AASM/Massimini2004"]:
            self.sw_detection_params['trough_duration'] = trough_duration
        else:
            self.sw_detection_params['min_dur'] = min_dur
            self.sw_detection_params['max_dur'] = max_dur
            # Add method-specific parameters
            if self.sw_method == "Ngo2015":
                self.sw_detection_params['peak_thresh_sigma'] = peak_thresh_sigma
                self.sw_detection_params['ptp_thresh_sigma'] = ptp_thresh_sigma
    
        
        # Start detection
        self.statusBar().showMessage("Detecting slow waves...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        # Disable button
        self.detect_sw_btn.setEnabled(False)
        
        # Log
        self.write_log("Starting slow wave detection...")
        
        # Start thread
        self.sw_thread = threading.Thread(target=self.detect_sw)
        self.sw_thread.daemon = True
        self.sw_thread.start()

    def detect_sw(self):
        """Detect slow waves (runs in a thread)"""
        try:
            # Create a GUI log handler
            gui_log_handler = GUILogHandler(self.write_log)

            data = self.dataset
            # Check if we should use existing annotations or load fresh
            if self.annotations and os.path.isfile(self.annot_file_path):
                annot = self.annotations  # Use existing if available
                self.write_log("Using existing loaded annotations")
            else:
                annot = CustomAnnotations(self.annot_file_path)
                self.annotations = annot  # Store for future use
                self.write_log(f"Loaded annotation file: {self.annot_file_path}")

            # Create sw results directory
            json_dir = os.path.join(self.output_dir, "wonambi", "sw_results")
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
                self.write_log(f"Created directory: {json_dir}")

            # Create ParalSWA instance
            event_processor = ParalSWA(dataset=data, annotations=annot,
                                       log_level=logging.INFO, log_file=None) 
            
            event_processor.logger.addHandler(gui_log_handler)
            
            # Get parameters from the thread preparation
            params = self.sw_detection_params.copy()

            # Detect slow waves
            self.write_log(f"Calling detect_slow_waves with method={params['method']}")
            self.write_log(f"Using {len(params['chan'])} channels")
            self.write_log(f"Sleep stages: {', '.join(params['stage'])}")

            # Pass the appropriate duration parameters based on the method 
            detect_kwargs = {k: v for k, v in params.items() if k not in ['min_dur', 'max_dur', 
                                                                          'trough_duration', 'peak_thresh_sigma', 
                                                                          'ptp_thresh_sigma']}
            # Method-specific parameters
            if params['method'] in ["Massimini2004", "AASM/Massimini2004"]:
                # For Massimini methods, use trough_duration
                self.write_log(f"Using trough_duration: {params['trough_duration']}")
                detect_kwargs['trough_duration'] = params['trough_duration']
            else:
                
                # For Ngo2015 and Staresina2015, use min_dur and max_dur
                self.write_log(f"Using min_dur: {params['min_dur']} and max_dur: {params['max_dur']}")
                detect_kwargs['min_dur'] = params['min_dur']
                detect_kwargs['max_dur'] = params['max_dur']
                if params['method'] == "Ngo2015" and 'peak_thresh_sigma' in params and 'ptp_thresh_sigma' in params:
                    self.write_log(f"Using adaptive thresholds: peak_thresh_sigma={params['peak_thresh_sigma']}, ptp_thresh_sigma={params['ptp_thresh_sigma']}")
                    detect_kwargs['peak_thresh_sigma'] = params['peak_thresh_sigma']
                    detect_kwargs['ptp_thresh_sigma'] = params['ptp_thresh_sigma']
            
            detect_kwargs['cat'] = (1, 1, 1, 0)  # concatenate within and between stages, cycles separate
            self.write_log("Using cat=(1, 1, 1, 0) for event concatenation")

            detect_kwargs['json_dir'] = json_dir  # Added parameter
            detect_kwargs['save_to_annotations'] = False  # Added parameter

            slow_waves = event_processor.detect_slow_waves(**detect_kwargs)

            
            # Export results
            # Format names for export
            freq_range_str = f"{params['frequency'][0]}-{params['frequency'][1]}Hz"
            stages_str = "".join(params['stage'])
            file_pattern = f"slowwaves_{params['method']}_{freq_range_str}_{stages_str}"

            # Export parameters to CSV
            param_csv = os.path.join(json_dir, f'sw_parameters_{params["method"]}_{freq_range_str}_{stages_str}.csv')
            self.write_log(f"Exporting parameters to {param_csv}")
            event_processor.export_slow_wave_parameters_to_csv(
                json_input=json_dir,
                csv_file=param_csv,
                frequency=params['frequency'],
                file_pattern=file_pattern
            )
            
            # Initialize and update database
            # Add database functionality
            db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
            self.write_log(f"Initializing/updating database at {db_path}")
            event_processor.initialize_sqlite_database(db_path)
            self.write_log(f"Importing slow wave parameters to database")
            stats = event_processor.import_parameters_csv_to_database(param_csv, db_path)
            self.write_log(f"Database update complete: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} skipped")

            # Export density to CSV
            density_csv = os.path.join(json_dir, f'sw_density_{params["method"]}_{freq_range_str}_{stages_str}.csv')
            self.write_log(f"Exporting density to {density_csv}")
            event_processor.export_slow_wave_density_to_csv(
                json_input=json_dir,
                csv_file=density_csv,
                stage=params['stage'],
                file_pattern=file_pattern
            )
            
            self.write_log(f"Slow wave parameters saved to {param_csv}")
            self.write_log(f"Slow wave parameters saved to {db_path}")
            self.write_log(f"Slow wave density saved to {density_csv}")
            self.write_log("Slow wave detection completed successfully")

            try:
                # Prepare parameters summary
                parameters_summary = {
                    'method': params['method'],
                    'frequency_range': params['frequency'],
                    'channels': params['chan'],
                    'stages': params['stage'],
                    'polar': params['polar'],
                    'reject_artifacts': params['reject_artifacts'],
                    'reject_arousals': params['reject_arousals']
                }
                
                # Add method-specific duration parameters
                if params['method'] in ["Massimini2004", "AASM/Massimini2004"]:
                    parameters_summary['trough_duration'] = params.get('trough_duration')
                else:
                    parameters_summary['min_dur'] = params.get('min_dur')
                    parameters_summary['max_dur'] = params.get('max_dur')
                    if params['method'] == "Ngo2015":
                        parameters_summary['peak_thresh_sigma'] = params.get('peak_thresh_sigma')
                        parameters_summary['ptp_thresh_sigma'] = params.get('ptp_thresh_sigma')
                
                # Prepare results summary
                results_summary = {
                    'total_slow_waves_detected': len(slow_waves) if 'slow_waves' in locals() else 0,
                    'channels_processed': len(params['chan']),
                    'csv_file': param_csv,
                    'density_file': density_csv,
                    'database_file': db_path
                }
                
                # Save detection summary
                event_processor.save_detection_summary(
                    output_dir=json_dir,
                    method=params['method'],
                    parameters=parameters_summary,
                    results_summary=results_summary
                )
                
            except Exception as e:
                self.write_log(f"Note: Could not save detection summary: {e}")

            QtCore.QMetaObject.invokeMethod(
                self, "finish_sw_detection", 
                QtCore.Qt.QueuedConnection
            )
            
        except Exception as e:
            self.write_log(f"Error detecting slow waves: {str(e)}")
            import traceback
            traceback.print_exc()
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Failed to detect slow waves: {str(e)}")
            )
            
            # Re-enable button in main thread
            QtCore.QMetaObject.invokeMethod(
                self.detect_sw_btn, "setEnabled", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True)
            )
            
            QtCore.QMetaObject.invokeMethod(
                self.progress, "setVisible", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False)
            )

    def setup_spindle_tab(self):
        # Main layout
        layout = QVBoxLayout(self.spindle_tab)
        
        # Top section split into two columns
        top_splitter = QSplitter(QtCore.Qt.Horizontal)
        
        # Left column - parameters
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Parameters group
        params_group = QGroupBox("Spindle Detection Parameters")
        self.spindle_params_form = QVBoxLayout()
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Detection Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Moelle2011", "Ferrarelli2007", "Lacourse2018","Ray2015","Martin2013","Wamsley2012","Nir2011"])
        method_layout.addWidget(self.method_combo)
        self.method_combo.currentTextChanged.connect(self.update_spindle_params_for_method)
        self.spindle_params_form.addLayout(method_layout)

        self.spindle_params_container = QGroupBox("Method-Specific Parameters")
        self.spindle_params_layout = QVBoxLayout(self.spindle_params_container)
        # Add it to the form layout right after the method selection
        self.spindle_params_form.addWidget(self.spindle_params_container)
        
        # Frequency range
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency Range (Hz):"))
        freq_layout.addWidget(QLabel("Min:"))
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(5, 20)
        self.min_freq_spin.setSingleStep(0.5)
        self.min_freq_spin.setValue(9.0)
        freq_layout.addWidget(self.min_freq_spin)
        
        freq_layout.addWidget(QLabel("Max:"))
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(5, 20)
        self.max_freq_spin.setSingleStep(0.5)
        self.max_freq_spin.setValue(12.0)
        freq_layout.addWidget(self.max_freq_spin)
        self.spindle_params_form.addLayout(freq_layout)
        
        # Duration range
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration Range (s):"))
        dur_layout.addWidget(QLabel("Min:"))
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(0.1, 5)
        self.min_dur_spin.setSingleStep(0.1)
        self.min_dur_spin.setValue(0.5)
        dur_layout.addWidget(self.min_dur_spin)
        
        dur_layout.addWidget(QLabel("Max:"))
        self.max_dur_spin = QDoubleSpinBox()
        self.max_dur_spin.setRange(0.5, 10)
        self.max_dur_spin.setSingleStep(0.1)
        self.max_dur_spin.setValue(3.0)
        dur_layout.addWidget(self.max_dur_spin)
        self.spindle_params_form.addLayout(dur_layout)
        
        # Options
        self.reject_artifacts_check = QCheckBox("Reject Artifacts")
        self.reject_artifacts_check.setChecked(True)
        self.spindle_params_form.addWidget(self.reject_artifacts_check)
        
        self.reject_arousals_check = QCheckBox("Reject Arousals")
        self.reject_arousals_check.setChecked(True)
        self.spindle_params_form.addWidget(self.reject_arousals_check)
        
       # Signal Processing Options
        options_group = QGroupBox("Signal Processing Options")
        options_layout = QHBoxLayout()
        
        self.invert_signal_check = QCheckBox("Invert Signal")
        self.invert_signal_check.setChecked(False)  # Default is normal polarity
        options_layout.addWidget(self.invert_signal_check)
        
        options_group.setLayout(options_layout)
        self.spindle_params_form.addWidget(options_group)

        params_group.setLayout(self.spindle_params_form)
        params_layout.addWidget(params_group)
                
        # Stage selection group
        stages_group = QGroupBox("Sleep Stage Selection")
        stages_layout = QHBoxLayout()
        
        self.stage_checks = {}
        stages = ["NREM1", "NREM2", "NREM3", "REM", "Wake"]
        default_selected = ["NREM2", "NREM3"]
        
        for stage in stages:
            check = QCheckBox(stage)
            check.setChecked(stage in default_selected)
            stages_layout.addWidget(check)
            self.stage_checks[stage] = check
        
        stages_group.setLayout(stages_layout)
        params_layout.addWidget(stages_group)
        
        params_layout.addStretch(1)
        top_splitter.addWidget(params_widget)
        

        
        # Right column - channel selection
        channels_widget = QWidget()
        channels_layout = QVBoxLayout(channels_widget)
        
        channels_group = QGroupBox("Channel Selection")
        channels_content = QHBoxLayout()
        
        # Available channels
        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Available Channels:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        avail_layout.addWidget(self.available_list)
        channels_content.addLayout(avail_layout)
        
       
        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.addStretch(1)
        
        self.add_btn = QPushButton("Add >")
        self.add_btn.clicked.connect(self.add_channels)
        btn_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("< Remove")
        self.remove_btn.clicked.connect(self.remove_channels)
        btn_layout.addWidget(self.remove_btn)
        
        self.add_all_btn = QPushButton("Add All >>")
        self.add_all_btn.clicked.connect(self.add_all_channels)
        btn_layout.addWidget(self.add_all_btn)
        
        self.remove_all_btn = QPushButton("<< Remove All")
        self.remove_all_btn.clicked.connect(self.remove_all_channels)
        btn_layout.addWidget(self.remove_all_btn)
        
        btn_layout.addStretch(1)
        channels_content.addLayout(btn_layout)
        
        # Selected channels
        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected Channels:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(self.selected_list)
        channels_content.addLayout(sel_layout)
        
        channels_group.setLayout(channels_content)
        channels_layout.addWidget(channels_group)
        
        top_splitter.addWidget(channels_widget)
        
        # Add the splitter to the main layout
        layout.addWidget(top_splitter)
        
        # Bottom section - action buttons
        action_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("Detect Spindles")
        self.detect_btn.clicked.connect(self.detect_spindles_thread)
        self.detect_btn.setStyleSheet("font-weight: bold;")
        action_layout.addWidget(self.detect_btn)
        
        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.clicked.connect(self.view_spindle_results)
        self.view_results_btn.setEnabled(False)
        action_layout.addWidget(self.view_results_btn)
        
        layout.addLayout(action_layout)
        
        #Initialize method-specific parameters for the default method
        self.update_spindle_params_for_method(self.method_combo.currentText())
       

    # update spindle parameters based on selected method
    def update_spindle_params_for_method(self, method_name):
        """Update spindle detection parameters based on selected method"""
        # Clear previous parameter widgets
        self.clear_layout(self.spindle_params_layout)

        
        # Import the detector class to access parameters
        try:
            from turtlewave_hdEEG.extensions import ImprovedDetectSpindle
            
            # Create a temporary detector with the selected method to access its parameters
            detector = ImprovedDetectSpindle(method=method_name)
            
            # Display info about the method
            method_descriptions = {
                "Moelle2011": "Detects spindles using bandpass filtering (12-15 Hz) with RMS and thresholding.",
                "Ferrarelli2007": "Uses a bandpass filter (11-15 Hz) followed by amplitude threshold detection.",
                "Nir2011": "Employs a bandpass filter with Hilbert transform and multiple thresholds.",
                "Wamsley2012": "Uses wavelet transform for spindle detection in the 12-15 Hz range.",
                "Martin2013": "Applies a remez filter with moving RMS and percentile thresholding.",
                "Ray2015": "Uses complex demodulation for precise spindle frequency targeting.",
                "Lacourse2018": "Multi-metric approach combining absolute power, relative power, covariance and correlation."
            }
            
            # Add description label
            if method_name in method_descriptions:
                desc_label = QLabel(method_descriptions[method_name])
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: #333; font-style: italic; background-color: #f0f4f7; padding: 8px; border-radius: 4px;")
                self.spindle_params_layout.addWidget(desc_label)
                self.spindle_params_layout.addSpacing(10)
                
                # Log the change
                self.write_log(f"Selected spindle detection method: {method_name}")

            # Create header
            info_label = QLabel("<b>Detection Parameters:</b>")
            info_label.setAlignment(QtCore.Qt.AlignCenter)
            self.spindle_params_layout.addWidget(info_label)
            
            # Initialize dict to store UI elements
            self.spindle_param_widgets = {}
            
            # Create different parameter groups based on method
            if method_name == "Moelle2011":
                # Detection threshold
                thresh_group = QGroupBox("Detection Threshold")
                thresh_layout = QHBoxLayout()
                thresh_layout.addWidget(QLabel("Threshold (σ):"))
                thresh_spin = QDoubleSpinBox()
                thresh_spin.setRange(0.5, 10.0)
                thresh_spin.setSingleStep(0.1)
                thresh_spin.setValue(detector.det_thresh)
                thresh_layout.addWidget(thresh_spin)
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = thresh_spin
                
                # RMS parameters
                rms_group = QGroupBox("RMS Parameters")
                rms_layout = QHBoxLayout()
                rms_layout.addWidget(QLabel("Window Duration (s):"))
                rms_spin = QDoubleSpinBox()
                rms_spin.setRange(0.05, 1.0)
                rms_spin.setSingleStep(0.05)
                rms_spin.setValue(detector.moving_rms['dur'])
                rms_layout.addWidget(rms_spin)
                rms_group.setLayout(rms_layout)
                self.spindle_params_layout.addWidget(rms_group)
                self.spindle_param_widgets["rms_dur"] = rms_spin
                
            elif method_name == "Ferrarelli2007":
                # Detection threshold
                thresh_group = QGroupBox("Thresholds")
                thresh_layout = QVBoxLayout()
                
                det_layout = QHBoxLayout()
                det_layout.addWidget(QLabel("Detection Threshold:"))
                det_thresh_spin = QDoubleSpinBox()
                det_thresh_spin.setRange(1.0, 20.0)
                det_thresh_spin.setSingleStep(0.5)
                det_thresh_spin.setValue(detector.det_thresh)
                det_layout.addWidget(det_thresh_spin)
                thresh_layout.addLayout(det_layout)
                
                sel_layout = QHBoxLayout()
                sel_layout.addWidget(QLabel("Selection Threshold:"))
                sel_thresh_spin = QDoubleSpinBox()
                sel_thresh_spin.setRange(0.5, 10.0)
                sel_thresh_spin.setSingleStep(0.1)
                sel_thresh_spin.setValue(detector.sel_thresh)
                sel_layout.addWidget(sel_thresh_spin)
                thresh_layout.addLayout(sel_layout)
                
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = det_thresh_spin
                self.spindle_param_widgets["sel_thresh"] = sel_thresh_spin
                
            elif method_name == "Nir2011":
                # Detection threshold
                thresh_group = QGroupBox("Thresholds")
                thresh_layout = QVBoxLayout()
                
                det_layout = QHBoxLayout()
                det_layout.addWidget(QLabel("Detection Threshold (σ):"))
                det_thresh_spin = QDoubleSpinBox()
                det_thresh_spin.setRange(1.0, 10.0)
                det_thresh_spin.setSingleStep(0.1)
                det_thresh_spin.setValue(detector.det_thresh)
                det_layout.addWidget(det_thresh_spin)
                thresh_layout.addLayout(det_layout)
                
                sel_layout = QHBoxLayout()
                sel_layout.addWidget(QLabel("Selection Threshold (σ):"))
                sel_thresh_spin = QDoubleSpinBox()
                sel_thresh_spin.setRange(0.5, 5.0)
                sel_thresh_spin.setSingleStep(0.1)
                sel_thresh_spin.setValue(detector.sel_thresh)
                sel_layout.addWidget(sel_thresh_spin)
                thresh_layout.addLayout(sel_layout)
                
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = det_thresh_spin
                self.spindle_param_widgets["sel_thresh"] = sel_thresh_spin
                
                # Tolerance 
                tol_group = QGroupBox("Signal Processing")
                tol_layout = QHBoxLayout()
                tol_layout.addWidget(QLabel("Tolerance (s):"))
                tol_spin = QDoubleSpinBox()
                tol_spin.setRange(0.0, 5.0)
                tol_spin.setSingleStep(0.1)
                tol_spin.setValue(detector.tolerance)
                tol_layout.addWidget(tol_spin)
                tol_group.setLayout(tol_layout)
                self.spindle_params_layout.addWidget(tol_group)
                self.spindle_param_widgets["tolerance"] = tol_spin
                
            elif method_name == "Wamsley2012":
                # Detection threshold
                thresh_group = QGroupBox("Detection Threshold")
                thresh_layout = QHBoxLayout()
                thresh_layout.addWidget(QLabel("Threshold:"))
                thresh_spin = QDoubleSpinBox()
                thresh_spin.setRange(1.0, 10.0)
                thresh_spin.setSingleStep(0.1)
                thresh_spin.setValue(detector.det_thresh)
                thresh_layout.addWidget(thresh_spin)
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = thresh_spin
                
                # Wavelet parameters
                wav_group = QGroupBox("Wavelet Parameters")
                wav_layout = QVBoxLayout()
                
                sd_layout = QHBoxLayout()
                sd_layout.addWidget(QLabel("Standard Deviation:"))
                sd_spin = QDoubleSpinBox()
                sd_spin.setRange(0.1, 5.0)
                sd_spin.setSingleStep(0.1)
                sd_spin.setValue(detector.det_wavelet['sd'])
                sd_layout.addWidget(sd_spin)
                wav_layout.addLayout(sd_layout)
                
                dur_layout = QHBoxLayout()
                dur_layout.addWidget(QLabel("Duration (s):"))
                dur_spin = QDoubleSpinBox()
                dur_spin.setRange(0.1, 5.0)
                dur_spin.setSingleStep(0.1)
                dur_spin.setValue(detector.det_wavelet['dur'])
                dur_layout.addWidget(dur_spin)
                wav_layout.addLayout(dur_layout)
                
                wav_group.setLayout(wav_layout)
                self.spindle_params_layout.addWidget(wav_group)
                self.spindle_param_widgets["wavelet_sd"] = sd_spin
                self.spindle_param_widgets["wavelet_dur"] = dur_spin
                
            elif method_name == "Martin2013":
                # Percentile threshold
                thresh_group = QGroupBox("Detection Threshold")
                thresh_layout = QHBoxLayout()
                thresh_layout.addWidget(QLabel("Percentile:"))
                thresh_spin = QSpinBox()
                thresh_spin.setRange(50, 99)
                thresh_spin.setValue(detector.det_thresh)
                thresh_layout.addWidget(thresh_spin)
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = thresh_spin
                
                # RMS parameters
                rms_group = QGroupBox("RMS Parameters")
                rms_layout = QHBoxLayout()
                rms_layout.addWidget(QLabel("Window Duration (s):"))
                rms_spin = QDoubleSpinBox()
                rms_spin.setRange(0.05, 1.0)
                rms_spin.setSingleStep(0.05)
                rms_spin.setValue(detector.moving_rms['dur'])
                rms_layout.addWidget(rms_spin)
                rms_group.setLayout(rms_layout)
                self.spindle_params_layout.addWidget(rms_group)
                self.spindle_param_widgets["rms_dur"] = rms_spin
                
            elif method_name == "Ray2015":
                # Z-score threshold
                thresh_group = QGroupBox("Thresholds")
                thresh_layout = QVBoxLayout()
                
                det_layout = QHBoxLayout()
                det_layout.addWidget(QLabel("Detection Threshold (Z):"))
                det_thresh_spin = QDoubleSpinBox()
                det_thresh_spin.setRange(0.5, 5.0)
                det_thresh_spin.setSingleStep(0.01)
                det_thresh_spin.setValue(detector.det_thresh)
                det_layout.addWidget(det_thresh_spin)
                thresh_layout.addLayout(det_layout)
                
                sel_layout = QHBoxLayout()
                sel_layout.addWidget(QLabel("Selection Threshold:"))
                sel_thresh_spin = QDoubleSpinBox()
                sel_thresh_spin.setRange(0.01, 1.0)
                sel_thresh_spin.setSingleStep(0.01)
                sel_thresh_spin.setValue(detector.sel_thresh)
                sel_layout.addWidget(sel_thresh_spin)
                thresh_layout.addLayout(sel_layout)
                
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = det_thresh_spin
                self.spindle_param_widgets["sel_thresh"] = sel_thresh_spin
                
                # Z-score window
                zscore_group = QGroupBox("Z-Score Window")
                zscore_layout = QHBoxLayout()
                zscore_layout.addWidget(QLabel("Window Duration (s):"))
                zscore_spin = QDoubleSpinBox()
                zscore_spin.setRange(10.0, 120.0)
                zscore_spin.setSingleStep(10.0)
                zscore_spin.setValue(detector.zscore['dur'])
                zscore_layout.addWidget(zscore_spin)
                zscore_group.setLayout(zscore_layout)
                self.spindle_params_layout.addWidget(zscore_group)
                self.spindle_param_widgets["zscore_dur"] = zscore_spin
                
            elif method_name == "Lacourse2018":
                # Multi-threshold approach
                thresh_group = QGroupBox("Detection Thresholds")
                thresh_layout = QVBoxLayout()
                
                abs_layout = QHBoxLayout()
                abs_layout.addWidget(QLabel("Absolute Power:"))
                abs_thresh_spin = QDoubleSpinBox()
                abs_thresh_spin.setRange(0.5, 5.0)
                abs_thresh_spin.setSingleStep(0.05)
                abs_thresh_spin.setValue(detector.abs_pow_thresh)
                abs_layout.addWidget(abs_thresh_spin)
                thresh_layout.addLayout(abs_layout)
                
                rel_layout = QHBoxLayout()
                rel_layout.addWidget(QLabel("Relative Power:"))
                rel_thresh_spin = QDoubleSpinBox()
                rel_thresh_spin.setRange(0.5, 5.0)
                rel_thresh_spin.setSingleStep(0.05)
                rel_thresh_spin.setValue(detector.rel_pow_thresh)
                rel_layout.addWidget(rel_thresh_spin)
                thresh_layout.addLayout(rel_layout)
                
                covar_layout = QHBoxLayout()
                covar_layout.addWidget(QLabel("Covariance:"))
                covar_thresh_spin = QDoubleSpinBox()
                covar_thresh_spin.setRange(0.5, 5.0)
                covar_thresh_spin.setSingleStep(0.05)
                covar_thresh_spin.setValue(detector.covar_thresh)
                covar_layout.addWidget(covar_thresh_spin)
                thresh_layout.addLayout(covar_layout)
                
                corr_layout = QHBoxLayout()
                corr_layout.addWidget(QLabel("Correlation:"))
                corr_thresh_spin = QDoubleSpinBox()
                corr_thresh_spin.setRange(0.1, 1.0)
                corr_thresh_spin.setSingleStep(0.01)
                corr_thresh_spin.setValue(detector.corr_thresh)
                corr_layout.addWidget(corr_thresh_spin)
                thresh_layout.addLayout(corr_layout)
                
                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["abs_thresh"] = abs_thresh_spin
                self.spindle_param_widgets["rel_thresh"] = rel_thresh_spin
                self.spindle_param_widgets["covar_thresh"] = covar_thresh_spin
                self.spindle_param_widgets["corr_thresh"] = corr_thresh_spin
                
                # Window settings
                window_group = QGroupBox("Window Settings")
                window_layout = QHBoxLayout()
                window_layout.addWidget(QLabel("Window Duration (s):"))
                window_spin = QDoubleSpinBox()
                window_spin.setRange(0.1, 1.0)
                window_spin.setSingleStep(0.05)
                window_spin.setValue(detector.windowing['dur'])
                window_layout.addWidget(window_spin)
                window_group.setLayout(window_layout)
                self.spindle_params_layout.addWidget(window_group)
                self.spindle_param_widgets["window_dur"] = window_spin
            
            else:
                # If method not recognized
                error_label = QLabel(f"Error: Parameters for method '{method_name}' not available.")
                error_label.setStyleSheet("color: red;")
                self.spindle_params_layout.addWidget(error_label)
            
            # Add a spacer at the end
            self.spindle_params_layout.addStretch(1)
            
        except Exception as e:
            # If we can't import or access the detector
            self.write_log(f"Error loading parameters from ImprovedDetectSpindle: {str(e)}")
            import traceback
            traceback.print_exc()
            
            error_label = QLabel(f"Error loading parameters: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.spindle_params_layout.addWidget(error_label)
        
        # Log the change
        self.write_log(f"Updated parameters for {method_name} spindle detection method")






    def setup_pac_tab(self):
        """Setup the Phase-Amplitude Coupling (PAC) analysis tab"""
        # Main layout
        layout = QVBoxLayout(self.pac_tab)
        
        # Top section split into two columns
        top_splitter = QSplitter(QtCore.Qt.Horizontal)
        
        # Left column - parameters
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Method selection group
        method_group = QGroupBox("PAC Analysis Method")
        method_layout = QVBoxLayout()
        
        # PAC method selection
        method_select_layout = QHBoxLayout()
        method_select_layout.addWidget(QLabel("PAC Type:"))
        self.pac_method_combo = QComboBox()
        self.pac_method_combo.addItems(["SW-Spindle", "Theta-Gamma"])
        method_select_layout.addWidget(self.pac_method_combo)
        method_layout.addLayout(method_select_layout)
        
        # Connect method change to update parameters
        self.pac_method_combo.currentTextChanged.connect(self.update_pac_params)
        
        method_group.setLayout(method_layout)
        params_layout.addWidget(method_group)
        
        # Event selection group
        event_group = QGroupBox("Event Selection")
        event_layout = QVBoxLayout()
        
        
        
        # SW method selection (for SW-Spindle coupling)
        sw_method_layout = QHBoxLayout()
        sw_method_layout.addWidget(QLabel("Slow Wave Method:"))
        self.sw_method_pac_combo = QComboBox()
        # Will be populated from database with method, freq range, and stage
        sw_method_layout.addWidget(self.sw_method_pac_combo)
        event_layout.addLayout(sw_method_layout)
        
        # Spindle method selection (for SW-Spindle coupling)
        spindle_method_layout = QHBoxLayout()
        spindle_method_layout.addWidget(QLabel("Spindle Method:"))
        self.spindle_method_pac_combo = QComboBox()
        # Will be populated from database with method, freq range, and stage
        spindle_method_layout.addWidget(self.spindle_method_pac_combo)
        event_layout.addLayout(spindle_method_layout)
        
   
        # Time window
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Time Window (s):"))
        self.time_window_spin = QDoubleSpinBox()
        self.time_window_spin.setRange(0.1, 10)
        self.time_window_spin.setSingleStep(0.1)
        self.time_window_spin.setValue(1.0)  # Default
        window_layout.addWidget(self.time_window_spin)
        event_layout.addLayout(window_layout)
        
        event_group.setLayout(event_layout)
        params_layout.addWidget(event_group)
        
        # For Theta-Gamma, manual frequency input
        # Create a stacked widget to switch between SW-Spindle and Theta-Gamma parameters
        self.pac_param_stack = QtWidgets.QStackedWidget()
        
        # Theta-Gamma frequency parameters widget
        theta_gamma_widget = QWidget()
        theta_gamma_layout = QVBoxLayout(theta_gamma_widget)
        
        theta_gamma_group = QGroupBox("Theta-Gamma Frequency Parameters")
        theta_gamma_form = QVBoxLayout()
        
        # Phase frequency range (theta)
        theta_layout = QHBoxLayout()
        theta_layout.addWidget(QLabel("Theta Frequency (Hz):"))
        theta_layout.addWidget(QLabel("Min:"))
        self.theta_min_spin = QDoubleSpinBox()
        self.theta_min_spin.setRange(3, 10)
        self.theta_min_spin.setSingleStep(0.5)
        self.theta_min_spin.setValue(4)  # Default for theta
        theta_layout.addWidget(self.theta_min_spin)
        
        theta_layout.addWidget(QLabel("Max:"))
        self.theta_max_spin = QDoubleSpinBox()
        self.theta_max_spin.setRange(3, 10)
        self.theta_max_spin.setSingleStep(0.5)
        self.theta_max_spin.setValue(8)  # Default for theta
        theta_layout.addWidget(self.theta_max_spin)
        theta_gamma_form.addLayout(theta_layout)
        
        # Amplitude frequency range (gamma)
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma Frequency (Hz):"))
        gamma_layout.addWidget(QLabel("Min:"))
        self.gamma_min_spin = QDoubleSpinBox()
        self.gamma_min_spin.setRange(30, 150)
        self.gamma_min_spin.setSingleStep(5)
        self.gamma_min_spin.setValue(30)  # Default for gamma
        gamma_layout.addWidget(self.gamma_min_spin)
        
        gamma_layout.addWidget(QLabel("Max:"))
        self.gamma_max_spin = QDoubleSpinBox()
        self.gamma_max_spin.setRange(30, 150)
        self.gamma_max_spin.setSingleStep(5)
        self.gamma_max_spin.setValue(80)  # Default for gamma
        gamma_layout.addWidget(self.gamma_max_spin)
        theta_gamma_form.addLayout(gamma_layout)
        

       # Add sleep stage selection for Theta-Gamma only
        stages_layout = QHBoxLayout()
        stages_layout.addWidget(QLabel("Sleep Stages:"))
        self.pac_stage_checks = {}
        stages = ["NREM1", "NREM2", "NREM3", "REM", "Wake"]
        default_selected = ["NREM2", "NREM3"]
        
        for stage in stages:
            check = QCheckBox(stage)
            check.setChecked(stage in default_selected)
            stages_layout.addWidget(check)
            self.pac_stage_checks[stage] = check
        
        theta_gamma_form.addLayout(stages_layout)


        theta_gamma_group.setLayout(theta_gamma_form)
        theta_gamma_layout.addWidget(theta_gamma_group)
        theta_gamma_layout.addStretch(1)


        # Placeholder widget for SW-Spindle (since we get frequencies from DB)
        sw_spindle_widget = QWidget()
        
        # Add widgets to stacked widget
        self.pac_param_stack.addWidget(sw_spindle_widget)  # Index 0 for SW-Spindle
        self.pac_param_stack.addWidget(theta_gamma_widget)  # Index 1 for Theta-Gamma
        
        params_layout.addWidget(self.pac_param_stack)


        # Advanced options (collapsible)
        advanced_group = QGroupBox("Advanced Options")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)  # Start collapsed
        advanced_layout = QVBoxLayout()
        
        # PAC method settings
        idpac_layout = QHBoxLayout()
        idpac_layout.addWidget(QLabel("PAC Method:"))
        self.idpac_method_combo = QComboBox()
        self.idpac_method_combo.addItems(["Mean over time (1)", "Mean over trials (2)", "Direct (3)"])
        self.idpac_method_combo.setCurrentIndex(0)  # Default
        idpac_layout.addWidget(self.idpac_method_combo)
        advanced_layout.addLayout(idpac_layout)
        
        # Surrogate method
        surrogate_layout = QHBoxLayout()
        surrogate_layout.addWidget(QLabel("Surrogate Method:"))
        self.surrogate_method_combo = QComboBox()
        self.surrogate_method_combo.addItems(["No surrogates (0)", "Swap phase/amp (1)", "Time lag (2)", "Blocks (3)"])
        self.surrogate_method_combo.setCurrentIndex(2)  # Default to time lag
        surrogate_layout.addWidget(self.surrogate_method_combo)
        advanced_layout.addLayout(surrogate_layout)
        
        # Correction method
        correction_layout = QHBoxLayout()
        correction_layout.addWidget(QLabel("Correction Method:"))
        self.correction_method_combo = QComboBox()
        self.correction_method_combo.addItems(["No correction (0)", "Substract mean (1)", "Divide by mean (2)", "Substract then divide (3)", "Z-score (4)"])
        self.correction_method_combo.setCurrentIndex(4)  # Default to Z-score
        correction_layout.addWidget(self.correction_method_combo)
        advanced_layout.addLayout(correction_layout)
        
        advanced_group.setLayout(advanced_layout)
        params_layout.addWidget(advanced_group)
        
        params_layout.addStretch(1)
        top_splitter.addWidget(params_widget)
        
        # Right column - channel selection
        channels_widget = QWidget()
        channels_layout = QVBoxLayout(channels_widget)
        
        channels_group = QGroupBox("Channel Selection")
        channels_content = QVBoxLayout()
        
        # Available channels
        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Available Channels:"))
        self.pac_available_list = QListWidget()
        self.pac_available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        avail_layout.addWidget(self.pac_available_list)
        channels_content.addLayout(avail_layout)
        
        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.addStretch(1)
        
        self.pac_add_btn = QPushButton("Add >")
        self.pac_add_btn.clicked.connect(self.add_pac_channels)
        btn_layout.addWidget(self.pac_add_btn)
        
        self.pac_remove_btn = QPushButton("< Remove")
        self.pac_remove_btn.clicked.connect(self.remove_pac_channels)
        btn_layout.addWidget(self.pac_remove_btn)
        
        self.pac_add_all_btn = QPushButton("Add All >>")
        self.pac_add_all_btn.clicked.connect(self.add_all_pac_channels)
        btn_layout.addWidget(self.pac_add_all_btn)
        
        self.pac_remove_all_btn = QPushButton("<< Remove All")
        self.pac_remove_all_btn.clicked.connect(self.remove_all_pac_channels)
        btn_layout.addWidget(self.pac_remove_all_btn)
        
        btn_layout.addStretch(1)
        channels_content.addLayout(btn_layout)
        
        # Selected channels
        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected Channels:"))
        self.pac_selected_list = QListWidget()
        self.pac_selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(self.pac_selected_list)
        channels_content.addLayout(sel_layout)
        
        channels_group.setLayout(channels_content)
        channels_layout.addWidget(channels_group)
        
        top_splitter.addWidget(channels_widget)
        
        # Add the splitter to the main layout
        layout.addWidget(top_splitter)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.run_pac_btn = QPushButton("Run PAC Analysis")
        self.run_pac_btn.clicked.connect(self.run_pac_analysis_thread)
        self.run_pac_btn.setStyleSheet("font-weight: bold;")
        action_layout.addWidget(self.run_pac_btn)
        
        self.view_pac_results_btn = QPushButton("View Results")
        self.view_pac_results_btn.clicked.connect(self.view_pac_results)
        self.view_pac_results_btn.setEnabled(False)
        action_layout.addWidget(self.view_pac_results_btn)
        
        # self.export_pac_btn = QPushButton("Export Results")
        # self.export_pac_btn.clicked.connect(self.export_pac_results)
        # self.export_pac_btn.setEnabled(False)
        # action_layout.addWidget(self.export_pac_btn)
        
        layout.addLayout(action_layout)

        # Initialize selected channels list
        self.pac_selected_channels = []
    
    def setup_log_tab(self):
        # Main layout
        layout = QVBoxLayout(self.log_tab)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Flush any buffered log messages
        if hasattr(self, '_log_buffer'):
            for message in self._log_buffer:
                self.log_text.append(message)
            del self._log_buffer

        # Clear button
        clear_layout = QHBoxLayout()
        clear_layout.addStretch(1)
        
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        clear_layout.addWidget(self.clear_log_btn)
        
        layout.addLayout(clear_layout)
    
    def browse_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EEG Data File", "", 
            "EEG Files (*.set *.edf *.bdf);;All Files (*)"
        )
        if file_path:
            self.data_file_path = file_path
            self.data_file_edit.setText(file_path)
            
            # Set default output directory
            default_output = os.path.dirname(file_path)
            self.output_dir = default_output
            self.output_dir_edit.setText(default_output)
    
    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
    
    def browse_annot_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", "", 
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.annot_file_path = file_path
            self.annot_file_edit.setText(file_path)
    
    def load_data_thread(self):
        """Start data loading in a separate thread"""
        # Check if file exists
        if not self.data_file_path or not os.path.isfile(self.data_file_path):
            QMessageBox.critical(self, "Error", "Please select a valid EEG data file.")
            return
        
        # Check if output directory exists
        if not self.output_dir or not os.path.isdir(self.output_dir):
            QMessageBox.critical(self, "Error", "Please select a valid output directory.")
            return
        
        # Update from the UI
        self.data_file_path = self.data_file_edit.text()
        self.output_dir = self.output_dir_edit.text()
        self.annot_file_path = self.annot_file_edit.text()
        
        # Disable load button
        self.load_btn.setEnabled(False)
        self.statusBar().showMessage("Loading data...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        # Log
        self.write_log("Loading EEG dataset...")
        
        # Start thread
        self.load_thread = threading.Thread(target=self.load_data)
        self.load_thread.daemon = True
        self.load_thread.start()
    
    def load_data(self):
        """Load the EEG dataset (runs in a thread)"""
        try:
            # Load dataset
            self.dataset = LargeDataset(self.data_file_path, create_memmap=False)
            
            # Create wonambi dir if it doesn't exist
            wonambi_dir = os.path.join(self.output_dir, "wonambi")
            if not os.path.exists(wonambi_dir):
                os.makedirs(wonambi_dir)
                self.write_log(f"Created directory: {wonambi_dir}")
            
            # Set default annotation file if not provided
            if not self.annot_file_path:
                base_name = os.path.splitext(os.path.basename(self.data_file_path))[0]
                self.annot_file_path = os.path.join(wonambi_dir, base_name + ".xml")
                QtCore.QMetaObject.invokeMethod(
                    self.annot_file_edit, "setText", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, self.annot_file_path)
                )
                self.write_log(f"Set default annotation file: {self.annot_file_path}")
            
            # Get available channels
            self.available_channels = self.dataset.channels
            
            # Update GUI in the main thread
            QtCore.QMetaObject.invokeMethod(
                self, "update_after_load", 
                QtCore.Qt.QueuedConnection
            )
            
            self.write_log(f"Successfully loaded dataset: {os.path.basename(self.data_file_path)}")
        
        except Exception as e:
            self.write_log(f"Error loading dataset: {str(e)}")
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Failed to load dataset: {str(e)}")
            )
        
        # Update UI in main thread
        QtCore.QMetaObject.invokeMethod(
            self, "finish_loading", 
            QtCore.Qt.QueuedConnection
        )
    
    @QtCore.pyqtSlot()
    def update_after_load(self):
        """Update UI after dataset is loaded"""
        # Update dataset info
        self.update_dataset_info()
        
        # Enable tabs
        self.tabs.setTabEnabled(1, True)  # Annotation tab
        self.tabs.setTabEnabled(2, True)  # Spindle tab
        self.tabs.setTabEnabled(3, True)  # SW tab
        self.tabs.setTabEnabled(4, True)  # PAC ta

        # Enable review button
        if hasattr(self, 'review_btn'):
            self.review_btn.setEnabled(True)


        # Update channel list
        self.update_channel_lists()
        
        # Populate detection methods from database if it exists
        self.populate_detection_methods()

        # Update status
        self.statusBar().showMessage("Data loaded successfully")
    
    @QtCore.pyqtSlot()
    def finish_loading(self):
        """Clean up after loading finishes"""
        self.load_btn.setEnabled(True)
        self.progress.setVisible(False)

        # Check for existing database
        db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
        if os.path.exists(db_path):
            self.write_log(f"Found existing neural events database: {db_path}")
         

    
    @QtCore.pyqtSlot(str)
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage("Error")
    
    def update_dataset_info(self):
        """Update dataset information display"""
        import datetime
        if self.dataset:
            try:
                n_channels = len(self.dataset.channels)
                sampling_rate = self.dataset.sampling_rate
                n_samples = self.dataset.header['n_samples']
                start_time = self.dataset.header['start_time']
                total_duration = n_samples / sampling_rate
                end_time = start_time + datetime.timedelta(seconds=total_duration)

                # Format info text
                info = (
                    f"Dataset Information:\n"
                    f"File: {os.path.basename(self.data_file_path)}\n"
                    f"Number of Channels: {n_channels}\n"
                    f"Sampling Rate: {sampling_rate} Hz\n"
                    f"Recording Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Recording End Time:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n"
                    f"Output Directory: {self.output_dir}\n"
                    f"Annotation File: {self.annot_file_path}\n\n"
                    f"Channels: {', '.join(self.dataset.channels[:10])}... (and {n_channels-10} more)"
                )
                
                self.info_text.setText(info)
            
            except Exception as e:
                self.write_log(f"Error getting dataset info: {str(e)}")
    
    def update_channel_lists(self):
        """Update channel selection listboxes"""
        # Clear spindle tab listboxes
        self.available_list.clear()
        self.selected_list.clear()
        
        # Clear SW tab listboxes if they exist (they might be created after this is called)
        if hasattr(self, 'sw_available_list') and self.sw_available_list is not None:
            self.sw_available_list.clear()
        if hasattr(self, 'sw_selected_list') and self.sw_selected_list is not None:
            self.sw_selected_list.clear()

        # eeg_channels = []
        # for channel in self.available_channels:
        #     if (channel.startswith('E') and len(channel) > 1 and channel[1:].isdigit()) or channel == 'Cz':
        #         eeg_channels.append(channel)
            
        # # If no EEG channels found, use all available channels
        # if not eeg_channels:
        #     eeg_channels = self.available_channels.copy()
        #     self.write_log(f"No specific EEG channels found. Including all {len(eeg_channels)} channels.")
        
        eeg_channels = self.available_channels.copy()
        # Filter selected channels to keep only EEG channels
        # self.selected_channels = [ch for ch in self.selected_channels if ch in eeg_channels]

        # Add available channels to spindle tab
        for channel in eeg_channels:
            if channel not in self.selected_channels:
                self.available_list.addItem(channel)
                # Also add to SW tab if it exists
                if hasattr(self, 'sw_available_list') and self.sw_available_list is not None:
                    self.sw_available_list.addItem(channel)
        
        # Add selected channels to spindle tab
        for channel in self.selected_channels:
            self.selected_list.addItem(channel)
            # Also add to SW tab if it exists
            if hasattr(self, 'sw_selected_list') and self.sw_selected_list is not None:
                self.sw_selected_list.addItem(channel)
    
    def add_channels(self):
        """Add selected channels to the selected list"""
        selected_items = self.available_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Add to selected channels
        for channel in selected:
            if channel not in self.selected_channels:
                self.selected_channels.append(channel)
        
        # Update listboxes
        self.update_channel_lists()
    
    def remove_channels(self):
        """Remove selected channels from the selected list"""
        selected_items = self.selected_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Remove from selected channels
        self.selected_channels = [ch for ch in self.selected_channels if ch not in selected]
        
        # Update listboxes
        self.update_channel_lists()
    
    def add_all_channels(self):
        """Add all channels to the selected list"""
        self.selected_channels = list(self.available_channels)
        self.update_channel_lists()
    
    def remove_all_channels(self):
        """Remove all channels from the selected list"""
        self.selected_channels = []
        self.update_channel_lists()
    

    def add_sw_channels(self):
        """Add selected channels to the SW selected list"""
        selected_items = self.sw_available_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Add to selected channels
        for channel in selected:
            if channel not in self.selected_channels:
                self.selected_channels.append(channel)
        
        # Update listboxes
        self.update_channel_lists()

    def remove_sw_channels(self):
        """Remove selected channels from the SW selected list"""
        selected_items = self.sw_selected_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Remove from selected channels
        self.selected_channels = [ch for ch in self.selected_channels if ch not in selected]
        
        # Update listboxes
        self.update_channel_lists()

    def add_all_sw_channels(self):
        """Add all channels to the SW selected list"""
        self.selected_channels = list(self.available_channels)
        self.update_channel_lists()

    def remove_all_sw_channels(self):
        """Remove all channels from the SW selected list"""
        self.selected_channels = []
        self.update_channel_lists()

    # Add channel selection methods for PAC
    def add_pac_channels(self):
        """Add selected channels to the PAC selected list"""
        selected_items = self.pac_available_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Add to selected channels
        for channel in selected:
            if channel not in self.pac_selected_channels:
                self.pac_selected_channels.append(channel)
        
        # Update listboxes
        self.update_pac_channel_lists()

    def remove_pac_channels(self):
        """Remove selected channels from the PAC selected list"""
        selected_items = self.pac_selected_list.selectedItems()
        if not selected_items:
            return
        
        # Get selected channels
        selected = [item.text() for item in selected_items]
        
        # Remove from selected channels
        self.pac_selected_channels = [ch for ch in self.pac_selected_channels if ch not in selected]
        
        # Update listboxes
        self.update_pac_channel_lists()

    def add_all_pac_channels(self):
        """Add all available channels to the PAC selected list"""
        if hasattr(self, 'pac_available_channels'):
            self.pac_selected_channels = list(self.pac_available_channels)
            self.update_pac_channel_lists()

    def remove_all_pac_channels(self):
        """Remove all channels from the PAC selected list"""
        self.pac_selected_channels = []
        self.update_pac_channel_lists()

    def update_pac_channel_lists(self):
        """Update PAC channel selection listboxes"""
        # Clear listboxes
        self.pac_available_list.clear()
        self.pac_selected_list.clear()
        
        # Add available channels
        if hasattr(self, 'pac_available_channels'):
            for channel in self.pac_available_channels:
                if channel not in self.pac_selected_channels:
                    self.pac_available_list.addItem(channel)
        
        # Add selected channels
        for channel in self.pac_selected_channels:
            self.pac_selected_list.addItem(channel)


    
    def update_pac_params(self, method_name):
        """Use stacked widget to switch between different parameter sets"""
        if method_name == "SW-Spindle":
           # Show SW-Spindle parameter page (no manual frequency input)
            self.pac_param_stack.setCurrentIndex(0)
        
            # Enable SW and spindle method selection
            self.sw_method_pac_combo.setEnabled(True)
            self.spindle_method_pac_combo.setEnabled(True)
            
            # Update frequencies from current selections
            if self.sw_method_pac_combo.count() > 0:
                self.update_sw_freq_from_db(self.sw_method_pac_combo.currentText())
            if self.spindle_method_pac_combo.count() > 0:
                self.update_spindle_freq_from_db(self.spindle_method_pac_combo.currentText())
            
            self.update_pac_available_channels()
        
        elif method_name == "Theta-Gamma":
            # Set defaults for Theta-Gamma coupling
            self.pac_param_stack.setCurrentIndex(1)
            
            # Disable SW and spindle method selection
            self.sw_method_pac_combo.setEnabled(False)
            self.spindle_method_pac_combo.setEnabled(False)
            
            # For Theta-Gamma, clear channel lists and show future implementation message
            self.pac_available_channels = []
            self.pac_selected_channels = []
            self.update_pac_channel_lists()
            
            # Add a placeholder item to indicate future implementation
            self.pac_available_list.addItem("-- Theta-Gamma PAC: Future Implementation --")
            self.pac_available_list.setSelectionMode(QAbstractItemView.NoSelection)
            
            self.write_log("Theta-Gamma PAC channel selection will be implemented in a future update")

    # update frequency ranges from database
    def update_sw_freq_from_db(self, display_name):
        """Update slow wave frequency range from database based on selected method"""
        if not display_name:
            self.sw_freq_label.setText("Not selected")
            return
        
        try:
            if hasattr(self, 'sw_methods_info') and display_name in self.sw_methods_info:
                # Get frequency range from stored info
                freq_range = self.sw_methods_info[display_name]['freq_range']
                
                # Store for later use in PAC analysis
                self.sw_freq_range = freq_range
            
                # Update available channels based on selection
                self.update_pac_available_channels()


            else:
                # Fall back to database query if method info not found
                # This is a fallback and shouldn't normally be needed
                method = display_name.split(" (")[0] if " (" in display_name else display_name
                
                db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
                if os.path.exists(db_path):
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "SELECT freq_lower, freq_upper FROM events WHERE event_type = 'slow_wave' AND method = ? LIMIT 1", 
                        (method,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        freq_lower, freq_upper = result
                        self.sw_freq_range = (freq_lower, freq_upper)
                        
                    else:
                        self.sw_freq_range = (0.5, 1.25)  # Default
                    
                    conn.close()
                else:
                    self.sw_freq_range = (0.5, 1.25)  # Default
        
        except Exception as e:
            self.write_log(f"Error getting SW frequency: {str(e)}")
            self.sw_freq_range = (0.5, 1.25)  # Default

    def update_spindle_freq_from_db(self, display_name):
        """Update spindle frequency range from database based on selected method"""
        if not display_name:
            self.spindle_freq_label.setText("Not selected")
            return
        
        try:
            if hasattr(self, 'spindle_methods_info') and display_name in self.spindle_methods_info:
                # Get frequency range from stored info
                freq_range = self.spindle_methods_info[display_name]['freq_range']
                
                # Store for later use in PAC analysis
                self.spindle_freq_range = freq_range
                
                self.update_pac_available_channels()
            else:
                # Fall back to database query if method info not found
                method = display_name.split(" (")[0] if " (" in display_name else display_name
                
                db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
                if os.path.exists(db_path):
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "SELECT freq_lower, freq_upper FROM events WHERE event_type = 'spindle' AND method = ? LIMIT 1", 
                        (method,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        freq_lower, freq_upper = result
                        self.spindle_freq_range = (freq_lower, freq_upper)
                        
                    else:
                        self.spindle_freq_range = (11, 16)  # Default
                    
                    conn.close()
                else:
                    self.spindle_freq_range = (11, 16)  # Default
        
        except Exception as e:
            self.write_log(f"Error getting spindle frequency: {str(e)}")
            self.spindle_freq_range = (11, 16)  # Default

    # Update method to get channels based on selected SW and spindle methods
    def update_pac_available_channels(self):
        """Update available channels for PAC analysis based on selected SW and Spindle methods"""
        if self.pac_method_combo.currentText() != "SW-Spindle":
            return  # Only relevant for SW-Spindle coupling
        
        # Get selected methods
        sw_method = self.sw_method_pac_combo.currentText()
        spindle_method = self.spindle_method_pac_combo.currentText()
        
        if not sw_method or not spindle_method:
            return  # Need both methods selected
        
        try:
            # Extract method info from display names
            sw_info = self.sw_methods_info.get(sw_method, {})
            spindle_info = self.spindle_methods_info.get(spindle_method, {})
            
            if not sw_info or not spindle_info:
                self.write_log("Could not find method information")
                return
            
            # Get method parameters
            sw_base_method = sw_info.get('method')
            sw_stage = sw_info.get('stage')
            spindle_base_method = spindle_info.get('method')
            spindle_stage = spindle_info.get('stage')
            
            if not sw_base_method or not spindle_base_method or not sw_stage or not spindle_stage:
                self.write_log("Missing method parameters")
                return
            
            # Check if stages match
            if sw_stage != spindle_stage:
                self.write_log(f"Warning: Sleep stages don't match - SW: {sw_stage}, Spindle: {spindle_stage}")
            
            # Query database for channels that have both slow waves and spindles with these methods and stages
            db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
            if not os.path.exists(db_path):
                self.write_log("Database not found. Cannot update available channels.")
                return
            
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Find channels with both SW and spindles for the selected methods and stages
            query = """
                SELECT DISTINCT sw.channel 
                FROM events sw
                JOIN events sp ON sw.channel = sp.channel AND sw.stage = sp.stage
                WHERE sw.event_type = 'slow_wave' AND sw.method = ? AND sw.stage = ?
                AND sp.event_type = 'spindle' AND sp.method = ? AND sp.stage = ?
                ORDER BY sw.channel
            """
            
            cursor.execute(query, (sw_base_method, sw_stage, spindle_base_method, spindle_stage))
            
            # Get matching channels
            matching_channels = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if matching_channels:
                #self.write_log(f"Found {len(matching_channels)} channels with both {sw_base_method} slow waves and {spindle_base_method} spindles in {sw_stage}")
                
                # Store and update available channels
                self.pac_available_channels = matching_channels
                
                # Update UI
                self.update_pac_channel_lists()
            else:
                #self.write_log(f"No channels found with both {sw_base_method} slow waves and {spindle_base_method} spindles in {sw_stage}")
                self.pac_available_channels = []
                self.update_pac_channel_lists()
        
        except Exception as e:
            self.write_log(f"Error updating available channels: {str(e)}")
            import traceback
            traceback.print_exc()


    def update_channel_selection_mode(self, mode):
        """Update channel selection UI based on the selected mode"""
        # Clear current selection
        self.pac_channel_list.clear()
        self.pac_selected_channels_label.setText("None")
        
        if mode == "Single Channel":
            self.pac_channel_list.setSelectionMode(QAbstractItemView.SingleSelection)
            # Add all channels from database
            self.populate_pac_channels()
        
        elif mode == "All Channels":
            self.pac_channel_list.setSelectionMode(QAbstractItemView.NoSelection)
            self.pac_selected_channels_label.setText("All Available Channels")
        
        elif mode == "Region Based":
            self.pac_channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
            # Add channel regions instead of individual channels
            regions = ["Frontal", "Central", "Parietal", "Temporal", "Occipital"]
            for region in regions:
                self.pac_channel_list.addItem(region)

    def populate_pac_channels(self):
        """Populate channel list for PAC analysis from database"""
        if not hasattr(self, 'database_channels'):
            # Get channels from database
            try:
                db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
                if os.path.exists(db_path):
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT channel FROM events ORDER BY channel")
                    self.database_channels = [row[0] for row in cursor.fetchall()]
                    conn.close()
                else:
                    self.database_channels = []
                    self.write_log("Database not found. No channels available for PAC analysis.")
            except Exception as e:
                self.database_channels = []
                self.write_log(f"Error loading channels from database: {str(e)}")
        
        # Apply filter if any
        filter_text = self.channel_filter_edit.text().strip().lower()
        filtered_channels = [ch for ch in self.database_channels if not filter_text or filter_text in ch.lower()]
        
        # Add to list
        self.pac_channel_list.clear()
        for channel in filtered_channels:
            self.pac_channel_list.addItem(channel)

    def filter_pac_channels(self):
        """Filter channel list based on user input"""
        self.populate_pac_channels()

    def populate_detection_methods(self):
        """Populate detection method lists from database"""
        db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
        if not os.path.exists(db_path):
            self.write_log("Database not found. Cannot load detection methods.")
            return
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            
            # Get slow wave methods with their frequency ranges and stages
            cursor.execute("""
                SELECT method, freq_lower, freq_upper, stage, COUNT(*) as event_count
                FROM events 
                WHERE event_type = 'slow_wave'
                GROUP BY method, freq_lower, freq_upper, stage
                ORDER BY method, freq_lower, freq_upper, stage
            """)
            sw_results = cursor.fetchall()

            # Get spindle methods with their frequency ranges and stages
            cursor.execute("""
                SELECT method, freq_lower, freq_upper, stage, COUNT(*) as event_count
                FROM events 
                WHERE event_type = 'spindle'
                GROUP BY method, freq_lower, freq_upper, stage
                ORDER BY method, freq_lower, freq_upper, stage
            """)
            spindle_results = cursor.fetchall()
            
                
            conn.close()
            
            # Create display names that include method and frequency range
            sw_display_names = []
            self.sw_methods_info = {}  # Store complete info for each display name
            
            for method, freq_lower, freq_upper, stage, count in sw_results:
                display_name = f"{method} ({freq_lower}-{freq_upper}Hz), {stage}"
                sw_display_names.append(display_name)
                self.sw_methods_info[display_name] = {
                    'method': method,
                    'freq_range': (freq_lower, freq_upper),
                    'stage': stage,
                    'count': count
              }

            # spindles
            spindle_display_names = []
            self.spindle_methods_info = {}

            for method, freq_lower, freq_upper, stage, count in spindle_results:
                display_name = f"{method} ({freq_lower}-{freq_upper}Hz), {stage}"
                spindle_display_names.append(display_name)
                self.spindle_methods_info[display_name] = {
                    'method': method,
                    'freq_range': (freq_lower, freq_upper),
                    'stage': stage,
                    'count': count
                }



            # Update combo boxes
            self.sw_method_pac_combo.clear()
            self.sw_method_pac_combo.addItems(sw_display_names)
            
            self.spindle_method_pac_combo.clear()
            self.spindle_method_pac_combo.addItems(spindle_display_names)
            
            # Connect selection changes to update channels
            self.sw_method_pac_combo.currentIndexChanged.connect(self.update_pac_available_channels)
            self.spindle_method_pac_combo.currentIndexChanged.connect(self.update_pac_available_channels)



            # Update frequency labels if methods are available
            if self.sw_method_pac_combo.count() > 0:
                self.update_sw_freq_from_db(self.sw_method_pac_combo.currentText())
            
            if self.spindle_method_pac_combo.count() > 0:
                self.update_spindle_freq_from_db(self.spindle_method_pac_combo.currentText())
            
            self.write_log(f"Loaded {len(sw_results)} slow wave methods and {len(spindle_results)} spindle methods from database")
        
        except Exception as e:
            self.write_log(f"Error loading detection methods: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_pac_analysis_thread(self):
        """Start PAC analysis in a separate thread"""
        if not self.dataset:
            QMessageBox.critical(self, "Error", "No dataset loaded. Please load a dataset first.")
            return
        
        # Check if database exists
        db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
        if not os.path.exists(db_path):
            QMessageBox.critical(self, "Error", "Database not found. Please run event detection first.")
            return
        
        # Get method and parameters
        pac_method = self.pac_method_combo.currentText()
        
        # Get method-specific parameters
        if pac_method == "SW-Spindle":
            # Check method selection
            if self.sw_method_pac_combo.count() == 0:
                QMessageBox.critical(self, "Error", "No slow wave detection methods available. Please run slow wave detection first.")
                return
            if self.spindle_method_pac_combo.count() == 0:
                QMessageBox.critical(self, "Error", "No spindle detection methods available. Please run spindle detection first.")
                return
            
            sw_method = self.sw_method_pac_combo.currentText()
            spindle_method = self.spindle_method_pac_combo.currentText()
            
            # Get method info from stored data
            sw_info = self.sw_methods_info.get(sw_method, {})
            spindle_info = self.spindle_methods_info.get(spindle_method, {})
            
            if not sw_info or not spindle_info:
                QMessageBox.critical(self, "Error", "Method information not found.")
                return

            # Extract base method names and parameters
            sw_base_method = sw_info.get('method')
            sw_stage = sw_info.get('stage')
            phase_freq = sw_info.get('freq_range', (0.5, 1.25))
            
            spindle_base_method = spindle_info.get('method')
            spindle_stage = spindle_info.get('stage')
            amp_freq = spindle_info.get('freq_range', (11, 16))
            
            # Verify stages match
            if sw_stage != spindle_stage:
                response = QMessageBox.question(
                    self, "Stage Mismatch", 
                    f"Sleep stages don't match: SW: {sw_stage}, Spindle: {spindle_stage}. Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.No:
                    return
            
            # Use the stage from SW for consistency
            selected_stages = [sw_stage]
            
        else:  # Theta-Gamma
            # Get frequency ranges from spinboxes
            phase_freq = (self.theta_min_spin.value(), self.theta_max_spin.value())
            amp_freq = (self.gamma_min_spin.value(), self.gamma_max_spin.value())
            sw_method = None
            spindle_method = None
        
            # Get selected stages
            selected_stages = [stage for stage, check in self.pac_stage_checks.items() if check.isChecked()]
            if not selected_stages:
                QMessageBox.critical(self, "Error", "No sleep stages selected. Please select at least one stage.")
                return
        
        # Get selected channels from the new selection interface
        if not hasattr(self, 'pac_selected_channels') or not self.pac_selected_channels:
            QMessageBox.critical(self, "Error", "No channels selected. Please select at least one channel.")
            return
        selected_channels = self.pac_selected_channels

        # Get time window
        time_window = self.time_window_spin.value()

        
        # Get IDPAC parameters
        idpac_method = self.idpac_method_combo.currentIndex() + 1  # 1-based indexing
        surrogate_method = self.surrogate_method_combo.currentIndex()
        correction_method = self.correction_method_combo.currentIndex()
        
        # Store parameters for analysis thread
        self.pac_analysis_params = {
            'method': pac_method,
            'sw_method': sw_base_method if pac_method == "SW-Spindle" else None,
            'spindle_method': spindle_base_method if pac_method == "SW-Spindle" else None,
            'phase_freq': phase_freq,
            'amp_freq': amp_freq,
            'channels': selected_channels,
            'stages': selected_stages,
            'idpac': (idpac_method, surrogate_method, correction_method),
            'time_window': time_window,
            'db_path': db_path
        }
        
        # Disable button and show progress
        self.run_pac_btn.setEnabled(False)
        self.statusBar().showMessage("Running PAC analysis...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        # Log
        self.write_log("Starting PAC analysis...")
        self.write_log(f"Method: {pac_method}")
        if pac_method == "SW-Spindle":
            self.write_log(f"SW Method: {sw_base_method}")
            self.write_log(f"Spindle Method: {spindle_base_method}")
            self.write_log(f"Stage: {', '.join(selected_stages)}")
        else:
            self.write_log(f"Stages: {', '.join(selected_stages)}")
        
        self.write_log(f"Phase Frequency: {phase_freq[0]}-{phase_freq[1]} Hz")
        self.write_log(f"Amplitude Frequency: {amp_freq[0]}-{amp_freq[1]} Hz")
        self.write_log(f"Channels: {len(selected_channels)} channels")
        self.write_log(f"IDPAC: {self.pac_analysis_params['idpac']}")
        
        # Start thread
        self.pac_thread = threading.Thread(target=self.run_pac_analysis)
        self.pac_thread.daemon = True
        self.pac_thread.start()

    def map_regions_to_channels(self, regions):
        """Map brain regions to actual channel names"""
        # This is a placeholder implementation
        # In a real implementation, you would have a mapping of regions to channels
        # based on the EEG montage
        
        # For now, use a simple prefix-based mapping
        if not hasattr(self, 'database_channels'):
            return []
        
        mapped_channels = []
        for region in regions:
            prefix = region[0]  # Use first letter as prefix (F for Frontal, etc.)
            for channel in self.database_channels:
                if channel.startswith(prefix):
                    mapped_channels.append(channel)
        
        return mapped_channels

    def run_pac_analysis(self):
        """Run PAC analysis (in a thread)"""
        try:
            # Create a GUI log handler
            gui_log_handler = GUILogHandler(self.write_log)
            
            # Get parameters
            params = self.pac_analysis_params
            
            # Import the PAC processor
            from turtlewave_hdEEG import ParalPAC
            
            # Create output directory
            pac_dir = os.path.join(self.output_dir, "wonambi", "pac_results")
            if not os.path.exists(pac_dir):
                os.makedirs(pac_dir)
            
            # Create PAC processor
            pac_processor = ParalPAC(
                dataset=self.dataset,
                annotations=self.annotations,
                rootpath=self.output_dir,
                log_level=logging.INFO
            )
            
            # Add GUI log handler
            pac_processor.logger.addHandler(gui_log_handler)
            
            # Setup event options if using SW-Spindle coupling
            event_opts = {}
            if params['method'] == "SW-Spindle":
                event_opts = {
                    'buffer': params['time_window'],
                    'sw_method': params['sw_method'],
                    'spindle_method': params['spindle_method']
                }
            
            # Run PAC analysis
            if params['method'] == "SW-Spindle":
                # For SW-Spindle coupling
                self.write_log(f"Running SW-Spindle coupling analysis...")
                results = pac_processor.analyze_pac(
                    chan=params['channels'],
                    stage=params['stages'],
                    phase_freq=params['phase_freq'],
                    amp_freq=params['amp_freq'],
                    idpac=params['idpac'],
                    use_detected_events=True,
                    event_type='slow_wave',
                    pair_with_spindles=True,
                    time_window=params['time_window'],
                    db_path=params['db_path'],
                    out_dir=pac_dir,
                    event_opts=event_opts
                )
            else:
                # For other coupling types (e.g., Theta-Gamma)
                self.write_log(f"Running {params['method']} coupling analysis...")
                results = pac_processor.analyze_pac(
                    chan=params['channels'],
                    stage=params['stages'],
                    phase_freq=params['phase_freq'],
                    amp_freq=params['amp_freq'],
                    idpac=params['idpac'],
                    use_detected_events=False,  # Use continuous data
                    time_window=params['time_window'],
                    db_path=params['db_path'],
                    out_dir=pac_dir
                )
            

            method_info = {
                'sw_method': event_opts.get('sw_method', 'unknown') if event_opts else 'unknown',
                'spindle_method': event_opts.get('spindle_method', 'unknown') if event_opts else 'unknown',
                'event_type': 'slow_wave' if params['method'] == 'SW-Spindle' else 'continuous',
                'stage': params['stages'],
                'pair_with_spindles': True if params['method'] == 'SW-Spindle' else False
            }

            # # Generate method-specific output name
            # method_name = params['method'].lower().replace("-", "_")
            # if params['method'] == "SW-Spindle":
            #     method_name += f"_{params['sw_method']}_{params['spindle_method']}"
            
            # Export results to CSV
            # csv_file = os.path.join(pac_dir, f"{params['method'].lower().replace('-', '_')}_pac_summary.csv")
            self.write_log(f"Exporting PAC results to CSV with method info: {method_info}")
            self.write_log(f"Base directory: {pac_dir}")

            pac_processor.export_pac_parameters_to_csv(
                csv_file= None,
                phase_freq=params['phase_freq'],
                amp_freq=params['amp_freq'],
                out_dir=pac_dir, 
                method_info= method_info
            )
            
            self.write_log(f"PAC analysis completed. Results saved to {pac_dir}")
            
            # Store results for later use
            self.pac_results = {
                'dir': pac_dir,
                'params': params
            }
            
            # Update UI in main thread
            QtCore.QMetaObject.invokeMethod(
                self, "finish_pac_analysis", 
                QtCore.Qt.QueuedConnection
            )
        
        except Exception as e:
            self.write_log(f"Error in PAC analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"PAC analysis failed: {str(e)}")
            )
            
            # Re-enable button in main thread
            QtCore.QMetaObject.invokeMethod(
                self.run_pac_btn, "setEnabled", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True)
            )
            
            QtCore.QMetaObject.invokeMethod(
                self.progress, "setVisible", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False)
            )

    @QtCore.pyqtSlot()
    def finish_pac_analysis(self):
        """Complete PAC analysis and update UI"""
        self.run_pac_btn.setEnabled(True)
        self.view_pac_results_btn.setEnabled(True)
        #self.export_pac_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("PAC analysis completed")
        QMessageBox.information(self, "Success", "PAC analysis completed successfully.")

    def view_pac_results(self):
        """View PAC analysis results"""
        if not hasattr(self, 'pac_results'):
            QMessageBox.critical(self, "Error", "No PAC results available.")
            return
        
        pac_dir = self.pac_results['dir']
        
    # Find all CSV files recursively in the pac_dir and subdirectories
        import glob
        csv_files = glob.glob(os.path.join(pac_dir, "**", "*.csv"), recursive=True)
        
        if not csv_files:
            QMessageBox.critical(self, "Error", "No CSV result files found.")
            return
        
        # Create file viewer dialog
        viewer = QtWidgets.QDialog(self)
        viewer.setWindowTitle("PAC Analysis Results")
        viewer.resize(800, 600)
        
        layout = QVBoxLayout(viewer)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select Result File:"))
        
        # Use relative paths for display
        display_paths = [os.path.relpath(f, pac_dir) for f in csv_files]
        
        file_combo = QComboBox()
        file_combo.addItems(display_paths)
        file_layout.addWidget(file_combo, 1)
        
        layout.addLayout(file_layout)
        
        # Text area
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)
        
        # Load function
        def load_file():
            selected_file = file_combo.currentText()
            if selected_file:
                try:
                    file_path = os.path.join(pac_dir, selected_file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    text_area.setText(content)
                except Exception as e:
                    QMessageBox.critical(viewer, "Error", f"Failed to load file: {str(e)}")
        
        # Load button
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(load_file)
        file_layout.addWidget(load_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(viewer.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
        
        # Load the first file by default
        file_combo.setCurrentIndex(0)
        load_file()
        
        viewer.exec_()

    # def export_pac_results(self):
    #     """Export PAC analysis results to user-selected location"""
    #     if not hasattr(self, 'pac_results'):
    #         QMessageBox.critical(self, "Error", "No PAC results available.")
    #         return
        
    #     # Ask for export directory
    #     export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
    #     if not export_dir:
    #         return
        
    #     try:
    #         # Copy all files from pac_results directory to export directory
    #         import shutil
    #         import glob
            
    #         pac_dir = self.pac_results['dir']
    #         files = glob.glob(os.path.join(pac_dir, "*.*"))
            
    #         for file in files:
    #             shutil.copy2(file, export_dir)
            
    #         QMessageBox.information(self, "Success", f"Results exported to {export_dir}")
        
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
    def process_annotations_thread(self):
        """Start annotation processing in a separate thread"""
        if not self.dataset:
            QMessageBox.critical(self, "Error", "No dataset loaded. Please load a dataset first.")
            return
        
        self.statusBar().showMessage("Generating annotations...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        # Disable buttons
        self.generate_annot_btn.setEnabled(False)
        
        # Log
        self.write_log("Starting annotation generation...")
        
        # Start thread
        self.annotation_thread = threading.Thread(target=self.process_annotations)
        self.annotation_thread.daemon = True
        self.annotation_thread.start()
    
    def process_annotations(self):
        """Process annotations (runs in a thread)"""
        try:
            # Create annotations
            annotations = XLAnnotations(self.dataset, self.annot_file_path)
            
            # Process artifacts, arousals, and sleep stages based on user selection
            process_all = (self.artifact_check.isChecked() and 
                          self.arousal_check.isChecked() and 
                          self.stage_check.isChecked())
            
            if process_all:
                annotations.process_all()
                self.write_log("Processed all annotation types")
            else:
                if self.artifact_check.isChecked():
                    annotations.process_artifact()
                    self.write_log("Processed artifacts")
                
                if self.arousal_check.isChecked():
                    annotations.process_arousal()
                    self.write_log("Processed arousals")
                
                if self.stage_check.isChecked():
                    annotations.process_stage()
                    self.write_log("Processed sleep stages")
            
            self.write_log(f"Annotations saved to {self.annot_file_path}")
            
            # Update UI in main thread
            QtCore.QMetaObject.invokeMethod(
                self, "finish_annotations", 
                QtCore.Qt.QueuedConnection
            )
        
        except Exception as e:
            self.write_log(f"Error generating annotations: {str(e)}")
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Failed to generate annotations: {str(e)}")
            )
            
            # Re-enable buttons in main thread
            QtCore.QMetaObject.invokeMethod(
                self.generate_annot_btn, "setEnabled", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True)
            )
            
            QtCore.QMetaObject.invokeMethod(
                self.progress, "setVisible", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False)
            )
    
    @QtCore.pyqtSlot()
    def finish_annotations(self):
        """Finish annotation generation"""
        self.generate_annot_btn.setEnabled(True)
        self.view_annot_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Annotations generated successfully")
        QMessageBox.information(self, "Success", "Annotations have been generated successfully.")
    
    def view_annotation_file(self):
        """View the annotation file"""
        if not os.path.isfile(self.annot_file_path):
            QMessageBox.critical(self, "Error", "Annotation file doesn't exist.")
            return
        
        try:
            # Simple file viewer
            viewer = QtWidgets.QDialog(self)
            viewer.setWindowTitle(f"Annotation File: {os.path.basename(self.annot_file_path)}")
            viewer.resize(800, 600)
            
            layout = QVBoxLayout(viewer)
            
            text_area = QTextEdit()
            text_area.setReadOnly(True)
            layout.addWidget(text_area)
            
            with open(self.annot_file_path, 'r') as f:
                content = f.read()
            
            text_area.setText(content)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(viewer.close)
            layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
            
            viewer.exec_()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open annotation file: {str(e)}")
    
    def detect_spindles_thread(self):
        """Start spindle detection in a separate thread"""
        if not self.dataset:
            QMessageBox.critical(self, "Error", "No dataset loaded. Please load a dataset first.")
            return
        
        # Check if annotation file exists
        if not os.path.isfile(self.annot_file_path):
            response = QMessageBox.question(
                self, "Annotation File Missing", 
                "No annotation file found. Would you like to generate annotations first?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                self.tabs.setCurrentIndex(1)  # Switch to annotation tab
                return
            else:
                return
        
        # Get current parameter values from UI
        self.spindle_method = self.method_combo.currentText()
        self.min_freq = self.min_freq_spin.value()
        self.max_freq = self.max_freq_spin.value()
        self.min_duration = self.min_dur_spin.value()
        self.max_duration = self.max_dur_spin.value()
        
        # Get method-specific parameters
        method_params = {}
        if hasattr(self, 'spindle_param_widgets'):
            for param, widget in self.spindle_param_widgets.items():
                method_params[param] = widget.value()
        
        # Check if channels are selected
        if not self.selected_channels:
            QMessageBox.critical(self, "Error", "No channels selected. Please select at least one channel.")
            return

        
        # Get selected sleep stages
        selected_stages = [stage for stage, check in self.stage_checks.items() if check.isChecked()]
        if not selected_stages:
            QMessageBox.critical(self, "Error", "No sleep stages selected. Please select at least one stage.")
            return
        
        self.statusBar().showMessage("Detecting spindles...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        # Disable button
        self.detect_btn.setEnabled(False)
        
        # Log
        self.write_log("Starting spindle detection...")
        self.write_log(f"Method: {self.spindle_method}")
        self.write_log(f"Frequency range: {self.min_freq}-{self.max_freq} Hz")
        self.write_log(f"Duration range: {self.min_duration}-{self.max_duration} seconds")        

        for param, value in method_params.items():
            self.write_log(f"Parameter {param}: {value}")


        # Get signal inversion setting
        invert_signal = self.invert_signal_check.isChecked()
        
        # Log the inversion setting
        self.write_log(f"Signal inversion: {'Enabled' if invert_signal else 'Disabled'}")
    

        # Start thread
        self.spindle_thread = threading.Thread(target=self.detect_spindles, 
                                          args=(selected_stages, method_params,invert_signal))
        self.spindle_thread.daemon = True
        self.spindle_thread.start()
    
    def detect_spindles(self, selected_stages,method_params=None,invert_signal=False):
        """Detect spindles (runs in a thread)"""
        try:
            # Create a GUI log handler
            gui_log_handler = GUILogHandler(self.write_log)
            # Load dataset and annotation for spindle detection
            data = self.dataset
            
            # Check if we should use existing annotations or load fresh
            if self.annotations and os.path.isfile(self.annot_file_path):
                annot = self.annotations  # Use existing if available
                self.write_log("Using existing loaded annotations")
            else:
                annot = CustomAnnotations(self.annot_file_path)
                self.annotations = annot  # Store for future use
                self.write_log(f"Loaded annotation file: {self.annot_file_path}")

            # Create spindle results directory
            json_dir = os.path.join(self.output_dir, "wonambi", "spindle_results")
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
                self.write_log(f"Created directory: {json_dir}")
            
            # Create ParalEvents instance with log handler
            event_processor = ParalEvents(dataset=data, annotations=annot,
                                          log_level=logging.INFO, log_file=None)
            
            event_processor.logger.addHandler(gui_log_handler)

            # Get frequency range
            freq_range = (self.min_freq, self.max_freq)
            
            # Get duration range
            duration_range = (self.min_duration, self.max_duration)
            
            self.write_log(f"Detecting spindles using {self.spindle_method} method")
            self.write_log(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
            self.write_log(f"Duration range: {duration_range[0]}-{duration_range[1]} seconds")
            self.write_log(f"Selected channels: {len(self.selected_channels)} channels")
            self.write_log(f"Selected stages: {', '.join(selected_stages)}")
            
            # Create custom params to pass to detect_spindles
            custom_params = {}
            if method_params:
                # Map widget parameter names to detector parameter names
                param_mapping = {
                    # Moelle2011
                    "det_thresh": "det_thresh",
                    "rms_dur": "moving_rms",
                    
                    # Ferrarelli2007
                    "sel_thresh": "sel_thresh",
                    
                    # Wamsley2012
                    "wavelet_sd": {"det_wavelet": {"sd": None}},
                    "wavelet_dur": {"det_wavelet": {"dur": None}},
                    
                    # Ray2015
                    "zscore_dur": {"zscore": {"dur": None}},
                    
                    # Lacourse2018
                    "abs_thresh": "abs_pow_thresh",
                    "rel_thresh": "rel_pow_thresh",
                    "covar_thresh": "covar_thresh",
                    "corr_thresh": "corr_thresh",
                    "window_dur": {"windowing": {"dur": None}, 
                                "moving_ms": {"dur": None},
                                "moving_power_ratio": {"dur": None},
                                "moving_covar": {"dur": None},
                                "moving_sd": {"dur": None}}
                }
                
                for param, value in method_params.items():
                    if param in param_mapping:
                        mapping = param_mapping[param]
                        if isinstance(mapping, str):
                            # Simple mapping
                            custom_params[mapping] = value
                        elif isinstance(mapping, dict):
                            # Nested parameter
                            for parent_key, nested in mapping.items():
                                if parent_key not in custom_params:
                                    custom_params[parent_key] = {}
                                if nested is None:
                                    # Direct value assignment
                                    custom_params[parent_key] = value
                                else:
                                    # Nested dictionary
                                    for nested_key, _ in nested.items():
                                        if isinstance(custom_params[parent_key], dict):
                                            custom_params[parent_key][nested_key] = value

             # Add polarity parameter
            polar = 'opposite' if invert_signal else 'normal'

            # Detect spindles
            spindles = event_processor.detect_spindles(
                method=self.spindle_method,
                chan=self.selected_channels,
                frequency=freq_range,
                duration=duration_range,
                stage=selected_stages,
                reject_artifacts=self.reject_artifacts_check.isChecked(),
                reject_arousals=self.reject_arousals_check.isChecked(),
                cat=(1, 1, 1, 0),  # concatenate within and between stages, cycles separate
                polar=polar,
                save_to_annotations=False,
                json_dir=json_dir,
                **custom_params
            )
            
            # Format names for export
            freq_range_str = f"{freq_range[0]}-{freq_range[1]}Hz"
            stages_str = "".join(selected_stages)
            file_pattern = f"spindles_{self.spindle_method}_{freq_range_str}_{stages_str}"
            
            # Export parameters to CSV
            param_csv = os.path.join(json_dir, f'spindle_parameters_{self.spindle_method}_{freq_range_str}_{stages_str}.csv')

            method_info = {
                'method': self.spindle_method,
                'frequency': freq_range,
                'duration': duration_range,
                'custom_params': custom_params,  # Add this
                'polar': polar,
                'stages': selected_stages
            }

            event_processor.export_spindle_parameters_to_csv(
                json_input=json_dir,
                csv_file=param_csv,
                file_pattern=file_pattern
            )
            
            # Initialize and update database
            # Add database functionality
            db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
            self.write_log(f"Initializing/updating database at {db_path}")
            event_processor.initialize_sqlite_database(db_path)
            self.write_log(f"Importing spindle parameters to database")
            stats = event_processor.import_parameters_csv_to_database(param_csv, db_path)
            self.write_log(f"Database update complete: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} skipped")

            # Export density to CSV
            density_csv = os.path.join(json_dir, f'spindle_density_{self.spindle_method}_{freq_range_str}_{stages_str}.csv')
            event_processor.export_spindle_density_to_csv(
                json_input=json_dir,
                csv_file=density_csv,
                stage=selected_stages,
                file_pattern=file_pattern
            )
            
            self.write_log(f"Spindle parameters saved to {param_csv}")
            self.write_log(f"Spindle density saved to {density_csv}")
            self.write_log("Spindle detection completed successfully")
            
            # Save detection summary
            try:
                # Prepare parameters summary
                parameters_summary = {
                    'method': self.spindle_method,
                    'frequency_range': freq_range,
                    'duration_range': duration_range,
                    'channels': self.selected_channels,
                    'stages': selected_stages,
                    'polar': polar,
                    'reject_artifacts': self.reject_artifacts_check.isChecked(),
                    'reject_arousals': self.reject_arousals_check.isChecked(),
                    'method_specific_parameters': method_params if 'method_params' in locals() else {}
                }
                
                # Prepare results summary
                results_summary = {
                    'total_spindles_detected': len(spindles) if 'spindles' in locals() else 0,
                    'channels_processed': len(self.selected_channels),
                    'csv_file': param_csv,
                    'density_file': density_csv,
                    'database_file': db_path
                }
                
                # Save detection summary
                event_processor.save_detection_summary(
                    output_dir=json_dir,
                    method=self.spindle_method,
                    parameters=parameters_summary,
                    results_summary=results_summary
                )
                
            except Exception as e:
                self.write_log(f"Note: Could not save detection summary: {e}")    

            # Update UI in main thread
            QtCore.QMetaObject.invokeMethod(
                self, "finish_spindle_detection", 
                QtCore.Qt.QueuedConnection
            )
        
        except Exception as e:
            self.write_log(f"Error detecting spindles: {str(e)}")
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Failed to detect spindles: {str(e)}")
            )
            
            # Re-enable button in main thread
            QtCore.QMetaObject.invokeMethod(
                self.detect_btn, "setEnabled", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True)
            )
            
            QtCore.QMetaObject.invokeMethod(
                self.progress, "setVisible", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False)
            )
    

    @QtCore.pyqtSlot()
    def finish_sw_detection(self):
        """Finish slow wave detection"""
        self.detect_sw_btn.setEnabled(True)
        self.view_sw_results_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Slow wave detection completed")
        QMessageBox.information(self, "Success", "Slow wave detection completed successfully.")
        # Update PAC detection methods
        self.populate_detection_methods()

    def view_sw_results(self):
        """View slow wave detection results"""
        # Similar to view_spindle_results but for slow waves
        # Change directory path to sw_results and adjust file patterns

    @QtCore.pyqtSlot()
    def finish_spindle_detection(self):
        """Finish spindle detection"""
        self.detect_btn.setEnabled(True)
        self.view_results_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Spindle detection completed")
        QMessageBox.information(self, "Success", "Spindle detection completed successfully.")
        self.populate_detection_methods()

    
    def view_spindle_results(self):
        """View spindle detection results"""
        json_dir = os.path.join(self.output_dir, "wonambi", "spindle_results")
        
        if not os.path.isdir(json_dir):
            QMessageBox.critical(self, "Error", "Spindle results directory doesn't exist.")
            return
        
        # Get list of CSV files
        csv_files = [f for f in os.listdir(json_dir) if f.endswith('.csv')]
        
        if not csv_files:
            QMessageBox.critical(self, "Error", "No CSV result files found.")
            return
        
        # Create file viewer dialog
        viewer = QtWidgets.QDialog(self)
        viewer.setWindowTitle("Spindle Detection Results")
        viewer.resize(800, 600)
        
        layout = QVBoxLayout(viewer)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select Result File:"))
        
        file_combo = QComboBox()
        file_combo.addItems(csv_files)
        file_layout.addWidget(file_combo, 1)
        
        layout.addLayout(file_layout)
        
        # Text area
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)
        
        # Load function
        def load_file():
            selected_file = file_combo.currentText()
            if selected_file:
                try:
                    file_path = os.path.join(json_dir, selected_file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    text_area.setText(content)
                except Exception as e:
                    QMessageBox.critical(viewer, "Error", f"Failed to load file: {str(e)}")
        
        # Load button
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(load_file)
        file_layout.addWidget(load_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(viewer.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
        
        # Load the first file by default
        file_combo.setCurrentIndex(0)
        load_file()
        
        viewer.exec_()


    def view_sw_results(self):
        """View slow wave detection results"""
        json_dir = os.path.join(self.output_dir, "wonambi", "sw_results")
        
        if not os.path.isdir(json_dir):
            QMessageBox.critical(self, "Error", "Slow wave results directory doesn't exist.")
            return
        
        # Get list of CSV files
        csv_files = [f for f in os.listdir(json_dir) if f.endswith('.csv')]
        
        if not csv_files:
            QMessageBox.critical(self, "Error", "No CSV result files found.")
            return
        
        # Create file viewer dialog
        viewer = QtWidgets.QDialog(self)
        viewer.setWindowTitle("Slow Wave Detection Results")
        viewer.resize(800, 600)
        
        layout = QVBoxLayout(viewer)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select Result File:"))
        
        file_combo = QComboBox()
        file_combo.addItems(csv_files)
        file_layout.addWidget(file_combo, 1)
        
        layout.addLayout(file_layout)
        
        # Text area
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)
        
        # Load function
        def load_file():
            selected_file = file_combo.currentText()
            if selected_file:
                try:
                    file_path = os.path.join(json_dir, selected_file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    text_area.setText(content)
                except Exception as e:
                    QMessageBox.critical(viewer, "Error", f"Failed to load file: {str(e)}")
        
        # Load button
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(load_file)
        file_layout.addWidget(load_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(viewer.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)
        
        # Load the first file by default
        file_combo.setCurrentIndex(0)
        load_file()
        
        viewer.exec_()


    def launch_event_review(self):
            """Launch the event review GUI as a separate window"""
            try:
                # Import from the frontend package (using relative import since we're in the same package)
                from .eeg_eventview import EventReviewInterface
                
                # Create and show the event review window
                self.review_window = EventReviewInterface()
                
                # Pre-populate with current data if available
                if self.data_file_path:
                    self.review_window.eeg_file_edit.setText(self.data_file_path)
                if self.annot_file_path:
                    self.review_window.annot_file_edit.setText(self.annot_file_path)
                
                # Look for database files in the output directory
                if self.output_dir:
                    import glob
                    db_files = glob.glob(os.path.join(self.output_dir, "wonambi", "**", "*.db"), recursive=True)
                    if db_files:
                        # Use the most recent database file
                        latest_db = max(db_files, key=os.path.getmtime)
                        self.review_window.db_file_edit.setText(latest_db)
                
                self.review_window.show()
                self.write_log("Launched Event Review GUI")
                
            except ImportError as e:
                QMessageBox.critical(self, "Error", f"Could not launch Event Review: {e}")
                self.write_log(f"Event Review not available: {e}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error launching Event Review: {e}")
                self.write_log(f"Error launching Event Review: {e}")


    def write_log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Store messages in a buffer if log_text isn't initialized yet
        if self.log_text is None:
            if not hasattr(self, '_log_buffer'):
                self._log_buffer = []
            self._log_buffer.append(log_message)
            print(log_message)  # Print to console as fallback
            return

        # Use invokeMethod to ensure thread safety
        QtCore.QMetaObject.invokeMethod(
            self.log_text, "append", 
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, log_message)
        )

    def clear_log(self):
        """Clear the log"""
        self.log_text.clear()

    # method to save the log
    def save_log_on_exit(self):
        """Save log to file on program exit """

        if not self.output_dir or not os.path.isdir(self.output_dir):
            print("Cannot save log: No valid output directory set")
            return
        
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"turtlewave_log_{timestamp}.txt"
            log_filepath = os.path.join(self.output_dir, "wonambi", log_filename)
            
            # Get the content from the log text area
            log_content = self.log_text.toPlainText()
            
            # Save to file
            with open(log_filepath, 'w') as f:
                f.write(log_content)
            
            print(f"Log saved to {log_filepath}")
        except Exception as e:
            print(f"Error saving log: {str(e)}")
    
    # Add the closeEvent method here
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(self, 'Exit', 
            "Are you sure you want to exit?\nThe log will be saved automatically.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes  # Default button)
        )
        if reply == QMessageBox.Yes:
            # User wants to exit and save log
            self.save_log_on_exit()
            event.accept()
        else:
            # User wants to exit without saving log
            event.accept()    
 

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create light palette with custom background color (RGB 247, 252, 253)
    light_palette = QtGui.QPalette()
    background_color = QtGui.QColor(247, 252, 253)
    light_palette.setColor(QtGui.QPalette.Window, background_color)
    light_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 0, 0))
    light_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
    light_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(240, 245, 250))
    light_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    light_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    light_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
    light_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(230, 240, 245))
    light_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(0, 0, 0))
    light_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    light_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    light_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    light_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    
    # Apply the palette
    app.setPalette(light_palette)
    
    # Apply stylesheet for nicer buttons with a light theme
    app.setStyleSheet("""
        QPushButton {
            background-color: #E6F0F5;
            border: 1px solid #C0D0E0;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #D0E0F0;
        }
        QPushButton:pressed {
            background-color: #B0C0D0;
        }
        QPushButton:disabled {
            background-color: #F0F0F0;
            color: #A0A0A0;
        }
        QGroupBox {
            border: 1px solid #C0D0E0;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
            background-color: rgba(247, 252, 253, 180);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
            color: #305070;
        }
        QTabWidget::pane {
            border: 1px solid #C0D0E0;
            background-color: rgb(247, 252, 253);
        }
        QTabBar::tab {
            background-color: #E6F0F5;
            border: 1px solid #C0D0E0;
            border-bottom-color: #C0D0E0;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 5px 10px;
        }
        QTabBar::tab:selected, QTabBar::tab:hover {
            background-color: rgb(247, 252, 253);
        }
        QTabBar::tab:selected {
            border-color: #C0D0E0;
            border-bottom-color: rgb(247, 252, 253);
        }
        QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
            border: 1px solid #C0D0E0;
            border-radius: 2px;
            padding: 2px;
            background-color: white;
            selection-background-color: #D0E0F0;
        }
        QProgressBar {
            border: 1px solid #C0D0E0;
            border-radius: 2px;
            background-color: white;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #6090C0;
            width: 10px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #C0D0E0;
            background-color: white;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #C0D0E0;
            background-color: #6090C0;
        }
    """)
    
    try:
        window = TurtleWaveGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        QMessageBox.critical(None, "Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()