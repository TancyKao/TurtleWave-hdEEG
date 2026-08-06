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
    from turtlewave_hdEEG import LargeDataset, XLAnnotations, ParalEvents, ParalSWA, ParalKC, CustomAnnotations
    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave, ImprovedDetectSpindle, ImprovedDetectKComplex
    # Single source of truth for the '{lo}-{hi}Hz' filename token, shared with
    # the detectors so any filename this GUI builds matches what the library
    # writes.
    #
    # The rest of these are the database-era result path. Detection no longer
    # writes per-channel JSON: neural_events.db is the store of record, so the
    # GUI resolves the target up front (resolve_db_target), checks afterwards
    # that every requested channel is accounted for (verify_channel_coverage),
    # derives density from the stored denominator (event_density), and offers
    # a flat CSV on demand (export_events_to_csv).
    from turtlewave_hdEEG.dbwrite import (fmt_freq_token, resolve_db_target,
                                          verify_channel_coverage,
                                          export_events_to_csv)
    from turtlewave_hdEEG.density import event_density

    #from wonambi.dataset import Dataset as WonambiDataset
except ImportError as e:
    print(f"Error importing TurtleWave hdEEG package: {e}")

try:
    from frontend.db_connect import connect_events_db
except ImportError:  # run as a script: frontend/ is on sys.path, not its parent
    from db_connect import connect_events_db


class LoggingOutput(QtCore.QObject):
    """Class to capture and redirect logging to the GUI"""
    text_written = QtCore.pyqtSignal(str)
    
    def write(self, text):
        if text.strip():  # Only emit if there's actual text
            self.text_written.emit(text.rstrip())
    
    def flush(self):
        pass

class GUILogHandler(logging.Handler):
    """Redirect library log records into the GUI log pane.

    The formatter deliberately carries no timestamp, logger name or level.
    ``TurtleWaveGUI.write_log`` already stamps every line it receives with
    ``[YYYY-MM-DD HH:MM:SS]``, so the library's own
    ``%(asctime)s - %(name)s - %(levelname)s`` prefix would give the GUI pane
    two timestamps for one message. That prefix is right for a log *file* and
    is left untouched on the library's own handlers; it is only stripped here,
    at the point where records enter the GUI.

    The level is still shown for WARNING and above, because a warning that
    reads like an ordinary progress line is worse than a slightly noisy one.
    """

    def __init__(self, signal_fn):
        """Initialize with a function to emit log messages to"""
        super().__init__()
        self.signal_fn = signal_fn
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        """Emit a log record to the GUI"""
        log_message = self.format(record)
        if record.levelno >= logging.WARNING:
            log_message = f"{record.levelname}: {log_message}"
        # Use the signal function to write to the GUI log
        self.signal_fn(log_message)

class TurtleWaveGUI(QMainWindow):

    #: Root of the library's logger tree. Every module and processor logger in
    #: ``turtlewave_hdEEG`` is a child of this name, so one handler here
    #: receives all of them - including modules like ``dataset`` that have no
    #: processor object for the GUI to reach into.
    LIBRARY_LOGGER_NAME = 'turtlewave_hdEEG'

    #: Serialises attach/detach of the log handler. Each detection runs in its
    #: own thread and calls ensure_gui_log_handler on entry, so the check-then-
    #: attach sequence below must not interleave: two threads both finding the
    #: handler absent would attach it twice and double every library log line.
    _log_handler_lock = threading.Lock()

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

        # The single handler that carries library log records into the pane.
        # Created lazily by ensure_gui_log_handler().
        self._gui_log_handler = None

        # How the last run of each detector ended, so the dialog that closes a
        # run can say the same thing the log did. Filled by log_run_outcome.
        self._last_run_outcome = {}

        # Setup UI
        self.setup_ui()

        # Listen to the library's logs from the moment the window exists, not
        # only once a detector is constructed: loading the dataset logs before
        # any processor is created, and those records are the ones that say why
        # a file failed to load.
        self.ensure_gui_log_handler()

        # Redirect stdout for logging
        self.log_output = LoggingOutput()
        self.log_output.text_written.connect(self.write_log)
        sys.stdout = self.log_output
    
    def setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.setup_menu_bar()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.setup_tab = QWidget()
        self.annotation_tab = QWidget()
        self.spindle_tab = QWidget()
        self.pac_tab = QWidget()  # Add PAC tab
        self.log_tab = QWidget()
        self.sw_tab = QWidget()
        self.kcomplex_tab = QWidget()
        #self.review_tab = EventReviewTab(self) <================

        # Add tabs to widget
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.annotation_tab, "Annotation")
        self.tabs.addTab(self.spindle_tab, "Spindle Detection")
        self.tabs.addTab(self.sw_tab, "Slow Wave Detection")
        self.tabs.addTab(self.kcomplex_tab, "K-Complex Detection")
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
        self.setup_kcomplex_tab()
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
        self.tabs.setTabEnabled(4, False)  # K-Complex tab
        self.tabs.setTabEnabled(5, False)  # PAC tab
        #self.tabs.setTabEnabled(self.tabs.indexOf(self.review_tab), False)<=================

    def setup_menu_bar(self):
        """Setup menu bar with documentation link"""
        menubar = self.menuBar()
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # Documentation action
        doc_action = QtWidgets.QAction('Documentation', self)
        doc_action.setShortcut('F1')
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)
        
        # About action
        about_action = QtWidgets.QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def open_documentation(self):
        """Open documentation in web browser"""
        import webbrowser
        webbrowser.open('https://turtlewave-hdeeg.readthedocs.io/en/latest/')
        self.write_log("Opened documentation in web browser")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About TurtleWave hdEEG", 
            "TurtleWave hdEEG - Sleep Event Detection and Coupling Suite\n\n"
            "Documentation: https://turtlewave-hdeeg.readthedocs.io/en/latest/\n\n"
            "A comprehensive tool for hd-EEG sleep event detection and analysis.")

    def handle_tab_change(self, index):
        """Handle tab changes"""
        # If switching to PAC tab, make sure methods are populated
        if index == 5:  # PAC tab index
            self.populate_detection_methods()

    def setup_setup_tab(self):
        # Main layout
        layout = QVBoxLayout(self.setup_tab)
        
        doc_group = QGroupBox("Documentation & Help")
        doc_layout = QHBoxLayout()
        
        doc_label = QLabel("📖 For detailed instructions and tutorials, visit:")
        doc_layout.addWidget(doc_label)
        
        doc_link = QPushButton("TurtleWave Documentation")
        doc_link.setStyleSheet("QPushButton { color: #2196F3; text-decoration: underline; border: none; background: none; }")
        doc_link.clicked.connect(self.open_documentation)
        doc_layout.addWidget(doc_link)
        
        doc_layout.addStretch(1)
        doc_group.setLayout(doc_layout)
        layout.addWidget(doc_group)

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
        # Tab-specific names. These were once called reject_artifacts_check /
        # reject_arousals_check here AND on the Spindle tab; setup_ui builds
        # the spindle tab first, so the slow wave widgets silently replaced the
        # spindle ones and a spindle run read whatever this tab said.
        self.sw_reject_artifacts_check = QCheckBox("Reject Artifacts")
        self.sw_reject_artifacts_check.setChecked(True)
        params_form.addWidget(self.sw_reject_artifacts_check)

        self.sw_reject_arousals_check = QCheckBox("Reject Arousals")
        self.sw_reject_arousals_check.setChecked(True)
        params_form.addWidget(self.sw_reject_arousals_check)
        
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

        # Detection writes to neural_events.db and nothing else, so the flat
        # file collaborators ask for is produced here, on demand, from the
        # database as it currently stands.
        self.export_sw_csv_btn = QPushButton("Export CSV")
        self.export_sw_csv_btn.setToolTip(
            "Write the stored slow wave events and their density out as CSV "
            "files, from the database. Detection itself writes no CSV.")
        self.export_sw_csv_btn.clicked.connect(self.export_sw_csv)
        action_layout.addWidget(self.export_sw_csv_btn)

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
            'reject_artifacts': self.sw_reject_artifacts_check.isChecked(),
            'reject_arousals': self.sw_reject_arousals_check.isChecked(),
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
            # Listen on the library logger, not on this processor's logger, so
            # dataset-level messages reach the pane too. Idempotent.
            self.ensure_gui_log_handler()

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

            # No per-processor handler: ParalSWA's logger is a child of
            # 'turtlewave_hdEEG' and propagates to the handler attached there.

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

            # Resolve the database BEFORE detecting: an unwritable target must
            # cost a dialog, not an hour of detection with nowhere to put the
            # results. write_db is left at its default (None = the database),
            # so no per-channel JSON is written.
            db_path = self.run_db_path()
            subject = self.resolve_subject()
            detect_kwargs['db_path'] = db_path
            detect_kwargs['subject'] = subject
            self.write_log(
                f"Slow wave rows will be keyed under subject '{subject}'")

            # Snapshot the runs already recorded for this scope, so the rows
            # this run writes can be told apart from rows an earlier run left.
            runs_before = self.db_run_ids(db_path, 'slow_wave', params['method'])

            slow_waves = event_processor.detect_slow_waves(**detect_kwargs)
            sw_count = self.log_event_count("Slow wave", slow_waves)

            # The method is kept UNESCAPED for every database query: that is how
            # the detector stores it, so 'AASM/Massimini2004' must not be
            # flattened to 'AASM_Massimini2004' here.
            runs_after = self.db_run_ids(db_path, 'slow_wave', params['method'])
            new_runs = (None if runs_before is None or runs_after is None
                        else runs_after - runs_before)
            if not new_runs:
                self.write_log(
                    "Could not identify this run's own database run id, so the "
                    "counts below cover every run stored for this method, band "
                    "and stage set - not only this one.")
            sw_rows, sw_channels = self.count_db_events(
                db_path, 'slow_wave', params['method'], params['frequency'],
                params['stage'], run_ids=new_runs)

            sw_coverage = self.verify_db_channels(
                "Slow wave", db_path, 'slow_wave', params['method'],
                params['chan'], params['frequency'], params['stage'])

            # Density from the database, not from JSON. The denominator is the
            # artefact-free time this run stored in analysed_time, selected by
            # the run's own rejection settings.
            self.report_db_density(
                "Slow wave", db_path, 'slow_wave', params['method'],
                params['frequency'], params['stage'], subject,
                params['reject_artifacts'], params['reject_arousals'])

            self.remember_run_scope(
                "Slow wave", db_path=db_path, event_type='slow_wave',
                method=params['method'], frequency=params['frequency'],
                stage=list(params['stage']), subject=subject,
                reject_artifacts=params['reject_artifacts'],
                reject_arousals=params['reject_arousals'],
                out_dir=json_dir)

            self.log_run_outcome("Slow wave", db_path, sw_count, sw_rows,
                                 sw_channels, sw_coverage)

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
                
                # Report the database, because that is where the results are.
                # Detection writes no CSV, so naming one here would send a
                # reader to a file that does not exist.
                results_summary = {
                    'total_slow_waves_detected': len(slow_waves) if 'slow_waves' in locals() else 0,
                    'channels_requested': len(params['chan']),
                    'channels_in_database': sw_channels,
                    'events_written_to_database': sw_rows,
                    'database_file': db_path,
                    'subject': subject,
                    'channels_missing_from_database': (
                        (sw_coverage or {}).get('missing') or []),
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

    # ============================================================
    # K-Complex Detection
    # ============================================================
    # KCs share the slow-wave detection pipeline (Wonambi's
    # AASM/Massimini2004 thresholds match AASM KC criteria), with one
    # extra knob: `min_isolation` enforces a gap between successive KCs
    # so a KC can't just be one cycle of an N3 slow-oscillation train.
    # The method combo is restricted to Massimini2004 / AASM/Massimini2004;
    # Ngo2015 and Staresina2015 target slow oscillations and are not
    # appropriate for KC scoring.

    def setup_kcomplex_tab(self):
        """Setup the K-complex detection tab"""
        layout = QVBoxLayout(self.kcomplex_tab)

        top_splitter = QSplitter(QtCore.Qt.Horizontal)

        # Left column - parameters
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)

        params_group = QGroupBox("K-Complex Detection Parameters")
        params_form = QVBoxLayout()

        # Method selection (KC-appropriate methods only)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Detection Method:"))
        self.kc_method_combo = QComboBox()
        self.kc_method_combo.addItems(["AASM/Massimini2004", "Massimini2004"])
        method_layout.addWidget(self.kc_method_combo)
        params_form.addLayout(method_layout)

        # Method-specific parameter container
        self.kc_method_params_container = QGroupBox("Method-Specific Parameters")
        self.kc_method_params_layout = QVBoxLayout(self.kc_method_params_container)
        params_form.addWidget(self.kc_method_params_container)
        self.kc_method_combo.currentTextChanged.connect(self.update_kc_params_for_method)

        # Isolation criterion (KC-only)
        iso_group = QGroupBox("Isolation")
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(QLabel("Min isolation (s):"))
        self.kc_min_isolation_spin = QDoubleSpinBox()
        self.kc_min_isolation_spin.setRange(0.0, 5.0)
        self.kc_min_isolation_spin.setSingleStep(0.1)
        self.kc_min_isolation_spin.setValue(1.0)
        self.kc_min_isolation_spin.setToolTip(
            "Minimum gap between successive K-complex troughs. KCs closer "
            "than this are dropped — distinguishes a KC from one cycle of "
            "a continuous slow-oscillation train. Set to 0 to disable.")
        iso_layout.addWidget(self.kc_min_isolation_spin)
        iso_group.setLayout(iso_layout)
        params_form.addWidget(iso_group)

        # Reject options
        self.kc_reject_artifacts_check = QCheckBox("Reject Artifacts")
        self.kc_reject_artifacts_check.setChecked(True)
        params_form.addWidget(self.kc_reject_artifacts_check)

        self.kc_reject_arousals_check = QCheckBox("Reject Arousals")
        self.kc_reject_arousals_check.setChecked(True)
        params_form.addWidget(self.kc_reject_arousals_check)

        params_group.setLayout(params_form)
        params_layout.addWidget(params_group)

        # Stage selection — N2 only by default
        stages_group = QGroupBox("Sleep Stage Selection")
        stages_layout = QHBoxLayout()
        self.kc_stage_checks = {}
        stages = ["NREM1", "NREM2", "NREM3", "REM", "Wake"]
        default_selected = ["NREM2"]
        for stage in stages:
            check = QCheckBox(stage)
            check.setChecked(stage in default_selected)
            stages_layout.addWidget(check)
            self.kc_stage_checks[stage] = check
        stages_group.setLayout(stages_layout)
        params_layout.addWidget(stages_group)

        params_layout.addStretch(1)
        top_splitter.addWidget(params_widget)

        # Right column - channel selection
        channels_widget = QWidget()
        channels_layout = QVBoxLayout(channels_widget)

        channels_group = QGroupBox("Channel Selection")
        channels_content = QHBoxLayout()

        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Available Channels:"))
        self.kc_available_list = QListWidget()
        self.kc_available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        avail_layout.addWidget(self.kc_available_list)
        channels_content.addLayout(avail_layout)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch(1)
        self.kc_add_btn = QPushButton("Add >")
        self.kc_add_btn.clicked.connect(self.add_kc_channels)
        btn_layout.addWidget(self.kc_add_btn)
        self.kc_remove_btn = QPushButton("< Remove")
        self.kc_remove_btn.clicked.connect(self.remove_kc_channels)
        btn_layout.addWidget(self.kc_remove_btn)
        self.kc_add_all_btn = QPushButton("Add All >>")
        self.kc_add_all_btn.clicked.connect(self.add_all_kc_channels)
        btn_layout.addWidget(self.kc_add_all_btn)
        self.kc_remove_all_btn = QPushButton("<< Remove All")
        self.kc_remove_all_btn.clicked.connect(self.remove_all_kc_channels)
        btn_layout.addWidget(self.kc_remove_all_btn)
        btn_layout.addStretch(1)
        channels_content.addLayout(btn_layout)

        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected Channels:"))
        self.kc_selected_list = QListWidget()
        self.kc_selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(self.kc_selected_list)
        channels_content.addLayout(sel_layout)

        channels_group.setLayout(channels_content)
        channels_layout.addWidget(channels_group)
        top_splitter.addWidget(channels_widget)

        layout.addWidget(top_splitter)

        action_layout = QHBoxLayout()
        self.detect_kc_btn = QPushButton("Detect K-Complexes")
        self.detect_kc_btn.clicked.connect(self.detect_kc_thread)
        self.detect_kc_btn.setStyleSheet("font-weight: bold;")
        action_layout.addWidget(self.detect_kc_btn)

        self.view_kc_results_btn = QPushButton("View Results")
        self.view_kc_results_btn.clicked.connect(self.view_kc_results)
        self.view_kc_results_btn.setEnabled(False)
        action_layout.addWidget(self.view_kc_results_btn)

        # See the slow wave tab: the flat file is produced on demand from the
        # database, not as a side effect of detection.
        self.export_kc_csv_btn = QPushButton("Export CSV")
        self.export_kc_csv_btn.setToolTip(
            "Write the stored K-complex events and their density out as CSV "
            "files, from the database. Detection itself writes no CSV.")
        self.export_kc_csv_btn.clicked.connect(self.export_kc_csv)
        action_layout.addWidget(self.export_kc_csv_btn)
        layout.addLayout(action_layout)

        self.update_kc_params_for_method(self.kc_method_combo.currentText())

    def update_kc_params_for_method(self, method_name):
        """Populate KC method-specific parameter widgets."""
        self.clear_layout(self.kc_method_params_layout)
        try:
            detector = ImprovedDetectKComplex(method=method_name)

            method_descriptions = {
                "AASM/Massimini2004":
                    "AASM K-complex criteria applied via the Massimini "
                    "method: ≥75 µV peak-to-peak, 0.25–1.0 s trough "
                    "duration. Recommended default for KC detection.",
                "Massimini2004":
                    "Original Massimini slow-wave thresholds — stricter "
                    "than AASM. Useful if you want to favour high-amplitude "
                    "KCs only.",
            }
            if method_name in method_descriptions:
                desc = QLabel(method_descriptions[method_name])
                desc.setWordWrap(True)
                desc.setStyleSheet(
                    "color: #333; font-style: italic; "
                    "background-color: #f0f4f7; padding: 8px; "
                    "border-radius: 4px;")
                self.kc_method_params_layout.addWidget(desc)
                self.kc_method_params_layout.addSpacing(10)

            info_label = QLabel(f"<b>Parameters for {method_name}:</b>")
            info_label.setAlignment(QtCore.Qt.AlignCenter)
            self.kc_method_params_layout.addWidget(info_label)

            self.kc_param_widgets = {}

            # Filter
            filter_group = QGroupBox("Filter Settings")
            filter_layout = QVBoxLayout()
            order_layout = QHBoxLayout()
            order_layout.addWidget(QLabel("Filter Order:"))
            order_spin = QSpinBox()
            order_spin.setRange(1, 10)
            order_spin.setValue(detector.det_filt.get('order', 2))
            order_layout.addWidget(order_spin)
            filter_layout.addLayout(order_layout)

            freq_range = detector.det_filt.get('freq', (0.1, 4.0))
            freq_layout = QHBoxLayout()
            freq_layout.addWidget(QLabel("Frequency Range (Hz):"))
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
            self.kc_method_params_layout.addWidget(filter_group)

            self.kc_param_widgets["filter"] = {
                "order": order_spin,
                "min_freq": min_freq_spin,
                "max_freq": max_freq_spin,
            }

            # Trough duration
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
            self.kc_method_params_layout.addWidget(trough_group)

            self.kc_param_widgets["trough_duration"] = {
                "min": min_trough_spin,
                "max": max_trough_spin,
            }

            # Amplitude thresholds
            threshold_group = QGroupBox("Amplitude Thresholds")
            threshold_layout = QVBoxLayout()
            neg_layout = QHBoxLayout()
            neg_layout.addWidget(QLabel("Negative Peak Threshold (μV):"))
            neg_peak_spin = QDoubleSpinBox()
            neg_peak_spin.setRange(-200, 0)
            neg_peak_spin.setValue(detector.max_trough_amp)
            neg_layout.addWidget(neg_peak_spin)
            threshold_layout.addLayout(neg_layout)
            p2p_layout = QHBoxLayout()
            p2p_layout.addWidget(QLabel("Peak-to-Peak Threshold (μV):"))
            p2p_spin = QDoubleSpinBox()
            p2p_spin.setRange(0, 300)
            p2p_spin.setValue(detector.min_ptp)
            p2p_layout.addWidget(p2p_spin)
            threshold_layout.addLayout(p2p_layout)
            threshold_group.setLayout(threshold_layout)
            self.kc_method_params_layout.addWidget(threshold_group)
            self.kc_param_widgets["max_trough_amp"] = neg_peak_spin
            self.kc_param_widgets["min_ptp"] = p2p_spin

            options_group = QGroupBox("Signal Processing Options")
            options_layout = QHBoxLayout()
            invert_check = QCheckBox("Invert Signal")
            invert_check.setChecked(False)
            options_layout.addWidget(invert_check)
            options_group.setLayout(options_layout)
            self.kc_method_params_layout.addWidget(options_group)
            self.kc_param_widgets["invert"] = invert_check

            self.kc_method_params_layout.addStretch(1)
        except Exception as e:
            self.write_log(
                f"Error loading parameters from ImprovedDetectKComplex: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"Error loading parameters: {e}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.kc_method_params_layout.addWidget(error_label)

    def detect_kc_thread(self):
        """Start K-complex detection in a separate thread."""
        if not self.dataset:
            QMessageBox.critical(self, "Error", "No dataset loaded. Please load a dataset first.")
            return

        if not os.path.isfile(self.annot_file_path):
            response = QMessageBox.question(
                self, "Annotation File Missing",
                "No annotation file found. Would you like to generate annotations first?",
                QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.Yes:
                self.tabs.setCurrentIndex(1)
                return
            return

        self.kc_method = self.kc_method_combo.currentText()

        if not self.selected_channels:
            QMessageBox.critical(self, "Error", "No channels selected. Please select at least one channel.")
            return

        self.kc_selected_stages = [s for s, c in self.kc_stage_checks.items() if c.isChecked()]
        if not self.kc_selected_stages:
            QMessageBox.critical(self, "Error", "No sleep stages selected. Please select at least one stage.")
            return

        try:
            polar = 'opposite' if self.kc_param_widgets.get(
                "invert", QCheckBox()).isChecked() else 'normal'

            filter_widgets = self.kc_param_widgets["filter"]
            frequency = (filter_widgets["min_freq"].value(),
                         filter_widgets["max_freq"].value())
            trough_widgets = self.kc_param_widgets["trough_duration"]
            trough_duration = (trough_widgets["min"].value(),
                               trough_widgets["max"].value())
            neg_peak_thresh = self.kc_param_widgets["max_trough_amp"].value()
            p2p_thresh = self.kc_param_widgets["min_ptp"].value()
            min_isolation = self.kc_min_isolation_spin.value()

            self.write_log(f"Using method: {self.kc_method}")
            self.write_log(f"Frequency range: {frequency} Hz")
            self.write_log(
                f"Trough duration: {trough_duration[0]:.2f}-{trough_duration[1]:.2f} s")
            self.write_log(f"Negative peak threshold: {neg_peak_thresh} μV")
            self.write_log(f"Peak-to-peak threshold: {p2p_thresh} μV")
            self.write_log(f"Min isolation: {min_isolation} s")
            self.write_log(f"Signal polarity: {polar}")
        except Exception as e:
            self.write_log(f"Error processing parameters: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error processing parameters: {e}")
            return

        self.kc_detection_params = {
            'method': self.kc_method,
            'chan': self.selected_channels,
            'frequency': frequency,
            'trough_duration': trough_duration,
            'neg_peak_thresh': neg_peak_thresh,
            'p2p_thresh': p2p_thresh,
            'min_isolation': min_isolation,
            'polar': polar,
            'reject_artifacts': self.kc_reject_artifacts_check.isChecked(),
            'reject_arousals': self.kc_reject_arousals_check.isChecked(),
            'stage': self.kc_selected_stages,
        }

        self.statusBar().showMessage("Detecting K-complexes...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.detect_kc_btn.setEnabled(False)
        self.write_log("Starting K-complex detection...")

        self.kc_thread = threading.Thread(target=self.detect_kc)
        self.kc_thread.daemon = True
        self.kc_thread.start()

    def detect_kc(self):
        """Detect K-complexes (runs in a thread)."""
        try:
            self.ensure_gui_log_handler()
            data = self.dataset
            if self.annotations and os.path.isfile(self.annot_file_path):
                annot = self.annotations
                self.write_log("Using existing loaded annotations")
            else:
                annot = CustomAnnotations(self.annot_file_path)
                self.annotations = annot
                self.write_log(f"Loaded annotation file: {self.annot_file_path}")

            json_dir = os.path.join(self.output_dir, "wonambi", "kc_results")
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
                self.write_log(f"Created directory: {json_dir}")

            event_processor = ParalKC(dataset=data, annotations=annot,
                                      log_level=logging.INFO, log_file=None)

            params = self.kc_detection_params.copy()

            self.write_log(
                f"Calling detect_kcomplexes with method={params['method']}")
            self.write_log(f"Using {len(params['chan'])} channels")
            self.write_log(f"Sleep stages: {', '.join(params['stage'])}")

            detect_kwargs = dict(params)
            detect_kwargs['cat'] = (1, 1, 1, 0)
            detect_kwargs['json_dir'] = json_dir
            detect_kwargs['save_to_annotations'] = False

            # See the slow wave path for why the database is resolved before
            # detection rather than after it.
            db_path = self.run_db_path()
            subject = self.resolve_subject()
            detect_kwargs['db_path'] = db_path
            detect_kwargs['subject'] = subject
            self.write_log(
                f"K-complex rows will be keyed under subject '{subject}'")

            runs_before = self.db_run_ids(db_path, 'k_complex', params['method'])

            kcomplexes = event_processor.detect_kcomplexes(**detect_kwargs)
            kc_count = self.log_event_count("K-complex", kcomplexes)

            # UNESCAPED method throughout: the shipped default is
            # 'AASM/Massimini2004' and that is exactly what is stored.
            runs_after = self.db_run_ids(db_path, 'k_complex', params['method'])
            new_runs = (None if runs_before is None or runs_after is None
                        else runs_after - runs_before)
            if not new_runs:
                self.write_log(
                    "Could not identify this run's own database run id, so the "
                    "counts below cover every run stored for this method, band "
                    "and stage set - not only this one.")
            kc_rows, kc_channels = self.count_db_events(
                db_path, 'k_complex', params['method'], params['frequency'],
                params['stage'], run_ids=new_runs)

            kc_coverage = self.verify_db_channels(
                "K-complex", db_path, 'k_complex', params['method'],
                params['chan'], params['frequency'], params['stage'])

            self.report_db_density(
                "K-complex", db_path, 'k_complex', params['method'],
                params['frequency'], params['stage'], subject,
                params['reject_artifacts'], params['reject_arousals'])

            self.remember_run_scope(
                "K-complex", db_path=db_path, event_type='k_complex',
                method=params['method'], frequency=params['frequency'],
                stage=list(params['stage']), subject=subject,
                reject_artifacts=params['reject_artifacts'],
                reject_arousals=params['reject_arousals'],
                out_dir=json_dir)

            self.log_run_outcome("K-complex", db_path, kc_count, kc_rows,
                                 kc_channels, kc_coverage)

            try:
                parameters_summary = {
                    'method': params['method'],
                    'frequency_range': params['frequency'],
                    'channels': params['chan'],
                    'stages': params['stage'],
                    'polar': params['polar'],
                    'reject_artifacts': params['reject_artifacts'],
                    'reject_arousals': params['reject_arousals'],
                    'trough_duration': params.get('trough_duration'),
                    'min_isolation': params.get('min_isolation'),
                }
                # The database is where the results are; detection writes no CSV.
                results_summary = {
                    'total_kcomplexes_detected': len(kcomplexes) if 'kcomplexes' in locals() else 0,
                    'channels_requested': len(params['chan']),
                    'channels_in_database': kc_channels,
                    'events_written_to_database': kc_rows,
                    'database_file': db_path,
                    'subject': subject,
                    'channels_missing_from_database': (
                        (kc_coverage or {}).get('missing') or []),
                }
                event_processor.save_detection_summary(
                    output_dir=json_dir, method=params['method'],
                    parameters=parameters_summary,
                    results_summary=results_summary)
            except Exception as e:
                self.write_log(f"Note: Could not save detection summary: {e}")

            QtCore.QMetaObject.invokeMethod(
                self, "finish_kc_detection",
                QtCore.Qt.QueuedConnection)
        except Exception as e:
            self.write_log(f"Error detecting K-complexes: {e}")
            import traceback
            traceback.print_exc()
            QtCore.QMetaObject.invokeMethod(
                self, "show_error", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Failed to detect K-complexes: {e}"))
            QtCore.QMetaObject.invokeMethod(
                self.detect_kc_btn, "setEnabled",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True))
            QtCore.QMetaObject.invokeMethod(
                self.progress, "setVisible",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False))

    @QtCore.pyqtSlot()
    def finish_kc_detection(self):
        """Finish K-complex detection."""
        self.detect_kc_btn.setEnabled(True)
        self.view_kc_results_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("K-complex detection completed")
        self.show_run_finished_dialog("K-complex")
        self.populate_detection_methods()

    def view_kc_results(self):
        """View K-complex detection results."""
        json_dir = os.path.join(self.output_dir, "wonambi", "kc_results")
        if not os.path.isdir(json_dir):
            QMessageBox.critical(
                self, "Error", "K-complex results directory doesn't exist.")
            return

        csv_files = [f for f in os.listdir(json_dir) if f.endswith('.csv')]
        if not csv_files:
            QMessageBox.information(self, "No CSV files", self._no_csv_message())
            return

        viewer = QtWidgets.QDialog(self)
        viewer.setWindowTitle("K-Complex Detection Results")
        viewer.resize(800, 600)
        layout = QVBoxLayout(viewer)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select Result File:"))
        file_combo = QComboBox()
        file_combo.addItems(csv_files)
        file_layout.addWidget(file_combo, 1)
        layout.addLayout(file_layout)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)

        def load_file():
            selected = file_combo.currentText()
            if selected:
                try:
                    with open(os.path.join(json_dir, selected), 'r') as f:
                        text_area.setText(f.read())
                except Exception as e:
                    QMessageBox.critical(viewer, "Error", f"Failed to load: {e}")

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(load_file)
        file_layout.addWidget(load_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(viewer.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        file_combo.setCurrentIndex(0)
        load_file()
        viewer.exec_()

    def add_kc_channels(self):
        selected_items = self.kc_available_list.selectedItems()
        if not selected_items:
            return
        for ch in [item.text() for item in selected_items]:
            if ch not in self.selected_channels:
                self.selected_channels.append(ch)
        self.update_channel_lists()

    def remove_kc_channels(self):
        selected_items = self.kc_selected_list.selectedItems()
        if not selected_items:
            return
        selected = [item.text() for item in selected_items]
        self.selected_channels = [c for c in self.selected_channels if c not in selected]
        self.update_channel_lists()

    def add_all_kc_channels(self):
        self.selected_channels = list(self.available_channels)
        self.update_channel_lists()

    def remove_all_kc_channels(self):
        self.selected_channels = []
        self.update_channel_lists()

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
        self.method_combo.addItems(["Moelle2011", "Ferrarelli2007", "Lacourse2018","Ray2015","Martin2013","Wamsley2012","Nir2011","CIRUS"])
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
        # Tab-specific names: see the note on the slow wave tab's pair.
        self.spindle_reject_artifacts_check = QCheckBox("Reject Artifacts")
        self.spindle_reject_artifacts_check.setChecked(True)
        self.spindle_params_form.addWidget(self.spindle_reject_artifacts_check)

        self.spindle_reject_arousals_check = QCheckBox("Reject Arousals")
        self.spindle_reject_arousals_check.setChecked(True)
        self.spindle_params_form.addWidget(self.spindle_reject_arousals_check)
        
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

        # See the slow wave tab: the flat file is produced on demand from the
        # database, not as a side effect of detection.
        self.export_spindle_csv_btn = QPushButton("Export CSV")
        self.export_spindle_csv_btn.setToolTip(
            "Write the stored spindle events and their density out as CSV "
            "files, from the database. Detection itself writes no CSV.")
        self.export_spindle_csv_btn.clicked.connect(self.export_spindle_csv)
        action_layout.addWidget(self.export_spindle_csv_btn)

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
                "Lacourse2018": "Multi-metric approach combining absolute power, relative power, covariance and correlation.",
                "CIRUS": "Hilbert-envelope thresholding ported from the qEEG_PSG Java tool. Threshold = median + alpha * std of the envelope. Validated in D'Rozario 2022 / Lam 2021. Designed for C3-M2 in N2/N3."
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

            elif method_name == "CIRUS":
                thresh_group = QGroupBox("CIRUS Thresholds")
                thresh_layout = QVBoxLayout()

                # Alpha (det_thresh) — multiplier on envelope std
                alpha_layout = QHBoxLayout()
                alpha_layout.addWidget(QLabel("Alpha (threshold sensitivity):"))
                alpha_spin = QDoubleSpinBox()
                alpha_spin.setRange(0.1, 5.0)
                alpha_spin.setSingleStep(0.1)
                alpha_spin.setValue(detector.det_thresh)
                alpha_spin.setToolTip(
                    "threshold = median + alpha * std of the Hilbert envelope. "
                    "Default 1.0 from D'Rozario 2022 / Lam 2021. "
                    "Use 1.4 for OSA cohorts (better F1 per CIRUS validation).")
                alpha_layout.addWidget(alpha_spin)
                thresh_layout.addLayout(alpha_layout)

                # Background ratio (sel_thresh)
                bg_layout = QHBoxLayout()
                bg_layout.addWidget(QLabel("Background ratio:"))
                bg_spin = QDoubleSpinBox()
                bg_spin.setRange(0.0, 1.0)
                bg_spin.setSingleStep(0.05)
                bg_spin.setValue(detector.sel_thresh)
                bg_spin.setToolTip(
                    "Reject candidate if surrounding-window mean is not below "
                    "this * spindle mean. Set to 0 to disable.")
                bg_layout.addWidget(bg_spin)
                thresh_layout.addLayout(bg_layout)

                thresh_group.setLayout(thresh_layout)
                self.spindle_params_layout.addWidget(thresh_group)
                self.spindle_param_widgets["det_thresh"] = alpha_spin
                self.spindle_param_widgets["sel_thresh"] = bg_spin

                # Filter pipeline mode
                filter_group = QGroupBox("Filter Pipeline")
                filter_layout = QHBoxLayout()
                filter_layout.addWidget(QLabel("Mode:"))
                filter_combo = QComboBox()
                filter_combo.addItems(["java", "wonambi"])
                filter_combo.setCurrentText(detector.filter_mode)
                filter_combo.setToolTip(
                    "'java' reproduces the original CIRUS implementation "
                    "(scipy firwin Hamming + fftconvolve + Hilbert). "
                    "'wonambi' uses Wonambi's remez+filtfilt pipeline.")
                filter_layout.addWidget(filter_combo)
                filter_group.setLayout(filter_layout)
                self.spindle_params_layout.addWidget(filter_group)
                self.spindle_param_widgets["filter_mode"] = filter_combo

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

        # Theta-Gamma coupling is not implemented yet. Keep the entry visible so
        # the roadmap stays legible, but grey it out: clearing Qt.ItemIsEnabled on
        # the combo's model item makes it unselectable by both mouse and keyboard,
        # so update_pac_params() can never be called with "Theta-Gamma".
        theta_gamma_index = self.pac_method_combo.findText("Theta-Gamma")
        theta_gamma_item = self.pac_method_combo.model().item(theta_gamma_index)
        theta_gamma_item.setFlags(theta_gamma_item.flags() & ~QtCore.Qt.ItemIsEnabled)
        self.pac_method_combo.setItemData(
            theta_gamma_index,
            "Theta-Gamma coupling is not implemented yet; only SW-Spindle "
            "coupling can be run in this version.",
            QtCore.Qt.ToolTipRole)
        # Make sure the enabled entry is the one selected on startup.
        self.pac_method_combo.setCurrentIndex(
            self.pac_method_combo.findText("SW-Spindle"))

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
        self.tabs.setTabEnabled(4, True)  # K-Complex tab
        self.tabs.setTabEnabled(5, True)  # PAC tab

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

        # Clear K-Complex tab listboxes if they exist
        if hasattr(self, 'kc_available_list') and self.kc_available_list is not None:
            self.kc_available_list.clear()
        if hasattr(self, 'kc_selected_list') and self.kc_selected_list is not None:
            self.kc_selected_list.clear()

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
                # Also add to K-Complex tab if it exists
                if hasattr(self, 'kc_available_list') and self.kc_available_list is not None:
                    self.kc_available_list.addItem(channel)

        # Add selected channels to spindle tab
        for channel in self.selected_channels:
            self.selected_list.addItem(channel)
            # Also add to SW tab if it exists
            if hasattr(self, 'sw_selected_list') and self.sw_selected_list is not None:
                self.sw_selected_list.addItem(channel)
            # Also add to K-Complex tab if it exists
            if hasattr(self, 'kc_selected_list') and self.kc_selected_list is not None:
                self.kc_selected_list.addItem(channel)
    
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
            # Currently unreachable: the "Theta-Gamma" combo entry is disabled in
            # setup_pac_tab() because the analysis is not implemented. Kept so the
            # parameter page wiring is ready for whoever implements it.
            self.pac_param_stack.setCurrentIndex(1)

            # Disable SW and spindle method selection
            self.sw_method_pac_combo.setEnabled(False)
            self.spindle_method_pac_combo.setEnabled(False)

            # Channel population is still to be written: update_pac_available_channels()
            # derives channels from the detected SW/spindle events in the database,
            # which does not apply to a continuous-band analysis.

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
                    conn = connect_events_db(db_path)
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
                    conn = connect_events_db(db_path)
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
                self.write_log_once('pac_channel_lookup',
                                    "Could not find method information")
                return
            
            # Get method parameters
            sw_base_method = sw_info.get('method')
            sw_stage = sw_info.get('stage')
            spindle_base_method = spindle_info.get('method')
            spindle_stage = spindle_info.get('stage')
            
            if not sw_base_method or not spindle_base_method or not sw_stage or not spindle_stage:
                self.write_log_once('pac_channel_lookup',
                                    "Missing method parameters")
                return

            # Check if stages match. Reported once per actual mismatch: this
            # method is re-entered on every combo change and refresh, and the
            # same warning about the same pair of selections says nothing new.
            # Passing None on a match clears the remembered state, so a later
            # mismatch is warned about again.
            self.write_log_once(
                'pac_stage_mismatch',
                (f"Warning: Sleep stages don't match - SW: {sw_stage}, "
                 f"Spindle: {spindle_stage}")
                if sw_stage != spindle_stage else None)

            # Query database for channels that have both slow waves and spindles with these methods and stages
            db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
            if not os.path.exists(db_path):
                self.write_log_once(
                    'pac_channel_lookup',
                    "Database not found. Cannot update available channels.")
                return
            
            conn = connect_events_db(db_path)
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

            # The lookup got through, so forget any earlier failure message:
            # if the same problem recurs it should be reported again.
            self.write_log_once('pac_channel_lookup', None)

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
            self.write_log_once('pac_channel_lookup',
                                f"Error updating available channels: {str(e)}")
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
        """Populate channel list for PAC analysis from database with optimized query"""
        if not hasattr(self, 'database_channels'):
            # Get channels from database
            try:
                db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
                if os.path.exists(db_path):
                    # write=True: _ensure_database_indexes issues CREATE INDEX.
                    conn = connect_events_db(db_path, write=True)
                    cursor = conn.cursor()

                    # Ensure indexes exist for better performance
                    self._ensure_database_indexes(cursor)
                    
                    # Optimized query with index hint
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

    def _ensure_database_indexes(self, cursor):
        """Create database indexes for improved query performance on large datasets"""
        try:
            # Create composite index for event_type + method + freq + stage queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_composite
                ON events(event_type, method, freq_lower, freq_upper, stage)
            """)
            
            # Create index for event_type filtering (most selective first)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type)
            """)
            
            # Create index for method filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_method
                ON events(method)
            """)
            
            # Create index for stage filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_stage
                ON events(stage)
            """)
            
            # Create index for channel queries (used in PAC analysis)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_channel
                ON events(channel)
            """)
            
            # Create composite index for PAC channel queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_pac_channels
                ON events(event_type, method, stage, channel)
            """)
            
            # Once per session: the statements are CREATE INDEX IF NOT EXISTS
            # and this runs on every method refresh, so repeating the line adds
            # nothing but noise.
            self.write_log_once(
                'db_indexes',
                "Database indexes created/verified for improved performance")

        except Exception as e:
            self.write_log_once(
                'db_indexes',
                f"Warning: Could not create database indexes: {str(e)}")

    def populate_detection_methods(self):
        """Populate detection method lists from database with optimized queries"""
        db_path = os.path.join(self.output_dir, "wonambi", "neural_events.db")
        if not os.path.exists(db_path):
            # This runs on every PAC tab switch and after every detection run;
            # only report when the answer changes.
            self.write_log_once(
                'pac_methods_loaded',
                "Database not found. Cannot load detection methods.")
            return

        try:
            # write=True: _ensure_database_indexes issues CREATE INDEX.
            conn = connect_events_db(db_path, write=True)
            cursor = conn.cursor()

            # Create performance indexes if they don't exist
            self._ensure_database_indexes(cursor)
            
            # Optimized single query to get both slow wave and spindle methods
            # Using UNION ALL for better performance than separate queries
            cursor.execute("""
                SELECT event_type, method, freq_lower, freq_upper, stage, COUNT(*) as event_count
                FROM events
                WHERE event_type IN ('slow_wave', 'spindle')
                GROUP BY event_type, method, freq_lower, freq_upper, stage
                ORDER BY event_type, method, freq_lower, freq_upper, stage
            """)
            all_results = cursor.fetchall()
            
            # Separate results by event type
            sw_results = [(method, freq_lower, freq_upper, stage, count)
                         for event_type, method, freq_lower, freq_upper, stage, count in all_results
                         if event_type == 'slow_wave']
            
            spindle_results = [(method, freq_lower, freq_upper, stage, count)
                              for event_type, method, freq_lower, freq_upper, stage, count in all_results
                              if event_type == 'spindle']
                
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



            # Update combo boxes. Signals are blocked across the rebuild: a
            # clear() followed by addItems() fires currentIndexChanged twice
            # per combo, and each of those re-ran the whole channel lookup.
            # The explicit update_*_freq_from_db calls below do that work once.
            for combo, names in ((self.sw_method_pac_combo, sw_display_names),
                                 (self.spindle_method_pac_combo, spindle_display_names)):
                blocked = combo.blockSignals(True)
                try:
                    combo.clear()
                    combo.addItems(names)
                finally:
                    combo.blockSignals(blocked)

            # Connect selection changes to update channels - exactly once per
            # session. This method is re-run after every detection and on every
            # PAC tab switch, and reconnecting each time stacked up duplicate
            # slots, so one combo change fired the handler N times.
            if not getattr(self, '_pac_combo_signals_connected', False):
                self.sw_method_pac_combo.currentIndexChanged.connect(self.update_pac_available_channels)
                self.spindle_method_pac_combo.currentIndexChanged.connect(self.update_pac_available_channels)
                self._pac_combo_signals_connected = True

            # Update frequency labels if methods are available
            if self.sw_method_pac_combo.count() > 0:
                self.update_sw_freq_from_db(self.sw_method_pac_combo.currentText())
            
            if self.spindle_method_pac_combo.count() > 0:
                self.update_spindle_freq_from_db(self.spindle_method_pac_combo.currentText())
            
            self.write_log_once(
                'pac_methods_loaded',
                f"Loaded {len(sw_results)} slow wave methods and "
                f"{len(spindle_results)} spindle methods from database")

        except Exception as e:
            self.write_log_once(
                'pac_methods_loaded',
                f"Error loading detection methods: {str(e)}")
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
            self.ensure_gui_log_handler()

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


            # Resolve the subject the PAC rows will be keyed under, and say so
            # in the log: without it the results go to CSV only and never
            # reach neural_events.db.
            subject = self.resolve_subject()
            self.write_log(
                f"PAC results will be written to the database at "
                f"{params['db_path']} under subject '{subject}'")

            # Setup event options if using SW-Spindle coupling
            event_opts = {}
            if params['method'] == "SW-Spindle":
                event_opts = {
                    'buffer': params['time_window'],
                    'sw_method': params['sw_method'],
                    'spindle_method': params['spindle_method']
                }

            # Label for a continuous-data run. There are no detected events to
            # derive a scope from, so analyze_pac would otherwise fall back to
            # event_type='slow_wave', method='unknown' and file a theta-gamma
            # result as slow-wave coupling. Describe what was actually
            # analysed, and derive it from the configured bands so the label
            # stays correct for any continuous band pair, not just Theta-Gamma.
            continuous_method = (
                f"{params['method']}"
                f"_phase{fmt_freq_token(*params['phase_freq'])}"
                f"_amp{fmt_freq_token(*params['amp_freq'])}")

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
                    event_opts=event_opts,
                    write_db=True,
                    subject=subject
                )
            else:
                # For other coupling types (e.g., Theta-Gamma)
                self.write_log(f"Running {params['method']} coupling analysis...")
                self.write_log(
                    f"Continuous-data run: rows will be stored as "
                    f"event_type='continuous', method='{continuous_method}'")
                results = pac_processor.analyze_pac(
                    chan=params['channels'],
                    stage=params['stages'],
                    phase_freq=params['phase_freq'],
                    amp_freq=params['amp_freq'],
                    idpac=params['idpac'],
                    use_detected_events=False,  # Use continuous data
                    time_window=params['time_window'],
                    db_path=params['db_path'],
                    out_dir=pac_dir,
                    write_db=True,
                    subject=subject,
                    stored_event_type='continuous',
                    stored_method=continuous_method
                )
            

            # For the continuous branch the CSV exporter names its output
            # directory after 'spindle_method', so give it the same descriptive
            # label the database row gets instead of letting it write to
            # pac_results/unknown/.
            method_info = {
                'sw_method': event_opts.get('sw_method', 'unknown') if event_opts else 'unknown',
                'spindle_method': (event_opts.get('spindle_method', 'unknown')
                                   if event_opts else continuous_method),
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

            # Confirm the rows actually landed under the labels this run
            # should have written, so a database write that silently did
            # nothing cannot look like a successful run. The SW-Spindle scope
            # is derived inside analyze_pac from event_opts, so it is checked
            # by breakdown rather than against a scope rebuilt here.
            expect_scope = (None if params['method'] == "SW-Spindle"
                            else ('continuous', continuous_method))
            self.verify_pac_rows(params['db_path'], subject,
                                 expect_scope=expect_scope)

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
                    with open(file_path, 'r', encoding='utf-8') as f:
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
            
            with open(self.annot_file_path, 'r', encoding='utf-8') as f:
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
        
        # Get method-specific parameters. Spinboxes use .value(); combos
        # (e.g. CIRUS filter_mode) use .currentText().
        method_params = {}
        if hasattr(self, 'spindle_param_widgets'):
            for param, widget in self.spindle_param_widgets.items():
                if isinstance(widget, QComboBox):
                    method_params[param] = widget.currentText()
                else:
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
            self.ensure_gui_log_handler()
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
            
            event_processor = ParalEvents(dataset=data, annotations=annot,
                                          log_level=logging.INFO, log_file=None)

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
                                "moving_sd": {"dur": None}},
                    # CIRUS — det_thresh / sel_thresh already mapped above
                    "filter_mode": "filter_mode",
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

            # Read once, then use that reading for the detection, the density
            # denominator and the summary alike, so the three cannot disagree
            # (and a checkbox toggled mid-run cannot change the meaning of a
            # run that is already going).
            reject_artifacts = self.spindle_reject_artifacts_check.isChecked()
            reject_arousals = self.spindle_reject_arousals_check.isChecked()

            # See the slow wave path for why the database is resolved before
            # detection rather than after it.
            db_path = self.run_db_path()
            subject = self.resolve_subject()
            self.write_log(
                f"Spindle rows will be keyed under subject '{subject}'")

            runs_before = self.db_run_ids(db_path, 'spindle',
                                          self.spindle_method)

            # Detect spindles. write_db is left at its default (None = the
            # database), so no per-channel JSON is written.
            spindles = event_processor.detect_spindles(
                method=self.spindle_method,
                chan=self.selected_channels,
                frequency=freq_range,
                duration=duration_range,
                stage=selected_stages,
                reject_artifacts=reject_artifacts,
                reject_arousals=reject_arousals,
                cat=(1, 1, 1, 0),  # concatenate within and between stages, cycles separate
                polar=polar,
                save_to_annotations=False,
                json_dir=json_dir,
                db_path=db_path,
                subject=subject,
                **custom_params
            )
            spindle_count = self.log_event_count("Spindle", spindles)

            # The method is used UNESCAPED for every database query, which is
            # how the detector stores it in events.method.
            runs_after = self.db_run_ids(db_path, 'spindle',
                                         self.spindle_method)
            new_runs = (None if runs_before is None or runs_after is None
                        else runs_after - runs_before)
            if not new_runs:
                self.write_log(
                    "Could not identify this run's own database run id, so the "
                    "counts below cover every run stored for this method, band "
                    "and stage set - not only this one.")
            spindle_rows, spindle_channels = self.count_db_events(
                db_path, 'spindle', self.spindle_method, freq_range,
                selected_stages, run_ids=new_runs)

            spindle_coverage = self.verify_db_channels(
                "Spindle", db_path, 'spindle', self.spindle_method,
                self.selected_channels, freq_range, selected_stages)

            self.report_db_density(
                "Spindle", db_path, 'spindle', self.spindle_method, freq_range,
                selected_stages, subject, reject_artifacts, reject_arousals)

            self.remember_run_scope(
                "Spindle", db_path=db_path, event_type='spindle',
                method=self.spindle_method, frequency=freq_range,
                stage=list(selected_stages), subject=subject,
                reject_artifacts=reject_artifacts,
                reject_arousals=reject_arousals,
                out_dir=json_dir)

            self.log_run_outcome("Spindle", db_path, spindle_count,
                                 spindle_rows, spindle_channels,
                                 spindle_coverage)
            
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
                    'reject_artifacts': reject_artifacts,
                    'reject_arousals': reject_arousals,
                    'method_specific_parameters': method_params if 'method_params' in locals() else {}
                }
                
                # The database is where the results are; detection writes no CSV.
                results_summary = {
                    'total_spindles_detected': len(spindles) if 'spindles' in locals() else 0,
                    'channels_requested': len(self.selected_channels),
                    'channels_in_database': spindle_channels,
                    'events_written_to_database': spindle_rows,
                    'database_file': db_path,
                    'subject': subject,
                    'channels_missing_from_database': (
                        (spindle_coverage or {}).get('missing') or []),
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
        self.show_run_finished_dialog("Slow wave")
        # Update PAC detection methods
        self.populate_detection_methods()

    @QtCore.pyqtSlot()
    def finish_spindle_detection(self):
        """Finish spindle detection"""
        self.detect_btn.setEnabled(True)
        self.view_results_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Spindle detection completed")
        self.show_run_finished_dialog("Spindle")
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
            QMessageBox.information(self, "No CSV files", self._no_csv_message())
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
                    with open(file_path, 'r', encoding='utf-8') as f:
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
            QMessageBox.information(self, "No CSV files", self._no_csv_message())
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
                    with open(file_path, 'r', encoding='utf-8') as f:
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


    # ------------------------------------------------------------------
    # Database-era run plumbing
    #
    # Detection writes straight into neural_events.db and no longer produces
    # per-channel JSON, so there is no CSV to export, no CSV to import, and
    # nothing on the filesystem to fingerprint. What replaces that chain is:
    # resolve the target before detecting, then read the database back
    # afterwards and report what is actually in it.
    # ------------------------------------------------------------------

    DB_FILENAME = "neural_events.db"

    def run_db_path(self):
        """Resolve, and log, the database a detection run will write to.

        ``resolve_db_target`` raises rather than downgrading an unwritable
        target to a no-op, and it is called here - before any channel is
        detected - so an unusable output directory costs a dialog rather than
        an hour of detection whose results have nowhere to go.

        Returns
        -------
        str
            Absolute path of the database file.
        """
        db_path = resolve_db_target(
            db_path=os.path.join(self.output_dir, "wonambi", self.DB_FILENAME))
        self.write_log(f"Results for this run go to the database: {db_path}")
        return db_path

    def db_run_ids(self, db_path, event_type, method):
        """The ``detection_runs`` ids already recorded for one scope.

        Snapshotted before detection and again after, so the difference names
        exactly the run that just happened. That matters for the closing
        report: ``events`` rows are upserted on a deterministic uuid5, so a
        scope-wide count cannot tell "this run wrote 12000 events" from "an
        earlier run left 12000 events and this one wrote none". Rows carry the
        run_id of whichever run last wrote them, so counting by the new run_id
        is the honest question.

        Parameters
        ----------
        db_path : str
            Database the run writes to. Need not exist yet.
        event_type : str
            ``'spindle'``, ``'slow_wave'`` or ``'k_complex'``.
        method : str
            Detection method as stored, UNESCAPED.

        Returns
        -------
        set of str or None
            The known run ids, and ``None`` when the database could not be read.
            A database file that does not exist yet returns an empty set, not
            ``None``: there are demonstrably no earlier runs, so the first run
            against a fresh output directory is identified exactly rather than
            warned about. ``None`` is reserved for "cannot tell" - an
            unreadable file, or one with no ``detection_runs`` table (a
            database predating the direct-write path) - and the caller then
            degrades to a scope-wide count and says so.
        """
        if not db_path:
            return None
        if not os.path.exists(db_path):
            return set()
        conn = None
        try:
            conn = connect_events_db(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name='detection_runs'")
            if cur.fetchone() is None:
                return None
            cur.execute("SELECT run_id FROM detection_runs "
                        "WHERE event_type = ? AND method = ?",
                        (str(event_type), str(method)))
            return {str(r[0]) for r in cur.fetchall()}
        except Exception:
            return None
        finally:
            if conn is not None:
                conn.close()

    def count_db_events(self, db_path, event_type, method, frequency,
                        stage_list, run_ids=None):
        """Count the rows a detection run left in ``events``.

        Parameters
        ----------
        db_path : str
            Database to read.
        event_type, method : str
            Scope of the run; `method` UNESCAPED, as stored.
        frequency : tuple of float
            Band ``(lo, hi)`` of the run.
        stage_list : list of str
            Stages the run was scoped to. ``events.stage`` holds each event's
            own epoch stage, so this is an ``IN`` filter, not the joined token.
        run_ids : set of str or None
            When given, restrict to these ``run_id`` values - the rows this run
            wrote. ``None`` counts the whole scope regardless of run, which
            over-reports after a re-run with a narrower channel set; callers
            pass ``None`` only when the run ids could not be determined, and
            say so in the log.

        Returns
        -------
        tuple of (int, int) or (None, None)
            ``(n_events, n_channels)``, or ``(None, None)`` when the database
            could not be read - which is reported as "cannot tell", never as
            zero.
        """
        if not db_path or not os.path.exists(db_path):
            return (None, None)
        where = ["event_type = ?", "method = ?"]
        params = [str(event_type), str(method)]
        if frequency is not None:
            where += ["freq_lower = ?", "freq_upper = ?"]
            params += [float(frequency[0]), float(frequency[1])]
        if stage_list:
            where.append("stage IN (%s)" % ",".join("?" * len(stage_list)))
            params += [str(s) for s in stage_list]
        if run_ids:
            run_ids = sorted(run_ids)
            where.append("run_id IN (%s)" % ",".join("?" * len(run_ids)))
            params += run_ids
        conn = None
        try:
            conn = connect_events_db(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*), COUNT(DISTINCT channel) FROM events "
                "WHERE " + " AND ".join(where), params)
            row = cur.fetchone()
        except Exception as e:
            self.write_log(
                f"Could not count events in {db_path}: {type(e).__name__}: {e}")
            return (None, None)
        finally:
            if conn is not None:
                conn.close()
        if row is None:
            return (0, 0)
        return (int(row[0] or 0), int(row[1] or 0))

    def verify_db_channels(self, label, db_path, event_type, method, channels,
                           frequency, stage_list):
        """Check every requested channel is accounted for in the database.

        Delegates to :func:`turtlewave_hdEEG.dbwrite.verify_channel_coverage`,
        which counts a channel as covered when it has events for the scope OR a
        ``processing_status`` row recording ``success = 1`` - so a channel that
        legitimately found nothing is not mistaken for one that crashed. The
        failures it reports are the partial-run case that a bare event count
        cannot see: 200 channels of 257 returning results still yields a large,
        healthy-looking total.

        Parameters
        ----------
        label : str
            Event name for the log lines, e.g. ``"Spindle"``.
        db_path : str
            Database the run wrote to.
        event_type, method : str
            Scope of the run; `method` UNESCAPED.
        channels : list of str
            Channels the run was asked to process.
        frequency : tuple of float
            Band ``(lo, hi)`` of the run.
        stage_list : list of str
            Stages of the run. Joined into the scope token the detectors write
            to ``processing_status`` (``"".join(stage)``), which is how all
            three processors key it.

        Returns
        -------
        dict or None
            The ``verify_channel_coverage`` result, or ``None`` when the check
            itself failed (reported, and treated by the caller as "unknown"
            rather than as success).
        """
        stage_key = "".join(str(s) for s in stage_list) if stage_list else "all"
        try:
            coverage = verify_channel_coverage(
                db_path, event_type, method, list(channels or []),
                frequency[0], frequency[1], stage_key)
        except Exception as e:
            self.write_log(
                f"Could not verify {label} channel coverage in {db_path}: "
                f"{type(e).__name__}: {e}. Treat the channel counts below as "
                f"unknown.")
            return None

        if not coverage.get('scoped_status', True):
            self.write_log(
                f"{label} coverage check ran against an unmigrated "
                f"processing_status table, so a status row from another method "
                f"or band cannot be told apart from this run's.")

        failed = coverage.get('failed') or []
        missing = coverage.get('missing') or []
        events_only = coverage.get('events_only') or []

        if failed:
            self.write_log(
                f"{label} channels that FAILED during detection "
                f"({len(failed)} of {coverage['requested']}): "
                f"{self._abbrev_channels(failed)}")
        silent = [c for c in missing if c not in set(failed)]
        if silent:
            self.write_log(
                f"{label} channels with neither events nor a completion record "
                f"({len(silent)} of {coverage['requested']}): "
                f"{self._abbrev_channels(silent)}")
        if events_only:
            self.write_log(
                f"{label}: {len(events_only)} channel(s) are counted as done on "
                f"the strength of stored events alone, with no completion "
                f"record for this exact stage scope. Those events may predate "
                f"this run.")
        return coverage

    @staticmethod
    def _abbrev_channels(channels, limit=12):
        """Join channel names for a log line, truncating a long list."""
        names = [str(c) for c in channels]
        if len(names) <= limit:
            return ", ".join(names)
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"

    def report_db_density(self, label, db_path, event_type, method, frequency,
                          stage_list, subject, reject_artifacts,
                          reject_arousals):
        """Log per-stage density read back from the database.

        Replaces the JSON-reading ``export_*_density_to_csv`` exporters, which
        can no longer work: their input was the per-channel JSON detection no
        longer writes. :func:`turtlewave_hdEEG.density.event_density` is the
        single definition of density now - its denominator is the
        ``analysed_time`` row the run itself stored, i.e. the artefact-free time
        the detector actually searched, not raw hypnogram time.

        ``missing='nan'`` is deliberate: a denominator that could not be stored
        is a reporting problem, and taking down the closing report of a
        detection run that otherwise succeeded would hide the more important
        news. The library warns, the density reads ``nan``, and the run is still
        reported.

        Parameters
        ----------
        label : str
            Event name for the log lines.
        db_path : str
            Database to read.
        event_type, method : str
            Scope of the run; `method` UNESCAPED.
        frequency : tuple of float
            Band ``(lo, hi)`` of the run.
        stage_list : list of str
            Stages of the run.
        subject : str
            Subject the ``analysed_time`` denominator is keyed under.
        reject_artifacts, reject_arousals : bool
            The run's rejection settings. They select the denominator row, so
            passing the run's own values is what keeps numerator and
            denominator on the same time base.

        Returns
        -------
        pandas.DataFrame or None
            The density table, or ``None`` when it could not be computed.
        """
        try:
            df = event_density(
                db_path, event_type=event_type, method=method,
                stage=list(stage_list) if stage_list else None,
                freq_lower=frequency[0], freq_upper=frequency[1],
                subject=subject,
                reject_artifacts=reject_artifacts,
                reject_arousals=reject_arousals,
                missing='nan')
        except Exception as e:
            self.write_log(
                f"{label} density could not be computed from {db_path}: "
                f"{type(e).__name__}: {e}. The detected events are unaffected.")
            return None

        if df is None or not len(df):
            self.write_log(
                f"No {label} density to report: no stored events matched this "
                f"run's scope.")
            return df

        self.write_log(
            f"{label} density (events per artefact-free minute, from "
            f"{db_path}):")
        for stage_name, grp in df.groupby('stage', dropna=False):
            dens = grp['density_per_min'].dropna()
            minutes = grp['analysed_minutes'].dropna()
            if not len(dens):
                self.write_log(
                    f"    {stage_name}: {len(grp)} channel(s), "
                    f"{int(grp['n_events'].sum())} events, but no stored "
                    f"denominator, so density is undefined for this stage.")
                continue
            self.write_log(
                f"    {stage_name}: {len(grp)} channel(s), "
                f"{int(grp['n_events'].sum())} events over "
                f"{minutes.iloc[0]:.1f} analysed min, "
                f"median {dens.median():.2f}/min "
                f"(range {dens.min():.2f}-{dens.max():.2f})")
        return df

    def remember_run_scope(self, label, **scope):
        """Store the scope of the last run so the CSV export can reuse it."""
        if getattr(self, '_last_run_scope', None) is None:
            self._last_run_scope = {}
        self._last_run_scope[label] = dict(scope)
        return self._last_run_scope[label]

    # ------------------------------------------------------------------
    # Explicit database -> CSV export
    #
    # Deliberately a button, not a side effect of detection. A CSV written
    # automatically after every run is a second copy of the results that goes
    # stale the moment anything is re-detected or a channel is dropped in
    # review, and the whole point of moving to one store of record was to stop
    # having two. Collaborators who need a flat file still get one, on request,
    # and it is dated from the database as it stands when they ask.
    # ------------------------------------------------------------------

    def _no_csv_message(self):
        """Explain an empty results directory instead of calling it an error.

        A results directory with no CSV in it used to mean detection had gone
        wrong. Since the database became the store of record it is the normal
        state, and reporting it as an error sends the reader looking for a
        failure that did not happen.
        """
        db_path = os.path.join(self.output_dir or "", "wonambi",
                               self.DB_FILENAME)
        return (
            "No CSV files here - and that is expected. Detection writes its "
            "results to the database:\n\n"
            f"{db_path}\n\n"
            "Use \"Export CSV\" on this tab to write a flat file from it, or "
            "open the Event Review GUI to inspect the events.")

    def db_scopes_for(self, db_path, event_type):
        """List the (method, band, stages) scopes stored for one event type.

        Parameters
        ----------
        db_path : str
            Database to read.
        event_type : str
            Event type to list scopes for.

        Returns
        -------
        list of dict
            One dict per scope with keys ``method``, ``frequency``, ``stage``
            and ``n_events``, ordered by method then band. Empty when there is
            nothing stored (or nothing readable).
        """
        if not db_path or not os.path.exists(db_path):
            return []
        conn = None
        try:
            conn = connect_events_db(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT method, freq_lower, freq_upper, "
                "       GROUP_CONCAT(DISTINCT stage), COUNT(*) "
                "FROM events WHERE event_type = ? "
                "GROUP BY method, freq_lower, freq_upper "
                "ORDER BY method, freq_lower, freq_upper", (str(event_type),))
            rows = cur.fetchall()
        except Exception as e:
            self.write_log(
                f"Could not list stored scopes in {db_path}: "
                f"{type(e).__name__}: {e}")
            return []
        finally:
            if conn is not None:
                conn.close()

        scopes = []
        for method, lo, hi, stages, count in rows:
            stage_list = sorted(s for s in str(stages or '').split(',') if s)
            scopes.append({
                'method': str(method),
                'frequency': (float(lo), float(hi)),
                'stage': stage_list,
                'n_events': int(count or 0),
            })
        return scopes

    def export_scope_csv(self, label, event_type):
        """Write the events and density of one stored scope out as CSV.

        Uses this session's last run for `label` when there was one, and
        otherwise offers whatever scopes the database already holds - so the
        button works after restarting the GUI, or on a database somebody else
        produced.

        Two files are written: the events table
        (:func:`turtlewave_hdEEG.dbwrite.export_events_to_csv`, the same column
        shape the old parameters CSV had) and the per-channel density
        (:func:`turtlewave_hdEEG.density.event_density`). Both are named after
        the scope, and both paths are reported.

        Parameters
        ----------
        label : str
            The run label, e.g. ``"Spindle"``.
        event_type : str
            Event type as stored: ``'spindle'``, ``'slow_wave'``,
            ``'k_complex'``.
        """
        if not self.output_dir:
            QMessageBox.warning(self, "No output directory",
                                "Set an output directory first.")
            return

        scope = (getattr(self, '_last_run_scope', None) or {}).get(label)
        if scope is None:
            try:
                db_path = resolve_db_target(
                    db_path=os.path.join(self.output_dir, "wonambi",
                                         self.DB_FILENAME))
            except ValueError as e:
                QMessageBox.critical(self, "No database", str(e))
                return
            scope = self._choose_db_scope(label, db_path, event_type)
            if scope is None:
                return

        db_path = scope['db_path']
        out_dir = scope.get('out_dir') or os.path.dirname(db_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Export failed",
                                 f"Could not create {out_dir}: {e}")
            return

        method_str = str(scope['method']).replace('/', '_')
        freq_str = fmt_freq_token(scope['frequency'][0], scope['frequency'][1])
        stages_str = "".join(scope['stage']) if scope['stage'] else "all"
        base = f"{event_type}_{method_str}_{freq_str}_{stages_str}"
        events_csv = os.path.join(out_dir, f"{base}_events.csv")
        density_csv = os.path.join(out_dir, f"{base}_density.csv")

        written = []
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.write_log(
                f"Exporting {label} scope method={scope['method']!r}, "
                f"{freq_str}, stages={stages_str} from {db_path} to CSV")
            path = export_events_to_csv(
                db_path, event_type, scope['method'], scope['frequency'],
                scope['stage'] or None, csv_file=events_csv)
            if path:
                written.append(path)
                self.write_log(f"{label} events CSV written: {path}")
            else:
                self.write_log(
                    f"No {label} events matched that scope, so no events CSV "
                    f"was written.")

            df = event_density(
                db_path, event_type=event_type, method=scope['method'],
                stage=scope['stage'] or None,
                freq_lower=scope['frequency'][0],
                freq_upper=scope['frequency'][1],
                subject=scope.get('subject'),
                reject_artifacts=scope.get('reject_artifacts', True),
                reject_arousals=scope.get('reject_arousals', True),
                missing='nan')
            if df is not None and len(df):
                df.to_csv(density_csv, index=False)
                written.append(density_csv)
                self.write_log(f"{label} density CSV written: {density_csv}")
            else:
                self.write_log(
                    f"No {label} density to export for that scope, so no "
                    f"density CSV was written.")
        except Exception as e:
            import traceback
            self.write_log("!" * 60)
            self.write_log(f"!!! {label} CSV export FAILED: "
                           f"{type(e).__name__}: {e}")
            for line in traceback.format_exc().rstrip().splitlines():
                self.write_log(f"!!!   {line}")
            self.write_log("!" * 60)
            QMessageBox.critical(
                self, "Export failed",
                f"Could not export {label} results to CSV:\n\n"
                f"{type(e).__name__}: {e}\n\nSee the Log tab.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if written:
            QMessageBox.information(
                self, "Export complete",
                f"{label} results exported from the database:\n\n"
                + "\n".join(written)
                + "\n\nThe database remains the store of record; these files "
                  "are a snapshot of it.")
        else:
            QMessageBox.warning(
                self, "Nothing exported",
                f"No {label} events matched that scope in\n{db_path}\n\n"
                f"Nothing was written. See the Log tab.")

    def _choose_db_scope(self, label, db_path, event_type):
        """Ask which stored scope to export when this session has no run.

        Returns
        -------
        dict or None
            A scope dict in the shape `remember_run_scope` stores, or ``None``
            when the user cancelled or there is nothing to export.
        """
        scopes = self.db_scopes_for(db_path, event_type)
        if not scopes:
            QMessageBox.information(
                self, "Nothing to export",
                f"No {label.lower()} events are stored in\n{db_path}\n\n"
                f"Run detection first.")
            return None

        entries = [
            f"{s['method']}  ·  "
            f"{fmt_freq_token(s['frequency'][0], s['frequency'][1])}  ·  "
            f"{'+'.join(s['stage']) if s['stage'] else 'all stages'}  ·  "
            f"{s['n_events']} events"
            for s in scopes]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, f"Export {label} results",
            "This session has not run a detection, so choose which stored "
            "result to export:", entries, 0, False)
        if not ok:
            return None
        selected = scopes[entries.index(choice)]
        # The rejection settings of the original run are not recoverable from
        # the events rows, and they select the density denominator. The
        # detector defaults are assumed and said so, rather than quietly
        # picking one: a wrong guess makes event_density report a missing
        # denominator, which is visible, not a silently biased number.
        self.write_log(
            f"Exporting a stored {label.lower()} scope from an earlier "
            f"session. The density denominator is looked up assuming the run "
            f"used reject_artifacts=True and reject_arousals=True (the "
            f"detector defaults). If it did not, the density columns will "
            f"report no stored denominator rather than a wrong number.")
        return {
            'db_path': db_path,
            'event_type': event_type,
            'method': selected['method'],
            'frequency': selected['frequency'],
            'stage': selected['stage'],
            'subject': None,
            'reject_artifacts': True,
            'reject_arousals': True,
            'out_dir': os.path.dirname(db_path),
        }

    def export_spindle_csv(self):
        """Export the spindle results from the database as CSV."""
        self.export_scope_csv("Spindle", 'spindle')

    def export_sw_csv(self):
        """Export the slow wave results from the database as CSV."""
        self.export_scope_csv("Slow wave", 'slow_wave')

    def export_kc_csv(self):
        """Export the K-complex results from the database as CSV."""
        self.export_scope_csv("K-complex", 'k_complex')

    def verify_pac_rows(self, db_path, subject, expect_scope=None):
        """Report the PAC rows the run left in the database, per scope.

        Read-only. A PAC run that writes CSV files but no database rows is
        the failure this check exists to surface, so zero rows (or a missing
        ``pac_coupling`` table) is reported prominently rather than passing
        for success. The breakdown is per ``(event_type, method)`` rather than
        a single count for the subject, because a bare count cannot tell a
        correctly labelled row from one filed under the wrong scope.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database that was written to.
        subject : str
            Subject identifier the rows were keyed under.
        expect_scope : tuple of (str, str) or None
            The ``(event_type, method)`` this run should have written. When
            given, that scope is required to be present and non-empty.

        Returns
        -------
        int or None
            Rows for `subject` in `expect_scope`, or across all scopes when no
            scope is given. None when the database could not be read.
        """
        conn = None
        try:
            conn = connect_events_db(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='pac_coupling'")
            if cur.fetchone() is None:
                self.write_log("!" * 60)
                self.write_log(
                    f"!!! No 'pac_coupling' table exists in {db_path}. The PAC "
                    f"results were NOT stored in the database; only the CSV "
                    f"files under the results directory were written.")
                self.write_log("!" * 60)
                return 0
            cur.execute(
                "SELECT event_type, method, COUNT(*) FROM pac_coupling "
                "WHERE subject = ? GROUP BY event_type, method "
                "ORDER BY event_type, method", (subject,))
            scopes = cur.fetchall()
        except Exception as e:
            self.write_log(
                f"Could not verify PAC rows in {db_path}: "
                f"{type(e).__name__}: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()

        total = sum(count for _, _, count in scopes)
        if scopes:
            self.write_log(
                f"Database check: {total} pac_coupling row(s) for subject "
                f"'{subject}' in {db_path}, by scope:")
            for event_type, method, count in scopes:
                self.write_log(
                    f"    event_type={event_type!r}, method={method!r}: "
                    f"{count} row(s)")

        if expect_scope is None:
            if not scopes:
                self.write_log("!" * 60)
                self.write_log(
                    f"!!! Database check: 0 pac_coupling rows for subject "
                    f"'{subject}' in {db_path}. The PAC results did NOT reach "
                    f"the database; only the CSV files were written.")
                self.write_log("!" * 60)
            return total

        n_rows = next((count for event_type, method, count in scopes
                       if (event_type, method) == tuple(expect_scope)), 0)
        if not n_rows:
            self.write_log("!" * 60)
            self.write_log(
                f"!!! Database check: 0 pac_coupling rows for subject "
                f"'{subject}' under the scope this run should have written, "
                f"event_type={expect_scope[0]!r}, method={expect_scope[1]!r}.")
            self.write_log(
                "!!! The results did not reach the database under the "
                "expected labels; only the CSV files were written.")
            self.write_log("!" * 60)
        return n_rows

    def resolve_subject(self, explicit=None):
        """Resolve the subject identifier used to key database rows.

        Uses ``turtlewave_hdEEG.utils.derive_subject``, whose precedence is
        an explicit value, then a BIDS ``sub-XXXX`` token in the annotation
        XML filename, then the basename of the output directory (prefixed
        with ``sub-`` when it lacks one). There is deliberately no
        whole-filename-stem fallback, so two annotation XMLs in the same
        output directory resolve to the same subject. It never raises; if
        nothing resolves it returns ``'unknown_subject'``.

        Parameters
        ----------
        explicit : str or None
            A subject id supplied by the caller, which wins over anything
            derived from paths.

        Returns
        -------
        str
            The resolved subject identifier.
        """
        from turtlewave_hdEEG.utils import derive_subject
        return derive_subject(
            explicit=explicit,
            annotation_path=self.annot_file_path,
            root_dir=self.output_dir,
        )

    @classmethod
    def _library_loggers(cls):
        """Return ``(name, logger)`` for the library logger and its children.

        Only loggers that already exist are returned; one created later by a
        module import is still covered, because it inherits the parent's
        handler by propagation rather than needing its own.
        """
        prefix = cls.LIBRARY_LOGGER_NAME + '.'
        names = [cls.LIBRARY_LOGGER_NAME] + sorted(
            name for name in list(logging.Logger.manager.loggerDict)
            if isinstance(name, str) and name.startswith(prefix))
        return [(name, logging.getLogger(name)) for name in names]

    def ensure_gui_log_handler(self):
        """Route the library's log records into the GUI log pane, exactly once.

        The handler is attached to the ``turtlewave_hdEEG`` logger rather than
        to each processor's own logger. Every library logger is a child of that
        name and propagates to it, so one attachment covers the processors *and*
        the module-level loggers - ``turtlewave_hdEEG.dataset`` above all - that
        no processor object exposes. Attaching per processor meant a dataset
        loading error had nowhere to go and fell through to stderr, invisible
        behind the window.

        Two levels are in play and both matter:

        * the handler is capped at INFO, so a processor constructed with
          ``log_level=logging.DEBUG`` still cannot flood the pane with debug
          chatter (the per-field EEGLAB metadata extraction messages, in
          particular, are DEBUG by design and must stay out);
        * the parent logger is raised to INFO, because module loggers such as
          ``dataset`` set no level of their own. Left at NOTSET the effective
          level would be inherited from the root logger's default WARNING and
          their INFO notices - "skipping large dataset", and the like - would
          never be created at all, let alone handled.

        The subtree is swept for stray ``GUILogHandler`` instances first, and
        only for those. A ``GUILogHandler`` on a child as well as on the parent
        would put every library line in the pane twice, because the child both
        handles the record and propagates it. Handlers of other kinds are left
        exactly as they are - which is a decision, not an oversight:

        * each ``Paral*`` processor gives its own logger a ``StreamHandler``
          in ``_setup_logger`` and leaves propagation on, so a GUI launched
          from a terminal writes every library line to that terminal as well as
          to the pane. That is the library's console output for command-line
          users, the example scripts rely on it, and it is a different sink
          from the pane, so it does not duplicate anything *in* the pane. It is
          left alone deliberately;
        * a child logger that has had propagation switched off cannot reach the
          parent's handler at all, so it is given the same handler directly.
          Its records stop there, so this cannot double-deliver either. Nothing
          in the library does this today; the branch means a later decision to
          isolate a module logger cannot quietly empty the pane.

        Returns
        -------
        GUILogHandler
            The handler now attached to the library logger.
        """
        parent = logging.getLogger(self.LIBRARY_LOGGER_NAME)

        with self._log_handler_lock:
            handler = getattr(self, '_gui_log_handler', None)
            if handler is None:
                handler = GUILogHandler(self.write_log)
                handler.setLevel(logging.INFO)
                self._gui_log_handler = handler

            for _, lg in self._library_loggers():
                if lg is parent:
                    continue
                for existing in list(lg.handlers):
                    if isinstance(existing, GUILogHandler) and existing is not handler:
                        lg.removeHandler(existing)
                        try:
                            existing.close()
                        except Exception:
                            pass
                # A child that propagates is served by the parent's handler;
                # one on the child as well would deliver the same record twice.
                # A child that does not propagate never reaches the parent, so
                # it needs the handler itself - and cannot double-deliver,
                # because its records stop there.
                if lg.propagate:
                    if handler in lg.handlers:
                        lg.removeHandler(handler)
                elif handler not in lg.handlers:
                    lg.addHandler(handler)

            for existing in list(parent.handlers):
                if isinstance(existing, GUILogHandler) and existing is not handler:
                    parent.removeHandler(existing)
                    try:
                        existing.close()
                    except Exception:
                        pass
            if handler not in parent.handlers:
                parent.addHandler(handler)

            if parent.level == logging.NOTSET or parent.level > logging.INFO:
                parent.setLevel(logging.INFO)

        return handler

    def detach_gui_log_handler(self):
        """Remove the GUI handler from the library logger tree.

        Called when the window closes: the handler writes into a Qt widget, and
        a background thread that logs after the window is gone would otherwise
        be writing to a deleted object.
        """
        with self._log_handler_lock:
            for _, lg in self._library_loggers():
                for existing in list(lg.handlers):
                    if isinstance(existing, GUILogHandler):
                        lg.removeHandler(existing)
                        try:
                            existing.close()
                        except Exception:
                            pass
            self._gui_log_handler = None

    def log_event_count(self, label, events):
        """Log how many events a detector returned, when that is knowable.

        Zero events is a legitimate result, not an error, and it is the single
        most important line in the log when it happens - everything downstream
        follows from it. The line states the finding and stops there: what the
        CSV export and the database then did is reported by those steps
        themselves, from what actually happened, so predicting it here would
        only be confirmed a line or two later.
        """
        try:
            count = len(events)
        except TypeError:
            self.write_log(f"{label} detection finished.")
            return None
        self.write_log(f"{label} detection finished: {count} events detected.")
        return count

    def _record_run_outcome(self, label, outcome):
        """Remember how the last `label` run ended, and return `outcome`.

        The detection threads finish by invoking a slot on the GUI thread that
        closes the run with a dialog. That slot has no other way to know what
        happened, which is why it used to announce success unconditionally.
        """
        if getattr(self, '_last_run_outcome', None) is None:
            self._last_run_outcome = {}
        self._last_run_outcome[label] = outcome
        return outcome

    def show_run_finished_dialog(self, label):
        """Close a detection run with a dialog that matches its outcome.

        The thread reaches its finish slot whenever it did not raise, which
        includes every run that detected events but failed to export or import
        them. A modal "Success" over one of those is the same false claim the
        log was cleaned up to stop making, in the more prominent place, so the
        dialog now reports the outcome the run actually had.

        A run that finished but wrote nothing is a warning rather than a
        notification: it needs the user to go and look at the log, and a
        warning icon says so before the text is read.

        Parameters
        ----------
        label : str
            The label the run reported under, e.g. ``"Spindle"``.
        """
        outcome = (getattr(self, '_last_run_outcome', None) or {}).get(label)
        scope = (getattr(self, '_last_run_scope', None) or {}).get(label) or {}
        db_path = scope.get('db_path')
        # The one thing this dialog must not leave unsaid. Detection no longer
        # writes a CSV anywhere, so a researcher who is only told "finished"
        # goes looking in the results folder and finds nothing.
        where = (f"\n\nResults are in the database:\n{db_path}\n\nNo CSV is "
                 f"written by detection. Use \"Export CSV\" on this tab if you "
                 f"need a flat file." if db_path else "")
        if outcome == 'written':
            QMessageBox.information(
                self, "Detection finished",
                f"{label} detection finished. Its events were written to the "
                f"database.{where}\n\nSee the Log tab for the per-stage counts "
                f"and density.")
        elif outcome == 'partial':
            QMessageBox.warning(
                self, "Detection finished with missing channels",
                f"{label} detection finished, but some channels produced no "
                f"result and are missing from the database.{where}\n\nSee the "
                f"Log tab for which channels, and re-run them before using "
                f"per-channel or topographic results.")
        elif outcome == 'no_events':
            QMessageBox.information(
                self, "Detection finished",
                f"{label} detection finished. Every channel ran and no events "
                f"were detected, so nothing was written to the database.")
        elif outcome == 'write_failed':
            QMessageBox.warning(
                self, "Detection finished with errors",
                f"{label} detection finished, but nothing reached the "
                f"database.{where}\n\nSee the Log tab: the errors are recorded "
                f"there.")
        elif outcome == 'unknown':
            QMessageBox.warning(
                self, "Detection finished, result unverified",
                f"{label} detection finished, but the database could not be "
                f"read back, so what reached it is unknown.{where}\n\nSee the "
                f"Log tab, and check the database before using these results.")
        else:
            # No outcome recorded: the run ended before the reporting step.
            QMessageBox.information(
                self, "Detection finished",
                f"{label} detection finished. See the Log tab for what was "
                f"written.")

    def log_run_outcome(self, label, db_path, event_count, rows_written,
                        channels_written, coverage):
        """Say how the run ended, in terms of what is actually in the database.

        The closing lines of a detection run are the ones a researcher reads,
        so they must not conflate outcomes that look alike from the inside.
        "No events were detected" and "every channel raised" both come back
        from the detector as an empty list; "12000 events written" and "12000
        events written but 57 channels never ran" both come back as a large,
        healthy-looking number. Only the database can tell those apart, so this
        reports from the database rather than from the return value.

        Now that detection writes straight to ``neural_events.db``, the closing
        report also has one job it did not have before: saying plainly where
        the results are. No CSV appears in the results directory any more, and a
        researcher who is not told will go looking for one.

        Parameters
        ----------
        label : str
            Event name for the log lines, e.g. ``"Spindle"``.
        db_path : str
            The database the run wrote to, named in every branch so the log
            always says which file is or is not affected.
        event_count : int or None
            What `log_event_count` returned: the number of events the detector
            collected, or None when its return value could not be counted.
        rows_written : int or None
            Rows this run left in ``events`` (`count_db_events`). ``None`` means
            the database could not be read, which is reported as unknown - never
            as zero.
        channels_written : int or None
            Distinct channels those rows cover.
        coverage : dict or None
            The `verify_db_channels` result, or ``None`` when that check could
            not run.

        Returns
        -------
        str
            One of ``'written'``, ``'partial'``, ``'no_events'``,
            ``'write_failed'`` or ``'unknown'`` - the outcome that was reported.
        """
        requested = (coverage or {}).get('requested')
        incomplete = bool(coverage) and not coverage.get('complete', True)

        if rows_written is None:
            self.write_log("!" * 60)
            self.write_log(
                f"!!! {label} detection finished, but the database at {db_path} "
                f"could not be read back, so what reached it is UNKNOWN. The "
                f"detector reported "
                f"{'an unknown number of' if event_count is None else event_count} "
                f"events. Open the database and check before using these "
                f"results.")
            self.write_log("!" * 60)
            return self._record_run_outcome(label, 'unknown')

        if rows_written > 0:
            where = (f"{rows_written} events across {channels_written} channel(s) "
                     f"written to {db_path}")
            if incomplete:
                n_missing = len(coverage.get('missing') or [])
                self.write_log("!" * 60)
                self.write_log(f"!!! {label} detection finished PARTIALLY: {where}.")
                self.write_log(
                    f"!!! {n_missing} of {requested} requested channels produced "
                    f"no result and are NOT in the database. The events that are "
                    f"there are valid, but this run does not cover the montage "
                    f"you asked for - see the channel errors earlier in this log, "
                    f"and re-run the missing channels before using per-channel or "
                    f"topographic results.")
                self.write_log("!" * 60)
                return self._record_run_outcome(label, 'partial')
            covered = (f" covering all {requested} requested channels"
                       if requested else "")
            self.write_log(
                f"{label} detection finished. {where}{covered}.")
            self.write_log(
                f"There is no results CSV: the database is the store of record. "
                f"Use \"Export CSV\" on this tab to write a flat file for "
                f"collaborators, or open the Event Review GUI to inspect the "
                f"events.")
            return self._record_run_outcome(label, 'written')

        # No rows. Distinguish an empty night from a run that fell over.
        if event_count == 0 and coverage is not None and coverage.get('complete'):
            self.write_log(
                f"{label} detection completed: 0 events detected. All "
                f"{requested} channels ran and reported nothing, so the "
                f"database at {db_path} holds no events for this scope. That is "
                f"a result, not a failure.")
            return self._record_run_outcome(label, 'no_events')

        self.write_log("!" * 60)
        counted = ("an unknown number of" if event_count is None
                   else str(event_count))
        self.write_log(
            f"!!! {label} detection wrote NO rows to {db_path}. The detector "
            f"reported {counted} events.")
        if coverage is None:
            self.write_log(
                "!!! The channel coverage check could not run either, so why "
                "is unknown. Check the errors earlier in this log.")
        elif incomplete:
            self.write_log(
                f"!!! {len(coverage.get('missing') or [])} of {requested} "
                f"channels produced no result. Treat this as a failed run, not "
                f"an empty night: a night with no events still records every "
                f"channel as completed.")
        else:
            self.write_log(
                "!!! Every channel is recorded as completed, yet no rows are "
                "stored for this scope. That combination points at the scope "
                "itself - method, band or stage set - not matching what was "
                "written. Check the detection parameters against the database.")
        self.write_log("!" * 60)
        return self._record_run_outcome(label, 'write_failed')

    def write_log_once(self, key, message):
        """Log `message` only when it differs from the last one under `key`.

        Several GUI paths are re-entered on every combo-box change, tab switch
        and detection run, and were logging the same line each time. Keying on
        the message means a real state change still gets reported.

        Parameters
        ----------
        key : str
            Identifies the piece of state being reported.
        message : str or None
            The line to log. None records "nothing to say about this state"
            without logging, so that when the condition recurs it is reported
            again.

        Returns
        -------
        bool
            True if the state changed (and the message, if any, was logged).
        """
        if getattr(self, '_log_once_state', None) is None:
            self._log_once_state = {}
        if key in self._log_once_state and self._log_once_state[key] == message:
            return False
        self._log_once_state[key] = message
        if message is not None:
            self.write_log(message)
        return True

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
            with open(log_filepath, 'w', encoding='utf-8') as f:
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
        # Stop feeding library log records into a widget that is going away.
        self.detach_gui_log_handler()
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