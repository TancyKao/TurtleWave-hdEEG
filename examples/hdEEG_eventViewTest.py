import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# In your test script, import like this:
from frontend.event_review import EventReviewTab


class EEGLABDataset:
    """Class to handle loading and processing EEGLAB .set files"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.sampling_frequency = None
        self.channel_names = None
        self.total_duration = None
        self.load_data()
    
    def load_data(self):
        """Load data from EEGLAB .set file using MNE"""
        try:
            # Load using MNE
            self.data = mne.io.read_raw_eeglab(self.file_path, preload=True)
            
            # Extract metadata
            self.sampling_frequency = self.data.info['sfreq']
            self.channel_names = self.data.ch_names
            self.total_duration = self.data.times[-1]
            
            print(f"Loaded EEG data: {len(self.channel_names)} channels, {self.sampling_frequency} Hz")
            print(f"Duration: {self.total_duration:.2f} seconds")
        
        except Exception as e:
            print(f"Error loading EEGLAB file: {str(e)}")
            raise
    
    def read_data(self, channel, start_idx, end_idx):
        """Read data segment for specified channel and indices"""
        if isinstance(channel, str):
            # Convert channel name to index
            if channel in self.channel_names:
                ch_idx = self.channel_names.index(channel)
            else:
                raise ValueError(f"Channel {channel} not found in dataset")
        else:
            ch_idx = channel
        
        # Get the data for the requested time window
        data, times = self.data[:, start_idx:end_idx]
        return data[ch_idx]

class AnnotationLoader:
    """Class to handle loading and processing annotations"""
    
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.annotations = []
        
        if file_path:
            self.load_annotations()
    
    def load_annotations(self):
        """Load annotations from file (CSV format)"""
        try:
            # Assuming annotations are in CSV format with columns:
            # onset, duration, stage/event_type, description
            annot_df = pd.read_csv(self.file_path)
            
            for _, row in annot_df.iterrows():
                self.annotations.append({
                    'onset': row.get('onset', 0),
                    'duration': row.get('duration', 0),
                    'type': row.get('type', 'unknown'),
                    'name': row.get('name', '')
                })
            
            print(f"Loaded {len(self.annotations)} annotations")
        
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
    
    def get_annotations(self):
        """Return list of annotations"""
        return self.annotations
    
    def add_annotation(self, onset, duration, annot_type, name):
        """Add a new annotation"""
        self.annotations.append({
            'onset': onset,
            'duration': duration,
            'type': annot_type,
            'name': name
        })

class EventsGenerator:
    """Class to generate or load sample events for testing"""
    
    @staticmethod
    def generate_dummy_events(num_events=100, channels=None):
        """Generate dummy events for testing"""
        if channels is None:
            channels = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
        
        events = []
        for i in range(num_events):
            event = {
                'start_time': np.random.uniform(0, 3600),  # Random start time within 1 hour
                'end_time': None,  # Will set below
                'channel': np.random.choice(channels),
                'confidence': np.random.uniform(0, 1),
                'amplitude': np.random.uniform(50, 150),
                'frequency': np.random.uniform(10, 15) if np.random.random() > 0.5 else np.random.uniform(0.5, 4)
            }
            
            # Set end time (0.5 to 2.5 sec duration)
            event['end_time'] = event['start_time'] + np.random.uniform(0.5, 2.5)
            
            # Add sleep stage
            stages = ['N1', 'N2', 'N3', 'REM', 'W']
            event['sleep_stage'] = np.random.choice(stages)
            
            events.append(event)
        
        # Create DataFrame
        events_df = pd.DataFrame(events)
        
        # Set index
        events_df.index.name = 'event_id'
        
        return events_df
    
    @staticmethod
    def load_events_from_csv(file_path):
        """Load events from CSV file"""
        try:
            events_df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['start_time', 'end_time', 'channel']
            missing_cols = [col for col in required_cols if col not in events_df.columns]
            
            if missing_cols:
                print(f"Warning: Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Add confidence column if not present
            if 'confidence' not in events_df.columns:
                events_df['confidence'] = 1.0
            
            return events_df
        
        except Exception as e:
            print(f"Error loading events from CSV: {str(e)}")
            return None

class TestApp(QtWidgets.QMainWindow):
    """Test application for EEG inspection"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("TurtleWave EEG Inspection Test")
        self.setGeometry(100, 100, 1200, 800)
        
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create the main widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Add menu bar
        self.setup_menu()
        
        # Add the review tab
        self.review_tab = EventReviewTab(self)
        self.main_layout.addWidget(self.review_tab)
        
        # Initialize with no data
        self.dataset = None
        self.annotations = None
        self.events_df = None
    
    def setup_menu(self):
        """Set up the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_eeg_action = QtWidgets.QAction('Load EEG (.set)', self)
        load_eeg_action.triggered.connect(self.load_eeg_file)
        file_menu.addAction(load_eeg_action)
        
        load_annot_action = QtWidgets.QAction('Load Annotations (.csv)', self)
        load_annot_action.triggered.connect(self.load_annotations)
        file_menu.addAction(load_annot_action)
        
        file_menu.addSeparator()
        
        load_events_action = QtWidgets.QAction('Load Events (.csv)', self)
        load_events_action.triggered.connect(self.load_events)
        file_menu.addAction(load_events_action)
        
        gen_events_action = QtWidgets.QAction('Generate Dummy Events', self)
        gen_events_action.triggered.connect(self.generate_events)
        file_menu.addAction(gen_events_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def load_eeg_file(self):
        """Load an EEGLAB .set file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open EEG File", "", "EEGLAB Files (*.set);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a progress dialog
                progress = QtWidgets.QProgressDialog("Loading EEG data...", "Cancel", 0, 100, self)
                progress.setWindowModality(QtCore.Qt.WindowModal)
                progress.show()
                progress.setValue(10)
                
                # Load the dataset
                self.dataset = EEGLABDataset(file_path)
                
                # Update progress
                progress.setValue(90)
                
                # If we already have annotations, update the review tab
                if self.annotations and self.events_df is not None:
                    self.review_tab.load_data(self.dataset, self.annotations)
                elif self.annotations:
                    self.review_tab.load_data(self.dataset, self.annotations)
                elif self.events_df is not None:
                    self.review_tab.load_data(self.dataset)
                
                # Complete progress
                progress.setValue(100)
                
                # Show success message
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Loaded EEG data from {os.path.basename(file_path)}"
                )
            
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to load EEG file: {str(e)}"
                )
    
    def load_annotations(self):
        """Load annotations from a CSV file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Annotations File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load annotations
                self.annotations = AnnotationLoader(file_path)
                
                # If we already have a dataset, update the review tab
                if self.dataset:
                    self.review_tab.load_data(self.dataset, self.annotations)
                
                # Show success message
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Loaded annotations from {os.path.basename(file_path)}"
                )
            
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to load annotations: {str(e)}"
                )
    
    def load_events(self):
        """Load events from a CSV file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Events File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load events
                events_df = EventsGenerator.load_events_from_csv(file_path)
                
                if events_df is not None:
                    self.events_df = events_df
                    
                    # Save to the appropriate directory for the GUI to find
                    os.makedirs(os.path.join(self.output_dir, "wonambi", "spindle_results"), exist_ok=True)
                    output_path = os.path.join(self.output_dir, "wonambi", "spindle_results", "spindle_parameters_test.csv")
                    self.events_df.to_csv(output_path)
                    
                    # Show success message
                    QtWidgets.QMessageBox.information(
                        self, "Success", f"Loaded {len(events_df)} events from {os.path.basename(file_path)}"
                    )
                    
                    # If we have a dataset, update the review tab
                    if self.dataset:
                        self.review_tab.load_data(self.dataset, self.annotations)
                        # The event loading happens through the GUI interaction
            
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to load events: {str(e)}"
                )
    
    def generate_events(self):
        """Generate dummy events for testing"""
        try:
            # Get channel names from dataset if available
            channels = None
            if self.dataset:
                channels = self.dataset.channel_names
            
            # Generate dummy events
            self.events_df = EventsGenerator.generate_dummy_events(100, channels)
            
            # Save to the appropriate directory for the GUI to find
            os.makedirs(os.path.join(self.output_dir, "wonambi", "spindle_results"), exist_ok=True)
            output_path = os.path.join(self.output_dir, "wonambi", "spindle_results", "spindle_parameters_test.csv")
            self.events_df.to_csv(output_path)
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self, "Success", f"Generated {len(self.events_df)} dummy events"
            )
            
            # If we have a dataset, update the review tab
            if self.dataset:
                self.review_tab.load_data(self.dataset, self.annotations)
                # The event loading happens through the GUI interaction
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to generate events: {str(e)}"
            )

def main():
    """Main function to run the test application"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Enable High DPI scaling
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    
    # Create and show the main window
    main_window = TestApp()
    main_window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()