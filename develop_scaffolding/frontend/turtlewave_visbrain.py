import os
import sys
import uuid
import json
import numpy as np
import pandas as pd
from scipy import signal
from datetime import datetime
from visbrain.gui import Sleep
import xml.etree.ElementTree as ET  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QWidget, QFrame,
                            QSplitter, QComboBox)
from PyQt5.QtCore import Qt

def load_spindle_event_data(dataset, channel_name, spindles, time_window=(1.0, 2.0)):
    """
    Selectively load data around spindle events for a specific channel
    
    Parameters:
    -----------
    dataset : Wonambi dataset object
    channel_name : str
        The name of the channel to load data for
    spindles : list
        List of spindle dictionaries
    time_window : tuple, optional
        (before, after) in seconds to load around each event
    
    Returns:
    --------
    events_data : dict
        Dictionary containing event times and corresponding data segments
    """
    # Get sampling rate from dataset
    sampling_rate = dataset.header['s_freq']
    
    # Get channel names and index
    channel_names = dataset.header['chan_name']
    try:
        channel_idx = channel_names.index(channel_name)
    except ValueError:
        print(f"Channel {channel_name} not found in dataset")
        return None
    
    # Filter spindles for the selected channel
    channel_spindles = [s for s in spindles if s['chan'] == channel_name]
    
    if not channel_spindles:
        print(f"No spindles found for channel {channel_name}")
        return None
    
    # Get event times
    event_times = [s['start time'] for s in channel_spindles]
    
    # Calculate samples before and after to extract
    before_samples = int(time_window[0] * sampling_rate)
    after_samples = int(time_window[1] * sampling_rate)
    
    # Extract data segments around each event
    events_data = {
        'times': event_times,
        'segments': [],
        'window': time_window,
        'spindles': channel_spindles,
        'channel': channel_name
    }
    
    # Get the total number of timepoints in the dataset
    data_obj = dataset.read_data()
    total_timepoints = data_obj.data[0].shape[1]
    
    for event_time in event_times:
        # Convert time to sample index
        event_sample = int(event_time * sampling_rate)
        
        # Calculate window boundaries
        start_idx = max(0, event_sample - before_samples)
        end_idx = min(total_timepoints, event_sample + after_samples)
        
        # Extract data for this time window
        segment_data = dataset.read_data(begtime=start_idx/sampling_rate, endtime=end_idx/sampling_rate)
        events_data['segments'].append(segment_data.data[0])
    
    return events_data

# ADDED: Function to parse sleep stages from XML file
def parse_sleep_stages_from_xml(xml_file, data_length, sampling_rate):
    """
    Parse sleep stages from XML file and convert to hypnogram
    
    Parameters:
    -----------
    xml_file : str
        Path to XML file containing sleep stages
    data_length : int
        Length of the EEG data in samples
    sampling_rate : float
        Sampling rate of the EEG data
    
    Returns:
    --------
    hypno : np.ndarray
        Hypnogram array with same length as the EEG data
    """
    # Map sleep stages to Visbrain's hypnogram values
    stage_map = {
        'Wake': 0,
        'NREM1': 1,
        'NREM2': 2, 
        'NREM3': 3,
        'REM': 4,
        'Artefact': -1,
        'Movement': -1,
        'Unknown': -1,
        'Undefined': -1
    }
    
    # Create empty hypnogram filled with undefined (-1)
    hypno = np.ones(data_length) * -1
    
    try:
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find all epoch elements
        epochs = root.findall(".//epoch")
        
        for epoch in epochs:
            # Get stage name
            stage = epoch.find('stage').text
            start = float(epoch.find('epoch_start').text)
            end = float(epoch.find('epoch_end').text)
            
            # Convert to sample indices
            start_idx = int(start * sampling_rate)
            end_idx = min(int(end * sampling_rate), data_length)
            
            # Set the hypnogram value for this epoch
            if stage in stage_map:
                hypno[start_idx:end_idx] = stage_map[stage]
        
        return hypno
    
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        import traceback
        traceback.print_exc()
        return hypno

# Function to parse events from XML file
def parse_events_from_xml(xml_file):
    """
    Parse arousal and artifact events from XML file
    
    Parameters:
    -----------
    xml_file : str
        Path to XML file containing events
    
    Returns:
    --------
    events : dict
        Dictionary with 'Arousal' and 'Artefact' events
    """
    events = {
        'Arousal': [],
        'Artefact': []
    }
    
    try:
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find all event_type elements
        event_types = root.findall(".//event_type")
        
        for event_type in event_types:
            type_name = event_type.get('type')
            
            # Only process Arousal and Artefact events
            if type_name in events:
                for event in event_type.findall('event'):
                    start = float(event.find('event_start').text)
                    end = float(event.find('event_end').text)
                    
                    events[type_name].append({
                        'start': start,
                        'end': end,
                        'duration': end - start,
                        'chan': event.find('event_chan').text,
                        'qual': event.find('event_qual').text
                    })
        
        return events
    
    except Exception as e:
        print(f"Error parsing XML events: {e}")
        import traceback
        traceback.print_exc()
        return events



class TurtlewaveVisbrainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Turtlewave Spindle Visualization")
        self.resize(1200, 800)
        
        # Data storage
        self.eeg_data = None
        self.spindles_json = []
        self.spindles_df = None
        self.eeg_file_path = None
        self.spindle_json_path = None
        self.spindle_csv_path = None
        self.sleep = None  # Will hold the Visbrain Sleep instance
        

        # ADDED: New variables for selective loading
        self.dataset = None  # Will hold the Wonambi Dataset object
        self.selected_channels_data = {}  # Will hold data for selected channels
        self.all_spindle_jsons = {}  # Will hold multiple spindle JSON files
        self.annotation_file = None  # ADDED: Will hold path to XML annotation file
        self.events = None  # ADDED: Will hold events from XML file
        self.hypno = None  # ADDED: Will hold hypnogram from XML file       

        # Setup UI
        self.init_ui()
    
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create top controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Add file loading buttons
        self.load_eeg_btn = QPushButton("Load EEG (.set)")
        # MODIFIED: Changed button text to reflect multiple JSON support
        self.load_spindles_btn = QPushButton("Load Spindles (Multiple JSON/CSV)")
        # ADDED: Button to load sleep annotations XML
        self.load_annotations_btn = QPushButton("Load Annotations (XML)")
 

        # Add file loading buttons
        self.load_eeg_btn = QPushButton("Load EEG (.set)")
        self.load_spindles_btn = QPushButton("Load Spindles (JSON/CSV)")
        self.load_eeg_btn.clicked.connect(self.load_eeg_data)
        self.load_spindles_btn.clicked.connect(self.load_spindle_files)
        
        # Add channel selector
        self.channel_selector = QComboBox()
        self.channel_label = QLabel("Channel:")
        self.channel_selector.currentIndexChanged.connect(self.on_channel_selected)


        # Add action buttons
        self.add_spindle_btn = QPushButton("Add Spindle")
        self.delete_spindle_btn = QPushButton("Delete Spindle")
        self.save_spindles_btn = QPushButton("Save Changes")
        self.save_as_btn = QPushButton("Save As...")

        self.add_spindle_btn.clicked.connect(self.add_spindle)
        self.delete_spindle_btn.clicked.connect(self.delete_spindle)
        self.save_spindles_btn.clicked.connect(self.save_spindles)
        self.save_as_btn.clicked.connect(self.save_spindles_as)

        # Disable buttons until data is loaded
        self.add_spindle_btn.setEnabled(False)
        self.delete_spindle_btn.setEnabled(False)
        self.save_spindles_btn.setEnabled(False)
        self.channel_selector.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        
        # Add controls to layout
        controls_layout.addWidget(self.load_eeg_btn)
        controls_layout.addWidget(self.load_spindles_btn)
        controls_layout.addWidget(self.load_annotations_btn) 
        controls_layout.addWidget(self.channel_label)
        controls_layout.addWidget(self.channel_selector)
        controls_layout.addWidget(self.add_spindle_btn)
        controls_layout.addWidget(self.delete_spindle_btn)
        controls_layout.addWidget(self.save_spindles_btn)
        controls_layout.addWidget(self.save_as_btn)


        # Create placeholder for Visbrain widget
        self.visbrain_container = QFrame()
        self.visbrain_layout = QVBoxLayout(self.visbrain_container)
        
        # Add widgets to main layout
        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.visbrain_container)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready. Please load EEG data.")
    
    def load_eeg_data(self):
        """Load EEG data from EEGLAB .set file using Wonambi directly"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load EEG Data", "", "EEGLAB Files (*.set)"
        )
        
        if not file_path:
            return
            
        self.eeg_file_path = file_path
        self.statusBar().showMessage(f"Loading EEG data from {file_path}...")
        
        try:
            # Import Wonambi Dataset
            from wonambi import Dataset
            
            # Use Wonambi's Dataset to load the file
            print(f"Loading EEG file with Wonambi: {file_path}")
            dataset = Dataset(file_path)
            
            # Get header information
            hdr = dataset.header
            # Check if sampling rate is in header
            if 's_freq' in hdr:
                sampling_rate = hdr['s_freq']
                print(f"Found sampling rate: {sampling_rate} Hz")
            else:
                # Handle missing sampling rate
                print("Sampling rate not found in header")
                from PyQt5.QtWidgets import QInputDialog, QMessageBox
                
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Sampling rate not found in file")
                msg.setInformativeText("Would you like to specify the sampling rate manually?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                response = msg.exec_()
                
                if response == QMessageBox.Yes:
                    sampling_rate, ok = QInputDialog.getDouble(
                        self, "Sampling Rate", 
                        "Please enter the EEG sampling rate (Hz):",
                        256.0, 500.0, 10000.0, 1)
                    
                    if not ok:
                        self.statusBar().showMessage("Loading canceled")
                        return
                        
                    # Update header with the manual sampling rate
                    hdr['s_freq'] = sampling_rate
                else:
                    self.statusBar().showMessage("Loading canceled")
                    return
            
            # Get channel names
            if 'chan_name' in hdr:
                channels = hdr['chan_name']
            else:
                # Create default channel names
                channels = [f"Channel_{i+1}" for i in range(32)]  # Default to 32 channels
                print("Channel names not found, using defaults")
            
            # Load a segment of data (first 60 seconds)
            try:
                # Try to load 60 seconds of data
                data_obj = dataset.read_data(begtime=0, endtime=10)
                
                # Extract the data array
                data = data_obj.data[0]  # First dimension is epochs (usually 1)
                
                # Get times
                times = np.linspace(0, data.shape[1] / sampling_rate, data.shape[1], endpoint=False)
                times[0] = int(times[0])
                
                print(f"Data loaded successfully: {data.shape}")
                
            except Exception as data_err:
                print(f"Error reading data: {data_err}")
                import traceback
                traceback.print_exc()
    
            
            # Store the dataset for later
            self.dataset = dataset
            
            # Store the data
            self.eeg_data = {
                'data': data,
                'sf': sampling_rate,
                'channels': channels,
                'times': times,
                'full_length': dataset.header['n_samples']  # ADDED: Store total length for hypnogram
            }
            
            # Filter to include only EEG channels
            self.filter_eeg_channels()
            
            # Update channel selector
            self.channel_selector.clear()
            self.channel_selector.addItems(channels)
            self.channel_selector.setEnabled(True)
            
            # ADDED: Check if we already have annotations, if so, load hypnogram
            if self.annotation_file:
                self.load_hypnogram_from_annotations()

            # Initialize Visbrain with EEG data
            self.initialize_visbrain()
            self.statusBar().showMessage(f"EEG data loaded: {len(channels)} channels, {data.shape[1]} samples at {sampling_rate} Hz")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading EEG data: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # ADDED: New method to load annotation XML file
    def load_annotation_file(self):
        """Load sleep stages and events from XML annotation file"""
        xml_path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotation XML File", "", "XML Files (*.xml)"
        )
        
        if not xml_path:
            return
            
        self.annotation_file = xml_path
        self.statusBar().showMessage(f"Loading annotations from {xml_path}...")
        
        # If EEG data is already loaded, load the hypnogram
        if self.eeg_data is not None:
            self.load_hypnogram_from_annotations()
            
            # Reload Visbrain with new hypnogram
            self.initialize_visbrain()
        else:
            self.statusBar().showMessage("Annotations loaded. Please load EEG data to visualize.")

    def filter_eeg_channels(self):
        """Filter data to include only EEG channels (starting with 'E') and Cz"""
        if not hasattr(self, 'eeg_data') or self.eeg_data is None:
            return False
            
        try:
            # Get indices of channels that start with 'E' or are 'Cz'
            eeg_indices = []
            excluded_channels = []
            for i, chan in enumerate(self.eeg_data['channels']):
                if chan.startswith('E') or chan == 'Cz'and not any(x in chan.upper() for x in ['EOG', 'EMG', 'ECG', 'EKG']):
                    eeg_indices.append(i)
                else:
                    excluded_channels.append(chan)
            
            if not eeg_indices:
                self.statusBar().showMessage("No EEG channels (starting with 'E' or 'Cz') found")
                return False
            
            # Extract only the EEG channels
            eeg_channels = [self.eeg_data['channels'][i] for i in eeg_indices]
            eeg_data = self.eeg_data['data'][eeg_indices]
            
            print(f"Filtered from {len(self.eeg_data['channels'])} to {len(eeg_channels)} EEG channels")
            print(f"Excluded channels: {excluded_channels}")
            print(f"New data shape: {eeg_data.shape}")

            # Check data amplitude range
            data_min = np.min(eeg_data)
            data_max = np.max(eeg_data)
            print(f"EEG data range: {data_min:.2f} to {data_max:.2f}, span: {data_max - data_min:.2f}")

            # Update the eeg_data with filtered channels
            self.eeg_data['original_channels'] = self.eeg_data['channels']
            self.eeg_data['original_data'] = self.eeg_data['data']
            self.eeg_data['channels'] = eeg_channels
            self.eeg_data['data'] = eeg_data
            
            # Update channel selector
            self.channel_selector.clear()
            self.channel_selector.addItems(eeg_channels)
            
            return True
            
        except Exception as e:
            self.statusBar().showMessage(f"Error filtering EEG channels: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    # ADDED: New method to load hypnogram from annotations
    def load_hypnogram_from_annotations(self):
        """Load hypnogram and events from annotation file"""
        if not self.annotation_file or not self.eeg_data:
            return
            
        try:
            # Parse sleep stages
            self.hypno = parse_sleep_stages_from_xml(
                self.annotation_file, 
                self.eeg_data['full_length'], 
                self.eeg_data['sf']
            )
            
            # Parse events
            self.events = parse_events_from_xml(self.annotation_file)
            
            self.statusBar().showMessage(f"Loaded sleep stages and {len(self.events['Arousal'])} arousals, {len(self.events['Artefact'])} artifacts")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading hypnogram from annotations: {str(e)}")
            import traceback
            traceback.print_exc()


    def load_spindle_files(self):
        """Load spindle events from multiple JSON and CSV files"""
        # Open file dialog for selecting multiple JSON files
        json_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Spindle JSON Files", "", "JSON Files (*.json)"
        )
        
        if not json_paths:
            return    
        
        # Keep track of all loaded JSONs
        self.all_spindle_jsons = {}
        self.spindles_json = []  # This will be a combined list
        
        for json_path in json_paths:
            # Try to find corresponding CSV file
            default_csv_path = json_path.replace('.json', '.csv')
            if os.path.exists(default_csv_path):
                csv_path = default_csv_path
            else:
                csv_path, _ = QFileDialog.getOpenFileName(
                    self, f"Load CSV File for {os.path.basename(json_path)}", "", "CSV Files (*.csv)"
                )
            if not csv_path:
                continue  # Skip this JSON if no CSV


            try:
                # Load JSON file
                with open(json_path, 'r') as f:
                    spindles_json = json.load(f)
                    
                # Store in our dictionary
                json_name = os.path.basename(json_path)
                self.all_spindle_jsons[json_name] = {
                    'json_path': json_path,
                    'csv_path': csv_path,
                    'spindles': spindles_json,
                    'df': pd.read_csv(csv_path)
                }
                
                # Add to combined list
                self.spindles_json.extend(spindles_json)
                
                # If this is the first one, also set it as the current spindle file
                if not hasattr(self, 'spindle_json_path') or self.spindle_json_path is None:
                    self.spindle_json_path = json_path
                    self.spindle_csv_path = csv_path
                    self.spindles_df = self.all_spindle_jsons[json_name]['df']
                    
            except Exception as e:
                self.statusBar().showMessage(f"Error loading spindle file {json_path}: {str(e)}")
            
        # Check if we loaded any files successfully
        if self.all_spindle_jsons:
            self.statusBar().showMessage(f"Loaded {len(self.all_spindle_jsons)} spindle files with {len(self.spindles_json)} total spindles")
            
            # Update the channel selector with channels that have spindles
            self.update_channel_selector_with_spindle_channels()
            
            # Enable buttons
            self.add_spindle_btn.setEnabled(True)
            self.delete_spindle_btn.setEnabled(True)
            self.save_spindles_btn.setEnabled(True)
            self.save_as_btn.setEnabled(True)
        else:
            self.statusBar().showMessage("No spindle files were loaded successfully")
    
    # ADDED: New method to update channel selector with spindle channels
    def update_channel_selector_with_spindle_channels(self):
        """Update the channel selector to highlight channels with spindles"""
        if not self.eeg_data or not self.spindles_json:
            return
            
        # Get all channels with spindles
        spindle_channels = set(spindle['chan'] for spindle in self.spindles_json)
        
        # Store current selection
        current_selection = self.channel_selector.currentText()
        
        # Clear and rebuild
        self.channel_selector.clear()
        
        # Add all EEG channels
        for channel in self.eeg_data['channels']:
            # Add indicator for channels with spindles
            if channel in spindle_channels:
                self.channel_selector.addItem(f"{channel} 🔍")  # Add magnifying glass icon
            else:
                self.channel_selector.addItem(channel)
        
        # Restore selection if possible
        index = self.channel_selector.findText(current_selection, Qt.MatchStartsWith)
        if index >= 0:
            self.channel_selector.setCurrentIndex(index)
        elif self.channel_selector.count() > 0:
            # Select the first channel with spindles
            for i in range(self.channel_selector.count()):
                if "🔍" in self.channel_selector.itemText(i):
                    self.channel_selector.setCurrentIndex(i)
                    break

    # ADDED: New method to handle channel selection
    def on_channel_selected(self, index):
        """Handle channel selection change"""
        if index < 0 or not self.dataset or not self.spindles_json:
            return
            
        # Get selected channel (remove the spindle indicator if present)
        channel_text = self.channel_selector.itemText(index)
        channel_name = channel_text.split(" 🔍")[0]  # Remove indicator if present
        
        self.statusBar().showMessage(f"Loading data for channel: {channel_name}")
        
        # Check if this channel has spindles
        channel_spindles = [s for s in self.spindles_json if s['chan'] == channel_name]
        has_spindles = len(channel_spindles) > 0
        
        if has_spindles:
            # Selectively load data around spindles
            try:
                events_data = load_spindle_event_data(
                    self.dataset, 
                    channel_name, 
                    self.spindles_json, 
                    time_window=(1.0, 2.0)  # 1 sec before, 2 sec after each spindle
                )
                
                if events_data and events_data['segments']:
                    # Store the selectively loaded data
                    self.selected_channels_data[channel_name] = events_data
                    
                    # Update visualization with this data
                    self.update_visbrain_with_events_data(events_data)
                    
                    self.statusBar().showMessage(
                        f"Loaded {len(events_data['segments'])} spindle events for channel {channel_name}"
                    )
                else:
                    self.statusBar().showMessage(f"No spindle events found for channel {channel_name}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading spindle data: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            # Regular channel with no spindles, use the initial data sample
            self.statusBar().showMessage(f"Channel {channel_name} has no spindle events")
            self.initialize_visbrain()  # Revert to regular view


    # ADDED: New method to update Visbrain with events data
    def update_visbrain_with_events_data(self, events_data):
        """Update Visbrain display with spindle event data"""
        if not events_data or not events_data['segments'] or len(events_data['segments']) == 0:
            return
            
        # Combine spindle segments with gaps between them
        sampling_rate = self.eeg_data['sf']
        gap_size = int(0.5 * sampling_rate)  # Half-second gap
        
        # Calculate total length needed
        total_length = sum(segment.shape[1] for segment in events_data['segments'])
        # Add gaps between segments
        total_length += gap_size * (len(events_data['segments']) - 1)
        
        # Get number of channels
        n_channels = events_data['segments'][0].shape[0]
        
        # Create combined data array
        combined_data = np.zeros((n_channels, total_length))
        
        # Fill in the data
        current_pos = 0
        event_positions = []  # To track where each spindle is in the combined array
        
        for i, segment in enumerate(events_data['segments']):
            seg_len = segment.shape[1]
            combined_data[:, current_pos:current_pos+seg_len] = segment
            
            # Calculate where the actual spindle event is within this segment
            # (It's at events_data['window'][0] seconds into each segment)
            event_mid = current_pos + int(events_data['window'][0] * sampling_rate)
            event_positions.append(event_mid)
            
            # Move position for next segment
            current_pos += seg_len + gap_size if i < len(events_data['segments'])-1 else seg_len
        
        # Create a flat hypnogram
        if self.hypno is not None:
            # Sample from the full hypnogram for each segment
            hypno = np.zeros(total_length)
            current_pos = 0
            
            for i, segment in enumerate(events_data['segments']):
                seg_len = segment.shape[1]
                event_time = events_data['spindles'][i]['start time']
                
                # Find corresponding position in the full hypnogram
                hypno_start_idx = max(0, int(event_time * sampling_rate) - int(events_data['window'][0] * sampling_rate))
                hypno_end_idx = min(len(self.hypno), hypno_start_idx + seg_len)
                
                # Get hypnogram segment
                if hypno_end_idx > hypno_start_idx:
                    hypno_segment = self.hypno[hypno_start_idx:hypno_end_idx]
                    hypno[current_pos:current_pos+len(hypno_segment)] = hypno_segment
                
                # Move position for next segment
                current_pos += seg_len + gap_size if i < len(events_data['segments'])-1 else seg_len
        else:
            # Create default hypnogram (all stage 2)
            hypno = np.ones(total_length) * 2  # Default to NREM2
        
        # Create or update the Sleep object
        if self.sleep is not None:
            # Clear previous widget if it exists
            if self.visbrain_layout.count() > 0:
                item = self.visbrain_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # Create a new Sleep object
        self.sleep = Sleep(data=combined_data, 
                            sf=sampling_rate,
                            channels=self.eeg_data['channels'],
                            hypno=hypno)
        
        # Add annotations for each spindle event
        for i, pos in enumerate(event_positions):
            spindle = events_data['spindles'][i]
            self.sleep.add_annotation(
                time=pos/sampling_rate,
                duration=spindle['duration'],
                text=f"Spindle {i+1}: {spindle.get('peak_freq', 0):.1f} Hz",
                color='#e74c3c'
            )
        
        # ADDED: Add arousal and artifact events if available
        if self.events is not None:
            # For spindle view, we don't add the events since they're not in this time window
            # They'll be shown in the full view
            pass
        
        # Get the canvas widget from Visbrain
        self.visbrain_widget = self.sleep._canvas_spec
        
        # Add widget to layout
        self.visbrain_layout.addWidget(self.visbrain_widget.canvas.native)
        
        # Connect to Visbrain's signal for event selection
        self.sleep._on_picked_line = self.on_event_selected
        
        # Update display
        self.sleep.update()


    def initialize_visbrain(self):
        """Initialize Visbrain Sleep module with EEG data"""
        if self.eeg_data is None:
            return

        if self.hypno is not None:
            # Use the part of hypnogram that corresponds to our sample data
            sample_length = self.eeg_data['data'].shape[1]
            hypno = self.hypno[:sample_length]
        else:
            # Create a default hypnogram (all zeros or -1 for undefined)
            hypno = np.zeros(self.eeg_data['data'].shape[1])  # or use -1 for undefined 

        
        # Now initialize Visbrain with the patched code
        try:            
            # Get data and convert time values to integers (milliseconds)
            data = self.eeg_data['data']
            channels = self.eeg_data['channels']
            sf = self.eeg_data['sf']
            
            # Create integer time values (sample indices)
            # Instead of trying to use actual time values in seconds/milliseconds,
            # simply use sample indices (0, 1, 2, etc.)
            n_samples = data.shape[1]
            # Create integer time array of sample indices
            time_indices = np.arange(n_samples, dtype=np.int32)
            
            print(f"Using integer sample indices as time values")
            print(f"Data shape: {data.shape}, Channels: {len(channels)}")
            print(f"Sample rate: {sf} Hz, Time points: {len(time_indices)}")
            print(f"Time values type: {time_indices.dtype}, First few values: {time_indices[:5]}")       



            # Create Sleep module with data
            self.sleep = Sleep(data=data, 
                            sf=sf,
                            channels=channels,
                            hypno=hypno)

            # ADDED: Add arousal and artifact events if available
            if self.events is not None:
                # Add arousal events
                for i, event in enumerate(self.events['Arousal']):
                    # Only add events that are within our sample time range
                    if event['start'] < 10:  # 10 seconds = length of our initial sample
                        self.sleep.add_annotation(
                            time=event['start'],
                            duration=event['duration'],
                            text=f"Arousal {i+1}",
                            color='#ff0000'  # Red
                        )

            # Add artifact events
                for i, event in enumerate(self.events['Artefact']):
                    # Only add events that are within our sample time range  
                    if event['start'] < 10:  # 10 seconds = length of our initial sample
                        self.sleep.add_annotation(
                            time=event['start'],
                            duration=event['duration'],
                            text=f"Artifact {i+1}",
                            color='#ff9900'  # Orange
                        )

            
            # Get the canvas widget from Visbrain
            self.visbrain_widget = self.sleep._canvas_spec
            
            # Clear previous widget if it exists
            if self.visbrain_layout.count() > 0:
                item = self.visbrain_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            # Add widget to layout
            self.visbrain_layout.addWidget(self.visbrain_widget.canvas.native)
            
            # Connect to Visbrain's signal for event selection
            self.sleep._on_picked_line = self.on_event_selected
            
            # Add annotations if spindles are loaded
            if self.spindles_json:
                self.update_visbrain_annotations()

            # Replace the central widget with Visbrain
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            layout.addWidget(self.sleep._canvas)
            
            # Set central widget
            self.setCentralWidget(central_widget)
            
            # Access and configure the Visbrain object
            self.sleep.show()
            
            # Customize visualization settings
            if len(self.eeg_data['channels']) > 0:
                self.sleep._chanChecks[0].setChecked(True)  # Enable first channel
                self.sleep._PanSpecChan.setCurrentIndex(0)  # Set spectrogram to first channel
            
            # Set appropriate time window (display 10 seconds initially)
            self.sleep.set_window(0, min(10000, self.eeg_data['times'][-1]))
            
            # Update status
            self.statusBar().showMessage(f"Visualization initialized with {len(self.eeg_data['channels'])} channels")
            return True
        
        except Exception as e:
            self.statusBar().showMessage(f"Error initializing Visbrain: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to custom visualization
            print("Falling back to custom EEG viewer...")
            self.initialize_custom_eeg_viewer()
            return False


    
    def update_visbrain_annotations(self):
        """Update Visbrain with spindle annotations"""
        if self.sleep is None or not self.spindles_json:
            return
            
        # Convert spindles to Visbrain annotation format
        annotations = self.convert_spindles_to_visbrain_format()
        
        # Clear existing annotations
        self.sleep._annot_table.model.clear()
        
        # Add annotations to Visbrain
        for annot in annotations:
            self.sleep.add_annotation(annot)
            
        # Refresh display
        self.sleep.update()
    
    def convert_spindles_to_visbrain_format(self):
        """Convert spindles from JSON to Visbrain annotations format"""
        visbrain_annotations = []
        
        for spindle in self.spindles_json:
            # Use only spindles for the currently selected channel
            if spindle['chan'] == self.channel_selector.currentText():
                annot = {
                    'name': f"Spindle-{spindle['uuid'][:8]}",
                    'time': spindle['start time'],
                    'duration': spindle['duration'],
                    'color': '#e74c3c',  # Red color
                    'text': f"Spindle {spindle['peak_freq']:.2f} Hz",
                    'uuid': spindle['uuid']
                }
                visbrain_annotations.append(annot)
                
        return visbrain_annotations
        
    def on_event_selected(self, index, xdata):
        """Handle event selection in Visbrain"""
        # Get selected annotation from Visbrain
        if hasattr(self.sleep, '_annot_table'):
            selected_annot = self.sleep._annot_table.selectedItems()
            if selected_annot:
                # Get UUID from the annotation name
                annot_name = selected_annot[0].text()
                
                # Extract UUID if it exists in the annotation
                if "Spindle" in annot_name and ":" in annot_name:
                    uuid_part = annot_name.split("Spindle ")[1].split(":")[0].strip()
                    
                    # Find corresponding spindle
                    for spindle in self.spindles_json:
                        if spindle['uuid'].startswith(uuid_part):
                            # Display info in status bar
                            msg = (f"Selected: {spindle['chan']}, "
                                   f"Duration: {spindle['duration']:.2f}s, "
                                   f"Frequency: {spindle.get('peak_freq', 0):.2f} Hz")
                            self.statusBar().showMessage(msg)
                            break
                # Handle arousal or artifact event selection
                elif "Arousal" in annot_name:
                    idx = int(annot_name.split("Arousal ")[1]) - 1
                    if idx < len(self.events['Arousal']):
                        event = self.events['Arousal'][idx]
                        self.statusBar().showMessage(f"Arousal: {event['start']:.2f}s - {event['end']:.2f}s, Duration: {event['duration']:.2f}s")
                elif "Artifact" in annot_name:
                    idx = int(annot_name.split("Artifact ")[1]) - 1
                    if idx < len(self.events['Artefact']):
                        event = self.events['Artefact'][idx]
                        self.statusBar().showMessage(f"Artifact: {event['start']:.2f}s - {event['end']:.2f}s, Duration: {event['duration']:.2f}s")
    
    
    # MODIFIED: Updated to work with selected channel
    def add_spindle(self):
        """Add a new spindle at current position"""
        if self.sleep is None:
            return
            
        # Get current time from Visbrain's display
        current_time = self.sleep.canvas.loc[0]
        
        # Get currently selected channel (remove indicator if present)
        channel = self.channel_selector.currentText().split(" 🔍")[0]
        
        # Create a new spindle (1 second duration by default)
        new_spindle = {
            'uuid': str(uuid.uuid4()),
            'chan': channel,
            'start time': current_time,
            'end time': current_time + 1.0,
            'peak_time': current_time + 0.5,
            'duration': 1.0,
            'method': "Ferrarelli2007",
            'stage': ["NREM2", "NREM3"],
            'freq_range': [11, 16],
            'ptp_det': 10.0,
            'peak_freq': 12.5  # Default frequency
        }
        
        # Calculate actual peak frequency if possible
        if self.dataset is not None:
            try:
                # Get the channel index
                channel_idx = self.eeg_data['channels'].index(channel)
                
                # Load a small segment for frequency analysis
                segment_data = self.dataset.read_data(
                    begtime=current_time, 
                    endtime=current_time + 1.0
                )
                
                # Get signal segment
                segment = segment_data.data[0][channel_idx, :]
                
                # Calculate frequency using FFT
                if len(segment) > 10:  # Check if we have enough data
                    freqs, psd = signal.welch(segment, self.eeg_data['sf'], nperseg=len(segment))
                    # Find peak frequency in spindle range (11-16 Hz)
                    spindle_range = (freqs >= 11) & (freqs <= 16)
                    if np.any(spindle_range):
                        peak_idx = np.argmax(psd[spindle_range])
                        new_spindle['peak_freq'] = freqs[spindle_range][peak_idx]
            except Exception as e:
                print(f"Could not calculate peak frequency: {e}")
        
        # Add to internal data structures
        self.spindles_json.append(new_spindle)
        
        # Find which JSON file this spindle belongs to
        # Default to the first one if we can't determine
        target_json_name = None
        for json_name, json_data in self.all_spindle_jsons.items():
            # Check if this JSON has spindles for this channel
            if any(s['chan'] == channel for s in json_data['spindles']):
                target_json_name = json_name
                break
        
        # If no matching JSON found, use the first one
        if target_json_name is None and self.all_spindle_jsons:
            target_json_name = list(self.all_spindle_jsons.keys())[0]
        
        # Add to the target JSON's spindle list
        if target_json_name is not None:
            self.all_spindle_jsons[target_json_name]['spindles'].append(new_spindle)
            
            # Add to CSV dataframe
            new_row = {
                'Segment index': 1,
                'Start time': new_spindle['start time'],
                'Start time (HH:MM:SS)': self.format_time(new_spindle['start time']),
                'End time': new_spindle['end time'],
                'Stitches': 0,
                'Stage': str(new_spindle['stage']),
                'Cycle': '',
                'Event type': 'spindle',
                'Channel': new_spindle['chan'],
                'Duration (s)': new_spindle['duration'],
                'UUID': new_spindle['uuid']
            }
            
            # Add to the JSON's dataframe
            self.all_spindle_jsons[target_json_name]['df'] = self.all_spindle_jsons[target_json_name]['df'].append(
                new_row, ignore_index=True
            )
            
            # If this is the currently active spindle file, update that too
            if self.spindles_df is not None:
                self.spindles_df = self.spindles_df.append(new_row, ignore_index=True)
        
        # Update visualization
        # If we're in the events view, reload the events
        current_channel = self.channel_selector.currentText().split(" 🔍")[0]
        if current_channel in self.selected_channels_data:
            # Reload this channel's data with the new spindle
            events_data = load_spindle_event_data(
                self.dataset, 
                current_channel, 
                self.spindles_json, 
                time_window=(1.0, 2.0)
            )
            if events_data:
                self.selected_channels_data[current_channel] = events_data
                self.update_visbrain_with_events_data(events_data)
        else:
            # Regular view, just update annotations
            self.update_visbrain_annotations()
        
        self.statusBar().showMessage(f"Added new spindle at {current_time:.2f}s")
    
    # MODIFIED: Updated to work with multiple JSON files
    def delete_spindle(self):
        """Delete selected spindle"""
        if self.sleep is None:
            return
            
        # Get selected annotation from Visbrain
        if hasattr(self.sleep, '_annot_table'):
            selected_annot = self.sleep._annot_table.selectedItems()
            if selected_annot:
                # Get UUID from the annotation name
                annot_name = selected_annot[0].text()
                
                # Extract UUID if it exists in the annotation
                if "Spindle" in annot_name and ":" in annot_name:
                    uuid_part = annot_name.split("Spindle ")[1].split(":")[0].strip()
                    
                    # Find and remove corresponding spindle
                    spindle_to_delete = None
                    spindle_index = -1
                    
                    for i, spindle in enumerate(self.spindles_json):
                        if spindle['uuid'].startswith(uuid_part):
                            spindle_to_delete = spindle
                            spindle_index = i
                            break
                    
                    if spindle_to_delete is not None:
                        # Remove from main list
                        del self.spindles_json[spindle_index]
                        
                        # Find which JSON file this spindle belongs to
                        for json_name, json_data in self.all_spindle_jsons.items():
                            # Look for this spindle in the JSON's spindle list
                            for i, s in enumerate(json_data['spindles']):
                                if s['uuid'].startswith(uuid_part):
                                    # Remove from JSON's spindle list
                                    del json_data['spindles'][i]
                                    
                                    # Remove from JSON's dataframe
                                    json_data['df'] = json_data['df'][
                                        ~json_data['df']['UUID'].str.startswith(uuid_part)
                                    ]
                                    
                                    # If this is the currently active spindle file, update that too
                                    if self.spindles_df is not None:
                                        self.spindles_df = self.spindles_df[
                                            ~self.spindles_df['UUID'].str.startswith(uuid_part)
                                        ]
                                    
                                    break
                            
                        # Update visualization
                        # If we're in the events view, reload the events
                        current_channel = self.channel_selector.currentText().split(" 🔍")[0]
                        if current_channel in self.selected_channels_data:
                            # Reload this channel's data without the deleted spindle
                            events_data = load_spindle_event_data(
                                self.dataset, 
                                current_channel, 
                                self.spindles_json, 
                                time_window=(1.0, 2.0)
                            )
                            if events_data:
                                self.selected_channels_data[current_channel] = events_data
                                self.update_visbrain_with_events_data(events_data)
                        else:
                            # Regular view, just update annotations
                            self.update_visbrain_annotations()
                        
                        self.statusBar().showMessage("Spindle deleted")

    # MODIFIED: Updated to work with multiple JSON files
    def save_spindles(self):
        """Save spindles to JSON and CSV files"""
        if not self.all_spindle_jsons:
            self.statusBar().showMessage("No spindle data to save")
            return
            
        # Create backups before overwriting
        self.backup_original_files()

        try:
            # Save each JSON file
            for json_name, json_data in self.all_spindle_jsons.items():
                # Save JSON file
                with open(json_data['json_path'], 'w') as f:
                    json.dump(json_data['spindles'], f, indent=2)
                    
                # Save CSV file
                json_data['df'].to_csv(json_data['csv_path'], index=False)
                    
            self.statusBar().showMessage(f"Saved {len(self.all_spindle_jsons)} spindle files successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving spindle files: {str(e)}")

    # MODIFIED: Updated to save all JSON files
    def save_spindles_as(self):
        """Save spindles to new JSON and CSV files"""
        if not self.all_spindle_jsons:
            self.statusBar().showMessage("No spindle data to save")
            return
        
        try:
            saved_count = 0
            
            # Save each JSON file individually
            for json_name, json_data in self.all_spindle_jsons.items():
                # Get filename for JSON file
                default_path = os.path.join(
                    os.path.dirname(json_data['json_path']),
                    f"EDITED_{os.path.basename(json_data['json_path'])}"
                )
                
                json_path, _ = QFileDialog.getSaveFileName(
                    self, f"Save {json_name} As", default_path, "JSON Files (*.json)"
                )
                
                if not json_path:
                    continue  # Skip this one if canceled
                    
                # Create default CSV path based on JSON path
                default_csv_path = json_path.replace('.json', '.csv')
                
                # Get filename for CSV file
                csv_path, _ = QFileDialog.getSaveFileName(
                    self, f"Save CSV for {json_name} As", default_csv_path, "CSV Files (*.csv)"
                )
                
                if not csv_path:
                    continue  # Skip this one if canceled
                    
                # Save JSON file
                with open(json_path, 'w') as f:
                    json.dump(json_data['spindles'], f, indent=2)
                    
                # Update internal path
                json_data['json_path'] = json_path
                    
                # Save CSV file
                json_data['df'].to_csv(csv_path, index=False)
                    
                # Update internal path
                json_data['csv_path'] = csv_path
                
                saved_count += 1
                    
            if saved_count > 0:
                self.statusBar().showMessage(f"Saved {saved_count} spindle files")
            else:
                self.statusBar().showMessage("No files were saved")
                
        except Exception as e:
            self.statusBar().showMessage(f"Error saving spindle files: {str(e)}")           

   # MODIFIED: Updated to backup all files
    def backup_original_files(self):
        """Create a backup of original files before overwriting"""
        if not self.all_spindle_jsons:
            return
            
        try:
            # Create timestamp for backup files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for json_name, json_data in self.all_spindle_jsons.items():
                # Create backup paths
                json_backup = f"{json_data['json_path']}.{timestamp}.bak"
                csv_backup = f"{json_data['csv_path']}.{timestamp}.bak"
                
                # Copy files
                if os.path.exists(json_data['json_path']):
                    with open(json_data['json_path'], 'r') as src, open(json_backup, 'w') as dst:
                        dst.write(src.read())
                        
                if os.path.exists(json_data['csv_path']):
                    with open(json_data['csv_path'], 'r') as src, open(csv_backup, 'w') as dst:
                        dst.write(src.read())
                    
            self.statusBar().showMessage(f"Backup created at {timestamp}")
            return True
        except Exception as e:
            self.statusBar().showMessage(f"Warning: Could not create backup: {str(e)}")
            return False

    def format_time(self, seconds):
        """Format seconds as HH:MM:SS.sss"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TurtlewaveVisbrainApp()
    window.show()
    sys.exit(app.exec_())