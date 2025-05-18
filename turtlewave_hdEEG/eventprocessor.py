
import numpy as np
import time
import os
import multiprocessing
import csv
from wonambi.trans import select, fetch, math
from wonambi.attr import Annotations
from turtlewave_hdEEG.extensions import ImprovedDetectSpindle as DetectSpindle
import json
import datetime
import logging


class ParalEvents:
    """
    A class for parallel detection and analysis of EEG events such as spindles,
    slow waves, and other neural events across multiple channels.
    """
    
    def __init__(self, dataset, annotations=None,log_level=logging.INFO, log_file=None):
        """
        Initialize the ParalEvents object.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset object containing EEG data
        annotations : XLAnnotations
            Annotations object for storing and retrieving events
        log_level : int
            Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file : str or None
            Path to log file. If None, logs to console only.
        """
        self.dataset = dataset
        self.annotations = annotations
        # Setup logging
        self.logger = self._setup_logger(log_level, log_file)
    
    def _setup_logger(self, log_level, log_file=None):
        """
        Set up a logger for the EventProcessor.
        
        Parameters
        ----------
        log_level : int
            Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file : str or None
            Path to log file. If None, logs to console only.
            
        Returns
        -------
        logger : logging.Logger
            Configured logger instance
        """
        # Create a logger
        logger = logging.getLogger('turtlewave_hdEEG.eventprocessor')
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_file specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def detect_spindles(self, method='Ferrarelli2007', chan=None, ref_chan=[], grp_name='eeg',
                       frequency=(11, 16), duration=(0.5, 3), polar='normal', 
                       reject_artifacts=True, reject_arousals=True,stage=None, cat=None,
                       save_to_annotations=False, json_dir=None, create_empty_json=True):
        """
        Detect spindles in the dataset while considering artifacts and arousals.
        
        Parameters
        ----------
        method : str or list
            Detection method(s) to use ('Ferrarelli2007', 'Wamsley2012', etc.)
        chan : list or str
            Channels to analyze
        ref_chan : list or str
            Reference channel(s) for re-referencing, or None to use original reference
        grp_name : str
            Group name for channel selection
        frequency : tuple
            Frequency range for spindle detection (min, max)
        duration : tuple
            Duration range for spindle detection in seconds (min, max)
        polar : str
            'normal' or 'opposite' for handling signal polarity
        reject_artifacts : bool
            Whether to exclude segments marked with artifact annotations
        reject_arousals : bool
            Whether to exclude segments marked with arousal annotations
        json_dir : str or None
            Directory to save individual channel JSON files (one per channel)
        create_empty_json : bool
            Whether to create empty JSON files when no spindles are found
        Returns
        -------
        list
            List of all detected spindles
        """
        import uuid 
        
        self.logger.info(r"""Whaling it... (searching for spindles)
                              .
                           ":"
                         ___:____     |"\/"|
                       ,'        `.    \  /
                       |  O        \___/  |
                     ~^~^~^~^~^~^~^~^~^~^~^~^~
                     """)
                     
        
        # Configure what to reject
        reject_types = []
        if reject_artifacts:
            reject_types.append('Artefact')
            self.logger.debug("Configured to reject artifacts")
        if reject_arousals:
            reject_types.extend(['Arousal'])
            self.logger.debug("Configured to reject arousals")

        # Make sure method is a list
        if isinstance(method, str):
            method = [method]
        
        # Make sure chan is a list
        if isinstance(chan, str):
            chan = [chan]
        
        # Make sure stage is a list
        if isinstance(stage, str):
            stage = [stage]
        
        # Create json_dir if specified
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
            self.logger.info(f"Channel JSONs will be saved to: {json_dir}")
        
        # Verify that we have all required components
        if self.dataset is None:
            self.logger.error("Error: No dataset provided for spindle detection")
            return []
        
        if self.annotations is None and save_to_annotations:
            self.logger.warning("Warning: No annotations provided but annotation saving requested.")
            self.logger.warning("Spindles will not be saved to annotations.")
            save_to_annotations = False

        # Convert method to string
        method_str = "_".join(method) if isinstance(method, list) else str(method)
        
        # Convert frequency to string
        freq_str = f"{frequency[0]}-{frequency[1]}Hz"

        self.logger.info(f"Starting spindle detection with method={method_str}, frequency={freq_str}")
        self.logger.debug(f"Parameters: channels={chan}, reject_artifacts={reject_artifacts}, reject_arousals={reject_arousals}")

        # Create a custom annotation file name if saving to annotations
        if save_to_annotations:
            # Convert channel list to string
            chan_str = "_".join(chan) if len(chan) <= 3 else f"{chan[0]}_plus_{len(chan)-1}_chans"
            
            
            # Create custom filename
            annotation_filename = f"spindles_{method_str}_{chan_str}_{freq_str}.xml"
             # Create full path if json_dir is specified
            if json_dir:
                annotation_file_path = os.path.join(json_dir, annotation_filename)
            else:
                # Use current directory
                annotation_file_path = annotation_filename
                
            # Create new annotation object if we're saving to a new file
            if self.annotations is not None:
                try:
                    # Create a copy of the original annota
                    import shutil
                    if hasattr(self.annotations, 'xml_file') and os.path.exists(self.annotations.xml_file):
                        shutil.copy(self.annotations.xml_file, annotation_file_path)
                        new_annotations = Annotations(annotation_file_path)
                        try:
                            spindle_events = new_annotations.get_events('spindle')
                            if spindle_events:
                                self.logger.info(f"Removing {len(spindle_events)} existing spindle events")
                                new_annotations.remove_event_type('spindle')
                        except Exception as e:
                            self.logger.error(f"Note: No existing spindle events to remove: {e}")
                    else:
                        # If we can't copy, create a new annotations file from scratch
                        # Create minimal XML structure
                        with open(annotation_file_path, 'w') as f:
                            f.write('<?xml version="1.0" ?>\n<annotations><dataset><filename>')
                            if hasattr(self.dataset, 'filename'):
                                f.write(self.dataset.filename)
                            f.write('</filename></dataset><rater><name>Wonambi</name></rater></annotations>')
                        new_annotations = Annotations(annotation_file_path)
                    print(f"Will save spindles to new annotation file: {annotation_file_path}")    

                except Exception as e:
                    self.logger.error(f"Error creating new annotation file: {e}")
                    save_to_annotations = False
                    new_annotations = None
            else:
                self.logger.warning("Warning: No annotations provided but annotation saving requested.")
                self.logger.error("Spindles will not be saved to annotations.")
                save_to_annotations = False
                new_annotations = None

        # Store all detected spindles
        all_spindles = []

        for ch in chan:
                try:
                    self.logger.info(f'Reading data for channel {ch}')
                    
                    # Fetch segments, filtering based on stage and artifacts
                    segments = fetch(self.dataset, self.annotations, cat=cat, stage=stage, cycle=None, 
                                    reject_epoch=True, reject_artf=reject_types)
                    segments.read_data(ch, ref_chan, grp_name=grp_name)

                    
                    # Process each detection method
                    channel_spindles = []
                    channel_json_spindles = []
                    ## Loop through methods (i.e. WHALE IT!)
                    for m, meth in enumerate(method):
                        self.logger.info(f"Applying method: {meth}")
                        ### define detection
                        detection = DetectSpindle(meth, frequency=frequency, duration=duration)
                            
                        for i, seg in enumerate(segments):
                            self.logger.info(f'Detecting events, segment {i + 1} of {len(segments)}')

                            # Apply polarity adjustment if needed
                            if polar == 'normal':
                                pass # No change needed
                            elif polar == 'opposite':
                                seg['data'].data[0][0] = seg['data'].data[0][0]*-1
                            # Run detection
                            spindles = detection(seg['data'])

                            if spindles and save_to_annotations and new_annotations is not None:
                                spindles.to_annot(new_annotations, 'spindle')
                            
                            # Add to our results
                            # Convert to dictionary format for consistency
                            for sp in spindles:
                                # Add UUID to each spindle
                                sp['uuid'] = str(uuid.uuid4())
                                # Add channel information
                                sp['chan'] = ch
                                channel_spindles.append(sp)
                                
                                # Add to JSON 
                                if json_dir:
                                    # Extract key properties in a serializable format
                                    sp_data = {
                                        'uuid': sp['uuid'],
                                        'chan': ch,
                                        'start_time': float(sp.get('start', 0)),
                                        'end_time': float(sp.get('end', 0)),
                                        'peak_time': float(sp.get('peak_time', 0)),
                                        'duration': float(sp.get('dur', 0)),
                                        'ptp_det': float(sp.get('ptp_det', 0)),
                                        'method': meth
                                    }
                                    
                                    sp_data['stage'] = stage
                                    sp_data['freq_range'] = frequency
                                    # Add frequency/power/amplitude if available
                                    if 'peak_freq' in sp:
                                        sp_data['peak_freq'] = float(sp['peak_freq'])
                                    if 'peak_val' in sp:
                                        sp_data['peak_val'] = float(sp['peak_val'])
                                    if 'power' in sp:
                                        sp_data['power'] = float(sp['power'])
                                        
                                    channel_json_spindles.append(sp_data)
                    all_spindles.extend(channel_spindles)
                    self.logger.info(f"Found {len(channel_spindles)} spindles in channel {ch}")
                    stages_str = "".join(stage)
                    if json_dir:
                        try:
                            ch_json_file = os.path.join(json_dir, f"spindles_{method_str}_{freq_str}_{stages_str}_{ch}.json")

                            # Create empty JSON if no spindles found but flag is set
                            if not channel_json_spindles and create_empty_json:
                                self.logger.info(f"Creating empty JSON file for channel {ch} (no spindles detected)")
                                with open(ch_json_file, 'w') as f:
                                    json.dump([], f)
                            elif channel_json_spindles:
                                with open(ch_json_file, 'w') as f:
                                    json.dump(channel_json_spindles, f, indent=2)
                                self.logger.info(f"Saved spindle data for channel {ch} to {ch_json_file}")
                        except Exception as e:
                            self.logger.error(f"Error saving channel JSON: {e}")
                except Exception as e:        
                        self.logger.warning(f'WARNING: No spin channel {ch}: {e}')
                        
                        # Create empty JSON file even in case of error
                        if json_dir and create_empty_json:
                            try:
                                stages_str = "".join(stage) if stage else "all"
                                ch_json_file = os.path.join(json_dir, f"spindles_{method_str}_{freq_str}_{stages_str}_{ch}.json")
                                with open(ch_json_file, 'w') as f:
                                    json.dump([], f)
                                self.logger.info(f"Created empty JSON file for channel {ch} after error")
                            except Exception as json_e:
                                self.logger.error(f"Error creating empty JSON for channel {ch}: {json_e}")
        
        # Save the new annotation file if needed
        if save_to_annotations and new_annotations is not None and all_spindles:
            try:
                new_annotations.save(annotation_file_path)
                self.logger.info(f"Saved {len(all_spindles)} spindles to new annotation file: {annotation_file_path}")
            except Exception as e:
                self.logger.error(f"Error saving annotation file: {e}")



        # Return all detected spindles
        self.logger.info(f"Total spindles detected across all channels: {len(all_spindles)}")
        return all_spindles
    
    
    def export_spindle_parameters_to_csv(self, json_input, csv_file, export_params='all', 
                              frequency=None, ref_chan=None, grp_name='eeg', n_fft_sec=4, file_pattern=None,skip_empty_files=True):
        """
        Calculate spindle parameters from JSON files and export to CSV.
        
        Parameters
        ----------
        json_input : str or list
            Path to JSON file, directory of JSON files, or list of JSON files
        csv_file : str
            Path to output CSV file
        export_params : dict or str
            Parameters to export. If 'all', exports all available parameters
        frequency : tuple or None
            Frequency range for power calculations (default: None, uses original range from JSON)
        ref_chan : list or None
            Reference channel(s) to use for parameter calculation
        n_fft_sec : int
            FFT window size in seconds for spectral analysis
        file_pattern : str or None
            Pattern to filter JSON files if json_input is a directory
        grp_name : str
            Group name for channel selection
        skip_empty_files : bool
            Whether to skip empty JSON files or include them in the report
        Returns
        -------
        dict
            Dictionary of calculated parameters
        """
        from wonambi.trans.analyze import event_params, export_event_params
        import glob
        
        self.logger.info("Calculating spindle parameters for CSV export...")
         
        # Load spindles from JSON file(s)
        json_files = []
        if file_pattern:
            # Get all JSON files in the directory
            all_json_files = glob.glob(os.path.join(json_input, "*.json"))
            # Match files where pattern is followed by underscore or dot
            json_files = [f for f in all_json_files if 
                        f"{file_pattern}_" in os.path.basename(f) or 
                        f"{file_pattern}." in os.path.basename(f)]
        else:
            # If no pattern, get all JSON files
            json_files = glob.glob(os.path.join(json_input, "*.json"))


        self.logger.info(f"Found {len(json_files)} JSON files matching pattern: {file_pattern}")
        
        
        # Load spindles from JSON files
        all_spindles = []
        empty_channels = []
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    spindles = json.load(f)
                    
                if isinstance(spindles, list):
                    if len(spindles) > 0:
                            all_spindles.extend(spindles)
                    else:
                        # Extract channel name from filename
                        filename = os.path.basename(file)
                        parts = filename.split('_')
                        if len(parts) > 1:
                            chan = parts[-1].replace('.json', '')
                            empty_channels.append(chan)
                        self.logger.info(f"File {file} contains an empty list (no spindles)")
                else:
                    self.logger.warning(f"Warning: Unexpected format in {file}")
                    
                self.logger.info(f"Loaded {len(spindles) if isinstance(spindles, list) else 0} spindles from {file}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        if not all_spindles:
            self.logger.info("No spindles found in the input files")
            # Create an empty CSV file with header to indicate processing was done
            if empty_channels and not skip_empty_files:
                try:
                    with open(csv_file, 'w', newline='') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(["No spindles were detected in the following channels:"])
                        for chan in empty_channels:
                            writer.writerow([chan])
                    self.logger.info(f"Created empty CSV file at {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error creating empty CSV: {e}")
            return None

        
        # Get frequency band from spindles if not provided
        if frequency is None:
            try:
                # Try to extract from the first spindle
                if 'freq_range' in all_spindles[0]:
                    freq_range = all_spindles[0]['freq_range']
                    if isinstance(freq_range, list) and len(freq_range) == 2:
                        frequency = tuple(freq_range)
                    elif isinstance(freq_range, str) and '-' in freq_range:
                        freq_parts = freq_range.split('-')
                        frequency = (float(freq_parts[0].replace('Hz', '').strip()), 
                                    float(freq_parts[1].replace('Hz', '').strip()))
                        self.logger.info(f"Using frequency range from JSON: {frequency}")
            except:
                # Default if we can't extract
                frequency = (11, 16)
                self.logger.info(f"Using default frequency range: {frequency}")
        

        # Get sampling frequency from dataset
        try:
            s_freq = self.dataset.header['s_freq']
            #print(f"Dataset sampling frequency: {s_freq} Hz")
        except:
            self.logger.info("Could not determine dataset sampling frequency")
            return None
        
        # Try to get recording start time if not provided
        recording_start_time = None
        try:
            # Get it from dataset header
            if hasattr(self.dataset, 'header'):
                header = self.dataset.header
                if hasattr(header, 'start_time'):
                    recording_start_time = header.start_time
                elif isinstance(header, dict) and 'start_time' in header:
                    recording_start_time = header['start_time']
                    
            if recording_start_time:
                self.logger.info(f"Found recording start time: {recording_start_time}")
            else:
                self.logger.warning("Warning: Could not find recording start time in dataset header. Using relative time only.")
        except Exception as e:
            self.logger.error(f"Error getting recording start time: {e}")
            self.logger.warning("Warning:Using relative time only.")

        
        # Group spindles by channel for more efficient processing
        spindles_by_chan = {}
        for sp in all_spindles:
            chan = sp.get('chan')
            if chan not in spindles_by_chan:
                spindles_by_chan[chan] = []
            spindles_by_chan[chan].append(sp)

        self.logger.info(f"Grouped spindles by {len(spindles_by_chan)} channels")

        # Process each channel in turn
        all_segments = []

        # Load data for each channel and create segments
        for chan, spindles in spindles_by_chan.items():
            self.logger.info(f"Processing {len(spindles)} spindles for channel {chan}")

            # Use fetch for proper segmentation - critical fix
            try:
                # Create a list of time windows for spindles
                spindle_windows = []
                for sp in spindles:
                    start_time = sp['start_time']
                    end_time = sp['end_time']
                    spindle_windows.append((start_time, end_time))
                

                # Use direct segment creation for better power calculation
                for i, (start_time, end_time) in enumerate(spindle_windows):
                    try:
                        # Add a small buffer for FFT calculation
                        buffer = 0.1  # 100ms buffer
                        start_with_buffer = max(0, start_time - buffer)
                        end_with_buffer = end_time + buffer
                        
                        # Read data for this specific spindle
                        data = self.dataset.read_data(chan=[chan], 
                                                    begtime=start_with_buffer, 
                                                    endtime=end_with_buffer)
                        # Create a segment for this spindle
                        seg = {
                            'data': data,
                            'name': 'spindle',
                            'start': start_time,
                            'end': end_time,
                            'n_stitch': 0,
                            'stage': spindles[i].get('stage'),
                            'cycle': None,
                            'chan': chan,  # Important: store the channel
                            'uuid': spindles[i].get('uuid', str(i))  # Store ID for tracking
                        }
                        all_segments.append(seg)

                    except Exception as e:
                        self.logger.error(f"Error creating segment for spindle {start_time}-{end_time}: {e}")

            except Exception as e:
                self.logger.error(f"Error processing channel {chan}: {e}")
    
        
        if not all_segments:
            self.logger.error("No valid segments created for parameter calculation")
            return None
        
        self.logger.info(f"Created {len(all_segments)} segments for parameter calculation")
        
        # Calculate parameters
        n_fft = None
        if all_segments and n_fft_sec is not None:
            n_fft = int(n_fft_sec * s_freq)                
        
        # Create a temporary file to use for the initial export
        temp_csv = csv_file + '.temp'

        try:
            # Calculate parameters with proper FFT settings
            self.logger.info(f"Calculating parameters with frequency band {frequency} and n_fft={n_fft}")
            params = event_params(all_segments, export_params, band=frequency, n_fft=n_fft)
            
            if not params:
                self.logger.info("No parameters calculated")
                return None
            
            # Export parameters to temporary CSV file
            self.logger.info(f"Exporting parameters to temporary file")            
            export_event_params(temp_csv, params, count=None, density=None)

            # Store UUIDs for later use (they're not included in the params for CSV export)
            uuid_dict = {}
            for i, segment in enumerate(all_segments):
                if 'uuid' in segment:
                    uuid_dict[i] = segment['uuid']

            # Now read the temporary CSV and process it
            self.logger.info(f"Processing CSV to remove summary rows and add HH:MM:SS format")
            with open(temp_csv, 'r', newline='') as infile, open(csv_file, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                # Read all rows
                all_rows = list(reader)

                # Find the header row (the one with 'Start time')
                header_row_index = None
                start_time_index = None
                for i, row in enumerate(all_rows):
                    if row and 'Start time' in row:
                        header_row_index = i
                        start_time_index = row.index('Start time')
                        break
                
                if header_row_index is None or start_time_index is None:
                    self.logger.info("Error: Could not find 'Start time' column in CSV")
                    # Copy the original file as fallback
                    with open(temp_csv, 'r') as src, open(csv_file, 'w') as dst:
                        dst.write(src.read())
                    return params
            
                # Create filtered rows without Mean, SD, Mean of ln, SD of ln
                filtered_rows = []
            
                # Add any prefix rows before the header (like 'Wonambi v7.15')
                for i in range(header_row_index):
                    filtered_rows.append(all_rows[i])

                # Add the header row and add 'Start time (HH:MM:SS)' and 'UUID' columns
                header_row = all_rows[header_row_index].copy()
                # Add 'Start time (HH:MM:SS)' right after 'Start time'
                header_row.insert(start_time_index + 1, 'Start time (HH:MM:SS)')

                # Add UUID column if not already present
                if 'UUID' not in header_row:
                    header_row.append('UUID')
                filtered_rows.append(header_row)

                # Skip the header row and the 4 statistic rows (Mean, SD, Mean of ln, SD of ln)
                # and add the rest of the data rows
                for i in range(header_row_index + 5, len(all_rows)):
                    row = all_rows[i]
                    if not row:  # Skip empty rows
                        continue
                        
                    # Make a copy of the row to modify
                    new_row = row.copy()
                    
                    # Add the HH:MM:SS time format after the start time
                    if len(row) > start_time_index:
                        try:
                            start_time_sec = float(row[start_time_index])
                            
                            # Convert to HH:MM:SS
                            def sec_to_time(seconds):
                                hours = int(seconds // 3600)
                                minutes = int((seconds % 3600) // 60)
                                sec = seconds % 60
                                return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"
                                
                            # Calculate clock time if recording start time is available
                            if recording_start_time is not None:
                                try:
                                    delta = datetime.timedelta(seconds=start_time_sec)
                                    event_time = recording_start_time + delta
                                    start_time_hms = event_time.strftime('%H:%M:%S.%f')[:-3]
                                except:
                                    start_time_hms = sec_to_time(start_time_sec)
                            else:
                                start_time_hms = sec_to_time(start_time_sec)
                            
                            # Insert the HH:MM:SS time
                            new_row.insert(start_time_index + 1, start_time_hms)
                        except (ValueError, IndexError):
                            # If we can't convert, insert empty cell
                            new_row.insert(start_time_index + 1, '')
                    else:
                        # Row is too short, insert empty cell
                        new_row.insert(start_time_index + 1, '')
                    
                    # Add UUID at the end 
                    # Calculate the segment index
                    segment_index = i - (header_row_index + 5)
                    if segment_index in uuid_dict:
                        new_row.append(uuid_dict[segment_index])
                    else:
                        new_row.append('')
                    
                    filtered_rows.append(new_row)
                
                # Write all filtered rows
                for row in filtered_rows:
                    writer.writerow(row)
                   # Remove the temporary file
            try:
                os.remove(temp_csv)
            except:
                self.logger.info(f"Note: Could not remove temporary file {temp_csv}")

            self.logger.info(f"Successfully exported to {csv_file} with HH:MM:SS time format")
            return params
        except Exception as e:
            self.logger.error(f"Error calculating parameters: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def export_spindle_density_to_csv(self, json_input, csv_file, stage=None, file_pattern=None):
        """
        Export spindle statistics to CSV with both whole night and stage-specific densities.
        
        Parameters
        ----------
        json_input : str or list
            Path to JSON file, directory of JSON files, or list of JSON files
        csv_file : str
            Path to output CSV file
        stage : str or list
            Sleep stage(s) to include (e.g., 'NREM2', ['NREM2', 'NREM3'])
            if None, will extract stages from spindles
        file_pattern : str or None
        Returns
        -------
        dict
            Dictionary with spindle statistics by channel
        """
        import os
        import json
        import glob
        import csv
        import numpy as np
        from collections import defaultdict
        
        # Load spindles from JSON file(s)
        json_files = []
        if file_pattern:
            # Get all JSON files in the directory
            all_json_files = glob.glob(os.path.join(json_input, "*.json"))
            # Match files where pattern is followed by underscore or dot
            json_files = [f for f in all_json_files if 
                        f"{file_pattern}_" in os.path.basename(f) or 
                        f"{file_pattern}." in os.path.basename(f)]
        else:
            # If no pattern, get all JSON files
            json_files = glob.glob(os.path.join(json_input, "*.json"))

        self.logger.info(f"Found {len(json_files)} JSON files matching pattern: {file_pattern}")
        if not json_files:
            self.logger.error(f"No JSON files found matching pattern: {file_pattern}")
            
            # Create an empty CSV file with a message
            try:
                with open(csv_file, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["No JSON files found matching pattern:", file_pattern])
                self.logger.info(f"Created empty CSV file at {csv_file}")
            except Exception as e:
                self.logger.error(f"Error creating empty CSV: {e}")
                
            return None



        # Prepare the stages as a list
        if stage is None:
            combined_stages = False
            stage_list = None
        elif isinstance(stage, list) and len(stage) > 1:
            combined_stages = True
            stage_list = stage
            combined_stage_name = "+".join(stage_list)
            self.logger.info(f"Calculating combined spindle density for stages: {combined_stage_name}")
        elif isinstance(stage, list) and len(stage) == 1:
            combined_stages = False
            stage_list = [stage[0]]
            self.logger.info(f"Calculating spindle density for stage: {stage_list[0]}")
        else:
            combined_stages = False
            stage_list = [stage]
            self.logger.info(f"Calculating spindle density for stage: {stage}")

        

        all_spindles = []
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    spindles = json.load(f)
                    all_spindles.extend(spindles if isinstance(spindles, list) else [])
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        # Get stage durations from annotations (assuming annotations are available)
        epoch_duration_sec = 30  # Standard epoch duration
        
        # Count epochs for each stage
        stage_counts = defaultdict(int)
        all_stages = self.annotations.get_stages()

                                
        # Count epochs for each stage
        for s in all_stages:
            if s in ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']:
                stage_counts[s] += 1


        # Calculate durations in minutes
        stage_durations = {stg: count * epoch_duration_sec / 60 for stg, count in stage_counts.items()}
         
        total_duration_min = sum(stage_durations.values())
    
        # Extract stages from spindles if stage is None
        spindle_stages = set()
        for sp in all_spindles:
            if not isinstance(sp, dict) or 'stage' not in sp:
                continue        
            sp_stage = sp['stage']
            if isinstance(sp_stage, list):
                for s in sp_stage:
                    spindle_stages.add(str(s))
            else:
                spindle_stages.add(str(sp_stage))
        
        # If stage is None, process all stages found in spindles
        if stage is None:
            stages_to_process = sorted(spindle_stages)
            combined_stages = False
        elif combined_stages:
            # Just process the combined stage set
            stages_to_process = [stage_list]  # List containing the list of stages
        else:
            # Process individual stages
            stages_to_process = stage_list

        # Group spindles by channel and stage
        spindles_by_chan_stage = defaultdict(lambda: defaultdict(list))
        spindles_by_chan = defaultdict(list)
        
        for sp in all_spindles:
            if not isinstance(sp, dict):
                continue
            # Get channel information
            chan = None
            if 'chan' in sp:
                chan = sp['chan']
            elif 'channel' in sp:
                chan = sp['channel']
            if not chan:
                continue
        
            
            # Add to whole night spindle count
            spindles_by_chan[chan].append(sp)
            
            if not combined_stages:
                # Process stage info, handling multiple stages per spindle
                if 'stage' in sp:
                    sp_stages = sp['stage'] if isinstance(sp['stage'], list) else [sp['stage']]
            
                for sp_stage in sp_stages:
                    sp_stage = str(sp_stage)  # Convert to string for consistency
                    # Add to stage-specific spindle count
                    spindles_by_chan_stage[chan][sp_stage].append(sp)
                

        # Calculate statistics by channel for each stage
        stage_channel_stats = defaultdict(dict)
        for chan in set(spindles_by_chan.keys()):
            # Whole night statistics
            all_chan_spindles = spindles_by_chan[chan]
        
            
            for process_stage in stages_to_process:
                # Get spindles for this channel and stage
                stage_spindles = []
                if combined_stages or (isinstance(process_stage, list) and len(process_stage) > 1):
                    stages_to_include = process_stage if isinstance(process_stage, list) else stage_list
                    stage_name_display = "+".join(stages_to_include)
                    # Create a set of stages to check against
                    stages_set = set(str(s) for s in stages_to_include)
                    # Find spindles that belong to ANY of the target stages, but count each spindle only once
                    stage_spindles = []
                    seen_spindles = set()  # Track spindles we've already counted

                    for sp in all_chan_spindles:
                        if 'stage' not in sp:
                            continue
                        # Get spindle's stages as a set
                        sp_stages = sp['stage'] if isinstance(sp['stage'], list) else [sp['stage']]
                        sp_stages = set(str(s) for s in sp_stages)

                        # Check if any of the spindle's stages match any target stage
                        if sp_stages.intersection(stages_set) and id(sp) not in seen_spindles:
                            stage_spindles.append(sp)
                            seen_spindles.add(id(sp))

                    # Sum durations for all specified stages
                    stage_duration_min = sum(stage_durations.get(s, 0) for s in stages_to_include)
        
                else:
                    # Single stage processing
                    s_str = str(process_stage)
                    stage_spindles = spindles_by_chan_stage[chan].get(s_str, [])
                    stage_name_display = process_stage
                    stage_duration_min = stage_durations.get(s_str, 0)
            
                # Skip if no spindles for this stage and channel
                if len(stage_spindles) == 0:
                    continue
            
                # Count spindles
                stage_count = len(stage_spindles)
                whole_night_count = len(all_chan_spindles)
                
                # Calculate density (spindles per minute)
                stage_density = stage_count / stage_duration_min if stage_duration_min > 0 else 0
                whole_night_density = whole_night_count / total_duration_min if total_duration_min > 0 else 0
                
                # Calculate mean duration of spindles
                durations = []
                for sp in stage_spindles:
                    if 'start_time' in sp and 'end_time' in sp:
                        durations.append(sp['end_time'] - sp['start_time'])
                
                mean_duration = np.mean(durations) if durations else 0
                
                # Store the statistics
                key = tuple(process_stage) if isinstance(process_stage, list) else process_stage
                stage_channel_stats[key][chan] = {
                    'count': stage_count,
                    'stage_density': stage_density,
                    'whole_night_density': whole_night_density,
                    'mean_duration': mean_duration,
                    'stage_name_display': stage_name_display,
                    'stage_duration_min': stage_duration_min,
                }
        
        # Export to CSV - each stage gets its own section
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Add whole night summary
            writer.writerow(['Whole Night Summary'])
            writer.writerow(['Total Recording Duration (min)', f'{total_duration_min:.2f}'])
            writer.writerow([])
            
            # Add stage duration summary
            writer.writerow(['Stage Duration Summary'])
            writer.writerow(['Stage', 'Duration (min)'])
            for stg in sorted(set(stage_durations.keys())):
                writer.writerow([stg, f"{stage_durations.get(stg, 0):.2f}"])
            # If combined stages were requested, add their summary too
            if combined_stages:
                combined_duration = sum(stage_durations.get(s, 0) for s in stage_list)
                writer.writerow([combined_stage_name, f"{combined_duration:.2f}"])

            writer.writerow([])
            
            # Process each stage
            for process_stage in stages_to_process:
                key = tuple(process_stage) if isinstance(process_stage, list) else process_stage
                # Skip if no data for this stage
                if key not in stage_channel_stats:
                    continue
                # Get any channel's stats to extract the stage name display
                any_chan = next(iter(stage_channel_stats[key].keys()))
                stage_name_display = stage_channel_stats[key][any_chan]['stage_name_display']

                # Add stage header
                writer.writerow([f"Sleep Stage: {stage_name_display}"])
                writer.writerow([
                    'Channel', 
                    'Count',
                    f'Density in {stage_name_display} (events/min)', 
                    'Whole Night Density (events/min)',
                    'Mean Duration (s)'
                ])

                
                # Write channel-specific statistics, sorted by channel name
                for chan in sorted(stage_channel_stats[key].keys()):
                    stats = stage_channel_stats[key][chan]
                    writer.writerow([
                        chan, 
                        stats['count'],
                        f"{stats['stage_density']:.4f}",
                        f"{stats['whole_night_density']:.4f}",
                        f"{stats['mean_duration']:.4f}"
                    ])
                
                writer.writerow([])
        
        self.logger.info(f"Exported spindle statistics to {csv_file}")
        return dict(stage_channel_stats)
            
