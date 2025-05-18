import numpy as np
import time
import os
#import multiprocessing
import csv
from wonambi.trans import select, fetch, math
from wonambi.attr import Annotations
from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave as DetectSlowWave
import json
import datetime
import logging


class ParalSWA:
    """
    A class for parallel detection and analysis of slow wave activity (SWA)
    across multiple channels.
    """
    
    def __init__(self, dataset, annotations=None, log_level=logging.INFO, log_file=None):
        """
        Initialize the ParalSWA object.
        
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
        Set up a logger for the SWAProcessor.
        
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
        logger = logging.getLogger('turtlewave_hdEEG.swaprocessor')
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

    def detect_slow_waves(self, method='Massimini2004', chan=None, ref_chan=[], grp_name='eeg',
                     frequency=(0.1, 4), trough_duration=(0.3, 1.5), 
                     neg_peak_thresh=-80.0,  
                     p2p_thresh=140.0,  
                     min_dur=None, max_dur=None,
                     detrend=False,
                     polar='normal', # normal vs opposite 
                     reject_artifacts=True, reject_arousals=True, 
                     stage=None, 
                     cat=None,
                     peak_thresh_sigma=None, 
                     ptp_thresh_sigma=None,
                     save_to_annotations=False, json_dir=None,
                     create_empty_json=True):
        """
        Detect slow waves in the dataset while considering artifacts and arousals.
        
        Parameters
        ----------
        method : str or list
            Detection method(s) to use ('Massimini2004', 'AASM/Massimini2004', 'Ngo2015', 'Staresina2015')
        chan : list or str
            Channels to analyze
        ref_chan : list or str
            Reference channel(s) for re-referencing
        grp_name : str
            Group name for channel selection
        frequency : tuple
            Frequency range for slow wave detection (min, max)
        trough_duration : tuple
            Duration range for slow wave trough in seconds (min, max)
        neg_peak_thresh : float
            Minimum negative peak threshold in μV
        p2p_thresh : float
            Minimum peak-to-peak amplitude threshold in μV
        peak_thresh_sigma : float or None
            Peak threshold in standard deviations (for Ngo2015 method)
        ptp_thresh_sigma : float or None
            Peak-to-peak threshold in standard deviations (for Ngo2015 method)
        invert : bool
            Whether to invert the signal polarity
        reject_artifacts : bool
            Whether to exclude segments marked with artifact annotations
        reject_arousals : bool
            Whether to exclude segments marked with arousal annotations
        stage : list or str
            Sleep stage(s) to analyze
        cat : tuple
            Category specification for data selection
        save_to_annotations : bool
            Whether to save detected slow waves to annotations
        json_dir : str or None
            Directory to save individual channel JSON files
        
        Returns
        -------
        list
            List of all detected slow waves
        """
        import uuid    
       
        self.logger.info(r"""
                   .==.                   
                  ()''()-.    Sweet Dreams...
                   .--''  (Detecting Slow Waves)
                 .'O_O  '._   ____  
                 _(z_z)_ /  |_||__|  
               ,'| |  /\   _//--'     
              /  |_|'||  |/|         
             |  | | |\   ||          
             |_/'\_|_| \_|'\         
        """)
        # Validate polar parameter
        if polar not in ['normal', 'opposite']:
            self.logger.warning(f"Invalid polar value '{polar}'. Using 'normal'.")
            polar = 'normal'
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
        
        # Verify required components
        if self.dataset is None:
            self.logger.error("Error: No dataset provided for slow wave detection")
            return []
        
        if self.annotations is None and save_to_annotations:
            self.logger.warning("Warning: No annotations provided but annotation saving requested.")
            self.logger.warning("Slow waves will not be saved to annotations.")
            save_to_annotations = False

        # Convert method to string
        method_str = "_".join(method).replace('/', '_') if isinstance(method, list) else str(method).replace('/', '_')

        # Convert frequency to string
        freq_str = f"{frequency[0]}-{frequency[1]}Hz"

        self.logger.info(f"Starting slow wave detection with method={method_str}, frequency={freq_str}")
        self.logger.debug(f"Parameters: channels={chan}, reject_artifacts={reject_artifacts}, reject_arousals={reject_arousals}")

        # Log adaptive threshold parameters if applicable
        first_method = method[0] if isinstance(method, list) and len(method) > 0 else method
        if first_method == 'Ngo2015' and peak_thresh_sigma is not None and ptp_thresh_sigma is not None:
            self.logger.info(f"Using adaptive thresholds: peak_thresh_sigma={peak_thresh_sigma}, ptp_thresh_sigma={ptp_thresh_sigma}")


        # Create custom annotation file name if saving to annotations
        if save_to_annotations:
            # Convert channel list to string
            chan_str = "_".join(chan) if len(chan) <= 3 else f"{chan[0]}_plus_{len(chan)-1}_chans"
            
            # Create custom filename
            annotation_filename = f"slowwaves_{method_str}_{chan_str}_{freq_str}.xml"
            
            # Create full path if json_dir is specified
            if json_dir:
                annotation_file_path = os.path.join(json_dir, annotation_filename)
            else:
                # Use current directory
                annotation_file_path = annotation_filename
                
            # Create new annotation object if we're saving to a new file
            if self.annotations is not None:
                try:
                    # Create a copy of the original annotations
                    import shutil
                    if hasattr(self.annotations, 'xml_file') and os.path.exists(self.annotations.xml_file):
                        shutil.copy(self.annotations.xml_file, annotation_file_path)
                        new_annotations = Annotations(annotation_file_path)
                        try:
                            sw_events = new_annotations.get_events('slow_wave')
                            if sw_events:
                                self.logger.info(f"Removing {len(sw_events)} existing slow wave events")
                                new_annotations.remove_event_type('slow_wave')
                        except Exception as e:
                            self.logger.error(f"Note: No existing slow wave events to remove: {e}")
                    else:
                        # Create new annotations file from scratch
                        with open(annotation_file_path, 'w') as f:
                            f.write('<?xml version="1.0" ?>\n<annotations><dataset><filename>')
                            if hasattr(self.dataset, 'filename'):
                                f.write(self.dataset.filename)
                            f.write('</filename></dataset><rater><name>Wonambi</name></rater></annotations>')
                        new_annotations = Annotations(annotation_file_path)
                    print(f"Will save slow waves to new annotation file: {annotation_file_path}")    

                except Exception as e:
                    self.logger.error(f"Error creating new annotation file: {e}")
                    save_to_annotations = False
                    new_annotations = None
            else:
                self.logger.warning("Warning: No annotations provided but annotation saving requested.")
                self.logger.error("Slow waves will not be saved to annotations.")
                save_to_annotations = False
                new_annotations = None



        # Store all detected slow waves
        all_slow_waves = []

        for ch in chan:
                try:
                    self.logger.info(f'Reading data for channel {ch}')
                    
                    # Fetch segments, filtering based on stage and artifacts
                    segments = fetch(self.dataset, self.annotations, cat=cat, stage=stage, cycle=None, 
                                  reject_epoch=True, reject_artf=reject_types)
                    segments.read_data(ch, ref_chan, grp_name=grp_name)

                    # Process each detection method
                    channel_slow_waves = []
                    channel_json_slow_waves = []
                    
                    ## Loop through methods
                    for m, meth in enumerate(method):
                        self.logger.info(f"Applying method: {meth}")
                            
                        for i, seg in enumerate(segments):
                            self.logger.info(f'Detecting events, segment {i + 1} of {len(segments)}')
                            # Create a copy of the segment for processing
                            processed_seg = seg.copy()

                            # Apply polarity adjustment if needed
                            if polar == 'opposite':
                                processed_seg['data'].data[0][0] = -processed_seg['data'].data[0][0]
                            elif polar == 'normal':
                                pass
                            self.logger.debug(f'Applied polarity inversion to segment {i + 1}')

                            if detrend:
                                self.logger.debug(f'Applying detrend to segment {i + 1}')
                                try:
                                    processed_seg['data'] = math(processed_seg['data'], operator='detrend', axis='time')
                                except Exception as e:
                                    self.logger.error(f"Error detrending data: {e}")

                            # Special handling for Ngo2015 with adaptive thresholds
                            detection_kwargs = {}
                            if meth == 'Ngo2015' and peak_thresh_sigma is not None and ptp_thresh_sigma is not None:
                                # Store sigma thresholds as class variables that the detector will use
                                detection_kwargs = {
                                    'peak_thresh': peak_thresh_sigma,
                                    'ptp_thresh': ptp_thresh_sigma
                                }
                                self.logger.debug(f"Using custom adaptive thresholds: {detection_kwargs}")

                            # Define detection with parameters
                            detection = DetectSlowWave(
                                meth,
                                frequency=frequency,
                                # MODIFIED: Use appropriate duration parameter based on method
                                duration=trough_duration if meth in ['Massimini2004', 'AASM/Massimini2004'] else None,
                                neg_peak_thresh=neg_peak_thresh,
                                p2p_thresh=p2p_thresh,
                                min_dur=min_dur if meth not in ['Massimini2004', 'AASM/Massimini2004'] else None,
                                max_dur=max_dur if meth not in ['Massimini2004', 'AASM/Massimini2004'] else None,
                                polar=polar,
                                **detection_kwargs  # Pass method-specific kwargs
                            )

                            # Run detection
                            slow_waves = detection(processed_seg['data'])

                            if slow_waves and save_to_annotations and new_annotations is not None:
                                slow_waves.to_annot(new_annotations, 'slow_wave')
                            
                            # Add to our results
                            # Convert to dictionary format for consistency
                            for sw in slow_waves:
                                # Add UUID to each slow wave
                                sw['uuid'] = str(uuid.uuid4())
                                # Add channel information
                                sw['chan'] = ch
                                channel_slow_waves.append(sw)
                                
                                # Add to JSON 
                                if json_dir:
                                    # Extract key properties in a serializable format
                                    sw_data = {
                                        'uuid': sw['uuid'],
                                        'chan': ch,
                                        'start_time': float(sw.get('start', 0)),
                                        'end_time': float(sw.get('end', 0)),
                                        'trough_time': float(sw.get('trough_time', 0)),
                                        'peak_time': float(sw.get('peak_time', 0)),
                                        'duration': float(sw.get('dur', 0)),
                                        'trough_val': float(sw.get('trough_val', 0)),
                                        'peak_val': float(sw.get('peak_val', 0)),
                                        'ptp': float(sw.get('ptp', 0)),
                                        'method': meth
                                    }
                                    
                                    sw_data['stage'] = stage
                                    sw_data['freq_range'] = frequency
                                    
                                    channel_json_slow_waves.append(sw_data)
                                    
                    all_slow_waves.extend(channel_slow_waves)
                    self.logger.info(f"Found {len(channel_slow_waves)} slow waves in channel {ch}")
                    
                    stages_str = "".join(stage) if stage else "all"
                    if json_dir :
                        try:
                            ch_json_file = os.path.join(json_dir, 
                                                      f"slowwaves_{method_str}_{freq_str}_{stages_str}_{ch}.json")
                            
                            # Create empty JSON if no waves found but flag is set
                            if not channel_json_slow_waves and create_empty_json:
                                self.logger.info(f"Creating empty JSON file for channel {ch} (no slow waves detected)")
                                with open(ch_json_file, 'w') as f:
                                    json.dump([], f)
                            elif channel_json_slow_waves:
                                with open(ch_json_file, 'w') as f:
                                    json.dump(channel_json_slow_waves, f, indent=2)
                                self.logger.info(f"Saved slow wave data for channel {ch} to {ch_json_file}")
                        except Exception as e:
                            self.logger.error(f"Error saving channel JSON: {e}")
                except Exception as e:        
                        self.logger.warning(f'WARNING: No slow waves in channel {ch}: {e}')
                        # Create empty JSON file even in case of error
                        if json_dir and create_empty_json:
                            try:
                                stages_str = "".join(stage) if stage else "all"
                                ch_json_file = os.path.join(json_dir, 
                                                        f"slowwaves_{method_str}_{freq_str}_{stages_str}_{ch}.json")
                                with open(ch_json_file, 'w') as f:
                                    json.dump([], f)
                                self.logger.info(f"Created empty JSON file for channel {ch} after error")
                            except Exception as json_e:
                                self.logger.error(f"Error creating empty JSON for channel {ch}: {json_e}")
        
        # Save the new annotation file if needed
        if save_to_annotations and new_annotations is not None and all_slow_waves:
            try:
                new_annotations.save(annotation_file_path)
                self.logger.info(f"Saved {len(all_slow_waves)} slow waves to new annotation file: {annotation_file_path}")
            except Exception as e:
                self.logger.error(f"Error saving annotation file: {e}")

        # Return all detected slow waves
        self.logger.info(f"Total slow waves detected across all channels: {len(all_slow_waves)}")
        return all_slow_waves
    
    def export_slow_wave_parameters_to_csv(self, json_input, csv_file, export_params='all', 
                                         frequency=None, ref_chan=None, grp_name='eeg', 
                                         n_fft_sec=4, file_pattern=None,skip_empty_files=True):
        """
        Calculate slow wave parameters from JSON files and export to CSV.
        
        Parameters
        ----------
        json_input : str or list
            Path to JSON file, directory of JSON files, or list of JSON files
        csv_file : str
            Path to output CSV file
        export_params : dict or str
            Parameters to export. If 'all', exports all available parameters
        frequency : tuple or None
            Frequency range for power calculations
        ref_chan : list or None
            Reference channel(s) for parameter calculation
        n_fft_sec : int
            FFT window size in seconds for spectral analysis
        file_pattern : str or None
            Pattern to filter JSON files if json_input is a directory
        """
        from wonambi.trans.analyze import event_params, export_event_params
        import glob
        
        self.logger.info("Calculating slow wave parameters for CSV export...")
         
        # Load slow waves from JSON file(s)
        json_files = []
        if file_pattern:
            all_json_files = glob.glob(os.path.join(json_input, "*.json"))
            json_files = [f for f in all_json_files if 
                        f"{file_pattern}_" in os.path.basename(f) or 
                        f"{file_pattern}." in os.path.basename(f)]
        else:
            json_files = glob.glob(os.path.join(json_input, "*.json"))

        self.logger.info(f"Found {len(json_files)} JSON files matching pattern: {file_pattern}")
        
        # Load slow waves from JSON files
        all_slow_waves = []
        empty_channels = [] 
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    slow_waves = json.load(f)
                    
                if isinstance(slow_waves, list):
                    if len(slow_waves) > 0:
                        all_slow_waves.extend(slow_waves)
                    else:
                        # Extract channel name from filename
                        filename = os.path.basename(file)
                        parts = filename.split('_')
                        if len(parts) > 1:
                            chan = parts[-1].replace('.json', '')
                            empty_channels.append(chan)
                        self.logger.info(f"File {file} contains an empty list (no slow waves)")

                else:
                    self.logger.warning(f"Warning: Unexpected format in {file}")
                    
                self.logger.info(f"Loaded {len(slow_waves) if isinstance(slow_waves, list) else 0} slow waves from {file}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        if not all_slow_waves:
            self.logger.info("No slow waves found in the input files")
             # Create an empty CSV file with header to indicate processing was done
            if empty_channels and not skip_empty_files:
                try:
                    with open(csv_file, 'w', newline='') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(["No slow waves were detected in the following channels:"])
                        for chan in empty_channels:
                            writer.writerow([chan])
                    self.logger.info(f"Created empty CSV file at {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error creating empty CSV: {e}")   
        
            return None
        
        # Get frequency band from slow waves if not provided
        if frequency is None:
            try:
                if 'freq_range' in all_slow_waves[0]:
                    freq_range = all_slow_waves[0]['freq_range']
                    if isinstance(freq_range, list) and len(freq_range) == 2:
                        frequency = tuple(freq_range)
                    elif isinstance(freq_range, str) and '-' in freq_range:
                        freq_parts = freq_range.split('-')
                        frequency = (float(freq_parts[0].replace('Hz', '').strip()), 
                                   float(freq_parts[1].replace('Hz', '').strip()))
                        self.logger.info(f"Using frequency range from JSON: {frequency}")
            except:
                frequency = (0.1, 4.0)  # Default for slow waves
                self.logger.info(f"Using default frequency range: {frequency}")

        # Get sampling frequency from dataset
        try:
            s_freq = self.dataset.header['s_freq']
        except:
            self.logger.error("Could not determine dataset sampling frequency")
            return None
        
        # Try to get recording start time
        recording_start_time = None
        try:
            if hasattr(self.dataset, 'header'):
                header = self.dataset.header
                if hasattr(header, 'start_time'):
                    recording_start_time = header.start_time
                elif isinstance(header, dict) and 'start_time' in header:
                    recording_start_time = header['start_time']
                    
            if recording_start_time:
                self.logger.info(f"Found recording start time: {recording_start_time}")
            else:
                self.logger.warning("Could not find recording start time in dataset header. Using relative time only.")
        except Exception as e:
            self.logger.error(f"Error getting recording start time: {e}")
            self.logger.warning("Using relative time only.")

        # Group slow waves by channel for more efficient processing
        waves_by_chan = {}
        for sw in all_slow_waves:
            chan = sw.get('chan')
            if chan not in waves_by_chan:
                waves_by_chan[chan] = []
            waves_by_chan[chan].append(sw)

        self.logger.info(f"Grouped slow waves by {len(waves_by_chan)} channels")

        # Process each channel
        all_segments = []

        # Load data for each channel and create segments
        for chan, waves in waves_by_chan.items():
            self.logger.info(f"Processing {len(waves)} slow waves for channel {chan}")

            try:
                # Create time windows for slow waves
                wave_windows = []
                for sw in waves:
                    start_time = sw['start_time']
                    end_time = sw['end_time']
                    wave_windows.append((start_time, end_time))

                # Create segments
                for i, (start_time, end_time) in enumerate(wave_windows):
                    try:
                        # Add buffer for FFT calculation
                        buffer = 0.1  # 100ms buffer
                        start_with_buffer = max(0, start_time - buffer)
                        end_with_buffer = end_time + buffer
                        
                        # Read data
                        data = self.dataset.read_data(chan=[chan], 
                                                    begtime=start_with_buffer, 
                                                    endtime=end_with_buffer)
                        
                        # Create segment
                        seg = {
                            'data': data,
                            'name': 'slow_wave',
                            'start': start_time,
                            'end': end_time,
                            'n_stitch': 0,
                            'stage': waves[i].get('stage'),
                            'cycle': None,
                            'chan': chan,
                            'uuid': waves[i].get('uuid', str(i))
                        }
                        all_segments.append(seg)

                    except Exception as e:
                        self.logger.error(f"Error creating segment for slow wave {start_time}-{end_time}: {e}")

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
        
        # Create temporary file
        temp_csv = csv_file + '.temp'

        try:
            # Calculate parameters
            self.logger.info(f"Calculating parameters with frequency band {frequency} and n_fft={n_fft}")
            params = event_params(all_segments, export_params, band=frequency, n_fft=n_fft)
            
            if not params:
                self.logger.info("No parameters calculated")
                return None
            
            # Export to temporary CSV
            self.logger.info("Exporting parameters to temporary file")            
            export_event_params(temp_csv, params, count=None, density=None)

            # Store UUIDs
            uuid_dict = {}
            for i, segment in enumerate(all_segments):
                if 'uuid' in segment:
                    uuid_dict[i] = segment['uuid']

            # Process CSV
            self.logger.info("Processing CSV to remove summary rows and add HH:MM:SS format")
            with open(temp_csv, 'r', newline='') as infile, open(csv_file, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                # Read all rows
                all_rows = list(reader)

                # Find header row
                header_row_index = None
                start_time_index = None
                for i, row in enumerate(all_rows):
                    if row and 'Start time' in row:
                        header_row_index = i
                        start_time_index = row.index('Start time')
                        break
                
                if header_row_index is None or start_time_index is None:
                    self.logger.error("Could not find 'Start time' column in CSV")
                    with open(temp_csv, 'r') as src, open(csv_file, 'w') as dst:
                        dst.write(src.read())
                    return params
            
                # Create filtered rows
                filtered_rows = []
            
                # Add prefix rows
                for i in range(header_row_index):
                    filtered_rows.append(all_rows[i])

                # Add header row with additional columns
                header_row = all_rows[header_row_index].copy()
                header_row.insert(start_time_index + 1, 'Start time (HH:MM:SS)')
                if 'UUID' not in header_row:
                    header_row.append('UUID')
                filtered_rows.append(header_row)

                # Add data rows
                for i in range(header_row_index + 5, len(all_rows)):
                    row = all_rows[i]
                    if not row:
                        continue
                        
                    new_row = row.copy()
                    
                    # Add HH:MM:SS time format
                    if len(row) > start_time_index:
                        try:
                            start_time_sec = float(row[start_time_index])
                            
                            def sec_to_time(seconds):
                                hours = int(seconds // 3600)
                                minutes = int((seconds % 3600) // 60)
                                sec = seconds % 60
                                return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"
                                
                            # Calculate clock time
                            if recording_start_time is not None:
                                try:
                                    delta = datetime.timedelta(seconds=start_time_sec)
                                    event_time = recording_start_time + delta
                                    start_time_hms = event_time.strftime('%H:%M:%S.%f')[:-3]
                                except:
                                    start_time_hms = sec_to_time(start_time_sec)
                            else:
                                start_time_hms = sec_to_time(start_time_sec)
                            
                            new_row.insert(start_time_index + 1, start_time_hms)
                        except (ValueError, IndexError):
                            new_row.insert(start_time_index + 1, '')
                    else:
                        new_row.insert(start_time_index + 1, '')
                    
                    # Add UUID
                    segment_index = i - (header_row_index + 5)
                    if segment_index in uuid_dict:
                        new_row.append(uuid_dict[segment_index])
                    else:
                        new_row.append('')
                    
                    filtered_rows.append(new_row)
                
                # Write filtered rows
                for row in filtered_rows:
                    writer.writerow(row)

            # Remove temporary file
            try:
                os.remove(temp_csv)
            except:
                self.logger.info(f"Could not remove temporary file {temp_csv}")

            self.logger.info(f"Successfully exported to {csv_file} with HH:MM:SS time format")
            return params
        except Exception as e:
            self.logger.error(f"Error calculating parameters: {e}")
            import traceback
            traceback.print_exc()
            return None

    def export_slow_wave_density_to_csv(self, json_input, csv_file, stage=None, file_pattern=None):
        """
        Export slow wave statistics to CSV with both whole night and stage-specific densities.
        
        Parameters
        ----------
        json_input : str or list
            Path to JSON file, directory of JSON files, or list of JSON files
        csv_file : str
            Path to output CSV file
        stage : str or list
            Sleep stage(s) to include
        file_pattern : str or None
            Pattern to filter JSON files
        """
        import glob
        from collections import defaultdict
        
        # Load slow waves from JSON file(s)
        json_files = []
        if file_pattern:
            all_json_files = glob.glob(os.path.join(json_input, "*.json"))
            json_files = [f for f in all_json_files if 
                        f"{file_pattern}_" in os.path.basename(f) or 
                        f"{file_pattern}." in os.path.basename(f)]
        else:
            json_files = glob.glob(os.path.join(json_input, "*.json"))

        self.logger.info(f"Found {len(json_files)} JSON files matching pattern: {file_pattern}")

        if not json_files:
            try:
                with open(csv_file, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["No JSON files found matching pattern:", file_pattern])
                self.logger.info(f"Created empty CSV file at {csv_file}")
            except Exception as e:
                self.logger.error(f"Error creating empty CSV: {e}")
                
            return None    
        # Prepare stages
        if stage is None:
            combined_stages = False
            stage_list = None
        elif isinstance(stage, list) and len(stage) > 1:
            combined_stages = True
            stage_list = stage
            combined_stage_name = "+".join(stage_list)
            self.logger.info(f"Calculating combined slow wave density for stages: {combined_stage_name}")
        elif isinstance(stage, list) and len(stage) == 1:
            combined_stages = False
            stage_list = [stage[0]]
            self.logger.info(f"Calculating slow wave density for stage: {stage_list[0]}")
        else:
            combined_stages = False
            stage_list = [stage]
            self.logger.info(f"Calculating slow wave density for stage: {stage}")

        # Load all slow waves
        all_slow_waves = []
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    waves = json.load(f)
                    all_slow_waves.extend(waves if isinstance(waves, list) else [])
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        # Get stage durations
        epoch_duration_sec = 30
        stage_counts = defaultdict(int)
        all_stages = self.annotations.get_stages()
                                
        # Count epochs
        for s in all_stages:
            if s in ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']:
                stage_counts[s] += 1

        # Calculate durations
        stage_durations = {stg: count * epoch_duration_sec / 60 for stg, count in stage_counts.items()}
        total_duration_min = sum(stage_durations.values())
    
        # Extract stages from slow waves if needed
        wave_stages = set()
        for sw in all_slow_waves:
            if not isinstance(sw, dict) or 'stage' not in sw:
                continue        
            sw_stage = sw['stage']
            if isinstance(sw_stage, list):
                for s in sw_stage:
                    wave_stages.add(str(s))
            else:
                wave_stages.add(str(sw_stage))
        
        # Determine stages to process
        if stage is None:
            stages_to_process = sorted(wave_stages)
            combined_stages = False
        elif combined_stages:
            stages_to_process = [stage_list]
        else:
            stages_to_process = stage_list

        # Group slow waves by channel and stage
        waves_by_chan_stage = defaultdict(lambda: defaultdict(list))
        waves_by_chan = defaultdict(list)
        
        for sw in all_slow_waves:
            if not isinstance(sw, dict):
                continue
            
            chan = sw.get('chan', sw.get('channel'))
            if not chan:
                continue
        
            waves_by_chan[chan].append(sw)
            
            if not combined_stages:
                if 'stage' in sw:
                    sw_stages = sw['stage'] if isinstance(sw['stage'], list) else [sw['stage']]
            
                    for sw_stage in sw_stages:
                        sw_stage = str(sw_stage)
                        waves_by_chan_stage[chan][sw_stage].append(sw)

        # Calculate statistics
        stage_channel_stats = defaultdict(dict)
        for chan in set(waves_by_chan.keys()):
            all_chan_waves = waves_by_chan[chan]
        
            for process_stage in stages_to_process:
                stage_waves = []
                if combined_stages or (isinstance(process_stage, list) and len(process_stage) > 1):
                    stages_to_include = process_stage if isinstance(process_stage, list) else stage_list
                    stage_name_display = "+".join(stages_to_include)
                    stages_set = set(str(s) for s in stages_to_include)
                    stage_waves = []
                    seen_waves = set()

                    for sw in all_chan_waves:
                        if 'stage' not in sw:
                            continue
                        sw_stages = sw['stage'] if isinstance(sw['stage'], list) else [sw['stage']]
                        sw_stages = set(str(s) for s in sw_stages)

                        if sw_stages.intersection(stages_set) and id(sw) not in seen_waves:
                            stage_waves.append(sw)
                            seen_waves.add(id(sw))

                    stage_duration_min = sum(stage_durations.get(s, 0) for s in stages_to_include)
        
                else:
                    s_str = str(process_stage)
                    stage_waves = waves_by_chan_stage[chan].get(s_str, [])
                    stage_name_display = process_stage
                    stage_duration_min = stage_durations.get(s_str, 0)
            
                if len(stage_waves) == 0:
                    continue
            
                # Calculate statistics
                stage_count = len(stage_waves)
                whole_night_count = len(all_chan_waves)
                
                stage_density = stage_count / stage_duration_min if stage_duration_min > 0 else 0
                whole_night_density = whole_night_count / total_duration_min if total_duration_min > 0 else 0
                
                # Calculate mean duration
                durations = []
                for sw in stage_waves:
                    if 'start_time' in sw and 'end_time' in sw:
                        durations.append(sw['end_time'] - sw['start_time'])
                
                mean_duration = np.mean(durations) if durations else 0
                
                # Store statistics
                key = tuple(process_stage) if isinstance(process_stage, list) else process_stage
                stage_channel_stats[key][chan] = {
                    'count': stage_count,
                    'stage_density': stage_density,
                    'whole_night_density': whole_night_density,
                    'mean_duration': mean_duration,
                    'stage_name_display': stage_name_display,
                    'stage_duration_min': stage_duration_min,
                }
        
        # Export to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Add summary sections
            writer.writerow(['Whole Night Summary'])
            writer.writerow(['Total Recording Duration (min)', f'{total_duration_min:.2f}'])
            writer.writerow([])
            
            writer.writerow(['Stage Duration Summary'])
            writer.writerow(['Stage', 'Duration (min)'])
            for stg in sorted(set(stage_durations.keys())):
                writer.writerow([stg, f"{stage_durations.get(stg, 0):.2f}"])
            if combined_stages:
                combined_duration = sum(stage_durations.get(s, 0) for s in stage_list)
                writer.writerow([combined_stage_name, f"{combined_duration:.2f}"])

            writer.writerow([])
            
            # Process each stage
            for process_stage in stages_to_process:
                key = tuple(process_stage) if isinstance(process_stage, list) else process_stage
                if key not in stage_channel_stats:
                    continue
                    
                any_chan = next(iter(stage_channel_stats[key].keys()))
                stage_name_display = stage_channel_stats[key][any_chan]['stage_name_display']

                writer.writerow([f"Sleep Stage: {stage_name_display}"])
                writer.writerow([
                    'Channel', 
                    'Count',
                    f'Density in {stage_name_display} (events/min)', 
                    'Whole Night Density (events/min)',
                    'Mean Duration (s)'
                ])

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
        
        self.logger.info(f"Exported slow wave statistics to {csv_file}")
        return dict(stage_channel_stats)