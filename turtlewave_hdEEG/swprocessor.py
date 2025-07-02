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

    def clean_memory(self):
        """
        Perform thorough memory cleanup to release resources
        """
        import gc
        import sys
        
        # Clear any large variables in the class
        if hasattr(self, '_temp_data'):
            del self._temp_data
        
        # Force garbage collection
        gc.collect()
        
        # For more aggressive cleanup on systems that support it
        if sys.platform == 'linux':
            try:
                import resource
                import psutil
                # Suggest to OS to release memory
                psutil.Process().memory_info()
                resource.RUSAGE_SELF
            except ImportError:
                self.logger.info("psutil not available for advanced memory cleanup")
        
        self.logger.info("Memory cleanup performed")



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
               ___    __,__,__,__, 
              /_@ \  /  /  \  \  \
               \__\/-<_>-<_>-<->-|-<
                    /\____________/~
                   / /===/ /=====\ \
                   ""    ""       "" ''''  
                searching for slow waves...
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                                # Use appropriate duration parameter based on method
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
        
        # Clean memory first
        self.clean_memory()

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
    

    def save_detection_summary(self, output_dir, method, parameters, results_summary):
        """
        Save a comprehensive summary of detection parameters and results.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the summary
        method : str
            Detection method used
        parameters : dict
            All parameters used for detection
        results_summary : dict
            Summary of detection results
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = os.path.join(output_dir, f"detection_summary_{method}_{timestamp}.json")
            
            summary_data = {
                'detection_method': method,
                'parameters': parameters,
                'results': results_summary,
                'timestamp': datetime.datetime.now().isoformat(),
                'software_version': 'TurtleWave hdEEG GUI'
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            self.logger.info(f"Saved detection summary to: {summary_file}")
            return summary_file
        except Exception as e:
            self.logger.error(f"Error saving detection summary: {e}")
            return None

    ############# SQLite Database Initialization and Import Functions #############

    def initialize_sqlite_database(self, db_path='neural_events.db'):
            """
            Create SQLite database optimized for storing calculated event parameters 
            from event_params() function.
            
            Parameters
            ----------
            db_path : str
                Path to SQLite database file
                
            Returns
            -------
            str
                Path to created database
            """
            import sqlite3
            import os

            # If db_path is a directory, append the default filename
            if os.path.isdir(db_path):
                db_path = os.path.join(db_path, 'neural_events.db')
                self.logger.info(f"Database path was a directory, using: {db_path}")
            
            # Create directory for database if it doesn't exist
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"Created directory for database: {db_dir}")
            
            # Check if database exists
            db_exists = os.path.exists(db_path)
            
            # Define the database initialization operation
            def init_db(conn):
                cursor = conn.cursor()
                # Main events table with common fields across all event types
                conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    uuid TEXT PRIMARY KEY,
                    event_type TEXT,           -- 'spindle', 'slow_wave', 'ripple', etc.
                    channel TEXT,
                    
                    -- Basic temporal properties
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    start_time_hms TEXT,       -- formatted time (HH:MM:SS)
                    stage TEXT,
                    cycle TEXT,                -- sleep cycle
                    method TEXT,


                    -- Frequency band information
                    freq_band TEXT,            -- Full text representation (e.g. "0.5-3Hz")
                    freq_lower REAL,           -- Lower bound of frequency band (e.g. 0.5)
                    freq_upper REAL,           -- Upper bound of frequency band (e.g. 3.0)
                                            

                    -- Amplitude metrics
                    min_amp REAL,          -- minimum amplitude
                    max_amp REAL,          -- maximum amplitude

                    peak2peak_amp REAL,    -- peak-to-peak amplitude

                    -- Processing metadata         
                    processing_timestamp TEXT,
                    n_fft_sec INTEGER,
                    
                    CONSTRAINT event_chan_time UNIQUE (event_type, channel, start_time, method, freq_lower, freq_upper, stage)
                )''')

                # Create tracking table for batch processing
                conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_status (
                    channel TEXT,
                    event_type TEXT,
                    json_file TEXT,
                    processed BOOLEAN DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    last_attempt_time TEXT,
                    success BOOLEAN DEFAULT 0,
                    error_message TEXT,
                            
                    PRIMARY KEY (channel, event_type)
                )''')

                # Create indexes for efficient querying
                conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_channel ON events(channel)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timerange ON events(start_time, end_time)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_stage ON events(stage)')
                
                
                conn.commit()


                # If database didn't exist, log creation
                if not db_exists:
                    self.logger.info(f"Created new database at: {db_path}")
                    
                return db_path
        
            # Use the safe database operation
            return self._safe_database_operation(db_path, init_db)



    def _safe_database_operation(self, db_path, operation_func):
        """Safely perform a database operation with proper connection handling"""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            result = operation_func(conn)
            return result
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()


    def import_parameters_csv_to_database(self, csv_file, db_path,  append=True):
        """
        Import event parameters from an existing CSV file into SQLite database.
        Supports multiple event types and incremental updates.
        
        Parameters
        ----------
        csv_file : str
            Path to existing parameters CSV file
        db_path : str
            Path to SQLite database
        append : bool
            If True, adds to existing database without replacing existing entries
            If False, replaces any existing entries with the same UUID
                
        Returns
        -------
        dict
            Summary of the operation with counts of added, updated, and skipped rows
        """
        import sqlite3
        import pandas as pd
        import os
        import glob
        

        self.clean_memory()

        # Initialize database if needed
        if not os.path.exists(db_path):
            self.initialize_sqlite_database(db_path)
        
        # Check if the file exists
        if not os.path.exists(csv_file):
            self.logger.error(f"CSV file not found: {csv_file}")
            return {"error": "CSV file not found", "added": 0, "updated": 0, "skipped": 0}
        
        # Track statistics
        stats = {
            "added": 0,
            "updated": 0,
            "skipped": 0
        }
        
        # Read the CSV file
        self.logger.info(f"Reading parameters from CSV: {csv_file}")
        try:
            # First determine how many rows to skip (header plus statistics)
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                
            # Find the header row (contains 'Start time')
            header_row = None
            for i, line in enumerate(lines):
                if 'Start time' in line:
                    header_row = i
                    break
            
            if header_row is None:
                self.logger.error("Could not find header row in CSV")
                return {"error": "Could not find header row", "added": 0, "updated": 0, "skipped": 0}
            
            # Check if there are statistic rows after the header
            has_stat_rows = False
            if header_row + 1 < len(lines):
                next_line = lines[header_row + 1]
                # Check if the next line starts with "Mean" or contains statistical summaries
                if next_line.strip().startswith('Mean') or 'Mean' in next_line:
                    has_stat_rows = True

            # Skip header row and 4 statistic rows
            skiprows = header_row + 4 if has_stat_rows else header_row
            
            # Read the CSV, skipping header and statistics
            df = pd.read_csv(csv_file, skiprows=skiprows)
            
            if df.empty:
                self.logger.warning("CSV file contains no data rows")
                return {"error": "Empty CSV file", "added": 0, "updated": 0, "skipped": 0}
                
            self.logger.info(f"Read {len(df)} parameter rows from CSV")
            

            # Define database operation function
            def process_csv_data(conn):
                cursor = conn.cursor()
            
                # Determine event type from CSV filename or content
                event_type = "slow_wave"  # Default
                filename = os.path.basename(csv_file).lower()
                if 'slow_wave' in filename or 'slowwave' in filename or 'sw' in filename:
                    event_type = "slow_wave"
                elif 'spindle' in filename:
                    event_type = "spindle"

                # Override event_type if 'Event type' column exists in CSV
                if 'Event type' in df.columns:
                    # Use the first non-null value in the Event type column
                    event_types = df['Event type'].dropna()
                    if len(event_types) > 0:
                        event_type = event_types.iloc[0]

                self.logger.info(f"Importing parameters for event type: {event_type}")

                # Map column names from CSV to database columns
                column_mapping = {
                    'Start time': 'start_time',
                    'Start time (HH:MM:SS)': 'start_time_hms',
                    'End time': 'end_time',
                    'Stage': 'stage',
                    'Cycle': 'cycle',
                    'Event type': 'event_type',
                    'Channel': 'channel',                
                    'Duration (s)': 'duration',                
                    'Min. amplitude (uV)':'min_amp',
                    'Max. amplitude (uV)': 'max_amp',
                    'Peak-to-peak amplitude (uV)': 'peak2peak_amp',
                    #'RMS (uV)': 'rms',
                    #'Power (uV^2)': 'power',
                    #'Peak power frequency (Hz)': 'peak_power_freq',
                    #'Energy (uV^2s)': 'energy',
                    #'Peak energy frequency': 'peak_energy_freq',
                    'UUID': 'uuid'
                }
                
                # Create a list of columns that exist in the dataframe
                existing_columns = []
                db_columns = []
                
                for csv_col, db_col in column_mapping.items():
                    if csv_col in df.columns:
                        existing_columns.append(csv_col)
                        db_columns.append(db_col)
                
                # Add processing timestamp
                import datetime
                now = datetime.datetime.now().isoformat()
                df['processing_timestamp'] = now
                existing_columns.append('processing_timestamp')
                db_columns.append('processing_timestamp')
                
                # Extract frequency band from filename if possible
                filename = os.path.basename(csv_file)
                freq_band = "unknown"
                
                # Try to extract frequency from filename (e.g., sw_parameters_Staresina2015_0.3-2.0Hz_NREM2NREM3.csv)
                if "_" in filename and "Hz" in filename:
                    parts = filename.split('_')
                    for part in parts:
                        if "Hz" in part:
                            freq_band = part
                            try:
                                # Handle formats like "9-12Hz" or "9.0-12.0Hz"
                                freq_parts = freq_band.replace("Hz", "").split("-")
                                if len(freq_parts) == 2:
                                    freq_lower = float(freq_parts[0])
                                    freq_upper = float(freq_parts[1])

                            except ValueError:
                                self.logger.warning(f"Could not parse frequency bounds from {freq_band}")

                            break
                
                df['freq_band'] = freq_band
                df['freq_lower'] = freq_lower
                df['freq_upper'] = freq_upper
                
                existing_columns.append('freq_band')
                existing_columns.append('freq_lower')
                existing_columns.append('freq_upper')
                
                db_columns.append('freq_band')
                db_columns.append('freq_lower')
                db_columns.append('freq_upper')
                
                # Extract method from filename if possible
                method = "unknown"
                if "_" in filename:
                    parts = filename.split('_')
                    if len(parts) > 2:
                        # Typically the format is sw_parameters_METHOD_freq_stages.csv
                        method = parts[2]
                
                df['method'] = method
                existing_columns.append('method')
                db_columns.append('method')
                
                # Set event_type from our detection
                df['event_type'] = event_type
                if 'event_type' not in db_columns:
                    existing_columns.append('event_type')
                    db_columns.append('event_type')

                # Check for UUID column, which is essential for avoiding duplicates
                uuid_col = 'UUID' if 'UUID' in df.columns else 'uuid' if 'uuid' in df.columns else None
                
                # If no UUID column, create one
                if uuid_col is None:
                    self.logger.warning("No UUID column found, creating UUIDs based on channel and time")
                    import uuid
                    df['uuid'] = [
                        str(uuid.uuid4()) for _ in range(len(df))
                    ]
                    uuid_col = 'uuid'
                    existing_columns.append('uuid')
                    db_columns.append('uuid')
                

                # Check if the required columns for uniqueness constraint exist
                if 'Channel' not in df.columns or 'Start time' not in df.columns:
                    self.logger.warning("Missing required columns for uniqueness check")

                # Pre-check existing events by unique constraint (event_type, channel, start_time, method)
                # rather than just UUID to avoid constraint violations
                existing_events = set()
                if append and 'Channel' in df.columns and 'Start time' in df.columns:
                    # Get all unique combinations of event_type, channel, start_time
                    channels = df['Channel'].astype(str).tolist()
                    start_times = df['Start time'].astype(float).tolist()
                    
                    # Build a query to get existing events matching these combinations
                    query_parts = []
                    query_params = []
                    
                    for i in range(len(channels)):
                        freq_lower = df['freq_lower'].iloc[i] if 'freq_lower' in df.columns else None
                        freq_upper = df['freq_upper'].iloc[i] if 'freq_upper' in df.columns else None
                        stage = df['Stage'].iloc[i] if 'Stage' in df.columns else None

                        query_parts.append("(event_type = ? AND channel = ? AND start_time = ? AND method = ? AND freq_lower = ? AND freq_upper = ? AND stage = ?)")
                        query_params.extend([event_type, channels[i], start_times[i],method, freq_lower, freq_upper, stage])
                    
                    if query_parts:
                        query = f"SELECT event_type, channel, start_time, method, freq_lower, freq_upper, stage FROM events WHERE {' OR '.join(query_parts)}"
                        cursor.execute(query, query_params)
                        
                        for row in cursor.fetchall():
                            # Create a tuple of (event_type, channel, start_time. method) to check against
                            existing_events.add((row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
                        
                        self.logger.info(f"Found {len(existing_events)} existing entries matching event type, channel, and start time")
                
                # Mark rows that exist in the database based on the uniqueness constraint
                df['exists_in_db'] = df.apply(
                    lambda row: (
                        event_type, 
                        str(row.get('Channel', '')), 
                        float(row.get('Start time', 0)), 
                        method,
                        row.get('freq_lower', None),
                        row.get('freq_upper', None),
                        str(row.get('Stage',''))
                        ) in existing_events, 
                    axis=1
                )



                # # If appending, we need to check which rows already exist in the database
                # if append and uuid_col:
                #     # Get all UUIDs from the dataframe
                #     all_uuids = df[uuid_col].astype(str).tolist()
                    
                #     # Check which UUIDs already exist in the database
                #     placeholders = ','.join(['?' for _ in all_uuids])
                #     cursor.execute(f"SELECT uuid FROM events WHERE uuid IN ({placeholders})", all_uuids)
                #     existing_uuids = {row[0] for row in cursor.fetchall()}
                    
                #     self.logger.info(f"Found {len(existing_uuids)} existing entries in database")
                    
                #     # Mark rows that already exist in the database
                #     df['exists_in_db'] = df[uuid_col].apply(lambda x: str(x) in existing_uuids)
                # else:
                #     # If not appending, mark all rows as not existing
                #     df['exists_in_db'] = False
                
                # Process each row based on whether it exists and append mode
                for _, row in df.iterrows():
                    if isinstance(row['Stage'], list):
                        row['Stage'] = '+'.join(row['Stage'])
                    elif isinstance(row['Stage'], str) and '[' in row['Stage']:
                        # Sometimes stage might be a string representation of a list like "['NREM2', 'NREM3']"
                        # Try to convert it to a proper list then join
                        try:
                            import ast
                            stage_list = ast.literal_eval(row['Stage'])
                            if isinstance(stage_list, list):
                                row['Stage'] = ''.join(stage_list)
                        except:
                            # If conversion fails, keep as is
                            pass
                    # Skip existing rows when in append mode
                    if append and row['exists_in_db']:
                        stats["skipped"] += 1
                        continue
                    values = [row[col] if col in row else None for col in existing_columns]
                    
                    # Handle NaN values
                    for i, val in enumerate(values):
                        # Check if value is NaN (using pandas or numpy's isnan)
                        if pd.isna(val) or (hasattr(val, 'isnan') and val.isnan()):
                            values[i] = None  # Convert NaN to None (which becomes NULL in SQLite)

                    try:
                        if append and row['exists_in_db']:
                            # Skip existing rows when in append mode
                            stats["skipped"] += 1
                            continue
                        if not append and row['exists_in_db']:
                            # Update existing row when not in append mode
                            update_columns = [col for col in db_columns if col != 'uuid']
                            update_values = [val for i, val in enumerate(values) if db_columns[i] != 'uuid']
                            
                            # Update based on the unique constraint, not just UUID
                            cursor.execute(f"""
                            UPDATE events
                            SET {', '.join([f'{col} = ?' for col in update_columns])}
                            WHERE event_type = ? AND channel = ? AND start_time = ? AND method = ?
                                AND freq_lower = ? AND freq_upper = ? AND stage = ?
                            """, update_values + [
                                event_type, 
                                row.get('Channel', ''), 
                                row.get('Start time', 0), 
                                method,
                                row.get('freq_lower', None),
                                row.get('freq_upper', None),
                                str(row.get('Stage', ''))
                                    ])
                            
                            stats["updated"] += 1
                        else:
                            # Insert new row - use REPLACE to handle any constraint violations
                            cursor.execute(f"""
                            INSERT OR REPLACE INTO events
                            ({', '.join(db_columns)})
                            VALUES ({', '.join(['?' for _ in db_columns])})
                            """, values)
                            
                            stats["added"] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing row: {e}")
                        stats["skipped"] += 1
                
                conn.commit()
                self.logger.info(f"Database updated: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} skipped")
                
                # Update processing status
                #cursor.execute("PRAGMA table_info(processing_status)")
                #columns = cursor.fetchall()
                #print("Columns in processing_status table:", columns)

                # Update processing status with handling for both channels with events and empty channels
                if 'Channel' in df.columns:
                    processed_channels = set(df['Channel'].unique())
                    
                    # Add channels that have events in the CSV
                    for channel in processed_channels:
                        cursor.execute('''
                        INSERT OR REPLACE INTO processing_status
                        (channel, event_type, processed, success, attempts, last_attempt_time)
                        VALUES (?, ?, 1, 1, 1, datetime('now'))
                        ''', (channel,event_type))
                    
                    # Try to identify empty channels from JSON filenames
                    # Note: This assumes the CSV file name contains information to identify related JSON files
                    csv_basename = os.path.basename(csv_file)
                    parts = csv_basename.split('_')
                    if len(parts) >= 3:
                        # For CSVs like: spindle_parameters_Ferrarelli2007_9-12Hz_NREM2NREM3.csv
                        # Matching JSONs like: spindles_Ferrarelli2007_9-12Hz_NREM2NREM3_E101.json
                        
                        # Extract the method and frequency-stage parts
                        method = parts[2]  # Ferrarelli2007
                        freq_stage = parts[3:]  # ['9-12Hz', 'NREM2NREM3']
                        freq_stage_str = '_'.join(freq_stage).replace('.csv', '')
                        
                        # Construct pattern to find related JSON files
                        json_pattern = f"{event_type}s_{method}_{freq_stage_str}_*"
                        
                        # Find JSON files matching the pattern
                        json_dir = os.path.dirname(csv_file)
                        all_json_files = glob.glob(os.path.join(json_dir, f"{json_pattern}.json"))
                        
                        self.logger.info(f"Looking for JSON files matching pattern: {json_pattern}.json")
                        self.logger.info(f"Found {len(all_json_files)} matching JSON files")

                        # Extract channel names from JSON files
                        empty_channels = set()
                        for file in all_json_files:
                            try:
                                # Extract channel name from filename
                                # Assuming format like "spindles_method_freq_stage_CHANNELNAME.json"
                                channel_name = os.path.basename(file).split('_')[-1].replace('.json', '')
                                # Skip if channel already in processed_channels
                                if channel_name in processed_channels:
                                    continue
                                
                                # Read JSON file to check if it's empty
                                with open(file, 'r') as f:
                                    content = json.load(f)
                                    
                                # If JSON file contains an empty array, add to empty_channels
                                if isinstance(content, list) and len(content) == 0:
                                    empty_channels.add(channel_name)
                                    self.logger.info(f"Found empty JSON file for channel: {channel_name}")
                            except Exception as e:
                                self.logger.warning(f"Error checking JSON file {file}: {e}")
    
                        
                        # Add empty channels to processing_status
                        for channel in empty_channels:
                            cursor.execute('''
                            INSERT OR REPLACE INTO processing_status
                            (channel, event_type, processed, success, attempts, last_attempt_time, error_message)
                            VALUES (?, ?, 1, 1, 1, datetime('now'), 'No events detected')
                            ''', (channel,event_type))
            
                        if empty_channels:
                            self.logger.info(f"Recorded {len(empty_channels)} channels with no events: {', '.join(empty_channels)}")
                        # Add empty channels count to stats
                        stats["empty_channels"] = len(empty_channels)
                    
                    conn.commit()

                # Get total count
                cursor.execute("SELECT COUNT(*) FROM events")
                total_count = cursor.fetchone()[0]
                self.logger.info(f"Total parameters in database: {total_count}")

                conn.close()

                return stats
            # Use the safe database operation
            return self._safe_database_operation(db_path, process_csv_data)
        
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "added": 0, "updated": 0, "skipped": 0}


