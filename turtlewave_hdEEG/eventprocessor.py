
import numpy as np
import time
import os
import multiprocessing
import csv
from wonambi.trans import select, fetch, math
from wonambi.attr import Annotations
from turtlewave_hdEEG.extensions import ImprovedDetectSpindle as DetectSpindle
from turtlewave_hdEEG import dbwrite
import json
import datetime
import logging
import uuid as _uuid_mod


class ParalEvents:
    """
    A class for parallel detection and analysis of EEG events such as spindles,
    and other neural events across multiple channels.
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

        # This logger name is a process-wide singleton. Clear any handlers left
        # by a previous instance so batch loops don't duplicate log lines or
        # leak file handles.
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

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


   # solve the issue of memory leaks by cleaning up large variables and forcing garbage collection
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


    def detect_spindles(self, method='Ferrarelli2007', chan=None, ref_chan=[], grp_name='eeg',
                       frequency=(11, 16), duration=(0.5, 3), polar='normal',
                       reject_artifacts=True, reject_arousals=True,stage=None, cat=None,
                       save_to_annotations=False, json_dir=None, create_empty_json=True,
                       *, write_db=False, db_path=None, resume=False, run_params=None,
                       replace_channels=None,
                       **detector_params):
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
        write_db : bool, keyword-only, default False
            When True, write detected events straight into a SQLite database
            (``db_path``) in addition to the JSON output, using the direct-write
            path (deterministic uuid5 rows, batched morphology, per-scope
            ``processing_status`` tracking and a ``detection_runs`` provenance
            row). When False the behaviour is byte-identical to the legacy
            JSON-only path.
        db_path : str or None, keyword-only
            Target SQLite database (or directory, in which case
            ``neural_events.db`` is used). Required when ``write_db`` is True.
        resume : bool, keyword-only, default False
            When True (and ``write_db``), channels already recorded as
            ``success = 1`` for the *same* scope (method, band, stage set) are
            skipped instead of re-detected.
        run_params : dict or None, keyword-only
            Extra parameters merged into the ``detection_runs.params_json``
            provenance record.
        replace_channels : optional, keyword-only
            Reserved for P3 (scoped channel re-detection); accepted but not yet
            wired.
        **detector_params : dict
        Additional parameters to pass to the detector. These are method-specific
        and can include parameters like det_thresh, sel_thresh, etc.
        Returns
        -------
        list
            List of all detected spindles

        Notes
        -----
        In the direct-write path each spindle's ``stage`` is resolved to the
        single scored epoch it falls in (via ``_stage_at``), matching the
        slow-wave/K-complex convention. This differs from the legacy spindle CSV
        path, which tags each event with the whole requested stage list.
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

        if detector_params:
            self.logger.info(f"Method-specific parameters: {detector_params}")

        # ------------------------------------------------------------------
        # Direct-to-DB write path setup (opt-in; JSON behaviour unchanged).
        # ------------------------------------------------------------------
        stages_key = "".join(stage) if stage else "all"
        db_conn = None
        run_id = None
        db_skip = set()
        db_n_fft_sec = 4  # matches the CSV exporter's default FFT window
        rec_start = None
        s_freq = None
        if write_db:
            if db_path is None:
                self.logger.error("write_db=True but db_path is None; skipping DB writes")
                write_db = False
            else:
                try:
                    if os.path.isdir(db_path):
                        db_path = os.path.join(db_path, 'neural_events.db')
                    # Create base tables (events/processing_status/...) if absent.
                    self.initialize_sqlite_database(db_path)
                    db_conn = dbwrite.open_write_connection(db_path)
                    dbwrite.ensure_direct_write_schema(db_conn, self.logger)
                    try:
                        s_freq = self.dataset.header['s_freq']
                    except Exception:
                        s_freq = None
                    try:
                        rec_start = self.dataset.header.get('start_time')
                    except Exception:
                        rec_start = None
                    run_id = str(_uuid_mod.uuid4())
                    params_dict = {
                        'frequency': list(frequency), 'duration': list(duration),
                        'polar': polar, 'method': method_str,
                        'detector_params': detector_params,
                        'reject_artifacts': reject_artifacts,
                        'reject_arousals': reject_arousals,
                        'n_fft_sec': db_n_fft_sec,
                    }
                    if run_params:
                        params_dict.update(run_params)
                    dbwrite.record_run(
                        db_conn, run_id, 'spindle', method_str,
                        dbwrite.method_citation(method_str),
                        json.dumps(params_dict, default=str),
                        ref_chan, polar, stage, reject_artifacts, reject_arousals)
                    if resume:
                        db_skip = dbwrite.resume_skip_channels(
                            db_conn, 'spindle', method_str,
                            frequency[0], frequency[1], stages_key)
                        if db_skip:
                            self.logger.info(
                                f"Resume: skipping {len(db_skip)} already-completed "
                                f"channels for this scope")
                except Exception as e:
                    self.logger.error(f"Could not set up direct-DB write: {e}", exc_info=True)
                    write_db = False
                    if db_conn is not None:
                        try:
                            db_conn.close()
                        except Exception:
                            pass
                        db_conn = None

        # Epoch -> stage lookup so each event is attributed to the single scored
        # epoch it falls in (matches the SW/KC convention).
        try:
            import bisect as _bisect
            _det_epochs = sorted(
                ((float(e['start']), float(e['end']), str(e['stage']))
                 for e in self.annotations.get_epochs()),
                key=lambda x: x[0]
            ) if self.annotations is not None else []
        except Exception as e:
            self.logger.warning(f"Could not build epoch stage lookup: {e}")
            _det_epochs = []
        _det_epoch_starts = [e[0] for e in _det_epochs]

        def _stage_at(t):
            """Return the scored stage of the epoch containing time t, or None."""
            if t is None or not _det_epochs:
                return None
            idx = _bisect.bisect_right(_det_epoch_starts, t) - 1
            if 0 <= idx < len(_det_epochs) and _det_epochs[idx][0] <= t < _det_epochs[idx][1]:
                return _det_epochs[idx][2]
            return None

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
                        with open(annotation_file_path, 'w', encoding='utf-8') as f:
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
                if write_db and resume and ch in db_skip:
                    self.logger.info(f"Resume: channel {ch} already complete for this scope; skipping")
                    continue
                try:
                    self.logger.info(f'Reading data for channel {ch}')

                    # Fetch segments, filtering based on stage and artifacts
                    segments = fetch(self.dataset, self.annotations, cat=cat, stage=stage, cycle=None,
                                    reject_epoch=True, reject_artf=reject_types)
                    segments.read_data(ch, ref_chan, grp_name=grp_name)


                    # Process each detection method
                    channel_spindles = []
                    channel_json_spindles = []
                    # Direct-write accumulators (populated only when write_db).
                    channel_db_events = []
                    channel_param_segments = []
                    ## Loop through methods (i.e. WHALE IT!)
                    for m, meth in enumerate(method):
                        self.logger.info(f"Applying method: {meth}")
                        ### define detection
                        detection = DetectSpindle(meth, frequency=frequency, duration=duration,
                        polar=polar, **detector_params)
                        
                        self.logger.debug(f"Detector parameters for {meth}: frequency={frequency}, duration={duration}")
                        if hasattr(detection, 'det_thresh'):
                            self.logger.debug(f"  det_thresh: {detection.det_thresh}")
                        if hasattr(detection, 'sel_thresh'):
                            self.logger.debug(f"  sel_thresh: {detection.sel_thresh}")


                        for i, seg in enumerate(segments):
                            self.logger.info(f'Detecting events, segment {i + 1} of {len(segments)}')

                            # Polarity is handled inside the detector (polar=...),
                            # which inverts on a copy. Do NOT invert here as well.
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

                                # Assemble the direct-DB event (independent of the
                                # JSON uuid above): deterministic uuid5, single
                                # resolved stage, detector-own morphology, and an
                                # in-memory window for batched re-measurement.
                                if write_db:
                                    sp_start = float(sp.get('start', 0))
                                    sp_end = float(sp.get('end', 0))
                                    sp_dur = float(sp.get('dur', sp_end - sp_start))
                                    single_stage = _stage_at(sp_start)
                                    if single_stage is None and isinstance(stage, (list, tuple)) and len(stage) == 1:
                                        single_stage = stage[0]
                                    morph = dbwrite.event_det_morphology(sp)
                                    ev = {
                                        'uuid': dbwrite.event_uuid5(
                                            'spindle', ch, sp_start, meth,
                                            frequency[0], frequency[1], single_stage),
                                        'start_time': sp_start, 'end_time': sp_end,
                                        'duration': sp_dur, 'stage': single_stage,
                                        'method': meth,
                                    }
                                    ev.update(morph)
                                    channel_db_events.append(ev)
                                    channel_param_segments.append(
                                        dbwrite.make_param_segment(
                                            seg['data'], sp_start, sp_end,
                                            'spindle', single_stage, ch))

                                # Add to JSON
                                if json_dir:
                                    # Extract key properties in a serializable format
                                    sp_data = {
                                        'uuid': sp['uuid'],
                                        'chan': ch,
                                        'start_time': float(sp.get('start', 0)),
                                        'end_time': float(sp.get('end', 0)),
                                    #    'peak_time': float(sp.get('peak_time', 0)),
                                    #    'duration': float(sp.get('dur', 0)),
                                    #    'ptp_det': float(sp.get('ptp_det', 0)),
                                        'method': meth
                                    }
                                    
                                    sp_data['stage'] = stage
                                    sp_data['freq_range'] = frequency
                                    # Add frequency/power/amplitude if available
                                    #if 'peak_freq' in sp:
                                    #    sp_data['peak_freq'] = float(sp['peak_freq'])
                                    #if 'peak_val' in sp:
                                    #    sp_data['peak_val'] = float(sp['peak_val'])
                                    #if 'power' in sp:
                                    #    sp_data['power'] = float(sp['power'])
                                        
                                    channel_json_spindles.append(sp_data)
                    all_spindles.extend(channel_spindles)
                    self.logger.info(f"Found {len(channel_spindles)} spindles in channel {ch}")

                    # Direct-DB write: one batched re-measurement + one
                    # transaction per channel, BEFORE the JSON write.
                    if write_db and db_conn is not None:
                        batched = dbwrite.compute_batched_params(
                            channel_param_segments, frequency, s_freq,
                            db_n_fft_sec, self.logger)
                        dbwrite.write_channel_events(
                            db_conn, run_id, 'spindle', ch, method_str,
                            frequency[0], frequency[1], stages_key,
                            channel_db_events, batched, rec_start,
                            db_n_fft_sec, self.logger)
                        self.logger.info(
                            f"Wrote {len(channel_db_events)} spindle rows for "
                            f"channel {ch} to the database")

                    stages_str = "".join(stage) if stage else "all"
                    if json_dir:
                        try:
                            ch_json_file = os.path.join(json_dir, f"spindles_{method_str}_{freq_str}_{stages_str}_{ch}.json")

                            # Create empty JSON if no spindles found but flag is set
                            if not channel_json_spindles and create_empty_json:
                                self.logger.info(f"Creating empty JSON file for channel {ch} (no spindles detected)")
                                with open(ch_json_file, 'w', encoding='utf-8') as f:
                                    json.dump([], f)
                            elif channel_json_spindles:
                                with open(ch_json_file, 'w', encoding='utf-8') as f:
                                    json.dump(channel_json_spindles, f, indent=2)
                                self.logger.info(f"Saved spindle data for channel {ch} to {ch_json_file}")
                        except Exception as e:
                            self.logger.error(f"Error saving channel JSON: {e}")
                except Exception as e:
                        # A real read/detection failure is logged as an error with
                        # a traceback so it is not mistaken for a channel that
                        # legitimately had no spindles.
                        self.logger.error(f'Failed to process channel {ch}: {e}', exc_info=True)

                        # In the direct-DB path, record the failure in
                        # processing_status (success=0) instead of an error
                        # sentinel JSON, so a resume re-runs only this channel.
                        if write_db and db_conn is not None:
                            dbwrite.record_channel_failure(
                                db_conn, 'spindle', ch, method_str,
                                frequency[0], frequency[1], stages_key, e)
                        # Write an error sentinel (not an empty list) so downstream
                        # import can tell a failed channel apart from one that
                        # legitimately had no spindles and re-run it.
                        elif json_dir and create_empty_json:
                            try:
                                stages_str = "".join(stage) if stage else "all"
                                ch_json_file = os.path.join(json_dir, f"spindles_{method_str}_{freq_str}_{stages_str}_{ch}.json")
                                with open(ch_json_file, 'w', encoding='utf-8') as f:
                                    json.dump({"error": str(e), "channel": ch}, f)
                                self.logger.info(f"Wrote error-sentinel JSON for channel {ch} after failure")
                            except Exception as json_e:
                                self.logger.error(f"Error creating sentinel JSON for channel {ch}: {json_e}")

        if write_db and db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass

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
                              frequency=None, ref_chan=None, grp_name='eeg', n_fft_sec=4, 
                              file_pattern=None,skip_empty_files=True):
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
        #self.logger.warning("export_spindle_parameters_to_csv is deprecated. Please use calculate_and_store_parameters() and export_parameters_to_csv() instead.")
        
        # Call the new methods as a migration path
        #db_path = os.path.join(os.path.dirname(csv_file), "spindle_parameters.db")
        #self.calculate_and_store_parameters(json_input, db_path, export_params, frequency, n_fft_sec=n_fft_sec, file_pattern=file_pattern)
        #self.export_parameters_to_csv(db_path, csv_file)
        
        #return None  # Original returned a dict of parameters

        from wonambi.trans.analyze import event_params, export_event_params
        import glob

        self.clean_memory()
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
        
        if not json_files:
            self.logger.warning(f"No JSON files found matching pattern: {file_pattern}")
            with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["No JSON files found matching pattern:", file_pattern])
            return None


        # Load spindles from JSON files
        all_spindles = []
        empty_channels = []
        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
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
                    with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
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

        # Process each channel
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
            params = event_params(all_segments, export_params, band=frequency, n_fft=n_fft) # can include 'slope' in event_params 
            
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
            with open(temp_csv, 'r', newline='', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
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
                    with open(temp_csv, 'r', encoding='utf-8') as src, open(csv_file, 'w', encoding='utf-8') as dst:
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

    
    def export_spindle_density_to_csv(self, json_input, csv_file, stage=None, file_pattern=None,
                                      reject_artifacts=True, reject_arousals=True):
        """
        Export spindle statistics to CSV with both whole night and stage-specific densities.

        The stage-specific density denominator is the artefact-free in-stage time
        actually fed to the detector (per channel), computed with
        :func:`turtlewave_hdEEG.utils.compute_analysed_seconds`, not the sum of
        all scored epochs of the stage. Detection rejects artefact/arousal epochs,
        so using all scored epochs as the denominator systematically
        under-estimates density in proportion to each recording's artefact load.

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
        reject_artifacts : bool, optional
            Subtract time overlapped by 'Artefact' events from the density
            denominator. Should match the detection run's setting. Default True.
        reject_arousals : bool, optional
            Subtract time overlapped by 'Arousal' events from the density
            denominator. Should match the detection run's setting. Default True.

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
        from turtlewave_hdEEG.utils import build_density_denominators

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
                with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
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
                with open(file, 'r', encoding='utf-8') as f:
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

        # Build an epoch time->stage lookup so an event detected over a
        # multi-stage span can be attributed to the single stage of the epoch
        # it actually falls in. Without this, an event tagged ['NREM2','NREM3']
        # is counted under BOTH stages, inflating per-stage density.
        try:
            _epochs = sorted(
                ((float(e['start']), float(e['end']), str(e['stage']))
                 for e in self.annotations.get_epochs()),
                key=lambda x: x[0]
            )
        except Exception as e:
            self.logger.warning(f"Could not build epoch stage lookup: {e}")
            _epochs = []
        _epoch_starts = [e[0] for e in _epochs]

        def _stage_at(t):
            """Return the scored stage of the epoch containing time t, or None."""
            if t is None or not _epochs:
                return None
            import bisect
            idx = bisect.bisect_right(_epoch_starts, t) - 1
            if 0 <= idx < len(_epochs) and _epochs[idx][0] <= t < _epochs[idx][1]:
                return _epochs[idx][2]
            return None

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

        # Build the artefact-free density denominators (per-stage analysed time,
        # detected-stage whole-night time, per-channel whole-night count). This
        # shared helper matches what the detector pooled and logs the reject-type
        # assumption so it is never silent. See utils.build_density_denominators.
        dd = build_density_denominators(
            self.annotations, self.dataset,
            reject_artifacts=reject_artifacts, reject_arousals=reject_arousals,
            stage_list=stage_list, stages_present=spindle_stages,
            logger=self.logger)
        reject_types = dd.reject_types
        detected_stage_set = dd.detected_stage_set
        whole_night_analysed_min = dd.whole_night_analysed_min

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
                sp_stages = []
                if 'stage' in sp:
                    sp_stages = sp['stage'] if isinstance(sp['stage'], list) else [sp['stage']]
                sp_stages = [str(s) for s in sp_stages]

                # If the event spans multiple requested stages, attribute it to
                # the single stage of the epoch it actually occurred in, so it is
                # not double-counted across stages.
                if len(sp_stages) > 1:
                    actual = _stage_at(sp.get('start_time', sp.get('start')))
                    if actual in sp_stages:
                        sp_stages = [actual]

                for sp_stage in sp_stages:
                    # Add to stage-specific spindle count
                    spindles_by_chan_stage[chan][sp_stage].append(sp)
                

        # Calculate statistics by channel for each stage
        stage_channel_stats = defaultdict(dict)
        for chan in set(spindles_by_chan.keys()):
            # Whole night statistics
            all_chan_spindles = spindles_by_chan[chan]

            # Whole-night count is stage-independent: compute once per channel,
            # restricted to the detected stages so it shares the same time base
            # as whole_night_analysed_min.
            whole_night_count = dd.whole_night_count(all_chan_spindles)
            whole_night_density = (whole_night_count / whole_night_analysed_min
                                   if whole_night_analysed_min > 0 else 0)

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

                    # Artefact-free analysed time is per-stage then summed, so
                    # a span shared across stages is not double-counted.
                    analysed_sec = 0.0
                    artefact_sec = 0.0
                    for s in stages_to_include:
                        a, ar = dd.analysed_seconds(s)
                        analysed_sec += a
                        artefact_sec += ar
                    stage_duration_min = analysed_sec / 60.0

                else:
                    # Single stage processing
                    s_str = str(process_stage)
                    stage_spindles = spindles_by_chan_stage[chan].get(s_str, [])
                    stage_name_display = process_stage
                    analysed_sec, artefact_sec = dd.analysed_seconds(s_str)
                    stage_duration_min = analysed_sec / 60.0

                # Skip if no spindles for this stage and channel
                if len(stage_spindles) == 0:
                    continue

                # Count spindles
                stage_count = len(stage_spindles)

                # Calculate density (spindles per minute). whole_night_density
                # is computed once per channel above (stage-independent).
                stage_density = stage_count / stage_duration_min if stage_duration_min > 0 else 0

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
                    'analysed_minutes': stage_duration_min,
                    'artefact_seconds_excluded': artefact_sec,
                }
        
        # Export to CSV - each stage gets its own section
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Add whole night summary
            writer.writerow(['Whole Night Summary'])
            writer.writerow(['Total Recording Duration (min)', f'{total_duration_min:.2f}'])
            writer.writerow(['Detected stages (density time base)',
                             ', '.join(sorted(detected_stage_set)) if detected_stage_set else 'none'])
            writer.writerow(['Whole-night analysed minutes (artefact-free, detected stages)',
                             f'{whole_night_analysed_min:.2f}'])
            writer.writerow(['Stage density denominator',
                             'artefact-free in-stage time fed to detector (per channel)'])
            writer.writerow(['Whole-night density denominator',
                             'artefact-free minutes summed over detected stages (Wake excluded unless detected)'])
            writer.writerow(['Reject types subtracted',
                             ', '.join(reject_types) if reject_types else 'none'])
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
                    'Mean Duration (s)',
                    'Analysed Minutes (artefact-free)',
                    'Artefact Seconds Excluded'
                ])


                # Write channel-specific statistics, sorted by channel name
                for chan in sorted(stage_channel_stats[key].keys()):
                    stats = stage_channel_stats[key][chan]
                    writer.writerow([
                        chan,
                        stats['count'],
                        f"{stats['stage_density']:.4f}",
                        f"{stats['whole_night_density']:.4f}",
                        f"{stats['mean_duration']:.4f}",
                        f"{stats['analysed_minutes']:.4f}",
                        f"{stats['artefact_seconds_excluded']:.4f}"
                    ])
                
                writer.writerow([])
        
        self.logger.info(f"Exported spindle statistics to {csv_file}")
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
                'software_version': dbwrite.provenance().get('turtlewave_version'),
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            self.logger.info(f"Saved detection summary to: {summary_file}")
            return summary_file
        except Exception as e:
            self.logger.error(f"Error saving detection summary: {e}")
            return None

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
                freq_band TEXT,            -- Full text representation (e.g. "9-12Hz")
                freq_lower REAL,           -- Lower bound of frequency band (e.g. 9.0)
                freq_upper REAL,           -- Upper bound of frequency band (e.g. 12.0)
                        
                -- Amplitude metrics
                min_amp REAL,          -- minimum amplitude
                max_amp REAL,          -- maximum amplitude

                peak2peak_amp REAL,    -- peak-to-peak amplitude

                -- Spectral / RMS metrics (from Wonambi event_params)
                rms REAL,              -- RMS (uV)
                power REAL,            -- band power (uV^2)
                peak_power_freq REAL,  -- peak power frequency (Hz)
                energy REAL,           -- energy (uV^2s)
                peak_energy_freq REAL, -- peak energy frequency (Hz)

                -- Processing metadata
                processing_timestamp TEXT,
                n_fft_sec INTEGER,
                
                CONSTRAINT event_chan_time UNIQUE (event_type, channel, start_time, method, freq_lower, freq_upper, stage)
            )''')

            # Create tracking table for batch processing. The primary key is the
            # full detection scope so a resume can skip only channels completed
            # for the same method/band/stage set; the scope columns default so
            # the legacy (channel, event_type)-only CSV-import markers stay
            # idempotent. Existing narrow-PK DBs are migrated in place by
            # dbwrite.ensure_direct_write_schema.
            conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_status (
                channel TEXT NOT NULL,
                event_type TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT '',
                freq_lower REAL NOT NULL DEFAULT 0,
                freq_upper REAL NOT NULL DEFAULT 0,
                stage TEXT NOT NULL DEFAULT '',
                json_file TEXT,
                processed BOOLEAN DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                last_attempt_time TEXT,
                success BOOLEAN DEFAULT 0,
                error_message TEXT,

                PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage)
            )''')

            # Per-cycle sleep-cycle structure (populated by ParalCycles).
            conn.execute('''
            CREATE TABLE IF NOT EXISTS sleep_cycles (
                subject TEXT,
                method TEXT,               -- cycle definition ('2022' or '1979')
                cycle_number INTEGER,      -- 1-based
                nrem_start REAL,           -- seconds from recording start
                nrem_end REAL,
                rem_start REAL,            -- inter-NREM (REM) segment start
                rem_end REAL,              -- cycle end
                nrem_dur_min REAL,         -- full period (N1+N2+N3+absorbed wake)
                nrem_n23_dur_min REAL,     -- N2+N3 only, within the period
                rem_dur_min REAL,
                cycle_dur_min REAL,
                PRIMARY KEY (subject, method, cycle_number)
            )''')

            # Per-subject sleep-stage durations (populated by ParalCycles).
            # DDL kept identical to ParalCycles._ensure_stage_durations_table.
            conn.execute('''
            CREATE TABLE IF NOT EXISTS stage_durations (
                subject TEXT,
                epoch_length REAL,
                wake_min REAL,
                n1_min REAL,
                n2_min REAL,
                n3_min REAL,
                rem_min REAL,
                artefact_min REAL,
                total_min REAL,
                PRIMARY KEY (subject)
            )''')

            # Create indexes for efficient querying
            conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_channel ON events(channel)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timerange ON events(start_time, end_time)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stage ON events(stage)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cycle ON events(cycle)')

            # Bring an existing events table up to the current column set.
            self._ensure_event_param_columns(conn)

            conn.commit()

            # If database didn't exist, log creation
            if not db_exists:
                self.logger.info(f"Created new database at: {db_path}")

            return db_path
        # Use the safe database operation
        return self._safe_database_operation(db_path, init_db)

    # Spectral / RMS parameter columns added to the events table after the
    # amplitude columns. Kept in sync with the CREATE TABLE definition above and
    # with swprocessor.ParalSWA (the k_complex path delegates to that exporter).
    _EVENT_PARAM_COLUMNS = (
        ('rms', 'REAL'),
        ('power', 'REAL'),
        ('peak_power_freq', 'REAL'),
        ('energy', 'REAL'),
        ('peak_energy_freq', 'REAL'),
    )

    def _ensure_event_param_columns(self, conn):
        """Additively migrate the ``events`` table to hold spectral/RMS columns.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so existing columns are read
        from ``PRAGMA table_info(events)`` and only absent ones are added. The
        operation is idempotent and touches no existing rows (new columns are
        ``NULL`` on legacy rows) or other tables.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open connection to the target database. The caller is responsible
            for committing; this method commits only when it actually alters the
            schema so a no-op run leaves the connection state unchanged.

        Returns
        -------
        list of str
            Names of the columns that were added (empty when already current).
        """
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        existing = {row[1] for row in cursor.fetchall()}
        # No events table yet (fresh DB before CREATE TABLE): nothing to migrate.
        if not existing:
            return []
        added = []
        for col, col_type in self._EVENT_PARAM_COLUMNS:
            if col not in existing:
                cursor.execute(f"ALTER TABLE events ADD COLUMN {col} {col_type}")
                added.append(col)
        if added:
            conn.commit()
            self.logger.info(
                f"Migrated events table: added spectral/RMS columns {added}"
            )
        return added

    
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
        
        # Clean memory before starting
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

        def _norm_stage(val):
            """Normalize a Stage cell to the joined form stored in the DB.

            The duplicate check and the INSERT must agree on this, otherwise
            re-importing a multi-stage CSV never matches existing rows and
            append mode silently overwrites instead of skipping.
            """
            import ast
            if isinstance(val, list):
                return "".join(str(s) for s in val)
            if isinstance(val, str) and '[' in val:
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return "".join(str(s) for s in parsed)
                except Exception:
                    pass
            return str(val)

        # Read the CSV file
        self.logger.info(f"Reading parameters from CSV: {csv_file}")
        try:
            # First determine how many rows to skip (header plus statistics)
            with open(csv_file, 'r', encoding='utf-8') as f:
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
                # Auto-upgrade an existing DB (initialize_sqlite_database is only
                # called when the file is absent, so migrate here for old DBs).
                self._ensure_event_param_columns(conn)
                # Determine event type from CSV filename or content
                event_type = "spindle"  # Default
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
                    'RMS (uV)': 'rms',
                    'Power (uV^2)': 'power',
                    'Peak power frequency (Hz)': 'peak_power_freq',
                    'Energy (uV^2s)': 'energy',
                    'Peak energy frequency (Hz)': 'peak_energy_freq',
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
                freq_lower = None
                freq_upper = None
                
                # Try to extract frequency from filename (e.g., spindle_parameters_Moelle2011_9.0-12.0Hz_NREM2NREM3.csv)
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
                
                # Resolve method with precedence: a truthful 'Method' CSV column
                # (written by the DB export; preserves slash-methods like
                # 'AASM/Massimini2004') over the lossy filename parse
                # (filename.split('_')[2] mangles slash-methods to 'AASM' and
                # would corrupt events.method on an INSERT OR REPLACE re-import).
                # Legacy JSON-exported CSVs have no 'Method' column, so they keep
                # the historical filename-parse behaviour unchanged.
                method = None
                if 'Method' in df.columns:
                    method_vals = df['Method'].dropna()
                    if len(method_vals) > 0:
                        method = str(method_vals.iloc[0])
                if method is None:
                    method = "unknown"
                    if "_" in filename:
                        parts = filename.split('_')
                        if len(parts) > 2:
                            # Typically the format is spindle_parameters_METHOD_freq_stages.csv
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
                    
                # Pre-check existing using UUID to avoid constraint violations
                existing_events = set()
                if append and 'Channel' in df.columns and 'Start time' in df.columns:
                    # Get all unique combinations of event_type, channel, start_time
                    channels = df['Channel'].astype(str).tolist()
                    start_times = df['Start time'].astype(float).tolist()
                    
                    batch_size = 100  # Process in batches to avoid memory issues
                    for batch_start in range(0, len(channels), batch_size):
                        batch_end = min(batch_start + batch_size, len(channels))
                        batch_channels = channels[batch_start:batch_end]
                        batch_start_times = start_times[batch_start:batch_end]
                        
                        # Build a query to get existing events matching these combinations
                        query_parts = []
                        query_params = []
                        
                        for batch_idx in range(len(batch_channels)):
                            original_idx = batch_start + batch_idx
                            freq_lower = df['freq_lower'].iloc[original_idx] if 'freq_lower' in df.columns else None
                            freq_upper = df['freq_upper'].iloc[original_idx] if 'freq_upper' in df.columns else None
                            stage = _norm_stage(df['Stage'].iloc[original_idx]) if 'Stage' in df.columns else None

                            # `IS` (not `=`) so NULL freq bounds match NULL, since `x = NULL` is never true in SQL.
                            query_parts.append("(event_type = ? AND channel = ? AND start_time = ? AND method = ? AND freq_lower IS ? AND freq_upper IS ? AND stage = ?)")
                            query_params.extend([event_type, batch_channels[batch_idx], batch_start_times[batch_idx], method, freq_lower, freq_upper, stage])
        
                        
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
                        _norm_stage(row.get('Stage',''))
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
                    row['Stage'] = _norm_stage(row['Stage'])

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
                        if not append and row['exists_in_db']:
                            # Update existing row when not in append mode
                            update_columns = [col for col in db_columns if col != 'uuid']
                            update_values = [val for i, val in enumerate(values) if db_columns[i] != 'uuid']

                            # Update based on the unique constraint, not just UUID.
                            # `IS` on freq bounds so NULL matches NULL; stage is already normalized above.
                            cursor.execute(f"""
                            UPDATE events
                            SET {', '.join([f'{col} = ?' for col in update_columns])}
                            WHERE event_type = ? AND channel = ? AND start_time = ? AND method = ?
                                AND freq_lower IS ? AND freq_upper IS ? AND stage = ?
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
                        failed_channels = {}
                        for file in all_json_files:
                            try:
                                # Extract channel name from filename
                                # Assuming format like "spindles_method_freq_stage_CHANNELNAME.json"
                                channel_name = os.path.basename(file).split('_')[-1].replace('.json', '')
                                # Skip if channel already in processed_channels
                                if channel_name in processed_channels:
                                    continue

                                # Read JSON file to check its contents
                                with open(file, 'r', encoding='utf-8') as f:
                                    content = json.load(f)

                                # An empty array means the channel had no events;
                                # an error-sentinel dict means detection failed and
                                # must be re-run, so record it as unsuccessful.
                                if isinstance(content, dict) and 'error' in content:
                                    failed_channels[channel_name] = str(content.get('error', 'unknown error'))
                                    self.logger.warning(f"Found error-sentinel JSON for channel: {channel_name}")
                                elif isinstance(content, list) and len(content) == 0:
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

                        # Record failed channels as unsuccessful so a resume re-runs them
                        for channel, err in failed_channels.items():
                            cursor.execute('''
                            INSERT OR REPLACE INTO processing_status
                            (channel, event_type, processed, success, attempts, last_attempt_time, error_message)
                            VALUES (?, ?, 1, 0, 1, datetime('now'), ?)
                            ''', (channel, event_type, err[:500]))

                        if empty_channels:
                            self.logger.info(f"Recorded {len(empty_channels)} channels with no events: {', '.join(empty_channels)}")
                        if failed_channels:
                            self.logger.warning(f"Recorded {len(failed_channels)} failed channels: {', '.join(failed_channels)}")
                        # Add empty channels count to stats
                        stats["empty_channels"] = len(empty_channels)
                        stats["failed_channels"] = len(failed_channels)
                    
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


