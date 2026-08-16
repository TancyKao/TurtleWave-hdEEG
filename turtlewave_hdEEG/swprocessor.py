import numpy as np
import time
import os
#import multiprocessing
import csv
from wonambi.trans import select, fetch, math
from wonambi.attr import Annotations
from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave as DetectSlowWave
from turtlewave_hdEEG import dbwrite
import json
import datetime
import logging
import uuid as _uuid_mod


#: Basename prefixes of JSON files this package writes into a results
#: directory that are NOT per-channel event files. The parameter exporters skip
#: them so an unfiltered glob cannot mistake one for a channel that failed.
NON_EVENT_JSON_PREFIXES = ('detection_summary_', 'redetect_request')


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
        # Named after the module (swprocessor), not 'swaprocessor': the logger
        # name is what every log line shows, so a typo here misnames the module
        # in every record and breaks per-module handler configuration.
        logger = logging.getLogger('turtlewave_hdEEG.swprocessor')
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
                self.logger.debug("psutil not available for advanced memory cleanup")
        
        self.logger.debug("Memory cleanup performed")



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
                     create_empty_json=True,
                     *, write_db=False, db_path=None, resume=False,
                     run_params=None, replace_channels=None,
                     event_type='slow_wave', citation=None, n_fft_sec=4):
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
        min_dur : float or None
            Minimum event duration in seconds (method-dependent override)
        max_dur : float or None
            Maximum event duration in seconds (method-dependent override)
        detrend : bool
            Whether to detrend the signal before detection
        polar : str
            'normal' or 'opposite' for handling signal polarity
        reject_artifacts : bool
            Whether to exclude segments marked with artifact annotations
        reject_arousals : bool
            Whether to exclude segments marked with arousal annotations
        stage : list or str
            Sleep stage(s) to analyze
        cat : tuple
            Category specification for data selection
        peak_thresh_sigma : float or None
            Peak threshold in standard deviations (for Ngo2015 method)
        ptp_thresh_sigma : float or None
            Peak-to-peak threshold in standard deviations (for Ngo2015 method)
        save_to_annotations : bool
            Whether to save detected slow waves to annotations
        json_dir : str or None
            Directory to save individual channel JSON files
        create_empty_json : bool
            Whether to create empty JSON files when no slow waves are found
        write_db : bool, keyword-only, default False
            When True, write detected events straight into a SQLite database
            (``db_path``) in addition to the JSON output, via the direct-write
            path (deterministic uuid5 rows, detector-own morphology in the
            ``det_*`` columns, batched re-measured amplitude/spectral columns,
            per-scope ``processing_status`` tracking and a ``detection_runs``
            provenance row). When False the behaviour is byte-identical to the
            legacy JSON-only path.
        db_path : str or None, keyword-only
            Target SQLite database (or directory -> ``neural_events.db``).
        resume : bool, keyword-only, default False
            When True (and ``write_db``), channels already recorded as
            ``success = 1`` for the same scope are skipped.
        run_params : dict or None, keyword-only
            Extra parameters merged into ``detection_runs.params_json``.
        replace_channels : iterable of str or None, keyword-only
            Scoped channel re-detection (P3). Channels in this set have their
            existing rows for this exact scope (event_type, method, band)
            DELETE-then-INSERT replaced in one transaction; channels not in the
            set keep the append/upsert path and are never touched. Only
            meaningful with ``write_db=True``. ``None`` (default) disables it.
        event_type : str, keyword-only, default 'slow_wave'
            Event type label used for DB rows / scope (``'k_complex'`` when a KC
            caller delegates here).
        citation : str or None, keyword-only
            Literature citation for the provenance row; auto-resolved from the
            method when None.
        n_fft_sec : int, keyword-only, default 4
            FFT window (seconds) for the batched spectral re-measurement.

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

        # Two forms of the method, deliberately kept apart because they have
        # conflicting requirements and one value cannot satisfy both:
        #   method_db  - canonical, UNESCAPED ('AASM/Massimini2004'). Everything
        #                stored in or queried from the database uses this, so
        #                direct-write rows match CSV-imported rows and the
        #                citation table (keyed on the unescaped name).
        #   method_str - filesystem-safe ('AASM_Massimini2004'). ONLY for
        #                filenames and path components; existing result
        #                directories are named this way.
        method_db = "_".join(method) if isinstance(method, list) else str(method)
        method_str = method_db.replace('/', '_')

        # Convert frequency to string. Single source of truth for the band
        # token so the JSON filenames written here and any file_pattern a
        # caller rebuilds later cannot drift (see dbwrite.fmt_freq_token).
        freq_str = dbwrite.fmt_freq_token(frequency[0], frequency[1])

        self.logger.info(f"Starting slow wave detection with method={method_db}, frequency={freq_str}")
        self.logger.debug(f"Parameters: channels={chan}, reject_artifacts={reject_artifacts}, reject_arousals={reject_arousals}")

        # Log adaptive threshold parameters if applicable
        first_method = method[0] if isinstance(method, list) and len(method) > 0 else method
        if first_method == 'Ngo2015' and peak_thresh_sigma is not None and ptp_thresh_sigma is not None:
            self.logger.info(f"Using adaptive thresholds: peak_thresh_sigma={peak_thresh_sigma}, ptp_thresh_sigma={ptp_thresh_sigma}")

        # ------------------------------------------------------------------
        # Direct-to-DB write path setup (opt-in; JSON behaviour unchanged).
        # ------------------------------------------------------------------
        stages_key = "".join(stage) if stage else "all"
        db_conn = None
        run_id = None
        db_skip = set()
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
                        'frequency': list(frequency),
                        'trough_duration': list(trough_duration),
                        'neg_peak_thresh': neg_peak_thresh,
                        'p2p_thresh': p2p_thresh,
                        'min_dur': min_dur, 'max_dur': max_dur,
                        'detrend': detrend, 'polar': polar,
                        'peak_thresh_sigma': peak_thresh_sigma,
                        'ptp_thresh_sigma': ptp_thresh_sigma,
                        'method': method_db,
                        'ref_chan': ref_chan, 'cat': cat,
                        'reject_artifacts': reject_artifacts,
                        'reject_arousals': reject_arousals,
                        'n_fft_sec': n_fft_sec,
                    }
                    if run_params:
                        params_dict.update(run_params)
                    run_citation = citation or dbwrite.method_citation(method_db)
                    dbwrite.record_run(
                        db_conn, run_id, event_type, method_db, run_citation,
                        json.dumps(params_dict, default=str),
                        ref_chan, polar, stage, reject_artifacts, reject_arousals)
                    if resume:
                        db_skip = dbwrite.resume_skip_channels(
                            db_conn, event_type, method_db,
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
                        with open(annotation_file_path, 'w', encoding='utf-8') as f:
                            f.write('<?xml version="1.0" ?>\n<annotations><dataset><filename>')
                            if hasattr(self.dataset, 'filename'):
                                f.write(self.dataset.filename)
                            f.write('</filename></dataset><rater><name>Wonambi</name></rater></annotations>')
                        new_annotations = Annotations(annotation_file_path)
                    self.logger.info(
                        f"Will save slow waves to new annotation file: "
                        f"{annotation_file_path}")

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

        # Build an epoch time->stage lookup so each detected wave can be tagged
        # with the single stage of the epoch it actually falls in. Detecting over
        # a multi-stage request (e.g. ['NREM2','NREM3']) otherwise tags every wave
        # with the whole list, which double-counts events across stages downstream.
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

        # Scoped channel re-detection (P3): channels whose existing rows are
        # DELETE-then-INSERT replaced for this scope; all others stay on P2's
        # append/upsert path untouched.
        replace_set = {str(c) for c in replace_channels} if replace_channels else set()

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
                    channel_slow_waves = []
                    channel_json_slow_waves = []
                    # Direct-write accumulators (populated only when write_db).
                    channel_db_events = []
                    channel_param_segments = []

                    ## Loop through methods
                    for m, meth in enumerate(method):
                        self.logger.info(f"Applying method: {meth}")
                            
                        for i, seg in enumerate(segments):
                            self.logger.info(f'Detecting events, segment {i + 1} of {len(segments)}')
                            # Create a copy of the segment for processing
                            processed_seg = seg.copy()

                            # Do NOT invert here. `polar` is passed to
                            # DetectSlowWave below, which forwards it as
                            # Wonambi's own `opts.invert`; each slow-wave
                            # method then negates its local copy of the signal
                            # exactly once (wonambi/detect/slowwave.py:192,
                            # :256, :322). Any inversion added here would
                            # cancel that and make polar='opposite' identical
                            # to polar='normal'. Note `seg.copy()` above is a
                            # shallow dict copy, so processed_seg['data'] is
                            # still the caller's ChanTime until detrend
                            # replaces it — an in-place negation here would
                            # also leak across methods.
                            # Locked down by test_slow_wave_polarity in
                            # tests/test_turtlewave.py.

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

                                # Assemble the direct-DB event: deterministic
                                # uuid5, single resolved stage, detector-own
                                # morphology, and an in-memory window (from the
                                # data the detector saw, i.e. processed_seg) for
                                # batched re-measurement.
                                if write_db:
                                    sw_start = float(sw.get('start', 0))
                                    sw_end = float(sw.get('end', 0))
                                    sw_dur = float(sw.get('dur', sw_end - sw_start))
                                    single_stage = _stage_at(sw_start)
                                    if single_stage is None and isinstance(stage, (list, tuple)) and len(stage) == 1:
                                        single_stage = stage[0]
                                    morph = dbwrite.event_det_morphology(sw)
                                    ev = {
                                        'uuid': dbwrite.event_uuid5(
                                            event_type, ch, sw_start, meth,
                                            frequency[0], frequency[1], single_stage),
                                        'start_time': sw_start, 'end_time': sw_end,
                                        'duration': sw_dur, 'stage': single_stage,
                                        'method': meth,
                                    }
                                    ev.update(morph)
                                    channel_db_events.append(ev)
                                    channel_param_segments.append(
                                        dbwrite.make_param_segment(
                                            processed_seg['data'], sw_start, sw_end,
                                            event_type, single_stage, ch))

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
                                    
                                    # Attribute the wave to the single stage of
                                    # the epoch it actually occurred in. Fall back
                                    # to the requested stage only if the epoch
                                    # lookup is unavailable, so behaviour degrades
                                    # to the old (list) form rather than crashing.
                                    actual_stage = _stage_at(sw_data['start_time'])
                                    if actual_stage is not None:
                                        sw_data['stage'] = actual_stage
                                    elif isinstance(stage, (list, tuple)) and len(stage) == 1:
                                        sw_data['stage'] = stage[0]
                                    else:
                                        sw_data['stage'] = stage
                                    sw_data['freq_range'] = frequency
                                    
                                    channel_json_slow_waves.append(sw_data)
                                    
                    all_slow_waves.extend(channel_slow_waves)
                    self.logger.info(f"Found {len(channel_slow_waves)} slow waves in channel {ch}")

                    # Direct-DB write: one batched re-measurement + one
                    # transaction per channel, BEFORE the JSON write.
                    if write_db and db_conn is not None:
                        batched = dbwrite.compute_batched_params(
                            channel_param_segments, frequency, s_freq,
                            n_fft_sec, self.logger)
                        dbwrite.write_channel_events(
                            db_conn, run_id, event_type, ch, method_db,
                            frequency[0], frequency[1], stages_key,
                            channel_db_events, batched, rec_start,
                            n_fft_sec, self.logger,
                            replace=(ch in replace_set), replace_methods=method)
                        self.logger.info(
                            f"Wrote {len(channel_db_events)} {event_type} rows for "
                            f"channel {ch} to the database")

                    stages_str = "".join(stage) if stage else "all"
                    if json_dir :
                        try:
                            ch_json_file = os.path.join(json_dir, 
                                                      f"slowwaves_{method_str}_{freq_str}_{stages_str}_{ch}.json")
                            
                            # Create empty JSON if no waves found but flag is set
                            if not channel_json_slow_waves and create_empty_json:
                                self.logger.debug(f"Creating empty JSON file for channel {ch} (no slow waves detected)")
                                with open(ch_json_file, 'w', encoding='utf-8') as f:
                                    json.dump([], f)
                            elif channel_json_slow_waves:
                                with open(ch_json_file, 'w', encoding='utf-8') as f:
                                    json.dump(channel_json_slow_waves, f, indent=2)
                                self.logger.info(f"Saved slow wave data for channel {ch} to {ch_json_file}")
                        except Exception as e:
                            self.logger.error(f"Error saving channel JSON: {e}")
                except Exception as e:
                        # A real read/detection failure is logged as an error with
                        # a traceback so it is not mistaken for a channel that
                        # legitimately had no slow waves.
                        self.logger.error(f'Failed to process channel {ch}: {e}', exc_info=True)
                        # In the direct-DB path, record the failure in
                        # processing_status (success=0) instead of an error
                        # sentinel JSON, so a resume re-runs only this channel.
                        if write_db and db_conn is not None:
                            dbwrite.record_channel_failure(
                                db_conn, event_type, ch, method_db,
                                frequency[0], frequency[1], stages_key, e)
                        # Write an error sentinel (not an empty list) so downstream
                        # import can tell a failed channel apart from one that
                        # legitimately had no slow waves and re-run it.
                        elif json_dir and create_empty_json:
                            try:
                                stages_str = "".join(stage) if stage else "all"
                                ch_json_file = os.path.join(json_dir,
                                                        f"slowwaves_{method_str}_{freq_str}_{stages_str}_{ch}.json")
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
                                         n_fft_sec=4, file_pattern=None, skip_empty_files=True,
                                         event_type='slow_wave', strict=True):
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
        skip_empty_files : bool
            Whether to list the channels whose JSON held no events in the log
            at INFO (``False``) or only at DEBUG (``True``, the default). It no
            longer changes the CSV: a run that detected nothing always writes a
            header-only CSV, so the file is present and parseable either way.
        event_type : str
            Value written into the CSV "Event type" column. ``ParalKC`` passes
            ``'k_complex'`` so K-complexes are not mislabelled as slow waves.
        strict : bool
            If True (default), raise ``FileNotFoundError`` when ``file_pattern``
            matches no JSON file. A zero-match export is almost always a
            filename round-trip bug (the band or method token in the pattern
            not matching what the detector wrote), and returning quietly made
            that indistinguishable from a genuinely empty run. Pass
            ``strict=False`` to restore the silent behaviour.

        Returns
        -------
        dict or None
            Dictionary of calculated parameters, or None when there was
            nothing to export.

        Raises
        ------
        FileNotFoundError
            If ``strict`` is True and no JSON file matches ``file_pattern``.

        Notes
        -----
        Three outcomes look similar but mean different things and are kept
        distinct on purpose.

        * ``file_pattern`` matches no JSON file at all: the detector's output
          could not be found, almost always a filename round-trip bug. Raises
          ``FileNotFoundError``.
        * Every matched JSON holds an empty list: the detector ran on every
          channel and genuinely found nothing. Writes a header-only CSV and
          returns None, so the import step reports a clean no-op.
        * Any matched JSON is an error sentinel (``{"error": ..., "channel":
          ...}``), an unrecognised payload, or unreadable: that channel FAILED
          and is not "no events detected". The failed channels are logged at
          ERROR by name. If no channel produced any event, no CSV is written at
          all, so the run cannot be mistaken downstream for an empty night; the
          import step then fails loudly on the missing file. If other channels
          did produce events, their parameters are still exported, with the
          failures reported.
        """
        from wonambi.trans.analyze import event_params, export_event_params
        import glob

        # Clean memory first
        self.clean_memory()

        self.logger.debug("Calculating slow wave parameters for CSV export...")

        # Load slow waves from JSON file(s)
        json_files = []
        if file_pattern:
            all_json_files = glob.glob(os.path.join(json_input, "*.json"))
            json_files = [f for f in all_json_files if
                        f"{file_pattern}_" in os.path.basename(f) or
                        f"{file_pattern}." in os.path.basename(f)]
        else:
            json_files = glob.glob(os.path.join(json_input, "*.json"))

        # Drop JSON files the pipeline writes into the same directory that are
        # not per-channel event files. With file_pattern given they are already
        # excluded; without one the glob takes every *.json, and a
        # detection_summary_*.json would then be read as a channel whose
        # payload is an unrecognised dict -- i.e. counted as a FAILED channel,
        # which suppresses the CSV of an otherwise clean zero-event run.
        skipped_non_event = [f for f in json_files
                             if os.path.basename(f).startswith(NON_EVENT_JSON_PREFIXES)]
        if skipped_non_event:
            json_files = [f for f in json_files if f not in skipped_non_event]
            self.logger.debug(
                f"Ignoring {len(skipped_non_event)} non-event JSON file(s) in "
                f"{json_input}: "
                f"{', '.join(sorted(os.path.basename(f) for f in skipped_non_event))}")

        self.logger.info(f"Found {len(json_files)} JSON files matching pattern: {file_pattern}")

        if not json_files:
            from .utils import missing_json_message
            msg = missing_json_message(json_input, file_pattern)
            if strict:
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            self.logger.warning(msg)
            return None


        # Load slow waves from JSON files
        all_slow_waves = []
        empty_channels = []
        # Channels whose result cannot be trusted as "no events": detection
        # failed, or the JSON is unreadable/not the expected shape. Kept apart
        # from empty_channels because folding the two together reports a total
        # detection failure as a clean night with no slow waves.
        failed_channels = {}

        def _chan_of(path):
            """Channel name from a per-channel JSON filename.

            The detector names files
            ``{prefix}_{method}_{freq}Hz_{stages}_{channel}.json``, so the
            channel is the last underscore-separated token.
            """
            base = os.path.splitext(os.path.basename(path))[0]
            parts = base.split('_')
            return parts[-1] if len(parts) > 1 else base

        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    slow_waves = json.load(f)

                if isinstance(slow_waves, list):
                    if len(slow_waves) > 0:
                        all_slow_waves.extend(slow_waves)
                    else:
                        empty_channels.append(_chan_of(file))
                        self.logger.debug(f"File {file} contains an empty list (no slow waves)")

                elif isinstance(slow_waves, dict) and 'error' in slow_waves:
                    # Error sentinel written by detect_slow_waves when a channel
                    # raised. Same shape the import side keys on when filling
                    # processing_status, so there is one rule, not two.
                    failed_channels[_chan_of(file)] = str(
                        slow_waves.get('error', 'unknown error'))
                else:
                    # Neither a list of events nor a recognised sentinel: the
                    # channel's result is unknown, which is not the same as
                    # empty.
                    failed_channels[_chan_of(file)] = (
                        f"unexpected JSON format ({type(slow_waves).__name__})")
                    self.logger.warning(f"Unexpected format in {file}")

                self.logger.debug(f"Loaded {len(slow_waves) if isinstance(slow_waves, list) else 0} slow waves from {file}")
            except Exception as e:
                failed_channels[_chan_of(file)] = f"unreadable JSON: {e}"
                self.logger.error(f"Error loading {file}: {e}")

        if failed_channels:
            self.logger.error(
                f"{len(failed_channels)} of {len(json_files)} channel(s) did "
                f"not produce a usable {event_type} result and are NOT "
                f"'no events detected': "
                f"{'; '.join(f'{c} ({r})' for c, r in sorted(failed_channels.items()))}. "
                f"Re-run these channels before using this export.")

        if not all_slow_waves:
            if failed_channels:
                # No channel produced an event AND at least one failed. Writing
                # the header-only CSV here would let a total detection failure
                # read downstream as a clean night with no slow waves. Write
                # nothing instead, so the import step fails loudly on the
                # missing file, which is the truthful outcome.
                self.logger.error(
                    f"No {event_type} parameters were exported: no channel "
                    f"produced an event and {len(failed_channels)} channel(s) "
                    f"failed. This is a FAILED run, not an empty one, so no "
                    f"CSV was written to {csv_file}.")
                return None
            # Every channel returned a genuine empty list: the detector ran
            # everywhere and found nothing. That is a valid result, not a
            # failure. Write a header-only CSV so the file exists and parses:
            # the import step then reads it, finds no rows and reports a clean
            # no-op instead of raising FileNotFoundError on a missing file.
            # This matches the density exporter, which already writes a file
            # for the same input.
            from .utils import write_empty_params_csv
            write_empty_params_csv(csv_file, event_type,
                                   channels=empty_channels or None,
                                   logger=self.logger)
            if empty_channels and not skip_empty_files:
                self.logger.info(
                    f"Channels with no {event_type} events: "
                    f"{', '.join(str(c) for c in empty_channels)}")
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

        self.logger.debug(f"Grouped slow waves by {len(waves_by_chan)} channels")

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
                        
                        # Create segment. `name` flows into the CSV's
                        # "Event type" column via Wonambi's export_event_params,
                        # so callers can override to e.g. 'k_complex'.
                        seg = {
                            'data': data,
                            'name': event_type,
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
        
        self.logger.debug(f"Created {len(all_segments)} segments for parameter calculation")
        
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
            self.logger.debug("Exporting parameters to temporary file")            
            export_event_params(temp_csv, params, count=None, density=None)

            # Store UUIDs
            uuid_dict = {}
            for i, segment in enumerate(all_segments):
                if 'uuid' in segment:
                    uuid_dict[i] = segment['uuid']

            # Process CSV
            self.logger.debug("Processing CSV to remove summary rows and add HH:MM:SS format")
            with open(temp_csv, 'r', newline='', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
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
                    with open(temp_csv, 'r', encoding='utf-8') as src, open(csv_file, 'w', encoding='utf-8') as dst:
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
                self.logger.debug(f"Could not remove temporary file {temp_csv}")

            self.logger.info(f"Successfully exported to {csv_file} with HH:MM:SS time format")
            return params
        except Exception as e:
            self.logger.error(f"Error calculating parameters: {e}", exc_info=True)
            return None

    def export_slow_wave_density_to_csv(self, json_input, csv_file, stage=None, file_pattern=None,
                                        reject_artifacts=None, reject_arousals=None):
        """
        Export slow wave statistics to CSV with both whole night and stage-specific densities.

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
            Sleep stage(s) to include
        file_pattern : str or None
            Pattern to filter JSON files
        reject_artifacts : bool or None, optional
            Subtract time overlapped by 'Artefact' events from the density
            denominator. Should match the detection run's setting. ``None``
            (the default) assumes True and logs a warning saying so; pass the
            value explicitly to confirm it matches the run and silence the
            warning.
        reject_arousals : bool or None, optional
            Subtract time overlapped by 'Arousal' events from the density
            denominator. Should match the detection run's setting. ``None``
            (the default) assumes True and logs a warning saying so; pass the
            value explicitly to confirm it matches the run and silence the
            warning.
        """
        import glob
        from collections import defaultdict
        from turtlewave_hdEEG.utils import build_density_denominators

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
                with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
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
                with open(file, 'r', encoding='utf-8') as f:
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

        # Build the artefact-free density denominators (per-stage analysed time,
        # detected-stage whole-night time, per-channel whole-night count). This
        # shared helper matches what the detector pooled and logs the reject-type
        # assumption so it is never silent. See utils.build_density_denominators.
        dd = build_density_denominators(
            self.annotations, self.dataset,
            reject_artifacts=reject_artifacts, reject_arousals=reject_arousals,
            stage_list=stage_list, stages_present=wave_stages,
            logger=self.logger)
        reject_types = dd.reject_types
        detected_stage_set = dd.detected_stage_set
        whole_night_analysed_min = dd.whole_night_analysed_min

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
                sw_stages = []
                if 'stage' in sw:
                    sw_stages = sw['stage'] if isinstance(sw['stage'], list) else [sw['stage']]
                sw_stages = [str(s) for s in sw_stages]

                # If the event spans multiple requested stages, attribute it to
                # the single stage of the epoch it actually occurred in, so it is
                # not double-counted across stages.
                if len(sw_stages) > 1:
                    actual = _stage_at(sw.get('start_time', sw.get('start')))
                    if actual in sw_stages:
                        sw_stages = [actual]

                for sw_stage in sw_stages:
                    waves_by_chan_stage[chan][sw_stage].append(sw)

        # Calculate statistics
        stage_channel_stats = defaultdict(dict)
        for chan in set(waves_by_chan.keys()):
            all_chan_waves = waves_by_chan[chan]

            # Whole-night count is stage-independent: compute once per channel,
            # restricted to the detected stages so it shares the same time base
            # as whole_night_analysed_min.
            whole_night_count = dd.whole_night_count(all_chan_waves)
            whole_night_density = (whole_night_count / whole_night_analysed_min
                                   if whole_night_analysed_min > 0 else 0)

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
                    s_str = str(process_stage)
                    stage_waves = waves_by_chan_stage[chan].get(s_str, [])
                    stage_name_display = process_stage
                    analysed_sec, artefact_sec = dd.analysed_seconds(s_str)
                    stage_duration_min = analysed_sec / 60.0

                if len(stage_waves) == 0:
                    continue

                # Calculate statistics
                stage_count = len(stage_waves)

                # whole_night_density is computed once per channel above
                # (stage-independent).
                stage_density = stage_count / stage_duration_min if stage_duration_min > 0 else 0
                
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
                    'analysed_minutes': stage_duration_min,
                    'artefact_seconds_excluded': artefact_sec,
                }
        
        # Export to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Add summary sections
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
                    'Mean Duration (s)',
                    'Analysed Minutes (artefact-free)',
                    'Artefact Seconds Excluded'
                ])

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
                'software_version': dbwrite.provenance().get('turtlewave_version'),
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
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
                self.logger.debug(f"Created directory for database: {db_dir}")
            
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

                # Create tracking table for batch processing. Primary key is the
                # full detection scope (see dbwrite.ensure_direct_write_schema);
                # scope columns default so legacy narrow CSV-import markers stay
                # idempotent, and existing narrow-PK DBs migrate in place.
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
    # with eventprocessor.ParalEvents; the k_complex path delegates to this
    # exporter, so this migration covers both slow_wave and k_complex rows.
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
        """Run a database operation on a properly configured write connection.

        Routed through :func:`~turtlewave_hdEEG.dbwrite.open_write_connection`
        so schema initialisation and CSV import get the same 60 s busy timeout
        and journal-mode handling as the direct-write path. A bare
        ``sqlite3.connect`` here used Python's 5 s default, which is not enough
        on a DELETE-mode database (network drives) where an open review GUI
        holds a read transaction and blocks the writer.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database.
        operation_func : callable
            Called with the open connection; its return value is returned.

        Returns
        -------
        object
            Whatever ``operation_func`` returns.
        """
        conn = None
        try:
            conn = dbwrite.open_write_connection(db_path, logger=self.logger)
            result = operation_func(conn)
            return result
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()


    def import_parameters_csv_to_database(self, csv_file, db_path, append=True,
                                          event_type=None, method=None,
                                          force=False):
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
        event_type : str or None
            Override the inferred event_type. The default heuristic (filename
            substring + CSV "Event type" column) doesn't recognise files like
            ``kc_parameters_*.csv``, so callers ingesting non-SW events
            (e.g. ParalKC) should pass ``event_type='k_complex'`` explicitly.
        method : str or None
            Override the method parsed from the filename. The filename parser
            underscore-splits and grabs ``parts[2]``, which mangles methods
            with embedded underscores (e.g. ``AASM_Massimini2004`` → ``AASM``).
            Pass the original method string here to bypass it.
        force : bool
            Allow the import to proceed even when the target scope already
            holds rows written by the direct-to-database path (rows with a
            non-NULL ``run_id``). The import is ``INSERT OR REPLACE`` on a
            deterministic event UUID, so importing a CSV over those rows blanks
            their ``run_id`` and severs them from their ``detection_runs``
            provenance. Default False refuses instead.

        Returns
        -------
        dict
            Summary of the operation with counts of added, updated and skipped
            rows, plus ``"ok": True`` on success. A CSV written by a run that
            detected no events (header row, no data rows) returns
            ``{"ok": True, "no_events": True, "added": 0, "updated": 0,
            "skipped": 0}`` with a plain-language ``"message"``: nothing was
            imported because there was nothing to import. Callers that treat
            "no rows reached the database" as a failure must check
            ``no_events`` first.

        Raises
        ------
        RuntimeError
            If the target scope already contains direct-written rows and
            ``force`` is False.
        Exception
            Any failure during parsing or the database transaction is
            re-raised rather than being swallowed into an error dict, so a
            failed import can never be mistaken for a clean re-run.
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
            raise FileNotFoundError(
                f"Parameters CSV not found: {csv_file}. The export step that "
                f"should have written it either failed or wrote a different "
                f"filename (check the file_pattern / band token).")
        
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
        self.logger.debug(f"Reading parameters from CSV: {csv_file}")
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
                self.logger.error(
                    f"Could not find a 'Start time' header row in {csv_file}; "
                    f"this is not a parameters CSV (a placeholder CSV written "
                    f"by a zero-match export looks like this). Nothing imported.")
                return {"ok": False, "error": "Could not find header row",
                        "added": 0, "updated": 0, "skipped": 0}
            
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
                # A parameters CSV holding nothing but its header row is what a
                # zero-event run writes: the detector ran and found no events.
                # That is a real result, not a failure, so it returns a clean
                # no-op ("ok": True, "no_events": True) and callers must not
                # present it as an error. A file that does have rows after the
                # header but still parses to an empty frame is a genuinely
                # malformed CSV and keeps reporting failure, as does a file
                # with no header row at all (handled above).
                trailing = [ln for ln in lines[header_row + 1:] if ln.strip()]
                if not trailing:
                    msg = (f"No events to import: "
                           f"{os.path.basename(csv_file)} contains a valid "
                           f"header and no event rows, because the detection "
                           f"run found no events. Nothing was added to the "
                           f"database.")
                    self.logger.info(msg)
                    return {"ok": True, "no_events": True, "message": msg,
                            "added": 0, "updated": 0, "skipped": 0}
                self.logger.warning(f"CSV file contains no data rows: {csv_file}")
                return {"ok": False, "error": "Empty CSV file",
                        "added": 0, "updated": 0, "skipped": 0}
                
            self.logger.info(f"Read {len(df)} parameter rows from CSV")
            

            # Capture caller overrides (rebound inside process_csv_data).
            event_type_override = event_type
            method_override = method

            # Define database operation function
            def process_csv_data(conn):
                cursor = conn.cursor()
                # Auto-upgrade an existing DB (initialize_sqlite_database is only
                # called when the file is absent, so migrate here for old DBs).
                self._ensure_event_param_columns(conn)

                # Determine event type. Caller override wins.
                if event_type_override is not None:
                    event_type = event_type_override
                    # The downstream INSERT pulls the event_type value from
                    # row['Event type'] (CSV column), not from the
                    # df['event_type'] assignment below. Stamp the override
                    # onto the CSV column too so the insert sees it.
                    if 'Event type' in df.columns:
                        df['Event type'] = event_type
                else:
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
                
                # Extract method. Caller override wins; otherwise the
                # filename heuristic underscore-splits and grabs parts[2],
                # which mangles methods that contain underscores in their
                # escaped form (e.g. AASM/Massimini2004 -> AASM_Massimini2004
                # -> 'AASM'). Pass `method=` explicitly to bypass.
                if method_override is not None:
                    method = method_override
                else:
                    # Prefer a truthful 'Method' CSV column (written by the DB
                    # export; preserves slash-methods) over the lossy filename
                    # parse, so a DB-exported CSV re-imports without corrupting
                    # events.method even when no method= arg is passed. Legacy
                    # JSON-exported CSVs lack this column and keep the historical
                    # filename-parse behaviour.
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
                                # Typically the format is sw_parameters_METHOD_freq_stages.csv
                                method = parts[2]

                df['method'] = method
                existing_columns.append('method')
                db_columns.append('method')

                # Refuse to overwrite direct-written rows, whose run_id this
                # INSERT OR REPLACE would silently blank.
                dbwrite.guard_run_id(conn, event_type, method,
                                     freq_lower, freq_upper,
                                     force=force, logger=self.logger)

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
                            
                    self.logger.debug(f"Found {len(existing_events)} existing entries matching event type, channel, and start time")
                
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
                        
                        # Extract the method and frequency-stage parts from the
                        # CSV name. Use a local (not the `method` arg) so the
                        # caller's override survives for the INSERT below.
                        file_method = parts[2]  # e.g. Ferrarelli2007
                        freq_stage = parts[3:]  # ['9-12Hz', 'NREM2NREM3']
                        freq_stage_str = '_'.join(freq_stage).replace('.csv', '')

                        # Map event_type -> the JSON prefix the detector actually
                        # wrote. `{event_type}s` is wrong for slow waves
                        # (slowwaves, not slow_waves) and k-complexes (kcomplex),
                        # which left their empty/failed channels undetected here.
                        json_prefix = {
                            'spindle': 'spindles',
                            'slow_wave': 'slowwaves',
                            'k_complex': 'kcomplex',
                        }.get(event_type, f"{event_type}s")

                        # Construct pattern to find related JSON files
                        json_pattern = f"{json_prefix}_{file_method}_{freq_stage_str}_*"
                        
                        # Find JSON files matching the pattern
                        json_dir = os.path.dirname(csv_file)
                        all_json_files = glob.glob(os.path.join(json_dir, f"{json_pattern}.json"))
                        
                        self.logger.debug(f"Looking for JSON files matching pattern: {json_pattern}.json")
                        self.logger.debug(f"Found {len(all_json_files)} matching JSON files")

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
                                    self.logger.debug(f"Found empty JSON file for channel: {channel_name}")
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

                stats["ok"] = True
                return stats
            # Use the safe database operation
            return self._safe_database_operation(db_path, process_csv_data)

        except Exception as e:
            # Re-raise. Swallowing this into {"error": ..., "added": 0} made a
            # failed import indistinguishable from a clean idempotent re-run,
            # and the driver scripts went on to report success.
            self.logger.error(f"Error processing CSV {csv_file}: {e}",
                              exc_info=True)
            raise


