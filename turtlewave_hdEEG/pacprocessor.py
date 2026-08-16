##pacprocessor.py


"""
pac_processor.py
A class for phase-amplitude coupling (PAC) analysis for high-density EEG data.
Based on the OCTOPUS method from the seapipe package.
"""

import os
import sys
import re
import math
import numpy as np
import time
import json
import csv
import logging
from wonambi.dataset import Dataset
from wonambi.attr import Annotations
from wonambi.trans import fetch
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime

from .utils import derive_subject
from . import dbwrite


# ---------------------------------------------------------------------------
# Canonical PAC metric columns and their aliases.
#
# There are three naming conventions in this codebase for the same quantities:
#   1. the in-memory per-channel results dict (``self.tracking['event_pac']``):
#      ``mi_norm``, ``pval``, ``preferred_phase_rad``, ``n_segments`` ...
#   2. the file-based per-channel ``*_pac_parameters.csv`` written by
#      :meth:`ParalPAC.analyze_pac`: ``mi_norm``, ``median_mi_pval``,
#      ``preferred_phase_rad`` ... (no event-count column; recovered from the
#      sibling ``*_mean_amps.npy``).
#   3. the tracking-derived ``pac_summary_*.csv``: ``MI``, ``MI_pval``,
#      ``PP_rad``, ``Mean_vector_length``, ``N_Segments`` ...
#
# BOTH ingest paths (live write and CSV back-fill) route every metric through
# this single alias map so no metric is silently dropped when a column happens
# to be named differently. This is the one choke point; add new aliases here.
PAC_COLUMN_ALIASES = {
    'mi_raw':              ('mi_raw',),
    'mi_norm':             ('mi_norm', 'MI'),
    'median_mi_pval':      ('median_mi_pval', 'MI_pval', 'pval'),
    'preferred_phase_rad': ('preferred_phase_rad', 'PP_rad'),
    'preferred_phase_deg': ('preferred_phase_deg', 'PP_degrees', 'PP_deg'),
    'mean_vector_length':  ('mean_vector_length', 'Mean_vector_length', 'MVL'),
    'rho':                 ('rho',),
    'rayleigh_z':          ('rayleigh_z', 'Rayleigh_z'),
    'rayleigh_p':          ('rayleigh_p', 'Rayleigh_p'),
    'n_events':            ('n_events', 'N_Segments', 'n_segments'),
}


class ParalPAC:
    """
    A class for parallel detection and analysis of phase-amplitude coupling (PAC)
    across multiple channels of high-density EEG data.
    """
    
    def __init__(self, dataset, annotations=None, rootpath=None, log_level=logging.INFO, log_file=None):
        """
        Initialize the ParalPAC object.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset object containing EEG data
        annotations : Annotations
            Annotations object for storing and retrieving events
        rootpath : str
            Root path for input/output operations
        log_level : int
            Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file : str or None
            Path to log file. If None, logs to console only.
        """
        self.dataset = dataset
        self.annotations = annotations
        self.rootpath = rootpath if rootpath else os.path.dirname(os.path.dirname(dataset.filename))
        
        # Setup logging
        self.logger = self._setup_logger(log_level, log_file)
        
        # Initialize the tracking dictionary
        self.tracking = {'event_pac': {}}
    
    def _setup_logger(self, log_level, log_file=None):
        """Set up a logger for the PAC processor."""
        # Create a logger
        logger = logging.getLogger('turtlewave_hdEEG.pacprocessor')
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

    def pac_method(self, method, surrogate, correction, list_methods=False):
        """
        Format the method and corrections to be applied through Tensorpac.
        Adapted from OCTOPUS module.
        
        Parameters
        ----------
        method : int
            PAC method number
        surrogate : int
            Surrogate method number
        correction : int
            Correction method number
        list_methods : bool
            If True, return a list of method descriptions
        
        Returns
        -------
        tuple or list
            Either a tuple of (method, surrogate, correction) or a list of descriptions
        """
        # Calculate Coupling Strength (idpac)
        methods = {1: 'Mean Vector Length (MVL) [Canolty et al. 2006 (Science)]',
                   2: 'Modulation Index (MI) [Tort 2010 (J Neurophys.)]',
                   3: 'Heights Ratio (HR) [Lakatos 2005 (J Neurophys.)]',
                   4: 'ndPAC [Ozkurt 2012 (IEEE)]',
                   5: 'Phase-Locking Value (PLV) [Penny 2008 (J. Neuro. Meth.), Lachaux 1999 (HBM)]',
                   6: 'Gaussian Copula PAC (GCPAC) `Ince 2017 (HBM)`'}
        surrogates = {0: 'No surrogates', 
                      1: 'Swap phase / amplitude across trials [Tort 2010 (J Neurophys.)]',
                      2: 'Swap amplitude time blocks [Bahramisharif 2013 (J. Neurosci.) ]',
                      3: 'Time lag [Canolty et al. 2006 (Science)]'}
        corrections = {0: 'No normalization',
                       1: 'Subtract the mean of surrogates',
                       2: 'Divide by the mean of surrogates',
                       3: 'Subtract then divide by the mean of surrogates',
                       4: 'Z-score'}
        
        if list_methods:
            return [methods, surrogates, corrections]
        else:
            return (method, surrogate, correction)

    def analyze_pac(self, chan=None, ref_chan=None, grp_name='eeg',
                stage=None, rater=None, reject_artf=['Artefact', 'Arousal'],
                cycle_idx=None, cat=(1,1,1,0), nbins=18,
                phase_freq=(0.5, 1.25), amp_freq=(11, 16),
                idpac=(2, 3, 4), min_dur=1,
                adap_bands_phase='Fixed', adap_bands_amplitude='Fixed',
                filter_opts=None, event_opts=None, invert=False,
                use_detected_events=True, event_type='slow_wave',
                pair_with_spindles=False, time_window=0.5,
                db_path=None, out_dir=None, progress=False,
                subject=None, write_db=False,
                stored_event_type=None, stored_method=None):
        """
        Analyze phase-amplitude coupling (PAC) in the dataset.
        
        Parameters
        ----------
        chan : list or str
            Channels to analyze
        ref_chan : list or str
            Reference channel(s) for re-referencing
        grp_name : str
            Group name for channel selection
        stage : list or str
            Sleep stage(s) to analyze
        rater : str
            Rater name for annotations
        reject_artf : list
            Event types to reject
        cycle_idx : list or None
            Sleep cycle indices to include
        cat : tuple
            Category specification for data selection
        nbins : int
            Number of phase bins
        phase_freq : tuple
            Frequency range for phase signal
        amp_freq : tuple
            Frequency range for amplitude signal
        idpac : tuple
            PAC method settings (method, surrogate, correction)
        min_dur : float
            Minimum event duration in seconds
        adap_bands_phase : str
            Type of frequency band adaptation for phase
        adap_bands_amplitude : str
            Type of frequency band adaptation for amplitude
        filter_opts : dict
            Signal filtering options
        event_opts : dict
            Event processing options
        invert : bool
            Whether to invert signal polarity
        use_detected_events : bool
            Whether to use detected events for PAC analysis
        event_type : str
            Type of events to use ('slow_wave' or 'spindle')
        pair_with_spindles : bool
            If True and event_type is 'slow_wave', will pair slow waves with spindles
        time_window : float
            Time window (in seconds) to search for spindles around slow waves
        db_path : str
            Path to the SQLite database containing events
        out_dir : str
            Output directory for results
        progress : bool
            Whether to show progress bar
        subject : str or None
            Subject identifier written into the ``pac_coupling`` table when
            ``write_db`` is True. If None, it is derived from the root path
            basename (with a warning); prefer passing it explicitly.
        write_db : bool
            If True, persist the in-memory per-channel PAC results directly to
            the ``pac_coupling`` table in ``db_path`` after analysis (see
            :meth:`store_pac_to_database`). Default False, so existing callers
            are unaffected.
        stored_event_type : str or None
            Value written to ``pac_coupling.event_type``. Normally derived
            from ``event_type`` / ``pair_with_spindles``, which only works on
            the event-locked path. On the continuous path
            (``use_detected_events=False``, e.g. theta-gamma coupling) there
            is no event scope to derive, so this must be given explicitly
            (e.g. ``'continuous'``) or the write is refused.
        stored_method : str or None
            Value written to ``pac_coupling.method``. Normally derived from
            ``event_opts['sw_method']`` / ``['spindle_method']``. Must be given
            explicitly whenever those are absent, e.g. ``'theta_gamma'`` for a
            continuous run.

        Returns
        -------
        dict
            Dictionary containing PAC results

        Raises
        ------
        ValueError
            If ``write_db`` is truthy but the stored scope cannot be named --
            no ``stored_event_type``/``stored_method`` and no derivable event
            scope. Writing a continuous result under the derived defaults
            would store it as ``event_type='slow_wave', method='unknown'``,
            i.e. a theta-gamma result indistinguishable from slow-wave
            coupling, so it is refused rather than mislabelled.
        """
        from tensorpac import Pac
        import sys
        import sqlite3

        # Set up logger
        logger = self.logger

        # Fail fast: resolve and validate the scope the rows would be stored
        # under BEFORE any analysis runs. Doing this at the end would burn
        # minutes of per-channel computation on this data before telling the
        # caller the scope cannot be named.
        resolved_event_type = resolved_method = None
        if write_db:
            resolved_event_type, resolved_method = self._resolve_pac_scope(
                event_type=event_type,
                pair_with_spindles=pair_with_spindles,
                event_opts=event_opts,
                use_detected_events=use_detected_events,
                stored_event_type=stored_event_type,
                stored_method=stored_method)
            logger.info(
                f"PAC results will be stored as event_type="
                f"'{resolved_event_type}', method='{resolved_method}'")

        # Get method descriptions
        pac_list = self.pac_method(0, 0, 0, list_methods=True)
        methods = pac_list[0]
        surrogates = pac_list[1]
        corrections = pac_list[2]
        
        # Set up tracking
        tracking = self.tracking
        flag = 0
        
        # Set up default filter options if not provided
        # https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html?highlight=cycle#tensorpac.Pac.cycle
        if filter_opts is None:
            filter_opts = {
                'notch': True,
                'notch_freq': 50,
                'notch_harmonics': True,
                'bandpass': True,
                'highpass': 0.1,
                'lowpass': 45,
                'laplacian': False,
                'dcomplex': 'hilbert',
                'filtcycle': [3, 6],
                'width': 7
            }
        
        # Set up default event options if not provided
        if event_opts is None:
            event_opts = {
                'buffer': 1.0  # Buffer in seconds
            }
        
        logger.info("")
        logger.info(r"""
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ___  ___ _____ ___________ _    _____
            / _ \/ __/_  _/__  / __/ _ | |/|/ / _ \\
        / // / _/  / /  _/ /_\ \/ __ |    / ___/
        /____/___/ /_/  /___/___/_/ |_/_/|_/_/

        Phase-Amplitude Coupling Analysis
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """)
        
        logger.info(f"Method: {methods[idpac[0]]}")
        logger.info(f"Surrogate: {surrogates[idpac[1]]}")
        logger.info(f"Correction: {corrections[idpac[2]]}")
        
        # Log filtering options
        logger.info(f"Using {adap_bands_phase} bands for phase frequency")
        logger.info(f"Using {adap_bands_amplitude} bands for amplitude frequency")
        if filter_opts['notch']:
            logger.info(f"Applying notch filtering: {filter_opts['notch_freq']} Hz")
        if filter_opts['notch_harmonics']: 
            logger.info("Applying notch harmonics filtering")
        if filter_opts['bandpass']:
            logger.info(f"Applying bandpass filtering: {filter_opts['highpass']} - {filter_opts['lowpass']} Hz")
        if filter_opts['laplacian']:
            logger.info("Applying Laplacian filtering")
        
        # 1. Check directories
        if out_dir:
            base_out_dir = out_dir
        else:
            base_out_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
        
        os.makedirs(base_out_dir, exist_ok=True)
        logger.info(f"Using base output directory: {base_out_dir}")

        # 2. Process channel input
        if isinstance(chan, str):
            chan = [chan]
        
        # 3. Process stage input
        if isinstance(stage, str):
            stage = [stage]
        
        # 4. Determine database path
        if db_path is None:
            db_path = os.path.join(self.rootpath, "wonambi", "neural_events.db")
            logger.info(f"Using default database path: {db_path}")
        
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            return None
        
        # 5. Begin channel processing
        for c, ch in enumerate(chan):
            chan_results = {}
            logger.info(f"Processing channel: {ch}")
            
            # Prepare output filename
            if adap_bands_phase == 'Fixed':
                phadap = '-fixed'
            else:
                phadap = '-adap'
                
            if adap_bands_amplitude == 'Fixed':
                ampadap = '-fixed'
            else:
                ampadap = '-adap'
                
            phaname1 = round(phase_freq[0], 2)
            phaname2 = round(phase_freq[1], 2)
            ampname1 = round(amp_freq[0], 2)
            ampname2 = round(amp_freq[1], 2)
            freqs = f'pha-{phaname1}-{phaname2}Hz{phadap}_amp-{ampname1}-{ampname2}Hz{ampadap}'
            

            # Extract method information before creating output directories
            sw_method = event_opts.get('sw_method', 'unknown') if event_opts else 'unknown'
            spindle_method = event_opts.get('spindle_method', 'unknown') if event_opts else 'unknown'
    
            # Create a method-specific output directory
            stage_str = ''.join(stage) if isinstance(stage, list) else str(stage)
            
            # Use consistent directory structure for all output files
            if pair_with_spindles and event_type == 'slow_wave':
                # For slow wave-spindle pairing, include both methods
                method_dir = f"{sw_method}_paired_{spindle_method}"
            else:
                # For single event type analysis
                method_dir = sw_method if event_type == 'slow_wave' else spindle_method

            # Create the full output directory path
            method_out_dir = os.path.join(base_out_dir, method_dir, stage_str)
            os.makedirs(method_out_dir, exist_ok=True)
            logger.info(f"Using method-specific output directory: {method_out_dir}")

            # Create output filenames using the method-specific directory
            if pair_with_spindles and event_type == 'slow_wave':
                outputfile = os.path.join(method_out_dir, f'{ch}_slowwave_spindle_coupling_{freqs}_pac_parameters.csv')
            else:
                outputfile = os.path.join(method_out_dir, f'{ch}_{event_type}_{freqs}_pac_parameters.csv')

            
            # 6. Fetch data segments
            try:
                logger.info(f"Fetching data segments for {ch}")
                
                if use_detected_events:
                    # Get events from SQLite database
                    logger.info(f"Using detected {event_type} events from database")
                    
                    # Connect to database. Read-only use, but with the same 60 s
                    # busy timeout as the writers: under DELETE journal mode
                    # (network drives) a writer blocks readers, and Python's 5 s
                    # default would fail this query while a detection export is
                    # mid-write.
                    conn = None
                    try:
                        conn = sqlite3.connect(db_path, timeout=60.0)
                        cursor = conn.cursor()
                        
                        # Construct SQL query based on parameters
                        if event_type == 'slow_wave':
                            # Get slow waves from the database
                            query = """
                            SELECT uuid, channel, start_time, end_time, duration, stage, method, freq_lower, freq_upper
                            FROM events 
                            WHERE event_type = 'slow_wave' AND channel = ? 
                            """

                            # Initialize params list
                            params = [ch]  # HERE IS MODIFY: Initialize params list with channel

                            # Add method filter if specified
                            if 'sw_method' in event_opts and event_opts['sw_method']:
                                query += " AND method = ?"
                                params.append(event_opts['sw_method'])
                            # Add frequency range filter if specified
                            if 'sw_freq_range' in event_opts and event_opts['sw_freq_range'] and len(event_opts['sw_freq_range']) == 2:
                                query += " AND freq_lower >= ? AND freq_upper <= ?"
                                params.extend(event_opts['sw_freq_range'])

                            # Add stage filter if specified
                            if stage and len(stage) > 0:
                                placeholders = ', '.join(['?' for _ in stage])
                                query += f" AND stage IN ({placeholders})"
                                params.extend(stage)
                                #params = [ch] + stage
                            #else:
                            #    params = [ch]
                            
                            # Execute query
                            cursor.execute(query, params)
                            slow_wave_events = cursor.fetchall()
                            
                            logger.info(f"Found {len(slow_wave_events)} slow wave events for channel {ch}")
                            
                            if pair_with_spindles:
                                logger.info("Looking for slow wave-spindle pairs")
                                
                                # Initialize list for paired events
                                paired_events = []
                                
                                # For each slow wave, find spindles that occur within the time window
                                for sw in slow_wave_events:
                                    sw_uuid, sw_chan, sw_start, sw_end, sw_dur, sw_stage, sw_method, sw_freq_lower, sw_freq_upper = sw
                                    
                                    # Define search window around the slow wave
                                    search_start = sw_start - time_window
                                    search_end = sw_end + time_window
                                    
                                    # Find spindles within this window
                                    spindle_query = """
                                    SELECT uuid, channel, start_time, end_time, duration, stage, method, freq_lower, freq_upper  
                                    FROM events 
                                    WHERE event_type = 'spindle' AND channel = ? 
                                    AND ((start_time >= ? AND start_time <= ?) OR
                                        (end_time >= ? AND end_time <= ?) OR
                                        (start_time <= ? AND end_time >= ?))
                                    """
                                                                
                                    # Initialize spindle_params list with search parameters
                                    spindle_params = [ch, search_start, search_end, 
                                                    search_start, search_end,
                                                    search_start, search_end]
                                      
                                          # Add method filter if specified
                                    if 'spindle_method' in event_opts and event_opts['spindle_method']:
                                        spindle_query += " AND method = ?"
                                        spindle_params.append(event_opts['spindle_method'])
                                    
                                    # Add frequency range filter if specified
                                    if 'spindle_freq_range' in event_opts and event_opts['spindle_freq_range'] and len(event_opts['spindle_freq_range']) == 2:
                                        spindle_query += " AND freq_lower >= ? AND freq_upper <= ?"
                                        spindle_params.extend(event_opts['spindle_freq_range'])

                                    cursor.execute(spindle_query, spindle_params)
                                    related_spindles = cursor.fetchall()
                                    
                                    if related_spindles:
                                         for sp in related_spindles:
                                            sp_uuid, sp_chan, sp_start, sp_end, sp_dur, sp_stage, sp_method, sp_freq_lower, sp_freq_upper = sp

                                            # Create a pair record
                                            paired_events.append({
                                                'sw_uuid': sw_uuid,
                                                'sp_uuid': sp_uuid,
                                                'channel': ch,
                                                'sw_start': sw_start,
                                                'sw_end': sw_end,
                                                'sp_start': sp_start,
                                                'sp_end': sp_end,
                                                'stage': sw_stage,
                                                'sw_method': sw_method,
                                                'sp_method': sp_method
                                            })
                                
                                logger.info(f"Found {len(paired_events)} slow wave-spindle pairs for channel {ch}")
                                
                                if len(paired_events) == 0:
                                    logger.warning(f"No slow wave-spindle pairs found for channel {ch}")
                                    continue
                                
                                # Create segments from paired events
                                segments = []
                                for pair in paired_events:
                                    try:
                                        # Define analysis window that encompasses both events
                                        start_time = min(pair['sw_start'], pair['sp_start'])
                                        end_time = max(pair['sw_end'], pair['sp_end'])
                                        
                                        # Add buffer
                                        buffer = event_opts['buffer']
                                        start_with_buffer = max(0, start_time - buffer)
                                        end_with_buffer = end_time + buffer
                                        
                                        # Read data
                                        data = self.dataset.read_data(chan=[ch], 
                                                                begtime=start_with_buffer, 
                                                                endtime=end_with_buffer)
                                        
                                        # Create segment
                                        seg = {
                                            'data': data,
                                            'name': 'sw_spindle_pair',
                                            'start': start_time,
                                            'end': end_time,
                                            'n_stitch': 0,
                                            'stage': pair['stage'],
                                            'cycle': None,
                                            'chan': ch,
                                            'sw_uuid': pair['sw_uuid'],
                                            'sp_uuid': pair['sp_uuid']
                                        }
                                        segments.append(seg)
                                    except Exception as e:
                                        logger.error(f"Error creating segment for paired events: {e}")
                                
                            else:
                                # Use slow waves directly
                                segments = []
                                for sw in slow_wave_events:
                                    sw_uuid, sw_chan, sw_start, sw_end, sw_dur, sw_stage, sw_method, sw_freq_lower, sw_freq_upper = sw
                                    
                                    try:
                                        # Add buffer
                                        buffer = event_opts['buffer']
                                        start_with_buffer = max(0, sw_start - buffer)
                                        end_with_buffer = sw_end + buffer
                                        
                                        # Read data
                                        data = self.dataset.read_data(chan=[ch], 
                                                                begtime=start_with_buffer, 
                                                                endtime=end_with_buffer)
                                        
                                        # Create segment
                                        seg = {
                                            'data': data,
                                            'name': 'slow_wave',
                                            'start': sw_start,
                                            'end': sw_end,
                                            'n_stitch': 0,
                                            'stage': sw_stage,
                                            'cycle': None,
                                            'chan': ch,
                                            'uuid': sw_uuid
                                        }
                                        segments.append(seg)
                                    except Exception as e:
                                        logger.error(f"Error creating segment for slow wave {sw_uuid}: {e}")
                        
                        elif event_type == 'spindle':
                            # Get spindles from the database
                            query = """
                            SELECT uuid, channel, start_time, end_time, duration, stage, method, freq_lower, freq_upper
                            FROM events 
                            WHERE event_type = 'spindle' AND channel = ? 
                            """
                            # Initialize params list
                            params = [ch]  # Initialize params list with channel

                            # Add method filter if specified
                            if 'spindle_method' in event_opts and event_opts['spindle_method']:
                                query += " AND method = ?"
                                params.append(event_opts['spindle_method'])

                            # Add frequency range filter if specified
                            if 'spindle_freq_range' in event_opts and event_opts['spindle_freq_range'] and len(event_opts['spindle_freq_range']) == 2:
                                query += " AND freq_lower >= ? AND freq_upper <= ?"
                                params.extend(event_opts['spindle_freq_range'])


                            # Add stage filter if specified
                            if stage and len(stage) > 0:
                                placeholders = ', '.join(['?' for _ in stage])
                                query += f" AND stage IN ({placeholders})"
                                params.extend(stage) 
                            
                            # Execute query
                            cursor.execute(query, params)
                            spindle_events = cursor.fetchall()
                            
                            logger.info(f"Found {len(spindle_events)} spindle events for channel {ch}")
                            
                            if len(spindle_events) == 0:
                                logger.warning(f"No spindle events found for channel {ch}")
                                continue
                            
                            # Create segments from spindles
                            segments = []
                            for sp in spindle_events:
                                sp_uuid, sp_chan, sp_start, sp_end, sp_dur, sp_stage, sp_method, sp_freq_lower, sp_freq_upper = sp
                                
                                try:
                                    # Add buffer
                                    buffer = event_opts['buffer']
                                    start_with_buffer = max(0, sp_start - buffer)
                                    end_with_buffer = sp_end + buffer
                                    
                                    # Read data
                                    data = self.dataset.read_data(chan=[ch], 
                                                            begtime=start_with_buffer, 
                                                            endtime=end_with_buffer)
                                    
                                    # Create segment
                                    seg = {
                                        'data': data,
                                        'name': 'spindle',
                                        'start': sp_start,
                                        'end': sp_end,
                                        'n_stitch': 0,
                                        'stage': sp_stage,
                                        'cycle': None,
                                        'chan': ch,
                                        'uuid': sp_uuid
                                    }
                                    segments.append(seg)
                                except Exception as e:
                                    logger.error(f"Error creating segment for spindle {sp_uuid}: {e}")
                        
                        else:
                            logger.error(f"Unknown event type: {event_type}")
                            continue

                        if not segments or len(segments) == 0:
                            logger.warning(f"No valid segments created from database events for {ch}")
                            continue

                        logger.info(f"Created {len(segments)} segments for PAC analysis")

                    except Exception as e:
                        logger.error(f"Error accessing database: {e}", exc_info=True)
                        continue
                    finally:
                        # Always release the DB handle, including on the
                        # early-continue paths above.
                        if conn is not None:
                            conn.close()
                else:
                    # Use standard fetch for continuous data
                    # NEED TO FIX STAGE ISN NREM2NREM3 <===============================
                    segments = fetch(self.dataset, self.annotations, cat=cat, 
                                evt_type=None, stage=stage, cycle=cycle_idx,
                                buffer=event_opts['buffer'])
                    
                    # Read data for the channel
                    segments.read_data(ch, ref_chan, grp_name=grp_name)
                
                if not segments or len(segments) == 0:
                    logger.warning(f"No valid data segments found for {ch}")
                    continue
                
                logger.info(f"Processing {len(segments)} data segments")
                
                # 6. Define PAC object
                pac = Pac(idpac=idpac, f_pha=phase_freq, f_amp=amp_freq, 
                        dcomplex=filter_opts['dcomplex'], 
                        cycle=filter_opts['filtcycle'], 
                        width=filter_opts['width'], 
                        n_bins=nbins,
                        verbose='ERROR')
                
                # 7. Process segments
                # Initialize arrays for results
                ampbin = np.zeros((len(segments), nbins))
                ms = int(np.ceil(len(segments)/50))
                longamp = np.zeros((ms, 50), dtype=object)  # Blocked amplitude series
                longpha = np.zeros((ms, 50), dtype=object)  # Blocked phase series
                
                for s, seg in enumerate(segments):
                    # Print progress
                    if progress:
                        j = s/len(segments)
                        sys.stdout.write('\r')
                        sys.stdout.write(f"Progress: [{'»' * int(50 * j):{50}s}] {int(100 * j)}%")
                        sys.stdout.flush()
                    
                    # Extract data
                    data = seg['data']
                    timeline = data.axis['time'][0]
                    
                    # Fix polarity of recording if needed
                    dat = data()[0][0]
                    if invert:
                        dat = dat * -1
                    
                    # Obtain phase signal
                    pha = np.squeeze(pac.filter(data.s_freq, dat, ftype='phase'))
                    if len(pha.shape) > 2:
                        pha = np.squeeze(pha)
                    
                    # Obtain amplitude signal
                    amp = np.squeeze(pac.filter(data.s_freq, dat, ftype='amplitude'))
                    if len(amp.shape) > 2:
                        amp = np.squeeze(amp)
                    
                    # Extract signal (minus buffer)
                    nbuff = int(event_opts['buffer'] * data.s_freq)
                    minlen = data.s_freq * min_dur
                    if len(pha) >= 2 * nbuff + minlen:
                        pha = pha[nbuff:-nbuff]
                        amp = amp[nbuff:-nbuff]
                    
                    # Put data in blocks (for surrogate testing)
                    longpha[s//50, s%50] = pha
                    longamp[s//50, s%50] = amp
                    
                    # Calculate mean amplitude per phase bin
                    ampbin[s, :] = self._mean_amp(pha, amp, nbins=nbins)
                
                # Clear progress line
                sys.stdout.write('\r')
                sys.stdout.flush()
                
                # 8. If number of events not divisible by block length,
                # pad incomplete final block with randomly resampled events
                rem = len(segments) % 50
                if rem > 0:
                    pads = 50 - rem
                    for pad in range(pads):
                        ran = np.random.randint(0, rem)
                        longpha[-1, rem+pad] = longpha[-1, ran]
                        longamp[-1, rem+pad] = longamp[-1, ran]
                
                # 9. Calculate Coupling Strength
                mi = np.zeros((longamp.shape[0], 1))
                mi_pv = np.zeros((longamp.shape[0], 1))
                
                for row in range(longamp.shape[0]):
                    pha_data = np.zeros((1))
                    amp_data = np.zeros((1))
                    
                    for col in range(longamp.shape[1]):
                        pha_data = np.concatenate((pha_data, longpha[row, col]))
                        amp_data = np.concatenate((amp_data, longamp[row, col]))
                    
                    pha_data = np.reshape(pha_data, (1, 1, len(pha_data)))
                    amp_data = np.reshape(amp_data, (1, 1, len(amp_data)))
                    
                    mi[row] = pac.fit(pha_data, amp_data, n_perm=400, random_state=5, verbose=False)[0][0]
                    mi_pv[row] = pac.infer_pvalues(p=0.95, mcp='fdr')[0][0]
                
                # 10. Calculate preferred phase
                # Normalize amplitude by sum (to get probability distribution)
                ampbin = ampbin / ampbin.sum(-1, keepdims=True)
                ampbin = ampbin.squeeze()
                # Remove NaN trials
                ampbin = ampbin[~np.isnan(ampbin[:, 0]), :]
                ab = ampbin
                
                # Create bins for preferred phase. Centres must match the
                # [-pi, pi] edges used in _mean_amp (np.linspace(-pi, pi, ...)),
                # otherwise the preferred phase is reported 180 deg off.
                vecbin = np.zeros(nbins)
                width = 2 * np.pi / nbins
                for n in range(nbins):
                    vecbin[n] = -np.pi + (n + 0.5) * width
                
                # Calculate circular statistics
                from scipy.stats import circmean, circvar
                
                # Find bin with max amplitude for each trial
                ab_pk = np.argmax(ab, axis=1)
                
                # Convert to angles
                angles = vecbin[ab_pk]
                
                # Calculate mean direction (theta) & mean vector length (rad).
                # Wrap into [-pi, pi] to match the vecbin / ppha convention so
                # up-state (~0) vs down-state (~+/-pi) coupling reads correctly;
                # scipy's default [0, 2pi) would silently disagree with it.
                theta = circmean(angles, low=-np.pi, high=np.pi)
                theta_deg = np.degrees(theta)

                # Calculate circular variance (1 - R)
                circ_var = circvar(angles)
                rad = 1 - circ_var  # Mean resultant length
                
                # Take mean across all segments/events
                ma = np.nanmean(ab, axis=0)
                
                # Correlation between mean amplitudes and phase-giving sine wave
                sine = np.sin(np.linspace(-np.pi, np.pi, nbins))
                sine = np.interp(sine, (sine.min(), sine.max()), (ma.min(), ma.max()))
                
                from scipy.stats import pearsonr
                rho, pv1 = pearsonr(ma, sine)
                
                # # Rayleigh test for non-uniformity of circular data
                ppha = vecbin[ab.argmax(axis=-1)]  # phase in radians
                n = len(ppha)
                r = np.abs(np.sum(np.exp(1j * ppha))) / n
                z = n * r**2  # Get test statistic from the rayleigh_test function
                pv2 = np.exp(-z) # Get p-value directly from the rayleigh_test function


                # 11. Export and save data
                # Save binned amplitudes to numpy file
                amp_file = outputfile.split('_pac_parameters.csv')[0] + '_mean_amps'
                np.save(amp_file, ab)
                
                # Save CFC metrics to dataframe.
                # NOTE: `pac.pac` only holds the LAST block from the loop above,
                # so average the per-block values in `mi` instead (mi_raw used to
                # report just the final block). With idpac normalization enabled,
                # mi_raw and mi_norm are now the same quantity.
                d = pd.DataFrame([
                    np.mean(mi),
                    np.mean(mi),
                    np.median(mi_pv), 
                    theta, 
                    theta_deg, 
                    rad, 
                    rho, 
                    z, 
                    pv2
                ]).transpose()
                
                d.columns = [
                    'mi_raw', 'mi_norm', 'median_mi_pval', 
                    'preferred_phase_rad', 'preferred_phase_deg', 'mean_vector_length',
                    'rho', 'rayleigh_z', 'rayleigh_p'
                ]
                
                d.to_csv(outputfile, sep=',')
                
                logger.info(f"Saved PAC results to {outputfile}")
                logger.info(f"Saved mean amplitudes to {amp_file}.npy")
                
                # Store results in channel_results
                chan_results = {
                    'mi_raw': float(np.mean(mi)),
                    'mi_norm': float(np.mean(mi)),
                    'pval': float(np.median(mi_pv)),
                    'preferred_phase_rad': float(theta),
                    'preferred_phase_deg': float(theta_deg),
                    'mean_vector_length': float(rad),
                    'rho': float(rho),
                    'rayleigh_z': float(z),
                    'rayleigh_p': float(pv2),
                    'n_segments': len(segments),
                    'outputfile': outputfile,
                    'amp_file': f"{amp_file}.npy"
                }
            
            except Exception as e:
                logger.error(f"Error processing channel {ch}: {e}", exc_info=True)
                flag += 1
                continue
            
            # Add results to tracking
            if ch not in tracking['event_pac']:
                tracking['event_pac'][ch] = {}
            
            # Create a key based on parameters
            key = f"{phase_freq[0]}-{phase_freq[1]}Hz_{amp_freq[0]}-{amp_freq[1]}Hz"
            
            tracking['event_pac'][ch][key] = chan_results
        
        # Check completion status
        if flag == 0:
            logger.info("Phase-amplitude coupling analysis finished without errors")
        else:
            logger.warning(f"Phase-amplitude coupling analysis finished with {flag} warnings/errors")

        # Optional direct-to-DB write of the in-memory per-channel results.
        # The scope was resolved and validated at function entry, before any
        # analysis ran.
        if write_db:
            stage_str = ''.join(stage) if isinstance(stage, list) else str(stage)

            if subject is None:
                subject = derive_subject(root_dir=self.rootpath)
                logger.warning(
                    f"write_db=True but no subject given; derived subject "
                    f"'{subject}' from rootpath. Pass subject= explicitly to be safe.")

            # Deliberately NOT wrapped in try/except: a database write that
            # fails must not return normally with the analysis "successful".
            # The CSV/.npy artefacts are already on disk at this point, so
            # raising loses no results.
            db_stats = self.store_pac_to_database(
                db_path=db_path, subject=subject,
                event_type=resolved_event_type, method=resolved_method,
                stage=stage_str, phase_freq=phase_freq, amp_freq=amp_freq,
                idpac=idpac, ref_chan=ref_chan, invert=invert,
                results=tracking['event_pac'])
            logger.info(f"PAC results written to database: {db_stats}")

        return tracking['event_pac']

    def _resolve_pac_scope(self, event_type, pair_with_spindles, event_opts,
                           use_detected_events, stored_event_type=None,
                           stored_method=None):
        """Resolve the ``(event_type, method)`` a PAC row will be stored under.

        ``pac_coupling`` keys on ``(subject, channel, event_type, method,
        stage, phase/amp bounds)``, so these two strings are what make a
        stored result identifiable. Deriving them only works on the
        event-locked path, where ``event_opts`` carries the detector methods.
        On the continuous path (``use_detected_events=False``, e.g.
        theta-gamma) there is nothing to derive: the previous code fell
        through to ``event_type='slow_wave', method='unknown'``, which stores
        a theta-gamma result as slow-wave coupling and reads back as a valid
        row. This method refuses instead.

        Parameters
        ----------
        event_type : str
            Event type analysed (``'slow_wave'`` / ``'spindle'``).
        pair_with_spindles : bool
            Whether slow waves were paired with spindles.
        event_opts : dict or None
            Event options; ``'sw_method'`` / ``'spindle_method'`` are read.
        use_detected_events : bool
            False for the continuous (non-event-locked) path.
        stored_event_type, stored_method : str or None
            Caller overrides. Each wins over derivation independently.

        Returns
        -------
        tuple of str
            ``(event_type, method)`` to store.

        Raises
        ------
        ValueError
            If either component is neither supplied nor derivable.
        """
        resolved_type = (str(stored_event_type)
                         if stored_event_type is not None else None)
        resolved_method = (str(stored_method)
                           if stored_method is not None else None)

        if resolved_type is not None and resolved_method is not None:
            return resolved_type, resolved_method

        if not use_detected_events:
            raise ValueError(
                "analyze_pac(write_db=True, use_detected_events=False) cannot "
                "name the scope of the row it would store: there is no "
                "detected-event method to derive it from, and the derived "
                "defaults would file this continuous result as "
                "event_type='slow_wave', method='unknown'. Pass "
                "stored_event_type= and stored_method= explicitly "
                "(e.g. stored_event_type='continuous', "
                "stored_method='theta_gamma'), or set write_db=False.")

        sw_method = (event_opts or {}).get('sw_method') or 'unknown'
        spindle_method = (event_opts or {}).get('spindle_method') or 'unknown'

        if pair_with_spindles and event_type == 'slow_wave':
            derived_type = 'sw_spindle'
            derived_method = f"{sw_method}_paired_{spindle_method}"
            components = {'sw_method': sw_method,
                          'spindle_method': spindle_method}
        else:
            derived_type = event_type
            if event_type == 'slow_wave':
                derived_method = sw_method
                components = {'sw_method': sw_method}
            else:
                derived_method = spindle_method
                components = {'spindle_method': spindle_method}

        if resolved_type is None:
            resolved_type = derived_type
        if resolved_method is None:
            missing = [k for k, v in components.items() if v == 'unknown']
            if missing:
                raise ValueError(
                    f"analyze_pac(write_db=True) cannot name the method of the "
                    f"row it would store: event_opts is missing {missing} and "
                    f"the derived method would be {derived_method!r}, which "
                    f"records the run as coming from an unidentifiable "
                    f"detector. Pass stored_method= explicitly, or supply "
                    f"event_opts={{'sw_method': ..., 'spindle_method': ...}}.")
            resolved_method = derived_method

        return resolved_type, resolved_method
    
    def _mean_amp(self, pha, amp, nbins=18):
        """
        Calculate mean amplitude in phase bins.
        
        Parameters
        ----------
        pha : array
            Phase time series
        amp : array
            Amplitude time series
        nbins : int
            Number of phase bins
        
        Returns
        -------
        array
            Mean amplitude in each phase bin
        """
        # Convert phase to bin indices
        phase_bins = np.linspace(-np.pi, np.pi, nbins + 1)
        phase_bins_indices = np.digitize(pha, phase_bins) - 1
        phase_bins_indices[phase_bins_indices == nbins] = 0
        
        # Calculate mean amplitude in each bin
        mean_amp_bins = np.zeros(nbins)
        for i in range(nbins):
            bin_mask = phase_bins_indices == i
            if np.any(bin_mask):
                mean_amp_bins[i] = np.mean(amp[bin_mask])
        
        return mean_amp_bins
    
    def generate_comodulogram(self, chan=None, stage=None, 
                            phase_freqs=None, amp_freqs=None,
                            idpac=(2, 3, 4), buffer=1.0,
                            out_dir=None, reject_artf=['Artefact', 'Arousal']):
        """
        Generate a comodulogram for the given channel and parameters.
        
        Parameters
        ----------
        chan : str
            Channel to analyze
        stage : list or str
            Sleep stage(s) to analyze
        phase_freqs : list of tuples
            List of phase frequency bands to analyze
        amp_freqs : list of tuples
            List of amplitude frequency bands to analyze
        idpac : tuple
            PAC method settings (method, surrogate, correction)
        buffer : float
            Buffer in seconds
        out_dir : str
            Output directory for results
        reject_artf : list
            Event types to reject
            
        Returns
        -------
        dict
            Dictionary containing comodulogram results
        """
        from tensorpac import Pac
        
        logger = self.logger
        
        # Process stage input - handle combined stages like "NREM2NREM3"
        if isinstance(stage, str):
            # Handle combined stages like "NREM2NREM3"
            if "NREM2NREM3" in stage:
                stage = ["NREM2", "NREM3"]
                logger.info(f"Parsed combined stage 'NREM2NREM3' into: {stage}")
            elif "NREM" in stage and len(stage) > 5:  # Handle other combined NREM stages
                parsed_stages = []
                # Common stage names to look for
                known_stages = ["NREM1", "NREM2", "NREM3", "REM", "Wake"]
                for known_stage in known_stages:
                    if known_stage in stage:
                        parsed_stages.append(known_stage)
                
                if parsed_stages:
                    logger.info(f"Parsed stage string '{stage}' into: {parsed_stages}")
                    stage = parsed_stages
                else:
                    # If no known stages found, treat it as a single stage
                    stage = [stage]
                    logger.warning(f"Could not parse stage string '{stage}', treating as a single stage")
            else:
                # Single stage or already properly formatted
                stage = [stage] if isinstance(stage, str) else stage
                logger.info(f"Using stage: {stage}")

            


        # Set default phase and amplitude frequencies if not provided
        if phase_freqs is None:
            phase_freqs = [(0.5, 1.5), (1.5, 4), (4, 8), (8, 13)]
        
        if amp_freqs is None:
            amp_freqs = [(8, 13), (13, 30), (30, 45), (55, 95)]
        
        # Set up output directory
        if out_dir is None:
            out_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Fetch data segments
        try:
            logger.info(f"Fetching data segments for channel {chan}")
            
            # Fetch segments based on sleep stage
            segments = fetch(self.dataset, self.annotations, cat=(1, 1,1,0), 
                          evt_type=None, stage=stage, cycle=None,
                          buffer=buffer, reject_artf=reject_artf)
            
            # Read data for the channel
            segments.read_data(chan)
            
            if not segments or len(segments) == 0:
                logger.warning(f"No valid data segments found for {chan}")
                return None
            
            logger.info(f"Processing {len(segments)} data segments")
            
            # Concatenate data from all segments
            all_data = []
            for seg in segments:
                data = seg['data']
                all_data.append(data()[0][0])
            
            # Concatenate data
            if all_data:
                data_array = np.concatenate(all_data)
                
                # Calculate sampling frequency
                s_freq = segments[0]['data'].s_freq
                
                # Create PAC object
                pac = Pac(idpac=idpac, verbose='ERROR')
                
                # Prepare phase and amplitude frequency ranges
                p_freqs = np.array([list(pf) for pf in phase_freqs])
                a_freqs = np.array([list(af) for af in amp_freqs])
                
                # Calculate comodulogram
                logger.info("Calculating comodulogram...")
                
                try:
                    # Try with permutations first
                    pac = Pac(idpac=idpac, verbose='ERROR')
                    comod = pac.filterfit(s_freq, data_array, p_freqs, a_freqs, n_perm=200, verbose=False)
                    logger.info("Comodulogram generated with statistical testing")
                except TypeError as e:
                    if "multiple values for argument 'n_perm'" in str(e):
                        # Fall back without n_perm if there's a parameter conflict
                        pac = Pac(idpac=idpac, verbose='ERROR')
                        comod = pac.filterfit(s_freq, data_array, p_freqs, a_freqs, verbose=False)
                        logger.warning(
                            "Statistical testing disabled due to a parameter "
                            "conflict; comodulogram computed without surrogates")
                    else:
                        logger.error(f"Error in filterfit: {e}")
                        raise e
               
                # Save results
                stagename = '-'.join(stage)
                output_file = f"{out_dir}/comodulogram_{chan}_{stagename}.npz"
                
                np.savez(output_file,
                       comod=comod,
                       p_freqs=p_freqs,
                       a_freqs=a_freqs,
                       idpac=idpac,
                       chan=chan,
                       stage=stage)
                
                logger.info(f"Saved comodulogram to {output_file}")
                
                # Create and save plot
                fig = Figure(figsize=(10, 8), dpi=100)
                ax = fig.add_subplot(111)
                
                # Create meshgrid for plotting
                p_centers = [(p[0] + p[1])/2 for p in phase_freqs]
                a_centers = [(a[0] + a[1])/2 for a in amp_freqs]
                
                # Plot comodulogram as heatmap
                im = ax.imshow(comod, cmap='viridis', aspect='auto', 
                             extent=[p_centers[0], p_centers[-1], a_centers[0], a_centers[-1]],
                             origin='lower')
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('PAC Strength')
                
                # Add labels
                ax.set_xlabel('Phase Frequency (Hz)')
                ax.set_ylabel('Amplitude Frequency (Hz)')
                ax.set_title(f'PAC Comodulogram - {chan} - {stagename}')
                
                # Set y-axis to log scale for better visualization
                ax.set_yscale('log')
                
                # Add frequency band labels
                ax.set_xticks([p[0] for p in phase_freqs] + [phase_freqs[-1][1]])
                ax.set_yticks([a[0] for a in amp_freqs] + [amp_freqs[-1][1]])
                
                # Save figure
                fig_file = f"{out_dir}/comodulogram_{chan}_{stagename}.png"
                fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                
                logger.info(f"Saved comodulogram plot to {fig_file}")
                
                return {
                    'comod': comod,
                    'p_freqs': p_freqs,
                    'a_freqs': a_freqs,
                    'output_file': output_file,
                    'fig_file': fig_file
                }
            
            else:
                logger.warning("No data segments to process")
                return None
            
        except Exception as e:
            logger.error(f"Error generating comodulogram: {e}", exc_info=True)
            return None
    
    def compare_conditions(self, condition1, condition2, test_type='watson_williams', 
                         alpha=0.05, out_dir=None):
        """
        Compare PAC between two conditions.
        
        Parameters
        ----------
        condition1 : dict
            First condition with keys 'amp_file', 'stage', etc.
        condition2 : dict
            Second condition with keys 'amp_file', 'stage', etc.
        test_type : str
            Type of statistical test ('watson_williams' or 'permutation')
        alpha : float
            Significance level
        out_dir : str
            Output directory for results
            
        Returns
        -------
        dict
            Dictionary containing comparison results
        """
        logger = self.logger
        
        # Set up output directory
        if out_dir is None:
            out_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Load data from condition files
        try:
            # Load amplitude data
            amp1 = np.load(condition1['amp_file'])
            amp2 = np.load(condition2['amp_file'])
            
            # Get number of bins
            nbins = amp1.shape[1]
            
            # Create bins for preferred phase. Centres must match the
            # [-pi, pi] edges used in _mean_amp, otherwise the reported
            # preferred phase is 180 deg off (the F/p test is unaffected).
            vecbin = np.zeros(nbins)
            width = 2 * np.pi / nbins
            for n in range(nbins):
                vecbin[n] = -np.pi + (n + 0.5) * width

            # Find preferred phase for each trial
            ab_pk1 = np.argmax(amp1, axis=1)
            ab_pk2 = np.argmax(amp2, axis=1)
            
            # Convert to angles
            angles1 = vecbin[ab_pk1]
            angles2 = vecbin[ab_pk2]
            
            # Perform statistical test
            if test_type == 'watson_williams':
                from scipy.stats import circmean
                from pingouin import circ_r
                
                # Calculate mean direction for each condition. Wrap into [-pi, pi]
                # to match the vecbin / up-down-state convention (scipy defaults
                # to [0, 2pi), which would disagree with analyze_pac).
                theta1 = circmean(angles1, low=-np.pi, high=np.pi)
                theta2 = circmean(angles2, low=-np.pi, high=np.pi)
                
                # Calculate mean vector length for each condition
                r1 = circ_r(vecbin, np.histogram(ab_pk1, bins=nbins)[0], d=width)
                r2 = circ_r(vecbin, np.histogram(ab_pk2, bins=nbins)[0], d=width)
                
                # Perform Watson-Williams test
                try:
                    from pingouin import circ_wwtest
                    
                    # Run Watson-Williams test
                    F, p = circ_wwtest(angles1, angles2, np.ones(angles1.shape), np.ones(angles2.shape))
                    
                    # Save results
                    cond1_name = condition1.get('name', 'Condition1')
                    cond2_name = condition2.get('name', 'Condition2')
                    
                    output_file = os.path.join(out_dir, f"pac_comparison_{cond1_name}_vs_{cond2_name}.csv")
                    
                    results_df = pd.DataFrame({
                        'Condition1': [cond1_name],
                        'Condition2': [cond2_name],
                        'Condition1_PP_rad': [theta1],
                        'Condition1_PP_deg': [np.degrees(theta1)],
                        'Condition1_MVL': [r1],
                        'Condition1_n': [len(angles1)],
                        'Condition2_PP_rad': [theta2],
                        'Condition2_PP_deg': [np.degrees(theta2)],
                        'Condition2_MVL': [r2],
                        'Condition2_n': [len(angles2)],
                        'F': [F],
                        'p': [p],
                        'Significant': [p < alpha]
                    })
                    
                    results_df.to_csv(output_file, index=False)
                    
                    logger.info(f"Saved comparison results to {output_file}")
                    
                    # Create and save plot
                    fig = Figure(figsize=(10, 8), dpi=100)
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Calculate mean amplitudes for each condition
                    mean_amp1 = np.nanmean(amp1, axis=0)
                    mean_amp1 = mean_amp1 / mean_amp1.sum()
                    
                    mean_amp2 = np.nanmean(amp2, axis=0)
                    mean_amp2 = mean_amp2 / mean_amp2.sum()
                    
                    # Create angles for plotting
                    angles = np.linspace(0, 2*np.pi, nbins, endpoint=False)
                    
                    # Plot data
                    ax.bar(angles, mean_amp1, width=width, alpha=0.5, label=cond1_name)
                    ax.bar(angles, mean_amp2, width=width, alpha=0.5, label=cond2_name)
                    
                    # Add preferred phase markers
                    ax.plot([theta1, theta1], [0, np.max(mean_amp1)*1.2], 'r-', linewidth=2)
                    ax.plot([theta2, theta2], [0, np.max(mean_amp2)*1.2], 'b-', linewidth=2)
                    
                    # Add labels and title
                    ax.set_title(f'PAC Comparison\n{cond1_name} vs {cond2_name}\nF={F:.2f}, p={p:.4f}')
                    ax.set_theta_zero_location('N')  # 0 at the top
                    ax.set_theta_direction(-1)  # clockwise
                    
                    # Add legend
                    ax.legend()
                    
                    # Save figure
                    fig_file = f"{out_dir}/pac_comparison_{cond1_name}_vs_{cond2_name}.png"
                    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                    
                    logger.info(f"Saved comparison plot to {fig_file}")
                    
                    return {
                        'condition1': cond1_name,
                        'condition2': cond2_name,
                        'theta1': theta1,
                        'theta2': theta2,
                        'r1': r1,
                        'r2': r2,
                        'F': F,
                        'p': p,
                        'significant': p < alpha,
                        'output_file': output_file,
                        'fig_file': fig_file
                    }
                
                except Exception as e:
                    logger.error(f"Error performing Watson-Williams test: {e}", exc_info=True)
                    return None
            
            elif test_type == 'permutation':
                # Implement permutation test for PAC comparison
                logger.error("Permutation test not implemented yet")
                return None
            
            else:
                logger.error(f"Unknown test type: {test_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error comparing conditions: {e}", exc_info=True)
            return None
    
    def export_pac_parameters_to_csv(self, json_dir=None, csv_file=None, 
                                channels=None, stages=None, 
                                phase_freq=None, amp_freq=None, append=True,
                                method_info=None, out_dir=None):
        """
        Export PAC parameters from tracking to a CSV file.
        
        Parameters
        ----------
        json_dir : str
            Directory containing JSON files or individual channel CSV files
        csv_file : str
            Output CSV file
        channels : list
            List of channels to include
        stages : list
            List of sleep stages to include
        phase_freq : tuple
            Phase frequency range
        amp_freq : tuple
            Amplitude frequency range
        append : bool
            If True, append to existing CSV file by channel rather than overwrite
        method_info : dict
            Dictionary containing method information (sw_method, spindle_method)
        out_dir : str
            Base output directory to use
        
        Returns
        -------
        dict
            Dictionary containing export results
        """
        logger = self.logger
        
        # First, determine the base directory
        base_dir = out_dir if out_dir else json_dir
        if base_dir is None:
            base_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
        
        # Create method-specific directory path
        method_dir = base_dir
        if method_info:
            sw_method = method_info.get('sw_method', 'unknown')
            spindle_method = method_info.get('spindle_method', 'unknown')
            event_type = method_info.get('event_type', 'unknown')
            stage = method_info.get('stage', 'all')
            
            # Create stage string
            stage_str = ''.join(stage) if isinstance(stage, list) else str(stage)
            
            # Determine method directory
            if event_type == 'slow_wave' and method_info.get('pair_with_spindles', False):
                method_dir_name = f"{sw_method}_paired_{spindle_method}"
            else:
                method_dir_name = sw_method if event_type == 'slow_wave' else spindle_method
            
            # Create full method directory path
            method_dir = os.path.join(base_dir, method_dir_name, stage_str)
        
        # Ensure directory exists
        os.makedirs(method_dir, exist_ok=True)
        logger.info(f"Using method directory: {method_dir}")
        
        # Create frequency string for filename
        freq_str = ""
        if phase_freq and amp_freq:
            ph_str = f"{phase_freq[0]}-{phase_freq[1]}Hz"
            amp_str = f"{amp_freq[0]}-{amp_freq[1]}Hz"
            freq_str = f"{ph_str}_{amp_str}"
        
        # Determine output CSV file
        if csv_file is None:
            if freq_str:
                csv_file = os.path.join(method_dir, f"pac_summary_{phase_freq[0]}-{phase_freq[1]}Hz_{amp_freq[0]}-{amp_freq[1]}Hz.csv")
            else:
                csv_file = os.path.join(method_dir, "pac_summary.csv")
        
        logger.info(f"Output summary CSV file: {csv_file}")
        
        # First approach: Look for individual channel result files
        # For PAC data, we need to look for files with pattern:
        # E*_slowwave_spindle_coupling_pha-FREQ-fixed_amp-FREQ-fixed_pac_parameters.csv
        
        if method_info and method_info.get('pair_with_spindles', False):
            # For SW-Spindle coupling
            file_pattern = f"*_slowwave_spindle_coupling_pha-{phase_freq[0]}-{phase_freq[1]}Hz-fixed_amp-{amp_freq[0]}-{amp_freq[1]}Hz-fixed_pac_parameters.csv"
        else:
            # For other coupling types
            file_pattern = f"*_pha-{phase_freq[0]}-{phase_freq[1]}Hz-fixed_amp-{amp_freq[0]}-{amp_freq[1]}Hz-fixed_pac_parameters.csv"
        
        # Find all matching channel CSV files
        channel_files = []
        try:
            import glob
            channel_files = glob.glob(os.path.join(method_dir, file_pattern))
            logger.info(f"Found {len(channel_files)} individual channel PAC parameter files")
        except Exception as e:
            logger.error(f"Error finding channel files: {e}")
        
        # If we found individual channel files, use them to build the summary
        if channel_files:
            try:
                import pandas as pd
                # Store all channel data
                all_data = []
                
                # Process each file
                for file in channel_files:
                    try:
                        # Extract channel name from filename
                        filename = os.path.basename(file)
                        channel = filename.split('_')[0]  # Assuming format: E101_slowwave_...
                        
                        # Read channel data
                        df = pd.read_csv(file)
                        if not df.empty:
                            # Add channel data to combined list
                            for _, row in df.iterrows():
                                # Create data row
                                data_row = {
                                    'Channel': channel,
                                    'Phase_Freq': f"{phase_freq[0]}-{phase_freq[1]}",
                                    'Amp_Freq': f"{amp_freq[0]}-{amp_freq[1]}",
                                }
                                
                                # Copy relevant metrics
                                metric_cols = [ 'mi_raw', 'mi_norm', 'median_mi_pval', 
                                                'preferred_phase_rad', 'preferred_phase_deg', 
                                                'mean_vector_length', 'rho', 'rayleigh_z', 'rayleigh_p'
                                ]
        
                                for col in metric_cols:
                                    if col in row:
                                        data_row[col] = row[col]
                                
                                all_data.append(data_row)
                                
                        logger.debug(f"Processed data from {file}")
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                
                # Create summary dataframe
                if all_data:
                    summary_df = pd.DataFrame(all_data)
                    
                    # Check if we should append to existing file
                    if append and os.path.exists(csv_file):
                        # Read existing data
                        try:
                            existing_df = pd.read_csv(csv_file)
                            # Create set of existing channels
                            existing_channels = set()
                            if 'Channel' in existing_df.columns:
                                for _, row in existing_df.iterrows():
                                    ch = row['Channel']
                                    ph_freq = row['Phase_Freq'] if 'Phase_Freq' in row else ""
                                    amp_freq = row['Amp_Freq'] if 'Amp_Freq' in row else ""
                                    existing_channels.add(f"{ch}_{ph_freq}_{amp_freq}")
                            
                            # Filter out channels that already exist
                            new_data = []
                            for row in all_data:
                                ch = row['Channel']
                                ph_freq = row['Phase_Freq']
                                amp_freq = row['Amp_Freq']
                                key = f"{ch}_{ph_freq}_{amp_freq}"
                                if key not in existing_channels:
                                    new_data.append(row)
                            
                            # Append new data to existing data
                            if new_data:
                                new_df = pd.DataFrame(new_data)
                                summary_df = pd.concat([existing_df, new_df])
                                logger.info(f"Appending {len(new_data)} new channels to existing summary")
                            else:
                                summary_df = existing_df
                                logger.info("No new data to append")
                        except Exception as e:
                            logger.error(f"Error appending to existing file: {e}, creating new file")
                    
                    # Write summary to CSV
                    summary_df.to_csv(csv_file, index=False)
                    logger.info(f"Exported PAC summary to {csv_file} with {len(summary_df)} entries")
                    
                    return {
                        'file': csv_file, 
                        'channels': len(summary_df['Channel'].unique()),
                        'rows': len(summary_df)
                    }
                else:
                    logger.warning("No PAC data to export")
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating summary from files: {e}", exc_info=True)
        
        # Second approach: Use tracking data if available and no files were found
        elif 'event_pac' in self.tracking and self.tracking['event_pac']:
            try:
                # Filter channels if specified
                if channels is None:
                    channels = list(self.tracking['event_pac'].keys())
                else:
                    channels = [ch for ch in channels if ch in self.tracking['event_pac']]
                
                # Create key based on frequency bands
                key = None
                if phase_freq and amp_freq:
                    key = f"{phase_freq[0]}-{phase_freq[1]}Hz_{amp_freq[0]}-{amp_freq[1]}Hz"
                
                # Read existing data if appending
                existing_data = {}
                if append and os.path.exists(csv_file):
                    try:
                        import pandas as pd
                        # Read existing CSV into DataFrame
                        existing_df = pd.read_csv(csv_file)
                        logger.info(f"Read {len(existing_df)} existing entries from {csv_file}")
                        
                        # Convert DataFrame to dictionary keyed by channel
                        for _, row in existing_df.iterrows():
                            ch = row['Channel']
                            if ch not in existing_data:
                                existing_data[ch] = {}
                            
                            # Create frequency key from Phase_Freq and Amp_Freq
                            ph_freq = row['Phase_Freq'] if 'Phase_Freq' in row else ""
                            amp_freq = row['Amp_Freq'] if 'Amp_Freq' in row else ""
                            freq_key = f"{ph_freq}_{amp_freq}"
                            
                            # Store row data
                            existing_data[ch][freq_key] = row.to_dict()
                            
                    except Exception as e:
                        logger.warning(f"Could not read existing CSV for appending: {e}")
                        existing_data = {}
                
                # Prepare data for export
                data = []
                for ch in channels:
                    if ch not in self.tracking['event_pac']:
                        continue
                    
                    ch_results = self.tracking['event_pac'][ch]
                    
                    if key and key in ch_results:
                        # Use specific frequency key
                        results = ch_results[key]
                        # Check if already in existing data
                        skip_channel = False
                        if append and ch in existing_data:
                            for ex_key, ex_data in existing_data[ch].items():
                                # See if there's a matching frequency entry
                                if ex_key.startswith(f"{phase_freq[0]}-{phase_freq[1]}") and \
                                ex_key.endswith(f"{amp_freq[0]}-{amp_freq[1]}"):
                                    # Check if existing has more segments
                                    if ex_data.get('N_Segments', 0) > results.get('n_segments', 0):
                                        logger.info(f"Skipping {ch}/{key}: existing has more segments")
                                        data.append(ex_data)
                                        skip_channel = True
                                        break
                        
                        if not skip_channel:
                            data.append({
                                'Channel': ch,
                                'Phase_Freq': f"{phase_freq[0]}-{phase_freq[1]}",
                                'Amp_Freq': f"{amp_freq[0]}-{amp_freq[1]}",
                                'MI': results.get('mi_norm', float('nan')),
                                'MI_pval': results.get('pval', float('nan')),
                                'PP_rad': results.get('preferred_phase_rad', float('nan')),
                                'PP_degrees': results.get('preferred_phase_deg', float('nan')),
                                'Mean_vector_length': results.get('mean_vector_length', float('nan')),
                                'rho': results.get('rho', float('nan')),
                                'Rayleigh_z': results.get('rayleigh_z', float('nan')),
                                'Rayleigh_p': results.get('rayleigh_p', float('nan')),
                                'N_Segments': results.get('n_segments', 0)
                            })
                    else:
                        # Export all frequency combinations
                        for freq_key, results in ch_results.items():
                            try:
                                # Parse frequency ranges from key
                                freq_parts = freq_key.split('_')
                                ph_freq = freq_parts[0]
                                amp_freq = freq_parts[1]
                                
                                # Check if already in existing data
                                skip_entry = False
                                if append and ch in existing_data:
                                    for ex_key, ex_data in existing_data[ch].items():
                                        if ex_key == freq_key:
                                            # Check if existing has more segments
                                            if ex_data.get('N_Segments', 0) > results.get('n_segments', 0):
                                                logger.info(f"Skipping {ch}/{freq_key}: existing has more segments")
                                                data.append(ex_data)
                                                skip_entry = True
                                                break
                                
                                if not skip_entry:
                                    data.append({
                                        'Channel': ch,
                                        'Phase_Freq': ph_freq,
                                        'Amp_Freq': amp_freq,
                                        'MI': results.get('mi_norm', float('nan')),
                                        'MI_pval': results.get('pval', float('nan')),
                                        'PP_rad': results.get('preferred_phase_rad', float('nan')),
                                        'PP_degrees': results.get('preferred_phase_deg', float('nan')),
                                        'Mean_vector_length': results.get('mean_vector_length', float('nan')),
                                        'rho': results.get('rho', float('nan')),
                                        'Rayleigh_z': results.get('rayleigh_z', float('nan')),
                                        'Rayleigh_p': results.get('rayleigh_p', float('nan')),
                                        'N_Segments': results.get('n_segments', 0)
                                    })
                            except Exception as e:
                                logger.warning(f"Could not parse frequency key: {freq_key} - {e}")
                
                # Create DataFrame and export to CSV
                if data:
                    import pandas as pd
                    df = pd.DataFrame(data)
                    
                    # If append and file exists, merge with existing data
                    if append and os.path.exists(csv_file):
                        try:
                            existing_df = pd.read_csv(csv_file)
                            # Only keep rows from existing_df that aren't already in our new data
                            combined_df = pd.concat([existing_df, df]).drop_duplicates(
                                subset=['Channel', 'Phase_Freq', 'Amp_Freq'], 
                                keep='last'
                            )
                            combined_df.to_csv(csv_file, index=False)
                            logger.info(f"Appended to existing CSV: {len(df)} new rows, {len(combined_df)} total rows")
                        except Exception as e:
                            logger.error(f"Error appending to existing CSV: {e}")
                            df.to_csv(csv_file, index=False)
                            logger.info(f"Created new CSV with {len(df)} rows")
                    else:
                        df.to_csv(csv_file, index=False)
                        logger.info(f"Created new CSV with {len(df)} rows")
                    
                    return {'file': csv_file, 'channels': len(channels), 'rows': len(data)}
                else:
                    logger.warning("No PAC data to export")
                    return None
            except Exception as e:
                logger.error(f"Error exporting PAC parameters from tracking: {e}", exc_info=True)
                return None
        else:
            logger.warning("No PAC results in tracking dictionary or individual files")
            return None

    # ------------------------------------------------------------------ #
    # PAC -> SQLite (pac_coupling table)
    # ------------------------------------------------------------------ #

    # Natural-key primary key (subject included to future-proof a merged DB).
    _PAC_PK_COLS = ['subject', 'channel', 'event_type', 'method', 'stage',
                    'phase_freq_lower', 'phase_freq_upper',
                    'amp_freq_lower', 'amp_freq_upper']
    _PAC_METRIC_COLS = ['mi_raw', 'mi_norm', 'median_mi_pval',
                        'preferred_phase_rad', 'preferred_phase_deg',
                        'mean_vector_length', 'rho', 'rayleigh_z', 'rayleigh_p']
    _PAC_PROV_COLS = ['n_events', 'idpac', 'ref_chan', 'invert',
                      'turtlewave_version', 'processing_timestamp', 'source_path']
    _PAC_ALL_COLS = _PAC_PK_COLS + _PAC_METRIC_COLS + _PAC_PROV_COLS
    _PAC_COLUMN_ALIASES = PAC_COLUMN_ALIASES

    def _init_pac_table(self, conn):
        """
        Create the ``pac_coupling`` table and its indexes if absent.

        Delegates to :func:`turtlewave_hdEEG.dbwrite.ensure_pac_schema`, which
        owns the DDL. The same function is called from
        ``dbwrite.ensure_direct_write_schema``, so the table exists on any
        database a detector has touched even when PAC has never been run and
        a reader never faces a missing table.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open connection to the target database. The caller owns the
            connection lifecycle (commit and close).

        Returns
        -------
        None
        """
        from turtlewave_hdEEG import dbwrite
        dbwrite.ensure_pac_schema(conn)

    def _pac_version(self):
        """
        Return the installed ``turtlewave_hdEEG`` version string.

        Returns
        -------
        str
            The package ``__version__``, or ``'unknown'`` if it cannot be read.
        """
        try:
            import turtlewave_hdEEG
            return getattr(turtlewave_hdEEG, '__version__', 'unknown')
        except Exception:
            return 'unknown'

    @staticmethod
    def _fmt_ref_chan(ref_chan):
        """
        Normalise a reference-channel specification to a text value for storage.

        Parameters
        ----------
        ref_chan : None, str, or list
            Reference channel(s).

        Returns
        -------
        str or None
            Comma-joined channel list, the string itself, or None.
        """
        if ref_chan is None:
            return None
        if isinstance(ref_chan, (list, tuple)):
            if len(ref_chan) == 0:
                return None
            return ','.join(str(c) for c in ref_chan)
        return str(ref_chan)

    @staticmethod
    def _nan_to_none(v):
        """
        Coerce a value to a Python float, mapping NaN/inf/None to None.

        Ensures numpy scalar types are converted to native Python types that
        the ``sqlite3`` driver can bind, and that NaN becomes SQL NULL.

        Parameters
        ----------
        v : Any
            Candidate metric value.

        Returns
        -------
        float or None
            Finite Python float, or None for NaN/inf/None/non-numeric.
        """
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        return f

    def _resolve_pac_row(self, row):
        """
        Map a source row onto the canonical PAC metric column set.

        Both ingest paths (live tracking dict and file-based CSV, in either the
        per-channel or the ``pac_summary`` layout) go through this single
        resolver so no metric is silently dropped when its column name differs.

        Parameters
        ----------
        row : dict
            A row keyed by any of the recognised aliases in
            :data:`PAC_COLUMN_ALIASES`.

        Returns
        -------
        dict
            Canonical column name -> value (or None if no alias present/finite).
        """
        out = {}
        for canon, aliases in self._PAC_COLUMN_ALIASES.items():
            val = None
            for a in aliases:
                if a not in row:
                    continue
                v = row[a]
                if v is None:
                    continue
                # Skip NaN (both Python and numpy floats).
                try:
                    if isinstance(v, float) and math.isnan(v):
                        continue
                except Exception:
                    pass
                try:
                    if isinstance(v, np.floating) and np.isnan(v):
                        continue
                except Exception:
                    pass
                val = v
                break
            out[canon] = val
        return out

    def _upsert_pac_row(self, conn, row):
        """
        Insert or replace a single row in ``pac_coupling`` on the natural key.

        Metric columns are passed through :meth:`_nan_to_none` (NaN -> SQL
        NULL). Existence is checked first so the caller can distinguish an
        insert from a replace for its added/updated tally.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open connection (caller commits).
        row : dict
            Dict providing every column in ``_PAC_ALL_COLS``.

        Returns
        -------
        str
            ``'updated'`` if a row with this natural key already existed,
            else ``'added'``.
        """
        cursor = conn.cursor()
        pk_vals = [row[c] for c in self._PAC_PK_COLS]
        cursor.execute(
            "SELECT 1 FROM pac_coupling WHERE " +
            " AND ".join(f"{c} = ?" for c in self._PAC_PK_COLS),
            pk_vals)
        exists = cursor.fetchone() is not None

        vals = []
        for c in self._PAC_ALL_COLS:
            v = row.get(c)
            if c in self._PAC_METRIC_COLS:
                v = self._nan_to_none(v)
            vals.append(v)

        cursor.execute(
            f"INSERT OR REPLACE INTO pac_coupling "
            f"({', '.join(self._PAC_ALL_COLS)}) "
            f"VALUES ({', '.join(['?'] * len(self._PAC_ALL_COLS))})",
            vals)
        return 'updated' if exists else 'added'

    def store_pac_to_database(self, db_path, subject, event_type, method, stage,
                              phase_freq, amp_freq, idpac, ref_chan=None,
                              invert=False, results=None):
        """
        Write in-memory per-channel PAC results directly to ``pac_coupling``.

        This is the live path used by :meth:`analyze_pac` (``write_db=True``).
        ``n_events`` is taken from the in-memory ``n_segments`` (always present
        on the live path); a channel whose event count is missing or
        non-positive is rejected and flagged for re-run rather than stored as
        0/NULL. Writes are idempotent (``INSERT OR REPLACE`` on the natural
        key).

        Parameters
        ----------
        db_path : str
            Path to the SQLite database.
        subject : str
            Subject identifier (part of the primary key).
        event_type : str
            Stored event type ('slow_wave', 'spindle', or 'sw_spindle').
        method : str
            Method / pairing token (mirrors the results directory name).
        stage : str
            Combined stage string (e.g. 'NREM2NREM3').
        phase_freq : tuple of float
            (lower, upper) phase frequency bounds in Hz.
        amp_freq : tuple of float
            (lower, upper) amplitude frequency bounds in Hz.
        idpac : tuple
            Tensorpac (method, surrogate, correction) triple, stored as text.
        ref_chan : None, str, or list, optional
            Reference channel(s) actually used.
        invert : bool, optional
            Polarity-inversion flag actually used.
        results : dict or None, optional
            Nested ``{channel: {freq_key: metrics}}`` mapping. Defaults to
            ``self.tracking['event_pac']``.

        Returns
        -------
        dict
            ``{'added': int, 'updated': int, 'skipped': int}``.
        """
        import sqlite3
        logger = self.logger

        if results is None:
            results = self.tracking.get('event_pac', {})

        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        if not results:
            logger.warning("store_pac_to_database: no in-memory PAC results to store")
            return stats

        key = f"{phase_freq[0]}-{phase_freq[1]}Hz_{amp_freq[0]}-{amp_freq[1]}Hz"
        version = self._pac_version()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = None
        try:
            # Writer: routed through dbwrite so it gets the 60 s busy timeout
            # and the preserve-existing-journal-mode behaviour.
            conn = dbwrite.open_write_connection(db_path, logger=logger)
            self._init_pac_table(conn)

            for ch, ch_results in results.items():
                if not isinstance(ch_results, dict):
                    continue

                if key in ch_results:
                    entry = ch_results[key]
                elif len(ch_results) == 1:
                    entry = next(iter(ch_results.values()))
                else:
                    logger.warning(
                        f"store_pac_to_database: channel {ch} has no entry for "
                        f"{key} and multiple freq entries; skipping")
                    stats['skipped'] += 1
                    continue

                if not isinstance(entry, dict):
                    stats['skipped'] += 1
                    continue

                n_events = entry.get('n_segments')
                if (n_events is None or
                        not np.isfinite(float(n_events)) or float(n_events) <= 0):
                    logger.error(
                        f"store_pac_to_database: channel {ch} has unrecoverable "
                        f"n_events ({n_events}); rejecting row and flagging for "
                        f"re-run (NOT stored as 0/NULL)")
                    stats['skipped'] += 1
                    continue

                row = {
                    'subject': str(subject),
                    'channel': str(ch),
                    'event_type': str(event_type),
                    'method': str(method),
                    'stage': str(stage),
                    'phase_freq_lower': float(phase_freq[0]),
                    'phase_freq_upper': float(phase_freq[1]),
                    'amp_freq_lower': float(amp_freq[0]),
                    'amp_freq_upper': float(amp_freq[1]),
                    'mi_raw': entry.get('mi_raw'),
                    'mi_norm': entry.get('mi_norm'),
                    'median_mi_pval': entry.get('pval'),
                    'preferred_phase_rad': entry.get('preferred_phase_rad'),
                    'preferred_phase_deg': entry.get('preferred_phase_deg'),
                    'mean_vector_length': entry.get('mean_vector_length'),
                    'rho': entry.get('rho'),
                    'rayleigh_z': entry.get('rayleigh_z'),
                    'rayleigh_p': entry.get('rayleigh_p'),
                    'n_events': int(n_events),
                    'idpac': str(tuple(idpac)) if idpac is not None else None,
                    'ref_chan': self._fmt_ref_chan(ref_chan),
                    'invert': int(bool(invert)) if invert is not None else None,
                    'turtlewave_version': version,
                    'processing_timestamp': timestamp,
                    'source_path': entry.get('outputfile'),
                }

                outcome = self._upsert_pac_row(conn, row)
                stats[outcome] += 1

            conn.commit()
            logger.info(
                f"store_pac_to_database: {stats['added']} added, "
                f"{stats['updated']} updated, {stats['skipped']} skipped")
        except Exception as e:
            # Re-raise. Returning the untouched stats dict here made a failed
            # database write indistinguishable from a run that stored nothing
            # because there was nothing to store -- the same silent-loss shape
            # as the event importers.
            logger.error(f"store_pac_to_database failed: {e}", exc_info=True)
            raise
        finally:
            if conn is not None:
                conn.close()

        stats['ok'] = True
        return stats

    def _infer_pac_context_from_path(self, csv_path):
        """
        Infer PAC context (method, stage, event type, channel, freqs) from path.

        Uses the results directory layout
        ``.../<method_dir>/<stage_str>/<channel>_<event>_pha-..-..Hz_amp-..-..Hz_pac_parameters.csv``
        rather than a lossy ``split('_')``: the method and stage come from the
        parent directory names, the event type from the filename token
        (``slowwave_spindle_coupling`` -> 'sw_spindle'), and the frequency
        bounds from the ``pha-``/``amp-`` tokens via regex.

        Parameters
        ----------
        csv_path : str
            Path to a per-channel PAC parameters CSV.

        Returns
        -------
        dict
            Keys: ``method``, ``stage``, ``event_type``, ``channel``,
            ``phase_freq`` (tuple or None), ``amp_freq`` (tuple or None).
        """
        fname = os.path.basename(csv_path)
        stage = os.path.basename(os.path.dirname(csv_path)) or None
        method = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or None

        # Split the channel off at the FIRST recognized event-type token, not
        # the first underscore: channel labels may themselves contain
        # underscores (e.g. 'E_101'), so `fname.split('_')[0]` would truncate
        # 'E_101' to 'E' and collide the natural keys of distinct channels.
        # The coupling token is checked first because it embeds 'spindle'.
        event_type = None
        channel = None
        for token, etype in (('slowwave_spindle_coupling', 'sw_spindle'),
                             ('slow_wave', 'slow_wave'),
                             ('spindle', 'spindle')):
            idx = fname.find(f'_{token}_')
            if idx > 0:
                event_type = etype
                channel = fname[:idx]
                break

        phase_freq = amp_freq = None
        m = re.search(r'pha-([0-9.]+)-([0-9.]+)Hz.*?amp-([0-9.]+)-([0-9.]+)Hz', fname)
        if m:
            phase_freq = (float(m.group(1)), float(m.group(2)))
            amp_freq = (float(m.group(3)), float(m.group(4)))

        return {'method': method, 'stage': stage, 'event_type': event_type,
                'channel': channel, 'phase_freq': phase_freq, 'amp_freq': amp_freq}

    def import_pac_csv_to_database(self, csv_path, db_path, subject,
                                   event_type=None, method=None, stage=None,
                                   phase_freq=None, amp_freq=None, idpac=None,
                                   ref_chan=None, invert=None, channel=None):
        """
        Back-fill one existing PAC CSV into ``pac_coupling``.

        Context (method, stage, event type, frequency bounds) is inferred from
        the path components (see :meth:`_infer_pac_context_from_path`); explicit
        caller arguments override the inference. ``n_events`` is recovered from
        the sibling ``*_mean_amps.npy`` (``shape[0]``) for per-channel CSVs, or
        from an ``N_Segments``/``n_events`` column for ``pac_summary`` CSVs. A
        row whose event count cannot be recovered is rejected, counted under
        ``n_events_missing`` and logged as flag-for-re-run; it is NOT inserted.

        The preferred-phase value stored is exactly what the CSV holds; the
        historical pre-180-degree-fix migration of old cluster outputs is a
        separate concern.

        Parameters
        ----------
        csv_path : str
            Path to the PAC CSV (per-channel ``*_pac_parameters.csv`` preferred).
        db_path : str
            Path to the SQLite database.
        subject : str
            Subject identifier (part of the primary key).
        event_type, method, stage : str, optional
            Override the path-inferred values.
        phase_freq, amp_freq : tuple of float, optional
            Override the path-inferred (lower, upper) frequency bounds.
        idpac : tuple, optional
            Tensorpac (method, surrogate, correction) triple, stored as text.
        ref_chan : None, str, or list, optional
            Reference channel(s) used.
        invert : bool or None, optional
            Polarity-inversion flag used (None -> stored NULL).
        channel : str or None, optional
            Explicit channel label for a per-channel CSV, overriding the
            filename-inferred value (ignored for ``pac_summary`` CSVs, which
            carry a per-row Channel column). Provide this when the channel
            cannot be unambiguously parsed from the filename.

        Returns
        -------
        dict
            ``{'added': int, 'updated': int, 'skipped': int,
            'n_events_missing': int}``. A row whose channel cannot be
            unambiguously resolved is counted under ``skipped`` and logged,
            never stored with a truncated/ambiguous channel.
        """
        import sqlite3
        logger = self.logger

        stats = {'added': 0, 'updated': 0, 'skipped': 0, 'n_events_missing': 0}

        if not os.path.exists(csv_path):
            logger.error(f"PAC CSV not found: {csv_path}")
            stats['skipped'] += 1
            return stats

        inferred = self._infer_pac_context_from_path(csv_path)
        method = method or inferred['method']
        stage = stage or inferred['stage']
        event_type = event_type or inferred['event_type']
        if phase_freq is None:
            phase_freq = inferred['phase_freq']
        if amp_freq is None:
            amp_freq = inferred['amp_freq']

        fname = os.path.basename(csv_path)
        is_summary = fname.startswith('pac_summary')

        if phase_freq is None or amp_freq is None:
            logger.error(
                f"Cannot resolve phase/amp frequency for {fname}; skipping. "
                f"Pass phase_freq=/amp_freq= explicitly.")
            stats['skipped'] += 1
            return stats
        if event_type is None or method is None or stage is None:
            logger.error(
                f"Cannot resolve event_type/method/stage for {fname} "
                f"(event_type={event_type}, method={method}, stage={stage}); "
                f"skipping. Pass them explicitly.")
            stats['skipped'] += 1
            return stats

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read {csv_path}: {e}")
            stats['skipped'] += 1
            return stats

        if df.empty:
            logger.warning(f"Empty PAC CSV: {csv_path}")
            return stats

        # Recover event count from the sibling mean-amplitude array (per-channel).
        npy_n_events = None
        npy_path = csv_path.replace('_pac_parameters.csv', '_mean_amps.npy')
        if not is_summary and os.path.exists(npy_path):
            try:
                npy_n_events = int(np.load(npy_path, mmap_mode='r').shape[0])
            except Exception as e:
                logger.warning(f"Could not read event count from {npy_path}: {e}")

        # Channel column present only in the tracking-derived summary layout.
        chan_col = None
        for cand in ('Channel', 'channel'):
            if cand in df.columns:
                chan_col = cand
                break

        version = self._pac_version()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = None
        try:
            # Writer: routed through dbwrite so it gets the 60 s busy timeout
            # and the preserve-existing-journal-mode behaviour.
            conn = dbwrite.open_write_connection(db_path, logger=logger)
            self._init_pac_table(conn)

            for _, r in df.iterrows():
                row_dict = r.to_dict()
                metrics = self._resolve_pac_row(row_dict)

                # Resolve the channel label. Summary CSVs carry a per-row
                # Channel column (authoritative). Per-channel CSVs use the
                # explicit `channel` arg if given, else the token-boundary
                # parse. A missing/blank channel is ambiguous -> skip-with-log
                # rather than store a truncated/colliding key.
                if chan_col:
                    row_channel = row_dict[chan_col]
                elif channel is not None:
                    row_channel = channel
                else:
                    row_channel = inferred['channel']

                if row_channel is None or str(row_channel).strip() == '':
                    logger.warning(
                        f"Cannot infer channel from filename {fname}; pass "
                        f"channel= explicitly or add a Channel column. Skipping "
                        f"row (NOT stored to avoid an ambiguous/colliding key).")
                    stats['skipped'] += 1
                    continue
                row_channel = str(row_channel)

                # n_events: prefer explicit column, else sibling npy.
                n_events = metrics.get('n_events')
                if n_events is None and npy_n_events is not None:
                    n_events = npy_n_events

                try:
                    n_ok = (n_events is not None and
                            np.isfinite(float(n_events)) and float(n_events) > 0)
                except (TypeError, ValueError):
                    n_ok = False
                if not n_ok:
                    logger.error(
                        f"n_events unrecoverable for {fname} (channel {row_channel}); "
                        f"rejecting row and flagging for re-run "
                        f"(need sibling {os.path.basename(npy_path)} or an "
                        f"N_Segments column). NOT stored as 0/NULL.")
                    stats['n_events_missing'] += 1
                    continue

                row = {
                    'subject': str(subject),
                    'channel': row_channel,
                    'event_type': str(event_type),
                    'method': str(method),
                    'stage': str(stage),
                    'phase_freq_lower': float(phase_freq[0]),
                    'phase_freq_upper': float(phase_freq[1]),
                    'amp_freq_lower': float(amp_freq[0]),
                    'amp_freq_upper': float(amp_freq[1]),
                    'mi_raw': metrics.get('mi_raw'),
                    'mi_norm': metrics.get('mi_norm'),
                    'median_mi_pval': metrics.get('median_mi_pval'),
                    'preferred_phase_rad': metrics.get('preferred_phase_rad'),
                    'preferred_phase_deg': metrics.get('preferred_phase_deg'),
                    'mean_vector_length': metrics.get('mean_vector_length'),
                    'rho': metrics.get('rho'),
                    'rayleigh_z': metrics.get('rayleigh_z'),
                    'rayleigh_p': metrics.get('rayleigh_p'),
                    'n_events': int(float(n_events)),
                    'idpac': str(tuple(idpac)) if idpac is not None else None,
                    'ref_chan': self._fmt_ref_chan(ref_chan),
                    'invert': int(bool(invert)) if invert is not None else None,
                    'turtlewave_version': version,
                    'processing_timestamp': timestamp,
                    'source_path': csv_path,
                }

                outcome = self._upsert_pac_row(conn, row)
                stats[outcome] += 1

            conn.commit()
            logger.info(
                f"import_pac_csv_to_database [{fname}]: {stats['added']} added, "
                f"{stats['updated']} updated, {stats['skipped']} skipped, "
                f"{stats['n_events_missing']} rejected (n_events missing)")
        except Exception as e:
            # Re-raise for the same reason as store_pac_to_database. The batch
            # walker (backfill_pac_directory) catches per file so one bad CSV
            # still does not abort a whole back-fill.
            logger.error(f"import_pac_csv_to_database failed for {csv_path}: {e}", exc_info=True)
            raise
        finally:
            if conn is not None:
                conn.close()

        stats['ok'] = True
        return stats

    def backfill_pac_directory(self, root_dir, db_path, subject_from='folder'):
        """
        Walk a PAC results directory and back-fill every per-channel CSV.

        The subject is the basename of ``root_dir`` when ``subject_from ==
        'folder'`` (logged), otherwise the literal ``subject_from`` value.
        ``pac_summary_*`` files are skipped whenever per-channel CSVs are
        present in the same directory (the per-channel files are the source of
        truth). Idempotent: re-running reports ``added=0`` for rows already
        stored.

        Parameters
        ----------
        root_dir : str
            Root of a subject's PAC results tree.
        db_path : str
            Path to the SQLite database.
        subject_from : str, optional
            ``'folder'`` (default) to use the ``root_dir`` basename as subject,
            or any other string to use it literally as the subject id.

        Returns
        -------
        dict
            ``{'files': int, 'added': int, 'updated': int, 'skipped': int,
            'n_events_missing': int, 'failed': int}``. ``failed`` counts CSVs
            whose import raised; those are logged and skipped rather than
            aborting the walk.
        """
        logger = self.logger

        if subject_from == 'folder':
            # Shared resolver, so a folder-derived subject keys the database
            # the same way every other path does ('sub-10sd', not '10sd').
            subject = derive_subject(root_dir=root_dir)
        else:
            subject = str(subject_from)
        logger.info(f"Back-filling PAC results under {root_dir} for subject '{subject}'")

        totals = {'files': 0, 'added': 0, 'updated': 0,
                  'skipped': 0, 'n_events_missing': 0, 'failed': 0}

        for dirpath, _dirs, filenames in os.walk(root_dir):
            per_chan = sorted(
                f for f in filenames
                if f.endswith('_pac_parameters.csv') and not f.startswith('pac_summary'))
            summaries = [f for f in filenames if f.startswith('pac_summary')]

            if per_chan and summaries:
                logger.info(
                    f"Skipping {len(summaries)} pac_summary file(s) in {dirpath}; "
                    f"using {len(per_chan)} per-channel CSV(s)")
            if not per_chan and summaries:
                logger.warning(
                    f"No per-channel CSVs in {dirpath}; {len(summaries)} "
                    f"pac_summary file(s) present but not back-filled "
                    f"(per-channel files are the source of truth)")

            for f in per_chan:
                csv_path = os.path.join(dirpath, f)
                # Defensive: resolve the channel here too (token-boundary parse)
                # and pass it explicitly, so the importer is not the sole line
                # of defense against underscore-containing channel labels.
                ctx = self._infer_pac_context_from_path(csv_path)
                totals['files'] += 1
                # import_pac_csv_to_database now raises on failure; catch per
                # file so one unreadable CSV does not abort a whole back-fill,
                # and count it so the caller sees a non-zero failure tally
                # instead of a silently short run.
                try:
                    r = self.import_pac_csv_to_database(
                        csv_path, db_path, subject, channel=ctx['channel'])
                except Exception as e:
                    totals['failed'] += 1
                    logger.error(f"Back-fill failed for {csv_path}: {e}")
                    continue
                for k in ('added', 'updated', 'skipped', 'n_events_missing'):
                    totals[k] += r.get(k, 0)

        if totals['failed']:
            logger.warning(
                f"PAC back-fill finished with {totals['failed']} failed "
                f"file(s) out of {totals['files']}")
        logger.info(f"PAC back-fill complete: {totals}")
        return totals

