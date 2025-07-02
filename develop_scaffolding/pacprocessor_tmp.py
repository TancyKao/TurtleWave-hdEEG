##pacprocessor.py


"""
pac_processor.py
A class for phase-amplitude coupling (PAC) analysis for high-density EEG data.
Based on the OCTOPUS method from the seapipe package.
"""

import os
import sys
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
                db_path=None, out_dir=None, progress=True):
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
            
        Returns
        -------
        dict
            Dictionary containing PAC results
        """
        from tensorpac import Pac
        import sys
        import sqlite3
        
        # Set up logger
        logger = self.logger
        
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
        logger.info("""                    
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
            os.makedirs(out_dir, exist_ok=True)
            logger.info(f"Output directory: {out_dir} created/exists")
        else:
            out_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
            os.makedirs(out_dir, exist_ok=True)
            logger.info(f"Using default output directory: {out_dir}")
        
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
        # Force pair_with_spindles to True if event_type is 'slow_wave'
        if event_type == 'slow_wave':
            pair_with_spindles = True
            logger.info("Forcing slow wave-spindle coupling analysis")
        elif event_type == 'spindle':
            # HERE MODIFY: Skip spindle-only analysis
            logger.info("Spindle-only analysis not supported in this modified version")
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
            
            # Determine output filename
            stagename = '-'.join(stage)
            
            outputfile = f'{out_dir}/{ch}_{stagename}_{event_type}_{freqs}_pac_parameters.csv'
            
            # 6. Fetch data segments
            try:
                logger.info(f"Fetching data segments for {ch}")
                
                if use_detected_events:
                    # Get events from SQLite database
                    logger.info(f"Using detected {event_type} events from database")
                    
                    # Connect to database
                    try:
                        conn = sqlite3.connect(db_path)
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

                            
                            # Execute query
                            cursor.execute(query, params)
                            slow_wave_events = cursor.fetchall()
                            
                            logger.info(f"Found {len(slow_wave_events)} slow wave events for channel {ch}")
                            # Always look for slow wave-spindle pairs
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
                            continue
                        else:
                            logger.error(f"Unknown event type: {event_type}")
                            continue
                        
                        # Close database connection
                        conn.close()
                        
                        if not segments or len(segments) == 0:
                            logger.warning(f"No valid segments created from database events for {ch}")
                            continue
                        
                        logger.info(f"Created {len(segments)} segments for PAC analysis")
                        
                    except Exception as e:
                        logger.error(f"Error accessing database: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    # Use standard fetch for continuous data
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
                
                # Create bins for preferred phase
                vecbin = np.zeros(nbins)
                width = 2 * np.pi / nbins
                for n in range(nbins):
                    vecbin[n] = n * width + width / 2
                
                # Calculate circular statistics
                from scipy.stats import circmean, circvar
                
                # Find bin with max amplitude for each trial
                ab_pk = np.argmax(ab, axis=1)
                
                # Convert to angles
                angles = vecbin[ab_pk]
                
                # Calculate mean direction (theta) & mean vector length (rad)
                theta = circmean(angles)
                theta_deg = np.degrees(theta)
                if theta_deg < 0:
                    theta_deg += 360
                    
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
                
                # Save CFC metrics to dataframe
                d = pd.DataFrame([
                    np.mean(pac.pac), 
                    np.mean(mi), 
                    np.median(mi_pv), 
                    theta, 
                    theta_deg, 
                    rad, 
                    rho, 
                    pv1, 
                    z, 
                    pv2
                ]).transpose()
                
                d.columns = [
                    'mi_raw', 'mi_norm', 'pval1', 
                    'pp_radians', 'ppdegrees', 'mvl',
                    'rho', 'pval2', 'rayl', 'pval3'
                ]
                
                d.to_csv(outputfile, sep=',')
                
                logger.info(f"Saved PAC results to {outputfile}")
                logger.info(f"Saved mean amplitudes to {amp_file}.npy")
                
                # Store results in channel_results
                chan_results = {
                    'mi_raw': float(np.mean(pac.pac)),
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
                logger.error(f"Error processing channel {ch}: {e}")
                import traceback
                traceback.print_exc()
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
        
        return tracking['event_pac']
    
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
        
        # Process stage input
        if isinstance(stage, str):
            stage = [stage]
        
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
            segments = fetch(self.dataset, self.annotations, cat=(1, 1), 
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
                
                comod = pac.filterfit(s_freq, data_array, p_freqs, a_freqs, n_perm=200, 
                                   progress_bar=True, random_state=42)
                
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
            logger.error(f"Error generating comodulogram: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # def compare_conditions(self, condition1, condition2, test_type='watson_williams', 
    #                      alpha=0.05, out_dir=None):
    #     """
    #     Compare PAC between two conditions.
        
    #     Parameters
    #     ----------
    #     condition1 : dict
    #         First condition with keys 'amp_file', 'stage', etc.
    #     condition2 : dict
    #         Second condition with keys 'amp_file', 'stage', etc.
    #     test_type : str
    #         Type of statistical test ('watson_williams' or 'permutation')
    #     alpha : float
    #         Significance level
    #     out_dir : str
    #         Output directory for results
            
    #     Returns
    #     -------
    #     dict
    #         Dictionary containing comparison results
    #     """
    #     logger = self.logger
        
    #     # Set up output directory
    #     if out_dir is None:
    #         out_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
        
    #     os.makedirs(out_dir, exist_ok=True)
        
    #     # Load data from condition files
    #     try:
    #         # Load amplitude data
    #         amp1 = np.load(condition1['amp_file'])
    #         amp2 = np.load(condition2['amp_file'])
            
    #         # Get number of bins
    #         nbins = amp1.shape[1]
            
    #         # Create bins for preferred phase
    #         vecbin = np.zeros(nbins)
    #         width = 2 * np.pi / nbins
    #         for n in range(nbins):
    #             vecbin[n] = n * width + width / 2
            
    #         # Find preferred phase for each trial
    #         ab_pk1 = np.argmax(amp1, axis=1)
    #         ab_pk2 = np.argmax(amp2, axis=1)
            
    #         # Convert to angles
    #         angles1 = vecbin[ab_pk1]
    #         angles2 = vecbin[ab_pk2]
            
    #         # Perform statistical test
    #         if test_type == 'watson_williams':
    #             from scipy.stats import circmean
    #             from pingouin import circ_r
                
    #             # Calculate mean direction for each condition
    #             theta1 = circmean(angles1)
    #             theta2 = circmean(angles2)
                
    #             # Calculate mean vector length for each condition
    #             r1 = circ_r(vecbin, np.histogram(ab_pk1, bins=nbins)[0], d=width)
    #             r2 = circ_r(vecbin, np.histogram(ab_pk2, bins=nbins)[0], d=width)
                
    #             # Perform Watson-Williams test
    #             try:
    #                 from pingouin import circ_wwtest
                    
    #                 # Run Watson-Williams test
    #                 F, p = circ_wwtest(angles1, angles2, np.ones(angles1.shape), np.ones(angles2.shape))
                    
    #                 # Save results
    #                 cond1_name = condition1.get('name', 'Condition1')
    #                 cond2_name = condition2.get('name', 'Condition2')
                    
    #                 output_file = f"{out_dir}/pac_comparison_{cond1_name}_vs_{cond2_name}.csv"
                    
    #                 results_df = pd.DataFrame({
    #                     'Condition1': [cond1_name],
    #                     'Condition2': [cond2_name],
    #                     'Condition1_PP_rad': [theta1],
    #                     'Condition1_PP_deg': [np.degrees(theta1)],
    #                     'Condition1_MVL': [r1],
    #                     'Condition1_n': [len(angles1)],
    #                     'Condition2_PP_rad': [theta2],
    #                     'Condition2_PP_deg': [np.degrees(theta2)],
    #                     'Condition2_MVL': [r2],
    #                     'Condition2_n': [len(angles2)],
    #                     'F': [F],
    #                     'p': [p],
    #                     'Significant': [p < alpha]
    #                 })
                    
    #                 results_df.to_csv(output_file, index=False)
                    
    #                 logger.info(f"Saved comparison results to {output_file}")
                    
    #                 # Create and save plot
    #                 fig = Figure(figsize=(10, 8), dpi=100)
    #                 ax = fig.add_subplot(111, polar=True)
                    
    #                 # Calculate mean amplitudes for each condition
    #                 mean_amp1 = np.nanmean(amp1, axis=0)
    #                 mean_amp1 = mean_amp1 / mean_amp1.sum()
                    
    #                 mean_amp2 = np.nanmean(amp2, axis=0)
    #                 mean_amp2 = mean_amp2 / mean_amp2.sum()
                    
    #                 # Create angles for plotting
    #                 angles = np.linspace(0, 2*np.pi, nbins, endpoint=False)
                    
    #                 # Plot data
    #                 ax.bar(angles, mean_amp1, width=width, alpha=0.5, label=cond1_name)
    #                 ax.bar(angles, mean_amp2, width=width, alpha=0.5, label=cond2_name)
                    
    #                 # Add preferred phase markers
    #                 ax.plot([theta1, theta1], [0, np.max(mean_amp1)*1.2], 'r-', linewidth=2)
    #                 ax.plot([theta2, theta2], [0, np.max(mean_amp2)*1.2], 'b-', linewidth=2)
                    
    #                 # Add labels and title
    #                 ax.set_title(f'PAC Comparison\n{cond1_name} vs {cond2_name}\nF={F:.2f}, p={p:.4f}')
    #                 ax.set_theta_zero_location('N')  # 0 at the top
    #                 ax.set_theta_direction(-1)  # clockwise
                    
    #                 # Add legend
    #                 ax.legend()
                    
    #                 # Save figure
    #                 fig_file = f"{out_dir}/pac_comparison_{cond1_name}_vs_{cond2_name}.png"
    #                 fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                    
    #                 logger.info(f"Saved comparison plot to {fig_file}")
                    
    #                 return {
    #                     'condition1': cond1_name,
    #                     'condition2': cond2_name,
    #                     'theta1': theta1,
    #                     'theta2': theta2,
    #                     'r1': r1,
    #                     'r2': r2,
    #                     'F': F,
    #                     'p': p,
    #                     'significant': p < alpha,
    #                     'output_file': output_file,
    #                     'fig_file': fig_file
    #                 }
                
    #             except Exception as e:
    #                 logger.error(f"Error performing Watson-Williams test: {e}")
    #                 import traceback
    #                 traceback.print_exc()
    #                 return None
            
    #         elif test_type == 'permutation':
    #             # Implement permutation test for PAC comparison
    #             logger.error("Permutation test not implemented yet")
    #             return None
            
    #         else:
    #             logger.error(f"Unknown test type: {test_type}")
    #             return None
        
    #     except Exception as e:
    #         logger.error(f"Error comparing conditions: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    
    def export_pac_parameters_to_csv(self, json_dir=None, csv_file=None, 
                                   channels=None, stages=None, 
                                   phase_freq=None, amp_freq=None):
        """
        Export PAC parameters from tracking to a CSV file.
        
        Parameters
        ----------
        json_dir : str
            Directory containing JSON files
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
            
        Returns
        -------
        dict
            Dictionary containing export results
        """
        logger = self.logger
        
        # Use tracking dictionary if available
        if 'event_pac' in self.tracking and self.tracking['event_pac']:
            try:
                # Set default output file if not provided
                if csv_file is None:
                    if json_dir is None:
                        json_dir = os.path.join(self.rootpath, "wonambi", "pac_results")
                    
                    # Create filename based on parameters
                    if phase_freq and amp_freq:
                        ph_str = f"{phase_freq[0]}-{phase_freq[1]}Hz"
                        amp_str = f"{amp_freq[0]}-{amp_freq[1]}Hz"
                        # Modified the CSV filename to reflect slow wave-spindle coupling
                        csv_file = f"{json_dir}/sw_spindle_pac_summary_{ph_str}_{amp_str}.csv"
                    else:
                        csv_file = f"{json_dir}/sw_spindle_pac_summary.csv"
                
                # Filter channels if specified
                if channels is None:
                    channels = list(self.tracking['event_pac'].keys())
                else:
                    channels = [ch for ch in channels if ch in self.tracking['event_pac']]
                
                # Create key based on frequency bands if provided
                key = None
                if phase_freq and amp_freq:
                    key = f"{phase_freq[0]}-{phase_freq[1]}Hz_{amp_freq[0]}-{amp_freq[1]}Hz"
                
                # Prepare data for export
                data = []
                for ch in channels:
                    if ch not in self.tracking['event_pac']:
                        continue
                    
                    ch_results = self.tracking['event_pac'][ch]
                    
                    if key:
                        # Use specific frequency key
                        if key in ch_results:
                            results = ch_results[key]
                            data.append({
                                'Channel': ch,
                                'Phase_Freq': f"{phase_freq[0]}-{phase_freq[1]}",
                                'Amp_Freq': f"{amp_freq[0]}-{amp_freq[1]}",
                                'MI': results.get('mi_norm', float('nan')),
                                'MI_pval': results.get('pval', float('nan')),
                                'PP_degrees': results.get('preferred_phase_deg', float('nan')),
                                'MVL': results.get('mean_vector_length', float('nan')),
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
                                
                                data.append({
                                    'Channel': ch,
                                    'Phase_Freq': ph_freq,
                                    'Amp_Freq': amp_freq,
                                    'MI': results.get('mi_norm', float('nan')),
                                    'MI_pval': results.get('pval', float('nan')),
                                    'PP_degrees': results.get('preferred_phase_deg', float('nan')),
                                    'MVL': results.get('mean_vector_length', float('nan')),
                                    'Rayleigh_p': results.get('rayleigh_p', float('nan')),
                                    'N_Segments': results.get('n_segments', 0)
                                })
                            except:
                                logger.warning(f"Could not parse frequency key: {freq_key}")
                
                # Create DataFrame and export to CSV
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(csv_file, index=False)
                    logger.info(f"Exported PAC parameters to {csv_file}")
                    return {'file': csv_file, 'channels': len(channels), 'rows': len(data)}
                else:
                    logger.warning("No PAC data to export")
                    return None
            
            except Exception as e:
                logger.error(f"Error exporting PAC parameters: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            logger.warning("No PAC results in tracking dictionary")
            return None


