"""
Custom extensions to Wonambi spindle detection
"""

from copy import copy
import logging

import numpy as np
import scipy
from wonambi.detect import DetectSpindle as OriginalDetectSpindle
from wonambi.detect import DetectSlowWave as OriginalDetectSlowWave
from wonambi.detect.spindle import transform_signal
from wonambi.graphoelement import Spindles

lg = logging.getLogger('turtlewave_hdEEG.extensions')


def _get_cirus_envelope_func(sfreq, filter_mode, filter_window, filter_order, transition_bw, frequency, det_remez=None):
    if filter_mode == 'wonambi':
        def _env(chunk):
            dat = transform_signal(chunk, sfreq, 'remez', det_remez)
            dat = transform_signal(dat, sfreq, 'hilbert')
            return transform_signal(dat, sfreq, 'abs')
        return _env
        
    elif filter_mode == 'java':
        half_fs = sfreq / 2.0
        _window_k = {
            'hann': 3.137, 'hamming': 3.320, 'blackman': 5.582,
            'bartlett': 3.391, 'bohman': 5.332, 'blackmanharris': 7.828,
        }
        if transition_bw:
            k = _window_k.get(filter_window)
            if k is None:
                raise ValueError(
                    f"transition_bw not supported for window "
                    f"'{filter_window}'. Supported: {list(_window_k)}"
                )
            n = int(np.ceil(k * half_fs / transition_bw))
            order = (n if n % 2 == 1 else n + 1) - 1
        else:
            order = filter_order
            
        bf = scipy.signal.firwin(order + 1,
                    [frequency[0] / half_fs, frequency[1] / half_fs],
                    window=filter_window, pass_zero='bandpass')
                    
        def _env(chunk):
            dat = scipy.signal.fftconvolve(chunk, bf, mode='same')
            n_fft = 1 << (len(dat) - 1).bit_length()
            return np.abs(scipy.signal.hilbert(dat, N=n_fft)[:len(chunk)])
            
        return _env
    else:
        raise ValueError(f"Unknown filter_mode '{filter_mode}'.")


def _detect_spindles_cirus(raw, sfreq, filter_mode, filter_window, filter_order, transition_bw, 
                           frequency, det_remez, epoch_len, epoch_overlap, 
                           alpha, duration, bg_spindle_thresh, dur_correction):
    """
    Core CIRUS spindle detection on a raw numpy array.

    Bandpass-filters the signal, extracts the Hilbert envelope, and detects
    threshold crossings of sufficient duration as spindle candidates.
    Candidates are optionally rejected if background amplitude is not
    sufficiently lower than the spindle.

    Intended for C3-M2 (or C3-CRef) during N2 and N3 sleep.

    Parameters
    ----------
    raw : np.ndarray, shape (n_samples,)
        Single-channel EEG signal in µV.
    sfreq : float
        Sampling frequency in Hz.
    filter_mode : 'java' or 'wonambi'
        Filtering pipeline to use. 'java' applies a Hamming FIR via
        fftconvolve (mode='same'), matching the original CIRUS implementation.
        'wonambi' applies a Parks-McClellan (remez) FIR via filtfilt, matching
        wonambi's filter pipeline.
    filter_window : str
        Window function for the FIR design when filter_mode='java'. Any window
        accepted by scipy firwin.
    filter_order : int
        FIR filter order for filter_mode='java'.
    transition_bw : float or None
        Transition bandwidth in Hz for filter_mode='java'. When >0, overrides
        filter_order using filter_length = ceil(K * Nyquist / transition_bw).
    frequency : tuple of float
        Spindle band filter range in Hz (low, high).
    det_remez : dict or None
        Dictionary with parameters for the remez design when filter_mode='wonambi'.
        E.g. {'freq': (11, 16), 'rolloff': 1.1, 'dur': 2.56}
    epoch_len : float or None
        Length of processing epochs in seconds. The signal is split into
        non-overlapping epochs; the Hilbert envelope is computed per epoch
        and a single threshold is derived from all epochs pooled. Spindles
        cannot span epoch boundaries. None or 0 processes the full signal as one block.
    epoch_overlap : float
        Fraction of epoch_len that consecutive epochs overlap, in [0, 1).
        Overlapping epochs reduce missed spindles near epoch boundaries.
        Duplicate detections from overlapping windows are removed.
    alpha : float
        Threshold sensitivity. Higher means fewer detections, higher amplitude required.
        Threshold = median + alpha * std of the Hilbert envelope.
    duration : tuple of float
        Valid spindle length in seconds (min, max), exclusive.
    bg_spindle_thresh : float or None
        Rejects spindles where the surrounding signal mean is not below
        bg_spindle_thresh * spindle mean. 0 or None disables the check.
    dur_correction : float
        Seconds added to each reported duration to compensate for edge
        clipping by the threshold.

    Returns
    -------
    list of dict
        One entry per detected spindle, each with keys 'start' (s), 'end' (s), 'dur' (s).
    """
    _env = _get_cirus_envelope_func(sfreq, filter_mode, filter_window, filter_order, transition_bw, frequency, det_remez)

    # --- epoch splitting ---
    if epoch_len:
        n_epoch = int(epoch_len * sfreq)
        step    = max(1, int(n_epoch * (1 - epoch_overlap)))
        starts  = list(range(0, len(raw) - n_epoch + 1, step))
        envelopes = [_env(raw[s:s + n_epoch]) for s in starts]
        offsets_s = [s / sfreq for s in starts]
    else:
        envelopes = [_env(raw)]
        offsets_s = [0.0]

    # --- threshold and detection ---
    all_env   = np.concatenate(envelopes)
    threshold = np.median(all_env) + alpha * np.std(all_env, ddof=1)
    min_samp  = int(duration[0] * sfreq)
    max_samp  = int(duration[1] * sfreq)

    spindles = []
    for envelope, epoch_offset in zip(envelopes, offsets_s):
        n = len(envelope)
        above = envelope > threshold
        trans = np.diff(above.view(np.int8), prepend=0)
        onsets = np.where(trans == 1)[0]
        offsets = np.where(trans == -1)[0]
        
        n_pairs = min(len(onsets), len(offsets))
        onsets = onsets[:n_pairs]
        offsets = offsets[:n_pairs]
        
        m_arr = offsets - onsets
        valid = (m_arr > min_samp) & (m_arr < max_samp)
        
        for onset, offset, m in zip(onsets[valid], offsets[valid], m_arr[valid]):
            if bg_spindle_thresh:
                pre  = envelope[max(0, onset - m):onset]
                post = envelope[offset:min(n, offset + m)]
                bg_n = len(pre) + len(post)
                if (bg_n == 0 or
                        (pre.sum() + post.sum()) / bg_n
                        >= bg_spindle_thresh * envelope[onset:offset].mean()):
                    continue
            peak = onset + int(np.argmax(envelope[onset:offset]))
            spindles.append({'start':     epoch_offset + onset / sfreq,
                             'end':       epoch_offset + offset / sfreq + dur_correction,
                             'dur':       m / sfreq + dur_correction,
                             'peak_time': epoch_offset + peak / sfreq})

    if epoch_overlap > 0.0:
        spindles.sort(key=lambda s: s['start'])
        deduped = []
        for s in spindles:
            if not deduped or s['start'] >= deduped[-1]['end']:
                deduped.append(s)
        spindles = deduped

    return spindles



class ImprovedDetectSpindle(OriginalDetectSpindle):
    def __init__(self, method='Moelle2011', frequency=None, duration=None, 
                 det_thresh=None, sel_thresh=None, moving_rms=None, 
                 smooth_dur=None, tolerance=None, min_interval=None, merge=False, 
                 polar='normal', **kwargs):
        """
        Initialize improved spindle detection.
        
        Parameters
        ----------
        method : str
            Detection method. Supported methods include: 'Ferrarelli2007', 
            'Moelle2011', 'Nir2011', 'Wamsley2012', 'Martin2013', 'Ray2015',
            'Lacourse2018'
        frequency : tuple of float
            Frequency range for spindle detection (low and high)
        duration : tuple of float
            Duration range for spindles in seconds (min and max)
        det_thresh : float or None
            Detection threshold (method-specific units)
        sel_thresh : float or None
            Selection threshold (method-specific units)
        moving_rms : dict or float or None
            Parameters for moving RMS, format: {'dur': float, 'step': float or None}
            or just duration as float
        smooth_dur : float or None
            Duration for smoothing window in seconds
        tolerance : float or None
            Tolerance for merging events in seconds
        min_interval : float or None
            Minimum interval between events in seconds
        merge : bool
            If True, merge events across channels
        polar : str
            Signal polarity - 'normal' or 'opposite'
        **kwargs : dict
            Additional method-specific parameters
        """
        if method == 'CIRUS':
            # Set the same base attributes as OriginalDetectSpindle.__init__
            # without going through wonambi's method validation
            self.method       = 'CIRUS'
            self.frequency    = frequency if frequency is not None else (11, 16)
            self.duration     = duration  if duration  is not None else (0.5, 3.0)
            self.merge        = merge
            self.tolerance    = 0
            self.min_interval = 0
            self.power_peaks  = 'interval'
            self.rolloff      = None
        else:
            super().__init__(method, frequency, duration, merge)
        
        # Store signal inversion
        if polar == 'normal':
            self.invert = False
        elif polar == 'opposite':
            self.invert = True
        
        # Store parameters that will be applied after default initialization
        self._custom_params = {
            'det_thresh': det_thresh,
            'sel_thresh': sel_thresh,
            'moving_rms_dur': moving_rms,
            'smooth_dur': smooth_dur,
            'tolerance': tolerance,
            'min_interval': min_interval,
            **kwargs  # Include any other custom parameters
        }
        
        # Set method-specific parameters
        self._set_method_params()
        # Apply custom parameters to override defaults
        self._apply_custom_parameters()

    def _set_method_params(self):
        """Set parameters specific to each detection method."""
        if self.method == 'Ferrarelli2007':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (11, 15)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (0.3, 3)
                
            self.det_remez = {'freq': self.frequency,
                              'rolloff': 0.9,
                              'dur': 2.56,
                              'step': None
                              }
            self.det_thresh = 8
            self.sel_thresh = 2
            
        elif self.method == 'Moelle2011':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (12, 15)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (0.5, 3)
                
            self.det_remez = {'freq': self.frequency,
                              'rolloff': 1.7,
                              'dur': 2.36,
                              'step': None
                               }
            self.moving_rms = {'dur': .2,
                               'step': None}
            self.smooth = {'dur': .2,
                           'win': 'flat'}
            self.det_thresh = 1.5
            
        elif self.method == 'Nir2011':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (9.2, 16.8)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (0.5, 2)
                
            self.det_butter = {'order': 2,
                               'freq': self.frequency,
                               'step': None
                               }
            self.tolerance = 1
            self.smooth = {'dur': .04}  # is in fact sigma
            self.det_thresh = 3
            self.sel_thresh = 1
            
        elif self.method == 'Wamsley2012':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (12, 15)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (0.3, 3)
                
            self.det_wavelet = {'f0': np.mean(self.frequency),
                                'sd': .8,
                                'dur': 1.,
                                'output': 'complex',
                                'step': None
                                }
            self.smooth = {'dur': .1,
                           'win': 'flat'}
            self.det_thresh = 4.5

        elif self.method == 'Martin2013':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (11.5, 14.5)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (.5, 3)
                
            self.det_remez = {'freq': self.frequency,
                              'rolloff': 1.1,
                              'dur': 2.56,
                              'step': None
                               }
            self.moving_rms = {'dur': .25,
                               'step': .25}
            self.det_thresh = 95
            
        elif self.method == 'Ray2015':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (11, 16)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (.49, None)
                
            self.cdemod = {'freq': np.mean(self.frequency)}
            self.det_butter = {'freq': (0.3, 35),
                               'order': 4,
                               'step': None}
            self.det_low_butter = {'freq': 5,
                                   'order': 4,
                                   'step': None}
            self.min_interval = 0.25 # they only start looking again after .25s
            self.smooth = {'dur': 2 / self.cdemod['freq'],
                           'win': 'triangle'}
            self.zscore = {'dur': 60,
                           'step': None,
                           'pcl_range': None}
            self.det_thresh = 2.33
            self.sel_thresh = 0.1
        
        elif self.method == 'Lacourse2018':
            if not hasattr(self, 'frequency') or self.frequency is None:
                self.frequency = (11, 16)
            if not hasattr(self, 'duration') or self.duration is None:
                self.duration = (.3, 2.5)
                
            self.det_butter = {'freq': self.frequency,
                               'order': 20,
                               'step': None}
            self.det_butter2 = {'freq': (.3, 30),
                                'order': 5,
                                'step': None}
            self.windowing = {'dur': .3,
                              'step': .1}
            win = self.windowing
            self.moving_ms = {'dur': win['dur'],
                              'step': win['step']}
            self.moving_power_ratio = {'dur': win['dur'],
                                     'step': win['step'],
                                     'freq_narrow': self.frequency,
                                     'freq_broad': (4.5, 30),
                                     'fft_dur': 2}
            self.zscore = {'dur': 30,
                           'step': None,
                           'pcl_range': (10, 90)}
            self.moving_covar = {'dur': win['dur'],
                                 'step': win['step']}
            self.moving_sd = {'dur': win['dur'],
                              'step': win['step']}
            self.smooth = {'dur': 0.3,
                           'win': 'flat_left'}
            self.abs_pow_thresh = 1.25
            self.rel_pow_thresh = 1.6
            self.covar_thresh = 1.3
            self.corr_thresh = 0.69
  
        elif self.method == 'CIRUS':
            # GUI default matches published papers 
            # (D'Rozario et al. 2022 SLEEP, Lam et al. 2021 Cereb Cortex). 
            # alpha=1.4 performs better for OSA specifically (F1 0.72 vs 0.65, 
            # validation PDF table 6) and can be passed via det_thresh if needed
            self.det_thresh    = 1.0
            # background ratio threshold hardcoded at 0.5 in qEEG_PSG CIRUS
            self.sel_thresh    = 0.5
            # duration correction hardcoded at 0.08s in qEEG_PSG CIRUS GUI
            self.dur_correction = 0.08
            # 'wonambi' mode defaults from Martin2013 (rolloff=1.1, dur=2.56)
            self.det_remez     = {'freq': self.frequency, 'rolloff': 1.1, 'dur': 2.56, 'step': None}
            # 30s epochs match the GUI's per-epoch processing and the scoring methodology
            self.epoch_len     = 30.0
            self.filter_mode   = 'java'
            self.filter_order  = 128
            self.filter_window = 'hamming'
            self.transition_bw = 0
            self.epoch_overlap = 0.0
        else:
            raise ValueError(f'Unknown method: {self.method}')

        # Safety checks for all methods - include step parameter checks here
        for param_name in ['moving_rms', 'moving_ms', 'moving_power_ratio', 
                        'moving_covar', 'moving_sd', 'windowing', 'zscore',
                        'det_butter', 'det_remez', 'det_wavelet']:
            if hasattr(self, param_name) and isinstance(getattr(self, param_name), dict):
                param_dict = getattr(self, param_name)
                if 'step' not in param_dict:
                    param_dict['step'] = None


    def _ensure_step_parameters(self):
        """
        Ensure all required parameters exist in method dictionaries with comprehensive check.
        """
        # CIRUS bypasses wonambi's detection pipeline entirely, so its params
        # don't follow wonambi's dict conventions and need no safety-filling.
        if self.method == 'CIRUS':
            return
        # Get all attributes of self that are dictionaries
        for attr_name in dir(self):
            # Skip private attributes and non-data attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            
            attr = getattr(self, attr_name)
            
            # Check if it's a dictionary
            if isinstance(attr, dict):
                # If it's a nested dictionary that contains parameters
                if any(k in attr for k in ['dur', 'freq', 'order']):
                    if 'step' not in attr:
                        attr['step'] = None
                # Ensure pcl_range exists for zscore dictionaries
                if attr_name == 'zscore' or (isinstance(attr, dict) and 'dur' in attr and 'pcl_range' not in attr):
                    attr['pcl_range'] = None
                
                # Handle other common missing parameters
                if 'freq' in attr and isinstance(attr['freq'], tuple) and 'rolloff' not in attr and attr_name.startswith('det_'):
                    attr['rolloff'] = 0.5

            # Handle moving_power_ratio parameters
            if attr_name == 'moving_power_ratio' or (isinstance(attr, dict) and 'dur' in attr and ('freq_narrow' not in attr or 'freq_broad' not in attr)):
                # Add default parameters for moving_power_ratio
                if 'freq_narrow' not in attr:
                    attr['freq_narrow'] = self.frequency if hasattr(self, 'frequency') else (11, 16)
                if 'freq_broad' not in attr:
                    attr['freq_broad'] = (4.5, 30)
                if 'fft_dur' not in attr:
                    attr['fft_dur'] = 2

                        
            # handle dictionaries in list attributes
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, dict):
                        if any(k in item for k in ['dur', 'freq', 'order']):
                            if 'step' not in item:
                                item['step'] = None
                            if 'dur' in item and 'pcl_range' not in item:
                                                    item['pcl_range'] = None
                            if 'sd' in item and 'output' not in item:
                                item['output'] = 'complex'
        # Specific method checks
        if self.method == 'Ray2015' and hasattr(self, 'zscore'):
            if 'pcl_range' not in self.zscore:
                self.zscore['pcl_range'] = None
        
        if self.method == 'Wamsley2012' and hasattr(self, 'det_wavelet'):
            if 'f0' not in self.det_wavelet:
                self.det_wavelet['f0'] = np.mean(self.frequency)
            if 'output' not in self.det_wavelet:
                self.det_wavelet['output'] = 'complex'
        
        # Lacourse2018-specific checks
        if self.method == 'Lacourse2018' and hasattr(self, 'moving_power_ratio'):
            # Ensure all required parameters exist
            if 'freq_narrow' not in self.moving_power_ratio:
                self.moving_power_ratio['freq_narrow'] = self.frequency
            if 'freq_broad' not in self.moving_power_ratio:
                self.moving_power_ratio['freq_broad'] = (4.5, 30)
            if 'fft_dur' not in self.moving_power_ratio:
                self.moving_power_ratio['fft_dur'] = 2
    
    def _apply_custom_parameters(self):
        """Apply custom parameters, overriding defaults"""
        # Simple parameter overrides
        if self._custom_params['det_thresh'] is not None:
            self.det_thresh = self._custom_params['det_thresh']
        
        if self._custom_params['sel_thresh'] is not None and hasattr(self, 'sel_thresh'):
            self.sel_thresh = self._custom_params['sel_thresh']
        
        if self._custom_params['tolerance'] is not None:
            self.tolerance = self._custom_params['tolerance']
        
        if self._custom_params['min_interval'] is not None:
            self.min_interval = self._custom_params['min_interval']
        
        # Update moving RMS duration if provided
        if self._custom_params['moving_rms_dur'] is not None and hasattr(self, 'moving_rms'):
            # Handle both dictionary and float inputs for moving_rms
            if isinstance(self._custom_params['moving_rms_dur'], dict):
                if 'dur' in self._custom_params['moving_rms_dur']:
                    self.moving_rms['dur'] = self._custom_params['moving_rms_dur']['dur']
                if 'step' in self._custom_params['moving_rms_dur']:
                    self.moving_rms['step'] = self._custom_params['moving_rms_dur']['step']
            else:
                # If just a float is provided, assume it's the duration
                self.moving_rms['dur'] = self._custom_params['moving_rms_dur']
        
        # Update smooth duration if provided
        if self._custom_params['smooth_dur'] is not None and hasattr(self, 'smooth'):
            self.smooth['dur'] = self._custom_params['smooth_dur']
        
        # Method-specific parameters
        if self.method == 'Lacourse2018':
            if 'abs_pow_thresh' in self._custom_params:
                self.abs_pow_thresh = self._custom_params['abs_pow_thresh']
            if 'rel_pow_thresh' in self._custom_params:
                self.rel_pow_thresh = self._custom_params['rel_pow_thresh']
            if 'covar_thresh' in self._custom_params:
                self.covar_thresh = self._custom_params['covar_thresh']
            if 'corr_thresh' in self._custom_params:
                self.corr_thresh = self._custom_params['corr_thresh']
            if 'window_dur' in self._custom_params and self._custom_params['window_dur'] is not None:
                # Update all window durations
                win_dur = self._custom_params['window_dur']
 
            for attr_name in ['windowing', 'moving_ms', 'moving_power_ratio', 'moving_covar', 'moving_sd']:
                if hasattr(self, attr_name):
                    attr = getattr(self, attr_name)
                    if isinstance(attr, dict):
                        # Set step equal to dur/2 if not specified (common default)
                        if 'step' not in attr or attr['step'] is None:
                            if 'dur' in attr:
                                attr['step'] = attr['dur'] / 2
    



        elif self.method == 'Ray2015':
            if 'zscore_dur' in self._custom_params and self._custom_params['zscore_dur'] is not None:
                if hasattr(self, 'zscore'):
                    self.zscore['dur'] = self._custom_params['zscore_dur']
                    # Always ensure step is present
                    if 'step' not in self.zscore:
                        self.zscore['step'] = None

        elif self.method == 'Wamsley2012':
            if 'wavelet_sd' in self._custom_params and self._custom_params['wavelet_sd'] is not None:
                if hasattr(self, 'det_wavelet'):
                    self.det_wavelet['sd'] = self._custom_params['wavelet_sd']
            if 'wavelet_dur' in self._custom_params and self._custom_params['wavelet_dur'] is not None:
                if hasattr(self, 'det_wavelet'):
                    self.det_wavelet['dur'] = self._custom_params['wavelet_dur']
            
            # Always ensure f0 is present for Wamsley2012
            if hasattr(self, 'det_wavelet'):
                self.det_wavelet['f0'] = np.mean(self.frequency)
                # Always ensure step is present
                if 'step' not in self.det_wavelet:
                    self.det_wavelet['step'] = None

        # Apply any additional custom parameters
        for key, value in self._custom_params.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        self._ensure_step_parameters()


    def _detect_cirus(self, data):
        sfreq  = data.s_freq
        chan   = data.axis['chan'][0][0]
        t_axis = np.concatenate([data.axis['time'][i] for i in range(len(data.data))])
        # wonambi preserves absolute recording times in the time axis even
        # after discontinuous concatenation 
        # index into t_axis by sample to get correct absolute times
        raw = np.concatenate([d[0] for d in data.data])

        def to_abs(t_s):
            idx = min(int(round(t_s * sfreq)), len(t_axis) - 1)
            return float(t_axis[idx])

        spindles = _detect_spindles_cirus(
            raw=raw,
            sfreq=sfreq,
            filter_mode=self.filter_mode,
            filter_window=self.filter_window,
            filter_order=self.filter_order,
            transition_bw=self.transition_bw,
            frequency=self.frequency,
            det_remez=self.det_remez,
            epoch_len=self.epoch_len,
            epoch_overlap=self.epoch_overlap,
            alpha=self.det_thresh,
            duration=self.duration,
            bg_spindle_thresh=self.sel_thresh,
            dur_correction=self.dur_correction
        )

        sp = Spindles()
        sp.events.clear()
        sp.chan_name = [chan]
        for s in spindles:
            sp.events.append({
                'start':     to_abs(s['start']),
                'end':       to_abs(s['end']),
                'peak_time': to_abs(s['peak_time']),
                'peak_val':  None,
                'chan':       chan,
            })
        return sp


    def __call__(self, data, parent=None): # 5 minutes timeout
        """
        Detect spindles in the data with optional signal inversion.
        
        Parameters
        ----------
        data : instance of Data
            The data to analyze
        parent : QWidget
            For use with GUI, as parent widget for the progress bar
        
        timeout : int
            Maximum time in seconds to allow for detection before timing out

            
        Returns
        -------
        instance of graphoelement.Spindles
            Detected spindles
        """

        
        if self.method == 'CIRUS':
            return self._detect_cirus(data)
        
        # Add comprehensive check for step parameters right before detection
        self._ensure_step_parameters()


        # Check if we need to invert the signal
        if hasattr(self, 'invert') and self.invert:
            # Make a copy to avoid modifying the original
            data_copy = data._copy(data=True)
            # Invert signal for all epochs
            for i in range(len(data_copy.data)):
                data_copy.data[i] = -data_copy.data[i]
            return super().__call__(data_copy, parent)
        else:
            # No inversion needed, call parent method directly
            return super().__call__(data, parent)
            


class ImprovedDetectSlowWave(OriginalDetectSlowWave):
    """Slow-wave detector that honours the criteria it is given.

    Wonambi's ``DetectSlowWave`` keeps the Massimini criteria in three
    attributes that its constructor hardcodes per method:

    ``trough_duration``
        min/max duration of the FIRST half-wave, in seconds — Massimini's
        *"a negative zero crossing and a subsequent positive zero crossing
        separated by 0.3–1.0 sec"*.
    ``max_trough_amp``
        depth the first half-wave's extremum must reach, in µV (negative) —
        *"a negative peak between the two zero crossings with voltage less
        than −80 µV"*.
    ``min_ptp``
        minimum negative-to-positive peak-to-peak amplitude, in µV — *"≥140
        µV"*.

    ``duration`` is a fourth, separate bound on the WHOLE wave (both
    half-waves), applied by ``detect_Massimini2004`` after the half-waves are
    joined. It is not one of the published criteria and defaults to
    ``(min_dur, max_dur)`` = ``(0, None)`` for the Massimini family.

    This subclass lets a caller override the three criteria, which is what
    Wonambi's own documentation describes and what its constructor does not
    allow. Passing a trough window as ``duration`` — which is what this class
    used to do — silently applies a half-wave limit to the whole wave and
    rejects everything below 1 Hz, i.e. most slow waves.

    References
    ----------
    Massimini, M., Huber, R., Ferrarelli, F., Hill, S. & Tononi, G.
    The sleep slow oscillation as a traveling wave.
    J Neurosci 24(31), 6862-70 (2004).
    """

    #: Methods that go through ``detect_Massimini2004`` and therefore use
    #: ``trough_duration`` / ``max_trough_amp`` / ``min_ptp``.
    MASSIMINI_METHODS = ('Massimini2004', 'AASM/Massimini2004')

    def __init__(self, method='Massimini2004', frequency=None,
                 duration=None, trough_duration=None,
                 neg_peak_thresh=None, p2p_thresh=None,
                 min_dur=None, max_dur=None, polar='normal'):
        """
        Initialize improved slow wave detection.

        Parameters
        ----------
        method : str
            Detection method. Supported methods:
            - 'Massimini2004': Traditional threshold-based detection
            - 'AASM/Massimini2004': AASM criteria with Massimini method
            - 'Ngo2015': Detection based on Ngo et al. 2015
            - 'Staresina2015': Detection based on Staresina et al. 2015
        frequency : tuple of float
            Frequency range for slow wave detection.
        duration : tuple of float or None
            Min/max duration of the WHOLE slow wave in seconds. For the
            Massimini family this is Wonambi's ``duration``, an extra bound
            on top of the published criteria; ``None`` leaves it at
            ``(min_dur, max_dur)``. For Ngo2015/Staresina2015 it is the
            zero-crossing interval and is derived from ``min_dur``/``max_dur``.
            **This is not the trough window** — pass that as
            ``trough_duration``.
        trough_duration : tuple of float or None
            Min/max duration of the first half-wave in seconds — Massimini's
            0.3–1.0 s criterion. ``None`` keeps the method's published window
            (0.3–1.0 s for Massimini2004, 0.25–1.0 s for AASM/Massimini2004).
            Ignored by Ngo2015 and Staresina2015, which are zero-crossing
            methods.
        neg_peak_thresh : float or None
            Depth the trough must reach, in µV. The sign is ignored (−80 and
            80 both mean "at least 80 µV deep"). ``None`` keeps the method's
            published criteria.

            For the Massimini family it is stored as Wonambi's negative
            ``max_trough_amp`` and enforced inside the detector. For
            Ngo2015/Staresina2015 — which publish no absolute µV criterion —
            ``None`` means **no amplitude floor at all**, and any explicit
            value is applied as a post-hoc µV floor on ``trough_val``, a
            deliberate deviation from the paper that is logged as a warning.
        p2p_thresh : float or None
            Minimum negative-to-positive peak-to-peak amplitude in µV.
            ``None`` keeps the method's published criteria.

            For the Massimini family it is stored as Wonambi's ``min_ptp``
            and enforced inside ``_add_halfwave``. For Ngo2015/Staresina2015
            it behaves exactly like ``neg_peak_thresh`` above: ``None`` means
            no floor, an explicit value is a post-hoc µV floor on the
            reported ``ptp``. It is **not** Staresina's percentile, which is
            ``self.ptp_thresh`` and is not settable here.
        min_dur : float or None
            Minimum duration of the whole slow wave in seconds. For the
            Massimini family this bounds the two joined half-waves and
            defaults to 0 (no lower bound), as in Wonambi.

            For Ngo2015/Staresina2015 it sets ``self.min_dur`` and through it
            ``det_filt['freq']``, but it does **not** reach
            ``find_intervals``, which gates on ``self.duration`` — the value
            Wonambi's constructor fixed from the published defaults. That is
            a real defect, left in place deliberately so those two methods
            keep producing exactly what they always have; see
            :meth:`_set_method_params`.
        max_dur : float or None
            Maximum duration of the whole slow wave in seconds, with the same
            Massimini-only caveat. ``None`` leaves the Massimini family
            unbounded (Wonambi then caps the search at its
            ``MAXIMUM_DURATION`` of 5 s).
        polar : str
            Signal polarity - 'normal' or 'opposite'.
        """
        # Keep the caller's requests aside: Wonambi's constructor overwrites
        # every criterion with its own per-method defaults, so they can only
        # be applied AFTER super().__init__ (in _set_method_params).
        self._duration_param = duration
        self._trough_duration_param = trough_duration
        self._neg_peak_thresh_param = neg_peak_thresh
        self._p2p_thresh_param = p2p_thresh

        super().__init__(method, duration)

        # Optional post-hoc amplitude floors in MICROVOLTS, applied only to
        # Ngo2015 and Staresina2015 — see __call__. `None` means "no floor",
        # which for those two methods is what their papers specify: neither
        # defines any absolute uV criterion. Both are normalised to a
        # magnitude so a caller's sign cannot change the meaning, matching the
        # `neg_peak_thresh` contract in the docstring above.
        self.min_neg_amp = (None if neg_peak_thresh is None
                            else abs(float(neg_peak_thresh)))
        self.min_ptp_amp = (None if p2p_thresh is None
                            else abs(float(p2p_thresh)))
        if (method not in self.MASSIMINI_METHODS
                and (self.min_neg_amp or self.min_ptp_amp)):
            lg.warning(
                "%s: neg_peak_thresh=%s / p2p_thresh=%s applied as absolute "
                "microvolt floors AFTER detection. Neither Ngo et al. 2015 "
                "nor Staresina et al. 2015 defines a fixed uV amplitude "
                "criterion (Ngo thresholds at 1.25x the mean, Staresina at "
                "the 75th percentile), so this is a deliberate deviation from "
                "the published method. Pass None to run the method as "
                "published.",
                method, neg_peak_thresh, p2p_thresh)
        if polar == 'normal':
            self.invert = False
        elif polar == 'opposite':
            self.invert = True

        # Store duration parameters
        self.min_dur_param = min_dur
        self.max_dur_param = max_dur

        # Override frequency if provided
        if frequency is not None:
            if method in ['Massimini2004', 'AASM/Massimini2004']:
                self.det_filt['freq'] = frequency
            elif method in ['Ngo2015', 'Staresina2015']:
                self.lowpass['freq'] = frequency[1]  # Use upper bound
                self.det_filt['freq'] = frequency

        # Set method-specific parameters
        self._set_method_params()

    def _set_method_params(self):
        """Set parameters specific to each detection method.

        Runs after ``super().__init__`` and is therefore the only place where
        a caller's criteria survive: it writes the published defaults first,
        then applies the overrides recorded by :meth:`__init__`, then
        recomputes ``self.duration`` from the final ``min_dur``/``max_dur``.
        """
        if self.method == 'Massimini2004':
            if not hasattr(self, 'det_filt'):
                self.det_filt = {
                    'order': 2,
                    'freq': (0.1, 4.0)
                }
            # Massimini et al. 2004, J Neurosci 24(31):6862-70, Methods.
            self.trough_duration = (0.3, 1.0)
            self.max_trough_amp = -80
            self.min_ptp = 140
            self.min_dur = 0
            self.max_dur = None


        elif self.method == 'AASM/Massimini2004':
            if not hasattr(self, 'det_filt'):
                self.det_filt = {
                    'order': 2,
                    'freq': (0.1, 1.0)
                }
            # AASM slow-wave activity criteria as configured by Wonambi's own
            # DetectSlowWave ('AASM/Massimini2004': -40 uV / 75 uV, 0.25-1.0 s).
            # The 75 uV peak-to-peak floor is the AASM number; turtlewave used
            # to use -37/70, which matches no published criterion.
            self.trough_duration = (0.25, 1.0)
            self.max_trough_amp = -40
            self.min_ptp = 75
            self.min_dur = 0
            self.max_dur = None

        elif self.method == 'Ngo2015':
            if not hasattr(self, 'lowpass'):
                self.lowpass = {
                    'order': 2,
                    'freq': 3.5
                }
            # Use provided min_dur and max_dur if available, otherwise use defaults
            self.min_dur = 0.833 if self.min_dur_param is None else self.min_dur_param
            self.max_dur = 2.0 if self.max_dur_param is None else self.max_dur_param

            if not hasattr(self, 'det_filt'):
                self.det_filt = {
                    'freq': (1 / self.max_dur, 1 / self.min_dur)
                }
            self.peak_thresh = 1.25
            self.ptp_thresh = 1.25


        elif self.method == 'Staresina2015':
            if not hasattr(self, 'lowpass'):
                self.lowpass = {
                    'order': 3,
                    'freq': 1.25
                }
            
            # Use provided min_dur and max_dur if available, otherwise use defaults
            self.min_dur = 0.8 if self.min_dur_param is None else self.min_dur_param
            self.max_dur = 2.0 if self.max_dur_param is None else self.max_dur_param

            if not hasattr(self, 'det_filt'):
                self.det_filt = {
                    'freq': (1 / self.max_dur, 1 / self.min_dur)
                }
            self.ptp_thresh = 75
 

        else:
            raise ValueError('Method must be one of: Massimini2004, AASM/Massimini2004, Ngo2015, or Staresina2015')

        # Always update filter frequency based on min_dur and max_dur for these methods
        if self.method in ['Ngo2015', 'Staresina2015'] and self.min_dur > 0 and self.max_dur > 0:
            self.det_filt['freq'] = (1 / self.max_dur, 1 / self.min_dur)

        # --- the caller's criteria, applied last so they survive -------------
        # Massimini's three published criteria live in three attributes that
        # Wonambi's constructor hardcodes; they are only settable here.
        if self.method in self.MASSIMINI_METHODS:
            if self._trough_duration_param is not None:
                self.trough_duration = tuple(self._trough_duration_param)
            if self._neg_peak_thresh_param is not None:
                # Wonambi's contract is a NEGATIVE max_trough_amp ("the trough
                # amplitude has a negative value, so this parameter sets the
                # minimum depth of the trough"). Callers in this codebase pass
                # -80/-37, the GUI spin box is clamped to [-200, 0], but the
                # old default here was +40. Normalise to a depth so neither
                # sign can silently mean something different.
                self.max_trough_amp = -abs(float(self._neg_peak_thresh_param))
            if self._p2p_thresh_param is not None:
                self.min_ptp = abs(float(self._p2p_thresh_param))
            # Massimini's branches hardcode min_dur=0 / max_dur=None, so
            # without this a caller has no way to bound the WHOLE wave under
            # its own name -- which is what made passing the trough window as
            # `duration` look like the only option.
            if self.min_dur_param is not None:
                self.min_dur = self.min_dur_param
            if self.max_dur_param is not None:
                self.max_dur = self.max_dur_param

            # self.duration is Wonambi's WHOLE-wave bound. The parent set it
            # from its own min_dur/max_dur BEFORE the overrides above ran, so
            # it has to be recomputed here or a Massimini caller's
            # min_dur/max_dur never reaches within_duration.
            #
            # MASSIMINI FAMILY ONLY, deliberately. Ngo2015 and Staresina2015
            # have the same latent defect -- the parent fixes self.duration
            # from its published defaults and _set_method_params then
            # overwrites min_dur/max_dur without it, so the GUI's "Slow Wave
            # Duration" control reaches det_filt['freq'] (cosmetic) but never
            # find_intervals, which is the actual gate. Repairing that would
            # be arguably more correct and is NOT done here: it moves the
            # detected set of two published methods that already have
            # hundreds of thousands of rows in the user's databases, and it
            # does not need exotic input to bite. frontend/turtlewave_gui.py
            # prefills the control with `setValue(detector.min_dur)` on a
            # 2-decimal spin box, so Ngo2015's 0.833 s default reads back as
            # 0.83 and every zero-crossing interval in [0.830, 0.833) would
            # flip from rejected to accepted on a GUI run with nothing typed.
            # Pinned by test_staresina_and_ngo_are_untouched.
            if self._duration_param is not None:
                self.duration = tuple(self._duration_param)
            else:
                self.duration = (self.min_dur, self.max_dur)

    @staticmethod
    def _as_negative_first(evt):
        """Report one event with a negative trough, positive peak and µV ptp.

        Applied to EVERY method, so ``neural_events.db`` carries one meaning
        per column no matter which detector wrote the row.

        **Signs.** ``detect_Massimini2004`` finds an ABOVE-zero run first
        (``detect_events(dat_det, 'above_thresh', value=0.)``) and stores that
        run's maximum as ``trough_val`` and the following minimum as
        ``peak_val``. The signs therefore come out opposite to Ngo2015 and
        Staresina2015, which are zero-crossing based, and opposite to
        Wonambi's own ``SlowWaves`` docstring ("trough_val: the lowest value,
        peak_val: the highest value"). Left alone, ``det_trough`` means
        +positive for one method family and -negative for the other, and any
        cross-method comparison of it is comparing opposite quantities. The
        swap is conditional, so this is idempotent, is a no-op on the
        zero-crossing methods, and stays correct if a future Wonambi anchors
        Massimini on the negative half-wave instead.

        **Units.** ``make_slow_waves`` computes ``'ptp': abs(ev[3] - ev[1])``
        on sample INDICES for all four methods — a sample count that scales
        with sampling rate and is independent of amplitude, stored in a
        column named ``det_ptp (uV)``. It is replaced with the real
        peak-to-peak amplitude, the quantity ``_add_halfwave`` already gates
        on with ``min_ptp``.

        This is a reporting change only: it runs after every criterion, so it
        cannot alter which events were detected.

        Parameters
        ----------
        evt : dict
            One event from ``wonambi.graphoelement.SlowWaves.events``.

        Returns
        -------
        dict
            The same dict, mutated in place.
        """
        trough_val = float(evt['trough_val'])
        peak_val = float(evt['peak_val'])

        if trough_val > peak_val:
            evt['trough_val'], evt['peak_val'] = peak_val, trough_val
            # Keep each time with its own value; for the Massimini family the
            # negative peak is the SECOND half-wave, so after the swap
            # trough_time > peak_time. That is what the detector actually
            # found, not a bookkeeping error.
            evt['trough_time'], evt['peak_time'] = (evt['peak_time'],
                                                    evt['trough_time'])

        evt['ptp'] = float(evt['peak_val']) - float(evt['trough_val'])
        return evt

    def _meets_amplitude_floor(self, evt):
        """Whether the event clears the optional µV floors. Non-Massimini only.

        Applies ``min_neg_amp`` to the negative peak and ``min_ptp_amp`` to
        the peak-to-peak amplitude, both in MICROVOLTS. It must therefore run
        AFTER :meth:`_as_negative_first`, which is what makes ``trough_val``
        the negative extremum and replaces Wonambi's ``ptp`` — a sample-index
        distance, ``abs(ev[3] - ev[1])`` (wonambi/detect/slowwave.py:418) —
        with the real amplitude.

        Until 4.3 this comparison ran BEFORE that conversion, so a µV
        threshold was tested against a sample count and the floor scaled with
        sampling rate instead of amplitude. ``ParalSWA`` passed 140.0 for
        these methods, which at 500 Hz rejected almost nothing but at 128 Hz
        rejected every event on the same signal (88 -> 0 for Staresina2015,
        11 -> 0 for Ngo2015 on the test oscillation). Both floors now default
        to ``None``, so Ngo2015 and Staresina2015 run on their published
        criteria alone.

        A threshold of ``0`` is accepted and can never reject anything, so the
        old workaround of passing ``p2p_thresh=0`` to neutralise the filter
        stays valid and is now identical to passing nothing.

        Parameters
        ----------
        evt : dict
            One event, AFTER :meth:`_as_negative_first`.

        Returns
        -------
        bool
            True if the event clears both floors (or neither is set).
        """
        if (self.min_neg_amp is not None
                and float(evt['trough_val']) > -self.min_neg_amp):
            return False
        if (self.min_ptp_amp is not None
                and float(evt['ptp']) < self.min_ptp_amp):
            return False
        return True

    def _meets_trough_depth(self, evt):
        """Whether the event's NEGATIVE trough reaches ``max_trough_amp``.

        Massimini et al. 2004 require *"a negative peak between the two zero
        crossings with voltage less than -80 µV"*. Wonambi enforces that with
        ``select_peaks``, which tests ``abs(data[events[:, 1]]) >=
        abs(limit)`` on column 1 — the extremum of the FIRST half-wave. Since
        ``detect_Massimini2004`` searches for the above-zero run first, that
        column is the POSITIVE peak, so the depth criterion lands on the
        wrong half-wave. On a symmetric wave the two are the same number and
        nothing shows; on a physiological slow wave (sharp deep negative,
        broad shallow positive) they are not, and 11 % of accepted events had
        a negative trough shallower than the stated threshold, the shallowest
        being 1.1 µV against a -80 µV criterion.

        This re-gates after detection rather than inverting the signal, so
        Wonambi's search order and candidate set are untouched and only the
        published criterion is added on top. ``max_trough_amp`` is normalised
        to a depth because callers pass either sign.

        The comparison is inclusive (``<=``), matching the ``>=`` that
        ``select_peaks`` already uses for the other half-wave.

        Parameters
        ----------
        evt : dict
            One event, AFTER :meth:`_as_negative_first`, so ``trough_val`` is
            the negative extremum.

        Returns
        -------
        bool
            True if the trough is at least ``abs(max_trough_amp)`` µV deep.
        """
        return float(evt['trough_val']) <= -abs(float(self.max_trough_amp))

    @staticmethod
    def negative_halfwave_duration(evt):
        """Duration of the negative half-wave, in seconds. Massimini only.

        ``_add_halfwave`` (wonambi/detect/slowwave.py:459-463) sets ``ev[4]``
        to the first zero crossing after ``ev[2]`` and ``ev[3]`` to the
        ``argmin`` between them, so ``[ev[2], ev[4])`` is exactly the negative
        half-wave. ``make_slow_waves`` exposes those two as ``zero_time`` and
        ``end``, which is why the span can be read straight off the event
        dict without re-deriving any crossings.

        Verified on every event of a synthetic run: the data over
        ``[ev[2], ev[4])`` is entirely negative, the reported trough lies
        inside it, and it is the ``argmin`` of that span.

        ``_as_negative_first`` swaps ``trough_*``/``peak_*`` but never touches
        ``zero_time`` or ``end``, so this is the same number before and after
        the relabel.

        **One-sample convention.** ``make_slow_waves`` stores
        ``'end': time[ev[4] - 1]`` while ``zero_time`` is ``time[ev[2]]``, so
        this span is one sample shorter than ``(ev[4] - ev[2]) / s_freq``.
        That is kept deliberately, NOT corrected: Wonambi's own
        ``within_duration`` — the function that applies this very same
        ``trough_duration`` tuple to the positive half-wave — measures
        ``time[ev[-1] - 1] - time[ev[0]]``, short by exactly the same one
        sample. Correcting one and not the other would make the same tuple
        mean two different things depending on which half-wave it lands on.
        The bias is conservative (it can only reject a borderline event) and
        is one sample: 3.9 ms at 256 Hz, 10 ms at 100 Hz, against a 300 ms
        lower bound.

        Not meaningful for Ngo2015/Staresina2015: those are zero-crossing
        methods whose events START at a positive-to-negative crossing, so for
        them ``zero_time`` is the crossing INSIDE the wave and
        ``end - zero_time`` is the POSITIVE half-wave — the opposite reading.

        Parameters
        ----------
        evt : dict
            One event from a Massimini-family detection.

        Returns
        -------
        float
            Negative half-wave duration in seconds.
        """
        return float(evt['end']) - float(evt['zero_time'])

    def _meets_trough_duration(self, evt):
        """Whether the NEGATIVE half-wave falls inside ``trough_duration``.

        Massimini: *"a negative zero crossing and a subsequent positive zero
        crossing separated by 0.3-1.0 sec"* — the negative half-wave.
        Wonambi applies ``trough_duration`` with ``within_duration`` to the
        ABOVE-zero run, because that is what its search finds first, so the
        published window lands on the positive half-wave instead. This
        re-gates on the negative one after detection, in the same style as
        :meth:`_meets_trough_depth`, leaving Wonambi's search and candidate
        set untouched.

        Bounds are inclusive on both sides, matching the ``>=`` / ``<=`` in
        ``within_duration``, and a ``None`` bound is ignored, matching its
        handling of ``None`` limits.

        Parameters
        ----------
        evt : dict
            One event from a Massimini-family detection.

        Returns
        -------
        bool
            True if the negative half-wave is within the window.
        """
        lo, hi = self.trough_duration
        dur = self.negative_halfwave_duration(evt)
        if lo is not None and dur < lo:
            return False
        if hi is not None and dur > hi:
            return False
        return True

    def _permissive_search(self):
        """A copy of this detector that pre-rejects nothing on the up-state.

        ``detect_Massimini2004`` applies two of the paper's criteria to the
        ABOVE-zero run, before any of our re-gates can see the candidate:

        * ``within_duration(above_zero, time, opts.trough_duration)`` requires
          the POSITIVE half-wave to fall inside the window. A wave with a
          0.45 s negative half-wave — paper-valid under any window — and a
          1.15 s positive one is discarded outright.
        * ``select_peaks(..., opts.max_trough_amp)`` requires the POSITIVE
          peak to reach the depth, so a wave with a deep trough and a shallow
          up-state is discarded outright.

        Post-detection re-gates can only remove, never recover, so leaving
        those in place biases which slow waves survive by their UP-state
        duration and amplitude — quantities Massimini does not constrain at
        all. This hands Wonambi a search that pre-rejects on neither, and
        lets :meth:`_meets_trough_duration` and :meth:`_meets_trough_depth`
        enforce the published criteria on the negative half-wave, where they
        belong.

        Only those two are relaxed. ``min_ptp`` is left at its real value:
        ``_add_halfwave`` already applies it in genuine microvolts to the
        true negative-to-positive excursion, so it is correct as it stands
        and still keeps the candidate set bounded. The whole-wave
        ``duration`` bound is likewise untouched.

        Returns a shallow COPY rather than mutating and restoring ``self``,
        so the user-facing ``trough_duration`` / ``max_trough_amp`` — which
        the GUI reads to prefill its spin boxes, and which the re-gates read
        back — are never briefly wrong, and a re-entrant or threaded caller
        cannot observe a half-configured detector.

        Returns
        -------
        instance of ImprovedDetectSlowWave
            A shallow copy with ``trough_duration = (None, None)`` (both
            limits ignored by ``within_duration``) and ``max_trough_amp = 0``
            (``abs(x) >= 0`` is always true in ``select_peaks``).
        """
        search = copy(self)
        search.trough_duration = (None, None)
        search.max_trough_amp = 0
        # copy() is shallow, so det_filt (and lowpass, where present) would
        # still be the SAME dict object as the detector's. Wonambi 7.15 only
        # reads them, so nothing aliases today, but a future in-place edit
        # anywhere in the detection path would silently reconfigure the live
        # detector. One line closes that permanently.
        for attr in ('det_filt', 'lowpass'):
            shared = getattr(search, attr, None)
            if isinstance(shared, dict):
                setattr(search, attr, dict(shared))
        return search

    def __call__(self, data):
        """
        Detect slow waves in the data.

        Parameters
        ----------
        data : instance of Data
            The data to analyze

        Returns
        -------
        instance of graphoelement.SlowWaves
            Detected slow waves
        """
        # Do NOT invert here. `self.invert` (set from polar='opposite' in
        # __init__) is Wonambi's own DetectSlowWave attribute, and every
        # slow-wave method applies it itself on its local copy of the signal:
        # detect_Massimini2004, detect_Ngo2015 and detect_Staresina2015 each
        # begin with `if opts.invert: dat_orig = -dat_orig`
        # (wonambi/detect/slowwave.py:192, :256, :322). Inverting here as well
        # would cancel the parent's inversion and make polar='opposite'
        # bit-identical to polar='normal'. See test_slow_wave_polarity in
        # tests/test_turtlewave.py, which locks this down.
        #
        # The parent also builds `dat_orig` fresh per channel
        # (hstack(data(chan=chan)) then a demean that allocates a new array),
        # so it never mutates the caller's segment in place.

        if self.method in self.MASSIMINI_METHODS:
            events = OriginalDetectSlowWave.__call__(
                self._permissive_search(), data)
        else:
            # Run detection using parent class
            events = super().__call__(data)

        # Reporting, every method: negative trough, positive peak, ptp in uV.
        # Every remaining criterion below is expressed in those normalised
        # fields, so nothing downstream can compare a microvolt threshold
        # against a sample index.
        events.events = [self._as_negative_first(evt)
                         for evt in events.events]

        if self.method not in self.MASSIMINI_METHODS:
            # Optional absolute amplitude floors for Ngo2015/Staresina2015.
            # Off by default (both thresholds None), because neither paper
            # defines one; when a caller sets them they are microvolts
            # compared against microvolts. See _meets_amplitude_floor for the
            # unit bug this replaces.
            if self.min_neg_amp is not None or self.min_ptp_amp is not None:
                events.events = [evt for evt in events.events
                                 if self._meets_amplitude_floor(evt)]

        if self.method in self.MASSIMINI_METHODS:
            # Two of the paper's three criteria, applied to the NEGATIVE
            # half-wave -- the only place they are enforced at all, since
            # _permissive_search stopped Wonambi applying them to the
            # above-zero run. The third, min_ptp, is already correct in uV
            # inside _add_halfwave and was left at its real value there.
            #
            # So the criteria the caller asked for are enforced here, on the
            # quantities Massimini names, and nowhere else:
            #   duration        _meets_trough_duration  (end - zero_time)
            #   trough depth    _meets_trough_depth     (trough_val)
            #   peak-to-peak    Wonambi's _add_halfwave (uV, unmodified)
            events.events = [evt for evt in events.events
                             if self._meets_trough_depth(evt)
                             and self._meets_trough_duration(evt)]

        return events


class ImprovedDetectKComplex(ImprovedDetectSlowWave):
    """K-complex detector built on the slow-wave detector.

    K-complexes are scored using the AASM criteria, which match Wonambi's
    `AASM/Massimini2004` configuration (≥75 µV peak-to-peak, 0.25–1.0 s
    trough duration). What distinguishes a KC from a free-running slow
    oscillation is **isolation**: a KC stands alone rather than being one
    cycle of a continuous train. This class adds a `min_isolation` filter
    on top of `ImprovedDetectSlowWave` to enforce that.

    The isolation criterion is this project's modelling choice, not a
    settled standard: AASM defines a KC by morphology and duration, not by a
    minimum gap to its neighbour.
    """

    SUPPORTED_METHODS = ('Massimini2004', 'AASM/Massimini2004')

    def __init__(self, method='AASM/Massimini2004', frequency=None,
                 duration=None, trough_duration=None,
                 neg_peak_thresh=None, p2p_thresh=None,
                 min_dur=None, max_dur=None, polar='normal',
                 min_isolation=1.0):
        """
        Parameters
        ----------
        method : str
            Detection method. Only 'Massimini2004' and 'AASM/Massimini2004'
            are supported for K-complexes (Ngo2015 / Staresina2015 are
            slow-oscillation algorithms that do not match AASM KC criteria).
        trough_duration : tuple of float or None
            Min/max duration of the negative half-wave in seconds. ``None``
            keeps the method's published window (0.25-1.0 s for AASM). This
            is NOT the whole-wave duration -- see
            :class:`ImprovedDetectSlowWave`.
        min_isolation : float
            Minimum gap in seconds between consecutive KCs, measured between
            successive ``trough_time`` values. KCs closer than this are
            dropped from the result.

            **The landmark it measures between changed in 4.3.** Wonambi's
            Massimini output labelled the POSITIVE peak ``trough_time``, so
            this gap used to be measured up-state to up-state;
            :meth:`ImprovedDetectSlowWave._as_negative_first` now puts
            ``trough_time`` on the negative peak, which is the correct
            landmark for a K-complex and the one the AASM description
            implies. The gap therefore shifts by roughly one half-wave per
            event, so K-complex counts move even at an unchanged
            ``min_isolation``. Together with the default threshold change
            below, a scripted ``ParalKC`` run changes on two axes: do not
            pool pre- and post-4.3 K-complexes.

        Other parameters are forwarded to ImprovedDetectSlowWave. Note the
        ``AASM/Massimini2004`` defaults also moved in 4.3, from -37 uV / 70 uV
        (which match no published criterion) to Wonambi's own -40 uV / 75 uV.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported KC method '{method}'. "
                f"Use one of: {self.SUPPORTED_METHODS}"
            )
        super().__init__(method=method, frequency=frequency, duration=duration,
                         trough_duration=trough_duration,
                         neg_peak_thresh=neg_peak_thresh,
                         p2p_thresh=p2p_thresh,
                         min_dur=min_dur, max_dur=max_dur, polar=polar)
        self.min_isolation = float(min_isolation)

    def __call__(self, data):
        """Detect K-complexes, then drop any that are not isolated.

        The isolation gap is measured between successive ``trough_time``
        values, which since 4.3 is the NEGATIVE peak rather than the positive
        one -- see ``min_isolation`` in :meth:`__init__` for what that means
        for counts.

        Returns
        -------
        instance of graphoelement.SlowWaves
            Detected K-complexes (as a `SlowWaves` graphoelement, so existing
            Wonambi machinery — `to_annot`, iteration over events as dicts —
            keeps working).
        """
        events = super().__call__(data)

        if self.min_isolation <= 0 or not events.events:
            return events

        sorted_events = sorted(events.events, key=lambda e: e['start'])
        isolated = []
        last_trough = -float('inf')
        for evt in sorted_events:
            t = evt.get('trough_time', evt.get('start'))
            if t - last_trough >= self.min_isolation:
                isolated.append(evt)
                last_trough = t
        events.events = isolated
        return events