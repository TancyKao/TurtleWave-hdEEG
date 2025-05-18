"""
Custom extensions to Wonambi spindle detection
"""

from numpy import mean, arange
from wonambi.detect import DetectSpindle as OriginalDetectSpindle
from wonambi.detect import DetectSlowWave as OriginalDetectSlowWave


class ImprovedDetectSpindle(OriginalDetectSpindle):
    def __init__(self, method='Moelle2011', frequency=None, duration=None, merge=False):
        super().__init__(method, frequency, duration, merge)
        
        # Fix the frequency issue by updating all relevant parameters
        if frequency is not None:
            # Update frequency in all relevant method parameters
            if hasattr(self, 'det_remez'):
                self.det_remez['freq'] = self.frequency
            if hasattr(self, 'det_butter'):
                self.det_butter['freq'] = self.frequency
            if hasattr(self, 'det_low_butter') and hasattr(self, 'cdemod'):
                self.cdemod['freq'] = mean(self.frequency)
            if hasattr(self, 'det_wavelet'):
                if 'f0' in self.det_wavelet:
                    self.det_wavelet['f0'] = mean(self.frequency)
                if 'freqs' in self.det_wavelet:
                    self.det_wavelet['freqs'] = arange(self.frequency[0],
                                                    self.frequency[1] + .5, .5)
            if hasattr(self, 'sel_wavelet'):
                if 'freqs' in self.sel_wavelet:
                    self.sel_wavelet['freqs'] = arange(self.frequency[0],
                                                    self.frequency[1] + .5, .5)
            if hasattr(self, 'moving_power_ratio'):
                self.moving_power_ratio['freq_narrow'] = self.frequency
        if duration is not None:
            # Update duration in all relevant method parameters
            if hasattr(self, 'det_wavelet'):
                if 'duration' in self.det_wavelet:
                    self.det_wavelet['duration'] = duration
                if 'duration' in self.sel_wavelet:
                    self.sel_wavelet['duration'] = duration
            if hasattr(self, 'moving_power_ratio'):
                self.moving_power_ratio['duration'] = duration

class ImprovedDetectSlowWave(OriginalDetectSlowWave):
    def __init__(self, method='Massimini2004', frequency=None, 
                 duration=None, neg_peak_thresh=40, p2p_thresh=75,
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
            Frequency range for slow wave detection
        duration : tuple of float
            Duration range for slow waves in seconds (used for trough_duration in Massimini methods)
        neg_peak_thresh : float
            Minimum negative peak amplitude in μV
        p2p_thresh : float
            Minimum peak-to-peak amplitude in μV
        min_dur : float or None
            Minimum duration of a slow wave in seconds (used for Ngo2015 and Staresina2015)
        max_dur : float or None
            Maximum duration of a slow wave in seconds (used for Ngo2015 and Staresina2015)
        polar : str
            Signal polarity - 'normal' or 'opposite'
        """
        super().__init__(method, duration)
        
        # Store additional parameters
        self.min_neg_amp = neg_peak_thresh
        self.min_ptp_amp = p2p_thresh
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
        """Set parameters specific to each detection method."""
        if self.method == 'Massimini2004':
            if not hasattr(self, 'det_filt'):
                self.det_filt = {
                    'order': 2,
                    'freq': (0.1, 4.0)
                }
            # Use default values unless overridden
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
            # Use default values unless overridden
            self.trough_duration = (0.25, 1.0)
            self.max_trough_amp = -37
            self.min_ptp = 70
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
        # Invert signal if requested
        if self.invert:
            data.data[0][0] = -data.data[0][0]
        
        # Run detection using parent class
        events = super().__call__(data)
        
        # Apply additional amplitude criteria if needed
        filtered_events = []
        for evt in events:
            if (abs(evt['trough_val']) >= self.min_neg_amp and 
                abs(evt['ptp']) >= self.min_ptp_amp):
                filtered_events.append(evt)
        
        # Update events
        events.events = filtered_events
        return events