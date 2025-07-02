"""
Custom extensions to Wonambi spindle detection
"""

from numpy import mean, arange
from wonambi.detect import DetectSpindle as OriginalDetectSpindle
from wonambi.detect import DetectSlowWave as OriginalDetectSlowWave



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
        # Call parent constructor
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
                
            self.det_wavelet = {'f0': mean(self.frequency),
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
                
            self.cdemod = {'freq': mean(self.frequency)}
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
                self.det_wavelet['f0'] = mean(self.frequency)
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
                self.det_wavelet['f0'] = mean(self.frequency)
                # Always ensure step is present
                if 'step' not in self.det_wavelet:
                    self.det_wavelet['step'] = None

        # Apply any additional custom parameters
        for key, value in self._custom_params.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        self._ensure_step_parameters()

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

        
        # Add comprehensive check for step parameters right before detection
        self._ensure_step_parameters()


        # Check if we need to invert the signal
        if hasattr(self, 'invert') and self.invert:
            # Make a copy to avoid modifying the original
            data_copy = data.copy()
            # Invert signal for all epochs
            for i in range(len(data_copy.data)):
                data_copy.data[i] = -data_copy.data[i]
            return super().__call__(data_copy, parent)
        else:
            # No inversion needed, call parent method directly
            return super().__call__(data, parent)
            


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