import numpy as np
import json
import scipy.io
import h5py
from pathlib import Path
from datetime import datetime
from wonambi import Dataset as WonambiDataset



class LargeDataset:
    """Dataset class optimized for large EEG recordings"""
    
    def __init__(self, filename, create_memmap=False, memmap_dir=None, extract_eeglab_metadata=True):
        """
        Initialize a large dataset handler
        
        Parameters
        ----------
        filename : str
            Path to the original EEG file
        create_memmap : bool
            Whether to create a memory-mapped version of the data
        memmap_dir : str or None
            Directory to store memory-mapped files, if None use same directory as input
        """
        self.filename = Path(filename)
        self.original_dataset = WonambiDataset(filename)
        self.memmap_info = None
        
        # Copy basic header info
        self.header = self.original_dataset.header
        self.channels = self.header['chan_name']
        self.sampling_rate = self.header['s_freq']

        # Fix EEGLAB start time if needed
        if extract_eeglab_metadata:
            self._extract_eeglab_metadata()

        # Create memory map if requested
        if create_memmap:
            self.create_memmap(memmap_dir)
    
    def _extract_eeglab_metadata(self):
        """
        Extract metadata from an EEGLAB .mat file.
        """
        """Extract metadata with failsafe timeout"""
        
        result = [None]
        error = [None]
        try:
            # Attempt to load the file using scipy.io.loadmat
            eeglab_data = scipy.io.loadmat(self.filename, struct_as_record=False, squeeze_me=True)
            is_h5py = False
        except NotImplementedError:
            # Handle MATLAB v7.3 files using h5py
            print("MATLAB v7.3 file detected. Using h5py to load the file.")
            with h5py.File(self.filename, 'r') as f:
                eeglab_data = self._load_hdf5_data(f)
            is_h5py = True

        try:
            if is_h5py:
                # Handle h5py data structure
                self._process_h5py_metadata(eeglab_data)
            else:
                # Handle scipy data structure
                self._process_scipy_metadata(eeglab_data)
            result[0] = eeglab_data
        except Exception as e:
            print(f"Error extracting EEGLAB metadata: {e}")
            error[0] = e
          
      
        return result[0]



    def _process_scipy_metadata(self, eeglab_data):
        """Process metadata from scipy.io.loadmat structure"""
    # Access the EEG structure
        eeg = eeglab_data.get('EEG', None)
        if eeg is None:
            print("Warning: Could not find EEG structure in the EEGLAB file")
            return
            
        # Extract additional metadata from the EEG structure
        try:
            for attr in ['group', 'condition', 'session']:
                if hasattr(eeg, attr):
                    self.header[attr] = getattr(eeg, attr)
            
            if hasattr(eeg, 'etc') and hasattr(eeg.etc, 'stages'):
                self.header['stages'] = eeg.etc.stages
                
            if hasattr(eeg, 'event'):
                # Extract event information
                event_onsets = [getattr(event, 'latency', None) for event in eeg.event] # sample points
                event_types = [getattr(event, 'type', None) for event in eeg.event]
                event_durations = [getattr(event, 'duration', None) for event in eeg.event]  
                event_isreject = [getattr(event, 'is_reject', None) for event in eeg.event]  
                # Create annotations object
                annotations = {
                    'onsets': event_onsets,
                    'types': event_types,
                    'durations': event_durations,
                    'isreject': event_isreject,
                }     
                self.header['event'] = annotations
            
            # Parse the date string into a datetime object
            if hasattr(eeg, 'etc') and hasattr(eeg.etc, 'rec_startdate'):
                print(f"Found rec_startdate in EEG.etc: {eeg.etc.rec_startdate}")
                parsed_date = self._parse_start_date(eeg.etc.rec_startdate)

                # Make sure we update both places where the start time might be stored
                if parsed_date is not None:
                    self.original_dataset.start_time = parsed_date
                    
        except Exception as e: 
            print(f"Error processing scipy metadata: {e}")

    def _process_h5py_metadata(self, eeglab_data):
        """Process metadata from h5py structure"""
        # Access the EEG structure
        eeg = eeglab_data.get('EEG', None)
        if eeg is None:
            print("Warning: Could not find EEG structure in the EEGLAB file")
            return
            
        # Extract additional metadata from the EEG structure
        try:
            print("Processing EEG metadata...")
            for attr in ['group', 'condition', 'session']:
                if attr in eeg:
                    print(f"Extracting field: {attr}")
                    value = eeg[attr]
                    # Resolve HDF5 references
                    value = self._resolve_h5py_value(value)
                    if value is not None:  # Skip None values
                        self.header[attr] = value
                    
            # Extract stages if available
            if 'etc' in eeg and 'stages' in eeg['etc']:
                print("Extracting stages")
                stages = eeg['etc']['stages']
                stages = self._resolve_h5py_value(stages)
                if stages is not None:
                    self.header['stages'] = stages
                    #print(f"  -> stages: {stages}")
                        
            # Extract events if available
            if 'event' in eeg:
                print("Processing events...")
                events = eeg['event']
                annotations = {
                    'onsets': [], 'types': [], 'durations': [], 'isreject': [],
                }
                for field, key in zip(['latency', 'type', 'duration','is_reject'], annotations):
                    if field in events:
                        value = self._resolve_h5py_value(events[field])
                        if value is not None:
                            annotations[key] = value
                            # print(f"  -> event[{key}]: {value}")
                if any(len(v) > 0 for v in annotations.values()):
                    self.header['event'] = annotations
        
            # Parse the date string into a datetime object                
            if 'etc' in eeg and 'rec_startdate' in eeg['etc']:
                print("Extracting rec_startdate")
                rec_startdate = self._resolve_h5py_value(eeg['etc']['rec_startdate'])
                if rec_startdate:
                    self._parse_start_date(rec_startdate)
        
        except Exception as e:
            print(f"Error processing h5py metadata: {e}")

    def _parse_start_date(self, rec_startdate):
        """Parse the recording start date with multiple format attempts"""
        try:
            # Print debug information
            print(f"Original rec_startdate type: {type(rec_startdate)}")
            if isinstance(rec_startdate, np.ndarray):
                print(f"Array shape: {rec_startdate.shape}, dtype: {rec_startdate.dtype}")
            
            # Special handling for uint16 arrays
            if isinstance(rec_startdate, np.ndarray) and rec_startdate.dtype == np.uint16:
                print("Processing uint16 date array")
                # Convert uint16 to characters, filtering out zeros
                chars = []
                # Flatten array if it's multi-dimensional
                flat_array = rec_startdate.flatten()
            
                # Get only non-zero values (zeros are usually null terminators)
                for val in flat_array:
                    if val != 0:
                        try:
                            chars.append(chr(val))
                        except ValueError:
                            # Skip invalid Unicode code points
                            pass
            
                # Join characters into a string
                date_str = ''.join(chars)
                print(f"Converted date string: {date_str}")
                
                # Continue with the normal date parsing using the converted string
                rec_startdate = date_str


            # Handle other different types that could come from h5py
            elif isinstance(rec_startdate, np.ndarray):
                # Convert ndarray to string
                if rec_startdate.dtype.kind in ['S', 'U']:  # String or Unicode
                    if rec_startdate.size == 1:
                        # Single element string array
                        rec_startdate = rec_startdate.item()
                    else:
                        # Array of strings, join them
                        rec_startdate = b''.join(rec_startdate).decode('utf-8') if rec_startdate.dtype.kind == 'S' else ''.join(rec_startdate)
                elif rec_startdate.dtype.kind in ['i', 'u']:  # Integer
                    # Convert array of integers to string (ASCII/Unicode code points)
                    if rec_startdate.ndim > 1:  # Multi-dimensional array
                        # Flatten array first
                        flat_array = rec_startdate.flatten()
                        # Filter out zeros and convert to characters
                        char_array = [chr(int(x)) for x in flat_array if x != 0]
                    else:  # 1D array
                        char_array = [chr(int(x)) for x in rec_startdate if x != 0]
                    
                    rec_startdate = ''.join(char_array)
                    print(f"Converted character array to string: {rec_startdate}")

            
            # Handle bytes
            if isinstance(rec_startdate, bytes):
                rec_startdate = rec_startdate.decode('utf-8')
                
            # Make sure we have a string at this point
            if not isinstance(rec_startdate, str):
                print(f"Warning: Could not parse date, unexpected type after conversion: {type(rec_startdate)}")
                return

            print(f"Converted date string: {rec_startdate}")
            
            # Try common EEGLAB date formats
            date_formats = [
                '%d-%b-%Y %H:%M:%S',  # 01-Jan-2020 12:00:00
                '%Y-%m-%d %H:%M:%S',  # 2020-01-01 12:00:00
                '%d.%m.%Y %H:%M:%S',  # 01.01.2020 12:00:00
                '%m/%d/%Y %H:%M:%S',  # 01/01/2020 12:00:00
                '%Y-%m-%dT%H:%M:%S',  # 2020-01-01T12:00:00
            ]
            
            # Try ISO format with timezone 
            try:
                # For strings like '2019-06-17T19:19:55.256234+10:00'
                import dateutil.parser
                parsed_date = dateutil.parser.parse(rec_startdate)
                self.header['start_time'] = parsed_date
                print(f"Updated header start_time to: {parsed_date}")
                return parsed_date
            except (ImportError, ValueError):
                pass
            
            # Try the list of formats
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(rec_startdate, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                print(f"Warning: Could not parse date format: {rec_startdate}")
                return
            
            # Update the header with the correct start time
            self.header['start_time'] = parsed_date
            print(f"Updated header start_time to: {parsed_date}")
            
        except Exception as e:
            print(f"Error parsing start date: {e} (type: {type(rec_startdate)})")
            # Store as string if we couldn't process it
            if isinstance(rec_startdate, np.ndarray):
                try:
                    self.header['date_array'] = rec_startdate.tolist()
                except:
                    pass


    def _load_hdf5_data(self, hdf5_group, depth=0, max_depth=10, path="root"):
        """
        Recursively load data from an HDF5 group into a nested dictionary.
         Parameters
        ----------
        hdf5_group : h5py.Group
            The HDF5 group to load
        depth : int
            Current recursion depth
        max_depth : int
            Maximum recursion depth to prevent infinite recursion
        path : str
            Current path in the HDF5 file (for debugging)
        
        Returns
        -------
        dict
            Nested dictionary containing the HDF5 data
        """
        # Guard against excessive recursion
        if depth >= max_depth:
            print(f"Maximum recursion depth reached at {path}, stopping recursion")
            return {"max_depth_reached": True}

        result = {}
        try:
            # Get keys before iteration to avoid any potential modifications
            keys = list(hdf5_group.keys())
            
            for key in keys:
                try:
                    item = hdf5_group[key]
                    new_path = f"{path}/{key}"
                    
                    if isinstance(item, h5py.Group):
                        # Recursively process group with depth tracking
                        result[key] = self._load_hdf5_data(item, depth + 1, max_depth, new_path)
                    elif isinstance(item, h5py.Dataset):
                        # Check for large datasets to avoid memory issues
                        size_mb = np.prod(item.shape) * item.dtype.itemsize / (1024*1024) if hasattr(item, 'shape') else 0
                        
                        if size_mb > 100:  # Skip loading datasets larger than 100MB
                            print(f"Skipping large dataset {new_path}: {size_mb:.2f} MB")
                            result[key] = {
                                "shape": item.shape,
                                "dtype": str(item.dtype),
                                "size_mb": size_mb,
                                "large_dataset": True
                            }
                        elif item.dtype == h5py.ref_dtype and item.size > 10000:
                            # Handle large reference arrays
                            print(f"Large reference array detected at {new_path}: {item.size} references")
                            result[key] = {
                                "shape": item.shape,
                                "dtype": str(item.dtype),
                                "reference_count": item.size,
                                "large_reference_array": True
                            }
                        else:
                            # Load normal datasets
                            result[key] = item[()]
                except Exception as e:
                    # Handle errors for specific items
                    print(f"Error loading {path}/{key}: {e}")
                    result[key] = {"error": str(e)}
        
        except Exception as e:
            # Handle errors for the entire group
            print(f"Error processing HDF5 group at {path}: {e}")
            return {"error": str(e)}

        return result


        # result = {}
        # for key, item in hdf5_group.items():
        #     if isinstance(item, h5py.Group):
        #         result[key] = self._load_hdf5_data(item)
        #     elif isinstance(item, h5py.Dataset):
        #         result[key] = item[()]
        # return result

    def _resolve_h5py_value(self, value, _depth=0, _max_depth=10):
        """
        Resolve HDF5 references in a value, which may be a single reference, an array of references, or a list.
        
        Parameters
        ----------
        value : any
            The value to resolve, which might contain HDF5 references
            
        Returns
        -------
        resolved_value : any
            The resolved value
        """
        # Circuit breaker to prevent infinite recursion
        if _depth > _max_depth:
            print(f"WARNING: Maximum recursion depth reached ({_depth}/{_max_depth}), stopping resolution")
            return None

        try:
            # Debug info
            #print(f"Resolving value of type: {type(value)}")
            # if isinstance(value, np.ndarray):
            #     print(f"  Array shape: {value.shape}, dtype: {value.dtype}")
            #     if value.size > 0:
            #         print(f"  First few elements: {value.flatten()[:min(5, value.size)]}")
            
            # Detect null reference arrays
            if isinstance(value, np.ndarray) and value.dtype == np.uint64 and np.all(value == 0):
                return None

            # Handle direct reference
            if isinstance(value, h5py.Reference):
                return self._resolve_single_reference(value)

            # Handle array of references
            if isinstance(value, np.ndarray) and value.dtype == object:
                result = []
                for ref in value.flatten():
                    if isinstance(ref, h5py.Reference) and ref:
                        resolved = self._resolve_single_reference(ref)
                        result.append(resolved)
                    else:
                        result.append(None)
                return result

            # Handle lists with circuit breaker
            if isinstance(value, list):
                return [self._resolve_h5py_value(item, _depth + 1, _max_depth) for item in value]

            # Special handling for uint16 arrays (like date strings, filenames, subjects)
            if isinstance(value, np.ndarray) and value.dtype == np.uint16:
                # Convert uint16 array to string
                try:
                    text = ''.join(chr(c) for c in value.flatten() if c != 0)
                    return text
                except:
                    return value.tolist()

            # Simplify (N, 1) or (1, N) arrays
            if isinstance(value, np.ndarray) and value.size == 1:
                return value.item()

            # Return other numpy arrays as lists
            if isinstance(value, np.ndarray):
                return value.tolist()

            return value

        except Exception as e:
            print(f"Error resolving HDF5 value: {e}")
            return value

    def _resolve_single_reference(self, reference):
        """
        Resolve a single HDF5 object reference.
        
        Parameters
        ----------
        reference : h5py.Reference
            The HDF5 object reference to resolve
            
        Returns
        -------
        data
            The resolved data
        """
        try:
            # We need to reopen the file to resolve references
            with h5py.File(self.filename, 'r') as f:
                # Get the referenced object
                obj = f[reference]
                # Return the data
                data = obj[()]
                #print(f"  Resolving object {obj.name}, dtype={data.dtype if isinstance(data, np.ndarray) else type(data)}")

                # Special handling for string data
                if isinstance(data, np.ndarray):
                    if data.dtype.kind in ['S', 'U']:
                        # Convert bytes to strings if needed
                        if data.dtype.kind == 'S':
                            # Handle single string or array of strings
                            if data.size == 1:
                                return data.item().decode('utf-8')
                            else:
                                return [s.decode('utf-8') if isinstance(s, bytes) else s for s in data]
                        else:
                            return data.tolist()  # Unicode strings
                        
                    elif data.dtype ==np.uint8:
                        if data.size == 1:
                            return int(data.item())  # Return 0 or 1
                        try:
                            return data.tobytes().decode('utf-8').rstrip('\x00')
                        except UnicodeDecodeError:
                            return [int(x) for x in data.flatten()]
                        
                    elif data.dtype == np.uint16:
                        # Convert uint16 array to string (ASCII codes)
                        try:
                            return ''.join(chr(c) for c in data.flatten() if c != 0)
                        except:
                            return data.tolist()
                    
                    elif data.dtype.kind in 'f':
                        # Handle floating-point data
                        if data.size == 1:
                            return data.item()  # Return as a Python float
                        else:
                            return data.tolist()  # Return as a list of floats
                
                # Convert numpy arrays to lists for better serialization
                if isinstance(data, np.ndarray):
                    return data.tolist()
                
                return data
        
        except Exception as e:
            print(f"Error resolving single HDF5 reference: {e}")
            return None

    def create_memmap(self, memmap_dir=None):
        """Create a memory-mapped version of the data for faster access"""
        if memmap_dir is None:
            memmap_dir = self.filename.parent
        else:
            memmap_dir = Path(memmap_dir)
            
        memmap_path = memmap_dir / f"{self.filename.stem}_memmap.dat"
        info_path = memmap_dir / f"{self.filename.stem}_memmap.json"
        
        # Get dataset dimensions
        n_channels = len(self.channels)
        n_samples = self.header.get('n_samples', 
                                    int(self.header.get('recording_duration', 8*3600) * 
                                        self.sampling_rate))
        
        # Create memory-mapped file
        shape = (n_channels, n_samples)
        mmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=shape)
        
        # Fill the memmap file in chunks
        chunk_size = 60 * self.sampling_rate  # 1 minute of data
        chunks = n_samples // chunk_size + (1 if n_samples % chunk_size > 0 else 0)
        
        print(f"Creating memory map with {chunks} chunks...")
        for i in range(chunks):
            start_sample = i * chunk_size
            end_sample = min((i + 1) * chunk_size, n_samples)
            start_time = start_sample / self.sampling_rate
            end_time = end_sample / self.sampling_rate
            
            print(f"Processing chunk {i+1}/{chunks}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Read chunk from original data
            try:
                data = self.original_dataset.read_data(begtime=start_time, endtime=end_time)
                mmap[:, start_sample:end_sample] = data.data[0]
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
        
        # Flush to disk
        mmap.flush()
        
        # Create info file
        self.memmap_info = {
            'filepath': str(memmap_path),
            'shape': shape,
            'channels': self.channels,
            'sampling_rate': self.sampling_rate,
            'dtype': 'float32'
        }
        
        # Save for later use
        with open(info_path, 'w') as f:
            json.dump(self.memmap_info, f)
        
        print(f"Memory map created at {memmap_path}")
        return self.memmap_info
    
    def read_data(self, begtime=None, endtime=None, chan=None):
        """
        Read data from the dataset, using memory map if available
        
        Parameters
        ----------
        begtime : float or None
            Start time in seconds
        endtime : float or None
            End time in seconds
        chan : list or None
            List of channels to load
            
        Returns
        -------
        data : ndarray
            Array containing the requested data
        """
        if self.memmap_info is not None:
            # Use memory map for faster access
            return self._read_from_memmap(begtime, endtime, chan)
        else:
            # Fall back to original Wonambi method
            return self.original_dataset.read_data(begtime=begtime, endtime=endtime, chan=chan)
    
    def _read_from_memmap(self, begtime=None, endtime=None, chan=None):
        """Read data from memory map"""
        mmap_path = self.memmap_info['filepath']
        shape = tuple(self.memmap_info['shape'])
        
        # Open memory map
        mmap_data = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=shape)
        
        # Calculate indices
        start_idx = 0 if begtime is None else int(begtime * self.sampling_rate)
        end_idx = shape[1] if endtime is None else int(endtime * self.sampling_rate)
        
        # Get channel indices
        if chan is None:
            chan_indices = slice(None)  # All channels
        else:
            # Convert channel names to indices if needed
            if isinstance(chan[0], str):
                chan_indices = [self.channels.index(ch) for ch in chan if ch in self.channels]
            else:
                chan_indices = chan
        
        # Get data slice
        data = mmap_data[chan_indices, start_idx:end_idx]
        
        # Create a copy to avoid reference issues when memmap is closed
        data_copy = data.copy()
        
        # Format similar to Wonambi's output
        from wonambi.datatype import ChanTime
        output = ChanTime()
        output.data = np.array([data_copy])
        output.axis['chan'] = [self.channels[i] for i in chan_indices] if isinstance(chan_indices, list) else self.channels
        output.axis['time'] = np.arange(start_idx, end_idx) / self.sampling_rate
        output.s_freq = self.sampling_rate
        
        return output