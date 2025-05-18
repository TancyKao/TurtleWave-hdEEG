import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
import csv



def process_events_parallel(events, data_source, window_size=5, n_workers=4, func=None):
    """
    Process EEG events in parallel
    
    Parameters
    ----------
    events : list of dict
        List of events with at least 'start_time' key
    data_source : LargeDataset or str
        Large dataset object or path to data file
    window_size : float
        Window size around event in seconds
    n_workers : int
        Number of parallel workers
    func : callable or None
        Function to apply to each event data, if None just return the data
        
    Returns
    -------
    results : list
        List of processed event data
    """
    from .dataset import LargeDataset
    
    # Initialize data source if needed
    if isinstance(data_source, str):
        data = LargeDataset(data_source)
    else:
        data = data_source
    
    def process_single_event(event):
        # Load data around event
        start = max(0, event['start_time'] - window_size/2)
        end = start + window_size
        event_data = data.read_data(begtime=start, endtime=end)
        
        # Apply custom function if provided
        result = {
            'event_id': event.get('id', None),
            'start_time': event['start_time'],
            'data': event_data,
        }
        
        if func is not None:
            result['analysis'] = func(event_data, event)
            
        return result
    
    # Process events in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_single_event, events))
    
    return results



def explore_eeglab_structure(filename):
    """
    Utility to explore the structure of an EEGLAB file
    
    Parameters
    ----------
    filename : str
        Path to EEGLAB .set file
    
    Returns
    -------
    structure : dict
        Dictionary representation of EEGLAB file structure
    """
    import scipy.io
    import numpy as np
    
    try:
        # Load the EEGLAB file
        eeglab_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        
        # Helper function to convert MATLAB structs to dictionaries
        def struct_to_dict(struct):
            if isinstance(struct, np.ndarray):
                return [struct_to_dict(s) for s in struct]
            
            if not hasattr(struct, '_fieldnames'):
                return struct
            
            result = {}
            for field in struct._fieldnames:
                value = getattr(struct, field)
                if hasattr(value, '_fieldnames'):
                    result[field] = struct_to_dict(value)
                elif isinstance(value, np.ndarray) and value.dtype.kind == 'O':
                    result[field] = struct_to_dict(value)
                else:
                    result[field] = value
            return result
        
        # Get the EEG structure
        if 'EEG' in eeglab_data:
            eeg = eeglab_data['EEG']
            eeg_dict = struct_to_dict(eeg)
            return eeg_dict
        else:
            print("EEG structure not found in file")
            return eeglab_data
    
    except Exception as e:
        print(f"Error exploring EEGLAB file: {e}")
        return None

# Function to read channels from CSV file
def read_channels_from_csv(csv_file_path):
    channels = []
    try:
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                # Check if the first cell contains a channel name
                if row and row[0].strip():  # Only add non-empty values
                    channels.append(row[0].strip())
        
        print(f"Found {len(channels)} channels in CSV: {channels}")
        
        return channels
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
