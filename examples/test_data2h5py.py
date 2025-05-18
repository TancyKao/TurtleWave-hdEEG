import h5py
import numpy as np
from pathlib import Path
import os
def safe_inspect_file(filename):
    """Safely inspect the structure of an HDF5 file with protections against common issues"""
    print(f"Inspecting file: {filename}")
    
    try:
        with h5py.File(filename, 'r') as f:
            # List top-level keys
            root_keys = list(f.keys())
            print(f"Root keys: {root_keys}")
            
            # Check if EEG structure exists
            if 'EEG' in root_keys:
                eeg = f['EEG']
                eeg_keys = list(eeg.keys())
                print(f"EEG keys: {eeg_keys}")
                
                # Check specifically for problematic areas
                problem_areas = []
                
                # Check references in events
                if 'event' in eeg_keys:
                    event = eeg['event']
                    event_keys = list(event.keys())
                    print(f"Event keys: {event_keys}")
                    
                    # Check if there are any reference datasets
                    for key in event_keys:
                        if key in event:
                            try:
                                dset = event[key]
                                if dset.dtype == h5py.ref_dtype:
                                    print(f"Found reference dataset: event/{key} (shape: {dset.shape})")
                                    # Check if there are too many references
                                    if np.prod(dset.shape) > 1000:
                                        problem_areas.append(f"Large reference array in event/{key}")
                            except Exception as e:
                                print(f"Error accessing event/{key}: {e}")
                
                # Check for unusual structures in etc
                if 'etc' in eeg_keys:
                    try:
                        etc = eeg['etc']
                        etc_keys = list(etc.keys())
                        print(f"etc keys: {etc_keys}")
                        
                        # Check for rec_startdate specifically
                        if 'rec_startdate' in etc_keys:
                            try:
                                rec_startdate = etc['rec_startdate']
                                print(f"rec_startdate type: {rec_startdate.dtype}, shape: {rec_startdate.shape}")
                                
                                # Try to understand the content
                                if rec_startdate.dtype.kind in ['S', 'U']:
                                    print("String date")
                                elif rec_startdate.dtype == np.uint16:
                                    # Show first few characters
                                    chars = ''.join(chr(c) for c in rec_startdate[:20] if c != 0)
                                    print(f"Uint16 date starting with: {chars}")
                                elif rec_startdate.dtype == h5py.ref_dtype:
                                    print("Reference type date")
                                else:
                                    print(f"Unusual date type: {rec_startdate.dtype}")
                            except Exception as e:
                                print(f"Error accessing rec_startdate: {e}")
                    except Exception as e:
                        print(f"Error accessing etc: {e}")
                
                # Return summary
                if problem_areas:
                    print(f"Potential problem areas: {problem_areas}")
                else:
                    print("No obvious structural issues detected")
    
    except Exception as e:
        print(f"Error opening file: {e}")

# Use this on your problematic file
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
#root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/OSA_BL13PR/"
#datafilename = "13PR_OSACPAP_BL_PSG_20171214.set"

datafile = os.path.join(root_dir, datafilename)

safe_inspect_file(datafile)