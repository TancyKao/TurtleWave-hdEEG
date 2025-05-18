#!/usr/bin/env python3
"""
HDF5 File Inspector for MATLAB v7.3 (HDF5) Files
================================================

This script safely inspects the contents of MATLAB v7.3 files that are
saved in HDF5 format, and detects potential problematic structures 
that might cause loading to hang or crash.

Usage:
    chmod +x inspect_hdf5.py
    python inspect_hdf5.py path/to/your/file.mat

"""

import os
import sys
import time
import threading
import numpy as np
import h5py
from pathlib import Path
import argparse
import traceback


def print_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")


def inspect_hdf5_file(filename, max_depth=3, timeout=300):
    """
    Safely inspect an HDF5 file with timeout protection and detailed diagnostics.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    max_depth : int
        Maximum recursion depth for exploration
    timeout : int
        Timeout in seconds for the inspection
    
    Returns
    -------
    dict
        Diagnostic information about the file
    """
    print_with_timestamp(f"Inspecting file: {filename}")
    
    # Container for the result
    result_container = [None]
    current_path = ["root"]
    inspection_complete = threading.Event()
    
    # Function to report progress
    def progress_reporter():
        start_time = time.time()
        while not inspection_complete.is_set():
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rInspecting... {elapsed:.1f}s - Current path: {current_path[0]}")
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line
    
    # Function to perform the inspection
    def file_inspector():
        try:
            diagnostic_info = {}
            
            with h5py.File(filename, 'r') as f:
                file_size_mb = os.path.getsize(filename) / (1024 * 1024)
                print_with_timestamp(f"File size: {file_size_mb:.2f} MB")
                diagnostic_info["file_size_mb"] = file_size_mb
                
                # Get top-level structure
                top_keys = list(f.keys())
                print_with_timestamp(f"Top-level keys: {top_keys}")
                diagnostic_info["top_keys"] = top_keys
                
                # Check if it looks like an EEGLAB file
                if 'EEG' in top_keys:
                    print_with_timestamp("Found EEGLAB structure")
                    diagnostic_info["file_type"] = "EEGLAB"
                    
                    # Examine EEG structure
                    eeg_info = examine_eeg_structure(f, current_path)
                    diagnostic_info["eeg_info"] = eeg_info
                else:
                    print_with_timestamp("Not an EEGLAB file or structure differs from standard")
                    diagnostic_info["file_type"] = "Unknown"
                    
                    # Just get basic info about each top key
                    diagnostic_info["structure"] = {}
                    for key in top_keys:
                        try:
                            current_path[0] = f"/{key}"
                            item = f[key]
                            
                            if isinstance(item, h5py.Group):
                                group_keys = list(item.keys())
                                item_type = "Group"
                                item_info = {"keys": group_keys, "count": len(group_keys)}
                            elif isinstance(item, h5py.Dataset):
                                item_type = "Dataset"
                                item_info = {
                                    "shape": item.shape,
                                    "dtype": str(item.dtype),
                                    "size_mb": np.prod(item.shape) * item.dtype.itemsize / (1024*1024) if hasattr(item, 'shape') else 0
                                }
                            else:
                                item_type = str(type(item))
                                item_info = {}
                                
                            diagnostic_info["structure"][key] = {
                                "type": item_type,
                                "info": item_info
                            }
                        except Exception as e:
                            diagnostic_info["structure"][key] = {
                                "error": str(e)
                            }
                
                # Attempt to load sample data
                try:
                    print_with_timestamp("\nSafe sampling of data structure...")
                    sample_data = safe_load_sample(f, max_depth, current_path)
                    diagnostic_info["sample_data"] = sample_data
                except Exception as e:
                    print_with_timestamp(f"Error during safe sampling: {e}")
                    diagnostic_info["sample_error"] = str(e)
                
            result_container[0] = diagnostic_info
            
        except Exception as e:
            print_with_timestamp(f"\nError inspecting file: {e}")
            traceback.print_exc()
            result_container[0] = {"error": str(e)}
        
        finally:
            inspection_complete.set()
    
    # Start the reporter thread
    reporter_thread = threading.Thread(target=progress_reporter)
    reporter_thread.daemon = True
    reporter_thread.start()
    
    # Start the inspector thread
    inspector_thread = threading.Thread(target=file_inspector)
    inspector_thread.daemon = True
    inspector_thread.start()
    
    # Wait with timeout
    inspector_thread.join(timeout)
    
    if inspector_thread.is_alive():
        print_with_timestamp(f"\nWARNING: Inspection timed out after {timeout} seconds")
        inspection_complete.set()  # Stop the progress reporter
        return {
            "error": "Inspection timed out",
            "last_path": current_path[0],
            "timeout": timeout
        }
    
    # Wait for progress reporter to finish
    reporter_thread.join(1)
    
    return result_container[0]


def examine_eeg_structure(f, current_path):
    """Examine the EEG structure in an EEGLAB file"""
    eeg_info = {}
    eeg = f['EEG']
    eeg_keys = list(eeg.keys())
    print_with_timestamp(f"EEG keys: {eeg_keys}")
    eeg_info["keys"] = eeg_keys
    
    # Check scalar values
    scalar_fields = ['nbchan', 'srate', 'trials', 'pnts', 'xmin', 'xmax']
    eeg_info["scalar_values"] = {}
    
    for field in scalar_fields:
        if field in eeg_keys:
            try:
                current_path[0] = f"/EEG/{field}"
                value = eeg[field][()]
                if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
                    scalar_value = float(value.item()) if isinstance(value, np.ndarray) else float(value)
                    print_with_timestamp(f"{field}: {scalar_value}")
                    eeg_info["scalar_values"][field] = scalar_value
            except Exception as e:
                print_with_timestamp(f"Error accessing {field}: {e}")
                eeg_info["scalar_values"][field] = {"error": str(e)}
    
    # Check for problematic structures
    problematic_structures = []
    
    # Check event structure
    if 'event' in eeg_keys:
        print_with_timestamp("Examining event structure...")
        current_path[0] = "/EEG/event"
        event_info = examine_event_structure(eeg['event'], current_path)
        eeg_info["event"] = event_info
        
        if "problematic_fields" in event_info and event_info["problematic_fields"]:
            problematic_structures.extend(event_info["problematic_fields"])
    
    # Check etc structure and date format
    if 'etc' in eeg_keys:
        print_with_timestamp("Examining etc structure...")
        current_path[0] = "/EEG/etc"
        etc_info = examine_etc_structure(eeg['etc'], current_path)
        eeg_info["etc"] = etc_info
        
        if "problematic_fields" in etc_info and etc_info["problematic_fields"]:
            problematic_structures.extend(etc_info["problematic_fields"])
    
    # Check chanlocs structure
    if 'chanlocs' in eeg_keys:
        print_with_timestamp("Examining chanlocs structure...")
        current_path[0] = "/EEG/chanlocs"
        chanlocs_info = examine_chanlocs_structure(eeg['chanlocs'], current_path)
        eeg_info["chanlocs"] = chanlocs_info
        
        if "problematic_fields" in chanlocs_info and chanlocs_info["problematic_fields"]:
            problematic_structures.extend(chanlocs_info["problematic_fields"])
    
    # Check for large data arrays that might cause memory issues
    large_arrays = []
    
    for key in ['data', 'icaact', 'icawinv']:
        if key in eeg_keys:
            try:
                current_path[0] = f"/EEG/{key}"
                data = eeg[key]
                shape = data.shape
                dtype = data.dtype
                size_mb = np.prod(shape) * dtype.itemsize / (1024 * 1024)
                
                print_with_timestamp(f"{key}: shape={shape}, dtype={dtype}, size={size_mb:.2f} MB")
                
                if size_mb > 100:  # More than 100MB
                    large_arrays.append(f"{key} ({size_mb:.2f} MB)")
            except Exception as e:
                print_with_timestamp(f"Error examining {key}: {e}")
    
    eeg_info["large_arrays"] = large_arrays
    eeg_info["problematic_structures"] = problematic_structures
    
    return eeg_info


def examine_event_structure(event, current_path):
    """Examine the event structure for potential issues"""
    event_info = {}
    
    try:
        event_keys = list(event.keys())
        print_with_timestamp(f"Event keys: {event_keys}")
        event_info["keys"] = event_keys
        
        # Check for reference arrays
        problematic_fields = []
        ref_fields = []
        
        for key in event_keys:
            try:
                current_path[0] = f"{current_path[0]}/{key}"
                field = event[key]
                
                if isinstance(field, h5py.Dataset):
                    shape = field.shape
                    dtype = field.dtype
                    size = np.prod(shape)
                    
                    field_info = {
                        "shape": shape,
                        "dtype": str(dtype),
                        "size": size
                    }
                    
                    if dtype == h5py.ref_dtype:
                        ref_fields.append(key)
                        field_info["is_reference"] = True
                        
                        if size > 1000:
                            problematic_fields.append(f"Large reference array in event/{key} ({size} items)")
                            field_info["problematic"] = True
                    
                    event_info[key] = field_info
                    
                    print_with_timestamp(f"event/{key}: shape={shape}, dtype={dtype}, size={size}")
                    
                    # Try to sample a few values
                    if size > 0 and size < 1000:
                        try:
                            if dtype == h5py.ref_dtype:
                                # For references, just check if they're valid
                                sample_size = min(5, size)
                                valid_count = 0
                                for i in range(sample_size):
                                    ref = field[i]
                                    if isinstance(ref, h5py.Reference) and ref:
                                        valid_count += 1
                                field_info["valid_references"] = valid_count
                            elif dtype.kind in ['S', 'U']:
                                # For strings, count non-empty
                                sample = field[0:min(5, size)]
                                field_info["sample"] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in sample]
                            elif size < 10:
                                # For small numeric arrays, get all values
                                sample = field[()].tolist()
                                field_info["sample"] = sample
                            else:
                                # For larger arrays, just get a few values
                                sample = field[0:5].tolist()
                                field_info["sample"] = sample
                        except Exception as e:
                            print_with_timestamp(f"Error sampling {key}: {e}")
                            field_info["sample_error"] = str(e)
            
            except Exception as e:
                print_with_timestamp(f"Error examining event/{key}: {e}")
                event_info[key] = {"error": str(e)}
        
        event_info["reference_fields"] = ref_fields
        event_info["problematic_fields"] = problematic_fields
    
    except Exception as e:
        print_with_timestamp(f"Error examining event structure: {e}")
        event_info["error"] = str(e)
    
    return event_info


def examine_etc_structure(etc, current_path):
    """Examine the etc structure for potential issues"""
    etc_info = {}
    
    try:
        etc_keys = list(etc.keys())
        print_with_timestamp(f"etc keys: {etc_keys}")
        etc_info["keys"] = etc_keys
        
        problematic_fields = []
        
        # Check specifically for rec_startdate
        if 'rec_startdate' in etc_keys:
            try:
                current_path[0] = f"{current_path[0]}/rec_startdate"
                rec_startdate = etc['rec_startdate']
                shape = rec_startdate.shape
                dtype = rec_startdate.dtype
                
                print_with_timestamp(f"rec_startdate: shape={shape}, dtype={dtype}")
                
                date_info = {
                    "shape": shape,
                    "dtype": str(dtype)
                }
                
                # Check if it's the problematic uint16 type
                if dtype == np.uint16:
                    print_with_timestamp("WARNING: Problematic uint16 date format detected")
                    problematic_fields.append("uint16 date format in etc/rec_startdate")
                    date_info["problematic"] = True
                    
                    # Try to convert to string
                    try:
                        date_array = rec_startdate[()]
                        date_str = ''.join(chr(c) for c in date_array.flatten() if c != 0)
                        print_with_timestamp(f"Converted date string: {date_str}")
                        date_info["converted_string"] = date_str
                    except Exception as e:
                        print_with_timestamp(f"Error converting date: {e}")
                        date_info["conversion_error"] = str(e)
                
                # Try to get sample values
                try:
                    if dtype.kind in ['S', 'U']:
                        sample = rec_startdate[()]
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8')
                        date_info["value"] = sample
                    elif dtype == np.uint16:
                        # Already handled above
                        pass
                    else:
                        sample = rec_startdate[()].tolist()
                        date_info["value"] = sample
                except Exception as e:
                    print_with_timestamp(f"Error getting date sample: {e}")
                    date_info["sample_error"] = str(e)
                
                etc_info["rec_startdate"] = date_info
            
            except Exception as e:
                print_with_timestamp(f"Error examining rec_startdate: {e}")
                etc_info["rec_startdate"] = {"error": str(e)}
        
        etc_info["problematic_fields"] = problematic_fields
    
    except Exception as e:
        print_with_timestamp(f"Error examining etc structure: {e}")
        etc_info["error"] = str(e)
    
    return etc_info


def examine_chanlocs_structure(chanlocs, current_path):
    """Examine the chanlocs structure for potential issues"""
    chanlocs_info = {}
    
    try:
        chanlocs_keys = list(chanlocs.keys())
        print_with_timestamp(f"chanlocs keys: {chanlocs_keys}")
        chanlocs_info["keys"] = chanlocs_keys
        
        problematic_fields = []
        
        # Check for channel labels
        if 'labels' in chanlocs_keys:
            try:
                current_path[0] = f"{current_path[0]}/labels"
                labels = chanlocs['labels']
                shape = labels.shape
                dtype = labels.dtype
                
                print_with_timestamp(f"labels: shape={shape}, dtype={dtype}")
                
                labels_info = {
                    "shape": shape,
                    "dtype": str(dtype)
                }
                
                # Check if it's a reference
                if dtype == h5py.ref_dtype:
                    labels_info["is_reference"] = True
                    
                    # Check if it's a large reference array
                    if np.prod(shape) > 1000:
                        problematic_fields.append(f"Large reference array in chanlocs/labels ({np.prod(shape)} items)")
                        labels_info["problematic"] = True
                    
                    # Try to sample a few values
                    try:
                        if np.prod(shape) > 0 and np.prod(shape) < 100:
                            with h5py.File(str(Path(current_path[0]).parent), 'r') as f:
                                sample_size = min(5, np.prod(shape))
                                channel_labels = []
                                for i in range(sample_size):
                                    ref = labels[i]
                                    if isinstance(ref, h5py.Reference) and ref:
                                        obj = f[ref]
                                        label = obj[()]
                                        if isinstance(label, bytes):
                                            label = label.decode('utf-8')
                                        channel_labels.append(label)
                                    else:
                                        channel_labels.append(None)
                                labels_info["sample_labels"] = channel_labels
                    except Exception as e:
                        print_with_timestamp(f"Error sampling channel labels: {e}")
                        labels_info["sample_error"] = str(e)
                
                # For string arrays
                elif dtype.kind in ['S', 'U']:
                    try:
                        sample_size = min(5, np.prod(shape))
                        sample = labels[0:sample_size]
                        if dtype.kind == 'S':
                            sample_labels = [s.decode('utf-8') if isinstance(s, bytes) else s for s in sample]
                        else:
                            sample_labels = sample.tolist()
                        labels_info["sample_labels"] = sample_labels
                    except Exception as e:
                        print_with_timestamp(f"Error sampling string labels: {e}")
                        labels_info["sample_error"] = str(e)
                
                chanlocs_info["labels"] = labels_info
            
            except Exception as e:
                print_with_timestamp(f"Error examining chanlocs/labels: {e}")
                chanlocs_info["labels"] = {"error": str(e)}
        
        chanlocs_info["problematic_fields"] = problematic_fields
    
    except Exception as e:
        print_with_timestamp(f"Error examining chanlocs structure: {e}")
        chanlocs_info["error"] = str(e)
    
    return chanlocs_info


def safe_load_sample(f, max_depth, current_path, path="/"):
    """Safely load a sample of the HDF5 file structure"""
    result = {}
    
    try:
        # Get item at current path
        item = f[path]
        
        if isinstance(item, h5py.Group):
            # Get keys and sort them
            keys = sorted(list(item.keys()))
            
            # Only process first 10 keys to avoid long execution
            if len(keys) > 10:
                print_with_timestamp(f"Group {path} has {len(keys)} keys, sampling first 10")
                keys = keys[:10]
                result["_note"] = f"Limited to first 10 of {len(keys)} keys"
            
            # Process each key
            for key in keys:
                current_path[0] = f"{path}{key}/"
                
                # Get subitem
                subpath = f"{path}{key}" if path.endswith("/") else f"{path}/{key}"
                
                # Check depth
                if subpath.count("/") <= max_depth:
                    result[key] = safe_load_sample(f, max_depth, current_path, subpath)
                else:
                    result[key] = {"_note": f"Max depth {max_depth} reached"}
        
        elif isinstance(item, h5py.Dataset):
            # Get basic info
            shape = item.shape
            dtype = item.dtype
            size = np.prod(shape)
            
            # Create base info
            result["_info"] = {
                "shape": shape,
                "dtype": str(dtype),
                "size": size
            }
            
            # Special cases
            if dtype == h5py.ref_dtype:
                result["_info"]["is_reference"] = True
                
                # For small reference arrays, try to get a sample
                if size > 0 and size < 10:
                    sample = []
                    for i in range(min(3, size)):
                        try:
                            ref = item[i]
                            if isinstance(ref, h5py.Reference) and ref:
                                ref_item = f[ref]
                                if isinstance(ref_item, h5py.Dataset) and np.prod(ref_item.shape) < 100:
                                    data = ref_item[()]
                                    if isinstance(data, bytes):
                                        data = data.decode('utf-8')
                                    elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U']:
                                        if data.dtype.kind == 'S':
                                            data = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data]
                                        else:
                                            data = data.tolist()
                                    sample.append(data)
                                else:
                                    sample.append(f"<ref to {ref_item.name}: shape={ref_item.shape}, dtype={ref_item.dtype}>")
                            else:
                                sample.append(None)
                        except Exception as e:
                            sample.append(f"<error: {e}>")
                    result["_sample"] = sample
            
            # For uint16 arrays (potential string data)
            elif dtype == np.uint16 and size < 1000:
                try:
                    data = item[()]
                    # Try to convert to string
                    text = ''.join(chr(c) for c in data.flatten() if c != 0)
                    result["_uint16_as_string"] = text
                    result["_sample"] = data.tolist() if size < 10 else data[:10].tolist()
                except Exception as e:
                    result["_error"] = str(e)
            
            # For normal datasets, get a sample if small enough
            elif size < 1000:
                try:
                    data = item[:]
                    
                    # Handle strings
                    if dtype.kind in ['S', 'U']:
                        if size < 10:
                            if dtype.kind == 'S':
                                data = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data]
                            result["_sample"] = data.tolist()
                        else:
                            if dtype.kind == 'S':
                                sample = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data[:5]]
                            else:
                                sample = data[:5].tolist()
                            result["_sample"] = sample
                    
                    # Handle numeric data
                    elif size < 10:
                        result["_sample"] = data.tolist()
                    else:
                        result["_sample"] = data[:5].tolist()
                
                except Exception as e:
                    result["_error"] = str(e)
    
    except Exception as e:
        result["_error"] = str(e)
    
    return result


def print_diagnostic_summary(diagnostic_info):
    """Print a summary of the diagnostic information"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    if "error" in diagnostic_info:
        print(f"ERROR: {diagnostic_info['error']}")
        return
    
    # Basic file info
    print(f"File size: {diagnostic_info.get('file_size_mb', 'unknown'):.2f} MB")
    print(f"File type: {diagnostic_info.get('file_type', 'unknown')}")
    print(f"Top-level keys: {', '.join(diagnostic_info.get('top_keys', []))}")
    
    # Check for EEG info
    if "eeg_info" in diagnostic_info:
        eeg_info = diagnostic_info["eeg_info"]
        
        # Print scalar values
        if "scalar_values" in eeg_info:
            print("\nEEG Parameters:")
            for key, value in eeg_info["scalar_values"].items():
                print(f"  {key}: {value}")
        
        # List problematic structures
        if "problematic_structures" in eeg_info and eeg_info["problematic_structures"]:
            print("\nPROBLEMATIC STRUCTURES DETECTED:")
            for issue in eeg_info["problematic_structures"]:
                print(f"  - {issue}")
        
        # List large arrays
        if "large_arrays" in eeg_info and eeg_info["large_arrays"]:
            print("\nLarge data arrays:")
            for array in eeg_info["large_arrays"]:
                print(f"  - {array}")
    
    print("\nRECOMMENDATIONS:")
    
    problematic_paths = []
    
    # Get event issues
    if ("eeg_info" in diagnostic_info and 
        "event" in diagnostic_info["eeg_info"] and 
        "problematic_fields" in diagnostic_info["eeg_info"]["event"] and 
        diagnostic_info["eeg_info"]["event"]["problematic_fields"]):
        problematic_paths.extend(diagnostic_info["eeg_info"]["event"]["problematic_fields"])
    
    # Get etc issues
    if ("eeg_info" in diagnostic_info and 
        "etc" in diagnostic_info["eeg_info"] and 
        "problematic_fields" in diagnostic_info["eeg_info"]["etc"] and 
        diagnostic_info["eeg_info"]["etc"]["problematic_fields"]):
        problematic_paths.extend(diagnostic_info["eeg_info"]["etc"]["problematic_fields"])
    
    # Get chanlocs issues
    if ("eeg_info" in diagnostic_info and 
        "chanlocs" in diagnostic_info["eeg_info"] and 
        "problematic_fields" in diagnostic_info["eeg_info"]["chanlocs"] and 
        diagnostic_info["eeg_info"]["chanlocs"]["problematic_fields"]):
        problematic_paths.extend(diagnostic_info["eeg_info"]["chanlocs"]["problematic_fields"])
    
    if problematic_paths:
        print("1. Use a specialized loader that handles these problematic structures:")
        for path in problematic_paths:
            print(f"   - {path}")
        
        print("\n2. Implement circuit breakers in recursive functions")
        print("3. Use chunked loading for large reference arrays")
        print("4. Add explicit timeout protection to prevent hangs")
    else:
        print("No major issues detected in file structure.")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Safely inspect HDF5/MATLAB v7.3 files")
    parser.add_argument("filename", help="Path to the HDF5/MATLAB file to inspect")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    parser.add_argument("--depth", type=int, default=5, help="Maximum recursion depth (default: 5)")
    
    args = parser.parse_args()
    
    try:
        # Check if file exists
        if not os.path.isfile(args.filename):
            print(f"Error: File not found: {args.filename}")
            return 1
        
        # Run the inspection
        diagnostic_info = inspect_hdf5_file(args.filename, args.depth, args.timeout)
        
        # Print a summary
        print_diagnostic_summary(diagnostic_info)
        
        # Save the diagnostic info to a file
        output_file = f"{os.path.splitext(args.filename)[0]}_diagnostic.txt"
        with open(output_file, 'w') as f:
            f.write(f"HDF5 File Diagnostic: {args.filename}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write nested dictionaries in a readable format
            def write_dict(d, indent=0):
                for key, value in d.items():
                    if isinstance(value, dict):
                        f.write(" " * indent + f"{key}:\n")
                        write_dict(value, indent + 2)
                    elif isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            f.write(" " * indent + f"{key}:\n")
                            for i, item in enumerate(value):
                                f.write(" " * (indent + 2) + f"[{i}]:\n")
                                write_dict(item, indent + 4)
                        else:
                            f.write(" " * indent + f"{key}: {value}\n")
                    else:
                        f.write(" " * indent + f"{key}: {value}\n")
            
            write_dict(diagnostic_info)
        
        print(f"\nDetailed diagnostic information saved to: {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())