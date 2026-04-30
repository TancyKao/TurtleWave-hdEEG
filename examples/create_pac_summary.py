"""
create_pac_summary.py

A standalone script to create summary CSV files from existing PAC analysis results.
This script scans a directory for individual channel PAC parameter files and combines
them into summary CSV files. It can automatically detect all frequency combinations
and create separate summary files for each.

Usage:
    # Process all frequency combinations automatically (recommended):
    python create_pac_summary.py --results_dir "/path/to/pac/results" --process_all
    
    # Process specific frequency combination:
    python create_pac_summary.py --results_dir "/path/to/pac/results" --phase_freq 1.0 4.54545454545455 --amp_freq 13 16
    
    # Custom output file:
    python create_pac_summary.py --results_dir "/path/to/pac/results" --phase_freq 1.0 4.54545454545455 --amp_freq 13 16 --output_file "/path/to/custom_summary.csv"

Examples:
    # Process all frequency combinations in a directory:
    python create_pac_summary.py --results_dir "/Users/user/pac_results/Staresina2015_paired_Moelle2011/NREM2NREM3" --process_all
    
    # This will automatically find all unique frequency combinations (e.g., 11-13Hz and 13-16Hz)
    # and create separate summary CSV files for each combination.
"""

import os
import sys
import argparse
import pandas as pd
import glob
import re
from pathlib import Path

def find_all_frequency_combinations(results_dir):
    """
    Find all unique frequency combinations in PAC parameter files.
    
    Parameters
    ----------
    results_dir : str
        Directory containing PAC results
        
    Returns
    -------
    dict
        Dictionary mapping frequency combinations to file lists
        Format: {(phase_freq, amp_freq): [file_list]}
    """
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return {}
    
    print(f"Scanning directory for all frequency combinations: {results_dir}")
    
    # Find all PAC parameter files
    pattern = "*_pac_parameters.csv"
    all_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not all_files:
        print("No PAC parameter files found")
        return {}
    
    print(f"Found {len(all_files)} PAC parameter files")
    
    # Extract frequency information from all files
    freq_combinations = {}
    freq_pattern = r'pha-([0-9.]+)-([0-9.]+)Hz.*amp-([0-9.]+)-([0-9.]+)Hz'
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = re.search(freq_pattern, filename)
        
        if match:
            phase_low, phase_high, amp_low, amp_high = match.groups()
            phase_freq = (float(phase_low), float(phase_high))
            amp_freq = (float(amp_low), float(amp_high))
            freq_key = (phase_freq, amp_freq)
            
            if freq_key not in freq_combinations:
                freq_combinations[freq_key] = []
            freq_combinations[freq_key].append(file_path)
            
            # Debug: Show which channel this file belongs to
            channel = filename.split('_')[0]
            print(f"  Found: {channel} -> Phase: {phase_freq}, Amp: {amp_freq}")
        else:
            print(f"Warning: Could not extract frequency info from {filename}")
    
    # Print summary of found combinations
    print(f"\nFound {len(freq_combinations)} unique frequency combinations:")
    for i, (freq_key, files) in enumerate(freq_combinations.items(), 1):
        phase_freq, amp_freq = freq_key
        print(f"  {i}. Phase: {phase_freq[0]}-{phase_freq[1]}Hz, Amplitude: {amp_freq[0]}-{amp_freq[1]}Hz ({len(files)} files)")
    
    return freq_combinations

def find_pac_files(results_dir, phase_freq=None, amp_freq=None, auto_detect=False):
    """
    Find PAC parameter files in the results directory.
    
    Parameters
    ----------
    results_dir : str
        Directory containing PAC results
    phase_freq : tuple
        Phase frequency range (low, high)
    amp_freq : tuple
        Amplitude frequency range (low, high)
    auto_detect : bool
        If True, automatically detect frequency ranges from filenames
        
    Returns
    -------
    list
        List of PAC parameter file paths
    dict
        Dictionary with detected frequency information
    """
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return [], {}
    
    print(f"Scanning directory: {results_dir}")
    
    # Look for PAC parameter files
    if auto_detect:
        # Find all frequency combinations
        freq_combinations = find_all_frequency_combinations(results_dir)
        
        if not freq_combinations:
            return [], {}
        
        # Return the first combination for backward compatibility
        first_key = list(freq_combinations.keys())[0]
        first_files = freq_combinations[first_key]
        phase_freq, amp_freq = first_key
        
        detected_freq = {
            'phase_freq': phase_freq,
            'amp_freq': amp_freq
        }
        
        return first_files, detected_freq
    
    else:
        # Use specified frequency ranges
        if phase_freq is None or amp_freq is None:
            print("Error: When not using auto-detection, both phase_freq and amp_freq must be specified")
            return [], {}
        
        # Create filename pattern based on frequency ranges
        ph_str = f"{phase_freq[0]}-{phase_freq[1]}Hz"
        amp_str = f"{amp_freq[0]}-{amp_freq[1]}Hz"
        
        # Try different filename patterns
        patterns = [
            f"*_slowwave_spindle_coupling_pha-{ph_str}-fixed_amp-{amp_str}-fixed_pac_parameters.csv",
            f"*_slowwave_pha-{ph_str}-fixed_amp-{amp_str}-fixed_pac_parameters.csv",
            f"*_spindle_pha-{ph_str}-fixed_amp-{amp_str}-fixed_pac_parameters.csv",
            f"*_pha-{ph_str}-fixed_amp-{amp_str}-fixed_pac_parameters.csv"
        ]
        
        files = []
        for pattern in patterns:
            found_files = glob.glob(os.path.join(results_dir, pattern))
            files.extend(found_files)
            if found_files:
                print(f"Found {len(found_files)} files with pattern: {pattern}")
        
        freq_info = {
            'phase_freq': phase_freq,
            'amp_freq': amp_freq
        }
        
        return files, freq_info

def extract_channel_from_filename(filename):
    """Extract channel name from PAC parameter filename."""
    basename = os.path.basename(filename)
    # Assume channel is the first part before underscore
    # Example: E1_slowwave_spindle_coupling_... -> E1
    channel = basename.split('_')[0]
    return channel

def create_summary_csv(pac_files, output_file, freq_info):
    """
    Create summary CSV from individual PAC parameter files.
    
    Parameters
    ----------
    pac_files : list
        List of PAC parameter file paths
    output_file : str
        Output CSV file path
    freq_info : dict
        Dictionary containing frequency information
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    
    if not pac_files:
        print("No PAC files to process")
        return False
    
    print(f"Processing {len(pac_files)} PAC parameter files...")
    
    all_data = []
    
    for file_path in pac_files:
        try:
            # Extract channel name from filename
            channel = extract_channel_from_filename(file_path)
            print(f"Processing {channel}: {os.path.basename(file_path)}")
            
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                print(f"  Error: File does not exist: {file_path}")
                continue
                
            if os.path.getsize(file_path) == 0:
                print(f"  Warning: File is empty: {file_path}")
                continue
            
            # Read PAC parameters
            df = pd.read_csv(file_path)
            print(f"  File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            if df.empty:
                print(f"  Warning: DataFrame is empty after loading: {file_path}")
                continue
            
            # Show available columns for debugging
            print(f"  Available columns: {list(df.columns)}")
            
            # Process each row in the file (usually just one row)
            for idx, row in df.iterrows():
                print(f"  Processing row {idx}")
                
                # Create summary row
                summary_row = {
                    'Channel': channel,
                    'Phase_Freq': f"{freq_info['phase_freq'][0]}-{freq_info['phase_freq'][1]}",
                    'Amp_Freq': f"{freq_info['amp_freq'][0]}-{freq_info['amp_freq'][1]}"
                }
                
                # Copy PAC metrics
                metric_columns = [
                    'mi_raw', 'mi_norm', 'median_mi_pval',
                    'preferred_phase_rad', 'preferred_phase_deg',
                    'mean_vector_length', 'rho', 'rayleigh_z', 'rayleigh_p'
                ]
                
                found_metrics = []
                for col in metric_columns:
                    if col in row:
                        summary_row[col] = row[col]
                        found_metrics.append(col)
                    else:
                        summary_row[col] = float('nan')
                
                print(f"  Found metrics: {found_metrics}")
                all_data.append(summary_row)
                print(f"  Added row for {channel}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        print("No valid data found in PAC files")
        return False
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_data)
    
    # Sort by channel name
    summary_df = summary_df.sort_values('Channel')
    
    # Write to CSV
    try:
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary CSV created successfully!")
        print(f"Output file: {output_file}")
        print(f"Channels: {len(summary_df['Channel'].unique())}")
        print(f"Total rows: {len(summary_df)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(summary_df.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Error writing summary CSV: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create summary CSV from PAC analysis results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing PAC parameter files')
    parser.add_argument('--output_file', type=str,
                       help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--phase_freq', type=float, nargs=2,
                       help='Phase frequency range (Hz)')
    parser.add_argument('--amp_freq', type=float, nargs=2,
                       help='Amplitude frequency range (Hz)')
    parser.add_argument('--process_all', action='store_true',
                       help='Process all frequency combinations found (creates multiple CSV files)')
    parser.add_argument('--auto_detect', action='store_true',
                       help='Automatically detect frequency ranges from filenames (legacy option)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.auto_detect and not args.process_all and (args.phase_freq is None or args.amp_freq is None):
        print("Error: Either use --process_all, --auto_detect, or specify both --phase_freq and --amp_freq")
        return
    
    # Process all frequency combinations
    if args.process_all or args.auto_detect:
        freq_combinations = find_all_frequency_combinations(args.results_dir)
        
        if not freq_combinations:
            print("No PAC parameter files found")
            return
        
        print(f"\nProcessing {len(freq_combinations)} frequency combinations...")
        
        success_count = 0
        for freq_key, pac_files in freq_combinations.items():
            phase_freq, amp_freq = freq_key
            
            freq_info = {
                'phase_freq': phase_freq,
                'amp_freq': amp_freq
            }
            
            # Auto-generate output filename for each combination
            ph_str = f"{phase_freq[0]}-{phase_freq[1]}Hz"
            amp_str = f"{amp_freq[0]}-{amp_freq[1]}Hz"
            output_filename = f"pac_summary_{ph_str}_{amp_str}.csv"
            output_file = os.path.join(args.results_dir, output_filename)
            
            print(f"\n--- Processing Phase: {phase_freq[0]}-{phase_freq[1]}Hz, Amplitude: {amp_freq[0]}-{amp_freq[1]}Hz ---")
            
            # Create summary CSV for this frequency combination
            success = create_summary_csv(pac_files, output_file, freq_info)
            
            if success:
                success_count += 1
                print(f"✓ Created: {output_filename}")
            else:
                print(f"✗ Failed: {output_filename}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Successfully created {success_count}/{len(freq_combinations)} summary CSV files")
        
    else:
        # Process single frequency combination
        pac_files, freq_info = find_pac_files(args.results_dir,
                                            tuple(args.phase_freq),
                                            tuple(args.amp_freq))
        
        if not pac_files:
            print("No PAC parameter files found")
            return
        
        if not freq_info:
            print("Could not determine frequency information")
            return
        
        # Determine output file
        if args.output_file is None:
            # Auto-generate output filename
            ph_str = f"{freq_info['phase_freq'][0]}-{freq_info['phase_freq'][1]}Hz"
            amp_str = f"{freq_info['amp_freq'][0]}-{freq_info['amp_freq'][1]}Hz"
            output_filename = f"pac_summary_{ph_str}_{amp_str}.csv"
            args.output_file = os.path.join(args.results_dir, output_filename)
        
        # Create summary CSV
        success = create_summary_csv(pac_files, args.output_file, freq_info)
        
        if success:
            print(f"\n✓ Summary CSV creation completed successfully!")
        else:
            print(f"\n✗ Summary CSV creation failed!")

if __name__ == "__main__":
    main()