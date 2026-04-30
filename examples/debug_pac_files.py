"""
debug_pac_files.py

A diagnostic script to help debug why certain channels are missing from PAC summary files.
This script will show detailed information about what files are found and how they're processed.
"""

import os
import glob
import re
import pandas as pd

def debug_pac_files(results_dir):
    """Debug PAC files in the directory to see what's happening with missing channels."""
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return
    
    print(f"=== DEBUGGING PAC FILES IN: {results_dir} ===\n")
    
    # Find all PAC parameter files
    pattern = "*_pac_parameters.csv"
    all_files = glob.glob(os.path.join(results_dir, pattern))
    
    print(f"1. TOTAL FILES FOUND: {len(all_files)}")
    if not all_files:
        print("   No PAC parameter files found!")
        return
    
    # Show all files
    print("\n2. ALL PAC PARAMETER FILES:")
    for i, file_path in enumerate(sorted(all_files), 1):
        filename = os.path.basename(file_path)
        print(f"   {i:2d}. {filename}")
    
    # Extract frequency information and group by combinations
    freq_pattern = r'pha-([0-9.]+)-([0-9.]+)Hz.*amp-([0-9.]+)-([0-9.]+)Hz'
    freq_combinations = {}
    unmatched_files = []
    
    print(f"\n3. FREQUENCY EXTRACTION:")
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
            
            # Extract channel name
            channel = filename.split('_')[0]
            print(f"   ✓ {filename} -> Channel: {channel}, Phase: {phase_freq}, Amp: {amp_freq}")
        else:
            unmatched_files.append(filename)
            print(f"   ✗ {filename} -> Could not extract frequency info")
    
    if unmatched_files:
        print(f"\n   WARNING: {len(unmatched_files)} files did not match frequency pattern!")
    
    # Show frequency combinations
    print(f"\n4. FREQUENCY COMBINATIONS FOUND: {len(freq_combinations)}")
    for i, (freq_key, files) in enumerate(freq_combinations.items(), 1):
        phase_freq, amp_freq = freq_key
        print(f"\n   Combination {i}: Phase {phase_freq[0]}-{phase_freq[1]}Hz, Amp {amp_freq[0]}-{amp_freq[1]}Hz")
        print(f"   Files: {len(files)}")
        
        # Extract and show channels for this combination
        channels = []
        for file_path in files:
            filename = os.path.basename(file_path)
            channel = filename.split('_')[0]
            channels.append(channel)
        
        channels.sort()
        print(f"   Channels: {channels}")
        
        # Check specifically for E2
        if 'E2' in channels:
            print(f"   ✓ E2 is present in this combination")
        else:
            print(f"   ✗ E2 is MISSING from this combination")
            
            # Look for E2 files that might not match the pattern
            e2_files = [f for f in all_files if os.path.basename(f).startswith('E2_')]
            if e2_files:
                print(f"   Found {len(e2_files)} E2 files in directory:")
                for e2_file in e2_files:
                    e2_filename = os.path.basename(e2_file)
                    print(f"     - {e2_filename}")
                    
                    # Check if this E2 file matches our frequency pattern
                    e2_match = re.search(freq_pattern, e2_filename)
                    if e2_match:
                        e2_phase_low, e2_phase_high, e2_amp_low, e2_amp_high = e2_match.groups()
                        e2_phase_freq = (float(e2_phase_low), float(e2_phase_high))
                        e2_amp_freq = (float(e2_amp_low), float(e2_amp_high))
                        print(f"       Frequencies: Phase {e2_phase_freq}, Amp {e2_amp_freq}")
                        
                        if (e2_phase_freq, e2_amp_freq) == freq_key:
                            print(f"       ✓ This E2 file SHOULD be included but wasn't!")
                        else:
                            print(f"       - This E2 file has different frequencies")
                    else:
                        print(f"       ✗ This E2 file doesn't match frequency pattern")
            else:
                print(f"   No E2 files found in directory at all")
    
    # Check file contents for a few examples
    print(f"\n5. SAMPLE FILE CONTENT CHECK:")
    for freq_key, files in list(freq_combinations.items())[:2]:  # Check first 2 combinations
        phase_freq, amp_freq = freq_key
        print(f"\n   Checking combination: Phase {phase_freq[0]}-{phase_freq[1]}Hz, Amp {amp_freq[0]}-{amp_freq[1]}Hz")
        
        for file_path in files[:3]:  # Check first 3 files in this combination
            filename = os.path.basename(file_path)
            channel = filename.split('_')[0]
            
            try:
                df = pd.read_csv(file_path)
                print(f"     {channel}: {len(df)} rows, columns: {list(df.columns)}")
                if df.empty:
                    print(f"       WARNING: File is empty!")
            except Exception as e:
                print(f"     {channel}: ERROR reading file - {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug PAC files to find missing channels')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing PAC parameter files')
    
    args = parser.parse_args()
    debug_pac_files(args.results_dir)

if __name__ == "__main__":
    main()