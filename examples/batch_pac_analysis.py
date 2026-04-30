"""
batch_pac_analysis.py

Batch processing script to run PAC analysis on multiple subjects with different
amplitude frequency ranges. This script calls hdEEG_pac_detector.py for each
subject and frequency range combination.

Usage:
    1. Edit the SUBJECTS and AMP_FREQ_RANGES lists below
    2. Run: python batch_pac_analysis.py
"""

import os
import sys
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION SECTION - Edit these parameters
# ============================================================================

# Base directory containing all subject folders
BASE_DIR = r"S:\Sleep\2. STAFF\Tancy\OSA CPAP Events"

# List of subjects to process
# Each subject should have a folder structure like: BASE_DIR/subject/BL/ or BASE_DIR/subject/TR/
SUBJECTS = [
    "sub-07DP",
    "sub-08RG",
    "sub-09AS",
    "sub-10DS",
    "sub-11NR",
    "sub-12SM",
    "sub-13PR",
    "sub-14CF"
    # Add more subjects here
]

# Subfolders to look for in each subject directory (e.g., 'BL', 'TR')
# The script will automatically process any subfolder that contains pac_channels.csv
# Set to None to auto-discover all subfolders, or specify a list like ['BL', 'TR']
SUBFOLDERS = None  # Auto-discover all subfolders
# SUBFOLDERS = ['BL', 'TR']  # Or specify which subfolders to check

# Amplitude frequency ranges to analyze
# Each subject will be analyzed with both frequency ranges
AMP_FREQ_RANGES = [
    [11.0, 13.0],  # Slow spindles
    [13.0, 16.0],  # Fast spindles
]

# Common parameters for all analyses
COMMON_PARAMS = {
    # Note: edf_file and xml_file will be constructed dynamically based on subfolder
    # Format: OSA_{subfolder}{subject}_clean_rebuilt.set
    # Example: OSA_BL10DS_clean_rebuilt.set or OSA_TR10DS_clean_rebuilt.set
    'db_file': "neural_events.db",
    'channels_file': "pac_channels.csv",
    'sw_method': "Staresina2015",
    'spindle_method': "Moelle2011",
    'stages': ['NREM2', 'NREM3'],
    'phase_freq': [1.0, 4.54545454545455],
}

# Optional: Filter events by frequency range
SW_FREQ_RANGE = None  # e.g., [0.5, 2.0] or None
SPINDLE_FREQ_RANGE = None  # e.g., [11, 16] or None

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def run_pac_analysis(subject, subfolder, root_dir, amp_freq_range):
    """
    Run PAC analysis for a single subject with specified amplitude frequency range.
    
    Parameters
    ----------
    subject : str
        Subject identifier (e.g., "sub-10DS")
    subfolder : str
        Subfolder name (e.g., "BL", "TR")
    root_dir : str
        Root directory for the subject's data
    amp_freq_range : list
        Amplitude frequency range [low, high]
    """
    # Extract subject code (e.g., "10DS" from "sub-10DS")
    subject_code = subject.replace("sub-", "")
    
    # Construct file names based on subfolder and subject
    # Format: OSA_{subfolder}{subject}_clean_rebuilt.set
    # Example: OSA_BL10DS_clean_rebuilt.set or OSA_TR10DS_clean_rebuilt.set
    edf_file = f"OSA_{subfolder}{subject_code}_clean_rebuilt.set"
    xml_file = f"OSA_{subfolder}{subject_code}_clean_rebuilt.xml"
    
    # Create output directory name based on frequency range
    freq_label = f"amp_{amp_freq_range[0]:.1f}-{amp_freq_range[1]:.1f}Hz"
    output_dir = os.path.join(root_dir, "wonambi", "pac_results", freq_label)
    
    # Build command
    script_path = os.path.join(os.path.dirname(__file__), "hdEEG_pac_detector.py")
    
    cmd = [
        sys.executable,  # Python interpreter
        script_path,
        "--root_dir", root_dir,
        "--edf_file", edf_file,
        "--xml_file", xml_file,
        "--db_file", COMMON_PARAMS['db_file'],
        "--channels_file", COMMON_PARAMS['channels_file'],
        "--sw_method", COMMON_PARAMS['sw_method'],
        "--spindle_method", COMMON_PARAMS['spindle_method'],
        "--stages"] + COMMON_PARAMS['stages'] + [
        "--phase_freq", str(COMMON_PARAMS['phase_freq'][0]), str(COMMON_PARAMS['phase_freq'][1]),
        "--amp_freq", str(amp_freq_range[0]), str(amp_freq_range[1]),
        "--output_dir", output_dir,
    ]
    
    # Add optional frequency range filters
    if SW_FREQ_RANGE:
        cmd.extend(["--sw_freq_range", str(SW_FREQ_RANGE[0]), str(SW_FREQ_RANGE[1])])
    if SPINDLE_FREQ_RANGE:
        cmd.extend(["--spindle_freq_range", str(SPINDLE_FREQ_RANGE[0]), str(SPINDLE_FREQ_RANGE[1])])
    
    # Print command info
    print("\n" + "=" * 80)
    print(f"Processing: {subject}")
    print(f"Amplitude frequency range: {amp_freq_range[0]}-{amp_freq_range[1]} Hz")
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Successfully completed: {subject} ({freq_label})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {subject} ({freq_label}): {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error for {subject} ({freq_label}): {e}")
        return False


def discover_subfolders(subject_dir):
    """
    Discover subfolders in a subject directory that contain pac_channels.csv.
    
    Parameters
    ----------
    subject_dir : str
        Path to subject directory
        
    Returns
    -------
    list
        List of subfolder names that contain pac_channels.csv
    """
    valid_subfolders = []
    
    if not os.path.exists(subject_dir):
        return valid_subfolders
    
    # Get list of subfolders to check
    if SUBFOLDERS is None:
        # Auto-discover: check all subdirectories
        try:
            all_items = os.listdir(subject_dir)
            subfolders_to_check = [item for item in all_items
                                   if os.path.isdir(os.path.join(subject_dir, item))]
        except Exception as e:
            print(f"Error listing directory {subject_dir}: {e}")
            return valid_subfolders
    else:
        # Use specified subfolders
        subfolders_to_check = SUBFOLDERS
    
    # Check each subfolder for pac_channels.csv
    for subfolder in subfolders_to_check:
        subfolder_path = os.path.join(subject_dir, subfolder)
        if not os.path.exists(subfolder_path):
            continue
            
        channels_file = os.path.join(subfolder_path, COMMON_PARAMS['channels_file'])
        if os.path.exists(channels_file):
            valid_subfolders.append(subfolder)
    
    return valid_subfolders


def main():
    """Main batch processing function."""
    print("=" * 80)
    print("PAC Analysis Batch Processing")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Number of subjects: {len(SUBJECTS)}")
    print(f"Amplitude frequency ranges: {AMP_FREQ_RANGES}")
    print(f"Subfolder mode: {'Auto-discover' if SUBFOLDERS is None else SUBFOLDERS}")
    print("=" * 80)
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # Process each subject
    for subject in SUBJECTS:
        subject_dir = os.path.join(BASE_DIR, subject)
        
        # Check if subject directory exists
        if not os.path.exists(subject_dir):
            print(f"\n⚠ Warning: Subject directory not found: {subject_dir}")
            results['skipped'].append(f"{subject} (subject directory not found)")
            continue
        
        # Discover valid subfolders (those with pac_channels.csv)
        valid_subfolders = discover_subfolders(subject_dir)
        
        if not valid_subfolders:
            print(f"\n⚠ Warning: No subfolders with {COMMON_PARAMS['channels_file']} found for {subject}")
            results['skipped'].append(f"{subject} (no valid subfolders)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Subject: {subject}")
        print(f"Valid subfolders found: {valid_subfolders}")
        print(f"{'='*80}")
        
        # Process each valid subfolder
        for subfolder in valid_subfolders:
            root_dir = os.path.join(subject_dir, subfolder)
            
            # Process each amplitude frequency range
            for amp_freq_range in AMP_FREQ_RANGES:
                freq_label = f"{amp_freq_range[0]:.1f}-{amp_freq_range[1]:.1f}Hz"
                analysis_id = f"{subject}/{subfolder} ({freq_label})"
                
                success = run_pac_analysis(subject, subfolder, root_dir, amp_freq_range)
                
                if success:
                    results['success'].append(analysis_id)
                else:
                    results['failed'].append(analysis_id)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(results['success'])}")
    for item in results['success']:
        print(f"  - {item}")
    
    if results['failed']:
        print(f"\n✗ Failed: {len(results['failed'])}")
        for item in results['failed']:
            print(f"  - {item}")
    
    if results['skipped']:
        print(f"\n⊘ Skipped: {len(results['skipped'])}")
        for item in results['skipped']:
            print(f"  - {item}")
    
    print("\n" + "=" * 80)
    print("Batch processing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()