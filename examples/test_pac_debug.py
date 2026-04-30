#!/usr/bin/env python3
"""
test_pac_debug.py

A simple test script to help debug the PAC summary issue.
This script will run both the debug tool and the summary creation tool
on the specified directory to help identify why E2 is missing.
"""

import os
import sys
import subprocess

def run_debug_analysis(results_dir):
    """Run the debug analysis on the PAC results directory."""
    
    print("=" * 80)
    print("RUNNING PAC DEBUG ANALYSIS")
    print("=" * 80)
    
    # Run the debug script
    debug_script = os.path.join(os.path.dirname(__file__), 'debug_pac_files.py')
    
    try:
        cmd = [sys.executable, debug_script, '--results_dir', results_dir]
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running debug script: {e}")

def run_summary_creation(results_dir):
    """Run the summary creation with verbose output."""
    
    print("\n" + "=" * 80)
    print("RUNNING PAC SUMMARY CREATION")
    print("=" * 80)
    
    # Run the summary creation script
    summary_script = os.path.join(os.path.dirname(__file__), 'create_pac_summary.py')
    
    try:
        cmd = [sys.executable, summary_script, '--results_dir', results_dir, '--process_all']
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running summary script: {e}")

def check_output_files(results_dir):
    """Check what output files were created."""
    
    print("\n" + "=" * 80)
    print("CHECKING OUTPUT FILES")
    print("=" * 80)
    
    import glob
    
    # Look for summary CSV files
    summary_files = glob.glob(os.path.join(results_dir, "pac_summary_*.csv"))
    
    if summary_files:
        print(f"Found {len(summary_files)} summary CSV files:")
        
        for summary_file in summary_files:
            filename = os.path.basename(summary_file)
            print(f"\n  File: {filename}")
            
            try:
                import pandas as pd
                df = pd.read_csv(summary_file)
                
                print(f"    Rows: {len(df)}")
                print(f"    Columns: {list(df.columns)}")
                
                if 'Channel' in df.columns:
                    channels = sorted(df['Channel'].unique())
                    print(f"    Channels: {channels}")
                    
                    if 'E2' in channels:
                        print(f"    ✓ E2 is present")
                    else:
                        print(f"    ✗ E2 is MISSING")
                        print(f"    Available channels: {channels}")
                
            except Exception as e:
                print(f"    Error reading file: {e}")
    else:
        print("No summary CSV files found")

def main():
    """Main function to run all debugging steps."""
    
    # Default directory - user can modify this
    default_dir = "/Volumes/Sleep/Sleep/2. STAFF/Tancy/OSA CPAP Events/sub-15DC/BL/wonambi/pac_results/Staresina2015_paired_Moelle2011/NREM2NREM3"
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = default_dir
        print(f"Using default directory: {results_dir}")
        print("To use a different directory, run: python test_pac_debug.py /path/to/your/directory")
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory does not exist: {results_dir}")
        return
    
    print(f"Analyzing PAC results in: {results_dir}")
    
    # Step 1: Run debug analysis
    run_debug_analysis(results_dir)
    
    # Step 2: Run summary creation
    run_summary_creation(results_dir)
    
    # Step 3: Check output files
    check_output_files(results_dir)
    
    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()