"""
hdEEG_pac_detector.py
This script is designed to analyze phase-amplitude coupling (PAC) in high-density EEG (hdEEG) data.
It uses the TurtleWave-hdEEG library and an integrated PAC processor to analyze coupling between
different frequency bands, specifically focusing on slow wave-spindle coupling from a SQLite database.

Functions:
    - analyze_pac: Analyzes phase-amplitude coupling in EEG data based on specified parameters.
    - export_pac_parameters_to_csv: Exports PAC parameters to a CSV file.
    - generate_comodulogram: Generates a comodulogram for visualizing PAC across frequency ranges.

Workflow:
    1. Define file paths for the EEG dataset, annotations, and database.
    2. Load the dataset and annotations.
    3. Create an instance of the ParalPAC class for analyzing PAC.
    4. Specify parameters for PAC analysis, including channels, frequency bands, and methods.
    5. Run the PAC analysis on slow wave-spindle pairs from the database.
    6. Export PAC parameters to CSV files for further analysis.
"""

import os
import sys
import logging
import argparse
import sqlite3
from wonambi.dataset import Dataset as WonambiDataset
#from wonambi.attr import Annotations
from turtlewave_hdEEG.utils import read_channels_from_csv
from turtlewave_hdEEG import ParalPAC, CustomAnnotations

def list_available_methods(db_path):
    """List available detection methods in the database for both slow waves and spindles."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return None, None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get slow wave methods
        cursor.execute("SELECT DISTINCT method FROM events WHERE event_type = 'slow_wave'")
        sw_methods = [row[0] for row in cursor.fetchall()]
        
        # Get spindle methods
        cursor.execute("SELECT DISTINCT method FROM events WHERE event_type = 'spindle'")
        spindle_methods = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        print("\nAvailable slow wave detection methods:")
        for i, method in enumerate(sw_methods):
            print(f"  {i+1}. {method}")
        
        print("\nAvailable spindle detection methods:")
        for i, method in enumerate(spindle_methods):
            print(f"  {i+1}. {method}")
        
        return sw_methods, spindle_methods
    
    except Exception as e:
        print(f"Error accessing database: {e}")
        return None, None

def get_event_stats(db_path, sw_method=None, spindle_method=None, channel=None, stage=None, 
                    sw_freq_range=None, spindle_freq_range=None):
    """Get statistics about available events in the database."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query for slow waves
        sw_query = "SELECT channel, stage, COUNT(*) FROM events WHERE event_type = 'slow_wave'"
        sw_params = []
        
        if sw_method:
            sw_query += " AND method = ?"
            sw_params.append(sw_method)
        
        if channel:
            sw_query += " AND channel = ?"
            sw_params.append(channel)
        
        if stage:
            if isinstance(stage, list):
                placeholders = ', '.join(['?' for _ in stage])
                sw_query += f" AND stage IN ({placeholders})"
                sw_params.extend(stage)
            else:
                sw_query += " AND stage = ?"
                sw_params.append(stage)
        
        # Add frequency range filtering for slow waves
        if sw_freq_range and len(sw_freq_range) == 2:
            sw_query += " AND freq_lower >= ? AND freq_upper <= ?"
            sw_params.extend(sw_freq_range)
        
        sw_query += " GROUP BY channel, stage"
        
        # Build query for spindles
        sp_query = "SELECT channel, stage, COUNT(*) FROM events WHERE event_type = 'spindle'"
        sp_params = []
        
        if spindle_method:
            sp_query += " AND method = ?"
            sp_params.append(spindle_method)
        
        if channel:
            sp_query += " AND channel = ?"
            sp_params.append(channel)
        
        if stage:
            if isinstance(stage, list):
                placeholders = ', '.join(['?' for _ in stage])
                sp_query += f" AND stage IN ({placeholders})"
                sp_params.extend(stage)
            else:
                sp_query += " AND stage = ?"
                sp_params.append(stage)
        
        # Add frequency range filtering for spindles
        if spindle_freq_range and len(spindle_freq_range) == 2:
            sp_query += " AND freq_lower >= ? AND freq_upper <= ?"
            sp_params.extend(spindle_freq_range)
        
        sp_query += " GROUP BY channel, stage"
        
        # Execute queries
        cursor.execute(sw_query, sw_params)
        sw_results = cursor.fetchall()
        
        cursor.execute(sp_query, sp_params)
        sp_results = cursor.fetchall()
        
        conn.close()
        
        # Print results
        sw_method_str = sw_method if sw_method else "all methods"
        sp_method_str = spindle_method if spindle_method else "all methods"
        channel_str = channel if channel else "all channels"
        stage_str = stage if stage else "all stages"
        sw_freq_str = f"{sw_freq_range[0]}-{sw_freq_range[1]}Hz" if sw_freq_range else "all frequencies"
        sp_freq_str = f"{spindle_freq_range[0]}-{spindle_freq_range[1]}Hz" if spindle_freq_range else "all frequencies"
        
        print(f"\nSlow Wave Statistics ({sw_method_str}, {channel_str}, {stage_str}, {sw_freq_str}):")
        if sw_results:
            for chan, stg, count in sw_results:
                print(f"  {chan}, {stg}: {count} events")
        else:
            print("  No slow wave events found matching criteria")
        
        print(f"\nSpindle Statistics ({sp_method_str}, {channel_str}, {stage_str}, {sp_freq_str}):")
        if sp_results:
            for chan, stg, count in sp_results:
                print(f"  {chan}, {stg}: {count} events")
        else:
            print("  No spindle events found matching criteria")
        
    except Exception as e:
        print(f"Error accessing database: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze phase-amplitude coupling in hdEEG data')
    parser.add_argument('--root_dir', type=str, help='Root directory for data files')
    parser.add_argument('--edf_file', type=str, help='EDF/SET file name')
    parser.add_argument('--xml_file', type=str, help='Annotations XML file name')
    parser.add_argument('--db_file', type=str, help='SQLite database file name')
    parser.add_argument('--channels_file', type=str, help='CSV file containing channels to analyze')
    parser.add_argument('--stages', type=str, nargs='+', default=['NREM2', 'NREM3'], help='Sleep stages to analyze')
    parser.add_argument('--phase_freq', type=float, nargs=2, default=[0.5, 1.25], help='Phase frequency range (Hz)')
    parser.add_argument('--amp_freq', type=float, nargs=2, default=[11, 16], help='Amplitude frequency range (Hz)')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--sw_method', type=str, help='Slow wave detection method to use')
    parser.add_argument('--spindle_method', type=str, help='Spindle detection method to use')
    parser.add_argument('--list_methods', action='store_true', help='List available detection methods in the database')
    parser.add_argument('--stats', action='store_true', help='Show statistics about available events')
    parser.add_argument('--channel', type=str, help='Specific channel to analyze (default: all)')
    parser.add_argument('--sw_freq_range', type=float, nargs=2, help='Slow wave frequency range for filtering events (Hz)')
    parser.add_argument('--spindle_freq_range', type=float, nargs=2, help='Spindle frequency range for filtering events (Hz)')

    args = parser.parse_args()

    # Set default root directory if not provided
    if args.root_dir is None:
        args.root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
    
    # Set default file names if not provided
    if args.edf_file is None:
        args.edf_file = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
    
    if args.xml_file is None:
        args.xml_file = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"
    
    if args.db_file is None:
        args.db_file = "neural_events.db"
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.root_dir, "wonambi", "pac_results")
    
    # Construct database path
    db_path = os.path.join(args.root_dir, "wonambi", args.db_file)
    
    # List methods if requested
    if args.list_methods:
        sw_methods, spindle_methods = list_available_methods(db_path)
        return
    
    # Show statistics if requested
    if args.stats:
        get_event_stats(db_path, args.sw_method, args.spindle_method, args.channel, 
                    args.stages, args.sw_freq_range, args.spindle_freq_range)
        return
    
    # Construct full paths
    data_file = os.path.join(args.root_dir, args.edf_file)
    annot_file = os.path.join(args.root_dir, "wonambi", args.xml_file)
    channels_file = os.path.join(args.root_dir, args.channels_file) if args.channels_file else None
    
    # Read channels from CSV if available
    if channels_file and os.path.exists(channels_file):
        channels = read_channels_from_csv(channels_file)
        print(f"Channels loaded from CSV: {channels}")
    elif args.channel:
        channels = [args.channel]
        print(f"Using specified channel: {args.channel}")
    else:
        print("No channels specified. Will load from dataset.")
        channels = None

    # Verify files exist
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    if not os.path.exists(annot_file):
        print(f"Warning: Annotation file not found: {annot_file}")
        print("Will proceed without annotations, using only database events.")
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return
    
    # Load dataset and annotations
    print("Loading dataset and annotations...")
    data = WonambiDataset(data_file)
    
    if os.path.exists(annot_file):
        annot = CustomAnnotations(annot_file)
    else:
        annot = None
    
    # # If no channels specified, use the first 10 channels from the dataset
    # if channels is None:
    #     channels = data.channels[:10]
    #     print(f"Using first 10 channels: {channels}")
    
    # Create ParalPAC instance
    pac_processor = ParalPAC(
        dataset=data,
        annotations=annot,
        rootpath=args.root_dir,
        log_level=logging.INFO
    )
    
    # # Setup filtering options
    # filter_opts = {
    #     'notch': True,
    #     'notch_freq': 50,  # 50 Hz for European data, 60 Hz for US data
    #     'notch_harmonics': True,
    #     'bandpass': True,
    #     'highpass': 0.1,
    #     'lowpass': 45,
    #     'laplacian': False,
    #     'dcomplex': 'hilbert',
    #     'filtcycle': 3,
    #     'width': 6
    # }
    
    # Setup event options
    event_opts = {
        'buffer': 1.0,  # 1 second buffer around events
        'sw_method': args.sw_method,  # Add detection method to event options
        'spindle_method': args.spindle_method,  # Add detection method to event options
        'sw_freq_range': args.sw_freq_range,  # Add frequency range for slow waves
        'spindle_freq_range': args.spindle_freq_range  # Add frequency range for spindles
    }
        
    # Create modified analyze_pac method to handle method selection in SQL queries
    def modified_analyze_pac(event_type, pair_with_spindles=False):
        """Wrapper for analyze_pac to handle method selection"""
        params = {
            'chan': channels,
            'stage': args.stages,
            'phase_freq': tuple(args.phase_freq),
            'amp_freq': tuple(args.amp_freq),
            'idpac': (1, 2, 4),  # Method: MI, Surrogate: Time lag, Correction: Z-score
            'use_detected_events': True,
            'event_type': event_type,
            'pair_with_spindles': pair_with_spindles,
            'time_window': 1.0,
            'db_path': db_path,
            'out_dir': args.output_dir,
            'event_opts': event_opts
        }
        
        # Create method-specific output directory
        method_dir = args.output_dir
        if args.sw_method or args.spindle_method:
            method_name = []
            if event_type == 'slow_wave' and args.sw_method:
                method_name.append(args.sw_method)
            if (event_type == 'spindle' or pair_with_spindles) and args.spindle_method:
                method_name.append(args.spindle_method)
            
            if method_name:
                method_dir = os.path.join(args.output_dir, '_'.join(method_name))
                os.makedirs(method_dir, exist_ok=True)
                params['out_dir'] = method_dir
        
        return pac_processor.analyze_pac(**params), method_dir
    
    # Run analyses
    results = {}
    output_dirs = {}
    
    # First check if the database has events with the specified methods
    if args.sw_method or args.spindle_method:
        get_event_stats(db_path, args.sw_method, args.spindle_method, args.channel, args.stages)
    
    # Run slow wave-spindle coupling analysis if both methods are specified
    if args.sw_method and args.spindle_method:
        print(f"\nRunning slow wave-spindle coupling analysis...")
        print(f"Using slow wave method: {args.sw_method}")
        print(f"Using spindle method: {args.spindle_method}")
        results['sw_spindle'], output_dirs['sw_spindle'] = modified_analyze_pac('slow_wave', True)
    
    # # Run slow wave analysis
    # if args.sw_method or not (args.sw_method or args.spindle_method):
    #     method_str = f"method: {args.sw_method}" if args.sw_method else "all methods"
    #     print(f"\nRunning slow wave PAC analysis ({method_str})...")
    #     results['sw'], output_dirs['sw'] = modified_analyze_pac('slow_wave', False)
    
    # # Run spindle analysis
    # if args.spindle_method or not (args.sw_method or args.spindle_method):
    #     method_str = f"method: {args.spindle_method}" if args.spindle_method else "all methods"
    #     print(f"\nRunning spindle PAC analysis ({method_str})...")
    #     results['spindle'], output_dirs['spindle'] = modified_analyze_pac('spindle', False)
    
    # Export results to CSV
    print("\nExporting PAC parameters to CSV...")
    
    # Export slow wave-spindle coupling results
    if 'sw_spindle' in results:
        sw_spindle_csv = os.path.join(output_dirs['sw_spindle'], "sw_spindle_coupling_pac_summary.csv")
        pac_processor.export_pac_parameters_to_csv(
            csv_file=sw_spindle_csv,
            phase_freq=tuple(args.phase_freq),
            amp_freq=tuple(args.amp_freq)
        )
    
    # # Export slow wave results
    # if 'sw' in results:
    #     sw_csv = os.path.join(output_dirs['sw'], "slow_wave_pac_summary.csv")
    #     pac_processor.export_pac_parameters_to_csv(
    #         csv_file=sw_csv,
    #         phase_freq=tuple(args.phase_freq),
    #         amp_freq=tuple(args.amp_freq)
    #     )
    
    # # Export spindle results
    # if 'spindle' in results:
    #     spindle_csv = os.path.join(output_dirs['spindle'], "spindle_pac_summary.csv")
    #     pac_processor.export_pac_parameters_to_csv(
    #         csv_file=spindle_csv,
    #         phase_freq=tuple(args.phase_freq),
    #         amp_freq=tuple(args.amp_freq)
    #     )
    
    # Generate comodulogram for a sample channel (if desired)
    import numpy as np
    do_comodulogram = True  # Set to True to generate comodulograms
    phase_freqs = [(i, i+0.5) for i in np.arange(0.2, 8.0, 0.5)]  # 0.5-1.0, 1.0-1.5, etc.
    amp_freqs = [(i, i+2) for i in np.arange(8, 30, 2)]   
    if do_comodulogram and channels:
        print("\nGenerating comodulogram for first channel...")
        first_channel = channels[0]
        
        comod_params = {
            'chan': first_channel,
            'stage': args.stages,
            'phase_freqs': phase_freqs,
            'amp_freqs': amp_freqs,
            'idpac': (1, 2, 4),
            'out_dir': args.output_dir
        }
        
        pac_processor.generate_comodulogram(**comod_params)
    
    print("\n~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"PAC analysis completed")
    print(f"Results saved to {args.output_dir}")
    print(f"ALL DONE")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")

if __name__ == "__main__":
    main()