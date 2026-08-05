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
from turtlewave_hdEEG import ParalPAC, CustomAnnotations, derive_subject

def verify_pac_rows(db_path, subject, stages, n_channels_requested):
    """Count the ``pac_coupling`` rows a run actually landed, and report.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    subject : str
        Subject id the rows should be keyed under.
    stages : list of str or str
        Stages analysed, used only for the printed scope.
    n_channels_requested : int
        Channels handed to the analysis, for the comparison line.

    Returns
    -------
    int
        Number of ``pac_coupling`` rows for this subject. 0 when the table
        does not exist, so an absent table reads as "nothing landed" rather
        than raising.
    """
    stage_str = ''.join(stages) if isinstance(stages, list) else str(stages)
    try:
        conn = sqlite3.connect(db_path)
        try:
            n_rows = conn.execute(
                "SELECT COUNT(*) FROM pac_coupling WHERE subject = ?",
                (subject,)).fetchone()[0]
            n_chan = conn.execute(
                "SELECT COUNT(DISTINCT channel) FROM pac_coupling "
                "WHERE subject = ?", (subject,)).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        print(f"Could not read pac_coupling in {db_path}: {e}")
        return 0

    print(f"\npac_coupling rows for subject '{subject}' (stages {stage_str}): "
          f"{n_rows} row(s) across {n_chan}/{n_channels_requested} channel(s)")
    return n_rows


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
            # Handle combined stage string (e.g., 'NREM2NREM3')
            if isinstance(stage, list):
                # Convert list to concatenated string
                stage_str = ''.join(stage)
                sw_query += " AND stage = ?"
                sw_params.append(stage_str)
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
            # Handle combined stage string (e.g., 'NREM2NREM3')
            if isinstance(stage, list):
                # Convert list to concatenated string
                stage_str = ''.join(stage)
                sp_query += " AND stage = ?"
                sp_params.append(stage_str)
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
        
        # Handle stage display for combined stages
        if stage:
            if isinstance(stage, list):
                stage_str = ''.join(stage)  # Show as combined string
            else:
                stage_str = stage
        else:
            stage_str = "all stages"

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

def get_common_channels(db_path, sw_method, spindle_method, stages):
    """Get channels that have both slow wave and spindle events for the specified methods and stages."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert stages to combined string format
        stage_str = ''.join(stages) if isinstance(stages, list) else stages
        print(f"Looking for events in stage: '{stage_str}'")

        # Get channels with slow wave events
        sw_query = """
            SELECT DISTINCT channel 
            FROM events 
            WHERE event_type = 'slow_wave' 
            AND method = ? 
            AND stage = ?
        """
        print(f"Slow wave query: {sw_query}")
        print(f"SW parameters: method='{sw_method}', stage='{stage_str}'")

        cursor.execute(sw_query, (sw_method, stage_str))
        sw_channels = set(row[0] for row in cursor.fetchall())
        print(f"SW channels found: {sorted(list(sw_channels))}")

        # Get channels with spindle events
        spindle_query = """
            SELECT DISTINCT channel 
            FROM events 
            WHERE event_type = 'spindle' 
            AND method = ? 
            AND stage = ?
        """
        print(f"Spindle query: {spindle_query}")
        print(f"Spindle parameters: method='{spindle_method}', stage='{stage_str}'")
        
        cursor.execute(spindle_query, (spindle_method, stage_str))
        spindle_channels = set(row[0] for row in cursor.fetchall())
        print(f"Spindle channels found: {sorted(list(spindle_channels))}")
        conn.close()
        
        # Find intersection (channels that have both event types)
        common_channels = list(sw_channels.intersection(spindle_channels))
        common_channels.sort()  # Sort for consistent ordering
        
        print(f"\nChannel analysis for stage '{stage_str}':")
        print(f"  Channels with slow waves ({sw_method}): {len(sw_channels)}")
        print(f"  Channels with spindles ({spindle_method}): {len(spindle_channels)}")
        print(f"  Common channels (both events): {len(common_channels)}")
        
        if common_channels:
            print(f"  Selected channels: {common_channels}")
        else:
            print("  Warning: No channels have both slow waves and spindles!")
        
        return common_channels
        
    except Exception as e:
        print(f"Error accessing database: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_comodulogram_fixed(pac_processor, chan, stage, phase_freqs, amp_freqs, idpac, out_dir):
    """Fixed comodulogram generation using the ParalPAC's built-in method"""
    try:
        print(f"Generating comodulogram for channel {chan}...")
        
        # Use the ParalPAC's built-in generate_comodulogram method
        result = pac_processor.generate_comodulogram(
            chan=chan,
            stage=stage,
            phase_freqs=phase_freqs,
            amp_freqs=amp_freqs,
            idpac=idpac,
            out_dir=out_dir
        )
        
        if result is not None:
            print("Comodulogram generated successfully!")
            return result['comod']
        else:
            print("Comodulogram generation failed.")
            return None
        
    except Exception as e:
        print(f"Error generating comodulogram: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# CONFIGURATION SECTION - Edit these parameters directly in the file
# ============================================================================

# Set USE_CONFIG = True to use the configuration below instead of command line arguments
# Set USE_CONFIG = False to use command line arguments
USE_CONFIG = True

CONFIG = {
    # Data file paths
    'root_dir': r"S:\Sleep\2. STAFF\Tancy\OSA CPAP Events\sub-10DS\BL",
    'edf_file': "OSA_10DS_clean_rebuilt.set",
    'xml_file': "OSA_10DS_clean_rebuilt.xml",
    'db_file': "neural_events.db",
    
    # Channel selection (choose one method):
    # Option 1: Use a CSV file with channel names
    'channels_file': "channels.csv",  # Set to None to use other options
    # Option 2: Use a single channel
    'channel': None,  # e.g., "E1" or None
    # Option 3: Auto-select channels (leave both above as None)
    
    # Detection methods (required for auto-selection or when analyzing events)
    'sw_method': "Staresina2015",
    'spindle_method': "Moelle2011",
    
    # Analysis parameters
    'stages': ['NREM2', 'NREM3'],  # Sleep stages to analyze
    'phase_freq': [1.0, 4.54545454545455],  # Phase frequency range (Hz) - slow waves
    'amp_freq': [13, 16],  # Amplitude frequency range (Hz) - spindles
    
    # Optional: Filter events by frequency range
    'sw_freq_range': None,  # e.g., [0.5, 2.0] or None
    'spindle_freq_range': None,  # e.g., [11, 16] or None
    
    # Output directory (None = auto-generate in root_dir/wonambi/pac_results)
    'output_dir': None,
    
    # Utility flags
    'list_methods': False,  # Set to True to list available methods and exit
    'stats': False,  # Set to True to show event statistics and exit
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


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
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject id used as the pac_coupling primary key. '
                             'Default: derived from the annotation XML filename, '
                             'else from the root directory name.')
    parser.add_argument('--no-write-db', dest='no_write_db', action='store_true',
                        help='Do not write PAC results into neural_events.db '
                             '(CSV/.npy files only). Results written only to '
                             'files are easy to lose track of; prefer the '
                             'default database write.')

    args = parser.parse_args()

    # Use configuration from CONFIG dict if USE_CONFIG is True
    if USE_CONFIG:
        print("=" * 80)
        print("Using configuration from CONFIG section in the script")
        print("=" * 80)
        
        # Override args with CONFIG values
        for key, value in CONFIG.items():
            if value is not None or not hasattr(args, key):
                setattr(args, key, value)
    
    # Set default root directory if not provided
    if args.root_dir is None:
        args.root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/sub-03RA/BL/"
    
    # Set default file names if not provided
    if args.edf_file is None:
        args.edf_file = "OSA_BL03RA_clean_rebuilt.set"
    
    if args.xml_file is None:
        args.xml_file = "OSA_BL03RA_clean_rebuilt.xml"
    
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
    
    channels = None

    # Read channels from CSV if available
    if channels_file and os.path.exists(channels_file):
        channels = read_channels_from_csv(channels_file)
        print(f"Channels loaded from CSV: {channels}")
    elif args.channel:
        channels = [args.channel]
        print(f"Using specified channel: {args.channel}")
    else:
        # Auto-select channels that have both slow waves and spindles
        if args.sw_method and args.spindle_method:
            print("No channels specified. Auto-selecting channels with both slow waves and spindles...")
            channels = get_common_channels(db_path, args.sw_method, args.spindle_method, args.stages)
            
            if not channels:
                print("Error: No channels found with both slow waves and spindles for the specified methods and stages.")
                return
        else:
            print("Error: When no channels are specified, both --sw_method and --spindle_method must be provided.")
            print("Use --list_methods to see available methods.")
            return

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
    
    if channels is None:
        print("Error: No channels selected. This should not happen.")
        return
    
    # Validate that channels exist in the dataset
    #dataset_channels = data.list_of_channels.label
    print(data.header)
    #print(data.header.chan_name)
    dataset_channels = data.header['chan_name']
   
    print(f"Dataset channels (first 10): {dataset_channels[:10] if len(dataset_channels) > 10 else dataset_channels}")
    print(f"Selected channels before validation: {channels}")
        
    valid_channels = [ch for ch in channels if ch in dataset_channels]
    
    if not valid_channels:
        print(f"Error: None of the selected channels {channels} exist in the dataset.")
        print(f"Available channels: {dataset_channels[:10]}... (showing first 10)")
        return
    
    if len(valid_channels) != len(channels):
        invalid_channels = [ch for ch in channels if ch not in dataset_channels]
        print(f"Warning: These channels don't exist in dataset: {invalid_channels}")
        channels = valid_channels
        print(f"Using valid channels: {channels}")

    channels = valid_channels
    print(f"Final channels to analyze: {channels}")
    if not channels:
        print("Error: No valid channels remain after validation.")
        return
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
        'spindle_freq_range': args.spindle_freq_range,  # Add frequency range for spindles
        'stages': ''.join(args.stages) if isinstance(args.stages, list) else args.stages  # Convert to combined string
    }
        
    # Resolve the subject id once. This is the primary key of pac_coupling, so
    # a PAC run without it cannot be stored; deriving it here (explicit ->
    # annotation XML stem -> root directory name) means the database write is
    # the default path rather than something the caller has to remember.
    subject = derive_subject(annotation_path=annot_file,
                             root_dir=args.root_dir,
                             explicit=args.subject)
    write_db = not args.no_write_db
    print(f"Subject: {subject}")
    print(f"Write PAC results to database: {write_db} ({db_path})")

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
            'event_opts': event_opts,
            # analyze_pac still defaults write_db=False (flipping that default
            # is a later stage); pass it explicitly so results reach the DB.
            'write_db': write_db,
            'subject': subject,
        }
        
        # Create method-specific output directory
        method_dir = args.output_dir
        if args.sw_method or args.spindle_method:
            method_name = []
            if event_type == 'slow_wave' and args.sw_method:
                method_name.append(args.sw_method)
            if (event_type == 'spindle' or pair_with_spindles) and args.spindle_method:
                method_name.append(args.spindle_method)
            
            #if method_name:
            #    method_dir = os.path.join(args.output_dir, '_'.join(method_name))
            #    os.makedirs(method_dir, exist_ok=True)
            #    params['out_dir'] = method_dir
        
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
        # Create method info for export
        method_info = {
            'sw_method': args.sw_method,
            'spindle_method': args.spindle_method,
            'event_type': 'slow_wave',
            'pair_with_spindles': True,
            'stage': args.stages
        }
        
        sw_spindle_csv = os.path.join(output_dirs['sw_spindle'], "sw_spindle_coupling_pac_summary.csv")
        export_result = pac_processor.export_pac_parameters_to_csv(
            csv_file=sw_spindle_csv,
            phase_freq=tuple(args.phase_freq),
            amp_freq=tuple(args.amp_freq),
            method_info=method_info,
            out_dir=args.output_dir
        )
        
        if export_result:
            print(f"Successfully exported summary to: {export_result['file']}")
            print(f"Exported {export_result['channels']} channels with {export_result['rows']} total rows")
        else:
            print("Warning: No PAC data was exported to summary CSV")
    
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
    
    # Generate comodulogram for a sample channel (FIXED VERSION)
    import numpy as np
    do_comodulogram = True  # Set to True to generate comodulograms
    phase_freqs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]  # Slow wave frequencies
    amp_freqs = [(8, 12), (12, 16), (16, 20), (20, 25), (25, 30)]  # Spindle and higher frequencies
    
    if do_comodulogram and channels:
        print("\nGenerating comodulogram for first channel (FIXED VERSION)...")
        first_channel = channels[0]
        
        # Use the fixed comodulogram function
        comod_result = generate_comodulogram_fixed(
            pac_processor=pac_processor,
            chan=first_channel,
            stage=args.stages,
            phase_freqs=phase_freqs,
            amp_freqs=amp_freqs,
            idpac=(1, 2, 4),
            out_dir=args.output_dir
        )
        
        if comod_result is not None:
            print("Comodulogram generated successfully!")
        else:
            print("Comodulogram generation failed.")
    
    # Post-run verification. The failure this whole change exists to stop was
    # a PAC run that completed cleanly, wrote CSVs, and never reached the
    # database -- and still printed "ALL DONE". Check the rows are actually
    # there before saying so, and exit non-zero if they are not.
    # Bound unconditionally: `results` stays empty whenever no analysis ran
    # (e.g. --sw_method without --spindle_method, since the sw_spindle branch
    # needs both), and the summary below reads it either way.
    n_rows = None
    if write_db and results:
        n_rows = verify_pac_rows(db_path, subject, args.stages, len(channels))
        if n_rows == 0:
            print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
            print(f"ERROR: PAC analysis produced results but NO rows landed in "
                  f"pac_coupling for subject '{subject}' in {db_path}.")
            print("Results exist only as CSV/.npy files under "
                  f"{args.output_dir} and are not in the database.")
            print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
            sys.exit(1)

    print("\n~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"PAC analysis completed")
    print(f"Results saved to {args.output_dir}")
    if not results:
        print("No PAC analysis ran: the slow-wave/spindle coupling branch "
              "needs BOTH --sw_method and --spindle_method.")
    elif write_db:
        print(f"PAC rows in database: {n_rows} (subject '{subject}')")
    print(f"ALL DONE")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")

if __name__ == "__main__":
    main()