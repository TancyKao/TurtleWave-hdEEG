
"""
hdEEG_spindle_detector.py
This script is designed to detect spindles in high-density EEG (hdEEG) data using the TurtleWave-hdEEG library. 
It processes EEG data, applies spindle detection algorithms, and exports the results in JSON and CSV format.
Modules:
    - wonambi.dataset: Used to load EEG datasets.
    - wonambi.attr: Used to handle annotations in EEG data.
    - turtlewave_hdEEG: Custom library for processing EEG events and annotations.
Functions:
    - detect_spindles: Detects spindles in EEG data based on specified parameters.
    - export_spindle_parameters_to_csv: Exports spindle parameters to a CSV file.
    - export_spindle_density_to_csv: Exports spindle density information to a CSV file.
Workflow:
    1. Define file paths for the EEG dataset and annotations.
    2. Load the dataset and annotations.
    3. Create an instance of the ParalEvents class for processing events.
    4. Specify test parameters for spindle detection, including method, channels, frequency range, and sleep stages.
    5. Run the spindle detection algorithm and save the results in JSON format.
    6. Export spindle parameters and density information to CSV files for further analysis.
Parameters:
    - root_dir (str): Root directory containing the EEG dataset and annotations.
    - datafilename (str): Name of the EEG dataset file.
    - annotfilename (str): Name of the annotation file.
    - test_method (str): Spindle detection method to use (e.g., 'Ferrarelli2007', 'Moelle2011').
    - test_stages (list): Sleep stages to include in the analysis (e.g., 'NREM3').
    - test_frequency (tuple): Frequency range for spindle detection (e.g., (9, 12)).
    - json_dir (str): Directory to save JSON results.
Outputs:
    - JSON files containing spindle detection results.
    - CSV files with spindle parameters and density information.
Usage:
    1. Ensure you have the TurtleWave-hdEEG library installed.
    Run this script to detect spindles in hdEEG data and export the results for further analysis.


"""

import os
import sys
from turtlewave_hdEEG.utils import read_channels_from_csv
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations, fmt_freq_token
import logging
import argparse as _ap

# Give the library's handler-less module loggers somewhere to write.
#
# The Paral* processors build their own console handler in `_setup_logger`, so
# their records already reach the terminal exactly once. The module-level
# loggers (`turtlewave_hdEEG.dataset`, `turtlewave_hdEEG.utils`) have no
# handler of their own, so without this their records go nowhere and the
# script looks silent while it loads the recording.
#
# The handler deliberately goes on those two loggers, not on the root via
# `logging.basicConfig` and not on the `turtlewave_hdEEG` parent. Both of
# those print every processor line TWICE, because a processor logger both
# handles its own records and propagates them upward; the root additionally
# pulls in INFO chatter from third-party libraries. Propagation is switched
# off here so a root handler installed later by another import cannot
# reintroduce the duplication.
for _tw_name in ('turtlewave_hdEEG.dataset', 'turtlewave_hdEEG.utils'):
    _tw_log = logging.getLogger(_tw_name)
    if not any(getattr(h, 'name', None) == 'turtlewave-console'
               for h in _tw_log.handlers):
        _tw_console = logging.StreamHandler()
        _tw_console.name = 'turtlewave-console'
        _tw_console.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        _tw_log.addHandler(_tw_console)
    _tw_log.setLevel(logging.INFO)
    _tw_log.propagate = False

# Optional CLI overrides (backward-compatible: no args => unchanged behaviour).
# Used by the eeg_review_gui "Export Re-run Package" QC handoff.
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--annot', default=None,
                help='override annotation XML (e.g. QC sidecar)')
_p.add_argument('--channels', default=None,
                help='CSV of channels to detect (no header, one per row)')
_p.add_argument('--write-db', dest='write_db', action='store_true',
                help='write events straight to neural_events.db and skip the '
                     'JSON->CSV->import steps (default: legacy JSON+CSV path)')
_p.add_argument('--resume', dest='resume', action='store_true',
                help='with --write-db, skip channels already completed for this '
                     'exact method/band/stage scope in the database')
_cli, _ = _p.parse_known_args()

# 1. Define the file paths for the dataset and annotations
# The root directory should contain the EEG dataset and the wonambi directory for annotations.
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"



#Read channels from CSV
# test_channels = read_channels_from_csv(channels_csv_path)
#print(f"Channels loaded from CSV: {test_channels}")

# Construct the full paths for the dataset and annotations
# The dataset file is located in the root directory
# The annotations are in the 'wonambi' subdirectory.
# The JSON files are in the 'wonambi'/'spindle_results' subdirectory.
data_file = os.path.join(root_dir, datafilename)
annot_file = _cli.annot if _cli.annot else os.path.join(root_dir, "wonambi",annotfilename)
json_dir = os.path.join(root_dir, "wonambi", "spindle_results")
db_path = os.path.join(root_dir, "wonambi",'neural_events.db')

# 2. Load dataset and annotations
print("Loading dataset and annotations...")
data = WonambiDataset(data_file)
annot = CustomAnnotations(annot_file)

# 3. Create ParalEvents instance
event_processor = ParalEvents(
    dataset=data, 
    annotations=annot,
    #log_level=logging.warning,  # Change to DEBUG for more detailed logs
    #log_file=os.path.join(root_dir, "wonambi", "spindle_detection.log"),
    )

# 4. Custom define parameters
test_method = 'Moelle2011' # 'Moelle2011', Ferrarelli2007
test_channels = ['E110','E111','E112']  # Channels
if _cli.channels:
    test_channels = read_channels_from_csv(_cli.channels)
    print(f"Channels from --channels: {len(test_channels)}")
test_stages = ['NREM2','NREM3'] # ['NREM2', 'NREM3']
test_frequency = (11, 13)  # Frequency range for spindles

# 5. Test detect_spindles with minimal parameters
print("Running detect_spindles...")

spindles = event_processor.detect_spindles(
    method               = test_method,
    chan                 = test_channels,
    frequency            = test_frequency,
    duration             = (0.5, 3),
    stage                = test_stages,
    reject_artifacts     = True,
    reject_arousals      = False,
    cat                  = (1, 1, 1, 0),# concatenate across cycles, stages, and discontinuities (event types separate)
    save_to_annotations  = False, # save to annotations
    json_dir             = json_dir,
    # Direct-to-DB path (opt-in). When --write-db is not passed these are
    # write_db=False / resume=False, so behaviour is byte-identical to before.
    write_db             = _cli.write_db,
    db_path              = db_path if _cli.write_db else None,
    resume               = _cli.resume,
)

""" 0 means no concatenation, 1 means concatenation
    position 1: cycle concatenation
    position 2: stage concatenation
    position 3: discontinuous signal concatenation
    position 4: event type concatenation (does not apply here)
"""


# After processing all channels, export parameters
freq_range = fmt_freq_token(*test_frequency)
stages_str = "".join(test_stages)

# for selecting proper json files
file_pattern = f"spindles_{test_method}_{freq_range}_{stages_str}"

if _cli.write_db:
    # Direct-to-DB run: events are already in neural_events.db (with det_* and
    # spectral columns and provenance). The JSON->CSV->import steps are not
    # needed. A flat CSV can still be produced on demand from the DB:
    #   from turtlewave_hdEEG import export_events_to_csv
    #   export_events_to_csv(db_path, 'spindle', test_method, test_frequency,
    #                        test_stages, output_dir=json_dir)
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"Spindle events written directly to DB: {db_path}")
    print(f"ALL DONE (direct-to-DB)")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
else:
    # 6. Test the new SQLite parameter calculation and storage
    # print("\nCalculating and storing parameters in SQLite database...")

    # Initialize the database
    #event_processor.initialize_sqlite_database(db_path)

    param2CSV = event_processor.export_spindle_parameters_to_csv(
        json_input   = json_dir,
        csv_file     = os.path.join(json_dir, f'spindle_parameters_{test_method}_{freq_range}_{stages_str}.csv'),
        file_pattern = file_pattern  # Pattern to match JSON files
    )

    # Pass the same rejection settings the detection call used. The density
    # denominator is the recording time the detector actually analysed, so a
    # mismatch here (detection kept arousal epochs, the denominator subtracts
    # them) biases every density. Detection above used reject_arousals=False.
    density2CSV = event_processor.export_spindle_density_to_csv(
        json_input       = json_dir,
        csv_file         = os.path.join(json_dir, f'spindle_density_{test_method}_{freq_range}_{stages_str}.csv'),
        stage            = test_stages,
        file_pattern     = file_pattern,
        reject_artifacts = True,
        reject_arousals  = False
    )

    csv2db = event_processor.import_parameters_csv_to_database(
        csv_file     = os.path.join(json_dir, f'spindle_parameters_{test_method}_{freq_range}_{stages_str}.csv'),
        db_path      = db_path
        )

    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"Spindle parameters saved")
    print(f"Spindle density saved")
    print(f"ALL DONE")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
