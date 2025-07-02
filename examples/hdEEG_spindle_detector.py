
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
from turtlewave_hdEEG import ParalEvents, CustomAnnotations
import logging

# 1. Define the file paths for the dataset and annotations
# The root directory should contain the EEG dataset and the wonambi directory for annotations.
root_dir = "//Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/MCI005_BL/"
datafilename = "MCI005_BL_clean_rebuilt.set"
annotfilename = "MCI005_BL_clean_rebuilt.xml"


channels_csv_path = os.path.join(root_dir, "channels.csv")  # Adjust the path as needed


#Read channels from CSV
test_channels = read_channels_from_csv(channels_csv_path)
#print(f"Channels loaded from CSV: {test_channels}")

# Construct the full paths for the dataset and annotations
# The dataset file is located in the root directory
# The annotations are in the 'wonambi' subdirectory.
# The JSON files are in the 'wonambi'/'spindle_results' subdirectory.
data_file = os.path.join(root_dir, datafilename)
annot_file = os.path.join(root_dir, "wonambi",annotfilename)
json_dir = os.path.join(root_dir, "wonambi", "spindle_results")
db_path = os.path.join(root_dir, "wonambi",'neural_events.db')

# 2. Load dataset and annotations
print("Loading dataset and annotations...")
data = WonambiDataset(data_file)
annot = CustomAnnotations(annot_file)

# 3. Create ParalEvents instance
event_processor = ParalEvents(
    dataset=data, 
    annotations=annot
    #log_level=logging.warning,  # Change to DEBUG for more detailed logs
    #log_file=os.path.join(root_dir, "wonambi", "spindle_detection.log")
    )

# 4. Custom define parameters
test_method = 'Moelle2011' # 'Moelle2011', Ferrarelli2007
#test_channels = ['E101','E110','E111','E112','E113']  # Channels
test_stages = ['NREM2','NREM3'] # ['NREM2', 'NREM3']
test_frequency = (9, 12)  # Frequency range for spindles

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
    cat                  = (1, 1, 1, 0),# concatenate within and between stages, cycles separate
    save_to_annotations  = False, # don't save to annotations
    json_dir             = json_dir
)

""" 0 means no concatenation, 1 means concatenation
    position 1: cycle concatenation
    position 2: stage concatenation
    position 3: discontinuous signal concatenation
    position 4: event type concatenation (does not apply here)
"""


# After processing all channels, export parameters
freq_range = f"{test_frequency[0]}-{test_frequency[1]}Hz"
stages_str = "".join(test_stages)

# for selecting proper json files
file_pattern = f"spindles_{test_method}_{freq_range}_{stages_str}"

# 6. Test the new SQLite parameter calculation and storage
print("\nCalculating and storing parameters in SQLite database...")

# Initialize the database
event_processor.initialize_sqlite_database(db_path)



param2CSV = event_processor.export_spindle_parameters_to_csv(
    json_input   = json_dir,  
    csv_file     = os.path.join(json_dir, f'spindle_parameters_{test_method}_{freq_range}_{stages_str}.csv'),
    file_pattern = file_pattern  # Pattern to match JSON files
)


csv2db = event_processor.import_parameters_csv_to_database(
    csv_file     = os.path.join(json_dir, f'spindle_parameters_{test_method}_{freq_range}_{stages_str}.csv'),
    db_path      = db_path
    )



density2CSV = event_processor.export_spindle_density_to_csv(
    json_input   = json_dir,  
    csv_file     = os.path.join(json_dir, f'spindle_density_{test_method}_{freq_range}_{stages_str}.csv'),
    stage        = test_stages,
    file_pattern = file_pattern
)


print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
print(f"Spindle parameters saved")
print(f"Spindle density saved")
print(f"ALL DONE")
print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
