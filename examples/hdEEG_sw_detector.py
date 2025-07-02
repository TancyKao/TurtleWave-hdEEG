"""
hdEEG_sw_detector.py
This script is designed to detect slow waves in high-density EEG (hdEEG) data using the TurtleWave-hdEEG library. 
It processes EEG data, applies slow wave detection algorithms, and exports the results in JSON and CSV format.
Modules:
    - wonambi.dataset: Used to load EEG datasets.
    - wonambi.attr: Used to handle annotations in EEG data.
    - turtlewave_hdEEG: Custom library for processing EEG events and annotations.
Functions:
    - detect_slow_waves: Detects slow waves in EEG data based on specified parameters.
    - export_slow_wave_parameters_to_csv: Exports slow wave parameters to a CSV file.
    - export_slow_wave_density_to_csv: Exports slow wave density information to a CSV file.
Workflow:
    1. Define file paths for the EEG dataset and annotations.
    2. Load the dataset and annotations.
    3. Create an instance of the ParalSWA class for processing events.
    4. Specify test parameters for slow wave detection, including method, channels, frequency range, and sleep stages.
    5. Run the slow wave detection algorithm and save the results in JSON format.
    6. Export slow wave parameters and density information to CSV files for further analysis.
"""

import os
import sys
from turtlewave_hdEEG.utils import read_channels_from_csv
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalSWA, CustomAnnotations
import logging

# 1. Define the file paths for the dataset and annotations
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"

channels_csv_path = os.path.join(root_dir, "channels.csv")

# Read channels from CSV
test_channels = read_channels_from_csv(channels_csv_path)
print(f"Channels loaded from CSV: {test_channels}")

# Construct the full paths
data_file = os.path.join(root_dir, datafilename)
annot_file = os.path.join(root_dir, "wonambi", annotfilename)
json_dir = os.path.join(root_dir, "wonambi", "sw_results")
db_path = os.path.join(root_dir, "wonambi",'neural_events.db')


# 2. Load dataset and annotations
print("Loading dataset and annotations...")
data = WonambiDataset(data_file)
annot = CustomAnnotations(annot_file)

# 3. Create ParalSWA instance
event_processor = ParalSWA(
    dataset=data, 
    annotations=annot
)

# 4. Custom define parameters
test_method = 'Staresina2015'  # 'Massimini2004','AASM/Massimini2004', 'Ngo2015', 'Staresina2015'
test_stages = ['NREM2', 'NREM3']  # Sleep stages to analyze
#test_channels = ['E101']  # Channels
test_frequency = (0.3, 2.0)  # Frequency range for slow waves
test_trough_duration = (0.8, 2)  # Min and max trough duration
test_amplitude = {
    'neg_peak_threshold': -20.0,  # Negative peak threshold (μV)
    'peak_to_peak_threshold': 40.0  # Min Peak-to-peak amplitude threshold (μV)
}

# 5. Run slow wave detection
print("Running slow wave detection...")

slow_waves = event_processor.detect_slow_waves(
    method=test_method,
    chan=test_channels,
    frequency=test_frequency,
    trough_duration=test_trough_duration,
    neg_peak_thresh=test_amplitude['neg_peak_threshold'],
    p2p_thresh=test_amplitude['peak_to_peak_threshold'],
    polar='normal', # 'normal' or 'opposite'
    stage=test_stages,
    reject_artifacts=True,
    reject_arousals=False,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir=json_dir,
    create_empty_json=True
)

# Export results
test_method_str = "_".join(test_method).replace('/', '_') if isinstance(test_method, list) else str(test_method).replace('/', '_')

freq_range = f"{test_frequency[0]}-{test_frequency[1]}Hz"
stages_str = "".join(test_stages)
file_pattern = f"slowwaves_{test_method_str}_{freq_range}_{stages_str}"



param2CSV = event_processor.export_slow_wave_parameters_to_csv(
    json_input=json_dir,
    csv_file=os.path.join(json_dir, f'sw_parameters_{test_method_str}_{freq_range}_{stages_str}.csv'),
    file_pattern=file_pattern
)

density2CSV = event_processor.export_slow_wave_density_to_csv(
    json_input=json_dir,
    csv_file=os.path.join(json_dir, f'sw_density_{test_method_str}_{freq_range}_{stages_str}.csv'),
    stage=test_stages,
    file_pattern=file_pattern
)

csv2db = event_processor.import_parameters_csv_to_database(
    csv_file     = os.path.join(json_dir, f'sw_parameters_{test_method}_{freq_range}_{stages_str}.csv'),
    db_path      = db_path
    )



print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
print(f"Slow wave parameters saved")
print(f"Slow wave density saved")
print(f"ALL DONE")
print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")