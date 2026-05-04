"""
hdEEG_kcomplex_detector.py

Detect K-complexes in high-density EEG data using TurtleWave-hdEEG.

KCs are detected with the AASM/Massimini2004 criteria (≥75 µV peak-to-peak,
0.25–1.0 s trough duration), restricted to N2, with a 1.0 s isolation
filter so a KC can't be one cycle of a continuous slow-oscillation train.
Override `test_stages` to also include N3 if needed.
"""

import os

from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalKC, CustomAnnotations


# 1. Paths --------------------------------------------------------------
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"

data_file = os.path.join(root_dir, datafilename)
annot_file = os.path.join(root_dir, "wonambi", annotfilename)
json_dir = os.path.join(root_dir, "wonambi", "kc_results")
db_path = os.path.join(root_dir, "wonambi", "neural_events.db")


# 2. Load dataset and annotations --------------------------------------
print("Loading dataset and annotations...")
data = WonambiDataset(data_file)
annot = CustomAnnotations(annot_file)


# 3. Create ParalKC instance -------------------------------------------
event_processor = ParalKC(dataset=data, annotations=annot)


# 4. Parameters --------------------------------------------------------
test_method = 'AASM/Massimini2004'        # or 'Massimini2004'
test_channels = ['E110', 'E111', 'E112']
test_stages = ['NREM2']                   # default; add 'NREM3' if needed
test_frequency = (0.1, 4.0)
test_trough_duration = (0.25, 1.0)
test_neg_peak_thresh = -37.0              # µV (AASM KC default)
test_p2p_thresh = 70.0                    # µV (AASM KC default)
test_min_isolation = 1.0                  # seconds between successive KC troughs


# 5. Run detection -----------------------------------------------------
print("Running K-complex detection...")
kcomplexes = event_processor.detect_kcomplexes(
    method=test_method,
    chan=test_channels,
    frequency=test_frequency,
    trough_duration=test_trough_duration,
    neg_peak_thresh=test_neg_peak_thresh,
    p2p_thresh=test_p2p_thresh,
    min_isolation=test_min_isolation,
    polar='normal',
    stage=test_stages,
    reject_artifacts=True,
    reject_arousals=True,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir=json_dir,
    create_empty_json=True,
)


# 6. Export ------------------------------------------------------------
method_str = str(test_method).replace('/', '_')
freq_range = f"{test_frequency[0]}-{test_frequency[1]}Hz"
stages_str = "".join(test_stages)
file_pattern = f"kcomplex_{method_str}_{freq_range}_{stages_str}"

param_csv = os.path.join(
    json_dir, f'kc_parameters_{method_str}_{freq_range}_{stages_str}.csv')
density_csv = os.path.join(
    json_dir, f'kc_density_{method_str}_{freq_range}_{stages_str}.csv')

event_processor.export_kc_parameters_to_csv(
    json_input=json_dir, csv_file=param_csv, file_pattern=file_pattern,
    frequency=test_frequency,
)
event_processor.export_kc_density_to_csv(
    json_input=json_dir, csv_file=density_csv, stage=test_stages,
    file_pattern=file_pattern,
)
event_processor.initialize_sqlite_database(db_path)
event_processor.import_parameters_csv_to_database(
    csv_file=param_csv, db_path=db_path, method=test_method)


print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
print(f"K-complex parameters saved: {param_csv}")
print(f"K-complex density saved:    {density_csv}")
print(f"Imported into database:     {db_path}")
print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
