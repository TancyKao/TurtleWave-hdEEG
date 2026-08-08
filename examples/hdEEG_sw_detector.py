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
from turtlewave_hdEEG import ParalSWA, CustomAnnotations, fmt_freq_token
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
_p.add_argument('--legacy-json', dest='legacy_json', action='store_true',
                help='opt back into the legacy JSON -> CSV -> import pipeline. '
                     'By default events go straight into neural_events.db and '
                     'no per-channel JSON or intermediate CSV is written.')
# Accepted so existing command lines keep working; it now names the default.
_p.add_argument('--write-db', dest='write_db_flag', action='store_true',
                help=_ap.SUPPRESS)
_p.add_argument('--subject', dest='subject', default=None,
                help='subject id keying the density denominator '
                     '(default: derived from the annotation/recording path)')
_p.add_argument('--resume', dest='resume', action='store_true',
                help='skip channels already completed for this exact '
                     'method/band/stage scope in the database')
_cli, _ = _p.parse_known_args()

# 1. Define the file paths for the dataset and annotations
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"

channels_csv_path = os.path.join(root_dir, "channels.csv")

# Read channels from CSV
#test_channels = read_channels_from_csv(channels_csv_path)
#print(f"Channels loaded from CSV: {test_channels}")

# Construct the full paths
data_file = os.path.join(root_dir, datafilename)
annot_file = _cli.annot if _cli.annot else os.path.join(root_dir, "wonambi", annotfilename)
out_dir = os.path.join(root_dir, "wonambi", "sw_results")
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
test_channels = ['E110','E111','E112']  # Channels
if _cli.channels:
    test_channels = read_channels_from_csv(_cli.channels)
    print(f"Channels from --channels: {len(test_channels)}")
test_stages = ['NREM2','NREM3'] # ['NREM2', 'NREM3']
test_frequency = (0.5, 1.25)  # Frequency range for slow waves
# Min/max duration of the NEGATIVE HALF-WAVE (Massimini's 0.3-1.0 s), not of
# the whole wave. None = the chosen method's published window, which is the
# only setting that is right for all four methods. Ignored by Ngo2015 and
# Staresina2015, which use min_dur/max_dur instead.
test_trough_duration = None
# For Massimini2004 / AASM/Massimini2004 these ARE the paper's amplitude
# criteria and they override it: set both to None to run the published values
# (-80/140 uV and -40/75 uV respectively). The values below are kept because
# this example's method is Staresina2015, which does not read them as
# amplitude criteria at all -- they only feed ParalSWA's legacy post-hoc
# filter. Change the method above and you should change these too.
test_amplitude = {
     'neg_peak_threshold': -75.0,  # trough depth (μV); sign ignored
     'peak_to_peak_threshold': 75.0  # min peak-to-peak amplitude (μV)
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
    reject_arousals=True,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir=out_dir,
    subject=_cli.subject,
    # neural_events.db is the store of record. --legacy-json opts back into
    # the per-channel JSON the export/import steps below consume.
    write_db=False if _cli.legacy_json else True,
    db_path=None if _cli.legacy_json else db_path,
    resume=_cli.resume,
)

# Export results
test_method_str = "_".join(test_method).replace('/', '_') if isinstance(test_method, list) else str(test_method).replace('/', '_')

freq_range = fmt_freq_token(*test_frequency)
stages_str = "".join(test_stages)

if not _cli.legacy_json:
    # Slow waves are already in neural_events.db with det_* and spectral
    # columns and a detection_runs provenance row. Density is derived from the
    # database on read -- its denominator is the artefact-free in-stage time
    # this run analysed, stored in analysed_time. A flat CSV can still be
    # produced on demand:
    #   from turtlewave_hdEEG import export_events_to_csv
    #   export_events_to_csv(db_path, 'slow_wave', test_method, test_frequency,
    #                        test_stages, output_dir=out_dir)
    from turtlewave_hdEEG.density import event_density, format_density_table

    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"Slow wave events written to: {db_path}")
    try:
        # Rejection settings must match the detection call above: they are part
        # of the analysed_time key, so a mismatch divides by a different amount
        # of recording time.
        density_df = event_density(
            db_path, event_type='slow_wave', method=test_method,
            stage=test_stages, subject=_cli.subject,
            reject_artifacts=True, reject_arousals=True)
        print("Slow wave density (events per minute of artefact-free in-stage time):")
        print(format_density_table(density_df))
    except (ValueError, FileNotFoundError) as e:
        print(f"Slow wave density unavailable: {e}")
    print("ALL DONE")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
else:
    # Legacy path: aggregate the per-channel JSON written above into CSVs and
    # import them. file_pattern must use the same band token the detector used.
    file_pattern = f"slowwaves_{test_method_str}_{freq_range}_{stages_str}"

    param2CSV = event_processor.export_slow_wave_parameters_to_csv(
        json_input=out_dir,
        csv_file=os.path.join(out_dir, f'sw_parameters_{test_method_str}_{freq_range}_{stages_str}.csv'),
        file_pattern=file_pattern
    )

    # Pass the same rejection settings the detection call used: the density
    # denominator must be the recording time the detector actually analysed,
    # otherwise every density is biased.
    density2CSV = event_processor.export_slow_wave_density_to_csv(
        json_input=out_dir,
        csv_file=os.path.join(out_dir, f'sw_density_{test_method_str}_{freq_range}_{stages_str}.csv'),
        stage=test_stages,
        file_pattern=file_pattern,
        reject_artifacts=True,
        reject_arousals=True
    )

    # Pass the UNESCAPED method. test_method_str ('AASM_Massimini2004') is the
    # filesystem-safe form used in filenames only; the direct-write path and
    # the other drivers store the original 'AASM/Massimini2004'. Storing both
    # spellings splits one run across two methods with no UNIQUE collision,
    # which double-counts densities and hides half the events from any
    # method-scoped query.
    csv2db = event_processor.import_parameters_csv_to_database(
        csv_file     = os.path.join(out_dir, f'sw_parameters_{test_method_str}_{freq_range}_{stages_str}.csv'),
        db_path      = db_path,
        event_type   = 'slow_wave',
        method       = test_method
        )

    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"Slow wave parameters saved")
    print(f"Slow wave density saved")
    print(f"ALL DONE (legacy JSON+CSV path)")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
