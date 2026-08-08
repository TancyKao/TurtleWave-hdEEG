"""
hdEEG_kcomplex_detector.py

Detect K-complexes in high-density EEG data using TurtleWave-hdEEG.

KCs are detected with the AASM/Massimini2004 criteria (≥75 µV peak-to-peak,
0.25–1.0 s trough duration), restricted to N2, with a 1.0 s isolation
filter so a KC can't be one cycle of a continuous slow-oscillation train.
Override `test_stages` to also include N3 if needed.
"""

import os
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

from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalKC, CustomAnnotations, fmt_freq_token

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


# 1. Paths --------------------------------------------------------------
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
datafilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"

data_file = os.path.join(root_dir, datafilename)
annot_file = _cli.annot if _cli.annot else os.path.join(root_dir, "wonambi", annotfilename)
out_dir = os.path.join(root_dir, "wonambi", "kc_results")
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
if _cli.channels:
    from turtlewave_hdEEG.utils import read_channels_from_csv
    test_channels = read_channels_from_csv(_cli.channels)
    print(f"Channels from --channels: {len(test_channels)}")
test_stages = ['NREM2']                   # default; add 'NREM3' if needed
test_frequency = (0.1, 4.0)
# Min/max duration of the NEGATIVE HALF-WAVE, not of the whole K-complex.
test_trough_duration = (0.25, 1.0)
test_neg_peak_thresh = -40.0              # µV trough depth (AASM/Massimini2004)
test_p2p_thresh = 75.0                    # µV peak-to-peak (AASM/Massimini2004)
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
    json_dir=out_dir,
    subject=_cli.subject,
    # neural_events.db is the store of record. --legacy-json opts back into
    # the per-channel JSON the export/import steps below consume.
    write_db=False if _cli.legacy_json else True,
    db_path=None if _cli.legacy_json else db_path,
    resume=_cli.resume,
)


# 6. Report / export ---------------------------------------------------
method_str = str(test_method).replace('/', '_')
freq_range = fmt_freq_token(*test_frequency)
stages_str = "".join(test_stages)

if not _cli.legacy_json:
    # K-complexes are already in neural_events.db with det_* and spectral
    # columns and a detection_runs provenance row. Density is derived from the
    # database on read -- its denominator is the artefact-free in-stage time
    # this run analysed, stored in analysed_time. A flat CSV can still be
    # produced on demand:
    #   from turtlewave_hdEEG import export_events_to_csv
    #   export_events_to_csv(db_path, 'k_complex', test_method, test_frequency,
    #                        test_stages, output_dir=out_dir)
    from turtlewave_hdEEG.density import event_density, format_density_table

    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"K-complex events written to: {db_path}")
    try:
        # Rejection settings must match the detection call above: they are part
        # of the analysed_time key, so a mismatch divides by a different amount
        # of recording time.
        density_df = event_density(
            db_path, event_type='k_complex', method=test_method,
            stage=test_stages, subject=_cli.subject,
            reject_artifacts=True, reject_arousals=True)
        print("K-complex density (events per minute of artefact-free in-stage time):")
        print(format_density_table(density_df))
    except (ValueError, FileNotFoundError) as e:
        print(f"K-complex density unavailable: {e}")
    print("ALL DONE")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
else:
    # Legacy path: aggregate the per-channel JSON written above into CSVs and
    # import them. file_pattern must use the same band token the detector used.
    file_pattern = f"kcomplex_{method_str}_{freq_range}_{stages_str}"
    param_csv = os.path.join(
        out_dir, f'kc_parameters_{method_str}_{freq_range}_{stages_str}.csv')
    density_csv = os.path.join(
        out_dir, f'kc_density_{method_str}_{freq_range}_{stages_str}.csv')

    event_processor.export_kc_parameters_to_csv(
        json_input=out_dir, csv_file=param_csv, file_pattern=file_pattern,
        frequency=test_frequency,
    )
    # Same rejection settings as the detection call above, so the density
    # denominator matches the recording time actually analysed.
    event_processor.export_kc_density_to_csv(
        json_input=out_dir, csv_file=density_csv, stage=test_stages,
        file_pattern=file_pattern,
        reject_artifacts=True, reject_arousals=True,
    )
    event_processor.initialize_sqlite_database(db_path)
    event_processor.import_parameters_csv_to_database(
        csv_file=param_csv, db_path=db_path, method=test_method)

    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
    print(f"K-complex parameters saved: {param_csv}")
    print(f"K-complex density saved:    {density_csv}")
    print(f"Imported into database:     {db_path}")
    print("~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^")
