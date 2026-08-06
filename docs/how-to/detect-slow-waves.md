# How to Detect Slow Waves

This guide shows you how to detect slow waves in your sleep EEG data using TurtleWave.

## Prerequisites

Before detecting slow waves, ensure you have:

- Loaded your EEG data file
- Generated sleep annotations
- Set an output directory

If you haven't done these steps, refer to the [Getting Started tutorial](../tutorials/getting-started.md).

## Using the GUI

### Basic Detection

To detect slow waves with default parameters:

1. Open the **Slow Wave Detection** tab
2. Click **"Detect Slow Waves"**
3. Wait for processing to complete

![Slow Wave Detection Interface](../images/gui-slow-wave-detection.png)
*Slow Wave Detection tab showing available parameters and channel selection*

Results are written straight into `neural_events.db` in your output
directory's `wonambi/` folder — there is no per-channel JSON or CSV step, and
no GUI toggle to opt back into one. If you need the legacy JSON/CSV files,
run `examples/hdEEG_sw_detector.py --legacy-json` instead of the GUI tab.

### Adjusting Detection Parameters

To customize slow wave detection for your specific needs:

**Frequency Range:**

1. Locate the **"Frequency Range (Hz)"** controls
2. Set the low frequency (default: 0.1 Hz)
3. Set the high frequency (default: 4 Hz)

Typical slow wave frequencies are 0.5-4 Hz, but you may adjust based on your research requirements.

**Trough Duration:**

1. Find the **"Trough Duration (Negative Half-Wave)"** group
2. Set the minimum and maximum trough duration in seconds (default: 0.3-1.5 s)

**Amplitude Thresholds:**

1. Locate the **"Amplitude Thresholds"** group
2. Set the negative peak threshold in µV (default: -80 µV)
3. Set the peak-to-peak threshold in µV (default: 140 µV)

More negative/higher values make detection more stringent.

**Channel Selection:**

Select the channels of interest before running detection. If no channels are
selected, all channels will be processed.

### Running Detection

After configuring parameters:

1. Click **"Detect Slow Waves"**
2. Monitor progress in the status panel
3. Review detection statistics when complete

## Using the Python API

For programmatic access or batch processing, mirror
[`examples/hdEEG_sw_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_sw_detector.py):

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalSWA, CustomAnnotations

# Load dataset and annotations
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Create the processor
event_processor = ParalSWA(dataset=data, annotations=annot)

# Run detection. write_db defaults to None (AUTO): events go straight into
# neural_events.db resolved from json_dir; no per-channel JSON is written.
slow_waves = event_processor.detect_slow_waves(
    method='Massimini2004',
    chan=['E110', 'E111', 'E112'],
    frequency=(0.5, 1.25),
    trough_duration=(0.3, 1.5),
    neg_peak_thresh=-75.0,
    p2p_thresh=75.0,
    stage=['NREM2', 'NREM3'],
    reject_artifacts=True,
    reject_arousals=True,
    json_dir='wonambi/sw_results',
    subject='sub-001',
)
```

`method` also accepts `'AASM/Massimini2004'`, `'Ngo2015'`, or `'Staresina2015'`.
`polar='opposite'` is available for inverted-reference recordings.

## Interpreting Results

Slow waves are in `neural_events.db` (`events` table, `event_type =
'slow_wave'`) as soon as detection returns — there's no export or import
step. Query it with pandas or R:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('wonambi/neural_events.db')
slow_waves = pd.read_sql_query(
    "SELECT channel, start_time, duration, stage, min_amp, peak2peak_amp, "
    "det_trough, det_peak FROM events "
    "WHERE event_type = 'slow_wave' AND method = 'Massimini2004'",
    conn,
)
```

and report density from the database directly — its denominator is the
artefact-free in-stage time this run actually analysed, stored automatically
in `analysed_time`:

```python
from turtlewave_hdEEG.density import event_density, format_density_table

density_df = event_density(
    'wonambi/neural_events.db', event_type='slow_wave', method='Massimini2004',
    stage=['NREM2', 'NREM3'], subject='sub-001',
    reject_artifacts=True, reject_arousals=True,  # must match the detection call
)
print(format_density_table(density_df))
```

See [Read the database with pandas and R](read-database-with-pandas-and-r.md)
for more query patterns, including how to pull a flat CSV back out with
`export_events_to_csv` if a downstream tool needs one.

!!! note "Using the legacy JSON → CSV → import path instead"
    If you passed `write_db=False` above, detection wrote one JSON file per
    channel to `json_dir` instead. Aggregate and import it the pre-4.2 way:
    ```python
    from turtlewave_hdEEG.dbwrite import fmt_freq_token

    freq_range = fmt_freq_token(0.5, 1.25)  # must match `frequency` above
    file_pattern = f"slowwaves_Massimini2004_{freq_range}_NREM2NREM3"

    event_processor.export_slow_wave_parameters_to_csv(
        json_input='wonambi/sw_results',
        csv_file='wonambi/sw_results/sw_parameters.csv',
        file_pattern=file_pattern,
    )
    event_processor.export_slow_wave_density_to_csv(  # deprecated; JSON-only
        json_input='wonambi/sw_results',
        csv_file='wonambi/sw_results/sw_density.csv',
        stage=['NREM2', 'NREM3'],
        file_pattern=file_pattern,
    )
    event_processor.import_parameters_csv_to_database(  # deprecated
        csv_file='wonambi/sw_results/sw_parameters.csv',
        db_path='wonambi/neural_events.db',
        method='Massimini2004',
    )
    ```
    Build `file_pattern`'s frequency segment with `fmt_freq_token`, not a
    hand-written f-string — a formatter that doesn't match what
    `detect_slow_waves` wrote matches zero JSON files and, by default, raises
    `FileNotFoundError` rather than silently exporting an empty CSV.
    `import_parameters_csv_to_database` now raises rather than returning
    `{"error": ..., "added": 0}` on failure, and refuses to import over rows
    already written by the direct-write path unless you pass `force=True` —
    see
    [Write Detection Results Directly to the Database](direct-to-database-detection.md#pull-a-csv-back-out-of-the-database)
    and
    [About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md).

## Optimizing Detection

### For High Sensitivity

If you want to detect more slow waves (higher sensitivity):

- Lower `neg_peak_thresh` / `p2p_thresh` (e.g. -60 / 70 µV)
- Widen the frequency range slightly
- Reduce the minimum trough duration

### For High Specificity

If you want only clear, unambiguous slow waves:

- Raise `neg_peak_thresh` / `p2p_thresh` (e.g. -90 / 150 µV)
- Narrow the frequency range
- Increase the minimum trough duration

### For Specific Sleep Stages

Pass the stages you want directly to `stage`:

```python
slow_waves = event_processor.detect_slow_waves(
    method='Massimini2004',
    chan=test_channels,
    stage=['NREM2', 'NREM3'],  # Only NREM stages 2 and 3
)
```

## Common Issues

### No Slow Waves Detected

If detection produces no results:

- **Check annotations**: Ensure sleep staging completed successfully
- **Verify data quality**: Check for excessive artifacts
- **Adjust thresholds**: Try relaxing `neg_peak_thresh` / `p2p_thresh`
- **Check frequency range**: Ensure it matches your data characteristics

### Too Many False Positives

If you're getting many artifacts detected as slow waves:

- **Raise thresholds**: Increase `neg_peak_thresh` / `p2p_thresh`
- **Improve preprocessing**: Run artifact detection first
- **Narrow frequency range**: Use more restrictive frequency bounds
- **Check specific channels**: Some channels may be noisier

### Performance Issues

If detection is slow:

- **Reduce channel count**: Process only channels of interest
- **Use batch processing**: Process multiple files overnight
- **Check system resources**: Ensure adequate RAM available

## Batch Processing Multiple Files

To process multiple files efficiently, loop over subjects and reuse the same
`ParalSWA` call shape:

```python
import os
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalSWA, CustomAnnotations

subjects = ['sub-001', 'sub-002', 'sub-003']

for subject in subjects:
    root_dir = f'data/{subject}/'
    data = WonambiDataset(os.path.join(root_dir, f'{subject}_eeg.set'))
    annot = CustomAnnotations(os.path.join(root_dir, 'wonambi', f'{subject}_annotations.xml'))

    event_processor = ParalSWA(dataset=data, annotations=annot)
    # AUTO resolves neural_events.db as a sibling of json_dir, i.e.
    # data/<subject>/wonambi/neural_events.db — one database per subject.
    slow_waves = event_processor.detect_slow_waves(
        method='Massimini2004',
        chan=['E110', 'E111', 'E112'],
        frequency=(0.5, 1.25),
        stage=['NREM2', 'NREM3'],
        json_dir=os.path.join(root_dir, 'wonambi', 'sw_results'),
        subject=subject,
    )
    print(f"{subject}: detected slow waves on {len(slow_waves)} channels")
```

For HPC batch runs across many subjects, see `examples/NCI_commands/` and the
`*_GADI.py` driver scripts referenced in the project README.

## Next Steps

After detecting slow waves, you might want to:

- Run spindle detection for comparison — [`examples/hdEEG_spindle_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_spindle_detector.py)
- Review detected events in the QC dashboard — [Review EEG Events](review-eeg-events.md)
- Read the database from pandas or R — [Read the Database with pandas and R](read-database-with-pandas-and-r.md)
- Resume interrupted runs, verify coverage, or opt out to legacy JSON — [Direct-to-Database Detection](direct-to-database-detection.md)
