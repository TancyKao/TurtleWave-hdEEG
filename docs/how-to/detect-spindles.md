# How to Detect Spindles

This guide shows you how to detect sleep spindles in your sleep EEG data using
TurtleWave.

## Prerequisites

Before detecting spindles, ensure you have:

- Loaded your EEG data file
- Generated sleep annotations
- Set an output directory

If you haven't done these steps, refer to the [Getting Started tutorial](../tutorials/getting-started.md).

## Using the GUI

### Basic Detection

To detect spindles with default parameters:

1. Open the **Spindle Detection** tab
2. Click **"Detect Spindles"**
3. Wait for processing to complete

![Spindle Detection Interface](../images/gui-spindle-detection.png)
*Spindle Detection tab showing available parameters and channel selection*

Results are written straight into `neural_events.db` in your output
directory's `wonambi/` folder — there is no per-channel JSON or CSV step, and
no GUI toggle to opt back into one. If you need the legacy JSON/CSV files,
run `examples/hdEEG_spindle_detector.py --legacy-json` instead of the GUI tab.

### Adjusting Detection Parameters

To customize spindle detection for your specific needs:

**Detection Method:**

1. Locate the **"Method"** dropdown
2. Choose a detector: `Ferrarelli2007`, `Moelle2011`, `Nir2011`,
   `Wamsley2012`, `Martin2013`, `Ray2015`, or `Lacourse2018`

Each method implements a different published algorithm; they differ in
threshold basis (RMS, sigma-band envelope) and smoothing, so detected counts
will vary across methods on the same data.

**Frequency Range:**

1. Locate the **"Frequency Range (Hz)"** controls
2. Set the low frequency (default: 11 Hz)
3. Set the high frequency (default: 16 Hz)

Typical spindle bands are 11-16 Hz (all spindles) or split into slow
(9-12 Hz) / fast (12-16 Hz) sub-bands depending on your research question.

**Duration:**

1. Find the **"Duration (s)"** group
2. Set the minimum and maximum spindle duration in seconds (default: 0.5-3 s)

**Channel Selection:**

Select the channels of interest before running detection. If no channels are
selected, all channels will be processed.

### Running Detection

After configuring parameters:

1. Click **"Detect Spindles"**
2. Monitor progress in the status panel
3. Review detection statistics when complete

## Using the Python API

For programmatic access or batch processing, mirror
[`examples/hdEEG_spindle_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_spindle_detector.py):

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

# Load dataset and annotations
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Create the processor
event_processor = ParalEvents(dataset=data, annotations=annot)

# Run detection. write_db defaults to None (AUTO): events go straight into
# neural_events.db resolved from json_dir (or an explicit db_path); no
# per-channel JSON is written.
spindles = event_processor.detect_spindles(
    method='Moelle2011',
    chan=['E110', 'E111', 'E112'],
    frequency=(11, 13),
    duration=(0.5, 3),
    stage=['NREM2', 'NREM3'],
    reject_artifacts=True,
    reject_arousals=False,
    cat=(1, 1, 1, 0),  # concatenate across cycles, stages, and discontinuities
    save_to_annotations=False,
    json_dir='wonambi/spindle_results',
    subject='sub-001',
)
```

`method` also accepts `'Ferrarelli2007'`, `'Nir2011'`, `'Wamsley2012'`,
`'Martin2013'`, `'Ray2015'`, or `'Lacourse2018'`. `polar='opposite'` is
available for inverted-reference recordings, and `ref_chan=[...]` re-references
before detection.

`cat` is passed straight through to Wonambi's `wonambi.trans.select.fetch`,
which defines it as a 4-position concatenation flag (`0` = keep separate,
`1` = concatenate):

- position 1 — concatenate across sleep **cycles**
- position 2 — concatenate across sleep **stages**
- position 3 — concatenate across **discontinuous** segments of signal
  (gaps within an otherwise matching stage/cycle/event-type condition)
- position 4 — concatenate across **event types** (not relevant here, since
  `detect_spindles` only fetches one event type at a time)

So `cat=(1, 1, 1, 0)` merges signal across cycles, stages, and
discontinuities into one continuous run per channel before detection, rather
than detecting cycle-by-cycle or stage-by-stage. Use `cat=(0, 0, 0, 0)` if you
want cycles and stages kept separate instead.

## Interpreting Results

Spindles are in `neural_events.db` (`events` table, `event_type = 'spindle'`)
as soon as detection returns — there's no export or import step. Query it
with pandas or R:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('wonambi/neural_events.db')
spindles = pd.read_sql_query(
    "SELECT channel, start_time, duration, stage, peak2peak_amp, rms "
    "FROM events WHERE event_type = 'spindle' AND method = 'Moelle2011'",
    conn,
)
```

and report density from the database directly — its denominator is the
artefact-free in-stage time this run actually analysed, stored automatically
in `analysed_time`:

```python
from turtlewave_hdEEG.density import event_density, format_density_table

density_df = event_density(
    'wonambi/neural_events.db', event_type='spindle', method='Moelle2011',
    stage=['NREM2', 'NREM3'], subject='sub-001',
    reject_artifacts=True, reject_arousals=False,  # must match the detection call
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

    freq_range = fmt_freq_token(11, 13)  # must match `frequency` above
    file_pattern = f"spindles_Moelle2011_{freq_range}_NREM2NREM3"

    event_processor.export_spindle_parameters_to_csv(
        json_input='wonambi/spindle_results',
        csv_file='wonambi/spindle_results/spindle_parameters.csv',
        file_pattern=file_pattern,
    )
    event_processor.export_spindle_density_to_csv(  # deprecated; JSON-only
        json_input='wonambi/spindle_results',
        csv_file='wonambi/spindle_results/spindle_density.csv',
        stage=['NREM2', 'NREM3'],
        file_pattern=file_pattern,
    )
    event_processor.import_parameters_csv_to_database(  # deprecated
        csv_file='wonambi/spindle_results/spindle_parameters.csv',
        db_path='wonambi/neural_events.db',
        method='Moelle2011',
    )
    ```
    Build `file_pattern`'s frequency segment with `fmt_freq_token`, not a
    hand-written f-string — a formatter that doesn't match what
    `detect_spindles` wrote (e.g. rounding `11` to `11.0`) matches zero JSON
    files and, by default, raises `FileNotFoundError` rather than silently
    exporting an empty CSV. `import_parameters_csv_to_database` accepts an
    optional `event_type=` / `method=` override and refuses to import over
    rows the direct-write path already wrote unless you pass `force=True` —
    see
    [Write Detection Results Directly to the Database](direct-to-database-detection.md#pull-a-csv-back-out-of-the-database)
    and
    [About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md).

## Optimizing Detection

### For High Sensitivity

If you want to detect more spindles (higher sensitivity):

- Widen the frequency range (e.g. 9-16 Hz)
- Widen the duration range, or lower the minimum duration
- Try `Wamsley2012` or `Lacourse2018`, which tend to be more permissive than
  amplitude-threshold methods like `Ferrarelli2007`

### For High Specificity

If you want only clear, unambiguous spindles:

- Narrow the frequency range to a sub-band (e.g. 12-14 Hz)
- Raise the minimum duration
- Pass method-specific threshold overrides via `**detector_params` (e.g.
  `det_thresh`, `sel_thresh` for RMS-based methods)

### For Specific Sleep Stages

Pass the stages you want directly to `stage`:

```python
spindles = event_processor.detect_spindles(
    method='Moelle2011',
    chan=test_channels,
    stage=['NREM2', 'NREM3'],  # Only NREM stages 2 and 3
)
```

## Common Issues

### No Spindles Detected

If detection produces no results:

- **Check annotations**: Ensure sleep staging completed successfully
- **Verify data quality**: Check for excessive artifacts
- **Check frequency range**: Ensure it matches your data's spindle band
- **Try a different method**: Methods differ substantially in sensitivity

### Too Many False Positives

If you're getting many artifacts detected as spindles:

- **Narrow frequency range**: Use a tighter sigma-band
- **Raise thresholds**: Pass a stricter `det_thresh` / `sel_thresh` via
  `**detector_params`
- **Improve preprocessing**: Run artifact detection first
- **Check specific channels**: Some channels may be noisier

### Performance Issues

If detection is slow:

- **Reduce channel count**: Process only channels of interest
- **Use batch processing**: Process multiple files overnight
- **Check system resources**: Ensure adequate RAM available

## Batch Processing Multiple Files

To process multiple files efficiently, loop over subjects and reuse the same
`ParalEvents` call shape:

```python
import os
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

subjects = ['sub-001', 'sub-002', 'sub-003']

for subject in subjects:
    root_dir = f'data/{subject}/'
    data = WonambiDataset(os.path.join(root_dir, f'{subject}_eeg.set'))
    annot = CustomAnnotations(os.path.join(root_dir, 'wonambi', f'{subject}_annotations.xml'))

    event_processor = ParalEvents(dataset=data, annotations=annot)
    # AUTO resolves neural_events.db as a sibling of json_dir, i.e.
    # data/<subject>/wonambi/neural_events.db — one database per subject.
    spindles = event_processor.detect_spindles(
        method='Moelle2011',
        chan=['E110', 'E111', 'E112'],
        frequency=(11, 13),
        stage=['NREM2', 'NREM3'],
        json_dir=os.path.join(root_dir, 'wonambi', 'spindle_results'),
        subject=subject,
    )
    print(f"{subject}: detected spindles on {len(spindles)} channels")
```

For HPC batch runs across many subjects, see `examples/NCI_commands/` and the
`*_GADI.py` driver scripts referenced in the project README.

## Next Steps

After detecting spindles, you might want to:

- Run slow wave detection for comparison — [Detect Slow Waves](detect-slow-waves.md)
- Analyze slow-wave/spindle coupling — [Run PAC Analysis](run-pac-analysis.md)
- Review detected events in the QC dashboard — [Review EEG Events](review-eeg-events.md)
- Read the database from pandas or R — [Read the Database with pandas and R](read-database-with-pandas-and-r.md)
- Resume interrupted runs, verify coverage, or opt out to legacy JSON — [Direct-to-Database Detection](direct-to-database-detection.md)
