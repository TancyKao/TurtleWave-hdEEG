# How to Detect K-Complexes

This guide shows you how to detect K-complexes in your sleep EEG data using
TurtleWave.

## Prerequisites

Before detecting K-complexes, ensure you have:

- Loaded your EEG data file
- Generated sleep annotations
- Set an output directory

If you haven't done these steps, refer to the [Getting Started tutorial](../tutorials/getting-started.md).

## Using the Python API

K-complex detection follows the same `Paral*` shape as spindles and slow
waves, but through the `ParalKC` class. Mirror
[`examples/hdEEG_kcomplex_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_kcomplex_detector.py):

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalKC, CustomAnnotations

# Load dataset and annotations
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Create the processor
event_processor = ParalKC(dataset=data, annotations=annot)

# Run detection
kcomplexes = event_processor.detect_kcomplexes(
    method='AASM/Massimini2004',
    chan=['E110', 'E111', 'E112'],
    frequency=(0.1, 4.0),
    trough_duration=(0.25, 1.0),
    neg_peak_thresh=-37.0,
    p2p_thresh=70.0,
    min_isolation=1.0,
    polar='normal',
    stage=['NREM2'],
    reject_artifacts=True,
    reject_arousals=True,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir='wonambi/kc_results',
)
```

`method` accepts `'AASM/Massimini2004'` (default) or `'Massimini2004'`. No
other Wonambi slow-wave methods are exposed here, since they target slow
oscillations rather than the isolated K-complex morphology. The AASM defaults
(`neg_peak_thresh=-37.0`, `p2p_thresh=70.0`, `trough_duration=(0.25, 1.0)`)
are the parameters shown above.

`min_isolation` is K-complex-specific: it is the minimum gap, in seconds,
required between successive K-complex trough times. It is what distinguishes
an isolated K-complex from one cycle of a continuous N3 slow-oscillation
train — set it to `0` to disable the isolation filter. K-complexes are
typically scored in N2 only; pass `stage=['NREM2', 'NREM3']` to also include
N3.

## Interpreting Results

Detection writes one JSON file per channel to `json_dir`, using the
`kcomplex_` filename prefix so results stay distinct from slow waves on disk.
Aggregate them into CSV, then import into the database. K-complex export
reuses the `ParalSWA` CSV helpers internally (K-complex parameters are
structurally identical to slow-wave parameters), so `export_kc_parameters_to_csv`
/ `export_kc_density_to_csv` pass the K-complex event type through for you:

```python
from turtlewave_hdEEG.dbwrite import fmt_freq_token

method_db = 'AASM/Massimini2004'   # canonical, unescaped — the value used in the DB
method_str = method_db.replace('/', '_')  # filesystem-safe — filenames/patterns ONLY
freq_range = fmt_freq_token(0.1, 4.0)     # must match the `frequency` passed to detect_kcomplexes
stages_str = 'NREM2'
file_pattern = f'kcomplex_{method_str}_{freq_range}_{stages_str}'

param_csv = f'wonambi/kc_results/kc_parameters_{method_str}_{freq_range}_{stages_str}.csv'
density_csv = f'wonambi/kc_results/kc_density_{method_str}_{freq_range}_{stages_str}.csv'

event_processor.export_kc_parameters_to_csv(
    json_input='wonambi/kc_results',
    csv_file=param_csv,
    file_pattern=file_pattern,
    frequency=(0.1, 4.0),
)

event_processor.export_kc_density_to_csv(
    json_input='wonambi/kc_results',
    csv_file=density_csv,
    stage=['NREM2'],
    file_pattern=file_pattern,
)

event_processor.initialize_sqlite_database('wonambi/neural_events.db')
event_processor.import_parameters_csv_to_database(
    csv_file=param_csv,
    db_path='wonambi/neural_events.db',
    method=method_db,  # the canonical, UNESCAPED method string — never method_str
)
```

!!! warning
    Always pass `method=` explicitly to `ParalKC.import_parameters_csv_to_database`
    with `method_db`, the original unescaped method string (e.g.
    `'AASM/Massimini2004'`, not the filename-escaped `'AASM_Massimini2004'`).
    The importer's filename parser breaks on the escaped form and would
    otherwise record just `'AASM'`. `method_db` (canonical, unescaped, for the
    database) vs. `method_str` (filesystem-safe, filenames only) is a
    project-wide convention — see
    [About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md).

    It also now raises rather than returning `{"error": ..., "added": 0}` on
    failure, and refuses to import over rows already written by the
    direct-to-database path unless you pass `force=True`.

The parameters CSV includes start/end time, channel, sleep stage, amplitude,
duration, and frequency for each K-complex, tagged with the `'k_complex'`
event type — this keeps K-complexes distinguishable from slow waves once both
are imported into the same `neural_events.db`.

Alternatively, pass `write_db=True` and `db_path=...` to `detect_kcomplexes`
to skip the JSON→CSV→import round-trip and write events straight into the
database under the `'k_complex'` event type — see
[Write Detection Results Directly to the Database](direct-to-database-detection.md).

## Optimizing Detection

### For High Sensitivity

If you want to detect more K-complexes:

- Lower `neg_peak_thresh` / `p2p_thresh` (e.g. −30 / 60 µV)
- Widen `trough_duration`
- Shorten `min_isolation`

### For High Specificity (AASM-strict)

If you want only unambiguous, AASM-conforming K-complexes:

- Use the AASM defaults: `neg_peak_thresh=-37.0`, `p2p_thresh=70.0`,
  `trough_duration=(0.25, 1.0)`
- Keep `min_isolation=1.0` so consecutive slow-oscillation cycles in N3 are
  not double-counted as separate K-complexes
- Restrict `stage=['NREM2']` only

## Common Issues

### No K-Complexes Detected

If detection produces no results:

- **Check annotations**: Ensure N2 sleep staging completed successfully
- **Verify data quality**: Check for excessive artifacts
- **Relax thresholds**: Try lowering `neg_peak_thresh` / `p2p_thresh`

### Too Many False Positives / Slow-Oscillation Contamination

If N3 slow oscillations are being counted as separate K-complexes:

- **Restrict to N2**: Keep `stage=['NREM2']`
- **Increase `min_isolation`**: A longer minimum gap between troughs filters
  out consecutive slow-oscillation cycles

## Next Steps

After detecting K-complexes, you might want to:

- Run slow wave detection for comparison — [Detect Slow Waves](detect-slow-waves.md)
- Review detected events in the QC dashboard — [Review EEG Events](review-eeg-events.md)
- Write results directly to the database — [Direct-to-Database Detection](direct-to-database-detection.md)
