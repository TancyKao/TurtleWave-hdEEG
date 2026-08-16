# API Reference

This section provides detailed technical documentation for all TurtleWave hdEEG modules and classes.

## Core Modules

### Event Processing

- [**Spindle Processor**](eventprocessor.md) - Core spindle detection functionality
- [**Slow Wave Processor**](swprocessor.md) - Slow wave detection algorithms
- [**PAC Processor**](pacprocessor.md) - Phase-amplitude coupling analysis

### Database & Provenance

- [**Direct-to-Database Write**](dbwrite.md) - `write_db=True` detection path, `detection_runs` provenance, DB → CSV export
- [**Re-run Guards**](rerun.md) - Correctness guards for scoped channel re-detection
- [**Utilities**](utils.md) - Artefact-free density denominators and other shared helpers

### User Interface

- [**GUI Components**](gui.md) - Graphical user interface

## Quick Navigation

**For event detection:**

- [`ParalEvents`](eventprocessor.md) - Spindle detection
- [`ParalSWA`](swprocessor.md) - Slow wave detection
- [`ParalPAC`](pacprocessor.md) - Phase-amplitude coupling

**For GUI usage:**

- [`TurtleWaveGUI`](gui.md) - Main detection/annotation application window

## Usage Pattern

Every `Paral*` processor takes a `wonambi.dataset.Dataset` and an annotations
object, then follows the same detect → export → import pipeline:

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

# Load
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Initialize
event_processor = ParalEvents(dataset=data, annotations=annot)

# Detect (writes one JSON file per channel to json_dir)
event_processor.detect_spindles(
    method='Ferrarelli2007',
    chan=['E110', 'E111'],
    frequency=(11, 16),
    stage=['NREM2', 'NREM3'],
    json_dir='wonambi/spindle_results',
)

# Aggregate the JSON into CSV, then import into the database
event_processor.export_spindle_parameters_to_csv(
    json_input='wonambi/spindle_results',
    csv_file='wonambi/spindle_results/spindle_parameters.csv',
    file_pattern='spindles_Ferrarelli2007',
)
event_processor.import_parameters_csv_to_database(
    csv_file='wonambi/spindle_results/spindle_parameters.csv',
    db_path='wonambi/neural_events.db',
)
```

`ParalSWA.detect_slow_waves` follows the same shape, via
`export_slow_wave_parameters_to_csv` and `import_parameters_csv_to_database`.
`ParalPAC.analyze_pac` detects and exports to CSV the same way, using its own
`export_pac_parameters_to_csv` / `import_pac_csv_to_database` pair. Passing
`write_db=True` to a `detect_*` call (or `write_db=True` to `analyze_pac`)
writes straight to the database instead, skipping the JSON/CSV round-trip —
see [Write Detection Results Directly to the Database](../../how-to/direct-to-database-detection.md).
Refer to individual module documentation for specific parameters and methods.