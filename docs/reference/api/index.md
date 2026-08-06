# API Reference

This section provides detailed technical documentation for all TurtleWave hdEEG modules and classes.

## Core Modules

### Event Processing

- [**Spindle Processor**](eventprocessor.md) - Core spindle detection functionality
- [**Slow Wave Processor**](swprocessor.md) - Slow wave detection algorithms
- [**PAC Processor**](pacprocessor.md) - Phase-amplitude coupling analysis

### Database & Provenance

- [**Direct-to-Database Write**](dbwrite.md) - the `write_db=None` (default) detection path, `detection_runs` provenance, DB → CSV export
- [**Event Density**](density.md) - per-channel density derived on read from `neural_events.db`
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
object, then detects straight into `neural_events.db` — no export or import
step:

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

# Load
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Initialize
event_processor = ParalEvents(dataset=data, annotations=annot)

# Detect. write_db defaults to None (AUTO): events go straight into
# neural_events.db in wonambi/, resolved from json_dir. No per-channel JSON
# is written.
event_processor.detect_spindles(
    method='Ferrarelli2007',
    chan=['E110', 'E111'],
    frequency=(11, 16),
    stage=['NREM2', 'NREM3'],
    json_dir='wonambi/spindle_results',
)
```

`ParalSWA.detect_slow_waves`, `ParalKC.detect_kcomplexes` and
`ParalPAC.analyze_pac` follow the same shape. Query the database directly
afterwards — see
[Read the database with pandas and R](../../how-to/read-database-with-pandas-and-r.md)
— or pull a CSV back out on demand with
[`export_events_to_csv`](dbwrite.md). Pass `write_db=False` (or
`--legacy-json` on the driver scripts) to restore the pre-4.2 JSON → CSV →
`import_parameters_csv_to_database` pipeline instead — see
[Write Detection Results Directly to the Database](../../how-to/direct-to-database-detection.md).
Refer to individual module documentation for specific parameters and methods.

!!! note
    The legacy `export_*_parameters_to_csv` / `export_*_density_to_csv`
    exporters (only relevant on the `write_db=False` path) default to
    `strict=True`: a `file_pattern` that matches zero JSON files raises
    `FileNotFoundError` instead of writing an empty placeholder CSV. The
    density exporters are additionally deprecated in favour of
    [`density.event_density`](density.md).
    `import_parameters_csv_to_database` raises rather than returning an
    error dict, and accepts `event_type=` / `method=` / `force=` — see
    [Upgrade to 4.0, Step 7](../../how-to/upgrade-to-4.0.md#step-7-exporters-and-importers-now-raise-instead-of-failing-silently)
    if you have existing scripts built around the old return-a-dict
    behaviour. Build `file_pattern`'s frequency segment with
    [`fmt_freq_token`](dbwrite.md) rather than a hand-written f-string — see
    [About naming, subject identity & provenance conventions](../../explanation/naming-and-identity-conventions.md).
