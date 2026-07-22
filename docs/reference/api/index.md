# API Reference

This section provides detailed technical documentation for all TurtleWave hdEEG modules and classes.

## Core Modules

### Event Processing

- [**Event Processor**](eventprocessor.md) - Core event detection functionality
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

- [`EventProcessor`](eventprocessor.md) - Base class for event detection
- [`SWProcessor`](swprocessor.md) - Slow wave detection
- [`PACProcessor`](pacprocessor.md) - Phase-amplitude coupling

**For GUI usage:**

- [`TurtleWaveGUI`](gui.md) - Main application window

## Usage Pattern

All processor modules follow a consistent pattern:

```python
from turtlewave_hdEEG import ProcessorClass

# Initialize
processor = ProcessorClass(
    eeg_file='path/to/data.edf',
    output_dir='path/to/output'
)

# Configure
processor.set_parameters(...)

# Process
results = processor.detect_events()

# Save
processor.save_results('output.h5')
```

Refer to individual module documentation for specific parameters and methods.