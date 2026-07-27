# TurtleWave hdEEG Documentation

Welcome to the TurtleWave hdEEG documentation! TurtleWave is a Python toolkit for high-density EEG event detection and analysis in sleep research.

## What is TurtleWave?

TurtleWave provides tools for:

- **Automated event detection** - Detect sleep spindles, slow waves, and phase-amplitude coupling
- **Manual review** - Validate detected events with an efficient GUI
- **Batch processing** - Process large datasets on HPC clusters
- **Visualization** - Explore EEG data and detected events interactively

## Quick Links

### 🎓 New to TurtleWave?

Start with our tutorials to learn by doing:

- [**Getting Started**](tutorials/getting-started.md) - Your first event detection
- [**EEG Review GUI Tutorial**](tutorials/eeg-review-gui-tutorial.md) - Learn to review detected events

### 🔧 Need to Solve a Specific Problem?

Check our how-to guides for practical solutions:

- [**Installation**](how-to/installation.md) - Set up TurtleWave
- [**Detect Spindles**](how-to/detect-spindles.md) - Run spindle detection
- [**Detect Slow Waves**](how-to/detect-slow-waves.md) - Run slow wave detection
- [**Review EEG Events**](how-to/review-eeg-events.md) - Validate detected events efficiently

### 📚 Looking for Technical Details?

Browse our reference documentation:

- [**API Reference**](reference/api/index.md) - Complete API documentation
- [**EEG Review GUI Reference**](reference/eeg-review-gui.md) - GUI components and features

### 🧠 Want to Understand How It Works?

Read our explanations:

- [**Overview**](explanation/overview.md) - System architecture and design
- [**EEG Review GUI Architecture**](explanation/eeg-review-gui-architecture.md) - GUI design principles

## Features

### Event Detection

- **Sleep Spindles** - Wavelet-based and bandpass detection methods
- **Slow Waves** - Amplitude and slope-based detection
- **Phase-Amplitude Coupling (PAC)** - Cross-frequency coupling analysis

### Review GUI

- **QC-driven triage** - Spot outlier channels first, then drill into their epochs
- **Keyboard-driven workflow** - Flag a channel for re-detection with `F`
- **Flexible filtering** - Filter by event type, channel, sleep stage, method, frequency band
- **Performance optimized** - Handle 100,000+ events smoothly

### Data Formats

- **Input:** EEGLAB (.set/.fdt), EDF, MNE-compatible formats
- **Output:** JSON per channel, aggregated CSV, and a SQLite database (`neural_events.db`)

## Installation

```bash
pip install turtlewave-hdEEG
```

For detailed installation instructions, see the [Installation Guide](how-to/installation.md).

## Quick Start

### Detect Sleep Spindles

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

# Load EEG data and annotations
data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

# Detect spindles
event_processor = ParalEvents(dataset=data, annotations=annot)
spindles = event_processor.detect_spindles(
    method='Ferrarelli2007',
    chan=['Cz', 'Fz'],
    frequency=(11, 16),
    stage=['NREM2', 'NREM3'],
    json_dir='wonambi/spindle_results',
)

# Aggregate the per-channel JSON into CSV, then into the database
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

See [`examples/hdEEG_spindle_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_spindle_detector.py)
for the full script, including density export. `write_db=True` can also
write straight to the database and skip the CSV step — see
[Write Detection Results Directly to the Database](how-to/direct-to-database-detection.md).

### Review Detected Events

```bash
eeg_review_gui
```

Then load your database, EEG file, and annotations, triage channels on the
**Channels (QC)** tab, drill into a channel's **Epochs** to inspect outliers,
and press **F** to flag a channel for re-detection.

See the [EEG Review GUI Tutorial](tutorials/eeg-review-gui-tutorial.md) for a complete walkthrough.

## Documentation Structure

This documentation follows the [Diátaxis framework](DIATAXIS_FRAMEWORK.md), organizing content into four types:

| Type | Purpose | When to Use |
|------|---------|-------------|
| **Tutorials** | Learning-oriented lessons | You're new and want to learn by doing |
| **How-to Guides** | Problem-solving recipes | You have a specific task to accomplish |
| **Reference** | Technical specifications | You need to look up details |
| **Explanation** | Understanding-oriented discussion | You want to understand how/why it works |

## Support

- **Issues and questions:** [GitHub Issues](https://github.com/TancyKao/TurtleWave-hdEEG/issues)

## Contributing

We welcome contributions! See our [repository](https://github.com/TancyKao/TurtleWave-hdEEG) for details.

## Citation

If you use TurtleWave in your research, please cite:

```bibtex
@software{turtlewave2024,
  title = {TurtleWave: High-density EEG Event Detection for Sleep Research},
  author = {TurtleWave Development Team},
  year = {2024},
  url = {https://github.com/TancyKao/TurtleWave-hdEEG}
}
```

## License

TurtleWave is released under the MIT License. See [LICENSE](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/LICENSE) for details.
