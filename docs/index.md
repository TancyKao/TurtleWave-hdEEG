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

- **High-throughput review** - Process thousands of events efficiently
- **Keyboard-driven workflow** - Rapid accept/reject decisions
- **Flexible filtering** - Filter by event type, channel, sleep stage, confidence
- **Performance optimized** - Handle 100,000+ events smoothly

### Data Formats

- **Input:** EEGLAB (.set/.fdt), EDF, MNE-compatible formats
- **Output:** SQLite databases, CSV files, XML annotations

## Installation

```bash
pip install turtlewave-hdEEG
```

For detailed installation instructions, see the [Installation Guide](how-to/installation.md).

## Quick Start

### Detect Sleep Spindles

```python
from turtlewave_hdEEG import LargeDataset, EventProcessor

# Load EEG data
dataset = LargeDataset('subject001.set')

# Detect spindles
processor = EventProcessor(dataset)
spindles = processor.detect_spindles(
    channels=['Cz', 'Fz'],
    freq_range=(11, 16),
    method='wavelet'
)

# Save results
spindles.to_csv('spindles.csv')
```

### Review Detected Events

```bash
python -m frontend.eeg_review_gui
```

Then:

1. Load your event database
2. Load your EEG file
3. Review events using keyboard shortcuts (A = accept, R = reject)
4. Export reviewed events

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

- **Issues:** [GitHub Issues](https://github.com/your-repo/turtlewave-hdEEG/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/turtlewave-hdEEG/discussions)
- **Email:** support@turtlewave.org

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/your-repo/turtlewave-hdEEG/blob/main/CONTRIBUTING.md) for details.

## Citation

If you use TurtleWave in your research, please cite:

```bibtex
@software{turtlewave2024,
  title = {TurtleWave: High-density EEG Event Detection for Sleep Research},
  author = {TurtleWave Development Team},
  year = {2024},
  url = {https://github.com/your-repo/turtlewave-hdEEG}
}
```

## License

TurtleWave is released under the MIT License. See [LICENSE](https://github.com/your-repo/turtlewave-hdEEG/blob/main/LICENSE) for details.
