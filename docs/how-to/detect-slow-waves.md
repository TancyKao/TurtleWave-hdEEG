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
2. Click **"Run Slow Wave Detection"**
3. Wait for processing to complete

![Slow Wave Detection Interface](../images/gui-slow-wave-detection.png)
*Slow Wave Detection tab showing available parameters and channel selection*

The results will be saved to your output directory as `*_slowwaves.h5`.

### Adjusting Detection Parameters

To customize slow wave detection for your specific needs:

**Frequency Range:**

1. Locate the **"Frequency Range"** controls
2. Set the low frequency (default: 0.5 Hz)
3. Set the high frequency (default: 4.0 Hz)

Typical slow wave frequencies are 0.5-4 Hz, but you may adjust based on your research requirements.

**Duration Criteria:**

1. Find the **"Duration"** settings
2. Set minimum duration (default: 0.5 seconds)
3. Set maximum duration (default: 2.0 seconds)

**Amplitude Threshold:**

1. Locate **"Amplitude Threshold"** controls
2. Adjust the threshold multiplier (default: 1.5)
3. Higher values = more stringent detection

**Channel Selection:**

To detect on specific channels:

1. Click **"Select Channels"**
2. Choose the channels of interest
3. Click **"Apply"**

If no channels are selected, all channels will be processed.

### Running Detection

After configuring parameters:

1. Click **"Run Slow Wave Detection"**
2. Monitor progress in the status panel
3. Review detection statistics when complete

![Detection Tabs Overview](../images/gui-detection-tabs.png)
*Overview of all detection tabs: Spindle Detection, Slow Wave Detection, and PAC Analysis*

## Using the Python API

For programmatic access or batch processing:

```python
from turtlewave_hdEEG import SWProcessor

# Initialize the processor
sw_processor = SWProcessor(
    eeg_file='path/to/data.edf',
    output_dir='path/to/output'
)

# Configure detection parameters
sw_processor.set_frequency_range(low=0.5, high=4.0)
sw_processor.set_duration_range(min_dur=0.5, max_dur=2.0)
sw_processor.set_amplitude_threshold(1.5)

# Run detection
results = sw_processor.detect_slow_waves()

# Save results
sw_processor.save_results('slowwaves.h5')
```

## Interpreting Results

The output HDF5 file contains:

- **Event times**: Start and end times for each detected slow wave
- **Channels**: Which channel(s) each event was detected on
- **Amplitude**: Peak-to-peak amplitude of each event
- **Frequency**: Dominant frequency of each event
- **Sleep stage**: Associated sleep stage for each event

To load and examine results:

```python
import h5py
import pandas as pd

# Load results
with h5py.File('output_slowwaves.h5', 'r') as f:
    events = pd.DataFrame({
        'start_time': f['events/start_time'][:],
        'end_time': f['events/end_time'][:],
        'channel': f['events/channel'][:],
        'amplitude': f['events/amplitude'][:],
        'frequency': f['events/frequency'][:]
    })

# View summary statistics
print(events.describe())
```

## Optimizing Detection

### For High Sensitivity

If you want to detect more slow waves (higher sensitivity):

- Lower the amplitude threshold (e.g., 1.0 or 1.25)
- Widen the frequency range slightly
- Reduce minimum duration requirement

### For High Specificity

If you want only clear, unambiguous slow waves:

- Increase the amplitude threshold (e.g., 2.0 or higher)
- Narrow the frequency range
- Increase minimum duration requirement

### For Specific Sleep Stages

To detect slow waves only during specific sleep stages:

```python
# Using Python API
sw_processor.set_sleep_stages(['N2', 'N3'])  # Only NREM stages 2 and 3
results = sw_processor.detect_slow_waves()
```

## Common Issues

### No Slow Waves Detected

If detection produces no results:

- **Check annotations**: Ensure sleep staging completed successfully
- **Verify data quality**: Check for excessive artifacts
- **Adjust thresholds**: Try lowering the amplitude threshold
- **Check frequency range**: Ensure it matches your data characteristics

### Too Many False Positives

If you're getting many artifacts detected as slow waves:

- **Increase threshold**: Raise the amplitude threshold
- **Improve preprocessing**: Run artifact detection first
- **Narrow frequency range**: Use more restrictive frequency bounds
- **Check specific channels**: Some channels may be noisier

### Performance Issues

If detection is slow:

- **Reduce channel count**: Process only channels of interest
- **Use batch processing**: Process multiple files overnight
- **Check system resources**: Ensure adequate RAM available

## Batch Processing Multiple Files

To process multiple files efficiently:

```python
from pathlib import Path
from turtlewave_hdEEG import SWProcessor

# Get all EEG files
eeg_files = Path('data/').glob('*.edf')

# Process each file
for eeg_file in eeg_files:
    print(f"Processing {eeg_file.name}...")
    
    sw_processor = SWProcessor(
        eeg_file=str(eeg_file),
        output_dir='results/'
    )
    
    # Use consistent parameters
    sw_processor.set_frequency_range(low=0.5, high=4.0)
    sw_processor.set_amplitude_threshold(1.5)
    
    # Detect and save
    results = sw_processor.detect_slow_waves()
    sw_processor.save_results(f"{eeg_file.stem}_slowwaves.h5")
    
    print(f"Detected {len(results)} slow waves")
```

## Next Steps

After detecting slow waves, you might want to:

- [Perform phase-amplitude coupling analysis](pac-analysis.md)
- [Export results for statistical analysis](export-results.md)
- [Visualize detected events](visualize-events.md)
- Compare with spindle detection results