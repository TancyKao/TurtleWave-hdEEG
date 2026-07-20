# TurtleWave hdEEG Overview

## What is TurtleWave?

TurtleWave is a Python package designed for event detection in high-density EEG sleep data. It extends the capabilities of Wonambi to efficiently handle large datasets, making it particularly suitable for research involving high-density EEG recordings during sleep.

## Purpose and Design

The package was developed to address the specific challenges of processing high-density EEG data:

- **Scalability**: Built to handle large datasets that are common in high-density EEG recordings
- **Sleep-specific**: Optimized for sleep research applications
- **Extended functionality**: Builds upon Wonambi's foundation while adding specialized features for event detection

## Core Capabilities

TurtleWave provides several key capabilities for sleep EEG analysis:

### Event Detection

The package specializes in detecting various sleep-related events:

- **Sleep spindles**: Transient oscillatory patterns during sleep
- **Slow waves**: Large amplitude, low-frequency oscillations
- **Phase-amplitude coupling**: Relationships between different frequency bands

### Data Processing

TurtleWave handles the complexities of high-density EEG data:

- Efficient processing of multi-channel recordings
- Artifact detection and handling
- Sleep stage annotation support
- Arousal detection

### Event Density is Artefact-Free

Event density (events per minute) is computed by dividing the event count by
the **artefact-free** in-stage time the detector actually pooled — the same
clean time `fetch` used during detection — not by the sum of all scored
epochs of a stage. Detection already excludes artefact/arousal epochs before
threshold estimation, so dividing by all scored epochs systematically
under-counted density in proportion to a recording's artefact load.

This means density values computed with the current exporters are **higher**
than densities computed before this fix, for the same detection run. The two
are not comparable: if you have older density CSVs, regenerate them from the
underlying detection JSON/database rather than mixing old and new values in
the same analysis. Whole-night density additionally restricts to the stages
that were actually detected on, excluding Wake unless Wake was itself a
detection stage.

The shared calculation lives in
[`compute_analysed_seconds` / `build_density_denominators`](../reference/api/utils.md),
which reproduce Wonambi's `fetch(reject_epoch=True, reject_artf=...)`
segmentation so the denominator matches the detector's input exactly.

### Analysis Workflow

The package supports a complete analysis pipeline from raw data to results, integrating:

- Data loading and preprocessing
- Automated event detection
- Statistical analysis
- Result visualization and export

## Technical Foundation

### Dependencies

TurtleWave is built on a foundation of established scientific Python libraries:

**Core Requirements:**

- Python ≥3.8
- NumPy ≥1.17.0 - Numerical computing
- SciPy ≥1.3.0 - Scientific computing
- Matplotlib ≥3.1.0 - Visualization
- h5py ≥2.10.0 - HDF5 file handling
- PyQt5 ≥5.12.0 - Graphical interface
- Wonambi ≥7.15 - EEG analysis foundation

**Additional Dependencies:**

- pandas - Data manipulation
- tensorpac - Phase-amplitude coupling analysis

### Architecture

The package is organized into specialized processors:

- **Event Processor**: Core event detection logic
- **SW Processor**: Slow wave detection algorithms
- **PAC Processor**: Phase-amplitude coupling analysis
- **GUI Components**: User interface for interactive analysis

## Use Cases

TurtleWave is particularly well-suited for:

### Research Applications

- Sleep neuroscience research requiring high-density EEG
- Studies investigating sleep oscillations and their coupling
- Large-scale sleep studies with multiple subjects
- Investigations of sleep microstructure

### Clinical Applications

- Sleep disorder research
- Neurological condition assessment during sleep
- Treatment effect monitoring

## Design Philosophy

The package follows several key design principles:

**Extensibility**: Built on Wonambi's architecture, allowing for customization and extension

**Efficiency**: Optimized for handling the computational demands of high-density recordings

**Usability**: Provides both programmatic API and graphical interface for different user needs

**Reproducibility**: Supports standardized workflows for consistent analysis across studies

## License

TurtleWave is released under the MIT License, an OSI-approved open source license that permits free use, modification, and distribution.

## Relationship to Wonambi

TurtleWave extends Wonambi specifically for high-density EEG applications. While Wonambi provides excellent general-purpose EEG analysis capabilities, TurtleWave adds:

- Enhanced scalability for high-channel-count recordings
- Specialized event detection algorithms optimized for sleep data
- Additional analysis tools for phase-amplitude coupling
- Workflow optimizations for batch processing

Users familiar with Wonambi will find TurtleWave's interface and concepts familiar, while benefiting from the additional capabilities for high-density sleep EEG analysis.