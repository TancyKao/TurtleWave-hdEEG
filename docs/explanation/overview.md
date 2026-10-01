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

### Which Annotation Events Are Rejected By Default

Every detector (`detect_spindles`, `detect_slow_waves`, `detect_kcomplexes`,
`analyze_pac`) and the density denominator share one `reject_types`
parameter: the annotation event types whose time is excluded from detection
and subtracted from the artefact-free time density is divided by. Since 4.4
the default set is `Artefact`, `Arousal`, `Move` — `Move` joined the other
two in this release. `Resp` (respiratory events — hypopnea, obstructive
apnea, SpO2 desaturation) and `Snore` are deliberately **not** in the default
set; both are opt-in via `reject_types=[..., 'Resp']` /
`reject_types=[..., 'Snore']`.

The reasoning, in brief:

- **`Move` is masked by default** because a movement annotation marks
  mechanical, non-neural signal — electrode and cable movement produce large
  low-frequency deflections, exactly the failure mode the amplitude-threshold
  slow-wave and K-complex detectors are most vulnerable to. This is also
  established practice: the largest published spindle study (Purcell et al.
  2017, *Nat Commun* 8:15930, 11,630 individuals) removed "any epoch with an
  overlapping arousal, movement or signal artefact annotation."
- **`Resp` is opt-in, not masked by default**, for two reasons. First, the
  EEG signature of a respiratory event is its terminating arousal, which
  `Arousal` already excludes — what remains inside a `Resp` window is
  ordinary scored sleep. Second, in a sleep-disordered-breathing cohort
  `Resp` time is proportional to disease severity: masking it by default
  would make the "clean sleep" that survives a severity-dependent
  subsample, which attenuates or inverts the very between-group effect a
  study is usually powered to detect. Published practice bears this out —
  Mohammadi et al. 2021 (*Front Neurol* 12:598632) state explicitly that
  apnea/hypopnea events were not filtered from their spindle analysis
  "because of the critical impact of these events on EEG activity," and
  D'Rozario et al. 2023 (*SLEEP* 46(12):zsad255), the closest published match
  to this toolkit's hd-EEG/OSA use case, excluded artefacts and arousals
  only. `Resp` is a legitimate opt-in for K-complex detection specifically —
  see the note in [Detect K-Complexes](../how-to/detect-kcomplexes.md) —
  because respiratory events can evoke a genuine cortical K-complex at
  termination, and masking that is a scientific choice about what counts as
  "spontaneous," not a clean-up step. Some published spindle work *does* mask
  `Resp`, but for a narrower, band-specific reason than a general artefact
  concern: apnea-related alpha intrusion contaminating slow-spindle
  (<13 Hz) detection specifically — the practice Mohammadi et al. 2021
  characterize and reject as a basis for a general exclusion. That is a
  reason to offer `Resp` as an opt-in for slow-spindle/alpha-adjacent
  analyses, not a reason to mask it by default for every analysis.
- **`Snore` is opt-in, not masked by default**, for the same
  severity-proportional-mask reason as `Resp`, and because the documented
  scalp signature of snoring-related artefact is a ~30 Hz burst at frontal
  electrodes — outside both the sigma band (11-16 Hz) and the slow-wave band
  (0.5-4 Hz), so the detectors' own bandpass already attenuates it.

One reject set is shared across all four detectors on purpose, not
per-event-type defaults: slow-wave/spindle coupling pairs events detected in
two separate runs, so if the two runs excluded different time the coupling
denominator would be undefined. PAC pays a second, PAC-specific cost for
every excluded type: `analyze_pac` concatenates the surviving fragments into
one continuous signal (`cat=(1, 1, 1, 0)`), so masked windows are not
analysed as separate pieces but cut out and rejoined — each join is a step
discontinuity that the phase filter smears across a neighbourhood on either
side, not a boundary the analysis respects. A wider reject set means more
such splices, not more dropped data (nothing is dropped by a duration floor
here — see [Run PAC Analysis](../how-to/run-pac-analysis.md) for the detail).

See [`resolve_reject_types` / `DEFAULT_REJECT_TYPES`](../reference/api/utils.md)
for the exact resolution rules (including the deprecated `reject_artifacts=`/
`reject_arousals=` boolean shims), and
[Write Detection Results Directly to the Database](../how-to/direct-to-database-detection.md)
for how the resolved set is recorded in `detection_runs.reject_types` and
keys the `analysed_time` denominator.

### EEGLAB Boundary Splices Are Masked as Artefact

`XLAnnotations.add_artefacts_from_events` also writes an `Artefact`
annotation around every EEGLAB `boundary` event — the marker EEGLAB writes
wherever a segment of data was cut out and the remaining samples spliced
together. The signal can step discontinuously at that instant: on one
subject (107 splices × 257 channels) the median step across a splice was
1.5 µV, but 7 boundaries stepped more than 30 µV on at least one channel —
large enough to seed a false slow wave, and exactly the kind of instant a
slow-oscillation/spindle coupling phase estimate must never straddle.

The masked window is a fixed ±2 s around the boundary's onset — **not** the
event's own `duration` field, which for a `boundary` event is the length of
data *removed*, expressed in the *original* recording's time base rather
than a span in the surviving, spliced recording; using it as the window end
would reject perfectly good post-splice data (about 36 minutes on one
measured subject). The cost of the ±2 s mask itself was about 1.6% of
analysable time on the same subject, so expect a small, splice-count-
dependent shift in density (both numerator and denominator move together)
on any recording with EEGLAB boundary events, relative to one processed
before this release.

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
- NumPy ==1.26.4 - Numerical computing (pinned for Wonambi compatibility)
- SciPy ≥1.3.0 - Scientific computing
- Matplotlib ≥3.1.0 - Visualization
- h5py ≥2.10.0 - HDF5 file handling
- PyQt5 ≥5.12.0 - Graphical interface
- Wonambi ==7.15 - EEG analysis foundation (pinned; its API is what the extension classes inherit from)

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