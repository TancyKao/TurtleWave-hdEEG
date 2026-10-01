# Explanation: EEG Review GUI Architecture

This document explains the design principles behind the TurtleWave EEG
Review GUI — why it's built around channel-level QC triage, and why the
per-event decisions added in 4.6 are a validation sample rather than a way to
curate every event.

## Why QC-by-Outlier-Triage, Not Per-Event Review?

Automated event detection on a high-density montage produces detections on
every channel. In practice, the review bottleneck is rarely "is this one
spindle real?" — it's "which channels have implausible physiology (dead,
noisy, or systematically over/under-detecting) and need to be excluded or
re-run?" A handful of bad channels can dominate a study's downstream
statistics even when every other channel is fine.

Earlier versions of this GUI supported per-event accept/reject at scale
(stratified sampling, confidence thresholds, a Compare-methods view). That
model assumes the unit of doubt is the event. In practice, reviewers were
using it to spot bad channels anyway — scrolling through hundreds of
individually "rejected" events on the same one or two noisy channels. 4.0.0
made that the actual workflow: the GUI computes an outlier score per
channel (robust z-score against the rest of the montage, plus a dead-channel
check) and reviewers triage at that granularity instead.

**What this trades away:** per-event review at scale. A night holds about
94,000 spindles, so no reviewer can decide each one. 4.6 adds per-event
decisions back at a scale a person can do: a drawn sample of about 120 events
per subject, labelled to estimate how often the detector is right, per region
and stage, with a second rater on a shared subset for inter-rater agreement.
It does not curate the output. Decisions are stored as evidence in their own
table and never remove an event; see
[Event figures and review sampling](event-figures-and-review-sampling.md).

## Design Principles

**1. Two Surfaces, Not Three**

A landing dashboard (**Channels (QC)**) for triage, and a drill-down
(**Epochs**) for inspecting *why* a channel was flagged. There is no
middle "event list" surface — once you've decided a channel needs a closer
look, you go straight to its epochs, not to a filtered table of its
individual events.

**2. Channel Verdicts for Action, Event Decisions for Evidence**

Decisions that change the analysis (keep / drop / mark-artefact /
queue-for-re-detect) are recorded per channel. This matches how they get used
downstream: a dropped channel is excluded wholesale; a re-detect-queued channel
gets re-run with different parameters. Event decisions (accept / reject /
unsure, with a reason) are different in kind. They are stored in
`event_reviews`, keyed by the event's uuid and the reviewer, and feed a
precision estimate. A rejected event stays in `events`, and density counts it
unless a reader asks to leave rejected events out.

**3. QC Triage Feeds Re-detection, Not Manual Correction**

The GUI has no way to hand-edit an event's boundaries or manually add a
missed event. Its output is a channel-level verdict and, optionally, a
scoped re-detect request — the fix for a bad channel is better detection
parameters or exclusion, not manual patching of individual events.

**4. Figures Computed Once, at Detection**

The numbers in the Event panel and the Channels-tab population checks are
stored with each event at detection time, not computed when you click. The GUI
only reads them, which keeps the montage-wide dashboard fast. A run from before
4.6 has none, and the GUI says so instead of computing something different.

**5. Performance Over Features**

Virtualized rendering, background waveform loading, and waveform caching
keep navigation responsive on large datasets. Features that would compromise
performance are deliberately omitted.

## Data Flow

```text
neural_events.db  →  per-channel QC aggregates  →  Channels (QC) table
                                                          │
                                            select + drill into a channel
                                                          ▼
                                                    Epochs panel
                                                          │
                                    mark artefact / queue re-detect
                                                          ▼
                                        channel_qc / qc_artefact_intervals
                                     (written back into neural_events.db)
                                                          │
                          Export QC report… (Markdown)  ◄─┤─►  Build re-detect
                                                             request… (JSON) /
                                                             Export Re-run
                                                             Package…
```

QC verdicts, artefact ranges, event decisions and sample data are written
straight back into `neural_events.db`, alongside the detected events they
describe — there's no separate reviews file to keep in sync.

## Key Components

### EventDatabase: The Data Layer

Abstracts querying events and channel-level QC state from SQLite. Verdicts
and artefact intervals are indexed the same way the events themselves are,
so filtering by channel, event type, method, or frequency band stays fast
even with a large montage and many detection runs in the same database.

### The QC Aggregation Layer

Per-channel QC metrics (event count, density, max peak-to-peak amplitude,
outlier flag) are computed from the events table on load and on threshold
change, not stored — so adjusting the outlier `z`-thresholds (**View →
Outlier threshold…**) recomputes flags immediately without touching the
database.

### Population Checks

Five per-channel columns (off-band share, low-prominence share, share at the
duration floor, median amplitude against background and against threshold) are
aggregated from the stored figures in one query per refresh, in a background
thread. They are flagged against the rest of the montage with the same robust z
as the amplitude columns, so a problem shared by every channel is not flagged
there and shows up in the precision estimate instead.

### Event Decisions and the Review Sample

Selecting an event, deciding it and undoing a decision are handled on the
Epochs tab; a sample of events is drawn and scored by the library
(`turtlewave_hdEEG.review_sampling`), so the sampling and the estimator can be
tested and run without Qt. Other reviewers' decisions are hidden by default so a
second rater stays independent.

### Epoch-Level Inspection

The Epochs panel exists because a channel-level flag alone doesn't tell you
*why* a channel looks bad — a channel can be flagged for one bad hour in an
otherwise clean recording. Stepping through 30-second windows (with outlier
epochs marked and `P`/`N` hopping between them) lets a reviewer distinguish
"globally noisy channel" from "channel with one contaminated stretch," which
determines whether the right fix is dropping the channel or marking an
artefact range.

### Background Waveform Loading

EEG data loads from disk in a background thread so navigating between
channels and epochs doesn't block the GUI. Loading takes 100-500ms for large
files; without threading every navigation step would stall. Recently viewed
waveforms are cached in memory for instant re-display.

## Design Trade-offs

**Memory vs. speed** — QC aggregates and the current event-type slice are
held in memory rather than re-queried per interaction, trading RAM for
instant sorting and filtering.

**Two fixed tabs vs. a customizable layout** — a consistent Channels →
Epochs progression reduces cognitive load and lets muscle memory develop,
at the cost of flexibility for workflows this GUI wasn't designed for.

**Channel-level granularity vs. event-level control** — this is the
central trade-off of the redesign (see "Why QC-by-Outlier-Triage, Not
Per-Event Review?" above). It buys throughput on the common case (spotting bad
channels across a high-density montage) at the cost of per-event manual
correction. The 4.6 sample recovers an honest per-event measure, precision,
without that cost.

## Why PyQt5 and pyqtgraph?

**PyQt5** gives native, cross-platform rendering and a mature ecosystem, at
the cost of a steeper API and a GPL/commercial licensing choice. Tkinter was
too slow for large datasets; a web-based UI (Dash/Streamlit) would add
network latency and deployment overhead for what is fundamentally a local,
single-user desktop tool.

**pyqtgraph** was chosen over matplotlib for the waveform and epoch strip
plots because it renders fast enough for interactive navigation — matplotlib
redraws became visibly laggy once the GUI needed to update a plot on every
epoch step or drag. The migration from matplotlib to pyqtgraph predates the
QC-triage redesign.

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md) - Learn by doing
- [How-to Guide: Review EEG Events](../how-to/review-eeg-events.md) - Solve specific problems
- [Reference: EEG Review GUI](../reference/eeg-review-gui.md) - Technical specifications
- [How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md) - What changed from the pre-4.0 per-event review workflow
- [Diátaxis Framework](../DIATAXIS_FRAMEWORK.md) - Documentation philosophy
