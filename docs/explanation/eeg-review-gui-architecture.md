# Explanation: EEG Review GUI Architecture

This document explains the design principles behind the TurtleWave EEG
Review GUI — why it's built around channel-level QC triage rather than
per-event review.

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

**What this trades away:** there's no built-in mechanism for one reviewer to
mark individual borderline events for a second opinion, and no per-event
provenance trail. If your study design needs inter-rater agreement on
individual events (rather than channel-level QC decisions), this GUI is not
the right tool for that layer of review.

## Design Principles

**1. Two Surfaces, Not Three**

A landing dashboard (**Channels (QC)**) for triage, and a drill-down
(**Epochs**) for inspecting *why* a channel was flagged. There is no
middle "event list" surface — once you've decided a channel needs a closer
look, you go straight to its epochs, not to a filtered table of its
individual events.

**2. Channel Verdicts, Not Event Verdicts**

Decisions (keep / drop / mark-artefact / queue-for-re-detect) are recorded
per channel, not per event. This matches how the decisions actually get
used downstream: a dropped channel is excluded from analysis wholesale; a
re-detect-queued channel gets re-run with different parameters, not
individually corrected event-by-event.

**3. QC Triage Feeds Re-detection, Not Manual Correction**

The GUI has no way to hand-edit an event's boundaries or manually add a
missed event. Its output is a channel-level verdict and, optionally, a
scoped re-detect request — the fix for a bad channel is better detection
parameters or exclusion, not manual patching of individual events.

**4. Performance Over Features**

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

QC verdicts and artefact ranges are written straight back into
`neural_events.db`, alongside the detected events they describe — there's no
separate reviews file to keep in sync.

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
central trade-off of the whole redesign (see "Why QC-by-Outlier-Triage,
Not Per-Event Review?" above). It buys throughput on the common case
(spotting bad channels across a high-density montage) at the cost of
fine-grained, per-event manual correction.

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
