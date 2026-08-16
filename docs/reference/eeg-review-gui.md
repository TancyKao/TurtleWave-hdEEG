# Reference: EEG Review GUI

The EEG Review GUI (`frontend.eeg_review_gui`) is a PyQt5/PyQtGraph application
for QC-triaging automatically detected EEG events (spindles, slow waves,
K-complexes, PAC) at the channel level.

**Module:** `frontend.eeg_review_gui`

**Main class:** `EventReviewGUI(QMainWindow)`

**Launch command:**

```bash
eeg_review_gui
```

## Layout

Two tabs, plus two docks:

- **1 · Channels (QC)** (landing tab) — a sortable per-channel table for the
  current event type, filtered by outlier flag (`hard` / `soft` / `dead` /
  `ok`), with actions to mark a channel as an artefact, queue it for
  re-detection, or drill into its epochs.
- **2 · Epochs** — steps through 30-second windows for a single channel, with
  a hypnogram strip, outlier markers, and range-marking for artefacts.
- **Filters dock** (left) — event type, detection method, frequency band, and
  channel selection, applied globally across both tabs.
- **Topography & detail dock** (right) — scalp topography for the current QC
  metric, the global worst-events list, and the selected channel's detail.

## Menu Bar

| Menu | Notable actions |
|------|------------------|
| File | Open Database…, Open EEG File…, Open Annotation File…, Exit |
| Edit | Flag selected channel for re-detect (`F`) |
| View | Outlier threshold…, toggle Filters dock / Topography & detail dock |
| Analysis | Refresh QC dashboard, Build re-detect request… |
| Export | Export QC report…, Export Re-run Package…, Export Figure… |
| Help | Design notes, About |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Flag the selected channel in the Channels (QC) table for re-detection |
| `P` / `N` | On the Epochs tab, jump to the previous / next outlier epoch |
| Prev / Next buttons | Step one epoch at a time |

There is no per-event accept/reject shortcut — QC verdicts are set per
channel, not per event.

## Data In / Out

**Input:** a `neural_events.db` SQLite database (created by the `Paral*`
detection pipeline), an EEGLAB `.set`/`.fdt` (or other Wonambi-supported)
EEG file, and optionally a Wonambi annotation XML for sleep stages.

**Output:**

- **Export QC report…** — a Markdown summary (per-channel QC table, flagged
  channels, marked artefact ranges) for the current event type.
- **Build re-detect request…** — a `redetect_request.json` written next to
  the annotation XML for `turtlewave_gui` to pick up.
- **Export Re-run Package…** — a snapshot of the current results plus
  `channels.csv` and a sidecar annotation XML for the local
  `--annot`/`--channels` detector scripts. See
  [Re-run Detection on Reviewer-Selected Channels](../how-to/rerun-detection-on-channels.md).

Channel-level QC verdicts (kept / dropped / marked-artefact) and the
re-detect queue live in the same database, in tables the GUI manages
internally (`channel_qc`, `qc_artefact_intervals`).

## What Changed From the Pre-4.0 GUI

4.0.0 dropped the per-event Events tab and everything built around it: no
more per-event accept/reject, stratified sampling, Compare-methods view, or
confidence/method/frequency-band per-event filtering. See
[How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md#step-5-adjust-to-the-review-gui-workflow-change)
for the full account of what moved and why.

This page intentionally stays high-level: the GUI's internals (widget names,
table columns, exact dialogs) are still evolving. For task-oriented recipes,
see the [how-to guide](../how-to/review-eeg-events.md).

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md)
- [How-to Guide: Review EEG Events](../how-to/review-eeg-events.md)
- [Explanation: Review GUI Architecture](../explanation/eeg-review-gui-architecture.md)
- [How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md)
