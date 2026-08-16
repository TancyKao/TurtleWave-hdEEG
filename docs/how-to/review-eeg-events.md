# How-to Guide: Review EEG Events

This guide provides practical solutions for specific QC-triage tasks in the TurtleWave EEG Review GUI. It assumes you've already loaded a database, EEG file, and (optionally) annotations — see the [tutorial](../tutorials/eeg-review-gui-tutorial.md) if you haven't.

## Prerequisites

The review GUI reads events out of `neural_events.db` — it doesn't detect
anything itself. Before you can review, events must already be detected into
that database:

- [Detect Spindles](detect-spindles.md) (or slow waves / K-complexes /
  PAC — any detector works)
- [Write Detection Results Directly to the Database](direct-to-database-detection.md)
  — the `write_db=True` path, or the legacy JSON → CSV →
  `import_parameters_csv_to_database` route documented on each detector's
  how-to page

## Triage Channels by QC Verdict

**Problem:** You want to see only the channels flagged as outliers before reviewing the rest.

**Solution:**

1. On the **1 · Channels (QC)** tab, use the **Outlier:** dropdown next to the event type selector
2. Choose `hard`, `soft`, `dead`, or `ok` to filter the table, or `any` to see every flagged channel
3. Click a column header to sort (e.g. by density or max peak-to-peak amplitude)

!!! tip
    The count strip above the table (HARD / SOFT / DEAD / OK) always reflects the full, unfiltered channel count for the current event type.

## Filter by Event Type, Method, or Frequency Band

**Problem:** You need to focus on one detector's output at a time.

**Solution:**

1. In the left **Filters** dock, check only the event type(s) you want (Slow wave, Spindle, K-complex, PAC)
2. Use the **Method** dropdown to narrow to a single detection method
3. Use the **Frequency Band** dropdown to narrow to a single band

These filters apply globally, across both the Channels (QC) and Epochs tabs.

## Adjust Outlier Sensitivity

**Problem:** The default outlier thresholds are flagging too many, or too few, channels.

**Solution:**

1. **View → Outlier threshold…**
2. Adjust `hard |z| >`, `soft |z| >`, and the dead-channel fraction
3. Click **OK** — the QC dashboard recomputes immediately

## Restrict to Specific Channels

**Problem:** You only care about a subset of channels (e.g. frontal).

**Solution:**

1. In the left **Filters** dock, type into the channel search box (e.g. `E33` or `Cz`) to narrow the list
2. Check the channels you want, or use **All** / **None** to bulk-select

## Drill into a Channel's Epochs

**Problem:** A channel is flagged and you want to see exactly which windows are driving it.

**Solution:**

1. Select the channel's row in the Channels (QC) table
2. Click **Drill into epochs ▸**
3. On the **2 · Epochs** tab, use **P** / **N** to jump between outlier epochs, or the prev/next buttons to step one epoch at a time

!!! tip
    Click a point in the global worst-events list (right dock) to jump straight to that channel and epoch without going through the table.

## Mark a Channel as an Artefact

**Problem:** A channel is unusable for the current event type and should be excluded going forward.

**Solution:**

1. Select the channel on the **1 · Channels (QC)** tab
2. Click **Mark channel artefact**

The button toggles — click **Unmark channel artefact** to reverse it. Marked channels are excluded from exports and get a ⚑ tag in the channel filter list.

## Mark a Time Range as Artefact

**Problem:** Only part of a channel's recording is bad, not the whole channel.

**Solution:**

1. On the **2 · Epochs** tab, shift-drag across the overview strip to select a range
2. Click **Mark N epochs as artefact** to confirm

## Flag Channels for Re-detection

**Problem:** A channel's detections look wrong and you want to re-run detection on it with different parameters.

**Solution:**

- Select the channel and press **F**, or click **Add to re-detect queue**
- To queue every currently HARD-flagged channel at once, click **Queue all HARD**

The re-detect queue count is shown in the status bar and in the **RE-DETECT QUEUE** section of the selection tray, where you can remove channels with the chip's ✕.

## Hand Off Flagged Channels to a Re-run

**Problem:** You've flagged channels and want to hand them off to a re-run.

**Solution:** there are two hand-off formats, depending on which detector script picks them up:

- **Analysis → Build re-detect request…** (or the **Build re-detect request…** button on the Channels tab) previews and saves `redetect_request.json` next to the annotation XML, for `turtlewave_gui` to pick up. This GUI never runs detection itself.
- **Export → Export Re-run Package…** snapshots the current database/CSVs, then writes a `channels.csv` and a sidecar annotation XML (with any marked artefact ranges folded in) for the local `--annot`/`--channels` detector scripts (e.g. `examples/hdEEG_spindle_detector.py`).

See [Re-run Detection on Reviewer-Selected Channels](rerun-detection-on-channels.md) for the full hand-off flow.

## Export a QC Report

**Problem:** You need a record of the QC pass to attach to a study log.

**Solution:**

1. **Export → Export QC report…**
2. Choose a location and filename

This writes a Markdown summary — channel count, dropped channels, global artefact windows, and the full flagged-channel table — for the current event type.

## Troubleshooting

### Channels Not Loading

**Problem:** The Channels (QC) table is empty after loading the database.

**Solution:**

1. Check that your database file is not corrupted (try opening it with a SQLite browser)
2. Verify the database contains events for the selected event type: `SELECT COUNT(*) FROM events WHERE event_type = 'spindle'`
3. Check the left Filters dock — an event type may be unchecked, or the channel list may be filtered to nothing

### Epochs Not Displaying

**Problem:** The Epochs tab is blank even though a channel is selected.

**Solution:**

1. Verify you loaded the correct EEG file (**File → Open EEG File…**)
2. Check that the EEG file path matches the one used during event detection
3. Ensure the EEG file format is supported (`.set`, `.fdt`, `.edf`)
4. Check the console for error messages

### GUI Running Slowly

**Problem:** The GUI is laggy when navigating between epochs.

**Solution:**

1. Narrow the channel filter to the channels you're actively reviewing
2. Filter to a single event type and method
3. Close other applications to free up memory

### `F` Doesn't Flag a Channel

**Problem:** Pressing **F** doesn't add the selected channel to the re-detect queue.

**Solution:**

1. Click on a row in the Channels (QC) table first — `F` acts on the current selection, and there's no selection until you click a row
2. Confirm a database is loaded (`Ready` should not still be showing in the status bar)

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md) - Learn the basics
- [Reference: EEG Review GUI](../reference/eeg-review-gui.md) - Technical specifications
- [Explanation: Review GUI Architecture](../explanation/eeg-review-gui-architecture.md) - Understand how it works
- [How to Upgrade to turtlewave-hdEEG 4.0](upgrade-to-4.0.md#step-5-adjust-to-the-review-gui-workflow-change) - What changed from the pre-4.0 per-event review workflow
