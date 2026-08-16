# Tutorial: Your First EEG Event Review Session

Welcome! In this tutorial, we'll walk through your first QC pass with the TurtleWave EEG Review GUI. By the end, you'll have triaged real channels and know how to flag one for re-detection.

!!! note "What you'll learn"
    - How to launch the EEG Review GUI
    - How to load your data files
    - How to read the Channels (QC) dashboard
    - How to drill into a channel's epochs
    - How to flag a channel for re-detection and export a QC report

!!! tip "Before you start"
    Make sure you have:

    - TurtleWave installed (`pip install turtlewave-hdEEG`)
    - An event database file (`neural_events.db` from event detection)
    - The corresponding EEG data file (`.set` or `.fdt` format)
    - Optional: Sleep stage annotation file (`.xml` format)

## Step 1: Launch the GUI

Open your terminal and run:

```bash
eeg_review_gui
```

The application window opens with two tabs — **1 · Channels (QC)** and
**2 · Epochs** — plus a left filter dock and a right dock carrying
topography, the global worst-events list, and channel detail.

!!! success "What you should see"
    A window titled "TurtleWave hdEEG · Event Review". The interface is
    empty because we haven't loaded any data yet.

## Step 2: Load Your Data

1. **File → Open Database…** and select your `neural_events.db`
2. **File → Open EEG File…** and select the matching `.set`/`.fdt` file
3. **File → Open Annotation File…** (optional) to load sleep stages

The toolbar LEDs (DB / XML / EEG) light up as each source loads, and the
Channels (QC) tab populates with one row per channel.

!!! success "What you should see"
    The Channels (QC) table fills with rows, each showing an outlier flag,
    event count, density, and amplitude for the current event type.

## Step 3: Read the QC Dashboard

The **Channels (QC)** tab is the landing surface. Each row is a channel, not
an individual event — this is a QC triage view, not a per-event review list.

Use the left filter dock to switch event type (spindle / slow wave /
K-complex / PAC) and to narrow by method or frequency band. Channels flagged
as outliers (hard or soft, based on a robust z-score against the rest of the
montage) sort to the top.

!!! tip "What to observe"
    Click a row to populate the right-hand detail dock with that channel's
    topography position and worst events. Click a row in the global
    worst-events list to jump straight to a channel.

## Step 4: Drill into a Channel's Epochs

1. Select a channel in the QC table
2. Click **Drill into epochs ▸** (this switches you to the **2 · Epochs** tab)

The Epochs panel steps through 30-second windows for that channel, with a
hypnogram strip and outlier markers. Use **P**/**N** to jump between outlier
epochs, or the prev/next buttons to step one epoch at a time.

!!! success "What you should see"
    The epoch strip highlights outlier windows. Stepping through lets you
    confirm whether a flagged channel's events look like real detections or
    artefact.

## Step 5: Flag a Channel for Re-detection

Once you've decided a channel needs re-running with different parameters
(or dropped from analysis):

1. Select the channel in the Channels (QC) table
2. Press **F** (or use **Edit → Flag selected channel for re-detect**)

The channel is added to the re-detect queue shown in the status bar. Repeat
for as many channels as needed, then use **Analysis → Build re-detect
request…** to hand them off to a re-run.

## Step 6: Export a QC Report

When you're done triaging:

1. **Export → Export QC report…**
2. Choose a location and filename

This writes a Markdown summary of the per-channel QC table, flagged
channels, and marked artefact ranges for the current event type.

!!! success "What you've created"
    A Markdown report you can attach to a study log or share with a
    collaborator, plus a re-detect queue ready to build into a scoped rerun.

## What You've Accomplished

Congratulations! You now know how to:

✅ Launch the GUI and load your data files
✅ Read the Channels (QC) dashboard and spot outlier channels
✅ Drill into a channel's epochs to inspect individual windows
✅ Flag a channel for re-detection with `F`
✅ Export a QC report

## Next Steps

- **Solve specific QC tasks** — see the [How-to Guide](../how-to/review-eeg-events.md)
- **Understand the design** — see the [Explanation](../explanation/eeg-review-gui-architecture.md)
- **Upgrading from a pre-4.0 project** — the workflow above replaced the old
  per-event review GUI; see
  [How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md#step-5-adjust-to-the-review-gui-workflow-change)
  for what changed and why.
