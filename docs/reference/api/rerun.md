# Re-run Guards API Reference

The `rerun` module holds the correctness guards for scoped channel
re-detection (re-running a detector on only the channels a reviewer flagged in
the review GUI, replacing just their rows in `neural_events.db`). None of these
functions touch the signal path; they decide whether a channel is re-detected
at all, and with which invariant parameters.

See [Re-run detection on reviewer-selected channels](../../how-to/rerun-detection-on-channels.md)
for the end-to-end hand-off from the review GUI.

::: turtlewave_hdEEG.rerun
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
