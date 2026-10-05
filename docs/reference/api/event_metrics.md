# Event Metrics API Reference

`turtlewave_hdEEG.event_metrics` computes the per-event review figures that
every 4.6 detection run stores on `events`: half-waves standing out from
background, nominal cycles, the 1/f-corrected peak frequency and prominence,
amplitude against background and against the detector's threshold, slow-wave
half-wave frequency, and the duration-bound flag. It uses only NumPy and SciPy
and imports no Qt, so the detectors, the review GUI and any backfill call the
same code.

The detectors call `apply_channel_figures` once per channel, after every method
has run, on the detector's own segment. To compute the figures for one event
from a single-channel read, call `event_figures`. The signal must be in the
run's reference (`ref_chan` applied), or the figures are not comparable.

See [Event figures and review sampling](../../explanation/event-figures-and-review-sampling.md)
for the definitions and their limits, and
[Decide whether an event is genuine](../../how-to/decide-if-an-event-is-genuine.md)
for how a reviewer uses them.

The stored columns are listed in
[Upgrade to 4.6](../../how-to/upgrade-to-4.6.md#what-changed). A figure that
cannot be computed is `None` (NULL in the database). `near_splice` events have
NULL signal figures by design.

::: turtlewave_hdEEG.event_metrics
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
