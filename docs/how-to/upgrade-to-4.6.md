# How to Upgrade to turtlewave-hdEEG 4.6

4.6 adds per-event review figures to every detection run, stores the
thresholds each run used, records reviewer decisions, and lets you draw and
score a review sample. All schema changes are additive. This guide covers what
changes for scripts and databases.

## Before you start

```bash
cp neural_events.db neural_events.db.bak-pre-4.6
```

Nothing is dropped or retyped, but back up before running new code against a
database you care about.

## What changed

- **`cat` defaults to `(1, 1, 1, 0)`** in `ParalEvents.detect_spindles`,
  `ParalSWA.detect_slow_waves` and `ParalKC.detect_kcomplexes`. It was `None`,
  which Wonambi's `fetch` rejects, so every channel failed. `cat=None` now means
  the default. Anything other than a four-flag tuple or list of 0 and 1 raises
  `ValueError` before detection starts. A script that already passed
  `cat=(1, 1, 1, 0)` is unchanged.
- **Six detector columns on `events`:** `peak_freq`, `peak_val_det`, `rms_det`,
  `rms_orig`, `power_orig`, `det_zero_time`. `peak_val_det` and `rms_det` are in
  the units of the method's detection signal, not always microvolts.
- **Fourteen figure columns on `events`:** `halfwaves_above_bg`,
  `cycles_nominal`, `peak_freq_ap`, `prominence_db`, `in_band`,
  `low_prominence`, `bg_rms`, `bg_n_windows`, `bg_stage_mixed`, `amp_ratio`,
  `thresh_ratio`, `near_bound`, `near_splice`, `wave_freq`. They are filled at
  detection time on the database path. Pass `compute_figures=False` to leave
  them NULL. The settings used are in `detection_runs.params_json['event_figures']`.
  See [Event figures and review sampling](../explanation/event-figures-and-review-sampling.md).
- **New table `detection_thresholds`:** the thresholds each run resolved, per
  channel and segment. Read it with `dbwrite.read_detection_thresholds`.
- **New table `event_reviews`:** one row per event and reviewer (`uuid`,
  `reviewer`, decision, reason, comment, `reviewed_at`, and a copy of the
  event's identity). It sits outside `events`, so a re-detection cannot erase it.
  A re-detection with the same parameters gives the same uuid, so a review stays
  attached.
- **New tables `review_sample_designs`, `review_samples`, `review_precision`:**
  the drawn samples, their events and weights, and the derived precision rows.
- **New views `events_reviewed` and `v_event_density_reviewed`:** `events` (and
  density) restricted to events no reviewer rejected, with review counts.
- **`exclude_rejected` and `reviewer` arguments** on `event_density` and
  `export_events_to_csv`. Both default to off, so existing results do not change.
- **New CSV columns** on `export_events_to_csv`, after `det_peak_time (s)`:
  `peak_freq (Hz)`, `peak_val_det`, `rms_det`, `rms_orig (uV)`, `power_orig`,
  `det_zero_time (s)`, then `halfwaves_above_bg`, `cycles_nominal`,
  `peak_freq_ap (Hz)`, `prominence_db (dB)`, `in_band`, `low_prominence`,
  `bg_rms (uV)`, `bg_n_windows`, `bg_stage_mixed`, `amp_ratio`, `thresh_ratio`,
  `near_bound`, `near_splice`, `wave_freq (Hz)`. Booleans are 0 or 1, and
  `near_bound` is -1, 0 or +1. The CSV importer ignores them. A script that
  reads columns by position, rather than by name, must be updated.
- **Review GUI:** population checks on the Channels tab, an Event panel,
  decision keys, neighbouring channels and a physiology strip on the Epochs tab,
  and a review sample with a precision report. See
  [Reference: EEG Review GUI](../reference/eeg-review-gui.md). Changes to the
  controls you may already use:
    - **Exclude channel** is one toggle that replaces *Drop channel* and *Mark
      channel artefact*. It writes the same stored verdict, so earlier choices
      carry over. An excluded channel is left out of review samples, the re-run
      export, the flag statistics and the topography; event density and exported
      events are unchanged in 4.6.0.
    - The *Selection* area and *Build re-detect request…* are gone, and nothing
      writes `redetect_request.json` any more. The re-detect queue is saved in the
      database (the `channel_qc` table gains a `redetect` column), shows in the
      Status column and the Show filter, and **File ▸ Export re-run package…**
      writes `redetect_channels.csv` for `examples/rerun_detection.py --channels`.
    - *Mark as artefact (writes XML)* is now **Exclude time range…**, and *Mark N
      epochs as artefact* is **Exclude N epochs…**. The reject reason *Artefact*
      keeps its name and labels one event only.
    - The Channels table says *Mean amp µV* (it was *Med amp µV*), and an amp flag
      names the measure that triggered it. Regions come from electrode names for
      10-20 and 10-5 labels.
    - The Event panel shows four rows (Signal vs background, Duration, Peak freq
      or Wave freq, Amplitude outlier); the rest moved to tooltips.
    - While a review sample is active the Channels tab shows a banner and hides
      the check columns.

## What you need to do

1. If you pass `cat=None` anywhere, nothing breaks, but the default is now
   explicit. Delete the argument or pass the tuple.
2. If you read the CSV export by column position, switch to column names.
3. To get figures and thresholds for an old run, re-detect with 4.6. There is no
   backfill in the package. Until then the figures are NULL and the GUI says
   "not recorded for this run (detected with 4.5 or earlier)".
4. Do not mix runs from different versions in one precision sample: the sample
   refuses to draw a scope whose runs differ in parameters unless you pass
   `allow_mixed_params=True`.

## Database changes happen on first use

`ensure_direct_write_schema` adds the `events` columns, `detection_thresholds`
and `event_reviews` when a 4.6 detection run opens the database.
`ensure_event_reviews_schema` and `ensure_review_sampling_schema` create the
review tables on their own write connections, so the review GUI can add them
without claiming to have written events. The three sampling tables are created
by `ensure_review_sampling_schema` the first time a sample is drawn.

If an `event_reviews` table written by a pre-release build has an older reason
vocabulary, it is rebuilt in one transaction: `arousal-alpha` becomes
`arousal`, and any other unknown reason or decision is kept as text in
`comment`. Every row is kept.

## Quiet two harmless Wonambi warnings

The pinned Wonambi 7.15 prints two `DeprecationWarning`s that no fix on your side
can remove:

- a notice that the `fooof` package is being deprecated, printed when Wonambi
  imports it (TurtleWave never uses `fooof`);
- NumPy's "Conversion of an array with ndim > 0 to a scalar is deprecated",
  raised from `wonambi/trans/analyze.py` once per call site.

The GUIs and the example and Gadi driver scripts now set the environment
variable `TURTLEWAVE_QUIET_WONAMBI=1` before they import the package, and
`import turtlewave_hdEEG` then calls `turtlewave_hdEEG.utils.quiet_wonambi_warnings()`
before it imports Wonambi. Nothing else is silenced. The library never does this
on its own: without the variable, your warnings stay visible.

To see the warnings again, set the variable to `0` before starting:

```bash
TURTLEWAVE_QUIET_WONAMBI=0 turtlewave_gui
```

In your own script, set it before the first import:

```python
import os
os.environ.setdefault('TURTLEWAVE_QUIET_WONAMBI', '1')
from turtlewave_hdEEG import ParalEvents
```

The order matters for the `fooof` notice: `fooof` switches Python's warning
filter to "always" just before it warns, so no filter set earlier can stop it.
The function imports `fooof` itself inside a quiet block, which only works before
Wonambi is first imported. Called later, it still silences the NumPy warning but
cannot take back a notice already printed. See
[Utilities API Reference](../reference/api/utils.md).

## See also

- [Validate a detection run](validate-a-detection-run.md)
- [Decide whether an event is genuine](decide-if-an-event-is-genuine.md)
- [dbwrite reference](../reference/api/dbwrite.md)
