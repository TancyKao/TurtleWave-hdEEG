# Reference: EEG Review GUI

The EEG Review GUI (`frontend.eeg_review_gui`) is a PyQt5/PyQtGraph application
for QC-triaging automatically detected EEG events (spindles, slow waves,
K-complexes, PAC) at the channel level, and, since 4.6, for recording
accept, reject or unsure decisions on individual events and scoring a review
sample.

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
- **2 · Epochs** — steps through the scored epochs for a single channel, with
  a hypnogram strip, outlier markers, and range-marking for artefacts. Epochs
  are the annotation file's own epochs: 30 s on a uniform grid, and on a cut
  recording whole-second epochs of 1 to 30 s. The window and the hypnogram
  strip follow each epoch's true length. Without an annotation file the tab
  uses a synthetic 30 s grid.
- **Filters dock** (left) — event type, detection method, frequency band, and
  channel selection, applied globally across both tabs.
- **Topography & detail dock** (right) — scalp topography for the current QC
  metric, the global worst-events list, the selected channel's detail, and the
  **Event** and **Decision** panels.

## Channel Marks and Defaults

- A channel the file names as interpolated (`header['interp_channels']`) is
  listed in the Filters dock with a trailing ` ~` and the tooltip
  "Interpolated channel (reconstructed from neighbours by the cleaning
  pipeline)". The channel name used for lookups is unchanged.
- A channel flagged as an artefact keeps its existing ⚑ tag.
- The waveform channels open as `E112`, `E118`, `Cz` when all three exist (EGI
  nets). Otherwise they are those of `Cz`, `Fz`, `Pz` that exist, topped up to
  three from the start of the list. Channels the file types as non-EEG are
  never chosen. A selection you made yourself is kept when every channel in it
  exists in the next file loaded.
- The dashed vertical gridlines in the trace are every 30 s, as a display
  guide. They are not epoch edges on a cut recording.

## Menu Bar

| Menu | Notable actions |
|------|------------------|
| File | Open Database…, Open EEG File…, Open Annotation File…, Exit |
| Edit | Flag selected channel for re-detect (`F`) |
| Review | Reviewer name…, Show other reviewers (checkable, off at every launch), and the review-sample entries (Draw review sample…, Resume review sample, Exit review sample, Precision report…) |
| View | Outlier threshold…, toggle Filters dock / Topography & detail dock |
| Analysis | Refresh QC dashboard, Build re-detect request… |
| Export | Export QC report…, Export Re-run Package…, Export Figure… |
| Help | Design notes, About |

## Population Checks (Channels tab)

Five columns per channel for the event type and run in view, computed from the
stored 4.6 figures with one aggregate query per refresh. Shares are whole
percentages; ratios have one decimal and a `×`.

| Column | Topography label | Definition | Event types |
|---|---|---|---|
| `off-band %` | off-band (% of events) | share of events with `in_band` false, among events with a measured peak | all |
| `low prom. %` | low prominence (% of events) | share with `low_prominence` true | spindles |
| `at floor %` | at the duration floor (% of events) | share within 0.05 s of the run's minimum duration | all |
| `amp/bg ×` | amp. vs background (median ×) | median `amp_ratio` | all |
| `amp/thr ×` | amp. vs threshold (median ×) | median `thresh_ratio` | methods that allow a ratio |
| `checks` | not a topography metric | `hard`, `soft` or empty | all |

Each of the five columns is flagged with a one-sided robust z across the
channels, using the same `hard` and `soft` limits as the amplitude flag. A
channel is flagged on a column only if it also differs from the montage median
by at least 10 percentage points (shares) or 0.3× (ratios). Channels with fewer
than 20 events with a value show `—` and are not flagged. The `checks` column
is separate from the amplitude `flag`; the **Outlier** filter gains
`checks: hard` and `checks: soft`.

On a run detected with 4.5 or earlier, the five values are `—`, the topography
items are disabled with the suffix ` — not recorded for this run`, and a line
under the topography says so. On a Lacourse2018, Ray2015, Wamsley2012,
Martin2013, CIRUS, Ngo2015 or Staresina2015 run, `amp/thr ×` is `—` with a
tooltip naming the reason.

In the right dock, the selected-channel block lists up to three flagged checks
in words, for example `PPOz: 34 % of spindles off-band`. Each phrase is a
link: it drills into the channel and shows a removable chip in the Epochs tab
title, such as `Showing off-band spindles only (117 of 344) ✕`.

## Event Panel

Shown in the right dock when an event is selected. Values come from the
`events` row; for a 4.5 run they read `not recorded for this run (detected with
4.5 or earlier)`. A missing value is `—` with a reason, never blank.

| Row | Shows |
|---|---|
| Time | clock time and seconds from recording start |
| Channel | name; `~PPOz` for an interpolated channel |
| Stage | `epoch_stage` |
| Detection | method, band, run date and short run id; tooltip has stages, reject types, reference and version |
| Duration | seconds, and position against the run's duration limits |
| Half-waves | half-waves standing out from background (2.5× bg RMS); spindles |
| Cycles (nominal) | sign changes ÷ 2 on the band-passed event; spindles |
| Peak frequency | 1/f-corrected peak with `in band` or `OFF BAND`, prominence, `low prominence`, `coarse` and the detector's own peak; spindles |
| Wave frequency | 1 / (2 × negative half-wave), with `in band` or `OFF BAND`; slow waves and K-complexes |
| Amp. vs background | ratio, band RMS against background RMS, number of windows, `near a stage change` |
| Amp. vs threshold | per method: a ratio, several thresholds, `no ratio`, or `Threshold not recorded for this run (detected with 4.5 or earlier)` |
| Trough, Peak-to-peak, Negative half-wave | slow waves and K-complexes; peak-to-peak only when `db_meta.det_ptp_units` is µV |
| Amplitude outlier | `yes` with the amplitude and limit, or `no` |

In sample mode a first row, `Sample`, shows the region, stage and flag words, for
example `parietal · NREM2 · flagged: off-band`, with the event's position in the
sample and a tooltip giving its sampling weight. It appears only after you have
accepted or rejected that event (not after unsure), so the stratum and flags
cannot steer the decision.

## Decision Controls

Buttons **Accept**, **Reject**, **Unsure**; a reason list; a comment field (500
characters); a `Current` line (`Not reviewed`, `Accepted by TK · 14:02`, and so
on); **Clear**; **Prev** and **Next** buttons; and a progress line. The
checkbox **Go to next unreviewed after deciding** is on by default and its
setting is kept.

Reasons, by key: `1` artefact, `2` arousal, `3` too short, `4` filter ringing,
`5` eye movement, `6` not visible in the raw trace, `7` outside the frequency
band, `8` only on this channel, `9` other (needs a comment). `not-isolated` and
`wrong-morphology` are available in the list without a digit. A reject needs a
reason; an unsure does not. See
[Decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md#reason-codes-and-their-ra-categories)
for the RA categories.

## Reviewer Name

The first decision of a session opens a prompt titled `Reviewer name`: *Your
name or initials. It is saved with every accept / reject decision, so two
reviewers' decisions on the same recording can be compared.* The name is
stripped, limited to 40 characters and remembered in the application settings.
Cancelling saves nothing: `Decision not saved — a reviewer name is needed.` The
status bar's first segment reads `Reviewer: TK` or `Reviewer: not set`. Changing
the name shows that reviewer's decisions and progress instead.

Bands, glyphs, the Current line and progress count only the current reviewer's
decisions. **Show other reviewers** reveals the rest and asks for confirmation
once per session.

## Keyboard Shortcuts

Active while the Epochs tab is current, including with focus in the right dock.
Suppressed while the comment field has focus.

| Key | Action |
|-----|--------|
| `F` | Flag the selected channel in the Channels (QC) table for re-detection |
| `P` / `N` | On the Epochs tab, jump to the previous / next outlier epoch |
| Prev / Next buttons | Step one epoch at a time |
| `A` | Accept the selected event |
| `R` / `U` | Arm reject / unsure, then choose a reason |
| `1`–`9` | Choose the reason while armed |
| `Enter` | Confirm an armed decision with the last reject reason; in the comment field, save the comment |
| `C` | Focus the comment field |
| `]` / `[` | Next / previous event not yet decided by you on this channel |
| `}` / `{` | Next / previous event of any status on this channel (only events passing a check filter, when one is on) |
| `Ctrl+Z` | Undo the last decision (at least 200 steps) |
| `Esc` | Cancel an armed decision; else leave the comment field; else clear the strip range; else remove the check filter; else clear the selection |

In sample mode `]` and `[` step through the undecided sample events in the
library's presentation order (balanced across groups, not by channel), drilling
into the channel as needed. `}` and `{` step through all events on the drilled
channel and keep the sample position, and `]` then returns to the sample. `[ ] { }` need AltGr on some keyboard layouts; the Prev and Next
buttons cover that.

Selecting an event: click its band on the raw trace, the filtered trace or the
ticker. A click selects the event containing the time, or the one whose nearest
edge is within 0.3 s.

## Neighbours and Physiology

**Neighbours** shows the selected event on the target channel and up to six
nearest EEG channels (by electrode position, else the same region) over 4 s
(spindles) or 6 s (slow waves, K-complexes), with their own detected events
shaded. **Physiology** shows EOG, chin EMG and ECG channels the file types as
such, filtered for display. Each group collapses, and the choice is remembered.

## Review Sample Bar

A one-row bar across the top of the Epochs tab.

| State | Text | Buttons |
|---|---|---|
| No sample | `REVIEW SAMPLE · No review sample for this run yet.` | Draw sample… |
| Sample, not active | events drawn, date, seed, number decided by you | Resume sample, Precision report… |
| Active | `37 of 120 in sample · 2 unsure`, plus the time left at your pace once there are enough decisions | Exit sample, Precision report…, Show other reviewers |
| Done | `120 of 120 in sample · 3 unsure · done` | as Active |
| Revisiting | `revisiting 3 unsure` | as Active |

`Draw sample…` opens a dialog with sample size (default 120), seed and a
preview of events per region and stage. Drawing again keeps the previous sample
and its decisions. Every reviewer who presses **Resume sample** gets the full
sample; the 30 shared events are a subset used for agreement.

## Precision Report

A non-modal dialog that reads the current sample's decisions and refreshes when
one is written in sample mode, or undone. A free-browsing decision or **Clear**
leaves an open report stale until it is reopened. It shows precision with a 95 % Wilson interval per region and
stage and for the whole night, a pooling rule with a threshold (default 0.80,
range 0.50 to 0.99) and an estimate choice (point estimate or lower 95 % bound),
a one-sentence verdict, the reasons for rejection with their FP-artifact and
FP-other counts, and agreement between two reviewers. Percent agreement is
three-way (accept, reject, unsure) over every event both reviewers decided;
Cohen's kappa leaves out events either reviewer marked unsure. A list of
disagreements can be opened. Groups with fewer than 10
decided events read `n too small` and are not judged. The report applies only
this pooling rule: the library's `TRUSTWORTHY`, `EXCLUDE` and `TOP_UP` verdicts
and `top_up_region` have no GUI yet. Opening it writes `review_precision` rows;
**Copy summary** and **Export CSV…** are in the footer. See
[Validate a detection run](../how-to/validate-a-detection-run.md).

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
internally (`channel_qc`, `qc_artefact_intervals`). Event decisions are written
to `event_reviews` at once, one row per event and reviewer, and sample data to
`review_sample_designs`, `review_samples` and `review_precision`. Decisions never
change `events`.

## What Changed From the Pre-4.0 GUI

4.0.0 dropped the per-event Events tab and everything built around it: no
more per-event accept/reject, stratified sampling, Compare-methods view, or
confidence/method/frequency-band per-event filtering. 4.6 brought event
decisions back in a different form: on the Epochs tab, and for validation
against a drawn sample rather than curation. See
[How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md#step-5-adjust-to-the-review-gui-workflow-change)
for the full account of what moved and why.

This page does not list widget names. For task-oriented recipes, see the
[how-to guide](../how-to/review-eeg-events.md).

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md)
- [How-to Guide: Review EEG Events](../how-to/review-eeg-events.md)
- [How to decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md)
- [How to validate a detection run](../how-to/validate-a-detection-run.md)
- [Explanation: Review GUI Architecture](../explanation/eeg-review-gui-architecture.md)
- [How to Upgrade to turtlewave-hdEEG 4.0](../how-to/upgrade-to-4.0.md)
