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
  current event type, with a Stage toggle, Show and Sort combos, population-check
  columns and a `Checks` flag beside the amplitude columns, and actions to mark a
  channel as an artefact, queue it for re-detection, drop it or open its epochs
  (see [Channels Tab](#channels-tab)).
- **2 · Epochs** — steps through the scored epochs for a single channel, with
  a hypnogram strip, outlier markers, and range-marking for artefacts. Epochs
  are the annotation file's own epochs: 30 s on a uniform grid, and on a cut
  recording whole-second epochs of 1 to 30 s. The window and the hypnogram
  strip follow each epoch's true length. Without an annotation file the tab
  uses a synthetic 30 s grid. Every event of the drilled channel that starts
  in the epoch is drawn as a band on the raw and filtered traces and can be
  selected. Below the filtered trace sit collapsible **Neighbours** and
  **Physiology** groups, and a `REVIEW SAMPLE` bar runs across the top.
- **Filters dock** (left) — event type, detection method, frequency band, and
  channel selection, applied globally across both tabs, then a REVIEW STATUS
  group that applies to the Epochs tab only.
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
| Help | Keyboard shortcuts…, Design notes, About |

## Channels Tab

Top to bottom: a control row, the channel table, a bottom action bar for the
selected channel and a footer line stating the flag rule. The right dock holds
the topography and, under it, the flagged-channel list.

**Control row.**

- **Event type** selects spindles, slow waves or K-complexes.
- **Stage** is a row of buttons, one per stage of the run (for example `NREM2`
  and `NREM3`) and a combined button (`NREM2 + NREM3`, the default). It
  changes only the check columns, the `Checks` flag, the check metrics on the
  topography and the flagged-channel list. Density and amplitude columns follow
  the Filters dock. The choice is remembered per event type.
- **Show** filters the rows: `All channels (n)`, `Flagged (n)` (a `Checks` flag
  or an amplitude flag of hard or soft), `Dropped (n)` and `Dead (n)`. The
  counts are over all channels.
- **Sort** orders the rows: `Checks (hard first)` (the default), `Off-band share
  ↓`, `At-floor share ↓`, `Amp / bg ↑`, `Amp / thr ↑`, `Low prominence share ↓`,
  `Amp z ↓`, `Channel` and `Region`. Clicking a column header also sorts; the
  combo then reads `Column header` unless the click matches an item.
- A **count line** at the right: `8 checks flagged · 3 amp flagged · 1 dead`,
  plus `· 1 dropped` when any channel is dropped. It counts channels over the
  whole montage, not the Show filter. On a run without stored figures it starts
  `checks not recorded`.

**Table columns**, in order:

| Header | Cell |
|---|---|
| `Channel`, `Region` | name; region |
| `Events` | count with thousands separator |
| `Density /min` | events per minute |
| `Med amp µV` | median amplitude |
| `Amp z` | signed robust z of the amplitude |
| `Amp flag` | `✓ OK`, `▲ SOFT`, `× HARD` or `DEAD` |
| `Checks` | `× HARD · off-band 34 %`, `▲ SOFT · …`, `—`, or `— dead channel` |
| `Off-band` | share of events whose peak lies outside the run band |
| `Low prom.` | share with `low_prominence`; context only, never tinted |
| `At floor` | share within 0.05 s of the run's minimum duration |
| `Amp / bg` | median event amplitude over background |
| `Amp / thr` | median detection peak over the detection threshold |
| `Status` | `kept` or `× dropped`, followed by ` · ↻ re-detect` when the channel is queued for re-detection |

There is no State column. A channel queued for re-detection reads `kept · ↻
re-detect` or `× dropped · ↻ re-detect` in the Status column. It also shows in
the **RE-DETECT QUEUE** chips of the selection tray, in the `re-detect queue: n`
count in the status bar and in the bottom-bar button, which reads `Remove from
re-detect queue` for a queued channel. A channel marked as an
artefact shows in the tray's **CHANNEL ARTEFACTS** chips and in the Status
column.

Hover tooltips: the `Checks` cell lists every flagged column with its value,
the montage median and the robust z. The `At floor` cell gives the floor share
and the ceiling share, or says the run has no upper limit. Shares are over
events for which the figure was computed, so a channel with fewer than 20 such
events shows `—` and is never flagged on that column.

**The flag rule, in words.** A channel is compared with the rest of the
montage, never with a fixed number. It is flagged when its off-band or at-floor
share is well above the montage median, or its median amp/bg or amp/thr is well
below it: hard when the robust z is above the hard limit (default 3.5), soft
above the soft limit (2.0). It also needs a difference from the montage median
of at least 10 percentage points (shares) or 0.3× (ratios) and at least 20
events. The limits are those of **View ▸ Outlier threshold…**, and the footer
line shows the current values. A problem every channel shares is not flagged;
see the Precision report.

**Low prominence is context only.** It stays as a column, a topography choice
and a montage median, but it never sets the `Checks` flag, never appears in the
`Checks` cell or the flagged-channel list, and its cells are not tinted. It
mostly follows a channel's signal-to-noise.

The `Checks` flag is separate from the amplitude `Amp flag`. **Queue all HARD**
acts on the amplitude flag only.

**Bottom action bar** (for the selected row): **Open in Epochs** (also turns on
the check filter for the channel's largest-z flagged column), **Drop channel**,
**Mark channel artefact**, **Add to re-detect queue** (`F`), then **Queue all
HARD** and **Build re-detect request…**. All are disabled with no row selected.

**Topography.** The combo offers event density, mean amplitude, maximum
peak-to-peak and the check metrics (off-band share, low-prominence share
(context), at-floor share, amp vs background, amp vs threshold). On a check
metric, each channel with a `Checks` flag has a ring (hard solid, soft dashed)
and up to 12 of them, those with the largest z, carry their name; a caption
under the colour bar defines the metric and says how many more are ringed.
Clicking a ringed electrode selects the channel.

**Flagged-channel list** (under the topography, facts only). One row per channel
with a `Checks` flag, hard first, for the chosen stage:

```text
× HARD  E75 · 34 % of spindles off-band · 61 % of those peak at 8–9 Hz
▲ SOFT  E62 · 22 % of spindles at the duration floor (0.5 s)
```

The off-band row adds the most common 1 Hz bin of the off-band peaks when the
channel has at least 10 off-band events (the library reports the bin from 5; the list needs 10). Its tooltip gives the share below the
band, the share above it and the bin. Other facts read `median amplitude 1.2×
background (montage median 2.4×)` and the same for threshold. The list never
interprets: it does not say what the off-band peaks are. Clicking a row selects
the channel and shows **Open in Epochs** and **Drop channel** on that row. A
dropped channel stays, greyed, with ` · dropped`. The **Selected channel** block
gives two lines of facts: name, region and event count, then `amp`, `checks` and
off-band share.

On a run detected with 4.5 or earlier the check columns read `—`, the check
topography items are disabled with the suffix ` — not recorded for this run`,
and the list says `Checks not recorded for this run (detected with 4.5 or
earlier).` On a Lacourse2018, Ray2015, Wamsley2012, Martin2013, CIRUS, Ngo2015
or Staresina2015 run, `Amp / thr` is `—` with a tooltip naming the reason.

**Hidden during live sample review.** While a review sample is active the right
dock shows no channel-level flags, so they cannot steer a decision. The Selected
channel block shows only channel, region and event count; the flagged-channel
list is replaced by `Channel checks are hidden while you review the sample.`
with the counts shown as `—`; the topography draws no rings or labels; and the
amplitude flag text is hidden. All return on **Exit sample**.

Clicking a flagged fact opens the Epochs tab on the channel with a removable
chip such as `Showing off-band spindles only (117 of 344) ✕`. Low prominence has
no filter chip.

## Event Panel

The header reads `EVENT i OF n IN EPOCH` (all events of the drilled channel and
type that start in the epoch), or `EVENT` with no selection. Values come from the
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

In sample mode a first row, `Sample`, shows the region and stage, with the
event's position in the sample and a tooltip giving its sampling weight.

**Flag words in live sample review.** For a sample event on which you have no
decision or an unsure decision, the panel hides every flag word and keeps every
number: `in band` / `OFF BAND`, `low prominence`, `at the floor of the run
limits`, `barely above background`, `barely crossed`, `meets` / `fails`, and the
`flagged: …` part of the Sample row (the Massimini and AASM value cell reads `2
criteria`). Nothing is coloured while hidden, and a line reads `Labels hidden
until you accept or reject this sample event.` The words appear straight after
your accept or reject is written, and are hidden again if it is undone or
cleared. Outside sample mode, and for events outside the sample, they always
show. The `outlier` row stays.

## Decision Controls

Buttons **Accept**, **Reject**, **Unsure**; a grid of reason buttons; a comment
field (500 characters); a `Current` line (`Not reviewed`, `Accepted by TK ·
14:02`, and so on); **Clear**; **Prev** and **Next** buttons; and a progress line.
The checkbox **Go to next unreviewed after deciding** is on by default and its
setting is kept.

The reason grid depends on the event type. `0` is always Other, which needs a
comment; `9` is unused.

| Key | Spindles | Slow waves and K-complexes |
|---|---|---|
| `1` | Artefact | Artefact |
| `2` | Eye movement | Eye movement |
| `3` | Not in raw | Not in raw |
| `4` | Filter ringing | Too short |
| `5` | Off-band | Arousal |
| `6` | Too short | Single channel |
| `7` | Arousal | Not isolated |
| `8` | Single channel | Wrong morphology |
| `0` | Other | Other |

A reject needs a reason; an unsure does not. Pressing `R` arms a reject and a
digit or a click on a grid button writes it. Clicking a grid button with nothing
armed arms Reject with that reason preselected (hint: `Click {label} again or
press Enter to reject ({label}).`); a second click on the same reason or `Enter`
writes, a click on a different reason switches the preselection, and `Esc`,
another event or paging cancels. Digits with nothing armed are ignored, so a stray key never writes. A stored reason that is not in the current
grid is still shown on the `Current` line. See
[Decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md#reason-codes-and-their-ra-categories)
for the RA categories.

## Reviewer Name

The first decision of a session opens a prompt titled `Reviewer name`: *Your
name or initials. It is saved with every accept / reject decision, so two
reviewers' decisions on the same recording can be compared.* The name is
stripped, limited to 40 characters and remembered in the application settings.
Cancelling saves nothing: `Decision not saved — a reviewer name is needed.`
Changing the name shows that reviewer's decisions and progress instead.

Bands, glyphs, the Current line and progress count only the current reviewer's
decisions. **Show other reviewers** reveals the rest and asks for confirmation
once per session.

## Left Dock: REVIEW STATUS

Under FREQUENCY BAND: checkboxes `unreviewed`, `reviewed` (a parent of the next
three), `accepted`, `rejected` and `unsure`, all checked at launch and not
remembered. The caption reads `Your decisions only. Applies to the Epochs tab.`
An event decided only by another reviewer counts as unreviewed. Events that do
not pass are drawn at half strength and skipped by `}` and `{`; clicking still
selects them. The filter does not affect `]` and `[`, sample navigation, epoch
counts, progress or the Channels tab.

## Keyboard Shortcuts

Active while the Epochs tab is current, including with focus in the right dock.
Suppressed while the comment field has focus. `?` works on both tabs.

| Key | Action |
|-----|--------|
| `F` | Add or remove the selected channel in the re-detect queue |
| `P` / `N` | On the Epochs tab, jump to the previous / next outlier epoch |
| `←` / `→`, Prev / Next buttons | Step one epoch at a time |
| `A` | Accept the selected event |
| `R` / `U` | Arm reject / unsure, then choose a reason |
| `1`–`8`, `0` | Choose the reason from the event type's grid while armed; `0` is Other |
| `Enter` | Confirm an armed decision with the last reject reason; in the comment field, save the comment |
| `C` | Focus the comment field |
| `]` / `[` | Next / previous event not yet decided by you on this channel |
| `}` / `{` | Next / previous event of any status on this channel (only events passing a check or REVIEW STATUS filter, when one is on) |
| `Ctrl+Z` | Undo the last decision (at least 200 steps) |
| `Esc` | Cancel an armed decision; else leave the comment field; else clear the strip range; else remove the check filter; else clear the selection |
| `?` | Open or close the keys cheat sheet |

In sample mode `]` and `[` step through the undecided sample events in the
library's presentation order (balanced across groups, not by channel), drilling
into the channel as needed. `}` and `{` step through all events on the drilled
channel and keep the sample position, and `]` then returns to the sample.
`[ ] { }` need AltGr on some keyboard layouts; the Prev and Next buttons cover
that.

**Key hints** run along the top bar. Epochs tab: `A accept · R reject · U unsure
· ] [ unreviewed · N P outlier · ? keys` (`] [ sample` in sample mode). Channels
tab: `F re-detect queue · ? keys`.

**Cheat sheet.** `?` or **Help ▸ Keyboard shortcuts…** opens a frameless sheet
listing the decide keys, the reason grid of the drilled event type (spindles
when no channel is drilled), the move keys and the channel keys. `?`, `Esc` or a
click outside closes it. `Ctrl+Z` is shown as `⌘Z` on macOS.

Selecting an event: click its band on the raw trace, the filtered trace or the
ticker. A click selects the event containing the time, or the one whose nearest
edge is within 0.3 s.

## Status Bar

Left to right: `Reviewer: TK` (or `not set`); the position (`PPOz · epoch 3 / 40`
on the Epochs tab, or the event type, method, band, channel count and check
counts on the Channels tab); the save line `Decisions save to neural_events.db
as you make them.` (`Set a reviewer name to save decisions.` without a name, and
a message when the library has no review store); and the re-detect queue count.
Decision confirmations appear as temporary messages.

Under the epoch strip a legend reads `grey bars = events per epoch · red =
amplitude outliers · purple dashes = marked artefact · white line = current
epoch`, with ` · blue ticks = sample events` in sample mode.

## Neighbours and Physiology

**Neighbours** shows the selected event on the target channel and up to six
nearest EEG channels (by electrode position, else the same region) over 4 s
(spindles) or 6 s (slow waves, K-complexes), with their own detected events
shaded. Rows are labelled by rank, not distance: `E75 · target`, then `E19 · 1`,
`E23 · 2` with 1 the nearest, and `~E19 · 1` for an interpolated channel. The
header says `(1 = nearest)`. The region and selected-channel fallbacks are not
ranked, so their rows carry the name only. No distance is shown, because EEGLAB
coordinates carry no reliable units.

**Physiology** shows EOG, chin EMG and ECG channels the file types as such,
filtered for display. The selected event is marked by two thin vertical lines
(its start and end) on each row. No box, fill or text covers the signal. Each
group collapses, and the choice is remembered.

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
