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
  current event type, with a Stage toggle, Show and Sort combos, an eight-column
  amplitude table, and actions to exclude a channel, queue it for re-detection
  or open its epochs. The channel checks are in the right dock
  (see [Channels Tab](#channels-tab)).
- **2 · Epochs** — steps through the scored epochs for a single channel, with
  a hypnogram strip, outlier markers, and time-range exclusion. Epochs
  are the annotation file's own epochs: 30 s on a uniform grid, and on a cut
  recording whole-second epochs of 1 to 30 s. The window and the hypnogram
  strip follow each epoch's true length. Without an annotation file the tab
  uses a synthetic 30 s grid. Every event of the drilled channel that starts
  in the epoch is drawn as a band on the raw and filtered traces and can be
  selected. Below the filtered trace sit collapsible **Neighbours** and
  **Physiology** groups, and a `REVIEW SAMPLE` bar runs across the top.
- **Filters dock** (left) — event type, detection method, frequency band, and
  channel selection, applied globally across both tabs. A caption `~ =
  interpolated` sits under the channel list.
- **Topography & detail dock** (right) — scalp topography for the current QC
  metric, the global worst-events list, the selected channel's detail, and the
  **Event** and **Decision** panels.

## Channel Marks and Defaults

- A channel the file names as interpolated (`header['interp_channels']`) is
  listed in the Filters dock with a trailing ` ~` and the tooltip
  "Interpolated channel (reconstructed from neighbours by the cleaning
  pipeline)". The channel name used for lookups is unchanged.
- An excluded channel shows `× excluded` in the Status column.
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
| File | Open Database…, Open EEG File…, Open Annotation File…, Export re-run package…, Exit |
| Edit | Flag selected channel for re-detect (`F`) |
| Review | Reviewer name…, Show other reviewers (checkable, off at every launch), the review-sample entries (Draw review sample…, Resume review sample, Exit review sample), Precision report… and Precision rule… |
| View | Outlier threshold…, toggle Filters dock / Topography & detail dock |
| Analysis | Refresh QC dashboard |
| Export | Export QC report…, Export Re-run Package…, Export Figure… |
| Help | Keyboard shortcuts…, What the event figures mean, Design notes, About |

## Channels Tab

Top to bottom: a banner (only while a review sample is active), a control row,
the channel table and a bottom action bar for the selected channel. The right dock holds the topography and, under it, the flagged-channel
list. The top bar shows `detector: Moelle2011 · 11–16 Hz` for the run in view,
or `detector: —` when several runs match the Filters dock.

**Banner during sample review.** While a review sample is active, a banner reads
`Review sample in progress: channel checks are hidden. Exit sample to see them.`
with an **Exit sample** button. The channel checks (the flagged-channel list,
the check metrics and rings on the topography) are hidden, the Stage buttons
are disabled, and the count line starts `checks hidden` and is not a link. All
return on Exit sample.

**Control row.**

- **Event type** selects spindles, slow waves or K-complexes.
- **Stage** is a row of buttons, one per stage of the run (for example `NREM2`
  and `NREM3`) and a combined button (`NREM2 + NREM3`, the default). It changes
  only the check metrics on the topography and the flagged-channel list. The
  table's density and amplitude columns follow the Filters dock. The choice is
  remembered per event type.
- **Show** filters the rows: `All channels (n)`, `Amp flagged (n)` (an amplitude
  flag of hard or soft), `Excluded (n)`, `Dead (n)` and `Queued for re-detect
  (n)`. The counts are over all channels.
- **Sort** orders the rows: `Amp flag (hard first)` (the default), `Amp z ↓`,
  `Mean amp ↓`, `Density ↓`, `Events ↓`, `Channel` and `Region`. Clicking a
  column header also sorts; the combo then reads `Column header` unless the click
  matches an item. A sort setting saved on a column that no longer exists opens
  as `Amp flag (hard first)`.
- A **count line** at the right: `8 checks flagged · 3 amp flagged · 1 dead`,
  plus `· 1 excluded` when any channel is excluded. It counts channels over the
  whole montage, not the Show filter. The `8 checks flagged` part is a link: it
  scrolls the right dock to **CHECKS — FLAGGED CHANNELS** and flashes its header.
  On a run without stored figures it starts `checks not recorded`.

**Table columns**, eight in order. The **Channel** column is pinned: it stays at
the left edge while the table scrolls sideways.

| Header | Cell |
|---|---|
| `Channel`, `Region` | name; region |
| `Events` | count with thousands separator |
| `Density /min` | events per minute |
| `Mean amp µV` | mean amplitude of the channel's events (detection-band signal) |
| `Amp z` | signed robust z of the amplitude measure furthest from the other channels (mean, 95th percentile or largest event); hover for all three |
| `Amp flag` | `✓ OK`, `▲ SOFT · mean`, `× HARD · largest event`, `· 95th pct`, or `DEAD`; the text after the dot names the measure that triggered it |
| `Status` | `kept` or `× excluded`, followed by ` · ↻ re-detect` when the channel is queued for re-detection |

There is no State column and no Selection area. A channel queued for
re-detection reads `kept · ↻ re-detect` or `× excluded · ↻ re-detect` in the
Status column, and the queue is saved in the database (the `channel_qc` table),
so it is still there when you reopen the GUI. The status bar shows the
`re-detect queue: n` count. An excluded channel's own `Amp flag` cell reads `—`:
an excluded channel is not judged. A `~` before a channel name in the left dock
means interpolated; a caption `~ = interpolated` under the channel list explains
it (shown only when a listed channel is interpolated).

The off-band, low-prominence, at-floor, amp/bg and amp/thr shares and ratios are
no longer table columns. They live in the right dock's flagged-channel list and
as metrics on the topography (below).

**Regions** come from the electrode name for 10-20 and 10-5 labels (`Fz`, `F1h`
and `F2h` are Frontal), and from coordinates only for EGI `E<n>` labels, or when
the name gives no region. The table, the topography and the review sample use
the same region.

**Where the rules are.** There is no footer line. Hover the `Amp flag` header for
the amplitude rule: a channel is compared with the other channels in view
(excluded channels are left out); `× hard` / `▲ soft` mean its mean event
amplitude, its 95th percentile or its largest event is far from the montage
median (robust z above the hard / soft limit, default 3.5 / 2), and the cell
names which one; `DEAD` means fewer than 15 % of the median event count. Hover
the **CHECKS — FLAGGED CHANNELS** header for the checks rule: a channel is listed
when its off-band or at-floor share is well above the montage median, or its
median signal vs background or amp vs threshold is well below it (the same
limits, and only if the difference is at least 10 percentage points for shares
or 0.3× for ratios and the channel has at least 20 events). Change the limits in
**View ▸ Outlier threshold…**. A problem every channel shares is not flagged; see
the Precision report. A channel with fewer than 20 events for a figure is never
flagged on it. `Amp z` shows the z of the measure that triggered the flag.

**Low prominence is context only.** It stays as a topography choice, but it
never sets a channel check and never appears in the flagged-channel list. It
mostly follows a channel's signal-to-noise.

The channel checks are separate from the amplitude `Amp flag`. **Queue all HARD**
acts on the amplitude flag only.

**Bottom action bar** (for the selected row): **Open in Epochs** (also turns on
the check filter for the channel's largest-z flagged column), **Exclude channel**
(**Include channel** once excluded), **Add to re-detect queue** (**Remove
from re-detect queue** once queued; the `F` key does the same, and its tooltip
ends `(F).`), then, when something is queued, a link
`n queued · Export re-run package…`, and **Queue all HARD (n)**. The buttons are
disabled with no row selected.

**Exclude channel** is one toggle. In 4.6.0 an excluded channel is left out of
review samples, of the re-run export (`redetect_channels.csv`), of the flag
statistics (the montage median and spread that channels are compared with) and
of the topography, where it is a hollow marker not used for the map. Event
density and exported events are unchanged, so excluding a channel does not
remove its events from `event_density` or from a CSV export. It is stored in
`channel_qc` and one click reverses it. **Exclude time range…** on the Epochs
tab is a different action; see below.

**Re-running queued channels.** The queue is not a request file. **File ▸ Export
re-run package…** (or the link in the bar) writes `redetect_channels.csv`, the
file `examples/rerun_detection.py --channels` reads. See
[Re-run Detection on Reviewer-Selected Channels](../how-to/rerun-detection-on-channels.md).

**Topography.** The combo offers event density, mean amplitude, maximum
peak-to-peak and the check metrics (off-band share, low-prominence share
(context), at-floor share, amp vs background, amp vs threshold). On a check
metric, each channel with a channel check has a ring (hard solid, soft dashed)
and up to 12 of them, those with the largest z, carry their name; excluded
channels are hollow, with a legend line `○ excluded channel (not used for the
map)`; a caption
under the colour bar defines the metric and says how many more are ringed.
Clicking a ringed electrode selects the channel.

**Flagged-channel list** (under the topography, facts only). One row per channel
with a channel check, hard first, for the chosen stage. Hover its header for the
checks rule:

```text
× HARD  E75 · 34 % of spindles off-band · 61 % of those peak at 8–9 Hz
▲ SOFT  E62 · 22 % of spindles at the duration floor (0.5 s)
```

The off-band row adds the most common 1 Hz bin of the off-band peaks when the
channel has at least 10 off-band events (the library reports the bin from 5; the list needs 10). Its tooltip gives the share below the
band, the share above it and the bin. Other facts read `median amplitude 1.2×
background (montage median 2.4×)` and the same for threshold. The list never
interprets: it does not say what the off-band peaks are. Clicking a row selects
the channel and shows **Open in Epochs** and **Exclude channel** on that row. An
excluded channel stays, greyed, with ` · excluded`. The **Selected channel** block
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
type that start in the epoch), or `EVENT` with no selection, with a **What do
these mean?** link at its right (also **Help ▸ What the event figures mean**,
which opens a short built-in explanation that needs no network). Under it, one
line: `01:16:37.5 · PPOz · NREM2 · Moelle2011 11–16 Hz` (the channel is `~PPOz`
when interpolated). Its tooltip gives the times in seconds, the run, stages,
excluded event types, reference and version.

Then four rows. Everything else is in tooltips.

| Row | Shows | Tooltip adds |
|---|---|---|
| `Signal vs background` | how many times bigger the event is than the surrounding signal in the detection band, for example `4.0×` | event and background RMS, the number of windows used, a stage-change note |
| `Duration` | seconds and the run's limits, for example `1.37 s · limits 0.5–3 s` | the half-waves above background and the nominal cycle count (spindles) |
| `Peak freq` | the dominant rhythm, with `in band` or `OFF BAND`; `≈` before the value for an event under 1 s | prominence in dB, a weak-peak or coarse-resolution note |
| `Wave freq` | one wave per event length, for slow waves and K-complexes (replaces `Peak freq`) | the searched band |
| `Amplitude outlier` | `yes · 984 µV` or `no · 41 µV` | the channel's outlier rule and the detector-threshold line; for slow waves the trough, peak-to-peak and negative half-wave |

Where a figure is missing the value reads `not recorded` (a 4.5 run),
`computing…`, `near a splice`, `too little background` or `not computed`, with
the reason in the tooltip. `Duration` and `Amplitude outlier` always have a
value. The detector's own first-difference frequency is not shown anywhere.

In sample mode a line `Sample event i of N · region · stage` sits under the
header.

**Reading words in live sample review.** For a sample event on which you have no
decision or an unsure decision, the panel hides the reading words and keeps every
number: `in band` / `OFF BAND`, `at the shortest allowed`, `at the longest
allowed`, `outside the limits`, `barely above background`, the `yes` / `no` of
the outlier row, the `flagged: …` part of the sample line, and `meets` / `fails`
in tooltips. Nothing is coloured while hidden, and a line reads `Labels hidden
until you accept or reject this sample event.` The words appear straight after
your accept or reject is written, and are hidden again if it is undone or
cleared. Outside sample mode, and for events outside the sample, they always
show. An undecided sample event is drawn without the outlier mark on the trace.
See [Decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md).

## Epochs Tab Layout

- The centre column is a vertical splitter. The top pane holds the epoch strip,
  navigation, the raw and filtered traces and the action rows; the bottom pane
  scrolls and holds Neighbours and Physiology, both closed on first use. Opening
  them never shrinks the traces below their minimum heights (raw 160 px, filtered
  110 px). The splitter position is remembered.
- Each trace's vertical range comes from a robust estimate of that epoch (the
  larger of a floor, 50 µV raw and 10 µV filtered, and about 8 robust standard
  deviations), so one large event does not flatten the trace. A sample beyond the
  range is drawn clipped at the edge, marked by a thin line, and a note reads
  `clipped at ±200 µV · largest 984 µV`. The **Full range** checkbox shows the
  whole range; it is off by default and resets when the epoch changes.
- **One selection tool: the brush on the trace.** There is no selection on the
  epoch strip (no Shift+drag, no "Exclude N epochs…"); a click on the strip
  pages. Drag on the raw trace to brush a range (blue, labelled `not saved`). The
  action row under the filtered trace holds a hint on the left, and **Clear
  range** and the primary button on the right. **Clear range** is enabled only
  while an unsaved range or a selected excluded range exists, and `Esc` does the
  same. **Exclude time range…** saves the brush: the time is excluded from
  analysis for every channel and saved to this review. It takes effect only when
  you export a re-run package (**File ▸ Export re-run package…**) and re-detect
  with it; events already detected are not changed. The `<stem>_review-qc.xml`
  file beside the annotation file is a record of the review and is not read by
  detection. This is separate from
  rejecting one event with the reason **Artefact**, which labels that event only.
- **Saved exclusions are visible and removable.** Excluded time you saved is
  drawn with a purple diagonal hatch and a dashed edge, labelled `excluded`, on
  the raw and filtered traces; events inside keep their own bands on top. A click
  inside the hatched range (away from any event) selects it, or you can click its
  row in the dock's **EXCLUDED TIME** list. The hint then reads `Excluded
  {start}–{end} ({d} s), saved by {rater} on {date}.` and the primary button
  becomes **Remove exclusion**. Removing deletes the range from the review
  (and from its review-qc record), and the time counts as analysed again here.
  Removing an exclusion does not change a package you already exported: the
  status line says `Export a new re-run package to apply this at re-detection.` There is no confirmation: brush the range
  again to restore it. Only exclusions made in the review GUI can be removed
  here, not artefacts from the scoring file.
- **Show events** is a row above the raw trace; see below.
- The strip legend reads `grey bars = events per epoch · red = amplitude outliers
  · purple = excluded time · white line = current epoch`, with ` · blue ticks = sample events` in sample mode and ` · grey ticks below = epochs with shown events` while Show events filters.

## Decision Controls

Buttons **Accept**, **Reject**, **Unsure**; a grid of reason buttons; a comment
field (500 characters); a `Current` line (`Not reviewed`, `Accepted by TK ·
14:02`, and so on); a flat **Clear** that removes your decision on the event; **Prev** and **Next** buttons; and a progress line.
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

The `Artefact` reason labels that one event only; its tooltip points to **Exclude time range…** for leaving time out of the analysis. Only a saved decision is highlighted: with no decision by you none of Accept,
Reject and Unsure looks selected, and after `A` only Accept does. While Reject or
Unsure is armed (waiting for a reason) that button shows a border only, so an
armed re-decision can sit beside the saved one. A reject needs a reason; an unsure does not. Pressing `R` arms a reject and a
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

## Show Events

A row directly above the raw trace: `Show events:` and four checkboxes,
`unreviewed`, `accepted`, `rejected` and `unsure`, all ticked at launch and not
remembered. It replaces the former REVIEW STATUS group in the left dock. It uses
your own decisions only: `unreviewed` means no decision by you, so an event
decided only by another reviewer counts as unreviewed. It applies to the Epochs
tab only.

While not all four are ticked:

- a chip reads `Showing: rejected · 3 of 1,847 on PPOz ✕` (`accepted, unsure`
  when two are ticked); **✕** ticks all four again;
- events that do not pass are drawn at half strength and skipped by `}` and `{`;
  clicking still selects them, and a rejected event that passes keeps its ✗ and
  dashed edge;
- the epoch strip draws a thin tick along its bottom edge under each epoch that
  holds a shown event on the channel;
- **◀ previous shown** and **next shown ▶** appear at the right of the row and do
  what `{` and `}` do (disabled when nothing matches; `}` then says `No rejected
  events on PPOz.`).

The filter does not affect `]` and `[`, sample navigation, `EVENT i OF n IN
EPOCH`, progress or the Channels tab. In live sample review it reads only your own
decisions, so it reveals nothing about undecided events. The left dock still has
`Filters apply globally to both tabs.` above the channel list.

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
| `}` / `{` | Next / previous event of any status on this channel (only events passing a check or Show events filter, when one is on) |
| `Ctrl+Z` | Undo the last decision (at least 200 steps) |
| `Esc` | Cancel an armed decision; else leave the comment field; else clear the unsaved range or deselect the excluded range (`Clear range`); else remove the check filter; else clear the selection |
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
on the Epochs tab, or the event type, method, band, channel count, check
counts and `n channel(s) excluded` on the Channels tab); the save line `Decisions save to neural_events.db
as you make them.` (`Set a reviewer name to save decisions.` without a name, and
a message when the library has no review store); and the re-detect queue count.
Decision confirmations appear as temporary messages.

Under the epoch strip a legend reads `grey bars = events per epoch · red =
amplitude outliers · purple dashes = excluded time · white line = current
epoch`, with ` · blue ticks = sample events` in sample mode.

## Neighbours and Physiology

**Neighbours** shows the selected event on the target channel and up to six
nearest EEG channels (by electrode position, else the same region) over 4 s
(spindles) or 6 s (slow waves, K-complexes). Two thin blue lines through all rows
mark the selected event's start and end, and a short bar under a neighbour's
trace marks an event detected on that channel. A one-line legend reads `blue
lines = the selected event · bar under a trace = an event detected on that
channel · all rows share one scale (±h µV)`, where `h` is taken
from the data in the window, with a floor (25 µV for spindles, 75 µV for slow
waves and K-complexes). Rows are labelled by rank, not distance: `E75 · target`,
then `E19 · 1`, `E23 · 2` with 1 the nearest, and `~E19 · 1` for an interpolated
channel. The region and selected-channel fallbacks are not ranked, so their rows
carry the name only. No distance is shown, because EEGLAB coordinates carry no
reliable units.

**Physiology** shows EOG, chin EMG and ECG channels the file types as such,
filtered for display (EOG 0.3–15 Hz, chin EMG above 10 Hz, ECG unfiltered). Each
row is scaled from its own data in the current epoch, and its right-aligned
label gives the half-range: `±50 µV` when the file states microvolts, `±0.5 mV`
for another stated unit, and `±0.05 · no unit in file` when the file states
none. The legend adds `no unit stated in this file for …` when that applies. The
selected event is marked by two thin vertical lines on each row; no box, fill or
text covers the signal. Each group collapses, and the choice is remembered.

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

A non-modal window, titled `Precision report · {subject}`, that refreshes when a
decision is written in sample mode, or undone. A free-browsing decision or
**Clear** leaves an open report stale until you reopen it. It fits one screen
with no scrolling. Top to bottom:

```text
Spindles · Moelle2011 11–16 Hz · reviewer TK
TK reviewed 120 of 120 sampled events: 119 accepted, 1 rejected (artefact 1).
Estimated precision: 99 % (95 % confidence 95–100 %)
Looks trustworthy (every region ≥ 80 %)

            NREM2   NREM3   All stages
frontal     100 %   100 %   100 %
parietal    100 %    93 %    99 %

No second reviewer yet.
                        [ Copy summary ] [ Export CSV… ] [ Close ]
```

- **Sentence.** The reasons are links (up to three, then `and n more`; `{u}
  unsure` is a link too). Clicking one opens a list under the table, `Rejected as
  artefact (1)`, with rows `channel · time · stage`; clicking it again closes it.
  Double-click a row to open that event in the Epochs tab; the report stays open.
- **Precision line.** Precision weighted to all of the night's events with a 95 %
  Wilson confidence range. Unsure events are left out.
- **Verdict.** `Looks trustworthy (every region ≥ 80 %)` or `Check parietal ·
  NREM2 (54 %): below 80 %.` (three groups listed, then `and n more`). Groups
  with fewer than 10 decided events show `—` and are not judged; the verdict says
  how many.
- **Table.** Cells are percentages only; a cell below the rule reads `54 % ▼`.
  Hover a cell for the confidence range and the counts.
- **Second reviewer.** With none: `No second reviewer yet.` With two reviewers a
  picker appears in the title line. Until you have decided every sampled event
  the picker is disabled and the line reads `{B} has also reviewed this sample.
  Agreement is shown once you have decided all {n} events.` so another
  reviewer's decisions cannot influence yours. After that the line gives percent
  agreement on the events you both decided (three-way, unsure included) and
  Cohen's kappa (which leaves out events either reviewer marked unsure).
- **Precision rule.** The threshold (default 80 %) and whether it applies to the
  point estimate or the lower 95 % bound are set in **Review ▸ Precision rule…**,
  not in the report. It is a lab convention, not a published standard. The
  library's `TRUSTWORTHY`, `EXCLUDE` and `TOP_UP` verdicts and `top_up_region`
  have no GUI yet.
- **Buttons.** **Copy summary** copies the title line, sentence, precision line
  and verdict as four lines of text. **Export CSV…** writes the table. Opening
  the report also writes `review_precision` rows to the database.

See [Validate a detection run](../how-to/validate-a-detection-run.md).

## Data In / Out

**Input:** a `neural_events.db` SQLite database (created by the `Paral*`
detection pipeline), an EEGLAB `.set`/`.fdt` (or other Wonambi-supported)
EEG file, and optionally a Wonambi annotation XML for sleep stages.

**Output:**

- **Export QC report…** — a Markdown summary (per-channel QC table, flagged
  channels, excluded time ranges) for the current event type.
- **Export re-run package…** (File menu, Export menu, or the `n queued` link) —
  a snapshot of the current results plus `channels.csv` (the channels that are
  not excluded), `redetect_channels.csv` (only the queued channels, for
  `examples/rerun_detection.py --channels`) and a `rerun_sidecar.xml` annotation copy
  that carries every current time exclusion (each package is complete, so a range
  exported earlier is included again). After the export the dialog suggests a
  command: `examples/rerun_detection.py … --channels redetect_channels.csv` when
  channels are queued, otherwise the event type's detector script with `--annot`;
  for PAC it says to re-run from `turtlewave_gui`. There is no re-detect request
  file any more. See
  [Re-run Detection on Reviewer-Selected Channels](../how-to/rerun-detection-on-channels.md).

Channel-level QC verdicts (kept / excluded) and the re-detect queue live in
the same database, in tables the GUI manages
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
