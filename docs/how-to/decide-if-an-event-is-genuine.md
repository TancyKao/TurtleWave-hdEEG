# How to Decide Whether a Detected Event Is Genuine

Use this when you have selected one event in the **Epochs** tab of the review
GUI and need to record accept, reject or unsure. It applies the same six checks
to every event, in the same order, so two reviewers reach the same call for the
same reasons.

The numbers in the Event panel are aids for these checks. They are flags, not
rejection rules: no figure rejects an event on its own. See
[Event figures and review sampling](../explanation/event-figures-and-review-sampling.md)
for what each figure is and where it fails.

## Before you start

- Open the database, the EEG file and the annotation file (**File** menu), then
  drill into a channel with **Drill into epochs ▸**.
- Set your name with **Review ▸ Reviewer name…**. Nothing is saved without it.
- Select an event by clicking its band on the trace, or press `]` for the next
  event you have not decided.
- The Event panel is in the right dock. Run the checks below from the top.

## The six checks

Work through them in order and stop at the first one that fails.

### 1. Look at the raw trace first

Look at the unfiltered trace before any number. If you cannot see the
oscillation in the raw signal, reject with **Not visible in the raw trace**. If
a sharp step or spike sits under it, reject with **Filter ringing**: a band-pass
filter turns a transient into a short oscillation.

Where: the raw trace above the filtered trace in the Epochs tab. A peak
frequency that is off-band or has low prominence backs up a ringing call.

### 2. Check the duration and the half-waves

A spindle should last at least half a second and show several waves that stand
out from the background. Few or no standing-out half-waves means the filter
found a rhythm your eye would not.

Where: the **Duration**, **Half-waves** and **Cycles (nominal)** rows. The
half-wave count is the number of band-passed peaks and troughs at least 2.5
times the background RMS. The nominal cycle count is not gated, so it can read
12 or more on noise; trust the half-wave count. An event at the floor of the
run's duration limits is a weaker call than one well above it.

### 3. Check shape and size against the background

A genuine event waxes and wanes and is clearly larger than the seconds around
it. An amplitude-versus-background figure near 1 means it does not stand out.

Where: the **Amp. vs background** and **Amp. vs threshold** rows, and the
filtered trace. An amplitude-versus-threshold figure of 1.0 to 1.2 means the
event barely crossed the detector's bar. For slow waves and K-complexes the
**Trough**, **Peak-to-peak** and **Negative half-wave** rows describe the shape.

### 4. Check the frequency and the site

The peak should sit inside the run band. A peak near 8 to 10 Hz on a posterior
channel in a 9 to 12 Hz run is usually posterior alpha, not a spindle. Reject it
as **Outside the frequency band** unless the frontal neighbours show the same
burst.

Where: the **Peak frequency** row (with an `in band` or `OFF BAND` badge), its
prominence line, and the **Channel** row. Under 1 s the peak frequency is
coarse and the low-prominence label is unreliable: a third to a half of weak
genuine spindles shorter than 1 s get it.

!!! warning "The run band can itself be wrong for the site"
    `in band` is true for alpha when the run band includes alpha. The call then
    rests on the site, the neighbours and the EOG and EMG, not on the badge.

### 5. Check the spread onto neighbours

A genuine spindle usually appears on adjacent channels at about the same time.
One channel only, with clean neighbours, is suspect but not proof.

Where: the **Neighbours** group under the filtered trace. The target is on top
and up to six nearest channels follow, with their own detected events shaded.

### 6. Check the EOG and EMG context

A rise in chin EMG, or eye movements, during the burst means an arousal or an
eye artefact.

Where: the **Physiology** strip under Neighbours. It shows only the kinds of
channel the file types as EOG, chin EMG or ECG.

## Record the decision

1. Press `A` to accept when all six agree.
2. Press `R` to reject, then a digit for the first reason that failed (table
   below). `Enter` repeats the last reject reason of the session.
3. Press `U` to mark unsure when the checks conflict. A reason is optional.
4. Press `C` to type a comment, and `Enter` to save it. Reason `9` (Other)
   needs a comment.
5. Press `Ctrl+Z` (`Cmd+Z` on macOS) to undo the last decision.

Nothing is written for a reject until you choose a reason.

## Reason codes and their RA categories

Each decision maps to one of four categories of the research-assistant (RA)
protocol. `turtlewave_hdEEG.dbwrite.review_category(decision, reason)` returns
the category.

| Key | Reason code | Meaning | Category when rejected |
|---|---|---|---|
| `1` | `artefact` | movement, electrode or muscle | FP-artifact |
| `2` | `arousal` | EEG speed-up, often with EMG | FP-other |
| `3` | `too-short` | too short | FP-other |
| `4` | `filter-ringing` | step or spike in the raw trace | FP-other |
| `5` | `eye-movement` | eye movement | FP-artifact |
| `6` | `not-in-raw` | not visible in the raw trace | FP-other |
| `7` | `off-band` | outside the frequency band | FP-other |
| `8` | `single-channel` | only on this channel | FP-other |
| `9` | `other` | described in the comment | FP-other |
| combo only | `not-isolated` | not isolated from other waves | FP-other |
| combo only | `wrong-morphology` | wrong shape for the event type | FP-other |

The other decisions map as follows: accept is TP (true positive), unsure is
Ambiguous, and a reject with no reason is FP-other. Posterior alpha without an
arousal is `off-band`, not `arousal`.

## Decisions are evidence, not deletions

A reject is stored in the `event_reviews` table, keyed by the event's uuid and
your name. It does not remove or alter the event in `events`. Density, CSV
exports and every other reader count the event as before unless you ask them to
leave rejected events out (`exclude_rejected=True`, off by default; see
[Validate a detection run](validate-a-detection-run.md#leave-rejected-events-out-of-density-or-an-export)).

Two consequences follow:

- Rejected events are the raw material of the precision estimate. Removing them
  would bias it.
- A reviewer who changes their mind overwrites only their own row. Other
  reviewers' decisions on the same event stay.

To remove the *time* around an event from the density denominator, brush it on
the trace and use **Mark as artefact**. Rejecting with reason `artefact` does
not do that.

## See also

- [Validate a detection run](validate-a-detection-run.md)
- [Event figures and review sampling](../explanation/event-figures-and-review-sampling.md)
- [Reference: EEG Review GUI](../reference/eeg-review-gui.md)
