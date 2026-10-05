# Explanation: Event Figures and Review Sampling

This page explains why 4.6 stores a set of figures for every detected event,
why a person labels only a drawn sample of events, and what neither of them can
tell you. For the steps, see
[Decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md)
and [Validate a detection run](../how-to/validate-a-detection-run.md).

## Why figures at detection time, and why a sample

A night's spindle detection on a 257-channel montage yields about 94,000 events.
The early plan for 4.6 was per-event accept and reject for curation. At that
scale it cannot work: nobody decides 94,000 events, and a handful of decisions
changes nothing in the totals.

So the question changed from "which events are wrong?" to "how often is the
detector wrong, and where?". Two tools answer it:

- A set of figures is computed for every event while the channel signal is
  already in memory, and stored beside it. These find whole-channel problems,
  such as a region that picks up alpha, without anyone looking at an event.
- A stratified random sample of about 120 events is labelled by a person. The
  weighted share accepted estimates precision per region and stage, with a
  confidence interval.

The figures are computed after each channel finishes detection, on the
detector's own segment, so detection output is unchanged. Computing them costs
about 0.4 to 0.5 s per channel, measured on the development fixtures, and they
can be switched off with `compute_figures=False`.

## The five figures

Each is defined in the Method Spec revision 3 and implemented in
`turtlewave_hdEEG.event_metrics`. All are computed on the signal the detector
saw: the run's reference applied, restricted to the run's stages, rejected time
removed, and the remaining bouts joined at splices.

**Half-waves standing out from background** (`halfwaves_above_bg`, spindles).
The event is band-passed with a second-order Butterworth filter over the run
band. The figure is the number of local maxima and minima whose absolute value
is at least 2.5 times the background RMS. It is shown beside the nominal cycle
count (`cycles_nominal`, sign changes divided by 2), which is not gated by the
background. The gate exists because the nominal count reads about 13 cycles on
pure noise (duration times band centre). Band-limited noise has a Rayleigh
envelope, so one extremum exceeds 2.5 times RMS with probability about 4 %. On
noise the half-wave count averaged 2.3, against 18 to 26 on a 13 Hz, 1 s
spindle. The figure is undefined when the background is.

**Peak frequency and prominence** (`peak_freq_ap`, `prominence_db`, `in_band`,
`low_prominence`, spindles). A Hann periodogram of the mean-removed raw event
samples over 4 to 30 Hz is fitted with a straight line in log-log coordinates.
The peak is the largest interior local maximum of the residual in dB, never an
endpoint. `prominence_db` is the residual there. `in_band` says the peak lies in
the run band, `low_prominence` that the prominence is under 10 dB, and an event
with no interior maximum is `no_peak` (stored as NULL, not as off-band). An
event under 1 s is `coarse`: its true frequency resolution is 1 over its
duration. The 4 to 30 Hz range exposes theta, alpha and beta on both sides of
every spindle band in use. The detector's own peak frequency
(a first-difference periodogram over 0 to 50 Hz, stored as `peak_freq`) is kept
for comparison: differencing multiplies power by about the square of the
frequency and returned 28 to 50 Hz on noise.

Slow waves and K-complexes have no spectrum figure. Their `wave_freq` is 1 over
twice the negative half-wave duration, compared with the run band. The negative
half-wave is the end minus the detector's zero crossing for Massimini-family
detectors (positive half-wave first), and the zero crossing minus the start for
Ngo2015 and Staresina2015.

**Amplitude against background** (`amp_ratio`, `bg_rms`, `bg_n_windows`,
`bg_stage_mixed`). `amp_ratio` is the band RMS of the event divided by the
median band RMS of non-overlapping background windows tiled across the
surround. The surround is 15 s each side for spindles (window 0.5 s, guard
0.25 s) and 30 s for slow waves and K-complexes (window 2 s, guard 0.5 s).
A window is dropped when it:

- overlaps the event or any other event on the channel in the same run, each
  widened by the guard;
- overlaps rejected time or a stage outside the run's stages;
- is not wholly inside one band-passed bout, or lies within the edge guard of
  that bout's edge (2 s for spindles, 10 s for slow waves and K-complexes).

With fewer than 10 windows (8 for slow waves and K-complexes) the background is
undefined and the figures that depend on it are NULL. The median keeps one
unflagged artefact window from setting the floor. A spindle train hides
undetected events in the background, so the ratio understates there.

`bg_stage_mixed` is stored as true only when a window that was otherwise
usable was dropped because it lay in a stage outside the run. It is not set
when out-of-stage time is simply absent from the detection segment, because
those windows were unavailable anyway. A true value means the event sits near a
stage change.

**Amplitude against threshold** (`thresh_ratio`). The event's stored detector
peak divided by the detector's own threshold, shown only where the two are the
same signal in linear units: Moelle2011, Ferrarelli2007 and Nir2011 (peak over
the lower detection threshold) and the Massimini family (the smaller of trough
over its criterion and peak-to-peak over its criterion). It is at least 1 by
construction, so 1.0 to 1.2 means barely crossed. Ray2015, Wamsley2012,
Martin2013 and Lacourse2018 show no ratio because the stored peak is a
different signal from the threshold; Ngo2015 and Staresina2015 do not return
their per-channel thresholds; CIRUS returns none.

**Duration against bounds** (`near_bound`). Minus 1 when the duration is within
0.05 s of the run's floor, plus 1 within 0.05 s of the ceiling, 0 otherwise. The
floor wins if both apply. It restates a bound the detector already enforced, so
it is display-only and not part of the sampling flag.

### Rounding and edge rules

- Event samples are those with `start <= t <= end`, found on a global sample
  grid with a tolerance of 1e-6 samples, so a one-continuous read and a
  concatenated detection segment tile windows identically.
- A signal is split into bouts where the time step exceeds 1.5 samples or does
  not advance. Each bout is band-passed on its own, so the filter never runs
  across a splice.
- An event within the edge guard of a bout edge, or spanning one, is
  `near_splice`. Its signal figures are NULL by design. `near_bound` and
  `thresh_ratio` need no signal and are still stored. In the sampling flag a
  `near_splice` event counts as flagged.

## Why thresholds are stored per run, channel and segment

`thresh_ratio` needs the threshold the detector used for that event. Wonambi
resolves thresholds from the data it is handed, so they differ by channel. With
a `cat` other than `(1, 1, 1, 0)` every contiguous bout is its own segment with
its own threshold. Two runs on one scope (for example a scoped re-run of a few
channels) can also differ. The `detection_thresholds` table therefore keys on
run, channel, method, segment and threshold name, with the segment's first and
last sample time, and a reader looks up the threshold by the event's own
`run_id` and start time. Run-wide criteria, such as the Massimini amplitude
limits, are stored once under channel `*`.

A run detected before 4.6 has no such rows, and the thresholds cannot be
recovered: the detector's cut-offs were local variables. The panel then says
"Threshold not recorded for this run (detected with 4.5 or earlier)". Nothing
is wrong with the events; re-detect with 4.6 to record them.

## The precision estimator

The sample is drawn in cells of region times stage, equal in size, because
per-region figures are the goal and proportional allocation would give a small
region such as occipital NREM3 about two events. Inside a cell, flagged events
are oversampled and each sampled event carries the weight `N_k / m_k`: the
number of events in its sub-cell divided by the number of that sub-cell's events
labelled. Precision is the weighted number accepted over the weighted number
decided, `sum(w a) / sum(w d)`, with unsure events left out of the denominator
and reported as their own weighted share. The draw is deterministic: each event
gets a number from a hash of the seed and its uuid, and a sub-cell takes its
events with the smallest numbers, so a redraw after a small re-detection keeps
most already-labelled events.

The interval is a Wilson score interval computed on the Kish effective sample
size, `(sum w)^2 / sum(w^2)` over decided events. Weights make the effective
size smaller than the label count, and the Wilson form behaves at the high
precisions and small cells seen here (1 to 30 events with precision near 1)
where a linearised variance under-covered in simulation (0.73 to 0.94 against
0.95 to 0.99 for Kish-Wilson, on simulated labels). The simulation of the
revised design is pre-registered and not yet run. No finite-population
correction is applied, which errs wide. The difference between two domains uses
the MOVER (square-and-add) interval.

## Why channel flags are relative, and why the list states only facts

The Channels tab flags a channel against the rest of the montage, with the same
robust z-score as the amplitude flag, and never against a fixed number. A
fixed cut-off such as "30 % off-band is bad" would hold only for one band, one
reference and one cohort, and the figures' own cutoffs already come from
synthetic noise. A relative rule asks a question that does not depend on those
choices: which channels behave unlike the rest? The price is that a problem
every channel shares is not flagged. It shows as a uniform topography, and the
Precision report is where it becomes a number.

Four columns can flag: the off-band share, the at-floor share, the median
amplitude over background and the median amplitude over threshold. The
low-prominence share cannot. It mostly tracks a channel's signal-to-noise, so
flagging on it would flag quiet but healthy channels; it stays on screen as
context. The at-ceiling share (events within 0.05 s of the upper duration
limit) is shown in a tooltip and also never flags.

For an off-band channel the flagged-channel list adds where the off-band peaks
lie: the most common 1 Hz bin, and the shares below and above the run band. The
list stops there. It does not say that a cluster of peaks at 8 to 9 Hz is
posterior alpha, or that a neighbour shows the same pattern. Whether an
off-band rhythm is alpha, an artefact or something real depends on the site,
the neighbours and the EOG and EMG, which is the reviewer's judgement. A label
placed by the software would carry authority the numbers do not have.

The same reasoning hides flag words during live sample review. The Event panel
keeps every number but withholds `OFF BAND`, `low prominence` and similar words
until the reviewer has accepted or rejected the event, because the sample was
stratified on those flags and showing them would let the stratifier steer the
label. Neighbours are labelled by rank (1 is the nearest) rather than distance,
because EEGLAB electrode coordinates carry no reliable units.

## What the figures cannot do

- **They are flags, not rejection rules.** No figure should drop an event.
  The half-wave and prominence cutoffs (2.5 times and 10 dB) come from synthetic
  pink noise. Real 1/f slopes, an average reference and low-amplitude cohorts
  will move them, so the value is always shown beside the label.
- **Low prominence misfires on weak short events.** On simulated spindles of
  0.5 s at an amplitude ratio near 2.5, a third to over half were flagged. A
  channel's low-prominence share tracks its signal-to-noise more than any
  off-band rhythm, which is why the off-band share leads.
- **`in_band` cannot see that the band is wrong.** In a 9 to 12 Hz run, alpha is
  in band. Site, neighbours and EOG and EMG decide that case.
- **Amplitudes do not compare across references.** An average reference lowers
  parietal amplitude.
- **A sample of 120 screens; it does not certify.** A region gets an effective
  sample of 18 to 24 events, so it can show that a region is bad but needs a
  point estimate near 0.90 to show a lower bound of 0.70. The trustworthiness
  and pooling thresholds are lab conventions, not published standards.
- **The labels are not blind to the stratifier.** A reviewer sees the figures
  that define the flag, so the stratum is not shown as a badge, and a flag that
  is shown in the Event panel is the same fact as the figure beside it.

## See also

- [Validate a detection run](../how-to/validate-a-detection-run.md)
- [Decide whether an event is genuine](../how-to/decide-if-an-event-is-genuine.md)
- [Event metrics API](../reference/api/event_metrics.md) and
  [review sampling API](../reference/api/review_sampling.md)
- [Upgrade to 4.6](../how-to/upgrade-to-4.6.md)
