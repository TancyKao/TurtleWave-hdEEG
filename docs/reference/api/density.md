# Event Density API Reference

`turtlewave_hdEEG.density` computes per-channel event density **derived on
read** from `neural_events.db` — there is no `density` table. The numerator
(`GROUP BY` count over `events`) is cheap to recompute on every call; the
denominator — the artefact-free in-stage time a detection run actually
analysed — is stored once per subject/stage/rejection-setting in
`analysed_time` (written by
[`turtlewave_hdEEG.dbwrite.store_analysed_time`](dbwrite.md) as part of every
direct-write detection run) and read back here.

`stage_durations` (see [Sleep Cycle Processor](cycleprocessor.md)) is
deliberately **not** an accepted denominator, even as a fallback: it holds raw
hypnogram time with no artefact subtraction, and dividing an artefact-free
event count by it under-estimates density in proportion to each recording's
artefact load — a per-subject bias that can look like a real group
difference. See
[Event density is artefact-free](../../explanation/overview.md#event-density-is-artefact-free).

A database written on the legacy `write_db=False` path has no `analysed_time`
table; `event_density` raises rather than silently falling back to
`stage_durations` or returning zeros. Use the (deprecated)
`export_*_density_to_csv` methods against that JSON directory instead, or
back-fill `analysed_time` for the recording.

`stage=None` prefers the stage set the matching detection run actually
searched (recovered from `processing_status` / `detection_runs`), which
includes a stage that was analysed and produced no events. Only when the
database records no detection scope at all does it fall back to the set of
stages that happen to appear in `events` instead — a `logger.warning` names
this fallback explicitly, since it can't represent a searched-but-empty
stage. `missing=` does not apply to the (recorded) implicit scope: a stage
there with no stored denominator (typically a `reject_artifacts=`/
`reject_arousals=` mismatch against `processing_status`, which isn't keyed
by them) is left out of the stage *scope* — no zero-event filler row, no
share of a pooled denominator — with a logged warning, regardless of
`missing=`. That stage's rows are **not** removed from the result wholesale,
though: any events actually detected in it are still returned, with
`analysed_minutes`/`density_per_min` as `NaN` and
`denominator_source='missing'` (`format_density_table` renders these as
`nan/min`). `missing='raise'`/`'nan'` governs only an *explicitly* requested
`stage=`: a missing denominator for one of those stages raises (default) or
returns `NaN` rows (`missing='nan'`) instead of being left out of scope.

See [How to read `neural_events.db` from pandas and R](../../how-to/read-database-with-pandas-and-r.md#report-event-density)
for a task-oriented walkthrough, and
[Write Detection Results Directly to the Database](../../how-to/direct-to-database-detection.md#what-lands-in-the-database)
for how `analysed_time` gets populated.

::: turtlewave_hdEEG.density
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
