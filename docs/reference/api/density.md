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

## `events.stage` is the run's joint stage token, not the event's epoch

Since 4.3, `stage` is a run-scope label — `'NREM2NREM3'` for a run over N2+N3,
`'NREM2'` for a single-stage run — stored the same on every event that run
wrote, not the epoch the individual event fell in. `analysed_time` stays keyed
per single scored stage (it is a physical quantity of that stage, not a run
label), so `event_density` pools the token's components on read via
[`dbwrite.pooled_denominator`](dbwrite.md): it sums `analysed_seconds` across
the components and returns `NaN` — never a partial sum — if any component has
no stored row. A legacy per-epoch database is a special case of the same
scheme (a token with exactly one component), so backward compatibility is
automatic: `stage=None`, `stage=['NREM2', 'NREM3']` and `stage='NREM2NREM3'`
all resolve to the same rows via `dbwrite.resolve_stage_tokens`.

**A strict subset of an identity's *only* stage token returns no row.**
`stage=['NREM2']` against an `(event_type, method)` identity whose events are
stored *only* under `'NREM2NREM3'` cannot be answered — those events were
never individually attributable to N2 or N3 — so it is reported as missing,
not as a number computed against the wrong denominator, with a
`logger.warning` explaining why. Use `events.epoch_stage` (additive since
4.3, the event's own scored epoch) if you need that split; it isn't part of
`event_density`'s pooling.

## `event_density` and a partially-covered identity

The rule above is not quite "a strict subset always returns nothing" — that's
only true when the identity's events are **entirely** under non-covering
tokens. An `(event_type, method)` identity can hold **both** a token the
request covers **and** one it doesn't — a per-epoch `'NREM2'` row alongside a
later joint-run `'NREM2NREM3'` row, say. That is exactly the state a
**partially-migrated database** is in (some scopes collapsed by
`examples/migrate_stage_to_joint.py`, others not yet), and it's also what a
scope detected twice under different stage sets produces. In that case
`event_density` does **not** refuse: it returns the density computed over the
**covered** tokens only, and emits a `logger.warning` naming how many events
were **excluded** (the ones under the non-covering token) and how many were
reported:

```text
WARNING stage=['NREM2'] does not cover stage token(s) ['NREM2NREM3'] that
event_type=spindle, method=Moelle2011 also holds, so 5760 event(s) are
EXCLUDED from this density and 945 are reported. A joint token is only
selected when every one of its stages was asked for, because its events
cannot be attributed to one of them. Ask for the full set to include them, or
read events.epoch_stage for the per-epoch split. This usually means the scope
was detected twice under different stage sets, or the database is only
partly migrated to the joint token.
```

So the full rule for `stage=['NREM2']` against one `(event_type, method)`
identity:

| Stored tokens for this identity | `event_density(stage=['NREM2'])` returns |
|---|---|
| Only `'NREM2'` (or other tokens `stage=['NREM2']` covers) | The normal density over those rows. No warning. |
| Only `'NREM2NREM3'` (or another token it does **not** cover) | No row for this identity. `logger.warning` explains why. |
| **Both** `'NREM2'` **and** `'NREM2NREM3'` | The density over the `'NREM2'` rows only. `logger.warning` names how many events were excluded and why — this is the partially-migrated-database case. |

## `v_event_density`: density in plain SQL, for R and non-Python callers

[`dbwrite.ensure_density_view`](dbwrite.md) creates a `v_event_density` SQL
view (part of `ensure_direct_write_schema`, so every current database has it)
that expresses the same component pooling in SQL — joining `events` to
`analysed_time` with no Python required. Query it directly:

```sql
SELECT channel, event_type, stage, n_events, density_per_min
FROM v_event_density
WHERE event_type = 'spindle' AND method = 'Moelle2011';
```

It is a **view**, not a table, so it recomputes at query time and cannot go
stale after a scoped re-detection. It diverges from `event_density` in three
ways that matter for a montage summary read from it alone:

- **No honest zeros.** A channel that ran and detected nothing has no row in
  `events`, so it has no row here either — `event_density`'s
  `include_zero_channels` has no SQL equivalent. A montage summary computed
  from `v_event_density` alone is over the channels that fired, not every
  channel processed.
- **No per-identity stage scope.** Rows appear for whatever
  `(event_type, method, stage)` combinations exist in `events`; nothing is
  filtered by which run actually searched which stage — the recorded-scope
  fallback `event_density` does with `processing_status` / `detection_runs`
  has no SQL equivalent either.
- **An out-of-vocabulary stage token reads as an incomplete denominator, even
  when it isn't.** Counting a token's components (how many of NREM1/NREM2/
  NREM3/REM/Wake it decomposes into) has to be done in raw SQL string
  arithmetic, since SQLite has no `split()`. A token outside that vocabulary
  — `'Undefined'`, a site-specific label, anything `split_stage_token` itself
  would raise on — decomposes to **zero** expected components in SQL, versus
  **one** (itself, whole) in Python's `stage_components`. If `analysed_time`
  actually holds a row for that exact label (a run detected on that stage
  literally), the view still counts it as 1 matched against 0 expected,
  `denominator_complete` comes out `0`, and `density_per_min` is `NULL` even
  though `event_density` would compute a real number for the same row. This
  fails safe — `NULL`, never a wrong number — but it means `v_event_density`
  under-reports density for any stage label outside the known vocabulary,
  where `event_density` does not.

`density_per_min` is `NULL` (`denominator_complete = 0`), never `0`, when a
component of the stage token has no matching `analysed_time` row. Rows with a
`NULL` stage are excluded — they have no denominator to join against.

::: turtlewave_hdEEG.density
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
