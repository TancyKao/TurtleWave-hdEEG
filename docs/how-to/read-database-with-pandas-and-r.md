# How to Read `neural_events.db` from pandas and R

Since 4.2, detection writes straight into `neural_events.db` and does not
produce a CSV. This guide shows the query patterns that replace loading a CSV
for downstream statistics, in both pandas and R.

## Where did my CSV go?

**Problem:** You upgraded to 4.2, ran detection the way you always have, and
there's no `spindle_parameters_*.csv` where you expect one.

**Solution:** As of 4.2, `write_db` defaults to `None` (AUTO), which means
"write to the database" — the JSON → CSV → import pipeline that used to
produce that CSV no longer runs unless you ask for it. You have three ways to
get tabular data back:

1. **Query the database directly** — the rest of this page. This is the
   recommended path: no intermediate file to go stale, and it's faster for
   anything beyond "give me one flat file."
2. **Export one scope to CSV on demand** —
   [`export_events_to_csv`](../reference/api/dbwrite.md), which writes the
   same column layout the legacy exporters did:
   ```python
   from turtlewave_hdEEG import export_events_to_csv

   csv_path = export_events_to_csv(
       db_path="wonambi/neural_events.db",
       event_type="spindle", method="Moelle2011",
       frequency=(11, 13), stage=["NREM2", "NREM3"],
       output_dir="wonambi/spindle_results",
   )
   ```
3. **Opt back into the legacy pipeline** for new runs — pass `write_db=False`
   (or `--legacy-json` on the driver scripts). See
   [Write Detection Results Directly to the Database](direct-to-database-detection.md#opt-out-the-legacy-json-csv-import-path).

Density CSVs are a special case: they're not just "the same numbers as a CSV
export" — see [Report density](#report-event-density) below, which uses
[`turtlewave_hdEEG.density.event_density`](../reference/api/density.md)
rather than a CSV at all.

## Prerequisites

- Events (and, if you want density, `analysed_time`) already detected into
  `neural_events.db` — see [Detect Spindles](detect-spindles.md) or the other
  detector how-to pages. A database produced with `--legacy-json` /
  `write_db=False` has no `analysed_time` table; density queries against it
  will fail (see [Report event density](#report-event-density)).
- Python: `pandas` (already a dependency of `turtlewave_hdEEG`).
- R: the [`DBI`](https://dbi.r-dbi.org/) and
  [`RSQLite`](https://rsqlite.r-dbi.org/) packages
  (`install.packages(c("DBI", "RSQLite"))`).

## The tables you'll query

| Table | One row per | Key columns |
|---|---|---|
| `events` | detected event | `event_type`, `channel`, `method`, `stage`, `start_time`, `duration`, `freq_lower`/`freq_upper`, `run_id` |
| `detection_runs` | detection invocation | `run_id`, `method`, `citation`, `params_json`, `reject_artifacts`, `reject_arousals` |
| `pac_coupling` | channel × scope PAC result | `subject`, `channel`, `event_type`, `method`, `stage`, `mi_norm`, `preferred_phase_deg`, `mean_vector_length` |
| `analysed_time` | subject × stage × rejection-setting | `analysed_seconds`, `artefact_seconds_excluded` — the density denominator |
| `sleep_cycles`, `stage_durations` | cycle / stage, per subject | populated by [Finalize Sleep Cycles & Stage Durations](detect-sleep-cycles.md) |
| `processing_status` | channel × detection scope | `success`, `error_message` — what `resume=True` reads |

`events.method` is the canonical, unescaped method string (e.g.
`'AASM/Massimini2004'`, not `'AASM_Massimini2004'`) — see
[About naming, subject identity & provenance conventions](../explanation/naming-and-identity-conventions.md).

## Read events into a pandas DataFrame

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("wonambi/neural_events.db")

spindles = pd.read_sql_query(
    """
    SELECT channel, start_time, duration, stage, method,
           freq_lower, freq_upper, peak2peak_amp, rms
    FROM events
    WHERE event_type = 'spindle'
      AND method = 'Moelle2011'
      AND stage IN ('NREM2', 'NREM3')
    """,
    conn,
)
conn.close()

spindles.groupby(['channel', 'stage']).size()
```

Join in the run's provenance (parameters, reference channel, citation) when
you need to report or verify what a run actually used — add `run_id` to the
event query and join it against `detection_runs`:

```python
spindles = pd.read_sql_query(
    "SELECT run_id, channel, start_time, duration, stage, method, "
    "freq_lower, freq_upper, peak2peak_amp, rms FROM events "
    "WHERE event_type = 'spindle' AND method = 'Moelle2011'",
    conn,
)
runs = pd.read_sql_query(
    "SELECT run_id, citation, params_json, reject_artifacts, "
    "reject_arousals FROM detection_runs",
    conn,
)
spindles_with_provenance = spindles.merge(runs, on='run_id', how='left')
```

## Read events into R

```r
library(DBI)
library(RSQLite)

con <- dbConnect(RSQLite::SQLite(), "wonambi/neural_events.db")

spindles <- dbGetQuery(con, "
    SELECT channel, start_time, duration, stage, method,
           freq_lower, freq_upper, peak2peak_amp, rms
    FROM events
    WHERE event_type = 'spindle'
      AND method = 'Moelle2011'
      AND stage IN ('NREM2', 'NREM3')
")

aggregate(start_time ~ channel + stage, data = spindles, FUN = length)

dbDisconnect(con)
```

`dbGetQuery` returns a plain `data.frame`; pipe it into `dplyr` /
`tidyverse` as usual. Parameterise with `dbGetQuery(con, sql, params = list(...))`
rather than pasting user-supplied values into the query string.

## Report event density

Density (events per minute) needs an artefact-free denominator that isn't a
column on `events` — use
[`turtlewave_hdEEG.density.event_density`](../reference/api/density.md)
rather than computing it yourself from `events` and `stage_durations`.
`stage_durations` holds **raw hypnogram time with no artefact subtraction**;
dividing by it under-estimates density in proportion to each recording's
artefact load. `event_density` reads the correct denominator from
`analysed_time`, which detection stores automatically on the direct-write
path:

```python
from turtlewave_hdEEG.density import event_density, format_density_table

density_df = event_density(
    "wonambi/neural_events.db",
    event_type="spindle", method="Moelle2011",
    stage=["NREM2", "NREM3"],
    reject_artifacts=True, reject_arousals=False,  # must match the detection run
)
print(format_density_table(density_df))
```

`density_df` is a regular DataFrame (`subject`, `channel`, `stage`,
`n_events`, `analysed_minutes`, `density_per_min`, ...) — write it to CSV with
`density_df.to_csv(...)` if you need a flat file, or hand it straight to R via
`density_df.to_csv()` / `pyreadr`, or query `analysed_time` from R directly if
you'd rather compute the ratio there:

```r
counts <- dbGetQuery(con, "
    SELECT channel, stage, COUNT(*) AS n_events
    FROM events WHERE event_type = 'spindle' AND method = 'Moelle2011'
    GROUP BY channel, stage
")
denom <- dbGetQuery(con, "
    SELECT stage, analysed_seconds FROM analysed_time
    WHERE subject = 'sub-001' AND reject_artifacts = 1 AND reject_arousals = 0
")
merged <- merge(counts, denom, by = "stage")
merged$density_per_min <- merged$n_events / (merged$analysed_seconds / 60)
```

`reject_artifacts` / `reject_arousals` must match the settings the detection
run used — they're part of `analysed_time`'s key, since a run that kept
arousal epochs analysed more seconds than one that dropped them.

!!! note "`missing=` only governs an EXPLICIT `stage=`"
    Omitting `stage` resolves the scope in two steps, both logged: it first
    tries the stage set the matching run actually searched (recovered from
    `processing_status` / `detection_runs`), which includes a stage that was
    analysed and found nothing; only if the database records **no** detection
    scope at all does it fall back to the stages that happen to appear in
    `events` — a weaker fallback that a `logger.warning` names explicitly,
    since a stage searched and found nothing is invisible to it.

    `stage=None` never *raises* over a missing denominator, regardless of
    `missing=`. A stage in the implicit (recorded) scope with no stored
    `analysed_time` row for your `reject_artifacts=`/`reject_arousals=`
    (commonly because `processing_status` carries a row from a run with
    *different* rejection settings — it isn't keyed by them) is left out of
    the stage *scope*: no zero-event filler row is added for it, and it takes
    no part in a pooled denominator. But it is **not** dropped from the
    output wholesale — if that stage actually has events, those rows are
    still returned, with `analysed_minutes` and `density_per_min` as `NaN`
    and `denominator_source='missing'` (`format_density_table` renders them
    as `nan/min`, easy to miss if you're skimming). `missing='raise'`/`'nan'`
    only takes effect when you pass `stage=` **explicitly** — then a missing
    denominator for one of *your* requested stages raises (`'raise'`, the
    default) or returns `NaN` rows (`'nan'`) instead of being left out of
    scope. Pass `stage=` explicitly if you want a missing denominator to be
    an error rather than a `NaN` row you have to notice yourself.

## Read PAC results

```python
pac = pd.read_sql_query(
    "SELECT channel, stage, mi_norm, preferred_phase_deg, mean_vector_length, "
    "rayleigh_p, n_events FROM pac_coupling "
    "WHERE subject = ? AND method = ?",
    conn, params=["sub-001", "Staresina2015_paired_Moelle2011"],
)
```

```r
pac <- dbGetQuery(con, "
    SELECT channel, stage, mi_norm, preferred_phase_deg, mean_vector_length,
           rayleigh_p, n_events
    FROM pac_coupling
    WHERE subject = ? AND method = ?
", params = list("sub-001", "Staresina2015_paired_Moelle2011"))
```

`pac_coupling` does not store the per-event modulogram matrix — that's in the
sibling `*_mean_amps.npy` file next to wherever PAC's `out_dir` pointed
(always written, regardless of `write_csv`); read it with `numpy.load` in
Python, or `reticulate`/`RcppCNPy` from R if you need it there.

## A note on concurrent access

`neural_events.db` is a single SQLite file. Reading it from pandas or R while
a detection run or the review GUI holds a write connection can block for the
duration of a busy-timeout, or fail outright on some network filesystems. See
[Database concurrency and journalling](../explanation/database-concurrency-and-journalling.md)
and [Run with the database on a network drive](run-with-database-on-a-network-drive.md)
if you're reading from a shared or mapped-drive copy. Reading is always safe
once no detection job is actively writing.

## Next steps

- [Reference: density module](../reference/api/density.md)
- [Reference: dbwrite module](../reference/api/dbwrite.md)
- [Write Detection Results Directly to the Database](direct-to-database-detection.md)
- [Review EEG Events](review-eeg-events.md)
