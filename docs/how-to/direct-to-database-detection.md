# Write Detection Results Directly to the Database

Since 4.2, `detect_spindles`, `detect_slow_waves`, `detect_kcomplexes` and
`analyze_pac` write straight into `neural_events.db` **by default** — no
per-channel JSON, no CSV, no separate import step. This guide covers the
mechanics of that default: what lands in the database, how to resume an
interrupted run, how to verify a batch job actually finished, how to opt back
out to the legacy JSON path, and how to pull a flat CSV back out when you need
one.

If you're looking for how to load `neural_events.db` into pandas or R for
statistics, see
[Read the database with pandas and R](read-database-with-pandas-and-r.md)
instead — that page is the direct replacement for what CSV used to do for
you.

## What happens by default

Each channel's events are written straight into `neural_events.db` in one
transaction, along with a `detection_runs` provenance row recording the
method, citation, full parameter dict, reference/polarity, artefact-rejection
settings and library versions used. No keyword argument is required — this is
what `write_db=None` (the default on all four detection/analysis calls) does:

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

data = WonambiDataset("sub-001_eeg.set")
annot = CustomAnnotations("sub-001_eeg.xml")
event_processor = ParalEvents(dataset=data, annotations=annot)

spindles = event_processor.detect_spindles(
    method="Moelle2011",
    chan=["E110", "E111", "E112"],
    frequency=(11, 13),
    duration=(0.5, 3),
    stage=["NREM2", "NREM3"],
    reject_artifacts=True,
    reject_arousals=False,
    cat=(1, 1, 1, 0),
    save_to_annotations=False,
    json_dir="wonambi/spindle_results",  # locates neural_events.db; no JSON written
    db_path="wonambi/neural_events.db",
    subject="sub-001",
    resume=False,
)
```

`json_dir` still matters on this path: it locates `neural_events.db` (via
`dbwrite.resolve_db_target`, which prefers an explicit `db_path`, then a
database beside `json_dir`) and holds the optional annotation XML, but no
per-channel JSON file is written there anymore. `db_path` can be given
explicitly instead, or omitted and resolved from `json_dir`; either way, an
unresolvable target **raises** rather than silently downgrading to a no-op —
the old behaviour that turned a demanded database write into a silently
discarded run.

`ParalSWA.detect_slow_waves`, `ParalKC.detect_kcomplexes` and
`ParalPAC.analyze_pac` accept the same `write_db` / `db_path` / `subject`
keywords (`analyze_pac` also takes `write_csv`; see
[Run PAC Analysis](run-pac-analysis.md#write-results-straight-to-the-database)).
`resume`, `run_params` and `replace_channels` are specific to the three event
detectors.

!!! tip
    `examples/hdEEG_spindle_detector.py`, `examples/hdEEG_sw_detector.py` and
    `examples/hdEEG_kcomplex_detector.py` already wire this up. Run any of
    them with no extra flags and events land in `neural_events.db`:
    ```bash
    python examples/hdEEG_spindle_detector.py --subject sub-001 --resume
    ```

## Resume an interrupted run

**Problem:** A batch job died partway through a subject with 128 channels and
you don't want to re-detect the 90 that already succeeded.

**Solution:** Pass `resume=True`. A channel is skipped only when a
`processing_status` row already recorded `success = 1` for the **exact same
scope** — event type, method, frequency band and stage set. A channel that
previously failed (`success = 0`) is retried, not skipped.

```python
event_processor.detect_spindles(
    method="Moelle2011",
    chan=all_channels,
    frequency=(11, 13),
    stage=["NREM2", "NREM3"],
    db_path="wonambi/neural_events.db",
    resume=True,
)
```

Crash-resume is the acceptance test for the direct-write path: `kill -9`
mid-run, re-run with `resume=True`, and confirm completed channels are
untouched while the killed channel re-detects cleanly. This works even
without `resume=True` — each event row's `uuid` is a deterministic `uuid5` of
its detection scope, so re-detecting an unchanged channel is a true row-level
no-op under `INSERT OR REPLACE` — but `resume=True` additionally skips the
re-computation, not just the re-insert.

## Run it on the NCI Gadi cluster

The `_GADI.py` driver scripts mirror the same flags:

```bash
python hdEEG_spindle_detector_GADI.py \
    --root /scratch/xx99/subjects --subject sub-001 \
    --method Moelle2011 --stages NREM2,NREM3 --freq 9.0,12.0 \
    --resume
```

See `examples/NCI_commands/hdEEG_spindle_detector_GADI.py` and
`examples/NCI_commands/hdEEG_sw_detector_GADI.py`.

Both drivers verify channel coverage after detection and **exit non-zero if
any requested channel is missing**, instead of printing "All done" over an
incomplete run:

```python
from turtlewave_hdEEG.dbwrite import verify_channel_coverage

coverage = verify_channel_coverage(
    db_path=db_path,
    event_type="spindle",
    method="Moelle2011",
    requested_channels=all_channels,
    freq_lower=9.0,
    freq_upper=12.0,
    stage_key="NREM2NREM3",
)
if not coverage["complete"]:
    print(f"Missing channels: {coverage['missing']}")
    sys.exit(1)
```

`coverage["failed"]` lists channels with an in-scope recorded failure (these
always count as missing, whatever event rows exist for them);
`coverage["events_only"]` lists channels credited by event rows alone with no
matching `processing_status` row for this exact scope — weaker evidence,
since `events.stage` cannot be scoped to the run's joined stage set the way
`processing_status` can. A caller reporting success should report
`events_only` too. `coverage["scoped_status"]` is `False` when the database
predates the per-scope `processing_status` schema; the check then falls back
to an event-type-only comparison that cannot distinguish a status row left by
a different method or band.

## What lands in the database

!!! note
    Writing straight to `neural_events.db` on a mapped network drive or a
    synced cloud folder can fail with `disk I/O error` — see
    [Run with the database on a network drive](run-with-database-on-a-network-drive.md).

Each event row in `events` carries:

- **Detector-own morphology** — `det_trough`, `det_peak`, `det_ptp`,
  `det_trough_time`, `det_peak_time`: the values the detector itself decided
  on (trough/peak/peak-to-peak amplitude and their timestamps). Spindle events
  are oscillatory rather than trough-based, so these are typically `NULL` for
  spindles and populated for slow waves / K-complexes. `det_trough` /
  `det_peak` are signed consistently (trough negative, peak positive) across
  every slow-wave and K-complex method. `det_ptp` is a real microvolt
  peak-to-peak amplitude on a database this release or later populates; a
  database written before it holds Wonambi's sample count in that same
  column instead — the two ranges overlap numerically, so nothing in the
  number itself tells you which it is. `dbwrite.ptp_units(conn)` reports
  `'microvolts'`, `None` (unmarked — treat as sample count if the database
  already holds slow-wave/K-complex rows), from the `db_meta.det_ptp_units`
  marker, seeded the same way as `db_meta.stage_format` below: only on a
  database with no events yet, since stamping an existing database would
  assert something false about rows already in it. `peak2peak_amp` is
  unaffected either way — it is re-measured from the signal, not read off
  the detector.
- **Re-measured spectral/RMS columns** — `rms`, `power`, `peak_power_freq`,
  `energy`, `peak_energy_freq`, plus `min_amp` / `max_amp` / `peak2peak_amp`:
  computed once per channel over all of that channel's in-memory event windows
  (`compute_batched_params`), rather than one raw-file re-read per event. These
  are the same quantities the legacy CSV exporter's `event_params` re-read
  produced, so column semantics are unchanged.
- **`run_id`** — links the event back to the `detection_runs` row for this
  invocation.
- **`stage`** — the run's joint stage token (`'NREM2NREM3'` for a run over
  N2+N3), not the event's own epoch. See
  [What `events.stage` means](#what-eventsstage-means) below.
- **`epoch_stage`** — the event's own scored epoch (`'NREM2'` or `'NREM3'`),
  additive since 4.3. See the same section.
- **`cycle`** — the NREM-REM cycle number the event fell in, tagged
  automatically after detection. See
  [Cycles and stage durations populate automatically](#cycles-and-stage-durations-populate-automatically)
  below.

A `detection_runs` row is written per invocation with the method, a literature
citation (resolved automatically for the built-in methods — Ferrarelli2007,
Moelle2011, Nir2011, Wamsley2012, Martin2013, Ray2015, Lacourse2018,
Massimini2004, Ngo2015, Staresina2015, etc.), the full parameter dict, the
reference channel(s), polarity, requested stages, artefact/arousal rejection
flags, and `turtlewave_hdEEG` / `wonambi` / `numpy` versions plus the git SHA.

The run also stores its density denominator — the artefact-free in-stage
seconds it actually analysed — in `analysed_time`, so
[`turtlewave_hdEEG.density.event_density`](../reference/api/density.md) can
derive density straight from the database. See
[Read the database with pandas and R](read-database-with-pandas-and-r.md).

## What `events.stage` means

Since 4.3, `events.stage` is **the run's stage set, not the event's epoch.**
A run over `stage=['NREM2', 'NREM3']` stores the joined token `'NREM2NREM3'`
on every row it writes — from all three detectors, on both the direct-write
and legacy JSON paths — regardless of whether a given event actually fell in
an N2 or an N3 epoch. A single-stage run (`stage=['NREM2']` or
`stage='NREM2'`) stores `'NREM2'`. The canonical token is built by
[`dbwrite.join_stage_token`](../reference/api/dbwrite.md); it sorts stages
into a fixed order (NREM1, NREM2, NREM3, REM, Wake, then anything else
alphabetically) before joining, so two callers requesting the same stage set
in a different order always produce the same token — `['NREM3', 'NREM2']`
and `['NREM2', 'NREM3']` both write `'NREM2NREM3'`.

Before 4.3, **all three detectors stored each event's own per-epoch stage on
the direct-write path** — `'NREM2'` on some rows, `'NREM3'` on others, for the
same N2+N3 run. Spindles and K-complexes were **no different from slow waves**
there: every direct-write database predating 4.3, including one written by
4.2, has per-epoch stages on every event type and needs migrating the same
way, spindles included.

The joined form only ever came from a *different* path: writing legacy
per-channel JSON (`write_db=False`), exporting it to CSV, and importing that
CSV back into the database. On that path, spindles and K-complexes stored the
run's whole *requested* stage list unconditionally in the JSON `stage` field,
which the CSV importer's `_norm_stage` then flattens into the joined token on
import — so a database built that way can already hold joined spindle/K-complex
tokens. Slow waves used the per-epoch stage on the JSON path too (falling back
to the requested list only when the epoch lookup failed), so a slow-wave CSV
import is per-epoch just like the direct-write case. What this change fixes is
the resulting three-way inconsistency — direct-write per-epoch, CSV-imported
spindles/K-complexes joined, CSV-imported slow waves per-epoch — replacing it
with one column, one meaning, on every path.

**A plain SQL match against a strict subset of a stored joint token matches
nothing, not a wrong number.** Once a run has stored `'NREM2NREM3'`, those
events cannot be attributed back to N2 or N3 individually — a raw
`WHERE stage = 'NREM2'` against those rows returns nothing, deliberately,
rather than a query someone has to remember divides part of a joint count by
an N2-only denominator. See
[`dbwrite.stage_tokens_covering`](../reference/api/dbwrite.md), the primitive
every reader uses to test whether a requested stage set is covered by a
stored token, for any query you write by hand.

[`density.event_density`](../reference/api/density.md) builds on this but is
not quite the same rule: asking it for `stage=['NREM2']` against an identity
whose **only** stored token is `'NREM2NREM3'` returns no row (same as the SQL
case above, with a warning explaining why). But if that identity holds
**both** a covering token (a per-epoch `'NREM2'` row, or an earlier run's
narrower joint token) **and** a non-covering one — which is exactly the state
a partially-migrated database is in, or a scope detected twice under
different stage sets — `event_density` returns the **narrower, covered
answer** instead of refusing, and warns naming how many events were excluded
and why. See
[`event_density` and a partially-covered identity](../reference/api/density.md#event_density-and-a-partially-covered-identity)
for the exact rule and an example.

If you need the N2-vs-N3 split back, use the additive **`events.epoch_stage`**
column instead — it holds each event's own scored epoch, computed for free at
detection time, and survives the join. A plain
`GROUP BY channel, epoch_stage` recovers the per-epoch breakdown that
`events.stage` no longer carries.

### A database refuses a re-detection that would duplicate events

`dbwrite.assert_stage_format_compatible` runs before any write on the
direct-write path, and refuses with a `ValueError` — rather than silently
duplicating a scope — in **two** situations. Both come down to the same root
cause: `event_uuid5` hashes the stage into every event's identity, and the
stage is also the last column of the `event_chan_time` UNIQUE constraint, so
a re-detection whose rows would carry a *different* stored `stage` value than
what's already there produces both a new uuid **and** a new unique key.
Neither the primary key nor the UNIQUE constraint then matches the existing
rows, so `INSERT OR REPLACE` **inserts** — the run appends a complete
duplicate set beside the old one, and every count and density in that scope
silently doubles.

**Check 1 — a pre-4.3 (per-epoch) database.** A database with no
`db_meta.stage_format` marker (every database written before 4.3) or one
marked `'per_epoch'` is treated as per-epoch. Re-detecting a scope such a
database already holds rows for is refused:

```text
ValueError: neural_events.db was written before the joint stage token
(db_meta.stage_format is None) and already holds 5760 spindle / method(s)
['Moelle2011'] / 11-16Hz row(s) across 64 channel(s), stored under stage
token(s) ['NREM2', 'NREM3']. Those rows carry uuids computed under the old
convention, so a re-detection cannot be proven to replace rather than
duplicate them. Convert the database first with
examples/migrate_stage_to_joint.py -- it rewrites the stored stages to the
run's token where needed and stamps db_meta.stage_format='joint' ...
```

**Check 2 — a joint database re-detected under a DIFFERENT stage set.** This
one is new, independent of check 1, and fires even on a database that is
already fully `'joint'` — the marker can't see it, because both the old rows
and the new run are "joint," just over different stage sets. Detect a scope
over `['NREM2']` first (stores `'NREM2'`), then re-detect the same channels
over `['NREM2', 'NREM3']` (would store `'NREM2NREM3'`), and this is what you
hit:

```text
ValueError: neural_events.db already holds 945 spindle / method(s)
['Moelle2011'] / 11-16Hz row(s) across 12 channel(s) under stage token(s)
['NREM2'], and this run would write 'NREM2NREM3'. The stage is part of both
the event uuid5 and the event_chan_time UNIQUE constraint, so those rows
would NOT be replaced: INSERT OR REPLACE would APPEND A COMPLETE DUPLICATE
SET beside them. Channels affected (first 10): ['E101', 'E102', ...]. To
proceed: re-detect with replace_channels=<the channels above> so the stale
rows are DELETEd in the same transaction; or delete that scope's rows first;
or write this run to a different database. Keeping both is not an option --
they overlap, so every count and density over this scope would be inflated.
(If the stored token is a per-epoch stage from a pre-4.3 run,
examples/migrate_stage_to_joint.py collapses it to the run's token instead.)
```

This is a natural thing to hit — widening or narrowing the stage set you
detect over, on channels you've already run, is an ordinary reason to
re-detect. It is not a sign anything is broken; it's the guard doing its job.

Ways past either refusal:

- **Migrate the database** with `examples/migrate_stage_to_joint.py` — see
  [Migrate a database to the joint stage token](migrate-stage-to-joint.md).
  Clears **check 1** by collapsing the stored per-epoch stages into the
  scope's own recorded stage set and stamping `db_meta.stage_format =
  'joint'` — it rewrites `events.stage` itself (the exact column check 2
  compares), so a **same**-stage-set re-run afterwards goes through cleanly.
  It does **not** clear check 2 for a genuinely *different* stage set,
  though: migrating doesn't change what any channel was actually searched
  over, only collapses the token that scope already recorded, so
  re-detecting under yet another stage set afterwards still raises check 2.
  Cheap, and preserves everything already in the database (`run_id`,
  provenance, cycles).
- **Re-run detection into a fresh database** — a brand-new `neural_events.db`
  has no rows yet, so `ensure_direct_write_schema` seeds
  `db_meta.stage_format = 'joint'` on it directly and neither check engages.
- **Re-run detection scoped to the affected channels with
  `replace_channels=<those channels>`.** Pass (at least) the same channels in
  `replace_channels` as you pass in `chan` for that call — this is what the
  error message itself names as the fix for check 2, and it also clears check
  1 the same way: the guard excludes any channel that is about to be replaced
  from its at-risk check, so if every channel you're about to write is also
  being replaced there is nothing left to check and it returns immediately.
  **`resume=False` does *not* clear either check** — `resume` only controls
  whether a channel with a prior `success=1` `processing_status` row is
  skipped; the stage-format check runs unconditionally before that, using the
  full `chan` list, so `resume=False` against an otherwise-unmigrated (or
  differently-scoped) database still raises.

Re-running (either of the last two) also re-computes `analysed_time` and
cycles for the scope from scratch — which the migration does not do on its
own unless you pass its backfill flags — but re-detects the events rather
than reusing them, so it costs whatever the original detection run cost.

If you just want to know which one you need: migrate if you have no scoring
file handy, or want to keep the exact events already stored (uuid, `run_id`
provenance) unchanged — but only when the problem is check 1 (a pre-4.3
database); re-run with `replace_channels` if the problem is check 2 (a
genuinely different stage set) or you'd rather regenerate `analysed_time` and
cycles as part of the same step, or the scope was never fully detected in the
first place.

### Cycles and stage durations populate automatically

Since 4.3, a detection run also fills `sleep_cycles`, `stage_durations` and
`events.cycle` itself — you no longer need a separate
[`finalize_cycles_and_durations`](detect-sleep-cycles.md) call just to get
`events.cycle` populated. Both cycle definitions (`'2022'` and `'1979'`) are
stored in `sleep_cycles`; `'2022'` owns `events.cycle`. This is a no-op on the
second and later detector run against a subject that already has cycles
stored, and any failure is caught and logged rather than losing a completed
detection — `events.cycle` just stays `NULL` for that run, recoverable later
with [`finalize_cycles_and_durations`](detect-sleep-cycles.md).

**A detection run never writes the annotation XML.** `write_xml=False` and
`plot=False` are passed unconditionally to the cycle finalizer and are not
exposed as arguments on `detect_spindles` / `detect_slow_waves` /
`detect_kcomplexes` — a detection run reads the scoring file, it does not
modify it. This is enforced by a regression test asserting the annotation
XML's mtime is byte-for-byte unchanged after a detection run.

## Opt out: the legacy JSON → CSV → import path

**Problem:** You need the per-channel JSON files — for a downstream tool that
still expects them, for archival, or to debug a single channel's raw detector
output — or you're running against a script that hasn't been updated yet and
you don't want its behaviour to change.

**Solution:** Pass `write_db=False` (or `--legacy-json` on the driver
scripts). This restores the pre-4.2 pipeline verbatim: one JSON file per
channel (including empty ones and error sentinels) is written to `json_dir`,
and nothing touches a database until you separately aggregate the JSON into a
CSV and import that CSV.

```python
spindles = event_processor.detect_spindles(
    method="Moelle2011",
    chan=["E110", "E111", "E112"],
    frequency=(11, 13),
    stage=["NREM2", "NREM3"],
    json_dir="wonambi/spindle_results",
    write_db=False,
)

event_processor.export_spindle_parameters_to_csv(
    json_input="wonambi/spindle_results",
    csv_file="wonambi/spindle_results/spindle_parameters.csv",
    file_pattern="spindles_Moelle2011_11-13Hz_NREM2NREM3",
)
event_processor.import_parameters_csv_to_database(
    csv_file="wonambi/spindle_results/spindle_parameters.csv",
    db_path="wonambi/neural_events.db",
    method="Moelle2011",
)
```

```bash
python examples/hdEEG_spindle_detector.py --legacy-json
```

A database written this way has no `analysed_time` table, so
`turtlewave_hdEEG.density.event_density` refuses to compute a density against
it — use the exporter's own `export_spindle_density_to_csv` /
`export_slow_wave_density_to_csv` / `export_kc_density_to_csv` instead (these
are deprecated but still work against a legacy JSON directory).

!!! warning "`--legacy-json` after a direct-write run on the same scope fails, on purpose"
    Since the database is now the default target, it's easy to end up running
    `--legacy-json` against the same `db_path`/scope (event type, method,
    band) that a previous default run already wrote directly. The CSV import
    step at the end of the legacy pipeline calls `guard_run_id`, which refuses
    to proceed when the scope already holds rows with a non-`NULL` `run_id` —
    importing would blank that provenance link with no other sign anything
    happened. The script exits with an uncaught `RuntimeError` (exit code 1),
    and the error message names `force=True` as the way past it if you really
    want to overwrite those rows and accept the lost `detection_runs` link.
    Re-running detection with the default (AUTO) path instead of
    `--legacy-json` is almost always what you want here.

`--write-db` is still accepted on the spindle/slow-wave/K-complex driver
scripts and the two `_GADI.py` cluster drivers — it's a true no-op on all
five: passing it or omitting it resolves to the identical AUTO behaviour,
because `write_db=True` and `write_db=None` behave identically for
`detect_spindles` / `detect_slow_waves` / `detect_kcomplexes`. Only the two
`_GADI.py` drivers actually print anything about it (a one-line warning that
it's a no-op); `examples/hdEEG_spindle_detector.py`, `hdEEG_sw_detector.py`
and `hdEEG_kcomplex_detector.py` register it with `help=SUPPRESS` and nothing
reads it, so passing it there is silent.

`analyze_pac` and its driver, `examples/hdEEG_pac_detector.py`, are
different: there is no `--write-db` flag — the PAC driver's opt-out flag is
`--no-write-db`, not the mirror-image of the other drivers'
`--legacy-json`. And passing `write_db=True` explicitly to `analyze_pac` is
**not** a no-op relative to the `write_db=None` default: on an unnameable
scope (continuous PAC with no `stored_event_type=`/`stored_method=`), AUTO
logs an error and skips the write, while explicit `write_db=True` raises
`ValueError` instead. See
[Write results straight to the database](run-pac-analysis.md#write-results-straight-to-the-database).

## Pull a CSV back out of the database

**Problem:** Your downstream stats pipeline expects a flat CSV, but events
are in the database.

**Solution:** Call `export_events_to_csv` for the scope you need. It writes
the SAME column layout as the legacy JSON → CSV exporters (plus the additional
`det_*` columns at the end), so the file round-trips back through
`import_parameters_csv_to_database` unchanged. If you want the *whole*
database as a data frame instead — the more common case for statistics — see
[Read the database with pandas and R](read-database-with-pandas-and-r.md).

```python
from turtlewave_hdEEG import export_events_to_csv

csv_path = export_events_to_csv(
    db_path="wonambi/neural_events.db",
    event_type="spindle",
    method="Moelle2011",
    frequency=(11, 13),
    stage=["NREM2", "NREM3"],
    output_dir="wonambi/spindle_results",
)
print(f"Wrote {csv_path}")
```

`csv_file=None` (the default) builds the standard filename with
`default_csv_path` — the same
`{event_type}_{method}_{freq_lo}-{freq_hi}Hz_{stages_joined}` convention used
everywhere else in the pipeline. If the scope genuinely has no matching
events, `export_events_to_csv` returns `None` and writes nothing; if the
`stage` filter excludes every row even though the same type/method/band DOES
have events under other stages, it raises `ValueError` instead of silently
writing an empty file, so a stage-token typo is caught immediately.

**Re-importing that CSV is guarded.** `import_parameters_csv_to_database`
does `INSERT OR REPLACE` on a deterministic event UUID with no `run_id` in
its column list, so importing a CSV over rows the direct-write path already
wrote would blank their `run_id` and sever them from their `detection_runs`
provenance — with no error, since the row count looks identical either way.
`import_parameters_csv_to_database` checks for this before writing and raises
`RuntimeError` if any in-scope rows carry a non-`NULL` run_id. Re-run
detection to update those rows properly, or pass `force=True` to proceed
anyway and accept the lost provenance link. It also now raises on a
missing/unreadable CSV or a bad database scope instead of returning
`{"error": ..., "added": 0}`, so a broken import can never be mistaken for a
clean, empty one. (`pac_coupling` has no `run_id` column and is keyed on its
own natural key instead, so this particular guard does not apply to
`import_pac_csv_to_database` / `backfill_pac_directory` — see
[Back-fill PAC results into the database](backfill-pac-to-database.md).)

`import_parameters_csv_to_database`, `import_pac_csv_to_database` and the
three `export_*_density_to_csv` methods are all deprecated (`5.0` removal) in
favour of the database being the store of record — each emits a
`DeprecationWarning` **and** a `logger.warning`, since deprecation warnings
alone are invisible to a script that isn't run under `-W`.

## See also

- [Read the database with pandas and R](read-database-with-pandas-and-r.md)
  — the replacement for what CSV used to do for downstream statistics.
- [Upgrade to 4.2](upgrade-to-4.2.md) — what changes for existing 4.0/4.1
  scripts and PBS jobs.
- [Re-run detection on reviewer-selected channels](rerun-detection-on-channels.md)
  — the direct-write path's `replace_channels` argument is what makes a scoped
  re-detection possible.
- [Migrate a database to the joint stage token](migrate-stage-to-joint.md) —
  clearing the pre-4.3 refusal without re-detecting.
- [Finalize sleep cycles & stage durations](detect-sleep-cycles.md) — the
  standalone call that a detection run now uses automatically.
- [Upgrade to 4.3](upgrade-to-4.3.md)
- [Explanation: Event density is artefact-free](../explanation/overview.md#event-density-is-artefact-free)
- [Reference: dbwrite module](../reference/api/dbwrite.md)
- [Reference: density module](../reference/api/density.md)
