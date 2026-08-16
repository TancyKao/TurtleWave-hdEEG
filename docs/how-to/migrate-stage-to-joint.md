# Migrate a Database to the Joint Stage Token

This guide shows you how to run `examples/migrate_stage_to_joint.py`, the
one-time migration that collapses a pre-4.3 database's per-epoch
`events.stage` values to the run's joint stage token, so a 4.3+ detector can
re-detect into it without refusing.

## When to use this

**Problem:** You try to re-detect an existing scope against a database
written before 4.3 (or with an unmarked `stage_format`), and the run raises:

```text
ValueError: neural_events.db was written before the joint stage token
(db_meta.stage_format is None) and already holds ... row(s) ... Convert the
database first with examples/migrate_stage_to_joint.py ...
```

This is [`dbwrite.assert_stage_format_compatible`](../reference/api/dbwrite.md)
refusing the write on purpose — see
[A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events)
for the full explanation. **This script fixes exactly one of the two things
that guard checks**: a pre-4.3 database whose rows still carry the per-epoch
stage. If you instead hit its *other* check — re-detecting a scope that's
already `'joint'` under a genuinely different stage set than it was written
with — this script has nothing to migrate; use `replace_channels` on the
detection call instead (see the section linked above). You have two ways past
the pre-4.3 case: migrate (this page), or re-run detection from scratch (see
[Cycles and stage durations populate automatically](direct-to-database-detection.md#cycles-and-stage-durations-populate-automatically)
and the comparison at the end of the section linked above). Migrate when you
want to keep the events already stored — their `uuid`, `run_id` provenance,
morphology columns — unchanged, and don't have (or don't want to re-read) the
scoring file.

**Solution:** Run the migration script against the database.

```bash
python examples/migrate_stage_to_joint.py /path/to/wonambi/neural_events.db
```

## Dry run first — this is the default

Without `--apply`, the script writes nothing. It reports what it would do,
per detection scope (`event_type` / `method` / frequency band), and — within
each scope — per **group of channels that share a target token**:

```text
INFO db_meta.stage_format = None
WARNING Scope slow_wave/Massimini2004/0.5-4.0Hz needs 2 DIFFERENT stage tokens ({'NREM2': 4, 'NREM2NREM3': 60}): its channels were not all searched over the same stage set. Each group is rewritten separately -- collapsing them to one token would divide the narrower group's events by the wider group's analysed time.
INFO /path/to/neural_events.db holds 12897 event row(s) across 2 detection scope(s):
INFO   [rewrite] slow_wave / Massimini2004 / 0.5-4.0Hz: 7137 row(s) {NREM2=945, NREM3=6192}
INFO       -> 'NREM2' for 4 channel(s) ['E101', 'E102', 'E103', 'E104'] (0 of 945 row(s) to change; target from processing_status (per channel))
INFO       -> 'NREM2NREM3' for 60 channel(s) ['E105', 'E106', 'E107', 'E108', 'E109', 'E110'] (6192 of 6192 row(s) to change; target from processing_status (per channel))
INFO   [noop] spindle / Moelle2011 / 11.0-16.0Hz: 5760 row(s) {NREM2NREM3=5760}
INFO       -> 'NREM2NREM3' for every channel (0 of 5760 row(s) to change; target from processing_status (per channel))
INFO DRY RUN. Would rewrite 6192 stage value(s) across 1 scope(s). Re-run with --apply to write; a backup is taken first at /path/to/neural_events.db.pre-joint-stage.bak.
INFO DRY RUN verdict: this would leave every scope re-detectable and would stamp db_meta.stage_format='joint'.
```

(Frequency bounds always print with a trailing `.0` here, even for a whole-number band — `freq_lower`/`freq_upper` come back from SQLite's `REAL` columns as Python floats regardless of what integer a caller originally passed to `detect_spindles`/`detect_slow_waves`. The "would rewrite" line and its "verdict" line are two separate `INFO` messages, not one combined sentence — the plan summary never mentions stamping, and the marker-stamping question always gets its own line.)

A scope whose rows already carry the joint token on every channel is reported
as `[noop]` and rewritten not at all — only `db_meta.stage_format` needs to
change for it. **This is not the common case, and don't assume it applies to
your spindles** — the spindle scope in the example above is a
CSV-imported one specifically: the 4.0.x legacy JSON → CSV → import route
stored spindle (and K-complex) events under the run's whole *requested*
stage list, which the importer's `_norm_stage` flattens into the same joined
form on the way in. A database populated the normal way since 4.2 — direct
write, which has been the default since then — has **per-epoch** stages on
every event type, spindles included, and every scope in it will show
`[rewrite]`, not `[noop]`. Check your own dry-run output rather than assuming
spindles are already fine.

The slow-wave scope above is where the grouping matters: 4 channels
(`E101`–`E104`) were detected over N2 alone, and the other 60 were detected
over N2+N3. Migrating this database with earlier code (before this planning
was per-channel) unioned the whole scope to `'NREM2NREM3'` and relabelled the
N2-only channels' events with it — silently and permanently understating
their density by dividing by N2+N3 analysed time instead of N2 alone. The
current script instead derives one target **per channel group**, rewrites
each group separately, and logs a `WARNING` naming every group's target and
size whenever a scope needs more than one token. If you see that warning,
check that the group sizes and targets match what you'd expect for that
scope — it means the scope genuinely was not searched consistently across
channels, not that anything is wrong.

Read the plan output before applying anything. The `target` for each group
comes from `processing_status`, per channel — the actual record of what stage
set that specific channel was searched over, not a scope-wide guess. If a
group's target looks wrong, pass `--stage-token` to force one token for the
**whole scope** (overriding all per-channel grouping — see below), or fix the
data some other way before migrating.

## Channels with no per-channel record

Two situations put a channel into a separate, explicitly-flagged fallback
group instead of a normal per-channel one:

- **The channel has no `processing_status` row for this scope at all** (a
  database whose `processing_status` predates the per-scope schema, or one
  built only from imported CSVs). These channels are collected into one
  fallback group, logged:
  ```text
  WARNING 3 channel(s) in slow_wave/Massimini2004 have no per-channel record
  of the stage set they were searched over (['E200', 'E201', 'E202']). Their
  target is derived from detection_runs. Check it, or pass --stage-token.
  ```
  Its target is derived from whatever scope-wide record exists
  (`detection_runs.stages`, falling back further to the union of stages these
  channels' own events actually hold if even that is absent) rather than a
  per-channel one — necessarily coarser, since there is nothing more precise
  on record.
- **The channel has MORE than one recorded stage set** for this exact scope
  (e.g. it was detected twice, at different times, under different stage
  scopes, both marked `success=1` in `processing_status`). No single token is
  correct for a channel like that, so it is logged and folded into the same
  fallback group rather than guessed at:
  ```text
  WARNING Channel E205 has 2 different recorded stage sets for
  slow_wave/Massimini2004 (['NREM2', 'NREM2NREM3']). No single token is
  correct for it, so it is planned from the stages its own events carry
  instead.
  ```
  A channel logged this way is then folded into the *first* fallback group
  above too (it now has no usable per-channel record either), so the
  "N channel(s) ... have no per-channel record" warning follows immediately
  after, naming the same channel and explaining where its target ultimately
  came from.

Either way, check the channels named in these warnings before applying —
they're the ones this script is least certain about.

## NULL-stage rows and when the marker is NOT stamped

Some rows have `stage IS NULL` — 4.2's direct-write path stores NULL whenever
no scored epoch contains an event, and the 4.0.x CSV importer stores NULL for
every row of a CSV with no `Stage` column. Both are ordinary, not corruption.

**These rows are rewritten by default, given their group's target token, the
same as any other row.** That is a deliberate reversal from how the script
first worked, and it is not inventing data: under the pre-4.3 convention
`events.stage` was the event's own scored epoch, so a NULL there meant "the
epoch could not be resolved," and writing a guess would have been a
confident, wrong label. Under 4.3, `events.stage` is the **run's stage
scope** instead — every row in the scope was genuinely detected within that
scope's fetched segments, whether or not its individual epoch lookup
succeeded — so the run's token is a correct label for a NULL row too, not a
guess. The per-epoch uncertainty a NULL used to carry isn't lost; it's
preserved in `events.epoch_stage`, which stays NULL. In the dry-run plan, a
group with NULL rows being rewritten shows it inline:

```text
INFO       -> 'NREM2NREM3' for 60 channel(s) ['E105', 'E106', 'E107', 'E108', 'E109', 'E110'] (6192 of 6192 row(s) to change; target from processing_status (per channel)), including 42 NULL-stage row(s) labelled with this token
```

**`--keep-null-stage` restores the old, conservative behaviour** — NULL rows
are left exactly as NULL, never labelled. Nothing is invented, but those rows
will keep refusing every future re-detection of their scope, because
`assert_stage_format_compatible` treats a NULL stage as "a different token
than what this run would write," the same as any other mismatched token. The
script reports this honestly rather than claiming success:

```text
ERROR WILL NOT be unblocked: slow_wave / Massimini2004 / 0.5-4.0Hz holds 3 row(s)
under ['NULL (no stage at all)'] on channel(s) ['E105', 'E106', 'E107'].
Re-detecting that scope will be REFUSED -- the duplicate guard treats these
as a different stage token. Resolve with --stage-token (when no stage set is
on record), by re-running without --keep-null-stage, by re-detecting those
channels with replace_channels=[...], or by deleting those rows.
ERROR DRY RUN verdict: this would NOT fully unblock the database -- 1
scope(s) above would still refuse re-detection, and db_meta.stage_format
would NOT be stamped.
```

(`channel(s)` here is the plain list, truncated silently to the first 10 —
the log never appends a `...` marker, so a database with more than 10
affected channels shows only the first 10 with nothing to indicate more
exist.)

**A NULL row is never given an invented token, either way.** A channel group
whose target can't be derived at all — no `processing_status` record, no
`detection_runs` row, and no non-NULL stage anywhere in the group to take a
union over — is left alone regardless of `--keep-null-stage`, reported as
having "NO TARGET could be derived," and needs `--stage-token` to resolve.

**The marker is stamped only when nothing is left blocked.** After writing,
the script re-checks every scope against what `assert_stage_format_compatible`
will actually see. If any scope still holds a NULL row (`--keep-null-stage`)
or a group with no derivable target, `db_meta.stage_format` is deliberately
**left unstamped** — even though the rewrite that *could* happen is committed
and kept. Stamping anyway would be worse than leaving it blank: it would
disable the guard for a database that is still genuinely part pre-4.3, while
the token check would keep refusing regardless, so the stamp would buy
nothing and destroy the one signal that a problem remains. This is what exit
code `3` means — see [Exit codes](#exit-codes).

**A run that ends up blocked with nothing to write skips the backup
entirely.** If nothing would actually change (every non-NULL row already
matches its target, and no back-fill is requested) but the database would
still be left blocked, the script refuses to write and exits `3` *before*
taking a backup — otherwise the corrective re-run it just told you to make
(typically with `--stage-token`) would immediately hit "a backup already
exists" and dead-end a second time.

**But `--keep-null-stage` no longer implies "nothing was written."** If
*anything else* changes in that run — other groups' non-NULL rows, or a
requested back-fill (`--annot` with `analysed_time`/`sleep_cycles` empty) —
real work happens: a backup **is** taken, the rewrite that could happen is
committed, and (since back-fills now run *before* the final verdict, not
after) any requested `analysed_time`/cycles back-fill is attempted and, if it
succeeds, committed too. The run can still end up returning `3` afterwards,
because the kept NULL rows are still blocked — but by then there is genuinely
something to protect, which is exactly why the backup exists. The log states
plainly what was committed and not undone, and — separately — what did not
finish, based on what actually happened rather than what was merely
requested (an earlier version of this message just said the back-fills "DID
run" whenever they'd been asked for, even on a run where both had thrown an
exception):

```text
ERROR NOT unblocked: slow_wave / Massimini2004 / 0.5-4.0Hz holds 1 row(s)
under ['NULL (no stage at all)'] on channel(s) ['E105']. Re-detecting that
scope is REFUSED -- the duplicate guard treats these as a different stage
token. Resolve with --stage-token (when no stage set is on record), by
re-running without --keep-null-stage, by re-detecting those channels with
replace_channels=[...], or by deleting those rows.
ERROR Requested back-fill did NOT complete -- analysed_time: FileNotFoundError:
[Errno 2] No such file or directory: '/path/to/sub-XXXX_scoring.xml'
ERROR Requested back-fill did NOT complete -- cycles: FileNotFoundError:
[Errno 2] No such file or directory: '/path/to/sub-XXXX_scoring.xml'
ERROR db_meta.stage_format was NOT stamped: 1 scope(s) above are still
blocked, so this database has not been migrated in full.
ERROR Committed and NOT undone: the stage rewrite (3 row(s)). NOT done: 1
blocked scope(s); analysed_time: FileNotFoundError: [Errno 2] No such file or
directory: '/path/to/sub-XXXX_scoring.xml'; cycles: FileNotFoundError: [Errno
2] No such file or directory: '/path/to/sub-XXXX_scoring.xml'. Re-run after
resolving the above (move or rename neural_events.db.pre-joint-stage.bak
first, or pass --backup-path, since a backup already exists).
```

A back-fill's outcome line names the exception type and message on a hard
failure (`FileNotFoundError` above). The two back-fills differ in whether
they can *also* fail without raising:

- **`analysed_time` can "run without raising but write nothing", and the
  script catches that case rather than calling it done.** Once
  `dbwrite.store_analysed_time` is reached, `strict=True` turns its failure
  modes into exceptions instead of empty, success-shaped returns: a
  computation error, and a denominator that comes out as zero seconds for
  every requested stage (an unreadable-but-parseable scoring), both raise
  `ValueError`. But the back-fill returns `{}` *before* that call when
  `events` holds no stage to compute against at all — the case where every
  row is NULL-staged and no target could be derived. The script treats that
  empty return as a failure, so the outcome is still one of: rows written,
  or a reason named in the log and a non-zero exit.
- **`cycles` can still "run without raising but write nothing."**
  `finalize_cycles_and_durations` has no equivalent strict mode here — an
  unreadable hypnogram can leave `stage_durations` empty without throwing, and
  the back-fill treats an empty `stage_durations` as the failure signal (a
  night can legitimately produce zero *cycles* and that's not a failure by
  itself, which is why cycle count isn't what's checked).

Either kind of failure — an exception or a reported "produced nothing" —
lands in the same `NOT done:` list, distinct from a still-blocked scope,
which is counted separately as `N blocked scope(s)`.

**A back-fill failure alone does not withhold the marker.** If no scope is
blocked but a requested back-fill failed, `db_meta.stage_format` **is**
stamped anyway — the stage migration itself genuinely succeeded, and
refusing to stamp it would leave you unable to re-detect for a reason that
has nothing to do with the stage rewrite. Only a still-blocked scope
withholds the marker; a failed back-fill on its own still returns `3` (see
[Exit codes](#exit-codes) — `0` means everything asked for succeeded, and a
failed back-fill is not "everything") but the database is otherwise usable
for detection again.

Either way — the pre-write shortcut above or this post-write case — a run
that did real work (stage rewrite and/or back-fills) but still returns `3`
leaves its backup behind on purpose, so a corrective re-run needs
`--backup-path` or the existing backup moved out of the way first: two
commands where, on a fully clean run, one would do. Read the log; it always
tells you when a backup already exists rather than failing with a bare
`FileExistsError`.

## Apply the migration

```bash
python examples/migrate_stage_to_joint.py \
    /path/to/wonambi/neural_events.db --apply \
    --annot /path/to/wonambi/sub-XXXX_scoring.xml
```

In order, this:

1. Refuses a path containing `archive` unless you pass `--allow-archive`
   (protects an archived, irreplaceable copy).
2. Runs `PRAGMA integrity_check` (skip with `--skip-integrity-check` on a
   large database on a slow network share).
3. Plans the rewrite per scope AND per channel group within each scope (as in
   the dry run above).
4. **Backs the database up** with `sqlite3.Connection.backup()` — WAL-safe,
   so a `-wal` sidecar with uncommitted-to-disk content is captured too — to
   `<db_path>.pre-joint-stage.bak` by default. **Refuses if that backup
   already exists**, so a second run can never overwrite the one good
   pre-migration copy with an already-migrated one. Pass `--backup-path` to
   write elsewhere.
5. Brings the schema current (`ensure_direct_write_schema`: adds
   `events.epoch_stage`, `db_meta`, the `v_event_density` view) before
   touching any row.
6. **Pre-checks for collisions, per group.** Within one group, six of the
   seven `event_chan_time` UNIQUE constraint components
   (`event_type, channel, start_time, method, freq_lower, freq_upper`) are
   already pinned by the group's own predicate, so once every row in the
   group carries the same target `stage`, the constraint reduces to **at most
   one row per `(channel, start_time)`**. The row set this checks is
   everything that will carry the target *after* the rewrite — the rows that
   move **and** any row already sitting on that exact token. Excluding the
   already-there row was the actual bug this replaced: it's precisely the row
   a moving row lands on top of, so a check that only looked at "rows about
   to change" could never see the collision it was about to cause, reported a
   clean dry run, and then died with an unhandled `IntegrityError` mid-write
   on `--apply`, after the backup was already taken. Two rows that are both
   already `NULL` at the same `(channel, start_time)` are legal today and
   **stay** legal — SQLite's UNIQUE index treats `NULL` as distinct from
   itself, so two NULL rows never collide — *unless* both are also being
   rewritten to the same non-NULL target (the default, since NULL rows move
   too), in which case that's exactly the same collision as any other pair
   and is caught the same way. Checked separately for each channel group —
   two channels being collapsed to *different* tokens can't collide with each
   other, so checking within the group is both sufficient and able to name
   which group a real collision belongs to. Structurally impossible from one
   consistent detection run, but possible in a database re-imported after
   re-scoring, or one that has accumulated rows from more than one run. If
   found, the script aborts **before writing anything** and lists up to 20
   colliding `(channel, start_time)` keys, naming the group's target.
7. Runs one `UPDATE` per channel group (not one per scope), all in **one
   transaction**, touching only the `stage` column. A group's `UPDATE` is
   scoped to exactly that group's channels via `channel IN (...)` (omitted
   when the group covers the whole scope), so two groups in the same scope
   are rewritten to different targets without touching each other's rows.
8. Asserts afterwards that the row count is unchanged and that every column
   except `stage` is byte-identical to before (an MD5 digest over a column
   list **pinned from the first call**, taken pre- and post-write — the
   second call reuses that exact list rather than re-deriving it, since
   step 5 adds new all-NULL columns that would otherwise make the digest
   compare different shapes and never match). If either check fails, it
   tells you to restore the backup and exits `2` — the database is left
   exactly as SQLite committed it.
9. Optionally back-fills `analysed_time` and `sleep_cycles` /
   `stage_durations` / `events.cycle` (see below). **This runs before the
   next step, on purpose.** A requested back-fill is work you asked for
   explicitly, and it's independent of the stage token — `analysed_time` is
   keyed per single scored stage, cycles are time windows, and neither cares
   whether the migration ends up leaving every scope re-detectable. It's also
   what makes the pre-write shortcut (skipping the backup entirely when
   nothing would change and every scope would stay blocked, described above)
   consistent: that shortcut only fires when no back-fill is requested
   either, precisely *because* a requested back-fill is real work that must
   still happen. Running the back-fills after step 10 instead would silently
   drop that work on exactly the runs where step 10 finds a blocker.
10. **Checks that the rewrite actually leaves every scope re-detectable, and
    stamps `db_meta.stage_format = 'joint'` only if it does — a decision
    based purely on whether any scope is still blocked, independent of
    whether step 9's back-fills succeeded.** A run where no scope is blocked
    but a back-fill failed still stamps the marker (the stage migration
    itself genuinely succeeded) and still returns exit code `3` (a requested
    back-fill did not complete, so it isn't "everything requested
    succeeded"). See
    [NULL-stage rows and when the marker is NOT stamped](#null-stage-rows-and-when-the-marker-is-not-stamped)
    below for the two ways a scope can still be left blocked, and
    [Exit codes](#exit-codes) for the full rc=3 rule (it covers a blocked
    scope and a failed back-fill without distinguishing them by exit code —
    the log does that). When the marker is stamped, that's what lets a 4.3+
    detector re-detect into the affected scopes without raising **check 1**
    of `assert_stage_format_compatible` — see
    [A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events).
    Migrating does not touch **check 2** (a scope re-detected under a
    different stage set than it was migrated to) — that's cleared with
    `replace_channels`, not this script.

## Back-filling density and cycles

A stage collapse by itself does nothing for density or cycles — those tables
are exactly as populated (or empty) after the migration as before it. Two
independent back-fills, **both default ON when their target table is
empty**, and both need `--annot <scoring.xml>`:

- `--backfill-analysed-time` / `--no-backfill-analysed-time` — recomputes the
  density denominator from the scoring, using the rejection settings
  (`reject_artifacts`/`reject_arousals`) recorded in `detection_runs` for
  each pair on record (falling back to the detector defaults, both `True`,
  with a warning, if none are recorded). Calls `dbwrite.store_analysed_time`
  with `strict=True` — unlike a detection run (which swallows a denominator
  problem so it never loses an otherwise-successful run), this back-fill's
  entire job is that write, so an unreadable scoring file **raises** instead
  of silently storing a zero-second denominator row. A zero-second row would
  have been worse than no row at all: `event_density` would have divided by
  it and silently reported every density in the scope as `NaN`, rather than
  the honest "denominator missing" a genuinely absent row produces.
- `--backfill-cycles` / `--no-backfill-cycles` — runs
  `finalize_cycles_and_durations` with `write_xml=False, plot=False` (the
  same contract a detection run itself follows — this script never modifies
  the rater's XML either) to fill `sleep_cycles`, `stage_durations` and
  `events.cycle`.

If neither table is empty (a database that already had density/cycles
before the collapse), neither back-fill runs unless you force it on.

**Missing `--annot` is handled two different ways, depending on whether you
asked for the back-fill explicitly.** They used to be handled the same way —
warn and silently skip both — which broke the "`0` means everything you
asked for succeeded" rule: a batch job whose `--annot` glob came back empty
would still mark the subject done with `analysed_time` still empty, and that
subject then silently dropped out of every density comparison for good.

- **You passed `--backfill-analysed-time` or `--backfill-cycles`
  explicitly, and `--annot` is missing.** This is a request that can never be
  honoured, so it's refused **before anything is written** — no backup, no
  connection opened for writing, database untouched — with exit code `1`:
  ```text
  ERROR --backfill-analysed-time was requested but --annot was not given, and
  a back-fill cannot run without the scoring file. Nothing has been written.
  Re-run with --annot <scoring.xml>, or drop the flag to migrate the stage
  tokens alone.
  ```
  Naming both flags if both were given (`--backfill-analysed-time,
  --backfill-cycles was requested but --annot was not given, ...`).
- **Neither flag was given, and a back-fill only defaulted ON because its
  table happened to be empty.** Nothing was explicitly asked for here, so the
  old behaviour is kept: the script logs a `WARNING` and proceeds without
  either back-fill, and a run that has no other blocker still returns `0`.
  ```text
  WARNING analysed_time is empty and sleep_cycles is empty, but --annot was
  not given, so neither can be back-filled. Density and cycles will stay
  unavailable. Re-run with --annot <scoring.xml> to fill them.
  ```

`--no-backfill-analysed-time` / `--no-backfill-cycles` sidestep both paths —
they're an explicit request *not* to back-fill, so a missing `--annot` never
comes up for the one you turned off.

Pass `--dataset /path/to/recording.set` alongside `--annot` if you want the
density back-fill to read `header['s_freq']` for a slightly more precise
segment floor; it's optional.

## Forcing the target token

By default the target token is derived **per channel group** within each
scope, from `processing_status` (falling back to `detection_runs`, and then
to the union of stages each channel's own events hold, when a channel has no
per-channel record). `--stage-token` overrides all of that: it forces **one**
token for the **entire scope**, collapsing every channel group into a single
group covering every channel — including channels that were genuinely
searched over a different, narrower stage set:

```bash
python examples/migrate_stage_to_joint.py neural_events.db --apply \
    --stage-token NREM2NREM3
```

Accepts `NREM2NREM3`, `NREM2,NREM3` or `NREM2 NREM3` — all parse to the same
canonical token, and an out-of-order spelling canonicalises rather than being
rejected: `--stage-token NREM3,NREM2` is accepted and logged

```text
INFO --stage-token 'NREM3,NREM2' canonicalised to 'NREM2NREM3' (the spelling a detector writes).
```

before it's used as the target. **Use this with the same caution the
per-channel grouping exists to avoid**: forcing `NREM2NREM3` onto a channel
that was actually detected over N2 alone reproduces exactly the
understated-density bug the per-channel grouping fixes, just deliberately
instead of by accident. Only reach for it when you're confident every channel
in that scope genuinely shares one stage set and the per-channel record is
what's wrong (e.g. a `processing_status` table that predates per-scope
tracking) — the per-channel default is what you want otherwise, including on
a database that has more than one channel group with different stage sets.

**`--stage-token` must spell out real stage names — it refuses anything else,
before the database is even opened for writing.** A value that doesn't
decompose into the known vocabulary (`NREM1`, `NREM2`, `NREM3`, `REM`,
`Wake`) is checked and rejected immediately, with exit code `1` and no
database touched — no backup taken, no connection opened, nothing written:

```text
ERROR --stage-token 'N2N3' is not a stage set this pipeline can produce
(Cannot split stage token 'N2N3' into known stages ['NREM1', 'NREM2',
'NREM3', 'Wake', 'REM']; pass an explicit list of stages instead.). A
detector writes tokens built from those stage names, so every row this run
wrote would still be refused by the duplicate guard -- while the marker
claimed the database was migrated. Spell the stages out, e.g. --stage-token
'NREM2,NREM3' or --stage-token NREM2NREM3.
```

This matters because `--stage-token` bypasses every other check that keeps
this script honest: it's the one path where the target isn't derived from
anything on record, so nothing else would have caught a typo like `N2N3`
(shorthand nobody's detector ever writes) or `NREM2NREM3x`. Before this
refusal existed, a bad token like that wrote its literal spelling into every
row, stamped `db_meta.stage_format = 'joint'`, and reported success — while
every real detector run against that scope kept raising, because nothing it
ever writes matches `'N2N3'`. `remaining_blockers` (the check behind the
normal exit-3 path) can't catch this case either: it compares each row
against the plan's *own* target, and here the plan's target *is* the bad
value, so the comparison always "passes." Spelling the stages out explicitly
— comma- or space-separated, or already joined in canonical order — is the
only form accepted.

## Reversal is the backup, and nothing else

There is no undo command. Collapsing `'NREM2'` and `'NREM3'` rows into
`'NREM2NREM3'` destroys the per-epoch distinction in `events.stage`;
recovering it means re-reading the hypnogram, which this script does not do
and structurally cannot do from the database alone. If a migrated result
looks wrong, restore the pre-migration backup file
(`<db_path>.pre-joint-stage.bak` by default) over the database — that backup,
taken before any write, is the only way back.

One partial exception: rows a 4.3 detector wrote (as opposed to a pre-4.3
one) already carry the per-epoch stage in the additive `events.epoch_stage`
column, so for those specifically the split survives the collapse even
without the backup. A pre-4.3 database being migrated for the first time has
no such column populated yet, which is exactly the case this script exists
for.

## Already-joint databases are a formality, not a blocker

A scope whose rows already carry the joint token needs no row rewritten — the
script detects this (`[noop]` in the plan output) and only stamps
`db_meta.stage_format = 'joint'`. This happens specifically for a spindle or
K-complex scope built via the legacy JSON → CSV → import route, where only
the marker was ever missing — **not** for the ordinary direct-write database
(the default since 4.2), where every event type, spindles and K-complexes
included, is per-epoch and needs an actual rewrite. Don't infer from one
`[noop]` scope that the rest of the database is also already joint — read
each scope's own status. Running the migration against a database where
every scope happens to already be a no-op still unblocks re-detection — it's
cheap to run even when you're not sure whether it's needed.

## Exit codes

**The governing rule: `0` means, and only means, everything you asked for
succeeded *and* the database is fully re-detectable.** Nothing less returns
`0` — not a stage rewrite that left one scope blocked, not a stage rewrite
that succeeded while a requested back-fill failed. This is deliberate and
aimed at batch use: a driver script over many subjects needs exactly one
rule it cannot get wrong when deciding whether a subject is done, and "check
the exit code" has to be sufficient without also parsing the log.

| Code | Meaning |
|---|---|
| `0` | Everything requested succeeded: every scope is (or, on a dry run, would be) fully re-detectable, and every requested back-fill completed. The marker is (or would be) stamped. Includes "nothing to do." |
| `1` | Refused before writing anything (archive path, an out-of-vocabulary `--stage-token`, an explicit `--backfill-analysed-time`/`--backfill-cycles` given without `--annot`, integrity check failed, no `events` table, collision found, backup already exists) |
| `2` | A post-write assertion failed (row count or non-`stage` column changed) — the database is left as SQLite committed it; restore the backup |
| `3` | The rewrite is committed but the job is not finished — for **either or both** of two reasons: at least one scope still refuses re-detection (a NULL row kept by `--keep-null-stage`, or a channel group with no derivable target), or a requested back-fill did not complete. **Deliberately not split into two codes**: both demand the identical response (read the log, fix, re-run), and collapsing them into one code is what makes the rule above true and simple. The log — not the exit code — distinguishes which happened: it names every still-blocked scope and channel, names each back-fill that failed with its exception type and message (either back-fill can also report "produced nothing" without an exception — see [NULL-stage rows and when the marker is NOT stamped](#null-stage-rows-and-when-the-marker-is-not-stamped)), and states exactly what was committed and NOT undone. `db_meta.stage_format` is stamped only if **no scope is blocked** — a back-fill failure on its own does not withhold it, since the stage migration itself genuinely succeeded. See [NULL-stage rows and when the marker is NOT stamped](#null-stage-rows-and-when-the-marker-is-not-stamped). |

## See also

- [What `events.stage` means](direct-to-database-detection.md#what-eventsstage-means)
- [A database refuses a re-detection that would duplicate events](direct-to-database-detection.md#a-database-refuses-a-re-detection-that-would-duplicate-events) —
  including the check this script does NOT clear.
- [`event_density` and a partially-covered identity](../reference/api/density.md#event_density-and-a-partially-covered-identity) —
  what density reports while a database is only partly migrated.
- [Upgrade to 4.3](upgrade-to-4.3.md)
- [Reference: dbwrite module](../reference/api/dbwrite.md)
- [Finalize sleep cycles & stage durations](detect-sleep-cycles.md)
