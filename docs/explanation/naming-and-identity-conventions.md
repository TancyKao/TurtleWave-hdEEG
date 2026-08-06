# About Naming, Subject Identity, and Provenance Conventions

This page is for anyone extending `turtlewave_hdEEG` with a new detector,
exporter, or database writer — it explains the conventions every existing
`Paral*` processor follows, and why departing from them has caused real,
silent data loss.

## Background

Since 4.2, detection writes straight into `neural_events.db` by default
(`write_db=None` resolves to a database write) and no per-channel JSON is
produced. The failure mode this page documents belongs to the **legacy path**
— `write_db=False` / `--legacy-json` — where the pipeline still writes
results in three places that must independently agree on the same identity: a
JSON file per channel on disk, a row in `neural_events.db`, and a CSV in
between. Nothing enforces that agreement at write time — a detector writes a
filename, and some other piece of code, often written months later by someone
else, has to reconstruct the same string to find that file again, or to key a
database row for the same recording. Every "zero errors, wrote nothing"
symptom this project has hit so far traces back to two of those independently
constructed strings drifting apart: a frequency token rebuilt with a
different formatter than the one that named the file, and a subject id
resolved two different ways for two XML files in the same subject folder.

The `method_db`/`method_str` split and `derive_subject` below still apply on
the direct-write path too — filenames aren't involved, but the same canonical
vs. filesystem-safe distinction, and the same subject-resolution precedence,
key `events`, `pac_coupling` and `analysed_time`.

## The `{event_type}_{method}_{freq_lo}-{freq_hi}Hz_{stages_joined}` convention

Every legacy-path detector's JSON output, and the `file_pattern` used to find
it again downstream, follows this template. It's also the naming convention
`dbwrite.default_csv_path` reuses when you pull a CSV back out of the database
with `export_events_to_csv`. Two helpers exist specifically so no caller has
to reproduce it by hand:

- **`turtlewave_hdEEG.dbwrite.fmt_freq_token(lo, hi)`** is the single source
  of truth for the `{freq_lo}-{freq_hi}Hz` segment. Use it on *both* sides of
  the round-trip — where a detector names its output and where a caller
  rebuilds the pattern to find it. The bug this fixes was never about the
  format itself (it is deliberately the plain `f"{lo}-{hi}Hz"`, so existing
  result directories keep matching); it was a second formatter
  (`f"{lo:.1f}"`, which turns `1.25` into `1.2`) silently matching zero files
  on the cluster and reporting success anyway.
- **`turtlewave_hdEEG.utils.derive_subject`** is the single source of truth
  for the subject id that keys `sleep_cycles`, `stage_durations` and
  `pac_coupling`. See [its reference entry](../reference/api/utils.md) for
  the exact precedence order — the short version is: an explicit id wins,
  then a `sub-XXXX` token in the annotation XML filename, then the recording
  root directory's basename. There is deliberately no whole-filename-stem
  fallback, because a subject folder routinely holds more than one XML (a
  raw scoring file and a `*_review-qc.xml` sidecar, say), and keying on the
  stem split one recording's rows across two different subjects.

If you add a detector or exporter that builds either of these tokens, call
the shared helper. Do not re-derive the format locally, even if it looks
like a one-line `f-string` — that is exactly how both of the bugs above were
introduced.

## `method_db` vs. `method_str`: two spellings of a method, on purpose

A detection method's name is not always filesystem-safe.
`'AASM/Massimini2004'` is a real, valid method token (it names a hybrid
detector), but `/` is a path separator. Every event processor
(`ParalEvents`, `ParalSWA`, `ParalKC`) therefore carries **two** spellings of
the method through a detection run, and keeps them scoped to different
things:

- **`method_db`** — the canonical, *unescaped* string. This is what goes into
  every database column (`events.method`, `detection_runs.method`,
  `processing_status.method`) and what `dbwrite.method_citation` and
  `dbwrite.event_uuid5` are keyed on.
- **`method_str`** — `method_db.replace('/', '_')`, a filesystem-safe
  variant used **only** for JSON/CSV filenames and for constructing a
  `file_pattern`.

```python
method_db = "_".join(method) if isinstance(method, list) else str(method)
method_str = method_db.replace('/', '_')
```

The reason the two must not be conflated is the same class of bug as the
frequency token above, just on the method axis: a CSV importer that
underscore-splits a *filename* to recover the method (`AASM_Massimini2004`
→ `parts[2]` → `'AASM'`) silently truncates a hybrid method name, and if two
constituent methods share a filename token, their rows collide on the
`events` table's UNIQUE constraint and one is dropped without an error. This
is why `import_parameters_csv_to_database` accepts an explicit `method=`
override — pass the original, unescaped `method_db` string, never the
filename-derived one — and why `event_uuid5` and every `events` row store
each event's *own* per-event method rather than the run's joined method set.

If you write a new detector, keep this split: build `method_db` once at the
top of the detection method, derive `method_str` from it immediately after,
and never let a filename round-trip stand in for the canonical value in a
database write.

## Why this matters more than it looks like it should

None of these conventions change what gets detected — they only govern how
a detector's output is *found again* and *identified* afterwards. That makes
violations quiet: detection itself completes with zero errors, because
finding the wrong (empty) set of files, or writing under the wrong subject
key, is not an exception. It just produces an empty result, or two half-rows
where a whole subject's data should be — the kind of failure a batch driver
reports as "All done" for.

## Further reading

- [Reference: `turtlewave_hdEEG.dbwrite`](../reference/api/dbwrite.md) —
  `fmt_freq_token`, `event_uuid5`, `method_citation`, `guard_run_id`,
  `verify_channel_coverage`
- [Reference: `turtlewave_hdEEG.utils`](../reference/api/utils.md) —
  `derive_subject`, `missing_json_message`
- [How to write detection results directly to the database](../how-to/direct-to-database-detection.md)
  — where `method_db`/`method_str` and `fmt_freq_token` are used end to end
- [Explanation: Event density is artefact-free](overview.md#event-density-is-artefact-free)
  — another shared-helper convention in the same spirit
