# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.3.0] — Unreleased

`events.stage` now records the stage set the run searched, as one joint token:
a run over N2 and N3 stores `NREM2NREM3` on every event, written identically by
all three detectors on both the database and the legacy paths. Two spellings
used to coexist: detection stored each event's own epoch stage, while the 4.0.x
CSV import route stored the joined request for spindles and the epoch stage for
slow waves. A database carrying both blocked the PAC tab outright. A detection
run now also fills sleep cycles and stage durations, which no run had ever done.
A database written before 4.3 must be migrated before it is re-detected — see
**Upgrading**.

The Massimini slow-wave and K-complex detectors also now implement the criteria
they document. The published duration and depth limits were applied to the wrong
half-wave, and the AASM window rejected every wave longer than a second, so those
methods returned few events or none. Slow-wave and K-complex counts change
substantially and must not be pooled across this release; Staresina2015 and
Ngo2015 are unaffected. Reference: Massimini et al. 2004, J Neurosci 24(31),
6862-70.

### Added

- `events.epoch_stage`, keeping each event's own scored epoch stage beside the run's token.
- Public API: `join_stage_token`, `split_stage_token`, `stage_components`, `stage_tokens_covering`, `resolve_stage_tokens`, `pooled_denominator`, `stage_format` and `assert_stage_format_compatible`, whose `stage_token` is keyword-only and has no default so the guard cannot be skipped by omission.
- `db_meta` table, carrying a `stage_format` marker that distinguishes a joint-token database from a pre-4.3 one.
- `idx_events_run` index on `events(run_id)`, so tagging a run's cycles visits only that run's rows instead of every run's events in the same time span.
- `db_meta.det_ptp_units`, recording whether `det_ptp` holds microvolts or a pre-4.3 sample count, seeded only on a database with no events because the two ranges overlap numerically.
- `v_event_density` SQL view, so R and plain SQL read the same densities `event_density` computes.
- `examples/migrate_stage_to_joint.py`: converts a pre-4.3 database and stamps the marker, dry-run by default, deriving the target token per channel group so channels searched over different stage sets are not collapsed into one.
- Detection fills `sleep_cycles`, `stage_durations` and `events.cycle` on its own connection, without touching the annotation XML.
- `run_id` and `tag_events` on `finalize_cycles_and_durations`, `ParalCycles.run` and `tag_events_with_cycles`, and `conn` on `finalize_cycles_and_durations` (the other two gained it in 4.0.2), so a detection shares one connection and tags only its own rows.
- `strict` on `store_analysed_time`, raising and rolling the write back when the denominator comes out as zero seconds for every stage rather than storing one that turns every density in the scope into an unexplained NaN. Detection keeps the tolerant default; the migration passes it.

### Changed

- `events.stage` holds the run's canonical stage token instead of the per-epoch stage or the requested list, in the database, the JSON and the exported filenames.
- Only `tag_method` writes `events.cycle`; every method used to tag it, last run winning.
- `finalize_cycles_and_durations` raises when `tag_method` is not one of `methods`, which previously wrote no XML markers at all and silently let the last method own the numbering.
- Detection stops when a requested stage has no scored epoch, instead of labelling every event with a stage the recording does not contain.
- Re-detecting a scope over a different stage set raises instead of writing, because the stage keys the stored rows and the new ones would be appended beside them rather than replacing them. Pass `replace_channels` for the affected channels to delete the old rows in the same transaction.
- `event_density` reports one row per stage token, and refuses a request for a strict subset of a stored joint token rather than returning a number it cannot attribute.
- The deprecated per-channel JSON carries the run's stage token plus a new `epoch_stage` field; the deprecated density exporters split the token, so their per-stage numbers are unchanged.
- The PAC channel lookup intersects two per-side queries instead of self-joining on stage, so a genuine stage mismatch is reported and confirmable rather than fatal.
- `AASM/Massimini2004` uses Wonambi's published -40 µV and 75 µV thresholds, replacing the -37 / 70 pair, which matches no published criterion.
- K-complex isolation is measured between successive negative peaks rather than positive ones.
- K-complex detection is turned off in the GUI for this release. Stored K-complexes stay reviewable and exportable, and `ParalKC.detect_kcomplexes` still runs, but what it returns moves with the two changes above.
- The Staresina2015 peak-to-peak control is labelled as the percentile it always was and made read-only, since nothing the GUI could send ever reached it. The values passed to the library are identical, so Staresina yields do not move.

### Fixed

- PAC was unusable on a database built by the 4.0.x CSV import route, where spindles carry the joined stage request and slow waves the per-epoch stage: the channel lookup joined the two on `stage`, which those spellings can never satisfy, so the channel list came back empty and every run failed with "No channels selected".
- Detection to the database was drastically slower than it needed to be, because each detected event's measurement window was cut from the whole-night segment with a scan whose cost grew with the length of the recording. A five-channel slow-wave run that took over an hour now takes seconds. The events and their measurements are unchanged, so nothing needs re-running for correctness. This affects 4.2.0, which made the database the default path.
- Ngo2015 slow-wave detection with adaptive thresholds failed on every channel with a `TypeError` and finished the run with zero events, in every released version. The sigma thresholds are read off the detector instance but were passed to its constructor, and the GUI prefills both fields, so the failure did not depend on typing a value.
- `trough_duration` now bounds the negative half-wave, as Massimini et al. 2004 define it, instead of being passed as the whole-wave duration. The published AASM window rejected every wave running past one second, which is most of them, so those runs returned no events at all.
- `neg_peak_thresh` and `p2p_thresh` now reach the detector; they were stored under attribute names Wonambi never reads, so neither threshold had any effect.
- The depth criterion is applied to the negative trough rather than the positive peak, so an accepted wave really is as deep as the threshold says.
- Candidates are no longer pre-rejected on the duration or amplitude of their positive half-wave, which the paper does not constrain. Against 216 slow waves injected into a synthetic 1/f background, recall rises from 0.025 to 0.525 for Massimini2004 and from 0.046 to 0.764 for AASM/Massimini2004, with precision holding above 0.98; re-run it as `test_permissive_search_recall_against_injected_ground_truth`. That ground truth is synthetic rather than expert-scored, so it shows recovery at high precision and is not a clinical validation.
- `det_ptp` holds peak-to-peak microvolts for all four methods, replacing a sample count that scaled with sampling rate and ignored amplitude despite the column being named for microvolts.
- `det_trough` and `det_peak` mean the same thing across methods: the negative extremum and the positive one. They were opposite between the Massimini family and the zero-crossing methods, so comparing them across methods compared opposite quantities.
- The GUI's duration spin boxes keep 3 decimals, so an Ngo2015 run with nothing typed no longer sends 0.83 s in place of the method's 0.833 s default and detects on a slightly shifted band.
- `sleep_cycles`, `stage_durations` and `events.cycle` were never populated by a detection run; only the standalone cycle script filled them.
- The GUI's post-run verification counted events by single stage, so a successful run would have reported "0 events written" once joint tokens landed.
- The review GUI's QC density and the legacy JSON exporters treated a joint token as a stage of its own, which matches no denominator and reads as zero everywhere.

### Upgrading

- Slow-wave and K-complex counts and densities change substantially, so rows detected before and after 4.3 must never be pooled and any comparison spanning the two is invalid. Re-detect rather than mix.
- Check a first production run before trusting it: N3 slow-wave density should land at roughly 5-15/min. A result far outside that means the run is misconfigured, not that the detector changed.
- Staresina2015 and Ngo2015 are unaffected: the same events, the same counts and densities, and every morphology column identical except `det_ptp`, which changes from a sample count to microvolts. That holds for scripted runs; a Ngo2015 run started from the GUI now filters on a band 0.4% lower, because the duration box can finally hold the method's own 0.833 s default, and no stored data is affected because GUI Ngo2015 runs raised `TypeError` and wrote nothing before 4.3.
- `det_ptp` written before 4.3 is a sample count rather than microvolts, and `db_meta.det_ptp_units` records which a database holds. `peak2peak_amp` was and remains microvolts and is unaffected.
- Detection refuses a pre-4.3 database that already holds rows for the scope it is about to write. `event_uuid5` hashes the stage, so the new token changes every event's identity and `INSERT OR REPLACE` appends a duplicate set instead of replacing it, doubling every count and density in that scope with no error. Run `examples/migrate_stage_to_joint.py`, or detect into a fresh database.
- Every row written by detection before 4.3 carries a per-epoch stage, spindles included, so a 4.1 or 4.2 database needs migrating in full. Only a database imported from CSV by 4.0.x already carries joint spindle tokens, and there the migration just stamps the marker.
- The migration reads `processing_status` to find which channels were searched over which stages, and warns when one scope needs more than one token. Check that warning before applying: relabelling a channel searched over N2 alone with a wider token divides its events by a larger denominator from then on.
- The migration relabels NULL-stage rows with their run's token, which is not invented data: `events.stage` used to be the event's own epoch and could be unresolved, and is now the run's stage scope, which is known for every row in it. The per-epoch uncertainty stays NULL in `events.epoch_stage`; `--keep-null-stage` restores the old behaviour.
- The migration exits 3 and leaves the marker unstamped when it could not unblock every scope, rather than reporting success while the surviving rows keep refusing re-detection. Treat any non-zero exit as failure, not only 1.
- The migration's backup is the only way back. A joint token cannot be un-collapsed without re-reading the hypnogram, so keep that file until you have checked the result.
- `processing_status` rows written by earlier versions are not in canonical stage order, so a resume re-detects those channels instead of skipping them, and coverage verification warns about channels that are not missing. Both are harmless and clear after one 4.3 run.
- `event_density` with `stage=['NREM2']` against a run detected over N2 and N3 returns no row instead of a wrong one. Use `events.epoch_stage` when you need the split.
- Schema changes are additive: `events.epoch_stage`, `db_meta` and the `v_event_density` view. No column is dropped, renamed or retyped.

## [4.2.0] — Unreleased

Makes `neural_events.db` the single store of record. Detection writes to the
database by default and no longer writes per-channel JSON, and the
export-import-density file round-trip is gone from the drivers, the example
scripts and the GUI. That round-trip is what lost events silently: detection
wrote files and a separately built string had to find them again. Density is now
derived from the database, which also closes a gap where `--write-db` runs
produced no density at all. `--legacy-json` keeps the old path — see **Upgrading**.

### Added

- `density.event_density`, deriving per-channel density from the database, and `density.format_density_table`.
- `analysed_time` table holding the artefact-free density denominator, written at detection time.
- Public API: `resolve_db_target`, `recording_root_from_db`, `ensure_analysed_time_schema`, `record_analysed_time`, `store_analysed_time`, `read_analysed_time`.
- `--legacy-json` on the detection drivers, and `--subject` on the three example detectors.
- `normalize_subject`, giving one canonical `sub-` subject form, and `subjects_in_database` / `assert_single_subject`, which refuse a write that would mix subjects in one database.
- `subject` column on `detection_runs`.
- `include_zero_channels` on `event_density`, on by default, so a channel that detected nothing appears as a zero rather than vanishing from a montage summary.
- `write_csv` on `analyze_pac`, gating the per-channel PAC parameter CSVs.
- An Export CSV button on each GUI detection tab, writing events and density for a chosen scope.
- Docs: [Read the database with pandas and R](docs/how-to/read-database-with-pandas-and-r.md) and [Upgrade to 4.2](docs/how-to/upgrade-to-4.2.md).

### Changed

- `write_db` defaults to `None`, meaning write to the database; `write_db=False` keeps the JSON and CSV path.
- Detection no longer writes per-channel JSON unless `write_db=False`. Crash-resume comes from `processing_status`, which records per-channel failures with reasons.
- `resolve_db_target` raises when a database write was asked for and no path resolves, instead of silently continuing without one.
- The GUI reports a run by reading the database back, and names the database path and the event and channel counts on completion.
- `analyze_pac` writes per-channel CSVs only when `write_csv=True` or `write_db=False`. The `*_mean_amps.npy` modulogram is always written; it is not reconstructable from `pac_coupling`.
- The three `export_*_density_to_csv` methods and the CSV import route are deprecated and warn through both `DeprecationWarning` and the logger.

### Removed

- The `create_empty_json` parameter on `detect_spindles`, `detect_slow_waves` and `detect_kcomplexes`.
- `--write-db` on the drivers, now the default. The flag is still accepted and does nothing; the two cluster drivers warn when it is passed.

### Fixed

- The review GUI's QC density assumed both artefact and arousal rejection regardless of what the run used, inflating density by 8% on a run detected with arousal rejection off. It now reads the run's own settings from `detection_runs`.
- A `--write-db` run produced no density at all, because the drivers returned before the density export.
- Detection continued after failing to read the sleep scoring, stamping every event with no stage; the run then read as an empty night while the database held it in full. It now stops and says so.
- Density pooled across stages divided by only the stages that produced events, over-estimating whenever a requested stage detected nothing.
- A joined stage token such as `NREM2NREM3` returned an empty density frame that read as "nothing detected".
- The drivers and the GUI disagreed on a subject's name, which put two denominators in one database and then failed every density query that did not name a subject.

### Upgrading

- Detection now writes to `neural_events.db` and no JSON. Pass `--legacy-json`, or `write_db=False`, to keep the old output.
- Scripts passing `create_empty_json` raise `TypeError`. Remove the argument.
- A database built by `--legacy-json` has no `analysed_time` table, so `event_density` refuses to compute on it rather than substituting raw hypnogram time. Use the legacy density CSV for those runs.
- Density read from the QC dashboard before this release was too high for any run detected with arousal rejection off, which is the default on the spindle drivers. Re-read affected numbers.
- Subject identifiers are normalised to the `sub-` form, so `--subject 10sd` and `--subject sub-10sd` now name one subject. The cycle and stage-duration writers stored whatever they were handed before this release, so a database written by an earlier version can hold the other spelling; re-running cycles rewrites those rows rather than adding a second set, and says how many it replaced.
- Detection refuses to write into a database that already holds another subject. A database written before 4.2 records no subject, so the first run claims it and says so in a warning; from then on a second subject is refused. One database per subject remains the expected layout.
- Detection now stops when the sleep scoring cannot be read, where it previously carried on and produced events with no stage.
- Schema changes are additive. An existing database gains an `analysed_time` table and a `subject` column on `detection_runs`; no column is dropped, renamed or retyped. The first 4.2 run fills that column in for rows that have none.

## [4.0.2] — 2026-08-16

Fixes `disk I/O error` when `neural_events.db` lives on a mapped network drive
or a synced folder. The databases were permanently in SQLite WAL journal mode,
which cannot work over a network filesystem, and every connection reimposed WAL
without checking whether it had been applied. The package now leaves an existing
database's journal mode alone, so converting a database once is enough — see
**Upgrading**.

### Added

- `TURTLEWAVE_SQLITE_JOURNAL` environment variable imposing a SQLite journal mode on every database this package opens, overriding the preserve rule in both directions; a blank value counts as unset, so a job template exporting an unset variable is harmless.
- Public API: `set_journal_mode`, converting an existing database to another journal mode, and `VALID_JOURNAL_MODES`, the accepted mode names.
- `turtlewave_set_journal_mode`: console script converting existing databases to another journal mode, reachable from a pip install where `examples/` is not. A directory argument converts `neural_events.db` files only; `--glob` reaches any other name.
- `examples/set_db_journal_mode.py`: converts one database or a whole `ROOT/*/wonambi/neural_events.db` tree.
- Docs: [Run with a database on a network drive](docs/how-to/run-with-database-on-a-network-drive.md) and [Database concurrency and journalling](docs/explanation/database-concurrency-and-journalling.md).

### Changed

- The package no longer overrides a journal mode you chose deliberately: `open_write_connection` preserves an existing database's mode and logs it, and picks a mode only for a database it creates. A database created before 4.0.2 is already WAL and stays WAL, wherever it lives.
- A database this package creates now starts in `DELETE` journal mode instead of `WAL`, so a database created straight onto a network drive or a synced folder works from its first write; `journal=` and `TURTLEWAVE_SQLITE_JOURNAL` still override it.
- `finalize_cycles_and_durations` and `ParalCycles.run` raise `FileNotFoundError` when the database does not exist instead of creating one; they annotate an existing `neural_events.db` and a missing file means a wrong path or detection that never ran.
- `open_write_connection` gains optional `journal` and `logger` arguments; existing calls are unaffected.
- `store_cycles_to_database`, `store_stage_durations`, `tag_events_with_cycles` and `ParalCycles.run` gain an optional trailing `conn` argument for sharing one connection.
- A `disk I/O error` from `open_write_connection` or `set_journal_mode` now names the database, the WAL-on-a-network-drive cause, `set_journal_mode` and the how-to page. Same exception class, with the original attached as `__cause__`.

### Fixed

- A detection run or a review GUI silently converted a database back to WAL, undoing a conversion made to keep it working on a network drive.
- The review GUIs no longer force `journal_mode=WAL` and `mmap_size`, both unusable over a network filesystem, and the detection GUI's connections now set a lock timeout. Outside WAL they set `synchronous=FULL`, since `NORMAL` is only corruption-safe under WAL.
- `open_write_connection` now checks that an imposed journal mode was actually applied and warns when it was not, instead of assuming the pragma took effect.
- The CSV importers, the PAC writers and several readers wait up to 60 seconds for a lock instead of five, so a reader and a writer on a `DELETE`-mode database no longer collide.
- A mistyped `db_path` in the cycle backfill wrote a stray empty database and then failed with `no such table: main.events`; it now fails immediately naming the path.
- `finalize_cycles_and_durations` uses one timed connection per subject instead of six untimed connect/close cycles, each of which recreated the WAL sidecar files.
- `import frontend` no longer raises `ImportError` without PyQt5 installed; the main GUI import is guarded like the review GUI's already was.

### Upgrading

- Journal mode is a persistent property of the database file, so a database created before 4.0.2 is in WAL until you convert it. Close every GUI and convert once with `turtlewave_set_journal_mode`; from 4.0.2 the choice sticks, and later runs will not switch it back. See [Run with a database on a network drive](docs/how-to/run-with-database-on-a-network-drive.md).
- New databases need no conversion, so `TURTLEWAVE_SQLITE_JOURNAL` is now only for opting *into* WAL, or for forcing `DELETE` on machines still running an older version.
- On Windows, set that variable with `setx TURTLEWAVE_SQLITE_JOURNAL DELETE` so it persists; a `$env:` assignment lives only until the shell closes.
- A cycle backfill pointed at a database that does not exist now raises `FileNotFoundError` instead of creating an empty one. Any script relying on that database being created must run detection first.
- Databases on local disk need no action, but a database created from 4.0.2 onward is in `DELETE` mode and loses WAL's concurrent reads: a review GUI opened during a detection run waits on the 60-second lock timeout instead of reading straight away. Set `TURTLEWAVE_SQLITE_JOURNAL=WAL` to keep the old behaviour.

## [4.0.1] — 2026-08-05

Closes a class of silent data loss where detection wrote files and a separately
constructed string had to find them again. Affected runs completed without
errors and wrote nothing to `neural_events.db`. Two density fixes change
exported numbers for some recordings — see **Upgrading** for who is affected.

### Added

- Public API: `derive_subject`, `fmt_freq_token`, `ensure_pac_schema`, `guard_run_id`, `verify_channel_coverage`, and `stored_event_type` / `stored_method` on `analyze_pac`.

### Changed

- Both HPC batch drivers verify channel coverage against the database and exit non-zero when channels are missing, instead of always logging success.
- Quieter, more accurate logs: dataset-loader diagnostics and per-file bookkeeping dropped to debug level, and the GUI log pane no longer double-timestamps library lines, repeats the same warning, or reports a CSV as saved when none was written.
- Two failure paths that were logged at info level are now logged as errors.

### Fixed

- PAC results now reach `neural_events.db`; the GUI and `examples/hdEEG_pac_detector.py` never requested a database write, so runs produced CSVs only and created no `pac_coupling` table.
- The cluster spindle driver built its frequency-band filename token with one-decimal formatting while detection wrote it unformatted, so any band bound needing two decimals matched zero files, imported nothing, and still logged success; all band tokens now come from one shared function.
- Detection methods containing a slash, such as `AASM/Massimini2004`, were truncated to `AASM` in `events.method`.
- A multi-method spindle run stamped every row with a single method, so the uniqueness constraint silently discarded the other method's events.
- Failed CSV imports returned "0 added" and were indistinguishable from a clean re-run; importers now raise, and parameter exporters raise when their pattern matches no files instead of writing a placeholder CSV.
- Continuous PAC was storable as slow-wave coupling; `analyze_pac` now refuses to write a row whose scope it cannot name.
- A subject identifier could differ depending on which annotation XML a caller pointed at, splitting one recording across two database keys.
- Density denominators no longer subtract time for epochs scored past the annotation's `last_second`; those epochs contributed negative durations that inflated density, and in one recording drove the denominator negative and reported negative densities without an error. The inconsistency is now logged once per stage at WARNING.
- The example scripts and both cluster drivers pass the run's real `reject_artifacts` / `reject_arousals` settings to the density export instead of assuming both were enabled, so a run with `reject_arousals=False` no longer has arousal time subtracted from its denominator.
- A detection run in which every channel succeeded and found zero events is now a clean no-op: the parameter exporter writes a header-only CSV, as the density exporter already did, and the importer records a successful import of nothing, instead of writing no file and failing the import with `FileNotFoundError`.
- A run in which no channel produced an event and at least one channel failed no longer looks like an empty night: no CSV is written and the import fails loudly, instead of a header-only CSV importing as a clean zero-event result.
- Channels that fail are now named at ERROR with their reason. A partially failed run still exports the channels that succeeded, so the presence of a parameters CSV does not mean every channel was detected — check the log before treating a montage as complete.

### Upgrading

- PAC results from 4.0.0 exist as CSVs only; back-fill or re-run them to populate `pac_coupling`.
- The first coverage check against a database created before 4.0.1 reports a large `events_only` warning, because legacy `processing_status` rows migrate in with an empty method, zero frequency bounds and an empty stage and never match the scoped query. It exits 0 and self-heals after one scoped run.
- Re-export densities for any run that used `reject_artifacts=False` or `reject_arousals=False`. Those denominators were computed as if both were True, so they subtracted time the detector never excluded. Runs that left both at their default True are unaffected by this change.
- Re-export densities for any recording whose annotation scores epochs beyond `last_second`; 4.0.1 logs a WARNING per stage when it finds this, so re-running the export tells you whether you were affected. Recordings whose epochs all fit inside `last_second` are unchanged. Where such epochs exist, the old error depended on whether the file happened to contain artefact or arousal events to reject: on one subject, `reject_arousals=False` densities moved by NREM2 −2.80%, Wake −5.03% and NREM1 −1.53%, while the same subject with both flags True was unchanged.

## [4.0.0] — 2026-07-22

The review GUI drops per-event triage: the Events tab is gone and review now
works at channel granularity. Event density is computed against artefact-free
analysed time, and PAC preferred phase was 180° off, so densities and PAC
phases from earlier releases must be regenerated. See
[Upgrade to 4.0](docs/how-to/upgrade-to-4.0.md).

### Added

- `ParalCycles` / `detect_cycles`: rule-based NREM-REM sleep-cycle detection from the hypnogram, supporting the NREM-based `'2022'` and Feinberg `'1979'` definitions.
- `neural_events.db`: new `sleep_cycles` table and an `idx_cycle` index on `events(cycle)`.
- `compute_stage_durations`: per-stage sleep durations from the hypnogram.
- `neural_events.db`: new `stage_durations` table, created at database initialization and written by `ParalCycles.run()`.
- `finalize_cycles_and_durations`: single post-detection step filling sleep cycles under both definitions plus stage durations, and tagging events by cycle.
- `turtlewave_hdEEG.cycleplot`: headless hypnogram + cycle-band plotting via `plot_hypnogram_cycles` / `plot_from_annotations`.
- `eeg_review_gui`: global left **Filters** dock and right **Topography & detail** dock, applied across both tabs.
- `eeg_review_gui`: live scalp topography of the active QC metric from EEGLAB `.set` channel locations, with a `label,x,y` montage CSV fallback.
- `eeg_review_gui`: right-dock global worst-events list, with impossible-scale events flagged as likely artefacts.
- Channels (QC) tab: `pac` event type, HARD/SOFT/DEAD/OK count strip, region column, per-metric heatmap shading, and Mark-channel-artefact / Queue-all-HARD / Build-re-detect actions.
- Epochs tab: paged 30-second epoch viewer with hypnogram strip, full-night amplitude overview, raw and band-filtered traces, and brush-to-mark artefact intervals.
- Per-channel artefact marks persisted to a sidecar `<stem>_review-qc.xml`; the original annotation XML is backed up and never modified.
- Re-detect request modal writing schema-v1 `redetect_request.json`; toolbar connection LEDs, segmented status bar, `View → Outlier threshold…`, `Help → Design notes`.
- `neural_events.db`: new `pac_coupling` table for PAC results, with a back-fill importer for existing CSV outputs.
- Spectral event columns on the events table: rms, power, peak-power-freq, energy, peak-energy-freq.
- Opt-in direct-to-database detection writes (`--write-db`) with deterministic event IDs, resume, and a `detection_runs` provenance table.
- `export_events_to_csv`: on-demand DB→CSV export.
- Re-run detection on selected channels (`--rerun`, `examples/rerun_detection.py`) with scoped replace and artefact-aware guards.
- `eeg_review_gui`: topography electrode hover labels and click-to-select; coordinate-based QC region.

### Changed

- `eeg_review_gui`: refocused on QC-driven outlier triage — `F` flags the selected channel for re-detect and `Export QC report…` replaces `Export Reviewed Events…`.
- `eeg_review_gui`: faster filter refresh on dense subjects.
- `eeg_review_gui`: neutral mid-grey chrome with pure-black plot interiors, so EEG traces and red outlier overlays read cleanly.
- `eeg_review_gui`: default to the Channels (QC) tab.
- `eeg_review_gui`: channel artefact / re-detect marks are visible and reversible; simplified the re-detect request dialog; removed the Reviewer1 placeholder.

### Removed

- `eeg_review_gui`: the per-event **Events tab**, including per-event accept/reject, stratified sampling, and the Compare-methods view. Two tabs remain: Channels (QC) and Epochs.
- Channels (QC) tab: the `ev/min`, `% in artf`, and `status` columns — verdict still shades rows and drives Drop/Keep, and density still appears in the Epochs-tab title.
- Right detail dock: the amplitude-trend sparkline.

### Fixed

- Slow-wave and K-complex detection with `polar='opposite'` inverted the signal twice — once in turtlewave and again inside Wonambi — so results were identical to `polar='normal'`. Inversion is now applied once.
- Spindle detection with `polar='opposite'` raised `AttributeError` on every channel and produced no events; it now runs.
- `tests/test_turtlewave.py` now covers detector polarity, which no test did before.
- `pacprocessor`: preferred phase (`preferred_phase_rad` / `preferred_phase_deg`) was reported 180° off, in both `analyze_pac` and `compare_conditions`; modulation index, mean vector length, and Rayleigh stats were unaffected.
- `eeg_review_gui`: METHOD / FREQUENCY BAND / event-type filters now refresh both the Channels (QC) tab and the active Epochs drill, without storming the refresh.
- Right detail dock: amplitude histogram now has mouse pan/zoom disabled.
- Event density now divides by artefact-free analysed time, fixing an artefact-scaled under-estimate; the review dashboard density matches and reflects live marks.
- CSV importer now preserves slash-methods (e.g. `AASM/Massimini2004`) instead of mangling the method name.

### Upgrading

- Back up `neural_events.db` before the first 4.0 write — `processing_status` auto-migrates and is not reversible.
- Regenerate every density CSV produced before 4.0.0; old values under-estimate density.
- Regenerate PAC preferred phase with `examples/fix_pac_preferred_phase.py`.
- Spindle detection with `polar='opposite'` produced no events at all before 4.0.0; re-run those detections to get output for the first time.
- Slow-wave and K-complex output from released versions is correct and comparable with 4.0.0; nothing to re-run.
- The review GUI no longer does per-event triage; review is channel-level.
- Full guide: [Upgrade to 4.0](docs/how-to/upgrade-to-4.0.md).

## [3.3.0] — 2026-05-04

### Added

- Windows as a first-class supported platform (CI matrix, classifiers, README).
- `pyproject.toml` (PEP 621) with `[dev]` and `[docs]` extras.
- `requirements.txt` lockfile via `uv pip compile` — 250 deps pinned.
- `LICENSE` (MIT).
- `.gitattributes` for cross-platform line endings.
- `.github/workflows/ci.yml` — install + smoke test on push/PR, three-OS matrix.
- CIRUS spindle method selectable from the GUI.

### Changed

- Minimum Python is now 3.10.
- Project moved off conda; `pip` + `venv` is canonical.
- Text-mode `open()` calls now pass `encoding='utf-8'`.
- `pacprocessor.py` uses `os.path.join` instead of f-string `/`.
- README rewritten for clone-and-run developer flow.
- `python-package.yml` is release-only; modern action versions.
- `tests/test_turtlewave.py` uses ASCII `[ok]`/`[FAIL]`.
- Root test scripts moved into `tests/`.

### Removed

- `setup.py`, `environment.yml`, `USEME.zip`, `resources/Woolcock.py`, `PYQTGRAPH_CONVERSION_SUMMARY.md`.
- `dist/`, `*.egg-info/`, `__pycache__/`, `.roo/` from version control.

### Security

- `.pypirc` scrubbed from git history (`git filter-repo`); PyPI tokens rotated.

## [3.2.0] — 2026-05-04

### Added

- K-complex detection — `ImprovedDetectKComplex`, `ParalKC`, GUI tab, review-GUI integration, example script.
- CIRUS spindle method (`ImprovedDetectSpindle(method='CIRUS')`) ported from qEEG_PSG Java.

### Changed

- `ParalSWA` exporter and importer accept `event_type` / `method` overrides; defaults preserved.
- Expanded `.gitignore` for build artifacts, virtual envs, IDE files.

## [3.1.0] — 2025

Initial PyPI release.
