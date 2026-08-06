# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] — 2026-08-06

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

## [4.0.2] — 2026-08-06

Fixes `disk I/O error` when `neural_events.db` lives on a mapped network drive
or a synced folder. The databases were permanently in SQLite WAL journal mode,
which cannot work over a network filesystem, and every connection reimposed WAL
without checking whether it had been applied. The package now leaves an existing
database's journal mode alone, so converting a database once is enough — see
**Upgrading**.

### Added

- `TURTLEWAVE_SQLITE_JOURNAL` environment variable imposing a SQLite journal mode on every database this package opens, overriding the preserve rule in both directions.
- Public API: `set_journal_mode`, converting an existing database to another journal mode, and `VALID_JOURNAL_MODES`, the accepted mode names.
- `examples/set_db_journal_mode.py`: converts one database or a whole `ROOT/*/wonambi/neural_events.db` tree.
- Docs: [Run with a database on a network drive](docs/how-to/run-with-database-on-a-network-drive.md) and [Database concurrency and journalling](docs/explanation/database-concurrency-and-journalling.md).

### Changed

- The package no longer overrides a journal mode you chose deliberately: `open_write_connection` preserves an existing database's mode and logs it, and applies WAL only to a database it creates. A database on local disk is already WAL and stays WAL.
- `finalize_cycles_and_durations` and `ParalCycles.run` raise `FileNotFoundError` when the database does not exist instead of creating one; they annotate an existing `neural_events.db` and a missing file means a wrong path or detection that never ran.
- `open_write_connection` gains optional `journal` and `logger` arguments; existing calls are unaffected.
- `store_cycles_to_database`, `store_stage_durations`, `tag_events_with_cycles` and `ParalCycles.run` gain an optional trailing `conn` argument for sharing one connection.
- A `disk I/O error` from `open_write_connection` or `set_journal_mode` now names the database, the WAL-on-a-network-drive cause, `set_journal_mode` and the how-to page. Same exception class, with the original attached as `__cause__`.

### Fixed

- A detection run or a review GUI silently converted a database back to WAL, undoing a conversion made to keep it working on a network drive.
- The review GUIs no longer force `journal_mode=WAL` and `mmap_size`, both unusable over a network filesystem, and the detection GUI's connections now set a lock timeout.
- `open_write_connection` now checks that an imposed journal mode was actually applied and warns when it was not, instead of assuming the pragma took effect.
- The CSV importers, the PAC writers and several readers wait up to 60 seconds for a lock instead of five, so a reader and a writer on a `DELETE`-mode database no longer collide.
- A mistyped `db_path` in the cycle backfill wrote a stray empty database and then failed with `no such table: main.events`; it now fails immediately naming the path.
- `finalize_cycles_and_durations` uses one timed connection per subject instead of six untimed connect/close cycles, each of which recreated the WAL sidecar files.
- `import frontend` no longer raises `ImportError` without PyQt5 installed; the main GUI import is guarded like the review GUI's already was.

### Upgrading

- Journal mode is a persistent property of the database file, so a database created before 4.0.2 is in WAL until you convert it. Close every GUI and convert once with `examples/set_db_journal_mode.py`; from 4.0.2 the choice sticks, and later runs will not switch it back. See [Run with a database on a network drive](docs/how-to/run-with-database-on-a-network-drive.md).
- Set `TURTLEWAVE_SQLITE_JOURNAL=DELETE` only if you are creating *new* databases on the share; a database created there would otherwise be created in WAL and fail the same way.
- A cycle backfill pointed at a database that does not exist now raises `FileNotFoundError` instead of creating an empty one. Any script relying on that database being created must run detection first.
- Databases on local disk need no action.

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
