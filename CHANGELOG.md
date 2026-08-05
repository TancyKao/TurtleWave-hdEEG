# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.1] — 2026-08-05

Closes a class of silent data loss where detection wrote files and a separately
constructed string had to find them again. Affected runs completed without
errors and wrote nothing to `neural_events.db`.

### Added

- Public API: `derive_subject`, `fmt_freq_token`, `ensure_pac_schema`, `guard_run_id`, `verify_channel_coverage`, and `stored_event_type` / `stored_method` on `analyze_pac`.

### Changed

- Both HPC batch drivers verify channel coverage against the database and exit non-zero when channels are missing, instead of always logging success.

### Fixed

- PAC results now reach `neural_events.db`; the GUI and `examples/hdEEG_pac_detector.py` never requested a database write, so runs produced CSVs only and created no `pac_coupling` table.
- The cluster spindle driver built its frequency-band filename token with one-decimal formatting while detection wrote it unformatted, so any band bound needing two decimals matched zero files, imported nothing, and still logged success; all band tokens now come from one shared function.
- Detection methods containing a slash, such as `AASM/Massimini2004`, were truncated to `AASM` in `events.method`.
- A multi-method spindle run stamped every row with a single method, so the uniqueness constraint silently discarded the other method's events.
- Failed CSV imports returned "0 added" and were indistinguishable from a clean re-run; importers now raise, and parameter exporters raise when their pattern matches no files instead of writing a placeholder CSV.
- Continuous PAC was storable as slow-wave coupling; `analyze_pac` now refuses to write a row whose scope it cannot name.
- A subject identifier could differ depending on which annotation XML a caller pointed at, splitting one recording across two database keys.

### Upgrading

- PAC results from 4.0.0 exist as CSVs only; back-fill or re-run them to populate `pac_coupling`.
- The first coverage check against a database created before 4.0.1 reports a large `events_only` warning, because legacy `processing_status` rows migrate in with an empty method, zero frequency bounds and an empty stage and never match the scoped query. It exits 0 and self-heals after one scoped run.

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
