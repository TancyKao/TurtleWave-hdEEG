# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ParalCycles` / `detect_cycles` (`cycleprocessor.py`): rule-based NREM-REM sleep-cycle detection from the hypnogram, ported from the MATLAB SpectraDynamic_Analysis detector. Supports the NREM-based `'2022'` and Feinberg `'1979'` definitions. Fills `events.cycle`, a new `sleep_cycles` table, and writes cycle markers to the annotation XML; `run()` also backfills existing databases. Example: `examples/hdEEG_cycle_detector.py`.
- `neural_events.db`: new `sleep_cycles` table (per-cycle NREM/REM boundaries and durations) and an `idx_cycle` index on `events(cycle)`.
- `compute_stage_durations` (re-exported from `turtlewave_hdEEG`): per-stage sleep durations from the hypnogram, moving stage summaries toward a DB-only workflow.
- `neural_events.db`: new `stage_durations` table (per-subject Wake/N1/N2/N3/REM, artefact, and total minutes), written automatically by `ParalCycles.run()`.
- `finalize_cycles_and_durations` (re-exported from `turtlewave_hdEEG`): a single post-detection step that fills `neural_events.db` with sleep cycles under both the `'2022'` and `'1979'` definitions and stage durations, and tags events by the default 2022 cycles.
- `turtlewave_hdEEG.cycleplot`: headless sleep-cycle plotting — `plot_hypnogram_cycles` / `plot_from_annotations` write a hypnogram + cycle-band PNG comparing both methods.
- `stage_durations` table now created at database initialization (`initialize_sqlite_database`), so the schema is complete before cycles are computed.
- `eeg_review_gui`: global left **Filters** dock and right **Topography & detail** dock, applied across both tabs.
- `eeg_review_gui`: **live scalp topography** of the active QC metric, interpolated from the loaded EEGLAB `.set` channel locations (`read_eeglab_chanlocs` in `dataset.py`); falls back to a `label,x,y` montage CSV.
- `eeg_review_gui`: right-dock **global worst-events list** — the most extreme events across all channels for the current event type; click a row to drill into that channel and epoch. Impossible-scale (>1000 µV) events flagged red as likely artefacts.
- Channels (QC) tab: `pac` event type, HARD/SOFT/DEAD/OK count strip, region column, per-metric robust-z heatmap shading, Mark-channel-artefact / Queue-all-HARD / Build-re-detect actions.
- Epochs tab: paged **30-second epoch viewer** — fixed window (no zoom), Prev/Next buttons + Left/Right + PageUp/PageDown keys aligned to the epoch grid; hypnogram strip with current-epoch box, slim full-night amplitude overview (click to jump), raw + band-filtered traces; brush a range on the trace and *Mark as artefact* to write a channel-scoped interval to the DB and sidecar XML.
- Per-channel artefact marks persisted to a sidecar `<stem>_review-qc.xml` (rater `review-qc`); original XML backed up to `*.xml.bak` and never modified.
- Re-detect request modal writing schema-v1 `redetect_request.json`; toolbar connection LEDs, segmented status bar, `View → Outlier threshold…`, `Help → Design notes`.
- `neural_events.db`: new `pac_coupling` table for PAC results, with a back-fill importer for existing CSV outputs.
- Spectral event columns on the events table: rms, power, peak-power-freq, energy, peak-energy-freq.
- Opt-in direct-to-database detection writes (`--write-db`) with deterministic event IDs, resume, and a `detection_runs` provenance table.
- `export_events_to_csv`: on-demand DB→CSV export.
- Re-run detection on selected channels (`--rerun`, `examples/rerun_detection.py`) with scoped replace and artefact-aware guards.
- `eeg_review_gui`: topography electrode hover labels and click-to-select; coordinate-based QC region.

### Changed

- `eeg_review_gui`: refocused on QC-driven outlier triage — the per-event **Events tab removed** (no per-event accept/reject, no stratified sampling, no Compare-methods view). Two tabs remain: the Channels (QC) landing surface and the Epochs drill; `F` now flags the selected channel for re-detect and `Export QC report…` replaces `Export Reviewed Events…`.
- `eeg_review_gui`: QC-table model is records-backed (no per-cell `DataFrame.iloc`) and the QC query uses a lean column projection — filter refresh on a dense subject dropped ~2.7× (2.3 s → 0.85 s).
- `eeg_review_gui`: chrome theme switched to PyQt5 Fusion-style neutral mid-grey via a module-level `THEME` dict + new `DARK_QSS`; PyQtGraph plot interiors are now pure black (`THEME['plot_bg']`) via a `_theme_plot(pw)` helper, so EEG traces and red outlier overlays read cleanly.
- `eeg_review_gui`: default to the Channels (QC) tab.
- `eeg_review_gui`: channel artefact / re-detect marks are visible and reversible; simplified the re-detect request dialog; removed the Reviewer1 placeholder.

### Fixed

- `pacprocessor`: preferred phase (`preferred_phase_rad` / `preferred_phase_deg`) was reported 180° off because the bin-centre vector (`vecbin`) spanned `[0, 2π)` while `_mean_amp` bins phase on `[-π, π]`. Corrected in both `analyze_pac` and `compare_conditions`; modulation index, mean vector length, and Rayleigh stats were unaffected.
- `eeg_review_gui`: METHOD / FREQUENCY BAND / event-type filters now refresh the Channels (QC) tab and the active Epochs drill (single `_refresh_all`; QC query threads method/freq through `get_events`). Event-type change no longer storms the refresh (combo-repopulation signals blocked).
- Channels (QC) tab: dropped `ev/min`, `% in artf`, and `status` columns from the table (verdict still shades rows and drives Drop/Keep; density still appears in the Epochs-tab title).
- Right detail dock: amplitude histogram now has mouse pan/zoom disabled (fixed reference); the 32-bin amplitude-trend sparkline was removed.
- Event density now divides by artefact-free analysed time (per-stage and whole-night), fixing an artefact-scaled under-estimate; the review dashboard density matches and reflects live marks.
- CSV importer now preserves slash-methods (e.g. `AASM/Massimini2004`) instead of mangling the method name.

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
