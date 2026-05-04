# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.3.0] — 2026-05-04

### Added

- Windows as a first-class supported platform (PyPI classifier, README, CI matrix).
- `pyproject.toml` (PEP 621) as the single source of truth for project metadata. Adds `[dev]` and `[docs]` extras.
- `requirements.txt` lockfile generated via `uv pip compile` — 250 dependencies fully pinned, including the load-bearing `numpy==1.26.4` / `wonambi==7.15` / `PyQt5==5.15.11`.
- `LICENSE` (MIT) file — declared previously but missing from the repo.
- `.gitattributes` enforcing LF for text files and explicit binary handling for fixtures (`.set`, `.fdt`, `.dat`, `.h5`).
- `.github/workflows/ci.yml` — install + smoke test on push/PR with `ubuntu-latest × macos-latest × windows-latest` matrix on Python 3.10.
- CIRUS spindle detection now selectable from the spindle detection GUI tab (method combo, alpha / background ratio / filter mode controls).

### Changed

- Minimum Python is now 3.10 (was 3.8).
- Project moved off conda. `pip` + `venv` is the canonical dev environment; HPC pipeline already used pip+venv.
- README rewritten with a clone → venv → install → smoke-test developer path; documents lockfile regeneration.
- 47 text-mode `open()` calls now pass `encoding='utf-8'` explicitly (Wonambi XML / EEGLAB metadata with non-ASCII content was Windows-incompatible under cp1252).
- Path constructions in `pacprocessor.py` switched from f-string `/` to `os.path.join`.
- `.github/workflows/python-package.yml` modernized: release-only triggers (push/PR triggers were silently broken, pointing at `main` instead of `master`), action versions bumped, duplicate test job removed (covered by `ci.yml`).
- Replaced `✓` / `✗` glyphs in `tests/test_turtlewave.py` with ASCII `[ok]` / `[FAIL]` (Windows cp1252 compatibility).
- Root-level `test_gui_performance.py` and `test_pyqtgraph_conversion.py` moved into `tests/`.

### Removed

- `setup.py` — superseded by `pyproject.toml`.
- `environment.yml` — conda no longer canonical.
- Orphan files at repo root: `USEME.zip`, `resources/Woolcock.py`, `PYQTGRAPH_CONVERSION_SUMMARY.md`.
- `dist/`, `*.egg-info/`, `__pycache__/`, `.roo/` removed from version control (now `.gitignore`d).

### Security

- `.pypirc` removed from the working tree and scrubbed from all of git history via `git filter-repo`. PyPI tokens were rotated; the GitHub repo secrets `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN` were updated to the new tokens.

## [3.2.0] — 2026-05-04

### Added

- K-complex detection: `ImprovedDetectKComplex` (subclasses `ImprovedDetectSlowWave` with AASM/Massimini2004 thresholds and a `min_isolation` filter), `ParalKC` orchestrator, "K-Complex Detection" tab in `turtlewave_gui`, K-complex visibility in `eeg_review_gui` (checkbox, timeline marker, table abbreviation), and `examples/hdEEG_kcomplex_detector.py`.
- CIRUS spindle detection method (`ImprovedDetectSpindle(method='CIRUS')`), ported from the qEEG_PSG Java tool. Threshold = `median + alpha * std` of the Hilbert envelope; supports `filter_mode='java'` (original implementation) and `filter_mode='wonambi'` (remez+filtfilt). Validated in D'Rozario 2022 / Lam 2021.

### Changed

- `ParalSWA.export_slow_wave_parameters_to_csv` and `ParalSWA.import_parameters_csv_to_database` accept optional `event_type` and `method` overrides so non-SW callers (e.g. `ParalKC`) can label their events correctly. Defaults preserve existing slow-wave behaviour.
- Expanded `.gitignore` to cover Python build artifacts, virtual envs, IDE files, and AI-assistant scratch directories.

## [3.1.0] — 2025

Initial public release on PyPI.
