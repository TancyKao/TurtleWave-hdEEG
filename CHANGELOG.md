# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- K-complex detection: `ImprovedDetectKComplex`, `ParalKC`, GUI tab in
  `turtlewave_gui`, K-Complex visibility in `eeg_review_gui`, example script.
- CIRUS spindle detection method (`ImprovedDetectSpindle(method='CIRUS')`),
  ported from the qEEG_PSG Java tool. Validated in D'Rozario 2022 / Lam 2021.
- GUI integration for CIRUS in the spindle detection tab.
- `pyproject.toml` (PEP 621) replaces `setup.py` as the single source of truth
  for project metadata and dependencies.
- Pinned-lockfile workflow via `uv pip compile` (`requirements.txt`).
- `LICENSE` file (MIT) and `CHANGELOG.md`.
- Expanded `.gitignore` for Python build artifacts, virtual envs, IDE files,
  and AI-assistant scratch directories.

### Changed

- `ParalSWA.export_slow_wave_parameters_to_csv` and
  `ParalSWA.import_parameters_csv_to_database` accept optional `event_type`
  and `method` overrides so non-SW callers (e.g. `ParalKC`) can label their
  events correctly. Defaults preserve existing slow-wave behaviour.
- Minimum supported Python is now 3.10 (was 3.8). Python 3.8 / 3.9 are EOL.
- Project moves off conda as the primary development environment toward
  plain `pip` + `venv`. `environment.yml` retained as an optional path.

### Removed

- `setup.py` — superseded by `pyproject.toml`.
- `.pypirc` removed from version control. Tokens still need rotating since
  they remain in pre-fix git history.

## [3.1.0] — 2025

Initial public release on PyPI. See `git log v3.1.0` for the full history of
this release.
