# Phase-Amplitude Coupling Processor API Reference

The PAC Processor module provides functionality for analyzing phase-amplitude coupling in EEG data.

`ParalPAC.analyze_pac` can persist its per-channel results directly to a
`pac_coupling` table in `neural_events.db` (`write_db=True`; see
`store_pac_to_database`). Existing PAC result CSVs can be back-filled into the
same table with `import_pac_csv_to_database` (one CSV) or
`backfill_pac_directory` (a whole results tree) — see
[Back-fill PAC results into the database](../../how-to/backfill-pac-to-database.md).

`write_db=True` requires a nameable stored scope. On the event-locked path
(the default, `use_detected_events=True`) `event_type`/`pair_with_spindles`/
`event_opts` resolve it automatically. On the **continuous** path
(`use_detected_events=False`, e.g. theta-gamma coupling with no anchoring
event) there is nothing to derive it from, so `stored_event_type` and
`stored_method` must be passed explicitly — `analyze_pac` raises `ValueError`
at entry, before any analysis runs, rather than storing a result under a
guessed or generic label.

::: turtlewave_hdEEG.pacprocessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2