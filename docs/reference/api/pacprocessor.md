# Phase-Amplitude Coupling Processor API Reference

The PAC Processor module provides functionality for analyzing phase-amplitude coupling in EEG data.

`ParalPAC.analyze_pac` can persist its per-channel results directly to a
`pac_coupling` table in `neural_events.db` (`write_db=True`; see
`store_pac_to_database`). Existing PAC result CSVs can be back-filled into the
same table with `import_pac_csv_to_database` (one CSV) or
`backfill_pac_directory` (a whole results tree) — see
[Back-fill PAC results into the database](../../how-to/backfill-pac-to-database.md).

::: turtlewave_hdEEG.pacprocessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2