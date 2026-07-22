# Direct-to-Database Write API Reference

The `dbwrite` module holds the shared primitives that `ParalEvents`, `ParalSWA`
and `ParalKC` use to write detected events straight into `neural_events.db`
(the `write_db=True` path), bypassing the JSON → CSV → import round-trip. It
also provides the on-demand DB → CSV export used to pull a flat file back out
of the database.

See [Direct-to-database detection](../../how-to/direct-to-database-detection.md)
for a task-oriented walkthrough.

::: turtlewave_hdEEG.dbwrite
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
