# Utilities API Reference

Shared helpers used across the detection processors, including
`compute_analysed_seconds` and `build_density_denominators`, which compute the
**artefact-free** in-stage time a detector actually pooled. Event density
(events per minute) is divided by this artefact-free time rather than by all
scored epochs of a stage — see
[Event density is artefact-free](../../explanation/overview.md#event-density-is-artefact-free)
for why this matters.

::: turtlewave_hdEEG.utils
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
