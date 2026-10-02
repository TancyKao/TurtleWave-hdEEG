# Utilities API Reference

Shared helpers used across the detection processors, including
`compute_analysed_seconds` and `build_density_denominators`, which compute the
**artefact-free** in-stage time a detector actually pooled. Event density
(events per minute) is divided by this artefact-free time rather than by all
scored epochs of a stage — see
[Event density is artefact-free](../../explanation/overview.md#event-density-is-artefact-free)
for why this matters.

`derive_subject` is the single, shared resolver for the subject id that keys
`sleep_cycles`, `stage_durations` and `pac_coupling` — every detector, PAC
run and back-fill script should call it rather than deriving a subject id
locally. See
[About naming, subject identity & provenance conventions](../../explanation/naming-and-identity-conventions.md)
for the precedence order and why it matters.

`region_from_label` maps a 10-20 / 10-5 electrode label to a coarse scalp
region. `interpolated_channels` and `warn_interpolated_channels` report which
selected channels the recording marks as interpolated; the detectors call the
latter once per run. See
[How to analyse Compumedics and other cut EEGLAB recordings](../../how-to/analyse-compumedics-recordings.md#recognise-interpolated-channels).

`quiet_wonambi_warnings()` silences two harmless `DeprecationWarning`s of the
pinned Wonambi 7.15: the `fooof` package-deprecation notice, printed when Wonambi
imports `fooof`, and NumPy's "Conversion of an array with ndim > 0 to a scalar is
deprecated" from `wonambi/trans/analyze.py`. It silences nothing else.

`import turtlewave_hdEEG` calls it, before importing Wonambi, only when the
environment variable named by `QUIET_WONAMBI_ENV` (`TURTLEWAVE_QUIET_WONAMBI`) is
`1`, `true`, `yes` or `on`. The GUIs and the example and Gadi scripts set the
variable to `1` with `os.environ.setdefault` before their imports, so exporting
`TURTLEWAVE_QUIET_WONAMBI=0` turns it off. The `fooof` notice can be stopped only
if the function runs before the first `import wonambi`, because `fooof` resets
the warning filter itself just before it warns; called later it still handles the
NumPy warning. See
[Upgrade to 4.6](../../how-to/upgrade-to-4.6.md#quiet-two-harmless-wonambi-warnings).

::: turtlewave_hdEEG.utils
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
