# EEGLAB I/O API Reference

`open_dataset` opens a recording as a Wonambi `Dataset` and reads both EEGLAB
`.set` layouts: the classic one with a single `EEG` variable, and the one used
by the Compumedics export, whose fields are top-level variables. It adds
`chan_type`, `reference` and `interp_channels` to the dataset header for every
format. The module imports no Qt.

For the task, see
[How to analyse Compumedics and other cut EEGLAB recordings](../../how-to/analyse-compumedics-recordings.md).

::: turtlewave_hdEEG.eeglab_io
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
