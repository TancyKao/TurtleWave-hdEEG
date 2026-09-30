"""
Channel-type rules and plain-text builders shared by the TurtleWave GUIs.

Everything here is pure Python: no Qt, no pyqtgraph. It is imported by
``turtlewave_gui.py``, ``eeg_review_gui.py`` and ``waveform_loader.py`` so the
three agree on one rule for which channels are non-EEG, which channels the
review GUI opens on, how the Setup tab describes a recording, and what a
researcher is told when a file cannot be loaded.

Only ``format_dataset_info`` and ``load_failure_message`` touch the library
(``turtlewave_hdEEG.timeline`` / ``turtlewave_hdEEG.eeglab_io``), and both
import it lazily, so this module imports headless and without the library.
"""

import datetime as _dt
import logging
import os

logger = logging.getLogger('frontend.channel_types')

#: Review-GUI default for EGI 256-channel nets, kept exactly as before.
EGI_DEFAULT_CHANNELS = ('E112', 'E118', 'Cz')

#: Midline sites preferred on any other montage, in this order.
MIDLINE_DEFAULT_CHANNELS = ('Cz', 'Fz', 'Pz')

#: How many channel names each Setup-tab preview line shows.
EEG_PREVIEW_COUNT = 10
OTHER_PREVIEW_COUNT = 20

#: Keys named in the "Found at the top level" line of a load-failure message.
FOUND_KEYS_SHOWN = 12

_REFERENCE_NOTE = '(as stored in the file; not changed by TurtleWave)'


# ---------------------------------------------------------------------------
# The non-EEG rule
# ---------------------------------------------------------------------------

def is_non_eeg_type(chan_type):
    """True only when a channel type positively says "not EEG".

    Parameters
    ----------
    chan_type : str or None
        One entry of ``header['chan_type']``.

    Returns
    -------
    bool
        ``False`` for ``'EEG'`` (any case, surrounding spaces ignored), for an
        empty string and for ``None``: an untyped channel is treated as EEG.
    """
    text = '' if chan_type is None else str(chan_type).strip()
    return bool(text) and text.upper() != 'EEG'


def _types_for(channels, chan_type):
    """``chan_type`` as a list parallel to ``channels``, or ``None`` when it
    is absent or does not line up with the channel list."""
    if chan_type is None:
        return None
    try:
        types = list(chan_type)
    except TypeError:
        return None
    if len(types) != len(channels):
        return None
    return types


class ChannelTypeSummary:
    """Channels of one recording split by the non-EEG rule.

    Parameters
    ----------
    channels : sequence of str
        Channel names as loaded.
    chan_type : sequence of str or None
        ``header['chan_type']``; ``None`` (or a list of the wrong length)
        means every channel is treated as EEG.

    Attributes
    ----------
    channels : list of str
        All channels, file order.
    eeg : list of str
        Channels not positively typed as non-EEG, file order.
    other : list of str
        Positively non-EEG channels, file order.
    type_of : dict
        ``{channel: type}`` for the non-EEG channels, type as in the file
        (surrounding spaces removed).
    type_counts : list of (str, int)
        Non-EEG types with their channel counts, in order of first appearance.
    has_types : bool
        Whether the file states any channel type at all.
    """

    def __init__(self, channels, chan_type=None):
        self.channels = [] if channels is None else [str(c) for c in channels]
        types = _types_for(self.channels, chan_type)
        self.has_types = bool(types) and any(
            str(t).strip() for t in types if t is not None)
        self.eeg, self.other, self.type_of = [], [], {}
        counts = {}
        for i, ch in enumerate(self.channels):
            t = types[i] if types is not None else None
            if is_non_eeg_type(t):
                label = str(t).strip()
                self.other.append(ch)
                self.type_of[ch] = label
                counts[label] = counts.get(label, 0) + 1
            else:
                self.eeg.append(ch)
        self.type_counts = list(counts.items())   # dicts keep insertion order

    @classmethod
    def from_dataset(cls, dataset):
        """Summary for a dataset object with ``.channels`` and ``.header``;
        an empty summary for ``None``."""
        if dataset is None:
            return cls([])
        channels = getattr(dataset, 'channels', None)
        channels = [] if channels is None else list(channels)
        header = getattr(dataset, 'header', None) or {}
        chan_type = header.get('chan_type') if hasattr(header, 'get') else None
        return cls(channels, chan_type)

    def is_non_eeg(self, channel):
        return channel in self.type_of

    def visible(self, show_non_eeg):
        """Channels to list, file order: all of them when ``show_non_eeg``,
        otherwise only the EEG ones."""
        return list(self.channels) if show_non_eeg else list(self.eeg)

    def count_line(self):
        """The Setup tab's ``Channels:`` count, e.g. ``277 (257 EEG, 20 other)``."""
        total = len(self.channels)
        if not self.has_types:
            return f"{total} (channel types not stated in the file; all listed as EEG)"
        if not self.other:
            return f"{total} (all EEG)"
        return f"{total} ({len(self.eeg)} EEG, {len(self.other)} other)"

    def checkbox_label(self):
        return f"Show non-EEG channels ({len(self.other)})"

    def checkbox_tooltip(self):
        kinds = ', '.join(f"{t} ({n})" for t, n in self.type_counts)
        return (f"The file marks {len(self.other)} channels as non-EEG: {kinds}. "
                f"They are hidden because the sleep-event detectors expect "
                f"scalp EEG. Tick to list them.")

    def item_tooltip(self, channel):
        """Tooltip for a Selected-list item, ``''`` for an EEG channel."""
        t = self.type_of.get(channel)
        return f"Non-EEG channel ({t})" if t else ''

    def run_note(self, selected):
        """Log line naming the non-EEG channels in ``selected``, or ``None``."""
        picked = [ch for ch in selected if ch in self.type_of]
        if not picked:
            return None
        noun = 'channel' if len(picked) == 1 else 'channels'
        return f"Note: {len(picked)} non-EEG {noun} selected: {', '.join(picked)}."


# ---------------------------------------------------------------------------
# Review GUI default channels
# ---------------------------------------------------------------------------

def default_review_channels(channels, chan_type=None):
    """Channels the review GUI opens on.

    Parameters
    ----------
    channels : sequence of str
        Channel names as loaded (EEG file or database).
    chan_type : sequence of str or None
        Parallel channel types; positively non-EEG channels are never chosen.

    Returns
    -------
    list of str
        ``['E112', 'E118', 'Cz']`` when all three are present (EGI nets);
        otherwise whichever of ``Cz``, ``Fz``, ``Pz`` are present, topped up
        to three from the start of the channel list. Fewer than three
        candidates returns them all; none returns ``[]``. Matching is exact
        and case-sensitive.
    """
    candidates = ChannelTypeSummary(channels, chan_type).eeg
    present = set(candidates)
    if all(ch in present for ch in EGI_DEFAULT_CHANNELS):
        return list(EGI_DEFAULT_CHANNELS)
    chosen = [ch for ch in MIDLINE_DEFAULT_CHANNELS if ch in present]
    for ch in candidates:
        if len(chosen) >= 3:
            break
        if ch not in chosen:
            chosen.append(ch)
    return chosen


# ---------------------------------------------------------------------------
# Setup tab: Dataset Information text
# ---------------------------------------------------------------------------

def _minutes(seconds):
    return f"{seconds / 60.0:.1f}"


def _preview(names, limit):
    shown = ', '.join(names[:limit])
    more = len(names) - limit
    return f"{shown} and {more} more" if more > 0 else shown


def _sampling_rate_text(s_freq):
    value = float(s_freq)
    return f"{int(value)} Hz" if value.is_integer() else f"{value:.1f} Hz"


def _reference_line(reference, n_eeg):
    """``Reference: ...`` from ``header['reference']`` (``None``, str or dict)."""
    n_good = None
    if isinstance(reference, dict):
        ref = reference.get('ref')
        n_good = reference.get('n_good')
    else:
        ref = reference
    ref = '' if ref is None else str(ref).strip()
    if not ref:
        return "Reference: not stated in the file"
    if ref.lower() == 'average':
        text = (f"average of {int(n_good)} of {n_eeg} EEG channels"
                if n_good is not None else "average")
    else:
        text = ref
    return f"Reference: {text} {_REFERENCE_NOTE}"


def _removed_text(timeline):
    n = timeline.n_boundaries
    if n == 0:
        return "none"
    noun = 'event' if n == 1 else 'events'
    if timeline.removed_seconds <= 0:
        return f"none ({n} boundary {noun} of zero length)"
    return (f"{_minutes(timeline.removed_seconds)} min at {n} boundary {noun} "
            f"(original recording {_minutes(timeline.original_seconds)} min)")


def format_dataset_info(dataset, data_file_path, output_dir, annot_file_path,
                        log=None):
    """The Setup tab's Dataset Information text for a loaded recording.

    Parameters
    ----------
    dataset : object
        Anything with ``.channels``, ``.sampling_rate`` and ``.header``
        (``n_samples``, ``start_time`` and, when present, ``chan_type``,
        ``reference`` and ``event``).
    data_file_path, output_dir, annot_file_path : str
        Shown as the file name and the two output paths.
    log : callable or None
        Receives one message per line that could not be built.

    Returns
    -------
    str
        One line per item, fixed order; a line that fails reads
        ``could not be read (see log)`` and never stops the others.
    """
    def note(message):
        logger.warning(message)
        if log is not None:
            log(message)

    header = getattr(dataset, 'header', None) or {}
    summary = ChannelTypeSummary.from_dataset(dataset)
    lines = []

    def guarded(label, build):
        try:
            lines.append(build())
        except Exception as err:
            note(f"Dataset information: {label.lower()} could not be read "
                 f"({type(err).__name__}: {err})")
            lines.append(f"{label}: could not be read (see log)")

    s_freq = getattr(dataset, 'sampling_rate', None)
    if s_freq is None:
        s_freq = header.get('s_freq')
    n_samples = header.get('n_samples')
    signal_seconds = (float(n_samples) / float(s_freq)
                      if n_samples is not None and s_freq else None)

    timeline, timeline_error = None, None
    try:
        from turtlewave_hdEEG.timeline import RecordingTimeline
        timeline = RecordingTimeline.from_header(header, s_freq, n_samples)
    except Exception as err:
        timeline_error = err
        note("Dataset information: the boundary events could not be read, so "
             f"the removed data is unknown ({type(err).__name__}: {err})")

    guarded('File', lambda: f"File: {os.path.basename(str(data_file_path or ''))}")

    def start_line():
        start = header['start_time']
        return f"Recording start: {start.strftime('%Y-%m-%d %H:%M:%S')}"
    guarded('Recording start', start_line)

    def end_line():
        if timeline_error is not None:
            raise RuntimeError("the removed data is unknown")
        start = header['start_time']
        end = start + _dt.timedelta(seconds=signal_seconds + timeline.removed_seconds)
        return f"Recording end: {end.strftime('%Y-%m-%d %H:%M:%S')}"
    guarded('Recording end', end_line)

    guarded('Signal duration', lambda: (
        f"Signal duration: {_minutes(signal_seconds)} min "
        f"({signal_seconds:,.1f} s)"))

    lines.append("Removed data: could not be read (see log)"
                 if timeline_error is not None
                 else f"Removed data: {_removed_text(timeline)}")

    guarded('Sampling rate', lambda: f"Sampling rate: {_sampling_rate_text(s_freq)}")
    guarded('Channels', lambda: f"Channels: {summary.count_line()}")
    guarded('Reference', lambda: _reference_line(header.get('reference'),
                                                 len(summary.eeg)))

    if summary.other:
        guarded('EEG channels', lambda: (
            f"EEG channels: {_preview(summary.eeg, EEG_PREVIEW_COUNT)}"))
        guarded('Other channels', lambda: (
            f"Other channels: {_preview(summary.other, OTHER_PREVIEW_COUNT)}"))
    else:
        guarded('Channels', lambda: (
            f"Channels: {_preview(summary.channels, EEG_PREVIEW_COUNT)}"))

    lines.append('')
    lines.append(f"Output directory: {output_dir or ''}")
    lines.append(f"Annotation file: {annot_file_path or ''}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Load-failure message
# ---------------------------------------------------------------------------

#: Fragments that mark a message as HDF5/h5py internals, never shown as-is.
_INTERNAL_MARKERS = ('synchronously', 'h5py', "object '", 'file signature')


def _first_line(exc):
    text = str(exc).strip()
    return text.splitlines()[0].strip() if text else ''


def load_failure_message(path, exc):
    """Dialog text for a recording that could not be loaded.

    Parameters
    ----------
    path : str
        The file the user chose; only its base name is shown.
    exc : BaseException
        What the loader raised. ``EEGLABFormatError`` and the library's
        missing-``.fdt`` ``FileNotFoundError`` get their own wording; anything
        else is reduced to its first line, with HDF5 internals replaced.

    Returns
    -------
    str
        Plain text, never a traceback or a Python class name.
    """
    name = os.path.basename(str(path or '')) or 'This file'
    head = "Could not load this recording."
    tail = "The full error is in the log panel."

    try:
        from turtlewave_hdEEG.eeglab_io import EEGLABFormatError
    except ImportError:                      # library absent: generic wording
        EEGLABFormatError = ()

    if EEGLABFormatError and isinstance(exc, EEGLABFormatError) and (
            exc.missing or exc.top_level_keys):
        fname = exc.filename or name
        kind = ('a MATLAB 7.3 (HDF5) EEGLAB file' if exc.is_hdf5
                else 'a MATLAB (pre-7.3) EEGLAB file')
        keys = [k for k in exc.top_level_keys if not str(k).startswith('#')]
        shown = ', '.join(keys[:FOUND_KEYS_SHOWN]) if keys else 'nothing'
        more = (f" ({FOUND_KEYS_SHOWN} of {len(keys)} shown)"
                if len(keys) > FOUND_KEYS_SHOWN else '')
        return (f"{head}\n\n"
                f"{fname} is {kind}, but TurtleWave could not find the sampling "
                f"rate (srate) and channel list (chanlocs), either at the top "
                f"level or inside an EEG structure.\n\n"
                f"Found at the top level: {shown}{more}\n\n"
                f"If the file was written by a script, check that it was saved "
                f"with EEGLAB's pop_saveset. {tail}")

    reason = _first_line(exc)
    if isinstance(exc, FileNotFoundError) and reason.startswith(
            'The signal data for'):
        return (f"{head}\n\n{reason}\n\n"
                f"Copy the .fdt file next to the .set file and load again. {tail}")

    if not reason or any(m in reason for m in _INTERNAL_MARKERS) or \
            type(exc).__module__.startswith('h5py'):
        reason = "the file's contents could not be read"
    return f"{head}\n\n{name}: {reason}\n\n{tail}"
