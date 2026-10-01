"""
Time map between a cut EEGLAB recording and the original night.

When data is removed from an EEGLAB dataset, EEGLAB splices the remaining
samples together and leaves a ``boundary`` event at each splice whose
``duration`` is the number of samples removed there. The signal file then
runs on a *cut* time base while a hypnogram scored on the full night (for
example ``EEG.etc.stages`` from a Compumedics export) runs on the *original*
time base. This module converts between the two, turns the full-night
stages into exact whole-second epochs on the cut file
(:func:`exact_cut_epochs`), and saves the map and those epochs as a JSON
sidecar so steps that have no dataset (sleep cycles, stage durations) can
still use the full night (:func:`load_sidecar_for`).

Pure numpy; no Wonambi, no Qt.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger('turtlewave_hdEEG.timeline')

#: EEGLAB stage-event names (lower-cased) -> Compumedics stage codes, the
#: codes Wonambi's ``import_staging(source='compumedics')`` reads.
STAGE_EVENT_CODES = {'ns': '?', 'wake': 'W', 'n1': 'N1', 'n2': 'N2',
                     'n3': 'N3', 'rem': 'R'}

#: Compumedics code for an unscored epoch.
UNDEFINED_CODE = '?'

#: Tolerance, in seconds, when placing an original-time stage onset into an
#: ``etc.stages`` epoch. Onsets sit a few ms either side of the 30 s grid
#: after the sample-level round trip, so a bare floor would put an onset at
#: 29.988 s into the previous epoch.
ONSET_TOLERANCE_S = 0.5

#: Compumedics stage codes -> one canonical code, for comparing hypnograms
#: that use different spellings of the same stage (``'2'`` and ``'N2'``).
#: Mirrors Wonambi's ``COMPUMEDICS_STAGE_KEY``.
_CANONICAL_CODE = {'?': '?', 'W': 'W', '0': 'W', 'N1': 'N1', '1': 'N1',
                   'N2': 'N2', '2': 'N2', 'N3': 'N3', '3': 'N3', '4': 'N3',
                   '5': 'N3', 'R': 'R'}


def canonical_stage_code(code):
    """Canonical Compumedics stage code, for comparing two spellings.

    Parameters
    ----------
    code : object
        A stage code as stored, e.g. ``'2'``, ``'N2'``, ``' w'``, ``4``.

    Returns
    -------
    str
        ``'W'``, ``'N1'``, ``'N2'``, ``'N3'``, ``'R'`` or ``'?'``
        (``'0'`` is Wake; ``'3'``, ``'4'`` and ``'5'`` are N3); anything
        unrecognised is returned as a stripped string.
    """
    text = str(code).strip()
    return _CANONICAL_CODE.get(text.upper(), text)


#: Version of the sidecar JSON layout written by :meth:`RecordingTimeline.to_json`.
SIDECAR_SCHEMA = 2


def _event_fields(event_info):
    """Onsets, lower-cased types and durations from ``header['event']``."""
    if not event_info:
        return np.array([]), [], []
    onsets = list(event_info.get('onsets', []) or [])
    types = [str(t).strip().lower() if t is not None else ''
             for t in (event_info.get('types', []) or [])]
    durations = list(event_info.get('durations', []) or [])
    n = min(len(onsets), len(types))
    return onsets[:n], types[:n], durations[:n]


def _as_float(value, default=np.nan):
    try:
        out = float(np.ravel(value)[0]) if np.ndim(value) else float(value)
    except (TypeError, ValueError, IndexError):
        return default
    return out


def boundaries_from_events(event_info, s_freq):
    """Splice points and removed lengths from EEGLAB ``boundary`` events.

    Parameters
    ----------
    event_info : dict or None
        ``header['event']`` with ``onsets`` (EEGLAB latency, 1-based samples),
        ``types`` and ``durations`` (samples).
    s_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    cut_onsets : ndarray
        Cut-file time, in seconds, of the first sample after each splice:
        ``(latency - 0.5) / s_freq`` (EEGLAB puts a boundary half-way
        between two samples). Sorted.
    removed : ndarray
        Seconds removed at each splice (``duration / s_freq``). A missing or
        non-finite duration (as left by concatenating two files) counts as 0.
    """
    onsets, types, durations = _event_fields(event_info)
    cut, removed = [], []
    for i, t in enumerate(types):
        if t != 'boundary':
            continue
        lat = _as_float(onsets[i])
        if not np.isfinite(lat):
            continue
        dur = _as_float(durations[i]) if i < len(durations) else np.nan
        cut.append((lat - 0.5) / float(s_freq))
        removed.append(dur / float(s_freq) if np.isfinite(dur) and dur > 0 else 0.0)
    cut = np.asarray(cut, dtype=float)
    removed = np.asarray(removed, dtype=float)
    order = np.argsort(cut, kind='stable')
    return cut[order], removed[order]


def stage_events_from_header(event_info, s_freq):
    """Stage events (``ns``, ``wake``, ``n1`` ...) as cut-time onsets and codes.

    Parameters
    ----------
    event_info : dict or None
        ``header['event']``.
    s_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    list of (float, str)
        ``(onset_seconds, compumedics_code)`` sorted by onset, onset
        ``(latency - 1) / s_freq``. Event durations are ignored: every stage
        event carries the nominal 30 s even when a cut truncated it.
    """
    onsets, types, _ = _event_fields(event_info)
    out = []
    for i, t in enumerate(types):
        code = STAGE_EVENT_CODES.get(t)
        if code is None:
            continue
        lat = _as_float(onsets[i])
        if np.isfinite(lat):
            out.append(((lat - 1.0) / float(s_freq), code))
    out.sort(key=lambda x: x[0])
    return out


def stage_event_intervals(stage_events, T, epoch_length=30.0, splices=None):
    """Label intervals from stage events, each ending at the next onset.

    Parameters
    ----------
    stage_events : list of (float, str)
        Output of :func:`stage_events_from_header`.
    T : float
        Cut-file duration in seconds.
    epoch_length : float
        Nominal stage-event length in seconds.
    splices : sequence of float or None
        Cut-file seconds of the splices (``boundary`` events). An interval is
        also ended at the first splice strictly inside it: the data after a
        splice comes from a later part of the night, and when its own stage
        event was removed with the cut it has no stage. ``None`` (default)
        clips nothing.

    Returns
    -------
    list of (float, float, str)
        ``[onset_i, min(onset_i + epoch_length, onset_{i+1}, next splice, T))``
        with its code; empty intervals are dropped.
    """
    cuts = np.sort(np.asarray([] if splices is None else list(splices),
                              dtype=float))
    out = []
    for i, (onset, code) in enumerate(stage_events):
        end = onset + epoch_length
        if i + 1 < len(stage_events):
            end = min(end, stage_events[i + 1][0])
        end = min(end, T)
        start = max(onset, 0.0)
        k = np.searchsorted(cuts, start + 1e-6, side='left')
        if k < cuts.size and cuts[k] < end:
            end = float(cuts[k])
        if end > start:
            out.append((start, end, code))
    return out


def round_edge(t):
    """Round an epoch edge to a whole second, halves up.

    The one rounding rule for every exact-epoch edge, so two pieces that
    share a raw edge always share the rounded edge.

    Parameters
    ----------
    t : float
        Edge in seconds.

    Returns
    -------
    int
        ``floor(t + 0.5)``, with a 1e-9 s allowance so an edge computed as
        ``x.4999999999`` from a sample-level round trip still rounds up.
    """
    return int(np.floor(float(t) + 0.5 + 1e-9))


def exact_cut_epochs(intervals, last_second, undefined=UNDEFINED_CODE):
    """Whole-second, variable-length epochs tiling ``[0, last_second)``.

    Parameters
    ----------
    intervals : iterable of tuple
        ``(start, end, code)`` or ``(start, end, code, orig_epoch)`` label
        intervals in cut-file seconds, as from
        :meth:`RecordingTimeline.stage_intervals_cut` or
        :func:`stage_event_intervals`. A missing ``orig_epoch`` is ``-1``.
    last_second : int
        End of the last epoch, ``int(n_samples / s_freq)``; the same value
        Wonambi writes as the annotation file's ``last_second``.
    undefined : str
        Code for gap-filling epochs.

    Returns
    -------
    list of (int, int, str, int)
        ``(start, end, code, orig_epoch)`` sorted, contiguous, starting at 0
        and ending at ``last_second``. Gap fillers carry ``undefined`` and
        ``orig_epoch = -1``.

    Notes
    -----
    Every edge goes through :func:`round_edge` and is clipped to
    ``[0, last_second]``, so an edge at or past the end of the signal becomes
    ``last_second``. A piece whose rounded end is not after its rounded start
    (a sliver under about 0.5 s) is dropped, never merged: its time goes to
    whichever neighbour the rounding gives it, at most 0.5 s per edge.
    Pieces that overlap after rounding (only possible when the input
    overlaps) are trimmed to start where the previous one ended.
    """
    last = int(last_second)
    pieces = []
    for iv in intervals:
        start, end, code = iv[0], iv[1], iv[2]
        orig = int(iv[3]) if len(iv) > 3 else -1
        s = min(max(round_edge(start), 0), last)
        e = min(max(round_edge(end), 0), last)
        pieces.append((s, e, str(code), orig))
    pieces.sort(key=lambda p: (p[0], p[1]))

    out = []
    cursor = 0
    for s, e, code, orig in pieces:
        s = max(s, cursor)
        if e <= s:
            continue
        if s > cursor:
            out.append((cursor, s, undefined, -1))
        out.append((s, e, code, orig))
        cursor = e
    if cursor < last:
        out.append((cursor, last, undefined, -1))
    return out


class SidecarMismatchError(ValueError):
    """The timeline sidecar is missing, outdated, or disagrees with its XML."""

class RecordingTimeline:
    """Map between the cut signal's time base and the original night.

    Parameters
    ----------
    cut_onsets : array-like of float
        Cut-file seconds of the first sample after each splice, sorted.
    removed : array-like of float
        Seconds removed at each splice.
    s_freq : float
        Sampling frequency in Hz.
    n_samples : int
        Samples in the cut signal.
    stages : sequence of str or None
        Full-night hypnogram (``etc.stages`` codes), one per original epoch.
    epoch_length : float
        Epoch length of ``stages`` in seconds.
    meta : dict or None
        Extra fields carried into the sidecar (source, versions, paths).
    last_second : int or None
        End of the cut file's exact epochs; ``int(n_samples / s_freq)`` when
        ``None``.
    cut_epochs : list of tuple or None
        The exact epochs written to the annotation file,
        ``(start, end, code, orig_epoch)``; ``None`` until staging sets it.

    Attributes
    ----------
    n_boundaries : int
        Number of boundary events (splices), including zero-length ones.
    removed_seconds : float
        Total seconds removed.
    cut_seconds : float
        Duration of the cut signal, ``n_samples / s_freq``.
    original_seconds : float
        ``cut_seconds + removed_seconds``.
    """

    def __init__(self, cut_onsets, removed, s_freq, n_samples, stages=None,
                 epoch_length=30.0, meta=None, last_second=None,
                 cut_epochs=None):
        self.cut_onsets = np.asarray(cut_onsets, dtype=float).ravel()
        self.removed = np.asarray(removed, dtype=float).ravel()
        if self.cut_onsets.shape != self.removed.shape:
            raise ValueError('cut_onsets and removed must have the same length')
        self.s_freq = float(s_freq)
        self.n_samples = int(n_samples)
        self.stages = None if stages is None else [str(s) for s in stages]
        self.epoch_length = float(epoch_length)
        self.meta = dict(meta or {})
        self.last_second = (int(self.n_samples / self.s_freq)
                            if last_second is None else int(last_second))
        self.cut_epochs = None if cut_epochs is None else [
            (int(e[0]), int(e[1]), str(e[2]), int(e[3])) for e in cut_epochs]
        # removed before each splice, and each splice's position in original time
        self._removed_before = np.concatenate(([0.0], np.cumsum(self.removed)[:-1])) \
            if self.removed.size else np.array([])
        self.original_onsets = self.cut_onsets + self._removed_before

        self.n_boundaries = int(self.cut_onsets.size)
        self.removed_seconds = float(self.removed.sum())
        self.cut_seconds = self.n_samples / self.s_freq
        self.original_seconds = self.cut_seconds + self.removed_seconds

    @classmethod
    def from_header(cls, header, s_freq, n_samples, epoch_length=30.0):
        """Build the map from a dataset header.

        Parameters
        ----------
        header : dict
            ``dataset.header``; reads ``event`` (boundary events, durations
            in samples) and ``stages`` (full-night codes) when present.
        s_freq : float
            Sampling frequency in Hz.
        n_samples : int
            Samples in the cut signal.
        epoch_length : float
            Epoch length of ``header['stages']`` in seconds.

        Returns
        -------
        RecordingTimeline
        """
        cut, removed = boundaries_from_events(header.get('event'), s_freq)
        stages = header.get('stages')
        if stages is not None:
            stages = [str(s).strip() for s in list(stages)]
        return cls(cut, removed, s_freq, n_samples, stages=stages,
                   epoch_length=epoch_length)

    def cut_to_original(self, t):
        """Convert cut-file seconds to original-night seconds.

        Parameters
        ----------
        t : float or array-like
            Seconds on the cut file. A time exactly at a splice is the first
            sample after the removed data.

        Returns
        -------
        float or ndarray
            Same shape as ``t``.
        """
        arr = np.asarray(t, dtype=float)
        n = np.searchsorted(self.cut_onsets, arr, side='right')
        cum = np.concatenate(([0.0], np.cumsum(self.removed)))
        out = arr + cum[n]
        return float(out) if np.ndim(out) == 0 else out

    def original_to_cut(self, t):
        """Convert original-night seconds to cut-file seconds.

        Parameters
        ----------
        t : float or array-like
            Seconds on the original night.

        Returns
        -------
        float or ndarray
            Same shape as ``t``. A time inside removed data clamps to the
            splice it was removed at.
        """
        arr = np.asarray(t, dtype=float)
        ends = self.original_onsets + self.removed
        # splices whose removed span is wholly at or before t
        n_done = np.searchsorted(ends, arr, side='right')
        cum = np.concatenate(([0.0], np.cumsum(self.removed)))
        out = arr - cum[n_done]
        # inside the removed span of splice n_done (if any) -> clamp
        idx = np.minimum(n_done, max(self.n_boundaries - 1, 0))
        if self.n_boundaries:
            inside = (n_done < self.n_boundaries) & \
                     (arr >= self.original_onsets[idx]) & (arr < ends[idx])
            out = np.where(inside, self.cut_onsets[idx], out)
        return float(out) if np.ndim(out) == 0 else out

    @property
    def consistency_tolerance(self):
        """Slack, in seconds, allowed at both edges of :attr:`is_consistent`.

        One sample per boundary event (``n_boundaries / s_freq``). EEGLAB
        removes whole samples at each cut and stores the removed length per
        boundary, so the summed removal can differ from the scorer's clock by
        up to about one sample per splice; without slack a night that ends
        exactly on an epoch multiple could read as inconsistent.
        """
        return self.n_boundaries / self.s_freq

    @property
    def is_consistent(self):
        """Whether ``etc.stages`` covers exactly the original night.

        ``True`` when ``-tol <= cut_seconds + removed_seconds -
        epoch_length * len(stages) < epoch_length + tol``, with ``tol`` from
        :attr:`consistency_tolerance`, i.e. the full-night hypnogram ends
        within the last (partial) epoch of the original recording. ``False``
        without stages.
        """
        if not self.stages:
            return False
        tol = self.consistency_tolerance
        diff = self.original_seconds - self.epoch_length * len(self.stages)
        return -tol - 1e-9 <= diff < self.epoch_length + tol

    def fullnight_hypnogram(self):
        """The full-night hypnogram.

        Returns
        -------
        list of str
            A copy of the ``etc.stages`` codes, one per original epoch of
            ``epoch_length`` seconds from the recording start, or ``[]``
            when the map has no stages.
        """
        return list(self.stages or [])

    def stage_intervals_cut(self):
        """Full-night epochs mapped onto the cut time base.

        Returns
        -------
        list of (float, float, str, int)
            For each continuous stretch of cut signal between splices, the
            parts of each original epoch that survived, as cut-file
            ``(start, end, code, orig_epoch)`` where ``orig_epoch`` indexes
            :meth:`fullnight_hypnogram`. Removed data contributes nothing, and
            an original time past the last stage has no label.
        """
        if not self.stages:
            return []
        T = self.cut_seconds
        L = self.epoch_length
        n_stages = len(self.stages)
        edges = np.concatenate(([0.0], self.cut_onsets[(self.cut_onsets > 0)
                                                       & (self.cut_onsets < T)], [T]))
        edges = np.unique(edges)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            shift = self.cut_to_original(a) - a
            i0 = int(np.floor((a + shift) / L))
            i1 = int(np.ceil((b + shift) / L))
            for i in range(max(i0, 0), min(i1, n_stages)):
                lo = max(a, i * L - shift)
                hi = min(b, (i + 1) * L - shift)
                if hi > lo:
                    out.append((lo, hi, self.stages[i], i))
        return out

    def exact_epochs(self):
        """Exact whole-second epochs of the full-night stages on the cut file.

        Returns
        -------
        list of (int, int, str, int)
            :func:`exact_cut_epochs` over :meth:`stage_intervals_cut`, ending
            at :attr:`last_second`.
        """
        return exact_cut_epochs(self.stage_intervals_cut(), self.last_second)

    def compare_stage_events(self, stage_events):
        """Check stage events against the full-night hypnogram.

        Parameters
        ----------
        stage_events : list of (float, str)
            Cut-time onsets and codes from :func:`stage_events_from_header`.

        Returns
        -------
        n_compared : int
            Stage events that fall on an ``etc.stages`` epoch.
        n_disagree : int
            Of those, events whose code differs from the epoch's, or whose
            mapped onset is more than :data:`ONSET_TOLERANCE_S` off the grid.
        """
        if not self.stages or not stage_events:
            return 0, 0
        L = self.epoch_length
        onsets = np.array([s[0] for s in stage_events])
        orig = np.atleast_1d(self.cut_to_original(onsets))
        n_compared = n_disagree = 0
        for (onset, code), u in zip(stage_events, orig):
            idx = int(np.floor((u + ONSET_TOLERANCE_S) / L))
            if idx < 0 or idx >= len(self.stages):
                continue
            n_compared += 1
            off_grid = abs(u - idx * L) > ONSET_TOLERANCE_S
            if off_grid or canonical_stage_code(self.stages[idx]) != \
                    canonical_stage_code(code):
                n_disagree += 1
        return n_compared, n_disagree

    def to_dict(self):
        """Serialisable form; the keys :meth:`to_json` writes.

        Returns
        -------
        dict
            JSON-serialisable: schema, signal and removal totals,
            ``boundaries``, ``fullnight_stages``, ``last_second`` and
            ``cut_epochs``, plus everything in ``meta``.
        """
        d = {
            'schema': SIDECAR_SCHEMA,
            's_freq': self.s_freq,
            'n_samples': self.n_samples,
            'epoch_length': self.epoch_length,
            'cut_seconds': self.cut_seconds,
            'removed_seconds': self.removed_seconds,
            'original_seconds': self.original_seconds,
            'n_boundaries': self.n_boundaries,
            'boundaries': [
                {'cut_onset_s': float(c), 'original_onset_s': float(o),
                 'removed_s': float(r)}
                for c, o, r in zip(self.cut_onsets, self.original_onsets,
                                   self.removed)],
            'fullnight_stages': self.fullnight_hypnogram(),
            'last_second': self.last_second,
            'cut_epochs': [list(e) for e in (self.cut_epochs or [])],
        }
        d.update(self.meta)
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuild from :meth:`to_dict` output; unknown keys go to ``meta``.

        Parameters
        ----------
        d : dict
            A dictionary as returned by :meth:`to_dict` or read from a
            sidecar JSON.

        Returns
        -------
        RecordingTimeline
        """
        known = {'schema', 's_freq', 'n_samples', 'epoch_length', 'cut_seconds',
                 'removed_seconds', 'original_seconds', 'n_boundaries',
                 'boundaries', 'fullnight_stages', 'last_second',
                 'cut_epochs'}
        b = d.get('boundaries', [])
        return cls([x['cut_onset_s'] for x in b], [x['removed_s'] for x in b],
                   d['s_freq'], d['n_samples'],
                   stages=d.get('fullnight_stages'),
                   epoch_length=d.get('epoch_length', 30.0),
                   meta={k: v for k, v in d.items() if k not in known},
                   last_second=d.get('last_second'),
                   cut_epochs=d.get('cut_epochs'))

    def to_json(self, path):
        """Write the sidecar JSON.

        Parameters
        ----------
        path : str or Path
            Output file, conventionally ``<annotation xml stem>_timeline.json``.

        Returns
        -------
        Path
            The path written.
        """
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    @classmethod
    def from_json(cls, path):
        """Read a sidecar written by :meth:`to_json`.

        Parameters
        ----------
        path : str or Path
            The sidecar JSON, conventionally ``<xml stem>_timeline.json``.

        Returns
        -------
        RecordingTimeline
            The map; keys other than the map's own (``source``,
            ``turtlewave_version``, ``annotation_file``...) are in ``meta``.
        """
        with open(path, 'r', encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))


def sidecar_path(annot_file):
    """Path of the timeline sidecar for an annotation file.

    Parameters
    ----------
    annot_file : str or Path
        The Wonambi annotation XML.

    Returns
    -------
    Path
        ``<annotation xml stem>_timeline.json`` in the same folder.
    """
    p = Path(annot_file)
    return p.with_name(f'{p.stem}_timeline.json')


def stage_name_for_code(code):
    """Wonambi stage name for a Compumedics stage code.

    Reads the code exactly as Wonambi's ``import_staging(source='compumedics')``
    reads one line of a staging file (the first two characters of
    ``code + newline``, looked up in ``COMPUMEDICS_STAGE_KEY``), so exact and
    grid imports name every code the same way.

    Parameters
    ----------
    code : object
        Stage code, e.g. ``'2'``, ``'N2'``, ``'R'``, ``'?'``.

    Returns
    -------
    str
        ``'Wake'``, ``'NREM1'``, ``'NREM2'``, ``'NREM3'``, ``'REM'``,
        ``'Undefined'``, or ``'Unknown'`` for an unrecognised code.
    """
    from wonambi.attr.annotations import COMPUMEDICS_STAGE_KEY
    key = (str(code).strip() + '\n')[0:2]
    return COMPUMEDICS_STAGE_KEY.get(key, 'Unknown')


def load_sidecar_for(annot_file, annotations=None, require=True):
    """Load the timeline sidecar of an annotation file and check it.

    Parameters
    ----------
    annot_file : str or Path
        The Wonambi annotation XML.
    annotations : object or None
        An annotation wrapper exposing ``get_stage_intervals()``
        (:class:`~turtlewave_hdEEG.annotation.CustomAnnotations`) already open
        on ``annot_file``. ``None`` opens one, on the sidecar's rater when the
        file has it.
    require : bool
        When True a missing sidecar raises; when False it returns ``None``.

    Returns
    -------
    RecordingTimeline or None
        The map with ``cut_epochs`` and ``last_second`` set, or ``None`` when
        the sidecar is absent and ``require`` is False.

    Raises
    ------
    SidecarMismatchError
        The sidecar is missing (``require`` True), unreadable, older than
        schema 2, written for a different annotation file, or its
        ``cut_epochs`` / ``last_second`` differ from the XML's epochs. The
        message lists up to three differing epochs and says to re-run the
        annotation step.
    """
    annot_file = Path(annot_file)
    path = sidecar_path(annot_file)
    redo = (f"Re-run the annotation step (XLAnnotations.add_stages_from_header) "
            f"to rewrite {annot_file.name} and its sidecar together.")
    if not path.exists():
        if require:
            raise SidecarMismatchError(
                f"No timeline sidecar {path.name} beside {annot_file.name}. "
                f"{redo}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise SidecarMismatchError(
            f"Timeline sidecar {path} cannot be read ({e}). {redo}") from e

    schema = data.get('schema')
    if not isinstance(schema, int) or schema < 2:
        raise SidecarMismatchError(
            f"Timeline sidecar {path.name} has schema {schema!r}; schema 2 or "
            f"later (exact epochs) is required. {redo}")
    named = data.get('annotation_file')
    if named != annot_file.name:
        raise SidecarMismatchError(
            f"Timeline sidecar {path.name} was written for {named!r}, not "
            f"{annot_file.name!r}. {redo}")

    if annotations is None:
        from .annotation import CustomAnnotations
        annotations = CustomAnnotations(str(annot_file))
        rater = data.get('rater')
        if rater and rater in (annotations.raters or []):
            annotations.get_rater(rater)
    xml = [(int(round(s)), int(round(e)), str(st))
           for s, e, st in annotations.get_stage_intervals()]
    side = [(int(e[0]), int(e[1]), stage_name_for_code(e[2]))
            for e in data.get('cut_epochs') or []]

    problems = []
    if len(side) != len(xml):
        problems.append(f"{len(side)} epochs in the sidecar, {len(xml)} in "
                        f"the XML")
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(side, xml)) if a != b]
    for i, a, b in diffs[:3]:
        problems.append(f"epoch {i}: sidecar {a[0]}-{a[1]} s {a[2]}, XML "
                        f"{b[0]}-{b[1]} s {b[2]}")
    if len(diffs) > 3:
        problems.append(f"... {len(diffs) - 3} more differing epochs")
    last = data.get('last_second')
    xml_last = xml[-1][1] if xml else None
    if last is None or (xml_last is not None and int(last) != xml_last):
        problems.append(f"last_second {last} in the sidecar, last epoch end "
                        f"{xml_last} in the XML")
    if problems:
        raise SidecarMismatchError(
            f"Timeline sidecar {path.name} does not match {annot_file.name}: "
            f"{'; '.join(problems)}. The XML was probably rescored or "
            f"re-annotated without the sidecar. {redo}")
    tl = RecordingTimeline.from_dict(data)
    logger.debug(f"Timeline sidecar {path.name} matches {annot_file.name} "
                 f"({len(xml)} epochs)")
    return tl


def nominal_epoch_length(annotations, default=30.0):
    """Epoch length to report for an annotation file.

    Parameters
    ----------
    annotations : object
        Annotation wrapper; read through ``has_uniform_epochs()`` and
        ``epoch_durations()`` when it has them.
    default : float
        Returned for uniform epochs, an empty file, or an object without
        those methods.

    Returns
    -------
    float
        ``default`` for a uniform grid; for variable-length epochs (a cut
        recording's exact epochs) the median epoch duration, which is a
        description of the file, not a length any epoch count may be
        multiplied by.
    """
    if annotations is None or not hasattr(annotations, 'has_uniform_epochs'):
        return float(default)
    try:
        if annotations.has_uniform_epochs():
            return float(default)
        durations = annotations.epoch_durations()
    except Exception:
        return float(default)
    return float(np.median(durations)) if durations else float(default)
