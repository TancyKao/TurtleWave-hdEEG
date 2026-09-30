"""
Time map between a cut EEGLAB recording and the original night.

When data is removed from an EEGLAB dataset, EEGLAB splices the remaining
samples together and leaves a ``boundary`` event at each splice whose
``duration`` is the number of samples removed there. The signal file then
runs on a *cut* time base while a hypnogram scored on the full night (for
example ``EEG.etc.stages`` from a Compumedics export) runs on the *original*
time base. This module converts between the two, relabels a regular 30 s
grid on the cut file, and saves the map as a JSON sidecar so steps that have
no dataset (sleep cycles, stage durations) can still use the full night.

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
SIDECAR_SCHEMA = 1


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


def stage_event_intervals(stage_events, T, epoch_length=30.0):
    """Label intervals from stage events, each ending at the next onset.

    Parameters
    ----------
    stage_events : list of (float, str)
        Output of :func:`stage_events_from_header`.
    T : float
        Cut-file duration in seconds.
    epoch_length : float
        Nominal stage-event length in seconds.

    Returns
    -------
    list of (float, float, str)
        ``[onset_i, min(onset_i + epoch_length, onset_{i+1}, T))`` with its
        code; empty intervals are dropped.
    """
    out = []
    for i, (onset, code) in enumerate(stage_events):
        end = onset + epoch_length
        if i + 1 < len(stage_events):
            end = min(end, stage_events[i + 1][0])
        end = min(end, T)
        start = max(onset, 0.0)
        if end > start:
            out.append((start, end, code))
    return out


def regrid_stages(label_intervals, T, epoch_length=30.0, undefined=UNDEFINED_CODE):
    """Label a regular epoch grid on ``[0, T)`` from arbitrary label intervals.

    Parameters
    ----------
    label_intervals : iterable of (float, float, str)
        ``(start, end, label)`` in seconds on the grid's time base.
    T : float
        Recording duration in seconds.
    epoch_length : float
        Grid epoch length in seconds.
    undefined : str
        Label for epochs that cannot be labelled.

    Returns
    -------
    list of str
        One label per epoch, ``ceil(T / epoch_length)`` of them. Each full
        epoch takes the label with the largest total overlap; a tie goes to
        the label whose overlap starts earlier in the epoch; an epoch less
        than half covered by any label is ``undefined``. The final partial
        epoch (when ``T`` is not a multiple of ``epoch_length``) is always
        ``undefined``.
    """
    L = float(epoch_length)
    n_full = int(np.floor(T / L + 1e-9))
    n_total = int(np.ceil(T / L - 1e-9))
    acc = [dict() for _ in range(n_full)]
    for start, end, label in sorted(label_intervals, key=lambda x: x[0]):
        s = max(float(start), 0.0)
        e = min(float(end), n_full * L)
        if e <= s:
            continue
        for k in range(int(s // L), min(int(np.ceil(e / L)), n_full)):
            lo, hi = max(s, k * L), min(e, (k + 1) * L)
            if hi <= lo:
                continue
            total, first = acc[k].get(label, (0.0, lo))
            acc[k][label] = (total + (hi - lo), min(first, lo))
    out = []
    for k in range(n_total):
        if k >= n_full or not acc[k]:
            out.append(undefined)
            continue
        covered = sum(v[0] for v in acc[k].values())
        if covered < 0.5 * L - 1e-9:
            out.append(undefined)
            continue
        best = max(acc[k].items(),
                   key=lambda kv: (round(kv[1][0], 9), -kv[1][1]))
        out.append(best[0])
    return out


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
                 epoch_length=30.0, meta=None):
        self.cut_onsets = np.asarray(cut_onsets, dtype=float).ravel()
        self.removed = np.asarray(removed, dtype=float).ravel()
        if self.cut_onsets.shape != self.removed.shape:
            raise ValueError('cut_onsets and removed must have the same length')
        self.s_freq = float(s_freq)
        self.n_samples = int(n_samples)
        self.stages = None if stages is None else [str(s) for s in stages]
        self.epoch_length = float(epoch_length)
        self.meta = dict(meta or {})
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
        list of (float, float, str)
            For each continuous stretch of cut signal between splices, the
            parts of each original epoch that survived, as cut-file
            ``(start, end, code)``. Removed data contributes nothing, and an
            original time past the last stage has no label.
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
                    out.append((lo, hi, self.stages[i]))
        return out

    def cut_hypnogram(self):
        """Full-night stages relabelled onto the cut file's 30 s grid.

        Returns
        -------
        list of str
            ``ceil(cut_seconds / epoch_length)`` codes, from
            :func:`regrid_stages` over :meth:`stage_intervals_cut`.
        """
        return regrid_stages(self.stage_intervals_cut(), self.cut_seconds,
                             self.epoch_length)

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
        """Serialisable form; the keys :meth:`to_json` writes."""
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
        }
        d.update(self.meta)
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuild from :meth:`to_dict` output; unknown keys go to ``meta``."""
        known = {'schema', 's_freq', 'n_samples', 'epoch_length', 'cut_seconds',
                 'removed_seconds', 'original_seconds', 'n_boundaries',
                 'boundaries', 'fullnight_stages'}
        b = d.get('boundaries', [])
        return cls([x['cut_onset_s'] for x in b], [x['removed_s'] for x in b],
                   d['s_freq'], d['n_samples'],
                   stages=d.get('fullnight_stages'),
                   epoch_length=d.get('epoch_length', 30.0),
                   meta={k: v for k, v in d.items() if k not in known})

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
            ``turtlewave_version``, ``cut_stages``...) are in ``meta``.
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
