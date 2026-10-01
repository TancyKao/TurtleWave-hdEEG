"""Per-event review figures, computed on the signal the detector saw.

Implements Method Spec revision 3 ("v2" figures) of
``_scratch/research/event-review/method_spec.md``: for every detected event,

* M1 ``halfwaves_above_bg`` (band-passed extrema with ``|y| >= 2.5 x bg_rms``)
  beside ``cycles_nominal`` (sign changes / 2), spindles only;
* M2 ``peak_freq_ap`` / ``prominence_db``: the largest interior peak of the
  event's 4-30 Hz periodogram after removing a log-log (aperiodic) fit, with
  ``in_band``, ``low_prominence`` (< 10 dB), ``coarse`` (< 1 s) and
  ``no_peak``, spindles only; slow waves and K-complexes get ``wave_freq`` =
  1 / (2 x negative half-wave duration) instead;
* M3 ``amp_ratio`` = band RMS of the event over the median band RMS of
  non-overlapping background windows, with ``bg_rms``, ``bg_n_windows`` and
  ``bg_stage_mixed``;
* M4 ``thresh_ratio``, the detector's own peak over its own threshold, only
  where the two are one signal in linear units;
* M6 ``near_bound`` (-1 floor, +1 ceiling, 0 neither).

The figures are computed on the data the detector was handed: the run's
``ref_chan`` re-reference applied, restricted to the run's stages, rejected
time removed, bouts concatenated with splices. Each contiguous run between
splices is band-passed on its own; an event within ``P`` of a run edge, or
spanning one, is ``near_splice`` and gets NULL signal figures, because the
filter and the surround would straddle a discontinuity.

Everything here is numpy/scipy only and imports no Qt, so the detectors, a GUI
recomputation and a backfill can all call the same code.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import NamedTuple, Optional

import numpy as np
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt

logger = logging.getLogger('turtlewave_hdEEG.event_metrics')

#: Method Spec revision tag written to ``detection_runs.params_json``.
SPEC_REVISION = 'v2'

#: Background / guard geometry per event family (seconds). ``S`` surround on
#: each side, ``P`` edge guard (and read padding), ``L`` window length, ``g``
#: guard around events, ``n_min`` fewest windows for a background estimate.
GEOMETRY = {
    'spindle': {'S': 15.0, 'P': 2.0, 'L': 0.5, 'g': 0.25, 'n_min': 10},
    'slow_wave': {'S': 30.0, 'P': 10.0, 'L': 2.0, 'g': 0.5, 'n_min': 8},
}
GEOMETRY['k_complex'] = GEOMETRY['slow_wave']

HALFWAVE_K = 2.5          #: M1 gate, multiples of background band RMS
LOW_PROMINENCE_DB = 10.0  #: M2 ``low_prominence`` cutoff
SEARCH_BAND = (4.0, 30.0)  #: M2 spectrum range (Hz)
COARSE_S = 1.0            #: M2 ``coarse`` when duration is below this (s)
NEAR_BOUND_S = 0.05       #: M6 tolerance (s)
FILTER_ORDER = 2          #: Butterworth order of the band filter
_MASSIMINI = ('Massimini2004', 'AASM/Massimini2004')

#: ``events`` columns written from :meth:`EventFigures.to_row`, in order.
FIGURE_COLUMNS = (
    'halfwaves_above_bg', 'cycles_nominal', 'peak_freq_ap', 'prominence_db',
    'in_band', 'low_prominence', 'bg_rms', 'bg_n_windows', 'bg_stage_mixed',
    'amp_ratio', 'thresh_ratio', 'near_bound', 'near_splice', 'wave_freq',
)


def figure_config(event_type):
    """The settings the figures were computed with, for provenance.

    Parameters
    ----------
    event_type : str
        ``'spindle'``, ``'slow_wave'`` or ``'k_complex'``.

    Returns
    -------
    dict
        Spec revision, geometry and cutoffs; stored in
        ``detection_runs.params_json['event_figures']``.
    """
    geo = _geometry(event_type)
    return {
        'spec_revision': SPEC_REVISION,
        'event_type': event_type,
        'surround_s': geo['S'], 'edge_guard_s': geo['P'],
        'window_s': geo['L'], 'event_guard_s': geo['g'],
        'min_windows': geo['n_min'],
        'halfwave_k': HALFWAVE_K, 'low_prominence_db': LOW_PROMINENCE_DB,
        'search_band_hz': list(SEARCH_BAND), 'coarse_s': COARSE_S,
        'near_bound_s': NEAR_BOUND_S,
        'filter': f'butter({FILTER_ORDER}) bandpass, sosfiltfilt, per '
                  f'contiguous run',
        'split_rule': 'diff(t) > 1.5/fs or diff(t) <= 0.5/fs',
    }


def figure_family(event_type):
    """Geometry family of an event type: ``'spindle'`` or ``'slow_wave'``.

    Parameters
    ----------
    event_type : str
        Any processor event type (``'spindle'``, ``'slow_wave'``,
        ``'k_complex'``, or a custom slow-wave label).

    Returns
    -------
    str
        ``'spindle'`` for spindles, ``'slow_wave'`` for everything else.
    """
    return 'spindle' if event_type == 'spindle' else 'slow_wave'


def _geometry(event_type):
    try:
        return GEOMETRY[event_type]
    except KeyError:
        raise ValueError(f"Unknown event_type {event_type!r}; expected one of "
                         f"{sorted(GEOMETRY)}") from None


# ---------------------------------------------------------------------------
# Elementary figures
# ---------------------------------------------------------------------------

def bandpass(x, fs, band, order=FILTER_ORDER):
    """Zero-phase Butterworth band-pass.

    Parameters
    ----------
    x : array_like
        Signal (1-D).
    fs : float
        Sampling frequency (Hz).
    band : tuple of float
        ``(low, high)`` in Hz, the run's ``freq_lower``-``freq_upper``.
    order : int, optional
        Butterworth order (default 2).

    Returns
    -------
    numpy.ndarray
        Filtered signal, float64.

    Raises
    ------
    ValueError
        If ``x`` is too short for ``sosfiltfilt``'s edge padding.
    """
    sos = butter(order, [float(band[0]), float(band[1])], 'bandpass',
                 fs=float(fs), output='sos')
    return sosfiltfilt(sos, np.asarray(x, dtype='f8'))


def _min_filter_len(band, fs, order=FILTER_ORDER):
    sos = butter(order, [float(band[0]), float(band[1])], 'bandpass',
                 fs=float(fs), output='sos')
    # sosfiltfilt's default padlen, plus one.
    n = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                    (sos[:, 5] == 0).sum()))
    return n + 1


def halfwaves_above_bg(y_event, bg_rms, k=HALFWAVE_K):
    """M1: band-passed extrema inside the event standing out from background.

    Parameters
    ----------
    y_event : array_like
        Band-passed event samples.
    bg_rms : float or None
        Background band RMS (M3 denominator).
    k : float, optional
        Gate in multiples of ``bg_rms`` (default 2.5).

    Returns
    -------
    int or None
        Number of local maxima and minima with ``|y| >= k * bg_rms``; ``None``
        when ``bg_rms`` is missing or not finite.
    """
    if bg_rms is None or not np.isfinite(bg_rms):
        return None
    y = np.asarray(y_event, dtype='f8')
    ext = np.r_[find_peaks(y)[0], find_peaks(-y)[0]]
    return int(np.count_nonzero(np.abs(y[ext]) >= k * bg_rms))


def cycles_nominal(y_event):
    """Ungated cycle count: sign changes of the band-passed event / 2.

    Parameters
    ----------
    y_event : array_like
        Band-passed event samples.

    Returns
    -------
    float
        Half the number of sign changes (to 0.5).
    """
    y = np.asarray(y_event, dtype='f8')
    if y.size < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(np.signbit(y)))) / 2.0


class PeakFreq(NamedTuple):
    """M2 result for one event.

    Attributes
    ----------
    freq : float
        Interior residual peak in Hz; NaN when ``no_peak``.
    prominence_db : float
        Residual at the peak in dB; NaN when ``no_peak``.
    in_band : bool
        ``band[0] <= freq <= band[1]``; False when ``no_peak``.
    low_prominence : bool
        ``prominence_db`` is under the cutoff (10 dB); False when ``no_peak``.
    coarse : bool
        The event is shorter than 1 s, so its frequency resolution is coarse.
    no_peak : bool
        The residual has no interior maximum.
    """

    freq: float            #: interior residual peak (Hz), NaN when no_peak
    prominence_db: float   #: residual at the peak (dB), NaN when no_peak
    in_band: bool          #: ``band[0] <= freq <= band[1]``; False when no_peak
    low_prominence: bool   #: ``prominence_db < 10``; False when no_peak
    coarse: bool           #: event shorter than 1 s
    no_peak: bool          #: no interior maximum of the residual


def peak_frequency_ap(x_event, fs, band, search=SEARCH_BAND,
                      low_prominence_db=LOW_PROMINENCE_DB, coarse_s=COARSE_S):
    """M2: aperiodic-corrected peak frequency of one event.

    Hann periodogram of the mean-removed raw event samples
    (``nfft = max(4 fs, n)``, linear detrend), restricted to ``search``; a
    least-squares line is fitted to ``log10 P`` against ``log10 f`` and the
    peak is the largest interior local maximum of the residual in dB.

    Parameters
    ----------
    x_event : array_like
        Raw (not band-passed) event samples, ``start <= t <= end``.
    fs : float
        Sampling frequency (Hz).
    band : tuple of float
        Run band, for ``in_band``.
    search : tuple of float, optional
        Spectrum range (default 4-30 Hz).
    low_prominence_db : float, optional
        ``low_prominence`` cutoff (default 10 dB).
    coarse_s : float, optional
        ``coarse`` below this duration (default 1 s).

    Returns
    -------
    PeakFreq
        Never raises on short input; a window with fewer than 4 samples or
        no interior maximum returns ``no_peak=True``.
    """
    x = np.asarray(x_event, dtype='f8')
    coarse = bool(x.size / float(fs) < coarse_s)
    none = PeakFreq(np.nan, np.nan, False, False, coarse, True)
    if x.size < 4:
        return none
    seg = x - x.mean()
    f, p = periodogram(seg, fs, window='hann',
                       nfft=max(int(4 * fs), seg.size), detrend='linear')
    m = (f >= search[0]) & (f <= search[1])
    f, p = f[m], p[m]
    if f.size < 3:
        return none
    lf, lp = np.log10(f), np.log10(p + 1e-30)
    resid = 10.0 * (lp - np.polyval(np.polyfit(lf, lp, 1), lf))
    pk, _ = find_peaks(resid)
    if not pk.size:
        return none
    i = pk[np.argmax(resid[pk])]
    freq, prom = float(f[i]), float(resid[i])
    return PeakFreq(freq, prom, bool(band[0] <= freq <= band[1]),
                    bool(prom < low_prominence_db), coarse, False)


def amplitude_ratio(y_event, bg_rms):
    """M3: band RMS of the event over the background band RMS.

    Parameters
    ----------
    y_event : array_like
        Band-passed event samples.
    bg_rms : float or None
        Background band RMS.

    Returns
    -------
    float or None
        ``None`` when ``bg_rms`` is missing, zero or not finite.
    """
    if bg_rms is None or not np.isfinite(bg_rms) or bg_rms <= 0:
        return None
    y = np.asarray(y_event, dtype='f8')
    if not y.size:
        return None
    return float(np.sqrt(np.mean(y * y)) / bg_rms)


def threshold_components(method, event_values, thresholds):
    """M4 ratio components for one event.

    Parameters
    ----------
    method : str
        Per-event detection method.
    event_values : dict
        The event's stored detector values (``peak_val_det``, ``det_trough``,
        ``det_ptp``).
    thresholds : dict or None
        Resolved threshold values ``{name: value}`` of the event's own
        detection segment (``extensions.detection_threshold_values``).

    Returns
    -------
    dict
        ``{threshold name: ratio}`` for each name the method allows a ratio
        for (``THRESHOLD_UNITS[method]['ratio_allowed']``) whose value and
        event value are both present. A ratio that is not finite or not
        positive (a sign mismatch, e.g. a positive trough against a negative
        criterion) is left out rather than reported.
    """
    from .extensions import THRESHOLD_UNITS

    spec = THRESHOLD_UNITS.get(method)
    if not spec or not thresholds:
        return {}
    out = {}
    for name, allowed in spec.get('ratio_allowed', {}).items():
        if not allowed:
            continue
        col = spec.get('ratio_value', {}).get(name)
        thr = thresholds.get(name)
        val = (event_values or {}).get(col) if col else None
        if thr is None or val is None:
            continue
        thr, val = float(thr), float(val)
        if thr == 0 or not np.isfinite(thr) or not np.isfinite(val):
            continue
        ratio = val / thr
        if np.isfinite(ratio) and ratio > 0:
            out[name] = float(ratio)
    return out


def threshold_ratio(method, event_values, thresholds):
    """M4: the event's detector peak over its detection threshold.

    For the Massimini family this is ``min(det_trough / max_trough_amp,
    det_ptp / min_ptp)``, the criterion the wave passed by the smallest
    margin; both criteria are negative/positive in step, so each component is
    positive and >= 1 for a wave that passed. For Moelle2011, Ferrarelli2007
    and Nir2011 it is ``peak_val_det / det_value_lo``. Every other method
    returns ``None`` (the detector peak and threshold are not one signal).

    Parameters
    ----------
    method : str
        Per-event detection method.
    event_values : dict
        The event's detector values.
    thresholds : dict or None
        Resolved thresholds of the event's own segment.

    Returns
    -------
    float or None
        The ratio, or ``None`` when the method allows none or a value is
        missing.
    """
    comps = threshold_components(method, event_values, thresholds)
    return min(comps.values()) if comps else None


def slow_wave_shape(method, start, end, zero_time):
    """M5: negative half-wave duration and the wave frequency it implies.

    Parameters
    ----------
    method : str
        Slow-wave / K-complex method.
    start, end : float
        Event bounds (s).
    zero_time : float or None
        Detector zero crossing (``det_zero_time``).

    Returns
    -------
    dict
        ``neg_halfwave_dur`` (s) -- ``end - zero_time`` for the Massimini
        family (positive half-wave first), ``zero_time - start`` for
        Ngo2015/Staresina2015 -- and ``wave_freq`` = 1 / (2 x that duration)
        in Hz. Both ``None`` when ``zero_time`` is missing or the duration is
        not positive.
    """
    out = {'neg_halfwave_dur': None, 'wave_freq': None}
    if zero_time is None or not np.isfinite(zero_time):
        return out
    if method in _MASSIMINI:
        dur = float(end) - float(zero_time)
    else:
        dur = float(zero_time) - float(start)
    if dur > 0:
        out['neg_halfwave_dur'] = dur
        out['wave_freq'] = 1.0 / (2.0 * dur)
    return out


def duration_bound_flag(duration, bounds, tol=NEAR_BOUND_S):
    """M6: is the duration within ``tol`` of the run's floor or ceiling?

    Parameters
    ----------
    duration : float
        Event duration (s).
    bounds : sequence or None
        ``(min, max)``; either may be ``None`` (no bound).
    tol : float, optional
        Tolerance (default 0.05 s).

    Returns
    -------
    int or None
        ``-1`` near the floor, ``+1`` near the ceiling, ``0`` neither (the
        floor wins if both apply); ``None`` when no bounds are known.
    """
    if bounds is None or duration is None:
        return None
    try:
        lo, hi = bounds[0], bounds[1]
    except (TypeError, IndexError):
        return None
    if lo is None and hi is None:
        return None
    if lo is not None and abs(float(duration) - float(lo)) <= tol:
        return -1
    if hi is not None and abs(float(hi) - float(duration)) <= tol:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Signal tracks: one band-passed copy per contiguous run of samples
# ---------------------------------------------------------------------------

class _Track:
    """A signal split into contiguous runs, each band-passed on its own.

    Runs are found where ``diff(t) > 1.5/fs`` (a gap) or ``diff(t) <=
    0.5/fs`` (time stepping back, as when a concatenation is not in time
    order). Every time is mapped to a global sample index ``rint(t * fs)``,
    which is how background windows are tiled identically whether the data
    is one continuous read or a concatenated detection segment.
    """

    def __init__(self, x, t, fs, band):
        x = np.asarray(x, dtype='f8').ravel()
        t = np.asarray(t, dtype='f8').ravel()
        if x.size != t.size:
            raise ValueError(f"x and t differ in length ({x.size} vs {t.size})")
        self.fs = float(fs)
        self.x = x
        n = x.size
        if n:
            dt = np.diff(t)
            cut = np.flatnonzero((dt > 1.5 / self.fs) | (dt <= 0.5 / self.fs)) + 1
            bounds = np.r_[0, cut, n]
        else:
            bounds = np.array([0, 0])
        xb = np.full(n, np.nan)
        min_len = _min_filter_len(band, self.fs)
        runs = []
        for l0, l1 in zip(bounds[:-1], bounds[1:]):
            if l1 <= l0:
                continue
            g0 = int(np.rint(t[l0] * self.fs))
            filtered = (l1 - l0) >= min_len
            if filtered:
                xb[l0:l1] = bandpass(x[l0:l1], self.fs, band)
            runs.append((g0, g0 + (l1 - l0) - 1, int(l0), filtered))
        runs.sort()
        self.xb = xb
        self.csum = np.r_[0.0, np.cumsum(np.nan_to_num(xb) ** 2)]
        self.run_g0 = np.array([r[0] for r in runs], dtype='i8')
        self.run_g1 = np.array([r[1] for r in runs], dtype='i8')
        self.run_l0 = np.array([r[2] for r in runs], dtype='i8')
        self.run_ok = np.array([r[3] for r in runs], dtype=bool)

    def locate(self, g):
        """Run index containing global sample ``g`` (array), -1 if none."""
        g = np.asarray(g, dtype='i8')
        if not self.run_g0.size:
            return np.full(g.shape, -1)
        r = np.searchsorted(self.run_g0, g, side='right') - 1
        ok = (r >= 0)
        r_safe = np.where(ok, r, 0)
        ok &= g <= self.run_g1[r_safe]
        return np.where(ok, r, -1)

    def local(self, r, g):
        """Local array index of global sample ``g`` in run ``r``."""
        return self.run_l0[r] + (np.asarray(g, dtype='i8') - self.run_g0[r])


def _merge_intervals(intervals):
    ivs = sorted((float(s), float(e)) for s, e in intervals if e > s)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return np.array(out, dtype='f8').reshape(-1, 2)


def _allowed_stage_intervals(stage_epochs, run_stages):
    """Merged ``[start, end)`` intervals whose scored stage is in the run."""
    if not stage_epochs or not run_stages:
        return None
    keep = {str(s) for s in run_stages}
    return _merge_intervals((e[0], e[1]) for e in stage_epochs
                            if str(e[2]) in keep)


def _overlaps_any(ws, we, ivs):
    """Vectorised closed-interval overlap of windows with intervals."""
    hit = np.zeros(ws.shape, dtype=bool)
    if ivs is None or not len(ivs):
        return hit
    ivs = np.asarray(ivs, dtype='f8').reshape(-1, 2)
    near = ivs[(ivs[:, 1] >= ws.min()) & (ivs[:, 0] <= we.max())]
    for s, e in near:
        hit |= (ws <= e) & (we >= s)
    return hit


def background_rms(track, start, end, event_type='spindle', others=None,
                   allowed=None, reject=None):
    """M3 denominator: median band RMS of the background windows.

    Windows of length ``L`` are tiled on the global sample grid from the first
    sample at or after ``start - S`` to the last at or before ``end + S``.
    A window is dropped when it overlaps (a) ``[start - g, end + g]``, (b)
    ``[s - g, e + g]`` of any other event, (c) a reject interval, (d) time
    outside the run's stages, or when its samples are not all present in one
    band-passed run at least ``P`` from that run's edges. Each window's RMS
    is O(1) from the track's cumulative sum of squares.

    Parameters
    ----------
    track : _Track
        Band-passed signal.
    start, end : float
        Event bounds (s).
    event_type : str, optional
        Selects the geometry (:data:`GEOMETRY`).
    others : numpy.ndarray or None, optional
        ``(n, 2)`` start/end of the channel's other events in the run, sorted
        by start.
    allowed : numpy.ndarray or None, optional
        Merged ``[start, end)`` in-stage intervals; ``None`` skips (d).
    reject : numpy.ndarray or None, optional
        ``(n, 2)`` reject intervals; ``None`` skips (c). On a detection
        segment rejected time is simply absent.

    Returns
    -------
    bg_rms : float
        NaN when fewer than ``n_min`` windows remain.
    n_windows : int
        Windows kept.
    stage_mixed : bool
        True when (d) dropped at least one window that every other rule
        would have kept (present in a band-passed run, not excluded, not
        rejected). Out-of-stage time absent from the signal does not count.
    """
    geo = _geometry(event_type)
    fs = track.fs
    w = int(round(geo['L'] * fs))
    g, S, P = geo['g'], geo['S'], geo['P']
    g_lo = int(np.ceil((start - S) * fs - 1e-6))
    g_hi = int(np.floor((end + S) * fs + 1e-6)) + 1  # exclusive
    s = np.arange(g_lo, g_hi - w + 1, w, dtype='i8')
    if not s.size:
        return np.nan, 0, False
    ws, we = s / fs, (s + w - 1) / fs

    excl = (ws <= end + g) & (we >= start - g)
    if others is not None and len(others):
        excl |= _overlaps_any(ws, we, others + np.array([-g, g]))

    stage_bad = np.zeros(s.shape, dtype=bool)
    if allowed is not None:
        if len(allowed):
            idx = np.searchsorted(allowed[:, 0], ws + 0.5 / fs, side='right') - 1
            ok = idx >= 0
            idx_s = np.where(ok, idx, 0)
            ok &= we < allowed[idx_s, 1]
            stage_bad = ~ok
        else:
            stage_bad[:] = True

    rej = _overlaps_any(ws, we, reject) if reject is not None else \
        np.zeros(s.shape, dtype=bool)

    r = track.locate(s)
    r_end = track.locate(s + w - 1)
    avail = (r >= 0) & (r == r_end)
    r_safe = np.where(avail, r, 0)
    avail &= track.run_ok[r_safe]
    run_t0 = track.run_g0[r_safe] / fs
    run_t1 = track.run_g1[r_safe] / fs
    tol = 0.5 / fs
    avail &= (ws >= run_t0 + P - tol) & (we <= run_t1 - P + tol)

    # Mixed only when the stage rule removed a window that would otherwise
    # have been used. On a detection segment out-of-stage time is simply
    # absent (those windows are unavailable anyway), so it is not material.
    mixed = bool(np.any(stage_bad & avail & ~excl & ~rej))
    keep = avail & ~excl & ~stage_bad & ~rej
    n = int(np.count_nonzero(keep))
    if n < geo['n_min']:
        return np.nan, n, mixed
    l0 = track.local(r_safe[keep], s[keep])
    rms = np.sqrt(np.maximum(track.csum[l0 + w] - track.csum[l0], 0.0) / w)
    return float(np.median(rms)), n, mixed


# ---------------------------------------------------------------------------
# One event
# ---------------------------------------------------------------------------

@dataclass
class EventFigures:
    """All figures for one event. ``None`` means not computable.

    Attributes
    ----------
    halfwaves_above_bg : int or None
        Band-passed extrema at least 2.5 times the background RMS (spindles).
    cycles_nominal : float or None
        Sign changes of the band-passed event divided by 2 (spindles).
    peak_freq_ap, prominence_db : float or None
        1/f-corrected peak frequency (Hz) and its prominence (dB), spindles.
    in_band, low_prominence : bool or None
        Peak inside the run band; prominence under 10 dB.
    bg_rms, bg_n_windows, bg_stage_mixed
        Background band RMS (µV), windows kept, and whether an otherwise usable
        window was dropped because it lay outside the run's stages.
    amp_ratio, thresh_ratio : float or None
        Event band RMS over background RMS; detector peak over its threshold.
    near_bound : int or None
        -1 near the duration floor, +1 near the ceiling, 0 neither.
    near_splice : bool or None
        Within the edge guard of a splice or spanning one; signal figures are
        then ``None``.
    wave_freq : float or None
        1 / (2 x negative half-wave duration) in Hz (slow waves, K-complexes).
    coarse, no_peak, bg_insufficient, neg_halfwave_dur, thresh_components, ref_note
        Display-only values, not stored.
    """

    halfwaves_above_bg: Optional[int] = None
    cycles_nominal: Optional[float] = None
    peak_freq_ap: Optional[float] = None
    prominence_db: Optional[float] = None
    in_band: Optional[bool] = None
    low_prominence: Optional[bool] = None
    bg_rms: Optional[float] = None
    bg_n_windows: Optional[int] = None
    bg_stage_mixed: Optional[bool] = None
    amp_ratio: Optional[float] = None
    thresh_ratio: Optional[float] = None
    near_bound: Optional[int] = None
    near_splice: Optional[bool] = None
    wave_freq: Optional[float] = None
    # Display-only, not stored.
    coarse: Optional[bool] = None
    no_peak: Optional[bool] = None
    bg_insufficient: Optional[bool] = None
    neg_halfwave_dur: Optional[float] = None
    thresh_components: dict = field(default_factory=dict)
    ref_note: Optional[str] = None

    def to_row(self):
        """The stored subset, keyed by ``events`` column.

        Returns
        -------
        dict
            :data:`FIGURE_COLUMNS` -> value; booleans as 0/1, NaN as ``None``.
        """
        row = {}
        for col in FIGURE_COLUMNS:
            v = getattr(self, col)
            if isinstance(v, (bool, np.bool_)):
                v = int(bool(v))
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v) if np.isfinite(v) else None
            row[col] = v
        return row

    def as_dict(self):
        """All fields, for display."""
        return asdict(self)


def _event_samples(track, start, end):
    """Run index and local slice of the event, or the splice condition.

    Returns
    -------
    tuple
        ``(status, r, l0, l1)``; status is ``'ok'``, ``'near'`` (within P of a
        run edge) or ``'spans'`` (not inside one run).
    """
    fs = track.fs
    e0 = int(np.ceil(start * fs - 1e-6))
    e1 = int(np.floor(end * fs + 1e-6))
    if e1 < e0:
        e1 = e0
    r0, r1 = (int(v) for v in track.locate(np.array([e0, e1])))
    if r0 < 0 or r0 != r1 or not track.run_ok[r0]:
        return 'spans', r0, None, None
    return 'ok', r0, int(track.local(r0, e0)), int(track.local(r0, e1)) + 1


def _figures_on_track(track, start, end, band, event_type, others, allowed,
                      reject, method=None, event_values=None, thresholds=None,
                      duration_bounds=None, ref_note=None):
    geo = _geometry(event_type)
    fig = EventFigures(ref_note=ref_note)
    duration = float(end) - float(start)
    fig.near_bound = duration_bound_flag(duration, duration_bounds)
    if method is not None:
        fig.thresh_components = threshold_components(
            method, event_values, thresholds)
        fig.thresh_ratio = (min(fig.thresh_components.values())
                            if fig.thresh_components else None)
    is_spindle = event_type == 'spindle'
    if is_spindle:
        fig.coarse = bool(duration < COARSE_S)
    else:
        shape = slow_wave_shape(method, start, end,
                                (event_values or {}).get('det_zero_time'))
        fig.neg_halfwave_dur = shape['neg_halfwave_dur']
        fig.wave_freq = shape['wave_freq']
        if fig.wave_freq is not None:
            fig.in_band = bool(band[0] <= fig.wave_freq <= band[1])

    status, r, l0, l1 = _event_samples(track, start, end)
    if status == 'ok':
        fs, P = track.fs, geo['P']
        t_r0 = track.run_g0[r] / fs
        t_r1 = track.run_g1[r] / fs
        if start - t_r0 < P - 0.5 / fs or t_r1 - end < P - 0.5 / fs:
            status = 'near'
    fig.near_splice = status != 'ok'
    if fig.near_splice:
        return fig

    y = track.xb[l0:l1]
    bg, n, mixed = background_rms(track, start, end, event_type, others,
                                  allowed, reject)
    fig.bg_n_windows = n
    fig.bg_stage_mixed = mixed
    fig.bg_insufficient = not np.isfinite(bg)
    fig.bg_rms = float(bg) if np.isfinite(bg) else None
    fig.amp_ratio = amplitude_ratio(y, fig.bg_rms)
    if is_spindle:
        fig.halfwaves_above_bg = halfwaves_above_bg(y, fig.bg_rms)
        fig.cycles_nominal = cycles_nominal(y)
        pf = peak_frequency_ap(track.x[l0:l1], track.fs, band)
        fig.no_peak = pf.no_peak
        if not pf.no_peak:
            fig.peak_freq_ap = pf.freq
            fig.prominence_db = pf.prominence_db
            fig.in_band = pf.in_band
            fig.low_prominence = pf.low_prominence
    return fig


def _others_array(others):
    if others is None or not len(others):
        return None
    arr = np.asarray(others, dtype='f8').reshape(-1, 2)
    return arr[np.argsort(arr[:, 0])]


def event_figures(x, t, fs, start, end, band, others=(), stage_epochs=None,
                  run_stages=None, thresholds=None, event_type='spindle',
                  ref_note=None, *, method=None, event_values=None,
                  duration_bounds=None, reject_intervals=None):
    """All figures for one event from a single-channel read.

    The entry point for a GUI recomputation or a backfill: pass the event
    channel over ``[start - S - P, end + S + P]`` (or a longer span), already
    re-referenced to the run's ``ref_chan``. The read may contain gaps; each
    contiguous run is band-passed on its own.

    Parameters
    ----------
    x : array_like
        Channel samples (µV).
    t : array_like
        Sample times (s from recording start), same length.
    fs : float
        Sampling frequency (Hz).
    start, end : float
        Event bounds (s).
    band : tuple of float
        Run band ``(freq_lower, freq_upper)``.
    others : sequence of (float, float), optional
        Other events of the same run on this channel.
    stage_epochs : sequence of (start, end, stage) or None, optional
        Scored epochs; with ``run_stages`` drives exclusion (d).
    run_stages : sequence of str or None, optional
        The run's stages; ``None`` or empty skips exclusion (d).
    thresholds : dict or None, optional
        Resolved thresholds of the event's own detection segment.
    event_type : str, optional
        ``'spindle'`` (default), ``'slow_wave'`` or ``'k_complex'``.
    ref_note : str or None, optional
        Free text describing the reference the data is in (display only).
    method : str or None, keyword-only
        Detection method, for ``thresh_ratio`` and the slow-wave half-wave.
    event_values : dict or None, keyword-only
        The event's detector values (``peak_val_det``, ``det_trough``,
        ``det_ptp``, ``det_zero_time``).
    duration_bounds : sequence or None, keyword-only
        ``(min, max)`` duration bounds for ``near_bound``.
    reject_intervals : sequence of (float, float) or None, keyword-only
        The run's reject-type intervals, exclusion (c).

    Returns
    -------
    EventFigures
        The figures of this event; fields that cannot be computed are
        ``None``.
    """
    track = _Track(x, t, fs, band)
    return _figures_on_track(
        track, float(start), float(end), band, event_type,
        _others_array(others), _allowed_stage_intervals(stage_epochs, run_stages),
        _others_array(reject_intervals), method=method,
        event_values=event_values, thresholds=thresholds,
        duration_bounds=duration_bounds, ref_note=ref_note)


def _segment_arrays(seg):
    """``(x, t, fs)`` of a single-channel Wonambi segment dict."""
    data = seg['data'] if isinstance(seg, dict) else seg
    x = np.asarray(data.data[0])
    if x.ndim == 2:
        x = x[0]
    t = np.asarray(data.axis['time'][0])
    return x, t, float(data.s_freq)


def channel_event_figures(segments, events, band, event_type, stage_epochs,
                          run_stages, thresholds_by_seg=None):
    """Figures for every event of one channel, on the detector's own segments.

    Called once per channel after every method has run (so the full event
    list, needed for background exclusion (b), is known) and before the
    events are written. Each segment is split into contiguous runs and
    band-passed once; per-event work is O(event length + windows).

    Parameters
    ----------
    segments : sequence
        The fetched Wonambi segments the detector ran on (``seg['data']`` a
        single-channel ChanTime, re-referenced, not detrended).
    events : list of dict
        The channel's direct-DB events (all methods). Each carries
        ``start_time``, ``end_time``, ``method``, the detector values, and
        the private keys ``_seg_idx`` (index into ``segments``),
        ``_thresholds`` (resolved threshold values of that segment) and
        ``_duration_bounds``.
    band : tuple of float
        Run band.
    event_type : str
        ``'spindle'``, ``'slow_wave'`` or ``'k_complex'``.
    stage_epochs : sequence of (start, end, stage)
        Scored epochs (the processor's epoch lookup).
    run_stages : sequence of str or None
        The run's stages.
    thresholds_by_seg : dict or None, optional
        ``{(method, seg_idx): thresholds}``, used when an event carries no
        ``_thresholds``.

    Returns
    -------
    list of dict
        One :meth:`EventFigures.to_row` dict per event, aligned with
        ``events``.
    """
    others = _others_array([(float(e['start_time']), float(e['end_time']))
                            for e in events])
    allowed = _allowed_stage_intervals(stage_epochs, run_stages)
    tracks = {}
    rows = []
    for i, ev in enumerate(events):
        si = int(ev.get('_seg_idx', 0))
        if si not in tracks:
            x, t, fs = _segment_arrays(segments[si])
            tracks[si] = _Track(x, t, fs, band)
        start, end = float(ev['start_time']), float(ev['end_time'])
        thr = ev.get('_thresholds')
        if thr is None and thresholds_by_seg:
            thr = thresholds_by_seg.get((ev.get('method'), si))
        # The event itself is in `others`; its own [start - g, end + g]
        # exclusion already covers it, so it is not removed.
        fig = _figures_on_track(
            tracks[si], start, end, band, event_type, others, allowed, None,
            method=ev.get('method'), event_values=ev, thresholds=thr,
            duration_bounds=ev.get('_duration_bounds'))
        rows.append(fig.to_row())
    return rows


def apply_channel_figures(segments, events, band, event_type, stage_epochs,
                          run_stages, log=None, channel=None):
    """Processor hook: compute and merge the figures into a channel's events.

    Wraps :func:`channel_event_figures` for the three ``Paral*`` processors.
    A failure here is logged at ERROR level with a traceback and leaves the
    figure columns NULL; it never costs the channel its events.

    Parameters
    ----------
    segments, events, band, stage_epochs, run_stages
        As for :func:`channel_event_figures`. ``events`` is updated in place.
    event_type : str
        The processor's event type; mapped with :func:`figure_family`.
    log : logging.Logger or None, optional
        Processor logger (defaults to this module's).
    channel : str or None, optional
        Channel name, for messages.

    Returns
    -------
    dict
        ``seconds`` (wall time), ``n_events``, ``n_near_splice`` and ``ok``.
    """
    import time

    log = log or logger
    t0 = time.perf_counter()
    summary = {'seconds': 0.0, 'n_events': len(events), 'n_near_splice': 0,
               'ok': True}
    if not events:
        return summary
    try:
        rows = channel_event_figures(segments, events, band,
                                     figure_family(event_type),
                                     stage_epochs, run_stages)
        for ev, row in zip(events, rows):
            ev.update(row)
        summary['n_near_splice'] = sum(1 for r in rows if r['near_splice'])
    except Exception as exc:
        summary['ok'] = False
        log.error(f"Event figures failed on channel {channel} ({exc}); its "
                  f"{len(events)} events are written with NULL figures.",
                  exc_info=True)
    summary['seconds'] = time.perf_counter() - t0
    if summary['ok']:
        log.info(f"Event figures for {len(events)} events on channel "
                 f"{channel} in {summary['seconds']:.2f} s; "
                 f"{summary['n_near_splice']} near a splice (NULL figures)")
    return summary
