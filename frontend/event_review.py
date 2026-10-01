"""Text, numbers and rules for per-event review in the review GUI.

Qt-free and library-optional: everything the Channels-tab population checks,
the Event panel and the decision status lines show is built here, so it can be
tested headless and the widgets in ``eeg_review_gui.py`` only render it.

Sources: UX spec ``_scratch/design/event-decision-spec.md`` (revision 2) and
Method Spec ``_scratch/research/event-review/method_spec.md`` (revision 3).
"""

import datetime as _dt
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------

#: ``(key, token, combo label, short label for status lines, tooltip)``.
#: Keys 1-9 follow the UX spec's numbering; the library vocabulary
#: (``dbwrite.REVIEW_REASONS``, Method Spec M7) has two more tokens,
#: ``not-isolated`` and ``wrong-morphology``, which have no digit and are
#: picked from the combo. ``arousal`` replaced the spec's ``arousal-alpha``
#: (M7: posterior alpha without an arousal is ``off-band``).
REASONS = (
    ('1', 'artefact', 'Artefact (movement, electrode, muscle)', 'artefact',
     'Rejects this event only. To also remove the time from the density '
     'denominator, brush it on the trace and use Mark as artefact.'),
    ('2', 'arousal', 'Arousal (EEG speed-up, often with EMG)', 'arousal', ''),
    ('3', 'too-short', 'Too short', 'too short', ''),
    ('4', 'filter-ringing', 'Filter ringing (step or spike in raw)',
     'filter ringing', ''),
    ('5', 'eye-movement', 'Eye movement', 'eye movement', ''),
    ('6', 'not-in-raw', 'Not visible in the raw trace',
     'not visible in the raw trace', ''),
    ('7', 'off-band', 'Outside the frequency band',
     'outside the frequency band', ''),
    ('8', 'single-channel', 'Only on this channel', 'only on this channel', ''),
    ('9', 'other', 'Other (describe in the comment)', 'other', ''),
    ('', 'not-isolated', 'Not isolated from other waves', 'not isolated', ''),
    ('', 'wrong-morphology', 'Wrong shape for this event type',
     'wrong shape', ''),
)
REASON_BY_DIGIT = {r[0]: r[1] for r in REASONS if r[0]}
REASON_SHORT = {r[1]: r[3] for r in REASONS}
REASON_LABEL = {r[1]: r[2] for r in REASONS}

DECISION_PAST = {'accept': 'accepted', 'reject': 'rejected',
                 'unsure': 'unsure'}
DECISION_TITLE = {'accept': 'Accepted', 'reject': 'Rejected',
                  'unsure': 'Unsure'}

EVENT_PLURAL = {'spindle': 'spindles', 'slow_wave': 'slow waves',
                'k_complex': 'K-complexes', 'pac': 'PAC events'}
EVENT_SINGULAR = {'spindle': 'spindle', 'slow_wave': 'slow wave',
                  'k_complex': 'K-complex', 'pac': 'PAC event'}

MINUS = '−'


def decision_word(decision, reason=None):
    """``not reviewed`` / ``accepted`` / ``rejected (artefact)`` / ``unsure``."""
    if not decision:
        return 'not reviewed'
    word = DECISION_PAST.get(decision, str(decision))
    if reason:
        word += f" ({REASON_SHORT.get(reason, reason)})"
    return word


# ---------------------------------------------------------------------------
# Number formats (spec section 4)
# ---------------------------------------------------------------------------

def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_uv(v):
    """µV: one decimal when |v| < 100, integer otherwise, U+2212 minus."""
    f = _finite(v)
    if f is None:
        return '—'
    s = f"{f:.1f}" if abs(f) < 100 else f"{f:.0f}"
    return s.replace('-', MINUS)


def fmt_num(v, nd=2):
    f = _finite(v)
    if f is None:
        return '—'
    return f"{f:.{nd}f}".replace('-', MINUS)


def fmt_hms1(seconds):
    """``HH:MM:SS.s`` of recording time."""
    f = _finite(seconds)
    if f is None:
        return '—'
    tenths = int(round(f * 10))
    s, d = divmod(tenths, 10)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{d}"


def fmt_when(iso, today=None):
    """``HH:MM`` today, else ``YYYY-MM-DD HH:MM``."""
    if not iso:
        return '—'
    try:
        t = _dt.datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)[:16]
    today = today or _dt.date.today()
    return (t.strftime('%H:%M') if t.date() == today
            else t.strftime('%Y-%m-%d %H:%M'))


# ---------------------------------------------------------------------------
# Population checks (spec section 1)
# ---------------------------------------------------------------------------

#: ``column -> (table header, topography combo label)``.
CHECK_COLUMNS = {
    'pct_off_band': ('off-band %', 'off-band (% of events)'),
    'pct_low_prom': ('low prom. %', 'low prominence (% of events)'),
    'pct_dur_floor': ('at floor %', 'at the duration floor (% of events)'),
    'med_amp_ratio': ('amp/bg ×', 'amp. vs background (median ×)'),
    'med_thresh_ratio': ('amp/thr ×', 'amp. vs threshold (median ×)'),
}
SHARE_COLUMNS = ('pct_off_band', 'pct_low_prom', 'pct_dur_floor')
RATIO_COLUMNS = ('med_amp_ratio', 'med_thresh_ratio')
#: ``value column -> the n column it is computed over``.
CHECK_N = {'pct_off_band': 'n_freq', 'pct_low_prom': 'n_prom',
           'pct_dur_floor': 'n_bound', 'med_amp_ratio': 'n_amp',
           'med_thresh_ratio': 'n_thresh'}
CHECK_MIN_N = 20          #: fewer events with a value -> '—', never flagged
SHARE_GUARD = 10.0        #: percentage points from the montage median
RATIO_GUARD = 0.3         #: × from the montage median (provisional)

#: Methods whose stored detector peak and threshold are one signal.
_RATIO_METHODS_FALLBACK = ('Moelle2011', 'Ferrarelli2007', 'Nir2011',
                           'Massimini2004', 'AASM/Massimini2004')

NOT_RECORDED_RUN = ('Event checks are not recorded for this run (detected '
                    'with 4.5 or earlier). Re-detect with 4.6, or run the '
                    'event-figures backfill example, to add them.')
THRESHOLD_NOT_RECORDED = ('Threshold not recorded for this run (detected '
                          'with 4.5 or earlier)')
THRESHOLD_NOT_RECORDED_TIP = ('Re-run detection with TurtleWave 4.6 or later '
                              'to record the detector\'s thresholds. Nothing '
                              'about this event is wrong.')
TOO_FEW_TIP = 'Too few events on this channel to judge (n = {n}).'


def method_has_ratio(method):
    """Whether :data:`THRESHOLD_UNITS` allows an amp/threshold ratio."""
    try:
        from turtlewave_hdEEG.extensions import THRESHOLD_UNITS
    except ImportError:
        return str(method) in _RATIO_METHODS_FALLBACK
    spec = THRESHOLD_UNITS.get(str(method))
    return bool(spec) and any(spec.get('ratio_allowed', {}).values())


def no_ratio_note(method):
    """Tooltip for a ``—`` amp/thr cell on a 4.6 run of ``method``."""
    m = str(method)
    if m == 'CIRUS':
        return 'Not available for CIRUS: the detector does not return a threshold.'
    if m in ('Ngo2015', 'Staresina2015'):
        return (f'No ratio for {m}: the per-channel threshold is not returned '
                f'by the detector.')
    return (f'No ratio for {m}: the stored peak and the detection threshold '
            f'are not the same signal.')


def pool_population(summary, medians=None):
    """Per-channel population figures from ``event_population_summary``.

    Shares are pooled over the stages exactly (``sum(share x n) / sum(n)``,
    each over its own ``n_*`` denominator). Medians cannot be pooled from
    per-stage medians, so they come from ``medians`` (``channel,
    med_amp_ratio, med_thresh_ratio``, computed over the channel's events);
    without it they fall back to the stage median of the largest stage.

    Returns
    -------
    pandas.DataFrame
        ``channel, n, n_no_peak, n_freq, pct_off_band, n_prom, pct_low_prom,
        n_bound, pct_dur_floor, n_amp, med_amp_ratio, n_thresh,
        med_thresh_ratio``; shares in percent.
    """
    cols = ['channel', 'n', 'n_no_peak', 'n_freq', 'pct_off_band', 'n_prom',
            'pct_low_prom', 'n_bound', 'pct_dur_floor', 'n_amp',
            'med_amp_ratio', 'n_thresh', 'med_thresh_ratio']
    if summary is None or len(summary) == 0:
        return pd.DataFrame(columns=cols)
    s = summary.copy()
    for c in ('n', 'n_near_splice', 'n_freq', 'n_prom', 'n_bound', 'n_amp',
              'n_thresh'):
        s[c] = pd.to_numeric(s[c], errors='coerce').fillna(0)
    for share, n in (('share_off_band', 'n_freq'),
                     ('share_low_prom', 'n_prom'),
                     ('share_at_floor', 'n_bound')):
        s['_w_' + share] = pd.to_numeric(s[share], errors='coerce').fillna(0) \
            * s[n]
    g = s.groupby('channel', sort=True)
    out = g[['n', 'n_near_splice', 'n_freq', 'n_prom', 'n_bound', 'n_amp',
             'n_thresh']].sum()
    for share, n, col in (('share_off_band', 'n_freq', 'pct_off_band'),
                          ('share_low_prom', 'n_prom', 'pct_low_prom'),
                          ('share_at_floor', 'n_bound', 'pct_dur_floor')):
        w = g['_w_' + share].sum()
        out[col] = np.where(out[n] > 0, 100.0 * w / out[n].where(out[n] > 0),
                            np.nan)
    out['n_no_peak'] = (out['n'] - out['n_near_splice'] - out['n_freq']).clip(
        lower=0)
    if medians is not None and len(medians):
        m = medians.set_index('channel')
        out['med_amp_ratio'] = m['med_amp_ratio'].reindex(out.index)
        out['med_thresh_ratio'] = m['med_thresh_ratio'].reindex(out.index)
    else:
        big = s.sort_values('n').groupby('channel').tail(1).set_index('channel')
        out['med_amp_ratio'] = big['amp_ratio_median'].reindex(out.index)
        out['med_thresh_ratio'] = big['thresh_ratio_median'].reindex(out.index)
    out = out.reset_index()
    for c in ('n', 'n_no_peak', 'n_freq', 'n_prom', 'n_bound', 'n_amp',
              'n_thresh'):
        out[c] = out[c].astype(int)
    return out[cols]


def _one_sided_z(x, higher_is_bad):
    """Robust one-sided z of each finite value against the finite median."""
    x = np.asarray(x, dtype=float)
    z = np.zeros(len(x))
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return z, np.nan
    v = x[ok]
    med = float(np.median(v))
    scale = 1.4826 * float(np.median(np.abs(v - med)))
    if not scale > 0:
        scale = 1.253314 * float(np.mean(np.abs(v - med)))
    if not scale > 0:
        return z, med
    d = (x - med) if higher_is_bad else (med - x)
    z[ok] = np.maximum(d[ok], 0.0) / scale
    return z, med


def population_flags(pop, hard_z=3.5, soft_z=2.0, event_type='spindle',
                     ratio_allowed=True):
    """Add ``z_*``, ``flag_*``, ``checks_flag`` and montage medians.

    A column is flagged for a channel when its one-sided robust z exceeds
    ``soft_z`` / ``hard_z`` AND the difference from the montage median is at
    least :data:`SHARE_GUARD` points (shares) or :data:`RATIO_GUARD` (ratios).
    Channels with fewer than :data:`CHECK_MIN_N` events with a value get NaN
    for that column (shown ``—``) and no flag. ``pct_low_prom`` is spindle
    only; ``med_thresh_ratio`` only when the method allows a ratio.

    Returns
    -------
    (pandas.DataFrame, dict)
        The frame, and ``{column: montage median}``.
    """
    df = pop.copy()
    medians = {}
    df['checks_flag'] = ''
    reasons = [[] for _ in range(len(df))]
    for col in CHECK_COLUMNS:
        n = df[CHECK_N[col]].to_numpy(dtype=float) if len(df) else np.array([])
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float) \
            if len(df) else np.array([])
        applicable = not ((col == 'pct_low_prom' and event_type != 'spindle')
                          or (col == 'med_thresh_ratio' and not ratio_allowed))
        if not applicable:
            vals = np.full(len(df), np.nan)
        vals = np.where(n >= CHECK_MIN_N, vals, np.nan)
        df[col] = vals
        higher_bad = col in SHARE_COLUMNS
        z, med = _one_sided_z(vals, higher_bad)
        medians[col] = med
        guard = SHARE_GUARD if higher_bad else RATIO_GUARD
        diff = np.abs(vals - med) if np.isfinite(med) else np.zeros(len(df))
        flag = np.array([''] * len(df), dtype=object)
        ok = np.isfinite(vals) & (diff >= guard - 1e-9)
        flag[ok & (z > soft_z)] = 'soft'
        flag[ok & (z > hard_z)] = 'hard'
        df['z_' + col] = z
        df['flag_' + col] = flag
        for i, f in enumerate(flag):
            if f:
                reasons[i].append(f"{col} z={z[i]:.1f}")
    if len(df):
        fl = df[['flag_' + c for c in CHECK_COLUMNS]].to_numpy()
        df['checks_flag'] = np.where((fl == 'hard').any(axis=1), 'hard',
                                     np.where((fl == 'soft').any(axis=1),
                                              'soft', ''))
    df['checks_reasons'] = ['; '.join(r) for r in reasons]
    return df, medians


def fmt_check(col, v):
    """Cell text of one check column: ``34 %`` or ``1.8×``; ``—`` missing."""
    f = _finite(v)
    if f is None:
        return '—'
    if col in SHARE_COLUMNS:
        return f"{f:.0f} %"
    return f"{f:.1f}×"


_PHRASE = {
    'pct_off_band': '{p} % of {events} off-band',
    'pct_low_prom': '{p} % low prominence',
    'pct_dur_floor': '{p} % at the duration floor',
    'med_amp_ratio': 'amplitude {r}× background (median)',
    'med_thresh_ratio': 'amplitude {r}× threshold (median)',
}


def dock_check_items(row, event_type, medians, limit=3):
    """Flagged phrases for the dock line, hard first then by z, at most 3.

    Returns
    -------
    list of dict
        ``key`` (check column), ``text``, ``severity`` ('hard'|'soft'),
        ``tooltip``.
    """
    items = []
    for col in CHECK_COLUMNS:
        f = str(row.get('flag_' + col, '') or '')
        if f not in ('hard', 'soft'):
            continue
        v = _finite(row.get(col))
        if v is None:
            continue
        text = _PHRASE[col].format(
            p=f"{v:.0f}", r=f"{v:.1f}",
            events=EVENT_PLURAL.get(event_type, event_type))
        med = medians.get(col)
        mtxt = fmt_check(col, med)
        items.append({'key': col, 'text': text, 'severity': f,
                      'z': float(row.get('z_' + col, 0) or 0),
                      'tooltip': f"Montage median {mtxt}; robust z "
                                 f"{float(row.get('z_' + col, 0) or 0):.1f}."})
    items.sort(key=lambda d: (d['severity'] != 'hard', -d['z']))
    return items[:limit]


DOCK_HINT = ('Most events failing? Drop channel. Run band wrong for this site? '
             'Add to re-detect queue. Unsure? Click a figure to look at those '
             'events.')


def dock_check_line(channel, items, recorded=True):
    """Plain text of the dock line (the widget renders the same as links)."""
    if not recorded:
        return f"{channel}: event checks not recorded for this run"
    if not items:
        return f"{channel}: event checks in line with the rest of the montage"
    return f"{channel}: " + ' · '.join(d['text'] for d in items)


def filter_chip_text(col, event_type, n_shown, n_total, ratio=None):
    """Epochs-tab chip text for a check filter."""
    ev = EVENT_PLURAL.get(event_type, event_type)
    if col == 'pct_off_band':
        head = f"Showing off-band {ev} only"
    elif col == 'pct_low_prom':
        head = f"Showing low-prominence {ev} only"
    elif col == 'pct_dur_floor':
        head = f"Showing at-floor {ev} only"
    elif col == 'med_amp_ratio':
        head = f"Showing {ev} with amplitude below {ratio:.1f}× background"
    else:
        head = f"Showing {ev} with amplitude below {ratio:.1f}× threshold"
    return f"{head} ({n_shown} of {n_total}) ✕"


def failing_mask(df, col, ratio=None):
    """Events of a drilled slice that fail check ``col`` (bool Series)."""
    def num(c):
        return (pd.to_numeric(df[c], errors='coerce') if c in df.columns
                else pd.Series(np.nan, index=df.index))
    if col == 'pct_off_band':
        return num('in_band') == 0
    if col == 'pct_low_prom':
        return num('low_prominence') == 1
    if col == 'pct_dur_floor':
        return num('near_bound') == -1
    if col == 'med_amp_ratio':
        return num('amp_ratio') < float(ratio)
    return num('thresh_ratio') < float(ratio)


def header_tooltips(band=None, min_dur=None, n_no_peak=None):
    """Header tooltips of the five check columns and ``checks``."""
    lo, hi = band if band else (None, None)
    btxt = f"{lo:g}–{hi:g} Hz" if lo is not None else 'run band'
    mtxt = f"{min_dur:g} s" if min_dur is not None else 'not recorded'
    ntxt = '—' if n_no_peak is None else str(int(n_no_peak))
    return {
        'pct_off_band': ("Share of this channel's events whose peak frequency, "
                         "after removing the 1/f background, lies outside the "
                         f"run band ({btxt}). Events with no spectral peak are "
                         f"not counted: {ntxt} on this channel."),
        'pct_low_prom': ("Share of events whose spectral peak stands less than "
                         "10 dB above the 1/f background. Under 1 s this label "
                         "is unreliable: a third to a half of weak genuine "
                         "spindles shorter than 1 s get it."),
        'pct_dur_floor': ("Share of events lasting no more than 0.05 s longer "
                          f"than the run's minimum duration ({mtxt}). Many "
                          "events at the floor means the detector is cutting "
                          "short bursts out of noise."),
        'med_amp_ratio': ("Median, over this channel's events, of event band "
                          "RMS divided by the median band RMS of the "
                          "surrounding ±15 s (other events, artefact and "
                          "other stages left out)."),
        'med_thresh_ratio': ("Median of the event's detection-signal peak "
                             "divided by the detection threshold for its run. "
                             "1.0–1.2 means most events barely crossed the "
                             "threshold."),
        'checks_flag': ("Flagged when a channel's off-band, low-prominence or "
                        "at-floor share is much higher, or its amplitude "
                        "ratios much lower, than the rest of the montage (same "
                        "z limits as the amplitude flag). A problem shared by "
                        "every channel is not flagged here; see the Precision "
                        "report."),
    }


# ---------------------------------------------------------------------------
# Run information helpers
# ---------------------------------------------------------------------------

def run_duration_bounds(run, method):
    """``(min, max)`` duration bounds of ``method`` in a run, or None."""
    params = (run or {}).get('params') or {}
    by = params.get('duration_by_method') or {}
    b = by.get(str(method)) if isinstance(by, dict) else None
    if b is None:
        b = params.get('duration')
    if not b:
        return None
    try:
        lo = None if b[0] is None else float(b[0])
        hi = None if len(b) < 2 or b[1] is None else float(b[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (lo, hi)


def run_has_figures(run):
    """True when the run stored 4.6 per-event figures."""
    params = (run or {}).get('params') or {}
    return bool(params.get('event_figures'))


def run_figures_switched_off(run):
    """True for a 4.6 run detected with the figures turned off
    (``params_json['event_figures']`` present but ``None``), as opposed to a
    run detected with 4.5 or earlier, which has no such key."""
    params = (run or {}).get('params') or {}
    return 'event_figures' in params and not params.get('event_figures')


def run_is_46(run):
    """True when the run was detected with 4.6 or later."""
    params = (run or {}).get('params') or {}
    if 'event_figures' in params:
        return True
    ver = str((run or {}).get('turtlewave_version') or '')
    try:
        major, minor = (int(x) for x in ver.split('.')[:2])
    except ValueError:
        return False
    return (major, minor) >= (4, 6)


def _parse_listish(raw):
    """A stored list, a ``str(list)`` repr, or a plain string, as either a
    list of str or a str; ``None`` for empty / ``'None'``.

    ``detection_runs`` stores ``stages`` and ``ref_chan`` with ``str()``, so
    a list arrives as ``"['NREM2', 'NREM3']"``.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if not s or s == 'None':
        return None
    if s[0] in '[(':
        import ast
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return str(v)
    if s[0] == '"' or s[0] == "'":
        return s.strip('"\'')
    return s


def _split_stage_text(text):
    """``'NREM2NREM3'`` / ``'NREM2+NREM3'`` / ``'NREM2'`` -> list."""
    parts = [p for p in str(text).replace(',', '+').replace(' ', '+')
             .split('+') if p]
    try:
        from turtlewave_hdEEG.dbwrite import split_stage_token
    except ImportError:
        return parts
    out = []
    for p in parts:
        try:
            out.extend(split_stage_token(p) or [])
        except ValueError:
            out.append(p)
    return out


def run_stages(run):
    """The run's stages as a list.

    ``params_json['stages']`` when present, else the ``stages`` column, which
    holds a ``str(list)`` repr (``"['NREM2', 'NREM3']"``), a joint token
    (``'NREM2NREM3'``) or a single stage.
    """
    params = (run or {}).get('params') or {}
    raw = params.get('stages') if params.get('stages') else (run or {}).get(
        'stages')
    v = _parse_listish(raw)
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(_split_stage_text(x))
        return out
    return _split_stage_text(v)


def run_ref_chan(run):
    """Re-reference channels of a run (list; empty = data as stored).

    ``params_json['ref_chan']`` when the key is present (``[]`` there means
    no re-reference), else the ``ref_chan`` column (a ``str(list)`` repr).
    """
    params = (run or {}).get('params') or {}
    raw = params['ref_chan'] if 'ref_chan' in params else (run or {}).get(
        'ref_chan')
    v = _parse_listish(raw)
    if not v:
        return []
    return [v] if isinstance(v, str) else list(v)


def rereference(data, labels, target, ref):
    """``target`` minus the mean of the ``ref`` rows, as Wonambi's
    ``montage(data, ref_chan=ref)`` does (``nanmean`` across the reference
    channels, the target included when it is one of them; NaN -> 0 after).

    Parameters
    ----------
    data : array_like, shape (n_chan, n_time)
    labels : sequence of str
        Row labels of ``data``.
    target : str
    ref : sequence of str
        Every reference channel must be a row of ``data``.

    Returns
    -------
    numpy.ndarray
    """
    data = np.asarray(data, dtype=float)
    labels = list(labels)
    x = data[labels.index(target)]
    if ref:
        rows = [labels.index(r) for r in ref]
        x = x - np.nanmean(data[rows], axis=0)
    return np.nan_to_num(x)


def run_label(run, run_id):
    """``run 2026-09-14 (3f2a9c1)`` or ``run not recorded``."""
    if not run:
        return 'run not recorded'
    date = str(run.get('timestamp') or '')[:10] or 'date unknown'
    if not run_id:
        return f"run {date}"
    return f"run {date} ({str(run_id)[:7]})"


# ---------------------------------------------------------------------------
# Event panel rows (spec section 4)
# ---------------------------------------------------------------------------

STORED_TIP = 'Computed by TurtleWave at detection time ({version}).'
COMPUTED_TIP = 'Computed now from the raw signal; not stored for this run.'
DETECTOR_TIP = 'Value stored by the detector at detection time.'
NOT_RECORDED_FIG = 'not recorded for this run (detected with 4.5 or earlier)'
FIGURES_OFF_FIG = ('not computed for this run (event figures were switched '
                   'off when it was detected)')
FIGURES_FAILED_FIG = 'figures not computed for this channel'
FIGURES_OFF_RUN = ('Event checks were switched off when this run was '
                   'detected. Re-detect with event figures on, or run the '
                   'event-figures backfill example, to add them.')
LOAD_EEG_NOTE = 'Load the EEG file to compute these figures for this older run.'
FIGURE_KEYS = ('halfwaves', 'cycles_nominal', 'peak_freq', 'amp_bg',
               'wave_freq')

_LACOURSE_NAMES = (('abs_pow_thresh', 'absolute sigma power'),
                   ('rel_pow_thresh', 'relative sigma power'),
                   ('covar_thresh', 'sigma covariance'),
                   ('corr_thresh', 'sigma correlation'))
_MASSIMINI = ('Massimini2004', 'AASM/Massimini2004', 'AASM')


def _row(key, label, value, sub=None, tip='', level=None):
    """One panel row; ``level`` 'warn' / 'bad' colours the value."""
    return {'key': key, 'label': label, 'value': value,
            'sub': [s for s in (sub or []) if s], 'tooltip': tip,
            'level': level}


def _thresholds_dict(thresholds):
    """``{name: value}`` from a ``read_detection_thresholds`` frame/dict."""
    if thresholds is None:
        return {}
    if isinstance(thresholds, dict):
        return {k: _finite(v) for k, v in thresholds.items()}
    if len(thresholds) == 0:
        return {}
    out = {}
    for _, r in thresholds.iterrows():
        out[str(r['name'])] = _finite(r['value'])
    return out


def threshold_row(method, ev, thresholds, run_recorded, ratio=None):
    """The ``amp_thr`` row for one event (spec section 4, by method).

    ``run_recorded`` is whether the run was detected with 4.6 or later
    (:func:`run_is_46`): with no stored thresholds, a 4.5 run reads
    :data:`THRESHOLD_NOT_RECORDED`, a 4.6 run (figures on or off) the plain
    ``Threshold not recorded for this run``.
    """
    th = _thresholds_dict(thresholds)
    m = str(method)
    label = 'Amp. vs threshold'
    if m == 'CIRUS':
        return _row('amp_thr', label, 'not available for CIRUS')
    if not th:
        if not run_recorded:
            return _row('amp_thr', label, THRESHOLD_NOT_RECORDED,
                        tip=THRESHOLD_NOT_RECORDED_TIP, level='muted')
        if m in ('Ngo2015', 'Staresina2015'):
            pass
        else:
            return _row('amp_thr', label, 'Threshold not recorded for this run',
                        tip=THRESHOLD_NOT_RECORDED_TIP, level='muted')

    def barely(r):
        return r is not None and 1.0 <= r <= 1.2

    if m in ('Moelle2011', 'Ferrarelli2007', 'Nir2011'):
        peak = _finite(ev.get('peak_val_det'))
        thr = th.get('det_value_lo')
        r = _finite(ratio)
        if r is None and peak is not None and thr:
            r = peak / thr
        sub = [f"peak {fmt_num(peak)} / threshold {fmt_num(thr)} µV"]
        if m != 'Moelle2011' and th.get('sel_value') is not None:
            sub.append(f"boundary threshold {fmt_num(th['sel_value'])} µV")
        if r is None:
            return _row('amp_thr', label, '—', sub, DETECTOR_TIP)
        if barely(r):
            sub[0] += ' · barely crossed'
        return _row('amp_thr', label, f"{r:.1f}×", sub, DETECTOR_TIP,
                    'warn' if barely(r) else None)
    if m == 'Ray2015':
        return _row('amp_thr', label, 'no ratio',
                    [f"detection threshold {fmt_num(th.get('det_value_lo'))} z",
                     f"boundary threshold {fmt_num(th.get('sel_value'))} z"],
                    DETECTOR_TIP)
    if m in ('Wamsley2012', 'Martin2013'):
        return _row('amp_thr', label, 'no ratio',
                    [f"threshold {fmt_num(th.get('det_value_lo'))}",
                     'detector peak is a different signal'], DETECTOR_TIP)
    if m == 'Lacourse2018':
        return _row('amp_thr', label, '4 thresholds',
                    [f"{lbl}  {fmt_num(th.get(name))}"
                     for name, lbl in _LACOURSE_NAMES], DETECTOR_TIP)
    if m in _MASSIMINI:
        tr, pp = _finite(ev.get('det_trough')), _finite(ev.get('det_ptp'))
        ctr, cpp = th.get('max_trough_amp'), th.get('min_ptp')
        ok_tr = tr is not None and ctr is not None and tr <= ctr
        ok_pp = pp is not None and cpp is not None and pp >= cpp
        fails = (not ok_tr) + (not ok_pp)
        value = 'meets both' if not fails else f"fails {fails} of 2"
        return _row('amp_thr', label, value,
                    [f"trough {fmt_uv(tr)} µV vs {fmt_uv(ctr)} µV · "
                     f"{'meets' if ok_tr else 'fails'}",
                     f"peak-to-peak {fmt_uv(pp)} µV vs {fmt_uv(cpp)} µV · "
                     f"{'meets' if ok_pp else 'fails'}"], DETECTOR_TIP,
                    'warn' if fails else None)
    if m == 'Ngo2015':
        f = th.get('peak_thresh_factor')
        return _row('amp_thr', label, 'relative',
                    [f"criterion {fmt_num(f, 2)} × channel mean (per-channel "
                     "value not returned by the detector)"], DETECTOR_TIP)
    if m == 'Staresina2015':
        return _row('amp_thr', label, 'relative',
                    ['criterion 75th percentile (per-channel value not '
                     'returned by the detector)'], DETECTOR_TIP)
    return _row('amp_thr', label, 'no ratio', tip=DETECTOR_TIP)


def build_event_rows(ev, *, event_type, run, run_id, thresholds=None,
                     figures=None, figure_state='stored', interpolated=False,
                     outlier_thr=None, amp_col='max_amp', ptp_units_uv=False,
                     sample_row=None, figure_note=None):
    """The Event panel rows for one event, in spec order.

    Parameters
    ----------
    ev : dict
        The ``events`` row.
    event_type : str
    run : dict
        ``EventDatabase.get_run_info`` of the row's own run (``{}`` unknown).
    run_id : str or None
    thresholds : pandas.DataFrame or dict or None
        ``read_detection_thresholds(run_id, channel, method, at_time=start)``.
    figures : dict or None
        Figure values: the stored ``events`` columns, or
        ``EventFigures.as_dict()`` when computed on selection.
    figure_state : {'stored', 'computed', 'computing', 'missing', 'unavailable'}
        Where ``figures`` came from; ``'missing'`` = pre-4.6 run, no EEG;
        ``'unavailable'`` = not computable, the reason in ``figure_note``.
    figure_note : str or None
        Reason shown on the figure rows when ``'unavailable'``.
    interpolated : bool
    outlier_thr : float or None
    amp_col : str
    ptp_units_uv : bool
        ``db_meta.det_ptp_units`` is microvolts.
    sample_row : dict or None
        Reserved for review-sample mode (``stratum``, ``flags``, ``pos``,
        ``total``, ``weight``); ignored while that mode is not built.

    Returns
    -------
    list of dict
        ``key, label, value, sub, tooltip, level``.
    """
    fig = dict(figures or {})
    m = str(ev.get('method') or '')
    start, end = _finite(ev.get('start_time')), _finite(ev.get('end_time'))
    dur = _finite(ev.get('duration'))
    if dur is None and start is not None and end is not None:
        dur = end - start
    lo, hi = _finite(ev.get('freq_lower')), _finite(ev.get('freq_upper'))
    version = (run or {}).get('turtlewave_version') or 'version unknown'
    fig_tip = (STORED_TIP.format(version=version) if figure_state == 'stored'
               else COMPUTED_TIP if figure_state == 'computed' else '')
    spindle = event_type == 'spindle'
    rows = []

    # time / channel / stage / detection
    rows.append(_row('time', 'Time',
                     f"{fmt_hms1(start)} · {fmt_num(start)}–{fmt_num(end)} s"))
    ch = str(ev.get('channel') or '—')
    rows.append(_row('channel', 'Channel', ('~' + ch) if interpolated else ch,
                     tip=('Interpolated channel (reconstructed from '
                          'neighbours by the cleaning pipeline)')
                     if interpolated else ''))
    st = ev.get('epoch_stage')
    rows.append(_row('stage', 'Stage', str(st) if st not in (None, '') and
                     st == st else '— (stage not stored with this event)'))
    band = (f"{lo:g}–{hi:g} Hz" if lo is not None and hi is not None
            else 'band not stored')
    tip = ''
    if run:
        tip = (f"Stages: {run.get('stages') or '—'}\n"
               f"Excluded: {run.get('reject_types') or '—'}\n"
               f"Reference: {', '.join(run_ref_chan(run)) or 'as stored'}\n"
               f"TurtleWave {run.get('turtlewave_version') or '—'}\n"
               f"Run {run_id}")
    rows.append(_row('detection', 'Detection',
                     f"{m or '—'} · {band} · {run_label(run, run_id)}", tip=tip))

    # duration
    bounds = run_duration_bounds(run, m)
    nb = ev.get('near_bound') if 'near_bound' in ev else fig.get('near_bound')
    nb = _finite(nb)
    if dur is None:
        rows.append(_row('duration', 'Duration', '— (duration not stored)'))
    elif bounds is None or (bounds[0] is None and bounds[1] is None):
        rows.append(_row('duration', 'Duration', f"{dur:.2f} s",
                         ['run limits not recorded']))
    else:
        bmin, bmax = bounds
        if nb is None:
            if bmin is not None and abs(dur - bmin) <= 0.05:
                nb = -1
            elif bmax is not None and abs(dur - bmax) <= 0.05:
                nb = 1
            else:
                nb = 0
        lim = (f"{bmin:g}–{bmax:g} s" if bmax is not None
               else f"{bmin:g} s – no upper bound")
        outside = ((bmin is not None and dur < bmin - 1e-9)
                   or (bmax is not None and dur > bmax + 1e-9))
        level = None
        if outside:
            sub = f"outside run limits {lim}"
            level = 'warn'
        elif nb == -1:
            sub = f"at the floor of the run limits ({bmin:g} s)"
            level = 'warn'
        elif nb == 1:
            sub = f"at the ceiling of the run limits ({bmax:g} s)"
            level = 'warn'
        elif bmax is None:
            sub = f"{bmin:g} s – no upper bound · {dur - bmin:.2f} s above the floor"
        else:
            sub = f"within run limits {lim} · {dur - bmin:.2f} s above the floor"
        rows.append(_row('duration', 'Duration', f"{dur:.2f} s", [sub],
                         level=level))

    def figure_missing_row(key, label):
        if figure_state == 'computing':
            return _row(key, label, 'computing…')
        if figure_state == 'unavailable':
            return _row(key, label, '— ' + (figure_note or FIGURES_FAILED_FIG),
                        level='muted')
        return _row(key, label, '— ' + NOT_RECORDED_FIG, level='muted')

    near_splice = _finite(fig.get('near_splice')) == 1
    splice_txt = '— (within 2 s of a recording splice; not computed)'
    have_figs = figure_state in ('stored', 'computed')
    bg_insuff = bool(fig.get('bg_insufficient')) or (
        have_figs and not near_splice and _finite(fig.get('amp_ratio')) is None)
    n_min = 10 if spindle else 8

    if spindle:
        if not have_figs:
            rows.append(figure_missing_row('halfwaves', 'Half-waves'))
            rows.append(figure_missing_row('cycles_nominal', 'Cycles (nominal)'))
            rows.append(figure_missing_row('peak_freq', 'Peak frequency'))
        else:
            h = _finite(fig.get('halfwaves_above_bg'))
            if near_splice:
                rows.append(_row('halfwaves', 'Half-waves', splice_txt))
            elif h is None:
                rows.append(_row('halfwaves', 'Half-waves',
                                 '— (too little clean background to measure)',
                                 tip=fig_tip))
            else:
                rows.append(_row(
                    'halfwaves', 'Half-waves', f"{int(h)}",
                    ['half-waves standing out from background (2.5× bg RMS)'],
                    fig_tip))
            c = _finite(fig.get('cycles_nominal'))
            rows.append(_row(
                'cycles_nominal', 'Cycles (nominal)',
                splice_txt if near_splice else
                ('—' if c is None else f"{c:.1f}"),
                [] if near_splice else
                ['zero crossings ÷ 2 on the band-passed event'], fig_tip))
            rows.append(_peak_row(ev, fig, dur, lo, hi, near_splice,
                                  splice_txt, fig_tip))
    else:
        if not have_figs:
            rows.append(figure_missing_row('wave_freq', 'Wave frequency'))
        else:
            wf = _finite(fig.get('wave_freq'))
            if wf is None:
                rows.append(_row('wave_freq', 'Wave frequency',
                                 '— (zero crossing not recorded for this run)'))
            else:
                ib = lo is not None and hi is not None and lo <= wf <= hi
                rows.append(_row('wave_freq', 'Wave frequency',
                                 f"{wf:.2f} Hz   {'in band' if ib else 'OFF BAND'}",
                                 tip=fig_tip, level=None if ib else 'warn'))

    # amplitude vs background
    if not have_figs:
        rows.append(figure_missing_row('amp_bg', 'Amp. vs background'))
    elif near_splice:
        rows.append(_row('amp_bg', 'Amp. vs background', splice_txt))
    elif bg_insuff:
        rows.append(_row('amp_bg', 'Amp. vs background',
                         f"— too little clean background (fewer than "
                         f"{n_min} windows)", tip=fig_tip))
    else:
        r = _finite(fig.get('amp_ratio'))
        bg = _finite(fig.get('bg_rms'))
        nwin = _finite(fig.get('bg_n_windows'))
        e = r * bg if (r is not None and bg is not None) else None
        sub = [f"band RMS {fmt_uv(e)} vs {fmt_uv(bg)} µV · "
               f"{'—' if nwin is None else int(nwin)} background windows"]
        level = None
        if r is not None and r < 1.5:
            sub[0] += ' · barely above background'
            level = 'warn'
        if _finite(fig.get('bg_stage_mixed')) == 1:
            stages = ', '.join(run_stages(run)) or 'the run stages'
            sub.append(f"near a stage change: background outside {stages} "
                       f"left out")
        rows.append(_row('amp_bg', 'Amp. vs background',
                         '—' if r is None else f"{r:.1f}×", sub, fig_tip, level))

    # amplitude vs threshold
    rows.append(threshold_row(m, ev, thresholds, run_is_46(run),
                              ratio=fig.get('thresh_ratio')))

    # slow-wave shape
    if not spindle:
        rows.append(_row('sw_trough', 'Trough',
                         f"{fmt_uv(ev.get('det_trough'))} µV", tip=DETECTOR_TIP))
        if ptp_units_uv:
            rows.append(_row('sw_ptp', 'Peak-to-peak',
                             f"{fmt_uv(ev.get('det_ptp'))} µV",
                             tip=DETECTOR_TIP))
        zt = _finite(ev.get('det_zero_time'))
        if zt is None or start is None or end is None:
            rows.append(_row('sw_neg_half', 'Negative half-wave',
                             '— not recorded for this run'))
        else:
            if m in _MASSIMINI:
                d = end - zt
                th = _thresholds_dict(thresholds)
                lo_c, hi_c = (th.get('trough_duration_lo'),
                              th.get('trough_duration_hi'))
                crit = ((lo_c, hi_c) if lo_c is not None and hi_c is not None
                        else (0.3, 1.0) if m == 'Massimini2004'
                        else (0.25, 1.0))
                ok = crit[0] <= d <= crit[1]
                rows.append(_row(
                    'sw_neg_half', 'Negative half-wave', f"{d:.2f} s",
                    [f"criterion {crit[0]:g}–{crit[1]:g} s · "
                     f"{'meets' if ok else 'fails'}"], DETECTOR_TIP,
                    None if ok else 'warn'))
            else:
                rows.append(_row('sw_neg_half', 'Negative half-wave',
                                 f"{zt - start:.2f} s",
                                 ['start to zero crossing · no criterion'],
                                 DETECTOR_TIP))

    # amplitude outlier
    amp = _finite(ev.get(amp_col))
    thr = _finite(outlier_thr)
    if amp is not None and thr is not None and amp > thr:
        rows.append(_row('outlier', 'Amplitude outlier',
                         f"yes   {fmt_uv(amp)} > {fmt_uv(thr)} µV "
                         f"(median + 3.5·MAD)", level='bad'))
    else:
        rows.append(_row('outlier', 'Amplitude outlier', 'no'))
    return rows


def _peak_row(ev, fig, dur, lo, hi, near_splice, splice_txt, fig_tip):
    label = 'Peak frequency'
    det = _finite(ev.get('peak_freq'))
    det_line = (f"detector (first-difference, 0–50 Hz): {det:.1f} Hz"
                if det is not None else
                'detector: not recorded (detected with 4.5 or earlier)')
    if near_splice:
        return _row('peak_freq', label, splice_txt, [det_line])
    f = _finite(fig.get('peak_freq_ap'))
    coarse = dur is not None and dur < 1.0
    if f is None:
        return _row('peak_freq', label, '— no peak above the 1/f background',
                    [det_line], fig_tip)
    ib = fig.get('in_band')
    ib = (lo is not None and hi is not None and lo <= f <= hi) \
        if ib is None else bool(_finite(ib))
    badge = 'in band' if ib else 'OFF BAND'
    if coarse and dur:
        badge += f" · coarse (resolution {1.0 / dur:.1f} Hz)"
    value = f"{'≈ ' if coarse else ''}{f:.1f} Hz   {badge}"
    p = _finite(fig.get('prominence_db'))
    prom = f"prominence {p:.1f} dB" if p is not None else 'prominence —'
    low = fig.get('low_prominence')
    low = bool(_finite(low)) if low is not None else (p is not None and p < 10)
    if low:
        prom += (' · low prominence (unreliable under 1 s)' if coarse
                 else ' · low prominence')
    return _row('peak_freq', label, value, [prom, det_line], fig_tip,
                None if ib else 'warn')


def has_stored_figures(ev, event_type):
    """True when the row carries any 4.6 figure."""
    keys = (('halfwaves_above_bg', 'cycles_nominal', 'peak_freq_ap',
             'amp_ratio', 'near_splice', 'bg_n_windows')
            if event_type == 'spindle' else
            ('wave_freq', 'amp_ratio', 'near_splice', 'bg_n_windows'))
    return any(_finite(ev.get(k)) is not None for k in keys)


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

def neighbour_header(target, chosen, source, window_s, region=None, k=6):
    """``NEIGHBOURS · PPOz + 6 nearest by electrode position · 4 s ...``."""
    n = len(chosen)
    win = f"{window_s:g} s around the event"
    if source == 'position':
        what = f"{n} nearest by electrode position"
        tail = win
    elif source == 'region':
        what = f"{n} from the same region ({region or 'unknown'})"
        tail = 'no electrode positions in this file'
    else:
        what = f"{n} of the selected channels"
        tail = 'no positions or region match'
    return f"NEIGHBOURS · {target} + {what} · {tail}"


# ---------------------------------------------------------------------------
# Population read (runs in a worker thread on its own connection)
# ---------------------------------------------------------------------------

def load_population(db_path, event_type, methods=None, freq_band=None):
    """Everything the population checks need for the events in view.

    Opens its own read connection (safe to call from a worker thread). The
    run is the most recent ``detection_runs`` row among the runs whose events
    match the dashboard filters, the same rule ``get_run_rejections`` uses.

    Returns
    -------
    dict
        ``run_id``, ``run`` (``get_run_info``-style dict), ``runs_in_view``
        (list of ``(run_id, n)``), ``recorded`` (bool), ``pop`` (per-channel
        :func:`pool_population` frame) and ``error`` (str or None).
    """
    import json
    import sqlite3
    out = {'run_id': None, 'run': {}, 'runs_in_view': [], 'recorded': False,
           'figures_off': False, 'pop': pool_population(None), 'error': None}
    from pathlib import Path
    try:
        con = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro',
                              uri=True, timeout=30.0)
    except sqlite3.Error as err:
        out['error'] = str(err)
        return out
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
        if 'run_id' not in cols:
            return out
        where, params = ["event_type = ?"], [str(event_type)]
        if methods:
            where.append(f"method IN ({','.join('?' * len(methods))})")
            params += [str(m) for m in methods]
        if freq_band:
            where.append("freq_lower = ? AND freq_upper = ?")
            params += [float(freq_band[0]), float(freq_band[1])]
        runs = con.execute(
            f"SELECT run_id, COUNT(*) FROM events WHERE {' AND '.join(where)} "
            f"AND run_id IS NOT NULL GROUP BY run_id", params).fetchall()
        out['runs_in_view'] = [(str(r), int(n)) for r, n in runs]
        if not runs:
            return out
        rcols = [r[1] for r in con.execute("PRAGMA table_info(detection_runs)")]
        info = {}
        if rcols:
            ids = [r for r, _ in runs]
            cur = con.execute(
                f"SELECT * FROM detection_runs WHERE run_id IN "
                f"({','.join('?' * len(ids))}) ORDER BY timestamp DESC", ids)
            names = [d[0] for d in cur.description]
            for row in cur.fetchall():
                d = dict(zip(names, row))
                try:
                    p = json.loads(d.get('params_json') or '{}')
                except (TypeError, ValueError):
                    p = {}
                d['params'] = p if isinstance(p, dict) else {}
                info[str(d['run_id'])] = d
        # newest recorded run; else the run with most events
        run_id = next(iter(info), None) or max(runs, key=lambda r: r[1])[0]
        out['run_id'] = str(run_id)
        out['run'] = info.get(str(run_id), {})
        out['figures_off'] = run_figures_switched_off(out['run'])
        if not run_has_figures(out['run']):
            return out
        from turtlewave_hdEEG.dbwrite import event_population_summary
        summary = event_population_summary(con, run_id, event_type)
        if summary is None or len(summary) == 0:
            return out
        med = pd.read_sql_query(
            "SELECT channel, amp_ratio, thresh_ratio FROM events "
            "WHERE run_id = ? AND event_type = ?", con,
            params=[str(run_id), str(event_type)])
        medians = (med.groupby('channel')[['amp_ratio', 'thresh_ratio']]
                   .median().rename(columns={'amp_ratio': 'med_amp_ratio',
                                             'thresh_ratio': 'med_thresh_ratio'})
                   .reset_index())
        out['pop'] = pool_population(summary, medians)
        out['recorded'] = True
    except Exception as err:   # never break the dashboard
        out['error'] = f"{type(err).__name__}: {err}"
    finally:
        con.close()
    return out
