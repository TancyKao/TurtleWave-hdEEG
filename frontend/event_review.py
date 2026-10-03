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

#: Grid label of every stored reason token (``dbwrite.REVIEW_REASONS``, 11
#: values, Method Spec M7). Used on the Current line and in the Precision
#: report, so a stored reason is shown even where the event type's grid has
#: no button for it.
REASON_LABEL = {
    'artefact': 'Artefact', 'eye-movement': 'Eye movement',
    'not-in-raw': 'Not in raw', 'filter-ringing': 'Filter ringing',
    'off-band': 'Off-band', 'too-short': 'Too short', 'arousal': 'Arousal',
    'single-channel': 'Single channel', 'not-isolated': 'Not isolated',
    'wrong-morphology': 'Wrong morphology', 'other': 'Other',
}
#: Lower-case words for status lines (``Rejected … (arousal)``).
REASON_SHORT = {t: l.lower() for t, l in REASON_LABEL.items()}
REASON_TIP = {
    'artefact': ('Movement, electrode or muscle artefact. This labels this '
                 'one event only. To leave the time out of analysis for '
                 'every channel, brush it on the trace and use "Exclude '
                 'time range…".'),
    'eye-movement': 'The deflection follows the EOG.',
    'not-in-raw': 'Not visible in the raw trace.',
    'filter-ringing': 'Ringing from a step or spike in the raw trace.',
    'off-band': "The rhythm is outside the run's frequency band.",
    'too-short': 'Too short to count as this event.',
    'arousal': 'Part of an arousal: EEG speed-up, often with EMG.',
    'single-channel': 'Only on this channel, not on its neighbours.',
    'not-isolated': 'Not isolated from other waves.',
    'wrong-morphology': 'Wrong shape for this event type.',
    'other': 'Describe the reason in the comment (required).',
}
#: Reason grids (UX spec revision 3, section 5): ``(digit, token)`` in
#: button order, two columns row-major; ``other`` is always ``0`` and last;
#: ``9`` is unused and ignored.
_GRID_SPINDLE = (('1', 'artefact'), ('2', 'eye-movement'),
                 ('3', 'not-in-raw'), ('4', 'filter-ringing'),
                 ('5', 'off-band'), ('6', 'too-short'), ('7', 'arousal'),
                 ('8', 'single-channel'), ('0', 'other'))
_GRID_SLOW = (('1', 'artefact'), ('2', 'eye-movement'), ('3', 'not-in-raw'),
              ('4', 'too-short'), ('5', 'arousal'), ('6', 'single-channel'),
              ('7', 'not-isolated'), ('8', 'wrong-morphology'),
              ('0', 'other'))


def reason_grid(event_type):
    """``[(digit, token, label, tooltip)]`` of the event type's grid."""
    grid = _GRID_SPINDLE if str(event_type) == 'spindle' else _GRID_SLOW
    return [(d, t, REASON_LABEL[t], REASON_TIP[t]) for d, t in grid]


def digit_reason(event_type, digit):
    """Token for a digit key in the event type's grid, or ``None``
    (``9`` and anything else is ignored)."""
    return dict((d, t) for d, t, _l, _tip in reason_grid(event_type)).get(
        str(digit))


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
    """``not reviewed`` / ``accepted`` / ``rejected (artefact)`` / ``unsure``.

    Parameters
    ----------
    decision : str or None
        ``'accept'``, ``'reject'``, ``'unsure'`` or empty.
    reason : str or None, optional
        Stored reason token, shown in brackets. Default ``None``.

    Returns
    -------
    str
        The status-line word.
    """
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
    """µV: one decimal when |v| < 100, integer otherwise, U+2212 minus.

    Parameters
    ----------
    v : float or None
        Amplitude in microvolts.

    Returns
    -------
    str
        Formatted value, or ``'—'`` when missing or not finite.
    """
    f = _finite(v)
    if f is None:
        return '—'
    s = f"{f:.1f}" if abs(f) < 100 else f"{f:.0f}"
    return s.replace('-', MINUS)


def fmt_num(v, nd=2):
    """A number with a fixed number of decimals and a U+2212 minus.

    Parameters
    ----------
    v : float or None
        The value.
    nd : int, optional
        Decimals. Default 2.

    Returns
    -------
    str
        Formatted value, or ``'—'`` when missing or not finite.
    """
    f = _finite(v)
    if f is None:
        return '—'
    return f"{f:.{nd}f}".replace('-', MINUS)


def fmt_hms1(seconds):
    """``HH:MM:SS.s`` of recording time.

    Parameters
    ----------
    seconds : float or None
        Seconds from recording start.

    Returns
    -------
    str
        ``HH:MM:SS.s``, or ``'—'`` when missing.
    """
    f = _finite(seconds)
    if f is None:
        return '—'
    tenths = int(round(f * 10))
    s, d = divmod(tenths, 10)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{d}"


def fmt_when(iso, today=None):
    """``HH:MM`` today, else ``YYYY-MM-DD HH:MM``.

    Parameters
    ----------
    iso : str or None
        ISO 8601 timestamp.
    today : datetime.date or None, optional
        The date treated as today. Default the current date.

    Returns
    -------
    str
        The formatted time, or ``'—'`` when ``iso`` is empty.
    """
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
# Population checks (UX spec revision 3, section 1)
# ---------------------------------------------------------------------------

#: ``column -> (table header, topography combo label)``.
CHECK_COLUMNS = {
    'pct_off_band': ('Off-band', 'Off-band share'),
    'pct_low_prom': ('Low prom.', 'Low-prominence share (context)'),
    'pct_dur_floor': ('At floor', 'At-floor share'),
    'med_amp_ratio': ('Amp / bg', 'Amp vs background (median)'),
    'med_thresh_ratio': ('Amp / thr', 'Amp vs threshold (median)'),
}
#: The four columns that can set ``checks_flag``. Low prominence is context
#: only (user decision, revision 3): it follows a channel's signal-to-noise.
FLAG_COLUMNS = ('pct_off_band', 'pct_dur_floor', 'med_amp_ratio',
                'med_thresh_ratio')
SHARE_COLUMNS = ('pct_off_band', 'pct_low_prom', 'pct_dur_floor')
RATIO_COLUMNS = ('med_amp_ratio', 'med_thresh_ratio')
#: ``value column -> the n column it is computed over``.
CHECK_N = {'pct_off_band': 'n_freq', 'pct_low_prom': 'n_prom',
           'pct_dur_floor': 'n_bound', 'med_amp_ratio': 'n_amp',
           'med_thresh_ratio': 'n_thresh'}
CHECK_MIN_N = 20          #: fewer events with a value -> '—', never flagged
SHARE_GUARD = 10.0        #: percentage points from the montage median
RATIO_GUARD = 0.3         #: × from the montage median (provisional)
MODE_MIN_OFF_BAND = 10    #: off-band events needed for the mode-bin clause

#: Methods whose stored detector peak and threshold are one signal.
_RATIO_METHODS_FALLBACK = ('Moelle2011', 'Ferrarelli2007', 'Nir2011',
                           'Massimini2004', 'AASM/Massimini2004')

NOT_RECORDED_RUN = ('Event checks are not recorded for this run (detected '
                    'with 4.5 or earlier). Figures are stored by detection '
                    'runs made with 4.6 or later; re-detect this run to get '
                    'them.')
NOT_RECORDED_LIST = ('Checks not recorded for this run (detected with 4.5 or '
                     'earlier).')
THRESHOLD_NOT_RECORDED = ('Threshold not recorded for this run (detected '
                          'with 4.5 or earlier)')
THRESHOLD_NOT_RECORDED_TIP = ('Re-run detection with TurtleWave 4.6 or later '
                              'to record the detector\'s thresholds. Nothing '
                              'about this event is wrong.')
TOO_FEW_TIP = 'Too few events on this channel to judge (n = {n}).'
STAGE_TOOLTIP = ('Stages used for the check columns, the checks flag and the '
                 'flagged-channel list. Density and amplitude columns follow '
                 'the Filters dock.')


def method_has_ratio(method):
    """Whether :data:`THRESHOLD_UNITS` allows an amp/threshold ratio.

    Parameters
    ----------
    method : str
        Detection method name.

    Returns
    -------
    bool
        True when at least one threshold of the method allows a ratio.
    """
    try:
        from turtlewave_hdEEG.extensions import THRESHOLD_UNITS
    except ImportError:
        return str(method) in _RATIO_METHODS_FALLBACK
    spec = THRESHOLD_UNITS.get(str(method))
    return bool(spec) and any(spec.get('ratio_allowed', {}).values())


def no_ratio_note(method):
    """Tooltip for a ``—`` amp/thr cell on a 4.6 run of ``method``.

    Parameters
    ----------
    method : str
        Detection method name.

    Returns
    -------
    str
        The reason no amp/threshold ratio is shown.
    """
    m = str(method)
    if m == 'CIRUS':
        return 'Not available for CIRUS: the detector does not return a threshold.'
    if m in ('Ngo2015', 'Staresina2015'):
        return (f'No ratio for {m}: the per-channel threshold is not returned '
                f'by the detector.')
    return (f'No ratio for {m}: the stored peak and the detection threshold '
            f'are not the same signal.')


#: Columns of :func:`population_from_summary`.
POP_COLUMNS = ['channel', 'n', 'n_no_peak', 'n_freq', 'pct_off_band',
               'n_prom', 'pct_low_prom', 'n_bound', 'pct_dur_floor',
               'pct_at_ceiling', 'n_amp', 'med_amp_ratio', 'n_thresh',
               'med_thresh_ratio', 'n_off_band', 'off_band_below_share',
               'off_band_above_share', 'off_band_mode_lo', 'off_band_mode_hi',
               'off_band_mode_share']


def population_from_summary(summary, stage=None):
    """Per-channel check values from ``dbwrite.event_population_summary``.

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows of the summary: per channel x stage, or the ``pooled=True``
        frame (stage ``'all'``).
    stage : str or None, optional
        Keep only this stage's rows. ``None`` keeps every row and expects
        one row per channel (the pooled frame).

    Returns
    -------
    pandas.DataFrame
        :data:`POP_COLUMNS`; shares in percent, medians as stored.
    """
    if summary is None or len(summary) == 0:
        return pd.DataFrame(columns=POP_COLUMNS)
    s = summary if stage is None else summary[summary['stage'] == stage]
    if s.empty:
        return pd.DataFrame(columns=POP_COLUMNS)
    s = s.drop_duplicates('channel')

    def num(c):
        return pd.to_numeric(s[c], errors='coerce') if c in s.columns \
            else pd.Series(np.nan, index=s.index)

    out = pd.DataFrame({'channel': s['channel'].astype(str).values})
    for c in ('n', 'n_near_splice', 'n_freq', 'n_prom', 'n_bound', 'n_amp',
              'n_thresh', 'n_off_band'):
        out[c] = num(c).fillna(0).astype(int).values
    out['n_no_peak'] = (out['n'] - out['n_near_splice']
                        - out['n_freq']).clip(lower=0)
    out['pct_off_band'] = (100.0 * num('share_off_band')).values
    out['pct_low_prom'] = (100.0 * num('share_low_prom')).values
    out['pct_dur_floor'] = (100.0 * num('share_at_floor')).values
    out['pct_at_ceiling'] = (100.0 * num('share_at_ceiling')).values
    out['med_amp_ratio'] = num('amp_ratio_median').values
    out['med_thresh_ratio'] = num('thresh_ratio_median').values
    for c in ('off_band_below_share', 'off_band_above_share',
              'off_band_mode_lo', 'off_band_mode_hi', 'off_band_mode_share'):
        out[c] = num(c).values
    return out[POP_COLUMNS].sort_values('channel').reset_index(drop=True)


def _one_sided_z(x, higher_is_bad, include=None):
    """Robust one-sided z of each finite value against the median and scale
    of the finite values in ``include`` (a boolean mask; default all)."""
    x = np.asarray(x, dtype=float)
    z = np.zeros(len(x))
    ok = np.isfinite(x)
    ref = ok if include is None else ok & np.asarray(include, dtype=bool)
    if ref.sum() < 3:
        return z, (float(np.median(x[ref])) if ref.any() else np.nan)
    v = x[ref]
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
                     ratio_allowed=True, excluded=None):
    """Add ``z_*``, ``flag_*``, ``checks_flag`` and montage medians.

    The four :data:`FLAG_COLUMNS` are flagged on a one-sided robust z above
    ``soft_z`` / ``hard_z`` (high is bad for the two shares, low for the two
    ratios) AND a difference from the montage median of at least
    :data:`SHARE_GUARD` points / :data:`RATIO_GUARD`; channels with fewer
    than :data:`CHECK_MIN_N` events with a value get NaN (shown ``—``) and
    no flag. ``pct_low_prom`` (spindles only) keeps its value and montage
    median and never flags; ``med_thresh_ratio`` is NaN when the method
    allows no ratio. ``excluded`` channels (a set of labels) are left out of
    the montage median and scale and are not flagged; their values and z
    (against the other channels) stay.

    Returns
    -------
    (pandas.DataFrame, dict)
        The frame, and ``{column: montage median}``.
    """
    df = pop.copy()
    excluded = {str(c) for c in (excluded or ())}
    inc = (~df['channel'].astype(str).isin(excluded).to_numpy()
           if len(df) else np.array([], dtype=bool))
    medians = {}
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
        z, med = _one_sided_z(vals, higher_bad, inc)
        medians[col] = med
        flag = np.array([''] * len(df), dtype=object)
        if col in FLAG_COLUMNS:
            guard = SHARE_GUARD if higher_bad else RATIO_GUARD
            diff = (np.abs(vals - med) if np.isfinite(med)
                    else np.zeros(len(df)))
            ok = np.isfinite(vals) & (diff >= guard - 1e-9) & inc
            flag[ok & (z > soft_z)] = 'soft'
            flag[ok & (z > hard_z)] = 'hard'
            for i, f in enumerate(flag):
                if f:
                    reasons[i].append(f"{col} z={z[i]:.1f}")
        else:
            z = np.zeros(len(df))       # context only: never tinted
        df['z_' + col] = z
        df['flag_' + col] = flag
    if len(df):
        fl = df[['flag_' + c for c in FLAG_COLUMNS]].to_numpy()
        df['checks_flag'] = np.where((fl == 'hard').any(axis=1), 'hard',
                                     np.where((fl == 'soft').any(axis=1),
                                              'soft', ''))
        df['checks_z'] = df[['z_' + c for c in FLAG_COLUMNS]].max(axis=1)
    else:
        df['checks_flag'] = []
        df['checks_z'] = []
    df['checks_reasons'] = ['; '.join(r) for r in reasons]
    return df, medians


def fmt_check(col, v):
    """Cell text of one check column: ``34 %`` or ``1.8×``; ``—`` missing."""
    f = _finite(v)
    if f is None:
        return '—'
    if col in SHARE_COLUMNS or col == 'pct_at_ceiling':
        return f"{f:.0f} %"
    return f"{f:.1f}×"


def flagged_columns(row):
    """The row's flagged columns, hard before soft, then by z descending."""
    cols = [c for c in FLAG_COLUMNS
            if str(row.get('flag_' + c, '') or '') in ('hard', 'soft')]
    return sorted(cols, key=lambda c: (row.get('flag_' + c) != 'hard',
                                       -float(row.get('z_' + c) or 0)))


def top_flag_column(row):
    """The flagged column with the largest z, or ``None``."""
    cols = [c for c in FLAG_COLUMNS
            if str(row.get('flag_' + c, '') or '') in ('hard', 'soft')]
    if not cols:
        return None
    return max(cols, key=lambda c: float(row.get('z_' + c) or 0))


def flagged_facts(row, event_type, medians, min_dur=None):
    """Facts of one flagged channel, hard first (spec section 1)."""
    ev = EVENT_PLURAL.get(event_type, event_type)
    out = []
    for col in flagged_columns(row):
        v = row.get(col)
        if col == 'pct_off_band':
            fact = f"{_finite(v):.0f} % of {ev} off-band"
            if (int(row.get('n_off_band') or 0) >= MODE_MIN_OFF_BAND
                    and _finite(row.get('off_band_mode_share')) is not None
                    and _finite(row.get('off_band_mode_lo')) is not None):
                fact += (f" · {100 * float(row['off_band_mode_share']):.0f} % "
                         f"of those peak at {float(row['off_band_mode_lo']):g}"
                         f"–{float(row['off_band_mode_hi']):g} Hz")
        elif col == 'pct_dur_floor':
            fact = (f"{_finite(v):.0f} % of {ev} at the duration floor"
                    + (f" ({min_dur:g} s)" if min_dur is not None else ''))
        elif col == 'med_amp_ratio':
            fact = (f"median amplitude {_finite(v):.1f}× background (montage "
                    f"median {fmt_check(col, medians.get(col))})")
        else:
            fact = (f"median amplitude {_finite(v):.1f}× threshold (montage "
                    f"median {fmt_check(col, medians.get(col))})")
        out.append(fact)
    return out


def flagged_tooltip(row):
    """Off-band row tooltip: below / above shares and the mode bin."""
    if 'pct_off_band' not in flagged_columns(row):
        return ''
    b, a = (_finite(row.get(k)) for k in ('off_band_below_share',
                                          'off_band_above_share'))
    text = (f"Off-band peaks below the band: {'—' if b is None else f'{100 * b:.0f} %'}"
            f" · above the band: {'—' if a is None else f'{100 * a:.0f} %'}.")
    lo, hi, m = (_finite(row.get(k)) for k in ('off_band_mode_lo',
                                               'off_band_mode_hi',
                                               'off_band_mode_share'))
    if lo is not None and m is not None:
        text += (f" Most common 1 Hz bin: {lo:g}–{hi:g} Hz ({100 * m:.0f} % "
                 f"of off-band events).")
    return text


def amp_flag_tooltip(hard_z, soft_z):
    """The ``Amp flag`` header tooltip: the amplitude rule (R5.0), with the
    live z limits."""
    return '\n'.join([
        "Amp flag compares this channel with the other channels in view "
        "(excluded channels are left out).",
        "× hard / ▲ soft: the channel's mean event amplitude, its 95th "
        "percentile or its largest event is far from the montage median "
        f"(robust z above {hard_z:g} / {soft_z:g}). The cell names which one.",
        "DEAD: fewer than 15 % of the median event count.",
        "Change the limits in View ▸ Outlier threshold….",
    ])


def checks_rule_tooltip(hard_z, soft_z):
    """The ``CHECKS — FLAGGED CHANNELS`` header tooltip: the checks rule
    (R5.0), with the live z limits."""
    return '\n'.join([
        "Channel checks compare each channel with the others in view "
        "(excluded channels are left out).",
        "A channel is listed when its off-band or at-floor share is well "
        "above the montage median, or its median signal vs background or "
        "amp vs threshold is well below it: hard above robust z "
        f"{hard_z:g}, soft above {soft_z:g}, and only if the difference is "
        "at least 10 percentage points (shares) or 0.3× (ratios) and the "
        "channel has at least 20 events.",
        "Low prominence is shown on the topography for context and never "
        "flags.",
        "A problem every channel shares is not listed; see the Precision "
        "report.",
    ])


#: tooltip of the header line's ``{c} checks flagged`` link
CHECKS_LINK_TIP = ('Listed under the topography in CHECKS — FLAGGED '
                   'CHANNELS. Click to go there.')


def excluded_mask(qc):
    """Rows of the QC frame whose channel is excluded (``verdict`` 'drop',
    or 'channel_artefact' from an older version)."""
    if not len(qc) or 'verdict' not in qc.columns:
        return pd.Series(False, index=qc.index)
    return qc['verdict'].isin(['drop', 'channel_artefact'])


def header_count_line(qc, recorded=True, sample=False):
    """``8 checks flagged · 3 amp flagged · 1 dead`` (+ `` · n excluded``);
    ``checks hidden · …`` during live sample review. Excluded channels are
    counted only in `` · n excluded``."""
    exc = excluded_mask(qc)
    inc = qc[~exc] if len(qc) else qc

    def count(mask):
        return int(mask.sum()) if len(inc) else 0
    amp = count(inc['flag'].isin(['hard', 'soft'])) if len(inc) else 0
    dead = count(inc['flag'] == 'dead') if len(inc) else 0
    excluded = int(exc.sum()) if len(qc) else 0
    if sample:
        head = 'checks hidden'
    elif recorded:
        chk = count(inc['checks_flag'].isin(['hard', 'soft'])) \
            if len(inc) and 'checks_flag' in inc.columns else 0
        head = f"{chk} checks flagged"
    else:
        head = 'checks not recorded'
    text = f"{head} · {amp} amp flagged · {dead} dead"
    if excluded:
        text += f" · {excluded} excluded"
    return text


SHOW_ITEMS = ('All channels', 'Amp flagged', 'Excluded', 'Dead',
              'Queued for re-detect')


def show_mask(qc, item, sample=False):
    """Rows of the QC frame kept by a Show item (R5.1): ``Amp flagged`` =
    amp flag hard or soft (dead has its own item); an excluded channel is
    never ``Amp flagged`` or ``Dead``. ``sample`` is accepted for callers;
    the items no longer depend on it."""
    if not len(qc):
        return pd.Series([], dtype=bool)
    exc = excluded_mask(qc)
    if item == 'Amp flagged':
        return qc['flag'].isin(['hard', 'soft']) & ~exc
    if item == 'Excluded':
        return exc
    if item == 'Queued for re-detect':
        return (qc['redetect'].astype(bool) if 'redetect' in qc.columns
                else pd.Series(False, index=qc.index))
    if item == 'Dead':
        return (qc['flag'] == 'dead') & ~exc
    return pd.Series(True, index=qc.index)


#: Sort combo (R5.0): ``(label, column key, descending)``; the first is the
#: default. ``Amp z ↓`` sorts by |z|.
SORT_ITEMS = (
    ('Amp flag (hard first)', 'flag', True),
    ('Amp z ↓', 'amp_z', True),
    ('Mean amp ↓', 'mean_amp', True),
    ('Density ↓', 'density', True),
    ('Events ↓', 'n', True),
    ('Channel', 'channel', False),
    ('Region', 'region', False),
)
DEFAULT_SORT = SORT_ITEMS[0][0]


def sort_label_or_default(label):
    """A stored Sort label if it still exists, else the default (a sort
    saved on a removed column opens as ``Amp flag (hard first)``)."""
    return label if any(lab == label for lab, _k, _d in SORT_ITEMS) \
        else DEFAULT_SORT


def topo_caption(col, event_type, stages_text, band=None, min_dur=None):
    """Caption under the colour bar for a check metric."""
    ev = EVENT_PLURAL.get(event_type, event_type)
    head = f"{CHECK_COLUMNS[col][1].split(' (')[0]} · {ev} · {stages_text}"
    lo, hi = band if band else (None, None)
    if col == 'pct_off_band':
        rng = f"{lo:g}–{hi:g} Hz" if lo is not None else 'the run band'
        tail = f"share of events whose 1/f-corrected peak lies outside {rng}"
    elif col == 'pct_low_prom':
        tail = ('share of events whose peak stands less than 10 dB above the '
                '1/f background · context only, never flags')
    elif col == 'pct_dur_floor':
        mn = f"{min_dur:g} s" if min_dur is not None else 'run'
        tail = f"share of events within 0.05 s of the {mn} minimum duration"
    elif col == 'med_amp_ratio':
        tail = 'median event band RMS over the surrounding band RMS'
    else:
        tail = "median detection peak over the run's threshold"
    return f"{head} · {tail}"


def filter_chip_text(col, event_type, n_shown, n_total, ratio=None):
    """Epochs-tab chip text for a check filter (low prominence has none).

    Parameters
    ----------
    col : str
        A key of :data:`FLAG_COLUMNS`.
    event_type : str
        Event type, for the plural.
    n_shown, n_total : int
        Events failing the check, and events in the slice.
    ratio : float or None, optional
        The montage median for a ratio column. Default ``None``.

    Returns
    -------
    str
        The chip text, ending in ``✕``.
    """
    ev = EVENT_PLURAL.get(event_type, event_type)
    if col == 'pct_off_band':
        head = f"Showing off-band {ev} only"
    elif col == 'pct_dur_floor':
        head = f"Showing at-floor {ev} only"
    elif col == 'med_amp_ratio':
        head = f"Showing {ev} with amplitude below {ratio:.1f}× background"
    else:
        head = f"Showing {ev} with amplitude below {ratio:.1f}× threshold"
    return f"{head} ({n_shown} of {n_total}) ✕"


def failing_mask(df, col, ratio=None):
    """Events of a drilled slice that fail check ``col`` (bool Series).

    Parameters
    ----------
    df : pandas.DataFrame
        The drilled channel's events, with the figure columns.
    col : str
        A key of :data:`FLAG_COLUMNS`.
    ratio : float or None, optional
        The montage median, for the two ratio columns. Default ``None``.

    Returns
    -------
    pandas.Series
        True for events that fail the check.
    """
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


# ---------------------------------------------------------------------------
# Run information helpers
# ---------------------------------------------------------------------------

def run_duration_bounds(run, method):
    """``(min, max)`` duration bounds of ``method`` in a run, or None.

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict with ``params``.
    method : str
        The event's method.

    Returns
    -------
    tuple or None
        ``(min, max)`` in seconds, either possibly ``None``, or ``None`` when the
        run records no limits.
    """
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
    """True when the run stored 4.6 per-event figures.

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.

    Returns
    -------
    bool
    """
    params = (run or {}).get('params') or {}
    return bool(params.get('event_figures'))


def run_figures_switched_off(run):
    """True for a 4.6 run detected with the figures turned off
    (``params_json['event_figures']`` present but ``None``), as opposed to a
    run detected with 4.5 or earlier, which has no such key.

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.

    Returns
    -------
    bool
    """
    params = (run or {}).get('params') or {}
    return 'event_figures' in params and not params.get('event_figures')


def run_is_46(run):
    """True when the run was detected with 4.6 or later.

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.

    Returns
    -------
    bool
    """
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

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.

    Returns
    -------
    list of str
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

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.

    Returns
    -------
    list of str
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


# ---------------------------------------------------------------------------
# Trace scaling (spec R4.9 - R4.11): pure functions, no Qt
# ---------------------------------------------------------------------------

def round_up_125(x):
    """The smallest 1-2-5 number (…, 0.5, 1, 2, 5, 10, 20, …) not below
    ``x``; 1 for a value that is zero, negative or not finite."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = 10.0 ** exp
    for m in (1.0, 2.0, 5.0, 10.0):
        if x <= m * base * (1 + 1e-9):
            return float(m * base)
    return float(10.0 * base)


def physio_range(x):
    """``(centre, half_range)`` of one physiology row in the epoch (R4.9):
    centre = median, half-range = 1.25 × the larger of the distances from
    the median to the 1st and the 99th percentile, rounded up to a 1-2-5
    number; a flat or empty row gets 1 (in its own unit)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return 0.0, 1.0
    med = float(np.median(x))
    lo, hi = np.percentile(x, [1, 99])
    spread = 1.25 * max(abs(lo - med), abs(hi - med))
    if not spread > 0:
        return med, 1.0
    return med, round_up_125(spread)


def physio_scale_label(half, unit):
    """``±50 µV``, ``±0.5 mV`` or ``±0.05 · no unit in file`` (R4.0)."""
    if unit:
        return f"±{half:g} {unit}"
    return f"±{half:g} · no unit in file"


#: floors of the shared neighbours half-range, µV (R4.10)
NEIGHBOUR_FLOOR = {'spindle': 25.0, 'slow_wave': 75.0, 'k_complex': 75.0}
NEIGHBOUR_FLOOR_FILTERED = 5.0


def neighbour_half_range(data, event_type='spindle', filtered=False):
    """The one half-range all neighbour rows share (R4.10): 1.25 × the 99th
    percentile of the distance from each row's median, over all rows,
    rounded up to a 1-2-5 number, and never below the floor (25 µV spindles,
    75 µV slow waves and K-complexes, 5 µV with the band filter on)."""
    floor = (NEIGHBOUR_FLOOR_FILTERED if filtered
             else NEIGHBOUR_FLOOR.get(str(event_type), 75.0))
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    dev = np.abs(x - np.nanmedian(x, axis=1, keepdims=True))
    dev = dev[np.isfinite(dev)]
    if not dev.size:
        return float(floor)
    # the floor is applied as it is (25 / 75 / 5), not rounded up
    return max(float(floor),
               round_up_125(1.25 * float(np.percentile(dev, 99))))


#: floors of the trace half-range, µV (R4.11)
RAW_FLOOR, FILTERED_FLOOR = 50.0, 10.0


def trace_range(x, floor):
    """``(centre, half_range, largest)`` of a trace in the epoch (R4.11).

    Centre = median; half-range = the larger of ``floor`` and 8 × 1.4826 ×
    MAD, but no larger than the largest distance from the median, rounded up
    to a 1-2-5 number. ``largest`` is that largest distance, so samples are
    clipped exactly when ``largest > half_range``. One large event therefore
    does not flatten the trace.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return 0.0, round_up_125(floor), 0.0
    med = float(np.median(x))
    dev = np.abs(x - med)
    largest = float(dev.max())
    sd = 1.4826 * float(np.median(dev))
    half = max(float(floor), 8.0 * sd)
    if largest > 0:
        half = min(half, largest)
    return med, round_up_125(half), largest


def clip_note(half, largest):
    """``clipped at ±200 µV · largest 984 µV`` (R4.0)."""
    return f"clipped at ±{half:g} µV · largest {round(float(largest)):g} µV"


def neighbours_legend(half):
    """The Neighbours legend line (R4.0)."""
    return ("blue lines = the selected event · bar under a trace = an event "
            "detected on that channel · all rows share one scale "
            f"(±{half:g} µV)")


PHYSIO_KIND_NAMES = {'eog': 'EOG', 'emg': 'chin EMG', 'ecg': 'ECG'}


def physio_legend(kinds, no_unit_kinds=()):
    """The Physiology legend (R4.0): the filters of the kinds shown, the
    own-scale clause, and the kinds whose unit the file does not state."""
    parts = []
    if 'eog' in kinds:
        parts.append('EOG 0.3–15 Hz')
    if 'emg' in kinds:
        parts.append('chin EMG above 10 Hz')
    if 'ecg' in kinds:
        parts.append('ECG unfiltered')
    parts.append('each row scaled to its own signal in this epoch')
    named = [PHYSIO_KIND_NAMES[k] for k in ('eog', 'emg', 'ecg')
             if k in no_unit_kinds]
    if named:
        parts.append('no unit stated in this file for ' + ', '.join(named))
    return ' · '.join(parts)


def excluded_any_type_line(excluded):
    """``Excluded for any event type: E29, E62`` (``none`` when empty): the
    channels the re-run export and the QC report leave out. Exclusion is
    stored per event type; these two use the channels excluded for any."""
    names = ', '.join(sorted(str(c) for c in excluded)) or 'none'
    return f"Excluded for any event type: {names}"


def rerun_summary(n_kept, excluded, queued_excluded):
    """The channel lines of the re-run package summary."""
    text = (f"channels.csv: {int(n_kept)} kept\n"
            f"{excluded_any_type_line(excluded)}\n")
    if queued_excluded:
        text += ("Queued but excluded, not re-detected: "
                 + ', '.join(sorted(str(c) for c in queued_excluded)) + "\n")
    return text


def detector_text(view):
    """Top-bar detector text: ``Moelle2011 · 11–16 Hz`` for the one run in
    view, ``—`` when several runs (or none) are in view or the run's method
    is not recorded.

    Parameters
    ----------
    view : dict or None
        The window's population view (``res`` with ``runs_in_view``,
        ``method``, ``band``).

    Returns
    -------
    str
    """
    return detector_label(view)[0]


DETECTOR_SEVERAL_TIP = ('Several detection runs are in view. Choose a method '
                        'and band in the Filters dock.')
DETECTOR_NONE_TIP = 'No events in view.'


def detector_label(view):
    """``(text, tooltip)`` of the top-bar detector label (spec R4.7)."""
    runs = ((view or {}).get('res') or {}).get('runs_in_view') or []
    if not runs:
        return '—', DETECTOR_NONE_TIP
    method = str(view.get('method') or '')
    if len(runs) != 1 or not method:
        return '—', DETECTOR_SEVERAL_TIP
    band = view.get('band')
    if not band:
        return method, ''
    return f"{method} · {band[0]:g}–{band[1]:g} Hz", ''


def run_label(run, run_id):
    """``run 2026-09-14 (3f2a9c1)`` or ``run not recorded``.

    Parameters
    ----------
    run : dict or None
        ``get_run_info``-style dict.
    run_id : str or None
        The run id; its first 7 characters are shown.

    Returns
    -------
    str
    """
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
                   'detected. Figures are stored by detection runs made with '
                   'event figures on; re-detect this run to get them.')
LOAD_EEG_NOTE = 'Load the EEG file to compute these figures for this older run.'
FIGURE_KEYS = ('signal_bg', 'peak_freq', 'wave_freq')

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

    Parameters
    ----------
    method : str
        The event's detection method.
    ev : dict
        The ``events`` row.
    thresholds : pandas.DataFrame or dict or None
        The run's thresholds for this event.
    run_recorded : bool
        Whether the run was detected with 4.6 or later.
    ratio : float or None, optional
        The stored ``thresh_ratio``. Default ``None``.

    Returns
    -------
    dict
        A row as built by :func:`build_event_rows`.
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


HIDDEN_LABELS_NOTE = 'Labels hidden until you accept or reject this sample event.'

# ---------------------------------------------------------------------------
# Event panel: a header line and four rows (spec R4.8)
# ---------------------------------------------------------------------------

HELP_LINK_TEXT = 'What do these mean?'
HELP_TITLE = 'What the event figures mean'
HELP_BODY = (
    "Signal vs background: how many times bigger the event is than the "
    "signal just before and after it, in the band the detector searched. "
    "About 1× means it does not stand out from its surroundings.\n\n"
    "Duration: how long the event lasts. The detection run only keeps "
    "events between its shortest and longest allowed length. An event right "
    "at the shortest length only just qualified.\n\n"
    "Peak freq: the rhythm that dominates the event. \"In band\" means it "
    "lies inside the band the detector searched. \"Off band\" means the "
    "strongest rhythm in the event is a different one. Slow waves and "
    "K-complexes show Wave freq instead: one wave per event length.\n\n"
    "Amplitude outlier: whether the event is much larger than the other "
    "events on the same channel. Very large events deserve a look at the "
    "raw trace.\n\n"
    "None of these decides for you. Look at the raw trace first, then the "
    "neighbouring channels and the EOG and EMG rows. Hover any figure for "
    "the numbers behind it.")
HELP_GUIDE_TEXT = ('Full guide: Decide if an event is genuine (opens in your '
                   'browser)')
HELP_GUIDE_URL = ('https://turtlewave-hdeeg.readthedocs.io/en/latest/how-to/'
                  'decide-if-an-event-is-genuine/')

#: compact states of a figure row: value text and tooltip first line
NOT_RECORDED_TIP = ('Not recorded for this run (detected with 4.5 or '
                    'earlier). Load the EEG file to compute it now.')
NEAR_SPLICE_TIP = 'Within 2 s of a recording splice; not computed.'
TOO_LITTLE_BG_TIP = 'Fewer than {n} clean background windows near this event.'


def event_header_line(ev, interpolated=False):
    """``01:16:37.5 · PPOz · NREM2 · Moelle2011 11–16 Hz`` (R4.0); an
    interpolated channel reads ``~PPOz``."""
    ch = str(ev.get('channel') or '—')
    st = ev.get('epoch_stage')
    stage = str(st) if st not in (None, '') and st == st else '—'
    lo, hi = _finite(ev.get('freq_lower')), _finite(ev.get('freq_upper'))
    band = f" {lo:g}–{hi:g} Hz" if lo is not None and hi is not None else ''
    return (f"{fmt_hms1(_finite(ev.get('start_time')))} · "
            f"{'~' if interpolated else ''}{ch} · {stage} · "
            f"{ev.get('method') or '—'}{band}")


def event_header_tooltip(ev, run, run_id):
    """``4597.50–4598.87 s · run 2026-10-02 (0322c41) · stages … · excluded:
    … · reference … · TurtleWave …``; parts the run does not record are
    left out."""
    start, end = _finite(ev.get('start_time')), _finite(ev.get('end_time'))
    parts = []
    if start is not None and end is not None:
        parts.append(f"{start:.2f}–{end:.2f} s")
    parts.append(run_label(run, run_id))
    if run:
        if run.get('stages'):
            st = run_stages(run)
            parts.append('stages ' + (', '.join(st) if st
                                      else str(run.get('stages'))))
        if run.get('reject_types'):
            parts.append(f"excluded: {run.get('reject_types')}")
        ref = run_ref_chan(run)
        parts.append('reference ' + (', '.join(ref) if ref else 'as stored'))
        if run.get('turtlewave_version'):
            parts.append(f"TurtleWave {run.get('turtlewave_version')}")
    return ' · '.join(parts)


def sample_line(pos, total, region, stage, flags_text=None):
    """``Sample event 3 of 120 · central · NREM2`` (+ `` · flagged: …`` /
    `` · not flagged`` when ``flags_text`` is given, i.e. only after this
    reviewer's accept or reject)."""
    text = f"Sample event {int(pos)} of {int(total)} · {region} · {stage}"
    return text + (f" · {flags_text}" if flags_text else '')


def _fig_row(key, label, number, word='', word_first=False, level=None,
             tip=(), tip_neutral=None, muted=False):
    """One Event-panel row. ``value`` is the whole text; ``number`` is the
    text without the reading word (what live sample review shows), ``word``
    the reading word, coloured by ``level`` ('warn' / 'bad') at weight 600;
    ``muted`` marks a compact state shown in ``text_3``."""
    tip = [t for t in tip if t]
    neutral = [t for t in (tip if tip_neutral is None else tip_neutral) if t]
    value = (word + number) if word_first else (number + word)
    return {'key': key, 'label': label, 'value': value, 'number': number,
            'word': word, 'word_first': bool(word_first), 'sub': [],
            'tooltip': '\n'.join(tip), '_tip_neutral': '\n'.join(neutral),
            'level': 'muted' if muted else level}


def hide_flag_words(rows):
    """The rows as live sample review shows them on an undecided sample
    event (R4.8): every reading word removed (``in band``, ``OFF BAND``, the
    three duration words, ``barely above background``, ``yes`` / ``no``),
    nothing coloured or bold, the tooltips without judgement wording; all
    numbers stay.

    Returns a new list; the input rows are not changed.
    """
    out = []
    for r in rows:
        r = dict(r, sub=list(r.get('sub') or []))
        if 'number' in r:
            r['value'] = r['number']
            r['word'] = ''
            r['tooltip'] = r.get('_tip_neutral', r.get('tooltip', ''))
        if r.get('level') in ('warn', 'bad'):
            r['level'] = None
        out.append(r)
    return out


def detector_line(method, ev, thresholds, run_recorded, ratio=None,
                  neutral=False):
    """The ``Detector threshold: …`` tooltip line of the outlier row, by
    method, in one line (R4.8). ``neutral`` drops the judgement wording
    (``meets`` / ``fails`` / ``only just crossed``) and keeps the numbers."""
    row = threshold_row(method, ev, thresholds, run_recorded, ratio=ratio)
    value, sub = str(row['value']), list(row['sub'])
    head = 'Detector threshold: '
    if value == THRESHOLD_NOT_RECORDED:
        return head + ('not recorded for this run (detected with 4.5 or '
                       'earlier).')
    if value == 'Threshold not recorded for this run':
        return head + 'not recorded for this run.'
    if value == 'not available for CIRUS':
        return head + 'no ratio for CIRUS.'
    if value.endswith('×') and sub:
        first = sub[0].replace(' · barely crossed', '').replace(' / ', ' ÷ ')
        rest = ''.join(f" · {t}" for t in sub[1:])
        tail = '' if neutral else ' (1.0× means it only just crossed)'
        return f"{head}{first} = {value}{tail}{rest}."
    if value == 'no ratio':
        detail = f" ({' · '.join(sub)})" if sub else ''
        return f"{head}no ratio for {method}{detail}."
    if neutral:
        if value == 'meets both' or value.startswith('fails '):
            value = '2 criteria'
        sub = [t.replace(' · meets', '').replace(' · fails', '') for t in sub]
    detail = f" ({' · '.join(sub)})" if sub else ''
    return f"{head}{value}{detail}."


def build_event_rows(ev, *, event_type, run, run_id, thresholds=None,
                     figures=None, figure_state='stored', interpolated=False,
                     outlier_thr=None, amp_col='max_amp', ptp_units_uv=False,
                     sample_row=None, figure_note=None, outlier_n=None):
    """The four Event panel rows for one event (spec R4.8).

    ``signal_bg``, ``duration``, ``peak_freq`` (``wave_freq`` for slow waves
    and K-complexes) and ``outlier``. Everything else the panel used to
    show (half-waves, nominal cycles, prominence, the detector threshold,
    the slow-wave shape) is in the tooltips; time, channel, stage and
    detection are the header line (:func:`event_header_line`).

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
        Reason given in the tooltip when ``'unavailable'``.
    interpolated : bool
        Unused here (the header line marks an interpolated channel).
    outlier_thr : float or None
        The channel's outlier threshold (median + 3.5 × MAD).
    amp_col : str
    ptp_units_uv : bool
        ``db_meta.det_ptp_units`` is microvolts.
    sample_row : dict or None
        Unused (the sample line is separate; see :func:`sample_line`).
    outlier_n : int or None
        Number of events behind ``outlier_thr``. Accepted for callers; the
        rule sentence no longer names the count.

    Returns
    -------
    list of dict
        Four rows: ``key, label, value, number, word, word_first, sub``
        (always empty), ``tooltip``, ``level`` (``'warn'``, ``'bad'``,
        ``'muted'`` or ``None``).
    """
    fig = dict(figures or {})
    m = str(ev.get('method') or '')
    start, end = _finite(ev.get('start_time')), _finite(ev.get('end_time'))
    dur = _finite(ev.get('duration'))
    if dur is None and start is not None and end is not None:
        dur = end - start
    lo, hi = _finite(ev.get('freq_lower')), _finite(ev.get('freq_upper'))
    band = (f"{lo:g}–{hi:g} Hz" if lo is not None and hi is not None
            else 'the run band')
    version = (run or {}).get('turtlewave_version') or 'version unknown'
    prov = (STORED_TIP.format(version=version) if figure_state == 'stored'
            else COMPUTED_TIP if figure_state == 'computed' else '')
    spindle = event_type == 'spindle'
    have_figs = figure_state in ('stored', 'computed')
    near_splice = have_figs and _finite(fig.get('near_splice')) == 1
    n_min = 10 if spindle else 8
    rows = []

    def compact(key, label, extra=()):
        """The row in a compact state, or ``None`` when it has a value."""
        if figure_state == 'computing':
            return _fig_row(key, label, 'computing…', muted=True)
        if figure_state == 'unavailable':
            return _fig_row(key, label, 'not computed', muted=True,
                            tip=[figure_note or FIGURES_FAILED_FIG, *extra])
        if not have_figs:
            return _fig_row(key, label, 'not recorded', muted=True,
                            tip=[NOT_RECORDED_TIP, *extra])
        if near_splice:
            return _fig_row(key, label, 'near a splice', muted=True,
                            tip=[NEAR_SPLICE_TIP, *extra, prov])
        return None

    # ---- Signal vs background ---------------------------------------------
    label = 'Signal vs background'
    what = (f"How many times bigger the event is than the signal around it, "
            f"in the detection band ({band}). About 1× means it does not "
            f"stand out.")
    row = compact('signal_bg', label, [what])
    r = _finite(fig.get('amp_ratio'))
    if row is None and (bool(_finite(fig.get('bg_insufficient'))) or r is None):
        row = _fig_row('signal_bg', label, 'too little background',
                       muted=True,
                       tip=[TOO_LITTLE_BG_TIP.format(n=n_min), what, prov])
    if row is None:
        bg = _finite(fig.get('bg_rms'))
        nwin = _finite(fig.get('bg_n_windows'))
        tip = [what]
        if bg is not None:
            windows = ('half-second windows in the surrounding 30 s'
                       if spindle else
                       '2-second windows in the surrounding 60 s')
            count = '' if nwin is None else f"{int(nwin)} "
            tip.append(
                f"Event {fmt_uv(r * bg)} µV RMS · background {fmt_uv(bg)} µV "
                f"RMS (the middle value of {count}{windows}; other events, "
                f"excluded time and other stages are left out).")
        if _finite(fig.get('bg_stage_mixed')) == 1:
            stages = ', '.join(run_stages(run)) or 'the run stages'
            tip.append(f"Near a stage change: background outside {stages} "
                       f"was left out.")
        tip.append(prov)
        barely = r < 1.5
        row = _fig_row('signal_bg', label, f"{r:.1f}×",
                       ' · barely above background' if barely else '',
                       level='warn' if barely else None, tip=tip)
    rows.append(row)

    # ---- Duration (always a value: it comes from the event row) -----------
    bounds = run_duration_bounds(run, m)
    bmin, bmax = bounds if bounds else (None, None)
    nb = ev.get('near_bound') if 'near_bound' in ev else fig.get('near_bound')
    nb = _finite(nb)
    tip = []
    if bmin is not None and bmax is not None:
        limits = f"limits {bmin:g}–{bmax:g} s"
        tip.append(f"How long the event lasts. This run keeps events between "
                   f"{bmin:g} and {bmax:g} s.")
    elif bmin is not None:
        limits = f"limits from {bmin:g} s"
        tip.append(f"How long the event lasts. This run keeps events of "
                   f"{bmin:g} s or longer.")
    else:
        limits = 'limits not recorded'
        tip.append("How long the event lasts.")
    if spindle and have_figs and not near_splice:
        h = _finite(fig.get('halfwaves_above_bg'))
        c = _finite(fig.get('cycles_nominal'))
        bits = []
        if h is not None:
            bits.append(f"{int(h)} half-waves stand out from the background "
                        f"(above 2.5× background RMS)")
        if c is not None:
            bits.append(f"{c:.1f} cycles counted from zero crossings")
        if bits:
            tip.append(' · '.join(bits) + '.')
            tip.append(prov)
    if dur is None:
        rows.append(_fig_row('duration', 'Duration', f"not stored · {limits}",
                             muted=True, tip=tip))
    else:
        word = ''
        if bmin is not None or bmax is not None:
            if nb is None:
                nb = (-1 if bmin is not None and abs(dur - bmin) <= 0.05 else
                      1 if bmax is not None and abs(dur - bmax) <= 0.05 else 0)
            if ((bmin is not None and dur < bmin - 1e-9)
                    or (bmax is not None and dur > bmax + 1e-9)):
                word = ' · outside the limits'
            elif nb == -1:
                word = ' · at the shortest allowed'
            elif nb == 1:
                word = ' · at the longest allowed'
        rows.append(_fig_row('duration', 'Duration',
                             f"{dur:.2f} s · {limits}", word,
                             level='warn' if word else None, tip=tip))

    # ---- Peak freq (spindles) / Wave freq (slow waves, K-complexes) --------
    if spindle:
        label = 'Peak freq'
        what = (f"The rhythm that dominates the event, once the slow "
                f"background that all EEG has (the \"1/f background\") is "
                f"removed. The detector searched {band}.")
        row = compact('peak_freq', label, [what])
        if row is None:
            f = _finite(fig.get('peak_freq_ap'))
            coarse = dur is not None and dur < 1.0
            tip = [what]
            p = _finite(fig.get('prominence_db'))
            if p is not None:
                tip.append(f"The peak stands {p:.1f} dB above the "
                           f"background. Under 10 dB is a weak peak, and for "
                           f"events under 1 s that label is unreliable.")
            if coarse and dur:
                tip.append(f"This event is shorter than 1 s, so the "
                           f"frequency is only accurate to about "
                           f"{1.0 / dur:.1f} Hz.")
            tip.append(prov)
            if f is None:
                row = _fig_row('peak_freq', label, 'no clear peak', tip=tip)
            else:
                ib = fig.get('in_band')
                ib = (lo is not None and hi is not None and lo <= f <= hi) \
                    if ib is None or _finite(ib) is None else bool(_finite(ib))
                row = _fig_row('peak_freq', label,
                               f"{'≈ ' if coarse else ''}{f:.1f} Hz",
                               ' · in band' if ib else ' · OFF BAND',
                               level=None if ib else 'warn', tip=tip)
        rows.append(row)
    else:
        label = 'Wave freq'
        tip = [f"One wave per event length (1 ÷ duration). The detector "
               f"searched {band}."]
        wf = _finite(fig.get('wave_freq')) if have_figs else None
        if wf is None and dur:
            wf = 1.0 / dur
        if wf is None:
            rows.append(_fig_row('wave_freq', label, 'not recorded',
                                 muted=True, tip=tip))
        else:
            ib = lo is not None and hi is not None and lo <= wf <= hi
            rows.append(_fig_row('wave_freq', label, f"{wf:.2f} Hz",
                                 ' · in band' if ib else ' · OFF BAND',
                                 level=None if ib else 'warn', tip=tip))

    # ---- Amplitude outlier (always a value) --------------------------------
    amp = _finite(ev.get(amp_col))
    thr = _finite(outlier_thr)
    events = EVENT_PLURAL.get(event_type, 'events')
    ch = str(ev.get('channel') or 'this channel')
    tip = []
    question = (f"Is this event much larger than the other {events} on "
                f"{ch}? A very large event deserves a look at the raw trace.")
    if thr is not None:
        tip.append(f"Is this event much larger than the other {events} on "
                   f"{ch}? Rule: amplitude above {fmt_uv(thr)} µV, which is "
                   f"the typical event on this channel plus 3.5 times the "
                   f"typical spread (median + 3.5 × MAD). A very large event "
                   f"deserves a look at the raw trace.")
    else:
        tip.append(question)
    det = {flag: detector_line(m, ev, thresholds, run_is_46(run),
                               ratio=fig.get('thresh_ratio'), neutral=flag)
           for flag in (False, True)}
    shape = ''
    if not spindle:
        bits = []
        if _finite(ev.get('det_trough')) is not None:
            bits.append(f"trough {fmt_uv(ev.get('det_trough'))} µV")
        if ptp_units_uv and _finite(ev.get('det_ptp')) is not None:
            bits.append(f"peak-to-peak {fmt_uv(ev.get('det_ptp'))} µV")
        zt = _finite(ev.get('det_zero_time'))
        if zt is not None and start is not None and end is not None:
            d = (end - zt) if m in _MASSIMINI else (zt - start)
            bits.append(f"negative half-wave {d:.2f} s")
        if bits:
            shape = 'Shape: ' + ' · '.join(bits) + '.'
    is_out = amp is not None and thr is not None and amp > thr
    number = f"{fmt_uv(amp)} µV" if amp is not None else 'amplitude not stored'
    rows.append(_fig_row(
        'outlier', 'Amplitude outlier', number,
        'yes · ' if is_out else 'no · ', word_first=True,
        level='bad' if is_out else None,
        # live sample review: the threshold stays out of the tooltip (with
        # the amplitude shown beside it, it would give the answer away)
        tip=tip + [det[False], shape],
        tip_neutral=[question, det[True], shape]))
    return rows


def has_stored_figures(ev, event_type):
    """True when the row carries any 4.6 figure.

    Parameters
    ----------
    ev : dict
        The ``events`` row.
    event_type : str
        ``'spindle'`` or a slow-wave type.

    Returns
    -------
    bool
    """
    keys = (('halfwaves_above_bg', 'cycles_nominal', 'peak_freq_ap',
             'amp_ratio', 'near_splice', 'bg_n_windows')
            if event_type == 'spindle' else
            ('wave_freq', 'amp_ratio', 'near_splice', 'bg_n_windows'))
    return any(_finite(ev.get(k)) is not None for k in keys)


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

def neighbour_header(target, chosen, source, window_s, region=None, k=6):
    """``NEIGHBOURS · PPOz + 6 nearest by electrode position · 4 s ...``.

    Parameters
    ----------
    target : str
        The selected channel.
    chosen : sequence of str
        The neighbour channels shown.
    source : str
        How they were chosen: electrode positions, same region, or selected channels.
    window_s : float
        Window length in seconds.
    region : str or None, optional
        Region name for the same-region case. Default ``None``.
    k : int, optional
        Neighbours asked for. Default 6.

    Returns
    -------
    str
        The header text.
    """
    n = len(chosen)
    win = f"{window_s:g} s around the event"
    if source == 'position':
        what = f"{n} nearest by electrode position (1 = nearest)"
        tail = win
    elif source == 'region':
        what = f"{n} from the same region ({region or 'unknown'})"
        tail = 'no electrode positions in this file'
    else:
        what = f"{n} of the selected channels"
        tail = 'no positions or region match'
    return f"NEIGHBOURS · {target} + {what} · {tail}"


def neighbour_labels(target, chosen, source, interpolated=()):
    """Row labels: ``{target} · target`` then ``{ch} · {rank}`` (rank 1 =
    nearest) for the position rule; region and selected-channel fallbacks
    are not ranked, so their rows read ``{ch}``. ``~`` marks interpolated
    channels. No distance anywhere (EEGLAB coordinates carry no reliable
    units)."""
    interp = set(interpolated or ())

    def name(c):
        return ('~' + c) if c in interp else c
    out = [f"{name(target)} · target"]
    for i, c in enumerate(chosen, 1):
        out.append(f"{name(c)} · {i}" if source == 'position' else name(c))
    return out


# ---------------------------------------------------------------------------
# Keys, cheat sheet, status bar, legends (UX spec revision 3, section 15)
# ---------------------------------------------------------------------------

KEY_HINTS = {
    'epochs': ('A accept · R reject · U unsure · ] [ unreviewed · N P outlier '
               '· ? keys'),
    # no "N P outlier": outlier marks and hops are hidden in sample review
    'sample': 'A accept · R reject · U unsure · ] [ sample · ? keys',
    'channels': 'F re-detect queue · ? keys',
}
#: Show events (R5.4): checkbox label -> decision ('unreviewed' = none)
SHOW_EVENTS_ITEMS = (('unreviewed', 'unreviewed'), ('accepted', 'accept'),
                     ('rejected', 'reject'), ('unsure', 'unsure'))
SHOW_EVENTS_TIP = ("Which of this channel's events to show, by your own "
                   "decisions. Applies to this tab only.")


def shown_chip_text(what, n, total, channel):
    """``Showing: rejected · 3 of 1,847 on PPOz ✕`` (R5.0)."""
    return (f"Showing: {what} · {int(n):,} of {int(total):,} on {channel} ✕")
STRIP_LEGEND = ('grey bars = events per epoch · red = amplitude outliers · '
                'purple = excluded time · white line = current epoch')
STRIP_LEGEND_SAMPLE = ' · blue ticks = sample events'
#: the strip legend while a review sample is active: no outlier colour
STRIP_LEGEND_NO_OUTLIERS = ('grey bars = events per epoch · purple = '
                            'excluded time · white line = current epoch')
#: appended while Show events filters (R5.0)
STRIP_LEGEND_SHOWN = ' · grey ticks below = epochs with shown events'
SAVE_LINE = 'Decisions save to {db} as you make them.'
SAVE_LINE_NO_NAME = 'Set a reviewer name to save decisions.'
SAVE_LINE_NO_STORE = ('Decisions cannot be saved: this TurtleWave library has '
                      'no review store.')
_GRID_HEAD = {'spindle': 'SPINDLES', 'slow_wave': 'SLOW WAVES',
              'k_complex': 'K-COMPLEXES'}


def cheat_sheet_text(event_type='spindle', undo='Ctrl+Z'):
    """The ``?`` dialog's text (spec section 15); ``undo`` is the platform's
    text for Ctrl+Z (``⌘Z`` on macOS)."""
    et = event_type if event_type in _GRID_HEAD else 'spindle'
    grid = reason_grid(et)
    cells = [f"{d}  {l}" for d, _t, l, _tip in grid]
    width = max(len(c) for c in cells) + 3
    pairs = ['  ' + ''.join(c.ljust(width) for c in cells[i:i + 2]).rstrip()
             for i in range(0, len(cells), 2)]
    u = f"{undo:<8}"
    return '\n'.join([
        'KEYS' + ' ' * 56 + '? or Esc to close', '',
        'DECIDE (Epochs tab)',
        '  A        accept',
        '  R        reject, then a reason',
        '  U        unsure, reason optional',
        '  1–8, 0   reason while Reject or Unsure is waiting; 0 = other '
        '(needs a comment)',
        '  Enter    save with the last reason, or save the comment',
        '  C        type a comment',
        '  Esc      cancel a waiting Reject or Unsure',
        f"  {u} undo the last decision", '',
        f"REASONS FOR {_GRID_HEAD[et]}", *pairs, '',
        'MOVE (Epochs tab)',
        '  ] [      next / previous sample event (in sample mode)',
        '           next / previous unreviewed event on this channel '
        '(otherwise)',
        '  } {      next / previous event on this channel',
        '  → ←      next / previous epoch',
        '  N P      next / previous epoch with outliers', '',
        'CHANNELS',
        '  F        add the selected channel to the re-detect queue',
    ])


# ---------------------------------------------------------------------------
# Population read (runs in a worker thread on its own connection)
# ---------------------------------------------------------------------------

def load_population(db_path, event_type, methods=None, freq_band=None):
    """Everything the population checks need for the events in view.

    Opens its own read connection (safe to call from a worker thread). The
    run is the most recent ``detection_runs`` row among the runs whose events
    match the dashboard filters, the same rule ``get_run_rejections`` uses.

    Parameters
    ----------
    db_path : str
        Path to ``neural_events.db``.
    event_type : str
        Event type in view.
    methods : sequence of str or None, optional
        Detection methods in view. Default ``None`` (all).
    freq_band : tuple of float or None, optional
        Band in view. Default ``None`` (all).

    Returns
    -------
    dict
        ``run_id``, ``run`` (``get_run_info``-style dict), ``runs_in_view``
        (list of ``(run_id, n)``), ``recorded`` (bool), ``stages`` (the
        run's stages, in order), ``by_stage`` (``{stage: frame}``) and
        ``pooled`` (all stages, ``event_population_summary(pooled=True)``),
        each a :func:`population_from_summary` frame; ``pop`` is
        ``pooled``; ``error`` (str or None).
    """
    import json
    import sqlite3
    out = {'run_id': None, 'run': {}, 'runs_in_view': [], 'recorded': False,
           'figures_off': False, 'stages': [], 'by_stage': {},
           'pooled': population_from_summary(None),
           'pop': population_from_summary(None), 'error': None}
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
        import inspect
        kw = {}
        if 'stages' in inspect.signature(event_population_summary).parameters:
            st = run_stages(out['run'])
            kw = {'stages': st} if st else {}   # pool only the run's stages
        pooled = event_population_summary(con, run_id, event_type,
                                          pooled=True, **kw)
        present = [x for x in summary['stage'].astype(str).unique()
                   if x != 'unscored']
        stages = [x for x in run_stages(out['run']) if x in present] or \
            sorted(present)
        out['stages'] = stages
        out['by_stage'] = {st: population_from_summary(summary, st)
                           for st in stages}
        out['pooled'] = out['pop'] = population_from_summary(pooled)
        out['recorded'] = True
    except Exception as err:   # never break the dashboard
        out['error'] = f"{type(err).__name__}: {err}"
    finally:
        con.close()
    return out
