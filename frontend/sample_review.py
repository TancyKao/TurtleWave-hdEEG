"""Review-sample mode and precision report: text, numbers and database reads.

Qt-free. The library (``turtlewave_hdEEG.review_sampling``) draws the sample
and estimates precision; this module turns its results into the exact text of
the UX spec (``_scratch/design/event-decision-spec.md`` revision 2, sections 2
and 11) and reads what the GUI needs from ``review_sample_designs`` /
``review_samples``. The library is imported lazily so ``frontend`` still
imports without it.
"""

import json
import math

#: ``review_samples.flag_components`` token -> words shown after a decision.
FLAG_WORDS = {'off_band': 'off-band', 'low_prominence': 'low prominence',
              'low_amp_ratio': 'amplitude under 2× background',
              'amp_ratio_missing': 'amplitude vs background missing',
              'near_splice': 'near a splice'}

#: The library's flag definition (review_sampling, frozen at draw time), in
#: words; the duration floor (``near_bound``) is display-only and never flags.
FLAG_DEFINITION = ('Flagged: off-band, low prominence (spindles only), '
                   'amplitude under 2× background or missing, or near a '
                   'splice. Being at the duration floor does not flag an '
                   'event; it is shown for information only.')

REGION_ORDER = ('frontal', 'central', 'parietal', 'temporal', 'occipital')
STAGE_ORDER = ('NREM2', 'NREM3')
MIN_DECIDED = 10          #: fewer decided events in a group: "n too small"

FREE_BROWSING_TIP = (
    'A decision on a sample event counts toward the sample whether or not '
    'sample mode was on when you made it: decisions are stored per event and '
    'reviewer, not per sample.')
NO_SAMPLE_TEXT = 'REVIEW SAMPLE · No review sample for this run yet.'
LEGACY_NOTE = ('Event checks are not recorded for this run, so flagged events '
               'cannot be oversampled. The sample is stratified by region and '
               'stage only.')
FLAGGED_NOTE = ('Flagged events (off-band, low prominence, amplitude under '
                '2× background or missing, or near a splice; the duration '
                'floor does not flag) are drawn more often than their share; '
                'each sampled event carries a weight so the precision '
                'estimate is not biased by this.')
OUTSIDE_SAMPLE = 'Saved, outside the review sample: not counted in precision.'
FILTER_OFF = 'Event filter off while reviewing the sample.'
NO_DECISIONS = 'No sample events decided yet. Press ] in the Epochs tab to start.'
NO_SECOND = 'No second reviewer yet.'
PICKER_LOCKED_TIP = ("Finish your own review first, so other reviewers' "
                     "decisions do not influence yours.")
EXPORT_CSV_TIP = ('Writes the table as CSV. The figures are also saved in '
                  'neural_events.db, table review_precision.')
LIST_HINT = 'Double-click an event to open it in the Epochs tab.'
CONVENTION_NOTE = 'A lab convention, not a published standard.'
#: the Precision rule setting (fraction, and 'point estimate' or
#: 'lower 95 % bound'); the same QSettings keys as before
POOL_THRESHOLD_KEY, POOL_ESTIMATE_KEY = ('review/pool_threshold',
                                         'review/pool_estimate')
ESTIMATES = ('point estimate', 'lower 95 % bound')

CSV_COLUMNS = ('subject', 'event_type', 'method', 'freq_lower', 'freq_upper',
               'run_id', 'sample_id', 'seed', 'reviewer', 'region', 'stage',
               'n_decided', 'n_accepted', 'n_rejected', 'n_unsure',
               'precision', 'ci_low', 'ci_high', 'weighted', 'pool_threshold',
               'pool_estimate', 'poolable', 'turtlewave_version')

EVENT_PLURAL = {'spindle': 'spindles', 'slow_wave': 'slow waves',
                'k_complex': 'K-complexes'}


def _rs():
    from turtlewave_hdEEG import review_sampling
    return review_sampling


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Database reads
# ---------------------------------------------------------------------------

def _cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    except Exception:
        return []


def scope_for_run(conn, run_id, event_type):
    """The detection scope of one run: ``event_type, method, freq_lower,
    freq_upper, stage`` (the ``events.stage`` token), or ``None``."""
    if run_id is None:
        return None
    rows = conn.execute(
        "SELECT DISTINCT event_type, method, freq_lower, freq_upper, stage "
        "FROM events WHERE run_id = ? AND event_type = ?",
        (str(run_id), str(event_type))).fetchall()
    if len(rows) != 1:
        return None
    et, m, lo, hi, st = rows[0]
    return {'event_type': et, 'method': m, 'freq_lower': lo,
            'freq_upper': hi, 'stage': st}


def samples_for_scope(conn, scope):
    """Designs of every sample drawn on ``scope``, newest first."""
    if not scope or 'review_sample_designs' not in {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}:
        return []
    cur = conn.execute(
        "SELECT * FROM review_sample_designs WHERE event_type = ? AND "
        "method = ? AND freq_lower IS ? AND freq_upper IS ? AND "
        "stage_scope IS ? ORDER BY drawn_at DESC, rowid DESC",
        (scope['event_type'], scope['method'], scope['freq_lower'],
         scope['freq_upper'], scope['stage']))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def sample_rows(conn, sample_id):
    """``{uuid: review_samples row}`` of one sample."""
    cur = conn.execute("SELECT * FROM review_samples WHERE sample_id = ?",
                       (sample_id,))
    names = [d[0] for d in cur.description]
    return {r[names.index('uuid')]: dict(zip(names, r)) for r in cur}


def presentation_order(conn, sample_id):
    """Every sample event in the library's presentation order (rank within
    sub-cell, then cell), from ``sample_progress`` for a reviewer with no
    labels (``''`` can never store a decision)."""
    return list(_rs().sample_progress(conn, sample_id,
                                      reviewer='')['next_uuids'])


def reviewer_labels(conn, sample_id):
    """``{reviewer: {uuid: (decision, reason)}}`` of the VALID labels on a
    sample's events, from the library's ``read_sample_labels`` (the same
    voiding rules as ``review_precision``: event gone, end time moved more
    than 0.05 s, or a label from another sample whose end time differs)."""
    df = _rs().read_sample_labels(conn, sample_id)
    out = {}
    for r in df[df['valid'].astype(bool)].itertuples(index=False):
        out.setdefault(r.reviewer, {})[r.uuid] = (
            r.decision, None if r.reason != r.reason else r.reason)
    return out


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def stratum_text(row, hide_flags=False):
    """``parietal · NREM2 · flagged: off-band`` (spec section 2); with
    ``hide_flags`` just ``parietal · NREM2`` (live sample review before this
    reviewer's accept or reject)."""
    head = f"{row.get('region')} · {row.get('stage')}"
    if hide_flags:
        return head
    flagged = row.get('flagged')
    cell = str(row.get('cell') or '')
    if flagged is None:
        return head + (' · figures not computed' if cell.endswith('|N')
                       else ' · not stratified by flags')
    if not int(flagged):
        return head + ' · not flagged'
    words = [FLAG_WORDS.get(t, t.replace('_', ' '))
             for t in str(row.get('flag_components') or '').split(',') if t]
    return head + ' · flagged: ' + (', '.join(words) or 'yes')


def weight_tooltip(weight):
    w = _finite(weight) or 1.0
    return (f"Sampling weight {w:.1f}: each event like this one stands for "
            f"about {w:.0f} events in its group.")


def fmt_minutes(seconds):
    m = int(round(seconds / 60.0))
    if m < 1:
        return 'under a minute'
    if m < 60:
        return f"about {m} min"
    return f"about {m // 60} h {m % 60:02d} min"


def session_eta(times, remaining):
    """Time left from this session's own pace: the median gap between its
    consecutive sample decisions (gaps over 5 min dropped) x remaining."""
    if remaining <= 0 or len(times) < 2:
        return None
    ts = sorted(times)
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if 0 < b - a <= 300)
    if not gaps:
        return None
    mid = gaps[len(gaps) // 2] if len(gaps) % 2 else \
        0.5 * (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2])
    return mid * remaining


def bar_text(state, *, design=None, n=0, total=0, unsure=0, reviewer='',
             n_revisit=0, eta_s=None, scope_tag=''):
    """REVIEW SAMPLE bar text for ``state`` in {'none', 'idle', 'active',
    'done', 'revisit'}."""
    head = 'REVIEW SAMPLE' + scope_tag
    if state == 'none':
        return head + ' · No review sample for this run yet.'
    if state == 'idle':
        date = str((design or {}).get('drawn_at') or '')[:10]
        seed = (design or {}).get('seed')
        who = f" · {n} decided by {reviewer}" if reviewer else ''
        return (f"{head} · {total} events drawn {date} (seed {seed}){who}")
    if state == 'revisit':
        return f"{head} · revisiting {n_revisit} unsure"
    text = f"{head} · {n} of {total} in sample · {unsure} unsure"
    if state == 'done':
        return text + ' · done'
    if eta_s is not None:
        text += f" · {fmt_minutes(eta_s)} left at your pace"
    return text


def end_text(total, reviewer, unsure):
    return f"All {total} sample events decided by {reviewer} ({unsure} unsure)."


# ---------------------------------------------------------------------------
# Precision report
# ---------------------------------------------------------------------------

def precision_frame(conn, sample_id, reviewer, write=True):
    """``compute_review_precision`` for one reviewer (primary labels)."""
    return _rs().compute_review_precision(conn, sample_id, reviewer=reviewer,
                                          write=write)


def _row(df, dtype, domain):
    hit = df[(df['domain_type'] == dtype) & (df['domain'] == domain)]
    return hit.iloc[0].to_dict() if len(hit) else None


def compared_value(row, estimate):
    """The number the pooling rule compares: P̂ or the lower bound."""
    return _finite(row.get('ci_lo') if estimate == 'lower 95 % bound'
                   else row.get('p_hat'))


def pct(v):
    """A fraction as a whole percent (``0.934`` -> ``93``)."""
    return int(round(100 * float(v)))


def cell_text(row, threshold=0.80, estimate='point estimate', mark=True):
    """One report table cell (R5.0): ``(text, level, tooltip)``.

    ``93 %``; below the rule ``54 % ▼`` (level ``'warn'``; only when
    ``mark``); too few decided ``—`` (level ``'muted'``). The tooltip gives
    the confidence range and the counts.
    """
    if row is None:
        return '—', 'muted', 'No events of the sample in this group.'
    n = int(row.get('n_decided') or 0)
    p, lo, hi = (_finite(row.get(k)) for k in ('p_hat', 'ci_lo', 'ci_hi'))
    if n < MIN_DECIDED or p is None:
        return ('—', 'muted', f"Only {n} decided here; at least "
                f"{MIN_DECIDED} are needed to judge.")
    a, r = int(row.get('n_accept') or 0), int(row.get('n_reject') or 0)
    u = int(row.get('n_unsure') or 0)
    tip = (f"{pct(p)} % (95 % confidence {pct(lo)}–{pct(hi)} %) · {n} "
           f"decided: {a} accepted, {r} rejected"
           + (f", {u} unsure" if u else ''))
    v = compared_value(row, estimate)
    if mark and v is not None and v < threshold:
        return f"{pct(p)} % ▼", 'warn', tip
    return f"{pct(p)} %", None, tip


def judged_groups(df):
    """Region × stage rows with enough decided events, and the count of the
    rest."""
    rs_rows = df[df['domain_type'] == 'region_stage']
    judged, small = [], 0
    for _, r in rs_rows.iterrows():
        if int(r.get('n_decided') or 0) >= MIN_DECIDED and _finite(
                r.get('p_hat')) is not None:
            judged.append(r.to_dict())
        else:
            small += 1
    return judged, small


def group_label(domain):
    region, stage = str(domain).split('|')
    return f"{region} · {stage}"


def below_groups(df, threshold=0.80, estimate='point estimate'):
    """Judged region × stage rows under the rule, lowest first."""
    judged, _small = judged_groups(df)
    rows = [r for r in judged if compared_value(r, estimate) < threshold]
    return sorted(rows, key=lambda r: compared_value(r, estimate))


def verdict_text(df, threshold=0.80, estimate='point estimate'):
    """The verdict line and its level (R5.5, with the user's wording for a
    pass): ``Every region ≥ 80 %`` (``ok``) or ``Check parietal · NREM2
    (54 %): below 80 %.`` (``warn``). Returns ``(text, level)``."""
    judged, small = judged_groups(df)
    lower = estimate == 'lower 95 % bound'
    t = pct(threshold)
    if not judged:
        return ('Not judged: no region × stage group has enough decided '
                'events yet.', 'warn')
    below = below_groups(df, threshold, estimate)
    if not below:
        text = (f"Every region's lower bound ≥ {t} %" if lower
                else f"Every region ≥ {t} %")
        if small:
            text += f" ({small} group(s) too small to judge)"
        return text, 'ok'

    def g(r):
        if lower:
            return (f"{group_label(r['domain'])} (lower bound "
                    f"{pct(r['ci_lo'])} %)")
        return f"{group_label(r['domain'])} ({pct(r['p_hat'])} %)"
    names = [g(r) for r in below[:3]]
    if len(below) == 1:
        body = names[0]
    elif len(below) == 2:
        body = f"{names[0]} and {names[1]}"
    elif len(below) == 3:
        body = f"{names[0]}, {names[1]} and {names[2]}"
    else:
        body = f"{', '.join(names)} and {len(below) - 3} more"
    return f"Check {body}: below {t} %.", 'warn'


def reason_counts(labels, reason_label):
    """``[(token, label lower case, count)]`` of rejected events, by count
    (then label)."""
    counts = {}
    for dec, reason in labels.values():
        if dec == 'reject':
            counts[reason or ''] = counts.get(reason or '', 0) + 1
    rows = [(tok, (reason_label.get(tok, tok) if tok else
                   'no reason given').lower(), n)
            for tok, n in counts.items()]
    return sorted(rows, key=lambda t: (-t[2], t[1]))


def report_sentence(name, labels, n_total, reason_label, links=False,
                    accent='#5a8fce'):
    """``TK reviewed 120 of 120 sampled events: 119 accepted, 1 rejected
    (artefact 1).`` (R5.0); with ``links`` the reasons and ``{u} unsure``
    are HTML links ``reason:{token}`` / ``unsure``."""
    import html as _html
    vals = list(labels.values())
    a = sum(1 for d, _r in vals if d == 'accept')
    r = sum(1 for d, _r in vals if d == 'reject')
    u = sum(1 for d, _r in vals if d == 'unsure')

    def link(href, text):
        t = _html.escape(text)
        return (f"<a href='{href}' style='color:{accent}'>{t}</a>"
                if links else text)
    reasons = reason_counts(labels, reason_label)
    parts = [link(f"reason:{tok}", f"{lab} {n}")
             for tok, lab, n in reasons[:3]]
    rtxt = ', '.join(parts)
    if len(reasons) > 3:
        rtxt += f" and {len(reasons) - 3} more"
    esc = _html.escape if links else (lambda x: x)
    text = (f"{esc(str(name))} reviewed {len(vals)} of {int(n_total)} "
            f"sampled events: {a} accepted, {r} rejected")
    if r:
        text += f" ({rtxt})"
    if u:
        text += ", " + link('unsure', f"{u} unsure")
    return text + '.'


def precision_line(df):
    """``Estimated precision: 99 % (95 % confidence 95–100 %)`` from the
    whole-night (weighted) row, or ``None``."""
    row = _row(df, 'scope', 'all')
    if row is None:
        return None
    p, lo, hi = (_finite(row.get(k)) for k in ('p_hat', 'ci_lo', 'ci_hi'))
    if p is None or lo is None or hi is None:
        return None
    return (f"Estimated precision: {pct(p)} % (95 % confidence {pct(lo)}–"
            f"{pct(hi)} %)")


def precision_tip(name):
    return (f"Of the events the detector found, the share {name} accepted, "
            f"weighted to all of the night's events. Unsure events are left "
            f"out.")


def report_title(design, reviewer):
    """``Spindles · Moelle2011 11–16 Hz · reviewer TK``."""
    ev = EVENT_PLURAL.get(design.get('event_type'), design.get('event_type'))
    ev = str(ev)[:1].upper() + str(ev)[1:]
    return (f"{ev} · {design.get('method')} "
            f"{float(design.get('freq_lower')):g}–"
            f"{float(design.get('freq_upper')):g} Hz · reviewer {reviewer}")


def second_reviewer_line(current, labels, n_total):
    """``(text, finished)`` of the second-reviewer line (R5.0); agreement is
    given only once ``current`` has decided every sample event."""
    mine = labels.get(current, {})
    finished = len(mine) >= int(n_total) > 0
    others = [o for o in labels if o not in (current, 'consensus')
              and labels[o]]
    if not others:
        return NO_SECOND, finished
    other = max(others, key=lambda o: (len(set(labels[o]) & set(mine)),
                                       len(labels[o])))
    if not finished:
        return (f"{other} has also reviewed this sample. Agreement is shown "
                f"once you have decided all {int(n_total)} events."), False
    res = agreement(mine, labels[other])
    if res is None:
        return (f"Second reviewer {other}: no events you both decided."), True
    n = res['n_shared']
    agree = int(round(res['percent_agreement'] * n))
    k = res['kappa']
    ktxt = (f"Cohen's kappa {k:.2f}" if k == k
            else "Cohen's kappa not defined")
    return (f"Second reviewer {other}: agreement {100 * agree / n:.0f} % on "
            f"{n} events you both decided ({ktxt})."), True


def agreement(a, b):
    """``label_agreement`` on the events both reviewers decided."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    res = _rs().label_agreement({u: a[u][0] for u in common},
                                {u: b[u][0] for u in common})
    res['uuids'] = common
    res['disagree'] = [u for u in common if a[u][0] != b[u][0]]
    return res


def csv_filename(design):
    lo, hi = design.get('freq_lower'), design.get('freq_upper')
    return (f"{design.get('subject')}_{design.get('event_type')}_"
            f"{design.get('method')}_{float(lo):g}-{float(hi):g}Hz_"
            f"review_precision.csv")


def csv_rows(design, frames, threshold, estimate):
    """One row per reviewer × region × stage plus each reviewer's whole-night
    row, with :data:`CSV_COLUMNS`."""
    run_ids = design.get('run_ids')
    try:
        run_ids = ';'.join(json.loads(run_ids))
    except (TypeError, ValueError):
        run_ids = str(run_ids or '')
    out = []
    for reviewer, df in frames.items():
        judged, _small = judged_groups(df)
        whole_ok = bool(judged) and not below_groups(df, threshold, estimate)
        for _, r in df.iterrows():
            if r['domain_type'] == 'region_stage':
                region, stage = str(r['domain']).split('|')
                v = compared_value(r.to_dict(), estimate)
                ok = (int(r['n_decided'] or 0) >= MIN_DECIDED and v is not None
                      and v >= threshold)
            elif r['domain_type'] == 'scope':
                region, stage, ok = 'all', 'all', whole_ok
            else:
                continue
            out.append({
                'subject': design.get('subject'),
                'event_type': design.get('event_type'),
                'method': design.get('method'),
                'freq_lower': design.get('freq_lower'),
                'freq_upper': design.get('freq_upper'),
                'run_id': run_ids, 'sample_id': design.get('sample_id'),
                'seed': design.get('seed'), 'reviewer': reviewer,
                'region': region, 'stage': stage,
                'n_decided': r['n_decided'], 'n_accepted': r['n_accept'],
                'n_rejected': r['n_reject'], 'n_unsure': r['n_unsure'],
                'precision': r['p_hat'], 'ci_low': r['ci_lo'],
                'ci_high': r['ci_hi'], 'weighted': 1,
                'pool_threshold': threshold, 'pool_estimate': estimate,
                'poolable': int(bool(ok)),
                'turtlewave_version': r['turtlewave_version']})
    return out


# ---------------------------------------------------------------------------
# Draw preview
# ---------------------------------------------------------------------------

def prepare(conn, scope):
    """The scope's population read once, when the library offers
    ``review_sampling.prepare_population`` (else ``None``). Raises
    ``ValueError`` as the draw would."""
    fn = getattr(_rs(), 'prepare_population', None)
    if fn is None:
        return None
    return fn(conn, scope=dict(scope))


def prepared_subject(prepared):
    """Subject of a prepared population: ``prepared['population']
    ['subject']`` (what ``prepare_population`` returns), else a top-level
    ``subject`` key or attribute."""
    if prepared is None:
        return None
    pop = (prepared.get('population') if isinstance(prepared, dict)
           else getattr(prepared, 'population', None))
    if isinstance(pop, dict) and pop.get('subject') is not None:
        return pop['subject']
    if isinstance(prepared, dict):
        return prepared.get('subject')
    return getattr(prepared, 'subject', None)


def preview(conn, scope, n_total, seed, n_shared=30, prepared=None):
    """What Draw would draw, from the library's ``preview_allocation``
    (reads only). Raises ``ValueError`` as the draw would.

    Returns
    -------
    (list of dict, dict, dict)
        Rows ``group, n_events, n_sample, n_flagged, census`` in region then
        stage order; totals; the library's preview dict (``subject``,
        ``flag_available``, ``exists``, ``sample_id`` …).
    """
    kw = {'prepared': prepared} if prepared is not None else {}
    pv = _rs().preview_allocation(conn, scope=dict(scope),
                                  n_total=int(n_total), seed=int(seed),
                                  n_shared=int(n_shared), **kw)
    flag = bool(pv['flag_available'])
    by = {(c['region'], c['stage']): c for c in pv['cells']}
    out = []
    for region in REGION_ORDER:
        for stage in STAGE_ORDER:
            c = by.get((region, stage))
            if c is None:
                continue
            out.append({'group': f"{region} · {stage}", 'n_events': c['N_h'],
                        'n_sample': c['n_h'],
                        'n_flagged': (c['parts'].get('F', {}).get('n', 0)
                                      if flag else None),
                        'census': bool(c['census'])})
    tot = {'n_events': sum(r['n_events'] for r in out),
           'n_sample': sum(r['n_sample'] for r in out),
           'n_flagged': (sum(r['n_flagged'] for r in out) if flag else None),
           'groups': len(out)}
    return out, tot, pv


def existing_sample_note(design, counts):
    """The dialog's note when a sample already exists for the scope."""
    date = str(design.get('drawn_at') or '')[:10]
    n = design.get('_n_rows') or design.get('n_total') or '?'
    who = [f"{rv} ({c})" for rv, c in sorted(counts.items()) if c]
    dec = (' and has decisions from ' + ', '.join(who[:-1])
           + (' and ' if len(who) > 1 else '') + who[-1]) if who else ''
    return (f"A sample of {n} was drawn on {date}{dec}. Drawing a new one "
            f"keeps that sample and its decisions; the Precision report will "
            f"use the new sample.")
