"""Review-sample mode and precision report: text, numbers and database reads.

Qt-free. The library (``turtlewave_hdEEG.review_sampling``) draws the sample
and estimates precision; this module turns its results into the exact text of
the UX spec (``_scratch/design/event-decision-spec.md`` revision 2, sections 2
and 11) and reads what the GUI needs from ``review_sample_designs`` /
``review_samples``. The library is imported lazily so ``frontend`` still
imports without it.
"""

import datetime as _dt
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
PRECISION_MEANING = ('What precision means here: of the events the detector '
                     'found, the share the reviewer accepted. Unsure events '
                     'are left out and counted separately.')
CONVENTION_NOTE = ('A lab convention, not a published standard. Change it '
                   'here; it is saved with the export.')
NO_DECISIONS = 'No sample events decided yet. Press ] in the Epochs tab to start.'
NO_SECOND = 'A second reviewer has not decided any of this sample yet.'

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
    sample's events.

    Uses the library's own label reader (``review_sampling._labels``), so a
    label is void here exactly when it is void in ``review_precision``: the
    event is gone, or its end time moved more than 0.05 s since the draw.
    """
    rs = _rs()
    rows = rs._sample_rows(conn, sample_id)
    labels, _stale, _missing = rs._labels(conn, sample_id, rows)
    return {rv: {u: (v[0], v[1]) for u, v in lab.items()}
            for rv, lab in labels.items()}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def stratum_text(row):
    """``parietal · NREM2 · flagged: off-band`` (spec section 2)."""
    head = f"{row.get('region')} · {row.get('stage')}"
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


def cell_text(row, threshold=0.80, estimate='point estimate'):
    """``0.93  (0.70–0.99)  n 15``; ``n too small (6)``; ``—``.

    Returns ``(text, level)``: level ``'warn'`` below the rule,
    ``'muted'`` too small, ``None`` otherwise.
    """
    if row is None:
        return '—', 'muted'
    n = int(row.get('n_decided') or 0)
    if n < MIN_DECIDED:
        return f"n too small ({n})", 'muted'
    p, lo, hi = (_finite(row.get(k)) for k in ('p_hat', 'ci_lo', 'ci_hi'))
    if p is None:
        return f"n too small ({n})", 'muted'
    text = f"{p:.2f}  ({lo:.2f}–{hi:.2f})  n {n}"
    v = compared_value(row, estimate)
    if v is not None and v < threshold:
        return text + ' below', 'warn'
    return text, None


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


def verdict_text(df, threshold=0.80, estimate='point estimate'):
    """The pooling verdict sentence (spec section 11)."""
    judged, small = judged_groups(df)
    lower = estimate == 'lower 95 % bound'
    below = [r for r in judged if compared_value(r, estimate) < threshold]
    tail = (f" {small} group{'s' if small != 1 else ''} with too few events "
            f"{'were' if small != 1 else 'was'} not judged." if small else '')
    if not judged:
        return ('Not judged: no region × stage group has enough decided '
                'events yet.' + tail)
    if not below:
        return (f"Poolable under this rule: all {len(judged)} region × stage "
                f"groups have precision of at least {threshold:.2f}." + tail)

    def g(r):
        if lower:
            return (f"{group_label(r['domain'])} lower bound "
                    f"{r['ci_lo']:.2f}")
        return (f"{group_label(r['domain'])} {r['p_hat']:.2f} "
                f"({r['ci_lo']:.2f}–{r['ci_hi']:.2f})")
    names = [g(r) for r in below[:3]]
    if len(below) == 1:
        body = f"{names[0]} is below {threshold:.2f}."
    else:
        more = f" and {len(below) - 3} more" if len(below) > 3 else ''
        body = (', '.join(names[:-1]) + f" and {names[-1]}{more} are below "
                f"{threshold:.2f}.")
    return 'Not poolable under this rule: ' + body + tail


def reason_lines(labels, reason_label):
    """``[(label, count)]`` of this reviewer's rejected sample events, by
    count descending, and the M7 line counts ``(fp_artifact, fp_other)``."""
    from turtlewave_hdEEG.dbwrite import review_category
    counts = {}
    fp = {'FP-artifact': 0, 'FP-other': 0}
    for dec, reason in labels.values():
        if dec != 'reject':
            continue
        counts[reason] = counts.get(reason, 0) + 1
        fp[review_category(dec, reason)] += 1
    rows = sorted(((reason_label.get(r, 'No reason given') if r else
                    'No reason given', n) for r, n in counts.items()),
                  key=lambda t: (-t[1], t[0]))
    return rows, (fp['FP-artifact'], fp['FP-other'])


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


def agreement_text(res):
    """``Agreed on 34 of 40 (85 %) …`` -- percent agreement three-way
    (accept / reject / unsure) over every event both reviewers decided;
    kappa over the events neither marked unsure."""
    n = res['n_shared']
    agree = int(round(res['percent_agreement'] * n))
    k, m = res['kappa'], res['n_both_decided']
    if m == 0:
        ktxt = ("Cohen's kappa not computed: no events both reviewers "
                "decided without an unsure")
    elif k != k:
        ktxt = (f"Cohen's kappa not defined on the {m} events neither "
                f"reviewer marked unsure (no variation in decisions)")
    else:
        ktxt = (f"Cohen's kappa {k:.2f} on the {m} events neither reviewer "
                f"marked unsure")
    return (f"Agreed on {agree} of {n} ({100 * agree / n:.0f} %, accept / "
            f"reject / unsure, over every event both reviewers decided) · "
            f"{ktxt}")


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
        whole_ok = verdict_text(df, threshold, estimate).startswith('Poolable')
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

def population_for(conn, scope):
    """The library's in-scope population of ``scope`` (no write).

    Uses ``review_sampling._population``, the function the draw itself
    reads, so the preview is exactly what Draw will draw. Raises
    ``ValueError`` as the draw would (empty or mixed-parameter scope).
    """
    return _rs()._population(conn, dict(scope))


def preview_allocation(pop, n_total, seed, n_shared=30):
    """Per region × stage: events in scope, in sample, of which flagged.

    Returns
    -------
    (list of dict, dict)
        Rows ``group, n_events, n_sample, n_flagged, census`` in region then
        stage order, and the totals.
    """
    rs = _rs()
    rows = rs.select_sample(
        [(e['uuid'], e['region'], e['stage'], e['flagged'])
         for e in pop['events']], int(seed), int(n_total), int(n_shared),
        flag_available=pop['flag_available'])
    sizes = {}
    for e in pop['events']:
        key = (e['region'], e['stage'])
        if e['region'] in REGION_ORDER and e['stage'] in STAGE_ORDER:
            sizes[key] = sizes.get(key, 0) + 1
    drawn, flagged = {}, {}
    for r in rows:
        key = (r['region'], r['stage'])
        drawn[key] = drawn.get(key, 0) + 1
        flagged[key] = flagged.get(key, 0) + bool(r['flagged'])
    out = []
    for region in REGION_ORDER:
        for stage in STAGE_ORDER:
            key = (region, stage)
            if key not in sizes:
                continue
            out.append({'group': f"{region} · {stage}",
                        'n_events': sizes[key], 'n_sample': drawn.get(key, 0),
                        'n_flagged': (flagged.get(key, 0)
                                      if pop['flag_available'] else None),
                        'census': drawn.get(key, 0) >= sizes[key]})
    tot = {'n_events': sum(sizes.values()), 'n_sample': len(rows),
           'n_flagged': (sum(flagged.values()) if pop['flag_available']
                         else None), 'groups': len(out)}
    return out, tot


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


def now_hhmm():
    return _dt.datetime.now().strftime('%H:%M')
