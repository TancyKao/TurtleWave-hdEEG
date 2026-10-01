"""Stratified review sample and design-weighted precision of detected events.

Implements revision 2 of ``_scratch/research/event-review/sample_spec.md``
(Gate A SOUND, 2026-10-01). Per subject x detection scope:

* **Population.** The current ``events`` rows of one detection scope --
  (subject, event_type, method, freq_lower, freq_upper, stage token) --
  across every ``run_id`` in it, minus channels marked ``drop`` in
  ``channel_qc`` (or passed in ``exclude_channels``), channels outside the
  five scalp regions of :func:`~turtlewave_hdEEG.utils.region_from_label`,
  and events whose own epoch stage is not NREM2 / NREM3.
* **Flag**, frozen at draw time from the stored review figures (spindles):
  ``in_band = 0`` or ``low_prominence = 1`` or ``amp_ratio < 2.0`` or
  ``amp_ratio`` NULL or ``near_splice = 1``. Slow waves / K-complexes drop the
  ``low_prominence`` term. ``near_bound`` is display-only. A scope without
  figures is stratified by region x stage only (``flagged`` NULL); inside a
  scope with figures, events whose figures were not computed
  (``near_splice`` NULL) are neither flagged nor unflagged and form a third
  sub-cell per region x stage with a proportional share.
* **Allocation.** Equal over region x stage cells (``floor(T/H)``, remainder to
  the largest cells, census when a cell is smaller than its share, surplus
  redistributed). Inside a cell the flagged half takes
  ``ceil(max(0.5, N_hF/N_h) * n_h)`` slots, leaving at least one for a
  non-empty unflagged half.
* **Draw.** Permanent random numbers: ``sha256(f"{seed}:main:{uuid}")``; the
  first ``n_k`` events of each sub-cell by (key, uuid) -- simple random
  sampling without replacement within the sub-cell.
* **Estimator.** ``P = sum(w a) / sum(w d)`` with ``w = N_k / m_k`` (``m_k`` =
  events actually labelled in the sub-cell), Wilson interval on the Kish
  effective sample size; MOVER (Newcombe square-and-add) for differences.

Everything here is numpy + hashlib + sqlite3 (pandas for the returned
frame); no Qt.
"""

import datetime
import hashlib
import json
import logging
import math
import sqlite3
import uuid as _uuid
from collections import Counter, defaultdict

import numpy as np

from . import dbwrite
from .utils import region_from_label

logger = logging.getLogger('turtlewave_hdEEG.review_sampling')

#: Strata of the design, in a fixed order (sets the remainder tie-break).
REGIONS = ('frontal', 'central', 'parietal', 'temporal', 'occipital')
STAGES = ('NREM2', 'NREM3')

#: Bumped when the draw rule changes; part of the design hash.
DESIGN_VERSION = 2

#: Namespace of :func:`draw_review_sample` sample ids (fixed forever).
SAMPLE_NAMESPACE = _uuid.UUID('6a3c2f1e-8d4b-5c7a-9e0f-1b2c3d4e5f60')

Z = 1.959963984540054  #: two-sided 95 % normal quantile
AMP_RATIO_CUTOFF = 2.0  #: ``amp_ratio`` below this flags an event
END_TIME_TOL_S = 0.05  #: a label survives a re-detection within this
CENSUS_LABEL_N = 8  #: cells with fewer events are labelled "census, N < 8"
TOP_UP_N = 12  #: events added by :func:`top_up_region`

#: Pooling-rule conventions (sample_spec section 3), user-changeable.
POOLING = {'scope_p': 0.80, 'scope_lo': 0.70, 'scope_unsure': 0.15,
           'region_hi': 0.70, 'region_p': 0.70}

# params_json keys that legitimately differ between runs of one scope.
_PARAMS_IGNORED = ('interpolated_channels', 'channels', 'timestamp',
                   'event_figures')

#: Columns of :func:`compute_review_precision`, in order.
PRECISION_COLUMNS = (
    'sample_id', 'reviewer', 'label_source', 'domain_type', 'domain',
    'n_sampled', 'n_reviewed', 'n_accept', 'n_reject', 'n_unsure',
    'n_decided', 'p_hat', 'ci_lo', 'ci_hi', 'n_eff', 'unsure_rate',
    'p_lo_bound', 'p_hi_bound', 'reason_counts', 'census', 'note', 'verdict',
    'computed_at', 'turtlewave_version', 'design_hash',
)

_SAMPLE_COLUMNS = (
    'sample_id', 'uuid', 'subject', 'run_id', 'event_type', 'channel',
    'start_time', 'end_time', 'cell', 'region', 'stage', 'flagged',
    'flag_components', 'pop_k', 'n_k', 'weight', 'prn', 'sort_key', 'irr_key',
    'is_shared', 'draw_round', 'seed', 'design_hash', 'drawn_at',
)

_DDL = (
    '''CREATE TABLE IF NOT EXISTS review_sample_designs (
        sample_id TEXT PRIMARY KEY,
        subject TEXT,
        event_type TEXT,
        method TEXT,
        freq_lower REAL,
        freq_upper REAL,
        stage_scope TEXT,            -- events.stage token of the scope
        run_ids TEXT,                -- JSON list of the scope's run_ids
        seed INTEGER,
        design_json TEXT,            -- canonical JSON of the design
        design_hash TEXT,
        population_hash TEXT,        -- sha256 of sorted (uuid, end, flag)
        n_population INTEGER,        -- scope events before exclusions
        n_in_scope INTEGER,          -- events the sample represents
        n_out_of_scope TEXT,         -- JSON {reason: count}
        flag_available INTEGER,
        n_no_figures TEXT,           -- JSON {region|stage: n}, figures NULL
        top_ups TEXT,                -- JSON {region: {n, at}}
        drawn_at TEXT,
        turtlewave_version TEXT
    )''',
    '''CREATE TABLE IF NOT EXISTS review_samples (
        sample_id TEXT NOT NULL,
        uuid TEXT NOT NULL,
        subject TEXT,
        run_id TEXT,                 -- the event's run at draw (provenance)
        event_type TEXT,
        channel TEXT,
        start_time REAL,
        end_time REAL,               -- at draw: the re-detection guard
        cell TEXT,                   -- 'region|stage|F' / '|U' / '|A'
        region TEXT,
        stage TEXT,                  -- the event's own epoch stage
        flagged INTEGER,             -- 0/1, NULL when no figures
        flag_components TEXT,        -- comma list of the terms that fired
        pop_k INTEGER,               -- N_k, sub-cell population size
                                     -- (not 'N_k': SQLite names are
                                     -- case-insensitive, n_k exists)
        n_k INTEGER,                 -- sub-cell sample size (after top-up)
        weight REAL,                 -- pop_k / n_k (estimator recomputes)
        prn REAL,                    -- key / 2**64
        sort_key TEXT,               -- 16-hex of the 64-bit key
        irr_key TEXT,
        is_shared INTEGER,           -- inter-rater subset
        draw_round INTEGER,          -- 0 main draw, 1 top-up
        seed INTEGER,
        design_hash TEXT,
        drawn_at TEXT,
        PRIMARY KEY (sample_id, uuid)
    )''',
    'CREATE INDEX IF NOT EXISTS idx_review_samples_cell '
    'ON review_samples(sample_id, cell, sort_key)',
    '''CREATE TABLE IF NOT EXISTS review_precision (
        sample_id TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        label_source TEXT NOT NULL,  -- 'primary' or 'consensus'
        domain_type TEXT NOT NULL,   -- scope/region/region_stage/flag/difference
        domain TEXT NOT NULL,
        n_sampled INTEGER, n_reviewed INTEGER, n_accept INTEGER,
        n_reject INTEGER, n_unsure INTEGER, n_decided INTEGER,
        p_hat REAL, ci_lo REAL, ci_hi REAL, n_eff REAL, unsure_rate REAL,
        p_lo_bound REAL, p_hi_bound REAL,
        reason_counts TEXT,          -- JSON {reason: n} over rejects
        census INTEGER, note TEXT, verdict TEXT,
        computed_at TEXT, turtlewave_version TEXT, design_hash TEXT,
        PRIMARY KEY (sample_id, reviewer, label_source, domain_type, domain)
    )''',
)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _fmt(x):
    """Stable text of a number (or None) for hashes and ids."""
    if x is None:
        return 'None'
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return 'nan'
    return repr(round(xf, 6))


def _sql_value(v):
    """numpy scalar -> Python; NaN -> None (SQLite NULL)."""
    if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def prn_key(seed, uuid, salt='main'):
    """Permanent random number of one event as a 64-bit integer.

    Parameters
    ----------
    seed : int
        Sample seed.
    uuid : str
        ``events.uuid``.
    salt : str, optional
        ``'main'`` for the draw, ``'irr'`` for the inter-rater subset.

    Returns
    -------
    int
        First 8 bytes of ``sha256(f"{seed}:{salt}:{uuid}")``, big-endian;
        divide by ``2**64`` for a uniform number in [0, 1).
    """
    digest = hashlib.sha256(f'{seed}:{salt}:{uuid}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def wilson(p, n, z=Z):
    """Wilson score interval for a proportion on an (effective) count.

    Parameters
    ----------
    p : float
        Point estimate.
    n : float
        Count, or Kish effective sample size.
    z : float, optional
        Normal quantile (default 95 %).

    Returns
    -------
    tuple of float
        ``(lower, upper)``; ``(nan, nan)`` when ``n <= 0`` or ``p`` is NaN.
    """
    if n is None or not n > 0 or p is None or not np.isfinite(p):
        return (float('nan'), float('nan'))
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(max(p * (1.0 - p), 0.0) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mover_difference(p1, lo1, hi1, p2, lo2, hi2):
    """MOVER (Newcombe square-and-add) interval for ``p1 - p2``.

    Parameters
    ----------
    p1, lo1, hi1, p2, lo2, hi2 : float
        Estimates and interval limits of the two (independent) domains.

    Returns
    -------
    tuple of float
        ``(difference, lower, upper)``; NaN when either input is NaN.
    """
    vals = (p1, lo1, hi1, p2, lo2, hi2)
    if any(v is None or not np.isfinite(v) for v in vals):
        return (float('nan'),) * 3
    d = p1 - p2
    return (d, d - math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2),
            d + math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2))


def weighted_precision(weights, decisions):
    """Design-weighted precision with a Kish-Wilson interval.

    Parameters
    ----------
    weights : array-like of float
        ``N_k / m_k`` of each labelled event.
    decisions : sequence of {'accept', 'reject', 'unsure'}
        The labels, aligned with ``weights``.

    Returns
    -------
    dict
        ``p_hat`` (accepts over decided, weighted), ``ci_lo``, ``ci_hi``,
        ``n_eff`` (Kish, over decided events), ``unsure_rate``,
        ``p_lo_bound`` (unsure counted as rejects), ``p_hi_bound`` (as
        accepts), and unweighted ``n_accept`` / ``n_reject`` / ``n_unsure``.
        Estimates are NaN when nothing was decided.
    """
    w = np.asarray(weights, dtype=float)
    lab = np.asarray(list(decisions), dtype=object)
    a = lab == 'accept'
    r = lab == 'reject'
    u = lab == 'unsure'
    d = a | r
    out = {'n_accept': int(a.sum()), 'n_reject': int(r.sum()),
           'n_unsure': int(u.sum()), 'n_decided': int(d.sum())}
    nan = float('nan')
    wsum = w.sum()
    wd = w[d].sum()
    if wd > 0:
        p = float((w * a).sum() / wd)
        neff = float(wd ** 2 / (w[d] ** 2).sum())
    else:
        p, neff = nan, 0.0
    lo, hi = wilson(p, neff)
    out.update(p_hat=p, ci_lo=lo, ci_hi=hi, n_eff=neff,
               unsure_rate=float((w * u).sum() / wsum) if wsum > 0 else nan,
               p_lo_bound=float((w * a).sum() / wsum) if wsum > 0 else nan,
               p_hi_bound=(float((w * (a | u)).sum() / wsum)
                           if wsum > 0 else nan))
    return out


def allocate_cells(sizes, total):
    """Equal allocation over non-empty cells with census and redistribution.

    Parameters
    ----------
    sizes : dict
        ``{cell: N_h}``; cells must be orderable for the remainder tie-break
        (largest ``N_h`` first, then cell order).
    total : int
        Events to allocate.

    Returns
    -------
    dict
        ``{cell: n_h}`` over non-empty cells. ``sum == min(total, sum(N))``.
    """
    alloc = {}
    open_ = {c for c, n in sizes.items() if n > 0}
    left = int(total)
    while open_:
        share = left // len(open_)
        small = {c for c in open_ if sizes[c] <= share}
        if not small:
            rem = left - share * len(open_)
            for i, c in enumerate(sorted(open_, key=lambda c: (-sizes[c], c))):
                alloc[c] = share + (1 if i < rem else 0)
            break
        for c in small:
            alloc[c] = sizes[c]
            left -= sizes[c]
        open_ -= small
    return alloc


def split_flagged(n_h, n_flag, n_unflag):
    """Split a cell's allocation between its flagged and unflagged halves.

    ``s = max(0.5, N_F / N_h)``; flagged take ``ceil(s * n_h)`` (at most
    ``n_h - 1`` when unflagged events exist), unflagged the rest, any
    shortfall goes back to the other half.

    Parameters
    ----------
    n_h : int
        Cell allocation.
    n_flag, n_unflag : int
        Population sizes of the two halves.

    Returns
    -------
    tuple of int
        ``(n_flagged, n_unflagged)``; each non-empty half gets at least one
        when ``n_h >= 2``.
    """
    if n_flag + n_unflag <= n_h:
        return n_flag, n_unflag
    s = max(0.5, n_flag / float(n_flag + n_unflag))
    nf = min(n_flag, int(math.ceil(s * n_h - 1e-12)))
    if n_unflag > 0:
        nf = min(nf, n_h - 1)
    nf = max(nf, 0)
    nu = min(n_unflag, n_h - nf)
    nf = min(n_flag, n_h - nu)
    return nf, nu


def split_cell(n_h, n_flag, n_unflag, n_nofig=0):
    """Split a cell's allocation over its flagged / unflagged / no-figure parts.

    Events whose figures were not computed (``near_splice`` NULL in a scope
    that has figures) are neither flagged nor unflagged; they form a third
    sub-cell taking its proportional share ``round(n_h N_N / N_h)`` (at
    least one, leaving one slot for each non-empty half). The remaining
    slots go to :func:`split_flagged`.

    Parameters
    ----------
    n_h : int
        Cell allocation.
    n_flag, n_unflag, n_nofig : int
        Population sizes of the three parts.

    Returns
    -------
    tuple of int
        ``(n_flagged, n_unflagged, n_nofig)``.

    Raises
    ------
    ValueError
        When ``n_h`` is too small to sample every non-empty part.
    """
    total = n_flag + n_unflag + n_nofig
    if total <= n_h:
        return n_flag, n_unflag, n_nofig
    nn = 0
    if n_nofig:
        need = (n_flag > 0) + (n_unflag > 0)
        nn = min(n_nofig, max(1, int(math.floor(n_h * n_nofig / total + 0.5))),
                 n_h - need)
        if nn < 1:
            raise ValueError(f"A cell allocation of {n_h} cannot sample its "
                             f"flagged, unflagged and no-figure parts; raise "
                             f"n_total.")
    nf, nu = split_flagged(n_h - nn, n_flag, n_unflag)
    if n_nofig:
        nn = min(n_nofig, n_h - nf - nu)
    return nf, nu, nn


def _half_of(flagged, flag_available):
    if not flag_available:
        return 'A'
    if flagged is None:
        return 'N'
    return 'F' if flagged else 'U'


def _cell_order(cell):
    region, stage = cell
    return (REGIONS.index(region), STAGES.index(stage))


def select_sample(events, seed, total=120, n_shared=30, flag_available=None):
    """The design's draw on an in-memory population (no database).

    Parameters
    ----------
    events : list of tuple
        ``(uuid, region, stage, flagged)`` of every in-scope event;
        ``flagged`` is True/False, or None when the event has no figures.
        Regions/stages outside :data:`REGIONS` / :data:`STAGES` are ignored.
    seed : int
        Sample seed.
    total : int, optional
        Sample size T (default 120). Must be ``>= 2 H``.
    n_shared : int, optional
        Inter-rater subset size (default 30; 0 for none).
    flag_available : bool or None, optional
        Whether the scope has figures. ``None`` infers it: True when any event
        has a non-None flag. Without figures every cell is one sub-cell
        (``|A``); with figures, events with ``flagged`` None form ``|N``.

    Returns
    -------
    list of dict
        One per sampled event: ``uuid``, ``region``, ``stage``, ``flagged``,
        ``cell``, ``N_k``, ``n_k``, ``weight``, ``key`` (int), ``irr_key``
        (int), ``is_shared``.

    Raises
    ------
    ValueError
        Empty population, ``total < 2 H``, or a cell too small for its parts.
    """
    cells = defaultdict(list)
    for uid, region, stage, flagged in events:
        if region in REGIONS and stage in STAGES:
            cells[(region, stage)].append((uid, flagged))
    if not cells:
        raise ValueError("No in-scope events to sample.")
    if flag_available is None:
        flag_available = any(f is not None for _, _, _, f in events)
    n_cells = len(cells)
    if total < 2 * n_cells:
        raise ValueError(f"total={total} is below 2 x {n_cells} cells; every "
                         f"cell needs room for a flagged and an unflagged "
                         f"event.")
    sizes = {c: len(v) for c, v in cells.items()}
    order = sorted(cells, key=_cell_order)
    rank = {c: i for i, c in enumerate(order)}
    alloc = allocate_cells({rank[c]: n for c, n in sizes.items()}, total)
    rows = []
    for c in order:
        n_h = alloc[rank[c]]
        groups = defaultdict(list)
        for u, f in cells[c]:
            groups[_half_of(f, flag_available)].append(u)
        if flag_available:
            nf, nu, nn = split_cell(n_h, len(groups['F']), len(groups['U']),
                                    len(groups['N']))
            take = {'F': nf, 'U': nu, 'N': nn}
        else:
            take = {'A': n_h}
        for half, uids in groups.items():
            if not uids:
                continue
            n_k = take[half]
            if n_k < 1:
                raise RuntimeError(f"Allocation left non-empty sub-cell "
                                   f"{c}|{half} unsampled.")
            ranked = sorted((prn_key(seed, u), u) for u in uids)
            for key, uid in ranked[:n_k]:
                rows.append({
                    'uuid': uid, 'region': c[0], 'stage': c[1],
                    'flagged': {'F': True, 'U': False}.get(half),
                    'cell': f'{c[0]}|{c[1]}|{half}', 'N_k': len(uids),
                    'n_k': n_k, 'weight': len(uids) / n_k, 'key': key,
                    'irr_key': prn_key(seed, uid, 'irr'), 'is_shared': False,
                })
    if n_shared > 0:
        per_cell = int(math.ceil(n_shared / float(n_cells)))
        by_cell = defaultdict(list)
        for r in rows:
            by_cell[(r['region'], r['stage'])].append(r)
        for members in by_cell.values():
            for r in sorted(members, key=lambda r: (r['irr_key'], r['uuid'])
                            )[:per_cell]:
                r['is_shared'] = True
    return rows


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_review_sampling_schema(conn, logger=None):
    """Create ``review_sample_designs``, ``review_samples``, ``review_precision``.

    Idempotent and additive; touches no other table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection (not closed).
    logger : logging.Logger or None, optional
        For a creation message.

    Returns
    -------
    bool
        True when any table was created by this call.
    """
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
    needed = {'review_sample_designs', 'review_samples', 'review_precision',
              'idx_review_samples_cell'}
    if needed <= have:
        if 'n_no_figures' not in _cols(conn, 'review_sample_designs'):
            conn.execute("ALTER TABLE review_sample_designs "
                         "ADD COLUMN n_no_figures TEXT")
            conn.commit()
        return False
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    (logger or globals()['logger']).info(
        "Created review sampling tables: %s", sorted(needed - have))
    return True


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _resolve_scope(conn, run_id, event_type, scope):
    if scope is not None:
        need = ('event_type', 'method', 'freq_lower', 'freq_upper', 'stage')
        missing = [k for k in need if k not in scope]
        if missing:
            raise ValueError(f"scope is missing {missing}")
        return {k: scope[k] for k in need}
    if run_id is None:
        raise ValueError("Give run_id (to resolve the scope) or scope=.")
    sql = ("SELECT DISTINCT event_type, method, freq_lower, freq_upper, stage "
           "FROM events WHERE run_id = ?")
    params = [str(run_id)]
    if event_type is not None:
        sql += " AND event_type = ?"
        params.append(event_type)
    found = conn.execute(sql, params).fetchall()
    if not found:
        raise ValueError(f"No events for run_id={run_id!r}"
                         + (f", event_type={event_type!r}" if event_type
                            else ''))
    if len(found) > 1:
        raise ValueError(f"run_id={run_id!r} spans {len(found)} detection "
                         f"scopes {found}; pass scope= explicitly.")
    et, method, lo, hi, stage = found[0]
    return {'event_type': et, 'method': method, 'freq_lower': lo,
            'freq_upper': hi, 'stage': stage}


def _normalised_params(text):
    try:
        params = json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {'_unparseable': str(text)}
    if not isinstance(params, dict):
        return {'_value': params}
    return {k: v for k, v in params.items() if k not in _PARAMS_IGNORED}


def _population(conn, sc, exclude_channels=None, use_channel_qc=True,
                amp_ratio_cutoff=AMP_RATIO_CUTOFF, allow_mixed_params=False,
                log=None):
    """Read and classify one scope's events. Raises on an empty scope."""
    log = log or logger
    ev_cols = set(_cols(conn, 'events'))
    if not ev_cols:
        raise ValueError("The database has no events table.")
    stage_expr = ("COALESCE(epoch_stage, stage)" if 'epoch_stage' in ev_cols
                  else 'stage')
    spindle = str(sc['event_type']).startswith('spindle')
    fig_needed = ['in_band', 'amp_ratio', 'near_splice'] + (
        ['low_prominence'] if spindle else [])
    fig_present = all(c in ev_cols for c in fig_needed)
    fig_sel = ', '.join(fig_needed) if fig_present else ', '.join(
        ['NULL'] * len(fig_needed))
    spellings = list(dbwrite.method_spellings(sc['method']))
    ph = ', '.join('?' * len(spellings))
    rows = conn.execute(
        f"SELECT uuid, channel, start_time, end_time, run_id, {stage_expr}, "
        f"{fig_sel} FROM events WHERE event_type = ? AND method IN ({ph}) "
        f"AND freq_lower IS ? AND freq_upper IS ? AND stage IS ?",
        [sc['event_type']] + spellings
        + [sc['freq_lower'], sc['freq_upper'], sc['stage']]).fetchall()
    if not rows:
        raise ValueError(f"No events in scope {sc}.")

    run_ids = sorted({r[4] for r in rows if r[4] is not None})
    subject = None
    if run_ids and 'subject' in _cols(conn, 'detection_runs'):
        ph_r = ', '.join('?' * len(run_ids))
        runs = conn.execute(
            f"SELECT run_id, subject, params_json FROM detection_runs "
            f"WHERE run_id IN ({ph_r})", run_ids).fetchall()
        subjects = sorted({r[1] for r in runs if r[1]})
        if len(subjects) > 1:
            raise ValueError(f"Scope spans subjects {subjects}.")
        subject = subjects[0] if subjects else None
        norm = {r[0]: json.dumps(_normalised_params(r[2]), sort_keys=True,
                                 default=str) for r in runs}
        if len(set(norm.values())) > 1:
            msg = (f"The scope's runs {sorted(norm)} were detected with "
                   f"different parameters (params_json differs beyond "
                   f"{_PARAMS_IGNORED}); their precision is not one number.")
            if not allow_mixed_params:
                raise ValueError(msg + " Pass allow_mixed_params=True to "
                                       "draw anyway (recorded in the design).")
            log.warning(msg + " Drawing anyway (allow_mixed_params=True).")

    excluded = set(exclude_channels or ())
    if use_channel_qc and {'channel', 'verdict'} <= set(
            _cols(conn, 'channel_qc')):
        has_et = 'event_type' in _cols(conn, 'channel_qc')
        sql = "SELECT channel FROM channel_qc WHERE verdict = 'drop'"
        params = []
        if has_et:
            sql += " AND (event_type = ? OR event_type IS NULL)"
            params.append(sc['event_type'])
        excluded |= {r[0] for r in conn.execute(sql, params)}

    fig_rows = [r for r in rows if r[6 + fig_needed.index('near_splice')]
                is not None] if fig_present else []
    flag_available = bool(fig_present and fig_rows)
    if not flag_available:
        log.warning("Scope %s has no stored review figures; stratifying by "
                    "region x stage only (flagged = NULL).", sc)

    out_counts = Counter()
    no_figures = Counter()
    events = []
    region_cache = {}
    for (uid, chan, start, end, run, stage, *figs) in rows:
        if chan in excluded:
            out_counts['channel_excluded'] += 1
            continue
        region = region_cache.get(chan)
        if region is None:
            region = region_cache[chan] = region_from_label(chan)
        if region not in REGIONS:
            out_counts[f'region:{region}'] += 1
            continue
        if stage not in STAGES:
            out_counts[f'stage:{stage}'] += 1
            continue
        comps = None
        f = dict(zip(fig_needed, figs))
        if flag_available and f['near_splice'] is None:
            # Figures not computed for this event (e.g. the channel's
            # figure step failed): neither flagged nor unflagged.
            no_figures[f'{region}|{stage}'] += 1
        elif flag_available:
            comps = []
            if f['in_band'] is not None and int(f['in_band']) == 0:
                comps.append('off_band')
            if spindle and f['low_prominence'] is not None and int(
                    f['low_prominence']) == 1:
                comps.append('low_prominence')
            amp = f['amp_ratio']
            if amp is None or not np.isfinite(float(amp)):
                comps.append('amp_ratio_missing')
            elif float(amp) < amp_ratio_cutoff:
                comps.append('low_amp_ratio')
            if f['near_splice'] is not None and int(f['near_splice']) == 1:
                comps.append('near_splice')
        events.append({'uuid': uid, 'channel': chan, 'start_time': start,
                       'end_time': end, 'run_id': run, 'region': region,
                       'stage': stage,
                       'flagged': None if comps is None else bool(comps),
                       'flag_components': None if comps is None
                       else ','.join(comps)})
    if not events:
        raise ValueError(
            f"No in-scope events in {sc}: every event fell outside the five "
            f"scalp regions x NREM2/NREM3 or on an excluded channel "
            f"({dict(out_counts)}). EGI 'E<n>' labels map to 'other' and need "
            f"a montage-specific region map; refusing to draw nothing.")
    n_other = sum(v for k, v in out_counts.items() if k.startswith('region:'))
    if n_other > 0.10 * len(rows):
        log.warning("%d of %d scope events are on channels outside the five "
                    "scalp regions (%s)", n_other, len(rows),
                    {k: v for k, v in out_counts.items()
                     if k.startswith('region:')})
    pop_hash = hashlib.sha256('\n'.join(
        f"{e['uuid']}|{_fmt(e['end_time'])}|{e['flagged']}"
        for e in sorted(events, key=lambda e: e['uuid'])).encode()).hexdigest()
    try:
        from .event_metrics import SPEC_REVISION
    except ImportError:
        SPEC_REVISION = None
    if no_figures:
        log.warning("%d in-scope events have no figures (near_splice NULL); "
                    "they are sampled as their own sub-cell per region x "
                    "stage: %s", sum(no_figures.values()), dict(no_figures))
    return {'events': events, 'n_population': len(rows),
            'no_figures': dict(sorted(no_figures.items())),
            'out_of_scope': dict(out_counts), 'excluded': sorted(excluded),
            'run_ids': run_ids, 'subject': subject,
            'flag_available': flag_available,
            'flag_components': (fig_needed if flag_available else []),
            'figures_revision': SPEC_REVISION, 'population_hash': pop_hash}


def _design(pop, total, n_shared, amp_ratio_cutoff, allow_mixed_params):
    return {
        'design_version': DESIGN_VERSION, 'total': int(total),
        'n_shared': int(n_shared),
        'flag_share_rule': 'ceil(max(0.5, N_hF/N_h) * n_h), <= n_h - 1',
        'flag_available': pop['flag_available'],
        'flag_terms': ('in_band=0 | low_prominence=1 | amp_ratio<cutoff | '
                       'amp_ratio NULL | near_splice=1'
                       if pop['flag_available'] else None),
        'flag_columns': pop['flag_components'],
        'amp_ratio_cutoff': float(amp_ratio_cutoff),
        'event_metrics_revision': pop['figures_revision'],
        'excluded_channels': pop['excluded'],
        'allow_mixed_params': bool(allow_mixed_params),
        'regions': list(REGIONS), 'stages': list(STAGES),
        'census_label_n': CENSUS_LABEL_N,
        'end_time_tolerance_s': END_TIME_TOL_S,
    }


def _sample_id(sc, subject, seed, design_hash, population_hash):
    key = '|'.join([str(subject), str(sc['event_type']), str(sc['method']),
                    _fmt(sc['freq_lower']), _fmt(sc['freq_upper']),
                    str(sc['stage']), str(int(seed)), design_hash,
                    population_hash])
    return str(_uuid.uuid5(SAMPLE_NAMESPACE, key))


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw_review_sample(conn, run_id=None, event_type=None, n_total=120,
                       seed=1, *, scope=None, n_shared=30,
                       amp_ratio_cutoff=AMP_RATIO_CUTOFF,
                       exclude_channels=None, allow_mixed_params=False,
                       logger=None):
    """Draw (or return the existing) stratified review sample of one scope.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection to ``neural_events.db``.
    run_id : str or None, optional
        Any run of the scope; used only to resolve the scope. The sample
        covers every run of that scope.
    event_type : str or None, optional
        Narrows the run to one event type when it carries several.
    n_total : int, optional
        Sample size T (default 120). Must be at least twice the number of
        non-empty region x stage cells.
    seed : int, optional
        Sample seed (default 1).
    scope : dict or None, optional
        ``event_type``, ``method``, ``freq_lower``, ``freq_upper``, ``stage``
        (``events.stage`` token); overrides ``run_id``.
    n_shared : int, optional
        Inter-rater subset size (default 30), ``ceil(n_shared / H)`` per cell.
    amp_ratio_cutoff : float, optional
        Flag cutoff on ``amp_ratio`` (default 2.0).
    exclude_channels : iterable of str or None, optional
        Channels to leave out, on top of ``channel_qc`` ``drop`` verdicts.
    allow_mixed_params : bool, optional
        Draw even when the scope's runs differ in ``params_json``.
    logger : logging.Logger or None, optional
        Defaults to ``turtlewave_hdEEG.review_sampling``.

    Returns
    -------
    str
        The ``sample_id``. Drawing again with identical inputs and an
        unchanged population returns the same id and writes nothing.

    Raises
    ------
    ValueError
        Unresolvable or empty scope (including EGI ``E<n>`` montages, whose
        labels map to region ``'other'``), mixed parameters without
        ``allow_mixed_params``, or ``n_total < 2 H``.
    """
    log = logger or globals()['logger']
    sc = _resolve_scope(conn, run_id, event_type, scope)
    pop = _population(conn, sc, exclude_channels=exclude_channels,
                      amp_ratio_cutoff=amp_ratio_cutoff,
                      allow_mixed_params=allow_mixed_params, log=log)
    design = _design(pop, n_total, n_shared, amp_ratio_cutoff,
                     allow_mixed_params)
    design_json = json.dumps(design, sort_keys=True)
    design_hash = hashlib.sha256(design_json.encode()).hexdigest()
    sample_id = _sample_id(sc, pop['subject'], seed, design_hash,
                           pop['population_hash'])

    ensure_review_sampling_schema(conn, logger=log)
    if conn.execute("SELECT 1 FROM review_sample_designs WHERE sample_id = ?",
                    (sample_id,)).fetchone():
        log.info("Review sample %s already drawn; returning it unchanged.",
                 sample_id)
        return sample_id

    by_uuid = {e['uuid']: e for e in pop['events']}
    drawn = select_sample([(e['uuid'], e['region'], e['stage'], e['flagged'])
                           for e in pop['events']], seed, n_total, n_shared,
                          flag_available=pop['flag_available'])
    now = _now()
    version = dbwrite.provenance()['turtlewave_version']
    records = []
    for r in drawn:
        e = by_uuid[r['uuid']]
        records.append((
            sample_id, r['uuid'], pop['subject'], e['run_id'],
            sc['event_type'], e['channel'], e['start_time'], e['end_time'],
            r['cell'], r['region'], r['stage'],
            None if r['flagged'] is None else int(r['flagged']),
            e['flag_components'], r['N_k'], r['n_k'], r['weight'],
            r['key'] / 2.0 ** 64, f"{r['key']:016x}", f"{r['irr_key']:016x}",
            int(r['is_shared']), 0, int(seed), design_hash, now))
    try:
        conn.execute('BEGIN')
        conn.execute(
            "INSERT INTO review_sample_designs (sample_id, subject, "
            "event_type, method, freq_lower, freq_upper, stage_scope, "
            "run_ids, seed, design_json, design_hash, population_hash, "
            "n_population, n_in_scope, n_out_of_scope, flag_available, "
            "n_no_figures, top_ups, drawn_at, turtlewave_version) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sample_id, pop['subject'], sc['event_type'], sc['method'],
             sc['freq_lower'], sc['freq_upper'], sc['stage'],
             json.dumps(pop['run_ids']), int(seed), design_json, design_hash,
             pop['population_hash'], pop['n_population'],
             len(pop['events']), json.dumps(pop['out_of_scope'],
                                            sort_keys=True),
             int(pop['flag_available']), json.dumps(pop['no_figures']),
             json.dumps({}), now, version))
        conn.executemany(
            f"INSERT INTO review_samples ({', '.join(_SAMPLE_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_SAMPLE_COLUMNS))})", records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    log.info("Drew review sample %s: %d events (%d shared) from %d in-scope "
             "of %d scope events over %d cells; out of scope %s",
             sample_id, len(records), sum(r[19] for r in records),
             len(pop['events']), pop['n_population'],
             len({(r[9], r[10]) for r in records}), pop['out_of_scope'])
    return sample_id


def _design_row(conn, sample_id):
    cur = conn.execute("SELECT * FROM review_sample_designs WHERE sample_id = ?",
                       (sample_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No review sample {sample_id!r}.")
    return dict(zip([d[0] for d in cur.description], row))


def _sample_rows(conn, sample_id):
    cur = conn.execute(
        f"SELECT {', '.join(_SAMPLE_COLUMNS)} FROM review_samples "
        f"WHERE sample_id = ? ORDER BY cell, sort_key", (sample_id,))
    return [dict(zip(_SAMPLE_COLUMNS, r)) for r in cur.fetchall()]


def top_up_region(conn, sample_id, region, n=TOP_UP_N, logger=None):
    """Add the next ``n`` events by key to one region of a sample (once).

    The extra events are spread equally over the region's region x stage
    cells with room left, re-split between flagged and unflagged halves by
    the draw rule (never shrinking a half), and taken as the next events by
    permanent random number, so each sub-cell stays a simple random sample.
    ``n_k`` and ``weight`` of the region's rows are updated.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection.
    sample_id : str
        Sample to extend.
    region : str
        One of :data:`REGIONS`.
    n : int, optional
        Events to add (default 12).
    logger : logging.Logger or None, optional

    Returns
    -------
    int
        Events added (fewer than ``n`` when the region runs out).

    Raises
    ------
    ValueError
        Unknown sample or region, a region already topped up, or a
        population that changed since the draw (draw a new sample instead).
    """
    log = logger or globals()['logger']
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    des = _design_row(conn, sample_id)
    top_ups = json.loads(des['top_ups'] or '{}')
    if region in top_ups:
        raise ValueError(f"Region {region!r} of sample {sample_id} was already "
                         f"topped up; one top-up only.")
    design = json.loads(des['design_json'])
    sc = {'event_type': des['event_type'], 'method': des['method'],
          'freq_lower': des['freq_lower'], 'freq_upper': des['freq_upper'],
          'stage': des['stage_scope']}
    pop = _population(conn, sc, exclude_channels=design['excluded_channels'],
                      use_channel_qc=False,
                      amp_ratio_cutoff=design['amp_ratio_cutoff'],
                      allow_mixed_params=True, log=log)
    if pop['population_hash'] != des['population_hash']:
        raise ValueError(f"The population of sample {sample_id} changed since "
                         f"it was drawn; draw a new sample instead of topping "
                         f"up.")
    seed = int(des['seed'])
    rows = _sample_rows(conn, sample_id)
    have = {r['uuid'] for r in rows}
    cur_k = Counter(r['cell'] for r in rows)
    halves = defaultdict(list)
    for e in pop['events']:
        if e['region'] != region:
            continue
        half = _half_of(e['flagged'], bool(des['flag_available']))
        halves[(e['stage'], half)].append(e)
    stages = sorted({s for s, _ in halves}, key=STAGES.index)
    spare = {}
    for s in stages:
        N_h = sum(len(halves.get((s, h), [])) for h in 'AFUN')
        n_h = sum(cur_k.get(f'{region}|{s}|{h}', 0) for h in 'AFUN')
        spare[STAGES.index(s)] = N_h - n_h
    extra = allocate_cells(spare, n)
    by_uuid = {e['uuid']: e for e in pop['events']}
    now = _now()
    new_records, updates = [], []
    for s in stages:
        add = extra.get(STAGES.index(s), 0)
        if add <= 0:
            continue
        cell = f'{region}|{s}'
        cur = {h: cur_k.get(f'{cell}|{h}', 0) for h in 'AFUN'}
        N = {h: len(halves.get((s, h), [])) for h in 'AFUN'}
        n_h = sum(cur.values()) + add
        if N['A']:
            want = {'A': n_h}
        else:
            nf, nu, nn = split_cell(n_h, N['F'], N['U'], N['N'])
            want = {'F': nf, 'U': nu, 'N': nn}
            # Never shrink a part already drawn; take the excess back from
            # the part with the most new slots.
            for h in want:
                want[h] = max(want[h], cur[h])
            while sum(want.values()) > n_h:
                h = max(want, key=lambda h: want[h] - cur[h])
                if want[h] <= cur[h]:
                    break
                want[h] -= 1
            want = {h: min(v, N[h]) for h, v in want.items()}
        for h, n_k in want.items():
            members = halves.get((s, h), [])
            if not members:
                continue
            ranked = sorted((prn_key(seed, e['uuid']), e['uuid'])
                            for e in members)[:n_k]
            for key, uid in ranked:
                if uid in have:
                    continue
                e = by_uuid[uid]
                new_records.append((
                    sample_id, uid, des['subject'], e['run_id'],
                    des['event_type'], e['channel'], e['start_time'],
                    e['end_time'], f'{cell}|{h}', region, s,
                    None if e['flagged'] is None else int(e['flagged']),
                    e['flag_components'], len(members), n_k,
                    len(members) / n_k, key / 2.0 ** 64, f'{key:016x}',
                    f"{prn_key(seed, uid, 'irr'):016x}", 0, 1, seed,
                    des['design_hash'], now))
            updates.append((n_k, len(members) / n_k, sample_id, f'{cell}|{h}'))
    top_ups[region] = {'n': len(new_records), 'at': now}
    try:
        conn.execute('BEGIN')
        conn.executemany(
            f"INSERT INTO review_samples ({', '.join(_SAMPLE_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_SAMPLE_COLUMNS))})", new_records)
        conn.executemany("UPDATE review_samples SET n_k = ?, weight = ? "
                         "WHERE sample_id = ? AND cell = ?", updates)
        conn.execute("UPDATE review_sample_designs SET top_ups = ? "
                     "WHERE sample_id = ?", (json.dumps(top_ups), sample_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    log.info("Topped up region %s of sample %s with %d events",
             region, sample_id, len(new_records))
    return len(new_records)


# ---------------------------------------------------------------------------
# Labels, progress and precision
# ---------------------------------------------------------------------------

def _labels(conn, sample_id, rows, reviewer=None):
    """Valid labels on the sample's events.

    Returns ``(labels, stale, missing)``: ``labels[reviewer][uuid] =
    (decision, reason, reviewed_at)``; ``stale`` uuids whose current end time
    moved more than :data:`END_TIME_TOL_S` from the sample's stored end time
    (all labels void); ``missing`` uuids no longer in ``events``. A label made
    in another sample (``event_reviews.sample_id``, when that column exists)
    counts only when that sample stored an end time within the tolerance of
    this sample's.
    """
    uuids = [r['uuid'] for r in rows]
    stored_end = {r['uuid']: r['end_time'] for r in rows}
    now_end = {}
    for i in range(0, len(uuids), 900):
        chunk = uuids[i:i + 900]
        now_end.update(conn.execute(
            f"SELECT uuid, end_time FROM events WHERE uuid IN "
            f"({', '.join('?' * len(chunk))})", chunk).fetchall())
    missing = {u for u in uuids if u not in now_end}
    stale = {u for u in uuids if u in now_end and stored_end[u] is not None
             and now_end[u] is not None
             and abs(now_end[u] - stored_end[u]) > END_TIME_TOL_S}
    labels = defaultdict(dict)
    rev_cols = _cols(conn, 'event_reviews')
    if not rev_cols:
        return labels, stale, missing
    has_sid = 'sample_id' in rev_cols
    sel = ("uuid, reviewer, decision, reason, reviewed_at, "
           + ("sample_id" if has_sid else "NULL"))
    other_end = {}
    for i in range(0, len(uuids), 900):
        chunk = uuids[i:i + 900]
        sql = (f"SELECT {sel} FROM event_reviews WHERE uuid IN "
               f"({', '.join('?' * len(chunk))})")
        params = list(chunk)
        if reviewer is not None:
            sql += " AND reviewer = ?"
            params.append(reviewer)
        for uid, rv, dec, reason, at, sid in conn.execute(sql, params):
            if uid in stale or uid in missing:
                continue
            if sid is not None and sid != sample_id:
                if (sid, uid) not in other_end:
                    hit = conn.execute(
                        "SELECT end_time FROM review_samples WHERE "
                        "sample_id = ? AND uuid = ?", (sid, uid)).fetchone()
                    other_end[(sid, uid)] = hit[0] if hit else None
                old = other_end[(sid, uid)]
                if old is None or stored_end[uid] is None or abs(
                        old - stored_end[uid]) > END_TIME_TOL_S:
                    continue
            labels[rv][uid] = (dec, reason, at)
    return labels, stale, missing


def _interleaved(rows):
    """Rows ordered by rank within their sub-cell, then cell order."""
    rank = Counter()
    keyed = []
    for r in sorted(rows, key=lambda r: (r['cell'], r['sort_key'])):
        keyed.append((rank[r['cell']], r['cell'], r))
        rank[r['cell']] += 1
    return [r for _, _, r in sorted(keyed, key=lambda t: (t[0], t[1]))]


def sample_progress(conn, sample_id, reviewer=None, shared_only=False):
    """How far a reviewer has got through a sample, and what to show next.

    Parameters
    ----------
    conn : sqlite3.Connection
    sample_id : str
    reviewer : str or None, optional
        Count this reviewer's labels; ``None`` counts an event labelled by
        anyone other than ``'consensus'``.
    shared_only : bool, optional
        Restrict to the inter-rater subset (the second rater's view).

    Returns
    -------
    dict
        ``n_total``, ``n_reviewed``, ``n_accept``, ``n_reject``,
        ``n_unsure``, ``n_shared``, ``n_shared_reviewed``, ``by_cell``
        (``{cell: {'n', 'reviewed'}}``), ``stale`` and ``missing`` (uuid
        lists), ``next_uuids`` (unlabelled events in presentation order:
        rank within sub-cell, then cell, so stopping early stays balanced),
        ``median_gap_s`` (median gap between this reviewer's consecutive
        decisions on the sample, gaps over 5 min dropped; None if < 2).
    """
    _design_row(conn, sample_id)
    rows = _sample_rows(conn, sample_id)
    if shared_only:
        rows = [r for r in rows if r['is_shared']]
    labels, stale, missing = _labels(conn, sample_id, rows, reviewer)
    if reviewer is None:
        merged = {}
        for rv, lab in labels.items():
            if rv != 'consensus':
                for u, v in lab.items():
                    merged.setdefault(u, v)
    else:
        merged = labels.get(reviewer, {})
    by_cell = defaultdict(lambda: {'n': 0, 'reviewed': 0})
    for r in rows:
        by_cell[r['cell']]['n'] += 1
        by_cell[r['cell']]['reviewed'] += r['uuid'] in merged
    dec = Counter(v[0] for v in merged.values())
    gaps = None
    times = sorted(datetime.datetime.fromisoformat(v[2])
                   for v in merged.values() if v[2])
    if len(times) >= 2:
        diffs = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        diffs = [d for d in diffs if d <= 300]
        gaps = float(np.median(diffs)) if diffs else None
    return {
        'sample_id': sample_id, 'n_total': len(rows),
        'n_reviewed': len(merged), 'n_accept': dec['accept'],
        'n_reject': dec['reject'], 'n_unsure': dec['unsure'],
        'n_shared': sum(1 for r in rows if r['is_shared']),
        'n_shared_reviewed': sum(1 for r in rows
                                 if r['is_shared'] and r['uuid'] in merged),
        'by_cell': dict(by_cell), 'stale': sorted(stale),
        'missing': sorted(missing),
        'next_uuids': [r['uuid'] for r in _interleaved(rows)
                       if r['uuid'] not in merged and r['uuid'] not in missing],
        'median_gap_s': gaps,
    }


def _resolve_primary(labels, rows):
    shared = {r['uuid'] for r in rows if r['is_shared']}
    counts = {rv: sum(1 for u in lab if u not in shared)
              for rv, lab in labels.items() if rv != 'consensus'}
    counts = {rv: n for rv, n in counts.items() if n > 0}
    if len(counts) != 1:
        raise ValueError(f"Cannot tell the primary reviewer from labels on "
                         f"non-shared events ({counts}); pass reviewer=.")
    return next(iter(counts))


def _domain_row(rows_lab, n_k_strata, m_k):
    """Estimate over one domain's labelled rows; flags missing strata.

    ``m_k`` counts the labelled events of each WHOLE sub-cell, not of the
    domain: a domain that cuts across sub-cells (``near_splice``) is a
    random subset of them, so its events carry the sub-cell's weight
    ``pop_k / m_k``.
    """
    w = [r['pop_k'] / m_k[r['cell']] for r, _ in rows_lab]
    est = weighted_precision(w, [lab[0] for _, lab in rows_lab])
    reasons = Counter(lab[1] or 'none' for _, lab in rows_lab
                      if lab[0] == 'reject')
    est['reason_counts'] = json.dumps(dict(sorted(reasons.items())))
    est['n_reviewed'] = len(rows_lab)
    est['missing_strata'] = sorted(k for k in n_k_strata if m_k[k] == 0)
    return est


def compute_review_precision(conn, sample_id, reviewer=None,
                             label_source='primary', write=True,
                             logger=None):
    """Design-weighted precision of a sample, per domain.

    Weights are ``N_k / m_k`` with ``m_k`` the events of the sub-cell that
    carry a valid label, so a partly reviewed sample is still estimated
    correctly as long as events were reviewed in key order (warned
    otherwise). Domains: ``scope/all``, ``region/<r>``,
    ``region_stage/<r>|<s>``, ``flag/flagged``, ``flag/unflagged``,
    ``flag/near_splice`` and ``flag/no_figures`` (events whose figures were
    not computed, sampled as their own ``|N`` sub-cell; when figures exist),
    and
    ``difference/flagged-unflagged`` (MOVER).

    Parameters
    ----------
    conn : sqlite3.Connection
    sample_id : str
    reviewer : str or None, optional
        Primary reviewer; ``None`` resolves the only non-``consensus``
        reviewer of the non-shared events (raises if ambiguous).
    label_source : {'primary', 'consensus'}, optional
        ``'consensus'`` substitutes reviewer ``consensus`` labels where they
        exist (sensitivity row beside the primary-only figure).
    write : bool, optional
        Replace this (sample, reviewer, label_source)'s rows in
        ``review_precision``. Default True.
    logger : logging.Logger or None, optional

    Returns
    -------
    pandas.DataFrame
        Columns :data:`PRECISION_COLUMNS`. ``verdict``: scope
        ``TRUSTWORTHY`` / ``NOT_TRUSTWORTHY``; region ``IN`` / ``EXCLUDE`` /
        ``TOP_UP``; any domain ``INCOMPLETE`` when a sub-cell has no label,
        ``NO_DATA`` when nothing was decided.
    """
    import pandas as pd

    log = logger or globals()['logger']
    if label_source not in ('primary', 'consensus'):
        raise ValueError("label_source must be 'primary' or 'consensus'")
    des = _design_row(conn, sample_id)
    rows = _sample_rows(conn, sample_id)
    labels, stale, missing = _labels(conn, sample_id, rows)
    if stale or missing:
        log.warning("Sample %s: %d stale and %d missing events carry no "
                    "valid label (re-review or draw anew).", sample_id,
                    len(stale), len(missing))
    if reviewer is None:
        reviewer = _resolve_primary(labels, rows)
    lab = dict(labels.get(reviewer, {}))
    if label_source == 'consensus':
        lab.update(labels.get('consensus', {}))
    top_ups = json.loads(des['top_ups'] or '{}')

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r['cell']].append(r)
    for cell, members in by_cell.items():
        done = [r['uuid'] in lab for r in members]   # members sorted by key
        if any(done) and not all(done) and done.index(False) < sum(done):
            log.warning("Sample %s cell %s was not reviewed in key order; the "
                        "within-cell sample is no longer a simple random "
                        "sample.", sample_id, cell)

    m_k = Counter(r['cell'] for r in rows if r['uuid'] in lab)

    def domain(name_type, name, pred):
        members = [r for r in rows if pred(r)]
        strata = {r['cell'] for r in members}
        rl = [(r, lab[r['uuid']]) for r in members if r['uuid'] in lab]
        est = _domain_row(rl, strata, m_k)
        est.update(domain_type=name_type, domain=name,
                   n_sampled=len(members))
        est['census'] = int(all(r['n_k'] >= r['pop_k'] for r in members))
        n_pop = {}
        for r in members:
            n_pop[r['cell']] = r['pop_k']
        est['note'] = ('census, N < 8' if name_type == 'region_stage'
                       and sum(n_pop.values()) < CENSUS_LABEL_N else None)
        return est

    out = [domain('scope', 'all', lambda r: True)]
    for region in REGIONS:
        if any(r['region'] == region for r in rows):
            out.append(domain('region', region,
                              lambda r, g=region: r['region'] == g))
    for region in REGIONS:
        for stage in STAGES:
            if any(r['region'] == region and r['stage'] == stage
                   for r in rows):
                out.append(domain(
                    'region_stage', f'{region}|{stage}',
                    lambda r, g=region, s=stage: (r['region'] == g
                                                  and r['stage'] == s)))
    if des['flag_available']:
        f = domain('flag', 'flagged', lambda r: r['flagged'] == 1)
        u = domain('flag', 'unflagged', lambda r: r['flagged'] == 0)
        out += [f, u]
        if any(r['cell'].endswith('|N') for r in rows):
            out.append(domain('flag', 'no_figures',
                              lambda r: r['cell'].endswith('|N')))
        if any('near_splice' in (r['flag_components'] or '') for r in rows):
            out.append(domain('flag', 'near_splice', lambda r: 'near_splice'
                              in (r['flag_components'] or '')))
        d, lo, hi = mover_difference(f['p_hat'], f['ci_lo'], f['ci_hi'],
                                     u['p_hat'], u['ci_lo'], u['ci_hi'])
        out.append({'domain_type': 'difference',
                    'domain': 'flagged-unflagged', 'p_hat': d, 'ci_lo': lo,
                    'ci_hi': hi, 'note': 'MOVER (Newcombe square-and-add)'})

    for est in out:
        p, lo, hi = est.get('p_hat'), est.get('ci_lo'), est.get('ci_hi')
        verdict = None
        if est['domain_type'] in ('scope', 'region'):
            if est.get('missing_strata'):
                verdict = 'INCOMPLETE'
            elif not est.get('n_decided'):
                verdict = 'NO_DATA'
            elif est['domain_type'] == 'scope':
                verdict = ('TRUSTWORTHY' if p >= POOLING['scope_p']
                           and lo >= POOLING['scope_lo']
                           and est['unsure_rate'] <= POOLING['scope_unsure']
                           else 'NOT_TRUSTWORTHY')
            elif hi < POOLING['region_hi']:
                verdict = 'EXCLUDE'
            elif p < POOLING['region_p'] and est['domain'] not in top_ups:
                verdict = 'TOP_UP'
            else:
                verdict = 'IN'
        elif est.get('missing_strata'):
            verdict = 'INCOMPLETE'
        est['verdict'] = verdict

    now = _now()
    version = dbwrite.provenance()['turtlewave_version']
    for est in out:
        est.update(sample_id=sample_id, reviewer=reviewer,
                   label_source=label_source, computed_at=now,
                   turtlewave_version=version, design_hash=des['design_hash'])
    df = pd.DataFrame([{c: est.get(c) for c in PRECISION_COLUMNS}
                       for est in out], columns=list(PRECISION_COLUMNS))
    if write:
        ensure_review_sampling_schema(conn, logger=log)
        try:
            conn.execute('BEGIN')
            conn.execute("DELETE FROM review_precision WHERE sample_id = ? "
                         "AND reviewer = ? AND label_source = ?",
                         (sample_id, reviewer, label_source))
            conn.executemany(
                f"INSERT INTO review_precision ({', '.join(PRECISION_COLUMNS)})"
                f" VALUES ({', '.join('?' * len(PRECISION_COLUMNS))})",
                [tuple(_sql_value(v) for v in rec)
                 for rec in df.itertuples(index=False, name=None)])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return df


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------

def _kappa(a, b):
    """Cohen's kappa on accept/reject pairs; NaN when undefined."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    n = len(a)
    if n == 0:
        return float('nan')
    po = float(np.mean(a == b))
    pa, pb = np.mean(a == 'accept'), np.mean(b == 'accept')
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    if pe >= 1.0:
        return float('nan')
    return (po - pe) / (1.0 - pe)


def _agreement_block(a, b):
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    n = len(a)
    agree = int(np.sum(a == b))
    pa = agree / n if n else float('nan')
    both = (a != 'unsure') & (b != 'unsure')
    ab, bb = a[both], b[both]
    n_aa = int(np.sum((ab == 'accept') & (bb == 'accept')))
    n_rr = int(np.sum((ab == 'reject') & (bb == 'reject')))
    n_off = int(both.sum()) - n_aa - n_rr
    p_pos = (2 * n_aa / (2 * n_aa + n_off)) if (2 * n_aa + n_off) else float('nan')
    p_neg = (2 * n_rr / (2 * n_rr + n_off)) if (2 * n_rr + n_off) else float('nan')
    lo, hi = wilson(pa, n)
    return {'n_shared': n, 'n_both_decided': int(both.sum()),
            'percent_agreement': pa, 'percent_agreement_ci': (lo, hi),
            'kappa': _kappa(ab, bb), 'positive_agreement': p_pos,
            'negative_agreement': p_neg}


def label_agreement(a, b, subjects=None, flags=None, n_boot=2000, seed=0):
    """Agreement between two raters' labels on the same events.

    Parameters
    ----------
    a, b : sequence or mapping
        Decisions (``'accept'``/``'reject'``/``'unsure'``) of the two raters.
        Two equal-length sequences are aligned by position; two mappings
        (``uuid -> decision``) are aligned on their common keys (sorted).
    subjects : sequence or mapping or None, optional
        Subject of each event, for the subject-clustered bootstrap of kappa
        (computed only with at least 10 subjects).
    flags : sequence or mapping or None, optional
        Flag of each event, for a by-flag breakdown.
    n_boot : int, optional
        Bootstrap resamples (default 2000).
    seed : int, optional
        Bootstrap seed (default 0), returned for provenance.

    Returns
    -------
    dict
        ``n_shared``, ``n_both_decided``, ``percent_agreement`` (three-way)
        with Wilson ``percent_agreement_ci``, ``confusion`` (``{a: {b: n}}``),
        ``kappa`` (Cohen, events both decided; NaN when chance agreement is
        1), ``kappa_ci`` (bootstrap percentile, or None) and ``kappa_ci_note``,
        ``positive_agreement`` and ``negative_agreement``, ``by_flag``,
        ``protocol_pass`` (kappa >= 0.60 and agreement >= 0.80),
        ``bootstrap_seed``.
    """
    def _align(x, keys):
        if x is None:
            return None
        if hasattr(x, 'keys'):
            return [x[k] for k in keys]
        return list(x)

    if hasattr(a, 'keys') and hasattr(b, 'keys'):
        keys = sorted(set(a.keys()) & set(b.keys()))
        la, lb = [a[k] for k in keys], [b[k] for k in keys]
        subs, fls = _align(subjects, keys), _align(flags, keys)
    else:
        la, lb = list(a), list(b)
        if len(la) != len(lb):
            raise ValueError("a and b differ in length")
        subs = None if subjects is None else list(subjects)
        fls = None if flags is None else list(flags)
    vocab = set(dbwrite.REVIEW_DECISIONS)
    bad = {x for x in la + lb if x not in vocab}
    if bad:
        raise ValueError(f"Unknown decisions {bad}; expected {sorted(vocab)}")
    out = _agreement_block(la, lb)
    conf = {x: {y: 0 for y in dbwrite.REVIEW_DECISIONS}
            for x in dbwrite.REVIEW_DECISIONS}
    for x, y in zip(la, lb):
        conf[x][y] += 1
    out['confusion'] = conf
    out['kappa_ci'] = None
    out['bootstrap_seed'] = int(seed)
    if subs is None:
        out['kappa_ci_note'] = 'no subjects given'
    else:
        uniq = sorted(set(subs), key=str)
        if len(uniq) < 10:
            out['kappa_ci_note'] = (f'{len(uniq)} subjects (< 10): clustered '
                                    f'bootstrap not reported')
        else:
            idx = defaultdict(list)
            for i, s in enumerate(subs):
                idx[s].append(i)
            arr_a = np.asarray(la, dtype=object)
            arr_b = np.asarray(lb, dtype=object)
            rng = np.random.default_rng(seed)
            ks = []
            for _ in range(int(n_boot)):
                pick = rng.integers(0, len(uniq), len(uniq))
                ii = np.concatenate([idx[uniq[j]] for j in pick])
                both = (arr_a[ii] != 'unsure') & (arr_b[ii] != 'unsure')
                ks.append(_kappa(arr_a[ii][both], arr_b[ii][both]))
            ks = np.asarray(ks, dtype=float)
            ks = ks[np.isfinite(ks)]
            out['kappa_ci'] = ((float(np.percentile(ks, 2.5)),
                                float(np.percentile(ks, 97.5)))
                               if len(ks) else None)
            out['kappa_ci_note'] = (f'subject-clustered bootstrap, '
                                    f'{len(uniq)} subjects, {n_boot} resamples')
    out['by_flag'] = None
    if fls is not None:
        out['by_flag'] = {}
        for fv in sorted(set(fls), key=str):
            sel = [i for i, f in enumerate(fls) if f == fv]
            out['by_flag'][fv] = _agreement_block([la[i] for i in sel],
                                                  [lb[i] for i in sel])
    k, pa = out['kappa'], out['percent_agreement']
    out['protocol_pass'] = bool(np.isfinite(k) and k >= 0.60
                                and np.isfinite(pa) and pa >= 0.80)
    return out
