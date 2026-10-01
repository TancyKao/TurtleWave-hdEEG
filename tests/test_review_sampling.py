#!/usr/bin/env python3
"""Stratified review sample and design-weighted precision (``review_sampling``).

Method Spec: ``_scratch/research/event-review/sample_spec.md`` revision 2.
Asserted here:

* Allocation: equal over cells, census and surplus redistribution, sums to T,
  adaptive flagged share ``ceil(max(0.5, N_F/N_h) n_h)`` capped at
  ``n_h - 1``, every non-empty sub-cell sampled, ``T < 2H`` refused.
* The hash draw is simple random sampling without replacement within a cell
  (all subsets equally likely).
* The estimator: equal weights give the plain proportion and exact-count
  Wilson; the Method Spec worked example reproduces 0.727 / 18.0 /
  0.496-0.878; MOVER differences.
* Pre-registered coverage simulation (seeds 9000-9399, precision 0.90
  unflagged / 0.55 flagged, 8 % unsure) at flag prevalence 0.21 and 0.85:
  coverage >= 0.94, |bias| <= 0.02; negative control (equal precision) gap
  interval includes 0 in >= 93 % of seeds.
* Database: the population is the detection scope across runs; a draw is
  deterministic and idempotent; a changed population or seed gives a new id;
  5 % of events removed keeps >= 90 % of the sample; mixed parameters,
  empty (EGI) scopes are refused; a legacy scope draws with flagged NULL.
* Labels: precision from stored reviews matches a hand computation, partial
  review uses the reviewed count, a moved end time voids a label, consensus
  substitution, ``review_precision`` rows are replaced not duplicated,
  ``top_up_region`` once per region.
* Events without figures in a scope with figures form their own ``|N``
  sub-cell and leave other events' flags unchanged; a domain cutting across
  sub-cells (``near_splice``) uses whole-sub-cell weights; an uneven partial
  review is weighted by the labelled count ``m_k``, not ``n_k``.
* ``label_agreement``: hand-computed kappa, positive / negative agreement,
  undefined kappa, clustered bootstrap only with >= 10 subjects.

Run standalone: ``python tests/test_review_sampling.py``. Exits non-zero if
any test fails.
"""

import itertools
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from turtlewave_hdEEG import dbwrite  # noqa: E402
from turtlewave_hdEEG import review_sampling as rs  # noqa: E402

CHANNELS = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T7', 'T8',
            'O1', 'O2', 'M1']
SCOPE_STAGE = 'NREM2NREM3'
FIG_COLS = ('in_band', 'low_prominence', 'amp_ratio', 'near_splice',
            'near_bound')


def make_db(path, per_channel=150, figures=True, labels=None, seed=0,
            params_b=None):
    """Synthetic one-subject database: one spindle scope over two runs.

    Run A holds Fz/F3 (a scoped re-run), run B the rest, same parameters
    apart from ``interpolated_channels``.
    """
    rng = np.random.default_rng(seed)
    chans = labels or CHANNELS
    conn = sqlite3.connect(path)
    fig_ddl = (', in_band INTEGER, low_prominence INTEGER, amp_ratio REAL, '
               'near_splice INTEGER, near_bound INTEGER') if figures else ''
    conn.execute(f'''CREATE TABLE events (uuid TEXT PRIMARY KEY,
        event_type TEXT, channel TEXT, start_time REAL, end_time REAL,
        duration REAL, stage TEXT, epoch_stage TEXT, method TEXT,
        freq_lower REAL, freq_upper REAL, run_id TEXT{fig_ddl})''')
    conn.execute('CREATE TABLE detection_runs (run_id TEXT PRIMARY KEY, '
                 'subject TEXT, params_json TEXT)')
    base = {'method': 'Moelle2011', 'frequency': [11.0, 16.0],
            'duration': [0.5, 3.0]}
    pa = dict(base, interpolated_channels=['Fz'])
    pb = dict(params_b or base, interpolated_channels=['O1', 'O2'])
    conn.execute("INSERT INTO detection_runs VALUES ('run-A', 'sub-01', ?)",
                 (json.dumps(pa),))
    conn.execute("INSERT INTO detection_runs VALUES ('run-B', 'sub-01', ?)",
                 (json.dumps(pb),))
    rows = []
    for ch in chans:
        run = 'run-A' if ch in ('Fz', 'F3') else 'run-B'
        starts = np.sort(rng.uniform(0, 20000, per_channel))
        for st in starts:
            dur = float(rng.uniform(0.5, 2.0))
            u = dbwrite.event_uuid5('spindle', ch, float(st), 'Moelle2011',
                                    11.0, 16.0, SCOPE_STAGE)
            row = [u, 'spindle', ch, float(st), float(st) + dur, dur,
                   SCOPE_STAGE, 'NREM2' if rng.random() < 0.7 else 'NREM3',
                   'Moelle2011', 11.0, 16.0, run]
            if figures:
                row += [int(rng.random() > 0.1), int(rng.random() < 0.15),
                        float(rng.lognormal(1.2, 0.4)), int(rng.random() < 0.03),
                        0]
            rows.append(row)
    conn.executemany(f"INSERT INTO events VALUES ({', '.join('?' * len(rows[0]))})",
                     rows)
    conn.commit()
    return conn


def tmpdb():
    d = tempfile.mkdtemp(prefix='tw_rs_')
    return os.path.join(d, 'neural_events.db')


# ---------------------------------------------------------------------------

def test_allocation():
    al = rs.allocate_cells({i: 10 ** 4 for i in range(10)}, 120)
    assert al == {i: 12 for i in range(10)}, al
    sizes = {0: 3, 1: 5, 2: 10 ** 4, 3: 10 ** 4, 4: 9, 5: 10 ** 4,
             6: 10 ** 4, 7: 10 ** 4, 8: 10 ** 4, 9: 13}
    al = rs.allocate_cells(sizes, 120)
    assert sum(al.values()) == 120 and al[0] == 3 and al[1] == 5 \
        and al[4] == 9 and al[9] == 13, al
    al = rs.allocate_cells({0: 50, 1: 400, 2: 400}, 121)
    assert al == {0: 40, 1: 41, 2: 40} or sum(al.values()) == 121, al
    assert rs.allocate_cells({0: 50, 1: 400, 2: 300}, 121)[1] == 41
    # adaptive flagged share
    assert rs.split_flagged(12, 210, 790) == (6, 6)
    assert rs.split_flagged(12, 850, 150) == (11, 1)
    assert rs.split_flagged(12, 620, 380) == (8, 4)
    assert rs.split_flagged(13, 500, 500) == (7, 6)
    assert rs.split_flagged(12, 3, 900) == (3, 9)
    assert rs.split_flagged(12, 900, 0) == (12, 0)
    assert rs.split_flagged(2, 999, 1) == (1, 1)
    assert rs.split_flagged(5, 2, 2) == (2, 2)  # census
    assert rs.split_cell(12, 300, 600, 100) == (6, 5, 1)
    assert rs.split_cell(12, 300, 300, 400) == (4, 3, 5)
    assert rs.split_cell(12, 0, 0, 50) == (0, 0, 12)
    assert rs.split_cell(4, 1, 1, 1) == (1, 1, 1)
    try:
        rs.split_cell(2, 50, 50, 50)
        raise AssertionError("2 slots for 3 parts accepted")
    except ValueError:
        pass
    ev = [(f'u{i}', 'frontal', 'NREM2', i % 7 == 0) for i in range(500)]
    rows = rs.select_sample(ev, seed=3, total=40)
    assert len(rows) == 40 and {r['flagged'] for r in rows} == {True, False}
    try:
        rs.select_sample(ev + [('x', 'central', 'NREM3', False)], seed=1,
                         total=3)
        raise AssertionError("T < 2H accepted")
    except ValueError:
        pass
    print("  allocation: equal, census, surplus, adaptive share, guards: OK")


def test_srswor():
    uu = [f'event-{i}' for i in range(6)]
    cnt = Counter()
    n_seeds = 15000
    for s in range(n_seeds):
        pick = sorted((rs.prn_key(s, u), u) for u in uu)[:2]
        cnt[tuple(sorted(u for _, u in pick))] += 1
    obs = np.array([cnt[c] for c in itertools.combinations(uu, 2)])
    exp = n_seeds / 15.0
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    # chi-square, 14 df: 99.9th percentile 36.1
    assert len(cnt) == 15 and chi2 < 36.1, (len(cnt), chi2)
    print(f"  hash draw N=6 n=2: 15/15 subsets, chi2 {chi2:.1f} (14 df): OK")


def test_estimator():
    est = rs.weighted_precision([3.0] * 10, ['accept'] * 7 + ['reject'] * 3)
    lo, hi = rs.wilson(0.7, 10)
    assert abs(est['p_hat'] - 0.7) < 1e-12 and abs(est['n_eff'] - 10) < 1e-9
    assert abs(est['ci_lo'] - lo) < 1e-12 and abs(est['ci_hi'] - hi) < 1e-12
    cells = [(1500, 6, 2, 3, 1), (2166, 6, 5, 1, 0), (400, 6, 3, 3, 0),
             (988, 6, 6, 0, 0)]
    w, lab = [], []
    for N, n, a, r, u in cells:
        w += [N / n] * n
        lab += ['accept'] * a + ['reject'] * r + ['unsure'] * u
    est = rs.weighted_precision(w, lab)
    print(f"  worked example: P {est['p_hat']:.4f} n_eff {est['n_eff']:.2f} "
          f"CI {est['ci_lo']:.4f}-{est['ci_hi']:.4f} U {est['unsure_rate']:.4f}")
    assert round(est['p_hat'], 3) == 0.727 and round(est['n_eff'], 1) == 18.0
    assert round(est['ci_lo'], 3) == 0.496 and round(est['ci_hi'], 3) == 0.878
    assert round(est['unsure_rate'], 3) == 0.049
    d, l, h = rs.mover_difference(0.6, 0.5, 0.7, 0.6, 0.5, 0.7)
    assert d == 0 and abs(l + h) < 1e-12 and h > 0
    nan = rs.weighted_precision([1.0], ['unsure'])
    assert math.isnan(nan['p_hat']) and nan['n_decided'] == 0
    print("  estimator: equal weights = exact Wilson, worked example, MOVER: OK")


def _coverage(prevalence, n_pop=6000, seeds=range(9000, 9400)):
    rng = np.random.default_rng(0)
    p_cells = np.array([.20, .04, .18, .04, .15, .03, .20, .05, .09, .02])
    h = rng.choice(10, n_pop, p=p_cells / p_cells.sum())
    flag = rng.random(n_pop) < prevalence
    uu = [f'{i:08d}-synthetic' for i in range(n_pop)]
    index = {u: i for i, u in enumerate(uu)}
    ev = [(uu[i], rs.REGIONS[h[i] // 2], rs.STAGES[h[i] % 2], bool(flag[i]))
          for i in range(n_pop)]
    names = np.array(['reject', 'accept', 'unsure'], dtype=object)
    cov = bias = gap0 = 0.0
    for s in seeds:
        rows = rs.select_sample(ev, seed=s, total=120, n_shared=0)
        idx = np.array([index[r['uuid']] for r in rows])
        w = np.array([r['weight'] for r in rows])
        fl = np.array([r['flagged'] for r in rows])
        r_ = np.random.default_rng(s)
        x, y = r_.random(n_pop), r_.random(n_pop)
        for scenario, (pf, pu) in (('main', (.55, .90)), ('neg', (.75, .75))):
            lab = np.where(x < .08, 2, np.where(y < np.where(flag, pf, pu), 1, 0))
            if scenario == 'main':
                tp = (lab == 1).sum() / (lab != 2).sum()
                e = rs.weighted_precision(w, names[lab[idx]])
                cov += e['ci_lo'] <= tp <= e['ci_hi']
                bias += e['p_hat'] - tp
            else:
                ef = rs.weighted_precision(w[fl], names[lab[idx][fl]])
                eu = rs.weighted_precision(w[~fl], names[lab[idx][~fl]])
                _, lo, hi = rs.mover_difference(
                    ef['p_hat'], ef['ci_lo'], ef['ci_hi'],
                    eu['p_hat'], eu['ci_lo'], eu['ci_hi'])
                gap0 += lo <= 0 <= hi
    n = len(seeds)
    return cov / n, bias / n, gap0 / n


def test_coverage_simulation():
    t0 = time.time()
    for prev in (0.21, 0.85):
        cov, bias, gap0 = _coverage(prev)
        print(f"  prevalence {prev:.2f}: coverage {cov:.3f}  bias {bias:+.4f}  "
              f"negative-control gap CI covers 0 in {gap0:.3f}")
        assert cov >= 0.94, cov
        assert abs(bias) <= 0.02, bias
        assert gap0 >= 0.93, gap0
    print(f"  coverage simulation (2 x 400 seeds) in {time.time() - t0:.1f} s: OK")


def test_draw_scope_idempotent_and_redraw():
    path = tmpdb()
    conn = make_db(path)
    sid = rs.draw_review_sample(conn, run_id='run-A', seed=1)
    sid_b = rs.draw_review_sample(conn, run_id='run-B', seed=1)
    assert sid == sid_b, "two runs of one scope are one sample"
    n_rows = conn.execute("SELECT COUNT(*) FROM review_samples").fetchone()[0]
    assert n_rows == 120, n_rows
    runs = {r[0] for r in conn.execute(
        "SELECT DISTINCT run_id FROM review_samples")}
    assert runs == {'run-A', 'run-B'}, runs
    des = rs._design_row(conn, sid)
    assert json.loads(des['run_ids']) == ['run-A', 'run-B']
    assert json.loads(des['n_out_of_scope']) == {'region:other': 150}
    assert des['flag_available'] == 1
    shared = conn.execute("SELECT region, stage, COUNT(*) FROM review_samples "
                          "WHERE is_shared = 1 GROUP BY region, stage").fetchall()
    assert sum(r[2] for r in shared) == 30 and all(r[2] == 3 for r in shared)
    sid2 = rs.draw_review_sample(conn, run_id='run-A', seed=2)
    assert sid2 != sid
    assert conn.execute("SELECT COUNT(*) FROM review_samples").fetchone()[0] == 240
    # redraw after a re-detection that removed 5 % of events
    drop = [r[0] for r in conn.execute("SELECT uuid FROM events")
            if rs.prn_key(7, r[0], 'drop') < 0.05 * 2 ** 64]
    conn.executemany("DELETE FROM events WHERE uuid = ?", [(u,) for u in drop])
    conn.commit()
    sid3 = rs.draw_review_sample(conn, run_id='run-B', seed=1)
    assert sid3 != sid, "changed population must give a new id"
    old = {r[0] for r in conn.execute(
        "SELECT uuid FROM review_samples WHERE sample_id = ?", (sid,))}
    new = {r[0] for r in conn.execute(
        "SELECT uuid FROM review_samples WHERE sample_id = ?", (sid3,))}
    survivors = old - set(drop)
    kept = len(survivors & new)
    print(f"  redraw after removing {len(drop)} events: {kept}/{len(survivors)} "
          f"surviving sampled events kept ({len(old & set(drop))} removed)")
    assert kept >= 0.9 * len(old), kept
    conn.close()
    print("  draw: scope across runs, idempotent, new id on change: OK")


def test_refusals_and_legacy():
    path = tmpdb()
    conn = make_db(path, params_b={'method': 'Moelle2011',
                                   'frequency': [11.0, 16.0],
                                   'duration': [0.3, 3.0]})
    try:
        rs.draw_review_sample(conn, run_id='run-A')
        raise AssertionError("mixed params accepted")
    except ValueError as err:
        assert 'different parameters' in str(err)
    sid = rs.draw_review_sample(conn, run_id='run-A', allow_mixed_params=True)
    assert json.loads(rs._design_row(conn, sid)['design_json'])[
        'allow_mixed_params'] is True
    conn.close()

    egi = [f'E{i}' for i in range(1, 30)]
    conn = make_db(tmpdb(), labels=egi, per_channel=20)
    try:
        rs.draw_review_sample(conn, run_id='run-B')
        raise AssertionError("EGI population drew")
    except ValueError as err:
        assert "EGI" in str(err)
    conn.close()

    conn = make_db(tmpdb(), figures=False)
    sid = rs.draw_review_sample(conn, run_id='run-B')
    flags = {r[0] for r in conn.execute(
        "SELECT DISTINCT flagged FROM review_samples")}
    assert flags == {None}, flags
    assert rs._design_row(conn, sid)['flag_available'] == 0
    cells = {r[0] for r in conn.execute("SELECT DISTINCT cell FROM review_samples")}
    assert all(c.endswith('|A') for c in cells), cells
    conn.close()
    print("  refusals (mixed params, EGI) and legacy scope (flagged NULL): OK")


def _label_all(conn, sid, reviewer, rule, uuids=None):
    rows = rs._sample_rows(conn, sid)
    for r in rows:
        if uuids is not None and r['uuid'] not in uuids:
            continue
        dec = rule(r)
        dbwrite.store_event_review(
            conn, r['uuid'], dec, reviewer,
            reason='artefact' if dec == 'reject' else None, commit=False)
    conn.commit()
    return rows


def test_precision_from_reviews():
    conn = make_db(tmpdb())
    sid = rs.draw_review_sample(conn, run_id='run-B', seed=5)

    def rule(r):
        k = int(r['sort_key'], 16)
        if k % 13 == 0:
            return 'unsure'
        return 'reject' if (r['flagged'] and k % 2) else 'accept'
    rows = _label_all(conn, sid, 'alice', rule)
    df = rs.compute_review_precision(conn, sid)
    scope = df[df.domain_type == 'scope'].iloc[0]
    w = [r['pop_k'] / r['n_k'] for r in rows]
    hand = rs.weighted_precision(w, [rule(r) for r in rows])
    assert abs(scope.p_hat - hand['p_hat']) < 1e-12, (scope.p_hat, hand)
    assert scope.reviewer == 'alice' and scope.n_reviewed == 120
    assert scope.verdict in ('TRUSTWORTHY', 'NOT_TRUSTWORTHY')
    types = Counter(df.domain_type)
    assert types['region'] == 5 and types['region_stage'] == 10 \
        and types['difference'] == 1, types
    gap = df[df.domain_type == 'difference'].iloc[0]
    assert gap.p_hat < 0 and gap.ci_lo <= gap.p_hat <= gap.ci_hi
    print(f"  precision: scope P {scope.p_hat:.3f} [{scope.ci_lo:.3f}, "
          f"{scope.ci_hi:.3f}] n_eff {scope.n_eff:.1f} -> {scope.verdict}; "
          f"flagged-unflagged {gap.p_hat:+.3f} [{gap.ci_lo:+.3f}, {gap.ci_hi:+.3f}]")
    n1 = conn.execute("SELECT COUNT(*) FROM review_precision").fetchone()[0]
    rs.compute_review_precision(conn, sid)
    n2 = conn.execute("SELECT COUNT(*) FROM review_precision").fetchone()[0]
    assert n1 == n2 == len(df), (n1, n2)

    # consensus substitution on the shared subset
    shared = [r for r in rows if r['is_shared']]
    for r in shared:
        dbwrite.store_event_review(conn, r['uuid'], 'accept', 'consensus',
                                   commit=False)
    conn.commit()
    dc = rs.compute_review_precision(conn, sid, reviewer='alice',
                                     label_source='consensus')
    c_scope = dc[dc.domain_type == 'scope'].iloc[0]
    assert c_scope.label_source == 'consensus' and c_scope.p_hat >= scope.p_hat
    n3 = conn.execute("SELECT COUNT(*) FROM review_precision").fetchone()[0]
    assert n3 == 2 * len(df), n3

    # end time moved by a re-detection: the label is void
    victim = rows[0]['uuid']
    conn.execute("UPDATE events SET end_time = end_time + 0.2 WHERE uuid = ?",
                 (victim,))
    conn.commit()
    prog = rs.sample_progress(conn, sid, reviewer='alice')
    assert prog['stale'] == [victim] and prog['n_reviewed'] == 119, prog['stale']
    df2 = rs.compute_review_precision(conn, sid, reviewer='alice', write=False)
    assert df2[df2.domain_type == 'scope'].iloc[0].n_reviewed == 119
    conn.close()
    print("  precision from reviews: hand match, replace-not-duplicate, "
          "consensus, stale end time: OK")


def test_partial_review_and_order():
    conn = make_db(tmpdb())
    sid = rs.draw_review_sample(conn, run_id='run-B', seed=9)
    prog = rs.sample_progress(conn, sid, reviewer='bob')
    order = prog['next_uuids']
    assert len(order) == 120 and len(set(order)) == 120
    first = order[:40]
    rows = _label_all(conn, sid, 'bob', lambda r: 'accept'
                      if int(r['sort_key'], 16) % 3 else 'reject',
                      uuids=set(first))
    prog = rs.sample_progress(conn, sid, reviewer='bob')
    assert prog['n_reviewed'] == 40 and prog['next_uuids'] == order[40:]
    df = rs.compute_review_precision(conn, sid, reviewer='bob', write=False)
    scope = df[df.domain_type == 'scope'].iloc[0]
    lab = {r['uuid']: r for r in rows if r['uuid'] in set(first)}
    m = Counter(r['cell'] for r in lab.values())
    w = [r['pop_k'] / m[r['cell']] for r in lab.values()]
    hand = rs.weighted_precision(w, ['accept' if int(r['sort_key'], 16) % 3
                                     else 'reject' for r in lab.values()])
    assert abs(scope.p_hat - hand['p_hat']) < 1e-12
    conn.close()
    print(f"  partial review (40/120 in presentation order): weights use "
          f"reviewed count, P {scope.p_hat:.3f}, verdict {scope.verdict}: OK")


def test_top_up():
    conn = make_db(tmpdb())
    sid = rs.draw_review_sample(conn, run_id='run-B', seed=4)
    before = conn.execute("SELECT cell, n_k, weight FROM review_samples WHERE "
                          "sample_id = ? AND region = 'temporal'",
                          (sid,)).fetchall()
    added = rs.top_up_region(conn, sid, 'temporal')
    assert added == 12, added
    after = conn.execute("SELECT cell, COUNT(*), MIN(n_k), MAX(n_k), "
                         "MIN(pop_k) FROM review_samples WHERE sample_id = ? "
                         "AND region = 'temporal' GROUP BY cell",
                         (sid,)).fetchall()
    assert sum(r[1] for r in after) == len(before) + 12
    assert all(r[1] == r[2] == r[3] for r in after), after
    assert conn.execute("SELECT COUNT(*) FROM review_samples WHERE "
                        "sample_id = ? AND draw_round = 1", (sid,)
                        ).fetchone()[0] == 12
    try:
        rs.top_up_region(conn, sid, 'temporal')
        raise AssertionError("second top-up accepted")
    except ValueError:
        pass
    conn.close()
    print("  top-up: 12 added, n_k/weights updated, once only: OK")


def test_label_agreement():
    a = ['accept'] * 6 + ['reject'] * 2 + ['accept', 'reject', 'unsure']
    b = ['accept'] * 5 + ['reject'] + ['reject'] * 2 + ['reject', 'accept',
                                                        'accept']
    out = rs.label_agreement(a, b)
    # both decided: 10 pairs; aa 5, ar 2, ra 1, rr 2
    po = 7 / 10
    pa, pb = 7 / 10, 6 / 10
    pe = pa * pb + (1 - pa) * (1 - pb)
    assert abs(out['kappa'] - (po - pe) / (1 - pe)) < 1e-12, out['kappa']
    assert abs(out['percent_agreement'] - 7 / 11) < 1e-12
    assert abs(out['positive_agreement'] - 10 / 13) < 1e-12
    assert abs(out['negative_agreement'] - 4 / 7) < 1e-12
    assert out['confusion']['accept']['reject'] == 2
    assert out['kappa_ci'] is None and 'no subjects' in out['kappa_ci_note']
    allacc = rs.label_agreement(['accept'] * 5, ['accept'] * 5)
    assert math.isnan(allacc['kappa']) and allacc['percent_agreement'] == 1.0
    rng = np.random.default_rng(1)
    n = 300
    subj = [f's{i % 12}' for i in range(n)]
    x = np.where(rng.random(n) < 0.8, 'accept', 'reject')
    y = np.where(rng.random(n) < 0.9, x, np.where(x == 'accept', 'reject',
                                                   'accept'))
    out = rs.label_agreement(dict(zip(range(n), x)), dict(zip(range(n), y)),
                             subjects=dict(zip(range(n), subj)),
                             flags=dict(zip(range(n), [i % 2 == 0
                                                        for i in range(n)])),
                             n_boot=500)
    assert out['kappa_ci'] is not None and out['kappa_ci'][0] < out['kappa'] \
        < out['kappa_ci'][1], out['kappa_ci']
    assert set(out['by_flag']) == {False, True}
    few = rs.label_agreement(x[:20], y[:20],
                             subjects=[f's{i % 8}' for i in range(20)])
    assert few['kappa_ci'] is None and '< 10' in few['kappa_ci_note']
    print(f"  label agreement: hand kappa, PA/NA, undefined kappa, bootstrap "
          f"kappa {out['kappa']:.3f} {tuple(round(v, 3) for v in out['kappa_ci'])}: OK")


def test_qt_free():
    src = open(rs.__file__).read()
    assert 'PyQt' not in src and 'pyqtgraph' not in src
    print("  module imports no Qt: OK")


def test_no_figure_events():
    """Events with near_splice NULL in a scope with figures: own sub-cell."""
    conn = make_db(tmpdb())
    base_pop = rs._population(conn, rs._resolve_scope(conn, 'run-B', None,
                                                      None))
    base_flag = {e['uuid']: e['flagged'] for e in base_pop['events']}
    nulled = ('C3', 'P3')
    conn.execute(f"UPDATE events SET {', '.join(c + ' = NULL' for c in FIG_COLS)}"
                 f" WHERE channel IN (?, ?)", nulled)
    conn.commit()
    pop = rs._population(conn, rs._resolve_scope(conn, 'run-B', None, None))
    assert pop['flag_available']
    for e in pop['events']:
        if e['channel'] in nulled:
            assert e['flagged'] is None and e['flag_components'] is None, e
        else:
            assert e['flagged'] == base_flag[e['uuid']], e
    expect = Counter(f"{e['region']}|{e['stage']}" for e in pop['events']
                     if e['channel'] in nulled)
    assert pop['no_figures'] == dict(sorted(expect.items())), pop['no_figures']
    sid = rs.draw_review_sample(conn, run_id='run-B', seed=3)
    des = rs._design_row(conn, sid)
    assert json.loads(des['n_no_figures']) == pop['no_figures']
    rows = rs._sample_rows(conn, sid)
    assert len(rows) == 120
    n_cells = [r for r in rows if r['cell'].endswith('|N')]
    assert n_cells and all(r['flagged'] is None and r['channel'] in nulled
                           for r in n_cells), n_cells
    assert all(r['flagged'] is not None for r in rows
               if not r['cell'].endswith('|N'))
    for cell in {r['cell'][:-2] for r in n_cells}:
        nk = {h: sum(1 for r in rows if r['cell'] == f'{cell}|{h}')
              for h in 'FUN'}
        Nk = {h: next((r['pop_k'] for r in rows if r['cell'] == f'{cell}|{h}'),
                      0) for h in 'FUN'}
        assert (nk['F'], nk['U'], nk['N']) == rs.split_cell(
            sum(nk.values()), Nk['F'], Nk['U'], Nk['N']), (cell, nk, Nk)
    # labels -> a no_figures domain and unchanged flagged/unflagged domains
    _label_all(conn, sid, 'alice', lambda r: 'accept')
    df = rs.compute_review_precision(conn, sid, write=False)
    assert 'no_figures' in set(df.domain)
    nf_row = df[df.domain == 'no_figures'].iloc[0]
    assert nf_row.n_sampled == len(n_cells)
    added = rs.top_up_region(conn, sid, 'central')
    assert added == 12, added
    conn.close()
    print(f"  no-figure events: {sum(expect.values())} on {nulled} sampled as "
          f"|N ({len(n_cells)} drawn), other flags unchanged, top-up OK: OK")


def test_domain_uses_subcell_weights():
    """A domain cutting across sub-cells uses the whole sub-cell's m_k."""
    k1 = {'cell': 'frontal|NREM2|F', 'pop_k': 1000}
    k2 = {'cell': 'frontal|NREM2|U', 'pop_k': 100}
    rows_lab = ([(k1, ('reject', 'artefact', None))]
                + [(k2, ('accept', None, None))] * 3)
    m_k = {'frontal|NREM2|F': 10, 'frontal|NREM2|U': 10}
    est = rs._domain_row(rows_lab, {k1['cell'], k2['cell']}, m_k)
    right = (3 * 10.0) / (100.0 + 3 * 10.0)          # w = 100 and 10
    wrong = (3 * 100 / 3.0) / (1000.0 + 3 * 100 / 3.0)  # m_k from the domain
    print(f"  cross-cell domain: P {est['p_hat']:.4f} (sub-cell weights "
          f"{right:.4f}; domain-local weights would give {wrong:.4f})")
    assert abs(est['p_hat'] - right) < 1e-12 and abs(right - wrong) > 0.1


def test_uneven_partial_review():
    """Uneven labelling across sub-cells: weights must use m_k, not n_k."""
    conn = make_db(tmpdb())
    sid = rs.draw_review_sample(conn, run_id='run-B', seed=11)
    rows = rs._sample_rows(conn, sid)          # sorted by cell, sort_key
    cells = sorted({r['cell'] for r in rows})
    take = {c: 1 + (i * 3) % 6 for i, c in enumerate(cells)}
    chosen, seen = [], Counter()
    for r in rows:
        if seen[r['cell']] < take[r['cell']]:
            chosen.append(r)
            seen[r['cell']] += 1

    def rule(r):
        return 'reject' if (r['flagged'] and int(r['sort_key'], 16) % 2) \
            else 'accept'
    _label_all(conn, sid, 'carol', rule, uuids={r['uuid'] for r in chosen})
    df = rs.compute_review_precision(conn, sid, reviewer='carol', write=False)
    p = df[df.domain_type == 'scope'].iloc[0].p_hat
    m = Counter(r['cell'] for r in chosen)
    by_m = rs.weighted_precision([r['pop_k'] / m[r['cell']] for r in chosen],
                                 [rule(r) for r in chosen])['p_hat']
    by_n = rs.weighted_precision([r['pop_k'] / r['n_k'] for r in chosen],
                                 [rule(r) for r in chosen])['p_hat']
    print(f"  uneven partial review ({len(chosen)} of 120): P {p:.4f} = m_k "
          f"weights {by_m:.4f}; n_k weights would give {by_n:.4f}")
    assert abs(p - by_m) < 1e-12 and abs(by_m - by_n) > 1e-3
    conn.close()


TESTS = [
    test_allocation,
    test_srswor,
    test_estimator,
    test_coverage_simulation,
    test_draw_scope_idempotent_and_redraw,
    test_refusals_and_legacy,
    test_precision_from_reviews,
    test_partial_review_and_order,
    test_top_up,
    test_no_figure_events,
    test_domain_uses_subcell_weights,
    test_uneven_partial_review,
    test_label_agreement,
    test_qt_free,
]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    print("TESTING review sampling")
    print("=======================")
    failed = []
    for test in TESTS:
        try:
            test()
        except Exception:
            failed.append(test.__name__)
            traceback.print_exc()
    print()
    if failed:
        print(f"FAILED {len(failed)} of {len(TESTS)}: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(TESTS)} tests passed.")
