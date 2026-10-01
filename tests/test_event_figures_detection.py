#!/usr/bin/env python3
"""Per-event review figures computed at detection time (4.6).

* A Moelle2011 run on ``tests/synthetic_sleep_eeg.set`` writes the 14 figure
  columns and records its settings in ``params_json['event_figures']``; with
  ``compute_figures=False`` it detects the identical events (counts,
  ``start_time`` lists and every other column byte-identical) and leaves the
  figures NULL.
* Detection-time figures equal :func:`event_metrics.event_figures` on a
  continuous read of the same samples to 1e-9; on a whole-record read they
  differ only through the per-run filter edges (unfiltered M2 figures stay
  identical, background windows can only be fewer at detection time).
* On a spliced synthetic segment, events farther than the filter settling
  time from a splice agree to 1e-9 with the continuous read, events within P
  of a splice are ``near_splice`` with NULL signal figures, and nothing else
  differs except the background window count near the splice.
* ``event_population_summary`` returns one row per channel x stage with the
  documented columns and denominators.
* Slow-wave and K-complex runs leave the spindle-only columns NULL, fill
  ``wave_freq`` from the negative half-wave, and store the Massimini
  ``thresh_ratio`` as ``min(det_trough / max_trough_amp, det_ptp / min_ptp)``,
  positive and >= 1, under both polarities.

Run standalone: ``python tests/test_event_figures_detection.py``. Exits
non-zero if any test fails.
"""

import json
import logging
import os
import sqlite3
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from turtlewave_hdEEG import ParalEvents, ParalKC, ParalSWA, dbwrite  # noqa: E402
from turtlewave_hdEEG import event_metrics as em  # noqa: E402
from test_detection_thresholds import Workdir, _set_recording  # noqa: E402

FIG_COLS = [c for c, _ in dbwrite._EVENT_FIGURE_COLUMNS]
SPINDLE_ONLY = ('halfwaves_above_bg', 'cycles_nominal', 'peak_freq_ap',
                'prominence_db', 'low_prominence')
STAGES = ['NREM2', 'NREM3']
REJECT = ('Artefact', 'Arousal', 'Move')
CHANS = ['C3', 'C4', 'O1']
BAND = (11.0, 16.0)


def _rows(db, sql, params=()):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _moelle(dataset, annot, db, tmp, **kw):
    return ParalEvents(dataset, annot, log_level=logging.CRITICAL).detect_spindles(
        method='Moelle2011', chan=CHANS, frequency=BAND, stage=STAGES,
        json_dir=tmp, db_path=db, subject='sub-T', **kw)


def _close(a, b, rtol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def test_spindle_columns_and_golden():
    """Figures written; compute_figures=False detects byte-identical events."""
    print("\n1. Moelle2011 on synthetic_sleep_eeg.set, figures on vs off:")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db_on = os.path.join(tmp, 'on.db')
        db_off = os.path.join(tmp, 'off.db')
        _moelle(dataset, annot, db_on, tmp)
        _moelle(dataset, annot, db_off, tmp, compute_figures=False)
        skip = {'processing_timestamp', 'run_id'}
        sql = "SELECT * FROM events ORDER BY channel, start_time, method"
        on, off = _rows(db_on, sql), _rows(db_off, sql)
        assert on and len(on) == len(off), (len(on), len(off))
        for col in on[0]:
            if col in skip or col in FIG_COLS:
                continue
            a = [repr(r[col]) for r in on]
            b = [repr(r[col]) for r in off]
            assert a == b, f"column {col} differs with figures on"
        assert all(r[c] is None for r in off for c in FIG_COLS), off[0]
        starts = [r['start_time'] for r in on]

        for r in on:
            assert r['near_splice'] in (0, 1), r
            if r['near_splice']:
                continue
            for c in ('halfwaves_above_bg', 'cycles_nominal', 'bg_rms',
                      'bg_n_windows', 'bg_stage_mixed', 'amp_ratio',
                      'thresh_ratio', 'near_bound', 'in_band'):
                assert r[c] is not None, (c, r)
            assert r['wave_freq'] is None, r
            assert r['thresh_ratio'] >= 1.0, r  # every Moelle event crossed
            assert abs(r['thresh_ratio'] - r['peak_val_det'] / _thr(
                db_on, r)['det_value_lo']) < 1e-12, r
        params = json.loads(_rows(db_on, "SELECT params_json FROM "
                                         "detection_runs")[0]['params_json'])
        cfg = params['event_figures']
        assert cfg['spec_revision'] == 'v2' and cfg['edge_guard_s'] == 2.0, cfg
        params_off = json.loads(_rows(db_off, "SELECT params_json FROM "
                                              "detection_runs")[0]['params_json'])
        assert params_off['event_figures'] is None, params_off
        print(f"[ok] {len(on)} events, starts {starts}: identical with "
              f"figures off; all 14 columns filled; thresh_ratio = "
              f"peak_val_det / det_value_lo >= 1; params_json records v2")


def _thr(db, row):
    df = dbwrite.read_detection_thresholds(db, row['run_id'], row['channel'],
                                           row['method'],
                                           at_time=row['start_time'])
    return dict(zip(df['name'], df['value']))


def _detection_runs(dataset, annot, chan):
    """Contiguous runs (t0, t1) of the segment the detector was handed."""
    from wonambi.trans import fetch
    segs = fetch(dataset, annot, cat=(1, 1, 1, 0), stage=STAGES, cycle=None,
                 reject_epoch=True, reject_artf=list(REJECT))
    segs.read_data(chan, [])
    t = np.asarray(segs[0]['data'].axis['time'][0])
    fs = float(segs[0]['data'].s_freq)
    cut = np.flatnonzero(np.diff(t) > 1.5 / fs) + 1
    edges = np.r_[0, cut, t.size]
    return [(t[a], t[b - 1]) for a, b in zip(edges[:-1], edges[1:])], fs


def _read(dataset, chan, t0, t1):
    data = dataset.read_data(chan=[chan], begtime=t0, endtime=t1)
    return np.asarray(data.data[0][0], dtype='f8'), \
        np.asarray(data.axis['time'][0]), float(data.s_freq)


def _continuous(dataset, annot, db, row, spans, all_rows):
    """event_figures on fresh reads of ``spans`` (concatenated with gaps)."""
    parts = [_read(dataset, row['channel'], a, b) for a, b in spans]
    x = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    fs = parts[0][2]
    params = json.loads(_rows(db, "SELECT params_json FROM detection_runs "
                                  "WHERE run_id = ?",
                              (row['run_id'],))[0]['params_json'])
    others = [(r['start_time'], r['end_time']) for r in all_rows
              if r['channel'] == row['channel'] and r['run_id'] == row['run_id']]
    reject = [(e['start'], e['end']) for e in annot.get_events()
              if e['name'] in REJECT]
    epochs = [(e['start'], e['end'], e['stage']) for e in annot.get_epochs()]
    return em.event_figures(
        x, t, fs, row['start_time'], row['end_time'], BAND, others=others,
        stage_epochs=epochs, run_stages=STAGES, thresholds=_thr(db, row),
        method=row['method'], event_values=row,
        duration_bounds=params['duration_by_method'][row['method']],
        reject_intervals=reject).to_row()


def test_detection_vs_continuous_read():
    """Same samples -> 1e-9; whole-record read -> only filter-edge effects."""
    print("\n2. Detection-time figures vs a continuous read:")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'n.db')
        _moelle(dataset, annot, db, tmp)
        rows = _rows(db, "SELECT * FROM events ORDER BY channel, start_time")
        runs = {ch: _detection_runs(dataset, annot, ch) for ch in CHANS}
        worst = 0.0
        for row in rows:
            spans, fs = runs[row['channel']]
            # The same samples the detector saw: every whole run reaching
            # into the event's surround, read afresh and concatenated.
            reach = em.GEOMETRY['spindle']['S'] + em.GEOMETRY['spindle']['P']
            same_spans = [(a, b + 1 / fs) for a, b in spans
                          if b >= row['start_time'] - reach
                          and a <= row['end_time'] + reach]
            same = _continuous(dataset, annot, db, row, same_spans, rows)
            for c in FIG_COLS:
                a, b = row[c], same[c]
                if isinstance(a, float) or isinstance(b, float):
                    assert _close(a, b), (c, a, b, row['start_time'])
                else:
                    assert a == b, (c, a, b, row['start_time'])

            whole = _continuous(dataset, annot, db, row, [(0.0, 600.0)], rows)
            for c in ('peak_freq_ap', 'prominence_db', 'in_band',
                      'low_prominence', 'thresh_ratio', 'near_bound',
                      'cycles_nominal'):
                if row['near_splice']:
                    break
                a, b = row[c], whole[c]
                assert (_close(a, b) if isinstance(a, float) else a == b), \
                    (c, a, b)
            if not row['near_splice']:
                assert row['bg_n_windows'] <= whole['bg_n_windows'], (row, whole)
                if row['bg_n_windows'] == whole['bg_n_windows']:
                    worst = max(worst, abs(row['amp_ratio'] - whole['amp_ratio'])
                                / whole['amp_ratio'])
        print(f"[ok] {len(rows)} events: all 14 figures equal (1e-9) to "
              f"event_figures on the same samples; on a whole-record read the "
              f"unfiltered figures are identical, background windows are never "
              f"more at detection time, and amp_ratio differs by at most "
              f"{worst:.1e} (relative) where the window sets match")


class _Seg(dict):
    """A minimal stand-in for a fetched Wonambi segment."""


def _segment(x, t, fs):
    from wonambi.datatype import ChanTime
    d = ChanTime()
    d.s_freq = fs
    for k in ('chan', 'time'):
        d.axis[k] = np.empty(1, dtype='O')
    d.data = np.empty(1, dtype='O')
    d.axis['chan'][0] = np.array(['Cz'])
    d.axis['time'][0] = t
    d.data[0] = x[None, :]
    return _Seg(data=d)


def test_splice_differences_are_local():
    """Spliced segment vs continuous read: differ only near the splice."""
    print("\n3. Spliced segment vs continuous read (250 Hz, 12-15 Hz):")
    fs, band = 250.0, (12.0, 15.0)
    rng = np.random.default_rng(9000)
    n = int(240 * fs)
    t = np.arange(n) / fs
    x = rng.standard_normal(n) * 10.0
    for c in np.arange(10.0, 230.0, 7.3):  # 13 Hz bursts every 7.3 s
        m = (t >= c) & (t < c + 1)
        x[m] += 25 * np.hanning(m.sum()) * np.sin(2 * np.pi * 13 * (t[m] - c))
    gap = (120.0, 150.0)  # scored Wake: not in the run, absent from the segment
    keep = (t < gap[0]) | (t >= gap[1])
    epochs = [(0.0, gap[0], 'NREM2'), (gap[0], gap[1], 'Wake'),
              (gap[1], 240.0, 'NREM2')]
    starts = np.arange(10.0, 230.0, 7.3)
    events = [{'start_time': float(s), 'end_time': float(s + 1.0),
               'method': 'Moelle2011', '_seg_idx': 0,
               '_duration_bounds': (0.5, 3.0)}
              for s in sorted(np.r_[starts, 118.4, 150.7])
              if not (gap[0] - 1 < s < gap[1])]
    seg = _segment(x[keep], t[keep], fs)
    det = em.channel_event_figures([seg], events, band, 'spindle', epochs,
                                   ['NREM2'])
    others = [(e['start_time'], e['end_time']) for e in events]
    P, S = em.GEOMETRY['spindle']['P'], em.GEOMETRY['spindle']['S']
    settle = 3.5  # 1e-9 settling of butter(2, 12-15 Hz) at 250 Hz
    n_far = n_near = n_mid = 0
    for ev, d in zip(events, det):
        c = em.event_figures(x, t, fs, ev['start_time'], ev['end_time'], band,
                             others=others, stage_epochs=epochs,
                             run_stages=['NREM2'], method='Moelle2011',
                             duration_bounds=(0.5, 3.0)).to_row()
        dist = min(abs(ev['start_time'] - gap[1]), abs(gap[0] - ev['end_time']),
                   abs(ev['start_time'] - gap[0]), abs(ev['end_time'] - gap[1]))
        if dist < P:
            n_near += 1
            assert d['near_splice'] == 1 and d['amp_ratio'] is None, (ev, d)
            assert c['near_splice'] == 0 and c['amp_ratio'] is not None, c
        elif dist > S + P + settle:
            n_far += 1
            for col in FIG_COLS:
                a, b = d[col], c[col]
                assert (_close(a, b) if isinstance(a, float) else a == b), \
                    (col, a, b, ev['start_time'])
        else:
            n_mid += 1
            assert d['near_splice'] == 0, d
            assert d['bg_n_windows'] <= c['bg_n_windows'], (d, c)
            # On the detection segment out-of-stage time is absent, so the
            # stage rule never removes a usable window there; the continuous
            # read holds that time and drops it. Detection may say "not
            # mixed" where the continuous read says "mixed", never the reverse.
            assert d['bg_stage_mixed'] <= c['bg_stage_mixed'], (d, c)
            for col in ('peak_freq_ap', 'prominence_db', 'cycles_nominal',
                        'near_bound', 'in_band', 'low_prominence'):
                a, b = d[col], c[col]
                assert (_close(a, b, 1e-6) if isinstance(a, float) else a == b), \
                    (col, a, b, ev['start_time'])
    assert n_far and n_near and n_mid, (n_far, n_near, n_mid)
    print(f"[ok] {n_far} far events identical to 1e-9; {n_near} within P of "
          f"the splice are near_splice with NULL signal figures; {n_mid} in "
          f"between differ only in the background (never more windows at "
          f"detection time)")


def test_population_summary():
    """One row per channel x stage, documented columns and denominators."""
    print("\n4. event_population_summary:")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'n.db')
        _moelle(dataset, annot, db, tmp)
        run_id = _rows(db, "SELECT run_id FROM events LIMIT 1")[0]['run_id']
        df = dbwrite.event_population_summary(db, run_id, 'spindle')
        assert list(df.columns) == list(dbwrite.POPULATION_SUMMARY_COLUMNS)
        n_events = _rows(db, "SELECT COUNT(*) AS n FROM events")[0]['n']
        pairs = _rows(db, "SELECT COUNT(DISTINCT channel || '|' || "
                          "COALESCE(epoch_stage, 'unscored')) AS n FROM events")
        assert len(df) == pairs[0]['n'] and df['n'].sum() == n_events, df
        for share, denom in (('share_off_band', 'n_freq'),
                             ('share_low_prom', 'n_prom'),
                             ('share_at_floor', 'n_bound')):
            ok = df[denom] > 0
            assert ((df.loc[ok, share] >= 0) & (df.loc[ok, share] <= 1)).all()
            assert (df[denom] <= df['n']).all()
        assert (df['amp_ratio_q25'] <= df['amp_ratio_median']).all()
        assert (df['amp_ratio_median'] <= df['amp_ratio_q75']).all()
        empty = dbwrite.event_population_summary(db, 'no-such-run', 'spindle')
        assert empty.empty and list(empty.columns) == list(df.columns)
        print(f"[ok] {len(df)} channel x stage rows, n sums to {n_events}; "
              f"shares within [0, 1] over their n_* denominators; an unknown "
              f"run gives an empty frame with the same columns")
        print(df[['channel', 'stage', 'n', 'share_off_band', 'share_low_prom',
                  'share_at_floor', 'amp_ratio_median',
                  'thresh_ratio_median']].to_string(index=False))


def test_slow_wave_and_kcomplex_figures():
    """SW/KC: spindle-only NULL, wave_freq from half-wave, Massimini ratio."""
    print("\n5. Slow-wave and K-complex figures:")
    from test_turtlewave import _synthetic_recording
    with Workdir() as tmp:
        dataset, annot = _synthetic_recording(tmp, ('NREM3',) * 6)
        db = os.path.join(tmp, 'neural_events.db')
        kw = dict(chan=['Cz'], stage=['NREM3'], json_dir=tmp, db_path=db,
                  subject='sub-T')
        for polar in ('normal', 'opposite'):
            ParalSWA(dataset, annot, log_level=logging.CRITICAL).detect_slow_waves(
                method='Massimini2004', frequency=(0.1, 4), polar=polar,
                replace_channels=['Cz'], **kw)
            rows = _rows(db, "SELECT * FROM events WHERE event_type = "
                             "'slow_wave'")
            assert rows, polar
            thr = {r['name']: r['value'] for r in _rows(
                db, "SELECT name, value FROM detection_thresholds WHERE "
                    "run_id = ?", (rows[0]['run_id'],))}
            for r in rows:
                assert all(r[c] is None for c in SPINDLE_ONLY), r
                half = r['end_time'] - r['det_zero_time']
                assert _close(r['wave_freq'], 1 / (2 * half)), r
                assert r['in_band'] == int(0.1 <= r['wave_freq'] <= 4), r
                if r['thresh_ratio'] is not None:
                    want = min(r['det_trough'] / thr['max_trough_amp'],
                               r['det_ptp'] / thr['min_ptp'])
                    assert _close(r['thresh_ratio'], want), (r, want)
                    assert r['thresh_ratio'] >= 1.0, r
                assert r['near_bound'] in (-1, 0, 1, None), r
                # A NULL amp_ratio away from a splice is an insufficient
                # background (a dense slow-oscillation train leaves few
                # event-free 2 s windows), never silently missing.
                if r['amp_ratio'] is None and not r['near_splice']:
                    assert r['bg_n_windows'] is not None, r
                    assert r['bg_n_windows'] < 8, r
            n_ratio = sum(r['thresh_ratio'] is not None for r in rows)
            n_bg = sum(r['amp_ratio'] is not None for r in rows)
            print(f"[ok] Massimini2004 polar={polar}: {len(rows)} waves, "
                  f"spindle-only columns NULL, wave_freq = 1/(2 x negative "
                  f"half-wave), thresh_ratio = min(trough, ptp) >= 1 on "
                  f"{n_ratio}, amp_ratio on {n_bg} (the rest: under 8 "
                  f"event-free background windows in this dense train)")

        ParalKC(dataset, annot, log_level=logging.CRITICAL).detect_kcomplexes(
            method='AASM/Massimini2004', frequency=(0.5, 4), **kw)
        kc = _rows(db, "SELECT * FROM events WHERE event_type = 'k_complex'")
        assert kc and all(r[c] is None for r in kc for c in SPINDLE_ONLY)
        assert all(r['thresh_ratio'] is None or r['thresh_ratio'] >= 1
                   for r in kc), kc
        assert all(r['wave_freq'] is not None for r in kc), kc
        cfg = json.loads(_rows(db, "SELECT params_json FROM detection_runs "
                                   "WHERE event_type = 'k_complex'")[0]
                         ['params_json'])['event_figures']
        assert cfg['edge_guard_s'] == 10.0 and cfg['window_s'] == 2.0, cfg
        print(f"[ok] K-complex: {len(kc)} events, spindle-only NULL, "
              f"thresh_ratio >= 1, slow-wave geometry recorded")


TESTS = [test_spindle_columns_and_golden, test_detection_vs_continuous_read,
         test_splice_differences_are_local, test_population_summary,
         test_slow_wave_and_kcomplex_figures]


if __name__ == '__main__':
    t_start = time.time()
    failed = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failed += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed in "
          f"{time.time() - t_start:.1f} s")
    sys.exit(1 if failed else 0)
