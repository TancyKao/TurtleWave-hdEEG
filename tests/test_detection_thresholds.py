#!/usr/bin/env python3
"""Detector thresholds and per-event detector values stored at detection time (4.6).

Covers the ``detection_thresholds`` table, the additive ``events`` columns
``peak_freq, peak_val_det, rms_det, rms_orig, power_orig, det_zero_time``, and
the three call sites (spindles, slow waves, K-complexes):

* Moelle2011 and Lacourse2018 on ``tests/synthetic_sleep_eeg.set`` write one
  row set per channel with the expected names and units; CIRUS writes none and
  does not fail.
* Wonambi spindle methods fill ``peak_freq`` / ``peak_val_det``, and for
  Moelle2011 ``peak_val_det / det_value_lo >= 1`` (every event crossed it).
* With ``cat=(0, 0, 0, 0)`` every bout is its own segment with its own
  threshold, and
  ``read_detection_thresholds(at_time=...)`` returns the event's own segment.
* A Massimini2004 slow-wave run writes run-wide ``channel='*'`` criteria and
  ``det_zero_time``, with ``end_time - det_zero_time`` the negative half-wave;
  a K-complex run adds ``min_isolation``.
* ``read_detection_thresholds`` returns the channel's rows plus the ``'*'``
  rows; re-running the same detection does not duplicate rows.
* A 4.5-style database gains the columns and the table additively, keeps its
  rows, and an old run reads back as an empty frame.

Run standalone: ``python tests/test_detection_thresholds.py``. Exits non-zero
if any test fails.
"""

import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from wonambi import Dataset  # noqa: E402
from wonambi.attr import Annotations  # noqa: E402

from turtlewave_hdEEG import ParalEvents, ParalKC, ParalSWA, dbwrite  # noqa: E402
from turtlewave_hdEEG.annotation import XLAnnotations  # noqa: E402
from turtlewave_hdEEG.dataset import LargeDataset  # noqa: E402
from turtlewave_hdEEG.extensions import THRESHOLD_UNITS  # noqa: E402

SET_FILE = os.path.join(HERE, 'synthetic_sleep_eeg.set')
NEW_COLUMNS = ('peak_freq', 'peak_val_det', 'rms_det', 'rms_orig',
               'power_orig', 'det_zero_time')
LACOURSE_NAMES = {'abs_pow_thresh', 'rel_pow_thresh', 'covar_thresh',
                  'corr_thresh'}


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_thresh_')
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def _set_recording(tmp):
    """The 3-channel synthetic .set, staged from its header."""
    xml = os.path.join(tmp, 'set_scoring.xml')
    xl = XLAnnotations(LargeDataset(SET_FILE), xml, rater_name='tester')
    assert xl.process_all() is True
    return Dataset(SET_FILE), Annotations(xml)


def _q(db, sql, params=()):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _spindles(dataset, annot, db, tmp, method, chan, cat=(1, 1, 1, 0),
              stage=('NREM2',), frequency=(11, 16)):
    return ParalEvents(dataset, annot, log_level=logging.CRITICAL).detect_spindles(
        method=method, chan=list(chan), frequency=frequency, stage=list(stage),
        json_dir=tmp, db_path=db, subject='sub-T', cat=cat)


def test_spindle_thresholds_on_set():
    """Moelle2011 / Lacourse2018 rows per channel; CIRUS none, no error."""
    print("\n1. Spindle thresholds on synthetic_sleep_eeg.set:")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'neural_events.db')
        for method in ('Moelle2011', 'Lacourse2018', 'CIRUS'):
            _spindles(dataset, annot, db, tmp, method, ('C3', 'C4'))

        rows = _q(db, "SELECT method, channel, name, units, value, seg_start, "
                      "seg_end FROM detection_thresholds")
        by = {}
        for method, chan, name, units, value, s0, s1 in rows:
            by.setdefault((method, chan), {})[name] = (units, value, s0, s1)

        for ch in ('C3', 'C4'):
            moelle = by[('Moelle2011', ch)]
            assert set(moelle) == {'det_value_lo'}, moelle
            assert moelle['det_value_lo'][0] == 'uV', moelle
            assert moelle['det_value_lo'][1] > 0, moelle
            lac = by[('Lacourse2018', ch)]
            assert set(lac) == LACOURSE_NAMES, lac
            assert {n: u for n, (u, *_rest) in lac.items()} == {
                'abs_pow_thresh': 'log10(uV^2)', 'rel_pow_thresh': 'z',
                'covar_thresh': 'z', 'corr_thresh': 'r'}, lac
            assert lac['abs_pow_thresh'][1] == 1.25, lac
            s0, s1 = moelle['det_value_lo'][2:]
            assert s0 is not None and s1 is not None and s0 < s1, (s0, s1)
        print(f"[ok] Moelle2011 det_value_lo per channel: "
              f"C3={by[('Moelle2011', 'C3')]['det_value_lo'][1]:.3f} uV, "
              f"C4={by[('Moelle2011', 'C4')]['det_value_lo'][1]:.3f} uV; "
              f"Lacourse2018 stores its four with units")

        assert not [k for k in by if k[0] == 'CIRUS'], by.keys()
        ok = _q(db, "SELECT COUNT(*) FROM processing_status "
                    "WHERE method = 'CIRUS' AND success = 1")[0][0]
        assert ok == 2, f"CIRUS channels not recorded as complete: {ok}"
        print("[ok] CIRUS writes no threshold rows and both channels complete")

        assert len(rows) == 2 * 1 + 2 * 4, len(rows)
        _spindles(dataset, annot, db, tmp, 'Moelle2011', ('C3', 'C4'))
        run_ids = {r[0] for r in _q(db, "SELECT DISTINCT run_id FROM "
                                        "detection_thresholds WHERE method="
                                        "'Moelle2011'")}
        per_run = _q(db, "SELECT run_id, COUNT(*) FROM detection_thresholds "
                         "WHERE method='Moelle2011' GROUP BY run_id")
        assert all(n == 2 for _, n in per_run), per_run
        print(f"[ok] a second Moelle2011 run adds its own rows under its own "
              f"run_id ({len(run_ids)} runs, 2 rows each)")


def test_segment_thresholds_and_at_time():
    """cat=(0,0,0,0): one threshold per bout; at_time picks the event's own.

    (Not ``cat=None``: Wonambi's ``fetch`` raises on it, so that default fails
    every channel before detection -- a separate, pre-existing defect.)
    """
    print("\n2. Per-segment thresholds (cat=(0,0,0,0)) and at_time lookup:")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'neural_events.db')
        _spindles(dataset, annot, db, tmp, 'Moelle2011', ('C3',),
                  cat=(0, 0, 0, 0))
        rows = _q(db, "SELECT segment_idx, value, seg_start, seg_end FROM "
                      "detection_thresholds ORDER BY segment_idx")
        assert len(rows) >= 2, f"expected one row per NREM2 bout, got {rows}"
        assert [r[0] for r in rows] == list(range(len(rows))), rows
        assert len({round(r[1], 9) for r in rows}) == len(rows), \
            f"segments share one threshold: {rows}"
        run_id = _q(db, "SELECT DISTINCT run_id FROM detection_thresholds")[0][0]
        _idx, value, s0, s1 = rows[1]
        df = dbwrite.read_detection_thresholds(db, run_id, channel='C3',
                                               method='Moelle2011',
                                               at_time=(s0 + s1) / 2)
        assert len(df) == 1 and df['segment_idx'].iloc[0] == 1, df
        assert abs(df['value'].iloc[0] - value) < 1e-12, df
        print(f"[ok] {len(rows)} NREM2 bouts -> {len(rows)} thresholds "
              f"({', '.join(f'{r[1]:.3f}' for r in rows)} uV); at_time "
              f"inside bout 1 returns only bout 1")


def test_spindle_event_values():
    """peak_freq / peak_val_det filled for Wonambi methods; Moelle ratio >= 1."""
    print("\n3. Per-event detector values for Wonambi spindle methods:")
    from test_turtlewave import _synthetic_recording
    with Workdir() as tmp:
        dataset, annot = _synthetic_recording(tmp, ('NREM2',) * 4)
        db = os.path.join(tmp, 'neural_events.db')
        for method in ('Moelle2011', 'Lacourse2018'):
            _spindles(dataset, annot, db, tmp, method, ('Cz',))
        stats = {m: (n, a, b, c, d, e, f) for m, n, a, b, c, d, e, f in _q(
            db, "SELECT method, COUNT(*), COUNT(peak_freq), COUNT(peak_val_det),"
                " COUNT(rms_det), COUNT(rms_orig), COUNT(power_orig), "
                "COUNT(det_zero_time) FROM events GROUP BY method")}
        for method in ('Moelle2011', 'Lacourse2018'):
            n, n_pf, n_pv, n_rd, n_ro, n_po, n_zt = stats[method]
            assert n > 0, f"{method} found nothing on the fixture"
            assert n_pf == n_pv == n_rd == n_ro == n_po == n, stats[method]
            assert n_zt == 0, f"spindles got det_zero_time: {stats[method]}"
        freqs = [r[0] for r in _q(db, "SELECT peak_freq FROM events WHERE "
                                      "method='Lacourse2018'")]
        assert all(10 <= f <= 17 for f in freqs), freqs
        ratio = _q(db, """
            SELECT MIN(e.peak_val_det / t.value) FROM events e
            JOIN detection_thresholds t ON t.run_id = e.run_id
             AND t.channel = e.channel AND t.method = e.method
             AND t.name = 'det_value_lo'
            WHERE e.method = 'Moelle2011'""")[0][0]
        assert ratio is not None and ratio >= 1.0, ratio
        print(f"[ok] {stats['Moelle2011'][0]} Moelle2011 + "
              f"{stats['Lacourse2018'][0]} Lacourse2018 events all carry "
              f"peak_freq/peak_val_det/rms_det/rms_orig/power_orig; Lacourse "
              f"peak_freq {min(freqs):.1f}-{max(freqs):.1f} Hz; min Moelle "
              f"peak_val_det/det_value_lo = {ratio:.2f} (>= 1)")


def test_slow_wave_and_kcomplex_criteria():
    """Massimini2004: '*' criteria + det_zero_time; K-complex adds isolation."""
    print("\n4. Slow-wave / K-complex run-wide criteria and det_zero_time:")
    from test_turtlewave import _synthetic_recording
    with Workdir() as tmp:
        dataset, annot = _synthetic_recording(tmp, ('NREM3',) * 4)
        db = os.path.join(tmp, 'neural_events.db')
        kw = dict(chan=['Cz'], stage=['NREM3'], json_dir=tmp, db_path=db,
                  subject='sub-T', cat=(1, 1, 1, 0))
        n_sw = len(ParalSWA(dataset, annot, log_level=logging.CRITICAL)
                   .detect_slow_waves(method='Massimini2004',
                                      frequency=(0.1, 4), **kw))
        n_kc = len(ParalKC(dataset, annot, log_level=logging.CRITICAL)
                   .detect_kcomplexes(method='AASM/Massimini2004',
                                      frequency=(0.5, 4), **kw))
        assert n_sw > 0 and n_kc > 0, (n_sw, n_kc)

        sw = {n: (v, u, c) for n, v, u, c in _q(
            db, "SELECT name, value, units, channel FROM detection_thresholds "
                "WHERE method = 'Massimini2004'")}
        assert {c for _v, _u, c in sw.values()} == {'*'}, sw
        assert sw['max_trough_amp'][:2] == (-80.0, 'uV'), sw
        assert sw['min_ptp'][:2] == (140.0, 'uV'), sw
        assert sw['trough_duration_lo'][0] == 0.3, sw
        assert sw['trough_duration_hi'][0] == 1.0, sw
        assert 'duration_lo' in sw and 'min_isolation' not in sw, sw
        kc = {n: v for n, v in _q(
            db, "SELECT name, value FROM detection_thresholds "
                "WHERE method = 'AASM/Massimini2004' AND channel = '*'")}
        assert kc['max_trough_amp'] == -40.0 and kc['min_ptp'] == 75.0, kc
        assert kc['min_isolation'] == 1.0, kc
        print(f"[ok] Massimini2004 '*' criteria {sorted(sw)}; "
              f"K-complex adds min_isolation={kc['min_isolation']}")

        rows = _q(db, "SELECT end_time - det_zero_time, det_trough, det_ptp "
                      "FROM events WHERE event_type = 'slow_wave'")
        assert len(rows) == n_sw and all(r[0] is not None for r in rows), rows
        assert all(0.3 <= r[0] <= 1.0 for r in rows), \
            [r[0] for r in rows if not 0.3 <= r[0] <= 1.0]
        tr = min(r[1] / sw['max_trough_amp'][0] for r in rows)
        pp = min(r[2] / sw['min_ptp'][0] for r in rows)
        assert tr >= 1.0 and pp >= 1.0, (tr, pp)
        print(f"[ok] {n_sw} slow waves carry det_zero_time; negative half-wave "
              f"end - det_zero_time within the 0.3-1.0 s criterion; min "
              f"det_trough/max_trough_amp = {tr:.2f}, det_ptp/min_ptp = {pp:.2f}")

        df = dbwrite.read_detection_thresholds(db, _q(
            db, "SELECT run_id FROM events WHERE event_type='slow_wave' "
                "LIMIT 1")[0][0], channel='Cz')
        assert set(df['channel']) == {'*'} and len(df) == len(sw), df
        print(f"[ok] read_detection_thresholds(channel='Cz') returns the "
              f"{len(df)} run-wide '*' rows")


def test_read_channel_plus_run_wide():
    """channel=X returns X's rows and the '*' rows, never another channel's."""
    print("\n5. read_detection_thresholds channel filter:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'neural_events.db')
        conn = sqlite3.connect(db)
        dbwrite.ensure_detection_thresholds_schema(conn)
        dbwrite.store_detection_thresholds(
            conn, 'r1', 'C3', 'Moelle2011',
            {'det_value_lo': 3.4, 'sel_value': float('nan')}, {'det_value_lo': 'uV'})
        dbwrite.store_detection_thresholds(
            conn, 'r1', 'C4', 'Moelle2011', {'det_value_lo': 2.9}, 'uV')
        dbwrite.store_detection_thresholds(
            conn, 'r1', '*', 'Moelle2011', {'note_factor': 1.5}, None)
        dbwrite.store_detection_thresholds(
            conn, 'r2', 'C3', 'Moelle2011', {'det_value_lo': 9.9}, 'uV')
        conn.commit()
        df = dbwrite.read_detection_thresholds(conn, 'r1', channel='C3')
        conn.close()
        assert sorted(df['channel']) == ['*', 'C3'], df
        assert 'sel_value' not in set(df['name']), "NaN value was stored"
        assert df.loc[df['channel'] == 'C3', 'value'].iloc[0] == 3.4, df
        assert list(df.columns) == list(dbwrite.THRESHOLD_COLUMNS), df.columns
        print(f"[ok] channel='C3' on run r1 -> rows {sorted(df['channel'])}; "
              f"NaN sel_value skipped; other run and channel excluded")


def test_old_database_migrates_additively():
    """A 4.5 database gains the columns + table, keeps rows, reads empty."""
    print("\n6. Additive migration of a 4.5-style database:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'neural_events.db')
        conn = sqlite3.connect(db)
        cols = list(dbwrite.EVENT_INSERT_COLUMNS) + ['epoch_stage']
        conn.execute(f"CREATE TABLE events ({', '.join(cols)}, "
                     f"PRIMARY KEY (uuid))")
        conn.execute(f"INSERT INTO events (uuid, event_type, channel, "
                     f"start_time, method, run_id) VALUES "
                     f"('u1', 'spindle', 'C3', 10.0, 'Moelle2011', 'old')")
        conn.commit()

        # A write before the schema is ensured still lands (presence-gated),
        # and thresholds create their table on demand.
        dbwrite.ensure_db_meta_schema(conn)
        conn.execute("CREATE TABLE processing_status (channel TEXT, "
                     "event_type TEXT, method TEXT, freq_lower REAL, "
                     "freq_upper REAL, stage TEXT, json_file TEXT, processed "
                     "BOOLEAN, attempts INTEGER, last_attempt_time TEXT, "
                     "success BOOLEAN, error_message TEXT, PRIMARY KEY "
                     "(channel, event_type, method, freq_lower, freq_upper, "
                     "stage))")
        conn.commit()
        ev = {'uuid': 'u2', 'start_time': 20.0, 'end_time': 21.0,
              'duration': 1.0, 'stage': 'NREM2', 'method': 'Moelle2011',
              'peak_freq': 13.0}
        dbwrite.write_channel_events(
            conn, 'new', 'spindle', 'C3', 'Moelle2011', 11, 16, 'NREM2', [ev],
            [{}], None, None,
            thresholds=[{'channel': 'C3', 'method': 'Moelle2011',
                         'values': {'det_value_lo': 2.0}, 'units': 'uV'}])
        assert _q(db, "SELECT COUNT(*) FROM events")[0][0] == 2
        assert _q(db, "SELECT COUNT(*) FROM detection_thresholds")[0][0] == 1
        print("[ok] pre-migration write: event lands without the new columns; "
              "thresholds table created on demand")

        dbwrite.ensure_direct_write_schema(conn)
        have = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        assert set(NEW_COLUMNS) <= have, set(NEW_COLUMNS) - have
        assert _q(db, "SELECT COUNT(*) FROM events")[0][0] == 2
        assert _q(db, "SELECT peak_freq FROM events WHERE uuid='u1'")[0][0] is None
        old = dbwrite.read_detection_thresholds(conn, 'old', channel='C3')
        assert old.empty and list(old.columns) == list(dbwrite.THRESHOLD_COLUMNS)
        before = conn.execute("SELECT sql FROM sqlite_master WHERE "
                              "name='detection_thresholds'").fetchone()
        dbwrite.ensure_direct_write_schema(conn)
        after = conn.execute("SELECT sql FROM sqlite_master WHERE "
                             "name='detection_thresholds'").fetchone()
        assert before == after
        conn.close()
        print(f"[ok] ensure_direct_write_schema adds {list(NEW_COLUMNS)} and "
              f"the table, keeps both rows (old row's new columns NULL), is "
              f"idempotent; the pre-4.6 run reads back empty")


def test_threshold_units_table():
    """Every method's ratio flags follow the Method Spec M4 table."""
    print("\n7. THRESHOLD_UNITS ratio flags:")
    allowed = {m: {n for n, ok in s['ratio_allowed'].items() if ok}
               for m, s in THRESHOLD_UNITS.items()}
    for m in ('Moelle2011', 'Ferrarelli2007', 'Nir2011'):
        assert allowed[m] == {'det_value_lo'}, (m, allowed[m])
        assert THRESHOLD_UNITS[m]['ratio_value'] == {
            'det_value_lo': 'peak_val_det'}, m
        assert 'sel_value' in THRESHOLD_UNITS[m]['units'], m
    for m in ('Ray2015', 'Wamsley2012', 'Martin2013', 'Lacourse2018',
              'CIRUS', 'Ngo2015', 'Staresina2015'):
        assert not allowed[m], (m, allowed[m])
    for m in ('Massimini2004', 'AASM/Massimini2004'):
        assert allowed[m] == {'max_trough_amp', 'min_ptp'}, allowed[m]
        assert THRESHOLD_UNITS[m]['ratio_value'] == {
            'max_trough_amp': 'det_trough', 'min_ptp': 'det_ptp'}
    for m, s in THRESHOLD_UNITS.items():
        assert set(s['ratio_allowed']) == set(s['units']), m
    print("[ok] ratio allowed: Moelle/Ferrarelli/Nir on det_value_lo only "
          "(sel_value stored, no ratio), "
          "Massimini/AASM on trough and ptp; none for Ray/Wamsley/Martin/"
          "Lacourse/CIRUS/Ngo/Staresina")


def test_csv_export_round_trip():
    """Export carries the new columns; the importer re-reads it (added=0)."""
    print("\n8. CSV export of the new columns and re-import:")
    import pandas as pd
    from test_turtlewave import _synthetic_recording
    with Workdir() as tmp:
        dataset, annot = _synthetic_recording(tmp, ('NREM2',) * 4)
        db = os.path.join(tmp, 'neural_events.db')
        _spindles(dataset, annot, db, tmp, 'Moelle2011', ('Cz',))
        csv = dbwrite.export_events_to_csv(db, 'spindle', 'Moelle2011',
                                           (11, 16), ['NREM2'], output_dir=tmp)
        csv = csv if isinstance(csv, str) else csv[0]
        assert os.path.basename(csv).startswith(
            'spindle_parameters_Moelle2011_11-16Hz_NREM2'), csv
        # Same header search as the importer: the export starts with a
        # provenance preamble and may carry summary rows under the header.
        with open(csv, encoding='utf-8') as f:
            header_row = next(i for i, ln in enumerate(f) if 'Start time' in ln)
        df = pd.read_csv(csv, skiprows=header_row)
        df = df[pd.to_numeric(df['Start time'], errors='coerce').notna()]
        for header in ('peak_freq (Hz)', 'peak_val_det', 'rms_det',
                       'rms_orig (uV)', 'power_orig', 'det_zero_time (s)'):
            assert header in df.columns, (header, list(df.columns))
        assert df['peak_freq (Hz)'].notna().all(), df['peak_freq (Hz)']
        assert df['det_zero_time (s)'].isna().all()
        # The source database refuses a CSV import over its direct-write rows
        # (it would blank run_id), so the importer side is checked on a fresh
        # database: every row lands and a second import adds none.
        fresh = os.path.join(tmp, 'fresh.db')
        pe = ParalEvents(None, None, log_level=logging.CRITICAL)
        pe.initialize_sqlite_database(fresh)
        res1 = pe.import_parameters_csv_to_database(csv, fresh)
        res2 = pe.import_parameters_csv_to_database(csv, fresh)
        n_fresh = _q(fresh, "SELECT COUNT(*) FROM events")[0][0]
        assert n_fresh == len(df), (n_fresh, len(df), res1)
        assert res2.get('added') == 0, res2
        print(f"[ok] {os.path.basename(csv)}: {len(df)} rows with the six new "
              f"columns; importer reads it into a fresh database "
              f"(added={res1.get('added')}), second import added="
              f"{res2.get('added')}, skipped={res2.get('skipped')}")


def test_default_cat_runs():
    """No cat argument: (1,1,1,0) is used and recorded; bad cat raises."""
    print("\n9. Default cat on synthetic_sleep_eeg.set and cat validation:")
    import json
    from turtlewave_hdEEG.utils import resolve_cat
    from test_turtlewave import _synthetic_recording
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'neural_events.db')
        ParalEvents(dataset, annot, log_level=logging.CRITICAL).detect_spindles(
            method='Moelle2011', chan=['C3', 'C4', 'O1'], frequency=(11, 16),
            stage=['NREM2'], json_dir=tmp, db_path=db, subject='sub-T')
        status = _q(db, "SELECT channel, success, error_message FROM "
                        "processing_status ORDER BY channel")
        assert [s[1] for s in status] == [1, 1, 1], status
        params = json.loads(_q(db, "SELECT params_json FROM detection_runs "
                                   "WHERE event_type='spindle'")[0][0])
        assert tuple(params['cat']) == (1, 1, 1, 0), params.get('cat')
        n_thr = _q(db, "SELECT COUNT(*) FROM detection_thresholds")[0][0]
        assert n_thr == 3, n_thr
        print(f"[ok] detect_spindles with no cat: all 3 channels succeed, "
              f"params_json.cat = {params['cat']}, one threshold per channel")

    with Workdir() as tmp:
        dataset, annot = _synthetic_recording(tmp, ('NREM3',) * 2)
        db = os.path.join(tmp, 'neural_events.db')
        kw = dict(chan=['Cz'], stage=['NREM3'], json_dir=tmp, db_path=db,
                  subject='sub-T')
        n_sw = len(ParalSWA(dataset, annot, log_level=logging.CRITICAL)
                   .detect_slow_waves(method='Massimini2004',
                                      frequency=(0.1, 4), **kw))
        n_kc = len(ParalKC(dataset, annot, log_level=logging.CRITICAL)
                   .detect_kcomplexes(method='AASM/Massimini2004',
                                      frequency=(0.5, 4), **kw))
        failed = _q(db, "SELECT COUNT(*) FROM processing_status "
                        "WHERE success = 0")[0][0]
        assert n_sw > 0 and n_kc > 0 and failed == 0, (n_sw, n_kc, failed)
        print(f"[ok] detect_slow_waves / detect_kcomplexes with no cat: "
              f"{n_sw} slow waves, {n_kc} K-complexes, no failed channel")

    assert resolve_cat(None) == (1, 1, 1, 0)
    assert resolve_cat([0, 1, 1, 0]) == (0, 1, 1, 0)
    for bad in ('1110', 1, (1, 1, 1), (1, 1, 2, 0), ('1', 1, 1, 0)):
        try:
            resolve_cat(bad)
        except ValueError:
            continue
        raise AssertionError(f"resolve_cat accepted {bad!r}")
    with Workdir() as tmp:
        dataset, annot = _set_recording(tmp)
        db = os.path.join(tmp, 'neural_events.db')
        try:
            ParalEvents(dataset, annot, log_level=logging.CRITICAL) \
                .detect_spindles(method='Moelle2011', chan=['C3'],
                                 stage=['NREM2'], json_dir=tmp, db_path=db,
                                 subject='sub-T', cat='1110')
        except ValueError as e:
            msg = str(e)
        else:
            raise AssertionError("cat='1110' did not raise")
        assert 'cat must be' in msg, msg
    print("[ok] None -> (1, 1, 1, 0); '1110', 1, 3-tuples, non-0/1 flags "
          "and strings raise ValueError, including from detect_spindles")


TESTS = [test_spindle_thresholds_on_set, test_segment_thresholds_and_at_time,
         test_spindle_event_values, test_slow_wave_and_kcomplex_criteria,
         test_read_channel_plus_run_wide, test_old_database_migrates_additively,
         test_threshold_units_table, test_csv_export_round_trip,
         test_default_cat_runs]


if __name__ == '__main__':
    logging.disable(logging.WARNING)
    failed = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failed += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
