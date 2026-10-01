#!/usr/bin/env python3
"""Stage B: sleep cycles and stage durations of cut recordings on the full night.

A cut recording is staged with exact variable-length epochs and a timeline
sidecar holding the full-night hypnogram. ``ParalCycles.run`` /
``finalize_cycles_and_durations`` with ``timeline='auto'`` detect cycles and
stage durations on that full-night hypnogram at 30 s, move the cycle bounds
onto the cut file's epoch edges, and record per-cycle coverage in
``analysed_time_cycles``.

The fixture night (142 epochs, 4260 s) has two cycles. Two splices remove
600.4 s at original 1000 s (inside the first N2 run) and 150 s at original
2000 s (inside the first REM period, leaving 210 s of its 360 s), so the
first cycle's REM is under the 5 min coverage floor.

Run standalone: ``python tests/test_stage_b_fullnight.py``. Exits non-zero if
any test fails.
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eeglab_fixture as fx  # noqa: E402
from turtlewave_hdEEG import dbwrite  # noqa: E402
from turtlewave_hdEEG.annotation import CustomAnnotations, XLAnnotations  # noqa: E402
from turtlewave_hdEEG.cycleprocessor import (  # noqa: E402
    ParalCycles, finalize_cycles_and_durations)
from turtlewave_hdEEG.dataset import LargeDataset  # noqa: E402
from turtlewave_hdEEG.density import read_cycle_analysed_time  # noqa: E402
from turtlewave_hdEEG.timeline import (  # noqa: E402
    SidecarMismatchError, load_sidecar_for, sidecar_path)

FS = 100.0
NIGHT = (['W'] * 4 + ['2'] * 40 + ['3'] * 20 + ['R'] * 12 + ['2'] * 40
         + ['3'] * 10 + ['R'] * 12 + ['W'] * 4)          # 142 epochs, 4260 s
REMOVED = (600.4, 150.0)
CUT_ONSETS = (1000.0, 2000.0 - 600.4)                     # cut-file seconds
T_CUT = 4260.0 - sum(REMOVED)                             # 3509.6 s
LAST_SECOND = int(T_CUT)


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_stage_b_')
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


class LogCapture(logging.Handler):
    """Collect records reaching one logger.

    Attach to the ``turtlewave_hdEEG`` parent: ``ParalCycles`` replaces the
    handlers of its own logger when it is constructed."""

    def __init__(self, name):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.logger = logging.getLogger(name)

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)

    def messages(self, level=None):
        return [r.getMessage() for r in self.records
                if level is None or r.levelno == level]


def _write_cut(tmp, name='cut', with_stages=True, stage_events=False):
    """The cut fixture recording, annotated; returns the XML path."""
    ev = [fx.event(c * FS + 0.5, 'boundary', r * FS)
          for c, r in zip(CUT_ONSETS, REMOVED)]
    ev.append(fx.event(500 * FS + 1, 'arousal', 20 * FS))
    if stage_events:
        names = {'W': 'wake', '2': 'n2', '3': 'n3', 'R': 'rem'}
        removed_at = 0.0
        spans = [(1000.0, 1600.4), (2000.0, 2150.0)]
        for i, code in enumerate(NIGHT):
            orig = 30.0 * i
            if any(a <= orig < b for a, b in spans):
                continue
            removed_at = sum(b - a for a, b in spans if b <= orig)
            ev.append(fx.event((orig - removed_at) * FS + 1, names[code], 30 * FS))
    path = os.path.join(tmp, f'{name}.set')
    fx.write_set(path, 'root', True, n_samples=int(round(T_CUT * FS)), srate=FS,
                 labels=['Cz'], types=None, ref=None, n_good=None,
                 stages=NIGHT if with_stages else None, events=ev)
    annot_file = os.path.join(tmp, f'{name}.xml')
    xl = XLAnnotations(LargeDataset(path), annot_file, rater_name='tester')
    assert xl.process_all() is True
    return annot_file


def _db(tmp, n_events=120, spacing=30.0):
    """A database with a bare events table (start_time, cycle, run_id)."""
    db = os.path.join(tmp, 'neural_events.db')
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE events (uuid TEXT PRIMARY KEY, start_time REAL, '
                 'cycle TEXT, run_id TEXT)')
    conn.executemany('INSERT INTO events VALUES (?, ?, NULL, NULL)',
                     [(f'e{i}', 5.0 + i * spacing) for i in range(n_events)])
    conn.commit()
    conn.close()
    return db


def _q(db, sql, params=()):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_sidecar_reader():
    """load_sidecar_for: match, rescored epoch, wrong file name, missing."""
    print("\n1. Sidecar reader:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        tl = load_sidecar_for(annot_file)
        assert tl.last_second == LAST_SECOND and tl.fullnight_hypnogram() == NIGHT
        print(f"   [ok] matching XML: {len(tl.cut_epochs)} epochs, last_second {tl.last_second}")

        other = os.path.join(tmp, 'renamed.xml')
        shutil.copy(annot_file, other)
        shutil.copy(sidecar_path(annot_file), sidecar_path(other))
        try:
            load_sidecar_for(other)
            raise AssertionError("a sidecar naming another XML must not pass")
        except SidecarMismatchError as e:
            assert "'cut.xml'" in str(e), e
        print("   [ok] sidecar written for another annotation file -> SidecarMismatchError")

        ca = CustomAnnotations(annot_file)
        ca.wonb_annot.set_stage_for_epoch(int(ca.get_stage_intervals()[10][0]), 'REM')
        try:
            load_sidecar_for(annot_file)
            raise AssertionError("a rescored epoch must not pass")
        except SidecarMismatchError as e:
            assert 'epoch 10' in str(e), e
        db = _db(tmp)
        try:
            finalize_cycles_and_durations(CustomAnnotations(annot_file), db,
                                          subject='sub-x', write_xml=False)
            raise AssertionError("finalize must refuse a mismatched sidecar")
        except SidecarMismatchError:
            pass
        assert _q(db, "SELECT name FROM sqlite_master WHERE name='sleep_cycles'") == []
        print("   [ok] rescored epoch -> SidecarMismatchError from the reader and from finalize; "
              "nothing written")

        os.remove(sidecar_path(annot_file))
        try:
            ParalCycles(annotations=CustomAnnotations(annot_file),
                        log_level=logging.ERROR).run(db, write_xml=False)
            raise AssertionError("variable epochs without a sidecar must raise")
        except ValueError as e:
            assert 'no timeline sidecar' in str(e), e
        print("   [ok] variable epochs and no sidecar -> ValueError")


def test_fullnight_cycles_on_cut_file():
    """finalize(timeline='auto'): full-night durations, cut-file bounds, coverage."""
    print("\n2. finalize_cycles_and_durations(timeline='auto') on the cut fixture:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        ca = CustomAnnotations(annot_file)
        with LogCapture('turtlewave_hdEEG') as log:
            out = finalize_cycles_and_durations(
                ca, db, subject='sub-x', timeline='auto', write_xml=True,
                plot=True, plot_path=os.path.join(tmp, 'c.png'),
                stages=['NREM2', 'NREM3', 'REM'], log_level=logging.INFO)
        notices = [m for m in log.messages(logging.INFO) if 'full-night hypnogram in the' in m]
        assert len(notices) == 1, notices
        print("   [ok] timeline notice logged once for two methods")
        assert os.path.getsize(os.path.join(tmp, 'c.png')) > 0

        sd = _q(db, "SELECT epoch_length, wake_min, n2_min, n3_min, rem_min, total_min, "
                    "time_base FROM stage_durations")
        assert sd == [(30.0, 4.0, 40.0, 15.0, 12.0, 71.0, 'original')], sd
        print(f"   [ok] stage_durations from the full night: {sd[0]}")

        assert [len(out[m]) for m in ('2022', '1979')] == [2, 2], {m: len(c) for m, c in out.items()}
        ivals = CustomAnnotations(annot_file).get_stage_intervals()
        edges = {s for s, _, _ in ivals} | {ivals[-1][1]}
        starts = {s for s, _, _ in ivals}
        rows = _q(db, "SELECT method, cycle_number, nrem_start, nrem_end, rem_end, "
                      "nrem_start_orig, nrem_end_orig, rem_end_orig, time_base, "
                      "cycle_dur_min FROM sleep_cycles ORDER BY method, cycle_number")
        for r in rows:
            assert all(x <= LAST_SECOND and x in edges for x in r[2:5]), r
            assert r[8] == 'cut' and r[5] <= r[6] <= r[7] <= 4260, r
        r22 = [r for r in rows if r[0] == '2022']
        assert r22[0][5:8] == (120.0, 1920.0, 2280.0), r22[0]
        assert r22[0][2:5] == (120.0, 1320.0, 1530.0), r22[0]     # cut times
        assert r22[0][9] == 36.0, "cycle_dur_min stays full-night"
        print(f"   [ok] sleep_cycles in cut seconds on epoch edges, *_orig on the night: {r22[0]}")

        mk = CustomAnnotations(annot_file).wonb_annot.get_cycles()
        assert mk and all(m[0] in starts and m[1] in starts for m in mk), mk
        assert [m[0] for m in mk] == [r[2] for r in r22], (mk, r22)
        print(f"   [ok] XML markers on epoch starts: {mk}")

        tagged = _q(db, "SELECT cycle, COUNT(*) FROM events GROUP BY cycle ORDER BY cycle")
        assert dict(tagged).get('1') and dict(tagged).get('2'), tagged
        lo2 = r22[1][2]
        assert _q(db, "SELECT COUNT(*) FROM events WHERE cycle='1' AND start_time >= ?",
                  (lo2,))[0][0] == 0
        print(f"   [ok] events tagged on cut seconds: {tagged}")

        df = read_cycle_analysed_time(db, 'sub-x')
        assert len(df) == 2 * 2 * 3, len(df)
        in_cut = df.fullnight_seconds - df.removed_seconds
        assert (df.fullnight_seconds >= in_cut).all() and (in_cut >= df.analysed_seconds - 1e-9).all()
        assert (df.coverage.dropna() <= 1).all()
        rem1 = df[(df.method == '2022') & (df.cycle_number == 1) & (df.stage == 'REM')].iloc[0]
        assert rem1.fullnight_seconds == 360 and rem1.removed_seconds == 150, rem1
        assert bool(rem1.low_coverage) and abs(rem1.masked_seconds - 4.0) < 0.05, rem1   # +/-2 s splice mask
        n2 = df[(df.method == '2022') & (df.cycle_number == 1) & (df.stage == 'NREM2')].iloc[0]
        # the 600.4 s cut at original 1000 s takes the last 320 s of N2 and
        # 280.4 s of N3; the 20 s arousal and 2 s of the splice mask fall in N2
        assert n2.fullnight_seconds == 1200 and n2.removed_seconds == 320, n2
        assert abs(n2.masked_seconds - 22.0) < 0.05 and not n2.low_coverage, n2
        n3 = df[(df.method == '2022') & (df.cycle_number == 1) & (df.stage == 'NREM3')].iloc[0]
        assert n3.fullnight_seconds == 600 and abs(n3.removed_seconds - 280.4) <= 0.6, n3
        print(f"   [ok] analysed_time_cycles: {len(df)} rows; cycle 1 REM {rem1.analysed_seconds:.1f} s analysed of 360 s (150 s cut, 4 s splice mask) -> low_coverage; "
              f"N2 masked {n2.masked_seconds:.1f} s by the arousal")


def test_uniform_without_sidecar_behaves_as_before():
    """A grid XML without a sidecar: legacy path, no analysed_time_cycles rows."""
    print("\n3. Uniform epochs without a sidecar:")
    with Workdir() as tmp:
        path = os.path.join(tmp, 'uncut.set')
        fx.write_set(path, 'root', True, n_samples=int(4260 * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=NIGHT)
        annot_file = os.path.join(tmp, 'uncut.xml')
        assert XLAnnotations(LargeDataset(path), annot_file, rater_name='tester').add_stages_from_header()
        assert not os.path.exists(sidecar_path(annot_file))
        db = _db(tmp)
        out = finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-u',
                                            write_xml=False)
        rows = _q(db, "SELECT nrem_start, rem_end, nrem_start_orig, rem_end_orig, time_base "
                      "FROM sleep_cycles WHERE method='2022' ORDER BY cycle_number")
        assert rows == [(120.0, 2280.0, 120.0, 2280.0, 'original'),
                        (2280.0, 4140.0, 2280.0, 4140.0, 'original')], rows
        assert _q(db, "SELECT time_base, total_min FROM stage_durations") == [('original', 71.0)]
        assert read_cycle_analysed_time(db).empty
        assert [len(out[m]) for m in ('2022', '1979')] == [2, 2]
        print(f"   [ok] legacy cycles {rows}; no analysed_time_cycles rows")

        try:
            ParalCycles(annotations=CustomAnnotations(_write_cut(tmp)),
                        log_level=logging.ERROR).run(db, write_xml=False, timeline='none')
            raise AssertionError("timeline='none' on variable epochs must raise")
        except ValueError as e:
            assert "timeline='none'" in str(e), e
        print("   [ok] timeline='none' on variable epochs -> ValueError")


def test_stage_event_source_has_no_cycles():
    """Staged from stage events: no full night, so no cycles; cut-time durations."""
    print("\n4. Stage-event source (no full-night hypnogram):")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp, name='ev', with_stages=False, stage_events=True)
        assert json.load(open(sidecar_path(annot_file)))['fullnight_stages'] == []
        db = _db(tmp)
        ca = CustomAnnotations(annot_file)
        with LogCapture('turtlewave_hdEEG') as log:
            out = finalize_cycles_and_durations(ca, db, subject='sub-e', write_xml=False,
                                                log_level=logging.INFO)
        errors = [m for m in log.messages(logging.ERROR) if 'NOT computed' in m]
        assert len(errors) == 1, log.messages()
        assert out == {'2022': [], '1979': []}
        sd = _q(db, "SELECT epoch_length, total_min, time_base FROM stage_durations")
        durs = ca.epoch_durations()
        med = float(np.median(durs))
        assert sd == [(med, sum(durs) / 60.0, 'cut')], (sd, med, sum(durs))
        assert _q(db, "SELECT COUNT(*) FROM sleep_cycles") == [(0,)]
        assert read_cycle_analysed_time(db).empty
        assert _q(db, "SELECT COUNT(*) FROM events WHERE cycle IS NOT NULL") == [(0,)]
        print(f"   [ok] one ERROR, no cycles, stage_durations {sd[0]}, no coverage rows")

        # epoch_length is the median epoch duration of the variable-epoch file,
        # not a length to multiply an epoch count by; the minutes are real sums.
        assert not ca.has_uniform_epochs() and min(durs) <= med <= max(durs)
        assert abs(sd[0][1] - sum(durs) / 60.0) < 1e-9
        print(f"   [ok] epoch_length is the median epoch duration ({med:g} s); "
              f"total_min is the sum of {len(durs)} real durations")

        # A second run must clear what an earlier full-night run left behind
        # for the same subject, and still write only one ERROR per run.
        pc = ParalCycles(annotations=ca, log_level=logging.ERROR)
        conn = sqlite3.connect(db)
        pc._ensure_sleep_cycles_table(conn)
        conn.execute("INSERT INTO sleep_cycles (subject, method, cycle_number, time_base) "
                     "VALUES ('sub-e', '2022', 1, 'cut')")
        conn.execute("UPDATE events SET cycle = '1' WHERE uuid = 'e0'")
        conn.commit()
        conn.close()
        finalize_cycles_and_durations(ca, db, subject='sub-e', write_xml=False,
                                      log_level=logging.ERROR)
        assert _q(db, "SELECT COUNT(*) FROM sleep_cycles WHERE subject='sub-e'") == [(0,)]
        assert _q(db, "SELECT COUNT(*) FROM events WHERE cycle IS NOT NULL") == [(0,)]
        print("   [ok] a stale cycle row and event tag are cleared by the stage-event run")


def test_ensure_cycles_populated_fills_missing_coverage():
    """A stored cut subject gets coverage rows for a new stage from its stored cycles."""
    print("\n5. ensure_cycles_populated and analysed_time_cycles:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        conn = dbwrite.open_write_connection(db)
        try:
            r1 = dbwrite.ensure_cycles_populated(conn, CustomAnnotations(annot_file), 'sub-x',
                                                 db_path=db, stages=['NREM2'],
                                                 reject_types=['Artefact'])
            assert set(r1) == {'2022', '1979'}
            r2 = dbwrite.ensure_cycles_populated(conn, CustomAnnotations(annot_file), 'sub-x',
                                                 db_path=db, stages=['NREM2'],
                                                 reject_types=['Artefact'])
            assert r2 == {}, r2
            r3 = dbwrite.ensure_cycles_populated(conn, CustomAnnotations(annot_file), 'sub-x',
                                                 db_path=db, stages=['NREM3'],
                                                 reject_types=['Artefact'])
            assert r3 == {}, r3          # filled from the stored cycles, not re-detected
        finally:
            conn.close()
        df = read_cycle_analysed_time(db, 'sub-x')
        assert set(df.stage) == {'NREM2', 'NREM3'} and set(df.reject_types) == {'Artefact'}, df
        print("   [ok] second call with the same stages is a no-op; a new stage is filled in")


def test_coverage_floor_and_low_coverage():
    """low_coverage follows coverage_floor_min; the floor is stored in seconds."""
    print("\n6. Coverage floor:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      methods=('2022',), write_xml=False,
                                      stages=['NREM2', 'NREM3', 'REM'],
                                      log_level=logging.ERROR)
        df = read_cycle_analysed_time(db, 'sub-x')
        assert (df.coverage_floor_seconds == 300.0).all(), "default floor is 5 min"
        under = df.analysed_seconds < 300.0
        assert (df.low_coverage == under).all(), df
        # cycle 1 REM: 210 s left in the file, 4 s masked -> 206 s < 300 s
        rem1 = df[(df.cycle_number == 1) & (df.stage == 'REM')].iloc[0]
        assert 0 < rem1.analysed_seconds < 300 and rem1.low_coverage
        assert df.low_coverage.any() and not df.low_coverage.all()
        print(f"   [ok] default floor 300 s; low_coverage rows: "
              f"{int(df.low_coverage.sum())} of {len(df)}")

        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      methods=('2022',), write_xml=False,
                                      stages=['NREM2', 'NREM3', 'REM'],
                                      coverage_floor_min=1.0, log_level=logging.ERROR)
        df = read_cycle_analysed_time(db, 'sub-x')
        assert (df.coverage_floor_seconds == 60.0).all()
        assert (df.low_coverage == (df.analysed_seconds < 60.0)).all()
        assert not df[(df.cycle_number == 1) & (df.stage == 'REM')].low_coverage.iloc[0]
        print("   [ok] coverage_floor_min=1 -> floor 60 s, cycle 1 REM no longer flagged")


def test_read_cycle_analysed_time():
    """The reader: columns, order, dtype, subject normalisation, absent table."""
    print("\n7. read_cycle_analysed_time:")
    from turtlewave_hdEEG.density import CYCLE_ANALYSED_TIME_COLUMNS
    with Workdir() as tmp:
        db = _db(tmp)
        empty = read_cycle_analysed_time(db)
        assert empty.empty and list(empty.columns) == list(CYCLE_ANALYSED_TIME_COLUMNS)
        print("   [ok] no table -> empty frame with the documented columns")

        annot_file = _write_cut(tmp)
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      write_xml=False, log_level=logging.ERROR)
        df = read_cycle_analysed_time(db)
        assert list(df.columns) == list(CYCLE_ANALYSED_TIME_COLUMNS)
        assert df.low_coverage.dtype == bool
        assert set(df.subject) == {'sub-x'} and set(df.method) == {'1979', '2022'}
        key = ['subject', 'method', 'reject_types', 'cycle_number', 'stage']
        assert df[key].equals(df.sort_values(key).reset_index(drop=True)[key]), "sorted"
        assert read_cycle_analysed_time(db, 'x').equals(df), "'x' finds 'sub-x'"
        assert read_cycle_analysed_time(db, 'sub-x').equals(df)
        assert read_cycle_analysed_time(db, 'sub-nobody').empty
        assert df.processing_timestamp.notna().all() and df.turtlewave_version.notna().all()
        assert df.annotation_file.str.endswith('cut.xml').all()
        print(f"   [ok] {len(df)} rows, sorted, bool low_coverage, subject normalised, "
              f"unknown subject -> empty")


def test_old_database_gains_columns_with_nulls():
    """A pre-4.5 database gains the additive columns; its old rows read NULL."""
    print("\n8. Old database migration:")
    with Workdir() as tmp:
        db = _db(tmp)
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE sleep_cycles (
            subject TEXT, method TEXT, cycle_number INTEGER, nrem_start REAL,
            nrem_end REAL, rem_start REAL, rem_end REAL, nrem_dur_min REAL,
            nrem_n23_dur_min REAL, rem_dur_min REAL, cycle_dur_min REAL,
            PRIMARY KEY (subject, method, cycle_number))""")
        conn.execute("""CREATE TABLE stage_durations (
            subject TEXT, epoch_length REAL, wake_min REAL, n1_min REAL,
            n2_min REAL, n3_min REAL, rem_min REAL, artefact_min REAL,
            total_min REAL, PRIMARY KEY (subject))""")
        conn.execute("INSERT INTO sleep_cycles VALUES ('sub-old','2022',1,0,900,900,1200,"
                     "15,15,5,20)")
        conn.execute("INSERT INTO stage_durations VALUES ('sub-old',30,1,1,1,1,1,0,5)")
        conn.commit()
        conn.close()
        new_cols = {'nrem_start_orig', 'nrem_end_orig', 'rem_end_orig', 'time_base'}
        assert not new_cols & {r[1] for r in _q(db, "PRAGMA table_info(sleep_cycles)")}

        finalize_cycles_and_durations(CustomAnnotations(_write_cut(tmp)), db,
                                      subject='sub-new', write_xml=False,
                                      log_level=logging.ERROR)
        cols = {r[1] for r in _q(db, "PRAGMA table_info(sleep_cycles)")}
        assert new_cols <= cols, new_cols - cols
        assert 'time_base' in {r[1] for r in _q(db, "PRAGMA table_info(stage_durations)")}
        old = _q(db, "SELECT nrem_start_orig, nrem_end_orig, rem_end_orig, time_base, "
                     "cycle_dur_min FROM sleep_cycles WHERE subject='sub-old'")
        assert old == [(None, None, None, None, 20.0)], old
        old_sd = _q(db, "SELECT time_base, total_min FROM stage_durations "
                        "WHERE subject='sub-old'")
        assert old_sd == [(None, 5.0)], old_sd
        new = _q(db, "SELECT DISTINCT time_base FROM sleep_cycles WHERE subject='sub-new'")
        assert new == [('cut',)], new
        print("   [ok] new columns added; sub-old rows keep their values with NULL "
              "time_base / *_orig; sub-new rows say 'cut'")

        # the coverage table is created on the same run, with the documented columns
        got = [r[1] for r in _q(db, "PRAGMA table_info(analysed_time_cycles)")]
        assert got == list(__import__('turtlewave_hdEEG.density', fromlist=['x'])
                           .CYCLE_ANALYSED_TIME_COLUMNS), got
        print("   [ok] analysed_time_cycles columns match CYCLE_ANALYSED_TIME_COLUMNS")


def test_run_is_idempotent_and_rerun_replaces():
    """Re-running replaces the subject's coverage rows instead of adding to them."""
    print("\n9. Re-run replaces rows:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        for _ in range(2):
            finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                          write_xml=True, log_level=logging.ERROR)
        assert _q(db, "SELECT COUNT(*) FROM analysed_time_cycles") == [(16,)]
        assert _q(db, "SELECT COUNT(*) FROM sleep_cycles") == [(4,)]
        assert _q(db, "SELECT COUNT(*) FROM stage_durations") == [(1,)]
        mk = CustomAnnotations(annot_file).wonb_annot.get_cycles()
        assert len(mk) == 2, mk
        print("   [ok] 16 coverage rows (2 methods x 2 cycles x 4 stages), 4 cycles, 1 "
              "stage_durations row, 2 markers after two runs")


def test_coverage_arithmetic_exact():
    """coverage = analysed / fullnight, capped at 1; removed floored at 0."""
    print("\n10. analysed_time_cycles arithmetic on a cycle that lost data to a splice:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      write_xml=False, stages=['REM'], reject_types=['Artefact'])
        row = _q(db, "SELECT fullnight_seconds, removed_seconds, masked_seconds, analysed_seconds, "
                     "coverage, low_coverage FROM analysed_time_cycles WHERE method='2022' "
                     "AND cycle_number=1 AND stage='REM'")[0]
        full, removed, masked, analysed, coverage, low = row
        assert (full, removed) == (360.0, 150.0), row
        assert abs(masked - 4.0) < 1e-6 and abs(analysed - 206.0) < 1e-6, row
        assert coverage == analysed / full, (coverage, analysed / full)      # not analysed / in_cut
        assert abs(coverage - 206.0 / 360.0) < 1e-12 and coverage < 206.0 / 210.0 - 0.3, row
        assert low == 1, row
        print(f"   [ok] REM cycle 1: full 360, removed 150, masked 4, analysed 206, "
              f"coverage {coverage:.4f} = 206/360, low_coverage")

        # A cycle whose cut span holds more N2 than its full-night range
        # (rounding can do this by a second or two; exaggerated here): removed
        # is floored at 0 and coverage capped at 1.
        conn = dbwrite.open_write_connection(db)
        try:
            ca = CustomAnnotations(annot_file)
            dbwrite.store_cycle_analysed_time(
                conn, 'sub-y', '2022',
                [{'cycle_number': 1, 'nrem_start_sec': 120.0, 'rem_end_sec': 1320.0,
                  'nrem_start_orig': 120.0, 'rem_end_orig': 150.0}],
                load_sidecar_for(annot_file), ca, ['NREM2'], reject_types=['Artefact'])
        finally:
            conn.close()
        row = _q(db, "SELECT fullnight_seconds, removed_seconds, analysed_seconds, coverage "
                     "FROM analysed_time_cycles WHERE subject='sub-y'")[0]
        assert row[0] == 30.0 and row[1] == 0.0 and row[2] > 30.0 and row[3] == 1.0, row
        print(f"   [ok] in-cut time above full-night time: removed {row[1]} (floored), "
              f"coverage {row[3]} (capped)")


def test_stored_cycles_never_redetected():
    """ensure_cycles_populated keeps user-chosen cycles and fills coverage from them."""
    print("\n11. Stored cycles are not re-detected by a detection run:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      write_xml=False, nrem_min=50, stages=['REM'],
                                      reject_types=['Artefact'])
        before = _q(db, "SELECT method, cycle_number, nrem_start, rem_end FROM sleep_cycles "
                        "ORDER BY method, cycle_number")
        assert [r for r in before if r[0] == '2022'] and \
            len([r for r in before if r[0] == '2022']) == 1, before
        tags_before = _q(db, "SELECT uuid, cycle FROM events ORDER BY uuid")
        rem_before = _q(db, "SELECT * FROM analysed_time_cycles WHERE stage='REM'")

        conn = dbwrite.open_write_connection(db)
        try:
            r = dbwrite.ensure_cycles_populated(conn, CustomAnnotations(annot_file), 'sub-x',
                                                db_path=db, stages=['NREM2'],
                                                reject_types=['Arousal'])
        finally:
            conn.close()
        assert r == {}, r
        assert _q(db, "SELECT method, cycle_number, nrem_start, rem_end FROM sleep_cycles "
                      "ORDER BY method, cycle_number") == before
        assert _q(db, "SELECT uuid, cycle FROM events ORDER BY uuid") == tags_before
        assert _q(db, "SELECT * FROM analysed_time_cycles WHERE stage='REM'") == rem_before
        new = _q(db, "SELECT method, cycle_number, fullnight_seconds, reject_types "
                     "FROM analysed_time_cycles WHERE stage='NREM2' ORDER BY method")
        assert {(m, c) for m, c, _, _ in new} == {(m, c) for m, c, _, _ in before}, (new, before)
        assert all(k == 'Arousal' for *_, k in new), new
        # nrem_min=50 cycle 1 of '2022' spans original 120-4140 s: 80 N2 epochs
        assert [f for m, c, f, _ in new if m == '2022'] == [2400.0], new
        print(f"   [ok] cycles (nrem_min=50) kept: {before}; NREM2/Arousal coverage rows "
              f"computed from them: {new}")


def test_restore_clears_all_coverage_rows():
    """Re-storing a method's cycles drops its coverage rows for every stage and reject set."""
    print("\n12. Re-stored cycles clear every analysed_time_cycles row of the method:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp)
        db = _db(tmp)
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      write_xml=False, stages=['REM'], reject_types=['Artefact'])
        conn = dbwrite.open_write_connection(db)
        try:
            dbwrite.ensure_cycles_populated(conn, CustomAnnotations(annot_file), 'sub-x',
                                            db_path=db, stages=['NREM2'], reject_types=['Arousal'])
        finally:
            conn.close()
        assert {r[0] for r in _q(db, "SELECT DISTINCT reject_types FROM analysed_time_cycles")} \
            == {'Artefact', 'Arousal'}
        finalize_cycles_and_durations(CustomAnnotations(annot_file), db, subject='sub-x',
                                      write_xml=False, stages=['NREM3'], reject_types=['Move'])
        left = _q(db, "SELECT DISTINCT stage, reject_types FROM analysed_time_cycles")
        assert left == [('NREM3', 'Move')], left
        print(f"   [ok] after re-storing cycles only the new scope remains: {left}")


def test_cycles_unavailable_recorded():
    """A stage-event-staged subject is skipped quietly on later detection runs."""
    print("\n13. 'Cycles unavailable' is recorded and later runs skip with one INFO line:")
    with Workdir() as tmp:
        annot_file = _write_cut(tmp, name='ev', with_stages=False, stage_events=True)
        db = _db(tmp)
        results = []
        test_log = logging.getLogger('turtlewave_hdEEG.test')
        test_log.setLevel(logging.INFO)
        with LogCapture('turtlewave_hdEEG') as log:
            for _ in range(3):
                conn = dbwrite.open_write_connection(db)
                try:
                    results.append(dbwrite.ensure_cycles_populated(
                        conn, CustomAnnotations(annot_file), 'sub-e', db_path=db,
                        stages=['NREM2'], logger=test_log))
                finally:
                    conn.close()
        errors = [m for m in log.messages(logging.ERROR) if 'NOT computed' in m]
        skips = [m for m in log.messages(logging.INFO) if 'unavailable' in m]
        assert results == [{'2022': [], '1979': []}, {}, {}], results
        assert len(errors) == 1 and len(skips) == 2, (errors, skips)
        assert dbwrite.subject_cycles_unavailable(sqlite3.connect(db), 'sub-e')
        print(f"   [ok] first run: one ERROR; later runs: '{skips[0][:70]}...'")


TESTS = [
    test_sidecar_reader,
    test_fullnight_cycles_on_cut_file,
    test_uniform_without_sidecar_behaves_as_before,
    test_stage_event_source_has_no_cycles,
    test_ensure_cycles_populated_fills_missing_coverage,
    test_coverage_floor_and_low_coverage,
    test_read_cycle_analysed_time,
    test_old_database_gains_columns_with_nulls,
    test_run_is_idempotent_and_rerun_replaces,
    test_coverage_arithmetic_exact,
    test_stored_cycles_never_redetected,
    test_restore_clears_all_coverage_rows,
    test_cycles_unavailable_recorded,
]


if __name__ == "__main__":
    print("TESTING Stage B: full-night cycles on cut recordings")
    print("====================================================")
    failed = []
    for test in TESTS:
        try:
            test()
        except Exception:
            failed.append(test.__name__)
            print(f"   [FAIL] {test.__name__}")
            traceback.print_exc()
    print()
    if failed:
        print(f"FAILED {len(failed)} of {len(TESTS)}: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(TESTS)} Stage B tests passed.")
