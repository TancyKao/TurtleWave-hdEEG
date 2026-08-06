## test_cycle_connection_reuse.py
#
# Verifies the cycleprocessor side of the network-drive SQLite fix:
# finalize_cycles_and_durations() must share ONE dbwrite.open_write_connection
# across the whole subject (previously six connect/close cycles -- three
# storage methods x two cycle methods -- each of which, on a WAL database,
# deletes and recreates the -wal/-shm sidecars on a network share). No pytest;
# plain functions with prints + asserts, matching tests/test_turtlewave.py.

import os
import shutil
import sqlite3
import tempfile

from turtlewave_hdEEG import cycleprocessor as cp
from turtlewave_hdEEG import dbwrite


class FakeAnnotations:
    """Minimal annotations stub: cycle detection only needs the hypnogram."""

    def __init__(self, hypnogram):
        self._hypnogram = hypnogram
        self.epochs = None   # no epoch grid -> XML marker writing is skipped

    def get_hypnogram(self):
        return list(self._hypnogram)


# Two clean NREM->REM cycles: 40 N2 epochs, 15 REM, 40 N2, 15 REM, at 30 s.
HYPNO = ([0] * 10) + ([2] * 40) + ([4] * 15) + ([2] * 40) + ([4] * 15) + ([0] * 5)


def _tmp_db(tag):
    d = tempfile.mkdtemp(prefix=f'twcycle_{tag}_')
    return d, os.path.join(d, 'neural_events.db')


def _make_db(db_path, n_events=70):
    """Create an events table with one event every 60 s across the night."""
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE events (uuid TEXT PRIMARY KEY, '
                 'start_time REAL, cycle TEXT)')
    conn.executemany('INSERT INTO events VALUES (?, ?, NULL)',
                     [(f'e{i}', float(i * 60)) for i in range(n_events)])
    conn.commit()
    conn.close()


def _dump(db_path):
    """Snapshot every table this pipeline writes, for equality comparison."""
    conn = sqlite3.connect(db_path)
    try:
        return {
            'sleep_cycles': conn.execute(
                'SELECT * FROM sleep_cycles ORDER BY method, cycle_number'
            ).fetchall(),
            'stage_durations': conn.execute(
                'SELECT * FROM stage_durations ORDER BY subject').fetchall(),
            'events': conn.execute(
                'SELECT uuid, cycle FROM events ORDER BY uuid').fetchall(),
        }
    finally:
        conn.close()


def test_finalize_opens_exactly_one_connection():
    """finalize_cycles_and_durations() calls open_write_connection exactly once.

    Regression guard for the six-connections-per-subject bug: three storage
    methods (sleep_cycles, stage_durations, events.cycle) x two cycle methods
    ('2022', '1979') used to each open and close their own connection.
    """
    print("\n1. Testing finalize_cycles_and_durations opens exactly one "
          "connection:")
    d, db = _tmp_db('count')
    try:
        _make_db(db)
        calls = {'n': 0}
        real_open = dbwrite.open_write_connection

        def counting_open(*a, **k):
            calls['n'] += 1
            return real_open(*a, **k)

        cp.dbwrite.open_write_connection = counting_open
        try:
            cp.finalize_cycles_and_durations(
                FakeAnnotations(HYPNO), db, subject='sub-test',
                write_xml=False)
        finally:
            cp.dbwrite.open_write_connection = real_open

        print(f"   open_write_connection call count: {calls['n']}")
        assert calls['n'] == 1, (
            f"expected exactly 1 connection for the whole subject "
            f"(2 cycle methods x 3 storage calls each used to open 6), "
            f"got {calls['n']}")
        print("[ok] exactly one connection opened for the whole subject")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_finalize_writes_and_tags_correctly():
    """The shared-connection run still writes cycles, durations, and tags."""
    print("\n2. Testing finalize_cycles_and_durations writes the right tables:")
    d, db = _tmp_db('write')
    try:
        _make_db(db)
        cycles_by_method = cp.finalize_cycles_and_durations(
            FakeAnnotations(HYPNO), db, subject='sub-test', write_xml=False)
        dumped = _dump(db)

        n_2022 = len(cycles_by_method.get('2022', []))
        n_1979 = len(cycles_by_method.get('1979', []))
        n_sleep_cycles_rows = len(dumped['sleep_cycles'])
        n_stage_rows = len(dumped['stage_durations'])
        n_tagged = sum(1 for _, c in dumped['events'] if c is not None)

        print(f"   cycles: 2022={n_2022}, 1979={n_1979}; "
              f"sleep_cycles rows={n_sleep_cycles_rows}, "
              f"stage_durations rows={n_stage_rows}, "
              f"events tagged={n_tagged}/70")

        assert n_2022 > 0 and n_1979 > 0, "expected cycles under both methods"
        assert n_sleep_cycles_rows == n_2022 + n_1979, (
            "sleep_cycles should hold both methods' cycles side by side")
        assert n_stage_rows == 1, "one stage_durations row per subject"
        assert n_tagged > 0, "events.cycle should be tagged by the run"
        print("[ok] sleep_cycles, stage_durations and events.cycle all "
              "populated correctly")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_finalize_is_idempotent():
    """Re-running finalize_cycles_and_durations on the same DB changes nothing."""
    print("\n3. Testing finalize_cycles_and_durations is idempotent:")
    d, db = _tmp_db('idempotent')
    try:
        _make_db(db)
        cp.finalize_cycles_and_durations(
            FakeAnnotations(HYPNO), db, subject='sub-test', write_xml=False)
        before = _dump(db)
        cp.finalize_cycles_and_durations(
            FakeAnnotations(HYPNO), db, subject='sub-test', write_xml=False)
        after = _dump(db)

        for table in ('sleep_cycles', 'stage_durations', 'events'):
            match = before[table] == after[table]
            print(f"   {table}: {len(after[table])} rows, "
                  f"{'unchanged' if match else 'CHANGED'}")
            assert match, f"{table} changed across a re-run"
        print("[ok] a second run leaves every table byte-identical")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_storage_methods_work_standalone():
    """Each storage method still works on its own with conn=None."""
    print("\n4. Testing each storage method still works standalone (conn=None):")
    d, db = _tmp_db('standalone')
    try:
        _make_db(db)
        pc = cp.ParalCycles(annotations=FakeAnnotations(HYPNO),
                            subject='sub-solo')
        cycles = pc.detect(method='2022')

        n_cyc = pc.store_cycles_to_database(cycles, db, subject='sub-solo',
                                            method='2022')
        n_tag = pc.tag_events_with_cycles(cycles, db)
        durations = cp.compute_stage_durations(HYPNO)
        n_dur = pc.store_stage_durations(durations, db, subject='sub-solo')

        print(f"   store_cycles_to_database -> {n_cyc}, "
              f"tag_events_with_cycles -> {n_tag}, "
              f"store_stage_durations -> {n_dur}")

        assert n_cyc == len(cycles) and n_cyc > 0
        assert n_tag > 0
        assert n_dur == 1
        print("[ok] store_cycles_to_database, tag_events_with_cycles and "
              "store_stage_durations all work with conn=None")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_shared_connection_matches_standalone_result():
    """A supplied conn produces the same rows as letting each call open its own."""
    print("\n5. Testing a shared conn= produces the same result as conn=None:")
    d, db_shared = _tmp_db('shared')
    _, db_solo = _tmp_db('solo')
    try:
        _make_db(db_shared)
        _make_db(db_solo)

        # Shared connection, passed explicitly through all three calls.
        pc_shared = cp.ParalCycles(annotations=FakeAnnotations(HYPNO),
                                   subject='sub-x')
        cycles = pc_shared.detect(method='2022')
        conn = dbwrite.open_write_connection(db_shared)
        try:
            pc_shared.store_cycles_to_database(
                cycles, db_shared, subject='sub-x', method='2022', conn=conn)
            pc_shared.tag_events_with_cycles(cycles, db_shared, conn=conn)
            durations = cp.compute_stage_durations(HYPNO)
            pc_shared.store_stage_durations(
                durations, db_shared, subject='sub-x', conn=conn)
        finally:
            conn.close()

        # Standalone: each call opens/closes its own connection (conn=None).
        pc_solo = cp.ParalCycles(annotations=FakeAnnotations(HYPNO),
                                 subject='sub-x')
        cycles_solo = pc_solo.detect(method='2022')
        pc_solo.store_cycles_to_database(
            cycles_solo, db_solo, subject='sub-x', method='2022')
        pc_solo.tag_events_with_cycles(cycles_solo, db_solo)
        pc_solo.store_stage_durations(
            cp.compute_stage_durations(HYPNO), db_solo, subject='sub-x')

        shared_dump, solo_dump = _dump(db_shared), _dump(db_solo)
        for table in ('sleep_cycles', 'stage_durations', 'events'):
            match = shared_dump[table] == solo_dump[table]
            print(f"   {table}: {'identical' if match else 'DIFFERS'} "
                  f"({len(shared_dump[table])} rows)")
            assert match, (
                f"{table} differs between a shared conn and conn=None -- the "
                f"conn plumbing changed behaviour, not just connection count")
        print("[ok] passing a shared conn produces byte-identical results to "
              "conn=None")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(os.path.dirname(db_solo), ignore_errors=True)


if __name__ == "__main__":
    print("TESTING CYCLE-PROCESSOR CONNECTION REUSE (network-drive fix)")
    print("==============================================================")

    test_finalize_opens_exactly_one_connection()
    test_finalize_writes_and_tags_correctly()
    test_finalize_is_idempotent()
    test_storage_methods_work_standalone()
    test_shared_connection_matches_standalone_result()

    print("\nAll tests completed!")
