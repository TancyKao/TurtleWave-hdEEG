#!/usr/bin/env python3
"""Review decisions carried over when a re-detection replaces a channel.

A re-detect with ``replace_channels=`` deletes the channel's events and
writes new ones, so every ``event_reviews`` decision on the channel loses its
event. :func:`turtlewave_hdEEG.dbwrite.write_channel_events` (the one write
all three processors use, ``replace=(ch in replace_set)``) now re-attaches
them through ``carry_over_reviews`` -> ``rematch_orphaned_reviews(dry_run=
False, channels=[ch], event_type=...)``. Asserted here:

1. Through ``write_channel_events`` directly (the processors' write, with
   hand-made event times): of three Cz decisions, the event re-detected at
   the same start (new uuid, because the stage set changed) and the one
   starting 0.05 s later get their decisions back with a ``[rematched
   from ...]`` comment; the one whose event is gone stays orphaned and
   unchanged; an orphaned decision on Fz and one on a Cz slow wave, both
   with a perfect successor, are NOT touched; nothing is deleted; one INFO
   line reports 2 / 1 / 0. The review-sample event follows its decision
   (``read_sample_labels`` / ``compute_review_precision`` count it), and
   the sample's 0.05 s end-time rule still voids it when the end moves.
   A later Wamsley2012 replace of Cz does not adopt the Moelle2011 orphan
   (reported as a near miss).
2. A database without ``event_reviews`` (written before 4.6), one with an
   empty table, and one whose table cannot be read: the replace commits,
   nothing raises.
3. Through the real ``ParalEvents.detect_spindles(replace_channels=['Cz'])``
   on a synthetic EEGLAB recording: a decision on an earlier detection 30 ms
   off the re-detected event is carried over.

Run standalone: ``python tests/test_review_carryover.py``. Exits non-zero if
any test fails.
"""

import gc
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback

# Windows consoles and CI pipes default to cp1252, which cannot encode some
# glyphs these checks print; replace rather than crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from turtlewave_hdEEG import dbwrite  # noqa: E402
from turtlewave_hdEEG import review_sampling as rs  # noqa: E402
from review_population_fixture import BASE_EVENTS_DDL  # noqa: E402

BAND = (11.0, 16.0)
STATUS_DDL = '''
CREATE TABLE IF NOT EXISTS processing_status (
    channel TEXT NOT NULL, event_type TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT '', freq_lower REAL NOT NULL DEFAULT 0,
    freq_upper REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT '',
    json_file TEXT, processed BOOLEAN DEFAULT 0, attempts INTEGER DEFAULT 0,
    last_attempt_time TEXT, success BOOLEAN DEFAULT 0, error_message TEXT,
    PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage)
)'''


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_carry_')
        return self.path

    def __exit__(self, *exc):
        gc.collect()   # Windows: drop datasets that still map files
        shutil.rmtree(self.path, ignore_errors=True)


class Capture(logging.Handler):
    """Collect formatted records of one logger."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelno, record.getMessage()))

    def carry_lines(self):
        return [m for lvl, m in self.lines
                if lvl == logging.INFO and m.startswith('Review carry-over')]


def _logger():
    log = logging.getLogger('turtlewave_hdEEG.test_carryover')
    log.handlers[:] = []
    log.propagate = False
    log.setLevel(logging.DEBUG)
    cap = Capture()
    log.addHandler(cap)
    return log, cap


def _db(tmp, with_reviews=True):
    """A database on the library's schema (events + processing_status)."""
    conn = sqlite3.connect(os.path.join(tmp, 'neural_events.db'))
    conn.execute(BASE_EVENTS_DDL)
    conn.execute(STATUS_DDL)
    conn.commit()
    dbwrite.ensure_direct_write_schema(conn)
    if not with_reviews:
        # A database written before 4.6: no event_reviews, no views over it.
        conn.execute("DROP VIEW IF EXISTS v_event_density_reviewed")
        conn.execute("DROP VIEW IF EXISTS events_reviewed")
        conn.execute("DROP TABLE IF EXISTS event_reviews")
        conn.commit()
    return conn


def _uid(ch, start, method='Moelle2011', stage='NREM2', event_type='spindle',
         band=BAND):
    return dbwrite.event_uuid5(event_type, ch, start, method, band[0],
                               band[1], stage)


def _write(conn, ch, events, stage='NREM2', method='Moelle2011',
           event_type='spindle', band=BAND, replace=False, run_id='run-1',
           logger=None, **kw):
    """``events`` = [(start, end), ...] for one channel, one method."""
    evs = [{'uuid': _uid(ch, s, method, stage, event_type, band),
            'start_time': s, 'end_time': e, 'duration': e - s,
            'stage': stage, 'epoch_stage': 'NREM2', 'method': method}
           for s, e in events]
    dbwrite.write_channel_events(
        conn, run_id, event_type, ch, method, band[0], band[1], stage, evs,
        [], None, None, logger=logger, replace=replace,
        replace_methods=[method], **kw)
    return [e['uuid'] for e in evs]


def _review(conn, uuid):
    return conn.execute(
        "SELECT uuid, decision, reason, comment, start_time, run_id "
        "FROM event_reviews WHERE uuid = ? AND reviewer = 'alice'",
        (uuid,)).fetchone()


def _draw_by_hand(conn, sample_id, rows):
    """A review sample of ``rows`` = [(uuid, start, end), ...].

    Written straight into the library's sample tables (one
    central|NREM2 unflagged sub-cell of 10 events, 3 sampled) so the test
    controls exactly which events are in it.
    """
    rs.ensure_review_sampling_schema(conn)
    conn.execute(
        "INSERT INTO review_sample_designs (sample_id, subject, event_type, "
        "method, freq_lower, freq_upper, stage_scope, run_ids, seed, "
        "design_json, design_hash, population_hash, n_population, "
        "n_in_scope, n_out_of_scope, flag_available, top_ups, drawn_at) "
        "VALUES (?, 'sub-T', 'spindle', 'Moelle2011', ?, ?, 'NREM2', "
        "'[\"run-1\"]', 1, '{}', 'h', 'p', 10, 10, '{}', 0, '{}', 'now')",
        (sample_id, BAND[0], BAND[1]))
    for k, (u, s, e) in enumerate(rows):
        conn.execute(
            "INSERT INTO review_samples (sample_id, uuid, subject, run_id, "
            "event_type, channel, start_time, end_time, cell, region, stage, "
            "flagged, pop_k, n_k, weight, prn, sort_key, irr_key, is_shared, "
            "draw_round, seed, design_hash, drawn_at) VALUES (?, ?, 'sub-T', "
            "'run-1', 'spindle', 'Cz', ?, ?, 'central|NREM2|U', 'central', "
            "'NREM2', NULL, 10, 3, 3.3333, ?, ?, ?, 0, 0, 1, 'h', 'now')",
            (sample_id, u, s, e, k / 10.0, f"{k:016x}", f"{k:016x}"))
    conn.commit()


# ---------------------------------------------------------------------------

def test_carry_over_on_replace():
    """Two decisions carried over, one left, other channel/type untouched."""
    print("\n1. write_channel_events replace carries decisions over:")
    log, cap = _logger()
    with Workdir() as tmp:
        conn = _db(tmp)
        cz = _write(conn, 'Cz', [(100.0, 100.8), (200.0, 200.8),
                                 (300.0, 300.8)])
        (fz,) = _write(conn, 'Fz', [(100.0, 100.8)])
        (sw,) = _write(conn, 'Cz', [(500.0, 500.9)], event_type='slow_wave',
                       method='Massimini2004', band=(0.5, 4.0))
        dbwrite.store_event_review(conn, cz[0], 'accept', 'alice',
                                   comment='clear')
        dbwrite.store_event_review(conn, cz[1], 'reject', 'alice',
                                   reason='artefact')
        dbwrite.store_event_review(conn, cz[2], 'unsure', 'alice')
        dbwrite.store_event_review(conn, fz, 'reject', 'alice',
                                   reason='artefact')
        dbwrite.store_event_review(conn, sw, 'accept', 'alice')
        _draw_by_hand(conn, 'S1', [(cz[0], 100.0, 100.8),
                                   (cz[1], 200.0, 200.8),
                                   (cz[2], 300.0, 300.8)])
        before = rs.read_sample_labels(conn, 'S1', reviewer='alice')
        assert len(before) == 3 and before['valid'].all(), before

        # Orphan the Fz and slow-wave decisions with a perfect successor each,
        # WITHOUT carrying them over: the Cz spindle replace below must not
        # pick them up, although an unscoped rematch would.
        (fz_new,) = _write(conn, 'Fz', [(100.02, 100.82)], replace=True,
                           carry_reviews=False)
        (sw_new,) = _write(conn, 'Cz', [(500.03, 500.93)],
                           event_type='slow_wave', method='Massimini2004',
                           band=(0.5, 4.0), replace=True, carry_reviews=False)
        n_rows = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
        assert n_rows == 5

        # Re-detect Cz over a different stage set: same start (new uuid,
        # the stage token is in it), +0.05 s (same end), gone, and new.
        stage2 = dbwrite.join_stage_token(['NREM2', 'NREM3'])
        new = _write(conn, 'Cz', [(100.0, 100.8), (200.05, 200.8),
                                  (400.0, 400.8)], stage=stage2,
                     replace=True, run_id='run-2', logger=log)
        assert new[0] != cz[0] and new[1] != cz[1]

        lines = cap.carry_lines()
        assert lines == ["Review carry-over (spindle on Cz): 2 decision(s) "
                         "carried over, 1 left unmatched, 0 near miss(es) "
                         "with a different method or band not applied (see "
                         "rematch_orphaned_reviews)"], lines

        r0, r1 = _review(conn, new[0]), _review(conn, new[1])
        assert r0[1] == 'accept' and r0[3] == (
            f"clear [rematched from {cz[0]}, dt=0.000s]"), r0
        assert r1[1] == 'reject' and r1[2] == 'artefact' and r1[3] == (
            f"[rematched from {cz[1]}, dt=0.050s]"), r1
        assert r0[5] == r1[5] == 'run-2' and abs(r1[4] - 200.05) < 1e-9
        assert _review(conn, cz[0]) is None and _review(conn, cz[1]) is None

        gone = _review(conn, cz[2])          # left as it was, orphaned
        assert gone is not None and gone[1] == 'unsure' and gone[3] is None
        assert gone[5] == 'run-1'
        assert _review(conn, new[2]) is None  # the new event: no decision

        for old, succ in ((fz, fz_new), (sw, sw_new)):
            row = _review(conn, old)
            assert row is not None and row[3] is None, row
            assert _review(conn, succ) is None
        n_after = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
        assert n_after == n_rows, "a decision was deleted"
        orph = dbwrite.read_event_reviews(conn, include_orphaned=True)
        assert set(orph.loc[orph['orphaned'], 'uuid']) == {cz[2], fz, sw}
        # ... and both were eligible: an unscoped dry run would move them
        dry = dbwrite.rematch_orphaned_reviews(conn)
        st = dict(zip(dry['old_uuid'], dry['status']))
        assert st == {cz[2]: 'no_match', fz: 'proposed', sw: 'proposed'}, st
        scoped = dbwrite.rematch_orphaned_reviews(conn, channels='Fz')
        assert list(scoped['old_uuid']) == [fz]
        scoped = dbwrite.rematch_orphaned_reviews(conn, event_type='slow_wave')
        assert list(scoped['old_uuid']) == [sw]
        n_after = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
        assert n_after == n_rows and _review(conn, fz) is not None  # dry runs
        print(f"   [ok] dt=0.000 and dt=0.050 decisions re-keyed with the "
              f"comment; the gone event's decision orphaned and unchanged; "
              f"Fz and slow-wave orphans untouched; {n_after} rows kept")
        print(f"   [ok] log: {lines[0]}")

        # The review sample follows the decision.
        smp = dict(conn.execute(
            "SELECT uuid, end_time FROM review_samples WHERE sample_id = 'S1'"
        ).fetchall())
        assert set(smp) == {new[0], new[1], cz[2]}, smp
        assert smp[new[1]] == 200.8           # draw-time end kept
        lab = rs.read_sample_labels(conn, 'S1', reviewer='alice')
        why = dict(zip(lab['uuid'], lab['void_reason']))
        assert len(lab) == 3 and why == {new[0]: None, new[1]: None,
                                         cz[2]: 'missing'}, why
        dec = dict(zip(lab['uuid'], lab['decision']))
        assert dec[new[0]] == 'accept' and dec[new[1]] == 'reject'
        prec = rs.compute_review_precision(conn, 'S1', reviewer='alice',
                                           write=False)
        scope = prec[prec['domain_type'] == 'scope'].iloc[0]
        assert (scope['n_reviewed'], scope['n_accept'],
                scope['n_reject']) == (2, 1, 1), scope.to_dict()
        prog = rs.sample_progress(conn, 'S1', reviewer='alice')
        assert prog['n_reviewed'] == 2, prog
        # the sample's end-time rule still applies to a carried label
        conn.execute("UPDATE events SET end_time = 201.0 WHERE uuid = ?",
                     (new[1],))
        conn.commit()
        lab = rs.read_sample_labels(conn, 'S1', reviewer='alice')
        why = dict(zip(lab['uuid'], lab['void_reason']))
        assert why[new[1]] == 'end_moved', why
        conn.execute("UPDATE events SET end_time = 200.8 WHERE uuid = ?",
                     (new[1],))
        conn.commit()
        print(f"   [ok] sample S1 re-keyed: read_sample_labels 2 valid + "
              f"1 missing; precision n_reviewed=2 (1 accept, 1 reject); an "
              f"end moved 0.2 s still voids the carried label")

        # A different-method replace of Cz near the orphan does not adopt it.
        cap.lines.clear()
        (wam,) = _write(conn, 'Cz', [(300.0, 300.8)], stage=stage2,
                        method='Wamsley2012', replace=True, run_id='run-3',
                        logger=log)
        lines = cap.carry_lines()
        assert lines and lines[0].startswith(
            "Review carry-over (spindle on Cz): 0 decision(s) carried over, "
            "0 left unmatched, 1 near miss(es)"), lines
        assert _review(conn, wam) is None
        assert _review(conn, cz[2])[3] is None
        # the carried decisions stay where they are
        assert _review(conn, new[0])[1] == 'accept'
        conn.close()
        print("   [ok] Wamsley2012 at the orphan's start: 1 near miss, "
              "not adopted")


def test_no_or_empty_or_broken_reviews_table():
    """The replace commits and nothing raises without usable reviews."""
    print("\n2. Replace on a database without usable event_reviews:")
    log, cap = _logger()
    with Workdir() as tmp:
        conn = _db(tmp, with_reviews=False)
        _write(conn, 'Cz', [(100.0, 100.8)])
        new = _write(conn, 'Cz', [(100.05, 100.85)], replace=True, logger=log)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert 'event_reviews' not in names, "carry-over created the table"
        assert [r[0] for r in conn.execute("SELECT uuid FROM events")] == new
        assert dbwrite.carry_over_reviews(conn, 'spindle', ['Cz']) == {
            'carried': 0, 'unmatched': 0, 'near_miss': 0}
        conn.close()
    print("   [ok] pre-4.6 database: replace written, no event_reviews "
          "created, no error")

    with Workdir() as tmp:
        conn = _db(tmp)
        _write(conn, 'Cz', [(100.0, 100.8)])
        _write(conn, 'Cz', [(100.05, 100.85)], replace=True, logger=log)
        assert conn.execute("SELECT COUNT(*) FROM event_reviews"
                            ).fetchone()[0] == 0
        assert not cap.carry_lines()      # nothing in scope: no INFO line
        conn.close()
    print("   [ok] empty event_reviews: replace written, silent")

    with Workdir() as tmp:
        conn = _db(tmp, with_reviews=False)
        # a table the reader cannot use (no decision/channel columns)
        conn.execute("CREATE TABLE event_reviews (uuid TEXT, reviewer TEXT)")
        conn.execute("INSERT INTO event_reviews VALUES ('x', 'alice')")
        conn.commit()
        _write(conn, 'Cz', [(100.0, 100.8)])
        cap.lines.clear()
        new = _write(conn, 'Cz', [(100.05, 100.85)], replace=True, logger=log)
        assert [r[0] for r in conn.execute("SELECT uuid FROM events")] == new
        warn = [m for lvl, m in cap.lines if lvl == logging.WARNING]
        assert warn and warn[0].startswith("Could not carry review decisions"), \
            cap.lines
        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM event_reviews"
                            ).fetchone()[0] == 1
        conn.close()
    print("   [ok] unreadable event_reviews: replace written, one WARNING, "
          "no raise, row kept")


def test_processor_replace_channels():
    """ParalEvents.detect_spindles(replace_channels=['Cz']) carries over."""
    print("\n3. ParalEvents re-detect with replace_channels=['Cz']:")
    import test_event_reviews as ter
    with Workdir() as tmp:
        dataset, annot, db = ter._recording(tmp)
        ter._detect(dataset, annot, db)
        conn = dbwrite.open_write_connection(db)
        row = conn.execute(
            "SELECT uuid, start_time FROM events WHERE channel = 'Cz' "
            "ORDER BY start_time LIMIT 1").fetchone()
        assert row is not None, "the fixture produced no Cz spindle"
        target, start = row
        # Pretend the earlier detection put this event 30 ms earlier, under
        # another uuid, and that alice rejected it there.
        old = 'earlier-detection-of-' + target
        conn.execute("UPDATE events SET uuid = ?, start_time = ? "
                     "WHERE uuid = ?", (old, start - 0.03, target))
        conn.commit()
        dbwrite.store_event_review(conn, old, 'reject', 'alice',
                                   reason='artefact')
        conn.close()

        ter._detect(dataset, annot, db, replace_channels=['Cz'])
        conn = sqlite3.connect(db)
        assert conn.execute("SELECT 1 FROM events WHERE uuid = ?",
                            (old,)).fetchone() is None
        rv = _review(conn, target)
        conn.close()
        assert rv is not None, "decision not carried over by the processor"
        assert rv[1] == 'reject' and rv[3] == (
            f"[rematched from {old}, dt=0.030s]"), rv
        assert dbwrite.read_event_reviews(db, include_orphaned=True)[
            'orphaned'].sum() == 0
    print(f"   [ok] the processor's replace carried the decision onto "
          f"{target[:8]}... (dt=0.030 s)")


TESTS = [
    test_carry_over_on_replace,
    test_no_or_empty_or_broken_reviews_table,
    test_processor_replace_channels,
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    print("TESTING review carry-over on replace_channels")
    print("=============================================")
    failed = []
    for test in TESTS:
        try:
            test()
        except Exception:
            failed.append(test.__name__)
            traceback.print_exc()
    print()
    if failed:
        print(f"{len(failed)} of {len(TESTS)} tests FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(TESTS)} tests passed.")
