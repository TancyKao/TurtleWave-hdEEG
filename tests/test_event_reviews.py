#!/usr/bin/env python3
"""Per-event review decisions: ``event_reviews``, ``events_reviewed`` and
``exclude_rejected``.

A reviewer's accept / reject / unsure call is stored in its own table keyed by
``(uuid, reviewer)``, never on the ``events`` row (re-detection rewrites those
rows). Asserted here:

* ``store_event_review`` round-trips through ``read_event_reviews`` and copies
  the event's identity (type, channel, start, run, subject) from the database.
* The CHECK constraints refuse an unknown decision or reason even from raw
  SQL; ``store_event_review`` refuses them with ``ValueError``.
* Two reviewers on one event coexist; a reviewer re-deciding overwrites only
  their own row.
* ``events_reviewed`` and ``v_event_density_reviewed`` drop an event any
  reviewer rejected; ``event_density`` and ``export_events_to_csv`` drop it
  only with ``exclude_rejected=True``, and ``reviewer=`` narrows the rule.
* A same-parameter re-detection (plain and ``replace_channels``) reproduces
  the event's uuid, so the review stays attached.
* A band change whose old rows are gone, and a replace that shifts the start,
  orphan the review; ``read_event_reviews(include_orphaned=True)`` reports it
  and ``rematch_orphaned_reviews`` proposes the successor (dry run writes
  nothing; applying re-keys it).
* An existing database without the table gains it and both views.
* ``delete_event_review`` removes one reviewer's row only; an explicit
  ``reviewed_at`` restores a decision exactly (undo); ``reviewer=`` alone
  (the blind view) is served by ``idx_event_reviews_reviewer``.
* The Method Spec reason vocabulary and ``review_category`` (RA protocol);
  ``reason='other'`` needs a comment; a table under an older CHECK is rebuilt
  with every row kept and ``arousal-alpha`` mapped to ``arousal``.

Run standalone: ``python tests/test_event_reviews.py``. Exits non-zero if any
test fails.
"""

import csv
import logging
import gc
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eeglab_fixture as fx  # noqa: E402
from turtlewave_hdEEG import dbwrite  # noqa: E402
from turtlewave_hdEEG.density import event_density  # noqa: E402

FS = 128.0
LABELS = ['Fz', 'Cz', 'Pz']
N_EPOCHS = 6
BAND = (11, 16)
OTHER_BAND = (12, 15)


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_reviews_')
        return self.path

    def __exit__(self, *exc):
        gc.collect()   # Windows: drop datasets that still map files
        shutil.rmtree(self.path, ignore_errors=True)


def _signal(n_samples, seed=0):
    """Slow oscillation plus one 13.5 Hz, 1 s burst per 30 s epoch (uV)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    base = 60.0 * np.sin(2 * np.pi * 0.8 * t) + 5.0 * rng.standard_normal(n_samples)
    for i in range(int(n_samples / FS // 30)):
        t0 = i * 30.0 + 12.0
        burst = (t >= t0) & (t < t0 + 1.0)
        base[burst] += 30.0 * np.sin(2 * np.pi * 13.5 * (t[burst] - t0))
    return np.vstack([base + 0.5 * k * rng.standard_normal(n_samples)
                      for k in range(len(LABELS))]).astype(np.float32)


def _recording(tmp):
    """Write the fixture, stage it, return (dataset, annotations, db path)."""
    from turtlewave_hdEEG.annotation import CustomAnnotations, XLAnnotations
    from turtlewave_hdEEG.dataset import LargeDataset
    n = int(30 * N_EPOCHS * FS)
    path = os.path.join(tmp, 'rec.set')
    fx.write_set(path, 'root', True, n_samples=n, srate=FS, labels=LABELS,
                 types=['EEG'] * len(LABELS), ref='average', n_good=3,
                 stages=['2'] * N_EPOCHS, data=_signal(n))
    dataset = LargeDataset(path)
    xml = os.path.join(tmp, 'sub-T_scoring.xml')
    xl = XLAnnotations(dataset, xml, rater_name='tester')
    assert xl.process_all() is True, "header staging was not imported"
    out = os.path.join(tmp, 'wonambi')
    os.makedirs(out, exist_ok=True)
    return dataset, CustomAnnotations(xml), os.path.join(out, 'neural_events.db')


def _detect(dataset, annot, db, frequency=BAND, **kw):
    from turtlewave_hdEEG import ParalEvents
    proc = ParalEvents(dataset, annot, log_level=logging.ERROR)
    return proc.detect_spindles(
        method='Moelle2011', frequency=frequency, chan=LABELS,
        stage=['NREM2'], json_dir=os.path.dirname(db), db_path=db,
        subject='sub-T', cat=(1, 1, 1, 0), **kw)


def _uuids(db, band=BAND):
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute(
            "SELECT uuid FROM events WHERE freq_lower = ? AND freq_upper = ? "
            "ORDER BY channel, start_time", band)]
    finally:
        conn.close()


def _event(db, uuid):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT uuid, event_type, channel, start_time, run_id FROM events "
            "WHERE uuid = ?", (uuid,)).fetchone()
    finally:
        conn.close()


def _density_by_channel(db, **kw):
    df = event_density(db, event_type='spindle', method='Moelle2011',
                       stage=['NREM2'], **kw)
    return {r.channel: (int(r.n_events), float(r.density_per_min))
            for r in df.itertuples()}


def _csv_starts(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    hdr = next(i for i, r in enumerate(rows) if 'Start time' in r)
    col = rows[hdr].index('Start time')
    return [float(r[col]) for r in rows[hdr + 1:] if r]


# ---------------------------------------------------------------- the table

def test_round_trip_and_checks():
    """Store, read back, CHECK constraints, two reviewers."""
    print("\n1. Round trip, CHECK constraints, two reviewers:")
    with Workdir() as tmp:
        dataset, annot, db = _recording(tmp)
        _detect(dataset, annot, db)
        uuids = _uuids(db)
        assert len(uuids) >= 6, f"fixture yielded only {len(uuids)} events"
        target = uuids[0]
        ev = _event(db, target)

        conn = dbwrite.open_write_connection(db)
        row = dbwrite.store_event_review(conn, target, 'reject', 'alice',
                                         reason='filter-ringing',
                                         comment='step artefact')
        assert row['event_type'] == 'spindle' and row['channel'] == ev[2]
        assert row['start_time'] == ev[3] and row['run_id'] == ev[4]
        assert row['subject'] == 'sub-T', row['subject']
        assert row['method'] == 'Moelle2011', row['method']
        assert (row['freq_lower'], row['freq_upper']) == BAND
        assert row['turtlewave_version'] == dbwrite.provenance()['turtlewave_version']
        assert row['reviewed_at'] and ('+' in row['reviewed_at'][10:]
                                       or '-' in row['reviewed_at'][10:])

        # a second reviewer on the same event coexists
        dbwrite.store_event_review(conn, target, 'accept', 'bob')
        # alice changes her mind: her row is replaced, bob's is not
        dbwrite.store_event_review(conn, target, 'unsure', 'alice')
        dbwrite.store_event_review(conn, target, 'reject', 'alice',
                                   reason='artefact')

        # unknown values: ValueError from the API, IntegrityError from raw SQL
        for kwargs in (dict(decision='maybe'),
                       dict(decision='reject', reason='boring'),
                       dict(decision='reject', reason='arousal-alpha'),
                       dict(decision='reject', reason='other'),
                       dict(decision='reject', reason='other', comment='  '),
                       dict(decision='accept', reviewer='  ')):
            args = dict(decision='accept', reviewer='carol')
            args.update(kwargs)
            try:
                dbwrite.store_event_review(conn, target, **args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"store_event_review accepted {kwargs}")
        for decision, reason in (('maybe', None), ('reject', 'boring'),
                                 ('reject', 'arousal-alpha')):
            try:
                conn.execute(
                    "INSERT INTO event_reviews (uuid, reviewer, decision, reason) "
                    "VALUES (?, 'raw', ?, ?)", (target, decision, reason))
            except sqlite3.IntegrityError:
                pass
            else:
                raise AssertionError(f"CHECK let ({decision}, {reason}) in")
        # a uuid not in events, without identity: refused
        try:
            dbwrite.store_event_review(conn, 'not-a-uuid', 'accept', 'alice')
        except ValueError:
            pass
        else:
            raise AssertionError("review of an unknown uuid was stored")
        conn.close()

        df = dbwrite.read_event_reviews(db, uuid=target)
        got = {r.reviewer: (r.decision, r.reason) for r in df.itertuples()}
        assert got == {'alice': ('reject', 'artefact'),
                       'bob': ('accept', None)}, got
        assert not df['orphaned'].any()
        assert len(dbwrite.read_event_reviews(db, decision='reject')) == 1
        assert len(dbwrite.read_event_reviews(db, reviewer=['bob'])) == 1
    print(f"   [ok] identity copied (subject='sub-T', run_id); CHECK refuses "
          f"'maybe' and 'boring'; alice+bob coexist, alice overwrote herself")


def test_exclusion_view_density_export():
    """Rejected events leave the view, the density and the CSV on request."""
    print("\n2. events_reviewed, v_event_density_reviewed, event_density, CSV:")
    with Workdir() as tmp:
        dataset, annot, db = _recording(tmp)
        _detect(dataset, annot, db)
        uuids = _uuids(db)
        rejected, accepted = uuids[0], uuids[1]
        ch = _event(db, rejected)[2]

        before = _density_by_channel(db)
        conn = dbwrite.open_write_connection(db)
        dbwrite.store_event_review(conn, rejected, 'reject', 'alice',
                                   reason='artefact')
        dbwrite.store_event_review(conn, rejected, 'accept', 'bob')
        dbwrite.store_event_review(conn, accepted, 'accept', 'alice')
        dbwrite.store_event_review(conn, uuids[2], 'unsure', 'alice')

        n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_view = conn.execute(
            "SELECT COUNT(*) FROM events_reviewed").fetchone()[0]
        assert n_view == n_events - 1, (n_view, n_events)
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed WHERE uuid = ?",
                            (rejected,)).fetchone()[0] == 0
        summ = conn.execute(
            "SELECT n_reviews, n_accept, n_reject, n_unsure FROM events_reviewed "
            "WHERE uuid = ?", (accepted,)).fetchone()
        assert summ == (1, 1, 0, 0), summ
        v_all = dict(conn.execute(
            "SELECT channel, n_events FROM v_event_density WHERE channel = ?",
            (ch,)).fetchall())
        v_rev = dict(conn.execute(
            "SELECT channel, n_events FROM v_event_density_reviewed "
            "WHERE channel = ?", (ch,)).fetchall())
        assert v_rev[ch] == v_all[ch] - 1, (v_all, v_rev)
        conn.close()

        default = _density_by_channel(db)
        assert default == before, (default, before)
        excl = _density_by_channel(db, exclude_rejected=True)
        assert excl[ch][0] == before[ch][0] - 1, (excl, before)
        assert excl[ch][1] < before[ch][1]
        for other in set(before) - {ch}:
            assert excl[other] == before[other], other
        # bob accepted it: by bob's rejections alone, nothing is excluded
        assert _density_by_channel(db, exclude_rejected=True,
                                   reviewer='bob') == before
        assert _density_by_channel(db, exclude_rejected=True,
                                   reviewer='alice') == excl
        try:
            _density_by_channel(db, reviewer='alice')
        except ValueError:
            pass
        else:
            raise AssertionError("reviewer without exclude_rejected accepted")

        rej_start = _event(db, rejected)[3]
        all_csv = dbwrite.export_events_to_csv(
            db, 'spindle', 'Moelle2011', BAND, ['NREM2'],
            csv_file=os.path.join(tmp, 'all.csv'))
        rev_csv = dbwrite.export_events_to_csv(
            db, 'spindle', 'Moelle2011', BAND, ['NREM2'],
            csv_file=os.path.join(tmp, 'rev.csv'), exclude_rejected=True)
        s_all, s_rev = _csv_starts(all_csv), _csv_starts(rev_csv)
        assert len(s_rev) == len(s_all) - 1, (len(s_all), len(s_rev))
        # The three fixture channels share the bursts, so the same start can
        # occur on several channels: exactly one occurrence must go.
        n_at = [sum(abs(s - rej_start) < 1e-6 for s in starts)
                for starts in (s_all, s_rev)]
        assert n_at[0] >= 1 and n_at[1] == n_at[0] - 1, n_at
    print(f"   [ok] {ch}: view, reviewed density view, event_density and CSV "
          f"drop exactly the rejected event ({before[ch][0]} -> "
          f"{excl[ch][0]}); defaults unchanged; reviewer= narrows")


def test_all_rejected_export_returns_none():
    """A scope emptied by rejections writes no CSV and does not raise."""
    print("\n3. Every row rejected:")
    with Workdir() as tmp:
        dataset, annot, db = _recording(tmp)
        _detect(dataset, annot, db)
        conn = dbwrite.open_write_connection(db)
        for u in _uuids(db):
            dbwrite.store_event_review(conn, u, 'reject', 'alice',
                                       reason='other', comment='test',
                                       commit=False)
        conn.commit()
        conn.close()
        out = dbwrite.export_events_to_csv(
            db, 'spindle', 'Moelle2011', BAND, ['NREM2'],
            csv_file=os.path.join(tmp, 'none.csv'), exclude_rejected=True)
        assert out is None and not os.path.exists(os.path.join(tmp, 'none.csv'))
        df = event_density(db, event_type='spindle', method='Moelle2011',
                           stage=['NREM2'], exclude_rejected=True)
        assert set(df['channel']) == set(LABELS), df
        assert (df['n_events'] == 0).all(), df[['channel', 'n_events']]
    print("   [ok] export returns None; density keeps a zero row per channel")


# -------------------------------------------------------------- re-detection

def test_redetect_keeps_review():
    """Same parameters reproduce the uuid; the review stays attached."""
    print("\n4. Same-parameter re-detection:")
    with Workdir() as tmp:
        dataset, annot, db = _recording(tmp)
        _detect(dataset, annot, db)
        first = _uuids(db)
        target = first[0]
        conn = dbwrite.open_write_connection(db)
        dbwrite.store_event_review(conn, target, 'reject', 'alice',
                                   reason='artefact')
        conn.close()

        _detect(dataset, annot, db)                       # INSERT OR REPLACE
        assert _uuids(db) == first
        _detect(dataset, annot, db, replace_channels=LABELS)  # scoped DELETE
        assert _uuids(db) == first
        df = dbwrite.read_event_reviews(db)
        assert list(df['uuid']) == [target] and not df['orphaned'].any(), df
        conn = sqlite3.connect(db)
        n_runs = conn.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0]
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed WHERE uuid = ?",
                            (target,)).fetchone()[0] == 0
        conn.close()
        assert n_runs == 3, n_runs
    print(f"   [ok] {len(first)} uuids identical over 3 runs (plain, replace); "
          f"review not orphaned, still excluded")


def test_band_change_orphans_and_rematch():
    """Old-band rows gone -> orphaned; rematch proposes the new-band event."""
    print("\n5. Band change, orphaning and rematch:")
    with Workdir() as tmp:
        dataset, annot, db = _recording(tmp)
        _detect(dataset, annot, db)
        target = _uuids(db)[0]
        _, _, ch, start, _ = _event(db, target)
        conn = dbwrite.open_write_connection(db)
        dbwrite.store_event_review(conn, target, 'reject', 'alice',
                                   reason='artefact', comment='first look')
        conn.close()

        _detect(dataset, annot, db, frequency=OTHER_BAND)
        new_uuids = _uuids(db, OTHER_BAND)
        assert target not in new_uuids
        # a band change alone leaves the old rows, so nothing is orphaned yet
        assert not dbwrite.read_event_reviews(db, include_orphaned=True)[
            'orphaned'].any()

        # the superseded run is removed -> the review is orphaned
        conn = dbwrite.open_write_connection(db)
        conn.execute("DELETE FROM events WHERE freq_lower = ? AND freq_upper = ?",
                     BAND)
        conn.commit()
        assert dbwrite.read_event_reviews(db).empty
        orph = dbwrite.read_event_reviews(db, include_orphaned=True)
        assert list(orph['uuid']) == [target] and orph['orphaned'].all()
        assert orph.iloc[0]['channel'] == ch and orph.iloc[0]['start_time'] == start

        assert orph.iloc[0]['method'] == 'Moelle2011'
        assert (orph.iloc[0]['freq_lower'], orph.iloc[0]['freq_upper']) == BAND

        nearest = conn.execute(
            "SELECT uuid, start_time, freq_lower, freq_upper FROM events "
            "WHERE channel = ? ORDER BY ABS(start_time - ?) LIMIT 1",
            (ch, start)).fetchone()
        assert (nearest[2], nearest[3]) == OTHER_BAND
        dt = nearest[1] - start
        tol = max(0.1, abs(dt) + 0.05)
        prop = dbwrite.rematch_orphaned_reviews(conn, tolerance_s=tol)
        assert len(prop) == 1, prop
        p = prop.iloc[0]
        # a band change is reported, with the nearest event shown, never proposed
        assert p['status'] == 'band_changed', p.to_dict()
        assert p['new_uuid'] == nearest[0] and p['n_candidates'] == 0
        assert p['old_method'] == 'Moelle2011'
        if abs(dt) > 1e-3:
            far = dbwrite.rematch_orphaned_reviews(
                conn, tolerance_s=abs(dt) / 2)
            assert far.iloc[0]['status'] == 'no_match', far.to_dict('records')

        applied = dbwrite.rematch_orphaned_reviews(conn, tolerance_s=tol,
                                                   dry_run=False)
        assert not applied['applied'].any()
        after = dbwrite.read_event_reviews(conn, include_orphaned=True)
        conn.close()
        assert list(after['uuid']) == [target] and after['orphaned'].all()
        assert after.iloc[0]['comment'] == 'first look'
    print(f"   [ok] orphan reported with its own method and band; nearest "
          f"12-15 Hz event (dt={dt:+.3f} s) is 'band_changed'; applying "
          f"writes nothing")


def test_shifted_replace_orphans():
    """A replace that moves an event's start re-keys it and orphans its review."""
    print("\n6. write_channel_events replace with a shifted start:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'neural_events.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute(LEGACY_STATUS_DDL)
        conn.commit()
        dbwrite.ensure_direct_write_schema(conn)

        def _events(starts):
            return [{'uuid': dbwrite.event_uuid5('spindle', 'Cz', s, 'Moelle2011',
                                                  11, 16, 'NREM2'),
                     'start_time': s, 'end_time': s + 0.8, 'duration': 0.8,
                     'stage': 'NREM2', 'epoch_stage': 'NREM2',
                     'method': 'Moelle2011'} for s in starts]

        def _write(starts, replace):
            dbwrite.write_channel_events(
                conn, 'run-x', 'spindle', 'Cz', 'Moelle2011', 11, 16, 'NREM2',
                _events(starts), [], None, None, replace=replace,
                carry_reviews=False)   # the manual rematch is under test

        _write([100.0, 200.0, 300.0], replace=False)
        old = dbwrite.event_uuid5('spindle', 'Cz', 200.0, 'Moelle2011', 11, 16,
                                  'NREM2')
        dbwrite.store_event_review(conn, old, 'accept', 'alice')
        _write([100.0, 200.0, 300.0], replace=True)       # identical rows
        assert not dbwrite.read_event_reviews(conn)['orphaned'].any()
        assert len(dbwrite.read_event_reviews(conn)) == 1
        _write([100.0, 200.04, 300.0], replace=True)      # start moved 40 ms
        orph = dbwrite.read_event_reviews(conn, include_orphaned=True)
        assert orph['orphaned'].all() and dbwrite.read_event_reviews(conn).empty
        prop = dbwrite.rematch_orphaned_reviews(conn)
        assert prop.iloc[0]['status'] == 'proposed'
        assert abs(prop.iloc[0]['dt'] - 0.04) < 1e-9
        new = dbwrite.event_uuid5('spindle', 'Cz', 200.04, 'Moelle2011', 11, 16,
                                  'NREM2')
        assert prop.iloc[0]['new_uuid'] == new
        assert len(dbwrite.read_event_reviews(conn)) == 0     # dry run
        done = dbwrite.rematch_orphaned_reviews(conn, dry_run=False)
        assert bool(done.iloc[0]['applied'])
        after = dbwrite.read_event_reviews(conn)
        assert list(after['uuid']) == [new]
        assert after.iloc[0]['comment'].startswith('[rematched from ' + old)
        assert after.iloc[0]['run_id'] == 'run-x'
        conn.close()
    print("   [ok] identical rows keep the review; a 40 ms shift orphans it; "
          "rematch proposes the shifted event and applying re-keys it")


# ------------------------------------------------------------------ migration

LEGACY_EVENTS_DDL = '''
CREATE TABLE events (
    uuid TEXT PRIMARY KEY, event_type TEXT, channel TEXT,
    start_time REAL, end_time REAL, duration REAL, start_time_hms TEXT,
    stage TEXT, cycle TEXT, method TEXT, freq_band TEXT,
    freq_lower REAL, freq_upper REAL, min_amp REAL, max_amp REAL,
    peak2peak_amp REAL, rms REAL, power REAL, peak_power_freq REAL,
    energy REAL, peak_energy_freq REAL, processing_timestamp TEXT,
    n_fft_sec INTEGER,
    CONSTRAINT event_chan_time UNIQUE (event_type, channel, start_time,
                                       method, freq_lower, freq_upper, stage)
)'''


LEGACY_STATUS_DDL = '''
CREATE TABLE processing_status (
    channel TEXT NOT NULL, event_type TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT '', freq_lower REAL NOT NULL DEFAULT 0,
    freq_upper REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT '',
    json_file TEXT, processed BOOLEAN DEFAULT 0, attempts INTEGER DEFAULT 0,
    last_attempt_time TEXT, success BOOLEAN DEFAULT 0, error_message TEXT,
    PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage)
)'''


def test_old_database_gains_table():
    """A pre-4.6 database: readers cope, the schema step adds table + views."""
    print("\n7. Existing database without event_reviews:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'old.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time, "
                     "end_time, duration, stage, method, freq_lower, freq_upper) "
                     "VALUES ('u1', 'spindle', 'Cz', 10.0, 10.8, 0.8, 'NREM2', "
                     "'Moelle2011', 11, 16)")
        conn.commit()
        conn.close()

        # readers on the untouched file
        assert dbwrite.read_event_reviews(db, include_orphaned=True).empty
        conn = sqlite3.connect(db)
        assert dbwrite.review_exclusion_clause(conn) == (None, [])
        conn.close()

        conn = sqlite3.connect(db)
        dbwrite.ensure_direct_write_schema(conn)
        names = {r[0]: r[1] for r in conn.execute(
            "SELECT name, type FROM sqlite_master")}
        assert names.get('event_reviews') == 'table', names
        assert names.get('events_reviewed') == 'view', names
        assert names.get('v_event_density_reviewed') == 'view', names
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 1
        # idempotent
        dbwrite.ensure_direct_write_schema(conn)
        assert dbwrite.ensure_event_reviews_schema(conn) is False
        ddl = conn.execute("SELECT sql FROM sqlite_master "
                           "WHERE name = 'event_reviews'").fetchone()[0]
        conn.close()
        assert 'PRIMARY KEY (uuid, reviewer)' in ddl
    print("   [ok] readers return empty / no clause; schema step adds the table "
          "and both views; second call is a no-op")


def test_delete_undo_and_blind_view():
    """Clear one reviewer's decision, restore it with its timestamp, read
    one reviewer's rows through the reviewer index."""
    print("\n8. delete_event_review, undo with reviewed_at, blind view:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'neural_events.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        for i, u in enumerate(('u1', 'u2', 'u3')):
            conn.execute(
                "INSERT INTO events (uuid, event_type, channel, start_time, "
                "end_time, duration, stage, method, freq_lower, freq_upper) "
                "VALUES (?, 'spindle', 'Cz', ?, ?, 0.8, 'NREM2', 'Moelle2011', "
                "11, 16)", (u, 10.0 * (i + 1), 10.0 * (i + 1) + 0.8))
        conn.commit()
        dbwrite.ensure_direct_write_schema(conn)

        first = dbwrite.store_event_review(conn, 'u1', 'reject', 'alice',
                                           reason='artefact')
        dbwrite.store_event_review(conn, 'u1', 'accept', 'bob')
        dbwrite.store_event_review(conn, 'u2', 'unsure', 'alice')
        dbwrite.store_event_review(conn, 'u3', 'accept', 'bob')

        # undo of a changed decision: alice re-decides, then restores
        dbwrite.store_event_review(conn, 'u1', 'accept', 'alice')
        restored = dbwrite.store_event_review(
            conn, 'u1', first['decision'], 'alice', reason=first['reason'],
            reviewed_at=first['reviewed_at'])
        assert restored['reviewed_at'] == first['reviewed_at']
        got = dbwrite.read_event_reviews(conn, uuid='u1', reviewer='alice')
        assert got.iloc[0]['reviewed_at'] == first['reviewed_at']
        assert got.iloc[0]['decision'] == 'reject'

        # clear: only alice's row on u1 goes
        assert dbwrite.delete_event_review(conn, 'u1', 'alice') is True
        assert dbwrite.delete_event_review(conn, 'u1', 'alice') is False
        assert dbwrite.delete_event_review(conn, 'nope', 'alice') is False
        left = dbwrite.read_event_reviews(conn)
        assert sorted(zip(left['uuid'], left['reviewer'])) == [
            ('u1', 'bob'), ('u2', 'alice'), ('u3', 'bob')], left
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 3

        # blind view: alice's rows only, through the reviewer index
        blind = dbwrite.read_event_reviews(conn, reviewer='alice')
        assert list(blind['uuid']) == ['u2'], blind
        plan = ' '.join(r[-1] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT uuid FROM event_reviews "
            "WHERE reviewer IN (?)", ('alice',)))
        assert 'idx_event_reviews_reviewer' in plan, plan
        conn.close()

        # a database without the table: delete is a no-op, not an error
        bare = os.path.join(tmp, 'bare.db')
        c2 = sqlite3.connect(bare)
        assert dbwrite.delete_event_review(c2, 'u1', 'alice') is False
        c2.close()
    print(f"   [ok] undo restored reviewed_at={first['reviewed_at']}; delete "
          f"removed only alice's row and reports False on a second call; "
          f"blind read uses idx_event_reviews_reviewer")


def test_store_on_database_without_runs_subject():
    """A pre-subject detection_runs table: the review is stored, subject NULL."""
    print("\n9. detection_runs without a subject column:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'old.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute("ALTER TABLE events ADD COLUMN run_id TEXT")
        conn.execute("CREATE TABLE detection_runs (run_id TEXT PRIMARY KEY, "
                     "event_type TEXT, method TEXT)")
        conn.execute("INSERT INTO detection_runs VALUES ('r1', 'spindle', "
                     "'Moelle2011')")
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time, "
                     "end_time, duration, stage, method, freq_lower, freq_upper, "
                     "run_id) VALUES ('u1', 'spindle', 'Cz', 10.0, 10.8, 0.8, "
                     "'NREM2', 'Moelle2011', 11, 16, 'r1')")
        conn.commit()
        # no ensure_direct_write_schema: the GUI path
        row = dbwrite.store_event_review(conn, 'u1', 'accept', 'alice')
        assert row['run_id'] == 'r1' and row['subject'] is None, row
        assert row['channel'] == 'Cz' and row['start_time'] == 10.0
        assert 'subject' not in [r[1] for r in conn.execute(
            "PRAGMA table_info(detection_runs)")]
        got = dbwrite.read_event_reviews(conn)
        assert list(got['uuid']) == ['u1'], got
        # and with no detection_runs table at all
        conn.execute("DROP TABLE detection_runs")
        row = dbwrite.store_event_review(conn, 'u1', 'reject', 'bob',
                                         reason='other', comment='noise')
        assert row['run_id'] == 'r1' and row['subject'] is None, row
        conn.close()
    print("   [ok] stored with run_id='r1', subject NULL; detection_runs left "
          "unmigrated; also without the table")


def test_vocabulary_and_categories():
    """Method Spec section 10 vocabulary and the RA-protocol mapping."""
    print("\n10. Reason vocabulary and review_category:")
    assert dbwrite.REVIEW_REASONS == (
        'artefact', 'eye-movement', 'not-in-raw', 'filter-ringing', 'off-band',
        'too-short', 'arousal', 'single-channel', 'not-isolated',
        'wrong-morphology', 'other')
    assert dbwrite.REVIEW_DECISIONS == ('accept', 'reject', 'unsure')
    cat = dbwrite.review_category
    for r in (None,) + dbwrite.REVIEW_REASONS:
        assert cat('accept', r) == 'TP'
        assert cat('unsure', r) == 'Ambiguous'
    assert cat('reject', 'artefact') == 'FP-artifact'
    assert cat('reject', 'eye-movement') == 'FP-artifact'
    for r in ('not-in-raw', 'filter-ringing', 'off-band', 'too-short',
              'arousal', 'single-channel', 'not-isolated', 'wrong-morphology',
              'other', None):
        assert cat('reject', r) == 'FP-other', r
    assert set(dbwrite.REVIEW_REASON_CATEGORY) == set(dbwrite.REVIEW_REASONS)
    for bad in (('maybe', None), ('reject', 'arousal-alpha')):
        try:
            cat(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"review_category accepted {bad}")
    # every reason passes the CHECK; 'other' with a comment is stored
    with Workdir() as tmp:
        db = os.path.join(tmp, 'v.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time) "
                     "VALUES ('u1', 'spindle', 'Cz', 1.0)")
        for i, r in enumerate(dbwrite.REVIEW_REASONS):
            dbwrite.store_event_review(conn, 'u1', 'reject', f'rater{i}',
                                       reason=r, comment='why')
        n = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
        conn.close()
        assert n == len(dbwrite.REVIEW_REASONS)
    print(f"   [ok] {len(dbwrite.REVIEW_REASONS)} reasons pass the CHECK; "
          f"accept->TP, unsure->Ambiguous, reject artefact/eye-movement->"
          f"FP-artifact, every other reject->FP-other")


def test_stale_check_is_rebuilt():
    """A table under the pre-spec CHECK is rebuilt, rows kept and mapped."""
    print("\n11. event_reviews with the old reason CHECK:")
    old_reasons = ("'artefact', 'arousal-alpha', 'too-short', 'filter-ringing', "
                   "'eye-movement', 'not-in-raw', 'off-band', 'single-channel', "
                   "'other'")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'stale.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute(LEGACY_STATUS_DDL)
        for u, t in (('u1', 1.0), ('u2', 2.0)):
            conn.execute("INSERT INTO events (uuid, event_type, channel, "
                         "start_time, stage, method, freq_lower, freq_upper) "
                         "VALUES (?, 'spindle', 'Cz', ?, 'NREM2', 'Moelle2011', "
                         "11, 16)", (u, t))
        conn.execute(f'''CREATE TABLE event_reviews (
            uuid TEXT NOT NULL, reviewer TEXT NOT NULL, run_id TEXT,
            subject TEXT, event_type TEXT, channel TEXT, start_time REAL,
            decision TEXT NOT NULL
                CHECK (decision IN ('accept', 'reject', 'unsure')),
            reason TEXT CHECK (reason IS NULL OR reason IN ({old_reasons})),
            comment TEXT, reviewed_at TEXT, turtlewave_version TEXT,
            PRIMARY KEY (uuid, reviewer))''')
        conn.execute("INSERT INTO event_reviews (uuid, reviewer, decision, "
                     "reason, reviewed_at) VALUES ('u1', 'alice', 'reject', "
                     "'arousal-alpha', '2026-09-30T10:00:00+10:00')")
        conn.execute("INSERT INTO event_reviews (uuid, reviewer, decision, "
                     "reason) VALUES ('u2', 'alice', 'reject', 'too-short')")
        conn.commit()
        # the old CHECK refuses a new reason
        try:
            conn.execute("INSERT INTO event_reviews (uuid, reviewer, decision, "
                         "reason) VALUES ('u2', 'bob', 'reject', 'not-isolated')")
        except sqlite3.IntegrityError:
            conn.rollback()
        else:
            raise AssertionError("fixture's old CHECK accepted a new reason")

        dbwrite.ensure_direct_write_schema(conn)   # rebuild, then views
        got = dbwrite.read_event_reviews(conn)
        rows = {r.uuid: (r.reason, r.reviewed_at) for r in got.itertuples()}
        assert rows == {'u1': ('arousal', '2026-09-30T10:00:00+10:00'),
                        'u2': ('too-short', None)}, rows
        dbwrite.store_event_review(conn, 'u2', 'reject', 'bob',
                                   reason='not-isolated')
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 0
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert {'events_reviewed', 'v_event_density_reviewed',
                'idx_event_reviews_reviewer'} <= names, names
        assert 'event_reviews_new' not in names
        # a second call finds the table current and changes nothing
        ddl = conn.execute("SELECT sql FROM sqlite_master "
                           "WHERE name='event_reviews'").fetchone()[0]
        dbwrite.ensure_event_reviews_schema(conn)
        assert conn.execute("SELECT sql FROM sqlite_master "
                            "WHERE name='event_reviews'").fetchone()[0] == ddl
        assert "'wrong-morphology'" in ddl and "'arousal-alpha'" not in ddl
        conn.close()
    print("   [ok] rebuilt under the new CHECK: 2 rows kept, 'arousal-alpha' -> "
          "'arousal', reviewed_at kept; views and index back; second call "
          "a no-op")


def test_stale_check_rebuilt_with_views_present():
    """The rebuild drops and restores the views that read the table."""
    print("\n12. Stale table while events_reviewed exists:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'stale2.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute(LEGACY_STATUS_DDL)
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time, "
                     "stage, method, freq_lower, freq_upper) VALUES ('u1', "
                     "'spindle', 'Cz', 1.0, 'NREM2', 'Moelle2011', 11, 16)")
        dbwrite.ensure_direct_write_schema(conn)
        dbwrite.store_event_review(conn, 'u1', 'reject', 'alice',
                                   reason='artefact')
        # swap in a table with a narrower CHECK, then put the two views
        # back over it verbatim, as a GUI-created 4.6 pre-release DB has them
        views = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='view' AND name IN "
            "('events_reviewed', 'v_event_density_reviewed') "
            "ORDER BY name = 'v_event_density_reviewed'").fetchall()
        assert [v[0] for v in views] == ['events_reviewed',
                                         'v_event_density_reviewed'], views
        conn.execute("DROP VIEW v_event_density_reviewed")
        conn.execute("DROP VIEW events_reviewed")
        conn.execute("DROP TABLE event_reviews")
        stale = dbwrite._event_reviews_ddl('event_reviews').replace(
            "'not-isolated', 'wrong-morphology', ", "")
        assert stale != dbwrite._event_reviews_ddl('event_reviews')
        conn.execute(stale)
        conn.execute("INSERT INTO event_reviews (uuid, reviewer, decision, "
                     "reason) VALUES ('u1', 'alice', 'reject', 'artefact')")
        for _name, sql in views:
            conn.execute(sql)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 0

        assert dbwrite.ensure_event_reviews_schema(conn) is False
        ddl = conn.execute("SELECT sql FROM sqlite_master "
                           "WHERE name='event_reviews'").fetchone()[0]
        assert "'wrong-morphology'" in ddl, "stale table was not rebuilt"
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'")}
        assert {'events_reviewed', 'v_event_density_reviewed'} <= names, names
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 0
        assert len(dbwrite.read_event_reviews(conn)) == 1
        conn.close()
    print("   [ok] stale table under live views rebuilt; both views restored, "
          "row kept, rejection still applied")


def _fresh_db(tmp, name='neural_events.db'):
    """Empty database with the legacy events/status DDL, schema ensured."""
    conn = sqlite3.connect(os.path.join(tmp, name))
    conn.execute(LEGACY_EVENTS_DDL)
    conn.execute(LEGACY_STATUS_DDL)
    conn.commit()
    dbwrite.ensure_direct_write_schema(conn)
    return conn


def _put(conn, rows, run_method, replace=False, replace_methods=None,
         band=(0.5, 4.0), event_type='slow_wave', channel='Cz',
         run_id='run-1'):
    """Write ``rows`` = [(start, per-event method), ...] for one channel."""
    evs = [{'uuid': dbwrite.event_uuid5(event_type, channel, s, m, band[0],
                                        band[1], 'NREM2'),
            'start_time': s, 'end_time': s + 0.8, 'duration': 0.8,
            'stage': 'NREM2', 'epoch_stage': 'NREM2', 'method': m}
           for s, m in rows]
    dbwrite.write_channel_events(
        conn, run_id, event_type, channel, run_method, band[0], band[1],
        'NREM2', evs, [], None, None, replace=replace,
        replace_methods=replace_methods,
        carry_reviews=False)   # the manual rematch is under test
    return [e['uuid'] for e in evs]


def test_rematch_multi_method_run():
    """A Massimini2004 review is never moved onto an Ngo2015 event."""
    print("\n13. Rematch inside a Massimini2004_Ngo2015 run:")
    with Workdir() as tmp:
        conn = _fresh_db(tmp)
        both = ['Massimini2004', 'Ngo2015']
        (mass, _ngo) = _put(conn, [(100.0, 'Massimini2004'),
                                   (100.0, 'Ngo2015')],
                            'Massimini2004_Ngo2015', replace_methods=both)
        dbwrite.store_event_review(conn, mass, 'reject', 'alice',
                                   reason='artefact')
        row = dbwrite.read_event_reviews(conn).iloc[0]
        assert row['method'] == 'Massimini2004', row.to_dict()

        # re-run: only Ngo2015 near 100 s -> method_changed, never applied
        _put(conn, [(100.01, 'Ngo2015')], 'Massimini2004_Ngo2015',
             replace=True, replace_methods=both, run_id='run-2')
        prop = dbwrite.rematch_orphaned_reviews(conn)
        p = prop.iloc[0]
        assert p['status'] == 'method_changed', p.to_dict()
        assert p['new_method'] == 'Ngo2015' and p['n_candidates'] == 0
        done = dbwrite.rematch_orphaned_reviews(conn, dry_run=False)
        assert not done['applied'].any()
        assert dbwrite.read_event_reviews(conn).empty

        # re-run: Ngo2015 nearer, Massimini2004 a little further -> the
        # Massimini2004 event is proposed, the nearer Ngo2015 one ignored
        (ngo2, mass2) = _put(conn, [(100.01, 'Ngo2015'),
                                    (100.03, 'Massimini2004')],
                             'Massimini2004_Ngo2015', replace=True,
                             replace_methods=both, run_id='run-3')
        p = dbwrite.rematch_orphaned_reviews(conn).iloc[0]
        assert p['status'] == 'proposed' and p['new_uuid'] == mass2, p.to_dict()
        conn.close()
    print("   [ok] Ngo2015 alone -> 'method_changed', not applied; with a "
          "Massimini2004 event further away, that one is proposed")


def test_rematch_aasm_spellings():
    """AASM/Massimini2004 matches its escaped spelling, not Massimini2004."""
    print("\n14. Rematch across AASM/Massimini2004 spellings:")
    with Workdir() as tmp:
        conn = _fresh_db(tmp)
        aasm = ['AASM/Massimini2004']
        (old,) = _put(conn, [(50.0, 'AASM/Massimini2004')],
                      'AASM/Massimini2004', replace_methods=aasm)
        dbwrite.store_event_review(conn, old, 'accept', 'alice')
        # replaced by a plain Massimini2004 detection: not the same method
        _put(conn, [(50.01, 'Massimini2004')], 'Massimini2004', replace=True,
             replace_methods=aasm + ['Massimini2004'], run_id='run-2')
        p = dbwrite.rematch_orphaned_reviews(conn).iloc[0]
        assert p['status'] == 'method_changed', p.to_dict()
        # the 4.0.x escaped spelling of the same method: proposed
        (esc,) = _put(conn, [(50.02, 'AASM_Massimini2004')],
                      'AASM_Massimini2004', replace=True,
                      replace_methods=aasm + ['Massimini2004'], run_id='run-3')
        p = dbwrite.rematch_orphaned_reviews(conn).iloc[0]
        assert p['status'] == 'proposed' and p['new_uuid'] == esc, p.to_dict()
        conn.close()
    print("   [ok] AASM/Massimini2004 -> Massimini2004 is 'method_changed'; "
          "-> AASM_Massimini2004 is proposed")


def test_rematch_duplicate_target_demoted():
    """Two orphans of one reviewer, one successor: both 'conflict'."""
    print("\n15. Duplicate target demotion:")
    with Workdir() as tmp:
        conn = _fresh_db(tmp)
        u_a, u_b = _put(conn, [(200.0, 'Massimini2004'),
                               (200.05, 'Massimini2004')], 'Massimini2004',
                        replace_methods=['Massimini2004'])
        dbwrite.store_event_review(conn, u_a, 'accept', 'alice')
        dbwrite.store_event_review(conn, u_b, 'reject', 'alice',
                                   reason='artefact')
        dbwrite.store_event_review(conn, u_b, 'accept', 'bob')
        (merged,) = _put(conn, [(200.02, 'Massimini2004')], 'Massimini2004',
                         replace=True, replace_methods=['Massimini2004'],
                         run_id='run-2')
        prop = dbwrite.rematch_orphaned_reviews(conn)
        by = {(r.old_uuid, r.reviewer): r.status for r in prop.itertuples()}
        assert by == {(u_a, 'alice'): 'conflict', (u_b, 'alice'): 'conflict',
                      (u_b, 'bob'): 'proposed'}, by
        done = dbwrite.rematch_orphaned_reviews(conn, dry_run=False)
        assert int(done['applied'].sum()) == 1
        live = dbwrite.read_event_reviews(conn)
        assert list(zip(live['uuid'], live['reviewer'])) == [(merged, 'bob')]
        assert len(dbwrite.read_event_reviews(conn, include_orphaned=True)) == 3
        conn.close()
    print("   [ok] alice's two orphans -> 'conflict' (not applied); bob's "
          "single orphan proposed and applied")


def test_commit_false_is_rollbackable():
    """store_event_review(commit=False) leaves the transaction open."""
    print("\n16. commit=False then rollback:")
    with Workdir() as tmp:
        conn = _fresh_db(tmp)
        u1, u2 = _put(conn, [(10.0, 'Massimini2004'), (20.0, 'Massimini2004')],
                      'Massimini2004', replace_methods=['Massimini2004'])
        dbwrite.store_event_review(conn, u1, 'accept', 'alice', commit=False)
        dbwrite.store_event_review(conn, u2, 'reject', 'alice',
                                   reason='artefact', commit=False)
        assert conn.in_transaction
        conn.rollback()
        n = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
        assert n == 0, n
        # and commit=False followed by commit keeps both
        dbwrite.store_event_review(conn, u1, 'accept', 'alice', commit=False)
        dbwrite.store_event_review(conn, u2, 'accept', 'alice', commit=False)
        conn.commit()
        assert conn.execute(
            "SELECT COUNT(*) FROM event_reviews").fetchone()[0] == 2
        conn.close()
    print("   [ok] two commit=False stores + rollback -> 0 rows; + commit -> 2")


def test_gui_path_creates_views():
    """ensure_event_reviews_schema alone gives a GUI database the views."""
    print("\n17. Schema step alone creates the views:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'gui.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time, "
                     "stage, method, freq_lower, freq_upper) VALUES ('u1', "
                     "'spindle', 'Cz', 1.0, 'NREM2', 'Moelle2011', 11, 16)")
        conn.commit()
        assert dbwrite.ensure_event_reviews_schema(conn) is True
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'")}
        assert {'events_reviewed', 'v_event_density_reviewed'} <= names, names
        assert 'v_event_density' not in names   # not this step's view
        assert not dbwrite._table_columns(conn, 'db_meta')
        assert dbwrite.ensure_event_reviews_schema(conn) is False
        conn.close()
    print("   [ok] events_reviewed + v_event_density_reviewed created; no "
          "db_meta stamp; second call a no-op")


def test_rebuild_maps_unknown_reason():
    """An unmapped reason becomes NULL with the value kept in the comment."""
    print("\n18. Rebuild with a reason outside every vocabulary:")
    with Workdir() as tmp:
        db = os.path.join(tmp, 'odd.db')
        conn = sqlite3.connect(db)
        conn.execute(LEGACY_EVENTS_DDL)
        conn.execute(LEGACY_STATUS_DDL)
        conn.execute("INSERT INTO events (uuid, event_type, channel, start_time, "
                     "stage, method, freq_lower, freq_upper) VALUES ('u1', "
                     "'spindle', 'Cz', 1.0, 'NREM2', 'Moelle2011', 11, 16)")
        # a table with no CHECK at all, holding values no vocabulary knows
        conn.execute("CREATE TABLE event_reviews (uuid TEXT NOT NULL, "
                     "reviewer TEXT NOT NULL, run_id TEXT, subject TEXT, "
                     "event_type TEXT, channel TEXT, start_time REAL, "
                     "decision TEXT NOT NULL, reason TEXT, comment TEXT, "
                     "reviewed_at TEXT, turtlewave_version TEXT, "
                     "PRIMARY KEY (uuid, reviewer))")
        conn.executemany(
            "INSERT INTO event_reviews (uuid, reviewer, decision, reason, "
            "comment) VALUES (?, ?, ?, ?, ?)",
            [('u1', 'alice', 'reject', 'spooky', 'looked odd'),
             ('u1', 'bob', 'maybe', 'arousal-alpha', None),
             ('u1', 'carol', 'reject', 'artefact', None)])
        conn.commit()
        log = logging.getLogger('tw_reviews_test')
        records = []

        class _H(logging.Handler):
            def emit(self, record):
                records.append(record)
        h = _H(level=logging.DEBUG)
        log.addHandler(h)
        log.setLevel(logging.DEBUG)
        try:
            dbwrite.ensure_direct_write_schema(conn, logger=log)  # must not raise
        finally:
            log.removeHandler(h)
        got = {r.reviewer: (r.decision, r.reason, r.comment)
               for r in dbwrite.read_event_reviews(conn).itertuples()}
        assert got == {
            'alice': ('reject', None, "looked odd [former reason 'spooky']"),
            'bob': ('unsure', 'arousal', "[former decision 'maybe']"),
            'carol': ('reject', 'artefact', None)}, got
        warns = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        assert any("'spooky'" in m for m in warns), warns
        assert any("'maybe'" in m for m in warns), warns
        cols = dbwrite._table_columns(conn, 'event_reviews')
        assert {'method', 'freq_lower', 'freq_upper'} <= set(cols), cols
        assert conn.execute("SELECT COUNT(*) FROM events_reviewed").fetchone()[0] == 0
        conn.close()
    print("   [ok] no raise; 'spooky' -> NULL + comment, 'maybe' -> 'unsure' + "
          "comment, 'arousal-alpha' -> 'arousal'; WARNINGs logged; method/band "
          "columns added")


TESTS = [
    test_round_trip_and_checks,
    test_exclusion_view_density_export,
    test_all_rejected_export_returns_none,
    test_redetect_keeps_review,
    test_band_change_orphans_and_rematch,
    test_shifted_replace_orphans,
    test_old_database_gains_table,
    test_delete_undo_and_blind_view,
    test_store_on_database_without_runs_subject,
    test_vocabulary_and_categories,
    test_stale_check_is_rebuilt,
    test_stale_check_rebuilt_with_views_present,
    test_rematch_multi_method_run,
    test_rematch_aasm_spellings,
    test_rematch_duplicate_target_demoted,
    test_commit_false_is_rollbackable,
    test_gui_path_creates_views,
    test_rebuild_maps_unknown_reason,
]


if __name__ == "__main__":
    print("TESTING event reviews")
    print("=====================")
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
