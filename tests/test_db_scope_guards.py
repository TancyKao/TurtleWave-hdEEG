#!/usr/bin/env python3
"""Coverage reporting, CSV-path status scope, and the method-spelling guard.

Four defects that a batch driver meets on its worst day, all of them silent or
misleading rather than loud:

* :func:`turtlewave_hdEEG.dbwrite.verify_channel_coverage` built its result
  dict in two places, so the early "no database" return lacked ``'failed'``
  and ``'events_only'``. Both cluster drivers dereference those keys
  unconditionally, so a typo'd ``--root`` produced a ``KeyError`` instead of
  the diagnostic the check exists to print. An existing-but-empty database was
  worse: it raised ``no such table: events`` from inside the check.
* The CSV import path wrote ``processing_status`` rows WITHOUT the run's
  method/band/stage, so they landed on the schema defaults and matched nothing
  in the scoped coverage query. A channel that legitimately detected zero
  events was therefore reported missing, and the drivers exited 1 on a
  successful run -- every run, with no way to self-heal.
* ``events.method`` changed spelling. Up to 4.0.2 the slow-wave processor
  stored the filename-escaped ``'AASM_Massimini2004'``; from 4.3 it stores
  ``'AASM/Massimini2004'``. The method is hashed into
  :func:`turtlewave_hdEEG.dbwrite.event_uuid5` and is part of the
  ``event_chan_time`` UNIQUE constraint, so re-detecting into a 4.0.x
  direct-write database APPENDED a complete duplicate set -- and the duplicate
  guard, keying on the new spelling alone, matched nothing and stayed quiet.
* ``open_write_connection`` caught only ``sqlite3.OperationalError``, missing
  ``DatabaseError`` ("file is not a database"), so a path pointing at an EDF
  or an XML escaped with the connection still open.

Run standalone: ``python tests/test_db_scope_guards.py``.
"""

import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlewave_hdEEG import dbwrite  # noqa: E402


# The events table as the detectors create it.
EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    uuid TEXT PRIMARY KEY, event_type TEXT, channel TEXT,
    start_time REAL, end_time REAL, duration REAL, start_time_hms TEXT,
    stage TEXT, cycle TEXT, method TEXT,
    freq_band TEXT, freq_lower REAL, freq_upper REAL,
    min_amp REAL, max_amp REAL, peak2peak_amp REAL,
    rms REAL, power REAL, peak_power_freq REAL, energy REAL,
    peak_energy_freq REAL, processing_timestamp TEXT, n_fft_sec INTEGER,
    CONSTRAINT event_chan_time UNIQUE
        (event_type, channel, start_time, method, freq_lower, freq_upper, stage)
)"""

#: The per-channel status table, as the detectors create it.
STATUS_DDL = """
CREATE TABLE IF NOT EXISTS processing_status (
    channel TEXT NOT NULL,
    event_type TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT '',
    freq_lower REAL NOT NULL DEFAULT 0,
    freq_upper REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '',
    json_file TEXT,
    processed BOOLEAN DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    last_attempt_time TEXT,
    success BOOLEAN DEFAULT 0,
    error_message TEXT,
    PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage)
)"""

#: The keys every caller of verify_channel_coverage may dereference. Both
#: cluster drivers read all of them without a .get(), so a result missing any
#: one is a KeyError in the middle of an error report.
COVERAGE_KEYS = {'requested', 'with_events', 'covered', 'missing', 'complete',
                 'scoped_status', 'failed', 'events_only'}


def _fresh_db(path):
    """Create a database with the detectors' schema and return a connection.

    Parameters
    ----------
    path : str
        Database file to create.

    Returns
    -------
    sqlite3.Connection
        Open write connection, schema ensured.
    """
    conn = dbwrite.open_write_connection(path)
    conn.execute(EVENTS_DDL)
    conn.execute(STATUS_DDL)
    dbwrite.ensure_direct_write_schema(conn, None)
    conn.commit()
    return conn


def test_coverage_result_always_has_every_key():
    """(1) Every return path yields the same keys, and none raises.

    The three "nothing was written" shapes a mistyped path produces: no file,
    a file with no tables, and a database with events but no
    ``processing_status``.
    """
    print("\n1. verify_channel_coverage returns a complete dict on every path:")

    tmp = tempfile.mkdtemp(prefix='tw_cov_keys_')
    try:
        cases = {}

        cases['missing file'] = os.path.join(tmp, 'nope.db')

        empty = os.path.join(tmp, 'empty.db')
        sqlite3.connect(empty).close()
        cases['empty database'] = empty

        no_status = os.path.join(tmp, 'no_status.db')
        conn = sqlite3.connect(no_status)
        conn.execute(EVENTS_DDL)
        conn.execute(
            "INSERT INTO events (uuid, event_type, channel, start_time, "
            "method, freq_lower, freq_upper, stage) "
            "VALUES ('u1','slow_wave','Cz',10.0,'Ngo2015',0.5,1.25,'NREM3')")
        conn.commit()
        conn.close()
        cases['events, no processing_status'] = no_status

        for label, path in cases.items():
            result = dbwrite.verify_channel_coverage(
                db_path=path, event_type='slow_wave', method='Ngo2015',
                requested_channels=['Cz', 'Pz'], freq_lower=0.5,
                freq_upper=1.25, stage_key='NREM3')
            assert set(result) == COVERAGE_KEYS, (
                f"{label}: keys {sorted(set(result) ^ COVERAGE_KEYS)} differ "
                f"from the contract; a driver reading result['failed'] would "
                f"raise KeyError while reporting the real problem")
            # Exactly what the drivers do, in order.
            _ = (result['failed'], result['complete'], result['missing'],
                 result['events_only'], result['covered'],
                 result['requested'], result['with_events'],
                 result['scoped_status'])
            print(f"   [ok] {label:<30} complete={result['complete']!s:<5} "
                  f"missing={result['missing']} failed={result['failed']} "
                  f"events_only={result['events_only']}")

        # And an empty request is complete, not "incomplete with 0 missing".
        result = dbwrite.verify_channel_coverage(
            db_path=cases['missing file'], event_type='slow_wave',
            method='Ngo2015', requested_channels=[], freq_lower=0.5,
            freq_upper=1.25, stage_key='NREM3')
        assert result['complete'] is True, result
        print("   [ok] a request for no channels is complete, not failed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_zero_event_csv(out_dir, event_type, method_str, freq_token,
                          stage_token, channels_with_events,
                          empty_channels, json_prefix):
    """Write a parameters CSV plus the per-channel JSON a run leaves behind.

    Parameters
    ----------
    out_dir : str
        Results directory.
    event_type : str
        ``'spindle'`` or ``'slow_wave'``.
    method_str : str
        Filename-escaped method token.
    freq_token, stage_token : str
        Band and stage components of the naming convention.
    channels_with_events : list of str
        Channels that get one event row in the CSV and a non-empty JSON.
    empty_channels : list of str
        Channels whose JSON is an empty list -- they ran and found nothing.
    json_prefix : str
        Per-channel JSON basename prefix the detector uses.

    Returns
    -------
    str
        Path of the parameters CSV.
    """
    import json

    csv_path = os.path.join(
        out_dir,
        f"{'sw' if event_type == 'slow_wave' else 'spindle'}_parameters_"
        f"{method_str}_{freq_token}_{stage_token}.csv")
    header = ("Start time,Start time (HH:MM:SS),End time,Duration (s),Stage,"
              "Cycle,Channel,Event type,Min. amplitude (uV),"
              "Max. amplitude (uV),Peak-to-peak amplitude (uV),UUID\n")
    lines = [header]
    for i, chan in enumerate(channels_with_events):
        lines.append(
            f"{10.0 + i},00:00:{10 + i:02d},{11.0 + i},1.0,{stage_token},1,"
            f"{chan},{event_type},-50.0,50.0,100.0,uuid-{chan}\n")
    with open(csv_path, 'w') as f:
        f.writelines(lines)

    for chan in channels_with_events:
        with open(os.path.join(
                out_dir,
                f"{json_prefix}_{method_str}_{freq_token}_{stage_token}_"
                f"{chan}.json"), 'w') as f:
            json.dump([{'start_time': 10.0, 'end_time': 11.0}], f)
    for chan in empty_channels:
        with open(os.path.join(
                out_dir,
                f"{json_prefix}_{method_str}_{freq_token}_{stage_token}_"
                f"{chan}.json"), 'w') as f:
            json.dump([], f)
    return csv_path


def test_zero_event_channel_is_covered_on_the_csv_path():
    """(2) A channel that detected nothing is covered, not 'missing'.

    The legacy ``--legacy-json`` route is detect -> JSON -> CSV -> import ->
    coverage check. A channel with no events has no row in the CSV, so its
    ONLY evidence is a ``processing_status`` row -- which the importer wrote
    without the run's method/band/stage, so the scoped coverage query could
    never match it. Both processors are exercised, since each carries its own
    copy of the importer.
    """
    print("\n2. A zero-event channel is covered on the CSV import path:")

    from wonambi import Dataset  # noqa: F401  (import cost only, not used)

    cases = [
        ('spindle', 'ParalEvents', 'Moelle2011', '11-16Hz', 'NREM2NREM3',
         'spindles'),
        ('slow_wave', 'ParalSWA', 'AASM_Massimini2004', '0.5-4Hz', 'NREM3',
         'slowwaves'),
    ]

    for event_type, cls_name, method_str, freq_token, stage_token, prefix \
            in cases:
        tmp = tempfile.mkdtemp(prefix='tw_zero_evt_')
        try:
            import logging

            from turtlewave_hdEEG import ParalEvents, ParalSWA

            proc_cls = {'ParalEvents': ParalEvents, 'ParalSWA': ParalSWA}[
                cls_name]
            proc = proc_cls(None, None, log_level=logging.CRITICAL)

            csv_path = _write_zero_event_csv(
                tmp, event_type, method_str, freq_token, stage_token,
                channels_with_events=['Cz'], empty_channels=['Pz', 'Fz'],
                json_prefix=prefix)

            db = os.path.join(tmp, 'neural_events.db')
            method = method_str.replace('AASM_', 'AASM/')
            proc.initialize_sqlite_database(db)
            stats = proc.import_parameters_csv_to_database(
                csv_file=csv_path, db_path=db, event_type=event_type,
                method=method)
            assert stats.get('ok'), stats

            lo, hi = [float(x) for x in
                      freq_token.replace('Hz', '').split('-')]
            coverage = dbwrite.verify_channel_coverage(
                db_path=db, event_type=event_type, method=method,
                requested_channels=['Cz', 'Pz', 'Fz'],
                freq_lower=lo, freq_upper=hi, stage_key=stage_token)

            assert coverage['complete'], (
                f"{cls_name}/{event_type}: {coverage['missing']} reported "
                f"missing, but they ran and found nothing -- the driver would "
                f"exit 1 on a successful run")
            assert coverage['covered'] == 3, coverage
            assert not coverage['events_only'], (
                f"{cls_name}: {coverage['events_only']} are credited by event "
                f"rows alone, so their processing_status row still does not "
                f"carry this scope")

            rows = sqlite3.connect(db).execute(
                "SELECT channel, method, freq_lower, freq_upper, stage, "
                "success FROM processing_status ORDER BY channel").fetchall()
            for chan, m, flo, fhi, stg, ok in rows:
                assert (m, flo, fhi, stg) == (method, lo, hi, stage_token), (
                    f"{cls_name}: {chan}'s status row carries scope "
                    f"{(m, flo, fhi, stg)}, not {(method, lo, hi, stage_token)}")
                assert ok == 1, f"{cls_name}: {chan} recorded success={ok}"
            print(f"   [ok] {cls_name:<12} {event_type:<10} "
                  f"{len(rows)} status rows all scoped "
                  f"{(method, lo, hi, stage_token)}; coverage complete "
                  f"({coverage['covered']}/{coverage['requested']}, "
                  f"{coverage['with_events']} with events)")

            # Idempotency: importing the same CSV again must not change the
            # verdict (this is the "self-heals after one scoped run" claim).
            proc.import_parameters_csv_to_database(
                csv_file=csv_path, db_path=db, event_type=event_type,
                method=method)
            again = dbwrite.verify_channel_coverage(
                db_path=db, event_type=event_type, method=method,
                requested_channels=['Cz', 'Pz', 'Fz'],
                freq_lower=lo, freq_upper=hi, stage_key=stage_token)
            assert again['complete'] and again['covered'] == 3, again
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("   [ok] a second import leaves the verdict unchanged")


def test_failed_channel_still_fails_on_the_csv_path():
    """(2b) The fix must not credit a channel whose detection ERRORED.

    An error-sentinel JSON records ``success = 0``, and an in-scope failure
    beats any event evidence. If the scope fix had made every status row look
    successful, this is the assertion that would catch it.
    """
    print("\n3. A failed channel is still reported, with the scope attached:")

    import json
    import logging

    from turtlewave_hdEEG import ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_failed_chan_')
    try:
        proc = ParalSWA(None, None, log_level=logging.CRITICAL)
        csv_path = _write_zero_event_csv(
            tmp, 'slow_wave', 'Ngo2015', '0.5-1.25Hz', 'NREM3',
            channels_with_events=['Cz'], empty_channels=['Pz'],
            json_prefix='slowwaves')
        with open(os.path.join(
                tmp,
                'slowwaves_Ngo2015_0.5-1.25Hz_NREM3_Fz.json'), 'w') as f:
            json.dump({'error': 'detector blew up'}, f)

        db = os.path.join(tmp, 'neural_events.db')
        proc.initialize_sqlite_database(db)
        proc.import_parameters_csv_to_database(
            csv_file=csv_path, db_path=db, event_type='slow_wave',
            method='Ngo2015')

        coverage = dbwrite.verify_channel_coverage(
            db_path=db, event_type='slow_wave', method='Ngo2015',
            requested_channels=['Cz', 'Pz', 'Fz'], freq_lower=0.5,
            freq_upper=1.25, stage_key='NREM3')
        assert coverage['failed'] == ['Fz'], coverage
        assert coverage['missing'] == ['Fz'], coverage
        assert not coverage['complete'], coverage
        print(f"   [ok] Cz (events) and Pz (empty) covered; Fz (error "
              f"sentinel) reported failed={coverage['failed']} and "
              f"missing={coverage['missing']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_method_spelling_cannot_silently_duplicate():
    """(3) Re-detecting into a 4.0.x escaped-method database is refused.

    Built as a 4.0.x direct-write database would be: escaped method, and the
    ``'joint'`` stage marker with a matching stage token, so BOTH existing
    checks pass and only the spelling check can catch it. Then the replace
    path is shown to actually replace, so the guard's own advice works.
    """
    print("\n4. The escaped/unescaped method spelling cannot duplicate:")

    assert dbwrite.method_spellings('AASM/Massimini2004') == [
        'AASM/Massimini2004', 'AASM_Massimini2004']
    assert dbwrite.method_spellings('AASM_Massimini2004') == [
        'AASM_Massimini2004', 'AASM/Massimini2004']
    assert dbwrite.method_spellings('Moelle2011') == ['Moelle2011']
    # A joined multi-method run label must NOT be un-escaped into nonsense.
    assert dbwrite.method_spellings('Moelle2011_Wamsley2012') == [
        'Moelle2011_Wamsley2012']
    print("   [ok] method_spellings pairs only the known slash-method")

    tmp = tempfile.mkdtemp(prefix='tw_method_spelling_')
    try:
        db = os.path.join(tmp, 'neural_events.db')
        conn = _fresh_db(db)
        old_uuid = dbwrite.event_uuid5('slow_wave', 'Cz', 10.0,
                                       'AASM_Massimini2004', 0.5, 4.0, 'NREM3')
        conn.execute(
            "INSERT INTO events (uuid, event_type, channel, start_time, "
            "end_time, duration, stage, method, freq_lower, freq_upper, "
            "run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (old_uuid, 'slow_wave', 'Cz', 10.0, 11.0, 1.0, 'NREM3',
             'AASM_Massimini2004', 0.5, 4.0, 'run-4.0'))
        # Marked joint, same stage token: the other two checks cannot fire.
        dbwrite.set_db_meta(conn, dbwrite.STAGE_FORMAT_KEY,
                            dbwrite.STAGE_FORMAT_JOINT)
        conn.commit()
        assert dbwrite.stage_format(conn) == dbwrite.STAGE_FORMAT_JOINT

        new_uuid = dbwrite.event_uuid5('slow_wave', 'Cz', 10.0,
                                       'AASM/Massimini2004', 0.5, 4.0, 'NREM3')
        assert old_uuid != new_uuid, (
            "the two spellings hash to one uuid, so this test cannot "
            "demonstrate the duplication it guards against")

        raised = None
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'slow_wave', ['AASM/Massimini2004'], 0.5, 4.0,
                stage_token='NREM3', channels=['Cz'], db_path=db)
        except ValueError as e:
            raised = str(e)
        assert raised is not None, (
            "the guard allowed a run that would append a duplicate set under "
            "the other method spelling")
        assert 'AASM_Massimini2004' in raised and 'DUPLICATE' in raised, raised
        print(f"   [ok] guard REFUSES: {raised.split('. ')[0][:96]}...")

        # The advice it gives must work: replace_channels deletes BOTH
        # spellings, so the re-detection replaces instead of duplicating.
        n = dbwrite.write_channel_events(
            conn, run_id='run-4.3', event_type='slow_wave', channel='Cz',
            method='AASM/Massimini2004', freq_lower=0.5, freq_upper=4.0,
            stage_key='NREM3',
            events=[{'uuid': new_uuid, 'start_time': 10.0, 'end_time': 11.0,
                     'duration': 1.0, 'stage': 'NREM3',
                     'method': 'AASM/Massimini2004'}],
            batched=[{}], recording_start_time=None, n_fft_sec=None,
            replace=True, replace_methods=['AASM/Massimini2004'])
        conn.commit()
        assert n == 1, n
        rows = conn.execute(
            "SELECT method FROM events ORDER BY method").fetchall()
        assert len(rows) == 1 and rows[0][0] == 'AASM/Massimini2004', (
            f"scoped replace left {len(rows)} row(s) {rows}; the escaped "
            f"spelling was not deleted, so the database still doubled")
        print(f"   [ok] replace_channels deletes both spellings: 1 row left, "
              f"method={rows[0][0]!r}")

        # And the guard is quiet once only the new spelling remains.
        assert dbwrite.assert_stage_format_compatible(
            conn, 'slow_wave', ['AASM/Massimini2004'], 0.5, 4.0,
            stage_token='NREM3', channels=['Cz'], db_path=db) == 0
        print("   [ok] guard passes once the old spelling is gone")
        conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_open_write_connection_reports_a_non_database():
    """(4) 'file is not a database' is caught, closed and reported.

    ``DatabaseError`` is a SIBLING of ``OperationalError``, so the old
    ``except sqlite3.OperationalError`` let it escape with the connection
    still open.
    """
    print("\n5. open_write_connection handles a file that is not a database:")

    tmp = tempfile.mkdtemp(prefix='tw_not_a_db_')
    try:
        bogus = os.path.join(tmp, 'sub-01.edf')
        with open(bogus, 'wb') as f:
            f.write(b'0' * 4096)  # plausible junk, not a SQLite header

        raised = None
        try:
            conn = dbwrite.open_write_connection(bogus)
            conn.close()
        except sqlite3.DatabaseError as e:
            raised = e
        assert raised is not None, (
            "opening a non-database returned a usable connection")
        assert 'not a database' in str(raised).lower(), str(raised)
        print(f"   [ok] raises {type(raised).__name__}: {raised}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage_token_helpers():
    """The filename/rows fallback that gives a status row its stage."""
    print("\n6. The CSV stage token comes from the name, then the rows:")

    assert dbwrite.stage_token_from_filename(
        'sw_parameters_Ngo2015_0.5-1.25Hz_NREM3.csv') == 'NREM3'
    assert dbwrite.stage_token_from_filename(
        '/a/b/spindle_parameters_Moelle2011_11-16Hz_NREM2NREM3.csv'
    ) == 'NREM2NREM3'
    assert dbwrite.stage_token_from_filename('whatever.csv') is None
    assert dbwrite.stage_token_from_filename(
        'sw_parameters_Ngo2015_0.5-1.25Hz.csv') is None
    print("   [ok] stage_token_from_filename reads the convention, and "
          "declines a name that does not follow it")

    # Filename wins; rows are the fallback; several stages with an unparseable
    # name gives '' rather than a guess.
    assert dbwrite.resolve_status_stage_token(
        'sw_parameters_Ngo2015_0.5-1.25Hz_NREM3.csv', ['NREM2']) == 'NREM3'
    assert dbwrite.resolve_status_stage_token(
        'renamed.csv', ['NREM2', 'NREM2']) == 'NREM2'
    assert dbwrite.resolve_status_stage_token(
        'renamed.csv', ["['NREM2', 'NREM3']"]) == 'NREM2NREM3'
    assert dbwrite.resolve_status_stage_token(
        'renamed.csv', ['NREM2', 'NREM3']) == ''
    assert dbwrite.resolve_status_stage_token('renamed.csv', None) == ''
    print("   [ok] resolve_status_stage_token: filename > single row stage > "
          "'' (never a guess)")


if __name__ == "__main__":
    print("TESTING DATABASE SCOPE GUARDS AND COVERAGE REPORTING")
    print("===================================================")

    test_coverage_result_always_has_every_key()
    test_zero_event_channel_is_covered_on_the_csv_path()
    test_failed_channel_still_fails_on_the_csv_path()
    test_method_spelling_cannot_silently_duplicate()
    test_open_write_connection_reports_a_non_database()
    test_stage_token_helpers()

    print("\nAll database scope-guard tests passed.")
