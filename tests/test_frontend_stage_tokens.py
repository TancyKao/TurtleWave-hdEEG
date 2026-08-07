#!/usr/bin/env python3
"""Frontend regression tests for joint stage tokens in ``events.stage``.

Since every detector stores the run's *joint* stage token (``NREM2NREM3``)
rather than each event's epoch stage, three GUI code paths that keyed on a
single stage had to change. Each is pinned here:

* :func:`frontend.eeg_review_gui.qc_density_stage_scope` -- splits the token
  back into scored stage labels before it becomes the QC density denominator.
  Left unsplit, ``build_density_denominators`` matches no scored epoch and
  density silently blanks.
* ``TurtleWaveGUI.count_db_events`` -- the post-run verification count. Left
  filtering ``stage IN ('NREM2','NREM3')`` it returns 0 against joint rows, so
  a successful detection reports "0 events written".
* ``TurtleWaveGUI.update_pac_available_channels`` -- the PAC channel lookup.
  Its old self-join required both event types to carry an identical stage
  string, which a database mixing per-epoch and joint stages cannot satisfy.

Runs standalone (``python tests/test_frontend_stage_tokens.py``) and under
pytest. Needs PyQt5 importable, but no display: set ``QT_QPA_PLATFORM=offscreen``.
"""

import os
import sys
import shutil
import sqlite3
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The events table as the detectors create it (eventprocessor.py / swprocessor.py).
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


def _make_events_db(path, rows):
    """Build a database of ``(event_type, method, stage, channel, run_id)`` rows."""
    from turtlewave_hdEEG.dbwrite import ensure_direct_write_schema

    conn = sqlite3.connect(path)
    try:
        conn.execute(EVENTS_DDL)
        ensure_direct_write_schema(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        for i, (event_type, method, stage, channel, run_id) in enumerate(rows):
            row = {'uuid': 'u%d' % i, 'event_type': event_type,
                   'method': method, 'stage': stage, 'channel': channel,
                   'start_time': float(i), 'end_time': float(i) + 1.0,
                   'freq_lower': 11.0, 'freq_upper': 16.0, 'run_id': run_id,
                   'subject': 'sub-10sd'}
            row = {k: v for k, v in row.items() if k in cols}
            conn.execute("INSERT INTO events (%s) VALUES (%s)"
                         % (",".join(row), ",".join("?" * len(row))),
                         list(row.values()))
        conn.commit()
    finally:
        conn.close()


class _Combo:
    def __init__(self, text=""):
        self._text = text

    def currentText(self):
        return self._text


def _fake_gui(output_dir=None, sw_info=None, sp_info=None):
    """A stand-in carrying only the attributes the methods under test touch.

    Instantiating TurtleWaveGUI needs a QApplication and builds the whole
    window; these two methods depend on a handful of plain attributes, so
    borrowing the unbound functions keeps the test to the logic being pinned.
    """
    from frontend.turtlewave_gui import TurtleWaveGUI

    class FakeGUI:
        count_db_events = TurtleWaveGUI.count_db_events
        update_pac_available_channels = TurtleWaveGUI.update_pac_available_channels
        write_log_once = TurtleWaveGUI.write_log_once

        def __init__(self):
            self.output_dir = output_dir
            self.pac_method_combo = _Combo("SW-Spindle")
            self.sw_method_pac_combo = _Combo("SW")
            self.spindle_method_pac_combo = _Combo("SP")
            self.sw_methods_info = {"SW": sw_info or {}}
            self.spindle_methods_info = {"SP": sp_info or {}}
            self.pac_available_channels = None
            self.logged = []

        def write_log(self, message):
            self.logged.append(message)

        def update_pac_channel_lists(self):
            pass

    return FakeGUI()


def test_qc_density_stage_scope():
    """The QC density scope must be scored stage labels, never a joint token."""
    print("\n1. Testing the QC density stage scope helper:")

    from frontend.eeg_review_gui import qc_density_stage_scope as scope

    assert scope(['NREM2NREM3']) == ['NREM2', 'NREM3'], scope(['NREM2NREM3'])
    print("[ok] a joint token splits into its components")

    # A database holding both forms (joint spindles, legacy per-epoch slow
    # waves) must not double-count or fragment the scope.
    assert scope(['NREM2NREM3', 'NREM2', 'NREM3']) == ['NREM2', 'NREM3']
    print("[ok] mixed joint + per-epoch values dedupe to one scope")

    assert scope(['NREM2']) == ['NREM2']
    assert scope(['NREM2NREM3REM']) == ['NREM2', 'NREM3', 'REM']
    print("[ok] single stage and three-stage token round-trip")

    # Wake is never a detection stage; including it would inflate the
    # denominator and deflate every density on the dashboard.
    assert scope(['NREM2NREM3', 'Wake', 'Undefined', '']) == ['NREM2', 'NREM3']
    assert scope(['Wake', '']) is None
    print("[ok] Wake-like labels dropped; an all-Wake scope returns None")

    # None means "no event-derived scope", the caller's signal to fall back to
    # the fixed N2+N3+REM set rather than to compute against nothing.
    assert scope(None) is None
    assert scope([]) is None
    print("[ok] None / empty return None so the caller falls back")

    # Labels outside the library vocabulary are passed through whole. Dropping
    # them would shrink the denominator; guessing at them would corrupt it.
    assert scope(['N2', 'N3']) == ['N2', 'N3']
    assert scope(['NREM2NREM3', 'SomeSiteLabel']) == ['NREM2', 'NREM3',
                                                      'SomeSiteLabel']
    print("[ok] unknown labels survive unsplit rather than being discarded")

    # numpy array input: this is fed straight from df['stage'].unique().
    import numpy as np
    assert scope(np.array(['NREM2NREM3'], dtype=object)) == ['NREM2', 'NREM3']
    print("[ok] accepts the numpy array the QC refresh actually passes")

    from frontend.eeg_review_gui import EventReviewGUI, QC_WAKE_STAGES
    assert EventReviewGUI._WAKE_STAGES is QC_WAKE_STAGES
    print("[ok] the widget and the helper share one Wake vocabulary")


def test_count_db_events_reads_joint_tokens():
    """A successful joint-token run must not be reported as "0 events written"."""
    print("\n2. Testing count_db_events against joint stage tokens:")

    tmp = tempfile.mkdtemp(prefix='tw_frontend_count_')
    db = os.path.join(tmp, 'neural_events.db')
    try:
        _make_events_db(db, [
            # A joint-token run: 4 spindles over 2 channels on NREM2+NREM3.
            ('spindle', 'Lacourse2018', 'NREM2NREM3', 'E1', 'run-A'),
            ('spindle', 'Lacourse2018', 'NREM2NREM3', 'E1', 'run-A'),
            ('spindle', 'Lacourse2018', 'NREM2NREM3', 'E2', 'run-A'),
            ('spindle', 'Lacourse2018', 'NREM2NREM3', 'E2', 'run-A'),
            # Legacy per-epoch rows from an older run of the same scope.
            ('spindle', 'Lacourse2018', 'NREM2', 'E3', 'run-legacy'),
            ('spindle', 'Lacourse2018', 'NREM3', 'E3', 'run-legacy'),
            # Out of scope: REM must never be counted for an N2+N3 run.
            ('spindle', 'Lacourse2018', 'REM', 'E4', 'run-rem'),
            # Out of scope: a different event type entirely.
            ('slow_wave', 'Massimini2004', 'NREM2NREM3', 'E1', 'run-sw'),
        ])
        gui = _fake_gui()

        n, nch = gui.count_db_events(db, 'spindle', 'Lacourse2018',
                                     (11.0, 16.0), ['NREM2', 'NREM3'])
        assert (n, nch) == (6, 3), \
            f"requested N2+N3 gave {n} events / {nch} channels, expected 6 / 3"
        print(f"[ok] a request for ['NREM2','NREM3'] counts {n} events over "
              f"{nch} channels - joint AND legacy per-epoch rows")

        # Scoped to the run that wrote the joint rows.
        n, nch = gui.count_db_events(db, 'spindle', 'Lacourse2018',
                                     (11.0, 16.0), ['NREM2', 'NREM3'],
                                     run_ids={'run-A'})
        assert (n, nch) == (4, 2), f"run-scoped count was {n} / {nch}"
        print(f"[ok] restricted to run-A: {n} events over {nch} channels")

        # A single-stage request must not sweep in the joint token: NREM2NREM3
        # covers time the request did not ask for.
        n, _ = gui.count_db_events(db, 'spindle', 'Lacourse2018',
                                   (11.0, 16.0), ['NREM2'])
        assert n == 1, f"an NREM2-only request counted {n}, expected 1"
        print(f"[ok] an NREM2-only request counts {n} (the joint token is NOT "
              f"a subset of NREM2)")

        # REM is a stage in the database but not in the request.
        n, _ = gui.count_db_events(db, 'spindle', 'Lacourse2018',
                                   (11.0, 16.0), ['REM'])
        assert n == 1, f"a REM request counted {n}, expected 1"
        print(f"[ok] a REM request counts {n} - scopes stay separate")

        # No stage filter counts the whole method/band scope.
        n, _ = gui.count_db_events(db, 'spindle', 'Lacourse2018',
                                   (11.0, 16.0), None)
        assert n == 7, f"an unfiltered count gave {n}, expected 7"
        print(f"[ok] no stage filter counts all {n} spindles")

        # An unreadable database reports "cannot tell", never zero.
        n, nch = gui.count_db_events(os.path.join(tmp, 'nope.db'), 'spindle',
                                     'Lacourse2018', (11.0, 16.0), ['NREM2'])
        assert (n, nch) == (None, None), (n, nch)
        print("[ok] a missing database returns (None, None), not (0, 0)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pac_channel_lookup_survives_stage_mismatch():
    """PAC channels must populate even when the two sides' stage tokens differ."""
    print("\n3. Testing the PAC channel lookup:")

    def _run(sw_stage, sw_chans, sp_stage, sp_chans):
        tmp = tempfile.mkdtemp(prefix='tw_frontend_pac_')
        os.makedirs(os.path.join(tmp, 'wonambi'))
        db = os.path.join(tmp, 'wonambi', 'neural_events.db')
        rows = [('slow_wave', 'Massimini2004', sw_stage, ch, 'run-sw')
                for ch in sw_chans]
        rows += [('spindle', 'Lacourse2018', sp_stage, ch, 'run-sp')
                 for ch in sp_chans]
        _make_events_db(db, rows)
        gui = _fake_gui(
            output_dir=tmp,
            sw_info={'method': 'Massimini2004', 'stage': sw_stage,
                     'freq_range': (0.5, 1.25)},
            sp_info={'method': 'Lacourse2018', 'stage': sp_stage,
                     'freq_range': (11, 16)})
        gui.update_pac_available_channels()
        shutil.rmtree(tmp, ignore_errors=True)
        return gui

    # The reported blocker: slow waves stored per-epoch, spindles joint. The
    # old self-join on sw.stage = sp.stage could never match, so PAC was dead.
    gui = _run('NREM3', ['E1', 'E2', 'E3'], 'NREM2NREM3', ['E2', 'E3', 'E9'])
    assert gui.pac_available_channels == ['E2', 'E3'], \
        gui.pac_available_channels
    print(f"[ok] mismatched stage tokens still yield "
          f"{gui.pac_available_channels} - the join returned nothing here")
    assert any("stages don't match" in m for m in gui.logged), gui.logged
    print("[ok] the mismatch is reported, so the run-time modal is expected")

    # Consistent joint tokens: the normal post-change case.
    gui = _run('NREM2NREM3', ['E1', 'E2'], 'NREM2NREM3', ['E1', 'E2'])
    assert gui.pac_available_channels == ['E1', 'E2'], \
        gui.pac_available_channels
    assert not gui.logged, gui.logged
    print("[ok] matching joint tokens populate the list with nothing logged")

    # Genuinely empty: the log must say which side and why, not stay silent.
    gui = _run('NREM2NREM3', ['E1', 'E2'], 'NREM2', [])
    assert gui.pac_available_channels == []
    diag = [m for m in gui.logged if m.startswith('PAC:')]
    assert len(diag) == 1, gui.logged
    assert 'no spindle rows' in diag[0], diag[0]
    assert 'stage scopes differ' in diag[0], diag[0]
    print(f"[ok] empty result is diagnosed: {diag[0]}")

    # Same stage, disjoint channels: a different cause, named differently.
    gui = _run('NREM2NREM3', ['E1'], 'NREM2NREM3', ['E9'])
    diag = [m for m in gui.logged if m.startswith('PAC:')]
    assert 'none in common' in diag[0], diag[0]
    assert 'stage scopes differ' not in diag[0], diag[0]
    print(f"[ok] disjoint channels diagnosed separately: {diag[0]}")

    # Re-entered on every combo change; the diagnostic must not spam the log.
    tmp = tempfile.mkdtemp(prefix='tw_frontend_pac_')
    os.makedirs(os.path.join(tmp, 'wonambi'))
    db = os.path.join(tmp, 'wonambi', 'neural_events.db')
    try:
        _make_events_db(db, [('slow_wave', 'Massimini2004', 'NREM2NREM3',
                              'E1', 'run-sw'),
                             ('spindle', 'Lacourse2018', 'NREM2NREM3',
                              'E9', 'run-sp')])
        gui = _fake_gui(
            output_dir=tmp,
            sw_info={'method': 'Massimini2004', 'stage': 'NREM2NREM3',
                     'freq_range': (0.5, 1.25)},
            sp_info={'method': 'Lacourse2018', 'stage': 'NREM2NREM3',
                     'freq_range': (11, 16)})
        for _ in range(4):
            gui.update_pac_available_channels()
        assert len(gui.logged) == 1, gui.logged
        print(f"[ok] 4 lookups logged the diagnostic {len(gui.logged)} time")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _FakeAnnot:
    """60 NREM2 epochs + 40 NREM3 epochs, all Good, no artefact events.

    Implements exactly the surface ``wonambi.trans.select.get_times`` uses, so
    the denominator below is computed by the real library code path.
    """

    last_second = 100 * 30.0

    def get_stages(self):
        return ['NREM2'] * 60 + ['NREM3'] * 40

    def get_epochs(self, name=None, time=None, chan=None, stage=None,
                   qual=None):
        out = []
        for i, s in enumerate(self.get_stages()):
            if stage is not None and s not in stage:
                continue
            out.append({'start': i * 30.0, 'end': (i + 1) * 30.0,
                        'stage': s, 'quality': 'Good'})
        return out

    def get_events(self, name=None, time=None, chan=None, stage=None,
                   qual=None):
        return []


class _FakeDataset:
    header = {'s_freq': 256.0}


def test_qc_denominator_end_to_end():
    """A joint token must not blank the QC density denominator."""
    print("\n4. Testing the QC density denominator end to end:")

    from turtlewave_hdEEG.utils import build_density_denominators
    from frontend.eeg_review_gui import qc_density_stage_scope

    def minutes_for(stage_list):
        d = build_density_denominators(
            _FakeAnnot(), _FakeDataset(), reject_artifacts=True,
            reject_arousals=True, stage_list=stage_list,
            stages_present=stage_list)
        return d.whole_night_analysed_min

    scope = qc_density_stage_scope(['NREM2NREM3'])
    assert scope == ['NREM2', 'NREM3'], scope
    split = minutes_for(scope)
    assert abs(split - 50.0) < 1e-6, split
    print(f"[ok] the split scope gives {split} analysed min (30 N2 + 20 N3)")

    # The regression itself: the raw token matches no scored epoch, so the
    # denominator collapses to zero and every density on the dashboard blanks
    # without a word said.
    raw = minutes_for(['NREM2NREM3'])
    assert not raw, raw
    print(f"[ok] the unsplit token gives {raw} min - the silent failure fixed")

    # And the stored row the dashboard checks itself against is keyed per
    # single stage, so _qc_stored_density_minutes pools the same 50 minutes
    # from components and (correctly) finds nothing for a raw token. This is
    # why that method needed no change.
    from turtlewave_hdEEG import dbwrite
    from frontend.eeg_review_gui import EventReviewGUI

    tmp = tempfile.mkdtemp(prefix='tw_frontend_denom_')
    try:
        db = os.path.join(tmp, 'neural_events.db')
        conn = dbwrite.open_write_connection(db)
        try:
            dbwrite.ensure_analysed_time_schema(conn)
            dbwrite.record_analysed_time(conn, 'sub-10sd', 'NREM2', 1800.0,
                                         reject_artifacts=True,
                                         reject_arousals=True)
            dbwrite.record_analysed_time(conn, 'sub-10sd', 'NREM3', 1200.0,
                                         reject_artifacts=True,
                                         reject_arousals=True)
            conn.commit()
        finally:
            conn.close()

        _handle = type('D', (), {'db_path': db})()

        class _Fake:
            _qc_stored_density_minutes = \
                EventReviewGUI._qc_stored_density_minutes
            db = _handle

        f = _Fake()
        assert f._qc_stored_density_minutes(scope, True, True) == 50.0
        assert f._qc_stored_density_minutes(['NREM2NREM3'], True, True) is None
        print("[ok] the stored denominator pools to 50.0 min over the same "
              "components, so _qc_stored_density_minutes is left untouched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING FRONTEND JOINT-STAGE-TOKEN HANDLING")
    print("===========================================")
    test_qc_density_stage_scope()
    test_count_db_events_reads_joint_tokens()
    test_pac_channel_lookup_survives_stage_mismatch()
    test_qc_denominator_end_to_end()
    print("\nAll frontend stage-token tests passed.")
