#!/usr/bin/env python3
"""Interpolated channels are read, flagged once per run and recorded.

A cleaning pipeline that interpolates a bad channel replaces its signal with a
weighted mix of its neighbours, and records the names in
``EEG.etc.interp_channels``. Events detected on such a channel are the
neighbours' events, spatially smoothed, so a density or coupling value from it
must never pass unremarked. TurtleWave flags these channels; it does not drop
them. Asserted here:

* ``header['interp_channels']`` (from ``open_dataset``) and
  ``read_eeglab_channel_info(...)['interp_channels']`` are the listed names in
  channel order, for both struct layouts and both MATLAB versions; ``[]`` when
  the field is absent or empty and for a non-``.set`` file; a listed name that
  is not a channel is dropped with an INFO message; non-text entries are
  skipped.
* ``utils.interpolated_channels(header, channels)`` returns the selection's
  interpolated channels in file order.
* ``detect_spindles``, ``detect_slow_waves`` and ``detect_kcomplexes`` log one
  WARNING naming them, and write ``detection_runs.interpolated_channels`` (a
  JSON list, ``'[]'`` when none) and the same list into ``params_json``;
  ``analyze_pac`` logs the WARNING.
* An existing database without the column gains it, old rows stay NULL.

Run standalone: ``python tests/test_interp_channels.py``. Exits non-zero if any
test fails.
"""

import json
import logging
import gc
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eeglab_fixture as fx  # noqa: E402
from turtlewave_hdEEG import open_dataset  # noqa: E402
from turtlewave_hdEEG.eeglab_io import read_eeglab_channel_info  # noqa: E402
from turtlewave_hdEEG.utils import (  # noqa: E402
    interpolated_channels, warn_interpolated_channels)

FS = 128.0
LABELS = ['Fz', 'Cz', 'Pz']
TYPES = ['EEG', 'EEG', 'EEG']
WARNING_TEXT = ("selected channels are interpolated in the file "
                "(signal reconstructed from neighbours)")


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_interp_')
        return self.path

    def __exit__(self, *exc):
        gc.collect()   # Windows: drop datasets that still map files
        shutil.rmtree(self.path, ignore_errors=True)


class Capture(logging.Handler):
    """Collect records from one logger for the duration of a ``with``."""

    def __init__(self, logger):
        super().__init__(level=logging.DEBUG)
        self.logger = logger
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self.logger.addHandler(self)
        self._level = self.logger.level
        if self.logger.level == logging.NOTSET or self.logger.level > logging.INFO:
            self.logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self._level)

    def messages(self, level=None):
        return [r.getMessage() for r in self.records
                if level is None or r.levelno == level]


def _signal(n_samples, seed=0):
    """Three channels of a slow oscillation plus one 13.5 Hz burst per epoch,
    in microvolts, so all three detectors have something to find."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    base = 60.0 * np.sin(2 * np.pi * 0.8 * t) + 5.0 * rng.standard_normal(n_samples)
    for i in range(int(n_samples / FS // 30)):
        t0 = i * 30.0 + 12.0
        burst = (t >= t0) & (t < t0 + 1.0)
        base[burst] += 30.0 * np.sin(2 * np.pi * 13.5 * (t[burst] - t0))
    return np.vstack([base + 0.5 * k * rng.standard_normal(n_samples)
                      for k in range(len(LABELS))]).astype(np.float32)


def _write(tmp, layout='root', v73=True, interp=('Cz',), name=None, n_epochs=4):
    n = int(30 * n_epochs * FS)
    path = os.path.join(tmp, name or f'rec_{layout}_{int(v73)}.set')
    fx.write_set(path, layout, v73, n_samples=n, srate=FS, labels=LABELS,
                 types=TYPES, ref='average', n_good=3, stages=['2'] * n_epochs,
                 data=_signal(n), interp_channels=(None if interp is None
                                                   else list(interp)))
    return path


# ------------------------------------------------------------------ reading

def test_header_key_every_layout():
    """Both layouts, both MATLAB versions, several shapes of the field."""
    print("\n1. header['interp_channels'] in every layout:")
    cases = [
        (None, []), ([], []), (['Cz'], ['Cz']),
        (['Pz', 'Fz'], ['Fz', 'Pz']),         # stored order -> channel order
        (['Pz', 'Xx'], ['Pz']),               # unknown name dropped, INFO
        (['Cz', 3.0], ['Cz']),                # numeric entry skipped, INFO
    ]
    log = logging.getLogger('turtlewave_hdEEG.eeglab_io')
    with Workdir() as tmp:
        for layout, v73 in fx.ALL_LAYOUTS:
            for k, (stored, expected) in enumerate(cases):
                path = _write(tmp, layout, v73, interp=stored,
                              name=f'c{k}_{layout}_{int(v73)}.set', n_epochs=1)
                with Capture(log) as cap:
                    header = open_dataset(path).header
                    info = read_eeglab_channel_info(path)
                got = header['interp_channels']
                assert isinstance(got, list) and all(isinstance(c, str) for c in got), got
                assert got == expected, (fx.layout_id(layout, v73), stored, got)
                assert info['interp_channels'] == expected, info['interp_channels']
                if stored and 'Xx' in stored:
                    assert any('not channels in the file' in m and 'Xx' in m
                               for m in cap.messages(logging.INFO)), cap.messages()
                if stored and any(not isinstance(s, str) for s in stored):
                    assert any('non-text entries' in m
                               for m in cap.messages(logging.INFO)), cap.messages()
            print(f"   [ok] {fx.layout_id(layout, v73)}: {len(cases)} cases")


def test_non_set_file_has_empty_list():
    """Any other format gets the key too, always ``[]``."""
    print("\n2. Non-.set recording:")
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data
    with Workdir() as tmp:
        data = create_data(datatype='ChanTime', n_trial=1, s_freq=FS,
                           chan_name=['Cz'], time=(0, 10))
        edf = os.path.join(tmp, 'x.edf')
        write_edf(data, edf)
        assert open_dataset(edf).header['interp_channels'] == []
    print("   [ok] EDF header['interp_channels'] == []")


def test_interpolated_channels_helper():
    """Selection filtering and file order."""
    print("\n3. utils.interpolated_channels:")
    header = {'chan_name': ['Fz', 'Cz', 'Pz', 'Oz'],
              'interp_channels': ['Pz', 'Cz']}
    assert interpolated_channels(header, ['Pz', 'Oz', 'Cz']) == ['Cz', 'Pz']
    assert interpolated_channels(header, ['Fz', 'Oz']) == []
    assert interpolated_channels(header, 'Cz') == ['Cz']
    assert interpolated_channels(header) == ['Cz', 'Pz']
    assert interpolated_channels({'chan_name': ['Cz']}, ['Cz']) == []
    assert interpolated_channels({'interp_channels': None}, ['Cz']) == []
    assert interpolated_channels(None, ['Cz']) == []
    # a selected name missing from chan_name still counts, after the rest
    assert interpolated_channels({'chan_name': ['Fz'], 'interp_channels': ['Fz', 'E9']},
                                 ['E9', 'Fz']) == ['Fz', 'E9']

    import types
    DS = types.SimpleNamespace(header=header)
    log = logging.getLogger('tw_interp_test')
    with Capture(log) as cap:
        assert warn_interpolated_channels(DS, ['Fz', 'Cz', 'Pz'], log) == ['Cz', 'Pz']
        assert warn_interpolated_channels(DS, ['Fz'], log) == []
        assert warn_interpolated_channels(DS, None, log) == []
        assert warn_interpolated_channels(object(), ['Cz'], log) == []
    warnings = cap.messages(logging.WARNING)
    assert warnings == ["2 of 3 selected channels are interpolated in the file "
                        "(signal reconstructed from neighbours): Cz, Pz"], warnings
    print("   [ok] filtering, file order, one WARNING only when there is one")


# ---------------------------------------------------------------- detection

def _annotate(path, tmp):
    from turtlewave_hdEEG.annotation import CustomAnnotations, XLAnnotations
    from turtlewave_hdEEG.dataset import LargeDataset
    dataset = LargeDataset(path)
    xml = os.path.join(tmp, 'sub-T_scoring.xml')
    xl = XLAnnotations(dataset, xml, rater_name='tester')
    assert xl.process_all() is True, "header staging was not imported"
    return dataset, CustomAnnotations(xml)


def _runs(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT event_type, interpolated_channels, params_json "
            "FROM detection_runs ORDER BY timestamp").fetchall()
    finally:
        conn.close()


def test_detectors_warn_and_record():
    """One WARNING per run, and the list in detection_runs and params_json."""
    print("\n4. Detectors flag and record interpolated channels:")
    from turtlewave_hdEEG import ParalEvents, ParalSWA
    from turtlewave_hdEEG.kcomplexprocessor import ParalKC
    with Workdir() as tmp:
        dataset, annot = _annotate(_write(tmp, interp=['Cz']), tmp)
        assert dataset.header['interp_channels'] == ['Cz']
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out)
        db = os.path.join(out, 'neural_events.db')
        kw = dict(chan=LABELS, stage=['NREM2'], json_dir=out, db_path=db,
                  subject='sub-T', cat=(1, 1, 1, 0))
        runs = [
            ('spindle', ParalEvents, 'detect_spindles',
             dict(method='Moelle2011', frequency=(11, 16))),
            ('slow_wave', ParalSWA, 'detect_slow_waves',
             dict(method='AASM/Massimini2004', frequency=(0.5, 4))),
            ('k_complex', ParalKC, 'detect_kcomplexes',
             dict(method='AASM/Massimini2004', frequency=(0.5, 4))),
        ]
        counts = {}
        for event_type, cls, fn, extra in runs:
            proc = cls(dataset, annot, log_level=logging.WARNING)
            with Capture(proc.logger) as cap:
                counts[event_type] = len(getattr(proc, fn)(**kw, **extra))
            hits = [m for m in cap.messages(logging.WARNING) if WARNING_TEXT in m]
            assert hits == ["1 of 3 selected channels are interpolated in the "
                            "file (signal reconstructed from neighbours): Cz"], (
                event_type, cap.messages(logging.WARNING))
        rows = _runs(db)
        assert [r[0] for r in rows] == ['spindle', 'slow_wave', 'k_complex'], rows
        for event_type, stored, params in rows:
            assert json.loads(stored) == ['Cz'], (event_type, stored)
            assert json.loads(params)['interpolated_channels'] == ['Cz'], event_type
        print(f"   [ok] WARNING once per run; detection_runs + params_json == ['Cz'] "
              f"(events: {counts})")

        # PAC: the warning only (it has no detection_runs row)
        from turtlewave_hdEEG import ParalPAC
        pac = ParalPAC(dataset, annot, rootpath=tmp, log_level=logging.WARNING)
        with Capture(pac.logger) as cap:
            try:
                pac.analyze_pac(chan=['Cz', 'Pz'], stage=['NREM2'], db_path=db,
                                out_dir=os.path.join(tmp, 'pac'),
                                event_type='slow_wave', write_db=False,
                                event_opts={'sw_method': 'AASM/Massimini2004',
                                            'sw_freq_range': (0.5, 4)})
            except Exception as err:  # the analysis itself is not under test
                print(f"   (PAC run ended with {type(err).__name__}: {err})")
        hits = [m for m in cap.messages(logging.WARNING) if WARNING_TEXT in m]
        assert hits == ["1 of 2 selected channels are interpolated in the file "
                        "(signal reconstructed from neighbours): Cz"], cap.messages()
        print("   [ok] analyze_pac logs the WARNING")


def test_no_interpolation_records_empty_list():
    """A clean selection logs nothing and stores '[]', not NULL."""
    print("\n5. No interpolated channel selected:")
    from turtlewave_hdEEG import ParalEvents
    with Workdir() as tmp:
        dataset, annot = _annotate(_write(tmp, interp=['Cz']), tmp)
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out)
        db = os.path.join(out, 'neural_events.db')
        proc = ParalEvents(dataset, annot, log_level=logging.WARNING)
        with Capture(proc.logger) as cap:
            proc.detect_spindles(method='Moelle2011', frequency=(11, 16),
                                 chan=['Fz', 'Pz'], stage=['NREM2'],
                                 json_dir=out, db_path=db, subject='sub-T',
                                 cat=(1, 1, 1, 0))
        assert not [m for m in cap.messages() if WARNING_TEXT in m], cap.messages()
        (_, stored, params), = _runs(db)
        assert stored == '[]' and json.loads(params)['interpolated_channels'] == []
    print("   [ok] no WARNING; detection_runs.interpolated_channels == '[]'")


def test_existing_database_gains_column():
    """The additive migration: old rows stay NULL, new rows are filled."""
    print("\n6. Existing database migrated:")
    from turtlewave_hdEEG import dbwrite
    with Workdir() as tmp:
        db = os.path.join(tmp, 'old.db')
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE detection_runs (run_id TEXT PRIMARY KEY, "
                     "subject TEXT, event_type TEXT, method TEXT, citation TEXT, "
                     "params_json TEXT, ref_chan TEXT, polar TEXT, stages TEXT, "
                     "reject_types TEXT, reject_artifacts INTEGER, "
                     "reject_arousals INTEGER, turtlewave_version TEXT, "
                     "wonambi_version TEXT, numpy_version TEXT, git_sha TEXT, "
                     "timestamp TEXT)")
        conn.execute("INSERT INTO detection_runs (run_id, event_type) "
                     "VALUES ('old', 'spindle')")
        conn.commit()
        dbwrite.ensure_direct_write_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detection_runs)")]
        assert 'interpolated_channels' in cols, cols
        dbwrite.record_run(conn, 'new', 'spindle', 'Moelle2011', '', '{}', [],
                           'normal', ['NREM2'], interpolated_channels=['Cz', 'Pz'])
        dbwrite.record_run(conn, 'unchecked', 'spindle', 'Moelle2011', '', '{}',
                           [], 'normal', ['NREM2'])
        got = dict(conn.execute(
            "SELECT run_id, interpolated_channels FROM detection_runs"))
        conn.close()
        assert got == {'old': None, 'new': '["Cz", "Pz"]', 'unchecked': None}, got
    print("   [ok] column added; old and unchecked rows NULL; new row a JSON list")


TESTS = [
    test_header_key_every_layout,
    test_non_set_file_has_empty_list,
    test_interpolated_channels_helper,
    test_detectors_warn_and_record,
    test_no_interpolation_records_empty_list,
    test_existing_database_gains_column,
]


if __name__ == "__main__":
    print("TESTING interpolated channels")
    print("=============================")
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
