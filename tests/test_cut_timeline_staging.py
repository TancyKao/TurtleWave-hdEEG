#!/usr/bin/env python3
"""Cut recordings: the time map, the regular-grid relabelling and the choice of
staging source.

A recording with data removed upstream (EEGLAB ``boundary`` events) keeps its
full-night hypnogram in ``etc.stages`` while the signal is shorter.
``turtlewave_hdEEG.timeline`` converts between the cut and the original time
base, and ``XLAnnotations.add_stages_from_header`` picks one of four staging
sources:

1. ``etc.stages`` as stored, when it is at most one epoch longer than the
   signal (uncut files; identical to the behaviour before cut support);
2. ``etc.stages`` through the boundary time map, when it is longer and the map
   accounts for the difference exactly;
3. the stage events (``wake``, ``n1``, ``n2``, ``n3``, ``rem``, ``ns``), when
   there is no usable map or no ``etc.stages``;
4. nothing: return ``False`` and log an ERROR rather than import stages that
   would sit on the wrong part of the signal.

Layout of the file:

* time-map tests (pure numpy, no files);
* ``regrid_stages`` tests;
* source-choice tests, driven through ``XLAnnotations`` on fixture ``.set``
  files. These assert which source was used, the return value, the sidecar
  and that epochs were written, but not the exact epoch layout;
* ``test_grid_*``: the exact epoch layout. Staging is currently a regular
  30 s grid over the cut file (final partial epoch Undefined). If the layout
  changes to exact variable-length epochs, replace these tests and leave the
  rest.

Run standalone: ``python tests/test_cut_timeline_staging.py``. Exits non-zero
if any test fails.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eeglab_fixture as fx  # noqa: E402
from turtlewave_hdEEG.annotation import (  # noqa: E402
    STAGING_SOURCE_EVENTS, STAGING_SOURCE_HEADER, STAGING_SOURCE_TIME_MAP,
    XLAnnotations)
from turtlewave_hdEEG.dataset import LargeDataset  # noqa: E402
from turtlewave_hdEEG.timeline import (  # noqa: E402
    RecordingTimeline, boundaries_from_events, canonical_stage_code,
    regrid_stages, sidecar_path, stage_event_intervals, stage_events_from_header)

FS = 100.0


class Workdir:
    """Temporary directory removed on exit."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_timeline_')
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


class LogCapture(logging.Handler):
    """Collect records from ``turtlewave_hdEEG.annotation`` at DEBUG and up."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.logger = logging.getLogger('turtlewave_hdEEG.annotation')

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self._old_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self._old_level)

    def messages(self, level=None):
        return [r.getMessage() for r in self.records
                if level is None or r.levelno == level]


def _timeline(cut_onsets, removed, n_samples, stages=None, s_freq=FS):
    return RecordingTimeline(cut_onsets, removed, s_freq, n_samples, stages=stages)


# ------------------------------------------------------------------ time map

def test_time_map_round_trip_and_clamping():
    """cut -> original -> cut is the identity; times in removed data clamp."""
    print("\n1. Time map: round trip and clamping inside removed data:")
    # splices at cut 100 s (50 s removed: original 100-150) and cut 300 s
    # (20 s removed: original 350-370); 600 s of signal, 670 s of night
    tl = _timeline([100.0, 300.0], [50.0, 20.0], 60000)
    assert tl.n_boundaries == 2
    assert tl.removed_seconds == 70.0 and tl.cut_seconds == 600.0
    assert tl.original_seconds == 670.0
    assert list(tl.original_onsets) == [100.0, 350.0], tl.original_onsets

    expected = {0.0: 0.0, 99.99: 99.99, 100.0: 150.0, 250.0: 300.0,
                299.99: 349.99, 300.0: 370.0, 599.0: 669.0}
    for cut, orig in expected.items():
        got = tl.cut_to_original(cut)
        assert abs(got - orig) < 1e-9, (cut, got, orig)
        assert isinstance(got, float)
    print("   [ok] cut_to_original: a time at a splice is the first sample after the gap")

    for cut in [0.0, 12.5, 99.99, 100.0, 100.01, 250.0, 299.99, 300.0, 450.0, 599.99]:
        back = tl.original_to_cut(tl.cut_to_original(cut))
        assert abs(back - cut) < 1e-9, (cut, back)
    print("   [ok] original_to_cut(cut_to_original(t)) == t on 10 points")

    inside = {100.0: 100.0, 125.0: 100.0, 149.999: 100.0,   # first gap
              350.0: 300.0, 360.0: 300.0, 369.999: 300.0}   # second gap
    for orig, cut in inside.items():
        got = tl.original_to_cut(orig)
        assert abs(got - cut) < 1e-9, (orig, got, cut)
    edges = {99.999: 99.999, 150.0: 100.0, 150.5: 100.5, 370.0: 300.0, 400.0: 330.0}
    for orig, cut in edges.items():
        got = tl.original_to_cut(orig)
        assert abs(got - cut) < 1e-9, (orig, got, cut)
    print("   [ok] inside a removed span -> the splice; at its end -> the splice; after -> shifted")

    arr = tl.original_to_cut(np.array([50.0, 120.0, 150.0, 360.0, 400.0]))
    assert isinstance(arr, np.ndarray) and arr.shape == (5,)
    assert np.allclose(arr, [50.0, 100.0, 100.0, 300.0, 330.0]), arr
    arr2 = tl.cut_to_original(np.array([[0.0, 100.0], [300.0, 400.0]]))
    assert arr2.shape == (2, 2) and np.allclose(arr2, [[0, 150], [370, 470]]), arr2
    print("   [ok] array input keeps its shape")

    # the map is monotone: a later original time never maps to an earlier cut time
    grid = np.linspace(0, 669.0, 4001)
    mapped = tl.original_to_cut(grid)
    assert np.all(np.diff(mapped) >= -1e-9), "original_to_cut is not monotone"
    print("   [ok] original_to_cut is non-decreasing over the whole night")


def test_time_map_without_boundaries():
    """No boundary events: both directions are the identity."""
    print("\n2. Time map with no boundaries:")
    tl = _timeline([], [], 30000)
    assert tl.n_boundaries == 0 and tl.removed_seconds == 0.0
    assert tl.original_seconds == tl.cut_seconds == 300.0
    for t in (0.0, 1.5, 299.0):
        assert tl.cut_to_original(t) == t and tl.original_to_cut(t) == t
    assert np.allclose(tl.original_to_cut(np.array([0.0, 10.0])), [0.0, 10.0])
    assert np.allclose(tl.cut_to_original(np.array([0.0, 10.0])), [0.0, 10.0])
    print("   [ok] identity for scalars and arrays")


def test_time_map_splice_at_time_zero():
    """A splice at cut time 0 (data removed before the first sample)."""
    print("\n3. Splice at time 0:")
    tl = _timeline([0.0], [10.0], 20000)          # night starts 10 s before the file
    assert tl.original_onsets[0] == 0.0
    assert tl.cut_to_original(0.0) == 10.0
    assert tl.cut_to_original(5.0) == 15.0
    assert tl.original_to_cut(0.0) == 0.0
    assert tl.original_to_cut(9.999) == 0.0
    assert tl.original_to_cut(10.0) == 0.0
    assert abs(tl.original_to_cut(20.0) - 10.0) < 1e-12
    print("   [ok] cut 0 -> original 10; original 0-10 clamps to 0")

    # stage intervals start at the original 10 s: epoch 0 survives 20 s of 30
    tl = _timeline([0.0], [10.0], 6000, stages=['W', '1', '2'])      # 60 s cut, 70 s night
    ivals = tl.stage_intervals_cut()
    assert ivals[0] == (0.0, 20.0, 'W'), ivals[0]
    assert ivals[1] == (20.0, 50.0, '1'), ivals[1]
    assert ivals[2] == (50.0, 60.0, '2'), ivals[2]
    assert len(ivals) == 3, ivals
    print(f"   [ok] stage_intervals_cut with a splice at 0: {ivals}")


def test_time_map_two_splices_at_same_cut_onset():
    """Two boundary events at one cut time are one gap of the summed length."""
    print("\n4. Two splices at the same cut onset:")
    tl = _timeline([100.0, 100.0], [10.0, 5.0], 30000)
    assert tl.n_boundaries == 2 and tl.removed_seconds == 15.0
    assert list(tl.original_onsets) == [100.0, 110.0]
    assert tl.cut_to_original(99.0) == 99.0
    assert tl.cut_to_original(100.0) == 115.0
    for orig in (100.0, 105.0, 110.0, 112.0, 114.999, 115.0):
        assert abs(tl.original_to_cut(orig) - 100.0) < 1e-9, orig
    assert abs(tl.original_to_cut(116.0) - 101.0) < 1e-9
    assert abs(tl.original_to_cut(99.0) - 99.0) < 1e-9
    print("   [ok] original 100-115 all map to cut 100; 116 -> 101")

    tl = _timeline([100.0, 100.0], [10.0, 5.0], 12000, stages=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    ivals = tl.stage_intervals_cut()
    # cut 0-100 <- original 0-100; cut 100-120 <- original 115-135
    assert ivals[0][:2] == (0.0, 30.0) and ivals[3] == (90.0, 100.0, 'D'), ivals
    assert ivals[4] == (100.0, 105.0, 'D'), ivals[4]         # original 115-120, still epoch 3
    assert ivals[5] == (105.0, 120.0, 'E'), ivals[5]         # original 120-135
    print("   [ok] stage_intervals_cut resumes at original 115 after the double splice")


def test_time_map_zero_length_removal():
    """A boundary that removed nothing (duration 0 or missing) is the identity."""
    print("\n5. Zero-length removal:")
    tl = _timeline([100.0], [0.0], 30000)
    assert tl.n_boundaries == 1 and tl.removed_seconds == 0.0
    for t in (0.0, 99.999, 100.0, 250.0):
        assert tl.cut_to_original(t) == t, t
        assert tl.original_to_cut(t) == t, t
    print("   [ok] one splice, nothing removed: identity, still counted as a boundary")

    # duration missing / NaN / negative in the file counts as 0 s removed
    event_info = {'onsets': [1001.5, 2001.5, 3001.5, 4001.5],
                  'types': ['boundary'] * 4,
                  'durations': [None, float('nan'), -50, 0]}
    cut, removed = boundaries_from_events(event_info, 100.0)
    assert len(cut) == 4 and list(removed) == [0.0] * 4, (cut, removed)
    print("   [ok] boundary durations None / NaN / negative / 0 -> 0 s removed")


def test_boundaries_from_events():
    """Durations are samples; onsets are latency - 0.5; result is sorted."""
    print("\n6. boundaries_from_events and from_header:")
    event_info = {
        'onsets': [20000.5, 101.0, 10000.5, 5.0],
        'types': ['Boundary', 'n2', 'boundary', 'arousal'],
        'durations': [500.0, 3000.0, 2000.0, 100.0],
    }
    cut, removed = boundaries_from_events(event_info, 100.0)
    assert np.allclose(cut, [100.0, 200.0]), cut         # (10000.5-0.5)/100, (20000.5-0.5)/100
    assert np.allclose(removed, [20.0, 5.0]), removed    # 2000/100, 500/100: samples, not seconds
    print(f"   [ok] cut onsets {list(cut)}, removed {list(removed)} (sorted, samples/s_freq)")

    assert boundaries_from_events(None, 100.0)[0].size == 0
    assert boundaries_from_events({}, 100.0)[0].size == 0
    assert boundaries_from_events({'onsets': [1], 'types': ['n2'], 'durations': [1]}, 100.0)[0].size == 0
    print("   [ok] no events / no boundary events -> empty")

    header = {'event': event_info, 'stages': np.array(['W', ' 1', '2'])}
    tl = RecordingTimeline.from_header(header, 100.0, 50000)
    assert tl.n_boundaries == 2 and abs(tl.removed_seconds - 25.0) < 1e-9
    assert tl.fullnight_hypnogram() == ['W', '1', '2'], tl.fullnight_hypnogram()
    tl2 = RecordingTimeline.from_header({}, 100.0, 50000)
    assert tl2.n_boundaries == 0 and tl2.fullnight_hypnogram() == [] and not tl2.is_consistent
    print("   [ok] from_header: stages stripped to strings; empty header is an empty map")

    hyp = tl.fullnight_hypnogram()
    hyp.append('X')
    assert tl.fullnight_hypnogram() == ['W', '1', '2'], "fullnight_hypnogram must return a copy"
    print("   [ok] fullnight_hypnogram returns a copy")


def test_is_consistent_edges():
    """``-tol <= T_cut + removed - 30 * n_stages < 30 + tol`` at both edges,
    ``tol`` = one sample per boundary event."""
    print("\n7. RecordingTimeline.is_consistent at both edges:")
    n_stages = 20                                   # 600 s of stages
    # One boundary at 100 Hz: tolerance one sample (0.01 s) at both edges.
    for delta, expected in [(0.0, True), (0.01, True), (29.99, True), (30.0, True),
                            (30.02, False), (45.0, False), (-0.01, True),
                            (-0.02, False), (-30.0, False)]:
        removed_samples = round((100.0 + delta) * FS)
        tl = _timeline([100.0], [removed_samples / FS], 50000, stages=['W'] * n_stages)
        diff = tl.original_seconds - 30 * n_stages
        assert abs(diff - delta) < 1e-6, (diff, delta)
        assert tl.is_consistent is expected, (delta, tl.is_consistent, expected)
        print(f"   [ok] night - stages = {delta:+.2f} s -> {expected}")

    assert not _timeline([100.0], [100.0], 50000, stages=None).is_consistent
    assert not _timeline([100.0], [100.0], 50000, stages=[]).is_consistent
    print("   [ok] no stages -> False")

    # tolerance scales with the number of boundary events: 5 splices each one
    # sample short leave the night 0.05 s short of the epoch multiple
    tl = _timeline([10.0, 20.0, 30.0, 40.0, 50.0], [19.99, 19.99, 19.99, 19.99, 19.99],
                   50000, stages=['W'] * 20)
    assert abs(tl.consistency_tolerance - 0.05) < 1e-12, tl.consistency_tolerance
    assert tl.is_consistent, tl.original_seconds - 600
    tl = _timeline([10.0, 20.0, 30.0, 40.0, 50.0], [19.98, 19.99, 19.99, 19.99, 19.99],
                   50000, stages=['W'] * 20)
    assert not tl.is_consistent, tl.original_seconds - 600
    print("   [ok] tolerance is one sample per boundary event (5 splices -> 0.05 s)")

    # the numbers from the Compumedics example: 858 epochs, 19,308.1 s + 6,460.1 s
    n = round(19308.1 * 250)
    tl = _timeline([1000.0], [6460.1], n, stages=['N2'] * 858, s_freq=250.0)
    assert tl.is_consistent, tl.original_seconds - 858 * 30
    print("   [ok] reference-file numbers (858 epochs, +28.2 s) are consistent")


def test_stage_intervals_cut_index_arithmetic():
    """Every cut-time sample gets the stage of the original epoch it came from."""
    print("\n8. stage_intervals_cut against an independent oracle:")
    cases = {
        'one splice mid-epoch': ([100.0], [50.0], 600.0),
        'two splices': ([100.0, 300.0], [50.0, 20.0], 600.0),
        'splice on an epoch edge': ([90.0], [60.0], 300.0),
        'splice at 0': ([0.0], [40.0], 300.0),
        'double splice': ([100.0, 100.0], [10.0, 5.0], 300.0),
        'zero length': ([100.0], [0.0], 300.0),
        'no splice': ([], [], 300.0),
        'splice 1 s from the end': ([299.0], [30.0], 300.0),
    }
    for label, (cuts, removed, T) in cases.items():
        n_ep = int(np.ceil((T + sum(removed)) / 30.0)) + 2
        stages = [f's{i}' for i in range(n_ep)]
        tl = _timeline(cuts, removed, int(T * FS), stages=stages)
        ivals = tl.stage_intervals_cut()

        # oracle: walk the cut file in 0.25 s steps and use the explicit segment list
        seg = []                      # (cut_start, cut_end, original_start)
        pos, orig = 0.0, 0.0
        for c, r in zip(cuts, removed):
            if c > pos:
                seg.append((pos, c, orig))
            orig += (c - pos) + r
            pos = c
        seg.append((pos, T, orig))
        n_checked = 0
        for t in np.arange(0.125, T, 0.25):
            o = next(o0 + (t - c0) for c0, c1, o0 in seg if c0 <= t < c1)
            want = stages[int(o // 30)]
            hit = [s for a, b, s in ivals if a <= t < b]
            assert hit == [want], (label, t, o, hit, want)
            n_checked += 1
        # intervals never overlap and stay inside the file
        flat = sorted((a, b) for a, b, _ in ivals)
        assert all(b1 <= a2 + 1e-9 for (_, b1), (a2, _) in zip(flat, flat[1:])), (label, flat)
        assert flat[0][0] >= 0 and flat[-1][1] <= T + 1e-9, (label, flat)
        print(f"   [ok] {label}: {n_checked} sample points, {len(ivals)} intervals")

    # original time past the last stage has no label
    tl = _timeline([], [], 12000, stages=['W', '1'])          # 120 s of signal, 60 s of stages
    ivals = tl.stage_intervals_cut()
    assert ivals == [(0.0, 30.0, 'W'), (30.0, 60.0, '1')], ivals
    print("   [ok] signal longer than etc.stages: nothing labelled past the last stage")


def test_compare_stage_events():
    """Agreement between stage events and etc.stages read through the map."""
    print("\n9. compare_stage_events:")
    stages = ['W', '1', '2', '3', 'R', 'W', '2', '2']          # 240 s
    # 30 s removed at cut 100 -> original 100-130; 210 s of signal
    tl = _timeline([100.0], [30.0], 21000, stages=stages)
    assert tl.is_consistent
    events = [(0.0, 'W'), (30.0, 'N1'), (60.0, 'N2'),           # epochs 0-2 ('N1' == '1')
              (90.0, 'N3'),                                     # epoch 3
              (120.0, 'R'),                                     # cut 120 -> original 150: epoch 5, stage 'W'
              (150.0, 'W')]                                     # cut 150 -> original 180: epoch 6, stage '2'
    n_cmp, n_bad = tl.compare_stage_events(events)
    assert (n_cmp, n_bad) == (6, 2), (n_cmp, n_bad)
    print(f"   [ok] 6 compared, 2 wrong codes -> {(n_cmp, n_bad)}")

    good = [(0.0, 'W'), (30.0, '1'), (60.0, '2'), (120.0, 'W'), (150.0, 'N2'), (180.0, 'N2')]
    assert tl.compare_stage_events(good) == (6, 0)
    print("   [ok] '1' vs 'N1' and '2' vs 'N2' agree (canonical codes)")

    # an onset 1 s off the grid counts as a disagreement, 0.4 s does not
    assert tl.compare_stage_events([(30.4, '1')]) == (1, 0)
    assert tl.compare_stage_events([(31.0, '1')]) == (1, 1)
    print("   [ok] onset 0.4 s off the epoch grid agrees; 1 s off does not")

    # events past the last stage are not compared; nothing to compare -> (0, 0)
    assert tl.compare_stage_events([(500.0, 'W')]) == (0, 0)
    assert tl.compare_stage_events([]) == (0, 0)
    assert _timeline([], [], 6000).compare_stage_events([(0.0, 'W')]) == (0, 0)
    print("   [ok] out of range / empty / no stages -> (0, 0)")


def test_stage_event_helpers():
    """Stage events from the header, and their intervals."""
    print("\n10. stage_events_from_header, stage_event_intervals, canonical_stage_code:")
    info = {'onsets': [301.0, 1.0, 101.0, 201.0, 401.0, 501.0, 601.0],
            'types': ['REM', 'Wake', 'N1', 'n2', 'boundary', 'ns', 'N3'],
            'durations': [3000] * 7}
    ev = stage_events_from_header(info, 100.0)
    assert ev == [(0.0, 'W'), (1.0, 'N1'), (2.0, 'N2'), (3.0, 'R'), (5.0, '?'), (6.0, 'N3')], ev
    assert stage_events_from_header(None, 100.0) == [] and stage_events_from_header({}, 100.0) == []
    print(f"   [ok] sorted (latency - 1)/fs, codes W/N1/N2/R/?/N3; boundary ignored")

    ev = [(0.0, 'W'), (10.0, 'N2'), (60.0, 'N3'), (200.0, 'R')]
    iv = stage_event_intervals(ev, T=215.0)
    assert iv == [(0.0, 10.0, 'W'), (10.0, 40.0, 'N2'), (60.0, 90.0, 'N3'),
                  (200.0, 215.0, 'R')], iv
    assert stage_event_intervals([(100.0, 'W')], T=100.0) == []
    assert stage_event_intervals([], T=100.0) == []
    print("   [ok] each interval ends at the next onset, +30 s, or the end of the file")

    for raw, want in [('1', 'N1'), ('2', 'N2'), ('3', 'N3'), ('4', 'N3'), ('0', 'W'),
                      ('N2', 'N2'), ('n2', 'N2'), (' 2 ', 'N2'), ('r', 'R'), ('R', 'R'),
                      ('?', '?'), ('W', 'W'), ('weird', 'weird')]:
        assert canonical_stage_code(raw) == want, (raw, canonical_stage_code(raw), want)
    print("   [ok] canonical_stage_code: numeric and N-style spellings agree")


def test_timeline_json_round_trip():
    """The sidecar keeps everything the cycle code will need."""
    print("\n11. to_json / from_json / sidecar_path:")
    with Workdir() as tmp:
        tl = RecordingTimeline([100.0, 300.0], [50.0, 20.0], 100.0, 60000,
                               stages=['W', '1', '2'], meta={'source': 'test', 'note': 'x'})
        path = tl.to_json(os.path.join(tmp, 't.json'))
        assert os.path.exists(path)
        raw = json.load(open(path))
        for key in ('schema', 's_freq', 'n_samples', 'epoch_length', 'cut_seconds',
                    'removed_seconds', 'original_seconds', 'n_boundaries', 'boundaries',
                    'fullnight_stages', 'source', 'note'):
            assert key in raw, key
        assert raw['boundaries'][1] == {'cut_onset_s': 300.0, 'original_onset_s': 350.0,
                                        'removed_s': 20.0}, raw['boundaries'][1]
        back = RecordingTimeline.from_json(path)
        assert back.n_boundaries == 2 and back.removed_seconds == 70.0
        assert back.fullnight_hypnogram() == ['W', '1', '2']
        assert back.meta == {'source': 'test', 'note': 'x'}, back.meta
        for t in (0.0, 99.0, 100.0, 450.0):
            assert back.cut_to_original(t) == tl.cut_to_original(t)
        for t in (0.0, 120.0, 360.0, 500.0):
            assert back.original_to_cut(t) == tl.original_to_cut(t)
        print("   [ok] round trip keeps boundaries, stages, meta and both conversions")

        assert sidecar_path('/a/b/sub-01_run-1.xml').as_posix() == '/a/b/sub-01_run-1_timeline.json'
        assert sidecar_path('x.xml').name == 'x_timeline.json'
        print("   [ok] sidecar_path: <xml stem>_timeline.json beside the annotation")


# ---------------------------------------------------------------- regrid_stages

def test_regrid_majority_tie_undefined_final_epoch():
    """Grid labelling rules (independent of the epoch layout in the XML)."""
    print("\n12. regrid_stages:")
    # majority: N2 20 s, N3 10 s in epoch 0
    got = regrid_stages([(0, 20, 'N2'), (20, 30, 'N3'), (30, 60, 'W')], T=60)
    assert got == ['N2', 'W'], got
    print("   [ok] the label with the largest overlap wins")

    # a label split into pieces counts as its total
    got = regrid_stages([(0, 8, 'A'), (8, 18, 'B'), (18, 30, 'A')], T=30)
    assert got == ['A'], got
    print("   [ok] pieces of one label add up (A 20 s vs B 10 s)")

    # tie -> the label that starts earlier in the epoch, whatever the list order
    for ivals in ([(0, 15, 'N2'), (15, 30, 'N3')], [(15, 30, 'N3'), (0, 15, 'N2')]):
        assert regrid_stages(ivals, T=30) == ['N2'], ivals
    assert regrid_stages([(0, 15, 'N2'), (15, 30, 'N3')], T=30) == ['N2']
    # N3 in two pieces (0-5, 20-30) ties N2 (5-20); N3 starts first
    assert regrid_stages([(0, 5, 'N3'), (5, 20, 'N2'), (20, 30, 'N3')], T=30) == ['N3']
    print("   [ok] tie goes to the label that starts earlier (either input order)")

    # under 50 % labelled -> undefined; exactly 50 % keeps the label
    assert regrid_stages([(0, 14.9, 'N2')], T=30) == ['?']
    assert regrid_stages([(0, 15.0, 'N2')], T=30) == ['N2']
    assert regrid_stages([(0, 7, 'N2'), (20, 27, 'N3')], T=30) == ['?'], "two labels, 14 s labelled in total"
    assert regrid_stages([(0, 10, 'N2'), (20, 30, 'N3')], T=30) == ['N2']
    assert regrid_stages([], T=60) == ['?', '?']
    print("   [ok] < 15 s labelled -> Undefined; 15 s keeps the label; no labels -> all Undefined")

    # final partial epoch is always undefined, even when fully labelled
    got = regrid_stages([(0, 75, 'N2')], T=75)
    assert got == ['N2', 'N2', '?'], got
    got = regrid_stages([(0, 61, 'N2')], T=61)
    assert got == ['N2', 'N2', '?'], got
    got = regrid_stages([(0, 60, 'N2')], T=60)
    assert got == ['N2', 'N2'], got
    got = regrid_stages([(0, 10, 'N2')], T=10)
    assert got == ['?'], got
    assert len(regrid_stages([], T=340.0)) == 12 and len(regrid_stages([], T=360.0)) == 12
    print("   [ok] final partial epoch Undefined; count is ceil(T / 30)")

    # intervals beyond T or before 0 are clipped; epoch_length and undefined are parameters
    assert regrid_stages([(-10, 5, 'W'), (5, 500, 'N2')], T=60) == ['N2', 'N2']
    assert regrid_stages([(0, 20, 'A'), (20, 40, 'B')], T=40, epoch_length=20, undefined='U') == ['A', 'B']
    assert regrid_stages([(0, 5, 'A')], T=40, epoch_length=20, undefined='U') == ['U', 'U']
    print("   [ok] clipping to [0, T); epoch_length and undefined arguments")


# ------------------------------------------------- fixture recordings on disk

#: Original night: 13 epochs (390 s) plus a 10 s tail. A gap of 60 s was cut at
#: cut time 100 s (original 100-160), so the signal is 340 s and 12 grid epochs.
#: The gap has to be well over one epoch, otherwise etc.stages is within the
#: one-epoch tolerance of the signal and the as-stored path is (correctly) used.
NIGHT_NUMERIC = ['W', '1', '2', '2', '3', '3', '2', '2', 'R', 'R', 'W', 'W', '2']
NIGHT_NAMED = ['W', 'N1', 'N2', 'N2', 'N3', 'N3', 'N2', 'N2', 'R', 'R', 'W', 'W', 'N2']
CUT_AT, GAP, T_CUT = 100.0, 60.0, 340.0
EVENT_NAME = {'W': 'wake', 'N1': 'n1', 'N2': 'n2', 'N3': 'n3', 'R': 'rem', '?': 'ns'}


def _stage_events(stages, cut_at=CUT_AT, gap=GAP, wrong=()):
    """Stage events as the upstream cut left them: none for epochs whose onset
    fell in removed data. ``wrong`` lists epoch indices given the wrong stage."""
    out = []
    for i, code in enumerate(stages):
        orig = 30.0 * i
        if cut_at <= orig < cut_at + gap:
            continue
        cut = orig if orig < cut_at else orig - gap
        canon = canonical_stage_code(code)
        if i in wrong:
            canon = 'N1' if canon != 'N1' else 'N2'
        out.append(fx.event(cut * FS + 1, EVENT_NAME[canon], 30 * FS))
    return out


def _write_cut(tmp, name='cut.set', stages=NIGHT_NUMERIC, events=True, wrong=(),
               boundary=True, n_samples=int(T_CUT * FS), layout='root', v73=True,
               with_stages=True, event_stages=None, cut_at=CUT_AT):
    """The 340 s cut recording, with the 13-epoch night as ``etc.stages``.

    ``event_stages`` are the stages the stage events are generated from
    (default ``stages``)."""
    ev = []
    if boundary:
        ev.append(fx.event(cut_at * FS + 0.5, 'boundary', GAP * FS))
    if events:
        ev += _stage_events(stages if event_stages is None else event_stages,
                              cut_at=cut_at, wrong=wrong)
    ev.append(fx.event(50 * FS, 'arousal', 3 * FS))       # a non-stage event
    path = os.path.join(tmp, name)
    fx.write_set(path, layout, v73, n_samples=n_samples, srate=FS,
                 labels=['Cz', 'Pz'], types=None, ref=None, n_good=None,
                 stages=stages if with_stages else None, events=ev)
    return path


def _write_uncut(tmp, n_stages, name='uncut.set', events=None, n_samples=36000):
    """A 360 s recording with no boundary events and ``n_stages`` epochs."""
    stages = [['W', '1', '2', '3', 'R'][i % 5] for i in range(n_stages)]
    path = os.path.join(tmp, name)
    fx.write_set(path, 'root', True, n_samples=n_samples, srate=FS,
                 labels=['Cz', 'Pz'], types=None, ref=None, n_good=None,
                 stages=stages, events=events)
    return path, stages


def _annotate(path, annot_name, tmp):
    annot_file = os.path.join(tmp, annot_name)
    xl = XLAnnotations(LargeDataset(path), annot_file, rater_name='tester')
    return xl, annot_file


def _epochs(annot_file):
    from wonambi.attr import Annotations
    ann = Annotations(annot_file, rater_name='tester')
    return [(round(float(e['start']), 3), round(float(e['end']), 3), e['stage'])
            for e in ann.get_epochs()]


def _nothing_imported(annot_file):
    """A fresh annotation file holds ``Unknown`` epochs; staging replaces them."""
    return all(stage == 'Unknown' for _, _, stage in _epochs(annot_file))


def _stage_names(annot_file):
    return [s for _, _, s in _epochs(annot_file)]


WONAMBI_NAME = {'W': 'Wake', '1': 'NREM1', '2': 'NREM2', '3': 'NREM3', 'R': 'REM',
                'N1': 'NREM1', 'N2': 'NREM2', 'N3': 'NREM3', '?': 'Undefined'}


def _oracle_time_map_grid(night):
    """Expected 30 s grid for the fixture cut, worked out by hand.

    Cut epochs 0-2 are original epochs 0-2. Cut epoch 3 (90-120 s) holds 10 s
    of original epoch 3 (90-100 s) and 20 s of original epoch 5 (160-180 s), so
    it takes epoch 5. From cut epoch 4 on the shift is exactly two epochs, so
    cut epoch k is original epoch k + 2. The last cut epoch is a 10 s
    remainder and is Undefined.
    """
    return [night[0], night[1], night[2], night[5]] + [night[k + 2] for k in range(4, 11)] + ['?']


def _oracle_events_grid(night):
    """Expected grid from the stage events alone: the event of original epoch
    3 (cut 90 s) runs to the next surviving onset (cut 120 s), so cut epoch 3
    keeps original epoch 3, whereas the time map gives it original epoch 5."""
    return [night[0], night[1], night[2], night[3]] + [night[k + 2] for k in range(4, 11)] + ['?']


# ------------------------------------------------------- source choice tests

def test_source_as_stored_for_uncut_files():
    """Uncut file: etc.stages is imported as stored, with no sidecar."""
    print("\n13. Source 1, etc.stages as stored (uncut file):")
    with Workdir() as tmp:
        path, stages = _write_uncut(tmp, 12)
        xl, annot_file = _annotate(path, 'uncut.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is True
        assert any(STAGING_SOURCE_HEADER in m for m in log.messages(logging.INFO)), log.messages()
        assert not os.path.exists(sidecar_path(annot_file)), "an uncut file must not get a sidecar"
        assert not log.messages(logging.ERROR) and not log.messages(logging.WARNING)
        names = _stage_names(annot_file)
        assert names == [WONAMBI_NAME[s] for s in stages], names
        print(f"   [ok] 12 epochs, {STAGING_SOURCE_HEADER!r}, no sidecar, no warning")

        # stage events that disagree with etc.stages are not consulted
        events = [fx.event(1, 'wake', 3000), fx.event(3001, 'wake', 3000)]
        path, stages = _write_uncut(tmp, 12, name='uncut_ev.set', events=events)
        xl, annot_file = _annotate(path, 'uncut_ev.xml', tmp)
        assert xl.add_stages_from_header() is True
        assert _stage_names(annot_file) == [WONAMBI_NAME[s] for s in stages]
        print("   [ok] disagreeing stage events do not change an as-stored import")


def test_source_threshold_between_stored_and_time_map():
    """``30 * n_stages - T <= 30`` is the as-stored path; one epoch more is not."""
    print("\n14. Threshold between the as-stored path and the time map:")
    with Workdir() as tmp:
        # T = 360 s. 12 epochs: 0 s over. 13: exactly 30 s over -> still as stored.
        for n_stages, expect_ok in [(11, True), (12, True), (13, True), (14, False)]:
            path, stages = _write_uncut(tmp, n_stages, name=f'thr{n_stages}.set')
            xl, annot_file = _annotate(path, f'thr{n_stages}.xml', tmp)
            with LogCapture() as log:
                ok = xl.add_stages_from_header()
            assert ok is expect_ok, (n_stages, ok)
            if expect_ok:
                assert any(STAGING_SOURCE_HEADER in m for m in log.messages()), (n_stages, log.messages())
                assert len(_epochs(annot_file)) == n_stages, (n_stages, len(_epochs(annot_file)))
            else:
                assert log.messages(logging.ERROR), "an over-long etc.stages with no map must log an ERROR"
                assert _nothing_imported(annot_file), "nothing may be imported"
            print(f"   [ok] {n_stages} stages over 360 s -> "
                  f"{'as stored' if expect_ok else 'rejected (ERROR, nothing imported)'}")

        # the same comparison at sample precision, on a stub (no file needed)
        from types import SimpleNamespace
        for n_samples, n_stages, want in [(36000, 13, STAGING_SOURCE_HEADER),
                                          (35999, 13, None),      # 359.99 s: 30.01 s over
                                          (35001, 13, None),
                                          (39000, 13, STAGING_SOURCE_HEADER)]:
            xl = XLAnnotations.__new__(XLAnnotations)
            xl.dataset = SimpleNamespace(header={'stages': ['W'] * n_stages, 'n_samples': n_samples,
                                                 's_freq': FS}, sampling_rate=FS)
            with LogCapture() as log:
                choice = xl._choose_staging_source(xl.dataset.header)
            got = None if choice is None else choice[0]
            assert got == want, (n_samples, n_stages, got, want)
        print("   [ok] 359.99 s vs 13 epochs is just over the limit and is not read as stored")

        # signal length unknown -> etc.stages as stored
        xl = XLAnnotations.__new__(XLAnnotations)
        xl.dataset = SimpleNamespace(header={'stages': ['W'] * 500}, sampling_rate=None)
        choice = xl._choose_staging_source(xl.dataset.header)
        assert choice is not None and choice[0] == STAGING_SOURCE_HEADER and choice[2] is None
        print("   [ok] signal length unknown -> as stored")


def test_source_time_map_writes_sidecar():
    """Cut file with a consistent map: etc.stages through the map, sidecar written."""
    print("\n15. Source 2, etc.stages through the time map:")
    with Workdir() as tmp:
        path = _write_cut(tmp)
        xl, annot_file = _annotate(path, 'cut.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is True
        info = log.messages(logging.INFO)
        assert any(STAGING_SOURCE_TIME_MAP in m for m in info), info
        assert not log.messages(logging.WARNING) and not log.messages(logging.ERROR), log.messages()
        line = next(m for m in info if STAGING_SOURCE_TIME_MAP in m)
        assert '13 full-night epochs' in line and '1 boundary events removing 60.0 s' in line, line
        assert 'agree on 11 of 11' in line, line
        print(f"   [ok] INFO: {line}")

        sc = sidecar_path(annot_file)
        assert os.path.exists(sc), "the sidecar is missing"
        assert sc.name == 'cut_timeline.json'
        data = json.load(open(sc))
        assert data['source'] == STAGING_SOURCE_TIME_MAP
        assert data['fullnight_stages'] == NIGHT_NUMERIC
        assert data['n_boundaries'] == 1
        assert data['boundaries'] == [{'cut_onset_s': 100.0, 'original_onset_s': 100.0, 'removed_s': 60.0}], \
            data['boundaries']
        assert abs(data['cut_seconds'] - 340.0) < 1e-9 and abs(data['removed_seconds'] - 60.0) < 1e-9
        assert abs(data['original_seconds'] - 400.0) < 1e-9
        assert data['s_freq'] == FS and data['n_samples'] == 34000 and data['epoch_length'] == 30.0
        assert data['stage_events_compared'] == 11 and data['stage_events_disagreeing'] == 0
        assert data['rater'] == 'tester' and data['annotation_file'] == 'cut.xml'
        assert data['data_file'].endswith('cut.set')
        assert data['recording_start'] == '2022-09-12T22:28:52', data['recording_start']
        for key in ('schema', 'turtlewave_version', 'created', 'cut_stages'):
            assert key in data, key
        assert len(data['cut_stages']) == len(_epochs(annot_file)) > 0
        print(f"   [ok] sidecar keys: {sorted(data)}")

        # the sidecar loads back into a working time map
        tl = RecordingTimeline.from_json(sc)
        assert tl.is_consistent and tl.cut_to_original(120.0) == 180.0
        print("   [ok] sidecar loads back into a consistent RecordingTimeline")

        # a second run overwrites rather than duplicates
        assert xl.add_stages_from_header() is True
        assert os.path.exists(sc)


def test_source_time_map_warns_when_stage_events_disagree():
    """More than 1 % of stage events off the mapped etc.stages logs a WARNING."""
    print("\n16. Time map: stage events that disagree:")
    with Workdir() as tmp:
        path = _write_cut(tmp, wrong=(7,))                 # 1 of 11 events wrong (9 %)
        xl, annot_file = _annotate(path, 'wrong.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        warns = log.messages(logging.WARNING)
        assert len(warns) == 1 and '1 of 11 stage events' in warns[0] and 'disagree' in warns[0], warns
        data = json.load(open(sidecar_path(annot_file)))
        assert data['stage_events_disagreeing'] == 1 and data['stage_events_compared'] == 11
        assert data['source'] == STAGING_SOURCE_TIME_MAP, "etc.stages stays the source"
        print(f"   [ok] WARNING: {warns[0]}")

        # 0 disagreements never warns; no stage events at all never warns
        path = _write_cut(tmp, name='none.set', events=False)
        xl, annot_file = _annotate(path, 'none.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert not log.messages(logging.WARNING), log.messages()
        data = json.load(open(sidecar_path(annot_file)))
        assert data['stage_events_compared'] == 0
        print("   [ok] no stage events in the file: time map still used, no warning")


def test_source_small_removal_uses_time_map():
    """A cut of under one epoch still goes through the time map.

    Rule changed with the coordinator's fix 3 (was
    ``test_source_small_removal_stays_as_stored``, which pinned the as-stored
    path): the time map is used whenever boundary events removed data and the
    map is consistent, whatever the size. A file that lost 25 s at a splice
    has etc.stages within one epoch of the signal, and importing it as stored
    would put every stage after the splice up to one epoch early.
    """
    print("\n17a. A 25 s cut goes through the time map:")
    with Workdir() as tmp:
        stages = ['W', '1', '2', '3', 'R', 'W', '2', '2', '3', 'R', 'W', '1']    # 360 s
        ev = [fx.event(100 * FS + 0.5, 'boundary', 25 * FS)]
        path = os.path.join(tmp, 'small.set')
        fx.write_set(path, 'root', True, n_samples=int(345 * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=stages, events=ev)
        xl, annot_file = _annotate(path, 'small.xml', tmp)
        tl = RecordingTimeline.from_header(xl.dataset.header, FS, 34500)
        assert tl.removed_seconds == 25.0 and tl.is_consistent, "the time map itself is valid"
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_TIME_MAP in m for m in log.messages(logging.INFO)), log.messages()
        assert os.path.exists(sidecar_path(annot_file))
        # cut epoch k covers original [30k + 25, 30k + 55) after the splice at 100 s
        expected = ['Wake', 'NREM1', 'NREM2', 'REM', 'Wake', 'NREM2', 'NREM2',
                    'NREM3', 'REM', 'Wake', 'NREM1', 'Undefined']
        got = [st for _, _, st in _epochs(annot_file)]
        assert got == expected, got
        print("   [ok] time map, sidecar written, stages after the splice shifted by the 25 s cut")


def test_source_no_removed_time_stays_as_stored():
    """Boundary events that remove no time keep the as-stored path."""
    print("\n17b. Zero-length boundaries stay on the as-stored path:")
    with Workdir() as tmp:
        stages = ['W', '1', '2', '3', 'R', 'W', '2', '2', '3', 'R', 'W', '1']    # 360 s
        ev = [fx.event(100 * FS + 0.5, 'boundary', 0)]
        path = os.path.join(tmp, 'joined.set')
        fx.write_set(path, 'root', True, n_samples=int(360 * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=stages, events=ev)
        xl, annot_file = _annotate(path, 'joined.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_HEADER in m for m in log.messages(logging.INFO)), log.messages()
        assert not os.path.exists(sidecar_path(annot_file))
        print("   [ok] as stored, no sidecar")


def test_source_scorer_stopped_early_uses_time_map():
    """etc.stages shorter than the night: the stage events pick the time map.

    A 900 s night, 60 s cut at 100 s, and a scorer who stopped after 25
    epochs (750 s). The map is not "consistent" (the night runs 150 s past
    the last stage) and 25 epochs fit the 840 s signal, so the as-stored
    reading is also plausible; it would shift every epoch after the cut by
    60 s. The stage events (23 of them survive the cut) agree with the
    full-night reading only.
    """
    print("\n17c. Scorer stopped early: the stage events choose the time map:")
    with Workdir() as tmp:
        night = ['W', 'W', '1', '2', '2', '3', '3', '2', 'R', 'R', 'W', '1', '2',
                 '2', '3', '2', 'R', 'R', '2', '2', 'W', '1', '2', '3', 'R']   # 25 epochs
        ev = [fx.event(CUT_AT * FS + 0.5, 'boundary', GAP * FS)] + _stage_events(night)
        path = os.path.join(tmp, 'stopped.set')
        fx.write_set(path, 'root', True, n_samples=int(840 * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=night, events=ev)
        xl, annot_file = _annotate(path, 'stopped.xml', tmp)
        tl = RecordingTimeline.from_header(xl.dataset.header, FS, int(840 * FS))
        assert not tl.is_consistent, "night runs 150 s past the last stage"
        assert tl.compare_stage_events(stage_events_from_header(
            xl.dataset.header['event'], FS)) == (23, 0)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        info = log.messages(logging.INFO)
        assert any(STAGING_SOURCE_TIME_MAP in m for m in info), log.messages()
        assert os.path.exists(sidecar_path(annot_file))
        expected = [WONAMBI_NAME[c] for c in tl.cut_hypnogram()]
        assert _stage_names(annot_file) == expected, _stage_names(annot_file)
        as_stored = [WONAMBI_NAME[c] for c in night]
        assert _stage_names(annot_file)[:len(night)] != as_stored
        print(f"   [ok] time map chosen ({[m for m in info if 'agree' in m][0].split('; ')[-1]})")


def _cut_base_file(tmp, name, with_events):
    """etc.stages already rescored on the cut signal: 28 epochs for 840 s, with
    a 20 s cut at 100 s, so the full-night reading is also consistent."""
    stages = ['W', '1', '2', '2', '3', '3', '2', 'R', 'R', 'W', '2', '2', '3', '3',
              '2', 'R', 'W', '1', '2', '2', '3', '2', 'R', 'R', '2', '2', 'W', 'W']
    ev = [fx.event(100 * FS + 0.5, 'boundary', 20 * FS)]
    if with_events:
        ev += [fx.event(30 * i * FS + 1, EVENT_NAME[canonical_stage_code(c)], 30 * FS)
               for i, c in enumerate(stages)]
    path = os.path.join(tmp, name)
    fx.write_set(path, 'root', True, n_samples=int(840 * FS), srate=FS, labels=['Cz'],
                 types=None, ref=None, n_good=None, stages=stages, events=ev)
    return path, stages


def test_source_events_choose_as_stored():
    """Both readings fit; stage events on the cut grid choose as stored."""
    print("\n17d. Stage events support the as-stored reading:")
    with Workdir() as tmp:
        path, stages = _cut_base_file(tmp, 'rescored.set', with_events=True)
        xl, annot_file = _annotate(path, 'rescored.xml', tmp)
        tl = RecordingTimeline.from_header(xl.dataset.header, FS, int(840 * FS))
        assert tl.is_consistent, "the full-night reading is also consistent"
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        info = log.messages(logging.INFO)
        assert any(STAGING_SOURCE_HEADER in m and '28 of 28 as stored' in m
                   for m in info), log.messages()
        assert not os.path.exists(sidecar_path(annot_file))
        assert _stage_names(annot_file) == [WONAMBI_NAME[c] for c in stages]
        print("   [ok] as stored chosen (events agree 28 of 28), no sidecar")


def test_source_no_events_ambiguous_warns():
    """Both readings fit, no stage events, >= half an epoch removed: WARNING."""
    print("\n17e. Ambiguous readings without stage events warn:")
    with Workdir() as tmp:
        path, _ = _cut_base_file(tmp, 'noev.set', with_events=False)
        xl, annot_file = _annotate(path, 'noev.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_TIME_MAP in m for m in log.messages(logging.INFO))
        warns = log.messages(logging.WARNING)
        assert any('no stage events' in m for m in warns), log.messages()
        print("   [ok] time map (the default) with a WARNING")


def test_source_misstated_boundary_falls_back_to_events():
    """Boundary table records 5 s where 20 s was cut: stage events take over.

    Reviewer's probe. The as-stored reading is the only candidate (the map is
    5 s short of the stages) and disagrees on 9 of 13 events, well above
    STAGE_EVENT_DISAGREEMENT_REJECT, so it must not be imported.
    """
    print("\n17f. Misstated boundary duration falls back to the stage events:")
    with Workdir() as tmp:
        night = ['W', '1', '2', '2', '3', '3', '2', 'R', 'R', 'W', '2', '3', 'R']  # 390 s
        cut_at, gap, T = 100.0, 20.0, 380.0
        ev = [fx.event(cut_at * FS + 0.5, 'boundary', 5 * FS)]       # records 5 s only
        for i, code in enumerate(night):
            orig = 30.0 * i
            if cut_at <= orig < cut_at + gap:
                continue
            cut = orig if orig < cut_at else orig - gap
            ev.append(fx.event(cut * FS + 1, EVENT_NAME[canonical_stage_code(code)], 30 * FS))
        path = os.path.join(tmp, 'misstated.set')
        fx.write_set(path, 'root', True, n_samples=int(T * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=night, events=ev)
        xl, annot_file = _annotate(path, 'misstated.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        warns = log.messages(logging.WARNING)
        assert any('9 of 13 stage events' in m and 'Falling back' in m for m in warns), log.messages()
        assert any(STAGING_SOURCE_EVENTS in m for m in log.messages(logging.INFO)), log.messages()
        assert not os.path.exists(sidecar_path(annot_file))
        se = stage_events_from_header(xl.dataset.header['event'], FS)
        expected = [WONAMBI_NAME[c] for c in regrid_stages(stage_event_intervals(se, T), T)]
        got = _stage_names(annot_file)
        assert got == expected, got
        assert got[:len(night)] != [WONAMBI_NAME[c] for c in night]
        print(f"   [ok] {[m for m in warns if 'Falling back' in m][0][:60]}... -> stage events")


def test_source_uncut_unmarked_gap_warns_output_unchanged():
    """No boundary events, but a 20 s unmarked gap: WARNING, as-stored unchanged."""
    print("\n17g. Unmarked gap on an uncut file warns, output unchanged:")
    with Workdir() as tmp:
        stages = ['W', '1', '2', '2', '3', '3', '2', 'R', 'R', 'W', '2', '3']   # 360 s
        ev = []
        for i, code in enumerate(stages):
            orig = 30.0 * i
            if 100.0 <= orig < 120.0:
                continue
            cut = orig if orig < 100.0 else orig - 20.0
            ev.append(fx.event(cut * FS + 1, EVENT_NAME[canonical_stage_code(code)], 30 * FS))
        path = os.path.join(tmp, 'gap.set')
        fx.write_set(path, 'root', True, n_samples=int(340 * FS), srate=FS, labels=['Cz'],
                     types=None, ref=None, n_good=None, stages=stages, events=ev)
        xl, annot_file = _annotate(path, 'gap.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_HEADER in m for m in log.messages(logging.INFO)), log.messages()
        warns = log.messages(logging.WARNING)
        assert len(warns) == 1 and 'unmarked gap' in warns[0], warns
        assert _stage_names(annot_file) == [WONAMBI_NAME[c] for c in stages]
        assert not os.path.exists(sidecar_path(annot_file))
        print(f"   [ok] WARNING: {warns[0][:70]}...; epochs as stored")


def test_source_time_map_on_all_layouts():
    """The same cut recording stages identically from every file layout."""
    print("\n17. Time-map staging from all four layouts:")
    with Workdir() as tmp:
        results = {}
        for layout, v73 in fx.ALL_LAYOUTS:
            name = fx.layout_id(layout, v73).replace('/', '_')
            path = _write_cut(tmp, name=f'{name}.set', layout=layout, v73=v73)
            xl, annot_file = _annotate(path, f'{name}.xml', tmp)
            assert xl.add_stages_from_header() is True, name
            data = json.load(open(sidecar_path(annot_file)))
            assert data['source'] == STAGING_SOURCE_TIME_MAP, name
            results[name] = (_epochs(annot_file), data['boundaries'], data['fullnight_stages'])
        first = next(iter(results.values()))
        assert all(v == first for v in results.values()), results
        print(f"   [ok] {len(results)} layouts -> the same epochs, boundary table and hypnogram")


def test_source_stage_events_when_no_usable_map():
    """Stage events are the source when there is no map or no etc.stages."""
    print("\n18. Source 3, stage events:")
    with Workdir() as tmp:
        # (a) no etc.stages at all, boundary present
        path = _write_cut(tmp, name='noetc.set', with_stages=False)
        xl, annot_file = _annotate(path, 'noetc.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is True
        line = next(m for m in log.messages(logging.INFO) if STAGING_SOURCE_EVENTS in m)
        assert 'the header has no etc.stages' in line, line
        assert not os.path.exists(sidecar_path(annot_file)), "the events path writes no sidecar"
        assert len(_epochs(annot_file)) == 12
        print(f"   [ok] (a) no etc.stages: {line}")

        # (b) etc.stages from a different night: too long, and the map cannot explain it
        long_night = NIGHT_NUMERIC * 2                      # 780 s vs 400 s
        path = _write_cut(tmp, name='inconsistent.set', stages=long_night,
                          event_stages=NIGHT_NUMERIC)
        xl, annot_file = _annotate(path, 'inconsistent.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is True
        line = next(m for m in log.messages(logging.INFO) if STAGING_SOURCE_EVENTS in m)
        assert 'etc.stages has 26 epochs' in line and '400.0 s' in line, line
        assert not os.path.exists(sidecar_path(annot_file))
        print(f"   [ok] (b) inconsistent map: {line[:120]}...")

        # (c) etc.stages over-long but no boundary events, stage events present
        events = _stage_events(NIGHT_NUMERIC, cut_at=1e9)
        path, stages = _write_uncut(tmp, 14, name='over.set', events=events)
        xl, annot_file = _annotate(path, 'over.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is True
        assert any(STAGING_SOURCE_EVENTS in m for m in log.messages(logging.INFO))
        assert not os.path.exists(sidecar_path(annot_file))
        print("   [ok] (c) 14 epochs over 360 s, no boundaries, events present -> stage events")

        # (d) no etc.stages, no boundary: header['stages'] absent, events only
        path = os.path.join(tmp, 'evonly.set')
        fx.write_set(path, 'root', True, n_samples=36000, srate=FS, labels=['Cz'], types=None,
                     ref=None, n_good=None, stages=None, events=_stage_events(NIGHT_NUMERIC, cut_at=1e9))
        xl, annot_file = _annotate(path, 'evonly.xml', tmp)
        assert 'stages' not in xl.dataset.header
        assert xl.add_stages_from_header() is True
        assert len(_epochs(annot_file)) == 12
        print("   [ok] (d) header without 'stages' key, events only")

        # a stale sidecar from an earlier time-map run is called out, not deleted
        path = _write_cut(tmp, name='stale.set', with_stages=False)
        xl, annot_file = _annotate(path, 'stale.xml', tmp)
        sidecar_path(annot_file).write_text('{}')
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        stale = [m for m in log.messages(logging.WARNING) if 'stale' in m]
        assert stale and sidecar_path(annot_file).exists(), log.messages()
        print(f"   [ok] stale sidecar: WARNING, file left in place")


def test_source_none_usable_returns_false():
    """Mismatch and no way to align: False, ERROR, nothing written, no sidecar."""
    print("\n19. Source 4, nothing usable:")
    with Workdir() as tmp:
        # over-long etc.stages, no boundaries, no stage events
        path, _ = _write_uncut(tmp, 20, name='none.set')
        xl, annot_file = _annotate(path, 'none.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is False
        errors = log.messages(logging.ERROR)
        assert len(errors) == 1 and 'Staging not imported' in errors[0], errors
        assert 'etc.stages has 20 epochs' in errors[0] and 'no stage events' in errors[0], errors[0]
        assert _nothing_imported(annot_file)
        assert not os.path.exists(sidecar_path(annot_file))
        print(f"   [ok] returns False; ERROR: {errors[0][:110]}...")

        # a map that does not add up (wrong removal length) and no stage events
        path = _write_cut(tmp, name='bad_map.set', events=False, stages=NIGHT_NUMERIC * 2)
        xl, annot_file = _annotate(path, 'bad_map.xml', tmp)
        with LogCapture() as log:
            ok = xl.add_stages_from_header()
        assert ok is False and _nothing_imported(annot_file) and log.messages(logging.ERROR)
        assert not os.path.exists(sidecar_path(annot_file))
        print("   [ok] boundary map that does not account for the length -> False")

        # neither stages nor events: the pre-existing message path, also False
        path = os.path.join(tmp, 'empty.set')
        fx.write_set(path, 'root', True, n_samples=36000, srate=FS, labels=['Cz'], types=None,
                     ref=None, n_good=None)
        xl, annot_file = _annotate(path, 'empty.xml', tmp)
        assert xl.add_stages_from_header() is False
        print("   [ok] no stages and no stage events -> False")


def test_stage_code_styles_agree():
    """'1','2','3' and 'N1','N2','N3' in etc.stages give the same staging."""
    print("\n20. Numeric vs N-style stage codes in etc.stages:")
    with Workdir() as tmp:
        out = {}
        for style, night in (('numeric', NIGHT_NUMERIC), ('named', NIGHT_NAMED)):
            path = _write_cut(tmp, name=f'{style}.set', stages=night)
            xl, annot_file = _annotate(path, f'{style}.xml', tmp)
            with LogCapture() as log:
                assert xl.add_stages_from_header() is True
            assert not log.messages(logging.WARNING), (style, log.messages())
            data = json.load(open(sidecar_path(annot_file)))
            assert data['source'] == STAGING_SOURCE_TIME_MAP, style
            assert data['stage_events_disagreeing'] == 0, style
            out[style] = _stage_names(annot_file)
            assert data['fullnight_stages'] == night, "the sidecar keeps the file's own codes"
        assert out['numeric'] == out['named'], out
        assert set(out['numeric']) <= {'Wake', 'NREM1', 'NREM2', 'NREM3', 'REM', 'Undefined'}, out
        print(f"   [ok] both spellings -> {out['numeric']}")

        # uncut path too
        for style, night in (('numeric', ['W', '1', '2', '3', 'R', 'W']),
                             ('named', ['W', 'N1', 'N2', 'N3', 'R', 'W'])):
            path = os.path.join(tmp, f'u_{style}.set')
            fx.write_set(path, 'root', True, n_samples=18000, srate=FS, labels=['Cz'], types=None,
                         ref=None, n_good=None, stages=night)
            xl, annot_file = _annotate(path, f'u_{style}.xml', tmp)
            assert xl.add_stages_from_header() is True
            out['u_' + style] = _stage_names(annot_file)
        assert out['u_numeric'] == out['u_named'] == ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM', 'Wake']
        print("   [ok] as-stored path: same for both spellings")


def test_as_stored_path_matches_the_old_import():
    """Regression: an as-stored file gets exactly the XML the old code wrote."""
    print("\n21. As-stored path vs the pre-cut-support import call:")
    import tempfile as _tf
    from wonambi.attr import Annotations

    def legacy_add_stages(xl):
        """The body of ``add_stages_from_header`` before cut support."""
        header = xl.dataset.header
        stages = header['stages']
        rec_start = header['start_time']
        with _tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as fh:
            name = fh.name
            for code in stages:
                fh.write(f"{code}\n")
        try:
            xl.annotations.import_staging(
                filename=name, source='compumedics', rater_name=xl.rater_name,
                rec_start=rec_start, staging_start=None, epoch_length=30,
                poor=['Artefact'], as_qual=False)
        finally:
            os.unlink(name)
        return True

    def dump(annot_file):
        ann = Annotations(annot_file, rater_name='tester')
        return ann.get_epochs(), ann.raters

    with Workdir() as tmp:
        for label, n_stages in (('12 epochs, 360 s', 12), ('13 epochs, one over', 13), ('5 epochs, short', 5)):
            path, stages = _write_uncut(tmp, n_stages, name=f'reg{n_stages}.set')
            new_xl, new_file = _annotate(path, f'new{n_stages}.xml', tmp)
            old_xl, old_file = _annotate(path, f'old{n_stages}.xml', tmp)
            assert new_xl.add_stages_from_header() is True
            assert legacy_add_stages(old_xl) is True
            new_xl.save()
            old_xl.save()
            assert dump(new_file) == dump(old_file), label
            with open(new_file) as a, open(old_file) as b:
                assert _strip_times(a.read()) == _strip_times(b.read()), label
            print(f"   [ok] {label}: epochs, raters and XML text identical")


def _strip_times(text):
    """Drop the wall-clock stamps and file names that differ between two runs."""
    import re
    return re.sub(r' (created|modified)="[^"]*"', '', text)


# ------------------------------------------------------------ epoch layout

# Everything below asserts the regular 30 s grid over the cut file. If staging
# moves to exact variable-length epochs, replace these functions.

def test_grid_time_map_epochs():
    """GRID: time-map staging gives one 30 s epoch per grid step over the cut file."""
    print("\n22. GRID layout, time-map source:")
    with Workdir() as tmp:
        for style, night in (('numeric', NIGHT_NUMERIC), ('named', NIGHT_NAMED)):
            path = _write_cut(tmp, name=f'g_{style}.set', stages=night)
            xl, annot_file = _annotate(path, f'g_{style}.xml', tmp)
            assert xl.add_stages_from_header() is True
            epochs = _epochs(annot_file)
            assert len(epochs) == 12, len(epochs)                      # ceil(340 / 30)
            assert (epochs[0][2], epochs[-1][2]) != ('Unknown', 'Unknown')
            assert [e[0] for e in epochs] == [30.0 * k for k in range(12)], [e[0] for e in epochs]
            assert all(abs(e[1] - e[0] - 30.0) < 1e-6 for e in epochs), epochs
            want = [WONAMBI_NAME[canon] for canon in
                    (canonical_stage_code(c) if c != '?' else '?' for c in _oracle_time_map_grid(night))]
            got = [e[2] for e in epochs]
            assert got == want, (style, got, want)
            data = json.load(open(sidecar_path(annot_file)))
            assert len(data['cut_stages']) == 12 and data['cut_stages'][-1] == '?'
            print(f"   [ok] {style}: {got}")

        # cut epoch 3 (90-120 s) is 10 s of original epoch 3 ('2') and 20 s of
        # original epoch 5 ('3'), so the time map labels it NREM3
        assert got[3] == 'NREM3' and got[-1] == 'Undefined'
        print("   [ok] cut epoch 3 straddles the splice and takes the majority (NREM3); last is Undefined")

        # after the splice cut epoch k is original epoch k + 2: cut 4 -> 6 ('2'), cut 10 -> 12 ('2')
        assert got[4] == 'NREM2' and got[8] == 'Wake' and got[10] == 'NREM2'
        print("   [ok] cut epochs after the splice read original epoch k + 2")


def test_grid_splice_at_time_zero_file():
    """GRID: a file whose first 60 s of night were removed (splice at cut 0)."""
    print("\n22a. GRID layout, splice at cut time 0:")
    with Workdir() as tmp:
        path = _write_cut(tmp, name='zero.set', cut_at=0.0)
        xl, annot_file = _annotate(path, 'zero.xml', tmp)
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_TIME_MAP in m for m in log.messages(logging.INFO)), log.messages()
        data = json.load(open(sidecar_path(annot_file)))
        assert data['boundaries'] == [{'cut_onset_s': 0.0, 'original_onset_s': 0.0, 'removed_s': 60.0}]
        got = _stage_names(annot_file)
        want = [WONAMBI_NAME[c] for c in [NIGHT_NUMERIC[k + 2] for k in range(11)] + ['?']]
        assert got == want, (got, want)
        # stage events for the two removed epochs are gone, the rest agree with the map
        assert data['stage_events_compared'] == 11 and data['stage_events_disagreeing'] == 0, data
        print(f"   [ok] cut epoch k = original epoch k + 2 from the first epoch: {got}")


def test_grid_stage_event_epochs():
    """GRID: the stage-event source on the same file, over the same grid."""
    print("\n23. GRID layout, stage-event source:")
    with Workdir() as tmp:
        path = _write_cut(tmp, name='g_ev.set', with_stages=False)
        xl, annot_file = _annotate(path, 'g_ev.xml', tmp)
        assert xl.add_stages_from_header() is True
        epochs = _epochs(annot_file)
        assert len(epochs) == 12 and [e[0] for e in epochs] == [30.0 * k for k in range(12)]
        want = [WONAMBI_NAME[c if c == '?' else canonical_stage_code(c)]
                for c in _oracle_events_grid(NIGHT_NUMERIC)]
        got = [e[2] for e in epochs]
        assert got == want, (got, want)
        print(f"   [ok] {got}")
        assert got[3] == 'NREM2' and got[3] != 'NREM3'
        print("   [ok] the event before the gap runs to the next onset, so cut epoch 3 "
              "keeps original epoch 3 (NREM2), unlike the time map (NREM3)")


def test_grid_as_stored_epochs():
    """GRID: as-stored staging writes one 30 s epoch per etc.stages entry."""
    print("\n24. GRID layout, as-stored source:")
    with Workdir() as tmp:
        path, stages = _write_uncut(tmp, 12)
        xl, annot_file = _annotate(path, 'g_st.xml', tmp)
        assert xl.add_stages_from_header() is True
        epochs = _epochs(annot_file)
        assert [(a, b) for a, b, _ in epochs] == [(30.0 * k, 30.0 * (k + 1)) for k in range(12)]
        assert [s for _, _, s in epochs] == [WONAMBI_NAME[s] for s in stages]
        print("   [ok] 12 contiguous 30 s epochs from time 0")


REFERENCE_FILE = ('/Volumes/Tancy_storage/Z_drug/sub-02dg/ses-1/'
                  'sub-02dg_ses-1_task-psg_run-1_desc-clean_eeg.set')


def test_reference_recording_if_reachable():
    """The real Compumedics example (skipped when the disk is not mounted).

    Read-only on the recording; the annotation and sidecar go to a temp folder.
    """
    print("\n25. Reference Compumedics recording:")
    if not os.path.exists(REFERENCE_FILE):
        print(f"   [skip] {REFERENCE_FILE} is not reachable")
        return
    with Workdir() as tmp:
        ds = LargeDataset(REFERENCE_FILE)
        h = ds.header
        assert (len(h['chan_name']), h['s_freq'], h['n_samples']) == (277, 250.0, 4827027)
        assert 'ECG_2' in h['chan_name'] and 'BodyPosition_2' in h['chan_name']
        assert sum(t == 'EEG' for t in h['chan_type']) == 257
        assert h['reference'] == {'ref': 'average', 'n_good': 220}
        assert len(h['stages']) == 858
        tl = RecordingTimeline.from_header(h, h['s_freq'], h['n_samples'])
        assert tl.n_boundaries == 189 and tl.is_consistent
        assert abs(tl.removed_seconds - 6460.1) < 0.1 and abs(tl.cut_seconds - 19308.1) < 0.1
        print("   [ok] 277 channels, 250 Hz, 4,827,027 samples, 189 boundaries, consistent map")

        xl = XLAnnotations(ds, os.path.join(tmp, 'ref.xml'), rater_name='tester')
        with LogCapture() as log:
            assert xl.add_stages_from_header() is True
        assert any(STAGING_SOURCE_TIME_MAP in m for m in log.messages(logging.INFO))
        names = _stage_names(os.path.join(tmp, 'ref.xml'))
        assert len(names) == 644, len(names)
        counts = {n: names.count(n) for n in sorted(set(names))}
        assert names[-1] == 'Undefined'
        plan = {'NREM2': 269, 'NREM3': 141, 'REM': 110, 'NREM1': 84, 'Wake': 35}
        for stage, n in plan.items():
            assert abs(counts.get(stage, 0) - n) <= 1, (stage, counts.get(stage), n)
        data = json.load(open(sidecar_path(os.path.join(tmp, 'ref.xml'))))
        assert data['n_boundaries'] == 189 and len(data['fullnight_stages']) == 858
        print(f"   [ok] time-map staging: 644 epochs {counts}; stage events disagree on "
              f"{data['stage_events_disagreeing']} of {data['stage_events_compared']}")


TESTS = [
    test_time_map_round_trip_and_clamping,
    test_time_map_without_boundaries,
    test_time_map_splice_at_time_zero,
    test_time_map_two_splices_at_same_cut_onset,
    test_time_map_zero_length_removal,
    test_boundaries_from_events,
    test_is_consistent_edges,
    test_stage_intervals_cut_index_arithmetic,
    test_compare_stage_events,
    test_stage_event_helpers,
    test_timeline_json_round_trip,
    test_regrid_majority_tie_undefined_final_epoch,
    test_source_as_stored_for_uncut_files,
    test_source_threshold_between_stored_and_time_map,
    test_source_time_map_writes_sidecar,
    test_source_time_map_warns_when_stage_events_disagree,
    test_source_small_removal_uses_time_map,
    test_source_no_removed_time_stays_as_stored,
    test_source_scorer_stopped_early_uses_time_map,
    test_source_events_choose_as_stored,
    test_source_no_events_ambiguous_warns,
    test_source_misstated_boundary_falls_back_to_events,
    test_source_uncut_unmarked_gap_warns_output_unchanged,
    test_source_time_map_on_all_layouts,
    test_source_stage_events_when_no_usable_map,
    test_source_none_usable_returns_false,
    test_stage_code_styles_agree,
    test_as_stored_path_matches_the_old_import,
    test_grid_time_map_epochs,
    test_grid_splice_at_time_zero_file,
    test_grid_stage_event_epochs,
    test_grid_as_stored_epochs,
    test_reference_recording_if_reachable,
]


if __name__ == "__main__":
    print("TESTING cut-recording time map and staging source")
    print("=================================================")
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
    print("All cut-timeline and staging tests passed.")
