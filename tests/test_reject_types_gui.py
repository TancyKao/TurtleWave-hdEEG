#!/usr/bin/env python3
"""Headless acceptance checklist for the reject-types GUI spec, items 1-14.

Promoted from the gui-engineer's acceptance script for
``_scratch/design/reject_types_gui_spec.md``. Covers the five reject-type
checkboxes on ``TurtleWaveGUI`` (Setup tab), the four per-tab echo labels,
that each detector run method is passed ``reject_types=`` (never the
deprecated ``reject_artifacts``/``reject_arousals`` booleans), the
``report_db_density`` header/mismatch note, ``_choose_db_scope`` reading
recorded exclusion sets back out of ``detection_runs``, and the QC dashboard's
``ChannelDetailDock`` denominator caption.

Run with:
    QT_QPA_PLATFORM=offscreen PYTHONPATH=$PWD python tests/test_reject_types_gui.py

TurtleWaveGUI.__init__ reassigns sys.stdout to its log widget, so the real
stdout is captured before any window is built and every report goes there.
"""
import os
import sqlite3
import sys
import tempfile
import types

REAL_STDOUT = sys.stdout          # captured BEFORE any TurtleWaveGUI exists


def say(*a):
    print(*a, file=REAL_STDOUT, flush=True)


FAILURES = []
CHECKS = [0]


def check(item, label, ok, detail=""):
    CHECKS[0] += 1
    tag = "PASS" if ok else "FAIL"
    say(f"  [{tag}] ({item}) {label}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"({item}) {label}: {detail}")


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()   # before any frontend import
from PyQt5 import QtWidgets                                   # noqa: E402
import frontend.turtlewave_gui as tg                          # noqa: E402
import frontend.eeg_review_gui as rg                          # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


class LogSpy:
    """Captures write_log lines off a TurtleWaveGUI without a log widget."""

    def __init__(self, win):
        self.win = win
        self.lines = []
        self._orig = win.write_log
        win.write_log = self._capture

    def _capture(self, message):
        self.lines.append(str(message))

    def clear(self):
        self.lines = []

    def text(self):
        return "\n".join(self.lines)


say("=" * 78)
say("Headless acceptance: reject_types_gui_spec.md items 1-14")
say("=" * 78)

win = tg.TurtleWaveGUI()
spy = LogSpy(win)

# ---------------------------------------------------------------- 1
say("\n-- 1. five checkboxes, correct default states")
keys = sorted(win.reject_type_checks)
check(1, "keys are exactly the five known types",
      keys == sorted(['Artefact', 'Arousal', 'Move', 'Resp', 'Snore']),
      str(keys))
states = {k: win.reject_type_checks[k].isChecked()
          for k in ['Artefact', 'Arousal', 'Move', 'Resp', 'Snore']}
check(1, "defaults are True, True, True, False, False",
      list(states.values()) == [True, True, True, False, False], str(states))

# ---------------------------------------------------------------- 2
say("\n-- 2. accessors on a freshly constructed window")
check(2, "selected_reject_types()",
      win.selected_reject_types() == ['Artefact', 'Arousal', 'Move'],
      repr(win.selected_reject_types()))
from turtlewave_hdEEG.utils import reject_key as _reject_key   # noqa: E402
# The GUI offers NO token accessor of its own: a display-ordered
# 'Artefact,Arousal,Move' and the library's sorted 'Arousal,Artefact,Move' are
# two comma-joined spellings of one set, which is the bug class this release
# removes. Human-facing strings come from reject_types_display(); stored or
# compared values come from the library's reject_key().
check(2, "no GUI-side reject_types_token()",
      hasattr(win, 'reject_types_token') is False,
      repr(getattr(win, 'reject_types_token', None)))
check(2, "the database key comes from turtlewave_hdEEG.utils.reject_key",
      _reject_key(win.selected_reject_types()) == 'Arousal,Artefact,Move',
      repr(_reject_key(win.selected_reject_types())))
check(2, "ticked_reject_types() matches selected_reject_types()",
      win.ticked_reject_types() == win.selected_reject_types(),
      repr(win.ticked_reject_types()))

# ---------------------------------------------------------------- 3
say("\n-- 3. order is stable and independent of click order")
win.reject_type_checks['Resp'].setChecked(True)
check(3, "ticking Resp appends it in canonical order",
      win.selected_reject_types() == ['Artefact', 'Arousal', 'Move', 'Resp'],
      repr(win.selected_reject_types()))
# untick everything, then retick in reverse order
for t in ['Artefact', 'Arousal', 'Move', 'Resp']:
    win.reject_type_checks[t].setChecked(False)
for t in ['Resp', 'Move', 'Arousal', 'Artefact']:
    win.reject_type_checks[t].setChecked(True)
check(3, "same set reticked backwards gives the same list",
      win.selected_reject_types() == ['Artefact', 'Arousal', 'Move', 'Resp'],
      repr(win.selected_reject_types()))

# ---------------------------------------------------------------- 4
say("\n-- 4. Restore defaults")
win.reject_type_checks['Snore'].setChecked(True)
win.reject_type_checks['Artefact'].setChecked(False)
spy.clear()
win.restore_reject_defaults()
after = {k: win.reject_type_checks[k].isChecked()
         for k in ['Artefact', 'Arousal', 'Move', 'Resp', 'Snore']}
check(4, "state matches item (1)",
      list(after.values()) == [True, True, True, False, False], str(after))
check(4, "logs the restored set",
      'Artefact, Arousal, Movement' in spy.text(), spy.text())

# ---------------------------------------------------------------- 5
say("\n-- 5. the old per-tab boolean widgets are gone")
stale = [a for a in dir(win)
         if 'reject_artifacts' in a or 'reject_arousals' in a]
check(5, "no *reject_artifacts* / *reject_arousals* attribute", stale == [],
      str(stale))

# ---------------------------------------------------------------- 6
# "after construction" is load-bearing here: checked on `win`, items 3 and 4
# would already have repainted the echoes and a blank-at-startup bug would go
# unseen (it did, once). So this runs against a window nothing has touched.
say("\n-- 6. four distinct echo labels, on a FRESH window")
names = ['spindle_reject_echo', 'sw_reject_echo', 'kc_reject_echo',
         'pac_reject_echo']
fresh = tg.TurtleWaveGUI()
have = [hasattr(fresh, n) for n in names]
check(6, "all four exist", all(have), str(dict(zip(names, have))))
echoes = [getattr(fresh, n) for n in names]
check(6, "all four are distinct objects", len({id(e) for e in echoes}) == 4)
bad = [n for n, e in zip(names, echoes)
       if 'Artefact, Arousal, Movement' not in e.text()]
check(6, "each text() names the default set with NO interaction", bad == [],
      str({n: getattr(fresh, n).text() for n in names}))
btns = [n + '_change_btn' for n in names]
check(6, "each has its own Change button",
      all(hasattr(fresh, b) for b in btns)
      and len({id(getattr(fresh, b)) for b in btns}) == 4)
check(6, "the Setup summary is painted at construction too",
      fresh.reject_summary_label.text()
      == 'Excluding: Artefact, Arousal, Movement',
      repr(fresh.reject_summary_label.text()))
# NOT fresh.close(): TurtleWaveGUI.closeEvent raises a modal
# "Are you sure you want to exit?" QMessageBox, which blocks forever with no
# display. Drop the reference and let Qt collect it.
del fresh

# ---------------------------------------------------------------- 7
say("\n-- 7. one toggle updates all four echoes")
win.reject_type_checks['Resp'].setChecked(True)
missing = [n for n in names if 'Respiratory' not in getattr(win, n).text()]
check(7, "all four echoes show Respiratory after ticking Resp", missing == [],
      str({n: getattr(win, n).text() for n in names}))
win.restore_reject_defaults()

# ---------------------------------------------------------------- 8
say("\n-- 8. each run method passes reject_types= and neither boolean")
captured = {}


def _capture_kwargs(name):
    def f(*args, **kwargs):
        captured[name] = kwargs
        raise RuntimeError(f"stop after {name}")
    return f


class _FakeProcessor:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr in ('detect_spindles', 'detect_slow_waves',
                    'detect_kcomplexes', 'analyze_pac'):
            return _capture_kwargs(attr)
        return lambda *a, **k: None


tmpdir = tempfile.mkdtemp(prefix='reject_gui_')
win.output_dir = tmpdir
win.annot_file_path = os.path.join(tmpdir, 'annot.xml')
open(win.annot_file_path, 'w').write('<annotations/>')
win.annotations = object()          # truthy: skip the CustomAnnotations load
win.dataset = object()
win.selected_channels = ['E1']
win.pac_selected_channels = ['E1']
win.resolve_subject = lambda explicit=None: 'sub-test'
win.run_db_path = lambda: os.path.join(tmpdir, 'neural_events.db')
win.db_run_ids = lambda *a, **k: set()

tg.ParalSWA = lambda **k: _FakeProcessor('sw')
tg.ParalKC = lambda **k: _FakeProcessor('kc')
tg.ParalEvents = lambda **k: _FakeProcessor('spindle')

run_specs = [
    ('detect_slow_waves', lambda: (
        setattr(win, 'sw_detection_params', {
            'method': 'Massimini2004', 'chan': ['E1'],
            'frequency': (0.5, 4.0), 'neg_peak_thresh': -80.0,
            'p2p_thresh': 140.0, 'polar': 'normal',
            'reject_types': win.selected_reject_types(),
            'stage': ['NREM2'], 'trough_duration': (0.3, 1.0)}),
        win.detect_sw())),
    ('detect_kcomplexes', lambda: (
        setattr(win, 'kc_detection_params', {
            'method': 'AASM/Massimini2004', 'chan': ['E1'],
            'frequency': (0.1, 4.0), 'trough_duration': (0.25, 1.0),
            'neg_peak_thresh': -40.0, 'p2p_thresh': 75.0,
            'min_isolation': 1.0, 'polar': 'normal',
            'reject_types': win.selected_reject_types(),
            'stage': ['NREM2']}),
        win.detect_kc())),
    ('detect_spindles', lambda: win.detect_spindles(
        ['NREM2'], {}, False, win.selected_reject_types())),
]
for name, run in run_specs:
    try:
        run()
    except Exception:
        pass

# PAC goes through its own processor import inside run_pac_analysis
import turtlewave_hdEEG as _tw                                # noqa: E402
_orig_pac = _tw.ParalPAC
_tw.ParalPAC = lambda **k: _FakeProcessor('pac')
win.pac_analysis_params = {
    'method': 'SW-Spindle', 'sw_method': 'Massimini2004',
    'spindle_method': 'Moelle2011', 'phase_freq': (0.5, 1.25),
    'amp_freq': (11, 16), 'channels': ['E1'], 'stages': ['NREM2'],
    'idpac': (2, 3, 4), 'time_window': 1.0,
    'db_path': os.path.join(tmpdir, 'neural_events.db'),
    'reject_types': win.selected_reject_types(),
}
try:
    win.run_pac_analysis()
except Exception:
    pass
_tw.ParalPAC = _orig_pac

for name in ('detect_spindles', 'detect_slow_waves', 'detect_kcomplexes',
             'analyze_pac'):
    kw = captured.get(name)
    if kw is None:
        check(8, f"{name} was called", False, "never reached")
        continue
    check(8, f"{name}(reject_types=['Artefact', 'Arousal', 'Move'])",
          list(kw.get('reject_types') or []) == ['Artefact', 'Arousal', 'Move'],
          repr(kw.get('reject_types')))
    check(8, f"{name} passes neither deprecated boolean",
          'reject_artifacts' not in kw and 'reject_arousals' not in kw,
          str(sorted(k for k in kw if 'reject' in k)))

# and the summary dicts the run methods build
sw_params = win.sw_detection_params
kc_params = win.kc_detection_params
for label, d in (('sw_detection_params', sw_params),
                 ('kc_detection_params', kc_params),
                 ('pac_analysis_params', win.pac_analysis_params)):
    check(8, f"{label} carries reject_types and neither boolean",
          d.get('reject_types') == ['Artefact', 'Arousal', 'Move']
          and 'reject_artifacts' not in d and 'reject_arousals' not in d,
          repr(d.get('reject_types')))

# ---------------------------------------------------------------- 9
say("\n-- 9. report_db_density header and mismatch note")


class _FakeDF(list):
    def groupby(self, *a, **k):
        return []

    def __len__(self):
        return 1


def _fake_density(*args, **kwargs):
    _fake_density.kwargs = kwargs
    return _FakeDF([1])


_real_density = tg.event_density
tg.event_density = _fake_density
_fake_db_path = os.path.join(tmpdir, 'neural_events.db')
spy.clear()
win.report_db_density("Spindle", _fake_db_path, 'spindle',
                      'Moelle2011', (11.0, 16.0), ['NREM2'], 'sub-test',
                      ['Artefact', 'Arousal', 'Move'])
hdr = [l for l in spy.lines if l.startswith("Spindle density")]
check(9, "header names the exclusion set",
      bool(hdr) and 'excluding Artefact, Arousal, Movement' in hdr[0],
      str(hdr))
check(9, "header says 'minute of searched time'",
      bool(hdr) and 'events per minute of searched time' in hdr[0], str(hdr))
check(9, "event_density got reject_types= and no booleans",
      list(_fake_density.kwargs.get('reject_types') or [])
      == ['Artefact', 'Arousal', 'Move']
      and 'reject_artifacts' not in _fake_density.kwargs
      and 'reject_arousals' not in _fake_density.kwargs,
      str(sorted(k for k in _fake_density.kwargs if 'reject' in k)))
check(9, "matching set emits NO mismatch note",
      sum('not the set currently ticked' in l for l in spy.lines) == 0)

spy.clear()
win.report_db_density("Spindle", _fake_db_path, 'spindle',
                      'Moelle2011', (11.0, 16.0), ['NREM2'], 'sub-test',
                      ['Artefact', 'Arousal'])
n_note = sum('not the set currently ticked' in l for l in spy.lines)
check(9, "differing set emits exactly one mismatch line", n_note == 1,
      f"{n_note} line(s)")
tg.event_density = _real_density

# ---------------------------------------------------------------- 10
say("\n-- 10. _choose_db_scope reads the recorded sets")


def make_db(path, with_types, run_rows):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE events (event_type TEXT, method TEXT, "
              "freq_lower REAL, freq_upper REAL, stage TEXT, channel TEXT)")
    c.execute("INSERT INTO events VALUES "
              "('spindle','Moelle2011',11.0,16.0,'NREM2','E1')")
    cols = ("run_id TEXT, event_type TEXT, method TEXT, params_json TEXT, "
            "reject_artifacts INTEGER, reject_arousals INTEGER, timestamp TEXT")
    if with_types:
        cols += ", reject_types TEXT"
    c.execute(f"CREATE TABLE detection_runs ({cols})")
    for row in run_rows:
        c.execute(
            "INSERT INTO detection_runs VALUES (%s)"
            % ",".join("?" * (8 if with_types else 7)), row)
    conn.commit()
    conn.close()


# (a) pre-4.4.0: no reject_types column at all
db_old = os.path.join(tmpdir, 'pre44.db')
make_db(db_old, False,
        [('r1', 'spindle', 'Moelle2011', '{"frequency": [11.0, 16.0]}',
          None, None, '2025-01-01T00:00:00')])
picked = {}
_real_getitem = QtWidgets.QInputDialog.getItem


def _auto_pick(parent, title, label, items, current=0, editable=False):
    picked['items'] = list(items)
    return items[0], True


QtWidgets.QInputDialog.getItem = staticmethod(_auto_pick)
spy.clear()
scope = win._choose_db_scope("Spindle", db_old, 'spindle')
check(10, "pre-4.4.0 database -> ['Artefact', 'Arousal']",
      scope is not None and scope['reject_types'] == ['Artefact', 'Arousal'],
      repr(scope and scope.get('reject_types')))
check(10, "logs the 'before version 4.4.0' disclosure",
      'before version 4.4.0' in spy.text(), spy.text()[:300])

# (b) two different recorded sets for one scope
db_two = os.path.join(tmpdir, 'twosets.db')
make_db(db_two, True,
        [('r1', 'spindle', 'Moelle2011', '{"frequency": [11.0, 16.0]}',
          1, 1, '2025-01-01T00:00:00', 'Arousal,Artefact'),
         ('r2', 'spindle', 'Moelle2011', '{"frequency": [11.0, 16.0]}',
          1, 1, '2025-02-01T00:00:00', 'Arousal,Artefact,Move')])
spy.clear()
scope2 = win._choose_db_scope("Spindle", db_two, 'spindle')
check(10, "dialog entries carry the 'excl. ...' field",
      all('excl. ' in e for e in picked['items']), str(picked['items']))
check(10, "one entry per recorded set", len(picked['items']) == 2,
      str(picked['items']))
check(10, "logs 'must not be pooled'", 'must not be pooled' in spy.text(),
      spy.text()[:400])
check(10, "the chosen scope carries a recorded set",
      scope2 is not None and scope2.get('reject_types_recorded') is True
      and scope2['reject_types'] in (['Artefact', 'Arousal'],
                                     ['Artefact', 'Arousal', 'Move']),
      repr(scope2 and scope2.get('reject_types')))
QtWidgets.QInputDialog.getItem = _real_getitem

# ---------------------------------------------------------------- 11
say("\n-- 11. EventDatabase.get_run_rejections returns a list")


class _Db:
    def __init__(self, conn):
        self.conn = conn


db44 = os.path.join(tmpdir, 'is44.db')
make_db(db44, True,
        [('r1', 'spindle', 'Moelle2011', '{}', 1, 1,
          '2025-02-01T00:00:00', 'Arousal,Artefact,Move')])
conn = sqlite3.connect(db44)
got = rg.EventDatabase.get_run_rejections(_Db(conn), event_type='spindle')
check(11, "4.4.0 fixture -> canonical list",
      got == ['Artefact', 'Arousal', 'Move'], repr(got))
conn.close()

conn = sqlite3.connect(db_old)          # pre-4.4: booleans only, no column
conn.execute("UPDATE detection_runs SET reject_artifacts = 1, "
             "reject_arousals = 1")
conn.commit()
got_old = rg.EventDatabase.get_run_rejections(_Db(conn), event_type='spindle')
check(11, "pre-4.4.0 fixture -> ['Artefact', 'Arousal']",
      got_old == ['Artefact', 'Arousal'], repr(got_old))
conn.close()

# ---------------------------------------------------------------- 12
# AMENDED: the lookup widens before it guesses. Three branches, three sources,
# and only the last is the warning colour.
say("\n-- 12. _qc_run_rejections: scoped hit, widened hit, nothing recorded")

warned = []


class _WarnSpy:
    def warning(self, msg, *args):
        warned.append(msg % args if args else msg)


class _ScriptedDb:
    """Answers only the queries listed in `answers`, keyed by the query shape."""

    db_path = None

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def get_run_rejections(self, event_type=None, methods=None):
        shape = ('scoped' if methods else
                 'by_type' if event_type else 'any')
        self.asked.append(shape)
        return self.answers.get(shape)


def run_rejections(db):
    """Call _qc_run_rejections on a stub with the logger captured."""
    stub = types.SimpleNamespace(
        db=db,
        qc_widget=types.SimpleNamespace(current_event_type=lambda: 'spindle'),
        _current_method_freq=lambda: (['Moelle2011'], None),
        annot_file_path=None,
        _set_qc_rejections=None)
    stub._set_qc_rejections = types.MethodType(
        rg.EventReviewGUI._set_qc_rejections, stub)
    real_logger, real_filter = rg._density_logger, rg._density_repeat_filter
    rg._density_logger = _WarnSpy()
    rg._density_repeat_filter = types.SimpleNamespace(context=None)
    try:
        got = rg.EventReviewGUI._qc_run_rejections(stub)
    finally:
        rg._density_logger = real_logger
        rg._density_repeat_filter = real_filter
    return got, stub


# (a) the scoped query answers: read from the run on screen, no warning.
warned = []
db_a = _ScriptedDb({'scoped': ['Arousal', 'Artefact', 'Move']})
got, stub = run_rejections(db_a)
check(12, "(a) scoped hit returns that set",
      got == ['Artefact', 'Arousal', 'Move'], repr(got))
check(12, "(a) source is 'from the detection run'",
      stub._qc_reject_source == 'from the detection run',
      repr(stub._qc_reject_source))
check(12, "(a) nothing is logged and no widening is attempted",
      warned == [] and db_a.asked == ['scoped'], f"{warned} {db_a.asked}")

# (b) the scoped query misses but the file records a set for another run.
warned = []
db_b = _ScriptedDb({'by_type': ['Arousal', 'Artefact']})
got, stub = run_rejections(db_b)
check(12, "(b) widened hit returns the recorded set, not a guess",
      got == ['Artefact', 'Arousal'], repr(got))
check(12, "(b) source is 'from another run in this file'",
      stub._qc_reject_source == 'from another run in this file',
      repr(stub._qc_reject_source))
check(12, "(b) that source renders in the NORMAL colour",
      'assumed' not in stub._qc_reject_source)
check(12, "(b) logs the widened-lookup line naming the set",
      len(warned) == 1
      and 'most recent run in this file' in warned[0]
      and 'Artefact, Arousal' in warned[0], str(warned))
check(12, "(b) it widened rather than stopping at the scoped miss",
      db_b.asked == ['scoped', 'by_type'], str(db_b.asked))

# (c) nothing in the file records a set anywhere -> pre-4.4.0 by definition.
warned = []
db_c = _ScriptedDb({})
got, stub = run_rejections(db_c)
check(12, "(c) fallback matches _choose_db_scope: ['Artefact', 'Arousal']",
      got == ['Artefact', 'Arousal'], repr(got))
check(12, "(c) source is 'assumed (not recorded)'",
      stub._qc_reject_source == 'assumed (not recorded)',
      repr(stub._qc_reject_source))
check(12, "(c) all three queries were tried before guessing",
      db_c.asked == ['scoped', 'by_type', 'any'], str(db_c.asked))
check(12, "(c) the warning names the assumed set",
      len(warned) == 1 and 'assumes Artefact, Arousal' in warned[0],
      str(warned))
check(12, "(c) bias stated the right way round: excluded LESS -> shown HIGHER",
      "if it excluded less, they are higher" in warned[0]
      and "excluded more than that, the densities shown are lower" in warned[0],
      str(warned))

# The repeat filter is what makes it once-per-file rather than once-per-refresh.
warned = []
run_rejections(_ScriptedDb({}))
run_rejections(_ScriptedDb({}))
check(12, "repeated refreshes produce the identical message",
      len(warned) == 2 and warned[0] == warned[1])
f = rg._RepeatSuppressingFilter()
rec = types.SimpleNamespace(levelno=30, getMessage=lambda: warned[0])
check(12, "_RepeatSuppressingFilter passes the first and drops the second",
      f.filter(rec) is True and f.filter(rec) is False)

# ---------------------------------------------------------------- 13
say("\n-- 13. a type with zero events stays ticked and enabled")


class _AnnotNoMove:
    def get_events(self, name=None):
        return [] if name == 'Move' else [object()] * 12


spy.clear()
win.write_log_once('reject_types_empty', None)     # reset the once-filter
counts = win.refresh_reject_counts(_AnnotNoMove())
check(13, "Movement still ticked", win.reject_type_checks['Move'].isChecked())
check(13, "Movement still enabled", win.reject_type_checks['Move'].isEnabled())
check(13, "count label reads 'none in this file'",
      win.reject_type_counts['Move'].text() == 'none in this file',
      repr(win.reject_type_counts['Move'].text()))
check(13, "a non-empty type reports its count",
      win.reject_type_counts['Artefact'].text() == '12 in this file',
      repr(win.reject_type_counts['Artefact'].text()))
check(13, "log explains why it stays ticked",
      'they stay ticked so this run is recorded with the same exclusion set'
      in spy.text(), spy.text())

# ---------------------------------------------------------------- 5b
say("\n-- extra: ChannelDetailDock.set_denominator_mask")
dock = rg.ChannelDetailDock()
dock.set_denominator_mask(['Artefact', 'Arousal', 'Move'],
                          'from the detection run')
check('5c', "caption text",
      dock.mask_caption.text() ==
      'density excludes Artefact, Arousal, Movement · from the '
      'detection run', repr(dock.mask_caption.text()))
check('5c', "normal colour when read from the run",
      '#6b7585' in dock.mask_caption.styleSheet(),
      dock.mask_caption.styleSheet())
# The widened-lookup source is a reading, not a guess, so it must NOT pick up
# the warning colour (only 'assumed ...' does).
dock.set_denominator_mask(['Artefact', 'Arousal'],
                          'from another run in this file')
check('5c', "widened source reads correctly",
      dock.mask_caption.text() ==
      'density excludes Artefact, Arousal · from another run in this file',
      repr(dock.mask_caption.text()))
check('5c', "widened source keeps the NORMAL colour",
      '#6b7585' in dock.mask_caption.styleSheet(),
      dock.mask_caption.styleSheet())
dock.set_denominator_mask(['Artefact', 'Arousal', 'Move'],
                          'assumed (not recorded)', pending_marks=3)
check('5c', "assumed + pending marks",
      dock.mask_caption.text().endswith('· + 3 marks you added')
      and 'assumed (not recorded)' in dock.mask_caption.text(),
      repr(dock.mask_caption.text()))
check('5c', "warning colour when assumed",
      '#d29922' in dock.mask_caption.styleSheet(),
      dock.mask_caption.styleSheet())

say("\n" + "=" * 78)
check('settings', "the real review-GUI preferences file was not "
      "touched", *gui_settings_guard.untouched())
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)
say("\n-- 14. run separately: python tests/test_gui_performance.py")

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
