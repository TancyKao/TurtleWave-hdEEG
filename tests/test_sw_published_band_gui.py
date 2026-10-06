#!/usr/bin/env python3
"""Headless checks: the slow-wave tab sends Ngo2015 / Staresina2015 their
published band, whatever the duration boxes hold (4.6.1).

Before 4.6.1 the tab sent ``frequency = (1/max_dur, 1/min_dur)`` for these
two methods. ``ImprovedDetectSlowWave`` uses the upper bound of
``frequency`` as the low-pass, and ``ParalSWA`` stores the pair as the run's
band, so Ngo2015 ran at a ~1.2 Hz low-pass instead of the published 3.5 Hz
and every duration edit moved the filter and the stored band.

The detector ignores ``min_dur`` / ``max_dur`` for these two methods (by
design, ``extensions.py``), so the duration boxes are read-only at
Wonambi's ``DetectSlowWave(method).duration`` and the tab sends ``None``.

For each method the kwargs the tab passes to
``ParalSWA.detect_slow_waves`` are captured (the processor is replaced by a
fake that records them and stops): ``frequency`` must be the band Wonambi
configures for the method (``det_filt`` lower edge, ``lowpass`` cut-off),
``min_dur`` / ``max_dur`` must be ``None`` even if a box value is changed,
and a run that finishes records the published duration in its summary. Massimini2004 and AASM/Massimini2004 still
send their filter boxes and trough window.

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_sw_published_band_gui.py

TurtleWaveGUI.__init__ reassigns sys.stdout to its log widget, so the real
stdout is captured before any window is built and every report goes there.
"""
import os
import sys
import tempfile

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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()   # before any frontend import
from PyQt5 import QtWidgets                                   # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
import frontend.turtlewave_gui as tg                          # noqa: E402
from wonambi.detect import DetectSlowWave                     # noqa: E402
from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

say("=" * 78)
say("Headless: slow-wave tab sends the published band (Ngo2015, Staresina2015)")
say("=" * 78)

# ---- the published bands, from Wonambi itself -----------------------------
PUBLISHED = {}
for m in ('Ngo2015', 'Staresina2015'):
    d = DetectSlowWave(m)
    PUBLISHED[m] = (float(d.det_filt['freq'][0]), float(d.lowpass['freq']))
check('0.1', "Wonambi's defaults are the published bands: Ngo2015 "
      "0.5-3.5 Hz, Staresina2015 0.5-1.25 Hz",
      PUBLISHED == {'Ngo2015': (0.5, 3.5), 'Staresina2015': (0.5, 1.25)},
      repr(PUBLISHED))
check('0.2', "published_sw_band() returns them",
      all(tg.published_sw_band(m) == b for m, b in PUBLISHED.items()),
      repr({m: tg.published_sw_band(m) for m in PUBLISHED}))
try:
    tg.published_sw_band('Massimini2004')
    refused = False
except ValueError:
    refused = True
check('0.3', "published_sw_band('Massimini2004') raises ValueError", refused)

# ---- drive the tab ----------------------------------------------------------
captured = []


class _Stop(Exception):
    """Raised by the fake processor once the kwargs are captured; the tab
    reports it in its error dialog as 'Failed to detect slow waves: STOP'."""

    def __str__(self):
        return 'STOP'


summaries = []
FULL_RUN = [False]      # True: let the run finish (summary + run scope)


class _FakeSWA:
    def __init__(self, **kwargs):
        pass

    def detect_slow_waves(self, **kwargs):
        captured.append(kwargs)
        if FULL_RUN[0]:
            return []
        raise _Stop()

    def save_detection_summary(self, **kwargs):
        summaries.append(kwargs)


class _SyncThread:
    """threading.Thread stand-in: runs the target on start(), in this
    thread, so the captured kwargs are there when start() returns."""

    def __init__(self, target=None, **kwargs):
        self._target = target
        self.daemon = True

    def start(self):
        self._target()


errors = []
tg.ParalSWA = _FakeSWA
tg.threading.Thread = _SyncThread
tg.QMessageBox.critical = staticmethod(
    lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))

tmp = tempfile.mkdtemp(prefix='tw_sw_band_')
win = tg.TurtleWaveGUI()
win.write_log = lambda message: None
win.output_dir = tmp
win.annot_file_path = os.path.join(tmp, 'annot.xml')
open(win.annot_file_path, 'w').write('<annotations/>')
win.annotations = object()          # truthy: skip the CustomAnnotations load
win.dataset = object()
win.selected_channels = ['E1']
win.resolve_subject = lambda explicit=None: 'sub-test'
win.run_db_path = lambda: os.path.join(tmp, 'neural_events.db')
win.db_run_ids = lambda *a, **k: set()
for st, box in win.sw_stage_checks.items():
    box.setChecked(st == 'NREM2')


def run(method, durations=None):
    """Select ``method``, optionally put ``durations`` into the boxes (set
    programmatically: they are read-only for the published-band methods),
    press Detect; return the kwargs sent to detect_slow_waves (None if
    none)."""
    win.sw_method_combo.setCurrentText(method)
    app.processEvents()
    if durations is not None:
        win.sw_param_widgets['duration']['min'].setValue(durations[0])
        win.sw_param_widgets['duration']['max'].setValue(durations[1])
    n = len(captured)
    win.detect_sw_thread()
    return captured[-1] if len(captured) > n else None


def forced_off(w):
    """Disabled by its own setEnabled(False), not by a disabled parent (the
    whole tab is off until data are loaded)."""
    return w.testAttribute(Qt.WA_ForceDisabled)


EDITED = (0.6, 1.5)
PUB_DUR = {m: tuple(float(x) for x in DetectSlowWave(m).duration)
           for m in PUBLISHED}
check('0.4', "Wonambi's published durations: Ngo2015 0.833-2 s, "
      "Staresina2015 0.8-2 s; published_sw_duration() returns them",
      PUB_DUR == {'Ngo2015': (0.833, 2.0), 'Staresina2015': (0.8, 2.0)}
      and all(tg.published_sw_duration(m) == d for m, d in PUB_DUR.items()),
      repr(PUB_DUR))
for method in ('Ngo2015', 'Staresina2015'):
    band = PUBLISHED[method]
    dur = PUB_DUR[method]
    kw_def = run(method)
    dw = win.sw_param_widgets['duration']
    check(f'{method}.1', f"{method}: frequency is the published "
          f"{band[0]:g}-{band[1]:g} Hz; min_dur/max_dur are None (the "
          f"detector uses the published duration)",
          kw_def is not None and tuple(kw_def['frequency']) == band
          and kw_def['min_dur'] is None and kw_def['max_dur'] is None,
          repr(kw_def and (kw_def['frequency'], kw_def['min_dur'],
                           kw_def['max_dur'])))
    check(f'{method}.2', f"{method}: the duration boxes are read-only at "
          f"the published {dur[0]:g}-{dur[1]:g} s, with the tooltip",
          forced_off(dw['min']) and forced_off(dw['max'])
          and (dw['min'].value(), dw['max'].value()) == dur
          and all(w.toolTip() == "Published value; the detector does not "
                  "take a custom duration for this method."
                  for w in (dw['min'], dw['max'])),
          repr((forced_off(dw['min']), dw['min'].value(), dw['max'].value(),
                dw['min'].toolTip())))
    kw_ed = run(method, EDITED)
    check(f'{method}.3', f"{method}: a box value changed anyway "
          f"{EDITED} is not sent; band unchanged",
          kw_ed is not None and tuple(kw_ed['frequency']) == band
          and kw_ed['min_dur'] is None and kw_ed['max_dur'] is None,
          repr(kw_ed and (kw_ed['frequency'], kw_ed['min_dur'],
                          kw_ed['max_dur'])))
    # what the detector is then built with: the published low-pass and the
    # published duration (None is accepted and means the method's own)
    built = ImprovedDetectSlowWave(method=method, frequency=kw_ed['frequency'],
                                   min_dur=kw_ed['min_dur'],
                                   max_dur=kw_ed['max_dur'])
    check(f'{method}.4', f"{method}: the detector built from these kwargs "
          f"low-passes at {band[1]:g} Hz and gates on {dur}",
          float(built.lowpass['freq']) == band[1]
          and tuple(float(x) for x in built.duration) == dur,
          repr((built.lowpass, built.duration)))
    lp = win.sw_param_widgets['lowpass']
    check(f'{method}.5', f"{method}: the tab shows the published band "
          f"read-only, and its low-pass boxes are read-only at "
          f"{band[1]:g} Hz", win.sw_band_label.text().startswith(
              f"{band[0]:g}–{band[1]:g} Hz · low-pass at {band[1]:g} Hz")
          and forced_off(lp['freq']) and forced_off(lp['order'])
          and lp['freq'].value() == band[1],
          repr((win.sw_band_label.text(), forced_off(lp['freq']),
                forced_off(lp['order']), lp['freq'].value())))

# ---- a run that finishes: summary and run scope -------------------------------
FULL_RUN[0] = True
win.count_db_events = lambda *a, **k: (0, 0)
win.verify_db_channels = lambda *a, **k: {}
win.report_db_density = lambda *a, **k: None
win.log_run_outcome = lambda *a, **k: None
finished = []
win.show_run_finished_dialog = lambda label, *a, **k: finished.append(label)
win.populate_detection_methods = lambda *a, **k: None   # reads the database
win._last_run_scope = None
for method in ('Ngo2015', 'Staresina2015'):
    n = len(summaries)
    run(method)
    for _ in range(3):
        app.processEvents()
    summ = summaries[-1]['parameters'] if len(summaries) > n else None
    scope = (win._last_run_scope or {}).get('Slow wave')
    check(f'{method}.6', f"{method}: the run finishes; the summary records "
          f"min_dur/max_dur None, the published duration and band, and the "
          f"run scope keeps the published band",
          summ is not None and summ['min_dur'] is None
          and summ['max_dur'] is None
          and summ['duration'] == list(PUB_DUR[method])
          and tuple(summ['frequency_range']) == PUBLISHED[method]
          and scope is not None and scope['method'] == method
          and tuple(scope['frequency']) == PUBLISHED[method],
          repr((summ, scope)))
FULL_RUN[0] = False

# ---- Massimini family unchanged ----------------------------------------------
for method in ('Massimini2004', 'AASM/Massimini2004'):
    win.sw_method_combo.setCurrentText(method)
    app.processEvents()
    fw = win.sw_param_widgets['filter']
    tw = win.sw_param_widgets['trough_duration']
    fw['min_freq'].setValue(0.2)
    fw['max_freq'].setValue(3.0)
    kw = run(method)
    check(f'{method}.1', f"{method}: frequency is the filter boxes "
          f"(0.2, 3.0), trough_duration the trough boxes, no "
          f"min_dur/max_dur", kw is not None
          and tuple(kw['frequency']) == (0.2, 3.0)
          and tuple(kw['trough_duration'])
          == (tw['min'].value(), tw['max'].value())
          and 'min_dur' not in kw and 'max_dur' not in kw,
          repr(kw and (kw['frequency'], kw.get('trough_duration'))))

for _ in range(3):
    app.processEvents()          # the tab posts its error dialog to the GUI thread
# every captured run stopped by the fake shows one STOP dialog; the two
# full runs show none
check('9', "the only error dialogs are the test's own stop after the "
      "capture (one per stopped run, none for the two full runs)",
      len(errors) == len(captured) - 2
      and all(str(e).endswith(': STOP') for e in errors),
      repr((len(captured), errors)))
ok_prefs, prefs_detail = gui_settings_guard.untouched()
check('10', "the real review-GUI preferences file was not touched",
      ok_prefs, prefs_detail)
del win          # not close(): closeEvent opens a modal confirm dialog

say("")
if FAILURES:
    say(f"{len(FAILURES)}/{CHECKS[0]} checks FAILED:")
    for f in FAILURES:
        say(f"  FAILED: {f}")
    sys.exit(1)
say(f"{CHECKS[0]}/{CHECKS[0]} checks passed")
