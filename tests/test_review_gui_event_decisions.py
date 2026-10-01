#!/usr/bin/env python3
"""Headless checks: per-event selection, the Event panel, decisions, keys,
reviewer name, neighbours and physiology in the review GUI (4.6.0).

Numbers in brackets are the acceptance criteria of
``_scratch/design/event-decision-spec.md`` (revision 2, section 13).
Population checks (8-15) are in ``test_review_gui_population.py``; review
sample (16-24) and precision report (45-51) are not built yet.

1. Data layer: lean montage columns, drill columns, ``event_reviews``
   readers, ``get_run_info``, the reason vocabulary.
2. Bands [1, 6]: one band per event starting in the epoch, clipped; spec
   fills, edges, glyphs; ticker agrees.
3. Selection [2-5]: click, 0.3 s tolerance, overlap cycling, brush drag and
   in-window selection never select / move / read.
4. Keys on the panel [7]: ``]`` ``[`` ``}`` ``{``, wrap, Right clears.
5. Event panel rows [25-33] from ``event_review.build_event_rows``.
6. Main window: reviewer prompt and QSettings, A / R / U flow, undo,
   comment focus, blind scoping, progress [34-39].
7. Neighbours and physiology [41-44].

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_review_gui_event_decisions.py
"""
import os
import re
import sys
import tempfile

# Windows consoles and CI pipes default to cp1252, which cannot encode some
# glyphs these checks print; replace rather than crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from PyQt5 import QtCore, QtWidgets                           # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
from PyQt5.QtTest import QTest                                # noqa: E402

TMP = tempfile.mkdtemp(prefix='tw_decisions_')
# Isolate QSettings before anything reads them.
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()     # before any frontend import

import frontend.eeg_review_gui as rg                          # noqa: E402
from frontend import event_review as er                       # noqa: E402
from frontend.channel_types import (neighbour_channels,       # noqa: E402
                                    physio_channels)
from turtlewave_hdEEG import dbwrite                          # noqa: E402
import review_population_fixture as fx                        # noqa: E402

REAL_STDOUT = sys.stdout
FAILURES = []
CHECKS = [0]


def say(*a):
    print(*a, file=REAL_STDOUT, flush=True)


def check(item, label, ok, detail=""):
    CHECKS[0] += 1
    say(f"  [{'PASS' if ok else 'FAIL'}] ({item}) {label}"
        + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"({item}) {label}: {detail}")


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# Cz spindles: five starting in epoch 0 [0, 30), three in epoch 1, one in
# epoch 3. 'u-out' is a 900 µV outlier; 'u-e' runs past the end of epoch 0.
STARTS = [2.0, 7.0, 12.0, 18.0, 29.6, 33.0, 41.0, 52.0, 95.0]
DURS = [0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8]
UUIDS = ['u-a', 'u-b', 'u-c', 'u-out', 'u-e', 'u-f', 'u-g', 'u-h', 'u-i']
AMPS = [40.0, 42.0, 38.0, 900.0, 41.0, 39.0, 43.0, 40.0, 41.0]
RUN = 'run-moelle-9-12'
RUN_B = 'run-moelle-9-12-b'


def ev_row(uuid, ch, start, dur, amp, run_id, *, in_band=1, low=0,
           near=0, amp_ratio=2.5, thresh_ratio=None, peak_val_det=5.16,
           epoch_stage='NREM2', figures=True, peak_freq_ap=10.5):
    fig = ((7, 6.0, peak_freq_ap, 15.0 if not low else 7.2, in_band, low,
            1.8, 55, 0, amp_ratio, thresh_ratio, near, 0, None)
           if figures else (None,) * 14)
    return ((uuid, 'spindle', ch, start, start + dur, dur, 'NREM2+NREM3',
             'Moelle2011', 9.0, 12.0, -amp / 2, amp, amp, run_id,
             epoch_stage) + fig + (9.8, peak_val_det, None, None, None))


def make_db(path):
    con = fx.open_schema(path)
    fx.add_run(con, RUN)
    fx.add_run(con, RUN_B, timestamp='2026-09-01T10:00:00')
    rows = [ev_row(u, 'Cz', s, d, a, RUN)
            for u, s, d, a in zip(UUIDS, STARTS, DURS, AMPS)]
    # second run on the same scope: one event on Fz
    rows.append(ev_row('u-fz', 'Fz', 20.0, 0.8, 40.0, RUN_B))
    fx.insert_rows(con, rows)
    dbwrite.store_detection_thresholds(con, RUN, 'Cz', 'Moelle2011',
                                       {'det_value_lo': 3.43}, 'uV')
    dbwrite.store_detection_thresholds(con, RUN_B, 'Fz', 'Moelle2011',
                                       {'det_value_lo': 5.16}, 'uV')
    con.commit()
    con.close()
    return path


say("=" * 78)
say("Headless: event selection, Event panel, decisions, keys, neighbours")
say("=" * 78)

# ======================================================================= 1
say("\n== 1. Data layer")
LEAN = ['channel', 'start_time', 'end_time', 'stage', 'min_amp', 'max_amp',
        'peak2peak_amp', 'freq_lower', 'freq_upper']
check('1.0', "QC_EVENT_COLS (montage-wide) is the original nine columns",
      rg.QC_EVENT_COLS == LEAN, repr(rg.QC_EVENT_COLS))
for c in ('uuid', 'duration', 'run_id', 'method', 'epoch_stage', 'in_band',
          'amp_ratio'):
    check('1.1', f"QC_DRILL_COLS includes {c}", c in rg.QC_DRILL_COLS)
check('1.1b', "local REVIEW_REASONS / DECISIONS equal the library's",
      rg.REVIEW_REASONS == tuple(dbwrite.REVIEW_REASONS)
      and rg.REVIEW_DECISIONS == tuple(dbwrite.REVIEW_DECISIONS))
check('1.1c', "every library reason has a label; both grids use library "
      "tokens, 9 buttons, 0 = other last, no 9",
      set(er.REASON_LABEL) == set(dbwrite.REVIEW_REASONS)
      and all({t for _d, t, _l, _tip in er.reason_grid(et)}
              <= set(dbwrite.REVIEW_REASONS)
              and len(er.reason_grid(et)) == 9
              and er.reason_grid(et)[-1][:2] == ('0', 'other')
              and er.digit_reason(et, '9') is None
              for et in ('spindle', 'slow_wave', 'k_complex')))
DB = make_db(os.path.join(TMP, 'review.db'))
db = rg.EventDatabase(DB)
df = db.get_events(event_type='spindle', channels=['Cz'],
                   columns=rg.QC_DRILL_COLS)
check('1.2', "drill fetch returns the drill columns and 9 Cz rows",
      list(df.columns) == rg.QC_DRILL_COLS and len(df) == 9,
      repr(list(df.columns)))
info = db.get_run_info(RUN)
check('1.3', "get_run_info parses params_json (duration_by_method)",
      info.get('params', {}).get('duration_by_method')
      == {'Moelle2011': [0.5, 3.0]} and info.get('method') == 'Moelle2011',
      repr(info.get('params')))
check('1.4b', "not-recorded texts name no backfill script and say how to "
      "get the figures", 'backfill' not in er.NOT_RECORDED_RUN
      and 'backfill' not in er.FIGURES_OFF_RUN
      and er.NOT_RECORDED_RUN.endswith('Figures are stored by detection runs '
                                       'made with 4.6 or later; re-detect '
                                       'this run to get them.'))
check('1.4', "no events-table review columns are added",
      not {'reviewed', 'review_decision'} & set(db._table_columns('events')))

# ======================================================================= 2
say("\n== 2. Bands [1, 6]")
panel = rg.EpochsPanel()
panel.resize(1400, 900)
panel.show()
app.processEvents()
selected, decisions, cleared = [], [], []
panel.eventSelected.connect(selected.append)
panel.decisionRequested.connect(decisions.append)
panel.selectionCleared.connect(lambda: cleared.append(1))
reads = []
panel.read_window = lambda ch, a, b: (reads.append((ch, a, b)) or
                                      (np.linspace(a, b, 600),
                                       np.sin(np.linspace(0, 60, 600)) * 30,
                                       20.0))
panel.set_channel('Cz', df, df, event_type='spindle', trec=120.0)


def bands(plot):
    return panel.band_items(plot)


def band_for(uuid, plot):
    s = STARTS[UUIDS.index(uuid)]
    return next((it for it in bands(plot)
                 if abs(it.getRegion()[0] - s) < 1e-9), None)


check('2.1', "[1] five events start in epoch 0 -> five bands on raw, five "
      "on filtered", len(bands(panel.raw_plot)) == 5
      and len(bands(panel.filt_plot)) == 5,
      repr((len(bands(panel.raw_plot)), len(bands(panel.filt_plot)))))
check('2.2', "u-e (29.6-30.6 s) is clipped at the epoch end (30 s)",
      abs(band_for('u-e', panel.raw_plot).getRegion()[1] - 30.0) < 1e-9,
      repr(band_for('u-e', panel.raw_plot).getRegion()))
panel._goto_epoch(1)
check('2.3', "u-e is not drawn in epoch 1 (starts in epoch 0); ticker "
      "agrees", len(bands(panel.raw_plot)) == 3
      and sum(len(it.opts['x']) for it in panel._ticker_items) == 3,
      repr(len(bands(panel.raw_plot))))
panel._goto_epoch(0)
c = band_for('u-out', panel.raw_plot)
check('2.4', "outlier: red fill alpha 46, red edge alpha 140",
      c.brush.color().getRgb() == (224, 83, 63, 46)
      and c.lines[0].pen.color().getRgb()[3] == 140,
      repr((c.brush.color().getRgb(), c.lines[0].pen.color().getRgb())))
c = band_for('u-a', panel.raw_plot)
check('2.5', "regular: text_3 fill alpha 30, no edge",
      c.brush.color().getRgb() == (136, 136, 136, 30)
      and c.lines[0].pen.style() == Qt.NoPen,
      repr((c.brush.color().getRgb(), c.lines[0].pen.style())))
panel.set_reviews({'u-a': ('accept', None, 'TK'),
                   'u-b': ('reject', 'artefact', 'TK'),
                   'u-out': ('unsure', None, 'TK')})
a_, r_, u_ = (band_for(u, panel.raw_plot) for u in ('u-a', 'u-b', 'u-out'))
check('2.6', "[6] accepted: ok fill alpha 40, solid ok edge",
      a_.brush.color().getRgb() == (105, 179, 93, 40)
      and a_.lines[0].pen.style() == Qt.SolidLine,
      repr(a_.brush.color().getRgb()))
check('2.7', "[6] rejected: grey fill alpha 20, dashed bad edge (decision "
      "replaces outlier style)", r_.brush.color().getRgb() == (136, 136, 136, 20)
      and r_.lines[0].pen.style() == Qt.DashLine
      and r_.lines[0].pen.color().name() == '#e0533f',
      repr((r_.brush.color().getRgb(), r_.lines[0].pen.style())))
check('2.8', "[6] unsure on an outlier: warn fill alpha 40, dotted edge",
      u_.brush.color().getRgb() == (224, 163, 52, 40)
      and u_.lines[0].pen.style() == Qt.DotLine,
      repr(u_.brush.color().getRgb()))
texts = [t.toPlainText() for t in panel.glyph_items()]
glyphs = sorted(t for t in texts if t != 'outlier')
check('2.9', "[6] decided bands carry ✓ ✗ ? TextItems on the raw trace",
      glyphs == sorted(['✓', '✗', '?']), repr(glyphs))
labels_on = {p: [it.toPlainText() for pp, it in panel._event_items
                 if pp is p and isinstance(it, rg.pg.TextItem)]
             for p in (panel.raw_plot, panel.filt_plot)}
check('2.9b', "[89] the outlier band carries an 'outlier' label on each "
      "plot; no 'selected' text anywhere",
      labels_on[panel.raw_plot].count('outlier') == 1
      and labels_on[panel.filt_plot].count('outlier') == 1
      and not any('selected' in t for v in labels_on.values() for t in v),
      repr(labels_on))
panel.select_event('u-c', emit=False)
s_ = band_for('u-c', panel.raw_plot)
check('2.10', "[6] selected: accent edge width 2, on top",
      s_.lines[0].pen.color().name() == rg.THEME['accent'].lower()
      and s_.lines[0].pen.widthF() == 2 and s_.zValue() == 8,
      repr((s_.lines[0].pen.color().name(), s_.lines[0].pen.widthF())))
tick_pens = [it.opts.get('pen') for it in panel._ticker_items]
check('2.11', "ticker: rejected bar is an outline, selected bar has an "
      "accent outline", any(it.opts.get('brush') is None
                            and it.opts.get('height') == 9
                            for it in panel._ticker_items)
      and any(it.opts.get('height') == 16 for it in panel._ticker_items))
panel.set_reviews({})
panel.clear_selection(emit=False)

# ======================================================================= 3
say("\n== 3. Selection [2-5]")


class FakeClick:
    def __init__(self, plot, x):
        vb = plot.getPlotItem().vb
        y = sum(vb.viewRange()[1]) / 2.0
        self._pos = vb.mapViewToScene(QtCore.QPointF(x, y))

    def button(self):
        return Qt.LeftButton

    def scenePos(self):
        return self._pos


panel._goto_epoch(0)
region_before = tuple(panel.region.getRegion())
n_reads = len(reads)
panel._on_trace_click(FakeClick(panel.raw_plot, 12.4), panel.raw_plot)
check('3.1', "[2] click at an event's midpoint emits eventSelected(uuid)",
      selected[-1:] == ['u-c'], repr(selected))
check('3.2', "[5] selecting within the epoch makes no read_window call and "
      "does not move the brush", len(reads) == n_reads
      and tuple(panel.region.getRegion()) == region_before)
n_sel = len(selected)
panel._on_trace_click(FakeClick(panel.raw_plot, 23.0), panel.raw_plot)
check('3.3', "[2] a click 1 s or more from any event emits nothing, keeps "
      "the selection", len(selected) == n_sel
      and panel._selected_uuid == 'u-c')
panel._on_trace_click(FakeClick(panel.filt_plot, 18.0 + 0.8 + 0.25),
                      panel.filt_plot)
check('3.4', "click on the filtered trace 0.25 s past u-out's end selects "
      "it (0.3 s tolerance)", selected[-1] == 'u-out', repr(selected[-1]))
check('3.5', "_event_at: 0.35 s from an edge selects nothing",
      panel._event_at(2.0 - 0.35) is None and panel._event_at(1.75) == 'u-a')
# [3] overlapping events: two runs on one scope
ov = pd.DataFrame({'channel': 'Cz', 'uuid': ['o-long', 'o-short'],
                   'start_time': [10.0, 10.9], 'end_time': [12.0, 11.3],
                   'max_amp': [40.0, 41.0]})
opanel = rg.EpochsPanel()
opanel.resize(1400, 900)
opanel.show()
app.processEvents()
osel = []
opanel.eventSelected.connect(osel.append)
opanel.set_channel('Cz', ov, ov, event_type='spindle', trec=60.0)
opanel._on_trace_click(FakeClick(opanel.raw_plot, 11.0), opanel.raw_plot)
opanel._on_trace_click(FakeClick(opanel.raw_plot, 11.0), opanel.raw_plot)
check('3.6', "[3] overlapping: first click nearest centre, second click at "
      "the same x the other", osel == ['o-long', 'o-short'], repr(osel))
opanel.close()
# [4] dragging the brush does not select
n_sel = len(selected)
panel.region.setRegion([5.0, 9.0])
app.processEvents()
check('3.7', "[4] moving the artefact brush emits no eventSelected",
      len(selected) == n_sel)

# ======================================================================= 4
say("\n== 4. Keys on the panel [7]")
panel.set_reviews({'u-e': ('accept', None, 'TK')})
panel.select_event('u-out')
panel.setFocus()
QTest.keyClick(panel, Qt.Key_BracketRight)
check('4.1', "] from u-out skips reviewed u-e, pages to epoch 1 -> u-f",
      panel._selected_uuid == 'u-f' and panel._epoch == 1,
      repr((panel._selected_uuid, panel._epoch)))
QTest.keyClick(panel, Qt.Key_BracketLeft)
check('4.2', "[ goes back to u-out", panel._selected_uuid == 'u-out')
QTest.keyClick(panel, Qt.Key_BraceRight)
check('4.3', "} selects the next event of any status (u-e, reviewed)",
      panel._selected_uuid == 'u-e', repr(panel._selected_uuid))
QTest.keyClick(panel, Qt.Key_BraceLeft)
check('4.4', "{ goes back to u-out", panel._selected_uuid == 'u-out')
panel.select_event('u-i')
QTest.keyClick(panel, Qt.Key_BracketRight)
check('4.5', "] past the last unreviewed wraps once to the earliest "
      "(u-a), last_nav 'wrapped'", panel._selected_uuid == 'u-a'
      and panel.last_nav == 'wrapped',
      repr((panel._selected_uuid, panel.last_nav)))
QTest.keyClick(panel, Qt.Key_A)
check('4.6', "A emits decisionRequested('accept')", decisions == ['accept'])
n_clear = len(cleared)
panel._next()        # Right
check('4.7', "[7] Right with the selected event in the epoch clears the "
      "selection", panel._selected_uuid is None and len(cleared) == n_clear + 1)
panel.select_event('u-c')
QTest.keyClick(panel, Qt.Key_Escape)
check('4.8', "Esc (no strip range, no filter) clears the selection",
      panel._selected_uuid is None)
panel.close()

# ======================================================================= 5
say("\n== 5. Event panel rows [25-33]")
run46 = {'method': 'Moelle2011', 'stages': 'NREM2+NREM3',
         'turtlewave_version': '4.6.0', 'timestamp': '2026-09-14T10:00:00',
         'params': {'duration_by_method': {'Moelle2011': [0.5, 3.0]},
                    'event_figures': {'spec_revision': 'v2'}}}
run45 = {'method': 'Moelle2011', 'stages': 'NREM2', 'timestamp':
         '2026-01-01T10:00:00', 'params': {'duration': [0.5, 3.0]}}
base = {'uuid': 'x', 'channel': 'PPOz', 'start_time': 6812.856,
        'end_time': 6813.456, 'duration': 0.6, 'method': 'Moelle2011',
        'freq_lower': 9.0, 'freq_upper': 12.0, 'epoch_stage': 'NREM2',
        'max_amp': 50.07, 'peak_val_det': 5.16, 'peak_freq': 9.8,
        'run_id': '3f2a9c1d', 'halfwaves_above_bg': 7, 'cycles_nominal': 6.0,
        'peak_freq_ap': 9.5, 'prominence_db': 7.2, 'in_band': 1,
        'low_prominence': 1, 'bg_rms': 1.8, 'bg_n_windows': 55,
        'bg_stage_mixed': 0, 'amp_ratio': 1.81, 'thresh_ratio': None,
        'near_bound': 0, 'near_splice': 0}


def rows_of(ev, **kw):
    kw.setdefault('event_type', 'spindle')
    kw.setdefault('run', run46)
    kw.setdefault('run_id', ev.get('run_id'))
    kw.setdefault('figures', ev)
    return er.build_event_rows(ev, **kw)


def text_of(rows, key):
    r = next((r for r in rows if r['key'] == key), None)
    return None if r is None else '\n'.join([r['value']] + r['sub'])


r = rows_of(base, outlier_thr=38.2)
keys = [x['key'] for x in r]
check('5.1', "[25] spindle row keys in order", keys ==
      ['time', 'channel', 'stage', 'detection', 'duration', 'halfwaves',
       'cycles_nominal', 'peak_freq', 'amp_bg', 'amp_thr', 'outlier'],
      repr(keys))
sw = dict(base, method='Massimini2004', det_trough=-80.0, det_ptp=120.0,
          det_zero_time=6813.0, wave_freq=0.9, freq_lower=0.5,
          freq_upper=4.0)
rsw = rows_of(sw, event_type='slow_wave', ptp_units_uv=True,
              thresholds={'max_trough_amp': -40.0, 'min_ptp': 75.0})
check('5.2', "[25] slow-wave row keys in order", [x['key'] for x in rsw] ==
      ['time', 'channel', 'stage', 'detection', 'duration', 'wave_freq',
       'amp_bg', 'amp_thr', 'sw_trough', 'sw_ptp', 'sw_neg_half', 'outlier'],
      repr([x['key'] for x in rsw]))
check('5.3', "[26] half-waves sub-line exact; cycles never an integer",
      text_of(r, 'halfwaves').split('\n')[1]
      == 'half-waves standing out from background (2.5× bg RMS)'
      and not re.match(r'^\d+$', text_of(r, 'cycles_nominal').split('\n')[0]),
      repr((text_of(r, 'halfwaves'), text_of(r, 'cycles_nominal'))))
pf = text_of(r, 'peak_freq')
check('5.4', "[27] 0.6 s, 7.2 dB: 'low prominence (unreliable under 1 s)' "
      "and 'coarse (resolution 1.7 Hz)'",
      'prominence 7.2 dB · low prominence (unreliable under 1 s)' in pf
      and 'coarse (resolution 1.7 Hz)' in pf, repr(pf))
long_ = dict(base, end_time=base['start_time'] + 1.2, duration=1.2)
pf12 = text_of(rows_of(long_), 'peak_freq')
pf15 = text_of(rows_of(dict(long_, prominence_db=15.0, low_prominence=0)),
               'peak_freq')
check('5.5', "[27] 1.2 s: 'low prominence' without 'unreliable'; 15 dB "
      "still shows 'prominence 15.0 dB'", 'low prominence' in pf12
      and 'unreliable' not in pf12 and 'prominence 15.0 dB' in pf15,
      repr((pf12, pf15)))
check('5.6', "[28] in_band false -> OFF BAND; no peak -> no-peak text",
      'OFF BAND' in text_of(rows_of(dict(base, in_band=0, peak_freq_ap=8.0)),
                            'peak_freq')
      and text_of(rows_of(dict(base, peak_freq_ap=None, in_band=None)),
                  'peak_freq').startswith('— no peak above the 1/f background'))
check('5.7', "[29] stage mix and insufficient background",
      'near a stage change' in text_of(rows_of(dict(base, bg_stage_mixed=1)),
                                       'amp_bg')
      and 'too little clean background' in text_of(
          rows_of(dict(base, amp_ratio=None, bg_rms=None, bg_n_windows=6)),
          'amp_bg'))
check('5.8', "[30] near_bound -1 -> 'at the floor of the run limits'",
      'at the floor of the run limits' in text_of(
          rows_of(dict(base, near_bound=-1, duration=0.52,
                       end_time=base['start_time'] + 0.52)), 'duration'))
t45 = text_of(rows_of(dict(base, halfwaves_above_bg=None, cycles_nominal=None,
                           peak_freq_ap=None, amp_ratio=None),
                      run=run45, thresholds=pd.DataFrame(),
                      figure_state='missing'), 'amp_thr')
check('5.9', "[31] pre-4.6 run, no thresholds: exact not-recorded text",
      t45 == 'Threshold not recorded for this run (detected with 4.5 or '
             'earlier)', repr(t45))
ta = text_of(rows_of(base, thresholds={'det_value_lo': 3.43}), 'amp_thr')
tb = text_of(rows_of(base, thresholds={'det_value_lo': 5.16}), 'amp_thr')
check('5.10', "[31] two runs on one scope give different amp_thr (1.5× vs "
      "1.0× barely crossed)", ta.startswith('1.5×') and tb.startswith('1.0×')
      and 'barely crossed' in tb, repr((ta, tb)))
branches = {
    'Moelle2011': ({'det_value_lo': 3.43}, '1.5×'),
    'Ferrarelli2007': ({'det_value_lo': 3.43, 'sel_value': 2.0}, '1.5×'),
    'Nir2011': ({'det_value_lo': 3.43}, '1.5×'),
    'Ray2015': ({'det_value_lo': 2.33, 'sel_value': 0.1}, 'no ratio'),
    'Wamsley2012': ({'det_value_lo': 12.0}, 'no ratio'),
    'Martin2013': ({'det_value_lo': 4.0}, 'no ratio'),
    'Lacourse2018': ({'abs_pow_thresh': 1.25, 'rel_pow_thresh': 1.6,
                      'covar_thresh': 1.3, 'corr_thresh': 0.69},
                     '4 thresholds'),
    'CIRUS': ({}, 'not available for CIRUS'),
    'Massimini2004': ({'max_trough_amp': -40.0, 'min_ptp': 75.0},
                      'meets both'),
    'Ngo2015': ({'peak_thresh_factor': 1.25}, 'relative'),
    'Staresina2015': ({'ptp_percentile': 75.0}, 'relative'),
}
for m, (th, want) in branches.items():
    et = 'slow_wave' if m in ('Massimini2004', 'Ngo2015',
                              'Staresina2015') else 'spindle'
    got = er.threshold_row(m, dict(sw if et == 'slow_wave' else base),
                           th, True)['value']
    check('5.11', f"[32] amp_thr value for {m}", got == want,
          repr((got, want)))
fails = er.threshold_row('Massimini2004', dict(sw, det_ptp=60.0),
                         {'max_trough_amp': -40.0, 'min_ptp': 75.0}, True)
check('5.12', "[32] Massimini failing one criterion: 'fails 1 of 2'",
      fails['value'] == 'fails 1 of 2', repr(fails['value']))
check('5.14', "run_stages: list repr, joint token, single stage, params",
      er.run_stages({'stages': "['NREM2', 'NREM3']"}) == ['NREM2', 'NREM3']
      and er.run_stages({'stages': 'NREM2NREM3'}) == ['NREM2', 'NREM3']
      and er.run_stages({'stages': 'NREM2'}) == ['NREM2']
      and er.run_stages({'stages': "['NREM2']"}) == ['NREM2']
      and er.run_stages({'stages': "['x']", 'params': {'stages': ['NREM3']}})
      == ['NREM3'],
      repr([er.run_stages({'stages': "['NREM2', 'NREM3']"}),
            er.run_stages({'stages': 'NREM2NREM3'})]))
check('5.15', "run_ref_chan: params key wins ([] = none); else the column "
      "repr", er.run_ref_chan({'params': {'ref_chan': []},
                               'ref_chan': "['M1']"}) == []
      and er.run_ref_chan({'params': {}, 'ref_chan': "['M1', 'M2']"})
      == ['M1', 'M2'] and er.run_ref_chan({'ref_chan': '[]'}) == []
      and er.run_ref_chan({'ref_chan': 'None'}) == [])
check('5.16', "rereference keeps the target in an average reference "
      "(Wonambi montage) and matches the formula",
      np.allclose(rg._rereference_like_wonambi(
          np.array([[1., 2, 3], [3, 3, 3], [5, 4, 3]]), ['Cz', 'Fz', 'Pz'],
          np.arange(3) / 100.0, 100.0, 'Cz', ['Cz', 'Fz', 'Pz']),
          np.array([1., 2, 3]) - np.array([3., 3, 3]))
      and np.allclose(er.rereference(
          np.array([[1., 2, 3], [3, 3, 3], [5, 4, 3]]), ['Cz', 'Fz', 'Pz'],
          'Cz', ['Cz', 'Fz', 'Pz']), [-2., -1, 0]))
off_run = dict(run46, params=dict(run46['params'], event_figures=None))
toff = text_of(rows_of(dict(base, halfwaves_above_bg=None), run=off_run,
                       thresholds=pd.DataFrame(), figure_state='unavailable',
                       figure_note=er.FIGURES_OFF_FIG), 'amp_thr')
hoff = text_of(rows_of(dict(base, halfwaves_above_bg=None), run=off_run,
                       figure_state='unavailable',
                       figure_note=er.FIGURES_OFF_FIG), 'halfwaves')
check('5.17', "4.6 run with figures off: not the 4.5 wording, figures say "
      "switched off", toff == 'Threshold not recorded for this run'
      and hoff == '— ' + er.FIGURES_OFF_FIG, repr((toff, hoff)))
crit = text_of(rows_of(sw, event_type='slow_wave', thresholds={
    'max_trough_amp': -40.0, 'min_ptp': 75.0, 'trough_duration_lo': 0.25,
    'trough_duration_hi': 1.0}), 'sw_neg_half')
check('5.18', "negative half-wave criterion read from the stored "
      "trough_duration_lo / hi", 'criterion 0.25–1 s' in crit, repr(crit))
nulls = {k: None for k in base}
nulls.update({'uuid': 'n', 'channel': 'Cz', 'method': 'Moelle2011'})
bad = []
for et, run_ in (('spindle', run46), ('spindle', run45),
                 ('slow_wave', run46), ('slow_wave', {})):
    for state in ('stored', 'missing', 'computing'):
        for row in er.build_event_rows(nulls, event_type=et, run=run_,
                                       run_id=None, figures=nulls,
                                       figure_state=state):
            for t in [row['value']] + row['sub']:
                if (not str(t).strip() or re.search(r'\bnan\b|\bNone\b',
                                                     str(t))):
                    bad.append((et, state, row['key'], t))
check('5.13', "[33] all-NULL rows: no nan / None / empty text anywhere",
      not bad, repr(bad[:4]))

# ======================================================================= 6
say("\n== 6. Main window: reviewer, decisions, undo, keys [34-39]")
win = rg.EventReviewGUI()
win.db = db
win.qc_widget.evt_combo.blockSignals(True)
win.qc_widget.evt_combo.setCurrentText('spindle')
win.qc_widget.evt_combo.blockSignals(False)
win._qc_events_df = db.get_events(event_type='spindle',
                                  columns=rg.QC_EVENT_COLS)
win.show()
win.activateWindow()
app.processEvents()
asked = []
win._ask_reviewer_name = lambda prefill: (asked.append(prefill) or ('', False))
win.on_qc_drill('Cz', switch_tab=True)
app.processEvents()
ep = win.epochs_panel
evp = win.detail_dock_w.event_panel
check('6.0', "reviewer segment reads 'Reviewer: not set'; review keys live "
      "on the Epochs tab", win.seg_reviewer.text() == 'Reviewer: not set'
      and win.review_shortcuts_enabled())


def rows_db(uuid):
    return db.conn.execute(
        "SELECT reviewer, decision, reason, comment, reviewed_at FROM "
        "event_reviews WHERE uuid = ? ORDER BY reviewer", (uuid,)).fetchall()


def key(k, text=''):
    QTest.keyClick(ep, k)
    app.processEvents()


ep.select_event('u-a')
app.processEvents()
key(Qt.Key_A)
check('6.1', "[34] no name: A opens the name dialog; Cancel writes nothing",
      len(asked) == 1 and rows_db('u-a') == []
      and win.status_bar.currentMessage()
      == 'Decision not saved — a reviewer name is needed.',
      repr((asked, win.status_bar.currentMessage())))
win._ask_reviewer_name = lambda prefill: ('  TK  ', True)
key(Qt.Key_A)
check('6.2', "name given: saved stripped, in QSettings, on the status "
      "segment; A writes accept", win.reviewer_name == 'TK'
      and rg._review_settings().value('review/reviewer_name') == 'TK'
      and win.seg_reviewer.text() == 'Reviewer: TK'
      and [r[:3] for r in rows_db('u-a')] == [('TK', 'accept', None)],
      repr(rows_db('u-a')))
check('6.3', "auto-advance selected the next unreviewed event (u-b)",
      ep._selected_uuid == 'u-b', repr(ep._selected_uuid))
key(Qt.Key_R)
check('6.4', "[35] R alone writes nothing", rows_db('u-b') == [])
key(Qt.Key_Escape)
check('6.5', "[35] R, Esc writes nothing; status says cancelled",
      rows_db('u-b') == [] and win._armed is None
      and win.status_bar.currentMessage()
      == 'Reject cancelled — no reason chosen.')
key(Qt.Key_R)
key(Qt.Key_1)
check('6.6', "[35] R, 1 writes reject / artefact",
      [r[:3] for r in rows_db('u-b')] == [('TK', 'reject', 'artefact')],
      repr(rows_db('u-b')))
msg = win.status_bar.currentMessage()
check('6.7', "status after a write names it, the next unreviewed and undo",
      msg.startswith('Rejected spindle on Cz at 00:00:07.0 (artefact) · next '
                     'unreviewed ') and msg.endswith(' · Ctrl+Z to undo'),
      repr(msg))
cur = ep._selected_uuid
key(Qt.Key_U)
key(Qt.Key_Return)
check('6.8', "[35] U, Enter writes unsure with no reason",
      [r[:3] for r in rows_db(cur)] == [('TK', 'unsure', None)],
      repr(rows_db(cur)))
cur = ep._selected_uuid
key(Qt.Key_R)
key(Qt.Key_9)
check('6.9a', "[85] R, 9 writes nothing (9 is not a reason key)",
      rows_db(cur) == [] and win._armed == 'reject', repr(rows_db(cur)))
key(Qt.Key_0)
check('6.9', "[35] R, 0 waits for a comment (nothing written)",
      rows_db(cur) == [] and evp.comment.hasFocus()
      and not win.review_shortcuts_enabled())
QTest.keyClicks(evp.comment, 'spike train')
QTest.keyClick(evp.comment, Qt.Key_Return)
app.processEvents()
check('6.10', "[35] … then the comment and Enter write reject / other",
      [r[:4] for r in rows_db(cur)] == [('TK', 'reject', 'other',
                                         'spike train')], repr(rows_db(cur)))
# [36] change then undo twice
ep.select_event('u-g')
app.processEvents()
key(Qt.Key_A)
# back-date the accept by one hour so a same-second coincidence cannot pass
import datetime as _dtm                                       # noqa: E402
past = (_dtm.datetime.now().astimezone() - _dtm.timedelta(hours=1)
        ).isoformat(timespec='seconds')
db.conn.execute("UPDATE event_reviews SET reviewed_at = ? WHERE uuid = 'u-g' "
                "AND reviewer = 'TK'", (past,))
db.conn.commit()
first = rows_db('u-g')
ep.select_event('u-g')
app.processEvents()
key(Qt.Key_R)
key(Qt.Key_7)
check('6.11', "[36] A then R,7 (spindle grid) leaves one row reject / "
      "arousal",
      [r[:3] for r in rows_db('u-g')] == [('TK', 'reject', 'arousal')],
      repr(rows_db('u-g')))
check('6.12', "status names the change",
      win.status_bar.currentMessage() == 'Changed from accepted to rejected '
                                         '(arousal) · Ctrl+Z to undo',
      repr(win.status_bar.currentMessage()))
QTest.keyClick(ep, Qt.Key_Z, Qt.ControlModifier)
app.processEvents()
check('6.13', "[36] Ctrl+Z restores accept and its reviewed_at (one hour "
      "old), selects it", rows_db('u-g') == first and first[0][4] == past
      and ep._selected_uuid == 'u-g'
      and win.status_bar.currentMessage().startswith(
          'Undid reject of Cz 00:00:41.0 — now accepted'),
      repr((rows_db('u-g'), win.status_bar.currentMessage())))
QTest.keyClick(ep, Qt.Key_Z, Qt.ControlModifier)
app.processEvents()
check('6.14', "[36] a second Ctrl+Z deletes the row",
      rows_db('u-g') == [], repr(rows_db('u-g')))
# undo of a decision on another channel re-drills
win.on_qc_drill('Fz', switch_tab=False)
ep.select_event('u-fz')
key(Qt.Key_A)
win.on_qc_drill('Cz', switch_tab=False)
QTest.keyClick(ep, Qt.Key_Z, Qt.ControlModifier)
app.processEvents()
check('6.15', "[36] undo of a decision on another channel re-drills it",
      ep._channel == 'Fz' and ep._selected_uuid == 'u-fz'
      and rows_db('u-fz') == [], repr((ep._channel, ep._selected_uuid)))
win.on_qc_drill('Cz', switch_tab=False)
ep.select_event('u-h')
evp.comment.setFocus()
app.processEvents()
QTest.keyClicks(evp.comment, 'ar')
app.processEvents()
check('6.16', "[37] typing a / r in the comment field writes no decision",
      rows_db('u-h') == [] and evp.comment.text() == 'ar'
      and not win.review_shortcuts_enabled())
evp.comment.clear()
ep.setFocus()
app.processEvents()
# [38] another reviewer's decision does not count as reviewed
dbwrite.store_event_review(db.conn, 'u-h', 'accept', 'JS')
win.on_qc_drill('Cz', switch_tab=False)
mine = {u for u, v in ep._reviews.items()}
ep.select_event('u-g')
key(Qt.Key_BracketRight)
check('6.17', "[38] ] skips events decided by TK, not those decided only "
      "by JS (u-h)", ep._selected_uuid == 'u-h' and 'u-h' not in mine,
      repr((ep._selected_uuid, sorted(mine))))
check('6.18', "blind by default: JS's decision is hidden on the Current line",
      'Also decided by 1 other reviewer(s); hidden so your decisions stay '
      'independent.' in evp.current_sub.text()
      and evp.current_lbl.text() == 'Not reviewed',
      repr((evp.current_lbl.text(), evp.current_sub.text())))
check('6.19', "Show other reviewers is off at launch", not
      win.act_show_others.isChecked())
win._confirm_show_others = lambda: True
win.act_show_others.setChecked(True)
app.processEvents()
check('6.20', "turning it on (confirmed) lists JS's decision",
      'JS: accepted · ' in evp.current_sub.text(),
      repr(evp.current_sub.text()))
win.act_show_others.setChecked(False)
# [39] progress
n_dec = len(ep._reviews)
counts = {d: sum(1 for v in ep._reviews.values() if v[0] == d)
          for d in ('accept', 'reject', 'unsure')}
check('6.21', "[39] progress line counts this reviewer only",
      evp.progress_lbl.text() ==
      f"Progress  {n_dec} reviewed / 9 on Cz · {counts['accept']} accepted · "
      f"{counts['reject']} rejected · {counts['unsure']} unsure",
      repr(evp.progress_lbl.text()))
win.set_reviewer_name('JS')
check('6.22', "changing the name rescopes bands and says so",
      set(ep._reviews) == {'u-h'} and win.status_bar.currentMessage() ==
      "Changing the name shows that reviewer's decisions and sample progress "
      "instead.", repr(set(ep._reviews)))
win.set_reviewer_name('TK')
win.tabs.setCurrentIndex(0)
app.processEvents()
check('6.23', "review keys are off on the Channels tab",
      not win.review_shortcuts_enabled())
win.tabs.setCurrentIndex(1)
app.processEvents()
ep.select_event('u-a')
evp.clear_btn.click()
app.processEvents()
check('6.24', "Clear deletes this reviewer's decision and Ctrl+Z restores it",
      rows_db('u-a') == [], repr(rows_db('u-a')))
QTest.keyClick(ep, Qt.Key_Z, Qt.ControlModifier)
app.processEvents()
check('6.25', "… restored", [r[:2] for r in rows_db('u-a')] == [('TK',
                                                                 'accept')])
ep.select_event('u-c')
check('6.26', "selection status names stage and decision",
      win.status_bar.currentMessage().startswith(
          'Selected spindle on Cz at 00:00:12.0 · NREM2 · ')
      and win.status_bar.currentMessage().endswith(
          ' — A accept · R reject · U unsure'),
      repr(win.status_bar.currentMessage()))
check('6.27', "the dock panel shows the spindle rows for the selection",
      evp.row_keys() == ['time', 'channel', 'stage', 'detection', 'duration',
                         'halfwaves', 'cycles_nominal', 'peak_freq', 'amp_bg',
                         'amp_thr', 'outlier']
      and evp.row_text('amp_thr').startswith('1.5×'), repr(evp.row_keys()))
win.close()

# ======================================================================= 7
say("\n== 7. Neighbours and physiology [41-44]")
CH = ['Fz', 'Cz', 'Pz', 'POz', 'PPOz', 'O1', 'O2', 'P3', 'P4', 'C3', 'VEOG',
      'HEOG', 'EMGChin', 'ECG']
TYPES = ['EEG'] * 10 + ['EOG', 'EOG', 'EMG', 'ECG']
coords = {'Fz': (0, 0.5), 'Cz': (0, 0), 'Pz': (0, -0.4), 'POz': (0, -0.6),
          'PPOz': (0, -0.5), 'O1': (-0.2, -0.8), 'O2': (0.2, -0.8),
          'P3': (-0.3, -0.4), 'P4': (0.3, -0.4), 'C3': (-0.4, 0)}
chs, src, _ = neighbour_channels('PPOz', [c for c, t in zip(CH, TYPES)
                                          if t == 'EEG'], coords, k=6)
check('7.1', "[42] by position: 6 nearest, no EOG/EMG/ECG",
      src == 'position' and len(chs) == 6 and chs[0] in ('POz', 'Pz')
      and not {'VEOG', 'HEOG', 'EMGChin', 'ECG'} & set(chs), repr(chs))
check('7.2', "physio_channels: none typed -> {}; typed -> EOG, EOG, chin "
      "EMG, ECG", physio_channels(CH, None) == {}
      and physio_channels(CH, TYPES) == {'eog': ['VEOG', 'HEOG'],
                                         'emg': 'EMGChin', 'ecg': 'ECG'})


class FakeData:
    def __init__(self, channels, types, interp=()):
        self.channels = list(channels)
        self.header = {'chan_type': list(types), 'interp_channels': list(interp),
                       's_freq': 100.0}
        self.n_reads = 0

    def read_data(self, chan, begtime, endtime):
        self.n_reads += 1
        n = max(2, int((endtime - begtime) * 100))
        ts = begtime + np.arange(n) / 100.0
        arr = np.vstack([np.sin(2 * np.pi * 10 * ts) * 20 for _ in chan])
        return type('W', (), {'data': [arr], 's_freq': 100.0,
                              'axis': {'time': [ts],
                                       'chan': [np.array(chan)]}})()


db = rg.EventDatabase(DB)          # the previous window closed it
win = rg.EventReviewGUI()
win.db = db
win.eeg_data = FakeData(CH, TYPES, interp=['P4'])
win.qc_widget.evt_combo.blockSignals(True)
win.qc_widget.evt_combo.setCurrentText('spindle')
win.qc_widget.evt_combo.blockSignals(False)
win._qc_events_df = db.get_events(event_type='spindle',
                                  columns=rg.QC_EVENT_COLS)
win._refresh_physio_channels()
win.detail_dock_w._coords = None
win.on_qc_drill('Cz', switch_tab=True)
ep = win.epochs_panel
ep.neighbours.set_open(True)
ep.select_event('u-a')
app.processEvents()
hdr = ep.neighbours.header.text()
check('7.3', "[41] without coordinates: 'from the same region'",
      'from the same region' in hdr, repr(hdr))
win.detail_dock_w._coords = dict(coords, Cz=(0.0, 0.0))
ep.select_event('u-b')
hdr = ep.neighbours.header.text()
labels = ep.neighbours.plot.row_labels()
check('7.4', "[41, 42] with coordinates: 'by electrode position', target "
      "first, ≤ 6 others, interpolated P4 shown as ~P4, no physio channels",
      'by electrode position' in hdr and labels[0].startswith('Cz')
      and len(labels) <= 7 and not any(l.startswith(('VEOG', 'HEOG', 'EMG',
                                                     'ECG')) for l in labels)
      and any(l.startswith('~P4') for l in labels), repr((hdr, labels)))
ep.neighbours.set_open(False)
n0 = ep.neighbours.n_reads
ep.select_event('u-c')
check('7.5', "[43] collapsed: selecting an event triggers no neighbour read",
      ep.neighbours.n_reads == n0)
check('7.6', "[44] physiology rows in the order EOG, EOG, Chin EMG, ECG",
      ep.physio.row_titles() == ['EOG · VEOG', 'EOG · HEOG',
                                 'Chin EMG · EMGChin', 'ECG · ECG']
      and ep.physio.isVisibleTo(ep), repr(ep.physio.row_titles()))
win.eeg_data = FakeData(CH[:10], TYPES[:10])
win._refresh_physio_channels()
check('7.7', "[44] no typed EOG/EMG/ECG: physiology group hidden",
      not ep.physio.isVisibleTo(ep) and ep.physio.row_titles() == [])
win.close()

# ======================================================================= 8
say("\n== 8. Pre-4.6 row: figures computed on selection with the run's ref")


class SpindleData(FakeData):
    """Pink-ish noise with a 10.5 Hz, 1 s Hann-tapered burst at 100.0 s on
    Cz; Fz carries a 3 µV offset that re-referencing must remove."""

    def __init__(self):
        super().__init__(['Cz', 'Fz', 'Pz'], ['EEG'] * 3)
        self.calls = []
        self.duration = 400.0

    def read_data(self, chan, begtime, endtime):
        self.calls.append(list(chan))
        fs = 250.0
        n = int(round((endtime - begtime) * fs))
        ts = begtime + np.arange(n) / fs
        r = np.random.default_rng(int(begtime * 10) % 1000)
        rows = []
        for c in chan:
            x = np.cumsum(r.normal(0, 1, n)) * 0.3
            x = x - np.convolve(x, np.ones(250) / 250, mode='same')
            if c == 'Cz':
                m = (ts >= 100.0) & (ts <= 101.0)
                x = x.copy()
                x[m] += 20 * np.hanning(m.sum()) * np.sin(
                    2 * np.pi * 10.5 * (ts[m] - 100.0))
            rows.append(x + (3.0 if c == 'Fz' else 0.0))
        return type('W', (), {'data': [np.vstack(rows)], 's_freq': fs,
                              'axis': {'time': [ts],
                                       'chan': [np.array(chan)]}})()


P_OLD = os.path.join(TMP, 'old.db')
con = fx.open_schema(P_OLD)
# written as the library writes a 4.5 run: stages "['NREM2', 'NREM3']" and
# ref_chan "['Fz']" as str(list) columns, ref_chan absent from params_json
fx.add_run(con, 'run-old', figures=False, version='4.5.0', ref_chan=['Fz'],
           ref_in_params=False)
fx.add_run(con, 'run-avg', figures=False, version='4.5.0',
           ref_chan=['Cz', 'Fz', 'Pz'], timestamp='2026-01-02T10:00:00')
fx.add_run(con, 'run-m1', figures=False, version='4.5.0', ref_chan=['M1'],
           timestamp='2026-01-03T10:00:00')
fx.add_run(con, 'run-new', timestamp='2026-01-04T10:00:00')
fx.insert_rows(con, [
    ev_row('old-1', 'Cz', 100.0, 1.0, 40.0, 'run-old', figures=False,
           peak_val_det=None),
    ev_row('avg-1', 'Cz', 160.0, 1.0, 40.0, 'run-avg', figures=False,
           peak_val_det=None),
    ev_row('m1-1', 'Cz', 220.0, 1.0, 40.0, 'run-m1', figures=False,
           peak_val_det=None),
    ev_row('new-1', 'Cz', 280.0, 1.0, 40.0, 'run-new', figures=False)])
con.commit()
con.close()
db = rg.EventDatabase(P_OLD)
win = rg.EventReviewGUI()
win.db = db
win.qc_widget.evt_combo.blockSignals(True)
win.qc_widget.evt_combo.setCurrentText('spindle')
win.qc_widget.evt_combo.blockSignals(False)
win._qc_events_df = db.get_events(event_type='spindle',
                                  columns=rg.QC_EVENT_COLS)
ep = win.epochs_panel
evp = win.detail_dock_w.event_panel
# scored NREM2 / NREM3 epochs over the whole read, so the background windows
# survive the run's stage exclusion only if the stored stages parse
win._epoch_table = lambda: rg.EpochTable(
    [(30.0 * i, 30.0 * (i + 1), 'NREM2' if i % 2 else 'NREM3')
     for i in range(14)])
win.on_qc_drill('Cz', switch_tab=True)
ep.select_event('old-1')
check('8.1', "pre-4.6 row without EEG: figures 'not recorded' and the load "
      "note", evp.row_text('halfwaves').startswith('— not recorded for this run')
      and evp.note_lbl.text() == er.LOAD_EEG_NOTE
      and evp.row_text('amp_thr') == er.THRESHOLD_NOT_RECORDED,
      repr((evp.row_text('halfwaves'), evp.row_text('amp_thr'))))
win.eeg_data = SpindleData()
ep.clear_selection()
ep.select_event('old-1')
check('8.2', "with EEG: the four figure rows show 'computing…' first",
      all(evp.row_text(k) == 'computing…'
          for k in ('halfwaves', 'cycles_nominal', 'peak_freq', 'amp_bg')),
      repr([evp.row_text(k) for k in ('halfwaves', 'peak_freq')]))
for _ in range(5):
    app.processEvents()
hw = evp.row_text('halfwaves')
pf = evp.row_text('peak_freq')
check('8.3', "then computed: a half-wave count and a peak near 10.5 Hz, "
      "labelled as computed now", re.match(r'^\d+\n', hw or '') is not None
      and re.search(r'1[01]\.\d Hz', pf or '') is not None
      and evp.row('halfwaves')['tooltip'] == er.COMPUTED_TIP,
      repr((hw, pf)))
check('8.4', "the read included the run's reference channel (Fz) with Cz "
      "(parsed from the str(list) column)",
      any(set(c) == {'Cz', 'Fz'} for c in win.eeg_data.calls),
      repr(win.eeg_data.calls))
bg = evp.row_text('amp_bg')
check('8.5', "stages stored as \"['NREM2', 'NREM3']\" keep the background: "
      "amp vs background is a ratio, not 'too little clean background'",
      re.match(r'^\d+\.\d×\n', bg or '') is not None
      and 'too little' not in bg, repr(bg))
win.eeg_data.calls.clear()
ep.select_event('avg-1')
for _ in range(5):
    app.processEvents()
check('8.6', "average reference including the target: the figure read is "
      "Cz, Fz, Pz in one call (target kept in the reference)",
      [c for c in win.eeg_data.calls if len(c) > 1] == [['Cz', 'Fz', 'Pz']]
      and re.match(r'^\d+\n', evp.row_text('halfwaves') or ''),
      repr((win.eeg_data.calls, evp.row_text('halfwaves'))))
import logging as _logging                                    # noqa: E402
caught = []
h = _logging.Handler()
h.emit = lambda rec: caught.append((rec.levelno, rec.getMessage()))
rg.logger.addHandler(h)
ep.select_event('m1-1')
for _ in range(5):
    app.processEvents()
rg.logger.removeHandler(h)
check('8.7', "reference channel not in the recording: WARNING naming it, no "
      "silent fall-back to the stored reference",
      any(lv == _logging.WARNING and 'M1' in m for lv, m in caught)
      and evp.row_text('halfwaves')
      == '— not computed: reference channel M1 not in this recording',
      repr((caught, evp.row_text('halfwaves'))))
ep.select_event('new-1')
check('8.8', "4.6 run with figures, this row all NULL: 'figures not "
      "computed for this channel', not 'stored'",
      evp.row_text('halfwaves') == '— ' + er.FIGURES_FAILED_FIG,
      repr(evp.row_text('halfwaves')))
win.close()


# ======================================================================= 9
say("\n== 9. Revision 3: grids, header, REVIEW STATUS, keys, status bar, marks")
P9 = os.path.join(TMP, 'rev3.db')
con = fx.open_schema(P9)
fx.add_run(con, RUN)
fx.add_run(con, 'run-sw', event_type='slow_wave', method='Massimini2004',
           band=(0.5, 4.0), timestamp='2026-09-15T10:00:00')
rows9 = [ev_row(u, 'Cz', s, d, a, RUN)
         for u, s, d, a in zip(UUIDS, STARTS, DURS, AMPS)]
for i, st in enumerate((5.0, 15.0, 25.0)):
    r = list(ev_row(f'sw-{i}', 'Cz', st, 1.0, 80.0, 'run-sw'))
    r[1], r[7], r[8], r[9] = 'slow_wave', 'Massimini2004', 0.5, 4.0
    rows9.append(tuple(r))
fx.insert_rows(con, rows9)
con.commit()
con.close()
db9 = rg.EventDatabase(P9)
win = rg.EventReviewGUI()
win.db = db9
win.eeg_data = FakeData(CH, TYPES)
win._refresh_physio_channels()
win._ask_reviewer_name = lambda prefill: ('TK', True)
win.set_reviewer_name('TK')


def use(evt):
    win.qc_widget.evt_combo.blockSignals(True)
    win.qc_widget.evt_combo.setCurrentText(evt)
    win.qc_widget.evt_combo.blockSignals(False)
    win._qc_events_df = db9.get_events(event_type=evt,
                                       columns=rg.QC_EVENT_COLS)
    win.on_qc_drill('Cz', switch_tab=True)
    app.processEvents()


win.show()
win.activateWindow()
app.processEvents()
use('spindle')
ep = win.epochs_panel
evp = win.detail_dock_w.event_panel
fd = win.filter_dock
check('9.84a', "[84] spindle grid texts", evp.grid_texts() ==
      ['1  Artefact', '2  Eye movement', '3  Not in raw', '4  Filter ringing',
       '5  Off-band', '6  Too short', '7  Arousal', '8  Single channel',
       '0  Other'], repr(evp.grid_texts()))
check('9.87', "[87] comment placeholder and grid header",
      evp.comment.placeholderText() == 'Comment (C) — required for "other"'
      and evp.grid_hdr.text() == 'Reason (required for Reject)')
ep._goto_epoch(0)
ep.select_event('u-c')
app.processEvents()
check('9.79a', "[79] header EVENT 3 OF 5 IN EPOCH for the third of five",
      evp.header_lbl.text() == 'EVENT 3 OF 5 IN EPOCH',
      repr(evp.header_lbl.text()))
press9 = lambda k: (QTest.keyClick(ep, k), app.processEvents())  # noqa: E731
press9(Qt.Key_R)
press9(Qt.Key_4)
press9(Qt.Key_Z) if False else None
check('9.85a', "[85] spindle R,4 writes filter-ringing",
      rows_db9 := [r[:3] for r in db9.conn.execute(
          "SELECT reviewer, decision, reason FROM event_reviews WHERE uuid = "
          "'u-c'")] == [('TK', 'reject', 'filter-ringing')],
      repr(list(db9.conn.execute("SELECT decision, reason FROM event_reviews"
                                 " WHERE uuid = 'u-c'"))))
ep.select_event('u-b')
press9(Qt.Key_R)
press9(Qt.Key_5)
check('9.85b', "[85] spindle R,5 writes off-band",
      list(db9.conn.execute("SELECT decision, reason FROM event_reviews "
                            "WHERE uuid = 'u-b'")) == [('reject', 'off-band')])
ep.select_event('u-a')
press9(Qt.Key_3)
check('9.86a', "[86] a digit with nothing armed writes nothing",
      list(db9.conn.execute("SELECT 1 FROM event_reviews WHERE uuid = 'u-a'"))
      == [])
evp.reason_buttons['not-in-raw'].click()
app.processEvents()
check('9.86b', "[86] one click on '3  Not in raw' with nothing armed writes "
      "nothing; it arms Reject with that reason preselected",
      list(db9.conn.execute("SELECT 1 FROM event_reviews WHERE uuid = 'u-a'"))
      == [] and win._armed == 'reject'
      and evp.reason_buttons['not-in-raw'].isChecked()
      and evp.hint_lbl.text() == 'Click Not in raw again or press Enter to '
                                 'reject (Not in raw).',
      repr(evp.hint_lbl.text()))
evp.reason_buttons['not-in-raw'].click()
app.processEvents()
check('9.86b2', "[86] a second click writes reject / not-in-raw",
      list(db9.conn.execute("SELECT decision, reason FROM event_reviews "
                            "WHERE uuid = 'u-a'")) == [('reject',
                                                        'not-in-raw')])
ep.select_event('u-f')
evp.reason_buttons['artefact'].click()
evp.reason_buttons['too-short'].click()
app.processEvents()
switched = (win._preselect == 'too-short' and list(db9.conn.execute(
    "SELECT 1 FROM event_reviews WHERE uuid = 'u-f' AND reviewer = 'TK'"))
    == [])
QTest.keyClick(ep, Qt.Key_Escape)
app.processEvents()
check('9.86d', "[86] a different reason switches the preselection; Esc "
      "cancels; nothing written", switched and win._armed is None
      and list(db9.conn.execute("SELECT 1 FROM event_reviews WHERE uuid = "
                                "'u-f' AND reviewer = 'TK'")) == [])
evp.reason_buttons['off-band'].click()
QTest.keyClick(ep, Qt.Key_Return)
app.processEvents()
check('9.86e', "[86] click then Enter writes the preselected reason",
      list(db9.conn.execute("SELECT decision, reason FROM event_reviews "
                            "WHERE uuid = 'u-f' AND reviewer = 'TK'"))
      == [('reject', 'off-band')])
ep.select_event('u-out')
evp.comment.clear()
evp.reason_buttons['other'].click()
evp.reason_buttons['other'].click()
app.processEvents()
check('9.86c', "[86] two clicks on '0  Other' wait for a comment",
      list(db9.conn.execute("SELECT 1 FROM event_reviews WHERE uuid = "
                            "'u-out'")) == [] and evp.comment.hasFocus())
QTest.keyClick(evp.comment, Qt.Key_Escape)
ep.setFocus()
app.processEvents()
# REVIEW STATUS
sc = fd.status_checks
check('9.72a', "[72] REVIEW STATUS: five boxes, all checked, reviewed "
      "tri-state", list(sc) == ['unreviewed', 'reviewed', 'accepted',
                                'rejected', 'unsure']
      and all(c.isChecked() for c in sc.values())
      and sc['reviewed'].isTristate())
sc['reviewed'].click()
app.processEvents()
off = [sc[k].isChecked() for k in ('accepted', 'rejected', 'unsure')]
sc['reviewed'].click()
app.processEvents()
on = [sc[k].isChecked() for k in ('accepted', 'rejected', 'unsure')]
sc['rejected'].setChecked(False)
app.processEvents()
check('9.72b', "[72] reviewed sets its three children; unchecking rejected "
      "alone makes it partial", off == [False] * 3 and on == [True] * 3
      and sc['reviewed'].checkState() == Qt.PartiallyChecked)
check('9.74', "[74] caption", fd.status_caption.text() ==
      'Your decisions only. Applies to the Epochs tab.')
for k in ('reviewed', 'accepted', 'rejected', 'unsure'):
    sc[k].setChecked(False)
sc['unreviewed'].setChecked(True)
app.processEvents()
dbwrite.store_event_review(db9.conn, 'u-e', 'accept', 'JS')
db9.conn.commit()
ep._goto_epoch(0)
alpha = {u: next((it.brush.color().alpha() for it in ep.band_items(
    ep.raw_plot) if abs(it.getRegion()[0] - STARTS[UUIDS.index(u)]) < 1e-9),
    None) for u in ('u-a', 'u-b', 'u-c', 'u-e')}
check('9.73a', "[73] only unreviewed: TK's decided events at half fill, "
      "JS-only u-e full", alpha['u-a'] == 10 and alpha['u-c'] == 10
      and alpha['u-e'] == 30, repr(alpha))
ep.select_event('u-a')
QTest.keyClick(ep, Qt.Key_BraceRight)
app.processEvents()
check('9.73b', "[73] } skips TK-decided events (u-b, u-c): lands on u-out",
      ep._selected_uuid == 'u-out', repr(ep._selected_uuid))
check('9.79b', "[79] the status filter does not change n in the header",
      evp.header_lbl.text() == 'EVENT 4 OF 5 IN EPOCH',
      repr(evp.header_lbl.text()))
# [73] ] / [ ignore the filter: with only 'rejected' shown, ] from u-a still
# goes to the next event TK has not decided (u-out), which the filter hides
for k in ('unreviewed', 'accepted', 'unsure'):
    sc[k].setChecked(False)
sc['rejected'].setChecked(True)
app.processEvents()
ep.select_event('u-a')
QTest.keyClick(ep, Qt.Key_BracketRight)
app.processEvents()
after_r = ep._selected_uuid
QTest.keyClick(ep, Qt.Key_BracketLeft)
app.processEvents()
check('9.73c', "[73] ] / [ ignore REVIEW STATUS: ] lands on an unreviewed "
      "event the filter dims, [ comes back", after_r == 'u-out'
      and ep._selected_uuid != 'u-out', repr((after_r, ep._selected_uuid)))
for c in sc.values():
    c.setChecked(True)
app.processEvents()
ep.clear_selection()
check('9.79c', "[79] no selection: header EVENT", evp.header_lbl.text() ==
      'EVENT', repr(evp.header_lbl.text()))
# key hints, cheat sheet, status bar, legend
check('9.75a', "[75] top-bar hint on the Epochs tab",
      win.key_hint_lbl.text() == 'A accept · R reject · U unsure · ] [ '
                                 'unreviewed · N P outlier · ? keys',
      repr(win.key_hint_lbl.text()))
QTest.keyClick(ep, Qt.Key_Question)
app.processEvents()
sheet = win._cheat_sheet
txt = sheet.text() if sheet is not None else ''
check('9.76a', "[76] ? opens the sheet with keys and the spindle grid",
      sheet is not None and sheet.isVisible()
      and all(k in txt for k in ('A        accept', '1–8, 0', 'Enter',
                                 'C        type a comment', '] [', '} {',
                                 'N P', 'Shift+drag', '4  Filter ringing'))
      and 'REASONS FOR SPINDLES' in txt, repr(txt[:80]))
QTest.keyClick(sheet, Qt.Key_Question)
app.processEvents()
check('9.76b', "[76] ? again closes it", not sheet.isVisible())
win.open_cheat_sheet()
QTest.keyClick(win._cheat_sheet, Qt.Key_Escape)
app.processEvents()
check('9.76c', "[76] Esc closes it", not win._cheat_sheet.isVisible())
segs = [w for w in win.status_bar.findChildren(QtWidgets.QWidget)
        if w in (win.seg_reviewer, win.seg_position, win.seg_save)]
order = sorted((w.mapTo(win, QtCore.QPoint(0, 0)).x(), w) for w in segs)
check('9.77a', "[77] status bar: reviewer, position, save line in order",
      [w for _x, w in order] == [win.seg_reviewer, win.seg_position,
                                 win.seg_save]
      and win.seg_position.text().startswith('Cz · epoch ')
      and win.seg_save.text() == 'Decisions save to rev3.db as you make '
                                 'them.', repr(win.seg_save.text()))
win.set_reviewer_name('')
check('9.77b', "[77] no reviewer name: save line asks for one",
      win.seg_save.text() == 'Set a reviewer name to save decisions.')
win.set_reviewer_name('TK')
check('9.78', "[78] strip legend", ep.strip_legend.text() ==
      'grey bars = events per epoch · red = amplitude outliers · purple '
      'dashes = marked artefact · white line = current epoch')
win.tabs.setCurrentIndex(0)
app.processEvents()
check('9.75b', "[75] Channels tab hint", win.key_hint_lbl.text() ==
      'F re-detect queue · ? keys', repr(win.key_hint_lbl.text()))
win.open_cheat_sheet()
check('9.76d', "[76] ? from the Channels tab (no drill change) opens it",
      win._cheat_sheet.isVisible())
win._cheat_sheet.close()
win.tabs.setCurrentIndex(1)
win.activateWindow()
app.processEvents()
# physiology marks [90, 91]
ep.physio.set_open(True)
ep._goto_epoch(0)
app.processEvents()
reads = ep.physio.n_reads
texts_before = [sum(isinstance(i, rg.pg.TextItem)
                    for i in w.getPlotItem().items)
                for _k, _c, w in ep.physio.rows]
ep.select_event('u-c')
app.processEvents()
lines_ok, extra = True, False
for i, (_k, _c, w) in enumerate(ep.physio.rows):
    its = w.getPlotItem().items
    vl = [x for x in its if isinstance(x, rg.pg.InfiniteLine)]
    lines_ok &= (len(vl) == 2
                 and sorted(round(x.value(), 3) for x in vl) == [12.0, 12.8]
                 and all(x.pen.widthF() == 1 for x in vl))
    extra |= any(isinstance(x, (rg.pg.LinearRegionItem,
                                QtWidgets.QGraphicsRectItem)) for x in its)
    extra |= sum(isinstance(x, rg.pg.TextItem) for x in its) != \
        texts_before[i]
check('9.90a', "[90] each physiology row: exactly two 1 px lines at the "
      "event's start and end; no region, rect or text for the selection",
      lines_ok and not extra and len(ep.physio.rows) == 4)
ep.select_event('u-b')
app.processEvents()
check('9.91', "[91] changing the selection reads no physiology data",
      ep.physio.n_reads == reads + 0 and sorted(
          round(x.value(), 3) for x in ep.physio.selection_lines(0))
      == [7.0, 7.8], repr(ep.physio.n_reads - reads))
ep.physio.set_open(False)
ep.select_event('u-c')
app.processEvents()
ep.physio.set_open(True)
app.processEvents()
check('9.90c', "selecting with the strip closed, then opening it: two "
      "lines per row", all(len(ep.physio.selection_lines(i)) == 2
                           for i in range(len(ep.physio.rows))),
      repr([len(ep.physio.selection_lines(i))
            for i in range(len(ep.physio.rows))]))
ep.clear_selection()
app.processEvents()
check('9.90b', "[90] nothing selected: the lines are gone", all(
    not [x for x in w.getPlotItem().items
         if isinstance(x, rg.pg.InfiniteLine)]
    for _k, _c, w in ep.physio.rows))
# neighbours [92, 93]
win.detail_dock_w._coords = dict(coords, Cz=(0.0, 0.0))
ep.neighbours.set_open(True)
ep.select_event('u-a')
app.processEvents()
labs = ep.neighbours.plot.row_labels()
hdr = ep.neighbours.header.text()
check('9.92', "[92] with coordinates: target, then ranks 1…6, nearest "
      "first; no cm / mm; header (1 = nearest)",
      labs[0] == 'Cz · target'
      and [l.split(' · ')[1] for l in labs[1:]] == [str(i) for i in
                                                     range(1, len(labs))]
      and not any(u in ' '.join(labs) + hdr for u in (' cm', ' mm'))
      and '(1 = nearest)' in hdr, repr((labs, hdr)))
win.detail_dock_w._coords = None
ep.select_event('u-b')
app.processEvents()
labs = ep.neighbours.plot.row_labels()
check('9.93', "[93] region fallback: no rank numbers",
      labs[0] == 'Cz · target'
      and not any(l.split(' · ')[-1].isdigit() for l in labs[1:]),
      repr(labs))
# slow-wave grid and digits [84, 85, 88]
use('slow_wave')
ep = win.epochs_panel
check('9.84b', "[84] slow-wave grid texts", evp.grid_texts() ==
      ['1  Artefact', '2  Eye movement', '3  Not in raw', '4  Too short',
       '5  Arousal', '6  Single channel', '7  Not isolated',
       '8  Wrong morphology', '0  Other'], repr(evp.grid_texts()))
for uid, key_, want in (('sw-0', Qt.Key_4, 'too-short'),
                        ('sw-1', Qt.Key_7, 'not-isolated'),
                        ('sw-2', Qt.Key_8, 'wrong-morphology')):
    ep.select_event(uid)
    press9(Qt.Key_R)
    press9(key_)
    got = list(db9.conn.execute("SELECT decision, reason FROM event_reviews "
                                "WHERE uuid = ? AND reviewer = 'TK'", (uid,)))
    check('9.85c', f"[85] slow wave R,{want}", got == [('reject', want)],
          repr(got))
dbwrite.store_event_review(db9.conn, 'sw-0', 'reject', 'TK',
                           reason='filter-ringing')
db9.conn.commit()
ep.select_event('sw-1')
ep.select_event('sw-0')
app.processEvents()
check('9.88', "[88] a stored filter-ringing reject on a slow wave shows "
      "its label on the Current line",
      evp.current_lbl.text().startswith('Rejected by TK · ')
      and evp.current_lbl.text().endswith(' · Filter ringing'),
      repr(evp.current_lbl.text()))
win.close()

say("\n" + "=" * 78)
check('settings', "the real review-GUI preferences file was not "
      "touched", *gui_settings_guard.untouched())
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
