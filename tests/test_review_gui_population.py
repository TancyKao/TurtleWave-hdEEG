#!/usr/bin/env python3
"""Headless checks: population checks in the Channels (QC) tab (4.6.0).

Acceptance criteria 8-15 of ``_scratch/design/event-decision-spec.md``
(revision 2, section 13):

8.  Per-channel off-band / low-prominence / at-floor shares and the two
    median ratios match a hand count from the inserted arrays.
9.  60 % off-band among 5-10 % channels -> ``hard``; 4 % against a 1 %
    median (difference under 10 points) -> not flagged; 19 events -> ``—``.
10. The amplitude ``flag``, the HARD / SOFT counts and "Queue all HARD" are
    unchanged by the new columns.
11. Slow-wave run: no low-prominence column or combo item. Lacourse2018:
    ``amp/thr ×`` is ``—`` with the method's tooltip.
12. Pre-4.6 run: the five combo items disabled with the suffix, caption
    ``Event checks are not recorded for this run``.
13. Dock line for the flagged channel and for an unflagged one.
14. The off-band link drills, shows the chip, ``}`` visits only off-band
    events, Esc removes the chip.
15. Dashboard refresh on a 257-channel, ~372k-event fixture: no more than
    10 % slower with the population checks (they are read in a background
    thread and cached per run).

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_review_gui_population.py
"""
import os
import sys
import tempfile
import time

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
from PyQt5 import QtCore, QtGui, QtWidgets                   # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
from PyQt5.QtTest import QTest                                # noqa: E402

TMP = tempfile.mkdtemp(prefix='tw_population_')
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()     # before any frontend import

import frontend.eeg_review_gui as rg                          # noqa: E402
from frontend import event_review as er                       # noqa: E402
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
rng = np.random.default_rng(7)
RUN = 'run-fixture'


def build(path, channels, run_kw=None, row_kw=None, per_channel=None):
    """``channels``: {name: (n, off_band share)}; returns per-channel arrays."""
    con = fx.open_schema(path)
    fx.add_run(con, RUN, **(run_kw or {}))
    arrays = {}
    t0 = 0.0
    for ch, (n, off) in channels.items():
        kw = dict(row_kw or {})
        kw.update((per_channel or {}).get(ch, {}))
        rows, arr = fx.make_rows(ch, n, RUN, rng, off_band=off, t0=t0, **kw)
        fx.insert_rows(con, rows)
        arrays[ch] = arr
    con.commit()
    con.close()
    return arrays


def open_window(path, evt='spindle'):
    win = rg.EventReviewGUI()
    win.db = rg.EventDatabase(path)
    win.qc_widget.evt_combo.blockSignals(True)
    win.qc_widget.evt_combo.setCurrentText(evt)
    win.qc_widget.evt_combo.blockSignals(False)
    return win


def refresh(win):
    win.refresh_qc_dashboard()
    win.wait_population()
    app.processEvents()


def col_cell(win, ch, key, role=Qt.DisplayRole):
    m = win.qc_widget.model
    for r in range(m.rowCount()):
        if m.channel_at(r) == ch:
            return m.data(m.index(r, rg._QC_COL_INDEX[key]), role)
    return None


say("=" * 78)
say("Headless: population checks in the Channels tab")
say("=" * 78)

# ===================================================================== 8-10
say("\n== 8-10. Shares, flags, unchanged amplitude flag")
chans = {f"E{i}": (120, float(rng.uniform(0.05, 0.10))) for i in range(1, 26)}
chans['E7'] = (120, 0.60)
chans['E19'] = (19, 0.60)
P1 = os.path.join(TMP, 'main.db')
arrays = build(P1, chans, per_channel={'E3': {'no_peak': 0.1}})

win = open_window(P1)
# baseline (no population checks) for criterion 10
orig = win._apply_population
win._apply_population = lambda qc, *a: qc
win.refresh_qc_dashboard()
flag_before = win._qc_df.set_index('channel')['flag'].to_dict()
counts_before = win.qc_widget.counts_lbl.text().split(' · ', 1)[1]
hard_btn_before = win.qc_widget.btn_queue_hard.text()
win._apply_population = orig
refresh(win)
qc = win._qc_df.set_index('channel')


def hand(arr):
    ib, low, near = arr['in_band'], arr['low'], arr['near']
    meas = ib != -1
    return {'pct_off_band': 100.0 * np.sum(ib == 0) / np.sum(meas),
            'pct_low_prom': 100.0 * np.sum((low == 1) & meas) / np.sum(meas),
            'pct_dur_floor': 100.0 * np.mean(near == -1),
            'med_amp_ratio': float(np.median(arr['amp'])),
            'med_thresh_ratio': float(np.median(arr['thr']))}


worst = 0.0
for ch in ('E1', 'E3', 'E7', 'E12'):
    h = hand(arrays[ch])
    for col, v in h.items():
        got = float(qc.loc[ch, col])
        tol = 0.1 if col.startswith('pct') else 0.01
        worst = max(worst, abs(got - v) / tol)
        check('8', f"{ch} {col} matches the hand count", abs(got - v) <= tol,
              f"{got:.3f} vs {v:.3f}")
check('8b', "E3's no-peak events are left out of the off-band denominator",
      int(qc.loc['E3', 'n_no_peak']) == 12 and int(qc.loc['E3', 'n_freq']) == 108,
      repr((qc.loc['E3', 'n_no_peak'], qc.loc['E3', 'n_freq'])))
check('9a', "E7 at 60 % off-band among 5-10 % channels -> checks_flag hard",
      qc.loc['E7', 'checks_flag'] == 'hard', repr(qc.loc['E7', 'checks_flag']))
check('9b', "E19 (19 events) shows — and is not flagged",
      col_cell(win, 'E19', 'pct_off_band') == '—'
      and qc.loc['E19', 'checks_flag'] == ''
      and col_cell(win, 'E19', 'pct_off_band', Qt.ToolTipRole)
      == 'Too few events on this channel to judge (n = 19).',
      repr(col_cell(win, 'E19', 'pct_off_band', Qt.ToolTipRole)))
check('9c', "only E7 is flagged on this fixture",
      sorted(qc.index[qc['checks_flag'] != '']) == ['E7'],
      repr(sorted(qc.index[qc['checks_flag'] != ''])))
synthetic = pd.DataFrame({
    'channel': [f"C{i}" for i in range(21)],
    'n': 200, 'n_no_peak': 0, 'n_freq': 200, 'n_prom': 200, 'n_bound': 200,
    'n_amp': 200, 'n_thresh': 200,
    'pct_off_band': list(np.linspace(0.9, 1.1, 20)) + [4.0],
    'pct_low_prom': 30.0, 'pct_dur_floor': 10.0, 'med_amp_ratio': 2.5,
    'med_thresh_ratio': 1.6})
flagged, med = er.population_flags(synthetic)
z4 = float(flagged.iloc[-1]['z_pct_off_band'])
check('9d', "4 % against a 1 % median: z large, under 10 points, not flagged",
      z4 > 3.5 and flagged.iloc[-1]['checks_flag'] == '',
      f"z={z4:.1f}, flag={flagged.iloc[-1]['checks_flag']!r}")
flag_after = win._qc_df.set_index('channel')['flag'].to_dict()
check('10', "[10] amplitude flag, amp-flag / dead counts and Queue all "
      "HARD unchanged by the check columns", flag_after == flag_before
      and win.qc_widget.counts_lbl.text().split(' · ', 1)[1] == counts_before
      and win.qc_widget.btn_queue_hard.text() == hard_btn_before,
      repr((counts_before, hard_btn_before)))
hdr = win.qc_widget.model.headerData(rg._QC_COL_INDEX['checks_flag'],
                                     Qt.Horizontal, Qt.ToolTipRole)
check('10d', "checks header tooltip is the spec text",
      hdr == er.header_tooltips()['checks_flag'], repr(hdr))
check('10e', "check cell text: share as '62 %', ratio as '2.5×'",
      col_cell(win, 'E7', 'pct_off_band') == f"{qc.loc['E7', 'pct_off_band']:.0f} %"
      and col_cell(win, 'E7', 'med_amp_ratio').endswith('×'),
      repr((col_cell(win, 'E7', 'pct_off_band'),
            col_cell(win, 'E7', 'med_amp_ratio'))))

# ===================================================================== 13
say("\n== 13. No interpretive dock text (revision 3)")
win.on_qc_channel_selected('E7')
dock = win.detail_dock_w
labels = [l.text() for l in dock.findChildren(QtWidgets.QLabel)]
bad = [t for t in labels if any(w in t for w in (
    'event checks in line with', 'Most events failing?', 'Looks like',
    'same pattern'))]
check('13', "[13] no dock label carries the revision 2 sentence or hint",
      not bad, repr(bad[:2]))
p7 = f"{qc.loc['E7', 'pct_off_band']:.0f}"
check('13b', "facts line for E7", dock.facts_line.text().endswith(
    f"amp ✓ ok · checks × hard · off-band {p7} %")
      and dock.facts_line.text().startswith('E7 · '),
      repr(dock.facts_line.text()))
items = [dock.topo_combo.itemText(i) for i in range(dock.topo_combo.count())]
check('13d', "[63-64] Topo items in order (spindle, ratio method)", items ==
      ['Event density', 'Mean amp (µV)', 'Max p2p (µV)', 'Off-band share',
       'Low-prominence share (context)', 'At-floor share',
       'Amp vs background (median)', 'Amp vs threshold (median)'],
      repr(items))

# ===================================================================== 14
say("\n== 14. Open in Epochs on an off-band channel")
win.show()
win.activateWindow()
app.processEvents()
win.qc_widget.select_channel('E7')
win.qc_widget.btn_open.click()
app.processEvents()
ep = win.epochs_panel
chip = ep.chip_text()
n_off = int(np.sum(arrays['E7']['in_band'] == 0))
check('14a', "[14] Open in Epochs drills E7 and shows the chip",
      ep._channel == 'E7' and win.tabs.currentIndex() == 1
      and chip == f"Showing off-band spindles only ({n_off} of 120) ✕",
      repr(chip))
off_uuids = set(ep._df.loc[pd.to_numeric(ep._df['in_band']) == 0, 'uuid'])
seen = []
ep._goto_epoch(0)
for _ in range(n_off + 3):
    QTest.keyClick(ep, Qt.Key_BraceRight)
    app.processEvents()
    if ep._selected_uuid is not None and ep._selected_uuid not in seen:
        seen.append(ep._selected_uuid)
check('14b', "} visits only off-band events, all of them",
      set(seen) == off_uuids, f"{len(seen)} visited, {len(off_uuids)} off-band")
other = next(u for u in ep._df['uuid'] if u not in off_uuids)
band_other = None
ep.select_event(other)
for it in ep.band_items(ep.raw_plot):
    s0 = float(ep._df.loc[ep._df['uuid'] == other, 'start_time'].iloc[0])
    if abs(it.getRegion()[0] - s0) < 1e-9:
        band_other = it
check('14c', "events passing the check are drawn at half fill, still "
      "clickable", band_other is not None
      and band_other.brush.color().alpha() in (15, 23),
      repr(None if band_other is None else band_other.brush.color().alpha()))
QTest.keyClick(ep, Qt.Key_Escape)
app.processEvents()
check('14d', "Esc (nothing armed, no strip range) removes the chip",
      ep.chip_text() == '' and ep._check_filter is None, repr(ep.chip_text()))
win.close()

# ===================================================================== 52-71
say("\n== 52-71. Channels tab layout, flagged list (revision 3)")


def build_rich(path, spec, stages_split=None):
    """``spec``: {ch: dict(n=…, **make_rows kwargs)}; ``stages_split``:
    {ch: [(stage, n, kwargs), …]} inserted per stage."""
    con = fx.open_schema(path)
    fx.add_run(con, RUN)
    arr = {}
    for ch, kw in spec.items():
        kw = dict(kw)
        n = kw.pop('n', 120)
        rows, a = fx.make_rows(ch, n, RUN, rng, **kw)
        fx.insert_rows(con, rows)
        arr[ch] = a
    for ch, parts in (stages_split or {}).items():
        t0 = 0.0
        for st, n, kw in parts:
            rows, _a = fx.make_rows(ch, n, RUN, rng, stages=(st,), t0=t0,
                                    **kw)
            fx.insert_rows(con, rows)
            t0 += 7.0 * n + 10
    con.commit()
    con.close()
    return arr


normal = {f"E{i}": dict(off_band=float(ob), at_floor=float(fl),
                        low_prom=0.2)
          for i, ob, fl in zip(range(1, 31), np.linspace(0.05, 0.10, 30),
                               np.linspace(0.04, 0.16, 30))}
rich = dict(normal)
rich['E75'] = dict(n=118, off_band=0.34, at_floor=0.10, low_prom=0.2,
                   off_peaks=[8.5] * 24 + [7.5] * 16)
rich['E70'] = dict(off_band=0.60, at_floor=0.22, low_prom=0.2)
rich['E62'] = dict(n=20, off_band=0.45, at_floor=0.10, low_prom=0.2)
rich['E40'] = dict(off_band=0.07, at_floor=0.10, low_prom=0.9)
rich['E41'] = dict(off_band=0.07, at_floor=0.12, at_ceiling=0.04,
                   low_prom=0.2)
rich['E42'] = dict(off_band=0.07, at_floor=0.0, at_ceiling=0.40,
                   low_prom=0.2)
rich['E90'] = dict(n=5, off_band=0.07, at_floor=0.10, low_prom=0.2)
split = {'E50': [('NREM2', 60, dict(off_band=0.5, at_floor=0.1,
                                    low_prom=0.2)),
                 ('NREM3', 60, dict(off_band=0.0, at_floor=0.1,
                                    low_prom=0.2))]}
P7 = os.path.join(TMP, 'rich.db')
rarr = build_rich(P7, rich, split)
win = open_window(P7)
win.show()
win.activateWindow()
refresh(win)
qcw = win.qc_widget
qc = win._qc_df.set_index('channel')
btns = [b.text() for b in qcw.stage_group.buttons()]
check('52a', "[52] Stage: NREM2, NREM3, NREM2 + NREM3; combined checked",
      btns == ['NREM2', 'NREM3', 'NREM2 + NREM3']
      and qcw.stage_buttons[qcw.COMBINED].isChecked()
      and sum(b.isChecked() for b in qcw.stage_group.buttons()) == 1,
      repr(btns))
check('52c', "Stage tooltip", qcw.stage_box.toolTip() ==
      'Stages used for the check columns, the checks flag and the '
      'flagged-channel list. Density and amplitude columns follow the '
      'Filters dock.')
amp_before = win._qc_df.set_index('channel')[['mean_amp', 'n', 'flag']]
vals = {}
for key in ('NREM2', 'NREM3', qcw.COMBINED):
    qcw.stage_buttons[key].click()
    app.processEvents()
    vals[key] = col_cell(win, 'E50', 'pct_off_band')
    same_amp = win._qc_df.set_index('channel')[
        ['mean_amp', 'n', 'flag']].equals(amp_before)
    check('53', f"[53] E50 off-band with {key}: amplitude columns unchanged",
          same_amp, repr(vals[key]))
check('53b', "[53] E50: 50 % with NREM2, 0 % with NREM3, 25 % pooled",
      vals == {'NREM2': '50 %', 'NREM3': '0 %', qcw.COMBINED: '25 %'},
      repr(vals))
check('53c', "the stage choice persists in QSettings per event type",
      rg._review_settings().value('review/check_stage/spindle')
      == qcw.COMBINED)
qc = win._qc_df.set_index('channel')
counts = {it: int(er.show_mask(win.qc_widget.model.df, it).sum())
          for it in er.SHOW_ITEMS}
texts = [qcw.show_combo.itemText(i) for i in range(qcw.show_combo.count())]
check('54a', "[54] Show items with live counts", texts ==
      [f"{it} ({counts[it]})" for it in er.SHOW_ITEMS], repr(texts))
qcw.show_combo.setCurrentIndex(1)
flagged = set(qc.index[(qc['checks_flag'].isin(['hard', 'soft']))
                       | (qc['flag'].isin(['hard', 'soft']))])
check('54b', "[54] Flagged keeps exactly checks- or amp-flagged rows",
      set(qcw.visible_channels()) == flagged, repr(sorted(flagged)))
qcw.show_combo.setCurrentIndex(2)
check('54c', "[54] no dropped channels: 'No channels match \"Dropped\".'",
      qcw.proxy.rowCount() == 0 and qcw.empty_lbl.isVisibleTo(qcw)
      and qcw.empty_lbl.text() == 'No channels match "Dropped".')
qcw.show_combo.setCurrentIndex(0)
# Sort
df = win.qc_widget.model.df


def expect(key, desc):
    if key is None:
        rank = df['checks_flag'].map({'hard': 2, 'soft': 1}).fillna(0)
        d = df.assign(_k=rank * 1000 + df['checks_z'].fillna(0))
        d = d.sort_values('_k', ascending=False, kind='mergesort')
        return d['channel'].iloc[0], d['channel'].iloc[-1]
    v = df[key]
    if key in ('channel', 'region'):
        d = df.assign(_k=v.astype(str)).sort_values('_k', ascending=not desc)
        return d['channel'].iloc[0], d['channel'].iloc[-1]
    d = df[v.notna()].sort_values(key, ascending=not desc)
    last = df[v.isna()]['channel'].iloc[-1] if v.isna().any() else \
        d['channel'].iloc[-1]
    return d['channel'].iloc[0], last


for label, key, desc in er.SORT_ITEMS:
    qcw.sort_combo.setCurrentText(label)
    app.processEvents()
    vis = qcw.visible_channels()
    f, l = expect(key, desc)
    di = df.set_index('channel')
    kk = key or 'checks_z'

    def same(a, b):
        va, vb = di.loc[a, kk], di.loc[b, kk]
        if key is None:
            return (di.loc[a, 'checks_flag'] == di.loc[b, 'checks_flag']
                    and (va == vb or (va != va and vb != vb)))
        return va == vb or (va != va and vb != vb)
    ok_first = vis[0] == f or same(vis[0], f)       # ties may come in any order
    ok_last = vis[-1] == l or same(vis[-1], l)
    if key is not None and key not in ('channel', 'region') and \
            df[key].isna().any():
        ok_last = df.set_index('channel').loc[vis[-1], key] != \
            df.set_index('channel').loc[vis[-1], key]       # missing last
    check('55', f"[55] Sort '{label}': first {f}, last as expected",
          ok_first and ok_last, repr((vis[0], vis[-1], f, l)))
hdr = qcw.table.horizontalHeader()
qcw.table.sortByColumn(rg._QC_COL_INDEX['pct_off_band'], Qt.DescendingOrder)
app.processEvents()
c1 = qcw.sort_combo.currentText()
qcw.table.sortByColumn(rg._QC_COL_INDEX['n'], Qt.AscendingOrder)
app.processEvents()
c2 = qcw.sort_combo.currentText()
check('55b', "[55] header Off-band descending selects 'Off-band share ↓'; "
      "Events selects 'Column header'",
      c1 == 'Off-band share ↓' and c2 == 'Column header', repr((c1, c2)))
qcw.sort_combo.setCurrentText('Checks (hard first)')
n_chk = int(qc['checks_flag'].isin(['hard', 'soft']).sum())
n_amp = int(qc['flag'].isin(['hard', 'soft']).sum())
n_dead = int((qc['flag'] == 'dead').sum())
line0 = qcw.counts_lbl.text()
check('56a', "[56] header count line", line0 ==
      f"{n_chk} checks flagged · {n_amp} amp flagged · {n_dead} dead",
      repr(line0))
check('57', "[57] Checks cells: hard off-band, unflagged, dead",
      col_cell(win, 'E70', 'checks_flag') ==
      f"× HARD · off-band {qc.loc['E70', 'pct_off_band']:.0f} %"
      and col_cell(win, 'E5', 'checks_flag') == '—'
      and col_cell(win, 'E90', 'checks_flag') == '— dead channel',
      repr((col_cell(win, 'E70', 'checks_flag'),
            col_cell(win, 'E90', 'checks_flag'))))
check('58a', "[58] low prominence 90 % alone: no checks flag, no reasons, "
      "untinted cell", qc.loc['E40', 'checks_flag'] == ''
      and qc.loc['E40', 'checks_reasons'] == ''
      and not isinstance(col_cell(win, 'E40', 'pct_low_prom',
                                  Qt.BackgroundRole), QtGui.QColor),
      repr(qc.loc['E40', 'checks_reasons']))
ftip = col_cell(win, 'E41', 'pct_dur_floor', Qt.ToolTipRole)
check('59a', "[59] At floor tooltip with floor and ceiling shares", ftip ==
      f"At the floor (0.5 s): 12 % · at the ceiling (3 s): 4 % · "
      f"{int(qc.loc['E41', 'n_bound'])} events with a duration bound.",
      repr(ftip))
check('59b', "[59] 0 % floor, 40 % ceiling: not flagged",
      qc.loc['E42', 'checks_flag'] == '')
# bottom bar
qcw.table.setCurrentIndex(QtCore.QModelIndex())
qcw.table.clearSelection()
qcw._update_action_state()
check('60a', "[60] no row: label — and every bar button disabled",
      qcw._sel_lbl.text() == '—' and not any(
          b.isEnabled() for b in (qcw.btn_open, qcw.btn_drop, qcw.btn_mark,
                                  qcw.btn_redetect)))
qcw.select_channel('E70')
lay_texts = []
bl = qcw.btn_open.parentWidget().layout()
for i in range(bl.count()):
    item = bl.itemAt(i)
    while item is not None and item.layout() is not None and False:
        pass
row = [qcw.btn_open, qcw.btn_drop, qcw.btn_mark, qcw.btn_redetect]
xs = [b.mapTo(qcw, QtCore.QPoint(0, 0)).x() for b in row]
check('60b', "[60] selected: Open in Epochs, Drop channel, Mark channel "
      "artefact, Add to re-detect queue (left to right); Queue all HARD and "
      "Build still there", [b.text() for b in row] ==
      ['Open in Epochs', 'Drop channel', 'Mark channel artefact',
       'Add to re-detect queue'] and xs == sorted(xs)
      and qcw.btn_queue_hard.text().startswith('Queue all HARD (')
      and qcw.btn_build.text().startswith('Build re-detect request…'),
      repr(xs))
qcw.select_channel('E62')
qcw.btn_drop.click()
app.processEvents()
win.epochs_panel.dropChannelRequested.emit('E29')
app.processEvents()
ver = win.db.get_channel_verdicts()
check('61', "[61] bottom-bar Drop sets the same verdict as the Epochs "
      "tab's Drop channel; Status reads × dropped",
      ver.get(('E62', 'spindle')) == ver.get(('E29', 'spindle')) == 'drop'
      and col_cell(win, 'E62', 'verdict') == '× dropped',
      repr((ver.get(('E62', 'spindle')), col_cell(win, 'E62', 'verdict'))))
qcw.select_channel('E62')
check('61b', "Drop on an already-dropped channel: disabled, with tooltip",
      not qcw.btn_drop.isEnabled() and qcw.btn_drop.toolTip() ==
      'Already dropped. Undo from the Selection tray.')
qcw.select_channel('E75')
qcw.btn_redetect.click()
app.processEvents()
check('rd', "a queued channel's Status reads 'kept · ↻ re-detect' with the "
      "tray tooltip; a dropped queued one '× dropped · ↻ re-detect'",
      col_cell(win, 'E75', 'verdict') == 'kept · ↻ re-detect'
      and col_cell(win, 'E75', 'verdict', Qt.ToolTipRole) ==
      'Queued for re-detection. Remove it from the Selection tray.'
      and '↻ re-detect' in col_cell(win, 'E75', 'verdict',
                                    rg._STATUS_HTML_ROLE),
      repr(col_cell(win, 'E75', 'verdict')))
qcw.select_channel('E62')
qcw.btn_redetect.click()
app.processEvents()
check('rd2', "dropped and queued", col_cell(win, 'E62', 'verdict') ==
      '× dropped · ↻ re-detect', repr(col_cell(win, 'E62', 'verdict')))
qcw.select_channel('E62')
qcw.btn_redetect.click()
qcw.select_channel('E75')
qcw.btn_redetect.click()
app.processEvents()
line1 = qcw.counts_lbl.text()
qcw.show_combo.setCurrentIndex(3)
line2 = qcw.counts_lbl.text()
qcw.show_combo.setCurrentIndex(0)
check('56b', "[56] header gains ' · 2 dropped' and ignores Show",
      line1.endswith(' · 2 dropped') and line2 == line1, repr(line1))
foot = qcw.footer.text()
check('62a', "[62] footer states the relative rule from live limits",
      all(t in foot for t in (
          'Checks compare each channel with the rest of the montage.',
          'above 3.5', 'above 2.0', '10 percentage points', '0.3×',
          'at least 20 events',
          'Low prominence is shown for context and never flags.'))
      and '≥ 25 %' not in foot and '≥ 15 %' not in foot, repr(foot))
win._qc_thresholds.update(hard_z=4.0, soft_z=2.5)
refresh(win)
foot = qcw.footer.text()
check('62b', "[62] after hard 4.0 / soft 2.5", 'above 4.0' in foot
      and 'above 2.5' in foot, repr(foot))
win._qc_thresholds.update(hard_z=3.5, soft_z=2.0)
refresh(win)
qc = win._qc_df.set_index('channel')
fl = win.detail_dock_w.flagged
n_h = int((qc['checks_flag'] == 'hard').sum())
n_s = int((qc['checks_flag'] == 'soft').sum())
check('65', "[65] list header and counts match the table",
      fl.header.text() == 'CHECKS — FLAGGED CHANNELS'
      and fl.counts.text() == f"{n_h} HARD · {n_s} SOFT",
      repr(fl.counts.text()))
r75 = fl.row('E75')
n75 = int(qc.loc['E75', 'n_off_band'])
m75 = 100 * 24 / n75
check('66a', "[66] E75 row: off-band share and its mode bin, facts only",
      r75 is not None and r75['text'] ==
      f"E75 · {qc.loc['E75', 'pct_off_band']:.0f} % of spindles off-band · "
      f"{m75:.0f} % of those peak at 8–9 Hz", repr(r75 and r75['text']))
r62 = fl.row('E62')
check('66b', "[66] with fewer than 10 off-band events the mode clause is "
      "absent", r62 is not None and 'of those peak' not in r62['text']
      and int(qc.loc['E62', 'n_off_band']) < 10, repr(r62 and r62['text']))
r70 = fl.row('E70')
check('67', "[67] E70 (off-band hard, at floor soft): one row, off-band "
      "fact first; hard before soft in the list",
      r70 is not None and r70['text'].split(' · ')[1].endswith('off-band')
      and 'at the duration floor (0.5 s)' in r70['text']
      and [r['badge'] for r in fl.rows] == sorted(
          [r['badge'] for r in fl.rows], key=lambda b: b != '× HARD'),
      repr(r70 and r70['text']))
check('58b', "[58] E40 (low prominence only) has no row",
      fl.row('E40') is None)
alltext = ' '.join([l.text() for l in win.detail_dock_w.findChildren(
    QtWidgets.QLabel)] + fl.texts())
check('68', "[68] no interpretation words in the dock or list",
      not any(w in alltext for w in ('Looks like', 'alpha',
                                     'check neighbours', 'Neighbour of',
                                     'same pattern')))
fl.click_row('E75')
app.processEvents()
vis1 = [r['channel'] for r in fl.rows if r['buttons'].isVisibleTo(fl)]
sel1 = qcw._current_channel()
fl.click_row('E70')
app.processEvents()
vis2 = [r['channel'] for r in fl.rows if r['buttons'].isVisibleTo(fl)]
check('69', "[69] a row click selects the channel and shows its two "
      "buttons only; another click moves them",
      vis1 == ['E75'] and sel1 == 'E75' and vis2 == ['E70']
      and fl.row('E70')['open'].text() == 'Open in Epochs'
      and fl.row('E70')['drop'].text() == 'Drop channel', repr((vis1, vis2)))
check('70', "[70] a dropped flagged channel: ' · dropped', no Drop button",
      fl.row('E62') is not None and fl.row('E62')['text'].startswith(
          'E62 · dropped · ') and fl.row('E62')['drop'] is None,
      repr(fl.row('E62') and fl.row('E62')['text']))
# topography rings, labels and caption
grid = {ch: ((i % 8) / 8.0 - 0.45, (i // 8) / 6.0 - 0.4)
        for i, ch in enumerate(sorted(qc.index))}
win.detail_dock_w.set_coords(grid)
dk = win.detail_dock_w
dk.topo_combo.setCurrentIndex(dk.topo_combo.findData('pct_off_band'))
app.processEvents()
ringed = {c for c in qc.index[qc['checks_flag'].isin(['hard', 'soft'])]
          if c != 'E62'}                    # E62 is dropped: no ring
pens = {r.channel: r.opts['pen'].style() for r in dk.ring_items}
check('63a', "[63] one ring per checks-flagged channel (dropped excluded); "
      "solid hard, dashed soft; one label each (≤ 12)",
      set(pens) == ringed and all(
          (pens[c] == Qt.SolidLine) == (qc.loc[c, 'checks_flag'] == 'hard')
          for c in pens) and len(dk.ring_labels) == len(ringed),
      repr(sorted(pens)))
cap = dk.topo_caption.text()
check('64', "[64] Off-band caption", cap ==
      'Off-band share · spindles · NREM2 + NREM3 · share of events whose '
      '1/f-corrected peak lies outside 9–12 Hz', repr(cap))
dk.topo_combo.setCurrentIndex(0)
app.processEvents()
check('63b', "[63] no rings on Event density", dk.ring_items == [])
win.close()
# 14 flagged channels: 12 labels, caption suffix
spec14 = {f"E{i}": dict(off_band=float(ob), at_floor=0.1, low_prom=0.2)
          for i, ob in zip(range(1, 41), np.linspace(0.05, 0.10, 40))}
for i in range(1, 15):
    spec14[f"E{i}"] = dict(off_band=0.6, at_floor=0.1, low_prom=0.2)
P8 = os.path.join(TMP, 'fourteen.db')
build_rich(P8, spec14)
win = open_window(P8)
refresh(win)
dk = win.detail_dock_w
chs = sorted(win._qc_df['channel'])
dk.set_coords({ch: ((i % 8) / 8.0 - 0.45, (i // 8) / 6.0 - 0.4)
               for i, ch in enumerate(chs)})
dk.topo_combo.setCurrentIndex(dk.topo_combo.findData('pct_off_band'))
app.processEvents()
check('63c', "[63] 14 flagged: 14 rings, 12 labels, caption suffix",
      len(dk.ring_items) == 14 and len(dk.ring_labels) == 12
      and dk.topo_caption.text().endswith(
          ' · 2 more flagged channels ringed, not labelled'),
      repr((len(dk.ring_items), len(dk.ring_labels))))
win.close()
# one-stage run; no channel flagged
P9 = os.path.join(TMP, 'onestage.db')
con = fx.open_schema(P9)
fx.add_run(con, RUN, stages=('NREM2',))
for i in range(1, 8):
    rows, _a = fx.make_rows(f"E{i}", 60, RUN, rng, off_band=0.05,
                            stages=('NREM2',))
    fx.insert_rows(con, rows)
con.commit()
con.close()
win = open_window(P9)
refresh(win)
b = win.qc_widget.stage_group.buttons()
check('52b', "[52] one-stage run: one checked, disabled button",
      len(b) == 1 and b[0].text() == 'NREM2' and b[0].isChecked()
      and not b[0].isEnabled(), repr([x.text() for x in b]))
check('71', "[71] no flagged channel: empty-state text",
      win.detail_dock_w.flagged.empty.text() ==
      'No channel is flagged by the checks for NREM2.',
      repr(win.detail_dock_w.flagged.empty.text()))
win.close()

# ===================================================================== 11-12
say("\n== 11-12. Event types, methods, pre-4.6 runs")
P2 = os.path.join(TMP, 'sw.db')
build(P2, {f"E{i}": (60, 0.05) for i in range(1, 8)},
      run_kw={'event_type': 'slow_wave', 'method': 'Massimini2004',
              'band': (0.5, 4.0)},
      row_kw={'event_type': 'slow_wave', 'method': 'Massimini2004',
              'band': (0.5, 4.0)})
win = open_window(P2, 'slow_wave')
refresh(win)
items = [win.detail_dock_w.topo_combo.itemText(i)
         for i in range(win.detail_dock_w.topo_combo.count())]
check('11a', "slow waves: low prom. % column hidden, combo item absent",
      win.qc_widget.table.isColumnHidden(rg._QC_COL_INDEX['pct_low_prom'])
      and not any('Low-prominence' in t for t in items), repr(items))
win.close()
P3 = os.path.join(TMP, 'lac.db')
build(P3, {f"E{i}": (60, 0.05) for i in range(1, 8)},
      run_kw={'method': 'Lacourse2018'}, row_kw={'method': 'Lacourse2018'})
win = open_window(P3)
refresh(win)
check('11b', "Lacourse2018: amp/thr × is — with the method's tooltip",
      col_cell(win, 'E1', 'med_thresh_ratio') == '—'
      and col_cell(win, 'E1', 'med_thresh_ratio', Qt.ToolTipRole)
      == 'No ratio for Lacourse2018: the stored peak and the detection '
         'threshold are not the same signal.',
      repr(col_cell(win, 'E1', 'med_thresh_ratio', Qt.ToolTipRole)))
check('11c', "spindles keep the low prom. % column",
      not win.qc_widget.table.isColumnHidden(rg._QC_COL_INDEX['pct_low_prom']))
# cache: another connection's commit (a re-detection) invalidates it, the
# GUI's own review write does not
k0 = win._population_key('spindle', None, None)
dbx = __import__('sqlite3').connect(P3)
dbx.execute("UPDATE events SET amp_ratio = amp_ratio + 0.01 WHERE rowid = 1")
dbx.commit()
dbx.close()
k1 = win._population_key('spindle', None, None)
win.db.add_review(win.db.conn.execute(
    "SELECT uuid FROM events LIMIT 1").fetchone()[0], 'accept', reviewer='TK')
k2 = win._population_key('spindle', None, None)
check('11d', "population cache: invalidated by another connection's "
      "commit, kept across the GUI's own review write",
      k0 != k1 and k1 == k2, repr((k0[-1], k1[-1], k2[-1])))
win.close()
# reopen: a fresh connection restarts PRAGMA data_version at 1, so the cache
# must not survive a database being replaced (File > Open)
P6 = os.path.join(TMP, 'reopen.db')
build(P6, {f"E{i}": (60, 0.05) for i in range(1, 8)})
win = open_window(P6)
refresh(win)
before = float(win._qc_df.set_index('channel').loc['E1', 'pct_off_band'])
dv_before = win._population_key('spindle', None, None)[-1]
other = __import__('sqlite3').connect(P6)
other.execute("UPDATE events SET in_band = 0 WHERE channel = 'E1'")
other.commit()
other.close()
win.db.conn.close()
win.db = rg.EventDatabase(P6)           # what File > Open does
check('11e', "replacing the database clears the population and figure "
      "caches", win._pop_cache == {} and win._fig_cache == {}
      and win._pop_view is None)
key_new = win._population_key('spindle', None, None)
refresh(win)
after = float(win._qc_df.set_index('channel').loc['E1', 'pct_off_band'])
check('11f', "after the other process's change and a reopen, the checks "
      "are read again (E1 off-band 5 % -> 100 %)",
      round(before) == 5 and after == 100.0,
      f"{before:.1f} -> {after:.1f}; data_version {dv_before} -> "
      f"{key_new[-1]}")
win.close()
P5b = os.path.join(TMP, 'off.db')
build(P5b, {f"E{i}": (60, 0.05) for i in range(1, 8)},
      run_kw={'figures_off': True}, row_kw={'figures': False})
win = open_window(P5b)
refresh(win)
check('12d', "4.6 run with figures switched off: caption says so (not the "
      "4.5 wording)", win.detail_dock_w.checks_caption.text()
      == er.FIGURES_OFF_RUN, repr(win.detail_dock_w.checks_caption.text()))
win.close()
P4 = os.path.join(TMP, 'old.db')
build(P4, {f"E{i}": (60, 0.05) for i in range(1, 8)},
      run_kw={'figures': False, 'version': '4.5.0'},
      row_kw={'figures': False})
win = open_window(P4)
refresh(win)
cb = win.detail_dock_w.topo_combo
model = cb.model()
check_items = [(cb.itemText(i), model.item(i).isEnabled())
               for i in range(3, cb.count())]
check('12a', "pre-4.6: all five combo items disabled with the suffix",
      len(check_items) == 5 and all(
          t.endswith(' — not recorded for this run') and not en
          for t, en in check_items), repr(check_items))
check('12b', "caption says the checks are not recorded",
      'Event checks are not recorded for this run'
      in win.detail_dock_w.checks_caption.text(),
      repr(win.detail_dock_w.checks_caption.text()))
check('12c', "cells read —, the flagged list says not recorded and the "
      "header count line starts 'checks not recorded'",
      col_cell(win, 'E1', 'pct_off_band') == '—'
      and win.detail_dock_w.flagged.empty.text() ==
      'Checks not recorded for this run (detected with 4.5 or earlier).'
      and win.qc_widget.counts_lbl.text().startswith('checks not recorded · '),
      repr((win.detail_dock_w.flagged.empty.text(),
            win.qc_widget.counts_lbl.text())))
win.close()

# ===================================================================== 15
say("\n== 15. Refresh time on 257 channels × 1450 events")
P5 = os.path.join(TMP, 'perf.db')
t = time.perf_counter()
build(P5, {f"E{i}": (1450, 0.07) for i in range(1, 258)})
say(f"  fixture built in {time.perf_counter() - t:.1f} s")
win = open_window(P5)


orig = win._apply_population
off = lambda qc, *a: qc                     # noqa: E731
win._apply_population = off
win.refresh_qc_dashboard()                 # warm the SQLite page cache
t = time.perf_counter()
win._apply_population = orig
win.refresh_qc_dashboard()                 # starts the background read
cold = time.perf_counter() - t
t_bg = time.perf_counter()
win.wait_population(120)
bg = time.perf_counter() - t_bg
# Interleave with / without, best of two each, and compare per pair: other
# processes on the machine move absolute times by tens of percent within one
# run, a pair taken back to back shares the same load.
base, warm, ratios = [], [], []
for _ in range(5):
    pair = []
    for lst, f in ((base, off), (warm, orig)):
        win._apply_population = f
        best = None
        for _rep in range(2):
            t = time.perf_counter()
            win.refresh_qc_dashboard()
            d = time.perf_counter() - t
            best = d if best is None else min(best, d)
        lst.append(best)
        pair.append(best)
    ratios.append(pair[1] / pair[0])
b, w = float(np.median(base)), float(np.median(warm))
r = float(np.median(ratios))
key = win._population_key('spindle', None, None)
qc0 = rg.compute_channel_qc(win._qc_events_df)
t = time.perf_counter()
for _ in range(5):
    win._merge_population(qc0, 'spindle', key, win._pop_cache[key])
merge = (time.perf_counter() - t) / 5
n_ev = int(win._qc_events_df.shape[0])
say(f"  {n_ev} events · refresh without checks median {b * 1000:.0f} ms "
    f"{[round(x * 1000) for x in base]}")
say(f"  refresh with cached checks median {w * 1000:.0f} ms "
    f"{[round(x * 1000) for x in warm]}")
say(f"  per-pair ratio with/without {[round(x, 3) for x in ratios]} -> "
    f"median {100 * (r - 1):+.1f} %")
say(f"  first refresh with checks {cold * 1000:.0f} ms (one sample, not "
    f"asserted: it only starts the background read); the read then took "
    f"{bg * 1000:.0f} ms off the GUI thread")
say(f"  merge step added to each refresh: {merge * 1000:.1f} ms "
    f"({100 * merge / b:.1f} % of a refresh)")
check('15a', "refresh with the checks cached is within 10 % of without "
      "(median of back-to-back pairs)", r <= 1.10, f"{100 * (r - 1):+.1f} %")
check('15d', "the work the checks add on the GUI thread is under 10 % of a "
      "refresh", merge <= 0.10 * b, f"{merge * 1000:.1f} ms")
check('15c', "the background read filled the columns",
      win._qc_df['pct_off_band'].notna().sum() == 257)
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
