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
import re
import sqlite3
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
check('54c', "[54] no excluded channels: 'No channels match \"Excluded\".'",
      qcw.proxy.rowCount() == 0 and qcw.empty_lbl.isVisibleTo(qcw)
      and qcw.empty_lbl.text() == 'No channels match "Excluded".')
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
# bottom bar (R4.4, R4.5)
qcw.table.setCurrentIndex(QtCore.QModelIndex())
qcw.table.clearSelection()
qcw._update_action_state()
check('60a', "[60] no row: label — and every bar button disabled",
      qcw._sel_lbl.text() == '—' and not any(
          b.isEnabled() for b in (qcw.btn_open, qcw.btn_exclude,
                                  qcw.btn_redetect)))
qcw.select_channel('E70')
row = [qcw.btn_open, qcw.btn_exclude, qcw.btn_redetect, qcw.btn_queue_hard]
# reading order (the bar wraps onto a second line when the tab is narrow)
xs = [(p.y(), p.x()) for p in (b.mapTo(qcw, QtCore.QPoint(0, 0))
                               for b in row)]
check('60b', "[R4.5] bar, left to right: Open in Epochs, Exclude channel, "
      "Add to re-detect queue, …, Queue all HARD (n) with its tooltip; no "
      "Build button",
      [b.text() for b in row[:3]] == ['Open in Epochs', 'Exclude channel',
                                      'Add to re-detect queue']
      and xs == sorted(xs)
      and qcw.btn_queue_hard.text().startswith('Queue all HARD (')
      and qcw.btn_queue_hard.toolTip() == 'Add every channel with a hard '
      'amp flag to the re-detect queue.'
      and not hasattr(qcw, 'btn_build') and not hasattr(qcw, 'tray_body'),
      repr(xs))
def n_checks_included(df):
    """Checks-flagged channels among the included ones."""
    inc = df[~er.excluded_mask(df)]
    return int(inc['checks_flag'].isin(['hard', 'soft']).sum())


EXC_TIP = ('Leaves this channel out of review samples, the re-run export, '
           'the flag statistics and the topography. Event density and '
           'exported events are unchanged in 4.6.0. Click again to include '
           'it.')
qcw.select_channel('E62')
check('103a', "[103] an included channel: 'Exclude channel', not checked, "
      "with the tooltip", qcw.btn_exclude.text() == 'Exclude channel'
      and not qcw.btn_exclude.isChecked()
      and qcw.btn_exclude.objectName() == 'danger'
      and qcw.btn_exclude.toolTip() == EXC_TIP + ' Applies to spindles '
      'only.', repr(qcw.btn_exclude.toolTip()))
qcw.btn_exclude.click()
app.processEvents()
msg_ex = win.status_bar.currentMessage()
win.on_qc_drill('E29', switch_tab=False)
win.epochs_panel.exclude_btn.click()
app.processEvents()
ver = win.db.get_channel_verdicts()
qcw.select_channel('E62')
check('104a', "[104] Exclude channel (bottom bar and Epochs tab) writes "
      "verdict 'drop'; Status reads × excluded; the buttons read 'Include "
      "channel', checked",
      ver.get(('E62', 'spindle')) == ver.get(('E29', 'spindle')) == 'drop'
      and col_cell(win, 'E62', 'verdict') == '× excluded'
      and qcw.btn_exclude.text() == 'Include channel'
      and qcw.btn_exclude.isChecked() and qcw.btn_exclude.objectName() == ''
      and win.epochs_panel.exclude_btn.text() == 'Include channel'
      and win.epochs_panel.exclude_btn.isChecked()
      and win.epochs_panel.exclude_btn.toolTip() == EXC_TIP
      + ' Applies to spindles only.'
      and msg_ex == 'Excluded E62 from spindles: left out of review '
      'samples, the re-run export, the flag statistics and the topography.',
      repr((ver.get(('E62', 'spindle')), col_cell(win, 'E62', 'verdict'),
            msg_ex)))
check('106a', "[106] an excluded row is not judged: Amp flag and Checks "
      "read —; the header counts it only in ' · 2 excluded'",
      col_cell(win, 'E62', 'flag') == '—'
      and col_cell(win, 'E62', 'checks_flag') == '—'
      and qcw.counts_lbl.text().endswith(' · 2 excluded')
      and qcw.counts_lbl.text().startswith(
          f"{n_checks_included(qcw.model.df)} checks flagged · "), repr(qcw.counts_lbl.text()))
win.db.set_channel_verdict('E30', 'spindle', 'channel_artefact', 'TK')
refresh(win)
qcw.select_channel('E30')
check('104b', "[104] a row stored as 'channel_artefact' reads × excluded "
      "and 'Include channel'", col_cell(win, 'E30', 'verdict') == '× excluded'
      and qcw.btn_exclude.text() == 'Include channel'
      and qcw.btn_exclude.isChecked())
qcw.btn_exclude.click()
app.processEvents()
check('104c', "[104] Include channel writes '' and reports it",
      win.db.get_channel_verdicts().get(('E30', 'spindle')) == ''
      and col_cell(win, 'E30', 'verdict') == 'kept'
      and win.status_bar.currentMessage() == 'Included E30 again.',
      repr(win.status_bar.currentMessage()))
shows = [qcw.show_combo.itemText(i) for i in range(qcw.show_combo.count())]


def window_texts(root):
    out = []
    for w in root.findChildren(QtWidgets.QWidget):
        if isinstance(w, (QtWidgets.QAbstractButton, QtWidgets.QLabel)):
            out.append(w.text())
        if isinstance(w, QtWidgets.QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
    out += [a.text() for a in root.findChildren(QtWidgets.QAction)]
    out.append(root.status_bar.currentMessage())
    return out


texts = window_texts(win)
check('105', "[105] Show has 'Excluded (2)' and no 'Dropped'; no visible "
      "string says 'dropped' or 'marked artefact'; [103] no 'Drop channel' "
      "or 'Mark channel artefact' anywhere",
      'Excluded (2)' in shows and not any(t.startswith('Dropped')
                                          for t in shows)
      and not [t for t in texts if 'dropped' in t.lower()
               or 'marked artefact' in t.lower() or t in (
                   'Drop channel', 'Mark channel artefact')],
      repr((shows, [t for t in texts if 'dropped' in t.lower()])))
menu_titles = [a.text() for a in win.menuBar().actions()]
analysis = next(a.menu() for a in win.menuBar().actions()
                if a.text().replace('&', '') == 'Analysis')
src = open(rg.__file__, encoding='utf-8').read()
check('109', "[109] no Selection tray, no 'Build re-detect request…' "
      "(button, menu item or JSON writer)",
      not [t for t in texts if t in ('Selection', 'Build re-detect request…')
           or t.startswith(('CHANNEL ARTEFACTS', 'RE-DETECT QUEUE',
                            'Build re-detect request'))]
      and not [a for a in analysis.actions() if 'detect' in a.text().lower()]
      and 'redetect_request' not in src
      and not hasattr(win, 'open_redetect_modal')
      and not hasattr(win, 'btn_build_redetect'), repr(menu_titles))
# re-detect queue: stored, shown in Status and Show, toggled by F
n_q0 = next(t for t in shows if t.startswith('Queued for re-detect ('))
qcw.select_channel('E75')
check('110a', "queue empty: Show reads 'Queued for re-detect (0)', the hint "
      "link is hidden, the button reads 'Add to re-detect queue'",
      n_q0 == 'Queued for re-detect (0)' and not qcw.queue_link.isVisibleTo(
          qcw) and qcw.btn_redetect.text() == 'Add to re-detect queue')
qcw.btn_redetect.click()
app.processEvents()
qcw.select_channel('E75')
shows = [qcw.show_combo.itemText(i) for i in range(qcw.show_combo.count())]
check('110b', "[110] Add to re-detect queue: button reads 'Remove from "
      "re-detect queue', Status gains ' · ↻ re-detect' with its tooltip, "
      "Show reads 'Queued for re-detect (1)'",
      qcw.btn_redetect.text() == 'Remove from re-detect queue'
      and qcw.btn_redetect.isChecked()
      and col_cell(win, 'E75', 'verdict') == 'kept · ↻ re-detect'
      and col_cell(win, 'E75', 'verdict', Qt.ToolTipRole) ==
      'Queued for re-detection. Use File ▸ Export re-run package… to re-run '
      'these channels.'
      and '↻ re-detect' in col_cell(win, 'E75', 'verdict',
                                    rg._STATUS_HTML_ROLE)
      and 'Queued for re-detect (1)' in shows,
      repr((col_cell(win, 'E75', 'verdict'), shows)))
qcw.select_channel('E62')
win._flag_selected_qc_row()              # the F key's slot
app.processEvents()
# The link runs the real export slot. With no annotation file loaded the
# export stops at its warning box; capture that instead of showing it.
emitted = []
_warn = QtWidgets.QMessageBox.warning
QtWidgets.QMessageBox.warning = staticmethod(
    lambda *a, **k: emitted.append(a[2]))
link_text, link_vis = qcw.queue_link.text(), qcw.queue_link.isVisibleTo(qcw)
qcw.queue_link.click()
app.processEvents()
QtWidgets.QMessageBox.warning = _warn
kept, excl, redet = win._rerun_channel_lists()
file_menu = next(a.menu() for a in win.menuBar().actions()
                 if a.text().replace('&', '') == 'File')
exp = [a for a in file_menu.actions() if a.text() == 'Export re-run package…']
check('111a', "[110, 111] F queues the selected row too: an excluded "
      "queued channel reads '× excluded · ↻ re-detect'; the bar shows the "
      "link '2 queued · Export re-run package…'; File has the same item "
      "with its status tip; the export list is the queue minus excluded",
      col_cell(win, 'E62', 'verdict') == '× excluded · ↻ re-detect'
      and link_text == '2 queued · Export re-run package…' and link_vis
      and len(exp) == 1 and exp[0].statusTip() == 'Writes the queued '
      'channels (redetect_channels.csv) for examples/rerun_detection.py '
      '--channels.' and len(emitted) == 1
      and emitted[0].startswith('Load the base annotation XML first')
      and redet == ['E75']
      and set(excl) == {'E62', 'E29'} and 'E62' not in kept
      and 'E75' in kept, repr((link_text, redet, excl)))
P1_path = win.db.db_path
win.close()
win = open_window(P1_path)
refresh(win)
qcw = win.qc_widget
con_chk = sqlite3.connect(P1_path)
stored = sorted(r[0] for r in con_chk.execute(
    "SELECT DISTINCT channel FROM channel_qc WHERE redetect = 1"))
con_chk.close()
check('110c', "[110] after closing and reopening the window on the same "
      "database both channels are still queued (channel_qc.redetect)",
      stored == ['E62', 'E75'] and win._redetect_queue == {'E62', 'E75'}
      and col_cell(win, 'E75', 'verdict') == 'kept · ↻ re-detect',
      repr((stored, win._redetect_queue)))
for ch in ('E62', 'E75'):
    qcw.select_channel(ch)
    qcw.btn_redetect.click()
    app.processEvents()
for ch in ('E62', 'E29'):                 # include both again
    win._set_channel_excluded(ch, False)
app.processEvents()
win.export_rerun_package()                # nothing queued or excluded
check('111b', "[111] empty queue: the link is hidden and the export says "
      "so", not qcw.queue_link.isVisibleTo(qcw)
      and win.status_bar.currentMessage() == 'Nothing is queued for '
      're-detection. Select a channel and use "Add to re-detect queue".',
      repr(win.status_bar.currentMessage()))
line1 = qcw.counts_lbl.text()
check('56b', "[56] with nothing excluded the header has no excluded part",
      'excluded' not in line1, repr(line1))
foot = qcw.footer.text()
ftip = qcw.footer.toolTip()
check('62a', "[102] footer: the one R4.0 line with the live limits, no "
      "line break; the full rule in the tooltip", foot ==
      '× hard / ▲ soft: the channel stands out from the others (robust z '
      'above 3.5 / 2). Low prominence never flags. Hover for the full rule.'
      and '\n' not in foot and not qcw.footer.wordWrap()
      and all(t in ftip for t in (
          'mean event amplitude, its 95th percentile or its largest event',
          '10 percentage points', '0.3×', 'at least 20 events',
          'Low prominence is shown for context and never flags.',
          'Excluded channels are left out of the comparison.',
          'Change the limits in View ▸ Outlier threshold….'))
      and len(ftip.split('\n')) == 6, repr(foot))
check('62c', "[R4.3] slow waves: no low-prominence sentence in the line, "
      "five tooltip lines; no amp/thr when the method has no ratio",
      er.footer_text(3.5, 2.0, 'slow_wave') == '× hard / ▲ soft: the '
      'channel stands out from the others (robust z above 3.5 / 2). Hover '
      'for the full rule.'
      and len(er.footer_tooltip(3.5, 2.0, 'slow_wave').split('\n')) == 5
      and 'amp/thr' not in er.footer_tooltip(3.5, 2.0, 'spindle', False))
hdrs = [qcw.model.headerData(i, Qt.Horizontal) for i in range(
    qcw.model.columnCount())]
check('113', "[113] header 'Mean amp µV' (no 'Med amp µV'), with the two "
      "amplitude header tooltips", 'Mean amp µV' in hdrs
      and 'Med amp µV' not in hdrs
      and qcw.model.headerData(rg._QC_COL_INDEX['mean_amp'], Qt.Horizontal,
                               Qt.ToolTipRole) == "Mean amplitude of this "
      "channel's events (detection-band signal)."
      and qcw.model.headerData(rg._QC_COL_INDEX['amp_z'], Qt.Horizontal,
                               Qt.ToolTipRole).startswith(
          'Robust z of the amplitude measure furthest from the other '
          'channels'), repr(hdrs))
check('114a', "[114] one run in view: the top bar names the detector",
      win.lbl_detector.text() == 'detector: Moelle2011 · 9–12 Hz'
      and win.lbl_detector.toolTip() == '', repr(win.lbl_detector.text()))
win._qc_thresholds.update(hard_z=4.0, soft_z=2.5)
refresh(win)
foot = qcw.footer.text()
check('62b', "[102] after hard 4.0 / soft 2.5", 'above 4 / 2.5)' in foot
      and 'above 4, soft above 2.5' in qcw.footer.toolTip(), repr(foot))
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
      and fl.row('E70')['drop'].text() == 'Exclude channel',
      repr((vis1, vis2)))
fl.row('E62')['drop'].click()
app.processEvents()
qc_x = win._qc_df.set_index('channel')
check('70', "[R4.4] the flagged list's Exclude channel excludes it; an "
      "excluded channel is not judged, so it leaves the list",
      win.db.get_channel_verdicts().get(('E62', 'spindle')) == 'drop'
      and win.detail_dock_w.flagged.row('E62') is None
      and qc_x.loc['E62', 'checks_flag'] == ''
      and bool(qc_x.loc['E62', 'excluded']))
# topography rings, labels and caption
grid = {ch: ((i % 8) / 8.0 - 0.45, (i // 8) / 6.0 - 0.4)
        for i, ch in enumerate(sorted(qc.index))}
win.detail_dock_w.set_coords(grid)
dk = win.detail_dock_w
dk.topo_combo.setCurrentIndex(dk.topo_combo.findData('pct_off_band'))
app.processEvents()
qc = win._qc_df.set_index('channel')        # E62 is excluded now
ringed = set(qc.index[qc['checks_flag'].isin(['hard', 'soft'])])
pens = {r.channel: r.opts['pen'].style() for r in dk.ring_items}
spots = dk.excluded_spots
check('107a', "[107] the excluded channel: a hollow marker (no fill), not "
      "in the interpolation input, legend shown; no ring",
      spots is not None and [p.data()[0] for p in spots.points()] == ['E62']
      and spots.opts['brush'].style() == Qt.NoBrush
      and 'E62' not in dk.topo_input_channels
      and set(dk.topo_input_channels) == set(
          qc.index[qc['pct_off_band'].notna()]) - {'E62'}
      and dk.topo_excluded_lbl.isVisibleTo(dk)
      and dk.topo_excluded_lbl.text()
      == '○ excluded channel (not used for the map)'
      and 'E62' not in ringed, repr((
          spots is not None and [p.data() for p in spots.points()],
          spots is not None and spots.opts['brush'],
          'E62' in dk.topo_input_channels, len(dk.topo_input_channels),
          len(qc), dk.topo_excluded_lbl.isVisibleTo(dk))))
check('63a', "[63] one ring per checks-flagged channel (none for the "
      "excluded one); solid hard, dashed soft; one label each (≤ 12)",
      'E62' not in pens and set(pens) == ringed and all(
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
dk.topo_combo.setCurrentIndex(dk.topo_combo.findData('pct_off_band'))
win._set_channel_excluded('E62', False)
app.processEvents()
qc_in = win._qc_df.set_index('channel')
check('107b', "[107] with no channel excluded the legend is hidden and "
      "E62 feeds the map again", not dk.topo_excluded_lbl.isVisibleTo(dk)
      and dk.excluded_spots is None and 'E62' in dk.topo_input_channels
      and set(dk.topo_input_channels) == set(
          qc_in.index[qc_in['pct_off_band'].notna()]), repr((
          dk.topo_excluded_lbl.isVisibleTo(dk), dk.excluded_spots,
          len(dk.topo_input_channels), len(qc))))
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

# =================================================================== R4
say("\n== R4. Checked style, sort guard, exclusion, regions, flag trigger")
qss = rg.DARK_QSS
m = re.search(r'QPushButton:checked, QToolButton:checked \{([^}]*)\}', qss)
body = m.group(1) if m else ''
check('100', "[100] the stylesheet has a QPushButton:checked and a "
      "QToolButton:checked rule: weight 600, accent border, soft accent "
      "fill; checked + disabled keeps the weight with a text_3 border; an "
      "armed button has the border and no fill",
      m is not None and 'font-weight: 600' in body
      and f"border: 1px solid {rg.THEME['accent']}" in body
      and f"background: {rg.THEME['accent_soft']}" in body
      and re.search(r'QPushButton:checked:disabled, QToolButton:checked:'
                    r'disabled \{[^}]*border: 1px solid '
                    + re.escape(rg.THEME['text_3']) + r'[^}]*font-weight: 600',
                    qss) is not None
      and 'background' not in rg.EventDecisionPanel.ARMED_QSS
      and rg.THEME['accent'] in rg.EventDecisionPanel.ARMED_QSS,
      repr(body.strip()))
marks = rg.checkbox_mark_qss()
check('100b', "[R4.2] checked checkboxes get a tick image and partly "
      "checked ones a dash (files written once, then cached)",
      'QCheckBox::indicator:checked' in marks
      and 'QCheckBox::indicator:indeterminate' in marks
      and len(re.findall(r'url\("([^"]+)"\)', marks)) == 2
      and all(os.path.exists(p)
              for p in re.findall(r'url\("([^"]+)"\)', marks))
      and rg.checkbox_mark_qss() is marks, repr(marks.strip()[:80]))
win = open_window(P1)
refresh(win)
qcw = win.qc_widget
b_ = {b.text(): b for b in qcw.stage_group.buttons()}
check('101', "[101] with no stored review/check_stage the combined Stage "
      "button is the checked one", b_['NREM2 + NREM3'].isChecked()
      and not b_['NREM2'].isChecked() and not b_['NREM3'].isChecked(),
      repr({k: v.isChecked() for k, v in b_.items()}))
hdr = qcw.table.horizontalHeader()
items0 = [qcw.sort_combo.itemText(i) for i in range(qcw.sort_combo.count())]
hdr.setSortIndicator(-1, Qt.AscendingOrder)
app.processEvents()
items1 = [qcw.sort_combo.itemText(i) for i in range(qcw.sort_combo.count())]
hdr.setSortIndicator(rg._QC_COL_INDEX['n'], Qt.DescendingOrder)   # a click
app.processEvents()
check('98', "[98] setSortIndicator(-1) never adds 'Column header'; a real "
      "header sort on Events does", 'Column header' not in items0
      and items1 == items0
      and qcw.sort_combo.currentText() == 'Column header',
      repr((items1, qcw.sort_combo.currentText())))
qcw.sort_combo.setCurrentText('Checks (hard first)')
win.close()

# two runs in view -> detector: — with the tooltip; none -> the other tooltip
P_two = os.path.join(TMP, 'two_runs.db')
con = fx.open_schema(P_two)
fx.add_run(con, RUN)
fx.add_run(con, 'run-b', timestamp='2026-09-01T10:00:00')
for ch, run_ in (('Cz', RUN), ('Fz', RUN), ('Pz', 'run-b')):
    rows, _a = fx.make_rows(ch, 40, run_, rng)
    fx.insert_rows(con, rows)
con.commit()
con.close()
win = open_window(P_two)
refresh(win)
check('114b', "[114] two runs in view: 'detector: —' with the tooltip",
      win.lbl_detector.text() == 'detector: —' and win.lbl_detector.toolTip()
      == 'Several detection runs are in view. Choose a method and band in '
      'the Filters dock.', repr(win.lbl_detector.text()))
check('114c', "no events in view: 'detector: —', 'No events in view.'",
      er.detector_label(None) == ('—', 'No events in view.')
      and er.detector_label({'res': {'runs_in_view': []}})[1]
      == 'No events in view.')
win.close()

# regions from 10-20 / 10-5 labels; coordinates only for EGI labels [108]
from turtlewave_hdEEG.utils import region_from_label          # noqa: E402
xy_parietal = {c: (0.1, -0.3) for c in ('F1h', 'F2h', 'Fz', 'E75', 'XYZ')}
ev_lab = pd.DataFrame({'channel': ['F1h', 'F2h', 'Fz', 'PPO1h', 'E75'] * 3,
                       'start_time': np.arange(15.0), 'end_time':
                       np.arange(15.0) + 1, 'max_amp': 30.0,
                       'peak2peak_amp': 60.0})
reg = rg.compute_channel_qc(ev_lab, coords=xy_parietal).set_index(
    'channel')['region']
check('108', "[108] F1h, F2h and Fz are frontal in the table whatever the "
      "coordinates say, the same string the sample strata use; E75 still "
      "takes its region from coordinates; an unknown label too",
      [reg[c] for c in ('F1h', 'F2h', 'Fz')] == ['frontal'] * 3
      and all(reg[c] == region_from_label(c) for c in ('F1h', 'F2h', 'Fz',
                                                       'PPO1h'))
      and rg._region_from_xy(0.1, -0.3) == 'parietal'
      and reg['E75'] == 'parietal'
      and rg._region_for_channel('XYZ', xy_parietal) == 'parietal'
      and rg._region_for_channel('XYZ') == 'other', repr(dict(reg)))


# amp flag: trigger, z and exclusion on a hand-made montage [106, 112]
def amp_events(spec, n=40):
    """``spec``: {channel: (amplitudes, peak-to-peak)} -> an events frame."""
    rows = []
    for ch, (amps, p2p) in spec.items():
        for i, (a, p) in enumerate(zip(amps, p2p)):
            rows.append({'channel': ch, 'start_time': float(i),
                         'end_time': float(i) + 1.0, 'max_amp': float(a),
                         'peak2peak_amp': float(p)})
    return pd.DataFrame(rows)


g = np.random.default_rng(11)
spec = {f"C{i:02d}": (30 + g.normal(0, 1.0, 40) + 0.2 * i,
                      60 + g.normal(0, 1.0, 40) + 0.2 * i) for i in range(12)}
low = np.sort(spec['C00'][0].copy())
low[:32] -= 20.0             # the lower 80 % of events: mean falls, p95 stays
spec['MEAN'] = (low, spec['C00'][1])
top = spec['C01'][0].copy()
top[:4] += 200.0                         # 10 % of events: p95 far, mean less
spec['P95'] = (top, spec['C01'][1])
big = spec['C02'][1].copy()
big[0] = 900.0                           # one event: only the largest
spec['MAXEV'] = (spec['C02'][0], big)
qa = rg.compute_channel_qc(amp_events(spec)).set_index('channel')
check('112a', "[112] the trigger is the measure with the largest |z| over "
      "the limit: mean, 95th pct, largest event",
      (qa.loc['MEAN', 'flag'], qa.loc['MEAN', 'flag_trigger'])
      == ('hard', 'mean_amp')
      and (qa.loc['P95', 'flag'], qa.loc['P95', 'flag_trigger'])
      == ('hard', 'p95_amp')
      and (qa.loc['MAXEV', 'flag'], qa.loc['MAXEV', 'flag_trigger'])
      == ('hard', 'max_p2p')
      and abs(qa.loc['MAXEV', 'amp_z'] - qa.loc['MAXEV', 'sz_max_p2p']) < 1e-9
      and abs(qa.loc['MAXEV', 'sz_mean_amp']) < qa.loc['MAXEV', 'amp_z']
      and qa.loc['C05', 'flag_trigger'] == ''
      and abs(qa.loc['C05', 'amp_z']) == max(
          abs(qa.loc['C05', k]) for k in ('sz_mean_amp', 'sz_p95_amp',
                                          'sz_max_p2p')),
      repr(qa.loc[['MEAN', 'P95', 'MAXEV'], ['flag', 'flag_trigger',
                                             'amp_z']].to_dict('index')))
model = rg.ChannelQCModel()
model.set_data(qa.reset_index(), {}, set(), 'spindle')


def cell(ch, key, role=Qt.DisplayRole):
    for r in range(model.rowCount()):
        if model.channel_at(r) == ch:
            return model.data(model.index(r, rg._QC_COL_INDEX[key]), role)


tipz = cell('MAXEV', 'amp_z', Qt.ToolTipRole)
check('112b', "[112] cells: '× HARD · largest event' / '· mean' / "
      "'· 95th pct'; Amp z shows that measure's z; its tooltip lists all "
      "three; Amp z sorts by |z|",
      cell('MAXEV', 'flag') == '× HARD · largest event'
      and cell('MEAN', 'flag') == '× HARD · mean'
      and cell('P95', 'flag') == '× HARD · 95th pct'
      and cell('C05', 'flag') == '✓ OK'
      and cell('MAXEV', 'amp_z') == f"{qa.loc['MAXEV', 'sz_max_p2p']:.1f}"
      and re.match(r'^mean z -?\d+\.\d · 95th pct z -?\d+\.\d · largest '
                   r'event z -?\d+\.\d$', tipz or '') is not None
      and cell('MEAN', 'amp_z', Qt.UserRole) == abs(qa.loc['MEAN', 'amp_z']),
      repr((cell('MAXEV', 'flag'), cell('MAXEV', 'amp_z'), tipz)))
# one channel so large that it widens the spread: excluding it changes the
# others' flags, exactly as if it were not in the montage
spec2 = {k: v for k, v in spec.items() if k.startswith('C')}
for i, name in enumerate(('X1', 'X2', 'X3', 'X4', 'X5', 'X6')):
    spec2[name] = (spec['C00'][0] + 6.0 + i, spec['C00'][1])
ev2 = amp_events(spec2)
q_all = rg.compute_channel_qc(ev2).set_index('channel')
q_exc = rg.compute_channel_qc(ev2, excluded={'X6'}).set_index('channel')
q_wo = rg.compute_channel_qc(ev2[ev2['channel'] != 'X6']).set_index('channel')
others = [c for c in q_wo.index]
check('106b', "[106] excluded={X6}: every other channel's flag, trigger and "
      "z equal a montage computed without X6 (and differ from the montage "
      "with it); X6 itself is not flagged",
      list(q_exc.loc[others, 'flag']) == list(q_wo.loc[others, 'flag'])
      and list(q_exc.loc[others, 'flag_trigger'])
      == list(q_wo.loc[others, 'flag_trigger'])
      and np.allclose(q_exc.loc[others, 'amp_z'], q_wo.loc[others, 'amp_z'])
      and not np.allclose(q_exc.loc[others, 'amp_z'],
                          q_all.loc[others, 'amp_z'])
      and q_exc.loc['X6', 'flag'] == '' and bool(q_exc.loc['X6', 'excluded'])
      and not q_exc.loc[others, 'excluded'].any(),
      repr((list(q_all.loc[others, 'flag']), list(q_exc.loc[others, 'flag']))))
pop = pd.DataFrame({'channel': [f"K{i}" for i in range(12)]})
for col in er.CHECK_COLUMNS:
    pop[col] = np.linspace(5.0, 8.0, 12)
    pop[er.CHECK_N[col]] = 100
pop.loc[11, 'pct_off_band'] = 70.0
pop.loc[10, 'pct_off_band'] = 30.0
p_all, _m = er.population_flags(pop)
p_exc, med_exc = er.population_flags(pop, excluded={'K11'})
p_wo, med_wo = er.population_flags(pop[pop['channel'] != 'K11'])
check('106c', "[106] checks: excluded={K11} gives the other channels the "
      "flags and medians of a montage without K11; K11 itself is not "
      "flagged", list(p_exc['checks_flag'][:11]) == list(p_wo['checks_flag'])
      and med_exc['pct_off_band'] == med_wo['pct_off_band']
      and p_all.loc[11, 'checks_flag'] == 'hard'
      and p_exc.loc[11, 'checks_flag'] == ''
      and p_exc.loc[10, 'checks_flag'] in ('hard', 'soft'),
      repr((list(p_all['checks_flag']), list(p_exc['checks_flag']))))

# channel_qc: an older table gains the redetect column; a verdict keeps it
P_mig = os.path.join(TMP, 'old_qc.db')
con = fx.open_schema(P_mig)
fx.add_run(con, RUN)
rows, _a = fx.make_rows('Cz', 30, RUN, rng)
fx.insert_rows(con, rows)
con.execute("CREATE TABLE channel_qc (channel TEXT, event_type TEXT, "
            "verdict TEXT, reviewer TEXT, qc_timestamp TEXT, "
            "PRIMARY KEY (channel, event_type))")
con.execute("INSERT INTO channel_qc VALUES ('Cz', 'spindle', 'drop', 'TK', "
            "'2026-09-01')")
con.commit()
con.close()
dbm = rg.EventDatabase(P_mig)
cols = [r[1] for r in dbm.conn.execute("PRAGMA table_info(channel_qc)")]
dbm.set_channel_redetect('Cz', True, 'spindle')
dbm.set_channel_redetect('Fz', True, 'spindle')
dbm.set_channel_verdict('Cz', 'spindle', '', 'TK')
q1 = dbm.get_redetect_queue()
dbm.set_channel_redetect('Cz', False)
check('mig', "an older channel_qc table gains 'redetect' (rows kept); a "
      "verdict change keeps the queue mark; a channel with no row can be "
      "queued; un-queue clears it",
      'redetect' in cols and q1 == {'Cz', 'Fz'}
      and dbm.get_channel_verdicts()[('Cz', 'spindle')] == ''
      and dbm.get_channel_verdicts()[('Fz', 'spindle')] == ''
      and dbm.get_redetect_queue() == {'Fz'}, repr((cols, q1)))
dbm.conn.close()

# designer sign-off: a 1366 x 768 window, both docks shown
# measured with the application stylesheet, as main() applies it
src = open(rg.__file__, encoding='utf-8').read()
app.setStyle('Fusion')
app.setStyleSheet(rg.DARK_QSS + rg.checkbox_mark_qss())
win = open_window(P1)
win._ask_reviewer_name = lambda prefill: ('TK', True)
win.set_reviewer_name('TK')
win.resize(1366, 768)
win.show()
refresh(win)
qcw = win.qc_widget
qcw.select_channel('E7')
win.on_qc_add_redetect('E7')
win._set_channel_excluded('E7', True)
win.on_qc_drill('E7', switch_tab=True)
ep = win.epochs_panel
ep.select_event(str(ep._ev.sort_values('_start')['uuid'].iloc[0]))
for _ in range(4):
    app.processEvents()
evp = win.detail_dock_w.event_panel
dock = win.detail_dock
scroll = dock.widget()
vals = {}
for k in evp.row_keys():
    v = evp._row_widgets[k][1]
    x0 = v.mapTo(win, QtCore.QPoint(0, 0)).x()
    plain = re.sub(r'<[^>]+>', '', v.text())
    widest = max(v.fontMetrics().horizontalAdvance(w)
                 for w in plain.split(' '))
    vals[k] = (x0, x0 + v.width(), widest <= v.width(),
               v.heightForWidth(v.width()) <= v.height() + 1, plain)
widths = {'window': (win.width(), win.height()),
          'right dock': (dock.x(), dock.x() + dock.width()),
          'dock h-scroll': scroll.horizontalScrollBar().maximum(),
          'channels min': qcw.minimumSizeHint().width(),
          'epochs min': ep.minimumSizeHint().width()}
say(f"  measured at 1366 x 768: {widths}")
check('w1', "[sign-off 1] at 1366 x 768 with both docks shown the window "
      "keeps that size, the right dock's right edge is inside it, the dock "
      "does not scroll sideways, and its minimum width is 300",
      widths['window'] == (1366, 768) and win.filter_dock.isVisible()
      and dock.isVisible() and widths['right dock'][1] <= win.width()
      and widths['dock h-scroll'] == 0 and dock.minimumWidth() == 300,
      repr(widths))
check('w2', "[sign-off 1] each of the four Event-panel value labels is "
      "fully visible: inside the window, wide enough for its longest word, "
      "tall enough for its wrapped lines",
      list(vals) == ['signal_bg', 'duration', 'peak_freq', 'outlier']
      and all(0 <= a and b <= win.width() and fits and tall
              for a, b, fits, tall, _t in vals.values()), repr(vals))
check('w3', "[sign-off 1] the Channels tab's minimum width is at most 760 "
      "px; Show and Sort can shrink to 140 px; the count line is its own "
      "row under the control row, left-aligned",
      widths['channels min'] <= 760
      and qcw.show_combo.minimumSizeHint().width() == 140
      and qcw.sort_combo.minimumSizeHint().width() == 140
      and qcw.counts_lbl.y() >= qcw.control_row.y() + qcw.control_row.height()
      and qcw.counts_lbl.x() == qcw.control_row.x()
      and qcw.counts_lbl.alignment() & Qt.AlignLeft
      and qcw.counts_lbl.parentWidget() is qcw, repr(widths))
raw_line = QtWidgets.QLabel.text(evp.event_line)
check('w4', "[sign-off 1] the Event panel header line wraps only at its "
      "' · ' separators (no-break spaces inside a part); text() is the "
      "plain line", evp.event_line.wordWrap()
      and evp.event_line.text().count(' · ') == 3
      and '\u00a0' not in evp.event_line.text()
      and raw_line.split(' ') == [p.replace(' ', '\u00a0') + '\u00a0·'
                                  for p in evp.event_line.text().split(
                                      ' · ')[:-1]]
      + [evp.event_line.text().split(' · ')[-1].replace(' ', '\u00a0')],
      repr(raw_line))
win.tabs.setCurrentIndex(0)
app.processEvents()
fm = qcw.table.fontMetrics()
wc = qcw.table.columnWidth(rg._QC_COL_INDEX['checks_flag'])
ws = qcw.table.columnWidth(rg._QC_COL_INDEX['verdict'])
check('w5', "[sign-off 2] Checks is at least 170 px and fits '× HARD · "
      "off-band 70 %'; Status is at least 170 px and fits '× excluded · "
      "↻ re-detect' (the cell shown for E7)",
      wc >= 170 and fm.horizontalAdvance('× HARD · off-band 70 %') + 10 <= wc
      and ws >= 170
      and fm.horizontalAdvance('× excluded · ↻ re-detect') + 10 <= ws
      and col_cell(win, 'E7', 'verdict') == '× excluded · ↻ re-detect',
      repr((wc, ws, fm.horizontalAdvance('× excluded · ↻ re-detect'))))
win.resize(2400, 1000)                  # room for the whole bar on one line
app.processEvents()
row = [qcw.btn_open, qcw.btn_exclude, qcw.btn_redetect, qcw.queue_link,
       qcw.btn_queue_hard]
pos = [b.mapTo(qcw, QtCore.QPoint(0, 0)) for b in row]
check('w6', "wide window: the bottom bar is one line in the R4.5 order, "
      "with the queue link and Queue all HARD at the right edge",
      qcw.action_row.width() >= qcw.action_row.sizeHint().width()
      and max(p.y() + b.height() // 2 for p, b in zip(pos, row))
      - min(p.y() + b.height() // 2 for p, b in zip(pos, row)) <= 2
      and [p.x() for p in pos] == sorted(p.x() for p in pos)
      and pos[-1].x() + row[-1].width() >= qcw.action_row.x()
      + qcw.action_row.width() - 2
      and pos[3].x() - (pos[2].x() + row[2].width()) > 40,
      repr([(p.x(), p.y()) for p in pos]))
win.on_qc_add_redetect('E7')
win._set_channel_excluded('E7', False)
dk = win.detail_dock_w
check('s3', "[sign-off 3, 5] strings: the Worst-events tooltip, Design "
      "notes, the export summary, the density warning; the old SELECTED "
      "CHANNEL subtitle is gone",
      rg._artefact_tooltip(1234.0) == '1234 µV peak-to-peak — exceeds '
      'physiological scale (>1000 µV).'
      and re.search(r'brush a time range and use \\"Exclude time range…\\" '
                    r'to "\s+"leave it out of analysis for every channel '
                    r'\(written to a "\s+"sidecar XML', src) is not None
      and 'to mark an artefact' not in src
      and 'Excluded time ranges appended (all channels): {n_iv}' in src
      and 'Sidecar artefacts appended' not in src
      and 'almost certainly artefact' not in src
      and '%d excluded time range(s) pending.' in src
      and 'artefact mark(s) pending' not in src
      and not hasattr(dk, 'subtitle')
      and not [l.text() for l in dk.findChildren(QtWidgets.QLabel)
               if re.search(r'· n=\d+$', l.text()) or 'flag ok' in l.text()])
win.close()
app.setStyleSheet('')

# gate low items: export wording, units from a later source, quoted url()
check('g3', "[gate 3] the export summary and the QC report say 'Excluded "
      "for any event type: …' and name a queued channel that is left out "
      "because it is excluded; the tooltip names the event type",
      er.rerun_summary(36, ['E62', 'E29'], ['E62']) ==
      'channels.csv: 36 kept\nExcluded for any event type: E29, E62\n'
      'Queued but excluded, not re-detected: E62\n'
      and er.rerun_summary(38, [], []) ==
      'channels.csv: 38 kept\nExcluded for any event type: none\n'
      and er.excluded_any_type_line({'Cz'}) ==
      'Excluded for any event type: Cz'
      and rg.exclude_tip('slow_wave').endswith(' Applies to slow waves only.')
      and rg.exclude_tip() == EXC_TIP
      and '_er.excluded_any_type_line(' in src and '_er.rerun_summary(' in src)
from frontend.channel_types import channel_units             # noqa: E402
bids = os.path.join(TMP, 'bids dir')
os.makedirs(bids, exist_ok=True)
with open(os.path.join(bids, 'sub-01_task-psg_channels.tsv'), 'w',
          encoding='utf-8') as fh:
    fh.write('name\ttype\tunits\nVEOG\tEOG\tmV\nECG\tECG\tn/a\n'
             'Cz\tEEG\tuV\n')
units = channel_units({'chan_name': ['VEOG', 'ECG', 'Cz', 'Fz'],
                       'chan_unit': ['', 'n/a', 'µV', None]},
                      os.path.join(bids, 'sub-01_task-psg_desc-clean_eeg.set'))
check('g4', "[gate 4] an empty or 'n/a' unit in the header does not block "
      "the unit a BIDS channels.tsv states; a stated header unit wins; "
      "'n/a' in every source stays None",
      units == {'VEOG': 'mV', 'ECG': None, 'Cz': 'µV', 'Fz': None},
      repr(units))
odd = os.path.join(TMP, 'marks dir (copy) 1')
os.makedirs(odd, exist_ok=True)
marks_odd = rg.checkbox_mark_qss(odd)
box = QtWidgets.QCheckBox('x')
box.setStyleSheet(rg.DARK_QSS + marks_odd)
box.setChecked(True)
box.resize(60, 24)
box.show()
app.processEvents()
img = box.grab().toImage()
light = sum(1 for x in range(0, 16) for y in range(img.height())
            if QtGui.QColor(img.pixel(x, y)).lightness() > 200)
box.setStyleSheet(rg.DARK_QSS)
app.processEvents()
img0 = box.grab().toImage()
light0 = sum(1 for x in range(0, 16) for y in range(img0.height())
             if QtGui.QColor(img0.pixel(x, y)).lightness() > 200)
box.close()
check('g5', "[gate 5] the mark images are referenced as url(\"…\") with "
      "the path quoted; in a folder with a space and parentheses the tick "
      "is still drawn on a checked box",
      re.findall(r'url\("([^"]+)"\)', marks_odd) == [
          odd.replace(os.sep, '/') + '/checked.png',
          odd.replace(os.sep, '/') + '/partial.png']
      and rg.qss_url('C:\\a b\\x"y.png') == 'url("C:/a b/x\\"y.png")'
      and light > light0 and light >= 6, repr((light, light0, marks_odd)))

# removed code stays removed [136-141]
er_src = open(er.__file__, encoding='utf-8').read()
check('136', "[136-141] removed: tray signals and layout, the re-detect "
      "modal, the tray glyph constants, fixed physiology scales, the "
      "Wonambi peak-frequency read; the footer is one line",
      not any(hasattr(rg.ChannelQCWidget, n) for n in (
          'requestDrop', 'unmarkArtefact', 'removeFromRedetect',
          'requestBuildRedetect', 'verdictChanged', '_rebuild_tray'))
      and not any(hasattr(rg, n) for n in (
          'FlowLayout', '_STATE_ART_GLYPH', '_STATE_RD_GLYPH',
          '_STATE_ART_COLOR', '_STATUS_TEXT'))
      and not any(hasattr(rg.EventReviewGUI, n) for n in (
          'open_redetect_modal', '_build_redetect_request',
          '_qc_unmark_artefact', '_qc_remove_redetect', '_drop_channel'))
      and all(len(v) == 2 for v in rg.PHYSIO_ROWS.values())
      and "ev.get('peak_freq')" not in er_src
      and 'first-difference' not in er_src and 'first-difference' not in src
      and '\n' not in er.footer_text(3.5, 2.0, 'spindle'))

say("\n" + "=" * 78)
check('settings', "the real review-GUI preferences file was not "
      "touched", *gui_settings_guard.untouched())
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
