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

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from PyQt5 import QtCore, QtWidgets                           # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
from PyQt5.QtTest import QTest                                # noqa: E402

TMP = tempfile.mkdtemp(prefix='tw_population_')
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.IniFormat,
                         QtCore.QSettings.UserScope, TMP)

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
counts_before = win.qc_widget.counts_lbl.text()
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
check('10', "[10] amplitude flag, HARD/SOFT counts and Queue all HARD "
      "unchanged", flag_after == flag_before
      and win.qc_widget.counts_lbl.text() == counts_before
      and win.qc_widget.btn_queue_hard.text() == hard_btn_before,
      repr((counts_before, hard_btn_before)))
check('10b', "Outlier filter offers checks: hard / checks: soft",
      [win.qc_widget.flag_combo.itemText(i)
       for i in range(win.qc_widget.flag_combo.count())][-2:]
      == ['checks: hard', 'checks: soft'])
win.qc_widget.flag_combo.setCurrentText('checks: hard')
check('10c', "checks: hard filter keeps E7 only",
      win.qc_widget.model.rowCount() == 1
      and win.qc_widget.model.channel_at(0) == 'E7')
win.qc_widget.flag_combo.setCurrentText('any')
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
say("\n== 13. Dock line")
win.on_qc_channel_selected('E7')
line = win.detail_dock_w.check_line_text()
p7 = f"{qc.loc['E7', 'pct_off_band']:.0f}"
check('13a', "flagged channel: exact dock line",
      line == f"E7: {p7} % of spindles off-band", repr(line))
check('13b', "the hint line is shown under a flagged line",
      win.detail_dock_w.checks_hint.isVisibleTo(win.detail_dock_w)
      and win.detail_dock_w.checks_hint.text().startswith(
          'Most events failing? Drop channel.'))
win.on_qc_channel_selected('E1')
line = win.detail_dock_w.check_line_text()
check('13c', "unflagged channel: in line with the rest",
      line == 'E1: event checks in line with the rest of the montage',
      repr(line))
items = [win.detail_dock_w.topo_combo.itemText(i)
         for i in range(win.detail_dock_w.topo_combo.count())]
check('13d', "topography offers the five checks for spindles",
      items[3:] == [v[1] for v in er.CHECK_COLUMNS.values()], repr(items))

# ===================================================================== 14
say("\n== 14. Off-band link into the Epochs tab")
win.show()
win.activateWindow()
app.processEvents()
win.on_qc_channel_selected('E7')
win.detail_dock_w._on_check_link('pct_off_band')
app.processEvents()
ep = win.epochs_panel
chip = ep.chip_text()
n_off = int(np.sum(arrays['E7']['in_band'] == 0))
check('14a', "link drills E7 and shows the chip",
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
    s = float(ep._df.loc[ep._df['uuid'] == other, 'start_time'].iloc[0])
    if abs(it.getRegion()[0] - s) < 1e-9:
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
      and not any('low prominence' in t for t in items), repr(items))
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
check('12c', "cells read — and the dock line says not recorded",
      col_cell(win, 'E1', 'pct_off_band') == '—'
      and (win.on_qc_channel_selected('E1') or True)
      and win.detail_dock_w.check_line_text()
      == 'E1: event checks not recorded for this run',
      repr(win.detail_dock_w.check_line_text()))
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
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
