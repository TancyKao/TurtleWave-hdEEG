#!/usr/bin/env python3
"""Headless checks: the review GUI on variable-length epochs (plan section D).

Cut recordings are staged as exact epochs 1-30 s long, so the review GUI
must take every staging quantity from the annotation file's own epoch table
instead of ``index * 30``. This builds two Wonambi XMLs, one with mixed
1/2/30 s epochs and one uniform 30 s file, and checks:

1. ``EpochTable``: ``index_at`` / ``span`` / ``snap`` / ``count_in`` on
   variable epochs, and the fallback that reads Wonambi epoch dicts.
2. ``_compute_epoch_outliers`` keys events by epoch id, a 1 s epoch included.
3. The timeline hypnogram draws one segment per epoch whose width equals the
   epoch's interval.
4. ``EpochsPanel.set_channel(epochs=...)`` then ``_goto_epoch(index_at(t))``
   on a 1 s epoch shows a 1 s window; strip bars, shift-drag snapping and
   the detail dock's marked list follow the table.
5. Main window: scored minutes are the sum of epoch durations, recording
   seconds are ``last_second``, the global worst-event jump lands on the
   right epoch.
6. A uniform 30 s XML behaves exactly as the old fixed grid.
7. No annotations: a synthetic 30 s grid.

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_review_gui_variable_epochs.py
"""
import os
import sys
import tempfile
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import pyqtgraph as pg                                        # noqa: E402
from PyQt5 import QtWidgets                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()   # before any frontend import
import frontend.eeg_review_gui as rg                          # noqa: E402
from turtlewave_hdEEG import CustomAnnotations                # noqa: E402

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
TMP = tempfile.mkdtemp(prefix='tw_varepochs_')

# (start, end, stage): 30 s, 30 s, a 1 s sliver, 30 s, a 2 s piece, 30 s
VAR_EPOCHS = [(0, 30, 'Wake'), (30, 60, 'NREM2'), (60, 61, 'NREM2'),
              (61, 91, 'NREM3'), (91, 93, 'NREM1'), (93, 123, 'REM')]
VAR_LAST_SECOND = 123
UNI_STAGES = ['Wake', 'NREM1', 'NREM2', 'NREM2', 'NREM3',
              'NREM3', 'REM', 'NREM2', 'Wake', 'NREM2']


def write_xml(path, epochs, last_second):
    """A Wonambi annotation XML holding exactly ``epochs``."""
    rows = ''.join(
        f"<epoch><epoch_start>{a}</epoch_start><epoch_end>{b}</epoch_end>"
        f"<stage>{s}</stage><quality>Good</quality></epoch>"
        for a, b, s in epochs)
    xml = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<annotations version="5"><dataset><filename>x.set</filename>'
        "<path>x.set</path><start_time>2022-09-12T22:28:52</start_time>"
        f"<first_second>0</first_second><last_second>{last_second}"
        "</last_second></dataset>"
        "<rater name='tester' created='2022-09-12T22:28:52' "
        "modified='2022-09-12T22:28:52'><bookmarks/><events/>"
        f"<stages>{rows}</stages><cycles/></rater></annotations>")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(xml)
    return path


var_xml = write_xml(os.path.join(TMP, 'var.xml'), VAR_EPOCHS, VAR_LAST_SECOND)
uni_epochs = [(30 * i, 30 * i + 30, s) for i, s in enumerate(UNI_STAGES)]
uni_xml = write_xml(os.path.join(TMP, 'uni.xml'), uni_epochs, 300)
var_ann = CustomAnnotations(var_xml)
uni_ann = CustomAnnotations(uni_xml)

say("=" * 78)
say("Headless: review GUI on variable-length epochs")
say("=" * 78)

# ======================================================================= 1
say("\n== 1. EpochTable")
tb = rg.EpochTable.from_annotations(var_ann)
check('1.1', "six epochs read from the XML",
      tb is not None and len(tb) == 6
      and list(tb.durations) == [30, 30, 1, 30, 2, 30],
      repr(None if tb is None else list(tb.durations)))
check('1.2', "index_at: 60.5 -> 2, 61 -> 3, 92 -> 4, 5000 -> 5, -3 -> 0",
      [tb.index_at(t) for t in (60.5, 61, 92, 5000, -3)] == [2, 3, 4, 5, 0],
      repr([tb.index_at(t) for t in (60.5, 61, 92, 5000, -3)]))
check('1.3', "span(2) is the 1 s epoch", tb.span(2) == (60.0, 61.0),
      repr(tb.span(2)))
check('1.4', "snap(60.3, 62) widens to (60, 91)", tb.snap(60.3, 62) == (60.0, 91.0),
      repr(tb.snap(60.3, 62)))
check('1.5', "count_in(60, 91) == 2 whole epochs", tb.count_in(60, 91) == 2,
      repr(tb.count_in(60, 91)))
check('1.6', "not uniform", not tb.is_uniform(), '')
fallback = types.SimpleNamespace(
    epochs=[{'start': a, 'end': b, 'stage': s, 'quality': 'Good'}
            for a, b, s in VAR_EPOCHS],
    last_second=VAR_LAST_SECOND)
fb = rg.EpochTable.from_annotations(fallback)
check('1.7', "fallback from Wonambi epoch dicts (no get_stage_intervals)",
      fb is not None and list(fb.starts) == list(tb.starts)
      and fb.stages == tb.stages, repr(None if fb is None else fb.stages))
check('1.8', "annotation_recording_seconds: library and fallback both 123",
      rg.annotation_recording_seconds(var_ann) == 123.0
      and rg.annotation_recording_seconds(fallback) == 123.0,
      repr((rg.annotation_recording_seconds(var_ann),
            rg.annotation_recording_seconds(fallback))))

# ======================================================================= 2
say("\n== 2. Outliers keyed by epoch id")
rng = np.random.default_rng(0)
starts = [5, 12, 20, 33, 45, 55, 60.2, 61.5, 70, 80, 92.0, 100, 110, 118]
amps = list(40 + rng.normal(0, 3, len(starts)))
amps[starts.index(60.2)] = 900.0          # the outlier sits in the 1 s epoch
events = pd.DataFrame({'channel': 'Cz', 'start_time': starts,
                       'end_time': [s + 0.5 for s in starts],
                       'max_amp': amps, 'peak2peak_amp': amps})
agg = rg._compute_epoch_outliers(events, epochs=tb, amp_col='max_amp')
row2 = agg[agg['idx'] == 2]
check('2.1', "event at 60.2 s lands in epoch 2 (the 1 s epoch), flagged",
      len(row2) == 1 and int(row2['n_events'].iloc[0]) == 1
      and int(row2['n_outliers'].iloc[0]) == 1
      and float(row2['t0'].iloc[0]) == 60.0
      and row2['stage'].iloc[0] == 'NREM2', repr(row2.to_dict('records')))
per_ep = dict(zip(agg['idx'], agg['n_events']))
check('2.2', "per-epoch counts follow the table (61.5 s is epoch 3, 92 s is 4)",
      per_ep == {0: 3, 1: 3, 2: 1, 3: 3, 4: 1, 5: 3}, repr(per_ep))
legacy = rg._compute_epoch_outliers(events, amp_col='max_amp')
check('2.3', "(contrast) the old 30 s grid lumps 60.2, 61.5, 70 and 80 s "
      "into one epoch", int(legacy.loc[legacy['idx'] == 2, 'n_events'].iloc[0]) == 4,
      repr(legacy[['idx', 'n_events']].to_dict('records')))

# ======================================================================= 3
say("\n== 3. Timeline hypnogram segment widths")
tl = rg.TimelineWidget()
tl.plot_timeline(events, current_index=0, annotations=var_ann)
segs = []
for item in tl.getPlotItem().listDataItems():
    x = item.xData
    if x is None:
        continue
    x = np.asarray(x, dtype=float)
    for k in range(0, len(x) - 1, 3):
        if np.isfinite(x[k]) and np.isfinite(x[k + 1]):
            segs.append((float(x[k]), float(x[k + 1])))
segs.sort()
check('3.1', "one segment per epoch at its true [start, end]",
      segs == [(float(a), float(b)) for a, b, _ in VAR_EPOCHS], repr(segs))
check('3.2', "segment widths include 1, 2 and 30 s",
      sorted({b - a for a, b in segs}) == [1.0, 2.0, 30.0],
      repr(sorted({b - a for a, b in segs})))
check('3.3', "x range ends at last_second",
      abs(tl.getPlotItem().vb.viewRange()[0][1] - 123.0) < 1e-6,
      repr(tl.getPlotItem().vb.viewRange()[0]))

# ======================================================================= 4
say("\n== 4. EpochsPanel on the exact epochs")
panel = rg.EpochsPanel()
panel.set_channel('Cz', events, events, event_type='spindle',
                  epochs=var_ann.get_stage_intervals(), trec=123.0)
i1 = panel.index_at(60.5)
panel._goto_epoch(i1)
xr = panel.raw_plot.getPlotItem().vb.viewRange()[0]
check('4.1', "_goto_epoch(index_at(60.5)) shows a 1 s window [60, 61]",
      i1 == 2 and abs(xr[0] - 60) < 1e-9 and abs(xr[1] - 61) < 1e-9, repr(xr))
check('4.2', "header names the epoch, its 1 s length and stage",
      panel.epoch_lbl.text().startswith('Epoch 3/6 · 00:01:00–00:01:01 (1 s) · NREM2'),
      repr(panel.epoch_lbl.text()))
r0, r1 = panel.region.getRegion()
check('4.3', "brush sits inside the 1 s window", 60 < r0 < r1 < 61,
      repr((r0, r1)))
check('4.4', "strip marker covers the epoch",
      tuple(panel._ov_marker.getRegion()) == (60.0, 61.0),
      repr(panel._ov_marker.getRegion()))
bars = [it for it in panel.plot.getPlotItem().items
        if isinstance(it, pg.BarGraphItem)]
w = np.asarray(bars[0].opts['width'], dtype=float) if bars else np.array([])
x = np.asarray(bars[0].opts['x'], dtype=float) if bars else np.array([])
check('4.5', "strip bars are 95 % of each epoch, centred on it",
      len(bars) == 2 and np.allclose(sorted(w), sorted(0.95 * np.array(
          [30, 30, 1, 30, 2, 30]))) and 60.5 in list(x), repr((list(w), list(x))))
panel._on_shift_drag(*panel.snap(60.3, 62), True)
check('4.6', "shift-drag snaps to epoch edges and counts epochs",
      tuple(panel._strip_range.getRegion()) == (60.0, 91.0)
      and panel.mark_n_btn.text() == 'Exclude 2 epochs…',
      repr((panel._strip_range.getRegion(), panel.mark_n_btn.text())))
check('4.7', "the strip view box snaps with the panel's table",
      panel._strip_vb._snapped(91.5, 92.5) == (91.0, 93.0),
      repr(panel._strip_vb._snapped(91.5, 92.5)))
panel._next()
check('4.8', "Next from the 1 s epoch goes to the 30 s epoch at 61 s",
      panel.raw_plot.getPlotItem().vb.viewRange()[0] == [61.0, 91.0],
      repr(panel.raw_plot.getPlotItem().vb.viewRange()[0]))
panel._set_ranges([{'id': 7, 'start_time': 60.0, 'end_time': 91.0}])
check('4.9', "marked-range row counts epochs from the table",
      panel.ranges_list.item(0).text().endswith('(2 ep)'),
      repr(panel.ranges_list.item(0).text()))

dock = rg.ChannelDetailDock()
dock.set_epoch_table(tb)
jumps = []
dock.gotoEpochRequested.connect(jumps.append)
dock.set_marked([{'id': 1, 'start_time': 60.0, 'end_time': 61.0}])
btn = dock._marked_layout.itemAt(0).widget().layout().itemAt(0).widget()
btn.click()
check('4.10', "detail dock: a 1 s mark reads '(1 ep)' and jumps to epoch 2",
      btn.text().endswith('(1 ep)') and jumps == [2], repr((btn.text(), jumps)))
dock.update_channel('Cz', events, {'_epochs': tb, '_event_type': 'spindle'})
worst = [dock.worst_list.item(i).data(0x0100)
         for i in range(dock.worst_list.count())]
check('4.11', "worst-epochs list keyed by epoch id (epoch 2 first)",
      worst[:1] == [2], repr(worst))

check('4.12', "header length text: 1 s, 2.5 s, 5 min 12 s, 2 min",
      [rg._epoch_len_text(x) for x in (1, 2.5, 312, 120)]
      == ['1 s', '2.5 s', '5 min 12 s', '2 min'],
      repr([rg._epoch_len_text(x) for x in (1, 2.5, 312, 120)]))

# ======================================================================= 5
say("\n== 5. Main window")
win = rg.EventReviewGUI()
win.annotations = var_ann
scored = sum(b - a for a, b, s in VAR_EPOCHS if s != 'Wake')
check('5.1', "scored minutes = sum of scored epoch durations (93 s)",
      abs((win._scored_minutes() or 0) - scored / 60.0) < 1e-9
      and scored == 93, repr((win._scored_minutes(), scored)))
check('5.2', "recording seconds = last_second (123), not 6 x 30",
      win._recording_seconds() == 123.0, repr(win._recording_seconds()))
win._refresh_toolbar_state()
check('5.3', "toolbar reads rec from last_second (2 min) and TST from "
      "durations (1 min), not 3 min / 2.5 min on a 30 s grid",
      'rec 0h 2m' in win.lbl_duration.text()
      and 'TST 0h 1m' in win.lbl_duration.text(),
      repr(win.lbl_duration.text()))
ev_nostage = events.assign(stage='')
rows = win._global_worst_rows(ev_nostage, 'spindle', limit=1)
check('5.4', "global worst rows are stage-tagged from the table",
      rows and rows[0]['start_time'] == 60.2 and rows[0]['stage'] == 'NREM2',
      repr(rows))
win._qc_events_df = events
win._on_global_worst_goto('Cz', 60.2)
check('5.5', "global worst jump lands on the 1 s epoch",
      win.epochs_panel._epoch == 2
      and win.epochs_panel.raw_plot.getPlotItem().vb.viewRange()[0] == [60.0, 61.0],
      repr((win.epochs_panel._epoch,
            win.epochs_panel.raw_plot.getPlotItem().vb.viewRange()[0])))

# ======================================================================= 6
say("\n== 6. Uniform 30 s XML behaves as before")
ut = rg.EpochTable.from_annotations(uni_ann)
ts = np.linspace(0, 299.9, 400)
check('6.1', "uniform table; index_at(t) == int(t // 30) everywhere",
      ut.is_uniform() and all(ut.index_at(t) == int(t // 30) for t in ts), '')
check('6.2', "span(i) == (30 i, 30 i + 30)",
      all(ut.span(i) == (30.0 * i, 30.0 * i + 30) for i in range(10)), '')
uev = pd.DataFrame({'channel': 'Cz', 'start_time': rng.uniform(0, 299, 80),
                    'max_amp': rng.normal(40, 5, 80)})
uev.loc[5, 'max_amp'] = 999.0
new = rg._compute_epoch_outliers(uev, epochs=ut)
old = rg._compute_epoch_outliers(uev, hypno=UNI_STAGES)
check('6.3', "outlier frame identical to the old fixed-grid path",
      new.reset_index(drop=True).equals(old.reset_index(drop=True)),
      repr((new.head(3).to_dict('records'), old.head(3).to_dict('records'))))
upanel = rg.EpochsPanel()
upanel.set_channel('Cz', uev, uev, epochs=ut, trec=300.0)
upanel._goto_epoch(upanel.index_at(75))
check('6.4', "30 s window, brush at t0+13..t0+17, no length in the header",
      upanel.raw_plot.getPlotItem().vb.viewRange()[0] == [60.0, 90.0]
      and tuple(upanel.region.getRegion()) == (73.0, 77.0)
      and '(30 s)' not in upanel.epoch_lbl.text()
      and upanel.epoch_lbl.text().startswith('Epoch 3/10 · 00:01:00–00:01:30 · NREM2'),
      repr((upanel.region.getRegion(), upanel.epoch_lbl.text())))
win.annotations = uni_ann
n_scored = sum(s != 'Wake' for s in UNI_STAGES)
check('6.5', "scored minutes = n_scored x 30 s; recording = 300 s",
      abs(win._scored_minutes() - n_scored * 0.5) < 1e-9
      and win._recording_seconds() == 300.0,
      repr((win._scored_minutes(), win._recording_seconds())))
check('6.6', "legacy hypno= argument still builds a 30 s grid",
      rg._as_epoch_table(hypno=UNI_STAGES).span(4) == (120.0, 150.0), '')

# ======================================================================= 7
say("\n== 7. No annotations: synthetic 30 s grid")
npanel = rg.EpochsPanel()
npanel.set_channel('Cz', events, events, trec=100.0)
npanel._goto_epoch(99)
check('7.1', "4 synthetic epochs over 100 s; the last is [90, 120]",
      npanel._n_epochs() == 4
      and npanel.raw_plot.getPlotItem().vb.viewRange()[0] == [90.0, 120.0],
      repr((npanel._n_epochs(),
            npanel.raw_plot.getPlotItem().vb.viewRange()[0])))
win.annotations = None
check('7.2', "main window without annotations: no table, no scored minutes",
      win._epoch_table() is None and win._scored_minutes() is None, '')

for w in (tl, panel, dock, upanel, npanel):
    w.close()
if win.background_loader is not None:
    win.background_loader.stop()
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
