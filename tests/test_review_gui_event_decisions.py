#!/usr/bin/env python3
"""Headless checks: per-event bands, selection and decisions in the review GUI
(4.6.0 plan, Phase 2 items D1 and D2).

1. Data: ``QC_EVENT_COLS`` (montage-wide) stays the lean nine columns;
   ``QC_DRILL_COLS`` (one drilled channel) adds uuid / duration / run_id /
   method / epoch_stage; ``get_events`` drops columns an older database lacks;
   ``EventDatabase`` no longer adds review columns to ``events``;
   ``get_run_info`` parses ``params_json``; ``get_reviews_for``,
   ``get_events(reviewed_only / unreviewed_only)``, ``get_review_stats`` and
   ``export_reviewed_events`` read ``event_reviews``.
2. ``EpochsPanel``: every event in the window gets a band on both traces,
   outliers keep their red, reviewed events take their decision colour.
3. Selection: a click at an event's time selects its uuid
   (``eventSelected``); a click within 0.25 s of an event picks it, a click
   far from every event picks nothing; selecting does not move the artefact
   brush.
4. Keys: ``]`` selects the next unreviewed event and skips a reviewed one,
   paging the epoch when needed; ``[`` goes back; ``A`` emits
   ``decisionRequested('accept')``.
5. Main window: drilling passes the review map, a selection is named in the
   status bar, a decision is stored only when the library can store it.

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_review_gui_event_decisions.py
"""
import os
import sqlite3
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pandas as pd                                           # noqa: E402
from PyQt5 import QtCore, QtWidgets                           # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
from PyQt5.QtTest import QTest                                # noqa: E402

import frontend.eeg_review_gui as rg                          # noqa: E402

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
TMP = tempfile.mkdtemp(prefix='tw_decisions_')

# Cz spindles: five in epoch 0 [0, 30), three in epoch 1, one in epoch 3.
# 'u-out' is a 900 µV amplitude outlier.
STARTS = [2.0, 7.0, 12.0, 18.0, 25.0, 33.0, 41.0, 52.0, 95.0]
UUIDS = ['u-a', 'u-b', 'u-c', 'u-out', 'u-e', 'u-f', 'u-g', 'u-h', 'u-i']
AMPS = [40.0, 42.0, 38.0, 900.0, 41.0, 39.0, 43.0, 40.0, 41.0]
RUN = 'run-moelle-9-12'
PARAMS = '{"duration_by_method": {"Moelle2011": [0.5, 3.0]}, "frequency": [9, 12]}'


def make_db(path, with_new_cols=True):
    con = sqlite3.connect(path)
    extra = ", run_id TEXT, epoch_stage TEXT" if with_new_cols else ""
    con.execute(f"""CREATE TABLE events (uuid TEXT PRIMARY KEY,
        event_type TEXT, channel TEXT, start_time REAL, end_time REAL,
        duration REAL, stage TEXT, method TEXT, freq_lower REAL,
        freq_upper REAL, min_amp REAL, max_amp REAL, peak2peak_amp REAL
        {extra})""")
    for u, s, a in zip(UUIDS, STARTS, AMPS):
        row = [u, 'spindle', 'Cz', s, s + 0.8, 0.8, 'NREM2', 'Moelle2011',
               9.0, 12.0, -a / 2, a, a]
        if with_new_cols:
            row += [RUN, 'NREM2']
        con.execute(f"INSERT INTO events VALUES ({','.join('?' * len(row))})",
                    row)
    if with_new_cols:
        con.execute("""CREATE TABLE detection_runs (run_id TEXT PRIMARY KEY,
            subject TEXT, event_type TEXT, method TEXT, params_json TEXT, stages TEXT,
            reject_types TEXT, reject_artifacts INTEGER,
            reject_arousals INTEGER, timestamp TEXT)""")
        con.execute("INSERT INTO detection_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (RUN, 'sub-test', 'spindle', 'Moelle2011', PARAMS, 'NREM2+NREM3',
                     'Artefact,Arousal', 1, 1, '2026-10-01T10:00:00'))
    con.commit()
    con.close()
    return path


def put_review(db, uuid, decision, reviewer='tester'):
    """Through the library when it can store reviews, else straight into a
    table of the planned schema (the GUI's no-op fallback writes nothing)."""
    if db.has_review_backend:
        return db.add_review(uuid, decision, reviewer=reviewer,
                             reason='artefact' if decision == 'reject' else None)
    db.conn.execute("""CREATE TABLE IF NOT EXISTS event_reviews (
        uuid TEXT, reviewer TEXT, run_id TEXT, subject TEXT,
        event_type TEXT, channel TEXT, start_time REAL,
        decision TEXT CHECK (decision IN ('accept','reject','unsure')),
        reason TEXT, comment TEXT, reviewed_at TEXT,
        turtlewave_version TEXT, PRIMARY KEY (uuid, reviewer))""")
    db.conn.execute(
        "INSERT OR REPLACE INTO event_reviews (uuid, reviewer, decision, "
        "reason, reviewed_at) VALUES (?,?,?,?,?)",
        (uuid, reviewer, decision, None, '2026-10-01T10:00:00'))
    db.conn.commit()
    return True


say("=" * 78)
say("Headless: event bands, selection and decisions")
say("=" * 78)

# ======================================================================= 1
say("\n== 1. Data layer")
LEAN = ['channel', 'start_time', 'end_time', 'stage', 'min_amp', 'max_amp',
        'peak2peak_amp', 'freq_lower', 'freq_upper']
check('1.0', "QC_EVENT_COLS (montage-wide) is the original nine columns",
      rg.QC_EVENT_COLS == LEAN, repr(rg.QC_EVENT_COLS))
for c in ('uuid', 'duration', 'run_id', 'method', 'epoch_stage'):
    check('1.1', f"QC_DRILL_COLS includes {c}", c in rg.QC_DRILL_COLS)
from turtlewave_hdEEG import dbwrite as _dw                   # noqa: E402
check('1.1b', "local fallback REVIEW_DECISIONS equals the library's",
      rg.REVIEW_DECISIONS == tuple(_dw.REVIEW_DECISIONS),
      repr((rg.REVIEW_DECISIONS, _dw.REVIEW_DECISIONS)))
check('1.1c', "local fallback REVIEW_REASONS equals the library's",
      rg.REVIEW_REASONS == tuple(_dw.REVIEW_REASONS),
      repr((rg.REVIEW_REASONS, _dw.REVIEW_REASONS)))
check('1.1d', "review_vocabulary() returns the library constants",
      rg.review_vocabulary() == (tuple(_dw.REVIEW_DECISIONS),
                                 tuple(_dw.REVIEW_REASONS)),
      repr(rg.review_vocabulary()))
say(f"  review backend in the library: "
    f"{rg._review_backend() is not None}")

db = rg.EventDatabase(make_db(os.path.join(TMP, 'new.db')))
ev_cols = db._table_columns('events')
check('1.2', "events is not altered (no reviewed/review_decision/reviewer)",
      not {'reviewed', 'review_decision', 'reviewer', 'review_timestamp',
           'review_comments'} & set(ev_cols), repr(ev_cols))
idx = [r[0] for r in db.conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index'")]
check('1.3', "no idx_reviewed / idx_review_decision index",
      'idx_reviewed' not in idx and 'idx_review_decision' not in idx, repr(idx))
df = db.get_events(event_type='spindle', channels=['Cz'],
                   columns=rg.QC_DRILL_COLS)
check('1.4', "drill fetch returns all 14 columns and 9 rows",
      list(df.columns) == rg.QC_DRILL_COLS and len(df) == 9,
      repr((list(df.columns), len(df))))
info = db.get_run_info(RUN)
check('1.5', "get_run_info parses params_json, keeps method and stages",
      info.get('method') == 'Moelle2011'
      and info.get('stages') == 'NREM2+NREM3'
      and info.get('params', {}).get('duration_by_method')
      == {'Moelle2011': [0.5, 3.0]}, repr(info))
check('1.6', "get_run_info: unknown run and None give {}",
      db.get_run_info('nope') == {} and db.get_run_info(None) == {})
check('1.7', "no reviews yet: reviewed_only empty, unreviewed_only all 9",
      len(db.get_events(reviewed_only=True)) == 0
      and len(db.get_events(unreviewed_only=True)) == 9
      and db.get_reviews_for(UUIDS) == {})

put_review(db, 'u-b', 'accept')
put_review(db, 'u-c', 'reject')
put_review(db, 'u-c', 'accept', reviewer='second')
rv = db.get_reviews_for(UUIDS, reviewer='tester')
check('1.8', "get_reviews_for(reviewer=) -> {uuid: (decision, reason, "
      "reviewer)}", set(rv) == {'u-b', 'u-c'}
      and rv['u-b'][0] == 'accept' and rv['u-c'][0] == 'reject'
      and rv['u-c'][2] == 'tester', repr(rv))
rv_any = db.get_reviews_for(UUIDS)
check('1.8b', "get_reviews_for(any reviewer): the latest write wins, even "
      "within one second", rv_any.get('u-c', (None,))[:3:2]
      == ('accept', 'second'), repr(rv_any))
check('1.9', "get_events reviewed_only 2 / unreviewed_only 7",
      len(db.get_events(reviewed_only=True)) == 2
      and len(db.get_events(unreviewed_only=True)) == 7)
st = db.get_review_stats()
check('1.10', "get_review_stats over event_reviews",
      st.get('total') == 9 and st.get('reviewed') == 2
      and st.get('accept_count') == 2 and st.get('reject_count') == 1,
      repr(st))
out_csv = os.path.join(TMP, 'reviewed.csv')
n = db.export_reviewed_events(out_csv)
exp = pd.read_csv(out_csv)
check('1.11', "export: one row per (event, reviewer) with review columns",
      n == 3 and len(exp) == 3
      and {'reviewer', 'review_decision', 'review_reason'} <= set(exp.columns),
      repr((n, list(exp.columns)[-5:])))

old = rg.EventDatabase(make_db(os.path.join(TMP, 'old.db'),
                               with_new_cols=False))
odf = old.get_events(event_type='spindle', columns=rg.QC_DRILL_COLS)
check('1.12', "older database: drill fetch drops run_id/epoch_stage, no error",
      'run_id' not in odf.columns and 'uuid' in odf.columns and len(odf) == 9,
      repr(list(odf.columns)))
check('1.13', "older database: no detection_runs -> get_run_info {}",
      old.get_run_info(RUN) == {})

# ======================================================================= 2
say("\n== 2. Bands")
panel = rg.EpochsPanel()
panel.resize(1400, 900)
panel.show()
app.processEvents()
selected, decisions = [], []
panel.eventSelected.connect(selected.append)
panel.decisionRequested.connect(decisions.append)
panel.set_channel('Cz', df, df, event_type='spindle', trec=120.0)
check('2.1', "drill opens on the outlier's epoch 0", panel._epoch == 0,
      repr(panel._epoch))


def bands(plot):
    return [it for p, it in panel._event_items if p is plot]


def band_brush(uuid, plot):
    s = float(df.loc[df['uuid'] == uuid, 'start_time'].iloc[0])
    for it in bands(plot):
        if abs(it.getRegion()[0] - s) < 1e-9:
            return it.brush.color().getRgb()
    return None


check('2.2', "five events in epoch 0 -> five bands on raw and five on "
      "filtered", len(bands(panel.raw_plot)) == 5
      and len(bands(panel.filt_plot)) == 5
      and all(it in panel.raw_plot.getPlotItem().items
              for it in bands(panel.raw_plot)),
      repr((len(bands(panel.raw_plot)), len(bands(panel.filt_plot)))))
check('2.3', "outlier band keeps the red (224, 83, 63)",
      band_brush('u-out', panel.raw_plot)[:3] == (224, 83, 63)
      and band_brush('u-out', panel.filt_plot)[:3] == (224, 83, 63),
      repr(band_brush('u-out', panel.raw_plot)))
check('2.4', "ordinary band is grey",
      band_brush('u-a', panel.raw_plot)[:3] == (136, 136, 136),
      repr(band_brush('u-a', panel.raw_plot)))
panel.set_reviews({'u-b': ('accept', None, 'tester')})
acc = rg.QtGui.QColor(rg.DECISION_COLOR['accept']).getRgb()[:3]
check('2.5', "reviewed band takes the decision colour; still 5 bands",
      band_brush('u-b', panel.raw_plot)[:3] == acc
      and len(bands(panel.raw_plot)) == 5,
      repr(band_brush('u-b', panel.raw_plot)))
tick_cols = {it.opts['brush'] for it in panel._ticker_items}
check('2.6', "ticker draws an accept-coloured bar for the reviewed event",
      rg.DECISION_COLOR['accept'] in tick_cols, repr(tick_cols))

# an event straddling from epoch 0 into epoch 1 gets a band AND a ticker bar
sdf = pd.DataFrame({'channel': 'Cz', 'uuid': ['s-1', 's-2'],
                    'start_time': [29.6, 40.0], 'end_time': [30.6, 40.8],
                    'max_amp': [40.0, 41.0]})
spanel = rg.EpochsPanel()
spanel.set_channel('Cz', sdf, sdf, event_type='spindle', trec=60.0)
spanel._goto_epoch(1)
s_bands = [it for p, it in spanel._event_items if p is spanel.raw_plot]
s_bars = sum(len(it.opts['x']) for it in spanel._ticker_items)
s_x = sorted(float(x) for it in spanel._ticker_items for x in it.opts['x'])
check('2.7', "straddling event: band and ticker agree (2 bands, 2 bars, the "
      "straddler's bar pinned inside the window)",
      len(s_bands) == 2 and s_bars == 2 and 30.0 < s_x[0] < 30.5,
      repr((len(s_bands), s_bars, s_x)))
spanel.close()

# ======================================================================= 3
say("\n== 3. Click selection")


class FakeClick:
    def __init__(self, plot, x):
        vb = plot.getPlotItem().vb
        y = sum(vb.viewRange()[1]) / 2.0
        self._pos = vb.mapViewToScene(QtCore.QPointF(x, y))

    def button(self):
        return Qt.LeftButton

    def scenePos(self):
        return self._pos


region_before = tuple(panel.region.getRegion())
panel._on_trace_click(FakeClick(panel.raw_plot, 12.4), panel.raw_plot)
check('3.1', "click at 12.4 s on the raw trace selects u-c",
      selected[-1:] == ['u-c'] and panel._selected_uuid == 'u-c',
      repr(selected))
check('3.2', "selecting does not move the artefact brush",
      tuple(panel.region.getRegion()) == region_before,
      repr((region_before, panel.region.getRegion())))
sel_pen = [it for it in bands(panel.raw_plot)
           if abs(it.getRegion()[0] - 12.0) < 1e-9][0]
check('3.3', "selected band: accent edge on top (z 8)",
      sel_pen.zValue() == 8
      and sel_pen.lines[0].pen.color().name() == rg.THEME['accent'].lower(),
      repr((sel_pen.zValue(), sel_pen.lines[0].pen.color().name())))
panel._on_trace_click(FakeClick(panel.ticker, 18.95), panel.ticker)
check('3.4', "click on the ticker 0.15 s after u-out's end selects it",
      selected[-1] == 'u-out', repr(selected))
n_sel = len(selected)
panel._on_trace_click(FakeClick(panel.raw_plot, 22.0), panel.raw_plot)
check('3.5', "click 3 s from every event selects nothing",
      len(selected) == n_sel and panel._selected_uuid == 'u-out',
      repr(selected))
check('3.6', "_event_at: containing event, nearest within 0.25 s, else None",
      panel._event_at(7.5) == 'u-b' and panel._event_at(1.8) == 'u-a'
      and panel._event_at(1.5) is None)

# ======================================================================= 4
say("\n== 4. Keys")
panel.set_reviews({'u-e': ('accept', None, 'tester')})
panel.select_event('u-out')
panel.setFocus()
QTest.keyClick(panel, Qt.Key_BracketRight)
check('4.1', "] from u-out skips reviewed u-e and pages to epoch 1 -> u-f",
      panel._selected_uuid == 'u-f' and panel._epoch == 1
      and selected[-1] == 'u-f', repr((panel._selected_uuid, panel._epoch)))
QTest.keyClick(panel, Qt.Key_BracketLeft)
check('4.2', "[ goes back to u-out (skipping u-e) on epoch 0",
      panel._selected_uuid == 'u-out' and panel._epoch == 0,
      repr((panel._selected_uuid, panel._epoch)))
QTest.keyClick(panel, Qt.Key_A)
check('4.3', "A emits decisionRequested('accept')",
      decisions == ['accept'], repr(decisions))
QTest.keyClick(panel, Qt.Key_R)
QTest.keyClick(panel, Qt.Key_U)
check('4.4', "R and U emit reject and unsure",
      decisions == ['accept', 'reject', 'unsure'], repr(decisions))
panel._goto_epoch(2)       # empty epoch, selection off screen
QTest.keyClick(panel, Qt.Key_BracketRight)
check('4.5', "] from an empty epoch selects the first unreviewed after it "
      "(u-i on epoch 3)", panel._selected_uuid == 'u-i' and panel._epoch == 3,
      repr((panel._selected_uuid, panel._epoch)))
QTest.keyClick(panel, Qt.Key_BracketRight)
check('4.6', "] past the last unreviewed event stays put",
      panel._selected_uuid == 'u-i', repr(panel._selected_uuid))
panel.set_channel('Cz', df, df, event_type='spindle', trec=120.0)
n_dec = len(decisions)
QTest.keyClick(panel, Qt.Key_A)
check('4.7', "a new drill clears the selection; A then emits nothing",
      panel._selected_uuid is None and len(decisions) == n_dec)

# ======================================================================= 5
say("\n== 5. Main window wiring")
win = rg.EventReviewGUI()
win.db = db
win.qc_widget.evt_combo.blockSignals(True)
win.qc_widget.evt_combo.setCurrentText('spindle')
win.qc_widget.evt_combo.blockSignals(False)
# montage-wide QC frame: the lean columns only, no uuid
win._qc_events_df = db.get_events(event_type='spindle',
                                  columns=rg.QC_EVENT_COLS)
check('5.0', "montage-wide QC frame carries no uuid",
      'uuid' not in win._qc_events_df.columns)
win.on_qc_drill('Cz', switch_tab=False)
dsl = win.epochs_panel._df
check('5.0b', "drill fetches Cz's slice with the identity columns",
      dsl is not None and len(dsl) == 9
      and {'uuid', 'run_id', 'epoch_stage'} <= set(dsl.columns)
      and list(win.epochs_panel._ev['uuid']) == UUIDS,
      repr(None if dsl is None else list(dsl.columns)))
check('5.1', "drill passes the stored reviews to the panel",
      set(win.epochs_panel._reviews) == {'u-b', 'u-c'},
      repr(win.epochs_panel._reviews))
win.epochs_panel.select_event('u-a')
msg = win.status_bar.currentMessage()
evt_word = str(win.epochs_panel._event_type).replace('_', ' ')
check('5.2', "selection named in the status bar",
      win.selected_event_uuid == 'u-a'
      and msg == f"Selected {evt_word} on Cz at 00:00:02 (1 of 9 on channel)",
      repr(msg))
win.show()
win.activateWindow()
app.processEvents()
win.tabs.setCurrentIndex(0)
win.qc_widget.setFocus()
app.processEvents()
win.on_qc_drill('Cz', switch_tab=True)
app.processEvents()
fw = QtWidgets.QApplication.focusWidget()
check('5.2b', "the panel has keyboard focus after a drill (no click needed)",
      fw is win.epochs_panel, repr(fw))
win.epochs_panel.select_event('u-a')
n_rows = db.conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
win.reviewer_name = ''
win._on_decision_requested('accept')
msg = win.status_bar.currentMessage()
n_after = db.conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
check('5.3', "no reviewer name: nothing written, status asks for a name",
      n_after == n_rows and 'u-a' not in db.get_reviews_for(['u-a'])
      and msg == "Decision not saved: set a reviewer name first "
                 "(Review ▸ Reviewer name…)", repr((n_rows, n_after, msg)))
check('5.3b', "Review menu carries 'Reviewer name…'",
      win.act_reviewer_name.text() == 'Reviewer name…')
win.set_reviewer_name('TK')
win._on_decision_requested('accept')
rows = db.conn.execute("SELECT reviewer, decision FROM event_reviews "
                       "WHERE uuid = 'u-a'").fetchall()
if db.has_review_backend:
    check('5.3c', "after set_reviewer_name('TK'): one row under 'TK', band "
          "tinted", rows == [('TK', 'accept')]
          and 'u-a' in win.epochs_panel._reviews, repr(rows))
else:
    check('5.3c', "library without event_reviews: nothing stored",
          rows == [], repr(rows))

qc_frame = win._qc_events_df
win.db = None
win.on_qc_drill('Cz', switch_tab=False)
check('5.4', "no database: drill falls back to the QC frame; nothing to "
      "select", len(win.epochs_panel._df) == len(qc_frame)
      and not win.epochs_panel.select_event('u-a'),
      repr(len(win.epochs_panel._df)))

for w in (panel,):
    w.close()
if win.background_loader is not None:
    win.background_loader.stop()
win.close()

say("\n" + "=" * 78)
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
