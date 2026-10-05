#!/usr/bin/env python3
"""Headless checks: review-sample mode and the precision report (4.6.0).

Acceptance criteria 16-24 and 45-51 of
``_scratch/design/event-decision-spec.md`` (revision 2, section 13), with the
coordinator's amendments: presentation order is the library's
(``sample_progress()['next_uuids']``), the stratum / flags row appears only
after a decision, and there is no ``event_reviews.blind`` column yet (the
"stored with blind = 0" half of 23 cannot be checked).

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_review_gui_sample.py
"""
import csv
import math
import os
import re
import shutil
import sqlite3
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
from PyQt5 import QtCore, QtWidgets                           # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402
from PyQt5.QtTest import QTest                                # noqa: E402

TMP = tempfile.mkdtemp(prefix='tw_sample_')
import gui_settings_guard                                    # noqa: E402
gui_settings_guard.isolate()     # before any frontend import

import frontend.eeg_review_gui as rg                          # noqa: E402
from frontend import sample_review as sr                      # noqa: E402
from frontend import event_review as er                       # noqa: E402
import frontend.review_sample_widgets as rsw                  # noqa: E402
import pandas as pd                                           # noqa: E402
from turtlewave_hdEEG import dbwrite, review_sampling as rs   # noqa: E402
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
rng = np.random.default_rng(11)
RUN = 'run-fixture'


def build(path, channels, n=80, figures=True, special=None, low_prom=0.3,
          gen=None):
    con = fx.open_schema(path)
    fx.add_run(con, RUN, figures=figures,
               version='4.6.0' if figures else '4.5.0')
    t0 = 0.0
    for i, ch in enumerate(channels):
        ob = (special or {}).get(ch, 0.06 + 0.01 * (i % 5))
        rows, _ = fx.make_rows(ch, n, RUN, gen or rng, off_band=ob,
                               low_prom=low_prom,
                               t0=t0, figures=figures)
        fx.insert_rows(con, rows)
    con.commit()
    con.close()


def window(path):
    win = rg.EventReviewGUI()
    win.db = rg.EventDatabase(path)
    win.qc_widget.evt_combo.blockSignals(True)
    win.qc_widget.evt_combo.setCurrentText('spindle')
    win.qc_widget.evt_combo.blockSignals(False)
    win._ask_reviewer_name = lambda prefill: ('TK', True)
    win.refresh_qc_dashboard()
    win.wait_population()
    win.show()
    win.activateWindow()
    app.processEvents()
    return win


def col_cell(win, ch, key, role=Qt.DisplayRole):
    m = win.qc_widget.model
    for r in range(m.rowCount()):
        if m.channel_at(r) == ch:
            return m.data(m.index(r, rg._QC_COL_INDEX[key]), role)
    return None


def press(win, key, mod=Qt.NoModifier):
    QTest.keyClick(win.epochs_panel, key, mod)
    app.processEvents()


say("=" * 78)
say("Headless: review sample and precision report")
say("=" * 78)

# ===================================================================== 16-24
CH10 = ['Fz', 'F3', 'Cz', 'C3', 'Pz', 'P3', 'T7', 'T8', 'O1', 'O2']
P1 = os.path.join(TMP, 'sample.db')
build(P1, CH10)
PRISTINE = os.path.join(TMP, 'pristine.db')
shutil.copy(P1, PRISTINE)

win = window(P1)
win.on_qc_drill('Fz', switch_tab=True)
app.processEvents()
bar = win.sample_bar
say("\n== 16-17. Bar with no sample; drawing")
check('16', "no sample: bar text and only 'Draw sample…'",
      bar.label.text() == 'REVIEW SAMPLE · No review sample for this run yet.'
      and bar.visible_buttons() == ['Draw sample…'],
      repr((bar.label.text(), bar.visible_buttons())))
check('16b', "bar tooltip says free-browsing decisions count",
      bar.label.toolTip() == sr.FREE_BROWSING_TIP)
seen = {}


def drive(dlg, size=120, seed=48213):
    seen['dlg'] = dlg
    seen['note'] = dlg.note.text()
    seen['existing'] = dlg.existing.text() if dlg.existing.isVisibleTo(dlg) \
        else ''
    seen['button'] = dlg.draw_btn.text()
    dlg.size_spin.setValue(size)
    dlg.seed_spin.setValue(seed)
    dlg.refresh_preview()
    seen['total_row'] = [dlg.cell(dlg.table.rowCount() - 1, j)
                         for j in range(4)]
    return QtWidgets.QDialog.Accepted


win._exec_dialog = drive
win._open_draw_dialog()
app.processEvents()
sid = win._sample['id']
n_rows = win.db.conn.execute(
    "SELECT COUNT(*) FROM review_samples WHERE sample_id = ?",
    (sid,)).fetchone()[0]
check('17a', "size 120, seed 48213 writes 120 review_samples rows",
      n_rows == 120, repr(n_rows))
check('17b', "the preview's total row matched (120 in sample, flagged "
      "counted) and the flag note was shown", seen['total_row'][2] == '120'
      and seen['total_row'][3] not in ('—', None)
      and seen['note'] == sr.FLAGGED_NOTE, repr(seen['total_row']))
# A population with few off-band events, so the library takes fewer flagged
# (F) than unflagged (U) events: the dialog's "of which flagged" must be F.
P_FU = os.path.join(TMP, 'sparse_flags.db')
# own generator so the shared one (and every later fixture) is unchanged
build(P_FU, CH10, special={ch: 0.02 for ch in CH10}, low_prom=0.02,
      gen=np.random.default_rng(7))
wfu = window(P_FU)
seen_fu = {}


def drive_fu(dlg):
    dlg.size_spin.setValue(120)
    dlg.seed_spin.setValue(48213)
    dlg.refresh_preview()
    seen_fu['total_row'] = [dlg.cell(dlg.table.rowCount() - 1, j)
                            for j in range(4)]
    seen_fu['subject'] = dlg.subject_lbl.text()
    return QtWidgets.QDialog.Rejected


wfu._exec_dialog = drive_fu
wfu.subject = 'WINDOW-SUBJECT'        # differs from the database's subject
wfu._open_draw_dialog()
app.processEvents()
pv17 = rs.preview_allocation(wfu.db.conn, scope=sr.scope_for_run(
    wfu.db.conn, RUN, 'spindle'), n_total=120, seed=48213)
nF = sum(c['parts'].get('F', {}).get('n', 0) for c in pv17['cells'])
nU = sum(c['parts'].get('U', {}).get('n', 0) for c in pv17['cells'])
check('17g', "unequal F and U: the total row's flagged count is the "
      "library's F, not U", nF != nU and seen_fu.get('total_row', [''] * 4)[3]
      == str(nF), repr((seen_fu.get('total_row'), nF, nU)))
db_subject = rs.prepare_population(wfu.db.conn, scope=sr.scope_for_run(
    wfu.db.conn, RUN, 'spindle'))['population']['subject']
check('17h', "the draw dialog names the database's subject (from the "
      "prepared population), not the window's", db_subject
      and db_subject != 'WINDOW-SUBJECT'
      and seen_fu.get('subject') == str(db_subject),
      repr((seen_fu.get('subject'), db_subject)))
wfu.close()
wfu.deleteLater()
win.raise_()
win.activateWindow()          # window-context shortcuts need the active window
app.processEvents()
check('17c', "status names the draw", win.status_bar.currentMessage()
      .startswith('Drew 120 events across 10 region × stage groups (seed '
                  '48213).') or win.status_bar.currentMessage()
      .startswith('Sample event'), repr(win.status_bar.currentMessage()))
con2 = sqlite3.connect(PRISTINE)
sid2 = rs.draw_review_sample(con2, scope=sr.scope_for_run(con2, RUN,
                                                          'spindle'),
                             n_total=120, seed=48213)
same = (sr.presentation_order(con2, sid2)
        == sr.presentation_order(win.db.conn, sid))
con2.close()
check('17d', "same seed on a copy of the database: same uuids, same order",
      same and sid2 == sid, repr((sid, sid2)))
lib_order = rs.sample_progress(win.db.conn, sid,
                               reviewer='TK')['next_uuids']
check('17f', "GUI presentation order is the library's next_uuids (a "
      "reversed list would not match)", win._sample['order'] == lib_order
      and list(reversed(win._sample['order'])) != lib_order
      and len(lib_order) == 120)

say("\n== 18-21. Sample mode")
order = win._sample['order']
ep = win.epochs_panel
evp = win.detail_dock_w.event_panel
check('18a', "starting selects the first sample event in presentation "
      "order", ep._selected_uuid == order[0]
      and ep._channel == win._sample['rows'][order[0]]['channel'],
      repr((ep._selected_uuid, order[0])))
press(win, Qt.Key_BracketRight)
r1 = win._sample['rows'][order[1]]
check('18b', "] selects the next undecided sample event, drilling its "
      "channel; the Channels table follows", ep._selected_uuid == order[1]
      and ep._channel == r1['channel']
      and win.qc_widget._current_channel() == r1['channel'],
      repr((ep._selected_uuid, ep._channel,
            win.qc_widget._current_channel())))
press(win, Qt.Key_BraceRight)
off = ep._selected_uuid
check('18c', "} selects the next event on the drilled channel, in the "
      "sample or not", off is not None and off != order[1]
      and ep._channel == r1['channel'], repr(off))
press(win, Qt.Key_BracketRight)
check('18d', "] after browsing with } returns to the undecided cursor event",
      ep._selected_uuid == order[1], repr(order.index(ep._selected_uuid)))
press(win, Qt.Key_BracketRight)
check('18e', "] while on that undecided event skips it deliberately",
      ep._selected_uuid == order[2], repr(order.index(ep._selected_uuid)))
press(win, Qt.Key_BracketLeft)
check('18h', "[ goes back to the previous undecided sample event",
      ep._selected_uuid == order[1], repr(order.index(ep._selected_uuid)))
win._exit_sample()
press(win, Qt.Key_BraceRight)                   # browse while out of sample
win._start_sample()
app.processEvents()
check('18i', "Exit then Resume returns to the undecided cursor event",
      ep._selected_uuid == order[1], repr(order.index(ep._selected_uuid)))
win._exit_sample()
win._start_sample()
app.processEvents()
check('18j', "… also when the selection never left it",
      ep._selected_uuid == order[1], repr(order.index(ep._selected_uuid)))
FOUR = ['signal_bg', 'duration', 'peak_freq', 'outlier']
srow0 = evp.sample_lbl.text()
check('20a', "[20, R4.0] before this reviewer's accept/reject the sample "
      "line reads 'Sample event i of N · region · stage' (no flags, no "
      "weight tooltip), under the header line; the rows are the four",
      evp.row_keys() == FOUR and evp.sample_lbl.isVisibleTo(evp)
      and re.match(r'^Sample event \d+ of 120 · [a-z-]+ · [A-Z0-9]+$',
                   srow0) is not None
      and not evp.sample_lbl.toolTip()
      and evp.event_line.isVisibleTo(evp)
      and evp.hidden_lbl.isVisibleTo(evp)
      and evp.hidden_lbl.text() == 'Labels hidden until you accept or '
                                   'reject this sample event.',
      repr((evp.row_keys(), srow0)))
check('75s', "[75] top-bar hint in sample mode", win.key_hint_lbl.text() ==
      'A accept · R reject · U unsure · ] [ sample · ? keys',
      repr(win.key_hint_lbl.text()))
check('78s', "[78] strip legend in sample mode ends with the blue ticks",
      ep.strip_legend.text().endswith(' · blue ticks = sample events'))
press(win, Qt.Key_A)
check('19a', "auto-advance: A selects the next undecided sample event",
      ep._selected_uuid == order[2], repr(ep._selected_uuid))
check('19b', "bar reads 1 of 120 in sample · 0 unsure",
      bar.label.text().startswith('REVIEW SAMPLE · 1 of 120 in sample · 0 '
                                  'unsure'), repr(bar.label.text()))
check('19c', "status names the next sample event",
      ' · next in sample: ' in win.status_bar.currentMessage(),
      repr(win.status_bar.currentMessage()))
ep.select_event(order[1]) if ep._channel == r1['channel'] else \
    win._goto_sample_event(order[1])
app.processEvents()
keys = evp.row_keys()
srow = evp.sample_lbl.text()
check('20b', "after the decision the sample line gains the flags and the "
      "weight tooltip", keys == FOUR and re.match(
          r'^Sample event 2 of 120 · [a-z-]+ · [A-Z0-9]+ · (not flagged|'
          r'flagged: .+)$', srow) is not None
      and evp.sample_lbl.toolTip().startswith('Sampling weight '),
      repr(srow))
check('20c', "is_shared is never shown", 'shared' not in srow.lower())
check('20d', "progress line in sample mode",
      evp.progress_lbl.text() == 'Progress  1 of 120 in sample · 1 accepted '
                                 '· 0 rejected · 0 unsure',
      repr(evp.progress_lbl.text()))
# browsing to a LATER sample event must not make ] or auto-advance skip
win._goto_sample_event(order[2])               # cursor = order[2]
far = order[6]
win._goto_sample_event(far, move_cursor=False)  # a click / } landing ahead
press(win, Qt.Key_BracketRight)
check('18f', "click on a later sample event, then ]: back to the undecided "
      "cursor event (order[2]), not past the click",
      ep._selected_uuid == order[2], repr(order.index(ep._selected_uuid)))
win._goto_sample_event(order[2])               # cursor back to order[2]
win._goto_sample_event(far, move_cursor=False)
press(win, Qt.Key_A)                            # decide the far event
check('18g', "deciding the clicked event advances from the cursor "
      "(order[2], still undecided), skipping nothing",
      ep._selected_uuid == order[2], repr(order.index(ep._selected_uuid)))
# outside the sample
outside = next(u for u in ep._ev['uuid'] if u not in win._sample['rows'])
ep.select_event(outside)
press(win, Qt.Key_A)
check('20e', "a decision outside the sample says it is not counted",
      win.status_bar.currentMessage() == sr.OUTSIDE_SAMPLE,
      repr(win.status_bar.currentMessage()))
# decide the rest: three unsure, the others accept
win._goto_sample_event(order[2])
n_unsure = 0
guard = 0
while win._sample_candidates() and guard < 400:
    guard += 1
    if n_unsure < 3:
        press(win, Qt.Key_U)
        press(win, Qt.Key_Return)
        n_unsure += 1
    else:
        press(win, Qt.Key_A)
check('21a', "deciding the last sample event shows the end message",
      evp.end_lbl.text() == 'All 120 sample events decided by TK (3 unsure).'
      and evp.end_box.isVisibleTo(evp), repr(evp.end_lbl.text()))
check('21b', "the Revisit button names the unsure count; bar reads done",
      evp.btn_revisit.text() == 'Revisit the 3 unsure'
      and evp.btn_revisit.isVisibleTo(evp)
      and bar.label.text() == 'REVIEW SAMPLE · 120 of 120 in sample · 3 '
                              'unsure · done', repr(bar.label.text()))
check('21c', "status says the sample is complete",
      win.status_bar.currentMessage()
      == 'Review sample complete: 120 of 120 decided by TK.',
      repr(win.status_bar.currentMessage()))
mine = sr.reviewer_labels(win.db.conn, sid)['TK']
unsure = {u for u, v in mine.items() if v[0] == 'unsure'}
evp.btn_revisit.click()
app.processEvents()
visited = {ep._selected_uuid}
for _ in range(4):
    press(win, Qt.Key_BracketRight)
    visited.add(ep._selected_uuid)
check('21f', "[81] an unsure sample event keeps its flags hidden (revisit "
      "stays unprimed)", re.match(
          r'^Sample event \d+ of 120 · [a-z-]+ · [A-Z0-9]+$',
          evp.sample_lbl.text()) is not None
      and not evp.sample_lbl.toolTip()
      and evp.hidden_lbl.isVisibleTo(evp), repr(evp.sample_lbl.text()))
check('21d', "after Revisit, ] visits only the unsure events",
      visited == unsure and bar.label.text() ==
      'REVIEW SAMPLE · revisiting 3 unsure', repr((len(visited),
                                                   bar.label.text())))
press(win, Qt.Key_A)
check('21e', "deciding an unsure one as accept removes it from the list",
      bar.label.text() == 'REVIEW SAMPLE · revisiting 2 unsure',
      repr(bar.label.text()))

say("\n== 22-24. Second rater, show others, filter")
win._ask_reviewer_name = lambda prefill: ('JS', True)
win._prompt_reviewer_name()
app.processEvents()
check('22a', "a second reviewer sees 0 of 120 in sample",
      bar.label.text().startswith('REVIEW SAMPLE · 0 of 120 in sample · 0 '
                                  'unsure'), repr(bar.label.text()))
check('22e', "the reviewer switch reset the sample cursor",
      win._sample_cursor is None, repr(win._sample_cursor))
press(win, Qt.Key_BracketRight)
check('22f', "JS's first ] selects position 0",
      ep._selected_uuid == order[0], repr(order.index(ep._selected_uuid)))
win._goto_sample_event(order[0])
app.processEvents()
check('22b', "TK's decisions do not tint, set Current or count as decided",
      ep._reviews == {} and evp.current_lbl.text() == 'Not reviewed'
      and win._sample_candidates() == set(order),
      repr((len(ep._reviews), evp.current_lbl.text())))
check('22c', "the Current line says another reviewer decided it, hidden",
      'Also decided by 1 other reviewer(s); hidden so your decisions stay '
      'independent.' in evp.current_sub.text(), repr(evp.current_sub.text()))
press(win, Qt.Key_A)
check('22d', "JS's own decision counts for JS: 1 of 120",
      bar.label.text().startswith('REVIEW SAMPLE · 1 of 120 in sample'),
      repr(bar.label.text()))
win._goto_sample_event(order[0])
check('23a', "Show other reviewers is unchecked (bar and menu) at launch",
      not bar.others_chk.isChecked() and not win.act_show_others.isChecked())
asked = []
win._confirm_show_others = lambda: (asked.append(1) or True)
bar.others_chk.setChecked(True)
app.processEvents()
check('23b', "turning it on asks once and syncs the menu",
      asked == [1] and win.act_show_others.isChecked()
      and 'TK: ' in evp.current_sub.text(), repr(evp.current_sub.text()))
bar.others_chk.setChecked(False)
win._exit_sample()
win.on_qc_drill('Fz', switch_tab=True)
win.detail_dock_w._check_channel = 'Fz'
win._on_check_link('Fz', 'pct_low_prom')
app.processEvents()
had_chip = ep.chip_text() != ''
win._start_sample()
app.processEvents()
check('24', "starting the sample removes the check filter and says so",
      had_chip and ep.chip_text() == ''
      and win.status_bar.currentMessage() == sr.FILTER_OFF,
      repr((had_chip, win.status_bar.currentMessage())))
check('16c', "with a sample, outside sample mode the bar offers Resume and "
      "Precision report", (win._exit_sample() or True)
      and bar.visible_buttons() == ['Resume sample', 'Precision report…']
      and bar.label.text().startswith('REVIEW SAMPLE · 120 events drawn '),
      repr((bar.visible_buttons(), bar.label.text())))
# a second draw keeps the first sample
win._ask_reviewer_name = lambda prefill: ('TK', True)
win.set_reviewer_name('TK')
win._exec_dialog = lambda dlg: drive(dlg, 120, 7)
win._open_draw_dialog()
app.processEvents()
kept = win.db.conn.execute(
    "SELECT COUNT(*) FROM review_samples WHERE sample_id = ?",
    (sid,)).fetchone()[0]
check('17e', "drawing again keeps the old sample; the dialog said so and "
      "offered 'Draw new sample'", kept == 120 and win._sample['id'] != sid
      and seen['button'] == 'Draw new sample'
      and seen['existing'].startswith('A sample of 120 was drawn on ')
      and 'decisions from JS (1) and TK (120)' in seen['existing'],
      repr((kept, seen['existing'])))
check('wording', "flag note and column tooltip state the library's flag "
      "definition; Show-others text promises no recording",
      'amplitude under 2× background or missing' in sr.FLAGGED_NOTE
      and 'duration floor does not flag' in sr.FLAGGED_NOTE
      and seen['dlg'].table.horizontalHeaderItem(3).toolTip()
      == sr.FLAG_DEFINITION
      and rg.SHOW_OTHERS_WARNING.endswith('This is not recorded '
                                          'automatically.')
      and 'recorded as not' not in rg.SHOW_OTHERS_WARNING
      and sr.stratum_text({'region': 'parietal', 'stage': 'NREM2',
                           'flagged': 1, 'cell': 'parietal|NREM2|F',
                           'flag_components': 'low_amp_ratio'})
      == 'parietal · NREM2 · flagged: amplitude under 2× background')
first_sid = sid
win._exec_dialog = lambda dlg: drive(dlg, 120, 48213)
win._open_draw_dialog()                       # reopens the first sample
app.processEvents()
reopened = win._sample['id'] == first_sid and 'reopened it' in \
    win.status_bar.currentMessage()
win._exit_sample()
win._start_sample()
win._exit_sample()
win.on_qc_drill('Cz', switch_tab=False)
rdlg = win._open_report()
check('5-sticky', "a reopened older sample stays the session's sample "
      "through Exit, Resume, a re-drill and the report",
      reopened and win._sample['id'] == first_sid
      and rdlg.src.design['sample_id'] == first_sid,
      repr((win._sample['id'][:8], first_sid[:8])))
rdlg.close()
moved = order[5]
win.db.conn.execute("UPDATE events SET end_time = end_time + 0.2 "
                    "WHERE uuid = ?", (moved,))
win.db.conn.commit()
labs = sr.reviewer_labels(win.db.conn, first_sid)
frame = rs.compute_review_precision(win.db.conn, first_sid, reviewer='TK',
                                    write=False)
n_scope = int(frame[frame['domain_type'] == 'scope']['n_reviewed'].iloc[0])
check('5-void', "labels use the library's voiding (end time moved > 0.05 s): "
      "the moved event drops out and the counts match review_precision",
      moved not in labs['TK'] and len(labs['TK']) == n_scope,
      repr((len(labs['TK']), n_scope)))
check('etas', "ETA from the session pace: median gap × remaining",
      sr.session_eta([0, 30, 60, 100], 10) == 300.0
      and sr.session_eta([0], 5) is None)
win.close()

# ===================================================================== 45-51
say("\n== 45-51. Precision report")
CH8 = ['Fz', 'F3', 'Cz', 'C3', 'O1', 'O2', 'T7', 'T8']
P2 = os.path.join(TMP, 'report.db')
build(P2, CH8, n=60, figures=False)
win = window(P2)
win.on_qc_drill('Fz', switch_tab=True)
win._exec_dialog = lambda dlg: drive(dlg, 120, 5)
win._open_draw_dialog()
app.processEvents()
check('45pre', "legacy scope: the dialog showed the region × stage only "
      "note and '—' flagged", seen['note'] == sr.LEGACY_NOTE
      and seen['total_row'][3] == '—', repr(seen['note']))
sid = win._sample['id']
rows = win._sample['rows']
by_cell = {}
for u, r in rows.items():
    by_cell.setdefault((r['region'], r['stage']), []).append(u)
for k in by_cell:
    by_cell[k].sort(key=lambda u: rows[u]['sort_key'])
check('45cells', "4 regions × 2 stages, 15 events each",
      sorted(len(v) for v in by_cell.values()) == [15] * 8,
      repr({k: len(v) for k, v in by_cell.items()}))
reasons = ['artefact'] * 4 + ['arousal'] * 2 + ['too-short', 'eye-movement']
con = win.db.conn
plan = {('frontal', 'NREM2'): ['accept'] * 14 + ['reject'],
        ('frontal', 'NREM3'): ['accept'] * 8 + ['reject'] * 7,
        ('occipital', 'NREM2'): ['accept'] * 6,
        ('occipital', 'NREM3'): ['accept'] * 4}
tk = {}
ri = 0
for cell, decs in plan.items():
    for u, d in zip(by_cell[cell], decs):
        reason = None
        if d == 'reject':
            reason = reasons[ri]
            ri += 1
        dbwrite.store_event_review(con, u, d, 'TK', reason=reason)
        tk[u] = d
both = list(tk)
assert len(both) == 40
flip = both[:6]
for u in both:
    d = tk[u]
    if u in flip:
        d = 'reject' if d == 'accept' else 'accept'
    dbwrite.store_event_review(con, u, d, 'JS',
                               reason='other' if d == 'reject' else None,
                               comment='x' if d == 'reject' else None)
con.commit()
before = con.execute("SELECT COUNT(*) FROM review_precision WHERE "
                     "sample_id = ?", (sid,)).fetchone()[0]
win.set_reviewer_name('TK')
dlg = win._open_report()
app.processEvents()
after = con.execute("SELECT COUNT(*) FROM review_precision WHERE "
                    "sample_id = ?", (sid,)).fetchone()[0]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


lo, hi = wilson(14, 15)
c_fn2 = dlg.cells[('frontal', 'NREM2')]
check('45', "[174] frontal NREM2 14 accept / 1 reject: '93 %' with the "
      "Wilson range and the counts in the tooltip",
      c_fn2 == (f"{round(100 * 14 / 15)} %",
                f"{round(100 * 14 / 15)} % (95 % confidence "
                f"{round(100 * lo)}–{round(100 * hi)} %) · 15 decided: 14 "
                f"accepted, 1 rejected"), repr(c_fn2))
check('46', "[174] a group with 6 decided reads '—' with the too-small "
      "tooltip; frontal NREM3 (53 %) is marked '▼'",
      dlg.cells[('occipital', 'NREM2')] == (
          '—', 'Only 6 decided here; at least 10 are needed to judge.')
      and dlg.cells[('frontal', 'NREM3')][0] == '53 % ▼'
      and all(re.match(r'^(\d+ %( ▼)?|—)$', t)
              for t, _tip in dlg.cells.values()), repr(dlg.cells))
# All stages column: pooled region value, and fully visible [real screen]
dlg.show()
app.processEvents()
_g = dlg.grid
_last = _g.columnCount() - 1
_df = dlg.frames[dlg.shown_reviewer()]


def _pooled(region):
    # independent of cell_text: the library's region row, as a whole percent
    hit = _df[(_df['domain_type'] == 'region') & (_df['domain'] == region)]
    if not len(hit) or int(hit.iloc[0]['n_decided']) < 10:
        return '—'
    return f"{int(round(100 * float(hit.iloc[0]['p_hat'])))} %"


_regions = [_g.verticalHeaderItem(i).text() for i in range(_g.rowCount())]
_stages = {k[1] for k in dlg.cells} - {'All stages'}
check('46b', "[real screen] All stages cells equal the pooled region value, "
      "the column header is not clipped, and the sample spans >= 2 stages",
      _g.horizontalHeaderItem(_last).text() == 'All stages'
      and len(_stages) >= 2
      and any(_pooled(r) != '—' for r in _regions)
      and all(_g.item(i, _last).text() == _pooled(r)
              and _g.item(i, _last).text() != '' for i, r in
              enumerate(_regions))
      and _g.columnViewportPosition(_last) + _g.columnWidth(_last)
      <= _g.viewport().width()
      and _g.horizontalHeader().fontMetrics().horizontalAdvance('All stages')
      < _g.columnWidth(_last),
      repr((_g.columnViewportPosition(_last), _g.columnWidth(_last),
            _g.viewport().width(), _stages)))
check('47a', "[173] rule 80 % with a group at 53 %: the verdict names it, "
      "in the warn colour", dlg.verdict.text()
      == 'Check frontal · NREM3 (53 %): below 80 %.'
      and dlg.verdict_level == 'warn' and '#e0a334' in dlg.verdict.styleSheet(),
      repr(dlg.verdict.text()))
# the verdict wording on synthetic frames [173, ruling]
def frame(groups):
    rows = [{'domain_type': 'region_stage', 'domain': f"{r}|{st}",
             'n_decided': 20, 'n_accept': 0, 'n_reject': 0, 'n_unsure': 0,
             'p_hat': p, 'ci_lo': p - 0.1, 'ci_hi': min(1.0, p + 0.05)}
            for (r, st), p in groups.items()]
    return pd.DataFrame(rows)


ok_df = frame({('frontal', 'NREM2'): 0.95, ('central', 'NREM2'): 0.9})
four = frame({('frontal', 'NREM2'): 0.5, ('central', 'NREM2'): 0.6,
              ('parietal', 'NREM2'): 0.7, ('occipital', 'NREM2'): 0.75})
small = pd.concat([ok_df, pd.DataFrame([{
    'domain_type': 'region_stage', 'domain': 'temporal|NREM3',
    'n_decided': 4, 'p_hat': 1.0, 'ci_lo': 0.5, 'ci_hi': 1.0}])])
check('173b', "[173, ruling] pass reads 'Every region ≥ 80 %' (ok); with "
      "groups too small it says how many; the lower-bound rule says so; "
      "four groups short: three listed then ' and 1 more'",
      sr.verdict_text(ok_df, 0.8) == ('Every region ≥ 80 %', 'ok')
      and sr.verdict_text(ok_df, 0.9, 'lower 95 % bound')
      == ("Check central · NREM2 (lower bound 80 %) and frontal · NREM2 "
          "(lower bound 85 %): below 90 %.", 'warn')
      and sr.verdict_text(ok_df, 0.7, 'lower 95 % bound')
      == ("Every region's lower bound ≥ 70 %", 'ok')
      and sr.verdict_text(small, 0.8)
      == ('Every region ≥ 80 % (1 group(s) too small to judge)', 'ok')
      and sr.verdict_text(four, 0.8)[0] == (
          'Check frontal · NREM2 (50 %), central · NREM2 (60 %), parietal · '
          'NREM2 (70 %) and 1 more: below 80 %.')
      and not any('Looks trustworthy' in t for t in (
          sr.verdict_text(ok_df, 0.8)[0], sr.verdict_text(small, 0.8)[0])),
      repr(sr.verdict_text(four, 0.8)))
# the sentence [171]
labs = {f"u{i}": ('accept', None) for i in range(119)}
labs['r1'] = ('reject', 'artefact')
check('171', "[171] the sentence: 119/1 with one reason; with 2 unsure; "
      "with five reasons, three then ' and 2 more'; the fixture's TK line",
      sr.report_sentence('TK', labs, 120, er.REASON_LABEL)
      == 'TK reviewed 120 of 120 sampled events: 119 accepted, 1 rejected '
         '(artefact 1).'
      and sr.report_sentence('TK', dict(labs, s1=('unsure', None),
                                        s2=('unsure', None)), 122,
                             er.REASON_LABEL).endswith(', 2 unsure.')
      and sr.report_sentence('TK', {'a': ('accept', None)}, 120,
                             er.REASON_LABEL)
      == 'TK reviewed 1 of 120 sampled events: 1 accepted, 0 rejected.'
      and sr.report_sentence('TK', {f"x{k}": ('reject', t) for k, t in
                                    enumerate(['artefact'] * 3
                                              + ['arousal'] * 2
                                              + ['too-short', 'other',
                                                 'eye-movement',
                                                 'not-in-raw'])},
                             9, er.REASON_LABEL).endswith(
          '(artefact 3, arousal 2, eye movement 1 and 3 more).')
      and dlg._sentence_text == 'TK reviewed 40 of 120 sampled events: 32 '
      'accepted, 8 rejected (artefact 4, arousal 2, eye movement 1 and 1 '
      'more).', repr(dlg._sentence_text))
scope = sr._row(dlg.frames['TK'], 'scope', 'all')
m = re.match(r'^Estimated precision: (\d+) % \(95 % confidence (\d+)–(\d+) '
             r'%\)$', dlg.precision.text())
check('172', "[172] the precision line: the weighted whole-night estimate "
      "in whole percent, with its tooltip", m is not None
      and [int(x) for x in m.groups()] == [round(100 * scope[k]) for k in
                                           ('p_hat', 'ci_lo', 'ci_hi')]
      and dlg.precision.toolTip().startswith(
          'Of the events the detector found, the share TK accepted'),
      repr(dlg.precision.text()))
BANNED_R = ('Pooling rule', 'As RA protocol classes',
            'What precision means here', 'Saved to', 'Why rejected',
            'Whole night', 'Looks trustworthy')
dlg.resize(1280, 800)
app.processEvents()
texts_r = [w.text() for w in dlg.findChildren(QtWidgets.QLabel)] + [
    w.text() for w in dlg.findChildren(QtWidgets.QPushButton)]
ys = [w.mapTo(dlg, QtCore.QPoint(0, 0)).y() for w in (
    dlg.title, dlg.sentence, dlg.precision, dlg.verdict, dlg.grid,
    dlg.second, dlg.copy_btn)]
btns = [b.text() for b in (dlg.copy_btn, dlg.export_btn, dlg.close_btn)]
check('170', "[170] title, sentence, precision line, verdict, table, "
      "second-reviewer line and the three buttons, in that order; none of "
      "the removed parts; window title names the subject",
      ys == sorted(ys) and btns == ['Copy summary', 'Export CSV…', 'Close']
      and not [t for t in texts_r for w in BANNED_R if w in t]
      and dlg.windowTitle() == 'Precision report · sub-fx'
      and dlg._title_text == 'Spindles · Moelle2011 9–12 Hz · reviewer TK'
      and dlg.export_btn.toolTip() == 'Writes the table as CSV. The figures '
      'are also saved in neural_events.db, table review_precision.',
      repr((ys, [t for t in texts_r for w in BANNED_R if w in t])))
check('176a', "[176] two reviewers, TK not finished: the picker is there "
      "but disabled with its tooltip; the line withholds agreement",
      dlg.reviewer_combo.isVisibleTo(dlg)
      and not dlg.reviewer_combo.isEnabled()
      and dlg.reviewer_combo.toolTip() == "Finish your own review first, so "
      "other reviewers' decisions do not influence yours."
      and dlg.second.text() == 'JS has also reviewed this sample. Agreement '
      'is shown once you have decided all 120 events.',
      repr(dlg.second.text()))
# the reason list [177]
dlg.sentence.linkActivated.emit('reason:artefact')
app.processEvents()
rows_l = [dlg.event_list.item(k).text()
          for k in range(dlg.event_list.count())]
hdr_l = dlg.list_hdr.text()
vis_l = dlg.event_list.isVisibleTo(dlg) and dlg.list_hint.isVisibleTo(dlg)
first_u = dlg.event_list.item(0).data(Qt.UserRole) if rows_l else None
dlg.event_list.itemDoubleClicked.emit(dlg.event_list.item(0))
app.processEvents()
selected = win.epochs_panel._selected_uuid
dlg.sentence.linkActivated.emit('reason:artefact')
app.processEvents()
check('177', "[177] the 'artefact 4' link lists the four events "
      "('{channel} · {hms1} · {stage}'); double-click selects one in the "
      "Epochs tab and the report stays open; the link again closes it",
      hdr_l == 'Rejected as artefact (4)' and vis_l and len(rows_l) == 4
      and all(re.match(r'^[A-Za-z0-9]+ · \d\d:\d\d:\d\d\.\d · NREM[23]$', r)
              for r in rows_l)
      and dlg.list_hint.text() == 'Double-click an event to open it in the '
      'Epochs tab.' and selected == first_u and dlg.isVisible()
      and win.tabs.currentIndex() == 1
      and not dlg.event_list.isVisibleTo(dlg), repr((hdr_l, rows_l[:2])))
dlg.copy_summary()
clip = QtWidgets.QApplication.clipboard().text().split('\n')
check('179', "[179] Copy summary: exactly the title line, sentence, "
      "precision line and verdict", clip == [
          'Spindles · Moelle2011 9–12 Hz · reviewer TK', dlg._sentence_text,
          dlg.precision.text(), dlg.verdict.text()], repr(clip))
dlg.show()
app.processEvents()
check('180', "[180] fits 1280 × 800 with the list closed: 4 regions × 3 "
      "columns, no table scroll bar", dlg.grid.rowCount() == 4
      and dlg.grid.columnCount() == 3
      and dlg.sizeHint().height() <= 800 and dlg.sizeHint().width() <= 1280
      and dlg.minimumSizeHint().height() <= 800
      and not dlg.grid.verticalScrollBar().isVisible()
      and not dlg.grid.horizontalScrollBar().isVisible(),
      repr((dlg.sizeHint(), dlg.grid.size())))
# the Precision rule [178]
t_def = (rg._review_settings().value('review/pool_threshold', 0.80),
         rg._review_settings().value('review/pool_estimate',
                                     'point estimate'))
seen_rule = {}


def drive_rule(d):
    seen_rule['title'] = d.windowTitle()
    seen_rule['start'] = (d.pct_spin.value(), d.est_combo.currentText())
    d.pct_spin.setValue(90)
    d.save()
    return QtWidgets.QDialog.Accepted


win._exec_dialog = drive_rule
win._open_precision_rule()
app.processEvents()
check('178', "[178] Review ▸ Precision rule… opens 'Precision rule' at the "
      "default (80 %, point estimate); Save at 90 % changes the open "
      "report's verdict and persists in the same QSettings keys",
      seen_rule == {'title': 'Precision rule',
                    'start': (80, 'point estimate')}
      and dlg.verdict.text() == 'Check frontal · NREM3 (53 %): below 90 %.'
      and abs(float(rg._review_settings().value('review/pool_threshold'))
              - 0.90) < 1e-9
      and float(t_def[0]) == 0.80
      and any(a.text() == 'Precision rule…' for a in
              win.findChildren(QtWidgets.QAction)),
      repr((seen_rule, dlg.verdict.text())))
rg._review_settings().setValue('review/pool_threshold', 0.80)
dlg._render()
check('50a', "[181] opening the report wrote review_precision rows",
      before == 0 and after > 0, repr((before, after)))
out = os.path.join(TMP, 'export.csv')
path = dlg.export_csv(out)
with open(path, newline='') as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    rows_csv = list(rd)
check('50b', "[181] the CSV has exactly the listed columns",
      tuple(cols) == sr.CSV_COLUMNS, repr(cols))
check('50c', "one row per reviewer × region × stage plus whole-night rows",
      len(rows_csv) == 2 * (8 + 1)
      and sum(r['region'] == 'all' for r in rows_csv) == 2,
      repr(len(rows_csv)))
check('50d', "default file name next to the database",
      os.path.basename(dlg.default_csv_path())
      == 'sub-fx_spindle_Moelle2011_9-12Hz_review_precision.csv'
      and os.path.dirname(dlg.default_csv_path()) == os.path.dirname(P2),
      repr(dlg.default_csv_path()))
check('50e', "export status names the reviewers and the path",
      win.status_bar.currentMessage() == f"Exported precision for 2 reviewers "
                                         f"to {out}.",
      repr(win.status_bar.currentMessage()))
# TK finishes the sample: agreement appears, the picker unlocks [176]
for u in rows:
    if u not in tk:
        dbwrite.store_event_review(con, u, 'accept', 'TK')
con.commit()
dlg.refresh()
app.processEvents()
a = np.array([tk[u] for u in both])
b = np.array([('reject' if tk[u] == 'accept' else 'accept') if u in flip
              else tk[u] for u in both])
po = np.mean(a == b)
pa, pb = np.mean(a == 'accept'), np.mean(b == 'accept')
pe = pa * pb + (1 - pa) * (1 - pb)
kappa = (po - pe) / (1 - pe)
check('176b', "[176] once TK has decided all 120: the picker is enabled "
      "and the line gives agreement and kappa (34 of 40, hand kappa)",
      dlg.reviewer_combo.isEnabled() and re.match(
          r"^Second reviewer .+: agreement \d+ % on \d+ events you both "
          r"decided \(Cohen's kappa -?\d\.\d\d\)\.$", dlg.second.text())
      is not None and dlg.second.text() == (
          f"Second reviewer JS: agreement 85 % on 40 events you both "
          f"decided (Cohen's kappa {kappa:.2f})."), repr(dlg.second.text()))
dlg.reviewer_combo.setCurrentText('JS')
app.processEvents()
check('176c', "the enabled picker switches the report to JS",
      dlg._title_text.endswith('reviewer JS')
      and dlg._sentence_text.startswith('JS reviewed 40 of 120'),
      repr(dlg._title_text))
dlg.close()


class _OneReviewer:
    """The report's source with only TK's decisions."""

    def __init__(self, src):
        self.__dict__.update(src.__dict__)
        self._src = src

    def labels(self):
        return {'TK': self._src.labels()['TK']}

    def __getattr__(self, name):
        return getattr(self._src, name)


one = rsw.PrecisionReportDialog(_OneReviewer(win._report.src), win)
check('175', "[175] one reviewer: no picker and 'No second reviewer yet.'",
      not one.reviewer_combo.isVisibleTo(one)
      and one.second.text() == 'No second reviewer yet.',
      repr(one.second.text()))
one.close()
win.close()


# ===================================================================== 80-83
say("\n== 80-83. Flag words hidden in live sample review")
P3 = os.path.join(TMP, 'hidden.db')
build(P3, CH10, special={'O2': 0.7})
con3 = sqlite3.connect(P3)
sc3 = sr.scope_for_run(con3, RUN, 'spindle')
sid3 = rs.draw_review_sample(con3, scope=sc3, n_total=120, seed=3)
order3 = sr.presentation_order(con3, sid3)
target = order3[0]
start3 = con3.execute("SELECT start_time FROM events WHERE uuid = ?",
                      (target,)).fetchone()[0]
con3.execute("UPDATE events SET max_amp = 40 + (rowid % 7)")  # a spread,
# so the median + 3.5·MAD outlier rule has a finite threshold
con3.execute(
    "UPDATE events SET in_band = 0, low_prominence = 1, prominence_db = 7.2,"
    " peak_freq_ap = 7.5, near_bound = -1, duration = 0.52, end_time = ?, "
    "amp_ratio = 1.2, max_amp = 900.0 WHERE uuid = ?",
    (start3 + 0.52, target))
con3.commit()
con3.close()
win = window(P3)
win.on_qc_drill(sr.sample_rows(win.db.conn, sid3)[target]['channel'],
                switch_tab=True)
dk = win.detail_dock_w
dk.set_coords({ch: ((i % 5) / 5.0 - 0.4, (i // 5) / 2.0 - 0.25)
               for i, ch in enumerate(CH10)})
dk.topo_combo.setCurrentIndex(dk.topo_combo.findData('pct_off_band'))
win.on_qc_channel_selected('O2')
app.processEvents()
rings_before = len(dk.ring_items)
rows_before = len(dk.flagged.rows)
qw = win.qc_widget
qw.sort_combo.setCurrentText('Amp flag (hard first)')
app.processEvents()
sorted_before = qw.visible_channels()
link_before = "<a href='checks'" in qw.counts_lbl.html()
win._start_sample()
app.processEvents()
win.on_qc_channel_selected('O2')
dock_txt = ' '.join(l.text() for l in dk.findChildren(QtWidgets.QLabel)
                    if l.isVisibleTo(dk))
leak = [w for w in ('HARD', 'SOFT', '× hard', '▲ soft', 'off-band')
        if w in dock_txt]
check('dock-s', "sample active: no channel flag words in the dock, the "
      "flagged list replaced by one line with counts —, no rings",
      rings_before > 0 and rows_before > 0 and not leak
      and dk.ring_items == [] and dk.flagged.rows == []
      and dk.flagged.empty.text() == 'Channel checks are hidden while you '
                                     'review the sample.'
      and dk.flagged.counts.text() == '—'
      and '\n' not in dk.facts_line.text(),
      repr((rings_before, rows_before, leak)))
qw = win.qc_widget
tops = [(dk.topo_combo.itemText(i), dk.topo_combo.model().item(i).isEnabled(),
         dk.topo_combo.itemData(i, Qt.ToolTipRole))
        for i in range(3, dk.topo_combo.count())]
check('95', "[95] sample active: the Channels tab shows the banner with "
      "its exact text and an 'Exit sample' button",
      qw.sample_banner.isVisibleTo(qw) and qw.banner_lbl.text() ==
      'Review sample in progress: channel checks are hidden. Exit sample to '
      'see them.' and qw.banner_exit_btn.text() == 'Exit sample',
      repr(qw.banner_lbl.text()))
visible_cells = [col_cell(win, ch, k)
                 for ch in qw.visible_channels()
                 for k, _h in rg._QC_COLS
                 # Density is — in this fixture with or without a sample
                 # (no scored minutes), so it says nothing about sample mode
                 if k != 'density'
                 and not qw.table.isColumnHidden(rg._QC_COL_INDEX[k])]
check('96', "[96, R5.1] the table has no check columns at all; no visible "
      "cell is — because of sample mode; Topo check items disabled",
      [h for _k, h in rg._QC_COLS] == ['Channel', 'Region', 'Events',
                                       'Density /min', 'Mean amp µV',
                                       'Amp z', 'Amp flag', 'Status']
      and '—' not in visible_cells
      and tops and all(not en and tip == 'Hidden while you review the '
                       'sample.' for _t, en, tip in tops),
      repr([c for c in visible_cells if c == '—']))
stage_b = qw.stage_group.buttons()
win.tabs.setCurrentIndex(0)              # the status bar's Channels summary
app.processEvents()
check('97', "[97, 145] Stage buttons disabled with the R4.0 tooltip (the "
      "checked one still checked); the header count starts 'checks hidden "
      "· ' and is not a link (it was before the sample)", stage_b and all(not b.isEnabled() for b in stage_b)
      and all(b.toolTip() == 'Stages apply to the channel checks, which are '
              'hidden while you review the sample.' for b in stage_b)
      and sum(b.isChecked() for b in stage_b) == 1
      and qw.counts_lbl.text().startswith('checks hidden · ')
      and '<a' not in qw.counts_lbl.html() and link_before
      and qw.counts_lbl.toolTip() == ''
      and ' · checks hidden · ' in win.seg_position.text()
      and 'soft checks' not in win.seg_position.text(),
      repr((qw.counts_lbl.text(), win.seg_position.text())))
sort_items = [qw.sort_combo.itemText(i) for i in range(qw.sort_combo.count())]
check('sort-s', "[R5.1, 98] sample mode leaves the sort as it was (no Sort "
      "item is check-based any more); no 'Column header' appeared",
      qw.visible_channels() == sorted_before
      and qw.sort_combo.currentText() == 'Amp flag (hard first)'
      and 'Column header' not in sort_items
      and all(qw.sort_combo.model().item(i).isEnabled()
              for i in range(qw.sort_combo.count())),
      repr((sorted_before[:2], qw.visible_channels()[:3])))
n_flag_s = qw.show_combo.itemText(1)
check('tbl-f', "[99, 144] sample active: Show > Amp flagged counts amplitude "
      "flags only", n_flag_s == f"Amp flagged "
      f"({int(qw.model.df['flag'].isin(['hard', 'soft']).sum())})",
      repr(n_flag_s))
# no check filter during sample review
win._open_in_epochs('O2')
app.processEvents()
ep_ = win.epochs_panel
msg_open = win.status_bar.currentMessage()
full = [it.brush.color().alpha() for it in ep_.band_items(ep_.raw_plot)]
ep_._goto_epoch(0)
ep_.clear_selection()
seen_all = set()
for _ in range(len(ep_._ev) + 2):
    QTest.keyClick(ep_, Qt.Key_BraceRight)
    app.processEvents()
    if ep_._selected_uuid:
        seen_all.add(ep_._selected_uuid)
check('flt-s', "sample active: Open in Epochs on a flagged channel applies "
      "no filter (no chip, full fill, } visits every event)",
      ep_._channel == 'O2' and ep_.chip_text() == '' and ep_._check_filter
      is None and len(full) > 0 and len(set(full)) == 1
      and len(seen_all) == len(ep_._ev)
      and msg_open
      == 'Check filter not applied while you review the sample.',
      repr((ep_._channel, ep_.chip_text(), ep_._check_filter,
            sorted(set(full)), len(seen_all), len(ep_._ev),
            msg_open)))
qw.banner_exit_btn.click()               # the banner's Exit sample
app.processEvents()
win.on_qc_channel_selected('O2')
app.processEvents()
check('95b', "[95] the banner's button ends the sample and the banner goes",
      not win._sample_active and not qw.sample_banner.isVisibleTo(qw))
check('sort-e', "after Exit: the sort ('Amp flag (hard first)') is as "
      "before, every Sort item is enabled, no 'Column header'",
      qw.sort_combo.currentText() == 'Amp flag (hard first)'
      and qw.visible_channels() == sorted_before
      and qw.sort_combo.findText('Column header') < 0
      and all(qw.sort_combo.model().item(i).isEnabled()
              for i in range(qw.sort_combo.count())),
      repr((qw.sort_combo.currentText(), qw.visible_channels()[:3])))
check('tbl-e', "[96, 145] after Exit: the header count (a link again), "
      "the Stage buttons and the Topo items are back",
      all(b.isEnabled() and b.toolTip() == ''
          for b in qw.stage_group.buttons())
      and qw.counts_lbl.text()[0].isdigit()
      and "<a href='checks'" in qw.counts_lbl.html()
      and all(dk.topo_combo.model().item(i).isEnabled()
              for i in range(3, dk.topo_combo.count())),
      repr(qw.counts_lbl.text()))
check('dock-e', "after Exit the rings, the list and the facts are back",
      len(dk.ring_items) == rings_before
      and len(dk.flagged.rows) == rows_before
      and 'checks × hard' in dk.facts_line.text(),
      repr((len(dk.ring_items), dk.facts_line.text())))
win.on_qc_drill(sr.sample_rows(win.db.conn, sid3)[target]['channel'],
                switch_tab=True)
win._start_sample()
app.processEvents()
ep = win.epochs_panel
evp = win.detail_dock_w.event_panel
BANNED = ('in band', 'OFF BAND', 'shortest allowed', 'longest allowed',
          'outside the limits', 'barely', 'yes', 'no · ', 'meets', 'fails',
          'only just crossed', 'flagged')


def panel_text():
    """Everything the panel shows for the event: the header and sample
    lines, the four values and their tooltips."""
    return '\n'.join([evp.event_line.text(), evp.sample_lbl.text()]
                     + [r['value'] + '\n' + r['tooltip'] for r in evp._rows])


def trace_marks():
    """'outlier' labels on the traces, the red-tinted bands, and the red
    (outlier) ticker bars."""
    bad = rg.QtGui.QColor(rg.THEME['bad'])
    labels = [it.toPlainText() for p, it in ep._event_items
              if isinstance(it, rg.pg.TextItem)]
    red = [it for it in ep.band_items(ep.raw_plot)
           if it.brush.color().red() == bad.red()
           and it.brush.color().green() == bad.green()]
    ticks = [it for it in ep._ticker_items
             if it.opts.get('brush') == rg.THEME['bad']]
    return labels.count('outlier'), len(red), len(ticks)


def coloured():
    return [r['key'] for r in evp._rows if r.get('level') in ('warn', 'bad')]


t = panel_text()
check('80', "[80] no flag word before a decision; numbers stay; no warn/bad "
      "colour; the hidden line is shown", ep._selected_uuid == target
      and not any(b in t for b in BANNED)
      and [r['value'] for r in evp._rows] == [
          '1.2×', '0.52 s · limits 0.5–3 s', '≈ 7.5 Hz', '900 µV']
      and coloured() == [] and evp.hidden_lbl.isVisibleTo(evp)
      and all(evp._row_widgets[k][1].textFormat() == Qt.PlainText
              for k in evp.row_keys()),
      repr(([b for b in BANNED if b in t], coloured(),
            [r['value'] for r in evp._rows])))
check('83', "[ruling] an undecided sample event that is an amplitude "
      "outlier is drawn as a regular band: no 'outlier' label, no red "
      "tint on the traces, no red ticker bar", trace_marks() == (0, 0, 0),
      repr(trace_marks()))
# the gate's probe: the first sample event is a 900 µV outlier, undecided.
# No surface of the Epochs tab or the right dock may say so.
dkp = win.detail_dock_w


def shown_texts():
    """Every text the Epochs tab and the right dock show: labels, buttons,
    list rows, and the text items on the plots."""
    out = []
    for root in (ep, dkp):
        for w in root.findChildren(QtWidgets.QWidget):
            if not w.isVisibleTo(root):
                continue
            if isinstance(w, (QtWidgets.QLabel, QtWidgets.QAbstractButton)):
                out.append(w.text())
            if isinstance(w, QtWidgets.QListWidget):
                out += [w.item(i).text() for i in range(w.count())]
    for plot in (ep.plot, ep.raw_plot, ep.filt_plot, ep.ticker):
        out += [it.toPlainText() for it in plot.getPlotItem().items
                if isinstance(it, rg.pg.TextItem)]
    return out


def strip_red():
    return [it for it in ep.plot.getPlotItem().items
            if isinstance(it, rg.pg.BarGraphItem)
            and it.opts.get('brush') == rg.THEME['bad']]


def rank_rows():
    return ([dkp.global_worst_list.item(i).text()
             for i in range(dkp.global_worst_list.count())]
            + [dkp.worst_list.item(i).text()
               for i in range(dkp.worst_list.count())])


# a refresh while the sample is active must not refill the rankings
win.refresh_qc_dashboard()
win.wait_population()
if dkp._last_channel is not None:
    dkp.update_channel(*dkp._last_channel)
app.processEvents()
texts_s = shown_texts()
# Fixed names, the same for every event, are not indications: the fourth
# Event-panel row's label, and the two disabled hop buttons (the ruling
# keeps them, disabled, with a tooltip)
FIXED = ('Amplitude outlier', '◀◀ Prev outlier', 'Next outlier ▶▶')
leaks = [t for t in texts_s if 'outlier' in t.lower() and t not in FIXED]
epoch_before = ep._epoch
press(win, Qt.Key_N)
press(win, Qt.Key_P)
ep._next_outlier()
ep._prev_outlier()
check('out-s', "[BLOCK] sample active, first sample event a 900 µV outlier, "
      "undecided: no text in the Epochs tab or the right dock says "
      "'outlier'; the two rankings are empty and replaced by one line; the "
      "strip has no red bar and its legend no red; the outlier rule line "
      "is hidden; N / P and the outlier buttons do nothing",
      ep._selected_uuid == target and not leaks
      and rank_rows() == [] and not any('900' in t for t in texts_s
                                        if t != evp.row_text('outlier'))
      and not dkp.global_worst_list.isVisibleTo(dkp)
      and not dkp.worst_list.isVisibleTo(dkp)
      and dkp.global_worst_hidden.isVisibleTo(dkp)
      and dkp.worst_hidden.isVisibleTo(dkp)
      and dkp.worst_hidden.text() == 'Hidden while you review the sample.'
      and strip_red() == []
      and 'red' not in ep.strip_legend.text()
      and 'outlier' not in ep.epoch_lbl.text()
      and not ep.strip_hdr.isVisibleTo(ep)
      and not ep.prev_out_btn.isEnabled() and not ep.next_out_btn.isEnabled()
      and ep.next_out_btn.toolTip() == 'Hidden while you review the sample.'
      and ep._epoch == epoch_before and ep._outlier_epoch_indices() == []
      and 'µV' not in evp.row('outlier')['tooltip'].split('Detector')[0],
      repr((leaks, rank_rows()[:2], ep.epoch_lbl.text())))
# [169] Show events in sample mode reads only TK's own decisions: an event
# only JS decided counts as unreviewed; the undecided outlier sample event
# passes 'unreviewed' and still shows no outlier tint or label
other_u = next(u for u in ep._ev['uuid'] if u != target
               and u not in ep._reviews)
dbwrite.store_event_review(win.db.conn, other_u, 'reject', 'JS',
                           reason='artefact')
win.db.conn.commit()
ep.set_reviews(win._reviews_for_slice(ep._df))
n_unrev = int(sum(1 for u in ep._ev['uuid'] if u not in ep._reviews))
for k in ('accepted', 'rejected', 'unsure'):
    ep.show_checks[k].setChecked(False)
app.processEvents()
ep.select_event(target)
app.processEvents()
chip169 = ep.shown_chip_text()
marks169 = trace_marks()
ticks169 = list(ep.shown_tick_epochs)
want169 = sorted({ep.index_at(t) for t in ep._ev.loc[
    ~ep._ev['uuid'].isin(list(ep._reviews)), '_start']})
ep.shown_chip_x.click()
app.processEvents()
check('169', "[169] sample mode, two reviewers: JS's decision counts as "
      "unreviewed for TK in the chip and the ticks; the undecided outlier "
      "sample event passing the filter has no outlier tint or label",
      chip169 == f"Showing: unreviewed · {n_unrev} of "
      f"{int(ep._ev['uuid'].notna().sum())} on {ep._channel} ✕"
      and other_u not in ep._reviews and ticks169 == want169
      and marks169 == (0, 0, 0), repr((chip169, marks169)))
ep.select_event(target)
app.processEvents()
win._auto_advance = False
press(win, Qt.Key_U)
press(win, Qt.Key_Return)
t = panel_text()
check('81a', "[81] after U + Enter the labels stay hidden",
      not any(b in t for b in BANNED) and evp.hidden_lbl.isVisibleTo(evp))
press(win, Qt.Key_A)
t = panel_text()
check('81b', "[81, 121] after A the reading words appear (coloured, "
      "weight 600), the hidden line goes, and the outlier label is on both "
      "traces", [r['value'] for r in evp._rows] == [
          '1.2× · barely above background',
          '0.52 s · limits 0.5–3 s · at the shortest allowed',
          '≈ 7.5 Hz · OFF BAND', 'yes · 900 µV']
      and coloured() == ['signal_bg', 'duration', 'peak_freq', 'outlier']
      and 'font-weight:600' in evp._row_widgets['peak_freq'][1].text()
      and ' · flagged: ' in evp.sample_lbl.text()
      and not evp.hidden_lbl.isVisibleTo(evp)
      and trace_marks()[0] == 2, repr((t[:120], trace_marks())))
bt = evp.btn
saved161 = {d: b.isChecked() for d, b in bt.items()}
press(win, Qt.Key_R)
armed161 = ({d: b.isChecked() for d, b in bt.items()},
            bt['reject'].styleSheet())
press(win, Qt.Key_Escape)
back161 = {d: b.isChecked() for d, b in bt.items()}
check('161', "[161] sample mode, sample event: after A only Accept is "
      "checked; R arms Reject (border, not checked) with Accept still "
      "checked; Esc returns to Accept only",
      saved161 == {'accept': True, 'reject': False, 'unsure': False}
      and armed161[0] == saved161
      and armed161[1] == rg.EventDecisionPanel.ARMED_QSS
      and back161 == saved161, repr((saved161, armed161, back161)))
press(win, Qt.Key_Z, Qt.ControlModifier)
t = panel_text()
check('81c', "[81] Ctrl+Z (back to unsure) hides them again",
      not any(b in t for b in BANNED) and evp.hidden_lbl.isVisibleTo(evp)
      and trace_marks()[0] == 0, repr([b for b in BANNED if b in t]))
win._exit_sample()
ep.select_event(target)
app.processEvents()
t = panel_text()
texts_e = shown_texts()
check('out-e', "[BLOCK] after Exit everything is back: the rankings (with "
      "the 900 µV event), the red strip bar and its legend, the outlier "
      "count in the epoch label, the rule line, N / P and the buttons",
      any('900' in r for r in rank_rows())
      and dkp.global_worst_list.isVisibleTo(dkp)
      and dkp.worst_list.isVisibleTo(dkp)
      and not dkp.global_worst_hidden.isVisibleTo(dkp)
      and len(strip_red()) == 1
      and 'red = amplitude outliers' in ep.strip_legend.text()
      and 'outlier' in ep.epoch_lbl.text()
      and ep.strip_hdr.isVisibleTo(ep)
      and ep.strip_hdr.text().startswith('Outlier rule: amp > ')
      and ep.prev_out_btn.isEnabled() and ep.next_out_btn.isEnabled()
      and ep.next_out_btn.toolTip() == ''
      and ep._outlier_epoch_indices() != []
      and 'Rule: amplitude above' in evp.row('outlier')['tooltip'],
      repr((rank_rows()[:1], ep.epoch_lbl.text())))
check('82a', "[82] the same event outside sample mode shows its labels "
      "and its outlier marks, and has no sample line",
      'OFF BAND' in t and 'yes · 900 µV' in t
      and not evp.hidden_lbl.isVisibleTo(evp)
      and not evp.sample_lbl.isVisibleTo(evp) and trace_marks()[0] == 2,
      repr(trace_marks()))
win._start_sample()
other = next(u for u in ep._ev['uuid'] if u not in win._sample['rows'])
ep.select_event(other)
app.processEvents()
check('82b', "[82] a non-sample event in sample mode shows labels, no "
      "hidden line, no sample line", not evp.hidden_lbl.isVisibleTo(evp)
      and not evp.sample_lbl.isVisibleTo(evp)
      and evp.row_keys() == FOUR)
# [item 2] a flagged and an unflagged undecided event in ONE cell: before a
# decision both rows are identical in form (no stratum, flags or weight);
# after accept the stratum and the weight tooltip appear.
rows3 = win._sample['rows']
undecided = [u for u in win._sample['order'] if u not in ep._reviews]
pair = None
for u in undecided:
    for v in undecided:
        if (rows3[u]['region'], rows3[u]['stage']) == (
                rows3[v]['region'], rows3[v]['stage']) \
                and rows3[u]['flagged'] == 1 and rows3[v]['flagged'] == 0:
            pair = (u, v)
            break
    if pair:
        break
before = {}
for u in pair or ():
    win._goto_sample_event(u)
    app.processEvents()
    before[u] = (evp.sample_lbl.text(), evp.sample_lbl.toolTip())
cell_txt = (f"{rows3[pair[0]]['region']} · {rows3[pair[0]]['stage']}"
            if pair else '')
check('wt-a', "flagged + unflagged in one cell, undecided: both sample "
      "lines read 'Sample event i of N · region · stage' and nothing "
      "else, with no weight tooltip", pair is not None
      and all(re.match(r'^Sample event \d+ of 120 · ' + re.escape(cell_txt)
                       + r'$', val) and not tip
              for val, tip in before.values()), repr(before))
after = {}
for u in pair or ():
    win._goto_sample_event(u)
    app.processEvents()
    press(win, Qt.Key_A)
    win._goto_sample_event(u)
    app.processEvents()
    after[u] = (evp.sample_lbl.text(), evp.sample_lbl.toolTip())
check('wt-b', "after accept: region · stage · flags, and the weight "
      "tooltip, which differs between the flagged and unflagged event",
      pair is not None
      and f" · {cell_txt} · flagged: " in after[pair[0]][0]
      and after[pair[1]][0].endswith(f" · {cell_txt} · not flagged")
      and all('weight' in tip.lower() for _v, tip in after.values())
      and after[pair[0]][1] != after[pair[1]][1], repr(after))
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
