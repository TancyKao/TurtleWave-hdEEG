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
srow0 = (evp.row_text('sample') or '')
check('20a', "[20] before this reviewer's accept/reject the sample row reads "
      "only 'event i of N' (no stratum, flags or weight)",
      evp.row_keys()[:1] == ['sample']
      and re.match(r'^event \d+ of 120$', srow0) is not None
      and not evp.row('sample')['tooltip']
      and evp.hidden_lbl.isVisibleTo(evp)
      and evp.hidden_lbl.text() == 'Labels hidden until you accept or '
                                   'reject this sample event.',
      repr((evp.row_keys()[:2], srow0)))
check('75s', "[75] top-bar hint in sample mode", win.key_hint_lbl.text() ==
      'A accept · R reject · U unsure · ] [ sample · N P outlier · ? keys',
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
srow = evp.row_text('sample') or ''
check('20b', "after the decision the first row is 'sample' with the stratum",
      keys[:1] == ['sample'] and re.match(
          r'^[a-z-]+ · [A-Z0-9]+ · (not flagged|flagged: .+)$',
          srow.split('\n')[0]) is not None
      and srow.split('\n')[1] == 'event 2 of 120', repr(srow))
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
      "stays unprimed)", re.match(r'^event \d+ of 120$',
                                  evp.row_text('sample') or '') is not None
      and evp.hidden_lbl.isVisibleTo(evp), repr(evp.row_text('sample')))
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
want = f"{14 / 15:.2f}  ({lo:.2f}–{hi:.2f})  n 15"
check('45', "frontal NREM2 14 accept / 1 reject: Wilson cell",
      dlg.cells[('frontal', 'NREM2')] == want,
      repr((dlg.cells[('frontal', 'NREM2')], want)))
check('46', "a group with 6 decided reads 'n too small (6)'; the verdict "
      "says groups were not judged",
      dlg.cells[('occipital', 'NREM2')] == 'n too small (6)'
      and 'were not judged.' in dlg.verdict.text(), repr(dlg.verdict.text()))
dlg.thr_spin.setValue(0.80)
check('47a', "rule 0.80 with a group at 0.53: not poolable",
      dlg.verdict.text().startswith('Verdict        Not poolable under this '
                                    'rule: frontal · NREM3 0.53'),
      repr(dlg.verdict.text()))
check('47b', "the cell below the rule reads ' below' in warn colour",
      dlg.cells[('frontal', 'NREM3')].endswith(' below'))
dlg.thr_spin.setValue(0.50)
check('47c', "rule 0.50: poolable", dlg.verdict.text().startswith(
    'Verdict        Poolable under this rule: all 2 region × stage groups '
    'have precision of at least 0.50.'), repr(dlg.verdict.text()))
dlg.close()
dlg2 = win._open_report()
check('47d', "the threshold survives a reopen (QSettings)",
      abs(dlg2.thr_spin.value() - 0.50) < 1e-9)
dlg2.est_combo.setCurrentText('lower 95 % bound')
check('47e', "lower-bound rule quotes the lower bound",
      'lower bound' in dlg2.verdict.text(), repr(dlg2.verdict.text()))
dlg2.est_combo.setCurrentText('point estimate')
dlg2.thr_spin.setValue(0.80)
counts = [n for _, n in dlg2.reason_rows]
fa_fo = re.search(r'FP-artifact (\d+) · FP-other (\d+)', dlg2.reasons.text())
check('48', "reasons by count, descending; M7 line sums to the rejected "
      "total (FP-artifact = artefact + eye movement)",
      counts == sorted(counts, reverse=True) and fa_fo is not None
      and int(fa_fo.group(1)) == 5 and int(fa_fo.group(2)) == 3
      and sum(counts) == 8, repr((dlg2.reason_rows, dlg2.reasons.text())))
a = np.array([tk[u] for u in both])
b = np.array([('reject' if tk[u] == 'accept' else 'accept') if u in flip
              else tk[u] for u in both])
po = np.mean(a == b)
pa, pb = np.mean(a == 'accept'), np.mean(b == 'accept')
pe = pa * pb + (1 - pa) * (1 - pb)
kappa = (po - pe) / (1 - pe)
check('49', "agreement 34 of 40 (85 %) and kappa equal to the hand value",
      dlg2.agree.text().startswith('Agreed on 34 of 40 (85 %, accept / '
                                   'reject / unsure, over every event both '
                                   'reviewers decided)')
      and f"Cohen's kappa {kappa:.2f} on the 40 events neither reviewer "
          f"marked unsure" in dlg2.agree.text(),
      repr((dlg2.agree.text(), round(kappa, 2))))
check('49c', "kappa wording when nobody decided without an unsure",
      "no events both reviewers decided" in sr.agreement_text(
          {'n_shared': 3, 'percent_agreement': 1.0, 'kappa': float('nan'),
           'n_both_decided': 0}))
check('49b', "Show the 6 disagreements lists six events",
      dlg2.dis_btn.text() == 'Show the 6 disagreements'
      and dlg2.dis_list.count() == 6)
check('50a', "opening the report wrote review_precision rows",
      before == 0 and after > 0, repr((before, after)))
out = os.path.join(TMP, 'export.csv')
path = dlg2.export_csv(out)
with open(path, newline='') as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    rows_csv = list(rd)
check('50b', "CSV has exactly the listed columns",
      tuple(cols) == sr.CSV_COLUMNS, repr(cols))
check('50c', "one row per reviewer × region × stage plus whole-night rows",
      len(rows_csv) == 2 * (8 + 1)
      and sum(r['region'] == 'all' for r in rows_csv) == 2,
      repr(len(rows_csv)))
check('50d', "default file name next to the database",
      os.path.basename(dlg2.default_csv_path())
      == 'sub-fx_spindle_Moelle2011_9-12Hz_review_precision.csv'
      and os.path.dirname(dlg2.default_csv_path()) == os.path.dirname(P2),
      repr(dlg2.default_csv_path()))
check('50e', "export status names the reviewers and the path",
      win.status_bar.currentMessage() == f"Exported precision for 2 reviewers "
                                         f"to {out}.",
      repr(win.status_bar.currentMessage()))
dlg2.copy_summary()
check('50f', "Copy summary puts the verdict on the clipboard",
      'Not poolable under this rule' in QtWidgets.QApplication.clipboard()
      .text())
dlg2.resize(1280, 800)
dlg2.show()
app.processEvents()
check('51', "fits 1280 × 800: size hint within it and no grid scroll bar "
      "for 4 regions × 3 columns", dlg2.grid.rowCount() == 4
      and dlg2.minimumSizeHint().height() <= 800
      and dlg2.minimumSizeHint().width() <= 1280
      and not dlg2.grid.verticalScrollBar().isVisible(),
      repr((dlg2.minimumSizeHint(), dlg2.grid.verticalScrollBar().isVisible())))
dlg2.close()
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
qw.sort_combo.setCurrentText('Checks (hard first)')
app.processEvents()
sorted_before = qw.visible_channels()
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
cells = [col_cell(win, 'O2', k) for k in ('checks_flag', 'pct_off_band',
                                          'pct_low_prom', 'pct_dur_floor',
                                          'med_amp_ratio', 'med_thresh_ratio')]
tops = [(dk.topo_combo.itemText(i), dk.topo_combo.model().item(i).isEnabled(),
         dk.topo_combo.itemData(i, Qt.ToolTipRole))
        for i in range(3, dk.topo_combo.count())]
check('tbl-s', "sample active: Checks and population cells read — with the "
      "tooltip; header '— checks flagged'; Topo check items disabled",
      cells == ['—'] * 6
      and col_cell(win, 'O2', 'pct_off_band', Qt.ToolTipRole)
      == 'Hidden while you review the sample.'
      and qw.counts_lbl.text().startswith('— checks flagged · ')
      and tops and all(not en and tip == 'Hidden while you review the '
                       'sample.' for _t, en, tip in tops),
      repr((cells, qw.counts_lbl.text())))
# montage order = the order of the coordinates loaded above (CH10), not
# the database's alphabetical order
chan_order = [c for c in CH10 if c in qw.visible_channels()]
sort_items = {qw.sort_combo.itemText(i): (
    qw.sort_combo.model().item(i).isEnabled(),
    qw.sort_combo.itemData(i, Qt.ToolTipRole))
    for i in range(qw.sort_combo.count())}
check_sorts = [lab for lab, k, _d in rg._er.SORT_ITEMS
               if k is None or k in rg._er.CHECK_COLUMNS]
check('sort-s', "sample active: rows in montage order (Fz, F3, Cz, ...; O2, "
      "flagged, is not on top), Sort reads 'Channel order', check sorts disabled with the "
      "tooltip, header clicks do not sort",
      sorted_before[0] == 'O2' and qw.visible_channels() == chan_order
      and qw.visible_channels()[0] != 'O2'
      and chan_order != sorted(chan_order)
      and qw.sort_combo.currentText() == 'Channel order'
      and all(not sort_items[l][0] and sort_items[l][1]
              == 'Hidden while you review the sample.' for l in check_sorts)
      and sort_items['Amp z ↓'][0] and sort_items['Channel'][0]
      and not qw.table.isSortingEnabled(),
      repr((sorted_before[:2], qw.visible_channels()[:3],
            qw.sort_combo.currentText())))
n_flag_s = qw.show_combo.itemText(1)
check('tbl-f', "sample active: Show > Flagged counts amplitude flags only",
      n_flag_s == f"Flagged ({int(qw.model.df['flag'].isin(['hard', 'soft']).sum())})",
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
win._exit_sample()
win.on_qc_channel_selected('O2')
app.processEvents()
check('sort-e', "after Exit: the previous sort ('Checks (hard first)', "
      "O2 on top) is back and every Sort item is enabled again",
      qw.sort_combo.currentText() == 'Checks (hard first)'
      and qw.visible_channels() == sorted_before
      and qw.sort_combo.findText('Channel order') < 0
      and all(qw.sort_combo.model().item(i).isEnabled()
              for i in range(qw.sort_combo.count()))
      and qw.table.isSortingEnabled(),
      repr((qw.sort_combo.currentText(), qw.visible_channels()[:3])))
check('tbl-e', "after Exit: check cells, header count and Topo items back",
      col_cell(win, 'O2', 'checks_flag').startswith('× HARD')
      and qw.counts_lbl.text()[0].isdigit()
      and all(dk.topo_combo.model().item(i).isEnabled()
              for i in range(3, dk.topo_combo.count())),
      repr((col_cell(win, 'O2', 'checks_flag'), qw.counts_lbl.text())))
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
BANNED = ('in band', 'OFF BAND', 'low prominence', 'at the floor',
          'at the ceiling', 'outside run limits', 'barely', 'meets', 'fails')


def panel_text():
    return '\n'.join((r['value'] + '\n' + '\n'.join(r['sub']))
                     for r in evp._rows)


def coloured():
    return [r['key'] for r in evp._rows if r.get('level') in ('warn', 'bad')]


t = panel_text()
check('80', "[80] no flag word before a decision; numbers stay; no warn/bad "
      "colour; the hidden line is shown", ep._selected_uuid == target
      and not any(b in t for b in BANNED)
      and all(u in t for u in ('Hz', 'dB', '×'))
      and coloured() == [] and evp.hidden_lbl.isVisibleTo(evp),
      repr(([b for b in BANNED if b in t], coloured())))
labels_on_trace = [it.toPlainText() for p, it in ep._event_items
                   if isinstance(it, rg.pg.TextItem)]
check('83', "[83] outlier row and the outlier trace label present before "
      "any decision", (evp.row_text('outlier') or '').startswith('yes')
      and labels_on_trace.count('outlier') == 2, repr(labels_on_trace))
win._auto_advance = False
press(win, Qt.Key_U)
press(win, Qt.Key_Return)
t = panel_text()
check('81a', "[81] after U + Enter the labels stay hidden",
      not any(b in t for b in BANNED) and evp.hidden_lbl.isVisibleTo(evp))
press(win, Qt.Key_A)
t = panel_text()
check('81b', "[81] after A they appear and the hidden line goes",
      'OFF BAND' in t and 'low prominence' in t
      and not evp.hidden_lbl.isVisibleTo(evp), repr(t[:120]))
press(win, Qt.Key_Z, Qt.ControlModifier)
t = panel_text()
check('81c', "[81] Ctrl+Z (back to unsure) hides them again",
      not any(b in t for b in BANNED) and evp.hidden_lbl.isVisibleTo(evp),
      repr([b for b in BANNED if b in t]))
win._exit_sample()
ep.select_event(target)
app.processEvents()
t = panel_text()
check('82a', "[82] the same event outside sample mode shows its labels",
      'OFF BAND' in t and not evp.hidden_lbl.isVisibleTo(evp))
win._start_sample()
other = next(u for u in ep._ev['uuid'] if u not in win._sample['rows'])
ep.select_event(other)
app.processEvents()
check('82b', "[82] a non-sample event in sample mode shows labels, no "
      "hidden line, no sample row", not evp.hidden_lbl.isVisibleTo(evp)
      and 'sample' not in evp.row_keys())
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
    r = evp.row('sample')
    before[u] = (r['value'], list(r['sub']), r['tooltip'])
check('wt-a', "flagged + unflagged in one cell, undecided: both rows read "
      "only 'event i of N', no sub-line, no tooltip", pair is not None
      and all(re.match(r'^event \d+ of 120$', val) and sub == [] and not tip
              for val, sub, tip in before.values()), repr(before))
after = {}
for u in pair or ():
    win._goto_sample_event(u)
    app.processEvents()
    press(win, Qt.Key_A)
    win._goto_sample_event(u)
    app.processEvents()
    r = evp.row('sample')
    after[u] = (r['value'], r['tooltip'])
check('wt-b', "after accept: region · stage · flags, and the weight "
      "tooltip, which differs between the flagged and unflagged event",
      pair is not None
      and after[pair[0]][0].startswith(
          f"{rows3[pair[0]]['region']} · {rows3[pair[0]]['stage']} · flagged")
      and after[pair[1]][0].endswith('not flagged')
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
