"""Widgets of review-sample mode: the REVIEW SAMPLE bar, the draw dialog and
the precision report (UX spec ``event-decision-spec.md`` revision 2,
sections 2 and 11). Logic and text live in :mod:`frontend.sample_review`.
"""

import csv
import os
import random

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

try:
    from frontend import sample_review as sr
    from frontend import event_review as er
except ImportError:  # run as a script
    import sample_review as sr
    import event_review as er

_MUTED = '#888888'
_WARN = '#e0a334'
_ACCENT = '#5a8fce'


def _settings():
    """Same store as ``eeg_review_gui._review_settings`` (default format)."""
    return QtCore.QSettings(QtCore.QSettings.defaultFormat(),
                            QtCore.QSettings.UserScope, 'turtlewave',
                            'eeg_review_gui')


class SampleBar(QtWidgets.QWidget):
    """One 28 px row across the top of the Epochs tab."""

    drawClicked = pyqtSignal()
    resumeClicked = pyqtSignal()
    exitClicked = pyqtSignal()
    reportClicked = pyqtSignal()
    showOthersToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        self.label = QtWidgets.QLabel(sr.NO_SAMPLE_TEXT)
        self.label.setStyleSheet("font-size:11px;color:#e5e5e5;")
        self.label.setToolTip(sr.FREE_BROWSING_TIP)
        lay.addWidget(self.label, 1)
        self.btn_draw = QtWidgets.QPushButton("Draw sample…")
        self.btn_resume = QtWidgets.QPushButton("Resume sample")
        self.btn_exit = QtWidgets.QPushButton("Exit sample")
        self.btn_report = QtWidgets.QPushButton("Precision report…")
        for b, sig in ((self.btn_draw, self.drawClicked),
                       (self.btn_resume, self.resumeClicked),
                       (self.btn_exit, self.exitClicked),
                       (self.btn_report, self.reportClicked)):
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(sig.emit)
            lay.addWidget(b)
        self.others_chk = QtWidgets.QCheckBox("Show other reviewers")
        self.others_chk.setFocusPolicy(Qt.NoFocus)
        self.others_chk.toggled.connect(self.showOthersToggled.emit)
        lay.addWidget(self.others_chk)
        self.set_state('none', sr.NO_SAMPLE_TEXT)

    def set_state(self, state, text):
        """Buttons per spec section 2 for ``none``/``idle``/``active``/
        ``done``/``revisit``."""
        self.label.setText(text)
        active = state in ('active', 'done', 'revisit')
        self.btn_draw.setVisible(state == 'none')
        self.btn_resume.setVisible(state == 'idle')
        self.btn_exit.setVisible(active)
        self.btn_report.setVisible(state != 'none')
        self.others_chk.setVisible(active)

    def visible_buttons(self):
        return [b.text() for b in (self.btn_draw, self.btn_resume,
                                   self.btn_exit, self.btn_report)
                if b.isVisibleTo(self)]


class DrawSampleDialog(QtWidgets.QDialog):
    """``Draw review sample``: size, seed, a live allocation preview.

    ``preview_fn(size, seed) -> (rows, totals, preview)`` is the library's
    ``preview_allocation`` (reads only; nothing is written until Draw). It
    runs 200 ms after the last size or seed change. Never deletes an
    existing sample.
    """

    def __init__(self, subject, events_line, preview_fn, existing_note=None,
                 seed=None, parent=None, error=None):
        super().__init__(parent)
        self.setWindowTitle('Draw review sample')
        self.preview_fn = preview_fn
        self.flag_available = None
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.subject_lbl = QtWidgets.QLabel(str(subject or '—'))
        form.addRow('Subject', self.subject_lbl)
        form.addRow('Events', QtWidgets.QLabel(events_line))
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(20, 2000)
        self.size_spin.setValue(120)
        form.addRow('Sample size', self.size_spin)
        seed_row = QtWidgets.QHBoxLayout()
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(1, 99999)
        self.seed_spin.setValue(int(seed) if seed else random.randint(1, 99999))
        seed_row.addWidget(self.seed_spin)
        seed_row.addWidget(QtWidgets.QLabel(
            '(random; keep it to redraw the same sample)'))
        form.addRow('Seed', seed_row)
        lay.addLayout(form)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ['Region × stage', 'events in run', 'in sample',
             'of which flagged'])
        self.table.horizontalHeaderItem(3).setToolTip(sr.FLAG_DEFINITION)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        lay.addWidget(self.table)
        self.note = QtWidgets.QLabel('')
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        lay.addWidget(self.note)
        self.existing = QtWidgets.QLabel(existing_note or '')
        self.existing.setWordWrap(True)
        self.existing.setVisible(bool(existing_note))
        lay.addWidget(self.existing)
        self.error = QtWidgets.QLabel(error or '')
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color:{_WARN};")
        self.error.setVisible(bool(error))
        lay.addWidget(self.error)
        box = QtWidgets.QDialogButtonBox()
        box.addButton('Cancel', QtWidgets.QDialogButtonBox.RejectRole)
        self.draw_btn = box.addButton(
            'Draw new sample' if existing_note else 'Draw',
            QtWidgets.QDialogButtonBox.AcceptRole)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.refresh_preview)
        self.size_spin.valueChanged.connect(lambda *_: self._timer.start())
        self.seed_spin.valueChanged.connect(lambda *_: self._timer.start())
        self.refresh_preview()

    def refresh_preview(self):
        """Recompute the allocation preview now."""
        self._timer.stop()
        if self.preview_fn is None:
            self.draw_btn.setEnabled(False)
            self.table.setRowCount(0)
            return
        try:
            rows, tot, pv = self.preview_fn(self.size_spin.value(),
                                            self.seed_spin.value())
        except Exception as err:       # refusals, or a closed database
            self.error.setText(str(err))
            self.error.setVisible(True)
            self.draw_btn.setEnabled(False)
            return
        self.error.setVisible(False)
        self.draw_btn.setEnabled(True)
        self.table.setRowCount(len(rows) + 1)

        def put(i, vals):
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if j:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, it)
        for i, r in enumerate(rows):
            put(i, [r['group'], f"{r['n_events']:,}".replace(',', ' '),
                    f"all {r['n_sample']}" if r['census'] else r['n_sample'],
                    '—' if r['n_flagged'] is None else r['n_flagged']])
        put(len(rows), ['Total', f"{tot['n_events']:,}".replace(',', ' '),
                        tot['n_sample'],
                        '—' if tot['n_flagged'] is None else tot['n_flagged']])
        self.n_groups = tot['groups']
        self.flag_available = bool(pv['flag_available'])
        self.note.setText(sr.FLAGGED_NOTE if self.flag_available
                          else sr.LEGACY_NOTE)

    def done(self, result):
        self._timer.stop()
        super().done(result)

    def cell(self, row, col):
        it = self.table.item(row, col)
        return None if it is None else it.text()


class PrecisionReportDialog(QtWidgets.QDialog):
    """Non-modal ``Precision report`` (spec section 11).

    ``source`` supplies the data: ``design`` (dict), ``labels()`` ->
    ``{reviewer: {uuid: (decision, reason)}}``, ``frame(reviewer)`` ->
    ``compute_review_precision`` frame (written to ``review_precision``),
    ``event_line(uuid)`` -> ``'PPOz 01:53:32.9'``, ``db_name`` and
    ``current_reviewer``. ``eventRequested(uuid)`` asks the window to select
    an event (double-click on a disagreement).
    """

    eventRequested = pyqtSignal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Precision report')
        self.setModal(False)
        self.src = source
        self.frames = {}
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel('Precision report')
        self.title.setStyleSheet("font-weight:600;font-size:14px;")
        lay.addWidget(self.title)
        self.header = QtWidgets.QLabel('')
        self.header.setWordWrap(True)
        lay.addWidget(self.header)
        self.sub = QtWidgets.QLabel('')
        self.sub.setWordWrap(True)
        self.sub.setStyleSheet(f"color:{_MUTED};")
        lay.addWidget(self.sub)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('Precision for'))
        self.reviewer_combo = QtWidgets.QComboBox()
        top.addWidget(self.reviewer_combo)
        meaning = QtWidgets.QLabel(sr.PRECISION_MEANING)
        meaning.setWordWrap(True)
        meaning.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        top.addWidget(meaning, 1)
        lay.addLayout(top)
        self.empty = QtWidgets.QLabel(sr.NO_DECISIONS)
        lay.addWidget(self.empty)
        self.body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        self.grid = QtWidgets.QTableWidget(0, 3)
        self.grid.setHorizontalHeaderLabels(['NREM2', 'NREM3', 'All stages'])
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.grid.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.grid.setMaximumHeight(200)
        bl.addWidget(self.grid)
        self.whole = QtWidgets.QLabel('')
        bl.addWidget(self.whole)
        rule = QtWidgets.QHBoxLayout()
        rule.addWidget(QtWidgets.QLabel('Pooling rule   precision of at least'))
        self.thr_spin = QtWidgets.QDoubleSpinBox()
        self.thr_spin.setRange(0.50, 0.99)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(float(_settings().value('review/pool_threshold',
                                                       0.80)))
        rule.addWidget(self.thr_spin)
        rule.addWidget(QtWidgets.QLabel('using the'))
        self.est_combo = QtWidgets.QComboBox()
        self.est_combo.addItems(['point estimate', 'lower 95 % bound'])
        self.est_combo.setCurrentText(str(_settings().value(
            'review/pool_estimate', 'point estimate')))
        rule.addWidget(self.est_combo)
        rule.addWidget(QtWidgets.QLabel('in every region × stage group'))
        rule.addStretch()
        bl.addLayout(rule)
        conv = QtWidgets.QLabel(sr.CONVENTION_NOTE)
        conv.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        bl.addWidget(conv)
        self.verdict = QtWidgets.QLabel('')
        self.verdict.setWordWrap(True)
        bl.addWidget(self.verdict)
        self.reasons_hdr = QtWidgets.QLabel('')
        self.reasons_hdr.setStyleSheet("font-weight:600;")
        bl.addWidget(self.reasons_hdr)
        self.reasons = QtWidgets.QLabel('')
        self.reasons.setStyleSheet(
            "font-family:'IBM Plex Mono',monospace;font-size:11px;")
        bl.addWidget(self.reasons)
        self.agree_hdr = QtWidgets.QLabel('')
        self.agree_hdr.setStyleSheet("font-weight:600;")
        bl.addWidget(self.agree_hdr)
        self.agree = QtWidgets.QLabel('')
        self.agree.setWordWrap(True)
        bl.addWidget(self.agree)
        self.dis_btn = QtWidgets.QPushButton('')
        self.dis_btn.clicked.connect(self._toggle_disagreements)
        bl.addWidget(self.dis_btn)
        self.dis_list = QtWidgets.QListWidget()
        self.dis_list.setMaximumHeight(120)
        self.dis_list.setVisible(False)
        self.dis_list.itemDoubleClicked.connect(
            lambda it: self.eventRequested.emit(str(it.data(Qt.UserRole))))
        bl.addWidget(self.dis_list)
        lay.addWidget(self.body)
        foot = QtWidgets.QHBoxLayout()
        self.footer = QtWidgets.QLabel('')
        self.footer.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        foot.addWidget(self.footer, 1)
        self.copy_btn = QtWidgets.QPushButton('Copy summary')
        self.copy_btn.clicked.connect(self.copy_summary)
        self.export_btn = QtWidgets.QPushButton('Export CSV…')
        self.export_btn.clicked.connect(lambda: self.export_csv())
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.close)
        for b in (self.copy_btn, self.export_btn, close):
            foot.addWidget(b)
        lay.addLayout(foot)
        self.thr_spin.valueChanged.connect(self._rule_changed)
        self.est_combo.currentTextChanged.connect(self._rule_changed)
        self.reviewer_combo.currentTextChanged.connect(
            lambda *_: self._render())
        self.resize(1100, 720)
        self.refresh()

    # ---- data ------------------------------------------------------------
    def threshold(self):
        return float(self.thr_spin.value())

    def estimate(self):
        return self.est_combo.currentText()

    def _rule_changed(self, *_):
        _settings().setValue('review/pool_threshold', self.threshold())
        _settings().setValue('review/pool_estimate', self.estimate())
        self._render()

    def refresh(self):
        """Re-read labels, recompute and write ``review_precision`` for
        every reviewer of the sample, then redraw."""
        self.labels = self.src.labels()
        self.frames = {}
        for rv, lab in sorted(self.labels.items()):
            if rv == 'consensus' or not lab:
                continue
            self.frames[rv] = self.src.frame(rv)
        keep = self.reviewer_combo.currentText() or self.src.current_reviewer
        self.reviewer_combo.blockSignals(True)
        self.reviewer_combo.clear()
        self.reviewer_combo.addItems(sorted(self.frames))
        i = self.reviewer_combo.findText(keep or '')
        self.reviewer_combo.setCurrentIndex(max(0, i))
        self.reviewer_combo.blockSignals(False)
        self.saved_at = sr.now_hhmm()
        self._render()

    # ---- rendering -------------------------------------------------------
    def _render(self):
        d = self.src.design
        ev = sr.EVENT_PLURAL.get(d.get('event_type'), d.get('event_type'))
        self.header.setText(
            f"{d.get('subject') or '—'} · {ev} · {d.get('method')} "
            f"{float(d.get('freq_lower')):g}–{float(d.get('freq_upper')):g} Hz"
            f" · {self.src.run_text}")
        who = []
        for rv in sorted(self.labels):
            lab = self.labels[rv]
            u = sum(1 for v in lab.values() if v[0] == 'unsure')
            who.append(f"{rv}: {len(lab)} decided, {u} unsure")
        self.sub.setText(
            f"Sample of {self.src.n_total} drawn "
            f"{str(d.get('drawn_at') or '')[:10]} (seed {d.get('seed')})"
            + (' · ' + ' · '.join(who) if who else ''))
        has = bool(self.frames)
        self.empty.setVisible(not has)
        self.body.setVisible(has)
        self.copy_btn.setEnabled(has)
        self.export_btn.setEnabled(has)
        self.footer.setText(
            f"Saved to {self.src.db_name} (table review_precision) at "
            f"{self.saved_at}." if has else '')
        if not has:
            return
        rv = self.reviewer_combo.currentText()
        df = self.frames[rv]
        self._fill_grid(df)
        scope = sr._row(df, 'scope', 'all')
        stxt, _lvl = sr.cell_text(scope, 0.0)
        self.whole.setText(f"Whole night   {stxt.rsplit('  n ', 1)[0]}  "
                           f"weighted to the night's events, all regions and "
                           f"stages")
        self.verdict.setText('Verdict        ' + sr.verdict_text(
            df, self.threshold(), self.estimate()))
        lines, (fa, fo) = sr.reason_lines(self.labels.get(rv, {}),
                                          er.REASON_LABEL)
        n_rej = sum(n for _, n in lines)
        self.reasons_hdr.setText(f"Why rejected ({rv}, {n_rej} rejected)")
        top = max([n for _, n in lines] or [1])
        width = max([len(l) for l, _ in lines] or [10])
        self.reason_rows = lines
        self.reasons.setText('\n'.join(
            f"  {l:<{width}}  {n:>3}  {'█' * max(1, round(12 * n / top))}"
            for l, n in lines)
            + f"\n  As RA protocol classes: FP-artifact {fa} · FP-other {fo}")
        self._fill_agreement(rv)

    def _fill_grid(self, df):
        rs = df[df['domain_type'] == 'region_stage']
        regions = [r for r in sr.REGION_ORDER
                   if any(str(x).startswith(r + '|') for x in rs['domain'])]
        self.grid.setRowCount(len(regions))
        self.grid.setVerticalHeaderLabels(regions)
        self.cells = {}
        for i, region in enumerate(regions):
            for j, stage in enumerate(('NREM2', 'NREM3', None)):
                row = (sr._row(df, 'region_stage', f"{region}|{stage}")
                       if stage else sr._row(df, 'region', region))
                text, lvl = sr.cell_text(row, self.threshold(),
                                         self.estimate())
                if stage is None and lvl == 'warn':
                    text, lvl = text[:-len(' below')], None
                it = QtWidgets.QTableWidgetItem(text)
                if lvl == 'warn':
                    it.setForeground(QtGui.QColor(_WARN))
                    f = it.font()
                    f.setWeight(QtGui.QFont.DemiBold)
                    it.setFont(f)
                elif lvl == 'muted':
                    it.setForeground(QtGui.QColor(_MUTED))
                self.grid.setItem(i, j, it)
                self.cells[(region, stage or 'All stages')] = text

    def _fill_agreement(self, rv):
        others = [o for o in self.labels if o not in (rv, 'consensus')
                  and set(self.labels[o]) & set(self.labels.get(rv, {}))]
        self.dis_list.clear()
        self.dis_list.setVisible(False)
        if not others:
            self.agree_hdr.setText('Agreement')
            self.agree.setText(sr.NO_SECOND)
            self.dis_btn.setVisible(False)
            self.agreement = None
            return
        other = max(others, key=lambda o: len(set(self.labels[o])
                                               & set(self.labels[rv])))
        res = sr.agreement(self.labels[rv], self.labels[other])
        self.agreement = res
        self.agree_hdr.setText(f"Agreement between {rv} and {other}   on "
                               f"{res['n_shared']} events both decided")
        self.agree.setText(sr.agreement_text(res))
        n_dis = len(res['disagree'])
        self.dis_btn.setVisible(n_dis > 0)
        self.dis_btn.setText(f"Show the {n_dis} disagreements")
        for u in res['disagree']:
            a, b = self.labels[rv][u], self.labels[other][u]
            it = QtWidgets.QListWidgetItem(
                f"{self.src.event_line(u)} · {rv} "
                f"{er.decision_word(a[0], a[1])} · {other} "
                f"{er.decision_word(b[0], b[1])}")
            it.setData(Qt.UserRole, u)
            self.dis_list.addItem(it)

    def _toggle_disagreements(self):
        self.dis_list.setVisible(not self.dis_list.isVisible())

    # ---- output ------------------------------------------------------------
    def summary_text(self):
        rv = self.reviewer_combo.currentText()
        parts = [self.title.text(), self.header.text(), self.sub.text(),
                 f"Precision for {rv}", self.whole.text(),
                 self.verdict.text(),
                 f"Pooling rule: precision of at least {self.threshold():.2f}"
                 f" using the {self.estimate()} (a lab convention)",
                 self.reasons_hdr.text(), self.reasons.text()]
        if self.agreement:
            parts += [self.agree_hdr.text(), self.agree.text()]
        return '\n'.join(p for p in parts if p)

    def copy_summary(self):
        QtWidgets.QApplication.clipboard().setText(self.summary_text())

    def default_csv_path(self):
        folder = os.path.dirname(os.path.abspath(self.src.db_path))
        return os.path.join(folder, sr.csv_filename(self.src.design))

    def export_csv(self, path=None):
        """Write the CSV; without ``path`` ask with a save dialog prefilled
        next to the database. Returns the path written or ``None``."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export precision', self.default_csv_path(),
                'CSV (*.csv)')
            if not path:
                return None
        rows = sr.csv_rows(self.src.design, self.frames, self.threshold(),
                           self.estimate())
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(sr.CSV_COLUMNS))
            w.writeheader()
            w.writerows(rows)
        self.last_export = (path, len(self.frames))
        if callable(getattr(self.src, 'status', None)):
            self.src.status(f"Exported precision for {len(self.frames)} "
                            f"reviewer{'s' if len(self.frames) != 1 else ''} "
                            f"to {path}.")
        return path
