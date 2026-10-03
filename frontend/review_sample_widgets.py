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


def pool_rule():
    """``(threshold fraction, estimate)`` of the Precision rule setting
    (default 80 % on the point estimate)."""
    st = _settings()
    try:
        t = float(st.value(sr.POOL_THRESHOLD_KEY, 0.80))
    except (TypeError, ValueError):
        t = 0.80
    est = str(st.value(sr.POOL_ESTIMATE_KEY, 'point estimate'))
    return t, (est if est in sr.ESTIMATES else 'point estimate')


class PrecisionRuleDialog(QtWidgets.QDialog):
    """Review ▸ ``Precision rule…`` (R5.5): the pooling rule, saved in the
    existing ``QSettings`` keys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Precision rule')
        lay = QtWidgets.QVBoxLayout(self)
        t, est = pool_rule()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Call a region × stage group '
                                       'trustworthy when its precision is '
                                       'at least'))
        self.pct_spin = QtWidgets.QSpinBox()
        self.pct_spin.setRange(50, 99)
        self.pct_spin.setValue(int(round(100 * t)))
        row.addWidget(self.pct_spin)
        row.addWidget(QtWidgets.QLabel('% using the'))
        self.est_combo = QtWidgets.QComboBox()
        self.est_combo.addItems(list(sr.ESTIMATES))
        self.est_combo.setCurrentText(est)
        row.addWidget(self.est_combo)
        row.addWidget(QtWidgets.QLabel('.'))
        lay.addLayout(row)
        note = QtWidgets.QLabel(sr.CONVENTION_NOTE)
        note.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        lay.addWidget(note)
        box = QtWidgets.QDialogButtonBox()
        box.addButton('Cancel', QtWidgets.QDialogButtonBox.RejectRole)
        self.save_btn = box.addButton('Save',
                                      QtWidgets.QDialogButtonBox.AcceptRole)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def save(self):
        st = _settings()
        st.setValue(sr.POOL_THRESHOLD_KEY, self.pct_spin.value() / 100.0)
        st.setValue(sr.POOL_ESTIMATE_KEY, self.est_combo.currentText())
        self.accept()


class PrecisionReportDialog(QtWidgets.QDialog):
    """Non-modal ``Precision report``, short form (UX spec R5.5).

    Title line, one sentence (its reasons and ``unsure`` open a list of
    those events), the weighted precision line, a verdict, a region × stage
    table of percentages, the second-reviewer line, and Copy summary /
    Export CSV… / Close. ``source`` supplies ``design``, ``n_total``,
    ``db_path``, ``current_reviewer``, ``labels()`` ->
    ``{reviewer: {uuid: (decision, reason)}}``, ``frame(reviewer)`` (the
    ``compute_review_precision`` frame, written to ``review_precision``),
    ``event_info(uuid)`` -> ``(channel, start_time, stage)`` and
    ``comments(reviewer)`` -> ``{uuid: comment}``. Double-clicking a listed
    event emits ``eventRequested(uuid)``.
    """

    eventRequested = pyqtSignal(str)
    MAX_LIST_ROWS = 8

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.src = source
        subject = source.design.get('subject') or '—'
        self.setWindowTitle(f'Precision report · {subject}')
        self.setModal(False)
        self.frames = {}
        self._open_list = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(6)
        trow = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel('')
        self.title.setStyleSheet("font-weight:600;font-size:14px;")
        trow.addWidget(self.title)
        # the reviewer picker, only with two or more reviewers (R5.5)
        self.reviewer_combo = QtWidgets.QComboBox()
        self.reviewer_combo.setVisible(False)
        self.reviewer_combo.currentTextChanged.connect(
            lambda *_: self._render())
        trow.addWidget(self.reviewer_combo)
        trow.addStretch()
        lay.addLayout(trow)
        self.empty = QtWidgets.QLabel(sr.NO_DECISIONS)
        lay.addWidget(self.empty)
        self.body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        self.sentence = QtWidgets.QLabel('')
        self.sentence.setTextFormat(Qt.RichText)
        self.sentence.setWordWrap(True)
        self.sentence.setStyleSheet("font-size:13px;")
        self.sentence.linkActivated.connect(self.toggle_list)
        bl.addWidget(self.sentence)
        self.precision = QtWidgets.QLabel('')
        self.precision.setStyleSheet("font-size:13px;")
        bl.addWidget(self.precision)
        self.verdict = QtWidgets.QLabel('')
        self.verdict.setWordWrap(True)
        bl.addWidget(self.verdict)
        self.grid = QtWidgets.QTableWidget(0, 0)
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.grid.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.grid.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bl.addWidget(self.grid, 0, Qt.AlignLeft)
        self.list_hdr = QtWidgets.QLabel('')
        self.list_hdr.setStyleSheet("font-weight:600;")
        bl.addWidget(self.list_hdr)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.itemDoubleClicked.connect(
            lambda it: self.eventRequested.emit(str(it.data(Qt.UserRole))))
        bl.addWidget(self.event_list)
        self.list_hint = QtWidgets.QLabel(sr.LIST_HINT)
        self.list_hint.setStyleSheet(f"color:{_MUTED};font-size:11px;")
        bl.addWidget(self.list_hint)
        self.second = QtWidgets.QLabel('')
        self.second.setWordWrap(True)
        bl.addWidget(self.second)
        lay.addWidget(self.body)
        lay.addStretch()
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch()
        self.copy_btn = QtWidgets.QPushButton('Copy summary')
        self.copy_btn.clicked.connect(self.copy_summary)
        self.export_btn = QtWidgets.QPushButton('Export CSV…')
        self.export_btn.setToolTip(sr.EXPORT_CSV_TIP)
        self.export_btn.clicked.connect(lambda: self.export_csv())
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.clicked.connect(self.close)
        for b in (self.copy_btn, self.export_btn, self.close_btn):
            foot.addWidget(b)
        lay.addLayout(foot)
        self._show_list(None)
        self.refresh()

    # ---- data ------------------------------------------------------------
    def threshold(self):
        return pool_rule()[0]

    def estimate(self):
        return pool_rule()[1]

    def refresh(self):
        """Re-read labels, recompute and write ``review_precision`` for
        every reviewer of the sample, then redraw."""
        self.labels = self.src.labels()
        self.frames = {}
        for rv, lab in sorted(self.labels.items()):
            if rv == 'consensus' or not lab:
                continue
            self.frames[rv] = self.src.frame(rv)
        cur = self.src.current_reviewer
        keep = self.reviewer_combo.currentText() or cur
        names = sorted(self.frames)
        if cur and cur not in names and cur in self.labels:
            names.append(cur)
        self.reviewer_combo.blockSignals(True)
        self.reviewer_combo.clear()
        self.reviewer_combo.addItems(names)
        i = self.reviewer_combo.findText(keep or '')
        if i < 0:
            i = self.reviewer_combo.findText(cur or '')
        self.reviewer_combo.setCurrentIndex(max(0, i))
        self.reviewer_combo.blockSignals(False)
        self._render()

    def shown_reviewer(self):
        """The reviewer the report is about: the current one, unless the
        (enabled) picker shows another."""
        if self.reviewer_combo.isVisible() and \
                self.reviewer_combo.isEnabled() and \
                self.reviewer_combo.currentText():
            return self.reviewer_combo.currentText()
        return self.src.current_reviewer

    # ---- rendering -------------------------------------------------------
    def _render(self):
        d = self.src.design
        cur = self.src.current_reviewer
        n_total = self.src.n_total
        second, finished = sr.second_reviewer_line(cur, self.labels, n_total)
        reviewers = [r for r in self.labels if r != 'consensus'
                     and self.labels[r]]
        many = len(reviewers) >= 2
        self.reviewer_combo.setVisible(many)
        self.reviewer_combo.setEnabled(many and finished)
        self.reviewer_combo.setToolTip('' if finished
                                       else sr.PICKER_LOCKED_TIP)
        if many and not finished:
            self.reviewer_combo.blockSignals(True)
            self.reviewer_combo.setCurrentText(cur or '')
            self.reviewer_combo.blockSignals(False)
        rv = self.shown_reviewer()
        self.title.setText(sr.report_title(d, rv)[:-len(str(rv))].rstrip()
                           if many else sr.report_title(d, rv))
        self._title_text = sr.report_title(d, rv)
        self.second.setText(second)
        has = rv in self.frames
        self.empty.setVisible(not has)
        self.body.setVisible(has)
        self.copy_btn.setEnabled(has)
        self.export_btn.setEnabled(has)
        if not has:
            return
        df = self.frames[rv]
        labels = self.labels.get(rv, {})
        self._sentence_text = sr.report_sentence(rv, labels, n_total,
                                                 er.REASON_LABEL)
        self.sentence.setText(sr.report_sentence(
            rv, labels, n_total, er.REASON_LABEL, links=True, accent=_ACCENT))
        line = sr.precision_line(df) or ''
        self.precision.setText(line)
        self.precision.setToolTip(sr.precision_tip(rv) if line else '')
        text, level = sr.verdict_text(df, self.threshold(), self.estimate())
        self.verdict.setText(text)
        self.verdict_level = level
        self.verdict.setStyleSheet(
            "font-size:13px;font-weight:600;color:"
            + ('#69b35d' if level == 'ok' else _WARN) + ";")
        self._fill_grid(df)
        if self._open_list is not None:
            self._show_list(self._open_list)

    def _fill_grid(self, df):
        rs_rows = df[df['domain_type'] == 'region_stage']
        regions = [r for r in sr.REGION_ORDER
                   if any(str(x).startswith(r + '|') for x in rs_rows['domain'])]
        stages = [st for st in sr.STAGE_ORDER
                  if any(str(x).endswith('|' + st) for x in rs_rows['domain'])]
        stages += sorted({str(x).split('|')[1] for x in rs_rows['domain']}
                         - set(stages))
        cols = stages + ['All stages']
        self.grid.clear()
        self.grid.setRowCount(len(regions))
        self.grid.setColumnCount(len(cols))
        self.grid.setHorizontalHeaderLabels(cols)
        self.grid.setVerticalHeaderLabels(regions)
        self.cells = {}
        for i, region in enumerate(regions):
            for j, stage in enumerate(stages + [None]):
                row = (sr._row(df, 'region_stage', f"{region}|{stage}")
                       if stage else sr._row(df, 'region', region))
                text, lvl, tip = sr.cell_text(row, self.threshold(),
                                              self.estimate(),
                                              mark=stage is not None)
                it = QtWidgets.QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                it.setToolTip(tip)
                if lvl == 'warn':
                    it.setForeground(QtGui.QColor(_WARN))
                    f = it.font()
                    f.setWeight(QtGui.QFont.DemiBold)
                    it.setFont(f)
                elif lvl == 'muted':
                    it.setForeground(QtGui.QColor(_MUTED))
                self.grid.setItem(i, j, it)
                self.cells[(region, stage or 'All stages')] = (text, tip)
        self.grid.resizeColumnsToContents()
        h = (self.grid.horizontalHeader().height()
             + sum(self.grid.rowHeight(i) for i in range(len(regions))) + 4)
        w = (self.grid.verticalHeader().width()
             + sum(self.grid.columnWidth(j) for j in range(len(cols))) + 4)
        self.grid.setFixedSize(max(w, 200), h)

    # ---- the reason / unsure list ------------------------------------------
    def toggle_list(self, key):
        """A link in the sentence: open its list, or close it when it is the
        one already open."""
        self._show_list(None if key == self._open_list else key)

    def _show_list(self, key):
        self._open_list = key
        self.event_list.clear()
        on = key is not None and self.shown_reviewer() in self.frames
        for w in (self.list_hdr, self.event_list, self.list_hint):
            w.setVisible(on)
        if not on:
            return
        rv = self.shown_reviewer()
        labels = self.labels.get(rv, {})
        if key == 'unsure':
            uuids = [u for u, (d, _r) in labels.items() if d == 'unsure']
            self.list_hdr.setText(f"Marked unsure ({len(uuids)})")
        else:
            tok = key.split(':', 1)[1]
            uuids = [u for u, (d, r) in labels.items()
                     if d == 'reject' and (r or '') == tok]
            lab = (er.REASON_LABEL.get(tok, tok) if tok
                   else 'no reason given').lower()
            self.list_hdr.setText(f"Rejected as {lab} ({len(uuids)})")
        comments = self.src.comments(rv) if callable(
            getattr(self.src, 'comments', None)) else {}
        rows = []
        for u in uuids:
            ch, start, stage = self.src.event_info(u)
            rows.append((float(start or 0), u, ch, start, stage))
        for _t, u, ch, start, stage in sorted(rows):
            text = f"{ch} · {er.fmt_hms1(start)} · {stage}"
            c = str(comments.get(u) or '').strip()
            if c:
                text += f"  “{c if len(c) <= 60 else c[:59] + '…'}”"
            it = QtWidgets.QListWidgetItem(text)
            it.setData(Qt.UserRole, u)
            self.event_list.addItem(it)
        rh = self.event_list.sizeHintForRow(0) if uuids else 18
        self.event_list.setFixedHeight(
            rh * min(max(len(uuids), 1), self.MAX_LIST_ROWS) + 6)

    # ---- output ------------------------------------------------------------
    def summary_text(self):
        """Title line, sentence, precision line and verdict, one per line."""
        rv = self.shown_reviewer()
        if rv not in self.frames:
            return ''
        return '\n'.join([self._title_text, self._sentence_text,
                          self.precision.text(), self.verdict.text()])

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
