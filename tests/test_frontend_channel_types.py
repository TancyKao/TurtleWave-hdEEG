#!/usr/bin/env python3
"""Headless acceptance checks for ``_scratch/design/compumedics-channels-spec.md``.

Section 6 covers the interpolated-channel marks (plan section C, GUI half):
the Dataset Information line, italic + tooltip items in turtlewave_gui, the
review GUI's " ~" suffix, and the summary / tooltip / run-note text.

Sections 1-5 of the spec: the "Show non-EEG channels" checkbox on the four
detection tabs, the Setup tab's Dataset Information text, scalp regions from
10-20 / 10-5 labels, the review GUI's default channels and filter-dock list,
and the load-failure message. Stub datasets stand in for real recordings;
section 5 writes small EEGLAB files with ``tests/eeglab_fixture.py`` into a
temporary folder.

Run with:
    QT_QPA_PLATFORM=offscreen python tests/test_frontend_channel_types.py

TurtleWaveGUI.__init__ reassigns sys.stdout to its log widget, so the real
stdout is captured before any window is built and every report goes there.
"""
import datetime
import os
import sys
import tempfile
import types

# Windows consoles and CI pipes default to cp1252, which cannot encode the
# flag glyph and middle dots these checks print; replace rather than crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

REAL_STDOUT = sys.stdout          # captured BEFORE any TurtleWaveGUI exists


def say(*a):
    print(*a, file=REAL_STDOUT, flush=True)


FAILURES = []
CHECKS = [0]


def check(item, label, ok, detail=""):
    CHECKS[0] += 1
    tag = "PASS" if ok else "FAIL"
    say(f"  [{tag}] ({item}) {label}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"({item}) {label}: {detail}")


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import pandas as pd                                           # noqa: E402
from PyQt5 import QtWidgets                                   # noqa: E402
from PyQt5.QtCore import Qt                                   # noqa: E402

import frontend.turtlewave_gui as tg                          # noqa: E402
import frontend.eeg_review_gui as rg                          # noqa: E402
from frontend.channel_types import (ChannelTypeSummary,       # noqa: E402
                                    default_review_channels)
from frontend.waveform_loader import WaveformBackgroundLoader  # noqa: E402
from turtlewave_hdEEG.utils import region_from_label          # noqa: E402
from eeglab_fixture import write_set                          # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
TMP = tempfile.mkdtemp(prefix='tw_chantypes_')

# ---------------------------------------------------------------------------
# The reference Compumedics file's channels (sub-02dg ses-1), as loaded
# ---------------------------------------------------------------------------
REF_LABELS = (
    'Fp1 Fpz Fp2 AF3 AF4 F11 F7 F5 F3 F1 Fz F2 F4 F6 F8 F12 FT11 FC5 FC3 FC1 '
    'FCz FC2 FC4 FC6 FT12 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPz CP2 '
    'CP4 CP6 TP8 M1 M2 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 '
    'Cb1 Cb2 AFp1 AFp2 AF7 AF5 AFz AF6 AF8 AFF5h AFF3h AFF1h AFF2h AFF4h AFF6h '
    'F9 F10 FFT7h FFC5h FFC3h FFC1h FFC2h FFC4h FFC6h FFT8h FT9 FT7 FT8 FT10 '
    'FTT7h FCC5h FCC3h FCC1h FCC2h FCC4h FCC6h FTT8h TTP7h CCP5h CCP3h CCP1h '
    'CCP2h CCP4h CCP6h TTP8h TPP7h CPP5h CPP3h CPP1h CPP2h CPP4h CPP6h TPP8h '
    'P9 P10 PPO3h PPO1h PPO2h PPO4h PO9 PO5 PO1 PO2 PO6 PO10 Cbz AFp7 AFpz '
    'AFp8 AFF7 AFF8 F7h F3h F4h F8h FFC5 FFC3 FFCz FFC4 FFC6 FT9h FT7h FC5h '
    'FC1h FC2h FC6h FT8h FT10h FTT9h FCC5 FCC1 FCC2 FCC6 FTT10h T7h C3h C4h '
    'T8h TTP7 CCP3 CCP4 TTP8 CP5h CP1h CP2h CP6h TPP7 CPP3 CPPz CPP4 TPP8 P7h '
    'P3h P4h P8h PPO9h PPO7h PPO5h PPO6h PPO8h POO10h PO9h PO10h POO9h POO5h '
    'POOz POO6h PPO10h OCb1h OCb2h FP1h Fp2h AFp3 AFp4 AF1 AF2 AFF7h AFF8h F9h '
    'F5h F1h F2h F6h F10h FFT11h FFt9h FFT7 FFC1 FFC2 FFT8 FFT10h FFT12h FC3h '
    'FC4h FTT11h FTT7 FCC3 FCCz FCC4 FTT8 FTT12h C5h C1h C2h C6h CCP5 CCP1 '
    'CCP2 CCP6 TP7h CP3h CP4h TP8h TPP9h CPP5 CPP1 CPP2 CPP6 TPP10h P9h P5h '
    'P1h P2h P6h P10h PPOz POO7 POO1 POO2 POO8 O1h O2h OCB1 OCb2 REF VEOG HEOG '
    'Abdo_Effort Thor_Effort Resp_Flow Snore do_not_use1 ECG EMGChin '
    'EMGLeftLeg EMGRightLeg BodyPosition Resp_Temp ECG_2 RightEDB LeftEDB '
    'OxStatus HR SpO2_OSat BodyPosition_2').split()
REF_TYPES = ['EEG'] * 257 + [
    'EOG', 'EOG', 'Respiratory', 'Respiratory', 'NasalPressure', 'Snoring',
    'MISC', 'ECG', 'EMG', 'EMG', 'EMG', 'MISC', 'Respiratory', 'ECG', 'EMG',
    'EMG', 'PNS', 'PNS', 'MISC', 'MISC']
assert len(REF_LABELS) == 277 and len(REF_TYPES) == 277
REF_FILE = 'sub-02dg_ses-1_task-psg_run-1_desc-clean_eeg.set'


def ref_events():
    """189 boundary events removing 6,460.1 s (1,615,025 samples at 250 Hz)."""
    durations = [8545] * 188
    durations.append(1615025 - sum(durations))
    onsets = [1000 + 20000 * i for i in range(189)]
    return {'onsets': onsets, 'types': ['boundary'] * 189,
            'durations': durations}


def stub(channels, chan_type=None, s_freq=250.0, n_samples=250 * 600,
         start=datetime.datetime(2022, 9, 12, 22, 28, 52), reference=None,
         event=None, omit=()):
    header = {'n_samples': n_samples, 'start_time': start, 's_freq': s_freq,
              'chan_type': chan_type, 'reference': reference, 'event': event}
    for key in omit:
        header.pop(key, None)
    return types.SimpleNamespace(channels=list(channels), sampling_rate=s_freq,
                                 header=header)


#: 37 interpolated channels, first three as in the plan's example line.
REF_INTERP = ['AF3', 'F3', 'F1', 'Cz'] + [
    lab for lab in REF_LABELS[:257]
    if lab not in ('AF3', 'F3', 'F1', 'Cz')][40:73]
assert len(REF_INTERP) == 37 and len(set(REF_INTERP)) == 37


def reference_stub():
    ds = stub(REF_LABELS, REF_TYPES, n_samples=4827027,
              reference={'ref': 'average', 'n_good': 220}, event=ref_events())
    ds.header['interp_channels'] = list(REF_INTERP)
    return ds


class LogSpy:
    """Captures write_log lines off a TurtleWaveGUI without a log widget."""

    def __init__(self, win):
        self.lines = []
        win.write_log = self.lines.append

    def clear(self):
        self.lines.clear()

    def text(self):
        return "\n".join(str(x) for x in self.lines)


class CriticalSpy:
    """Replaces QMessageBox.critical and records each dialog body."""

    def __init__(self):
        self.messages = []
        self._orig = QtWidgets.QMessageBox.critical

    def __enter__(self):
        def fake(parent, title, text, *a, **k):
            self.messages.append((title, text))
            return QtWidgets.QMessageBox.Ok
        QtWidgets.QMessageBox.critical = staticmethod(fake)
        return self

    def __exit__(self, *exc):
        QtWidgets.QMessageBox.critical = self._orig


say("=" * 78)
say("Headless acceptance: compumedics-channels-spec.md sections 1-5")
say("=" * 78)

win = tg.TurtleWaveGUI()
spy = LogSpy(win)
TAB_KEYS = ('spindle', 'sw', 'kc', 'pac')


def load(window, ds, path=None):
    """What load_data does on success, then the main-thread slot."""
    window.dataset = ds
    window.available_channels = ds.channels
    window.data_file_path = path or os.path.join(TMP, 'stub.set')
    window.output_dir = TMP
    window.annot_file_path = os.path.join(TMP, 'wonambi', 'stub.xml')
    window.update_after_load()


def page_of(window, widget):
    for i in range(window.tabs.count()):
        page = window.tabs.widget(i)
        if page.isAncestorOf(widget):
            return page
    return None


def texts(lst):
    return [lst.item(i).text() for i in range(lst.count())]


DET_AVAIL = lambda w: {'spindle': texts(w.available_list),   # noqa: E731
                       'sw': texts(w.sw_available_list),
                       'kc': texts(w.kc_available_list)}
DET_SEL = lambda w: {'spindle': w.selected_list,              # noqa: E731
                     'sw': w.sw_selected_list, 'kc': w.kc_selected_list}

# ======================================================================= 1
say("\n== Section 1: non-EEG channels in the detection channel lists")
five = stub(['Fp1', 'Cz', 'VEOG', 'ECG', 'Pz'],
            ['EEG', 'EEG', 'EOG', 'ECG', 'EEG'])
load(win, five)
win.update_channel_lists()
avail = DET_AVAIL(win)
check('1.1', "three detection Available lists hold Fp1, Cz, Pz",
      all(v == ['Fp1', 'Cz', 'Pz'] for v in avail.values()), repr(avail))

checks_ok = []
for key in TAB_KEYS:
    cb = win.non_eeg_checks[key]
    page = page_of(win, cb)
    checks_ok.append((key, page is not None and cb.isVisibleTo(page),
                      cb.isChecked(), cb.text()))
check('1.2', "checkbox visible on all four tabs, unticked, 'Show non-EEG channels (2)'",
      all(p and not c and t == 'Show non-EEG channels (2)'
          for _, p, c, t in checks_ok), repr(checks_ok))

win.non_eeg_checks['sw'].setChecked(True)
avail = DET_AVAIL(win)
check('1.3', "ticking Slow Wave ticks the other three",
      all(win.non_eeg_checks[k].isChecked() for k in TAB_KEYS),
      repr({k: win.non_eeg_checks[k].isChecked() for k in TAB_KEYS}))
check('1.3', "all three Available lists list every channel in file order",
      all(v == ['Fp1', 'Cz', 'VEOG', 'ECG', 'Pz'] for v in avail.values()),
      repr(avail))

win.non_eeg_checks['spindle'].setChecked(False)
win.selected_channels = []
win.update_channel_lists()
win.add_all_channels()
check('1.4', "unticked Add All >> on Spindle selects Fp1, Cz, Pz",
      win.selected_channels == ['Fp1', 'Cz', 'Pz'], repr(win.selected_channels))

win.selected_channels = []
win.non_eeg_checks['kc'].setChecked(True)
for i in range(win.available_list.count()):
    it = win.available_list.item(i)
    it.setSelected(it.text() == 'ECG')
win.add_channels()
win.non_eeg_checks['kc'].setChecked(False)
sel_lists = DET_SEL(win)
avail = DET_AVAIL(win)
ecg_tips = {k: [lst.item(i).toolTip() for i in range(lst.count())
                if lst.item(i).text() == 'ECG'] for k, lst in sel_lists.items()}
check('1.5', "ECG stays in selected_channels after unticking",
      'ECG' in win.selected_channels, repr(win.selected_channels))
check('1.5', "ECG in all three Selected lists with tooltip 'Non-EEG channel (ECG)'",
      all(v == ['Non-EEG channel (ECG)'] for v in ecg_tips.values()),
      repr(ecg_tips))
check('1.5', "ECG absent from all Available lists",
      all('ECG' not in v for v in avail.values()), repr(avail))
for i in range(win.selected_list.count()):
    it = win.selected_list.item(i)
    it.setSelected(it.text() == 'ECG')
win.remove_channels()
check('1.5', "removing ECG while unticked: gone from Selected, still hidden",
      'ECG' not in win.selected_channels
      and all('ECG' not in v for v in DET_AVAIL(win).values()),
      repr((win.selected_channels, DET_AVAIL(win)['spindle'])))

spy.clear()
win._log_non_eeg_selection(['Cz', 'ECG', 'VEOG'])
check('1.5', "run-start note names the non-EEG channels",
      spy.lines == ['Note: 2 non-EEG channels selected: ECG, VEOG.'],
      repr(spy.lines))

for label, ct in (("chan_type=None", None), ("chan_type all ''", ['', '', ''])):
    load(win, stub(['C3', 'C4', 'ECG'], ct))
    avail = DET_AVAIL(win)
    hidden = [not win.non_eeg_checks[k].isVisibleTo(page_of(win, win.non_eeg_checks[k]))
              for k in TAB_KEYS]
    check('1.6', f"{label}: every channel listed",
          all(v == ['C3', 'C4', 'ECG'] for v in avail.values()), repr(avail))
    check('1.6', f"{label}: checkbox hidden on all four tabs", all(hidden),
          repr(hidden))

load(win, five)
win.non_eeg_checks['pac'].setChecked(True)
load(win, stub(['Fz', 'EMG1'], ['EEG', 'EMG']))
check('1.7', "loading a second file resets the checkbox to unticked",
      not any(win.non_eeg_checks[k].isChecked() for k in TAB_KEYS)
      and win.show_non_eeg is False,
      repr({k: win.non_eeg_checks[k].isChecked() for k in TAB_KEYS}))

load(win, five)
win.pac_available_channels = ['Cz', 'ECG', 'E999']
win.pac_selected_channels = []
win.update_pac_channel_lists()
pac_unticked = texts(win.pac_available_list)
win.non_eeg_checks['pac'].setChecked(True)
pac_ticked = texts(win.pac_available_list)
check('1.8', "PAC unticked shows Cz, E999", pac_unticked == ['Cz', 'E999'],
      repr(pac_unticked))
check('1.8', "PAC ticked shows Cz, ECG, E999",
      pac_ticked == ['Cz', 'ECG', 'E999'], repr(pac_ticked))
win.non_eeg_checks['pac'].setChecked(False)
win.dataset = None
win.update_pac_channel_lists()
check('1.8', "PAC with no dataset: unfiltered, checkbox hidden",
      texts(win.pac_available_list) == ['Cz', 'ECG', 'E999']
      and not win.non_eeg_checks['pac'].isVisibleTo(
          page_of(win, win.non_eeg_checks['pac'])),
      repr(texts(win.pac_available_list)))

# ======================================================================= 2
say("\n== Section 2: Dataset Information text")
out_dir = os.path.join(TMP, 'out')
annot = os.path.join(out_dir, 'wonambi', 'ref.xml')


def info_for(ds, path):
    win.dataset = ds
    win.available_channels = ds.channels
    win.data_file_path = path
    win.output_dir = out_dir
    win.annot_file_path = annot
    win.info_text.setText("No dataset loaded.")
    win.update_dataset_info()
    return win.info_text.toPlainText()


EXPECTED_1 = f"""File: {REF_FILE}
Recording start: 2022-09-12 22:28:52
Recording end: 2022-09-13 05:38:20
Signal duration: 321.8 min (19,308.1 s)
Removed data: 107.7 min at 189 boundary events (original recording 429.5 min)
Sampling rate: 250 Hz
Channels: 277 (257 EEG, 20 other)
Reference: average of 220 of 257 EEG channels (as stored in the file; not changed by TurtleWave)
Interpolated channels: 37 ({', '.join(REF_INTERP[:10])} and 27 more)
EEG channels: Fp1, Fpz, Fp2, AF3, AF4, F11, F7, F5, F3, F1 and 247 more
Other channels: VEOG, HEOG, Abdo_Effort, Thor_Effort, Resp_Flow, Snore, do_not_use1, ECG, EMGChin, EMGLeftLeg, EMGRightLeg, BodyPosition, Resp_Temp, ECG_2, RightEDB, LeftEDB, OxStatus, HR, SpO2_OSat, BodyPosition_2

Output directory: {out_dir}
Annotation file: {annot}"""

got = info_for(reference_stub(), os.path.join('/data', REF_FILE))
diff = [(a, b) for a, b in zip(got.splitlines(), EXPECTED_1.splitlines()) if a != b]
check('2.1', "reference file text matches block 1 line for line",
      got == EXPECTED_1, repr(diff) or repr(got))

egi = [f"E{i}" for i in range(1, 257)] + ['Cz']
EXPECTED_2 = f"""File: sub-01_task-sleep_eeg.set
Recording start: 2021-03-04 22:10:05
Recording end: 2021-03-05 06:12:35
Signal duration: 482.5 min (28,950.0 s)
Removed data: none
Sampling rate: 500 Hz
Channels: 257 (channel types not stated in the file; all listed as EEG)
Reference: Cz (as stored in the file; not changed by TurtleWave)
Interpolated channels: none stated in the file
Channels: E1, E2, E3, E4, E5, E6, E7, E8, E9, E10 and 247 more

Output directory: {out_dir}
Annotation file: {annot}"""
got = info_for(stub(egi, None, s_freq=500.0, n_samples=500 * 28950,
                    start=datetime.datetime(2021, 3, 4, 22, 10, 5),
                    reference='Cz', event=None),
               '/data/sub-01_task-sleep_eeg.set')
diff = [(a, b) for a, b in zip(got.splitlines(), EXPECTED_2.splitlines()) if a != b]
check('2.2', "uncut EGI text matches block 2 line for line",
      got == EXPECTED_2, repr(diff) or repr(got))


def preview_line(text):
    lines = [ln for ln in text.splitlines() if ln.startswith('Channels:')]
    return lines[-1] if lines else ''


import re                                                     # noqa: E402
got = info_for(stub(['Fz', 'Cz', 'Pz', 'Oz']), '/d/four.set')
line = preview_line(got)
check('2.3', "4 channels: preview lists all four, no 'more', no negative",
      line == 'Channels: Fz, Cz, Pz, Oz' and 'more' not in line
      and not re.search(r'-\d', line), repr(line))

got = info_for(stub([f"C{i}" for i in range(1, 12)]), '/d/eleven.set')
check('2.4', "11 channels: preview ends 'and 1 more'",
      preview_line(got).endswith('and 1 more'), repr(preview_line(got)))

try:
    got = info_for(stub(['Fz', 'Cz'], omit=('chan_type', 'reference', 'event')),
                   '/d/bare.set')
    ok = ('channel types not stated' in got
          and 'Reference: not stated in the file' in got
          and 'Removed data: none' in got)
    check('2.5', "missing header keys fall through to 'not stated' / 'none'",
          ok, repr(got))
except Exception as e:                                        # noqa: BLE001
    check('2.5', "missing header keys fall through without an exception",
          False, repr(e))

spy.clear()
bad = reference_stub()
bad.header['event'] = 'not an event table'
got = info_for(bad, os.path.join('/data', REF_FILE))
labels = [ln.split(':', 1)[0] for ln in got.splitlines() if ln]
check('2.6', "unreadable timeline: 'Removed data: could not be read (see log)'",
      'Removed data: could not be read (see log)' in got, repr(got))
check('2.6', "every other line still present",
      labels == ['File', 'Recording start', 'Recording end', 'Signal duration',
                 'Removed data', 'Sampling rate', 'Channels', 'Reference',
                 'Interpolated channels', 'EEG channels', 'Other channels', 'Output directory',
                 'Annotation file'], repr(labels))
check('2.6', "the reason went to the log",
      'boundary events could not be read' in spy.text(), spy.text())

summary = ChannelTypeSummary(REF_LABELS, REF_TYPES)
load(win, reference_stub(), path=os.path.join('/data', REF_FILE))
info = win.info_text.toPlainText()
check('2.7', "count line and checkbox label from one helper",
      summary.count_line() == '277 (257 EEG, 20 other)'
      and 'Channels: 277 (257 EEG, 20 other)' in info
      and win.non_eeg_checks['spindle'].text() == summary.checkbox_label()
      == 'Show non-EEG channels (20)',
      repr((summary.count_line(), win.non_eeg_checks['spindle'].text())))
check('1.x', "reference tooltip lists types in first-appearance order",
      win.non_eeg_checks['pac'].toolTip() ==
      'The file marks 20 channels as non-EEG: EOG (2), Respiratory (3), '
      'NasalPressure (1), Snoring (1), MISC (4), ECG (2), EMG (5), PNS (2). '
      'They are hidden because the sleep-event detectors expect scalp EEG. '
      'Tick to list them.', repr(win.non_eeg_checks['pac'].toolTip()))

# ======================================================================= 3
say("\n== Section 3: scalp regions from 10-20 / 10-5 labels")
TABLE = {
    'frontal': 'Fp1 Fpz FP1h AFp3 AF7 AFz AFF5h F3 Fz F11 F9h FFC3h FFCz',
    'central': 'FC3 FCz FC5h FCC3h FCCz C3 Cz C5h CCP5h CP5 CPz CP1h',
    'parietal': 'CPP3h CPPz P3 Pz P9 PPO1h PPOz PO3 POz PO9 PO10h',
    'occipital': 'POO1 POOz POO9h O1 Oz O1h Iz OCb1h OCB1 OCb2',
    'temporal': 'T7 T8h FT9 FT11 FT7h FFT7h FFt9h FTT9h TTP7h TP7 TP8h TPP9h',
    'neck': 'Cb1 Cb2 Cbz',
}
cases = [(lab, reg) for reg, labs in TABLE.items() for lab in labs.split()]
cases += [(lab, 'other') for lab in
          ['M1', 'M2', 'A1', 'REF', 'Nz', 'E12', 'ECG', 'VEOG', 'EMGChin',
           'do_not_use1', 'SpO2_OSat', '', None]]
cases += [('EEG C3-M2', 'central'), ('C4-A1', 'central')]
wrong = [(lab, reg, region_from_label(lab)) for lab, reg in cases
         if region_from_label(lab) != reg]
check('3.1', f"table test, {len(cases)} labels", not wrong, repr(wrong))

eeg_labels = summary.eeg
counts = {}
for lab in eeg_labels:
    r = region_from_label(lab)
    counts[r] = counts.get(r, 0) + 1
others = [lab for lab in eeg_labels if region_from_label(lab) == 'other']
check('3.2', "257 EEG labels: only M1, M2, REF are 'other'",
      len(eeg_labels) == 257 and others == ['M1', 'M2', 'REF'], repr(others))
check('3.2', "region counts 67/64/58/44/18/3/3",
      counts == {'frontal': 67, 'central': 64, 'parietal': 58, 'temporal': 44,
                 'occipital': 18, 'neck': 3, 'other': 3}, repr(counts))

check('3.3', "_region_for('E60') still 'frontal'", rg._region_for('E60') == 'frontal',
      rg._region_for('E60'))
check('3.3', "_region_for('FC3') is 'central'", rg._region_for('FC3') == 'central',
      rg._region_for('FC3'))

rows = []
for ch in ['Fz', 'Cz', 'Pz', 'T7', 'O1']:
    for k in range(5):
        rows.append({'channel': ch, 'start_time': 10.0 * k + 1,
                     'end_time': 10.0 * k + 2, 'max_amp': 40.0 + k,
                     'min_amp': -40.0, 'peak2peak_amp': 80.0 + k})
qc = rg.compute_channel_qc(pd.DataFrame(rows), scored_minutes=60.0)
qc_regions = (dict(zip(qc['channel'], qc['region'])) if 'channel' in qc.columns
              else dict(zip(qc.index, qc['region'])))
check('3.4', "compute_channel_qc regions without coordinates",
      [qc_regions.get(c) for c in ['Fz', 'Cz', 'Pz', 'T7', 'O1']]
      == ['frontal', 'central', 'parietal', 'temporal', 'occipital'],
      repr(qc_regions))

win.database_channels = ['T7', 'FT9', 'F3']
check('3.5', "map_regions_to_channels(['Temporal'])",
      win.map_regions_to_channels(['Temporal']) == ['T7', 'FT9'],
      repr(win.map_regions_to_channels(['Temporal'])))

# ======================================================================= 4
say("\n== Section 4: review GUI default channels and channel list")
egi_plus = [f"E{i}" for i in range(1, 257)] + ['Cz']
cases4 = [
    ('4.1', default_review_channels(egi_plus), ['E112', 'E118', 'Cz']),
    ('4.2', default_review_channels(REF_LABELS, REF_TYPES), ['Cz', 'Fz', 'Pz']),
    ('4.3', default_review_channels(['Fz', 'C3', 'C4', 'Pz']), ['Fz', 'Pz', 'C3']),
    ('4.4', default_review_channels(['VEOG', 'ECG', 'C3', 'C4'],
                                    ['EOG', 'ECG', 'EEG', 'EEG']), ['C3', 'C4']),
    ('4.4', default_review_channels(['VEOG', 'ECG', 'C3', 'C4'], None),
     ['VEOG', 'ECG', 'C3']),
    ('4.5', default_review_channels([]), []),
]
for item, got4, want in cases4:
    check(item, f"default_review_channels -> {want}", got4 == want, repr(got4))

rwin = rg.EventReviewGUI()
check('4.8', "no database, no EEG file: channel list empty",
      rwin.channel_list.count() == 0, str(rwin.channel_list.count()))
check('4.8', "status bar hint",
      rwin.status_bar.currentMessage()
      == 'No channels yet - open an event database or an EEG file.',
      repr(rwin.status_bar.currentMessage()))


class EEGStub:
    """LargeDataset stand-in that records the channels each read asks for."""

    def __init__(self, channels, chan_type):
        self.channels = list(channels)
        self.sampling_rate = 250.0
        self.header = {'chan_type': chan_type, 'n_samples': 250 * 600,
                       's_freq': 250.0}
        self.reads = []

    def read_data(self, chan=None, begtime=None, endtime=None):
        self.reads.append(list(chan))
        raise RuntimeError("stub has no signal")


orig_ld = rg.LargeDataset
NEXT_FILE = {'labels': REF_LABELS, 'types': REF_TYPES}
rg.LargeDataset = lambda path, create_memmap=False: EEGStub(
    NEXT_FILE['labels'], NEXT_FILE['types'])


class InfoSpy(__import__('logging').Handler):
    """Collects the review GUI's INFO lines."""

    def __init__(self):
        super().__init__(level=20)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


info_spy = InfoSpy()
rg.logger.addHandler(info_spy)
rg.logger.setLevel(20)
try:
    rwin.load_eeg_file(os.path.join(TMP, 'ref_stub.edf'))
    check('4.6', "EEG stub load selects Cz, Fz, Pz",
          rwin.selected_channels == ['Cz', 'Fz', 'Pz'],
          repr(rwin.selected_channels))
    listed = [rwin.channel_list.item(i).data(Qt.UserRole)
              for i in range(rwin.channel_list.count())]
    ticked = [rwin.channel_list.item(i).data(Qt.UserRole)
              for i in range(rwin.channel_list.count())
              if rwin.channel_list.item(i).checkState() == Qt.Checked]
    check('4.8', "EEG stub only: list holds its 257 EEG channels",
          listed == summary.eeg, f"{len(listed)} items")
    check('4.6', "the default channels are ticked in the list",
          ticked == ['Fz', 'Cz', 'Pz'], repr(ticked))

    # the user ticks C3 themselves; a second load must not undo it
    for i in range(rwin.channel_list.count()):
        if rwin.channel_list.item(i).data(Qt.UserRole) == 'C3':
            rwin.channel_list.item(i).setCheckState(Qt.Checked)
    before = list(rwin.selected_channels)
    rwin.load_eeg_file(os.path.join(TMP, 'ref_stub2.edf'))
    check('4.6', "file B has every picked channel: the user's selection is kept",
          'C3' in before and rwin.selected_channels == before,
          repr((before, rwin.selected_channels)))
    check('4.6', "one INFO line says the selection was kept",
          sum('Keeping your channel selection' in m for m in info_spy.lines) == 1,
          repr(info_spy.lines))

    # file B with a different montage (EGI): the pick cannot be read there
    info_spy.lines.clear()
    NEXT_FILE.update(labels=[f"E{i}" for i in range(1, 257)] + ['Cz'],
                     types=None)
    rwin.load_eeg_file(os.path.join(TMP, 'egi_stub.edf'))
    check('4.6', "file B lacks the picked channels: defaults apply",
          rwin.selected_channels == ['E112', 'E118', 'Cz']
          and rwin._channels_user_set is False,
          repr(rwin.selected_channels))
    check('4.6', "one INFO line says the selection was reset and why",
          len(info_spy.lines) == 1
          and 'reset to defaults' in info_spy.lines[0]
          and 'C3' in info_spy.lines[0], repr(info_spy.lines))
    NEXT_FILE.update(labels=REF_LABELS, types=REF_TYPES)
    rwin.load_eeg_file(os.path.join(TMP, 'ref_stub3.edf'))
    check('4.6', "back on the reference montage, its defaults apply again",
          rwin.selected_channels == ['Cz', 'Fz', 'Pz'],
          repr(rwin.selected_channels))

    # 4.7: the loader falls back to the helper, not a literal EGI list
    rwin.selected_channels = []
    loader = WaveformBackgroundLoader(rwin)
    loader.load_waveform({'uuid': 'u1', 'start_time': 100.0, 'end_time': 101.0})
    reads = rwin.eeg_data.reads
    check('4.7', "waveform_loader with no selection reads the helper's channels",
          reads == [['Cz', 'Fz', 'Pz']], repr(reads))
finally:
    rg.LargeDataset = orig_ld
    rg.logger.removeHandler(info_spy)
    if rwin.background_loader is not None:
        rwin.background_loader.stop()
        rwin.background_loader = None

frontend_dir = os.path.join(os.path.dirname(HERE), 'frontend')
hits = []
for name in ('turtlewave_gui.py', 'eeg_review_gui.py', 'waveform_loader.py',
             'channel_types.py'):
    with open(os.path.join(frontend_dir, name), encoding='utf-8') as fh:
        for n, ln in enumerate(fh, 1):
            if "'E112'" in ln:
                hits.append(f"{name}:{n}")
check('4.7', "the string 'E112' appears only inside the helper",
      bool(hits) and all(h.startswith('channel_types.py') for h in hits),
      repr(hits))

# ======================================================================= 5
say("\n== Section 5: load-failure message")
broken = os.path.join(TMP, REF_FILE)
write_set(broken, layout='root', v73=True, omit=('srate', 'chanlocs'))
try:
    tg.LargeDataset(broken, create_memmap=False)
    broken_err = None
except Exception as e:                                        # noqa: BLE001
    broken_err = e
check('5.0', "fixture: the broken file raises EEGLABFormatError",
      type(broken_err).__name__ == 'EEGLABFormatError', repr(broken_err))


def run_load_data(window, path):
    window.data_file_path = path
    window.output_dir = TMP
    window.annot_file_path = ''
    spy.clear()
    with CriticalSpy() as dialogs:
        window.load_data()
        app.processEvents()
    return dialogs.messages


msgs = run_load_data(win, broken)
body = msgs[0][1] if msgs else ''
check('5.1', "one dialog titled 'Error'", len(msgs) == 1 and msgs[0][0] == 'Error',
      repr(msgs))
check('5.1', "text starts 'Could not load this recording.' and names the file, srate, "
      "chanlocs, 'Found at the top level:'",
      body.startswith('Could not load this recording.') and REF_FILE in body
      and 'srate' in body and 'chanlocs' in body
      and 'Found at the top level:' in body, repr(body))
check('5.1', "no HDF5 internals, class names or traceback",
      not any(w in body for w in ('synchronously', 'h5py', 'Traceback',
                                  'KeyError', 'EEGLABFormatError')), repr(body))
check('5.3', "log has the original exception text",
      str(broken_err) in spy.text() and 'Traceback' in spy.text(),
      spy.text()[:300])
say("\n  dialog body (broken file):\n    " + body.replace("\n", "\n    "))

nofdt = write_set(os.path.join(TMP, 'sub-x_nofdt_eeg.set'), layout='root', v73=True)
os.remove(nofdt['fdt'])
try:
    tg.LargeDataset(nofdt['path'], create_memmap=False)
    fdt_err = None
except Exception as e:                                        # noqa: BLE001
    fdt_err = e
msgs = run_load_data(win, nofdt['path'])
body2 = msgs[0][1] if msgs else ''
check('5.2', "missing .fdt: signal-file message naming the .fdt",
      body2.startswith('Could not load this recording.')
      and 'sub-x_nofdt_eeg.fdt' in body2
      and 'Copy the .fdt file next to the .set file and load again.' in body2,
      repr(body2))
check('5.3', "log has the original FileNotFoundError text",
      fdt_err is not None and str(fdt_err) in spy.text(), spy.text()[:300])
say("\n  dialog body (missing .fdt):\n    " + body2.replace("\n", "\n    "))

rwin2 = rg.EventReviewGUI()
with CriticalSpy() as dialogs:
    rwin2.load_eeg_file(broken)
rbody = dialogs.messages[0][1] if dialogs.messages else ''
check('5.4', "review GUI: same first lines as turtlewave_gui, not an MNE error",
      rbody.splitlines()[:3] == body.splitlines()[:3] and rwin2.eeg_data is None,
      repr(rbody[:300]))
check('5.5', "review GUI dialog points at the terminal, not a log panel",
      rbody.endswith('The full error is printed in the terminal window that '
                     'started the review GUI.')
      and 'log panel' not in rbody, repr(rbody[-120:]))
check('5.5', "turtlewave_gui keeps 'The full error is in the log panel.'",
      body.endswith('The full error is in the log panel.'), repr(body[-80:]))

# ======================================================================= 6
say("\n== Section 6: interpolated channels")
TIP = ('Interpolated channel (reconstructed from neighbours by the cleaning '
       'pipeline)')
check('6.0', "Dataset Information shows the interpolated line after Reference",
      'Reference: average of 220 of 257 EEG channels (as stored in the file; '
      'not changed by TurtleWave)\nInterpolated channels: 37 (AF3, F3, F1, Cz, '
      in info_for(reference_stub(), os.path.join('/data', REF_FILE)),
      'see 2.1')
got = info_for(stub(['Fz', 'Cz', 'Pz']), '/d/none.set')
check('6.0', "interp_channels absent: 'none stated in the file'",
      'Interpolated channels: none stated in the file' in got, repr(got))
three = stub(['Fz', 'Cz', 'Pz'])
three.header['interp_channels'] = ['Cz', ' Pz ', 'Cz', '']
got = info_for(three, '/d/three.set')
check('6.0', "3 names: listed in full, duplicates/blanks dropped",
      'Interpolated channels: 2 (Cz, Pz)' in got, repr(got))
empty = stub(['Fz', 'Cz'])
empty.header['interp_channels'] = []
check('6.0', "interp_channels == []: 'none stated in the file'",
      'Interpolated channels: none stated in the file'
      in info_for(empty, '/d/empty.set'), '')

s6 = ChannelTypeSummary(['Fp1', 'Cz', 'VEOG', 'ECG', 'Pz'],
                        ['EEG', 'EEG', 'EOG', 'ECG', 'EEG'],
                        interp_channels=['Cz', 'ECG'])
check('6.1', "summary.interpolated / is_interpolated",
      s6.interpolated == ['Cz', 'ECG'] and s6.is_interpolated('Cz')
      and not s6.is_interpolated('Pz'), repr(s6.interpolated))
check('6.1', "tooltip text for an interpolated EEG channel",
      s6.item_tooltip('Cz') == TIP, repr(s6.item_tooltip('Cz')))
check('6.1', "non-EEG + interpolated: both lines",
      s6.item_tooltip('ECG') == 'Non-EEG channel (ECG)\n' + TIP,
      repr(s6.item_tooltip('ECG')))
check('6.1', "plain EEG channel: no tooltip", s6.item_tooltip('Pz') == '',
      repr(s6.item_tooltip('Pz')))
check('6.1', "interp run note names the picked ones, singular noun",
      s6.interp_run_note(['Pz', 'Cz']) ==
      'Note: 1 interpolated channel selected: Cz (reconstructed from '
      'neighbours by the cleaning pipeline).', repr(s6.interp_run_note(['Cz'])))
check('6.1', "interp run note is None when none picked",
      s6.interp_run_note(['Pz']) is None, '')
check('6.1', "from_dataset reads header['interp_channels']",
      ChannelTypeSummary.from_dataset(three).interpolated == ['Cz', 'Pz'],
      repr(ChannelTypeSummary.from_dataset(three).interpolated))

five_i = stub(['Fp1', 'Cz', 'VEOG', 'ECG', 'Pz'],
              ['EEG', 'EEG', 'EOG', 'ECG', 'EEG'])
five_i.header['interp_channels'] = ['Cz', 'Pz']
load(win, five_i)
win.non_eeg_checks['spindle'].setChecked(False)
win.selected_channels = ['Pz']
win.update_channel_lists()


def item_state(lst, name):
    for i in range(lst.count()):
        it = lst.item(i)
        if it.text() == name:
            return (it.text(), it.font().italic(), it.toolTip())
    return None


avail_states = {k: item_state(lst, 'Cz') for k, lst in
                (('spindle', win.available_list),
                 ('sw', win.sw_available_list),
                 ('kc', win.kc_available_list))}
check('6.2', "Available lists: Cz italic, tooltip, text unchanged",
      all(v == ('Cz', True, TIP) for v in avail_states.values()),
      repr(avail_states))
check('6.2', "Available lists: Fp1 not italic, no tooltip",
      item_state(win.available_list, 'Fp1') == ('Fp1', False, ''),
      repr(item_state(win.available_list, 'Fp1')))
sel_states = {k: item_state(lst, 'Pz') for k, lst in DET_SEL(win).items()}
check('6.2', "Selected lists: Pz italic, tooltip, text unchanged",
      all(v == ('Pz', True, TIP) for v in sel_states.values()),
      repr(sel_states))
for i in range(win.available_list.count()):
    it = win.available_list.item(i)
    it.setSelected(it.text() == 'Cz')
win.add_channels()
check('6.2', "add_channels reads the bare name back from an italic item",
      win.selected_channels == ['Pz', 'Cz'], repr(win.selected_channels))
win.pac_available_channels = ['Cz', 'Fp1']
win.pac_selected_channels = ['Pz']
win.update_pac_channel_lists()
check('6.2', "PAC lists decorated too",
      item_state(win.pac_available_list, 'Cz') == ('Cz', True, TIP)
      and item_state(win.pac_selected_list, 'Pz') == ('Pz', True, TIP),
      repr((item_state(win.pac_available_list, 'Cz'),
            item_state(win.pac_selected_list, 'Pz'))))
spy.clear()
win._log_non_eeg_selection(['Cz', 'ECG', 'Pz'])
check('6.3', "run start logs the non-EEG note, then the interpolated note",
      spy.lines == ['Note: 1 non-EEG channel selected: ECG.',
                    'Note: 2 interpolated channels selected: Cz, Pz '
                    '(reconstructed from neighbours by the cleaning pipeline).'],
      repr(spy.lines))

# review GUI: " ~" suffix on the filter-dock list
rwin3 = rg.EventReviewGUI()


class InterpStub(EEGStub):
    def __init__(self, channels, chan_type, interp):
        super().__init__(channels, chan_type)
        self.header['interp_channels'] = list(interp)


rg.LargeDataset = lambda path, create_memmap=False: InterpStub(
    ['Fz', 'Cz', 'Pz', 'ECG'], ['EEG', 'EEG', 'EEG', 'ECG'], ['Cz'])
try:
    rwin3.load_eeg_file(os.path.join(TMP, 'interp_stub.edf'))
    names, ctypes, interp = rwin3._eeg_channel_info()
    check('6.4', "_eeg_channel_info returns the interpolated set",
          interp == {'Cz'} and names == ['Fz', 'Cz', 'Pz', 'ECG'],
          repr((names, interp)))
    rows = [(rwin3.channel_list.item(i).text(),
             rwin3.channel_list.item(i).data(Qt.UserRole),
             rwin3.channel_list.item(i).toolTip())
            for i in range(rwin3.channel_list.count())]
    check('6.4', "filter-dock list: 'Cz ~' with tooltip, others bare",
          rows == [('Fz', 'Fz', ''), ('Cz ~', 'Cz', TIP), ('Pz', 'Pz', '')],
          repr(rows))
    rwin3.filter_dock.decorate_channels({'Cz'}, {'Pz'})
    rows = [rwin3.channel_list.item(i).text()
            for i in range(rwin3.channel_list.count())]
    check('6.4', "later verdict decoration keeps the ' ~' mark",
          rows == ['Fz', 'Cz ~ \u2691', 'Pz \u21bb'], repr(rows))
    rwin3.filter_dock.decorate_channels(interp_set=set())
    check('6.4', "interp_set=set() clears the mark",
          rwin3.channel_list.item(1).text() == 'Cz'
          and rwin3.channel_list.item(1).toolTip() == '',
          repr(rwin3.channel_list.item(1).text()))

    # 6.5: defaults that include an interpolated channel are named once
    rwin4 = rg.EventReviewGUI()
    rg.LargeDataset = lambda path, create_memmap=False: InterpStub(
        REF_LABELS, REF_TYPES, ['Cz'])
    rwin4.load_eeg_file(os.path.join(TMP, 'ref_interp.edf'))
    msg = rwin4.status_bar.currentMessage()
    check('6.5', "defaults include interpolated Cz: status bar says so",
          msg == 'Showing Cz, Fz, Pz. Cz is interpolated (reconstructed from '
                 'neighbours by the cleaning pipeline).', repr(msg))
    rwin4._channels_user_set = True
    rwin4.selected_channels = ['Cz', 'C3']
    rwin4.load_eeg_file(os.path.join(TMP, 'ref_interp2.edf'))
    msg = rwin4.status_bar.currentMessage()
    check('6.5', "user's own selection kept: no interpolated note",
          'interpolated' not in msg, repr(msg))
    plural = rg.interpolated_defaults_note(['Cz', 'Fz', 'Pz'], {'Cz', 'Pz'})
    check('6.5', "plural wording",
          plural == 'Showing Cz, Fz, Pz. Cz and Pz are interpolated '
                    '(reconstructed from neighbours by the cleaning pipeline).',
          repr(plural))

    # F6: channel set-up failure after a successful open
    def boom():
        raise RuntimeError("channel table unreadable\nsecond line")
    rwin4.load_channels = boom
    with CriticalSpy() as dialogs:
        rwin4.load_eeg_file(os.path.join(TMP, 'ref_interp3.edf'))
    fbody = dialogs.messages[0][1] if dialogs.messages else ''
    check('6.6', "set-up failure dialog wording",
          fbody == 'The recording opened, but its channels could not be set '
                   'up: channel table unreadable. The full error is printed in '
                   'the terminal window that started the review GUI.',
          repr(fbody))
    if rwin4.background_loader is not None:
        rwin4.background_loader.stop()
        rwin4.background_loader = None
    rwin4.close()
finally:
    rg.LargeDataset = orig_ld
    if rwin3.background_loader is not None:
        rwin3.background_loader.stop()
        rwin3.background_loader = None

rwin.close()
rwin2.close()
rwin3.close()
say("\n" + "=" * 78)
say(f"{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
for f in FAILURES:
    say("  FAILED: " + f)
say("=" * 78)

if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
