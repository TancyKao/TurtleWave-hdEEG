"""Keep GUI tests out of the user's real preferences.

Import and call :func:`isolate` before importing anything from ``frontend``:
it points every ``QSettings`` in the default format at a temporary INI
directory. :func:`untouched` then says whether the real review-GUI
preferences file (macOS: ``~/Library/Preferences/
com.turtlewave.eeg_review_gui.plist``) kept its modification time and size.

Importing this module also gives Qt's ``offscreen`` platform the system
fonts on Windows. With the PyQt5 wheels that platform finds no fonts there
(``QFontDatabase().families()`` is empty): every character measures as a
fixed 17 px box and nothing is painted, so text widths come out about twice
the real ones and glyph images are blank. ``QT_QPA_FONTDIR`` points it at
``%WINDIR%\\Fonts`` (82 families on windows-latest); it must be set before
the ``QApplication`` exists, and an explicit value is kept. The native
``windows`` platform is unaffected.
"""
import os
import sys
import tempfile

if (sys.platform == 'win32'
        and os.environ.get('QT_QPA_PLATFORM', '').startswith('offscreen')):
    os.environ.setdefault('QT_QPA_FONTDIR', os.path.join(
        os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))

from PyQt5 import QtCore                                    # noqa: E402

REAL_PLIST = os.path.expanduser(
    '~/Library/Preferences/com.turtlewave.eeg_review_gui.plist')
_before = None


def _stamp():
    try:
        st = os.stat(REAL_PLIST)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def isolate():
    """Temporary INI settings for this process; returns the directory."""
    global _before
    tmp = tempfile.mkdtemp(prefix='tw_qsettings_')
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    for scope in (QtCore.QSettings.UserScope, QtCore.QSettings.SystemScope):
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, scope, tmp)
    _before = _stamp()
    return tmp


def untouched():
    """``(ok, detail)``: the real preferences file is as it was."""
    now = _stamp()
    return now == _before, f"before {_before}, after {now}"
