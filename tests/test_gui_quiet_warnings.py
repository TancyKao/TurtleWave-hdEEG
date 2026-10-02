#!/usr/bin/env python3
"""The GUI entry paths hide Wonambi's fooof notice; the library alone does not.

Both GUI modules set ``TURTLEWAVE_QUIET_WONAMBI`` (``setdefault``) before
their first ``turtlewave_hdEEG`` import, because the library reads it before
its first Wonambi import and a later call cannot take the fooof notice back.

Each case runs in a subprocess with ``-W always`` and a stand-in ``fooof``
package on ``PYTHONPATH`` (the pattern of ``tests/test_quiet_warnings.py``):
``simplefilter('always')`` then a DeprecationWarning at import.

* Importing ``frontend.eeg_review_gui`` or ``frontend.turtlewave_gui`` the way
  the console scripts do (``from frontend.<module> import main``), then
  importing ``wonambi.widgets.analysis`` (which imports fooof): no notice.
* The same with ``TURTLEWAVE_QUIET_WONAMBI=0`` exported: the notice prints,
  so the user can switch it back on and the stand-in is shown to be reached.
* ``import turtlewave_hdEEG`` alone: the variable is not set and no filter is
  installed.

No window is built and no QSettings is read. Run standalone:
``python tests/test_gui_quiet_warnings.py``. Exits non-zero on failure.
"""

import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ENV_NAME = 'TURTLEWAVE_QUIET_WONAMBI'

FAKE_FOOOF = '''
from warnings import warn, simplefilter
simplefilter('always')
warn("\\nThe `fooof` package is being deprecated and replaced by the "
     "`specparam` (spectral parameterization) package.", DeprecationWarning,
     stacklevel=2)
FOOOF = object
FOOOFGroup = object
'''

SCRIPT = r'''
import os, sys, warnings
target = sys.argv[1]
if target == 'library':
    import turtlewave_hdEEG
else:
    module = __import__('frontend.' + target, fromlist=['main'])
    assert callable(module.main)
print('ENV', os.environ.get('TURTLEWAVE_QUIET_WONAMBI'))
try:
    import wonambi.widgets.analysis  # noqa: F401  (imports fooof)
except Exception as err:
    print('WIDGETS_UNAVAILABLE', type(err).__name__)
print('FOOOF_IMPORTED', 'fooof' in sys.modules)
ours = [f for f in warnings.filters if f[0] == 'ignore' and f[1] is not None
        and ('fooof' in f[1].pattern or 'ndim > 0' in f[1].pattern)]
print('OUR_FILTERS', len(ours))
'''

NOTICE = 'The `fooof` package is being deprecated'


def _run(target, exported=None):
    with tempfile.TemporaryDirectory(prefix='tw_gui_quiet_') as tmp:
        os.makedirs(os.path.join(tmp, 'fooof'))
        with open(os.path.join(tmp, 'fooof', '__init__.py'), 'w') as fh:
            fh.write(FAKE_FOOOF)
        script = os.path.join(tmp, 'case.py')
        with open(script, 'w') as fh:
            fh.write(SCRIPT)
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([tmp, REPO])
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env.pop('PYTHONWARNINGS', None)
        env.pop(ENV_NAME, None)
        if exported is not None:
            env[ENV_NAME] = exported
        proc = subprocess.run([sys.executable, '-W', 'always', script, target],
                              capture_output=True, text=True, env=env,
                              cwd=tmp, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout, proc.stderr


def _fooof_reached(out):
    return 'FOOOF_IMPORTED True' in out


def _gui_quiet(target):
    out, err = _run(target)
    assert 'ENV 1' in out, out
    assert NOTICE not in err, err[-1500:]
    if _fooof_reached(out):
        assert 'OUR_FILTERS 2' in out, out
        print(f"  {target}: variable set before the library import; the "
              f"stand-in fooof was imported and printed no notice: OK")
    else:
        print(f"  {target}: variable set, no notice; fooof part SKIPPED "
              f"(wonambi.widgets not importable here): OK")


def test_review_gui_entry_is_quiet():
    _gui_quiet('eeg_review_gui')


def test_batch_gui_entry_is_quiet():
    _gui_quiet('turtlewave_gui')


def test_user_can_switch_warnings_back_on():
    for target in ('eeg_review_gui', 'turtlewave_gui'):
        out, err = _run(target, exported='0')
        assert 'ENV 0' in out, out
        assert 'OUR_FILTERS 0' in out, out
        if _fooof_reached(out):
            assert NOTICE in err, err[-1500:]
    print("  TURTLEWAVE_QUIET_WONAMBI=0 exported: setdefault keeps it, no "
          "filter is installed and the notice prints: OK")


def test_library_alone_installs_nothing():
    out, err = _run('library')
    assert 'ENV None' in out, out
    assert 'OUR_FILTERS 0' in out, out
    if _fooof_reached(out):
        assert NOTICE in err, err[-1500:]
    print("  import turtlewave_hdEEG alone: variable not set, no filter, "
          "the notice still prints: OK")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAILED {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
