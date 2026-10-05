#!/usr/bin/env python3
"""``utils.quiet_wonambi_warnings``: two Wonambi 7.15 warnings, nothing else.

Each case runs in a subprocess with ``-W always`` (so every warning would
print unless a specific filter ignores it) and a stand-in ``fooof`` package
on ``PYTHONPATH`` that behaves like fooof 1.1: ``simplefilter('always')``
then a DeprecationWarning at import. The real Wonambi imports it from
``wonambi/widgets/analysis.py``.

* Control (helper not called): the ``fooof`` notice and the NumPy
  "Conversion of an array with ndim > 0" warning attributed to
  ``wonambi.trans.analyze`` both print, so the cases below can see them.
* Opt-in (``TURTLEWAVE_QUIET_WONAMBI=1``, helper runs before Wonambi is
  imported): neither prints; the same NumPy warning from another module and
  an unrelated DeprecationWarning still print; a non-fooof warning raised
  while fooof is imported is re-issued.
* Late call (after ``import turtlewave_hdEEG``): the NumPy warning is
  silenced even though fooof's blanket ``'always'`` filter is installed; the
  notice already printed cannot be taken back; two calls leave one copy of
  each filter.
* Without the opt-in variable, importing the library installs no filter.

Run standalone: ``python tests/test_quiet_warnings.py``. Exits non-zero if
any test fails.
"""

import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

FAKE_FOOOF = '''
from warnings import warn, simplefilter
simplefilter('always')
warn("\\nThe `fooof` package is being deprecated and replaced by the "
     "`specparam` (spectral parameterization) package.", DeprecationWarning,
     stacklevel=2)
warn("fooof side warning that must survive", UserWarning, stacklevel=2)
FOOOF = object
FOOOFGroup = object
'''

SCRIPT = r'''
import sys, warnings
mode = sys.argv[1]
import turtlewave_hdEEG
if mode == 'late':
    turtlewave_hdEEG.quiet_wonambi_warnings()
    turtlewave_hdEEG.quiet_wonambi_warnings()
import wonambi.trans.analyze as an
try:
    import wonambi.widgets.analysis  # noqa: F401
except Exception as err:
    print('WIDGETS_UNAVAILABLE', type(err).__name__)
print('FOOOF_IMPORTED', 'fooof' in sys.modules)
conv = "import numpy as np\nfloat(np.array([1.0]))"
exec(compile(conv, an.__file__, 'exec'), {'__name__': 'wonambi.trans.analyze'})
exec(compile(conv, 'my_analysis.py', 'exec'), {'__name__': 'my_analysis'})
warnings.warn('unrelated own deprecation', DeprecationWarning)
ours = [f for f in warnings.filters if f[0] == 'ignore' and f[1] is not None
        and ('fooof' in f[1].pattern or 'ndim > 0' in f[1].pattern)]
print('OUR_FILTERS', len(ours))
'''

NOTICE = 'The `fooof` package is being deprecated'
CONV = 'Conversion of an array with ndim > 0 to a scalar is deprecated'


def _run(mode, opt_in):
    with tempfile.TemporaryDirectory(prefix='tw_quiet_') as tmp:
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
        env.pop('TURTLEWAVE_QUIET_WONAMBI', None)
        if opt_in:
            env['TURTLEWAVE_QUIET_WONAMBI'] = '1'
        proc = subprocess.run([sys.executable, '-W', 'always', script, mode],
                              capture_output=True, text=True, env=env,
                              cwd=tmp, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout, proc.stderr


def _conv_lines(err):
    return [ln for ln in err.splitlines() if CONV in ln]


def test_control_shows_both():
    out, err = _run('plain', opt_in=False)
    assert 'OUR_FILTERS 0' in out, out
    conv = _conv_lines(err)
    assert any('analyze.py' in ln for ln in conv), err[-1500:]
    assert any('my_analysis.py' in ln for ln in conv), err[-1500:]
    assert 'unrelated own deprecation' in err
    if 'FOOOF_IMPORTED True' in out:
        assert NOTICE in err, err[-1500:]
        print("  control: fooof notice and the wonambi.trans.analyze NumPy "
              "warning both print; no filter installed at import: OK")
    else:
        print("  control: NumPy warning prints; no filter at import; fooof "
              "part SKIPPED (wonambi.widgets not importable here): OK")


def test_opt_in_before_wonambi():
    out, err = _run('plain', opt_in=True)
    assert 'OUR_FILTERS 2' in out, out
    assert NOTICE not in err, err[-1500:]
    conv = _conv_lines(err)
    assert not any('analyze.py' in ln for ln in conv), conv
    assert len(conv) == 1 and 'my_analysis.py' in conv[0], conv
    assert 'unrelated own deprecation' in err
    assert 'fooof side warning that must survive' in err, err[-1500:]
    assert 'FOOOF_IMPORTED True' in out
    print("  opt-in before Wonambi: neither warning prints; the same NumPy "
          "warning from my_analysis.py, an unrelated DeprecationWarning and "
          "fooof's other warning still print: OK")


def test_late_call_and_idempotent():
    out, err = _run('late', opt_in=False)
    assert 'OUR_FILTERS 2' in out, out
    conv = _conv_lines(err)
    assert len(conv) == 1 and 'my_analysis.py' in conv[0], conv
    assert 'unrelated own deprecation' in err
    late_notice = err.count(NOTICE)
    assert late_notice <= 1, err[-1500:]
    print(f"  late call: NumPy warning silenced behind fooof's 'always' "
          f"filter; fooof notice already printed {late_notice}x (cannot be "
          f"undone); two calls leave 2 filters: OK")


TESTS = [test_control_shows_both, test_opt_in_before_wonambi,
         test_late_call_and_idempotent]


if __name__ == '__main__':
    print("TESTING quiet_wonambi_warnings")
    print("==============================")
    failed = []
    for test in TESTS:
        try:
            test()
        except Exception:
            failed.append(test.__name__)
            traceback.print_exc()
    print()
    if failed:
        print(f"FAILED {len(failed)} of {len(TESTS)}: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(TESTS)} tests passed.")
