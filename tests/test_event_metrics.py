#!/usr/bin/env python3
"""M8 fixtures of the event-review Method Spec (revision 3), pre-registered.

Runs ``turtlewave_hdEEG.event_metrics`` on synthetic pink-noise fixtures at
250 and 500 Hz, on the spec's untouched test seeds: 9000-9019 for each case,
9000-9199 for F7. F9 runs a slow wave through the real
``ImprovedDetectSlowWave('Massimini2004')`` and asserts on the detector's own
output event.

* F1 13 Hz x 1 s spindle: peak 13 +- 0.5 Hz, in band, not low prominence,
  17-27 half-waves standing out, nominal cycles 13 +- 1.5, amp_ratio >= 3,
  not coarse.
* F2 9 Hz burst against a 12-15 Hz band: peak 9 +- 0.5 Hz, never in band,
  amp_ratio < 2.5.
* F3 2 ms pulse / step, no spindle: flagged (off-band or low prominence) in
  >= 19/20 seeds per rate; nominal cycles >= 10.
* F4 13 Hz x 0.5 s: coarse, peak 13.5 +- 1 Hz, in band, >= 10 half-waves.
* F5 F1 on a -60 uV slow wave: peak 13 +- 0.5 Hz.
* F6 a second spindle 3 s later passed as another event: 3-4 fewer windows,
  ratio with exclusion >= without.
* F6b the surround's last 10 s scored outside the run's stages:
  bg_stage_mixed, >= 18 fewer windows.
* F7 200 noise windows: mean half-waves <= 3.0, 95th percentile <= 11,
  flagged >= 90 %, negative control <= 5 %.
* F9 Massimini2004 at exactly 80 uV / -100 uV: negative half-wave 0.5 +- 0.03 s,
  det_trough -100 +- 5 uV; a +50 uV variant yields no event.

Also: SW geometry, splice handling and the threshold-ratio helpers on direct
inputs. Run standalone: ``python tests/test_event_metrics.py`` (< 30 s).
Exits non-zero if any test fails.
"""

import logging
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from turtlewave_hdEEG import event_metrics as em  # noqa: E402

BAND = (12.0, 15.0)
RATES = (250, 500)
SEEDS = range(9000, 9020)
F7_SEEDS = range(9000, 9200)


def pink(n, fs, rng, rms_uv=10.0):
    """1/f-power noise with the given RMS (spec generator)."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / y.std() * rms_uv


def fixture(fs, f0=13.0, dur=1.0, amp=20.0, seed=0, extra=None, total=40.0):
    """40 s pink noise with a Hann-windowed burst at t = 20 s (spec generator)."""
    rng = np.random.default_rng(seed)
    n = int(total * fs)
    t = np.arange(n) / fs
    x = pink(n, fs, rng)
    t0 = total / 2
    ie = (t >= t0) & (t < t0 + dur)
    k = np.count_nonzero(ie)
    x[ie] += amp * np.hanning(k) * np.sin(2 * np.pi * f0 * (t[ie] - t0))
    if extra is not None:
        x += extra(t, t0)
    return x, t, t0, t0 + dur


def figs(x, t, fs, a, b, **kw):
    return em.event_figures(x, t, fs, a, b, BAND, **kw)


def flagged(f):
    return (not f.in_band) or bool(f.low_prominence)


def test_spindle_fixtures():
    """F1-F6b on seeds 9000-9019 at both rates."""
    print("\n1. Spindle fixtures F1-F6b (seeds 9000-9019, 250 and 500 Hz):")
    pulse = lambda t, t0: 100.0 * ((t >= t0 + .5) & (t < t0 + .502))  # noqa: E731
    step = lambda t, t0: 100.0 * (t >= t0 + .5)  # noqa: E731
    sw = lambda t, t0: -60.0 * np.exp(-((t - t0 - 0.5) / 0.25) ** 2)  # noqa: E731
    for fs in RATES:
        hw1, f3 = [], {'pulse': 0, 'step': 0}
        for s in SEEDS:
            x, t, a, b = fixture(fs, seed=s)
            f = figs(x, t, fs, a, b)
            assert abs(f.peak_freq_ap - 13) <= .5, (fs, s, f)
            assert f.in_band and not f.low_prominence, (fs, s, f)
            assert 17 <= f.halfwaves_above_bg <= 27, (fs, s, f)
            assert abs(f.cycles_nominal - 13) <= 1.5, (fs, s, f)
            assert f.amp_ratio >= 3 and f.coarse is False, (fs, s, f)
            assert f.near_splice is False and f.bg_stage_mixed is False, f
            hw1.append(f.halfwaves_above_bg)

            # F6: a second spindle 3 s later, excluded as another event.
            x2 = x.copy()
            m = (t >= a + 3) & (t < a + 4)
            x2[m] += 20 * np.hanning(m.sum()) * np.sin(2 * np.pi * 13 * (t[m] - a - 3))
            fe = figs(x2, t, fs, a, b, others=[(a, b), (a + 3, a + 4)])
            fn = figs(x2, t, fs, a, b)
            assert 3 <= fn.bg_n_windows - fe.bg_n_windows <= 4, (fs, s, fn, fe)
            assert fe.amp_ratio >= fn.amp_ratio, (fs, s, fe.amp_ratio, fn.amp_ratio)

            # F6b: last 10 s of the surround scored outside the run.
            epochs = [(0.0, b + 5, 'NREM2'), (b + 5, 40.0, 'Wake')]
            fm = figs(x, t, fs, a, b, stage_epochs=epochs, run_stages=['NREM2'])
            assert fm.bg_stage_mixed is True, (fs, s, fm)
            assert f.bg_n_windows - fm.bg_n_windows >= 18, (fs, s, f, fm)
            # Same, but the out-of-stage time is absent (a detection
            # segment): nothing usable was dropped, so not mixed.
            ins = t < b + 5
            fa = figs(x[ins], t[ins], fs, a, b, stage_epochs=epochs,
                      run_stages=['NREM2'])
            assert fa.bg_stage_mixed is False, (fs, s, fa)
            # The run edge guard P also removes windows next to the cut.
            assert fa.bg_n_windows <= fm.bg_n_windows, (fs, s, fa, fm)

            x, t, a, b = fixture(fs, f0=9.0, seed=s)
            f = figs(x, t, fs, a, b)
            assert abs(f.peak_freq_ap - 9) <= .5 and not f.in_band, (fs, s, f)
            assert f.amp_ratio < 2.5, (fs, s, f)

            for name, ex in (('pulse', pulse), ('step', step)):
                x, t, a, b = fixture(fs, amp=0.0, seed=s, extra=ex)
                f = figs(x, t, fs, a, b)
                assert f.cycles_nominal >= 10, (fs, s, name, f)
                f3[name] += flagged(f)

            x, t, a, b = fixture(fs, dur=0.5, seed=s)
            f = figs(x, t, fs, a, b)
            assert f.coarse and f.in_band, (fs, s, f)
            assert abs(f.peak_freq_ap - 13.5) <= 1, (fs, s, f)
            assert f.halfwaves_above_bg >= 10, (fs, s, f)

            x, t, a, b = fixture(fs, seed=s, extra=sw)
            f = figs(x, t, fs, a, b)
            assert abs(f.peak_freq_ap - 13) <= .5, (fs, s, f)
        assert f3['pulse'] >= 19 and f3['step'] >= 19, (fs, f3)
        print(f"[ok] {fs} Hz: F1 half-waves {min(hw1)}-{max(hw1)}; F3 flagged "
              f"pulse {f3['pulse']}/20, step {f3['step']}/20; F2, F4, F5, F6, "
              f"F6b pass on all 20 seeds")


def test_noise_f7():
    """F7, negative control and the kill condition on 200 noise seeds."""
    print("\n2. F7 pure noise (seeds 9000-9199):")
    for fs in RATES:
        hw, flg, neg = [], [], []
        for s in F7_SEEDS:
            x, t, a, b = fixture(fs, amp=0.0, seed=s)
            f = figs(x, t, fs, a, b)
            hw.append(f.halfwaves_above_bg)
            flg.append(flagged(f))
            neg.append(bool(f.in_band) and not f.low_prominence
                       and f.halfwaves_above_bg >= 10)
        mean, p95 = float(np.mean(hw)), float(np.percentile(hw, 95))
        assert mean <= 3.0, (fs, mean)
        assert p95 <= 11, (fs, p95)
        assert np.mean(flg) >= .9, (fs, np.mean(flg))
        assert np.mean(neg) <= .05, (fs, np.mean(neg))
        print(f"[ok] {fs} Hz: mean half-waves {mean:.2f}, p95 {p95:.2f}, "
              f"flagged {100 * np.mean(flg):.1f} %, negative control "
              f"{100 * np.mean(neg):.1f} %")


def test_f9_massimini_detector():
    """F9 through the real ImprovedDetectSlowWave('Massimini2004')."""
    print("\n3. F9 Massimini2004 through the real detector:")
    from wonambi.datatype import ChanTime
    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    def chantime(sig, fs):
        d = ChanTime()
        d.s_freq = fs
        for k in ('chan', 'time'):
            d.axis[k] = np.empty(1, dtype='O')
        d.data = np.empty(1, dtype='O')
        d.axis['chan'][0] = np.array(['Cz'])
        d.axis['time'][0] = np.arange(len(sig)) / fs
        d.data[0] = np.asarray(sig, 'f')[None, :]
        return d

    logging.disable(logging.WARNING)
    try:
        for fs in (250.0, 500.0):
            for pos in (80.0, 50.0):
                n = int(40 * fs)
                t = np.arange(n) / fs
                x = np.zeros(n)
                t0 = 20.0
                p = (t >= t0) & (t < t0 + 0.4)
                x[p] = pos * np.sin(np.pi * (t[p] - t0) / 0.4)
                q = (t >= t0 + 0.4) & (t < t0 + 0.9)
                x[q] = -100 * np.sin(np.pi * (t[q] - t0 - 0.4) / 0.5)
                ev = ImprovedDetectSlowWave('Massimini2004')(chantime(x, fs)).events
                if pos == 50.0:
                    assert not ev, (fs, ev)
                    continue
                assert len(ev) == 1, (fs, ev)
                e = ev[0]
                shape = em.slow_wave_shape('Massimini2004', e['start'], e['end'],
                                           e['zero_time'])
                assert abs(shape['neg_halfwave_dur'] - 0.5) <= 0.03, (fs, shape)
                assert abs(shape['wave_freq'] - 1.0) <= 0.07, (fs, shape)
                assert abs(e['trough_val'] + 100) <= 5, (fs, e['trough_val'])
                print(f"[ok] {fs:.0f} Hz: negative half-wave "
                      f"{shape['neg_halfwave_dur']:.3f} s, wave_freq "
                      f"{shape['wave_freq']:.3f} Hz, trough {e['trough_val']:.1f} uV; "
                      f"+50 uV variant yields no event")
    finally:
        logging.disable(logging.NOTSET)


def test_splice_and_sw_geometry():
    """near_splice, per-run filtering and slow-wave geometry on direct input."""
    print("\n4. Splices and slow-wave geometry:")
    fs = 250
    x, t, a, b = fixture(fs, seed=9000, total=120.0)
    # Cut 10 s out at 70-80 s: two runs.
    keep = (t < 70) | (t >= 80)
    xs, ts = x[keep], t[keep]
    far = em.event_figures(xs, ts, fs, 60.0 - 31, 60.0 - 30, BAND)
    near = em.event_figures(xs, ts, fs, 68.5, 69.5, BAND)
    span = em.event_figures(xs, ts, fs, 69.5, 80.5, BAND)
    assert far.near_splice is False and far.amp_ratio is not None, far
    assert near.near_splice is True and near.amp_ratio is None, near
    assert near.peak_freq_ap is None and near.halfwaves_above_bg is None, near
    assert span.near_splice is True and span.bg_n_windows is None, span
    row = near.to_row()
    assert row['near_splice'] == 1 and row['bg_n_windows'] is None, row
    print("[ok] events within P of a splice or spanning one are near_splice "
          "with NULL signal figures; a far event is untouched")

    # Slow-wave geometry: no spectrum, wave_freq from the half-wave, in_band
    # against the run band, 2 s windows.
    f = em.event_figures(x, t, fs, 50.0, 51.2, (0.1, 4.0), event_type='slow_wave',
                         method='Massimini2004',
                         event_values={'det_zero_time': 50.6, 'det_trough': -90.0,
                                       'det_ptp': 160.0},
                         thresholds={'max_trough_amp': -80.0, 'min_ptp': 140.0},
                         duration_bounds=(None, None))
    assert f.peak_freq_ap is None and f.halfwaves_above_bg is None, f
    assert abs(f.wave_freq - 1 / 1.2) < 1e-12 and f.in_band is True, f
    assert abs(f.thresh_ratio - min(90 / 80, 160 / 140)) < 1e-12, f
    assert f.bg_n_windows is not None and f.bg_n_windows <= 30, f
    assert f.near_bound is None, f
    print(f"[ok] slow wave: wave_freq {f.wave_freq:.3f} Hz in band, "
          f"thresh_ratio {f.thresh_ratio:.3f} = min(trough, ptp), "
          f"{f.bg_n_windows} background windows")


def test_threshold_and_bound_helpers():
    """thresh_ratio sign handling, method coverage and near_bound."""
    print("\n5. Threshold-ratio and duration-bound helpers:")
    tr = em.threshold_ratio
    assert tr('Moelle2011', {'peak_val_det': 5.16}, {'det_value_lo': 3.43}) == \
        5.16 / 3.43
    assert tr('Massimini2004', {'det_trough': -97.0, 'det_ptp': 178.4},
              {'max_trough_amp': -80.0, 'min_ptp': 140.0}) == 97 / 80
    assert tr('AASM/Massimini2004', {'det_trough': -45.0, 'det_ptp': 76.0},
              {'max_trough_amp': -40.0, 'min_ptp': 75.0}) == 76 / 75
    # A trough on the wrong side of zero is not reported as a ratio.
    assert tr('Massimini2004', {'det_trough': 10.0},
              {'max_trough_amp': -80.0}) is None
    for m in ('Wamsley2012', 'Martin2013', 'Ray2015', 'Lacourse2018', 'CIRUS',
              'Ngo2015', 'Staresina2015'):
        assert tr(m, {'peak_val_det': 5.0}, {'det_value_lo': 2.0}) is None, m
    assert tr('Moelle2011', {'peak_val_det': 5.0}, None) is None
    db = em.duration_bound_flag
    assert db(0.52, (0.5, 3.0)) == -1 and db(2.96, (0.5, 3.0)) == 1
    assert db(1.2, (0.5, 3.0)) == 0 and db(9.0, (0.5, None)) == 0
    assert db(1.0, None) is None
    print("[ok] Massimini min(trough, ptp) with negative criteria; non-ratio "
          "methods return None; near_bound is -1/0/+1")


TESTS = [test_spindle_fixtures, test_noise_f7, test_f9_massimini_detector,
         test_splice_and_sw_geometry, test_threshold_and_bound_helpers]


if __name__ == '__main__':
    t_start = time.time()
    failed = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failed += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed in "
          f"{time.time() - t_start:.1f} s")
    sys.exit(1 if failed else 0)
