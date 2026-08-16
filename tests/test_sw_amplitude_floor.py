#!/usr/bin/env python3
"""The optional slow-wave amplitude floor must be microvolts, not samples.

``ImprovedDetectSlowWave.__call__`` used to run a post-hoc amplitude filter on
Ngo2015 and Staresina2015 BEFORE ``_as_negative_first`` converted the event
fields to microvolts. At that point Wonambi's ``ptp`` is a SAMPLE-INDEX
distance — ``'ptp': abs(ev[3] - ev[1])`` on indices,
wonambi/detect/slowwave.py:418 — so ``ParalSWA``'s 140.0 default (a microvolt
number borrowed from the Massimini family) was compared against a sample
count. The floor therefore scaled with sampling rate instead of amplitude:

* at 500 Hz a half-cycle spans ~310 samples, so 140 rejected almost nothing;
* at 128 Hz the same half-cycle spans ~80 samples, so 140 rejected
  **every event** — measured here at 88 -> 0 for Staresina2015 and 11 -> 0 for
  Ngo2015 on the same signal.

Neither Ngo et al. 2015 (J Neurosci 35(17):6630-8, threshold at 1.25x the mean
trough / mean peak-to-peak) nor Staresina et al. 2015 (Nat Neurosci
18(11):1679-86, 75th percentile of peak-to-peak) defines a fixed microvolt
amplitude criterion, so the floor has no basis for these two methods and is
now off by default. An explicitly requested floor is honoured, in microvolts,
after normalisation — and logged as a deviation from the published method.

Covers four contracts:

1. a valid asymmetric wave at a low sampling rate is kept (it was rejected);
2. the published defaults and the old ``p2p_thresh=0`` workaround give
   identical event sets;
3. an explicit threshold filters in microvolts — invariant to sampling rate,
   sensitive to signal amplitude;
4. ``ParalSWA.detect_slow_waves`` forwards ``None`` unchanged for all four
   methods, so "no arguments" means "the published criteria" end to end.

The Massimini family is untouched: its thresholds are enforced inside the
detector (``max_trough_amp`` / ``min_ptp``) and never reach this floor.

Run standalone: ``python tests/test_sw_amplitude_floor.py``.
"""

import logging
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wonambi.detect import DetectSlowWave as OriginalDetectSlowWave  # noqa: E402

from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave  # noqa: E402


# The value ParalSWA used to substitute for Ngo2015/Staresina2015, and which
# the old code compared against a sample count.
LEGACY_PTP = 140.0


def _make_chantime(sig, s_freq):
    """Wrap a 1-D signal in a single-trial, single-channel ChanTime.

    Parameters
    ----------
    sig : ndarray
        Signal for one channel, shape (n_samples,).
    s_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    instance of wonambi.datatype.ChanTime
        One trial, one channel named 'Cz'.
    """
    from wonambi.datatype import ChanTime

    data = ChanTime()
    data.s_freq = s_freq
    data.axis['chan'] = np.empty(1, dtype='O')
    data.axis['time'] = np.empty(1, dtype='O')
    data.data = np.empty(1, dtype='O')
    data.axis['chan'][0] = np.array(['Cz'])
    data.axis['time'][0] = np.arange(len(sig)) / s_freq
    data.data[0] = np.asarray(sig, dtype='f')[None, :]
    return data


def _asymmetric_so(s_freq, duration=240.0, period=1.25, neg_frac=0.44,
                   neg_amp=130.0, pos_amp=85.0, seed=1):
    """A deterministic slow oscillation with a short trough-to-peak distance.

    Every cycle is a sharp negative lobe followed by a broader positive one,
    so the trough sits early and the peak late-ish within the cycle: their
    separation is about half the period, ~0.62 s. In SAMPLES that is ~80 at
    128 Hz and ~310 at 500 Hz — the quantity the old filter thresholded at
    140 — while the peak-to-peak AMPLITUDE is the same ~200 µV at every rate.
    That is the whole point of the test: one wave, one amplitude, and a
    "threshold" whose verdict flips with the sampling rate.

    The zero-crossing interval is the full ``period``, which sits inside both
    published gates (Staresina 0.8–2.0 s, Ngo 0.833–2.0 s).

    Per-cycle amplitude jitter is added because Ngo2015 thresholds at 1.25x
    the MEAN trough amplitude: on perfectly identical cycles no event can
    exceed the mean and the method returns nothing.

    Parameters
    ----------
    s_freq : float
        Sampling frequency in Hz.
    duration : float, optional
        Length of the signal in seconds. Default ``240.0``.
    period : float, optional
        Cycle period in seconds. Default ``1.25`` (0.8 Hz).
    neg_frac : float, optional
        Fraction of the cycle occupied by the negative lobe. Default ``0.44``.
    neg_amp, pos_amp : float, optional
        Peak amplitude of each lobe in µV, before jitter.
    seed : int, optional
        Seed for the per-cycle jitter and the additive noise. Default ``1``,
        chosen because Ngo2015's 1.25x-the-mean threshold needs cycles that
        actually exceed the mean: on seed 0 it returns nothing at all.

    Returns
    -------
    ndarray
        Signal in microvolts, shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * s_freq)
    t = np.arange(n) / s_freq
    cycle = (t // period).astype(int)
    u = (t % period) / period

    neg = u < neg_frac
    sig = np.where(
        neg,
        -neg_amp * np.sin(np.pi * np.clip(u, 0, neg_frac) / neg_frac),
        pos_amp * np.sin(np.pi * (np.clip(u, neg_frac, 1.0) - neg_frac)
                         / (1.0 - neg_frac)))

    gain = 1.0 + 0.45 * rng.standard_normal(cycle.max() + 2)[cycle]
    return sig * gain + rng.standard_normal(n) * 1.5


def _raw_events(method, data):
    """Wonambi's own output for one method, before any turtlewave filtering.

    Parameters
    ----------
    method : str
        Slow-wave method name.
    data : instance of ChanTime
        Signal to run on.

    Returns
    -------
    list of dict
        The parent detector's events, with ``ptp`` still a sample count.
    """
    det = ImprovedDetectSlowWave(method=method)
    return OriginalDetectSlowWave.__call__(det, data).events


def _starts(events):
    """Rounded start times of an event collection, for set comparison.

    Parameters
    ----------
    events : iterable of dict
        Detected events.

    Returns
    -------
    list of float
        ``start`` rounded to 6 decimals, in order.
    """
    return [round(float(e['start']), 6) for e in events]


def test_low_rate_asymmetric_wave_survives():
    """(a) A valid wave the sample-count filter rejected is now kept.

    Runs the same 200 µV asymmetric oscillation at 128, 256 and 500 Hz and
    compares three verdicts per rate: Wonambi's own candidate set, what the
    old sample-count rule would have kept from it, and what the detector
    returns now. The old rule's yield collapses at low rates while the
    amplitudes are unchanged; the new default keeps the full published set at
    every rate.
    """
    print("\n1. A low-rate asymmetric wave is no longer rejected on a "
          "sample count:")

    for method in ('Staresina2015', 'Ngo2015'):
        old_kept, new_kept = {}, {}
        for s_freq in (128.0, 256.0, 500.0):
            data = _make_chantime(_asymmetric_so(s_freq), s_freq)
            raw = _raw_events(method, data)
            assert raw, (f"{method} at {s_freq} Hz: Wonambi found nothing, so "
                         f"the comparison would be vacuous")

            # What the pre-4.3 code did: compare a uV number to `ptp`, which
            # at this point is abs(ev[3] - ev[1]) on sample INDICES.
            sample_ptp = [abs(float(e['ptp'])) for e in raw]
            old = [p for p in sample_ptp if p >= LEGACY_PTP]

            new = ImprovedDetectSlowWave(method=method)(data)
            assert len(new) == len(raw), (
                f"{method} at {s_freq} Hz: the default configuration dropped "
                f"{len(raw) - len(new)} of Wonambi's {len(raw)} events, so an "
                f"amplitude floor is still being applied by default")

            uv_ptp = np.array([float(e['ptp']) for e in new])
            assert (uv_ptp > 100.0).all(), (
                f"{method} at {s_freq} Hz: an event with only "
                f"{uv_ptp.min():.1f} uV peak-to-peak got through, so this "
                f"signal is not the large-amplitude wave the test assumes")

            old_kept[s_freq] = len(old)
            new_kept[s_freq] = len(new)
            print(f"   {method:<14} {s_freq:>5.0f} Hz  candidates={len(raw):>3}  "
                  f"old rule (ptp_samples >= {LEGACY_PTP:.0f}) kept "
                  f"{len(old):>3}  now kept {len(new):>3}  "
                  f"(median sample-ptp {np.median(sample_ptp):.0f}, "
                  f"median uV-ptp {np.median(uv_ptp):.0f})")

        assert old_kept[128.0] == 0, (
            f"{method}: the old rule kept {old_kept[128.0]} events at 128 Hz, "
            f"so this signal no longer reproduces the reported failure")
        assert old_kept[500.0] > 0, (
            f"{method}: the old rule kept nothing at 500 Hz either, so the "
            f"rate dependence is not what is being demonstrated")
        assert len(set(new_kept.values())) == 1, (
            f"{method}: the new default still yields different counts across "
            f"sampling rates {new_kept} -- something rate-dependent survives")

    print("   [ok] the same wave is rejected by the old rule only at low "
          "sampling rates; the published criteria now keep it at all three")


def test_defaults_equal_the_old_zero_workaround():
    """(b) Nothing passed == the old ``p2p_thresh=0`` neutralising workaround.

    Callers used to pass ``neg_peak_thresh=0`` / ``p2p_thresh=0`` (or a
    negative ``neg_peak_thresh``, which the old ``abs(x) >= thresh`` arm could
    never reject) purely to switch the filter off. With the floor off by
    default those calls must be exact no-ops, and all of them must equal
    Wonambi's own candidate set — otherwise something is still filtering.
    """
    print("\n2. Defaults, explicit zeros and the raw parent output agree:")

    variants = {
        'defaults': {},
        'p2p=0': dict(p2p_thresh=0.0),
        'both=0': dict(neg_peak_thresh=0.0, p2p_thresh=0.0),
    }

    for method in ('Staresina2015', 'Ngo2015'):
        for s_freq in (128.0, 256.0):
            data = _make_chantime(_asymmetric_so(s_freq), s_freq)
            reference = _starts(_raw_events(method, data))
            assert reference, f"{method} at {s_freq} Hz: no candidates"

            for label, kw in variants.items():
                got = _starts(ImprovedDetectSlowWave(method=method, **kw)(
                    _make_chantime(_asymmetric_so(s_freq), s_freq)))
                assert got == reference, (
                    f"{method} at {s_freq} Hz: '{label}' gave {len(got)} "
                    f"events against the parent's {len(reference)}; a zero "
                    f"threshold must not filter anything")
            print(f"   [ok] {method:<14} {s_freq:>5.0f} Hz  "
                  f"{len(reference)} events from all of: "
                  f"{', '.join(variants)}, parent")


def test_explicit_threshold_filters_in_microvolts():
    """(c) An explicit floor is microvolts: rate-invariant, amplitude-sensitive.

    Three properties distinguish a microvolt threshold from a sample count:

    * raising it monotonically removes events;
    * every survivor genuinely clears it, in µV, on both arms;
    * the retained FRACTION does not move with the sampling rate, but does
      move when the recording is scaled. The old rule had this exactly
      backwards.
    """
    print("\n3. An explicit threshold filters in microvolts:")

    method = 'Staresina2015'
    s_freq = 256.0
    data = _make_chantime(_asymmetric_so(s_freq), s_freq)
    open_events = ImprovedDetectSlowWave(method=method)(data)
    n_open = len(open_events)
    assert n_open, "no events to filter"

    # The sweep is taken from the detected amplitude distribution itself, so
    # the expected yield at each step is known in advance: a floor at the Nth
    # percentile of a quantity must keep the events above it and nothing else.
    # A hardcoded uV ladder would only assert "some number went down".
    uv_ptp = np.array([float(e['ptp']) for e in open_events])
    uv_depth = np.array([-float(e['trough_val']) for e in open_events])

    for label, values, kwarg in (('p2p_thresh', uv_ptp, 'p2p_thresh'),
                                 ('neg_peak_thresh', uv_depth,
                                  'neg_peak_thresh')):
        sweep = [0.0] + [float(np.percentile(values, q))
                         for q in (25, 50, 75)] + [float(values.max()) * 1.01]
        counts = [len(ImprovedDetectSlowWave(method=method,
                                             **{kwarg: t})(data))
                  for t in sweep]
        assert counts[0] == n_open, (
            f"{label}=0 dropped {n_open - counts[0]} of {n_open} events")
        assert counts == sorted(counts, reverse=True), (
            f"{label} sweep {[round(s, 1) for s in sweep]} is not monotone: "
            f"{counts}")
        assert counts[-1] == 0, (
            f"{label} above the largest observed value still kept "
            f"{counts[-1]} events")
        for q, got in zip((25, 50, 75), counts[1:4]):
            expected = int((values >= np.percentile(values, q)).sum())
            assert got == expected, (
                f"{label} at the {q}th percentile kept {got} events, not the "
                f"{expected} that clear it -- the filter is not reading this "
                f"quantity")
        print(f"   [ok] {label:<16} p0/p25/p50/p75/max sweep "
              f"{[round(s, 1) for s in sweep]} -> {counts} events")

    # Every survivor clears the threshold in uV, on both arms at once. The two
    # arms measure different quantities (depth vs excursion), so each gets its
    # own median rather than one shared number that would empty the set.
    thresh = float(np.percentile(uv_ptp, 50))
    depth = float(np.percentile(uv_depth, 50))
    kept = ImprovedDetectSlowWave(method=method, neg_peak_thresh=depth,
                                  p2p_thresh=thresh)(data)
    assert len(kept), (f"nothing survives {depth:.1f}/{thresh:.1f} uV, so the "
                       f"check is vacuous")
    for e in kept:
        assert float(e['trough_val']) <= -depth, (
            f"kept an event with a {float(e['trough_val']):.1f} uV trough "
            f"against a {depth:.1f} uV floor")
        assert float(e['ptp']) >= thresh, (
            f"kept an event with {float(e['ptp']):.1f} uV peak-to-peak "
            f"against a {thresh:.1f} uV floor")
    print(f"   [ok] every one of the {len(kept)} survivors clears "
          f"{depth:.1f} uV of depth AND {thresh:.1f} uV peak-to-peak")

    # Rate-invariant, amplitude-sensitive -- and the old rule's inverse.
    fractions, old_fractions = {}, {}
    for s_freq in (128.0, 256.0, 500.0):
        sig = _asymmetric_so(s_freq)
        data = _make_chantime(sig, s_freq)
        raw = _raw_events(method, data)
        n_all = len(raw)
        fractions[s_freq] = len(
            ImprovedDetectSlowWave(method=method,
                                   p2p_thresh=thresh)(data)) / n_all
        old_fractions[s_freq] = sum(
            abs(float(e['ptp'])) >= LEGACY_PTP for e in raw) / n_all
    spread = max(fractions.values()) - min(fractions.values())
    old_spread = max(old_fractions.values()) - min(old_fractions.values())
    assert spread < 0.05, (
        f"a {thresh:.1f} uV floor keeps "
        f"{ {k: round(v, 3) for k, v in fractions.items()} } across sampling "
        f"rates -- a spread of {spread:.1%}, so it is still rate-dependent")
    assert old_spread > 0.5, (
        f"the old sample-count rule varied by only {old_spread:.1%} here, so "
        f"this signal does not demonstrate the defect")
    print(f"   [ok] retained fraction across 128/256/500 Hz: "
          f"{ {k: round(v, 3) for k, v in fractions.items()} } "
          f"(spread {spread:.1%}); the old rule: "
          f"{ {k: round(v, 3) for k, v in old_fractions.items()} } "
          f"(spread {old_spread:.1%})")

    s_freq = 256.0
    sig = _asymmetric_so(s_freq)
    n_1x = len(ImprovedDetectSlowWave(method=method, p2p_thresh=thresh)(
        _make_chantime(sig, s_freq)))
    n_3x = len(ImprovedDetectSlowWave(method=method, p2p_thresh=thresh)(
        _make_chantime(sig * 3.0, s_freq)))
    assert n_3x > n_1x, (
        f"tripling the recording's amplitude did not change what a {thresh:.1f} "
        f"uV floor keeps ({n_1x} vs {n_3x}), so it is not reading microvolts")
    print(f"   [ok] a 3x louder recording moves the same uV floor from "
          f"{n_1x} to {n_3x} events")


def test_massimini_family_never_sees_the_floor():
    """(3) The Massimini path keeps enforcing its criteria inside the detector.

    Its thresholds go to ``max_trough_amp`` / ``min_ptp`` and are applied by
    Wonambi and by the negative-half-wave re-gates, never by the post-hoc
    floor. Asserted two ways: the floor helper is not called at all, and
    passing Massimini2004's published −80/140 explicitly gives exactly the
    same events as passing nothing.
    """
    print("\n4. The Massimini family does not go through the floor:")

    s_freq = 256.0
    data = _make_chantime(_asymmetric_so(s_freq), s_freq)

    for method in ('Massimini2004', 'AASM/Massimini2004'):
        det = ImprovedDetectSlowWave(method=method)
        calls = []
        real = det._meets_amplitude_floor
        det._meets_amplitude_floor = lambda evt, _c=calls, _r=real: (
            _c.append(evt) or _r(evt))
        events = det(data)
        assert not calls, (
            f"{method}: the post-hoc uV floor ran on {len(calls)} events; it "
            f"must never touch a method whose thresholds are enforced inside "
            f"the detector")
        print(f"   [ok] {method:<20} {len(events):>3} events, floor called "
              f"{len(calls)} times")

    published = ImprovedDetectSlowWave(method='Massimini2004')(data)
    explicit = ImprovedDetectSlowWave(method='Massimini2004',
                                      neg_peak_thresh=-80.0,
                                      p2p_thresh=140.0)(data)
    assert _starts(published) == _starts(explicit), (
        f"Massimini2004 with its published -80/140 passed explicitly gave "
        f"{len(explicit)} events against {len(published)} by default, so the "
        f"explicit route is applying an extra filter")
    print(f"   [ok] Massimini2004: explicit -80/140 == published defaults "
          f"({len(published)} events)")


def _synthetic_recording(tmp, stages, s_freq=128.0, seed=1):
    """Write a scored synthetic EDF and open it as a (Dataset, Annotations).

    Deliberately 128 Hz: the sampling rate at which the sample-count filter
    rejected 100 % of Ngo2015/Staresina2015 events.

    Parameters
    ----------
    tmp : str
        Directory to write the EDF and scoring XML into.
    stages : sequence of str
        Stage of each consecutive 30 s epoch.
    s_freq : float, optional
        Sampling frequency in Hz. Default ``128.0``.
    seed : int, optional
        Seed for the signal jitter and noise. Default ``1``.

    Returns
    -------
    tuple
        ``(dataset, annotations)``.
    """
    from wonambi import Dataset
    from wonambi.attr import Annotations
    from wonambi.attr.annotations import create_empty_annotations
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data

    duration = 30.0 * len(stages)
    sig = _asymmetric_so(s_freq, duration=duration, seed=seed)

    data = create_data(datatype='ChanTime', n_trial=1, s_freq=s_freq,
                       chan_name=['Cz'], time=(0, duration))
    data.data[0] = np.asarray(sig, dtype='f')[None, :len(data.axis['time'][0])]

    edf = os.path.join(tmp, 'sub-T.edf')
    write_edf(data, edf)
    dataset = Dataset(edf)

    xml = os.path.join(tmp, 'sub-T_scoring.xml')
    create_empty_annotations(xml, dataset)
    annot = Annotations(xml)
    annot.add_rater('tester')
    for i, stg in enumerate(stages):
        annot.set_stage_for_epoch(i * 30.0, stg, save=True)
    return dataset, annot


def test_paralswa_forwards_none_for_every_method():
    """(4) ``None`` reaches the detector unchanged, for all four methods.

    ``ParalSWA.detect_slow_waves`` used to substitute −80.0/140.0 for
    Ngo2015/Staresina2015 before constructing the detector, which is how the
    140 reached the sample-count comparison. Checked at both levels: the
    arguments the detector is constructed with, and a real 128 Hz run that
    returns events where the old plumbing returned none.
    """
    print("\n5. ParalSWA forwards None for every method:")

    import logging

    import turtlewave_hdEEG.swprocessor as swp
    from turtlewave_hdEEG import ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_amp_floor_')
    real_cls = swp.DetectSlowWave
    try:
        dataset, annot = _synthetic_recording(tmp, ('NREM3',) * 4)
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out, exist_ok=True)

        captured = []

        class Spy(real_cls):
            """Records the kwargs ParalSWA constructs the detector with."""

            def __init__(self, *args, **kwargs):
                captured.append(dict(kwargs))
                super().__init__(*args, **kwargs)

        swp.DetectSlowWave = Spy

        for i, method in enumerate(('Massimini2004', 'AASM/Massimini2004',
                                    'Ngo2015', 'Staresina2015')):
            captured.clear()
            db = os.path.join(out, f'events_{i}.db')
            events = ParalSWA(
                dataset, annot,
                log_level=logging.CRITICAL).detect_slow_waves(
                    method=method, chan=['Cz'], frequency=(0.5, 4),
                    stage=['NREM3'], json_dir=out, db_path=db,
                    subject='sub-T', cat=(1, 1, 1, 0))

            assert captured, f"{method}: the detector was never constructed"
            for kw in captured:
                assert kw.get('neg_peak_thresh', 'missing') is None, (
                    f"{method}: ParalSWA substituted "
                    f"neg_peak_thresh={kw.get('neg_peak_thresh')} for the "
                    f"caller's None")
                assert kw.get('p2p_thresh', 'missing') is None, (
                    f"{method}: ParalSWA substituted "
                    f"p2p_thresh={kw.get('p2p_thresh')} for the caller's None")

            det = real_cls(method=method)
            assert det.min_neg_amp is None and det.min_ptp_amp is None, (
                f"{method}: the default detector still carries a floor "
                f"({det.min_neg_amp}, {det.min_ptp_amp})")

            print(f"   [ok] {method:<20} neg_peak_thresh=None, "
                  f"p2p_thresh=None -> {len(events)} events at 128 Hz")

            if method in ('Ngo2015', 'Staresina2015'):
                assert events, (
                    f"{method}: a 128 Hz run through ParalSWA returned no "
                    f"events -- the sample-count floor is back")
    finally:
        swp.DetectSlowWave = real_cls
        shutil.rmtree(tmp, ignore_errors=True)


class _LogCatcher(logging.Handler):
    """Collects records so a test can assert on what the resolver said."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text(self, level=None):
        """Return the formatted messages, optionally filtered by level.

        Parameters
        ----------
        level : int or None, optional
            Only include records at this exact level when given.

        Returns
        -------
        list of str
            The rendered messages.
        """
        return [r.getMessage() for r in self.records
                if level is None or r.levelno == level]


def test_rerun_replays_legacy_thresholds_as_published_criteria():
    """(rerun) A pre-4.3 recorded threshold must not become a µV floor.

    ``examples/rerun_detection.py`` re-applies the original run's recorded
    ``neg_peak_thresh``/``p2p_thresh``. For Ngo2015/Staresina2015 every shipped
    entry point injected a value the caller never chose, and that value was
    compared against a sample-index distance rather than microvolts. Replaying
    it verbatim would impose a criterion the original run never applied, so
    :func:`turtlewave_hdEEG.rerun.resolve_sw_amplitude_thresholds` reads it in
    the semantics it had at the time.

    Both branches are pinned: a known legacy default resolves to the published
    criteria, an unrecognised value is kept with a loud warning.
    """
    print("\n6. Re-run replay of pre-4.3 amplitude thresholds:")

    import logging as _logging

    from turtlewave_hdEEG.rerun import (LEGACY_INERT_SW_THRESHOLDS,
                                        resolve_sw_amplitude_thresholds)

    def resolve(method, neg, p2p, version=None):
        catcher = _LogCatcher()
        log = _logging.getLogger('turtlewave_hdEEG.rerun.test')
        log.handlers = [catcher]
        log.propagate = False
        log.setLevel(_logging.DEBUG)
        out = resolve_sw_amplitude_thresholds(method, neg, p2p,
                                              recorded_version=version,
                                              logger=log)
        return out, catcher

    # (1) every known legacy default, both methods, both a missing version and
    #     an explicit pre-fix one.
    for method in ('Ngo2015', 'Staresina2015'):
        for neg, p2p in sorted(LEGACY_INERT_SW_THRESHOLDS):
            for version in (None, '4.2.0', '4.0.1'):
                out, catcher = resolve(method, neg, p2p, version)
                assert out == (None, None), (
                    f"{method} {version}: recorded ({neg}, {p2p}) resolved to "
                    f"{out}, so a pre-4.3 default would bind as a uV floor")
                assert catcher.text(_logging.INFO), (
                    f"{method}: the resolution was silent")
                assert not catcher.text(_logging.WARNING), (
                    f"{method}: a known legacy default should not warn")
        print(f"   [ok] {method:<14} {len(LEGACY_INERT_SW_THRESHOLDS)} known "
              f"legacy defaults x 3 version strings -> (None, None), logged")

    # (2) an unrecognised pair is AMBIGUOUS: kept, and warned about loudly.
    out, catcher = resolve('Staresina2015', -60.0, 90.0)
    assert out == (None, 90.0), (
        f"an unrecognised pair resolved to {out}; the p2p value must be kept "
        f"and the provably-inert negative depth dropped")
    warnings = catcher.text(_logging.WARNING)
    assert warnings, "an unrecognised recorded threshold was kept silently"
    assert 'MICROVOLT' in warnings[0], warnings[0]
    print(f"   [ok] unrecognised (-60.0, 90.0) -> {out} with a warning: "
          f"{warnings[0][:72]}...")

    # (3) a lone positive p2p is kept as-is and still warns.
    out, catcher = resolve('Ngo2015', None, 200.0)
    assert out == (None, 200.0), out
    assert catcher.text(_logging.WARNING), "kept a uV floor silently"
    print(f"   [ok] unrecognised (None, 200.0) -> {out} with a warning")

    # (4) a run recorded AFTER the fix meant microvolts, so it is replayed.
    out, catcher = resolve('Staresina2015', -75.0, 75.0, '4.3.0')
    assert out == (-75.0, 75.0), (
        f"a 4.3.0 run's deliberate uV floor was discarded: {out}")
    assert not catcher.text(_logging.WARNING), catcher.text()
    print(f"   [ok] the same pair recorded by 4.3.0 is replayed unchanged: "
          f"{out}")

    # (5) the Massimini family is a passthrough -- its thresholds always meant
    #     microvolts and are enforced inside the detector.
    for method in ('Massimini2004', 'AASM/Massimini2004'):
        out, catcher = resolve(method, -80.0, 140.0)
        assert out == (-80.0, 140.0), f"{method} was rewritten to {out}"
        assert not catcher.records, f"{method}: unexpected log {catcher.text()}"
    print("   [ok] Massimini2004 / AASM/Massimini2004 pass through untouched "
          "and silently")

    # (6) nothing recorded stays nothing.
    out, catcher = resolve('Ngo2015', None, None)
    assert out == (None, None) and not catcher.records, (out, catcher.text())
    print("   [ok] (None, None) stays (None, None), silently")

    # (7) a params_json holding something non-numeric must not crash the
    #     re-run from inside a provenance helper.
    out, catcher = resolve('Staresina2015', 'default', 140.0)
    assert out == ('default', 140.0), out
    assert catcher.text(_logging.WARNING), "a non-numeric threshold was silent"
    print(f"   [ok] non-numeric recorded threshold passes through with a "
          f"warning: {out}")


def test_recover_run_scope_exposes_the_writing_version():
    """The resolver's version gate needs the version, so it must be recovered.

    ``recover_run_scope`` did not select ``detection_runs.turtlewave_version``,
    so a re-run could not tell a pre-4.3 recorded threshold (a sample count)
    from a post-4.3 one (real microvolts).
    """
    print("\n7. recover_run_scope returns the writing library version:")

    import logging

    import turtlewave_hdEEG
    from turtlewave_hdEEG import ParalSWA, dbwrite

    tmp = tempfile.mkdtemp(prefix='tw_runver_')
    try:
        dataset, annot = _synthetic_recording(tmp, ('NREM3',) * 4)
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out, exist_ok=True)
        db = os.path.join(out, 'neural_events.db')

        ParalSWA(dataset, annot,
                 log_level=logging.CRITICAL).detect_slow_waves(
                     method='Staresina2015', chan=['Cz'], frequency=(0.5, 1.25),
                     stage=['NREM3'], json_dir=out, db_path=db,
                     subject='sub-T', cat=(1, 1, 1, 0))

        rec = dbwrite.recover_run_scope(db, 'slow_wave', 'Staresina2015')
        assert rec is not None, "no run was recorded"
        assert 'turtlewave_version' in rec, sorted(rec)
        assert rec['turtlewave_version'] == turtlewave_hdEEG.__version__, (
            f"recovered {rec['turtlewave_version']!r}, expected "
            f"{turtlewave_hdEEG.__version__!r}")
        # And the params this run recorded are the published-criteria None,
        # so a re-run of a 4.3 run needs no legacy handling at all.
        assert rec['params'].get('neg_peak_thresh') is None, rec['params']
        assert rec['params'].get('p2p_thresh') is None, rec['params']
        print(f"   [ok] turtlewave_version={rec['turtlewave_version']!r} "
              f"recovered, with neg_peak_thresh/p2p_thresh recorded as None")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING THE SLOW-WAVE AMPLITUDE FLOOR (uV, not samples)")
    print("=======================================================")

    test_low_rate_asymmetric_wave_survives()
    test_defaults_equal_the_old_zero_workaround()
    test_explicit_threshold_filters_in_microvolts()
    test_massimini_family_never_sees_the_floor()
    test_paralswa_forwards_none_for_every_method()
    test_rerun_replays_legacy_thresholds_as_published_criteria()
    test_recover_run_scope_exposes_the_writing_version()

    print("\nAll amplitude-floor tests passed.")
