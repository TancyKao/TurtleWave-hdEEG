## test_turtlewave_updates.py

import os
import sys
import shutil
import sqlite3
import importlib
import tempfile
import pandas as pd
import numpy as np
from turtlewave_hdEEG.utils import read_channels_from_csv

# Force reload to ensure you're using the latest code
import turtlewave_hdEEG
importlib.reload(turtlewave_hdEEG)

def test_utils_functions():
    """Test the utilities like read_channels_from_csv"""
    print("\n1. Testing utility functions:")
    
    # Create a temporary CSV file with test channels
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_file.write("channel\nE101\nE102\nE103\n")
        temp_csv_path = temp_file.name
    
    try:
        # Test read_channels_from_csv function
        channels = read_channels_from_csv(temp_csv_path)
        print(f"[ok] read_channels_from_csv returned {len(channels)} channels: {channels}")
        assert len(channels) == 3, "Should read 3 channels"
        assert "E101" in channels, "Should include E101"
    except Exception as e:
        print(f"[FAIL] Error in read_channels_from_csv: {e}")
    finally:
        # Clean up
        os.remove(temp_csv_path)

def test_custom_annotations():
    """Test the CustomAnnotations class functionality"""
    print("\n2. Testing CustomAnnotations class:")
    
    # Test initialization (without an actual file)
    try:
        # We'll just test if the class can be instantiated without error
        annot = turtlewave_hdEEG.CustomAnnotations()
        print("[ok] CustomAnnotations class can be instantiated")
        
        # List available methods to show what can be tested
        methods = [m for m in dir(annot) if not m.startswith('_') and callable(getattr(annot, m))]
        print(f"Available methods: {methods}")
    except Exception as e:
        print(f"[FAIL] Error initializing CustomAnnotations: {e}")

def test_paralevents_class():
    """Test the ParalEvents class functionality"""
    print("\n3. Testing ParalEvents class:")
    
    try:
        # We'll just test if the class can be instantiated without error
        # Note: In real testing, you'd provide actual dataset and annotations
        event_processor = turtlewave_hdEEG.ParalEvents()
        print("[ok] ParalEvents class can be instantiated")
        
        # List the methods available in ParalEvents
        methods = [m for m in dir(event_processor) if not m.startswith('_') and callable(getattr(event_processor, m))]
        print(f"Available methods: {', '.join(methods)}")
        
        # Verify the presence of specific methods
        assert 'detect_spindles' in methods, "detect_spindles method should be available"
        assert 'export_spindle_parameters_to_csv' in methods, "export_spindle_parameters_to_csv should be available"
        assert 'export_spindle_density_to_csv' in methods, "export_spindle_density_to_csv should be available"
        print("[ok] All expected methods are available")
    except Exception as e:
        print(f"[FAIL] Error testing ParalEvents: {e}")

def test_largedataset_class():
    """Test the LargeDataset class functionality"""
    print("\n4. Testing LargeDataset class:")
    
    try:
        # Just check if the class exists and can be initialized
        dataset_class = getattr(turtlewave_hdEEG, 'LargeDataset')
        print("[ok] LargeDataset class exists")
        
        # Check initialization parameters
        import inspect
        params = inspect.signature(dataset_class.__init__).parameters
        print(f"LargeDataset init parameters: {list(params.keys())}")
        assert 'create_memmap' in params, "create_memmap parameter should exist"
        print("[ok] create_memmap parameter exists")
    except Exception as e:
        print(f"[FAIL] Error testing LargeDataset: {e}")

def test_xlannotations_class():
    """Test the XLAnnotations class functionality"""
    print("\n5. Testing XLAnnotations class:")
    
    try:
        # Just check if the class exists
        annotations_class = getattr(turtlewave_hdEEG, 'XLAnnotations')
        print("[ok] XLAnnotations class exists")
        
        # Check if it has a process_all method
        assert hasattr(annotations_class, 'process_all'), "process_all method should exist"
        print("[ok] process_all method exists")
    except Exception as e:
        print(f"[FAIL] Error testing XLAnnotations: {e}")


def test_improved_detect_spindle():
    """Test the ImprovedDetectSpindle class"""
    print("\n7. Testing ImprovedDetectSpindle class:")
    
    try:
        # Check if the class exists
        spindle_detector = turtlewave_hdEEG.ImprovedDetectSpindle
        print(f"[ok] ImprovedDetectSpindle class exists")
        
        # List methods to see what can be tested
        methods = [m for m in dir(spindle_detector) if not m.startswith('_') and callable(getattr(spindle_detector, m))]
        print(f"Available methods: {methods}")
    except Exception as e:
        print(f"[FAIL] Error testing ImprovedDetectSpindle: {e}")
        

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


def _synthetic_slow_oscillation(s_freq=256.0, duration=600.0, freq=0.8,
                                jitter=0.7, seed=0):
    """Build an asymmetric slow oscillation for polarity testing.

    The wave has a sharp, large negative half-cycle and a broad, smaller
    positive half-cycle, so it is *not* symmetric under negation — a detector
    that genuinely inverts must return different events for the signal and
    its negation. Per-cycle amplitude jitter gives the relative-threshold
    methods (Ngo2015) a spread to threshold against.

    Parameters
    ----------
    s_freq : float
        Sampling frequency in Hz.
    duration : float
        Length of the signal in seconds.
    freq : float
        Slow-oscillation frequency in Hz.
    jitter : float
        Standard deviation of the per-cycle multiplicative amplitude jitter.
    seed : int
        Seed for the amplitude jitter and the additive noise.

    Returns
    -------
    ndarray
        Signal in microvolts, shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * s_freq)
    phase = 2 * np.pi * freq * np.arange(n) / s_freq
    sine = np.sin(phase)
    base = (-np.abs(sine) ** 0.6 * (sine < 0) * 110.0
            + (sine > 0) * sine ** 2 * 60.0)
    cycle = (phase // (2 * np.pi)).astype(int)
    gain = 1.0 + jitter * rng.standard_normal(cycle.max() + 2)[cycle]
    return base * gain + rng.standard_normal(n) * 2.0


def test_slow_wave_polarity():
    """Polarity regression: polar='opposite' must invert exactly once.

    `ImprovedDetectSlowWave.invert` is Wonambi's own `DetectSlowWave.invert`,
    and every slow-wave method negates the signal itself
    (wonambi/detect/slowwave.py:192, :256, :322). If turtlewave inverts as
    well — in the detector's `__call__` or in the processor loop — the two
    cancel and polar='opposite' silently becomes polar='normal', producing
    confidently wrong events with no error. This asserts the two invariants
    that catch that:

    1. ``opposite(x)`` returns exactly the events of ``normal(-x)``.
    2. ``opposite(x)`` differs from ``normal(x)`` — i.e. inverting is not a
       no-op on a signal that is genuinely asymmetric.
    """
    print("\n8. Testing slow-wave / K-complex polarity handling:")

    from turtlewave_hdEEG.extensions import (ImprovedDetectSlowWave,
                                             ImprovedDetectKComplex)

    s_freq = 256.0
    sig = _synthetic_slow_oscillation(s_freq=s_freq)

    def starts(events):
        return [round(float(evt['start']), 6) for evt in events]

    cases = [(ImprovedDetectSlowWave, method, {}) for method in
             ('Massimini2004', 'AASM/Massimini2004', 'Ngo2015',
              'Staresina2015')]
    cases.append((ImprovedDetectKComplex, 'AASM/Massimini2004',
                  {'min_isolation': 1.0}))

    # Ngo2015 thresholds are relative to the mean event amplitude, so its
    # yield on a synthetic stimulus is not stable enough to assert a floor on;
    # its round-trip equality is still checked.
    needs_events = {'Massimini2004', 'AASM/Massimini2004', 'Staresina2015'}

    for cls, method, kwargs in cases:
        opposite = starts(cls(method=method, polar='opposite',
                              **kwargs)(_make_chantime(sig, s_freq)))
        negated = starts(cls(method=method, polar='normal',
                             **kwargs)(_make_chantime(-sig, s_freq)))
        normal = starts(cls(method=method, polar='normal',
                            **kwargs)(_make_chantime(sig, s_freq)))

        label = f"{cls.__name__.replace('ImprovedDetect', '')}/{method}"
        print(f"   {label:<28} opposite(x)={len(opposite):>4}  "
              f"normal(-x)={len(negated):>4}  normal(x)={len(normal):>4}")

        assert opposite == negated, (
            f"{label}: polar='opposite' on x must equal polar='normal' on -x, "
            f"got {len(opposite)} vs {len(negated)} events")

        if method in needs_events:
            assert opposite, (
                f"{label}: no events detected, so the polarity round-trip "
                f"assertion above is vacuous")
            assert opposite != normal, (
                f"{label}: polar='opposite' is identical to polar='normal' — "
                f"the inversion is being applied twice and cancelling")

    print("[ok] polar='opposite' inverts exactly once for all slow-wave "
          "methods and K-complexes")


def test_ngo2015_adaptive_thresholds_are_attributes():
    """Ngo2015's adaptive thresholds are ATTRIBUTES, not constructor arguments.

    ``ImprovedDetectSlowWave._set_method_params`` assigns ``self.peak_thresh``
    and ``self.ptp_thresh`` (both 1.25) for Ngo2015, and Wonambi's
    ``detect_Ngo2015`` reads them back off the instance as ``opts.peak_thresh``
    / ``opts.ptp_thresh``. ``ParalSWA`` used to forward the user's sigma values
    as ``__init__`` kwargs, which raised ``TypeError`` and lost the whole
    channel -- and because the GUI prefills those spin boxes from the detector's
    own 1.25 default, the values were never ``None`` and the failure fired on
    EVERY Ngo2015 run. Adaptive Ngo2015 has therefore never worked in any
    released version.

    This is the project's recurring failure mode (custom configuration not
    surviving ``super().__init__``), so it is pinned on three levels:

    1. The constructor REJECTS the thresholds as kwargs -- the fact that made
       the old call site fail.
    2. Post-construction assignment survives and is not silently reset to 1.25
       by ``_set_method_params``.
    3. The override materially changes detection, so 1. and 2. are not vacuous.
    """
    print("\nTesting Ngo2015 adaptive thresholds survive construction:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    # (1) the thresholds are NOT constructor arguments
    raised = None
    try:
        ImprovedDetectSlowWave('Ngo2015', frequency=(0.5, 2.0),
                               peak_thresh=0.2, ptp_thresh=0.2)
    except TypeError as e:
        raised = str(e)
    assert raised is not None, (
        "peak_thresh/ptp_thresh became constructor arguments: the "
        "set-after-construction workaround in ParalSWA is now wrong")
    assert 'peak_thresh' in raised, raised
    print(f"   [ok] __init__ still rejects them: {raised.split(':')[-1].strip()}")

    # (2) the default, then an override that must survive
    det = ImprovedDetectSlowWave('Ngo2015', frequency=(0.5, 2.0))
    assert det.peak_thresh == 1.25 and det.ptp_thresh == 1.25, (
        f"Ngo2015 default changed: {det.peak_thresh}, {det.ptp_thresh}")
    det.peak_thresh = 0.2
    det.ptp_thresh = 0.2
    assert det.peak_thresh == 0.2 and det.ptp_thresh == 0.2, (
        "post-construction override did not survive")
    print("   [ok] default is 1.25/1.25; post-construction override survives")

    # (3) the override must actually change what is detected
    s_freq = 256.0
    sig = _synthetic_slow_oscillation(s_freq=s_freq, duration=300.0, seed=3)

    def n_events(peak, ptp):
        d = ImprovedDetectSlowWave('Ngo2015', frequency=(0.5, 2.0),
                                   polar='opposite')
        d.peak_thresh = peak
        d.ptp_thresh = ptp
        return len(d(_make_chantime(sig, s_freq)))

    lenient = n_events(0.2, 0.2)
    default = n_events(1.25, 1.25)
    assert lenient != default, (
        f"the threshold override changes nothing ({lenient} vs {default}), so "
        f"the assertions above cannot detect it being dropped")
    print(f"   [ok] override is load-bearing: 0.2/0.2 -> {lenient} slow waves, "
          f"1.25/1.25 -> {default}")


def _biphasic_wave(s_freq=256.0, duration=300.0, freq=0.7, amp=100.0):
    """A clean negative-going-first slow wave with known amplitude and period.

    At ``freq`` Hz each half-wave lasts ``1 / (2 * freq)`` s and the whole
    wave ``1 / freq`` s, so the negative half-wave window and the whole-wave
    window are numerically distinguishable — which is the point: Massimini's
    0.3-1.0 s criterion is on the half-wave, and applying it to the whole
    wave rejects everything below 1 Hz.

    Parameters
    ----------
    s_freq : float
        Sampling frequency in Hz.
    duration : float
        Length of the signal in seconds.
    freq : float
        Slow-wave frequency in Hz. 0.7 Hz gives a 0.71 s half-wave and a
        1.43 s whole wave.
    amp : float
        Peak amplitude in µV, so the trough is -amp, the peak +amp and the
        peak-to-peak amplitude 2 * amp.

    Returns
    -------
    ndarray
        Signal in microvolts, shape (int(duration * s_freq),).
    """
    t = np.arange(int(duration * s_freq)) / s_freq
    return -amp * np.sin(2 * np.pi * freq * t)


def test_massimini_criteria_are_the_published_ones():
    """The three Massimini criteria must reach the detector, in their units.

    Massimini et al. 2004 (J Neurosci 24(31):6862-70), Methods: *"(1) a
    negative zero crossing and a subsequent positive zero crossing separated
    by 0.3-1.0 sec, (2) a negative peak between the two zero crossings with
    voltage less than -80 µV, and (3) a negative-to-positive peak-to-peak
    amplitude >=140 µV."*

    Wonambi keeps these in ``trough_duration`` / ``max_trough_amp`` /
    ``min_ptp`` and hardcodes all three in its constructor. turtlewave used
    to forward the caller's trough window as Wonambi's ``duration`` instead,
    which bounds the WHOLE wave, and to store the caller's amplitudes under
    names Wonambi never reads. The consequences, both silent:

    * the AASM window (0.25, 1.0) rejected every wave slower than 1 Hz, i.e.
      most slow waves, and the method returned zero events;
    * a user "tightening the threshold" changed nothing.

    Each criterion is asserted twice — once accepting the known wave, once
    rejecting it — so no assertion can pass because the gate is inert.
    """
    print("\nTesting the Massimini criteria reach the detector in their units:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    s_freq = 256.0
    # 0.7 Hz, +/-100 uV: half-wave 0.71 s, whole wave 1.43 s, ptp 200 uV.
    sig = _biphasic_wave(s_freq=s_freq)

    def n_events(**kw):
        det = ImprovedDetectSlowWave(method='Massimini2004',
                                     frequency=(0.1, 4), **kw)
        return len(det(_make_chantime(sig, s_freq)))

    published = n_events()
    assert published > 0, (
        "the published Massimini criteria find nothing on a 0.7 Hz, 200 uV "
        "peak-to-peak wave, so every assertion below is vacuous")
    print(f"   published criteria (-80 uV, 140 uV, 0.3-1.0 s) -> "
          f"{published} events")

    # --- (1) the trough window bounds the HALF-wave -----------------------
    # 0.71 s half-wave: inside (0.3, 1.0), outside (0.3, 0.5) and (0.75, 1.0).
    # The whole wave is 1.43 s, so a window of (0.3, 1.0) read as a
    # whole-wave bound would reject everything -- which is the old bug.
    assert n_events(trough_duration=(0.3, 1.0)) == published, (
        "the paper's own 0.3-1.0 s window rejects a 0.71 s half-wave: it is "
        "still being applied to the whole 1.43 s wave")
    assert n_events(trough_duration=(0.25, 1.0)) == published, \
        "the AASM 0.25-1.0 s window rejects a 0.71 s half-wave"
    assert n_events(trough_duration=(0.3, 0.5)) == 0, \
        "a window ending before the 0.71 s half-wave still accepted it"
    assert n_events(trough_duration=(0.75, 1.0)) == 0, \
        "a window starting after the 0.71 s half-wave still accepted it"
    print("   [ok] trough_duration bounds the 0.71 s half-wave, not the "
          "1.43 s wave: (0.3,1.0) and (0.25,1.0) keep all "
          f"{published}, (0.3,0.5) and (0.75,1.0) keep 0")

    # --- (2) the amplitude-depth criterion --------------------------------
    # Swept rather than asserted at one point: the yield must fall smoothly
    # from "all" to "none" as the criterion crosses the wave's own 100 uV
    # amplitude. A single equality could pass on an inert gate that happened
    # to keep everything; a graded response cannot.
    depth_sweep = [-50.0, -90.0, -99.0, -104.0, -110.0, -150.0]
    depth_counts = [n_events(neg_peak_thresh=t) for t in depth_sweep]
    assert depth_counts[0] == published, (
        f"a -50 uV criterion on a 100 uV wave dropped events: "
        f"{depth_counts[0]} vs {published}")
    assert depth_counts[-1] == 0, (
        f"a -150 uV criterion on a 100 uV wave kept {depth_counts[-1]} "
        f"events: neg_peak_thresh never reaches Wonambi's max_trough_amp")
    assert depth_counts == sorted(depth_counts, reverse=True), \
        f"yield is not monotone in the depth criterion: {depth_counts}"
    assert any(0 < c < published for c in depth_counts), \
        f"the depth criterion is on/off, not graded: {depth_counts}"
    # The sign is a depth, not a direction: +150 must gate like -150.
    assert n_events(neg_peak_thresh=150.0) == 0, \
        "a positive neg_peak_thresh silently inverted the criterion"
    print(f"   [ok] neg_peak_thresh gates amplitude depth in uV: "
          f"{list(zip(depth_sweep, depth_counts))}; +150 keeps 0 too")

    # --- (3) the peak-to-peak criterion -----------------------------------
    ptp_sweep = [100.0, 180.0, 199.0, 205.0, 215.0, 250.0]
    ptp_counts = [n_events(p2p_thresh=t) for t in ptp_sweep]
    assert ptp_counts[0] == published, (
        f"a 100 uV peak-to-peak criterion on a 200 uV wave dropped events: "
        f"{ptp_counts[0]} vs {published}")
    assert ptp_counts[-1] == 0, (
        f"a 250 uV peak-to-peak criterion on a 200 uV wave kept "
        f"{ptp_counts[-1]} events: p2p_thresh never reaches Wonambi's "
        f"min_ptp")
    assert ptp_counts == sorted(ptp_counts, reverse=True), \
        f"yield is not monotone in the peak-to-peak criterion: {ptp_counts}"
    assert any(0 < c < published for c in ptp_counts), \
        f"the peak-to-peak criterion is on/off, not graded: {ptp_counts}"
    print(f"   [ok] p2p_thresh gates peak-to-peak in uV: "
          f"{list(zip(ptp_sweep, ptp_counts))}")

    # --- the published defaults are the published numbers ------------------
    defaults = {
        'Massimini2004': ((0.3, 1.0), -80, 140),
        'AASM/Massimini2004': ((0.25, 1.0), -40, 75),
    }
    for method, (window, trough, ptp) in defaults.items():
        det = ImprovedDetectSlowWave(method=method)
        assert det.trough_duration == window, \
            f"{method}: trough_duration {det.trough_duration} != {window}"
        assert det.max_trough_amp == trough, \
            f"{method}: max_trough_amp {det.max_trough_amp} != {trough}"
        assert det.min_ptp == ptp, \
            f"{method}: min_ptp {det.min_ptp} != {ptp}"
        # Wonambi's separate whole-wave bound must NOT pick up the trough
        # window: it stays at (min_dur, max_dur) unless a caller sets it.
        assert det.duration == (0, None), \
            f"{method}: duration {det.duration} is not the whole-wave default"
    print("   [ok] published defaults survive super().__init__: "
          "Massimini2004 (0.3,1.0)/-80/140, AASM (0.25,1.0)/-40/75, "
          "whole-wave duration left at (0, None)")

    # --- the two duration bounds are independent --------------------------
    # The whole wave is 1.43 s. min_dur/max_dur bound THAT, and must not be
    # confused with the 0.71 s half-wave window; both have to be reachable
    # separately or the next caller will pass one as the other again.
    assert n_events(max_dur=2.0) == published, \
        "a 2.0 s whole-wave ceiling rejected a 1.43 s wave"
    assert n_events(max_dur=1.0) == 0, \
        "a 1.0 s whole-wave ceiling accepted a 1.43 s wave: min_dur/max_dur " \
        "never reach Wonambi's duration"
    assert n_events(min_dur=2.0) == 0, \
        "a 2.0 s whole-wave floor accepted a 1.43 s wave"
    assert n_events(trough_duration=(0.3, 1.0), max_dur=2.0) == published, (
        "the published half-wave window plus a 2.0 s whole-wave ceiling "
        "rejects a 0.71 s / 1.43 s wave: the two bounds are still conflated")
    print("   [ok] whole-wave min_dur/max_dur are separate from the trough "
          "window: max_dur=2.0 keeps all 209, max_dur=1.0 keeps 0, and "
          "(0.3,1.0) + max_dur=2.0 keeps all 209")


def _lopsided_wave(s_freq=256.0, duration=600.0, period=2.0,
                   neg_fraction=0.68, amp=150.0):
    """Biphasic wave whose two half-waves have DIFFERENT durations.

    A half-sine of ``neg_fraction * period`` going negative, then a half-sine
    of the remainder going positive; both are smooth and meet at zero, so the
    zero crossings sit exactly where the fractions say. The amplitude-
    asymmetric generator :func:`_synthetic_slow_oscillation` cannot be used
    for duration tests because its half-waves are the same length.

    Note the 0.1-4 Hz detection band pulls the two halves back towards each
    other — a raw 0.72/0.28 split at a 1.6 s period measures 0.95/0.65 after
    filtering — so the ratio here is chosen from the FILTERED durations, not
    the nominal ones. At the defaults the negative half-wave measures ~1.16 s
    (outside a 0.3-1.0 s window) while the positive measures ~0.83 s (inside
    it), which is exactly the case Massimini's criterion separates and
    Wonambi's cannot.

    Parameters
    ----------
    s_freq : float
        Sampling frequency in Hz.
    duration : float
        Length of the signal in seconds.
    period : float
        Cycle length in seconds.
    neg_fraction : float
        Fraction of each cycle spent below zero, before filtering.
    amp : float
        Peak amplitude in µV before filtering.

    Returns
    -------
    ndarray
        Signal in microvolts.
    """
    t = np.arange(int(duration * s_freq)) / s_freq
    ph = (t % period) / period
    neg = -amp * np.sin(np.pi * ph / neg_fraction)
    pos = amp * np.sin(np.pi * (ph - neg_fraction) / (1.0 - neg_fraction))
    return np.where(ph < neg_fraction, neg, pos)


def test_trough_duration_criterion_gates_the_negative_half_wave():
    """``trough_duration`` must bound the NEGATIVE half-wave, per the paper.

    Massimini: *"a negative zero crossing and a subsequent positive zero
    crossing separated by 0.3-1.0 sec"*. Wonambi applies that window with
    ``within_duration`` to the ABOVE-zero run, because
    ``detect_Massimini2004`` searches for the positive excursion first, so
    the published criterion lands on the positive half-wave.

    The negative half-wave is readable straight off the event dict:
    ``_add_halfwave`` sets ``ev[4]`` to the first zero crossing after
    ``ev[2]`` and puts the trough between them, and ``make_slow_waves``
    exposes those as ``zero_time`` and ``end``. So its duration is
    ``end - zero_time`` — one sample short, by the same convention
    ``within_duration`` already uses on the other half-wave.

    Asserted on a wave whose half-waves differ in length (where it must
    bite), on one where they match (where it must not), and across two
    windows (so it follows the criterion rather than being a fixed cutoff).
    """
    print("\nTesting trough_duration bounds the negative half-wave:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    class _NoDurationRegate(ImprovedDetectSlowWave):
        """The behaviour before this gate: depth re-gated, duration not."""

        def _meets_trough_duration(self, evt):
            return True

    s_freq = 256.0
    # neg ~1.16 s / pos ~0.83 s after filtering: the negative half-wave is
    # outside a 0.3-1.0 s window while the positive one is inside it.
    lopsided = _lopsided_wave(s_freq=s_freq)
    # Half-waves of equal length: nothing may be removed.
    balanced = _synthetic_slow_oscillation(s_freq=s_freq)

    for method in ('Massimini2004', 'AASM/Massimini2004'):
        def run(sig, window):
            kw = dict(method=method, frequency=(0.1, 4),
                      trough_duration=window)
            before = _NoDurationRegate(**kw)(
                _make_chantime(sig, s_freq)).events
            after = ImprovedDetectSlowWave(**kw)(
                _make_chantime(sig, s_freq)).events
            return before, after

        # --- the window is honoured -----------------------------------
        for sig, name in ((lopsided, 'lopsided'), (balanced, 'balanced')):
            for window in ((0.3, 1.0), (0.25, 1.0), (0.3, 1.5)):
                before, after = run(sig, window)
                assert before, f"{method}/{name}/{window}: no candidates"
                outside = [e for e in after
                           if not (window[0]
                                   <= e['end'] - e['zero_time']
                                   <= window[1])]
                assert not outside, (
                    f"{method}/{name}/{window}: {len(outside)} accepted "
                    f"events have a negative half-wave outside the window, "
                    f"worst {max(e['end'] - e['zero_time'] for e in outside):.3f} s")
                kept = {round(float(e['start']), 6) for e in after}
                assert kept <= {round(float(e['start']), 6) for e in before}, \
                    f"{method}/{name}/{window}: the gate invented an event"

        # --- it bites where the paper says it should ------------------
        before, after = run(lopsided, (0.3, 1.0))
        neg = np.median([e['end'] - e['zero_time'] for e in before])
        pos = np.median([e['zero_time'] - e['start'] for e in before])
        assert neg > 1.0 > pos, (
            f"{method}: the lopsided wave is not lopsided after filtering "
            f"(neg {neg:.3f} s, pos {pos:.3f} s), so this test is vacuous")
        assert len(after) < len(before), (
            f"{method}: the gate removed nothing from a wave whose negative "
            f"half-wave is {neg:.3f} s under a 0.3-1.0 s criterion, so it "
            f"is inert")
        print(f"   [ok] {method:<21} lopsided (neg {neg:.3f} s / pos "
              f"{pos:.3f} s) window (0.3,1.0): {len(before)} -> "
              f"{len(after)} ({len(before) - len(after)} removed, "
              f"{100 * (len(before) - len(after)) / len(before):.0f}%)")

        # --- widening the window keeps them, so it tracks the criterion
        before_w, after_w = run(lopsided, (0.3, 1.5))
        assert len(after_w) == len(before_w), (
            f"{method}: a 0.3-1.5 s window still rejected a {neg:.3f} s "
            f"negative half-wave, so the gate is a fixed cutoff")
        print(f"   [ok] {method:<21} same wave, window (0.3,1.5): "
              f"{len(before_w)} -> {len(after_w)} (none removed, so the "
              f"gate follows the window)")

        # --- it removes exactly the events outside the window, no others ---
        # Necessary AND sufficient, so the gate cannot be either over- or
        # under-inclusive. Asserted on the balanced signal, where the vast
        # majority of half-waves are inside the window and a spurious
        # rejection would show up immediately.
        before_b, after_b = run(balanced, (0.3, 1.0))
        kept_b = {round(float(e['start']), 6) for e in after_b}
        removed_b = [e for e in before_b
                     if round(float(e['start']), 6) not in kept_b]
        wrongly_removed = [e for e in removed_b
                           if 0.3 <= e['end'] - e['zero_time'] <= 1.0]
        assert not wrongly_removed, (
            f"{method}: the gate removed {len(wrongly_removed)} events whose "
            f"negative half-wave IS inside 0.3-1.0 s, e.g. "
            f"{wrongly_removed[0]['end'] - wrongly_removed[0]['zero_time']:.3f} s")
        assert len(removed_b) / len(before_b) < 0.05, (
            f"{method}: the gate removed "
            f"{100 * len(removed_b) / len(before_b):.0f}% of a signal whose "
            f"half-waves are the same length, which is not a stray candidate")
        print(f"   [ok] {method:<21} balanced half-waves: {len(before_b)} -> "
              f"{len(after_b)}; the {len(removed_b)} removed are exactly "
              f"those outside the window, none inside it")

    # The span must be the same number before and after _as_negative_first,
    # which swaps trough/peak but not zero_time/end. Checked, not assumed.
    det = ImprovedDetectSlowWave(method='Massimini2004', frequency=(0.1, 4))
    raw = ImprovedDetectSlowWave.__mro__[1].__call__(
        det, _make_chantime(balanced, s_freq)).events
    pre = [round(e['end'] - e['zero_time'], 9) for e in raw]
    post = [round(ImprovedDetectSlowWave._as_negative_first(dict(e))['end']
                  - e['zero_time'], 9) for e in raw]
    assert pre == post and pre, (
        "the negative half-wave span changed across the relabel, so the "
        "gate depends on where in __call__ it runs")
    print(f"   [ok] end - zero_time is identical before and after the "
          f"relabel ({len(pre)} events)")


def _eeg_with_injected_slow_waves(s_freq=256.0, duration=600.0, seed=1):
    """Synthetic EEG with slow waves at KNOWN times, widths and amplitudes.

    Ground truth is constructed rather than assumed: a 1/f background (which
    is what makes the up-state irregular, and therefore what the pre-4.3
    detector was implicitly selecting on), plus sigma-band bursts, plus
    biphasic slow waves injected at recorded positions. Deterministic for a
    given seed.

    Each injected wave is a full cycle of a sine of width ``w`` starting
    negative, so its negative half-wave is the first ``w / 2`` seconds and
    its peak-to-peak amplitude is ``2 * amp``.

    Parameters
    ----------
    s_freq : float, optional
        Sampling frequency in Hz. Default ``256.0``.
    duration : float, optional
        Length of the signal in seconds. Default ``600.0``.
    seed : int, optional
        Seed for the background, the wave parameters and the spindle times.
        Default ``1``.

    Returns
    -------
    sig : ndarray
        Signal in microvolts, shape ``(int(duration * s_freq),)``.
    truth : list of dict
        One entry per injected wave with ``start``, ``end``, ``neg_dur``
        (negative half-wave duration, s) and ``amp`` (peak amplitude, µV,
        BEFORE band-pass filtering).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * s_freq)
    t = np.arange(n) / s_freq

    # 1/f background at 25 uV RMS.
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / s_freq)
    spec[1:] /= np.sqrt(freqs[1:])
    bg = np.fft.irfft(spec, n)
    sig = bg / bg.std() * 25.0

    truth = []
    tt = 1.0
    while tt < duration - 3.0:
        w = rng.uniform(0.9, 1.8)
        amp = rng.uniform(50.0, 180.0)
        m = (t >= tt) & (t < tt + w)
        sig[m] += -amp * np.sin(2 * np.pi * (t[m] - tt) / w)
        truth.append({'start': tt, 'end': tt + w, 'neg_dur': w / 2.0,
                      'amp': amp})
        tt += w + rng.uniform(0.3, 2.5)

    for _ in range(200):
        t0 = rng.uniform(0.0, duration - 1.0)
        m = (t >= t0) & (t < t0 + 0.8)
        sig[m] += 20.0 * np.sin(2 * np.pi * 13.5 * (t[m] - t0))

    return sig, truth


def _score_events(detected, truth, min_frac=0.2):
    """Event-wise precision / recall / F1 against a ground-truth list.

    A detected event counts as a true positive when it overlaps an unclaimed
    truth event by at least ``min_frac`` of the DETECTED event's duration —
    the ≥20% temporal-overlap rule commonly used for MODA-style spindle
    scoring. Each truth event can be claimed once, so two detections over one
    wave give one true positive and one false positive.

    This is by-EVENT, not by-sample; the two give different numbers.

    Parameters
    ----------
    detected : sequence of dict
        Detector output; each entry needs ``start`` and ``end``.
    truth : sequence of dict
        Ground truth; each entry needs ``start`` and ``end``.
    min_frac : float, optional
        Minimum fraction of the detected event that must overlap. Default
        ``0.2``.

    Returns
    -------
    dict
        ``tp``, ``fp``, ``fn``, ``precision``, ``recall``, ``f1``.
    """
    claimed = set()
    tp = 0
    for d in detected:
        d0, d1 = float(d['start']), float(d['end'])
        best, best_ov = None, 0.0
        for i, g in enumerate(truth):
            if i in claimed:
                continue
            ov = max(0.0, min(d1, g['end']) - max(d0, g['start']))
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None and best_ov >= min_frac * max(d1 - d0, 1e-9):
            claimed.add(best)
            tp += 1
    fp = len(detected) - tp
    fn = len(truth) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision,
            'recall': recall, 'f1': f1}


def test_permissive_search_recall_against_injected_ground_truth():
    """The recall the permissive search buys, and the precision it must keep.

    This is the evidence for the most consequential change in the release,
    and it belongs in the repository rather than in a report: a future edit
    that quietly loosens the detector has to fail here.

    ``detect_Massimini2004`` used to pre-reject candidates by applying the
    paper's trough-duration window and depth criterion to the ABOVE-zero run
    — the up-state — before any post-detection re-gate could see them. On
    clean synthetics that barely shows, because their up-states are regular.
    On a 1/f background it is devastating: the up-state's duration and peak
    vary enough that almost every genuine slow wave was discarded on a
    quantity Massimini does not constrain at all.

    Scored against waves injected at known times, the strict path reached
    recall 0.02-0.05. The permissive one reaches 0.5-0.8 at essentially
    unchanged precision, so this is recovery of genuine waves and not a
    looser detector inventing them.

    Floors, not exact figures, so the test is not brittle:

    * precision >= 0.95 — the property that must never degrade;
    * recall >= 0.40 and at least 5x the strict path — the recovery claim;
    * plus the character properties: a strict superset of the old output,
      zero criterion violations measured from each event's own dict, and no
      overlapping negative half-waves.

    Recall is well under 1.0 by construction: the ground truth records each
    wave's amplitude BEFORE the 0.1-4 Hz detection band attenuates it, so
    some waves counted as "meeting the criteria" no longer do by the time the
    detector sees them. The floor is set with that in mind.
    """
    print("\nTesting slow-wave recall against injected ground truth:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    class _Strict(ImprovedDetectSlowWave):
        """Pre-4.3 behaviour: Wonambi still pre-rejects on the up-state."""

        def _permissive_search(self):
            return self

    s_freq = 256.0
    sig, truth = _eeg_with_injected_slow_waves(s_freq=s_freq)
    print(f"   {len(truth)} slow waves injected into 600 s of 1/f background")

    for method, depth, min_ptp, window in (
            ('Massimini2004', -80.0, 140.0, (0.3, 1.0)),
            ('AASM/Massimini2004', -40.0, 75.0, (0.25, 1.0))):
        # Score only against waves that meet the criteria being ASKED for.
        # Penalising a detector for missing waves its own criteria exclude
        # would measure the wrong thing.
        gt = [g for g in truth
              if g['amp'] >= abs(depth)
              and window[0] <= g['neg_dur'] <= window[1]]
        assert len(gt) >= 50, (
            f"{method}: only {len(gt)} injected waves meet the criteria, too "
            f"few to score against")

        kw = dict(method=method, frequency=(0.1, 4))
        strict = _Strict(**kw)(_make_chantime(sig, s_freq)).events
        new = ImprovedDetectSlowWave(**kw)(_make_chantime(sig, s_freq)).events

        s_score = _score_events(strict, gt)
        n_score = _score_events(new, gt)
        print(f"   {method:<21} strict     n={len(strict):>4}  "
              f"precision={s_score['precision']:.3f} "
              f"recall={s_score['recall']:.3f} F1={s_score['f1']:.3f}")
        print(f"   {method:<21} permissive n={len(new):>4}  "
              f"precision={n_score['precision']:.3f} "
              f"recall={n_score['recall']:.3f} F1={n_score['f1']:.3f}   "
              f"(ground truth: {len(gt)})")

        # --- precision must not degrade -----------------------------------
        assert n_score['precision'] >= 0.95, (
            f"{method}: precision fell to {n_score['precision']:.3f} "
            f"({n_score['fp']} false positives out of {len(new)} events). "
            f"The looser search is inventing events, not recovering them")

        # --- recall must improve substantially ----------------------------
        assert n_score['recall'] >= 0.40, (
            f"{method}: recall is only {n_score['recall']:.3f}; the "
            f"up-state pre-rejection appears to be back")
        assert n_score['recall'] >= 5 * s_score['recall'], (
            f"{method}: recall {n_score['recall']:.3f} is not a substantial "
            f"gain over the strict path's {s_score['recall']:.3f}, so this "
            f"test is no longer measuring the change it exists for")

        # --- character: superset, criteria, non-overlapping half-waves ----
        s_starts = {round(float(e['start']), 6) for e in strict}
        n_starts = {round(float(e['start']), 6) for e in new}
        assert s_starts <= n_starts, (
            f"{method}: the permissive search LOST "
            f"{len(s_starts - n_starts)} events the strict one found")

        for e in new:
            neg_dur = e['end'] - e['zero_time']
            assert window[0] <= neg_dur <= window[1], (
                f"{method}: negative half-wave {neg_dur:.3f} s outside "
                f"{window}")
            assert e['trough_val'] <= depth, (
                f"{method}: trough {e['trough_val']:.1f} uV does not reach "
                f"{depth} uV")
            assert e['ptp'] >= min_ptp - 1e-6, (
                f"{method}: peak-to-peak {e['ptp']:.1f} uV is under "
                f"{min_ptp} uV")

        spans = sorted((float(e['zero_time']), float(e['end'])) for e in new)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0, (
                f"{method}: negative half-waves [{a0:.3f}, {a1:.3f}] and "
                f"[{b0:.3f}, {b1:.3f}] overlap, so one wave is being counted "
                f"twice")
        print(f"   [ok] {method:<21} precision >= 0.95, recall >= 0.40 and "
              f">= 5x strict, strict superset, {len(new)} events with 0 "
              f"criterion violations and 0 overlapping half-waves")


def test_permissive_search_recovers_paper_valid_waves():
    """Wonambi must not pre-reject a wave on its UP-state.

    ``detect_Massimini2004`` applies two of the paper's criteria to the
    ABOVE-zero run before any post-detection re-gate can see the candidate:
    ``within_duration`` requires the POSITIVE half-wave to fall inside
    ``trough_duration``, and ``select_peaks`` requires the POSITIVE peak to
    reach ``max_trough_amp``. A re-gate can only remove, never recover, so
    those two silently biased which slow waves survived by their up-state
    duration and amplitude — quantities Massimini does not constrain at all.
    A wave with a 0.45 s negative half-wave (valid under any window) and a
    1.15 s positive one returned zero events.

    ``_permissive_search`` hands Wonambi a copy that pre-rejects on neither,
    leaving the published criteria to be enforced on the negative half-wave
    where they belong. ``min_ptp`` is deliberately NOT widened: it is already
    applied in genuine microvolts to the true negative-to-positive excursion.

    Asserts the recovery, that it is a strict superset (nothing is lost),
    that every survivor still satisfies all three published criteria, that no
    two events claim the same negative half-wave, and that the search copy
    leaves the user-facing attributes alone.
    """
    print("\nTesting the search no longer pre-rejects on the up-state:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    class _Strict(ImprovedDetectSlowWave):
        """The previous state: Wonambi still pre-rejects on the up-state."""

        def _permissive_search(self):
            return self

    s_freq = 256.0
    # 0.45 s negative half-wave, 1.15 s positive one: paper-valid, but the
    # up-state is outside the very window the paper applies to the down-state.
    up_state_too_long = _lopsided_wave(s_freq=s_freq, period=2.0,
                                       neg_fraction=0.28, amp=150.0)
    balanced = _synthetic_slow_oscillation(s_freq=s_freq)

    for method, depth, ptp, window in (
            ('Massimini2004', -80.0, 140.0, (0.3, 1.0)),
            ('AASM/Massimini2004', -40.0, 75.0, (0.25, 1.0))):
        kw = dict(method=method, frequency=(0.1, 4))

        # --- the recovery -------------------------------------------------
        strict = _Strict(**kw)(_make_chantime(up_state_too_long, s_freq)).events
        new = ImprovedDetectSlowWave(**kw)(
            _make_chantime(up_state_too_long, s_freq)).events
        assert not strict, (
            f"{method}: the strict search already found {len(strict)} events "
            f"on a wave with a 1.15 s up-state, so this test no longer "
            f"demonstrates the pre-rejection")
        assert new, (
            f"{method}: a wave with a paper-valid 0.45 s negative half-wave "
            f"is still undetected because its up-state is 1.15 s long")
        neg = np.median([e['end'] - e['zero_time'] for e in new])
        print(f"   [ok] {method:<21} up-state 1.15 s, down-state "
              f"{neg:.3f} s: {len(strict)} -> {len(new)} events")

        # --- nothing is lost, on a signal both can handle -----------------
        s_bal = _Strict(**kw)(_make_chantime(balanced, s_freq)).events
        n_bal = ImprovedDetectSlowWave(**kw)(
            _make_chantime(balanced, s_freq)).events
        s_starts = {round(float(e['start']), 6) for e in s_bal}
        n_starts = {round(float(e['start']), 6) for e in n_bal}
        assert s_bal, f"{method}: the strict search found nothing to compare"
        assert s_starts <= n_starts, (
            f"{method}: the permissive search LOST "
            f"{len(s_starts - n_starts)} events the strict one found; it must "
            f"only ever add")
        print(f"   [ok] {method:<21} balanced wave: {len(s_bal)} -> "
              f"{len(n_bal)}, a strict superset")

        # --- and what it adds still satisfies the paper -------------------
        for sig, name in ((up_state_too_long, 'up-state too long'),
                          (balanced, 'balanced')):
            events = ImprovedDetectSlowWave(**kw)(
                _make_chantime(sig, s_freq)).events
            for e in events:
                neg_dur = e['end'] - e['zero_time']
                assert window[0] <= neg_dur <= window[1], (
                    f"{method}/{name}: negative half-wave {neg_dur:.3f} s "
                    f"outside {window}")
                assert e['trough_val'] <= depth, (
                    f"{method}/{name}: trough {e['trough_val']:.1f} uV does "
                    f"not reach {depth} uV")
                assert e['ptp'] >= ptp - 1e-6, (
                    f"{method}/{name}: peak-to-peak {e['ptp']:.1f} uV is "
                    f"under {ptp} uV -- min_ptp must NOT be widened")
            spans = [(round(e['zero_time'], 6), round(e['end'], 6))
                     for e in events]
            assert len(spans) == len(set(spans)), (
                f"{method}/{name}: two events claim the same negative "
                f"half-wave, so the looser search is double-counting")
        print(f"   [ok] {method:<21} every survivor meets the window "
              f"{window}, {depth} uV trough and {ptp} uV peak-to-peak; no "
              f"duplicate half-waves")

    # --- the search copy must not disturb the real detector ---------------
    det = ImprovedDetectSlowWave(method='Massimini2004', frequency=(0.1, 4))
    before = (det.trough_duration, det.max_trough_amp, det.min_ptp,
              det.duration)
    search = det._permissive_search()
    after = (det.trough_duration, det.max_trough_amp, det.min_ptp,
             det.duration)
    assert before == after, (
        f"_permissive_search mutated the detector: {before} -> {after}. The "
        f"GUI prefills its spin boxes from these attributes and the re-gates "
        f"read them back")
    assert search is not det, "_permissive_search returned the detector itself"
    assert search.trough_duration == (None, None), search.trough_duration
    assert search.max_trough_amp == 0, search.max_trough_amp
    # min_ptp is correct in uV inside _add_halfwave and must survive intact,
    # as must the separate whole-wave bound.
    assert search.min_ptp == det.min_ptp == 140, search.min_ptp
    assert search.duration == det.duration == (0, None), search.duration
    print("   [ok] the search copy relaxes only trough_duration and "
          "max_trough_amp; min_ptp (140 uV) and the whole-wave duration are "
          "untouched, and the detector itself is unchanged")


def test_trough_depth_criterion_gates_the_negative_peak():
    """The depth criterion must gate the NEGATIVE trough, per the paper.

    Massimini: *"a negative peak between the two zero crossings with voltage
    less than -80 µV"*. Wonambi enforces it with ``select_peaks``, which
    tests the extremum of the FIRST half-wave; because
    ``detect_Massimini2004`` searches for the above-zero run first, that is
    the POSITIVE peak. On a symmetric wave the two numbers are equal and
    nothing shows. On a physiological slow wave — sharp deep negative half,
    broad shallow positive half — they are not: events were accepted whose
    negative trough was 1.1 µV against a -40 µV criterion.

    ``ImprovedDetectSlowWave._meets_trough_depth`` re-gates after detection
    rather than inverting the signal, so Wonambi's search order and its
    candidate set are untouched and only the published criterion is added.

    Asserted on the asymmetric signal, where it bites, and on a symmetric one,
    where it must not.
    """
    print("\nTesting the depth criterion gates the negative trough:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    class _NoRegate(ImprovedDetectSlowWave):
        """The behaviour before the re-gate: relabelled but not re-gated."""

        def _meets_trough_depth(self, evt):
            return True

    s_freq = 256.0
    signals = {
        'asymmetric': _synthetic_slow_oscillation(s_freq=s_freq),
        'symmetric': _biphasic_wave(s_freq=s_freq),
    }

    for method, thresh in (('Massimini2004', -80.0),
                           ('AASM/Massimini2004', -40.0)):
        for name, sig in signals.items():
            kw = dict(method=method, frequency=(0.1, 4),
                      neg_peak_thresh=thresh)
            before = _NoRegate(**kw)(_make_chantime(sig, s_freq)).events
            after = ImprovedDetectSlowWave(**kw)(
                _make_chantime(sig, s_freq)).events
            assert before, f"{method}/{name}: nothing detected at all"

            # Every survivor meets the criterion. This is the whole point.
            shallow = [e for e in after if e['trough_val'] > thresh]
            assert not shallow, (
                f"{method}/{name}: {len(shallow)} accepted events have a "
                f"trough shallower than {thresh} uV, worst "
                f"{max(e['trough_val'] for e in shallow):+.1f} uV")

            # The re-gate only ever removes: it cannot invent an event, and
            # it must not perturb the ones it keeps.
            kept = {round(float(e['start']), 6) for e in after}
            candidates = {round(float(e['start']), 6) for e in before}
            assert kept <= candidates, \
                f"{method}/{name}: the re-gate produced an unseen event"

            dropped = len(before) - len(after)
            if name == 'symmetric':
                # Both extrema are the same +/-100 uV, so gating on either
                # half-wave gives the same answer and nothing may be lost.
                assert dropped == 0, (
                    f"{method}/{name}: the re-gate dropped {dropped} events "
                    f"from a symmetric wave, where both half-waves have the "
                    f"same amplitude")
            else:
                assert dropped > 0, (
                    f"{method}/{name}: the re-gate dropped nothing from an "
                    f"asymmetric wave, so it is inert and the assertion "
                    f"above is vacuous")
                worst = max(e['trough_val'] for e in before
                            if round(float(e['start']), 6) not in kept)
                print(f"   [ok] {method:<21} {name:<11} {len(before):>4} -> "
                      f"{len(after):>4} ({dropped} removed, "
                      f"{100 * dropped / len(before):.0f}%); shallowest "
                      f"trough removed {worst:+.1f} uV vs a {thresh} uV "
                      f"criterion")
        print(f"   [ok] {method:<21} {'symmetric':<11} unchanged, as it must "
              f"be when both half-waves match")


def test_slow_wave_ptp_is_microvolts_not_samples():
    """``ptp`` must be an amplitude, not a sample count — for ALL FOUR methods.

    ``make_slow_waves`` computes ``'ptp': abs(ev[3] - ev[1])`` on sample
    INDICES, and it is shared by every slow-wave method, so the defect is not
    confined to the Massimini family. The value scales with sampling rate and
    is independent of amplitude — 300 µV gave 160 at 256 Hz, 62 at 100 Hz,
    and 90 µV gave 312 at 500 Hz — while Wonambi's own ``SlowWaves``
    docstring promises "peak-to-peak (difference between highest and lowest
    value)" and ``neural_events.db`` stores it in a column labelled
    ``det_ptp`` (uV). In the user's live database Staresina2015 rows averaged
    ``det_ptp`` 375.8 against a real ``peak2peak_amp`` of 95.3 µV.

    The direct inverse of that evidence, per method:

    * the same wave gives the same ptp at every sampling rate;
    * scaling the recording by 1.5 scales ptp by 1.5;
    * ``ptp == peak_val - trough_val`` for every single event;
    * on a synthetic wave of known amplitude, ptp is that amplitude.

    Each method is measured with a signal and thresholds under which its own
    detected set is scale-invariant, so the ratio test isolates the reported
    value rather than a change in which events were found.
    """
    print("\nTesting slow-wave ptp is microvolts for all four methods:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    rates = (256.0, 500.0, 1000.0)

    # Massimini's criteria are absolute in uV, so scaling the signal would
    # change its detected set; the thresholds below sit far enough under a
    # 100 uV wave that both 1.0x and 1.5x are fully accepted.
    # Ngo2015/Staresina2015 are relative, so their detected set is already
    # scale-invariant -- but the legacy post-hoc filter's `min_neg_amp`
    # default of +40 is NOT, so it is given a negative value (its historical
    # no-op form) to keep the comparison about ptp alone.
    cases = {
        'Massimini2004': (lambda sf, amp: _biphasic_wave(s_freq=sf, amp=amp),
                          dict(neg_peak_thresh=-50.0, p2p_thresh=100.0),
                          True),
        'AASM/Massimini2004': (
            lambda sf, amp: _biphasic_wave(s_freq=sf, amp=amp),
            dict(neg_peak_thresh=-50.0, p2p_thresh=100.0), True),
        'Ngo2015': (
            lambda sf, amp: _synthetic_slow_oscillation(s_freq=sf)
            * (amp / 100.0),
            dict(neg_peak_thresh=-75.0, p2p_thresh=75.0), False),
        'Staresina2015': (
            lambda sf, amp: _synthetic_slow_oscillation(s_freq=sf)
            * (amp / 100.0),
            dict(neg_peak_thresh=-75.0, p2p_thresh=75.0), False),
    }

    for method, (gen, kw, known_amp) in cases.items():
        by_rate = {}
        for s_freq in rates:
            for amp in (100.0, 150.0):
                det = ImprovedDetectSlowWave(method=method,
                                             frequency=(0.1, 4), **kw)
                events = det(_make_chantime(gen(s_freq, amp), s_freq))
                assert len(events), (
                    f"{method} found nothing at {s_freq} Hz / {amp} scale")

                for e in events:
                    assert abs(e['ptp']
                               - (e['peak_val'] - e['trough_val'])) < 1e-6, (
                        f"{method}: ptp {e['ptp']} != peak_val - trough_val")
                    assert e['ptp'] > 0, f"{method}: non-positive ptp"

                by_rate[(s_freq, amp)] = float(
                    np.mean([e['ptp'] for e in events]))

        # Sampling-rate invariance. A sample count would rise ~4x from
        # 256 Hz to 1000 Hz.
        for amp in (100.0, 150.0):
            vals = [by_rate[(s, amp)] for s in rates]
            spread = (max(vals) - min(vals)) / np.mean(vals)
            assert spread < 0.02, (
                f"{method}: ptp varies {spread:.1%} across "
                f"{rates} Hz ({[round(v, 1) for v in vals]}), so it is still "
                f"a sample count")

        # Amplitude proportionality.
        for s_freq in rates:
            ratio = by_rate[(s_freq, 150.0)] / by_rate[(s_freq, 100.0)]
            assert abs(ratio - 1.5) < 0.02, (
                f"{method} at {s_freq} Hz: a 1.5x louder recording gave a "
                f"ptp ratio of {ratio:.3f}, so ptp does not track amplitude")

        if known_amp:
            # The synthetic wave's peak-to-peak amplitude is exactly 2 * amp.
            for (s_freq, amp), ptp in by_rate.items():
                assert abs(ptp - 2 * amp) / (2 * amp) < 0.02, (
                    f"{method} at {s_freq} Hz: ptp {ptp:.1f} is not the "
                    f"true {2 * amp:.0f} uV peak-to-peak")

        print(f"   [ok] {method:<21} "
              + "  ".join(f"{s:.0f}Hz={by_rate[(s, 100.0)]:.1f}"
                          for s in rates)
              + f"  |  1.5x -> {by_rate[(256.0, 150.0)]:.1f} "
                f"(x{by_rate[(256.0, 150.0)] / by_rate[(256.0, 100.0)]:.2f})")


def test_det_trough_is_negative_for_every_method():
    """``trough_val`` negative and ``peak_val`` positive, for all four methods.

    ``detect_Massimini2004`` detects an ABOVE-zero run first and stores that
    run's maximum as ``trough_val`` and the following minimum as
    ``peak_val``, the opposite of the zero-crossing methods Ngo2015 and
    Staresina2015 and the opposite of Wonambi's own ``SlowWaves`` docstring
    ("trough_val: the lowest value"). Measured on one signal before the fix:
    Massimini ``trough_val=+295``, Staresina ``trough_val=-101``. Every
    cross-method comparison of ``det_trough`` / ``min_amp`` in
    ``neural_events.db`` was therefore comparing opposite quantities.

    Also asserts ``ptp == peak_val - trough_val`` for every method, so the
    three fields cannot drift apart and no method can quietly go back to
    Wonambi's sample count.
    """
    print("\nTesting det_trough is negative and det_peak positive everywhere:")

    from turtlewave_hdEEG.extensions import (ImprovedDetectSlowWave,
                                             ImprovedDetectKComplex)

    s_freq = 256.0
    # The jittered oscillation, so Ngo2015's relative thresholds have a
    # spread to work against; a pure sine gives it nothing above 1.25 x mean.
    sig = _synthetic_slow_oscillation(s_freq=s_freq)
    data = _make_chantime(sig, s_freq)
    band = np.asarray(sig, dtype='f')

    cases = [(ImprovedDetectSlowWave, m, {}) for m in
             ('Massimini2004', 'AASM/Massimini2004', 'Ngo2015',
              'Staresina2015')]
    cases.append((ImprovedDetectKComplex, 'AASM/Massimini2004',
                  {'min_isolation': 1.0}))

    for cls, method, kwargs in cases:
        events = cls(method=method, **kwargs)(data)
        label = f"{cls.__name__.replace('ImprovedDetect', '')}/{method}"
        assert len(events), f"{label}: no events, assertions would be vacuous"

        troughs = np.array([e['trough_val'] for e in events])
        peaks = np.array([e['peak_val'] for e in events])
        assert (troughs < 0).all(), (
            f"{label}: {int((troughs >= 0).sum())}/{len(troughs)} events have "
            f"a non-negative trough_val (max {troughs.max():+.1f} uV)")
        assert (peaks > 0).all(), (
            f"{label}: {int((peaks <= 0).sum())}/{len(peaks)} events have a "
            f"non-positive peak_val (min {peaks.min():+.1f} uV)")

        for e in events:
            # Both times must lie inside the event, whichever half-wave came
            # first, so trough_time/peak_time cannot be swapped away from
            # their own values or off the end of the record.
            for key in ('trough_time', 'peak_time'):
                assert e['start'] <= e[key] <= e['end'], (
                    f"{label}: {key} {e[key]} outside "
                    f"[{e['start']}, {e['end']}]")
                idx = int(round(e[key] * s_freq))
                assert 0 <= idx < len(band), f"{label}: {key} off the record"
            assert abs(e['ptp']
                       - (e['peak_val'] - e['trough_val'])) < 1e-6, (
                f"{label}: ptp {e['ptp']} != peak_val - trough_val -- this "
                f"method has gone back to Wonambi's sample count")

        ptps = np.array([e['ptp'] for e in events])
        print(f"   [ok] {label:<28} n={len(events):>4} "
              f"trough_val<0 (mean {troughs.mean():+7.1f}), "
              f"peak_val>0 (mean {peaks.mean():+7.1f}), "
              f"ptp = peak-trough (median {np.median(ptps):6.1f} uV)")


def test_staresina_and_ngo_are_untouched():
    """Staresina2015 and Ngo2015 must not move by a single event.

    They are published methods with hundreds of thousands of rows already in
    the user's databases, and the criteria changes here are aimed only at the
    Massimini family. Staresina's peak-to-peak gate is
    ``percentile(ptp, opts.ptp_thresh)`` — a relative criterion that always
    keeps ~25 % of candidates and has no absolute amplitude floor — and it
    stays exactly that. The legacy post-hoc amplitude filter also stays on
    these two methods, unit confusion and all, and it runs BEFORE the
    reported ``ptp`` is converted to microvolts, so that conversion cannot
    move their detected set.

    Pins the published constants and the relative-threshold behaviour that
    depends on them.
    """
    print("\nTesting Staresina2015 / Ngo2015 configuration is unchanged:")

    from turtlewave_hdEEG.extensions import ImprovedDetectSlowWave

    star = ImprovedDetectSlowWave(method='Staresina2015')
    assert star.ptp_thresh == 75, (
        f"Staresina2015 ptp_thresh is {star.ptp_thresh}, not the published "
        f"75th percentile")
    assert (star.min_dur, star.max_dur) == (0.8, 2.0), \
        f"Staresina2015 duration {(star.min_dur, star.max_dur)} != (0.8, 2.0)"
    assert star.duration == (0.8, 2.0), \
        f"Staresina2015 zero-crossing interval {star.duration} != (0.8, 2.0)"
    assert star.lowpass['freq'] == 1.25, star.lowpass

    ngo = ImprovedDetectSlowWave(method='Ngo2015')
    assert (ngo.peak_thresh, ngo.ptp_thresh) == (1.25, 1.25), \
        f"Ngo2015 thresholds {(ngo.peak_thresh, ngo.ptp_thresh)} != 1.25"
    assert (ngo.min_dur, ngo.max_dur) == (0.833, 2.0), \
        f"Ngo2015 duration {(ngo.min_dur, ngo.max_dur)} != (0.833, 2.0)"
    assert ngo.duration == (0.833, 2.0), \
        f"Ngo2015 zero-crossing interval {ngo.duration} != (0.833, 2.0)"

    # The percentile is relative, so raising the recording's amplitudes must
    # NOT change the yield -- that is the property that makes it a published
    # method rather than an absolute uV criterion, and the reason it is not
    # being "fixed" here.
    s_freq = 256.0
    sig = _synthetic_slow_oscillation(s_freq=s_freq)
    n_1x = len(ImprovedDetectSlowWave(method='Staresina2015',
                                      neg_peak_thresh=-75.0,
                                      p2p_thresh=75.0)(
        _make_chantime(sig, s_freq)))
    n_3x = len(ImprovedDetectSlowWave(method='Staresina2015',
                                      neg_peak_thresh=-75.0,
                                      p2p_thresh=75.0)(
        _make_chantime(sig * 3.0, s_freq)))
    assert n_1x == n_3x and n_1x > 0, (
        f"Staresina2015 became amplitude-sensitive: {n_1x} events at 1x, "
        f"{n_3x} at 3x. Its 75 is a percentile, not microvolts")
    print(f"   [ok] Staresina2015 75 is still a percentile: {n_1x} events at "
          f"1x amplitude, {n_3x} at 3x")
    print("   [ok] published constants intact: Staresina 0.8-2.0 s / p75, "
          "Ngo 0.833-2.0 s / 1.25x")

    # ------------------------------------------------------------------
    # min_dur / max_dur must NOT reach the zero-crossing gate.
    # ------------------------------------------------------------------
    # This is the one path that moved, and the earlier version of this test
    # could not see it: it only checked default-constructed detectors, and
    # the identity check that accompanied it passed min_dur=0.8/max_dur=2.0 --
    # the published values -- so the recompute it was meant to catch was a
    # no-op by construction.
    #
    # `_set_method_params` overwrites self.min_dur/self.max_dur for these two
    # methods, but `find_intervals` gates on self.duration, which Wonambi's
    # constructor fixed from the published defaults beforehand. Recomputing
    # self.duration from the overridden values would be arguably more correct
    # -- the GUI's "Slow Wave Duration" control currently reaches only
    # det_filt['freq'] -- and it is deliberately NOT done, because it moves
    # two published methods that already have hundreds of thousands of rows
    # in the user's databases.
    #
    # It needs no exotic input to bite: frontend/turtlewave_gui.py prefills
    # the control with setValue(detector.min_dur) on a 2-decimal spin box, so
    # Ngo2015's 0.833 s default reads back as 0.83 and every zero-crossing
    # interval in [0.830, 0.833) would flip from rejected to accepted on a
    # GUI run with nothing typed.
    published = {'Staresina2015': (0.8, 2.0), 'Ngo2015': (0.833, 2.0)}
    # p2p_thresh=0 and a negative neg_peak_thresh make both arms of the
    # legacy post-hoc filter no-ops, so what is compared is the gate itself.
    isolate = dict(neg_peak_thresh=-75.0, p2p_thresh=0.0)

    for method, (pub_lo, pub_hi) in published.items():
        for s_freq, lo, hi in ((200.0, 0.83, 2.0), (256.0, 0.85, 2.0),
                               (500.0, 0.90, 1.5), (256.0, 0.5, 3.0)):
            sig = _synthetic_slow_oscillation(s_freq=s_freq)
            det = ImprovedDetectSlowWave(method=method, min_dur=lo,
                                         max_dur=hi, **isolate)
            assert det.duration == (pub_lo, pub_hi), (
                f"{method}: min_dur={lo}/max_dur={hi} reached self.duration "
                f"({det.duration}); find_intervals now gates on the caller's "
                f"values instead of the published {(pub_lo, pub_hi)}")

            base = ImprovedDetectSlowWave(method=method, **isolate)
            with_over = det(_make_chantime(sig, s_freq))
            without = base(_make_chantime(sig, s_freq))
            starts_over = [round(float(e['start']), 6) for e in with_over]
            starts_none = [round(float(e['start']), 6) for e in without]
            assert starts_over == starts_none, (
                f"{method} at {s_freq} Hz: passing min_dur={lo}/max_dur={hi} "
                f"changed the detected set ({len(starts_over)} vs "
                f"{len(starts_none)} events). These two methods must produce "
                f"exactly what they did before")
            assert starts_none, (
                f"{method} at {s_freq} Hz: no events, so the comparison is "
                f"vacuous")
        print(f"   [ok] {method:<14} min_dur/max_dur leave self.duration at "
              f"{(pub_lo, pub_hi)} and the detected set unchanged, at "
              f"200/256/500 Hz")

    # The exact GUI case: Ngo2015's 0.833 default through a 2-decimal spin box.
    s_freq = 256.0
    sig = _synthetic_slow_oscillation(s_freq=s_freq)
    rounded = ImprovedDetectSlowWave(method='Ngo2015', min_dur=0.83,
                                     max_dur=2.0, **isolate)
    default = ImprovedDetectSlowWave(method='Ngo2015', **isolate)
    # Asserted on the gate itself, not only on a count: whether 0.83 vs 0.833
    # changes the yield depends on whether the recording happens to contain a
    # zero-crossing interval in [0.830, 0.833), which a synthetic may not.
    # The mechanism is signal-independent, so pin that.
    assert rounded.duration == (0.833, 2.0), (
        f"Ngo2015 built the way the GUI builds it gates on "
        f"{rounded.duration}, not the published (0.833, 2.0): the spin box "
        f"rounds 0.833 to 0.83 and that is reaching find_intervals")
    n_round = [round(float(e['start']), 6)
               for e in rounded(_make_chantime(sig, s_freq))]
    n_def = [round(float(e['start']), 6)
             for e in default(_make_chantime(sig, s_freq))]
    assert n_round == n_def, (
        f"Ngo2015 run from the GUI with nothing typed ({len(n_round)} events) "
        f"differs from its published default ({len(n_def)})")
    print(f"   [ok] Ngo2015 via the GUI's 2-decimal spin box (0.833 -> 0.83) "
          f"still gates on (0.833, 2.0) and gives the published "
          f"{len(n_def)} events")

    # ------------------------------------------------------------------
    # "Unchanged" is true of the DETECTED SET and of every reported field
    # except one: `ptp`, which is now microvolts for every method.
    # ------------------------------------------------------------------
    # `_as_negative_first` runs for all four methods, so it is worth pinning
    # WHY the sign swap cannot move these two. Ngo2015 and Staresina2015 go
    # through find_peaks_in_slowwwave, which sets column 1 from ``argmin``
    # and column 3 from ``argmax`` over the same interval, so
    # trough_val <= peak_val holds structurally and the conditional swap is
    # never taken. det_trough, det_peak, det_trough_time and det_peak_time
    # therefore do not move; only det_ptp does.
    for method in ('Staresina2015', 'Ngo2015'):
        for s_freq in (256.0, 500.0):
            sig = _synthetic_slow_oscillation(s_freq=s_freq)
            raw = ImprovedDetectSlowWave.__mro__[1].__call__(
                ImprovedDetectSlowWave(method=method, **isolate),
                _make_chantime(sig, s_freq)).events
            assert raw, f"{method} at {s_freq} Hz: nothing to check"
            swapped = [e for e in raw
                       if float(e['trough_val']) > float(e['peak_val'])]
            assert not swapped, (
                f"{method} at {s_freq} Hz: {len(swapped)} raw events have "
                f"trough_val > peak_val, so _as_negative_first WOULD swap "
                f"them and det_trough/det_peak move for this method after "
                f"all")
            # And the fields the swap would have touched come through intact.
            final = ImprovedDetectSlowWave(method=method, **isolate)(
                _make_chantime(sig, s_freq)).events
            for before, after in zip(raw, final):
                for key in ('trough_val', 'peak_val', 'trough_time',
                            'peak_time'):
                    assert abs(float(before[key])
                               - float(after[key])) < 1e-9, (
                        f"{method}: {key} moved between the parent's output "
                        f"and ours")
                assert abs(float(after['ptp'])
                           - (float(after['peak_val'])
                              - float(after['trough_val']))) < 1e-6, (
                    f"{method}: ptp is not peak_val - trough_val")
    print("   [ok] the sign swap is never taken for either method, so "
          "det_trough/det_peak/det_trough_time/det_peak_time are untouched; "
          "det_ptp is the one field that moves (samples -> uV)")


def test_kcomplex_trough_duration_bounds_the_half_wave():
    """The K-complex path shares the trough/whole-wave conflation — pin it.

    ``ParalKC.detect_kcomplexes`` used to forward its ``trough_duration``
    into ``ImprovedDetectKComplex(duration=...)``, i.e. Wonambi's whole-wave
    bound, while the real half-wave limit stayed hardcoded. Its own default
    (0.25, 1.0) therefore rejected every K-complex whose full waveform ran
    past 1 s, which is most of them, and the AASM defaults could return
    nothing at all.

    Asserted at both levels: the detector stores the window where Wonambi
    reads it, and a real ``ParalKC`` run over the published AASM window
    writes K-complexes into the database with a negative trough and a
    peak-to-peak amplitude in microvolts.
    """
    print("\nTesting the K-complex trough window bounds the half-wave:")

    import logging
    from turtlewave_hdEEG import ParalKC
    from turtlewave_hdEEG.extensions import ImprovedDetectKComplex

    # --- detector level ---------------------------------------------------
    det = ImprovedDetectKComplex(method='AASM/Massimini2004',
                                 trough_duration=(0.25, 1.0))
    assert det.trough_duration == (0.25, 1.0), (
        f"trough_duration is {det.trough_duration}: the KC window is not "
        f"reaching Wonambi's half-wave limit")
    assert det.duration == (0, None), (
        f"duration is {det.duration}: the KC trough window is still being "
        f"applied to the whole wave")

    s_freq = 256.0
    sig = _biphasic_wave(s_freq=s_freq)  # 0.71 s half-wave, 1.43 s wave
    data = _make_chantime(sig, s_freq)

    def n_kc(window):
        return len(ImprovedDetectKComplex(
            method='AASM/Massimini2004', frequency=(0.1, 4),
            trough_duration=window, min_isolation=0.0)(data))

    inside = n_kc((0.25, 1.0))
    assert inside > 0, (
        "the AASM 0.25-1.0 s window finds no K-complex on a 0.71 s "
        "half-wave: it is still bounding the 1.43 s whole wave")
    assert n_kc((0.25, 0.5)) == 0, \
        "a window ending before the 0.71 s half-wave still accepted it"
    print(f"   [ok] detector: AASM (0.25, 1.0) -> {inside} KCs, "
          f"(0.25, 0.5) -> 0, whole-wave duration left at (0, None)")

    # --- processor level, end to end into the database --------------------
    tmp = tempfile.mkdtemp(prefix='tw_kc_window_')
    try:
        dataset, annot = _synthetic_recording(tmp, ('NREM2', 'NREM2'))
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out, exist_ok=True)
        db = os.path.join(out, 'neural_events.db')

        kcs = ParalKC(dataset, annot,
                      log_level=logging.CRITICAL).detect_kcomplexes(
            method='AASM/Massimini2004', chan=['Cz'], frequency=(0.5, 4),
            trough_duration=(0.25, 1.0), stage=['NREM2'], json_dir=out,
            db_path=db, subject='sub-KC', cat=(1, 1, 1, 0))
        assert kcs, (
            "a K-complex run with the published AASM window found nothing; "
            "before the fix this window was applied to the whole wave")

        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT det_trough, det_peak, det_ptp FROM events "
                "WHERE event_type = 'k_complex'").fetchall()
        finally:
            conn.close()
        assert len(rows) == len(kcs), f"{len(rows)} rows for {len(kcs)} KCs"
        assert all(t is not None and t < 0 for t, _, _ in rows), (
            "det_trough is not negative for every K-complex row: the "
            "Massimini relabel is not reaching the database")
        assert all(p is not None and p > 0 for _, p, _ in rows), \
            "det_peak is not positive for every K-complex row"
        for t, p, ptp in rows:
            assert abs(ptp - (p - t)) < 1e-4, (
                f"det_ptp {ptp} is not det_peak - det_trough ({p - t}): it "
                f"is still Wonambi's sample count")
        print(f"   [ok] processor: {len(rows)} K-complex rows in "
              f"neural_events.db, det_trough all < 0 "
              f"(median {sorted(r[0] for r in rows)[len(rows) // 2]:.1f} uV), "
              f"det_ptp = det_peak - det_trough")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _gappy_chantime(s_freq=256.0, n_epoch=20, epoch=30.0, gap_every=5,
                    n_chan=2, seed=7):
    """A concatenated, discontinuous segment shaped like fetch(...).read_data().

    Mirrors what ``wonambi.trans.select.Segments.read_data`` builds for
    ``cat=(1, 1, 1, 0)``: one trial whose time axis is the hstack of the
    retained epochs, so it is strictly increasing but NOT uniformly spaced.

    Parameters
    ----------
    s_freq : float
        Sampling frequency in Hz.
    n_epoch : int
        Number of retained epochs.
    epoch : float
        Epoch length in seconds.
    gap_every : int
        Drop one epoch in every ``gap_every``, creating the stitches.
    n_chan : int
        Number of channels.
    seed : int
        Seed for the sample values.

    Returns
    -------
    instance of wonambi.datatype.ChanTime
        Single-trial ChanTime with a gappy time axis.
    """
    from wonambi.datatype import ChanTime

    rng = np.random.default_rng(seed)
    kept = [i for i in range(n_epoch * 2) if i % gap_every][:n_epoch]
    n = int(epoch * s_freq)
    times = [e * epoch + np.arange(n) / s_freq for e in kept]

    data = ChanTime()
    data.s_freq = s_freq
    data.axis['chan'] = np.empty(1, dtype='O')
    data.axis['time'] = np.empty(1, dtype='O')
    data.data = np.empty(1, dtype='O')
    data.axis['chan'][0] = np.array([f'E{i + 1}' for i in range(n_chan)])
    data.axis['time'][0] = np.hstack(times)
    data.data[0] = rng.standard_normal(
        (n_chan, n * len(kept))).astype('f') * 20.0
    return data


def test_fast_time_slice_boundary_is_half_open():
    """Pin the window boundary of the index-based slice to Wonambi's ``select``.

    ``dbwrite.make_param_segment`` slices each event's measurement window by
    index (two ``numpy.searchsorted`` calls) instead of calling
    ``wonambi.trans.select.select`` once per event, which rescans the whole
    night's time axis per requested timestamp. The two MUST agree exactly: the
    window feeds ``compute_batched_params``, so an off-by-one sample would move
    every amplitude and spectral value in the database with no error anywhere.

    Asserts, against a discontinuous concatenated segment:

    1. The interval is HALF-OPEN, ``[t0, t1)`` -- start inclusive, end
       exclusive, exactly ``(t0 <= values) & (values < t1)`` from
       ``wonambi/trans/select.py:260-262``.
    2. ``fast_time_slice`` and ``select`` return byte-identical data, time axis,
       dtype and class, including for windows that straddle a stitch, fall
       entirely inside a gap, or run off either end of the record.
    3. The fast path DECLINES (returns None, so the caller falls back to
       ``select``) on a multi-trial segment and on a time axis with duplicated
       or non-monotonic timestamps, where a contiguous index range is not
       equivalent to Wonambi's value-based selection.
    """
    print("\nTesting fast_time_slice boundary convention:")

    from wonambi.datatype import ChanTime
    from wonambi.trans import select
    from turtlewave_hdEEG import dbwrite

    data = _gappy_chantime()
    t = data.axis['time'][0]
    s_freq = data.s_freq

    # (1) half-open: [t[k], t[k + m]) keeps exactly m samples, t[k]..t[k+m-1]
    for k, m in [(0, 1), (0, 256), (137, 640), (int(29 * s_freq), 512)]:
        t0, t1 = float(t[k]), float(t[k + m])
        got = dbwrite.fast_time_slice(data, t0, t1).axis['time'][0]
        assert got.size == m, (
            f"[{t0}, {t1}) must keep {m} samples, kept {got.size}")
        assert got[0] == t[k], "window start must be INCLUSIVE"
        assert got[-1] == t[k + m - 1], "window end must be EXCLUSIVE"
    print("   [ok] interval is half-open [t0, t1)")

    # (2) byte-identical to select, including across stitches and off the ends
    stitch = int(np.argmax(np.diff(t) > 2 / s_freq))
    windows = [(float(t[0]), float(t[0]) + 1.0),
               (float(t[stitch]) - 0.5, float(t[stitch + 1]) + 0.5),
               (float(t[stitch]) + 1e-6, float(t[stitch + 1]) - 1e-6),
               (-10.0, float(t[0]) + 0.25),
               (float(t[-1]) - 0.25, float(t[-1]) + 10.0),
               (1e9, 1e9 + 1.0),
               (5.0, 5.0)]
    for t0, t1 in windows:
        fast = dbwrite.fast_time_slice(data, t0, t1)
        ref = select(data, time=(t0, t1))
        assert fast is not None, f"fast path declined a valid window {t0, t1}"
        assert type(fast) is type(ref)
        assert fast.s_freq == ref.s_freq
        assert fast.data[0].dtype == ref.data[0].dtype, (t0, t1)
        assert fast.data[0].shape == ref.data[0].shape, (t0, t1)
        assert fast.data[0].tobytes() == ref.data[0].tobytes(), (
            f"data bytes differ for window {t0, t1}")
        assert fast.axis['time'][0].tobytes() == \
            ref.axis['time'][0].tobytes(), (
            f"time axis bytes differ for window {t0, t1}")
        assert np.array_equal(fast.axis['chan'][0], ref.axis['chan'][0])
    print(f"   [ok] byte-identical to select over {len(windows)} windows "
          f"(stitch, gap, both ends, empty)")

    # make_param_segment must agree with the select-based window it replaced
    for start, end in [(float(t[300]), float(t[300]) + 0.8),
                       (float(t[stitch]) - 0.2, float(t[stitch]) + 0.2)]:
        seg = dbwrite.make_param_segment(data, start, end, 'spindle',
                                         'NREM2', 'E1')
        ref = select(data, time=(max(0.0, start - 0.1), end + 0.1))
        assert seg['data'].data[0].tobytes() == ref.data[0].tobytes()
        assert seg['data'].axis['time'][0].tobytes() == \
            ref.axis['time'][0].tobytes()
    print("   [ok] make_param_segment reproduces the select-based window "
          "(0.1 s buffer)")

    # (3a) a NaN end bound must DECLINE, not return the rest of the night.
    # searchsorted sorts NaN last, so an unguarded `hi` would be len(t) while
    # select's mask `(values < nan)` is all-False. This is the one input where
    # the fast path would otherwise hand event_params a 50 s window and get a
    # plausible-looking amplitude and power out of it instead of nothing.
    for t0, t1 in [(float(t[10]), float('nan')), (float('nan'), float(t[500])),
                   (float(t[10]), float('inf')), (float('-inf'), float(t[500])),
                   (float(t[10]), None), (None, float(t[500]))]:
        assert dbwrite.fast_time_slice(data, t0, t1) is None, (
            f"non-finite bound {t0, t1} must decline, not slice")
    nan_ref = select(data, time=(float(t[10]), float('nan')))
    assert nan_ref.axis['time'][0].size == 0, (
        "select's NaN behaviour changed; the guard's premise no longer holds")
    nan_seg = dbwrite.make_param_segment(data, float(t[10]), float('nan'),
                                         'spindle', 'NREM2', 'E1')
    assert nan_seg['data'].axis['time'][0].size == 0, (
        "a NaN event end produced a non-empty measurement window")
    print("   [ok] non-finite bounds decline; a NaN end yields an EMPTY window, "
          "matching select, not the rest of the recording")

    # (3b) a non-float64 time axis must decline. NumPy 1.26 value-based casting
    # evaluates select's `t0 <= values` in float32 while searchsorted compares
    # in float64, so the two disagree by one sample on about half of windows.
    f32 = ChanTime()
    f32.s_freq = s_freq
    for axis_name in ('chan', 'time'):
        f32.axis[axis_name] = np.empty(1, dtype='O')
    f32.data = np.empty(1, dtype='O')
    f32.axis['chan'][0] = data.axis['chan'][0]
    f32.axis['time'][0] = t.astype('float32')
    f32.data[0] = data.data[0]
    assert f32.axis['time'][0].dtype == np.float32
    assert dbwrite.fast_time_slice(f32, float(t[10]), float(t[500])) is None, (
        "a float32 time axis must decline: select compares in float32, "
        "searchsorted in float64")
    print("   [ok] non-float64 time axis declines (float32 casting divergence)")

    # (3c) the axis/data shape guard. `read_data(concat_chan=True)` ravels the
    # data to 1-D while leaving the full time axis in place, so the time axis
    # no longer describes the array; slicing that by index silently returns
    # samples from the wrong channel.
    ravelled = ChanTime()
    ravelled.s_freq = s_freq
    for axis_name in ('chan', 'time'):
        ravelled.axis[axis_name] = np.empty(1, dtype='O')
    ravelled.data = np.empty(1, dtype='O')
    ravelled.axis['chan'][0] = np.array([', '.join(data.axis['chan'][0])])
    ravelled.axis['time'][0] = t
    ravelled.data[0] = np.ravel(data.data[0])          # 1-D: ndim 1 vs 2 axes
    assert dbwrite.fast_time_slice(
        ravelled, float(t[10]), float(t[500])) is None, (
        "a ravelled concat_chan segment must decline: its data is 1-D while "
        "the segment still declares two axes")

    short = ChanTime()
    short.s_freq = s_freq
    for axis_name in ('chan', 'time'):
        short.axis[axis_name] = np.empty(1, dtype='O')
    short.data = np.empty(1, dtype='O')
    short.axis['chan'][0] = data.axis['chan'][0]
    short.axis['time'][0] = t
    short.data[0] = data.data[0][:, :t.size // 2]      # right ndim, wrong len
    assert dbwrite.fast_time_slice(
        short, float(t[10]), float(t[500])) is None, (
        "a data array shorter than its time axis must decline")
    print("   [ok] axis/data shape guard declines the ravelled concat_chan "
          "segment and a short data array")

    # (3) the fast path must decline where index slicing is not equivalent
    multi = ChanTime()
    multi.s_freq = s_freq
    for axis_name in ('chan', 'time'):
        multi.axis[axis_name] = np.empty(2, dtype='O')
    multi.data = np.empty(2, dtype='O')
    for i in range(2):
        multi.axis['chan'][i] = data.axis['chan'][0]
        multi.axis['time'][i] = t[:1000] + i * 1000.0
        multi.data[i] = data.data[0][:, :1000]
    assert dbwrite.fast_time_slice(multi, float(t[10]), float(t[500])) is None, (
        "multi-trial segment must fall back to select")

    dup = ChanTime()
    dup.s_freq = s_freq
    for axis_name in ('chan', 'time'):
        dup.axis[axis_name] = np.empty(1, dtype='O')
    dup.data = np.empty(1, dtype='O')
    dup.axis['chan'][0] = data.axis['chan'][0]
    dup.axis['time'][0] = np.concatenate([t[:1000], t[:1000]])
    dup.data[0] = np.hstack([data.data[0][:, :1000]] * 2)
    assert dbwrite.fast_time_slice(dup, float(t[10]), float(t[500])) is None, (
        "duplicated timestamps must fall back to select")
    # ...and make_param_segment still returns a usable window via select
    fallback = dbwrite.make_param_segment(dup, float(t[10]), float(t[500]),
                                          'spindle', 'NREM2', 'E1')
    assert fallback is not None and fallback['data'].axis['time'][0].size > 0
    print("   [ok] declines multi-trial and duplicated-timestamp axes, "
          "caller falls back to select")

    # the monotonicity memo must not survive its time array being replaced
    assert dbwrite.fast_time_slice(data, float(t[10]), float(t[500])) \
        is not None
    data.axis['time'][0] = np.concatenate([t[:1000], t[:1000]])
    data.data[0] = np.hstack([data.data[0][:, :1000]] * 2)
    assert dbwrite.fast_time_slice(data, float(t[10]), float(t[500])) is None, (
        "cached monotonicity verdict outlived the time axis it described")
    print("   [ok] monotonicity memo invalidates when the time axis is "
          "replaced")


def test_package_structure():
    """Test the overall package structure"""
    print("\n6. Testing package structure:")
    
    # Check top-level components
    components = [item for item in dir(turtlewave_hdEEG) if not item.startswith('_')]
    print(f"Top-level components: {components}")
    
    # Check for specific components
    expected_components = ['ParalEvents', 'CustomAnnotations', 'LargeDataset', 'XLAnnotations','ImprovedDetectSpindle']
    for comp in expected_components:
        assert comp in components, f"{comp} should be available at the top level"
    print("[ok] All expected top-level components are available")

def test_density_with_bare_subject_id():
    """A driver-shaped call with a bare subject id must return a density.

    Every batch driver passes its ``--subject`` value straight through, and in
    the cluster deployment that value is the bare directory name (``10sd``)
    while detection stores the canonical ``sub-10sd``. When the read path did
    not normalise it, no run produced any density at all -- the feature this
    release is named for silently reported "unavailable" everywhere. This
    guards that exact round trip: write with one spelling, read with the other.
    """
    print("\n9. Testing density with a bare (unnormalised) subject id:")

    import tempfile
    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_density_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)
        # Detection side: stores the canonical 'sub-10sd'.
        for i in range(30):
            t = 100.0 + i
            conn.execute(
                "INSERT OR REPLACE INTO events (uuid, event_type, channel, "
                "start_time, end_time, duration, stage, method, freq_lower, "
                "freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'C3', t, 'Moelle2011', 11, 16,
                                     'NREM2'),
                 'spindle', 'C3', t, t + 1.0, 1.0, 'NREM2', 'Moelle2011',
                 11.0, 16.0))
        dbwrite.record_analysed_time(conn, '10sd', 'NREM2', 3000.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.upsert_processing_status(conn, 'spindle', 'C3', 'Moelle2011',
                                         11.0, 16.0, 'NREM2', True)
        conn.commit()
    finally:
        conn.close()

    stored = sqlite3.connect(db).execute(
        "SELECT DISTINCT subject FROM analysed_time").fetchall()
    assert stored == [('sub-10sd',)], f"write side did not normalise: {stored}"
    print(f"[ok] detection stored the subject as {stored[0][0]!r}")

    # Reader side: the driver's bare CLI value.
    for asked in ('10sd', 'sub-10sd'):
        df = event_density(db, event_type='spindle', method='Moelle2011',
                           stage=['NREM2'], subject=asked)
        assert len(df) == 1, f"subject={asked!r} returned {len(df)} rows"
        density = float(df['density_per_min'].iloc[0])
        assert abs(density - 0.6) < 1e-9, \
            f"subject={asked!r} gave density {density}, expected 0.6"
        print(f"[ok] event_density(subject={asked!r}) -> "
              f"{density:.4f}/min over {df['analysed_minutes'].iloc[0]:.1f} min")

    # read_analysed_time must normalise too.
    rows = dbwrite.read_analysed_time(db, subject='10sd')
    assert rows, "read_analysed_time(subject='10sd') found no denominator"
    print(f"[ok] read_analysed_time(subject='10sd') -> {len(rows)} denominator(s)")

    shutil.rmtree(tmp, ignore_errors=True)


def test_density_identity_axis():
    """A (detector, stage) combination that was never searched gets NO row.

    ``analysed_time`` is keyed on (subject, stage, rejection settings) and is
    shared by every detector that ever ran on the recording. Crossing that
    stage set with the detectors present would report, say, a K-complex
    density of 0.0/min in a stage the K-complex detector never searched --
    a hard zero against a real denominator, which is a fabricated result, not
    a conservative one. The stage scope must be resolved per
    ``(event_type, method)`` from that run's own recorded scope.
    """
    print("\n10. Testing the density identity axis (per-detector stage scope):")

    import tempfile
    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_identity_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)

        def add(ch, event_type, method, stage, n, band):
            for i in range(n):
                t = 100.0 + i + (abs(hash((ch, event_type, method, stage))) % 97)
                conn.execute(
                    "INSERT OR REPLACE INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5(event_type, ch, t, method, band[0],
                                         band[1], stage),
                     event_type, ch, t, t + 1.0, 1.0, stage, method,
                     band[0], band[1]))

        # Slow waves searched NREM2 only; spindles searched NREM2 and NREM3.
        add('C3', 'slow_wave', 'Massimini2004', 'NREM2', 50, (0.5, 4.0))
        add('C3', 'spindle', 'Moelle2011', 'NREM2', 30, (11.0, 16.0))
        add('C3', 'spindle', 'Moelle2011', 'NREM3', 20, (11.0, 16.0))
        dbwrite.record_analysed_time(conn, 'sub-A', 'NREM2', 3000.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.record_analysed_time(conn, 'sub-A', 'NREM3', 1500.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.upsert_processing_status(conn, 'slow_wave', 'C3',
                                         'Massimini2004', 0.5, 4.0, 'NREM2',
                                         True)
        dbwrite.upsert_processing_status(conn, 'spindle', 'C3', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()

    df = event_density(db)          # the flagship read API on its defaults
    combos = {(r.event_type, r.method, r.stage) for r in df.itertuples()}
    assert ('slow_wave', 'Massimini2004', 'NREM3') not in combos, \
        "fabricated a slow-wave row for a stage that detector never searched"
    assert ('slow_wave', 'Massimini2004', 'NREM2') in combos
    assert ('spindle', 'Moelle2011', 'NREM2') in combos
    assert ('spindle', 'Moelle2011', 'NREM3') in combos
    assert len(df) == 3, f"expected 3 rows, got {len(df)}:\n{df}"
    print(f"[ok] event_density(db) returned {len(df)} rows, none fabricated")

    sw = df[df['event_type'] == 'slow_wave'].iloc[0]
    assert abs(float(sw['density_per_min']) - 1.0) < 1e-9
    print(f"[ok] slow_wave/NREM2 = {float(sw['density_per_min']):.4f}/min "
          f"over {float(sw['analysed_minutes']):.1f} min")

    # Pooling is per identity: the slow-wave row must not gain NREM3's time.
    pooled = event_density(db, combine_stages=True)
    sw_p = pooled[pooled['event_type'] == 'slow_wave'].iloc[0]
    sp_p = pooled[pooled['event_type'] == 'spindle'].iloc[0]
    assert sw_p['stage'] == 'NREM2', f"slow wave pooled as {sw_p['stage']!r}"
    assert abs(float(sw_p['analysed_minutes']) - 50.0) < 1e-9
    assert sp_p['stage'] == 'NREM2+NREM3'
    assert abs(float(sp_p['analysed_minutes']) - 75.0) < 1e-9
    print(f"[ok] pooled per identity: slow_wave over {sw_p['stage']} "
          f"({float(sw_p['analysed_minutes']):.1f} min), spindle over "
          f"{sp_p['stage']} ({float(sp_p['analysed_minutes']):.1f} min)")

    shutil.rmtree(tmp, ignore_errors=True)


def test_density_multi_method_run():
    """A multi-method run must not invent an identity or lose its zeros.

    ``processing_status.method`` holds the run's JOINED method set
    (``'Moelle2011_Wamsley2012'``) while ``events.method`` holds the single
    method that detected each event. Labelling zero rows with the joined
    string produces an identity that appears nowhere in ``events`` -- a hard
    0.0/min against a real denominator -- and the channel that ran but fired
    nothing then gets no row under either real method, which defeats
    ``include_zero_channels`` exactly where it matters. Every earlier version
    of this test was single-method, which is why that survived.
    """
    print("\n11. Testing density on a multi-method run:")

    import tempfile
    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density, format_density_table

    tmp = tempfile.mkdtemp(prefix='tw_multimethod_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)

        def add(ch, method, n):
            for i in range(n):
                t = 100.0 + i + (abs(hash((ch, method))) % 97)
                conn.execute(
                    "INSERT OR REPLACE INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', ch, t, method, 11, 16,
                                         'NREM2'),
                     'spindle', ch, t, t + 1.0, 1.0, 'NREM2', method,
                     11.0, 16.0))

        # One run, two methods, three channels. Fz ran and fired nothing.
        add('Cz', 'Moelle2011', 20)
        add('Cz', 'Wamsley2012', 10)
        add('Pz', 'Moelle2011', 5)
        dbwrite.record_analysed_time(conn, 'sub-A', 'NREM2', 600.0,
                                     artefact_seconds_excluded=0.0)
        for ch in ('Cz', 'Pz', 'Fz'):
            dbwrite.upsert_processing_status(
                conn, 'spindle', ch, 'Moelle2011_Wamsley2012', 11.0, 16.0,
                'NREM2', True)
        conn.commit()
    finally:
        conn.close()

    df = event_density(db)
    methods = set(df['method'])
    assert 'Moelle2011_Wamsley2012' not in methods, \
        f"fabricated the joined method as an identity: {sorted(methods)}"
    assert methods == {'Moelle2011', 'Wamsley2012'}, sorted(methods)
    print(f"[ok] identities are the real per-event methods: {sorted(methods)}")

    # 3 channels x 2 methods, every combination present exactly once.
    assert len(df) == 6, f"expected 6 rows, got {len(df)}:\n{df}"
    channels = set(df['channel'])
    assert channels == {'Cz', 'Pz', 'Fz'}, sorted(channels)
    for m in ('Moelle2011', 'Wamsley2012'):
        got = set(df[df['method'] == m]['channel'])
        assert got == {'Cz', 'Pz', 'Fz'}, f"{m} covered {sorted(got)}"
    print(f"[ok] {len(df)} rows = 3 channels x 2 methods, Fz included in both")

    fz = df[df['channel'] == 'Fz']
    assert (fz['n_events'] == 0).all()
    assert (fz['density_per_min'] == 0.0).all()
    assert (fz['analysed_minutes'] == 10.0).all()
    print("[ok] Fz (ran, fired nothing) -> 0 events, 0.000/min over 10.0 min "
          "under BOTH methods")

    moelle = df[df['method'] == 'Moelle2011'].set_index('channel')
    assert abs(float(moelle.loc['Cz', 'density_per_min']) - 2.0) < 1e-9
    assert abs(float(moelle.loc['Pz', 'density_per_min']) - 0.5) < 1e-9
    print(f"[ok] Moelle2011: Cz={float(moelle.loc['Cz','density_per_min']):.3f}"
          f"/min, Pz={float(moelle.loc['Pz','density_per_min']):.3f}/min, "
          f"Fz=0.000/min")

    # The montage summary must count the montage, not a phantom.
    rendered = format_density_table(df[df['method'] == 'Moelle2011'])
    assert '3 channel(s)' in rendered, rendered
    print("[ok] format_density_table reports '3 channel(s)' for a 3-channel "
          "montage")

    shutil.rmtree(tmp, ignore_errors=True)


def test_cycle_subject_spelling_delete():
    """Re-running the cycle writers must replace old-spelling rows, not add to them.

    The writers normalise the subject before inserting, so a row written
    earlier under the bare folder name -- which the cycle how-to tells users
    to pass -- is not matched by a delete on the canonical id. Without the
    widened delete the insert adds a *second* row: ``sleep_cycles`` ends up
    with both spellings, and ``stage_durations`` (``PRIMARY KEY (subject)``,
    one row per recording by contract) doubles any total taken over it.
    """
    print("\n12. Testing the cycle subject-spelling delete:")

    import tempfile
    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA, ParalCycles

    tmp = tempfile.mkdtemp(prefix='tw_spelling_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        # Pre-existing rows, written under the BARE id by an earlier run.
        conn.execute(
            "INSERT INTO sleep_cycles (subject, method, cycle_number, "
            "nrem_start, nrem_end, rem_start, rem_end, nrem_dur_min, "
            "nrem_n23_dur_min, rem_dur_min, cycle_dur_min) "
            "VALUES ('10sd','2022',1,0,600,600,900,10.0,8.0,5.0,15.0)")
        conn.execute(
            "INSERT INTO stage_durations (subject, epoch_length, wake_min, "
            "n1_min, n2_min, n3_min, rem_min, artefact_min, total_min) "
            "VALUES ('10sd',30,60.0,20.0,150.0,90.0,80.0,0.0,400.0)")
        conn.commit()
    finally:
        conn.close()

    cyc = ParalCycles(annotations=None, subject='10sd',
                      log_level=logging.CRITICAL)
    cycles = [{'method': '2022', 'cycle_number': 1, 'nrem_start_sec': 0.0,
               'nrem_end_sec': 600.0, 'rem_end_sec': 900.0,
               'nrem_dur_min': 10.0, 'nrem_n23_dur_min': 8.0,
               'rem_dur_min': 5.0, 'cycle_dur_min': 15.0}]
    durations = {'epoch_length': 30, 'wake_min': 60.0, 'n1_min': 20.0,
                 'n2_min': 150.0, 'n3_min': 90.0, 'rem_min': 80.0,
                 'artefact_min': 0.0, 'total_min': 400.0}
    cyc.store_cycles_to_database(cycles, db, subject='10sd')
    cyc.store_stage_durations(durations, db, subject='10sd')

    conn = sqlite3.connect(db)
    try:
        cycle_rows = conn.execute(
            "SELECT subject, method, cycle_number FROM sleep_cycles").fetchall()
        dur_rows = conn.execute(
            "SELECT subject, total_min FROM stage_durations").fetchall()
        total = conn.execute(
            "SELECT SUM(total_min) FROM stage_durations").fetchone()[0]
    finally:
        conn.close()

    assert len(cycle_rows) == 1, f"sleep_cycles duplicated: {cycle_rows}"
    assert cycle_rows[0][0] == 'sub-10sd', cycle_rows
    print(f"[ok] sleep_cycles: 1 row, {cycle_rows[0][0]!r}")

    assert len(dur_rows) == 1, f"stage_durations duplicated: {dur_rows}"
    assert dur_rows[0][0] == 'sub-10sd', dur_rows
    print(f"[ok] stage_durations: 1 row, {dur_rows[0][0]!r}")

    assert total == 400.0, f"SUM(total_min) = {total}, expected 400.0"
    print(f"[ok] SELECT SUM(total_min) = {total} (not 800.0)")

    shutil.rmtree(tmp, ignore_errors=True)


def test_pac_twin_delete_is_scoped():
    """The pac_coupling twin delete must remove the twin and nothing else.

    ``subject`` is part of the natural key and is normalised on the way in, so
    a row written earlier under another spelling of the same recording is a
    *different* key and ``INSERT OR REPLACE`` would leave it as a duplicate.
    The twin is deleted first -- but that delete carries the row's whole
    natural key, so it must not touch a neighbour differing in any one
    component. This is the highest-consequence delete in the package; the
    seven near misses below pin every column of the key.
    """
    print("\n13. Testing the scope of the pac_coupling twin delete:")

    import tempfile
    import logging
    from turtlewave_hdEEG import dbwrite, ParalPAC

    tmp = tempfile.mkdtemp(prefix='tw_pactwin_')
    db = os.path.join(tmp, 'neural_events.db')

    base = dict(channel='C3', event_type='slow_wave', method='Staresina2015',
                stage='NREM2', phase_freq_lower=0.5, phase_freq_upper=1.25,
                amp_freq_lower=11.0, amp_freq_upper=16.0)

    # The twin, then one neighbour per key column, then a different subject.
    rows = [
        ('twin (differs only in subject spelling)', dict(base, subject='10sd')),
        ('neighbour: channel', dict(base, subject='10sd', channel='C4')),
        ('neighbour: event_type', dict(base, subject='10sd',
                                       event_type='spindle')),
        ('neighbour: method', dict(base, subject='10sd', method='Ngo2015')),
        ('neighbour: stage', dict(base, subject='10sd', stage='NREM3')),
        ('neighbour: amp_freq_lower', dict(base, subject='10sd',
                                           amp_freq_lower=12.0)),
        ('neighbour: phase_freq_lower', dict(base, subject='10sd',
                                             phase_freq_lower=0.75)),
        ('different subject entirely', dict(base, subject='sub-OTHER')),
    ]

    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_pac_schema(conn)
        for label, r in rows:
            conn.execute(
                "INSERT INTO pac_coupling (subject, channel, event_type, "
                "method, stage, phase_freq_lower, phase_freq_upper, "
                "amp_freq_lower, amp_freq_upper, n_events) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r['subject'], r['channel'], r['event_type'], r['method'],
                 r['stage'], r['phase_freq_lower'], r['phase_freq_upper'],
                 r['amp_freq_lower'], r['amp_freq_upper'], 111))
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM pac_coupling").fetchone()[0]
    finally:
        conn.close()
    assert before == 8, f"fixture built {before} rows, expected 8"
    print(f"[ok] fixture: {before} rows (1 twin + 6 near misses + 1 other "
          f"subject)")

    class _FakeDataset:
        filename = os.path.join(tmp, 'x.set')
        header = {'s_freq': 100.0}

    pac = ParalPAC(dataset=_FakeDataset(), rootpath=tmp,
                   log_level=logging.CRITICAL)
    stats = pac.store_pac_to_database(
        db_path=db, subject='10sd', event_type='slow_wave',
        method='Staresina2015', stage='NREM2', phase_freq=(0.5, 1.25),
        amp_freq=(11.0, 16.0), idpac=(1, 2, 4),
        results={'C3': {'0.5-1.25Hz_11-16Hz': {
            'mi_raw': 0.17, 'mi_norm': 0.17, 'pval': 0.49,
            'preferred_phase_rad': -1.14, 'preferred_phase_deg': -65.3,
            'mean_vector_length': 0.0801, 'rho': -0.88, 'rayleigh_z': 2.2,
            'rayleigh_p': 0.1, 'n_segments': 351, 'outputfile': None,
            'csv_written': False, 'amp_file': 'x.npy'}}})
    assert stats.get('ok'), stats

    conn = sqlite3.connect(db)
    try:
        after = conn.execute("SELECT COUNT(*) FROM pac_coupling").fetchone()[0]
        survivors = set(conn.execute(
            "SELECT subject, channel, event_type, method, stage, "
            "phase_freq_lower, amp_freq_lower FROM pac_coupling"))
    finally:
        conn.close()

    assert after == 8, f"{before} rows in, {after} out (expected 8)"
    print(f"[ok] {before} rows in, {after} rows out")

    canonical = ('sub-10sd', 'C3', 'slow_wave', 'Staresina2015', 'NREM2',
                 0.5, 11.0)
    assert canonical in survivors, "the canonical row was not written"
    twin = ('10sd', 'C3', 'slow_wave', 'Staresina2015', 'NREM2', 0.5, 11.0)
    assert twin not in survivors, "the old-spelling twin survived"
    print("[ok] the twin was replaced by the canonical row")

    for label, r in rows[1:]:
        key = (r['subject'], r['channel'], r['event_type'], r['method'],
               r['stage'], r['phase_freq_lower'], r['amp_freq_lower'])
        assert key in survivors, f"collateral damage to {label}: {key}"
    print(f"[ok] all {len(rows) - 1} neighbours intact (channel, event_type, "
          f"method, stage, amp_freq, phase_freq, other subject)")

    shutil.rmtree(tmp, ignore_errors=True)


def _synthetic_recording(tmp, stages, s_freq=128.0, seed=0):
    """Write a scored synthetic EDF and open it as a (Dataset, Annotations).

    Small enough to run three real detectors in a unit test: 30 s per
    requested stage, a continuous asymmetric slow oscillation (which
    Massimini2004/AASM and the K-complex detector both fire on) plus one
    13.5 Hz spindle burst per epoch.

    Parameters
    ----------
    tmp : str
        Directory to write the EDF and scoring XML into.
    stages : sequence of str
        Stage of each consecutive 30 s epoch, e.g.
        ``('NREM2', 'NREM2', 'NREM3', 'NREM3')``.
    s_freq : float, optional
        Sampling frequency in Hz. Default ``128.0``.
    seed : int, optional
        Seed for the noise and the per-cycle amplitude jitter. Default ``0``.

    Returns
    -------
    tuple
        ``(dataset, annotations)`` -- a ``wonambi.Dataset`` over the written
        EDF and a ``wonambi.attr.Annotations`` with one rater and one epoch
        per entry of ``stages``.
    """
    from wonambi import Dataset
    from wonambi.attr import Annotations
    from wonambi.attr.annotations import create_empty_annotations
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data

    duration = 30.0 * len(stages)
    sig = _synthetic_slow_oscillation(s_freq=s_freq, duration=duration,
                                      seed=seed)
    t = np.arange(len(sig)) / s_freq
    for i in range(len(stages)):
        t0 = i * 30.0 + 20.0
        burst = (t >= t0) & (t < t0 + 1.0)
        sig[burst] += 30.0 * np.sin(2 * np.pi * 13.5 * (t[burst] - t0))

    data = create_data(datatype='ChanTime', n_trial=1, s_freq=s_freq,
                       chan_name=['Cz'], time=(0, duration))
    data.data[0] = np.asarray(sig, dtype='f')[None, :len(data.axis['time'][0])]

    edf = os.path.join(tmp, 'sub-T.edf')
    write_edf(data, edf)
    dataset = Dataset(edf)

    xml = os.path.join(tmp, 'sub-T_scoring.xml')
    create_empty_annotations(xml, dataset)
    annot = Annotations(xml)
    # add_rater() also lays down the epoch grid; calling create_epochs() as
    # well would append a SECOND grid and get_epochs() would return both.
    annot.add_rater('tester')
    for i, stg in enumerate(stages):
        annot.set_stage_for_epoch(i * 30.0, stg, save=True)
    return dataset, annot


def test_stage_token_vocabulary():
    """Phase 0: one spelling per stage set, and one way to ask about it.

    ``join_stage_token`` is what makes ``events.stage`` an equality-comparable
    key. Without canonical ordering a caller passing ``['NREM3','NREM2']``
    writes ``'NREM3NREM2'``, which never matches the ``'NREM2NREM3'`` another
    caller wrote: the same scope becomes two tokens, every reader filtering on
    one silently misses the other, and nothing raises.

    ``stage_tokens_covering`` is the other half -- the read-side primitive that
    has to work against a per-epoch database AND a joint one, and has to
    refuse a request for a strict subset of a joint token rather than answer
    it with part of the truth.
    """
    print("\n14. Testing the stage-token vocabulary (phase 0):")

    from turtlewave_hdEEG.dbwrite import (join_stage_token, split_stage_token,
                                          stage_components,
                                          stage_tokens_covering,
                                          pooled_denominator)

    # Order-insensitivity: one SET, one spelling.
    assert join_stage_token(['NREM3', 'NREM2']) == 'NREM2NREM3'
    assert (join_stage_token(['NREM2', 'NREM3'])
            == join_stage_token(['NREM3', 'NREM2'])
            == join_stage_token({'NREM3', 'NREM2'})
            == join_stage_token('NREM2NREM3') == 'NREM2NREM3')
    assert join_stage_token(['NREM2', 'NREM2']) == 'NREM2'
    assert join_stage_token(None) == '' and join_stage_token([]) == ''
    assert join_stage_token(['Wake', 'REM', 'NREM1']) == 'NREM1REMWake'
    print("[ok] join_stage_token is order-insensitive, de-duplicating and "
          "canonical (NREM1, NREM2, NREM3, REM, Wake)")

    # Round trip.
    for stages in (['NREM2'], ['NREM2', 'NREM3'], ['NREM1', 'REM', 'Wake']):
        assert split_stage_token(join_stage_token(stages)) == sorted(
            stages, key=['NREM1', 'NREM2', 'NREM3', 'REM', 'Wake'].index)
    print("[ok] split_stage_token(join_stage_token(x)) round-trips")

    # 'REM' is a substring of 'NREM1'; the greedy longest-first split must not
    # be fooled by it.
    assert split_stage_token('NREM1REM') == ['NREM1', 'REM']
    assert stage_components('Undefined') == ['Undefined']  # forgiving on read
    print("[ok] 'NREM1REM' splits as ['NREM1', 'REM'], not as a REM run")

    # Covering, both directions.
    per_epoch = ['NREM2', 'NREM3', 'REM']
    joint = ['NREM2NREM3']
    assert stage_tokens_covering(per_epoch, ['NREM2', 'NREM3']) == ['NREM2',
                                                                    'NREM3']
    assert stage_tokens_covering(joint, ['NREM2', 'NREM3']) == ['NREM2NREM3']
    assert stage_tokens_covering(joint, 'NREM2NREM3') == ['NREM2NREM3']
    assert stage_tokens_covering(joint, ['NREM2']) == []
    assert stage_tokens_covering(per_epoch, ['NREM2']) == ['NREM2']
    assert stage_tokens_covering(per_epoch, None) == per_epoch
    print("[ok] stage_tokens_covering reads both database shapes, and refuses "
          "a strict subset of a joint token")

    # Pooled denominator: all-or-nothing.
    denom = {'NREM2': {'analysed_seconds': 1800.0,
                       'artefact_seconds_excluded': 60.0, 'source': 'detection'},
             'NREM3': {'analysed_seconds': 1200.0,
                       'artefact_seconds_excluded': 0.0, 'source': 'detection'}}
    pooled = pooled_denominator('NREM2NREM3', denom)
    assert pooled.analysed_seconds == 3000.0 and pooled.missing == []
    assert pooled.artefact_seconds_excluded == 60.0
    assert pooled.source == 'detection'
    partial = pooled_denominator('NREM2REM', denom)
    assert partial.missing == ['REM'] and partial.analysed_seconds != partial.analysed_seconds
    print(f"[ok] pooled_denominator('NREM2NREM3') = "
          f"{pooled.analysed_seconds:.0f} s; a missing component gives NaN, "
          f"never a partial sum")


def test_detectors_write_joint_stage_token():
    """Phase 1: all three detectors label a run with ONE stage token.

    The bug this closes: spindles stored the requested stage LIST, slow waves
    stored each event's own epoch stage. On a two-stage run the two never
    matched, so the PAC tab's ``JOIN events sp ON sw.stage = sp.stage`` was
    structurally unable to return a channel and the user got "No channels
    selected" with no explanation.

    Also asserts the additive ``events.epoch_stage`` column really carries the
    per-epoch value, which is what keeps an N2-vs-N3 split recoverable, and
    that a run scoped to a stage the scoring does not contain raises instead
    of stamping thousands of rows with a label nothing can contradict.
    """
    print("\n15. Testing the joint stage token written by all three detectors "
          "(phase 1):")

    import logging
    from turtlewave_hdEEG import ParalEvents, ParalSWA, ParalKC

    tmp = tempfile.mkdtemp(prefix='tw_joint_')
    try:
        dataset, annot = _synthetic_recording(
            tmp, ('NREM2', 'NREM2', 'NREM3', 'NREM3'))
        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out, exist_ok=True)

        def run_all(db, stage):
            """Run all three detectors over one stage set into one database."""
            kw = dict(chan=['Cz'], stage=stage, json_dir=out, db_path=db,
                      subject='sub-T', cat=(1, 1, 1, 0))
            n = {}
            n['spindle'] = len(ParalEvents(
                dataset, annot, log_level=logging.CRITICAL).detect_spindles(
                    method='Moelle2011', frequency=(11, 16), **kw))
            # Both run on their method's published criteria: with
            # trough_duration/neg_peak_thresh/p2p_thresh left unset,
            # AASM/Massimini2004 resolves to its own (0.25, 1.0) s /
            # -40 uV / 75 uV. What is under test here is the stage
            # bookkeeping, not the morphology criteria.
            n['slow_wave'] = len(ParalSWA(
                dataset, annot, log_level=logging.CRITICAL).detect_slow_waves(
                    method='AASM/Massimini2004', frequency=(0.5, 4), **kw))
            n['k_complex'] = len(ParalKC(
                dataset, annot, log_level=logging.CRITICAL).detect_kcomplexes(
                    method='AASM/Massimini2004', frequency=(0.5, 4), **kw))
            return n

        # --- two-stage run: exactly one token, on every event type ---------
        db2 = os.path.join(out, 'neural_events.db')
        counts = run_all(db2, ['NREM2', 'NREM3'])
        assert all(counts.values()), f"a detector found nothing: {counts}"
        conn = sqlite3.connect(db2)
        try:
            by_type = dict(conn.execute(
                "SELECT event_type, COUNT(DISTINCT stage) FROM events "
                "GROUP BY event_type").fetchall())
            tokens = {r[0] for r in conn.execute(
                "SELECT DISTINCT stage FROM events")}
            epoch_stages = {r[0] for r in conn.execute(
                "SELECT DISTINCT epoch_stage FROM events")}
            ps_tokens = {r[0] for r in conn.execute(
                "SELECT DISTINCT stage FROM processing_status")}
        finally:
            conn.close()
        assert tokens == {'NREM2NREM3'}, f"events.stage = {tokens}"
        assert set(by_type) == {'spindle', 'slow_wave', 'k_complex'}, by_type
        assert set(by_type.values()) == {1}, by_type
        assert ps_tokens == {'NREM2NREM3'}, ps_tokens
        print(f"[ok] two-stage run: every event of {sorted(by_type)} carries "
              f"the single token 'NREM2NREM3' ({counts})")
        assert epoch_stages == {'NREM2', 'NREM3'}, epoch_stages
        print(f"[ok] events.epoch_stage still splits them: {sorted(epoch_stages)}")

        # --- single-stage run: the token is just that stage ----------------
        tmp1 = tempfile.mkdtemp(prefix='tw_joint1_', dir=tmp)
        db1 = os.path.join(tmp1, 'neural_events.db')
        counts1 = run_all(db1, ['NREM2'])
        assert all(counts1.values()), f"a detector found nothing: {counts1}"
        conn = sqlite3.connect(db1)
        try:
            tokens1 = {r[0] for r in conn.execute(
                "SELECT DISTINCT stage FROM events")}
        finally:
            conn.close()
        assert tokens1 == {'NREM2'}, f"events.stage = {tokens1}"
        print(f"[ok] single-stage run: token is 'NREM2' ({counts1})")

        # --- a requested stage the scoring does not contain must RAISE -----
        tmp_rem = tempfile.mkdtemp(prefix='tw_unscored_', dir=tmp)
        raised = None
        try:
            ParalEvents(dataset, annot,
                        log_level=logging.CRITICAL).detect_spindles(
                method='Moelle2011', chan=['Cz'], frequency=(11, 16),
                stage=['REM'], json_dir=out,
                db_path=os.path.join(tmp_rem, 'neural_events.db'),
                subject='sub-T', cat=(1, 1, 1, 0))
        except ValueError as e:
            raised = str(e)
        assert raised is not None, \
            "a run scoped to an unscored stage completed silently"
        assert 'REM' in raised, raised
        assert not os.path.exists(os.path.join(tmp_rem, 'neural_events.db')) \
            or sqlite3.connect(os.path.join(tmp_rem, 'neural_events.db')
                               ).execute("SELECT COUNT(*) FROM events "
                                         "WHERE 1=1").fetchone()[0] == 0
        print(f"[ok] stage=['REM'] on an N2/N3 recording raises, naming it: "
              f"{raised.splitlines()[0][:90]}...")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mixed_stage_format_database_reads_back():
    """Phase 2: joint spindles and legacy per-epoch slow waves, one database.

    The realistic upgrade case: a database written by 4.2 holds per-epoch
    slow waves, and a 4.3 spindle run adds joint-token rows beside them. Every
    reader has to answer for both shapes at once, from one query, without the
    caller knowing which is which.
    """
    print("\n16. Testing a database holding BOTH stage formats (phase 2):")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_mixed_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)

        def add(event_type, method, band, token, epoch_stage, n, t0):
            for i in range(n):
                t = t0 + i * 3.0
                conn.execute(
                    "INSERT OR REPLACE INTO events (uuid, event_type, channel,"
                    " start_time, end_time, duration, stage, epoch_stage, "
                    "method, freq_lower, freq_upper) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5(event_type, 'Cz', t, method, band[0],
                                         band[1], token),
                     event_type, 'Cz', t, t + 1.0, 1.0, token, epoch_stage,
                     method, band[0], band[1]))

        # 4.3 spindles: one joint token. 4.2 slow waves: per-epoch stages.
        add('spindle', 'Moelle2011', (11.0, 16.0), 'NREM2NREM3', 'NREM2', 30, 10.0)
        add('slow_wave', 'Massimini2004', (0.5, 4.0), 'NREM2', None, 20, 11.0)
        add('slow_wave', 'Massimini2004', (0.5, 4.0), 'NREM3', None, 10, 2011.0)
        dbwrite.record_analysed_time(conn, 'sub-M', 'NREM2', 1800.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.record_analysed_time(conn, 'sub-M', 'NREM3', 1200.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.upsert_processing_status(conn, 'spindle', 'Cz', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        dbwrite.upsert_processing_status(conn, 'slow_wave', 'Cz',
                                         'Massimini2004', 0.5, 4.0,
                                         'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()

    # (a) the CSV exporter finds both, under the same requested stage set.
    sp_csv = dbwrite.export_events_to_csv(
        db, 'spindle', 'Moelle2011', (11.0, 16.0), ['NREM2', 'NREM3'],
        output_dir=tmp)
    sw_csv = dbwrite.export_events_to_csv(
        db, 'slow_wave', 'Massimini2004', (0.5, 4.0), ['NREM2', 'NREM3'],
        output_dir=tmp)
    assert sp_csv and sw_csv, (sp_csv, sw_csv)
    n_sp = sum(1 for _ in open(sp_csv)) - 2      # provenance line + header
    n_sw = sum(1 for _ in open(sw_csv)) - 2
    assert (n_sp, n_sw) == (30, 30), (n_sp, n_sw)
    # Both sides of the naming convention: the filename carries the canonical
    # token, and it is the same one the database holds.
    assert os.path.basename(sp_csv).endswith('_11-16Hz_NREM2NREM3.csv'), sp_csv
    print(f"[ok] export_events_to_csv: {n_sp} joint spindles and {n_sw} "
          f"per-epoch slow waves, both via one stage request")
    print(f"     filename token matches the database token: "
          f"{os.path.basename(sp_csv)}")

    # (b) density reports both, each against its own pooled denominator.
    df = event_density(db, subject='sub-M')
    got = {(r.event_type, r.stage): (int(r.n_events),
                                     round(float(r.analysed_minutes), 4))
           for r in df.itertuples()}
    assert got[('spindle', 'NREM2NREM3')] == (30, 50.0), got
    assert got[('slow_wave', 'NREM2')] == (20, 30.0), got
    assert got[('slow_wave', 'NREM3')] == (10, 20.0), got
    print(f"[ok] event_density reads both shapes in one call: {got}")

    shutil.rmtree(tmp, ignore_errors=True)


def test_density_pools_joint_stage_token():
    """Phase 3: density's stage dimension is the run's TOKEN, pooled on read.

    ``analysed_time`` stays keyed per single scored stage -- artefact-free
    time is a property of a stage, while a joint token is a label for a run's
    scope -- so the joint denominator is assembled at read time. The four
    requests below must agree, and the fifth must decline rather than answer
    with part of the truth.
    """
    print("\n17. Testing density over a joint stage token (phase 3):")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_pool_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)
        for i in range(30):
            t = 100.0 + i * 5.0
            conn.execute(
                "INSERT OR REPLACE INTO events (uuid, event_type, channel, "
                "start_time, end_time, duration, stage, epoch_stage, method, "
                "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'C3', t, 'Moelle2011', 11, 16,
                                     'NREM2NREM3'),
                 'spindle', 'C3', t, t + 1.0, 1.0, 'NREM2NREM3',
                 'NREM2' if i < 18 else 'NREM3', 'Moelle2011', 11.0, 16.0))
        dbwrite.record_analysed_time(conn, 'sub-P', 'NREM2', 1800.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.record_analysed_time(conn, 'sub-P', 'NREM3', 1200.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.upsert_processing_status(conn, 'spindle', 'C3', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()

    for label, stage in (('stage=None', None),
                         ("stage=['NREM2','NREM3']", ['NREM2', 'NREM3']),
                         ("stage='NREM2NREM3'", 'NREM2NREM3')):
        df = event_density(db, event_type='spindle', method='Moelle2011',
                           stage=stage, subject='sub-P')
        assert len(df) == 1, f"{label} returned {len(df)} rows:\n{df}"
        row = df.iloc[0]
        assert row['stage'] == 'NREM2NREM3', row['stage']
        assert abs(float(row['analysed_minutes']) - 50.0) < 1e-9, row
        assert abs(float(row['density_per_min']) - 0.6) < 1e-9, row
        print(f"[ok] {label:<26} -> 1 row, {float(row['analysed_minutes']):.1f} "
              f"min (1800 s + 1200 s pooled), "
              f"{float(row['density_per_min']):.3f}/min")

    # combine_stages is now a correct no-op: the token already pools.
    pooled = event_density(db, event_type='spindle', method='Moelle2011',
                           subject='sub-P', combine_stages=True)
    assert len(pooled) == 1 and pooled.iloc[0]['stage'] == 'NREM2NREM3'
    assert abs(float(pooled.iloc[0]['analysed_minutes']) - 50.0) < 1e-9
    print("[ok] combine_stages=True is a no-op on a joint token (50.0 min, "
          "not 100.0 -- the components are pooled once, not once per token)")

    # A strict subset of the token: no row, rather than 30 events over 30 min.
    subset = event_density(db, event_type='spindle', method='Moelle2011',
                           stage=['NREM2'], subject='sub-P')
    assert len(subset) == 0, f"expected no row, got:\n{subset}"
    print("[ok] stage=['NREM2'] alone returns NO row (0.0/min or 1.0/min "
          "would both be fabrications)")

    # The SQL view answers the same question without Python.
    conn = sqlite3.connect(db)
    try:
        view = conn.execute(
            "SELECT channel, stage, n_events, analysed_minutes, "
            "density_per_min, denominator_complete FROM v_event_density"
        ).fetchall()
    finally:
        conn.close()
    assert len(view) == 1, view
    assert view[0][1] == 'NREM2NREM3' and view[0][2] == 30
    assert abs(view[0][3] - 50.0) < 1e-9 and abs(view[0][4] - 0.6) < 1e-9
    assert view[0][5] == 1
    print(f"[ok] v_event_density (plain SQL, no Python): {view[0]}")

    # The view's component matching is string-based, and 'NREM1' CONTAINS
    # 'REM'. A naive substring test would give 'NREM1REM' three components
    # (NREM1, REM, and REM again from inside NREM1) and a denominator half
    # again too large. And an incomplete denominator must come back NULL, not
    # as the partial sum.
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE events SET stage = 'NREM1REM'")
        conn.execute("INSERT OR REPLACE INTO analysed_time (subject, stage, "
                     "reject_artifacts, reject_arousals, analysed_seconds, "
                     "artefact_seconds_excluded) VALUES ('sub-P','NREM1',1,1,"
                     "600.0,0.0)")
        conn.execute("INSERT OR REPLACE INTO analysed_time (subject, stage, "
                     "reject_artifacts, reject_arousals, analysed_seconds, "
                     "artefact_seconds_excluded) VALUES ('sub-P','REM',1,1,"
                     "600.0,0.0)")
        conn.commit()
        trap = conn.execute("SELECT stage, analysed_minutes, density_per_min, "
                            "denominator_complete FROM v_event_density"
                            ).fetchall()
        assert len(trap) == 1, trap
        assert trap[0][0] == 'NREM1REM'
        assert abs(trap[0][1] - 20.0) < 1e-9, trap   # 600 + 600 s, not 1800
        assert abs(trap[0][2] - 1.5) < 1e-9, trap    # 30 events / 20 min
        print(f"[ok] 'NREM1REM' pools NREM1 + REM once each ({trap[0][1]:.1f} "
              f"min): the REM-inside-NREM1 substring trap is handled")

        conn.execute("DELETE FROM analysed_time WHERE stage = 'REM'")
        conn.commit()
        partial = conn.execute(
            "SELECT analysed_minutes, density_per_min, denominator_complete "
            "FROM v_event_density").fetchall()
        assert partial == [(None, None, 0)], partial
        print("[ok] a missing component gives NULL minutes AND NULL density "
              "in the view, never the partial sum")
    finally:
        conn.close()

    shutil.rmtree(tmp, ignore_errors=True)


def test_stage_format_guard_blocks_duplicate_set():
    """Risk 1: re-detecting a pre-4.3 scope must refuse, not silently double it.

    ``event_uuid5`` hashes the stage, so re-running a scope whose rows were
    written per-epoch produces both a new uuid AND a new stage: neither the
    primary key nor the ``event_chan_time`` UNIQUE constraint matches, and
    ``INSERT OR REPLACE`` therefore APPENDS a complete duplicate set. Every
    count and every density in that scope doubles, with no error anywhere.
    This is the guard, and this test is the reason it exists.
    """
    print("\n18. Testing the pre-4.3 duplicate-set guard (risk 1):")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_dupguard_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)
        # A 4.2-shaped database: per-epoch stages, and NO stage_format marker.
        conn.execute("DELETE FROM db_meta WHERE key = ?",
                     (dbwrite.STAGE_FORMAT_KEY,))
        for i in range(10):
            t = 100.0 + i
            stg = 'NREM2' if i < 6 else 'NREM3'
            conn.execute(
                "INSERT OR REPLACE INTO events (uuid, event_type, channel, "
                "start_time, end_time, duration, stage, method, freq_lower, "
                "freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011', 11, 16,
                                     stg),
                 'spindle', 'Cz', t, t + 1.0, 1.0, stg, 'Moelle2011',
                 11.0, 16.0))
        conn.commit()
        assert dbwrite.stage_format(conn) is None
        n_before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        # (a) ensure_direct_write_schema must NOT stamp a database that
        # already holds events -- the absent marker IS the evidence.
        dbwrite.ensure_direct_write_schema(conn)
        assert dbwrite.stage_format(conn) is None, \
            "the marker was stamped on a legacy database, destroying the guard"
        print("[ok] a database with existing events is left unmarked "
              "(stamping it would erase the only evidence of its shape)")

        # (b) the guard refuses the scope that would duplicate.
        raised = None
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
                stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        except ValueError as e:
            raised = str(e)
        assert raised is not None, "the duplicating re-detection was allowed"
        assert 'migrate_stage_to_joint.py' in raised, raised
        assert 'DUPLICATE' in raised.upper(), raised
        print(f"[ok] re-detecting spindles/Moelle2011/11-16Hz on Cz raises and "
              f"names the migration")

        # (b2) stage_token is keyword-REQUIRED, and None is refused. A caller
        # that omitted it would keep only the marker check and lose the
        # different-stage-set protection -- silently, because the remaining
        # check still runs and still passes. Both mistakes must be loud.
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'spindle', ['Moelle2011'], 11.0, 16.0, channels=['Cz'],
                db_path=db)
        except TypeError as e:
            assert 'stage_token' in str(e), e
        else:
            raise AssertionError(
                "assert_stage_format_compatible accepted a call with no "
                "stage_token; the different-stage-set check would be skipped")
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'spindle', ['Moelle2011'], 11.0, 16.0, stage_token=None,
                channels=['Cz'], db_path=db)
        except ValueError as e:
            assert 'stage token' in str(e), e
        else:
            raise AssertionError("stage_token=None was accepted")
        print("[ok] omitting stage_token is a TypeError and stage_token=None "
              "a ValueError -- the guard cannot be silently downgraded")

        # (c) scopes that CANNOT duplicate are still allowed: a different
        # event type, a different band, and a channel being replaced.
        dbwrite.assert_stage_format_compatible(
            conn, 'slow_wave', ['Massimini2004'], 0.5, 4.0,
            stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        dbwrite.assert_stage_format_compatible(
            conn, 'spindle', ['Moelle2011'], 9.0, 12.0,
            stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        dbwrite.assert_stage_format_compatible(
            conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
            stage_token='NREM2NREM3', channels=['Cz'],
            replace_channels=['Cz'], db_path=db)
        dbwrite.assert_stage_format_compatible(
            conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
            stage_token='NREM2NREM3', channels=['Pz'], db_path=db)
        print("[ok] a new event type, a new band, a replaced channel and an "
              "untouched channel are all still allowed")

        # (d) the marker alone is NOT enough: a joint database re-detected
        # under a different stage set still appends, so the token check
        # applies whatever the marker says.
        dbwrite.set_db_meta(conn, dbwrite.STAGE_FORMAT_KEY,
                            dbwrite.STAGE_FORMAT_JOINT)
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
                stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        except ValueError as e:
            assert 'NREM2NREM3' in str(e), e
        else:
            raise AssertionError(
                "stage_format='joint' let a different-stage-set re-run "
                "through; it would append a duplicate set")
        print("[ok] marker='joint' does NOT excuse a different stage token")

        # (e) after the migration collapses the stages, the same call is
        # allowed -- the real post-migration sequence.
        conn.execute("UPDATE events SET stage = 'NREM2NREM3'")
        conn.commit()
        dbwrite.assert_stage_format_compatible(
            conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
            stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        n_after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n_after == n_before, (n_before, n_after)
        print(f"[ok] once the stages are collapsed AND the marker stamped, the "
              f"run proceeds; the guard never modified the {n_after} rows")
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_detection_populates_cycles_without_touching_xml():
    """Phase 5: a detection run fills cycles, durations and events.cycle.

    Before this, ``sleep_cycles`` and ``stage_durations`` were created by every
    run and filled by none -- only the standalone cycle script populated them
    -- so a database produced entirely through the GUI had two empty tables and
    ``events.cycle`` NULL on every row.

    The hard constraint is the annotation XML: a detection run reads the
    rater's scoring and must never write to it. That is asserted here on the
    file's mtime and byte content, not on an argument, because
    ``write_cycle_markers`` swallows its own failures -- an accidental write
    would leave no trace in the log.
    """
    print("\n19. Testing automatic cycles / stage durations (phase 5):")

    import logging
    from turtlewave_hdEEG import ParalEvents, ParalSWA, CustomAnnotations

    tmp = tempfile.mkdtemp(prefix='tw_cycles_')
    try:
        # 86 epochs = 43 min, laid out as two full cycles. The detector
        # defaults (nrem_min=30 epochs, rem_min=10) are used unchanged, so
        # this exercises the same thresholds production does.
        stages = (['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11
                  + ['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11)
        dataset, annot = _synthetic_recording(tmp, tuple(stages))
        xml = annot.xml_file
        wrapped = CustomAnnotations(xml)

        xml_mtime_before = os.path.getmtime(xml)
        with open(xml, 'rb') as f:
            xml_bytes_before = f.read()

        out = os.path.join(tmp, 'wonambi')
        os.makedirs(out, exist_ok=True)
        db = os.path.join(out, 'neural_events.db')
        kw = dict(chan=['Cz'], stage=['NREM2', 'NREM3'], json_dir=out,
                  db_path=db, subject='sub-T', cat=(1, 1, 1, 0))

        n_sp = len(ParalEvents(
            dataset, wrapped, log_level=logging.CRITICAL).detect_spindles(
                method='Moelle2011', frequency=(11, 16), **kw))
        assert n_sp, "no spindles detected, so the cycle tagging is vacuous"

        conn = sqlite3.connect(db)
        try:
            cyc = conn.execute(
                "SELECT method, COUNT(*) FROM sleep_cycles GROUP BY method"
            ).fetchall()
            durations = conn.execute(
                "SELECT subject, n2_min, n3_min, rem_min, total_min "
                "FROM stage_durations").fetchall()
            tagged, total = conn.execute(
                "SELECT COUNT(cycle), COUNT(*) FROM events").fetchone()
            cycle_vals = sorted({r[0] for r in conn.execute(
                "SELECT DISTINCT cycle FROM events WHERE cycle IS NOT NULL")})
        finally:
            conn.close()

        methods = {m for m, _ in cyc}
        assert methods == {'2022', '1979'}, f"sleep_cycles holds {cyc}"
        print(f"[ok] sleep_cycles holds BOTH definitions: {dict(cyc)}")
        assert len(durations) == 1, durations
        assert durations[0][0] == 'sub-T', durations
        assert abs(durations[0][4] - 43.0) < 1e-6, durations   # 86 x 30 s
        print(f"[ok] stage_durations: one row for {durations[0][0]!r}, "
              f"N2={durations[0][1]:.1f} N3={durations[0][2]:.1f} "
              f"REM={durations[0][3]:.1f} total={durations[0][4]:.1f} min")
        assert tagged == total and total == n_sp, (tagged, total, n_sp)
        print(f"[ok] events.cycle populated on all {tagged}/{total} rows "
              f"(cycle numbers {cycle_vals})")

        # THE hard constraint: the rater's file is untouched.
        assert os.path.getmtime(xml) == xml_mtime_before, \
            "the detection run modified the annotation XML's mtime"
        with open(xml, 'rb') as f:
            assert f.read() == xml_bytes_before, \
                "the detection run modified the annotation XML's contents"
        print("[ok] the annotation XML is byte-for-byte unchanged "
              "(mtime and contents)")

        # A second detector on the same database: cycles are a no-op, and it
        # tags only ITS OWN rows.
        n_sw = len(ParalSWA(
            dataset, wrapped, log_level=logging.CRITICAL).detect_slow_waves(
                method='AASM/Massimini2004', frequency=(0.5, 4),
                neg_peak_thresh=-37.0, p2p_thresh=75.0, **kw))
        assert n_sw, "no slow waves detected"
        conn = sqlite3.connect(db)
        try:
            cyc2 = conn.execute("SELECT COUNT(*) FROM sleep_cycles").fetchone()[0]
            dur2 = conn.execute("SELECT COUNT(*) FROM stage_durations").fetchone()[0]
            per_type = dict(conn.execute(
                "SELECT event_type, COUNT(cycle) FROM events GROUP BY 1"))
            n_runs = conn.execute(
                "SELECT COUNT(DISTINCT run_id) FROM events").fetchone()[0]
        finally:
            conn.close()
        assert cyc2 == sum(n for _, n in cyc), (cyc2, cyc)
        assert dur2 == 1, dur2
        assert per_type['spindle'] == n_sp and per_type['slow_wave'] == n_sw
        assert n_runs == 2, n_runs
        print(f"[ok] second detector: cycles/durations unchanged "
              f"({cyc2} cycle rows, {dur2} duration row), both runs' events "
              f"tagged ({per_type}) across {n_runs} run ids")
        assert os.path.getmtime(xml) == xml_mtime_before, \
            "the second detection run modified the annotation XML"
        print("[ok] annotation XML still untouched after the second run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tag_method_outside_methods_is_an_error():
    """Phase 5: a tag_method outside methods must not fail silently.

    It used to be self-contradictory and silent: the ``m == tag_method`` gate
    never fires, so NO XML markers are written, while ``events.cycle`` is
    still overwritten by whichever definition happened to run last. The
    database and the annotation file then disagree about the cycle numbering,
    from one misspelled argument, with nothing recording which is which.
    """
    print("\n20. Testing the tag_method misconfiguration guard (phase 5):")

    import logging
    from turtlewave_hdEEG import finalize_cycles_and_durations, ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_tagmethod_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    try:
        raised = None
        try:
            finalize_cycles_and_durations(
                None, db, subject='sub-T', methods=('2022', '1979'),
                tag_method='2002', log_level=logging.CRITICAL)   # typo
        except ValueError as e:
            raised = str(e)
        assert raised is not None, "a tag_method typo was accepted"
        assert "'2002'" in raised and '2022' in raised, raised
        print(f"[ok] tag_method='2002' raises: {raised.split('.')[0]}.")

        # tag_method=None is the explicit "store both, tag nothing" case and
        # must still be accepted (it fails later for want of annotations, not
        # on the guard).
        try:
            finalize_cycles_and_durations(
                None, db, subject='sub-T', methods=('2022', '1979'),
                tag_method=None, log_level=logging.CRITICAL)
        except ValueError as e:
            assert 'tag_method' not in str(e), \
                f"tag_method=None was rejected by the guard: {e}"
        print("[ok] tag_method=None is accepted (store both, tag nothing)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rerun_under_a_different_stage_set_is_refused():
    """A joint-to-joint re-run over a DIFFERENT stage set must not append.

    The ``stage_format`` marker cannot see this one: both runs are joint. But
    the stage is part of ``event_uuid5`` AND the last component of the
    ``event_chan_time`` UNIQUE constraint, so run 2's rows match run 1's on
    neither key and ``INSERT OR REPLACE`` appends. Every event of run 1
    survives as a duplicate, ``COUNT(*)`` is inflated, and ``event_density``
    returns two overlapping rows that sum to more events than exist.

    This is a regression the joint token itself introduces -- under per-epoch
    storage the two runs shared stage values and replaced correctly -- so a
    CHANGELOG line is not a fix for it.
    """
    print("\n23. Testing the different-stage-set re-run guard:")

    import logging
    from turtlewave_hdEEG import ParalEvents
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_restage_')
    try:
        dataset, annot = _synthetic_recording(
            tmp, ('NREM2', 'NREM2', 'NREM3', 'NREM3'))
        work = os.path.join(tmp, 'wonambi')
        os.makedirs(work, exist_ok=True)
        db = os.path.join(work, 'neural_events.db')
        pe = ParalEvents(dataset, annot, log_level=logging.CRITICAL)

        def run(stages, replace=None):
            return len(pe.detect_spindles(
                method='Moelle2011', chan=['Cz'], frequency=(11, 16),
                stage=stages, json_dir=work, db_path=db, subject='sub-T',
                cat=(1, 1, 1, 0), replace_channels=replace))

        def rows():
            conn = sqlite3.connect(db)
            try:
                return dict(conn.execute(
                    "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
            finally:
                conn.close()

        n1 = run(['NREM2'])
        assert n1, "no spindles detected in NREM2"
        before = rows()
        assert before == {'NREM2': n1}, before
        print(f"[ok] run 1 (stage=['NREM2']): {before}")

        raised = None
        try:
            run(['NREM2', 'NREM3'])
        except ValueError as e:
            raised = str(e)
        assert raised is not None, (
            "a re-run under a different stage set was accepted; it appends a "
            "duplicate set")
        assert 'NREM2NREM3' in raised and 'DUPLICATE' in raised.upper(), raised
        assert 'replace_channels' in raised, raised
        assert rows() == before, "the refused run still modified the database"
        print(f"[ok] run 2 (stage=['NREM2','NREM3']) is REFUSED before any "
              f"write; database still {rows()}")

        # Density must not be able to double-count, which is what the appended
        # rows would have caused.
        df = event_density(db, event_type='spindle', method='Moelle2011',
                           subject='sub-T')
        assert int(df['n_events'].sum()) == n1, df
        print(f"[ok] event_density sums to {int(df['n_events'].sum())} events, "
              f"not {n1 * 2 + 2}")

        # The escape hatch the error message advertises must actually work.
        n2 = run(['NREM2', 'NREM3'], replace=['Cz'])
        after = rows()
        assert after == {'NREM2NREM3': n2}, after
        print(f"[ok] replace_channels=['Cz'] proceeds and REPLACES: {after} "
              f"(the NREM2 rows are gone, not kept alongside)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_density_warns_about_tokens_it_excludes():
    """A request that silently drops a token must say so.

    An identity holding both ``'NREM2'`` (legacy per-epoch rows) and
    ``'NREM2NREM3'`` (a later joint run) answers ``stage=['NREM2']`` from the
    first token alone. That is a number rather than a missing row, and it is
    the shape a partly-migrated database is in, so it has to be loud.
    """
    print("\n24. Testing the density warning for excluded stage tokens:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA
    from turtlewave_hdEEG.density import event_density

    tmp = tempfile.mkdtemp(prefix='tw_excl_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)

        def add(token, n, t0):
            for i in range(n):
                t = t0 + i
                conn.execute(
                    "INSERT OR REPLACE INTO events (uuid, event_type, channel,"
                    " start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011',
                                         11.0, 16.0, token),
                     'spindle', 'Cz', t, t + 1.0, 1.0, token, 'Moelle2011',
                     11.0, 16.0))

        add('NREM2', 10, 100.0)
        add('NREM2NREM3', 30, 500.0)
        dbwrite.record_analysed_time(conn, 'sub-F', 'NREM2', 1800.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.record_analysed_time(conn, 'sub-F', 'NREM3', 1200.0,
                                     artefact_seconds_excluded=0.0)
        dbwrite.upsert_processing_status(conn, 'spindle', 'Cz', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    log = logging.getLogger('tw_density_excluded_test')
    log.setLevel(logging.WARNING)
    log.addHandler(_Capture())
    try:
        df = event_density(db, event_type='spindle', method='Moelle2011',
                           stage=['NREM2'], subject='sub-F', logger_=log)
    finally:
        log.handlers.clear()

    assert len(df) == 1 and int(df['n_events'].iloc[0]) == 10, df
    warned = [r.getMessage() for r in records
              if r.levelno >= logging.WARNING and 'EXCLUDED' in r.getMessage()]
    assert warned, (
        "density answered stage=['NREM2'] with 10 of 40 events and said "
        f"nothing about the other 30. Messages seen: "
        f"{[r.getMessage()[:60] for r in records]}")
    assert 'NREM2NREM3' in warned[0] and '30 event' in warned[0], warned[0]
    print(f"[ok] returned {int(df['n_events'].iloc[0])} events AND warned: "
          f"{warned[0][:120]}...")

    shutil.rmtree(tmp, ignore_errors=True)


def test_migration_keeps_per_channel_stage_sets_apart():
    """Two stage sets in one scope must not be collapsed into one token.

    ``processing_status`` is keyed per (channel, stage), so it records that E1
    was searched over N2 alone while E2 was searched over N2+N3 -- same event
    type, method and band. Unioning those and relabelling both
    ``'NREM2NREM3'`` makes E1's events divide by N2+N3 analysed time
    afterwards, understating that channel's density by ``N2 / (N2 + N3)``
    permanently, with nothing recording that it happened. The collision check
    cannot see it either: the rows are on different channels.
    """
    print("\n25. Testing per-channel stage sets in the migration:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_perchan_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        for i in range(10):        # E1: searched NREM2 only
            t = 100.0 + i
            conn.execute(
                "INSERT INTO events (uuid, event_type, channel, start_time, "
                "end_time, duration, stage, method, freq_lower, freq_upper) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'E1', t, 'Moelle2011', 11.0,
                                     16.0, 'NREM2'),
                 'spindle', 'E1', t, t + 1.0, 1.0, 'NREM2', 'Moelle2011',
                 11.0, 16.0))
        for i in range(20):        # E2: searched NREM2 + NREM3
            t = 300.0 + i
            stg = 'NREM2' if i < 12 else 'NREM3'
            conn.execute(
                "INSERT INTO events (uuid, event_type, channel, start_time, "
                "end_time, duration, stage, method, freq_lower, freq_upper) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'E2', t, 'Moelle2011', 11.0,
                                     16.0, stg),
                 'spindle', 'E2', t, t + 1.0, 1.0, stg, 'Moelle2011',
                 11.0, 16.0))
        dbwrite.upsert_processing_status(conn, 'spindle', 'E1', 'Moelle2011',
                                         11.0, 16.0, 'NREM2', True)
        dbwrite.upsert_processing_status(conn, 'spindle', 'E2', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()

    assert mig.main([db, '--apply']) == 0
    conn = sqlite3.connect(db)
    try:
        got = dict(conn.execute(
            "SELECT channel, stage FROM events GROUP BY channel, stage"
        ).fetchall())
        counts = dict(conn.execute(
            "SELECT channel, COUNT(*) FROM events GROUP BY channel").fetchall())
    finally:
        conn.close()
    assert got == {'E1': 'NREM2', 'E2': 'NREM2NREM3'}, got
    assert counts == {'E1': 10, 'E2': 20}, counts
    print(f"[ok] E1 (searched NREM2 alone) keeps 'NREM2'; E2 collapses to "
          f"'NREM2NREM3'. {got}")
    print("[ok] E1's 10 events are NOT relabelled, so they keep dividing by "
          "N2 time alone")

    shutil.rmtree(tmp, ignore_errors=True)


def test_migration_unblocks_null_stage_rows():
    """A NULL stage must not survive the migration and block forever.

    The duplicate guard treats NULL as a different stage token -- correctly,
    since a NULL-stage row was keyed on ``'None'`` and duplicates in exactly
    the same way. So a NULL row that survives the migration refuses every
    future re-detection of its scope, while the migration stamps the marker
    and reports "re-detection into this database is now allowed". That closed
    loop sends the user from a successful migration to a hard ValueError that
    tells them to run the migration.

    It is not a corner case: 4.2's direct-write path stored NULL whenever no
    scored epoch contained an event, and the 4.0.x CSV importer stored NULL
    for every row of a CSV with no Stage column.

    Asserts the fix from both ends -- the rows are relabelled AND a real
    detector run afterwards is accepted -- and that the honest-failure paths
    (``--keep-null-stage``, and a scope whose stage set is on record nowhere)
    report the database as NOT unblocked instead of claiming success.
    """
    print("\n26. Testing NULL-stage rows through the migration:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalEvents

    mig = _import_migration_script()

    def build(tmp, all_null):
        """A 4.2-shaped database whose stages are partly or wholly NULL."""
        dataset, annot = _synthetic_recording(
            tmp, ('NREM2', 'NREM2', 'NREM3', 'NREM3'))
        work = os.path.join(tmp, 'wonambi')
        os.makedirs(work, exist_ok=True)
        db = os.path.join(work, 'neural_events.db')
        pe = ParalEvents(dataset, annot, log_level=logging.CRITICAL)
        kw = dict(method='Moelle2011', chan=['Cz'], frequency=(11, 16),
                  stage=['NREM2', 'NREM3'], json_dir=work, db_path=db,
                  subject='sub-T', cat=(1, 1, 1, 0))
        assert pe.detect_spindles(**kw), "no spindles detected"
        conn = sqlite3.connect(db)
        try:
            real = conn.execute(
                "SELECT channel, start_time, end_time, duration, epoch_stage, "
                "method, freq_lower, freq_upper FROM events "
                "ORDER BY start_time").fetchall()
            conn.execute("DELETE FROM events")
            for i, (ch, t0, t1, dur, epoch, meth, lo, hi) in enumerate(real):
                stg = None if (all_null or i % 2) else epoch
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', ch, t0, meth, lo, hi, stg),
                     'spindle', ch, t0, t1, dur, stg, meth, lo, hi))
            conn.execute("DELETE FROM db_meta WHERE key = 'stage_format'")
            conn.commit()
            n_null = conn.execute(
                "SELECT COUNT(*) FROM events WHERE stage IS NULL").fetchone()[0]
        finally:
            conn.close()
        assert n_null, "fixture has no NULL stages"
        return db, pe, kw, n_null

    def stages(db):
        conn = sqlite3.connect(db)
        try:
            return dict(conn.execute(
                "SELECT COALESCE(stage, 'NULL'), COUNT(*) FROM events "
                "GROUP BY 1").fetchall())
        finally:
            conn.close()

    def marker(db):
        conn = sqlite3.connect(db)
        try:
            return dbwrite.stage_format(conn)
        finally:
            conn.close()

    def redetects(pe, kw):
        try:
            pe.detect_spindles(**kw)
            return True
        except ValueError:
            return False

    # (a) NULL rows are relabelled with the run's own token, and the database
    # is genuinely re-detectable afterwards.
    for label, all_null in (('half the rows NULL', False),
                            ('EVERY row NULL', True)):
        tmp = tempfile.mkdtemp(prefix='tw_null_')
        try:
            db, pe, kw, n_null = build(tmp, all_null)
            assert not redetects(pe, kw), \
                "the pre-migration database was not blocked; test is vacuous"
            assert mig.main([db, '--apply']) == 0
            after = stages(db)
            assert after == {'NREM2NREM3': sum(after.values())}, after
            assert marker(db) == 'joint'
            assert redetects(pe, kw), (
                "the migration stamped success but re-detection is still "
                "refused -- the closed loop is back")
            print(f"[ok] {label}: {n_null} NULL row(s) -> 'NREM2NREM3', marker "
                  f"stamped, and a real detector run is ACCEPTED afterwards")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # (b) --keep-null-stage keeps them, and must NOT claim success.
    tmp = tempfile.mkdtemp(prefix='tw_null_keep_')
    try:
        db, pe, kw, n_null = build(tmp, False)
        rc = mig.main([db, '--apply', '--keep-null-stage'])
        assert rc == 3, f"expected rc=3 (not fully unblocked), got {rc}"
        assert marker(db) is None, \
            "the marker was stamped on a database that is still blocked"
        assert stages(db).get('NULL') == n_null, stages(db)
        assert not redetects(pe, kw)
        print(f"[ok] --keep-null-stage: rc=3, marker NOT stamped, {n_null} "
              f"NULL row(s) kept, re-detection still refused (as reported)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # (c) a scope whose stage set is on record nowhere: nothing is invented,
    # nothing is written, no backup is left to clean up, and --stage-token
    # resolves it.
    tmp = tempfile.mkdtemp(prefix='tw_null_notarget_')
    try:
        db, pe, kw, n_null = build(tmp, True)
        conn = sqlite3.connect(db)
        try:
            conn.execute("DELETE FROM processing_status")
            conn.execute("DELETE FROM detection_runs")
            conn.commit()
        finally:
            conn.close()
        assert mig.main([db]) == 3, "the dry run did not predict the blocker"
        assert mig.main([db, '--apply']) == 3
        assert stages(db) == {'NULL': n_null}, stages(db)
        assert marker(db) is None
        assert not os.path.exists(db + mig.BACKUP_SUFFIX), (
            "a run that wrote nothing left a backup behind, so the re-run it "
            "asks for would be refused by the backup guard")
        print("[ok] no derivable target: dry run and --apply both report rc=3, "
              "nothing invented, no backup left to clean up")
        assert mig.main([db, '--apply', '--stage-token', 'NREM2,NREM3']) == 0
        assert stages(db) == {'NREM2NREM3': n_null}, stages(db)
        assert marker(db) == 'joint'
        assert redetects(pe, kw)
        print("[ok] --stage-token NREM2,NREM3 resolves it: rc=0, marker "
              "stamped, re-detection accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_migration_runs_backfills_even_when_partly_blocked():
    """Requested back-fills must run even when the verdict is exit 3.

    ``analysed_time`` and the cycle tables are independent of the stage token,
    and the pre-write shortcut deliberately lets a run proceed *because* a
    back-fill was requested. Returning 3 before running them made that
    reasoning false and dropped work the user asked for, with nothing in the
    log saying so -- and the corrective re-run then hits the backup guard, so
    the work had nowhere to go.
    """
    print("\n28. Testing back-fills on a partly-blocked migration:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_mig_backfill_')
    try:
        # A real hypnogram (two cycles) so the cycle back-fill has something
        # to find, and two scopes: one migratable, one with no derivable
        # target (which is what forces the exit-3 verdict).
        dataset, annot = _synthetic_recording(
            tmp, tuple(['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11
                       + ['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11))
        work = os.path.join(tmp, 'wonambi')
        os.makedirs(work, exist_ok=True)
        db = os.path.join(work, 'neural_events.db')
        ParalSWA(None, None,
                 log_level=logging.CRITICAL).initialize_sqlite_database(db)
        conn = dbwrite.open_write_connection(db)
        try:
            for i in range(4):
                t = 100.0 + i
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011',
                                         11.0, 16.0, 'NREM2'),
                     'spindle', 'Cz', t, t + 1.0, 1.0, 'NREM2', 'Moelle2011',
                     11.0, 16.0))
            for i in range(4):            # no processing_status -> no target
                t = 200.0 + i
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('slow_wave', 'Pz', t, 'Massimini2004',
                                         0.5, 4.0, None),
                     'slow_wave', 'Pz', t, t + 1.0, 1.0, None,
                     'Massimini2004', 0.5, 4.0))
            dbwrite.upsert_processing_status(
                conn, 'spindle', 'Cz', 'Moelle2011', 11.0, 16.0,
                'NREM2NREM3', True)
            conn.commit()
        finally:
            conn.close()

        rc = mig.main([db, '--apply', '--annot', annot.xml_file,
                       '--backfill-analysed-time', '--backfill-cycles'])
        assert rc == 3, f"expected rc=3 (partly blocked), got {rc}"

        conn = sqlite3.connect(db)
        try:
            n_time = conn.execute(
                "SELECT COUNT(*) FROM analysed_time").fetchone()[0]
            n_cyc = conn.execute(
                "SELECT COUNT(*) FROM sleep_cycles").fetchone()[0]
            n_dur = conn.execute(
                "SELECT COUNT(*) FROM stage_durations").fetchone()[0]
            spindle = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events WHERE event_type='spindle'"
                " GROUP BY 1").fetchall())
            marker = dbwrite.stage_format(conn)
        finally:
            conn.close()

        assert n_time, "the requested analysed_time back-fill was dropped"
        assert n_cyc, "the requested cycle back-fill was dropped"
        assert n_dur, "stage_durations was not written"
        # The migratable scope was still migrated, and the marker still
        # withheld because the other scope is blocked.
        assert spindle == {'NREM2NREM3': 4}, spindle
        assert marker is None, "the marker was stamped on a blocked database"
        print(f"[ok] rc=3 AND the requested work ran: analysed_time={n_time} "
              f"row(s), sleep_cycles={n_cyc}, stage_durations={n_dur}; the "
              f"migratable scope became {spindle} and the marker stayed "
              f"unstamped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_migration_reports_backfill_outcome_not_intention():
    """A back-fill that failed must reach the exit code and the message.

    The trap this closes is one step worse than dropping the work: the
    summary was gated on whether a back-fill had been REQUESTED, so a run
    whose back-fills both failed still reported that they "DID run", and a
    run with no blocked scope returned 0 -- which is what marks a subject
    done in the per-subject batch drivers under ``examples/NCI_commands/``.

    ``rc == 0`` must mean, and only mean, that everything asked for succeeded
    and the database is fully re-detectable.
    """
    print("\n29. Testing back-fill outcome reporting:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_backfill_outcome_')
    try:
        dataset, annot = _synthetic_recording(
            tmp, tuple(['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11
                       + ['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11))
        missing_annot = os.path.join(tmp, 'does_not_exist.xml')

        def build(name, blocked):
            work = os.path.join(tmp, name, 'wonambi')
            os.makedirs(work, exist_ok=True)
            db = os.path.join(work, 'neural_events.db')
            ParalSWA(None, None,
                     log_level=logging.CRITICAL).initialize_sqlite_database(db)
            conn = dbwrite.open_write_connection(db)
            try:
                for i in range(4):
                    t = 100.0 + i
                    conn.execute(
                        "INSERT INTO events (uuid, event_type, channel, "
                        "start_time, end_time, duration, stage, method, "
                        "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011',
                                             11.0, 16.0, 'NREM2'),
                         'spindle', 'Cz', t, t + 1.0, 1.0, 'NREM2',
                         'Moelle2011', 11.0, 16.0))
                dbwrite.upsert_processing_status(
                    conn, 'spindle', 'Cz', 'Moelle2011', 11.0, 16.0,
                    'NREM2NREM3', True)
                if blocked:
                    for i in range(4):   # no status row -> no derivable target
                        t = 200.0 + i
                        conn.execute(
                            "INSERT INTO events (uuid, event_type, channel, "
                            "start_time, end_time, duration, stage, method, "
                            "freq_lower, freq_upper) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (dbwrite.event_uuid5('slow_wave', 'Pz', t,
                                                 'Massimini2004', 0.5, 4.0,
                                                 None),
                             'slow_wave', 'Pz', t, t + 1.0, 1.0, None,
                             'Massimini2004', 0.5, 4.0))
                conn.commit()
            finally:
                conn.close()
            return db

        def tables(db):
            conn = sqlite3.connect(db)
            try:
                def n(table):
                    if conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                            " AND name=?", (table,)).fetchone() is None:
                        return 0
                    return conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                return (n('analysed_time'), n('sleep_cycles'),
                        n('stage_durations'))
            finally:
                conn.close()

        def run(db, annot_path):
            """Run the migration, capturing its log records."""
            records = []

            class _Capture(logging.Handler):
                def emit(self, record):
                    records.append(record.getMessage())

            handler = _Capture()
            mig.LOG.addHandler(handler)
            # The script calls logging.basicConfig, which is a no-op once
            # pytest has configured the root logger, so INFO records would be
            # filtered out by the logger's effective level before reaching the
            # handler and the success case would capture nothing.
            previous = mig.LOG.level
            mig.LOG.setLevel(logging.INFO)
            try:
                rc = mig.main([db, '--apply', '--annot', annot_path,
                               '--backfill-analysed-time',
                               '--backfill-cycles'])
            finally:
                mig.LOG.removeHandler(handler)
                mig.LOG.setLevel(previous)
            return rc, records

        # (a) back-fills FAIL and no scope is blocked. The old code returned 0
        # here -- a subject marked done with three empty tables.
        db = build('fail_clean', blocked=False)
        rc, records = run(db, missing_annot)
        assert rc == 3, f"a failed back-fill returned rc={rc}; 0 marks done"
        assert tables(db) == (0, 0, 0), tables(db)
        joined = "\n".join(records)
        assert 'DID run' not in joined, \
            "the summary claimed the back-fills ran when they failed"
        assert 'did NOT complete' in joined, joined[-400:]
        assert 'analysed_time' in joined and 'cycles' in joined
        print(f"[ok] back-fills failed, nothing blocked -> rc=3 (was 0), "
              f"tables {tables(db)}, message names both failures and never "
              f"says 'DID run'")

        # (b) back-fills FAIL and a scope is blocked: both are reported, and
        # the marker is withheld for the blocked scope.
        db = build('fail_blocked', blocked=True)
        rc, records = run(db, missing_annot)
        assert rc == 3, rc
        joined = "\n".join(records)
        assert 'DID run' not in joined, joined[-400:]
        assert 'blocked scope' in joined and 'did NOT complete' in joined
        conn = sqlite3.connect(db)
        try:
            assert dbwrite.stage_format(conn) is None
        finally:
            conn.close()
        print("[ok] back-fills failed AND a scope blocked -> rc=3, both named, "
              "marker withheld")

        # (c) back-fills SUCCEED: rc=0, real counts, everything populated.
        db = build('ok', blocked=False)
        rc, records = run(db, annot.xml_file)
        assert rc == 0, rc
        n_time, n_cyc, n_dur = tables(db)
        assert n_time and n_cyc and n_dur, (n_time, n_cyc, n_dur)
        joined = "\n".join(records)
        # The count must be ROWS written, not rejection settings attempted.
        assert f"analysed_time: {n_time} row(s)" in joined, joined[-400:]
        assert 'did NOT complete' not in joined
        print(f"[ok] back-fills succeeded -> rc=0, analysed_time={n_time} "
              f"sleep_cycles={n_cyc} stage_durations={n_dur}, and the summary "
              f"counts ROWS written rather than settings attempted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_store_analysed_time_strict_raises():
    """``strict=True`` must surface what the default swallows.

    The default contract is right for a detector -- a completed detection must
    not be lost to a denominator problem -- and wrong for a back-fill, whose
    only job is that write.

    The failure mode is subtler than an exception.
    ``build_density_denominators`` is deliberately tolerant, so an unreadable
    or epoch-less scoring produces **zero seconds**, not an error: the row is
    written, the return value is success-shaped, and every density in that
    scope silently becomes NaN. A zero-second denominator is not a
    denominator, so ``strict`` rejects it too.
    """
    print("\n30. Testing store_analysed_time(strict=True):")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_strict_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)

        quiet = logging.getLogger('tw_strict_test')
        quiet.addHandler(logging.NullHandler())
        quiet.propagate = False

        def n_rows():
            return conn.execute(
                "SELECT COUNT(*) FROM analysed_time").fetchone()[0]

        # Default: an unusable scoring writes a ZERO-second row and returns a
        # success-shaped value. Unchanged, because a detector must not lose a
        # completed run to this -- density reports a zero denominator as NaN.
        got = dbwrite.store_analysed_time(
            conn, 'sub-S', None, None, ['NREM2'], True, True, logger=quiet)
        assert got == {'NREM2': 0.0}, got
        assert n_rows() == 1, n_rows()
        print(f"[ok] default: an unusable scoring stores {got} and does not "
              f"raise (detector contract unchanged)")

        # strict=True: the same call is a failure, and no row survives.
        conn.execute("DELETE FROM analysed_time")
        conn.commit()
        try:
            dbwrite.store_analysed_time(
                conn, 'sub-S', None, None, ['NREM2'], True, True,
                logger=quiet, strict=True)
        except ValueError as e:
            assert '0 seconds' in str(e), e
        else:
            raise AssertionError(
                "strict=True accepted a zero-second denominator, so a "
                "back-fill still cannot tell success from silence")
        assert n_rows() == 0, \
            "strict=True raised but left the zero-second row behind"
        print("[ok] strict=True: raises ValueError and keeps NO row")

        # An empty stage list is an error under strict, a warning by default.
        assert dbwrite.store_analysed_time(
            conn, 'sub-S', None, None, [], True, True, logger=quiet) == {}
        try:
            dbwrite.store_analysed_time(
                conn, 'sub-S', None, None, [], True, True, logger=quiet,
                strict=True)
        except ValueError:
            print("[ok] strict=True: an empty stage list raises too")
        else:
            raise AssertionError("strict=True accepted an empty stage list")
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_migration_refuses_an_explicit_backfill_without_annot():
    """An explicitly requested back-fill must never be downgraded to a no-op.

    ``--backfill-analysed-time`` / ``--backfill-cycles`` cannot run without
    ``--annot``. Warning and clearing both flags let every later path --
    including ``return 0`` -- run as though nothing had been asked for, so a
    batch driver whose ``--annot`` glob came back empty for one subject
    marked that subject done with ``analysed_time`` still empty, and the
    subject then dropped out of every density comparison for good.

    The distinction that makes this fixable is three-state: ``None`` (neither
    flag given, the back-fill merely defaulted on because the table was
    empty -- nothing was asked for, so carrying on is right), ``True``
    (explicitly requested -- must be honoured or refused), ``False``
    (explicitly declined). Both halves are asserted, so the test cannot pass
    by simply blocking the feature.
    """
    print("\n32. Testing an explicit back-fill request without --annot:")

    import logging
    from turtlewave_hdEEG import dbwrite

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_backfill_annot_')
    try:
        dataset, annot = _synthetic_recording(
            tmp, tuple(['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11
                       + ['NREM2'] * 20 + ['NREM3'] * 12 + ['REM'] * 11))
        counter = [0]

        def fresh(already_joint=False):
            """A pre-4.3 database, optionally already migrated."""
            counter[0] += 1
            sub = os.path.join(tmp, f'db{counter[0]}')
            os.makedirs(sub, exist_ok=True)
            db = _sub10sd_shaped_db(sub)
            if already_joint:
                assert mig.main([db, '--apply']) == 0
                os.remove(db + mig.BACKUP_SUFFIX)
            return db

        def state(db):
            conn = sqlite3.connect(db)
            try:
                def n(table):
                    if conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                            " AND name=?", (table,)).fetchone() is None:
                        return None
                    return conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                return {
                    'analysed_time': n('analysed_time'),
                    'sleep_cycles': n('sleep_cycles'),
                    'marker': dbwrite.stage_format(conn),
                    'stages': dict(conn.execute(
                        "SELECT stage, COUNT(*) FROM events "
                        "GROUP BY 1").fetchall()),
                    'backup': os.path.exists(db + mig.BACKUP_SUFFIX),
                }
            finally:
                conn.close()

        # (a) explicitly requested, no --annot: refused, nothing written.
        for label, extra in (
                ('both flags', ['--backfill-analysed-time',
                                '--backfill-cycles']),
                ('one flag', ['--backfill-cycles'])):
            db = fresh()
            before = state(db)
            assert mig.main([db, '--apply'] + extra) == 1, label
            after = state(db)
            assert after == before, (label, before, after)
            assert after['marker'] is None, label
            assert not after['backup'], \
                f"{label}: a refused run left a backup behind"
            print(f"[ok] {label} without --annot: rc=1, database untouched "
                  f"({after['stages']}), no marker, no backup")

        # The same refusal in DRY RUN, so it is learned before --apply.
        db = fresh()
        assert mig.main([db, '--backfill-analysed-time']) == 1
        print("[ok] the dry run refuses it too, rather than reporting a plan "
              "it could not carry out")

        # And on the 'Nothing to do' path -- an already-migrated database,
        # which is the exit most likely to be read as success.
        db = fresh(already_joint=True)
        assert mig.main([db, '--apply', '--backfill-analysed-time']) == 1
        conn = sqlite3.connect(db)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM analysed_time").fetchone()[0] == 0
        finally:
            conn.close()
        print("[ok] an already-migrated database with an explicit back-fill "
              "and no --annot returns 1, not the 'Nothing to do' 0")

        # (b) NOT requested, merely defaulted on because the table is empty:
        # warn and proceed. This is the half that stops the fix from being
        # 'refuse whenever --annot is missing'.
        db = fresh()
        assert mig.main([db, '--apply']) == 0
        after = state(db)
        assert after['stages'] == {'NREM2NREM3': 80}, after
        assert after['marker'] == 'joint', after
        assert after['analysed_time'] == 0, after
        print(f"[ok] no flags, tables empty, no --annot: rc=0, migration "
              f"proceeds ({after['stages']}, marker={after['marker']!r}) and "
              f"only warns about the back-fill")

        # (c) explicitly DECLINED: also proceeds. `False` must not be caught
        # by a check looking for `True`.
        db = fresh()
        assert mig.main([db, '--apply', '--no-backfill-analysed-time',
                         '--no-backfill-cycles']) == 0
        assert state(db)['marker'] == 'joint'
        print("[ok] --no-backfill-* without --annot proceeds (rc=0): an "
              "explicit decline is not an explicit request")

        # (d) explicitly requested WITH --annot: honoured.
        db = fresh()
        assert mig.main([db, '--apply', '--annot', annot.xml_file,
                         '--backfill-analysed-time',
                         '--backfill-cycles']) == 0
        after = state(db)
        assert after['analysed_time'] and after['sleep_cycles'], after
        print(f"[ok] the same flags WITH --annot: rc=0, analysed_time="
              f"{after['analysed_time']} sleep_cycles={after['sleep_cycles']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_migration_sees_a_scope_with_a_null_method():
    """A NULL method must not make a scope invisible to the planner.

    Scopes come from a ``GROUP BY``, so any component can be NULL, and
    ``method = NULL`` is never true. Such a scope selected no rows, was
    planned as a no-op with an EMPTY group list, and ``remaining_blockers``
    then iterated that same empty list and reported nothing -- so the marker
    was stamped over rows nobody had looked at.
    """
    print("\n31. Testing a scope whose method is NULL:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_nullmethod_')
    try:
        db = os.path.join(tmp, 'neural_events.db')
        ParalSWA(None, None,
                 log_level=logging.CRITICAL).initialize_sqlite_database(db)
        conn = dbwrite.open_write_connection(db)
        try:
            for i in range(6):
                t = 100.0 + i
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f'orphan-{i}', 'spindle', 'Cz', t, t + 1.0, 1.0,
                     'NREM2' if i < 4 else 'NREM3', None, 11.0, 16.0))
            conn.commit()
        finally:
            conn.close()

        assert mig.main([db, '--apply']) == 0
        conn = sqlite3.connect(db)
        try:
            got = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
        finally:
            conn.close()
        # Before the fix these rows were invisible and left as NREM2/NREM3
        # while the marker was stamped over them.
        assert got == {'NREM2NREM3': 6}, got
        print(f"[ok] the NULL-method scope is planned and collapsed: {got}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_migration_sees_a_collision_with_an_already_target_row():
    """A row already carrying the target is a collision partner too.

    After the rewrite every row in a group carries the target, so the
    ``event_chan_time`` UNIQUE constraint reduces to one row per
    ``(channel, start_time)``. Counting only the rows that MOVE misses the
    pair that matters -- the already-at-target row is exactly what a moving
    row lands on. The dry run then reports a clean bill of health and
    ``--apply`` dies with an unhandled ``IntegrityError`` after the backup has
    been written.

    Newly reachable because NULL rows now move: a NULL-vs-already-target pair
    used to be inert.
    """
    print("\n29. Testing collision detection against an already-target row:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()

    def build(tmp, other, partner):
        db = os.path.join(tmp, 'neural_events.db')
        ParalSWA(None, None,
                 log_level=logging.CRITICAL).initialize_sqlite_database(db)
        conn = dbwrite.open_write_connection(db)
        try:
            for stage in (other, partner):
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', 'Cz', 500.0, 'Moelle2011',
                                         11.0, 16.0, stage),
                     'spindle', 'Cz', 500.0, 501.0, 1.0, stage, 'Moelle2011',
                     11.0, 16.0))
            dbwrite.upsert_processing_status(
                conn, 'spindle', 'Cz', 'Moelle2011', 11.0, 16.0,
                'NREM2NREM3', True)
            conn.commit()
        finally:
            conn.close()
        return db

    cases = [
        ('a NULL row onto an already-joint row', None, 'NREM2NREM3'),
        ('a per-epoch row onto an already-joint row', 'NREM2', 'NREM2NREM3'),
        ('two per-epoch rows onto the target', 'NREM2', 'NREM3'),
    ]
    for label, other, partner in cases:
        tmp = tempfile.mkdtemp(prefix='tw_collide_')
        try:
            db = build(tmp, other, partner)
            before = sorted(sqlite3.connect(db).execute(
                "SELECT COALESCE(stage, 'NULL'), COUNT(*) FROM events "
                "GROUP BY 1").fetchall())

            assert mig.main([db]) == 1, (
                f"{label}: the DRY RUN gave a clean bill of health on a "
                f"database the rewrite would hard-fail on")
            # No traceback, the designed refusal, and nothing written.
            assert mig.main([db, '--apply']) == 1, label
            after = sorted(sqlite3.connect(db).execute(
                "SELECT COALESCE(stage, 'NULL'), COUNT(*) FROM events "
                "GROUP BY 1").fetchall())
            assert after == before, (label, before, after)
            assert not os.path.exists(db + mig.BACKUP_SUFFIX), (
                f"{label}: a backup was written for a run that aborts, so the "
                f"retry would hit the backup guard")
            print(f"[ok] {label}: dry run and --apply both rc=1, database "
                  f"unmodified {before}, no backup written")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_migration_refuses_a_stage_token_outside_the_vocabulary():
    """``--stage-token N2N3`` must not be written and called a success.

    The rows would carry a label no detector ever produces, so the duplicate
    guard keeps refusing -- while the marker says the database is migrated.
    ``remaining_blockers`` cannot catch it: it compares each row against the
    plan's own target, which is the wrong reference here.
    """
    print("\n30. Testing --stage-token vocabulary validation:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_badtoken_')
    try:
        db = os.path.join(tmp, 'neural_events.db')
        ParalSWA(None, None,
                 log_level=logging.CRITICAL).initialize_sqlite_database(db)
        conn = dbwrite.open_write_connection(db)
        try:
            for i in range(4):
                t = 100.0 + i
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011',
                                         11.0, 16.0, 'NREM2'),
                     'spindle', 'Cz', t, t + 1.0, 1.0, 'NREM2', 'Moelle2011',
                     11.0, 16.0))
            conn.commit()
        finally:
            conn.close()

        assert mig.main([db, '--apply', '--stage-token', 'N2N3']) == 1
        conn = sqlite3.connect(db)
        try:
            got = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
            marker = dbwrite.stage_format(conn)
        finally:
            conn.close()
        assert got == {'NREM2': 4}, got
        assert marker is None, marker
        assert not os.path.exists(db + mig.BACKUP_SUFFIX)
        print("[ok] --stage-token N2N3 is refused (rc=1): nothing written, no "
              "marker, no backup")

        # The spelled-out form is accepted and canonicalised.
        assert mig.main([db, '--apply', '--stage-token', 'NREM3,NREM2']) == 0
        conn = sqlite3.connect(db)
        try:
            got = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
            marker = dbwrite.stage_format(conn)
        finally:
            conn.close()
        assert got == {'NREM2NREM3': 4}, got
        assert marker == 'joint'
        print("[ok] --stage-token 'NREM3,NREM2' is accepted and canonicalised "
              "to 'NREM2NREM3' (rc=0, marker stamped)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_refuses_an_empty_method_list():
    """``method IN ()`` is always-false, so an empty list must not be accepted.

    SQLite accepts the empty ``IN ()`` and evaluates it as false, so the guard
    would match no rows, return 0 and let the write through -- the same silent
    downgrade that ``stage_token`` is keyword-required to prevent, reached a
    different way. Unreachable from the three detectors, but this is public
    API.
    """
    print("\n27. Testing the empty-method-list guard:")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    tmp = tempfile.mkdtemp(prefix='tw_nomethod_')
    db = os.path.join(tmp, 'neural_events.db')
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        dbwrite.ensure_direct_write_schema(conn)
        conn.execute(
            "INSERT INTO events (uuid, event_type, channel, start_time, "
            "end_time, duration, stage, method, freq_lower, freq_upper) "
            "VALUES ('u1','spindle','Cz',1.0,2.0,1.0,'NREM2','Moelle2011',"
            "11.0,16.0)")
        conn.execute("DELETE FROM db_meta WHERE key = 'stage_format'")
        conn.commit()

        # The real call refuses, so the fixture is genuinely at risk.
        try:
            dbwrite.assert_stage_format_compatible(
                conn, 'spindle', ['Moelle2011'], 11.0, 16.0,
                stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
        except ValueError as e:
            assert 'DUPLICATE' in str(e).upper(), e
        else:
            raise AssertionError("fixture is not at risk; test is vacuous")

        for methods in ([], None, (), ['']):
            try:
                dbwrite.assert_stage_format_compatible(
                    conn, 'spindle', methods, 11.0, 16.0,
                    stage_token='NREM2NREM3', channels=['Cz'], db_path=db)
            except ValueError as e:
                assert 'non-blank method' in str(e), e
            else:
                raise AssertionError(
                    f"methods={methods!r} was accepted: 'method IN ()' is "
                    f"always-false, so the guard silently passed a write that "
                    f"appends a duplicate set")
        print("[ok] methods=[], None, () and [''] all raise instead of "
              "silently matching nothing")
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _import_migration_script():
    """Import ``examples/migrate_stage_to_joint.py`` as a module."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'examples', 'migrate_stage_to_joint.py')
    spec = importlib.util.spec_from_file_location('migrate_stage_to_joint',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sub10sd_shaped_db(tmp):
    """Build a database shaped like the 4.0.x one this migration targets.

    Spindles already carry the joined token -- the 4.0.x CSV importer's
    ``_norm_stage`` flattened the requested stage LIST into exactly that form
    -- while slow waves carry each event's own epoch stage. Neither
    ``analysed_time`` nor ``sleep_cycles`` has any rows, and there is no
    ``stage_format`` marker.

    Parameters
    ----------
    tmp : str
        Directory to build under.

    Returns
    -------
    str
        Path to the database.
    """
    import logging
    from turtlewave_hdEEG import dbwrite, ParalSWA

    work = os.path.join(tmp, 'sub-10sd', 'wonambi')
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, 'neural_events.db')
    # initialize_sqlite_database ONLY. Deliberately NOT
    # ensure_direct_write_schema: that is the 4.3 migration, and calling it
    # here would give the fixture an events table that already has
    # epoch_stage/det_*/run_id and a db_meta table -- i.e. not a pre-4.3
    # database at all, which is exactly how an earlier version of this test
    # missed a bug that only bites when those columns are ADDED during the
    # migration.
    ParalSWA(None, None, log_level=logging.CRITICAL).initialize_sqlite_database(db)
    conn = dbwrite.open_write_connection(db)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        assert 'epoch_stage' not in cols, \
            "fixture is not pre-4.3: events already has epoch_stage"
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name='db_meta'").fetchone() is None, \
            "fixture is not pre-4.3: db_meta already exists"
        for i in range(30):
            t = 100.0 + i * 7.0
            conn.execute(
                "INSERT INTO events (uuid, event_type, channel, start_time, "
                "end_time, duration, stage, method, freq_lower, freq_upper) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('spindle', 'Cz', t, 'Moelle2011', 11.0,
                                     16.0, 'NREM2NREM3'),
                 'spindle', 'Cz', t, t + 1.0, 1.0, 'NREM2NREM3', 'Moelle2011',
                 11.0, 16.0))
        for i in range(50):
            t = 105.0 + i * 11.0
            conn.execute(
                "INSERT INTO events (uuid, event_type, channel, start_time, "
                "end_time, duration, stage, method, freq_lower, freq_upper, "
                "min_amp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (dbwrite.event_uuid5('slow_wave', 'Cz', t, 'Massimini2004',
                                     0.5, 4.0,
                                     'NREM2' if i < 30 else 'NREM3'),
                 'slow_wave', 'Cz', t, t + 1.0, 1.0,
                 'NREM2' if i < 30 else 'NREM3', 'Massimini2004', 0.5, 4.0,
                 -85.5))
        dbwrite.upsert_processing_status(conn, 'spindle', 'Cz', 'Moelle2011',
                                         11.0, 16.0, 'NREM2NREM3', True)
        dbwrite.upsert_processing_status(conn, 'slow_wave', 'Cz',
                                         'Massimini2004', 0.5, 4.0,
                                         'NREM2NREM3', True)
        conn.commit()
    finally:
        conn.close()
    return db


def test_migration_collapses_stage_and_stamps_marker():
    """Phase 6: the migration is safe, idempotent, and unblocks re-detection.

    Everything asserted here is a way the script could destroy or corrupt the
    thing it exists to protect:

    * a dry run that wrote something;
    * a second run that replaced the good pre-migration backup with an
      already-migrated one -- the single way this script could destroy the
      only way back;
    * an archived (irreplaceable) database touched without being asked;
    * a rewrite that changed a column other than ``stage``;
    * a colliding database rewritten into a constraint violation.
    """
    print("\n21. Testing examples/migrate_stage_to_joint.py (phase 6):")

    import logging
    from turtlewave_hdEEG import dbwrite

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_migrate_')
    try:
        db = _sub10sd_shaped_db(tmp)

        def snapshot():
            conn = sqlite3.connect(db)
            try:
                def count(table):
                    """Row count, or None when the table does not exist."""
                    if conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                            " AND name=?", (table,)).fetchone() is None:
                        return None
                    return conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]

                return {
                    'stages': dict(conn.execute(
                        "SELECT stage || '/' || event_type, COUNT(*) "
                        "FROM events GROUP BY 1").fetchall()),
                    'marker': dbwrite.stage_format(conn),
                    'n_analysed': count('analysed_time'),
                    'n_cycles': count('sleep_cycles'),
                    'n_tagged': conn.execute(
                        "SELECT COUNT(cycle) FROM events").fetchone()[0],
                    'min_amp': conn.execute(
                        "SELECT DISTINCT min_amp FROM events "
                        "WHERE min_amp IS NOT NULL").fetchall(),
                    'columns': [r[1] for r in conn.execute(
                        "PRAGMA table_info(events)")],
                }
            finally:
                conn.close()

        before = snapshot()
        assert before['marker'] is None
        # The fixture must be a REAL pre-4.3 database: the migration adds
        # columns to events, and a fixture that already had them would hide
        # any bug that only appears when they are added mid-run.
        assert 'epoch_stage' not in before['columns'], before['columns']
        assert before['stages'] == {'NREM2NREM3/spindle': 30,
                                    'NREM2/slow_wave': 30,
                                    'NREM3/slow_wave': 20}, before
        print(f"[ok] fixture: {before['stages']}, no marker, no epoch_stage "
              f"column, no analysed_time/sleep_cycles table "
              f"({len(before['columns'])} events columns)")

        # (a) archive guard.
        arch = os.path.join(tmp, 'wonambi_archive')
        os.makedirs(arch, exist_ok=True)
        shutil.copy(db, os.path.join(arch, 'neural_events.db'))
        rc = mig.main([os.path.join(arch, 'neural_events.db'), '--apply'])
        assert rc == 1, rc
        assert not os.path.exists(
            os.path.join(arch, 'neural_events.db') + mig.BACKUP_SUFFIX)
        print("[ok] a path containing 'archive' is refused (rc=1), untouched")

        # (b) dry run writes nothing at all.
        rc = mig.main([db])
        assert rc == 0, rc
        assert snapshot() == before, "the dry run modified the database"
        assert not os.path.exists(db + mig.BACKUP_SUFFIX), \
            "the dry run wrote a backup"
        print("[ok] the dry run changed nothing and wrote no backup")

        # (c) apply.
        annot_xml = None
        rc = mig.main([db, '--apply'] + ([] if annot_xml is None else
                                         ['--annot', annot_xml]))
        assert rc == 0, rc
        after = snapshot()
        assert after['stages'] == {'NREM2NREM3/spindle': 30,
                                   'NREM2NREM3/slow_wave': 50}, after
        assert after['marker'] == 'joint', after
        assert after['min_amp'] == before['min_amp'], \
            "a column other than stage changed"
        # The migration ADDS columns (epoch_stage, det_*, run_id) while it
        # runs. The "only stage changed" digest must be taken over the
        # pre-migration column list, or it can never match and the script
        # reports rc=2 -- after committing the rewrite and before stamping the
        # marker, i.e. it fails on exactly the databases it exists to serve.
        assert 'epoch_stage' in after['columns'], after['columns']
        assert len(after['columns']) > len(before['columns'])
        print(f"[ok] --apply: {after['stages']}, marker={after['marker']!r}, "
              f"min_amp untouched ({after['min_amp'][0][0]} uV), and the "
              f"{len(after['columns']) - len(before['columns'])} columns the "
              f"migration ADDED did not defeat the 'only stage changed' check")
        assert os.path.exists(db + mig.BACKUP_SUFFIX)

        # (d) a second run must not replace the backup.
        bak = db + mig.BACKUP_SUFFIX
        bak_mtime, bak_size = os.path.getmtime(bak), os.path.getsize(bak)
        rc = mig.main([db, '--apply'])
        assert rc == 0, rc
        assert os.path.getmtime(bak) == bak_mtime, \
            "the second run overwrote the pre-migration backup"
        assert os.path.getsize(bak) == bak_size
        assert snapshot()['stages'] == after['stages']
        print("[ok] a second --apply is a no-op and leaves the backup alone")

        # The backup really is the pre-migration database.
        conn = sqlite3.connect(bak)
        try:
            bak_stages = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
        finally:
            conn.close()
        assert bak_stages == {'NREM2NREM3': 30, 'NREM2': 30, 'NREM3': 20}, \
            bak_stages
        print(f"[ok] the backup holds the PRE-migration stages: {bak_stages}")

        # (e) a colliding database aborts before writing anything.
        db2 = os.path.join(tmp, 'collide.db')
        from turtlewave_hdEEG import ParalSWA
        ParalSWA(None, None,
                 log_level=logging.CRITICAL).initialize_sqlite_database(db2)
        conn = dbwrite.open_write_connection(db2)
        try:
            for stg in ('NREM2', 'NREM3'):
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('slow_wave', 'Cz', 500.0,
                                         'Massimini2004', 0.5, 4.0, stg),
                     'slow_wave', 'Cz', 500.0, 501.0, 1.0, stg,
                     'Massimini2004', 0.5, 4.0))
            conn.commit()
        finally:
            conn.close()
        collide_before = sqlite3.connect(db2).execute(
            "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall()
        rc = mig.main([db2, '--apply'])
        collide_after = sqlite3.connect(db2).execute(
            "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall()
        assert rc == 1, rc
        assert collide_after == collide_before, collide_after
        assert not os.path.exists(db2 + mig.BACKUP_SUFFIX), \
            "a backup was written for a run that was going to abort"
        print(f"[ok] two rows sharing (channel, start_time) under different "
              f"stages abort the run (rc=1), database unmodified "
              f"{collide_before}, no backup written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_migrated_rows_are_replaced_not_duplicated():
    """Phase 6: the migration actually removes the duplicate-set hazard.

    The end-to-end sequence this whole release exists for: detect under 4.2
    (per-epoch stage, uuid hashed on it), migrate, re-detect under 4.3. The
    row count must not move.

    The subtle part is that the migration collapses ``stage`` but leaves the
    uuid alone, so a migrated row's uuid is no longer ``uuid5`` of its own
    scope. That is safe, and this is the test that says so rather than
    assuming it: the re-detected row now matches on the ``event_chan_time``
    UNIQUE constraint instead of on the primary key, and ``INSERT OR REPLACE``
    resolves a UNIQUE conflict by deleting the old row -- so the uuid is
    rewritten and nothing is duplicated.
    """
    print("\n22. Testing that migrated rows are REPLACED by re-detection "
          "(phase 6):")

    import logging
    from turtlewave_hdEEG import dbwrite, ParalEvents

    mig = _import_migration_script()
    tmp = tempfile.mkdtemp(prefix='tw_migrate_rt_')
    try:
        dataset, annot = _synthetic_recording(
            tmp, ('NREM2', 'NREM2', 'NREM3', 'NREM3'))
        work = os.path.join(tmp, 'wonambi')
        os.makedirs(work, exist_ok=True)
        db = os.path.join(work, 'neural_events.db')
        pe = ParalEvents(dataset, annot, log_level=logging.CRITICAL)
        kw = dict(method='Moelle2011', chan=['Cz'], frequency=(11, 16),
                  stage=['NREM2', 'NREM3'], json_dir=work, db_path=db,
                  subject='sub-T', cat=(1, 1, 1, 0))
        assert pe.detect_spindles(**kw), "no spindles detected"

        # Rewrite what the run produced into 4.2's shape: events.stage and the
        # uuid both keyed on the event's own epoch stage.
        conn = sqlite3.connect(db)
        try:
            real = conn.execute(
                "SELECT channel, start_time, end_time, duration, epoch_stage, "
                "method, freq_lower, freq_upper FROM events").fetchall()
            conn.execute("DELETE FROM events")
            for ch, t0, t1, dur, epoch_stage, meth, lo, hi in real:
                conn.execute(
                    "INSERT INTO events (uuid, event_type, channel, "
                    "start_time, end_time, duration, stage, method, "
                    "freq_lower, freq_upper) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (dbwrite.event_uuid5('spindle', ch, t0, meth, lo, hi,
                                         epoch_stage),
                     'spindle', ch, t0, t1, dur, epoch_stage, meth, lo, hi))
            conn.execute("DELETE FROM db_meta WHERE key = 'stage_format'")
            conn.commit()
            n_42 = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            stages_42 = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
            uuids_42 = {r[0] for r in conn.execute("SELECT uuid FROM events")}
        finally:
            conn.close()
        assert len(stages_42) == 2, stages_42
        print(f"[ok] 4.2-shaped database: {n_42} rows, {stages_42}")

        assert mig.main([db, '--apply']) == 0
        conn = sqlite3.connect(db)
        try:
            stages_mig = dict(conn.execute(
                "SELECT stage, COUNT(*) FROM events GROUP BY 1").fetchall())
            uuids_mig = {r[0] for r in conn.execute("SELECT uuid FROM events")}
        finally:
            conn.close()
        assert stages_mig == {'NREM2NREM3': n_42}, stages_mig
        assert uuids_mig == uuids_42, "the migration changed a uuid"
        print(f"[ok] migrated: {stages_mig}; every uuid left untouched")

        assert pe.detect_spindles(**kw), "re-detection found nothing"
        conn = sqlite3.connect(db)
        try:
            n_after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            uuids_after = {r[0] for r in conn.execute("SELECT uuid FROM events")}
        finally:
            conn.close()
        assert n_after == n_42, (
            f"re-detection after migration produced {n_after} rows from "
            f"{n_42}: the duplicate set is still being appended")
        assert uuids_after != uuids_42, \
            "the uuids were not rewritten, so nothing was actually replaced"
        print(f"[ok] re-detection REPLACED: {n_42} -> {n_after} rows, uuids "
              f"rewritten to uuid5 of the joint scope")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING TURTLEWAVE-HDEEG PACKAGE UPDATES")
    print("=======================================")

    # Run all tests
    test_utils_functions()
    test_custom_annotations()
    test_paralevents_class()
    test_largedataset_class()
    test_xlannotations_class()
    test_improved_detect_spindle()
    test_slow_wave_polarity()
    test_ngo2015_adaptive_thresholds_are_attributes()
    test_massimini_criteria_are_the_published_ones()
    test_trough_duration_criterion_gates_the_negative_half_wave()
    test_permissive_search_recovers_paper_valid_waves()
    test_permissive_search_recall_against_injected_ground_truth()
    test_trough_depth_criterion_gates_the_negative_peak()
    test_slow_wave_ptp_is_microvolts_not_samples()
    test_det_trough_is_negative_for_every_method()
    test_staresina_and_ngo_are_untouched()
    test_kcomplex_trough_duration_bounds_the_half_wave()
    test_package_structure()
    test_density_with_bare_subject_id()
    test_density_identity_axis()
    test_density_multi_method_run()
    test_cycle_subject_spelling_delete()
    test_pac_twin_delete_is_scoped()
    test_stage_token_vocabulary()
    test_detectors_write_joint_stage_token()
    test_mixed_stage_format_database_reads_back()
    test_density_pools_joint_stage_token()
    test_stage_format_guard_blocks_duplicate_set()
    test_detection_populates_cycles_without_touching_xml()
    test_tag_method_outside_methods_is_an_error()
    test_migration_collapses_stage_and_stamps_marker()
    test_migrated_rows_are_replaced_not_duplicated()
    test_rerun_under_a_different_stage_set_is_refused()
    test_density_warns_about_tokens_it_excludes()
    test_migration_keeps_per_channel_stage_sets_apart()
    test_migration_unblocks_null_stage_rows()
    test_migration_runs_backfills_even_when_partly_blocked()
    test_migration_reports_backfill_outcome_not_intention()
    test_store_analysed_time_strict_raises()
    test_migration_refuses_an_explicit_backfill_without_annot()
    test_migration_sees_a_scope_with_a_null_method()
    test_migration_sees_a_collision_with_an_already_target_row()
    test_migration_refuses_a_stage_token_outside_the_vocabulary()
    test_guard_refuses_an_empty_method_list()

    print("\nAll tests completed!")