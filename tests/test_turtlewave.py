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
    test_package_structure()
    test_density_with_bare_subject_id()
    test_density_identity_axis()
    test_density_multi_method_run()

    print("\nAll tests completed!")