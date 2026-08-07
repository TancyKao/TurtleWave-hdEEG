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
            # AASM thresholds explicitly: detect_slow_waves defaults to
            # Massimini2004's -80 uV / 140 uV, which this synthetic wave (a
            # sharp -110 uV negative half against a broad +60 uV positive
            # one) does not reach on peak-to-peak.
            n['slow_wave'] = len(ParalSWA(
                dataset, annot, log_level=logging.CRITICAL).detect_slow_waves(
                    method='AASM/Massimini2004', frequency=(0.5, 4),
                    neg_peak_thresh=-37.0, p2p_thresh=75.0, **kw))
            # trough_duration widened from the AASM KC window (0.25-1.0 s):
            # the negative half-wave of this synthetic oscillation measures
            # longer than that once band-passed, and what is under test here
            # is the stage bookkeeping, not the KC morphology criteria.
            n['k_complex'] = len(ParalKC(
                dataset, annot, log_level=logging.CRITICAL).detect_kcomplexes(
                    method='AASM/Massimini2004', frequency=(0.5, 4),
                    trough_duration=(0.3, 1.5), **kw))
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