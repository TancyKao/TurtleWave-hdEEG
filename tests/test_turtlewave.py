## test_turtlewave_updates.py

import os
import sys
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
    
    print("\nAll tests completed!")