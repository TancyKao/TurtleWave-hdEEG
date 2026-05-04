#!/usr/bin/env python3
"""
Test script for PyQtGraph conversion of EEG Review GUI
Tests performance and functionality of the new PyQtGraph-based widgets
"""

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Simple mock classes for testing
class MockWaveformData:
    """Mock waveform data for testing"""
    def __init__(self, n_channels=3, n_samples=15000, sampling_rate=500):
        self.data = [np.random.randn(n_channels, n_samples) * 50]  # µV scale
        self.axis = {
            'chan': [np.array(['E112', 'E118', 'Cz'])],
            's_freq': sampling_rate
        }

class MockAnnotations:
    """Mock annotations for testing"""
    def __init__(self):
        self.wonb_annot = type('obj', (object,), {
            'start_time': '2024-01-01T22:00:00'
        })()
    
    def get_stages(self):
        # 8 hours of sleep stages (30-second epochs)
        stages = []
        for i in range(960):  # 8 hours * 120 epochs/hour
            if i < 60:
                stages.append('Wake')
            elif i < 120:
                stages.append('NREM1')
            elif i < 480:
                stages.append('NREM2')
            elif i < 600:
                stages.append('NREM3')
            elif i < 720:
                stages.append('NREM2')
            elif i < 840:
                stages.append('REM')
            else:
                stages.append('Wake')
        return stages

def test_timeline_widget():
    """Test TimelineWidget performance"""
    print("Testing TimelineWidget...")
    
    from frontend.eeg_review_gui import TimelineWidget
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create widget
    timeline = TimelineWidget()
    timeline.setWindowTitle("Timeline Widget Test")
    timeline.resize(1200, 200)
    
    # Create mock data
    events_df = pd.DataFrame({
        'start_time': np.random.uniform(0, 28800, 1000),  # 1000 events over 8 hours
        'end_time': np.random.uniform(0, 28800, 1000) + 1,
        'event_type': np.random.choice(['spindle', 'slow_wave'], 1000),
        'channel': np.random.choice(['E112', 'E118', 'Cz'], 1000)
    })
    
    annotations = MockAnnotations()
    
    # Measure plotting time
    import time
    start = time.time()
    timeline.plot_timeline(events_df, current_index=0, annotations=annotations)
    elapsed = time.time() - start
    
    print(f"  Timeline plotting time: {elapsed:.3f}s for {len(events_df)} events")
    print(f"  ✓ Timeline widget created successfully")
    
    timeline.show()
    
    # Close after 2 seconds
    QTimer.singleShot(2000, app.quit)
    app.exec_()
    
    return True

def test_eeg_detail_widget():
    """Test EEGDetailWidget performance"""
    print("\nTesting EEGDetailWidget...")
    
    from frontend.eeg_review_gui import EEGDetailWidget
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create widget
    eeg_widget = EEGDetailWidget()
    eeg_widget.setWindowTitle("EEG Detail Widget Test")
    eeg_widget.resize(1200, 600)
    
    # Create mock event
    event_row = pd.Series({
        'start_time': 1000.0,
        'end_time': 1001.5,
        'duration': 1.5,
        'channel': 'E112',
        'event_type': 'spindle'
    })
    
    # Create mock waveform data
    waveform_data = MockWaveformData(n_channels=3, n_samples=15000)
    channels = ['E112', 'E118', 'Cz']
    
    # Measure plotting time
    import time
    start = time.time()
    eeg_widget.plot_event(event_row, waveform_data, channels)
    elapsed = time.time() - start
    
    print(f"  EEG plotting time: {elapsed:.3f}s for 3 channels, 15000 samples")
    print(f"  ✓ EEG detail widget created successfully")
    
    eeg_widget.show()
    
    # Close after 2 seconds
    QTimer.singleShot(2000, app.quit)
    app.exec_()
    
    return True

def test_multi_channel_performance():
    """Test performance with many channels"""
    print("\nTesting multi-channel performance...")
    
    from frontend.eeg_review_gui import EEGDetailWidget
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create widget
    eeg_widget = EEGDetailWidget()
    eeg_widget.setWindowTitle("Multi-Channel Performance Test")
    eeg_widget.resize(1200, 800)
    
    # Test with increasing number of channels
    for n_channels in [3, 10, 20, 50]:
        # Create mock waveform data
        channel_names = [f'E{i}' for i in range(1, n_channels + 1)]
        waveform_data = MockWaveformData(n_channels=n_channels, n_samples=15000)
        waveform_data.axis['chan'] = [np.array(channel_names)]
        
        event_row = pd.Series({
            'start_time': 1000.0,
            'end_time': 1001.5,
            'duration': 1.5,
            'channel': channel_names[0],
            'event_type': 'spindle'
        })
        
        # Measure plotting time
        import time
        start = time.time()
        eeg_widget.plot_event(event_row, waveform_data, channel_names)
        elapsed = time.time() - start
        
        print(f"  {n_channels} channels: {elapsed:.3f}s")
    
    print(f"  ✓ Multi-channel performance test completed")
    
    eeg_widget.show()
    
    # Close after 2 seconds
    QTimer.singleShot(2000, app.quit)
    app.exec_()
    
    return True

def test_rapid_navigation():
    """Test rapid event navigation performance"""
    print("\nTesting rapid navigation...")
    
    from frontend.eeg_review_gui import EEGDetailWidget
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create widget
    eeg_widget = EEGDetailWidget()
    eeg_widget.setWindowTitle("Rapid Navigation Test")
    eeg_widget.resize(1200, 600)
    
    # Create mock waveform data
    waveform_data = MockWaveformData(n_channels=3, n_samples=15000)
    channels = ['E112', 'E118', 'Cz']
    
    # Simulate rapid navigation through 50 events
    import time
    start = time.time()
    
    for i in range(50):
        event_row = pd.Series({
            'start_time': 1000.0 + i * 10,
            'end_time': 1001.5 + i * 10,
            'duration': 1.5,
            'channel': 'E112',
            'event_type': 'spindle'
        })
        eeg_widget.plot_event(event_row, waveform_data, channels)
    
    elapsed = time.time() - start
    avg_time = elapsed / 50
    
    print(f"  50 rapid updates: {elapsed:.3f}s total, {avg_time:.3f}s average")
    print(f"  ✓ Rapid navigation test completed")
    
    eeg_widget.show()
    
    # Close after 2 seconds
    QTimer.singleShot(2000, app.quit)
    app.exec_()
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("PyQtGraph Conversion Performance Tests")
    print("=" * 60)
    
    try:
        # Test timeline widget
        test_timeline_widget()
        
        # Test EEG detail widget
        test_eeg_detail_widget()
        
        # Test multi-channel performance
        test_multi_channel_performance()
        
        # Test rapid navigation
        test_rapid_navigation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
