#!/usr/bin/env python3
"""
Waveform Background Loader
Loads EEG waveforms in background thread with caching
"""

from PyQt5.QtCore import QThread, pyqtSignal, QMutex
import numpy as np
from collections import OrderedDict
import time


class WaveformCache:
    """LRU cache for waveform data"""
    
    def __init__(self, max_size_mb=500):
        self.cache = OrderedDict()
        self.max_size = max_size_mb
        self.current_size_mb = 0
        self.lock = QMutex()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get waveform from cache"""
        self.lock.lock()
        try:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
        finally:
            self.lock.unlock()
    
    def put(self, key, value):
        """Put waveform in cache"""
        self.lock.lock()
        try:
            # Estimate size (rough approximation)
            if hasattr(value, 'data'):
                size_mb = value.data[0].nbytes / (1024 * 1024)
            else:
                size_mb = 1  # Default 1MB
            
            # Remove old entries if needed
            while self.current_size_mb + size_mb > self.max_size and self.cache:
                oldest_key, oldest_value = self.cache.popitem(last=False)
                if hasattr(oldest_value, 'data'):
                    self.current_size_mb -= oldest_value.data[0].nbytes / (1024 * 1024)
            
            # Add new entry
            self.cache[key] = value
            self.current_size_mb += size_mb
            
        finally:
            self.lock.unlock()
    
    def clear(self):
        """Clear cache"""
        self.lock.lock()
        try:
            self.cache.clear()
            self.current_size_mb = 0
            self.hits = 0
            self.misses = 0
        finally:
            self.lock.unlock()
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size_mb': self.current_size_mb,
            'max_size_mb': self.max_size,
            'num_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class WaveformBackgroundLoader(QThread):
    """Background thread for loading waveforms"""
    
    waveform_loaded = pyqtSignal(str, object)  # (event_uuid, waveform_data)
    progress_update = pyqtSignal(int, int)  # (current, total)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.cache = WaveformCache(max_size_mb=500)
        self.running = True
        self.load_queue = []
        self.queue_lock = QMutex()
    
    def run(self):
        """Main thread loop"""
        while self.running:
            # Get next event to load
            event_to_load = None
            
            self.queue_lock.lock()
            if self.load_queue:
                event_to_load = self.load_queue.pop(0)
            self.queue_lock.unlock()
            
            if event_to_load is not None:
                self.load_waveform(event_to_load)
            else:
                # Sleep briefly if queue is empty
                time.sleep(0.1)
    
    def load_waveform(self, event_row):
        """Load waveform for an event - ONLY loads selected channels for performance"""
        try:
            event_uuid = event_row['uuid']
            
            # Check cache first
            cached = self.cache.get(event_uuid)
            if cached is not None:
                self.waveform_loaded.emit(event_uuid, cached)
                return
            
            # Load from EEG file
            if self.parent_gui.eeg_data is None:
                return
            
            # Get selected channels from parent GUI
            selected_channels = getattr(self.parent_gui, 'selected_channels', None)
            if not selected_channels:
                selected_channels = ['E112', 'E118', 'Cz']  # Default fallback
            
            start_time = event_row['start_time']
            end_time = event_row['end_time']
            
            # Fixed 30-second window centered on event (same as plot)
            event_center = (start_time + end_time) / 2
            window_start = event_center - 15  # 15 seconds before
            window_end = event_center + 15    # 15 seconds after
            
            # Load data
            try:
                # Try TurtleWave LargeDataset method
                if hasattr(self.parent_gui.eeg_data, 'read_data'):
                    waveform = self.parent_gui.eeg_data.read_data(
                        chan=selected_channels,  # ONLY load selected channels
                        begtime=window_start,
                        endtime=window_end
                    )
                    
                # Try MNE method
                elif hasattr(self.parent_gui.eeg_data, 'get_data'):
                    sfreq = self.parent_gui.eeg_data.info['sfreq']
                    start_sample = int(window_start * sfreq)
                    stop_sample = int(window_end * sfreq)
                    
                    # Get channel indices
                    ch_indices = [self.parent_gui.eeg_data.ch_names.index(ch)
                                 for ch in selected_channels
                                 if ch in self.parent_gui.eeg_data.ch_names]
                    
                    if not ch_indices:
                        return
                    
                    data = self.parent_gui.eeg_data.get_data(
                        picks=ch_indices,  # ONLY load selected channels
                        start=start_sample,
                        stop=stop_sample
                    )
                    
                    # Wrap in TurtleWave-like structure
                    class WaveformData:
                        def __init__(self, data, channels, sfreq):
                            self.data = [data]
                            self.axis = {
                                'chan': [np.array(channels)],
                                's_freq': sfreq
                            }
                    
                    waveform = WaveformData(
                        data,
                        [self.parent_gui.eeg_data.ch_names[i] for i in ch_indices],
                        sfreq
                    )
                else:
                    return
                
                # Cache it
                self.cache.put(event_uuid, waveform)
                
                # Emit signal
                self.waveform_loaded.emit(event_uuid, waveform)
                
            except Exception as e:
                pass  # Silently fail to avoid blocking
                
        except Exception as e:
            pass  # Silently fail to avoid blocking
    
    def queue_event(self, event_row):
        """Add event to load queue"""
        self.queue_lock.lock()
        self.load_queue.append(event_row)
        self.queue_lock.unlock()
    
    def queue_events_around(self, events_df, current_index, num_before=5, num_after=5):
        """Queue events around current index for prefetching"""
        if events_df.empty:
            return
        
        start_idx = max(0, current_index - num_before)
        end_idx = min(len(events_df), current_index + num_after + 1)
        
        self.queue_lock.lock()
        # Clear queue and add new events
        self.load_queue.clear()
        
        # Prioritize current event
        if 0 <= current_index < len(events_df):
            # Convert Series to dict to avoid pandas ambiguity
            event_dict = events_df.iloc[current_index].to_dict()
            self.load_queue.append(event_dict)
        
        # Add surrounding events
        for i in range(start_idx, end_idx):
            if i != current_index:
                event_dict = events_df.iloc[i].to_dict()
                self.load_queue.append(event_dict)
        
        self.queue_lock.unlock()
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return self.cache.get_stats()
