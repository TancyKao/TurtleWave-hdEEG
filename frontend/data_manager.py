#!/usr/bin/env python3
"""
TurtleWave Data Manager - Core Intelligence Layer
Provides three-level caching, intelligent prefetching, and optimized data access
"""

import sqlite3
import numpy as np
import pandas as pd
from collections import OrderedDict
from threading import Thread, Lock
from queue import Queue
import time
from typing import Dict, List, Optional, Tuple, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventMetadata:
    """Lightweight event metadata for fast filtering and display"""
    
    __slots__ = ['uuid', 'channel', 'start_time', 'end_time', 'duration',
                 'event_type', 'review_status', 'stage']
    
    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {key: getattr(self, key) for key in self.__slots__}


class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key, value):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class MetadataCache:
    """Level 1: Metadata cache for fast event filtering"""
    
    def __init__(self):
        self.events = []  # List of EventMetadata objects
        self.channel_index = {}  # {channel: [event_indices]}
        self.stage_index = {}  # {stage: [event_indices]}
        self.review_index = {}  # {review_status: [event_indices]}
        self.lock = Lock()
    
    def load_from_database(self, db_conn, channels=None, event_types=None, limit=10000):
        """Load metadata from database"""
        with self.lock:
            query = """
                SELECT uuid, channel, start_time, end_time, duration,
                       event_type,
                       CASE WHEN reviewed IS NULL THEN 0 ELSE reviewed END as review_status,
                       stage
                FROM events
                WHERE 1=1
            """
            params = []
            
            if channels:
                placeholders = ','.join(['?' for _ in channels])
                query += f" AND channel IN ({placeholders})"
                params.extend(channels)
            
            if event_types:
                placeholders = ','.join(['?' for _ in event_types])
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_types)
            
            query += " ORDER BY channel, start_time LIMIT ?"
            params.append(limit)
            
            cursor = db_conn.execute(query, params)
            
            # Clear existing data
            self.events = []
            self.channel_index = {}
            self.stage_index = {}
            self.review_index = {}
            
            # Load events
            for row in cursor.fetchall():
                event = EventMetadata(
                    uuid=row[0],
                    channel=row[1],
                    start_time=row[2],
                    end_time=row[3],
                    duration=row[4],
                    event_type=row[5],
                    review_status='reviewed' if row[6] else 'pending',
                    stage=row[7]
                )
                
                idx = len(self.events)
                self.events.append(event)
                
                # Build indexes
                if event.channel not in self.channel_index:
                    self.channel_index[event.channel] = []
                self.channel_index[event.channel].append(idx)
                
                if event.stage:
                    if event.stage not in self.stage_index:
                        self.stage_index[event.stage] = []
                    self.stage_index[event.stage].append(idx)
                
                if event.review_status not in self.review_index:
                    self.review_index[event.review_status] = []
                self.review_index[event.review_status].append(idx)
            
            logger.info(f"Loaded {len(self.events)} events into metadata cache")
    
    def filter_events(self, channels=None, stages=None, review_status=None):
        """Fast filtering using indexes"""
        with self.lock:
            # Start with all events
            result_indices = set(range(len(self.events)))
            
            # Filter by channel
            if channels:
                channel_indices = set()
                for ch in channels:
                    if ch in self.channel_index:
                        channel_indices.update(self.channel_index[ch])
                result_indices &= channel_indices
            
            # Filter by stage
            if stages:
                stage_indices = set()
                for stage in stages:
                    if stage in self.stage_index:
                        stage_indices.update(self.stage_index[stage])
                result_indices &= stage_indices
            
            # Filter by review status
            if review_status:
                if isinstance(review_status, list):
                    status_indices = set()
                    for status in review_status:
                        if status in self.review_index:
                            status_indices.update(self.review_index[status])
                    result_indices &= status_indices
                else:
                    if review_status in self.review_index:
                        result_indices &= set(self.review_index[review_status])
            
            # Return filtered events
            return [self.events[idx] for idx in sorted(result_indices)]
    
    def get_event_by_index(self, index):
        """Get event by index"""
        with self.lock:
            if 0 <= index < len(self.events):
                return self.events[index]
            return None
    
    def get_channel_stats(self):
        """Get statistics by channel"""
        with self.lock:
            stats = {}
            for channel, indices in self.channel_index.items():
                stats[channel] = {
                    'total': len(indices),
                    'reviewed': sum(1 for idx in indices if self.events[idx].review_status == 'reviewed'),
                    'pending': sum(1 for idx in indices if self.events[idx].review_status == 'pending')
                }
            return stats


class WaveformCache:
    """Level 2: Waveform cache with LRU eviction"""
    
    def __init__(self, max_size_mb=500):
        # Estimate: 1 event waveform ~= 256 channels × 256 samples × 4 bytes = 256KB
        # 500MB / 256KB = ~2000 events
        max_events = int(max_size_mb * 1024 / 256)
        self.cache = LRUCache(max_size=max_events)
        self.lock = Lock()
    
    def get_waveform(self, event_uuid):
        """Get waveform from cache"""
        return self.cache.get(event_uuid)
    
    def put_waveform(self, event_uuid, waveform_data):
        """Put waveform in cache"""
        self.cache.put(event_uuid, waveform_data)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self):
        """Get cache statistics"""
        return self.cache.get_stats()


class BackgroundPrefetcher:
    """Background thread for prefetching data"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.queue = Queue()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start prefetcher thread"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._worker, daemon=True)
            self.thread.start()
            logger.info("Background prefetcher started")
    
    def stop(self):
        """Stop prefetcher thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def prefetch_around_event(self, event_index, num_before=5, num_after=5):
        """Queue prefetch request for events around current event"""
        self.queue.put({
            'type': 'around_event',
            'event_index': event_index,
            'num_before': num_before,
            'num_after': num_after
        })
    
    def _worker(self):
        """Worker thread that processes prefetch requests"""
        while self.running:
            try:
                # Get request with timeout
                request = self.queue.get(timeout=0.1)
                
                if request['type'] == 'around_event':
                    self._prefetch_around_event(
                        request['event_index'],
                        request['num_before'],
                        request['num_after']
                    )
                
                self.queue.task_done()
                
            except:
                continue  # Timeout or empty queue
    
    def _prefetch_around_event(self, event_index, num_before, num_after):
        """Prefetch events around the given index"""
        # Get event list from data manager
        events = self.data_manager.get_current_events()
        
        if not events or event_index >= len(events):
            return
        
        # Calculate range
        start_idx = max(0, event_index - num_before)
        end_idx = min(len(events), event_index + num_after + 1)
        
        # Prefetch events
        for idx in range(start_idx, end_idx):
            if idx == event_index:
                continue  # Skip current event (already loaded)
            
            event = events[idx]
            
            # Check if already in cache
            if self.data_manager.waveform_cache.get_waveform(event.uuid):
                continue
            
            # Load waveform
            try:
                waveform = self.data_manager._load_event_waveform(event)
                if waveform is not None:
                    self.data_manager.waveform_cache.put_waveform(event.uuid, waveform)
                    logger.debug(f"Prefetched event {event.uuid}")
            except Exception as e:
                logger.warning(f"Failed to prefetch event {event.uuid}: {e}")


class DataManager:
    """Core intelligence layer for data management"""
    
    def __init__(self, db_path, eeg_data=None):
        self.db_path = db_path
        self.eeg_data = eeg_data
        
        # Three-level caching
        self.metadata_cache = MetadataCache()
        self.waveform_cache = WaveformCache(max_size_mb=500)
        
        # Background prefetcher
        self.prefetcher = BackgroundPrefetcher(self)
        
        # Current session state
        self.current_events = []  # Filtered event list
        self.current_filters = {}
        
        # Database connection
        self.db_conn = None
        
        logger.info("DataManager initialized")
    
    def connect(self):
        """Connect to database"""
        if not self.db_conn:
            self.db_conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
            logger.info("Disconnected from database")
    
    def set_eeg_data(self, eeg_data):
        """Set EEG data source"""
        self.eeg_data = eeg_data
        logger.info("EEG data source updated")
    
    def load_initial_data(self, channels=None, event_types=None, limit=10000):
        """Load initial metadata into cache"""
        self.connect()
        self.metadata_cache.load_from_database(self.db_conn, channels, event_types, limit)
        
        # Start background prefetcher
        self.prefetcher.start()
        
        logger.info(f"Initial data loaded for channels: {channels}")
    
    def apply_filters(self, channels=None, stages=None, review_status=None):
        """Apply filters and update current events"""
        # Store current filters
        self.current_filters = {
            'channels': channels,
            'stages': stages,
            'review_status': review_status
        }
        
        # Filter events using metadata cache
        self.current_events = self.metadata_cache.filter_events(
            channels=channels,
            stages=stages,
            review_status=review_status
        )
        
        logger.info(f"Filters applied: {len(self.current_events)} events match")
        
        return self.current_events
    
    def get_current_events(self):
        """Get current filtered events"""
        return self.current_events
    
    def get_event_by_index(self, index):
        """Get event by index in current filtered list"""
        if 0 <= index < len(self.current_events):
            return self.current_events[index]
        return None
    
    def get_event_waveform(self, event, channels=None, context_seconds=2.0):
        """Get event waveform with caching and prefetching"""
        # Check cache first
        cached = self.waveform_cache.get_waveform(event.uuid)
        if cached is not None:
            logger.debug(f"Waveform cache hit for {event.uuid}")
            return cached
        
        # Load from disk
        logger.debug(f"Waveform cache miss for {event.uuid}, loading from disk")
        waveform = self._load_event_waveform(event, channels, context_seconds)
        
        # Cache it
        if waveform is not None:
            self.waveform_cache.put_waveform(event.uuid, waveform)
        
        return waveform
    
    def _load_event_waveform(self, event, channels=None, context_seconds=2.0):
        """Load event waveform from EEG data"""
        if not self.eeg_data:
            logger.warning("No EEG data source available")
            return None
        
        try:
            # Calculate time window with context
            start_time = max(0, event.start_time - context_seconds)
            end_time = event.end_time + context_seconds
            
            # Read data
            data = self.eeg_data.read_data(
                chan=channels,
                begtime=start_time,
                endtime=end_time
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load waveform for {event.uuid}: {e}")
            return None
    
    def prefetch_around_current(self, current_index, num_before=5, num_after=5):
        """Trigger background prefetch around current event"""
        self.prefetcher.prefetch_around_event(current_index, num_before, num_after)
    
    def get_channel_statistics(self):
        """Get statistics by channel"""
        return self.metadata_cache.get_channel_stats()
    
    def get_cache_statistics(self):
        """Get cache performance statistics"""
        return {
            'metadata': {
                'events_loaded': len(self.metadata_cache.events),
                'channels': len(self.metadata_cache.channel_index),
                'stages': len(self.metadata_cache.stage_index)
            },
            'waveform': self.waveform_cache.get_stats()
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.waveform_cache.clear()
        logger.info("Caches cleared")
    
    def shutdown(self):
        """Shutdown data manager"""
        self.prefetcher.stop()
        self.disconnect()
        self.clear_caches()
        logger.info("DataManager shutdown complete")


# Example usage
if __name__ == "__main__":
    # Create data manager
    dm = DataManager("path/to/database.db")
    
    # Load initial data (both slow waves and spindles)
    dm.load_initial_data(channels=['E1', 'E2', 'E3'], event_types=['slow_wave', 'spindle'])
    
    # Apply filters
    events = dm.apply_filters(
        channels=['E1', 'E2'],
        review_status=['pending']
    )
    
    print(f"Found {len(events)} events")
    
    # Get event waveform
    if events:
        event = events[0]
        waveform = dm.get_event_waveform(event)
        print(f"Loaded waveform for event {event.uuid}")
        
        # Trigger prefetch
        dm.prefetch_around_current(0)
    
    # Get statistics
    stats = dm.get_cache_statistics()
    print(f"Cache stats: {stats}")
    
    # Shutdown
    dm.shutdown()
