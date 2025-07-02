import pandas as pd
import numpy as np

class EventNavigator:
    """Class to manage event navigation and filtering"""
    
    def __init__(self, events_df, filter_settings=None):
        self.events_df = events_df
        self.filtered_events = []
        self.filter_settings = filter_settings or {
            'confidence_threshold': 0.5,
            'filter_low': 0.5,
            'filter_high': 45.0,
            'selected_stages': ['N1', 'N2', 'N3', 'REM', 'W']
        }
        
        # Apply initial filtering
        self.apply_filters(self.filter_settings)
    
    def apply_filters(self, filter_settings):
        """Apply filters to events dataframe"""
        self.filter_settings = filter_settings
        
        # Start with all events
        filtered_df = self.events_df.copy()
        
        # Apply confidence threshold
        if 'confidence' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['confidence'] >= filter_settings['confidence_threshold']]
        
        # Apply sleep stage filter if stage column exists
        if 'sleep_stage' in filtered_df.columns and filter_settings.get('selected_stages'):
            filtered_df = filtered_df[filtered_df['sleep_stage'].isin(filter_settings['selected_stages'])]
        
        # Store filtered event IDs
        self.filtered_events = filtered_df.index.tolist()
    
    def get_filtered_events(self):
        """Get list of filtered event IDs"""
        return self.filtered_events
    
    def get_event(self, event_id):
        """Get a specific event by ID"""
        if event_id in self.events_df.index:
            return self.events_df.loc[event_id]
        return None
    
    def get_event_count(self):
        """Get count of filtered events"""
        return len(self.filtered_events)