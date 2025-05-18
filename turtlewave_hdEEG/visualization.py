import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from wonambi import Dataset

class EventViewer:
    """Interactive viewer for EEG events in large datasets"""
    
    def __init__(self, data_source, annotation_file, window_size=5):
        """
        Parameters
        ----------
        data_source : LargeDataset or str
            Large dataset object or path to data file
        annotation_file : str
            Path to annotation file with events
        window_size : float
            Initial window size in seconds
        """
        from .dataset import LargeDataset
        
        # Initialize data source
        if isinstance(data_source, str):
            self.data = LargeDataset(data_source)
        else:
            self.data = data_source
            
        self.window_size = window_size
        
        # Load annotations
        self.events = self._load_annotations(annotation_file)
        self.current_event_idx = 0 if self.events else None
        
        if not self.events:
            print("No events found in annotation file")
            return
            
        # Set up the figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Add sliders and buttons
        ax_event = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.event_slider = Slider(ax_event, 'Event', 0, len(self.events)-1, 
                                  valinit=0, valstep=1)
        self.event_slider.on_changed(self.update_event)
        
        ax_channels = plt.axes([0.2, 0.05, 0.65, 0.03])
        total_channels = len(self.data.channels)
        self.channel_slider = Slider(ax_channels, 'Channels', 0, total_channels - 1, 
                                    valinit=0, valstep=10)
        self.channel_slider.on_changed(self.update_display)
        
        ax_prev = plt.axes([0.05, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_event)
        
        ax_next = plt.axes([0.85, 0.05, 0.1, 0.075])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_event)
        
        # Plot initial event
        self.update_display(None)
        
    def _load_annotations(self, annotation_file):
        """Load events from annotation file"""
        try:
            dataset = Dataset(annotation_file)
            annotations = dataset.read_annotations()
            
            events = []
            for annotation in annotations.annotations:
                events.append({
                    'start_time': annotation['start'],
                    'end_time': annotation['end'],
                    'label': annotation['name']
                })
            
            return events
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return []
    
    def update_event(self, val):
        """Update when event slider changes"""
        self.current_event_idx = int(self.event_slider.val)
        self.update_display(None)
    
    def update_display(self, val):
        """Update the EEG display"""
        self.ax.clear()
        
        # Get current event
        event = self.events[self.current_event_idx]
        start_time = max(0, event['start_time'] - self.window_size/2)
        
        # Load data for the current window and selected channels
        channel_start = int(self.channel_slider.val)
        channel_end = min(channel_start + 20, len(self.data.channels))  # Show 20 channels at a time
        channels = self.data.channels[channel_start:channel_end]
        
        data = self.data.read_data(begtime=start_time, 
                                  endtime=start_time + self.window_size, 
                                  chan=channels)
        
        # Plot the data
        times = data.axis['time']
        for i, ch_name in enumerate(data.axis['chan']):
            # Offset each channel for visibility
            offset = i * 100  # Adjust scale as needed
            self.ax.plot(times, data.data[0][i] + offset, lw=0.5)
            self.ax.text(times[0], offset, f'{ch_name}', fontsize=8)
        
        # Mark the event
        event_start_rel = event['start_time'] 
        event_end_rel = event['end_time']
        self.ax.axvline(x=event_start_rel, color='r', linestyle='--', alpha=0.7)
        self.ax.axvline(x=event_end_rel, color='r', linestyle='--', alpha=0.7)
        
        # Set labels and title
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Channel')
        self.ax.set_title(f"Event {self.current_event_idx+1}/{len(self.events)}: {event['label']} "
                         f"at {event['start_time']:.2f}s")
        
        # Remove y ticks for cleaner display
        self.ax.set_yticks([])
        
        self.fig.canvas.draw_idle()
    
    def next_event(self, event):
        """Go to next event"""
        if self.current_event_idx < len(self.events) - 1:
            self.current_event_idx += 1
            self.event_slider.set_val(self.current_event_idx)
    
    def prev_event(self, event):
        """Go to previous event"""
        if self.current_event_idx > 0:
            self.current_event_idx -= 1
            self.event_slider.set_val(self.current_event_idx)
    
    def show(self):
        """Display the viewer"""
        plt.show()