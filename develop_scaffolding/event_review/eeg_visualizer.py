import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

class EEGVisualizerWidget(QtWidgets.QWidget):
    """Widget for displaying raw and filtered EEG data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.dataset = None
        self.raw_data = None
        self.filtered_data = None
        self.times = None
        self.filter_low = 0.5
        self.filter_high = 45.0
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Raw EEG plot
        self.raw_label = QtWidgets.QLabel("Raw EEG")
        layout.addWidget(self.raw_label)
        
        self.raw_plot_widget = pg.PlotWidget()
        self.raw_plot_widget.setBackground('w')
        self.raw_plot_widget.setLabel('bottom', 'Time', 's')
        self.raw_plot_widget.setLabel('left', 'Amplitude', 'μV')
        layout.addWidget(self.raw_plot_widget)
        
        # Filtered EEG plot
        self.filtered_label = QtWidgets.QLabel("Filtered EEG (0.5-45 Hz)")
        layout.addWidget(self.filtered_label)
        
        self.filtered_plot_widget = pg.PlotWidget()
        self.filtered_plot_widget.setBackground('w')
        self.filtered_plot_widget.setLabel('bottom', 'Time', 's')
        self.filtered_plot_widget.setLabel('left', 'Amplitude', 'μV')
        layout.addWidget(self.filtered_plot_widget)
        
        # Link x-axes for synchronized panning/zooming
        self.filtered_plot_widget.setXLink(self.raw_plot_widget)
    
    def set_dataset(self, dataset):
        """Set the dataset for visualization"""
        self.dataset = dataset
    
    def update_filter(self, low_freq, high_freq):
        """Update filter settings and reapply to current data"""
        self.filter_low = low_freq
        self.filter_high = high_freq
        
        # Update label
        self.filtered_label.setText(f"Filtered EEG ({low_freq}-{high_freq} Hz)")
        
        # Reapply filter if we have data
        if self.raw_data is not None and self.dataset is not None:
            self._filter_data()
            self._update_filtered_plot()
    
    def _filter_data(self):
        """Apply bandpass filter to raw data"""
        if self.dataset is None or self.raw_data is None:
            return
            
        try:
            # Get sampling frequency
            fs = self.dataset.sampling_frequency
            
            # Create copy of raw data for filtering
            self.filtered_data = self.raw_data.copy()
            
            # Apply filter (using scipy for simplicity)
            from scipy.signal import butter, filtfilt
            
            # Design Butterworth bandpass filter
            nyquist = 0.5 * fs
            low = self.filter_low / nyquist
            high = self.filter_high / nyquist
            b, a = butter(2, [low, high], btype='band')
            
            # Apply filter
            for i in range(len(self.filtered_data)):
                self.filtered_data[i] = filtfilt(b, a, self.filtered_data[i])
                
        except Exception as e:
            print(f"Error filtering data: {str(e)}")
    
    def display_event(self, dataset, channel, context_start, context_end, event_start, event_end):
        """Display an event in the visualizer"""
        if dataset is None:
            return
            
        self.dataset = dataset
        
        try:
            # Get channel indices
            channels = [channel]
            
            # Calculate time points
            start_idx = int(context_start * self.dataset.sampling_frequency)
            end_idx = int(context_end * self.dataset.sampling_frequency)
            duration = context_end - context_start
            num_points = end_idx - start_idx
            
            # Create time array
            self.times = np.linspace(context_start, context_end, num_points)
            
            # Get data for selected channels
            self.raw_data = []
            for ch in channels:
                # Use the dataset's method to get the data
                ch_data = self.dataset.read_data(ch, start_idx, end_idx)
                self.raw_data.append(ch_data)
            
            # Apply filter
            self._filter_data()
            
            # Update plots
            self._update_raw_plot()
            self._update_filtered_plot()
            
            # Highlight event region
            self._highlight_event_region(event_start, event_end)
            
        except Exception as e:
            print(f"Error displaying event: {str(e)}")
    
    def _update_raw_plot(self):
        """Update the raw EEG plot"""
        self.raw_plot_widget.clear()
        
        if self.raw_data is None or self.times is None:
            return
            
        # Plot each channel
        for i, data in enumerate(self.raw_data):
            pen = pg.mkPen(color=(0, 0, 255), width=1)
            self.raw_plot_widget.plot(self.times, data, pen=pen)
    
    def _update_filtered_plot(self):
        """Update the filtered EEG plot"""
        self.filtered_plot_widget.clear()
        
        if self.filtered_data is None or self.times is None:
            return
            
        # Plot each channel
        for i, data in enumerate(self.filtered_data):
            pen = pg.mkPen(color=(255, 0, 0), width=1)
            self.filtered_plot_widget.plot(self.times, data, pen=pen)
    
    def _highlight_event_region(self, start_time, end_time):
        """Highlight the event region in both plots"""
        # Create region items
        raw_region = pg.LinearRegionItem(
            values=[start_time, end_time],
            brush=pg.mkBrush(255, 0, 0, 50),
            movable=False
        )
        
        filtered_region = pg.LinearRegionItem(
            values=[start_time, end_time],
            brush=pg.mkBrush(255, 0, 0, 50),
            movable=False
        )
        
        # Add to plots
        self.raw_plot_widget.addItem(raw_region)
        self.filtered_plot_widget.addItem(filtered_region)