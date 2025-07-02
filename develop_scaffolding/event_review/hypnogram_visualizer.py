import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

class HypnogramWidget(QtWidgets.QWidget):
    """Widget for displaying sleep stage hypnogram"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.dataset = None
        self.annotations = None
        self.stages = None
        self.times = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Sleep Stage')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setFixedHeight(150)
        
        # Set y-axis ticks for sleep stages
        stage_ticks = [(0, 'W'), (1, 'REM'), (2, 'N1'), (3, 'N2'), (4, 'N3')]
        self.plot_widget.getAxis('left').setTicks([stage_ticks])
        
        layout.addWidget(self.plot_widget)
        
        # Line to mark current position
        self.current_time_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine)
        )
        self.plot_widget.addItem(self.current_time_line)
    
    def set_dataset(self, dataset, annotations=None):
        """Set the dataset and annotations for visualization"""
        self.dataset = dataset
        self.annotations = annotations
        
        # Extract sleep stages if annotations are provided
        if annotations is not None:
            self._extract_sleep_stages()
            self._update_plot()
    
    def _extract_sleep_stages(self):
        """Extract sleep stages from annotations"""
        if self.annotations is None:
            return
            
        try:
            # Get all annotations
            sleep_annots = self.annotations.get_annotations()
            
            # Extract stages
            stages = []
            times = []
            
            # Map for converting stage labels to numeric values
            stage_map = {'W': 0, 'REM': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'N4': 4,
                        'NREM1': 2, 'NREM2': 3, 'NREM3': 4}
            
            for annot in sleep_annots:
                # Check if it's a sleep stage annotation
                if 'stage' in annot['type'].lower():
                    stage = annot['name']
                    if stage in stage_map:
                        stages.append(stage_map[stage])
                        times.append(annot['onset'])
            
            if stages and times:
                self.stages = stages
                self.times = times
            
        except Exception as e:
            print(f"Error extracting sleep stages: {str(e)}")
    
    def _update_plot(self):
        """Update the hypnogram plot"""
        self.plot_widget.clear()
        
        if not self.stages or not self.times:
            return
            
        # Create step line for hypnogram
        x_values = []
        y_values = []
        
        # Add initial point at time 0 if necessary
        if self.times[0] > 0:
            x_values.append(0)
            y_values.append(0)  # Default to Wake
        
        # Convert to step function points (each stage needs two points)
        for i in range(len(self.times)):
            if i > 0:
                # Add point at same time as previous end time
                x_values.append(self.times[i])
                y_values.append(self.stages[i-1])
            
            # Add the actual stage point
            x_values.append(self.times[i])
            y_values.append(self.stages[i])
        
        # Add final point at end of recording if available
        if self.dataset and hasattr(self.dataset, 'total_duration'):
            x_values.append(self.dataset.total_duration)
            y_values.append(self.stages[-1])
        
        # Create the plot
        pen = pg.mkPen(color=(0, 0, 0), width=2)
        self.plot_widget.plot(x_values, y_values, pen=pen, stepMode='right')
        
        # Add the current time line back
        self.plot_widget.addItem(self.current_time_line)
    
    def display_current_time(self, time_point):
        """Update the current time marker"""
        self.current_time_line.setValue(time_point)