# gui_import.py

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QComboBox
from gui_import import EEGLoadWindow  # import from gui_import.py
from gui_visual import EEGVisualization  # import from gui_visual.py
from wonambi import Dataset
from wonambi.trans import filter_data
from wonambi.detect import Detect
from wonambi.analyze import Analyze
from wonambi.viz import plot
import numpy as np

class EEGMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleWave EEG Analysis")

        # Button to load EEG data
        self.load_button = QPushButton("Load EEG Data")
        self.load_button.clicked.connect(self.load_eeg_data)

        # Button to launch EEG Visualization
        self.visual_button = QPushButton("Show EEG Visualization")
        self.visual_button.clicked.connect(self.show_visualization)

        # ComboBox for detection selection
        self.detect_combo = QComboBox()
        self.detect_combo.addItems(["Select Detection", "Spindle", "K-Complex", "Slow Wave"])

        # Button to detect events
        self.detect_button = QPushButton("Detect Events")
        self.detect_button.clicked.connect(self.detect_events)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.visual_button)
        layout.addWidget(self.detect_combo)
        layout.addWidget(self.detect_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_eeg_data(self):
        # Implement the logic to load EEG data
        self.data = Dataset('path_to_your_eeg_file')  # Replace with your EEG file path
        print("EEG data loaded")

    def show_visualization(self):
        # Implement the logic to show EEG visualization
        plot(self.data.read_data())
        print("EEG visualization shown")

    def detect_events(self):
        detection_type = self.detect_combo.currentText()
        if detection_type == "Select Detection":
            print("Please select a detection type.")
            return

        # Define chunk size (e.g., 10 minutes)
        chunk_size = 10 * 60  # 10 minutes in seconds

        # Get the total duration of the recording
        total_duration = self.data.header['n_samples'] / self.data.header['s_freq']

        # Initialize an empty list to store detected events
        all_events = []

        # Process data in chunks
        for start_time in np.arange(0, total_duration, chunk_size):
            end_time = min(start_time + chunk_size, total_duration)
            dat = self.data.read_data(begtime=start_time, endtime=end_time)
            filtered_data = filter_data(dat, fmin=0.5, fmax=30)  # Filter between 0.5 and 30 Hz

            # Detect events
            if detection_type == "Spindle":
                detector = Detect('spindle')
            elif detection_type == "K-Complex":
                detector = Detect('kcomplex')
            elif detection_type == "Slow Wave":
                detector = Detect('slowwave')
            else:
                print("Invalid detection type selected.")
                return

            events = detector(filtered_data)
            all_events.extend(events)  # Append detected events to the list

        print(all_events)  # Print all detected events to the console

        # Analyze the data (optional)
        analyzer = Analyze()
        tf_analysis = analyzer.timefrequency(filtered_data, method='wavelet')
        print(tf_analysis)  # Print the results of the time-frequency analysis

        # Visualize the data (optional)
        plot(filtered_data)
        plot(tf_analysis)

if __name__ == "__main__":
    app = QApplication([])
    window = EEGMainWindow()
    window.resize(800, 600)
    window.show()
    app.exec_()