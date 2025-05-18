# gui_visual.py

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
import numpy as np
import sys


class EEGVisualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleWave EEG Visualization")

        self.plot_widget = pg.PlotWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Simulate EEG data plot
        eeg_data = np.random.randn(1000)
        self.plot_widget.plot(eeg_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGVisualization()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

class EEGVisualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleWave EEG Visualization")

        self.plot_widget = pg.PlotWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Simulate EEG data plot
        eeg_data = np.random.randn(1000)
        self.plot_widget.plot(eeg_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGVisualization()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())