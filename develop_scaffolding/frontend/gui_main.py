# gui_main.py

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from gui_import import EEGLoadWindow  # import from gui_import.py
from gui_visual import EEGVisualization  # import from gui_visual.py


class EEGMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TurtleWave EEG Analysis")

        # Button to load EEG dataw
        self.load_button = QPushButton("Load EEG Data")
        self.load_button.clicked.connect(self.open_load_window)

        # Button to launch EEG Visualization
        self.visual_button = QPushButton("Show EEG Visualization")
        self.visual_button.clicked.connect(self.open_visualization)


        layout = QVBoxLayout()
        layout.addWidget(self.visual_button)
        layout.addWidget(self.load_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_visualization(self):
        self.visual_window = EEGVisualization()
        self.visual_window.show()

    def open_load_window(self):
        self.load_window = EEGLoadWindow()
        self.load_window.show()

if __name__ == "__main__":
    app = QApplication([])
    window = EEGMainWindow()
    window.resize(800, 600)
    window.show()
    app.exec_()
