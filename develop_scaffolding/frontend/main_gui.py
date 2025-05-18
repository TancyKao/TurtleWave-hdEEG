# frontend/main_gui.py

def main():
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    gui = TurtleWaveGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()