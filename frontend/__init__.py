"""
turtlewave_hdEEG - GUI for HD EEG Analysis
"""
# frontend/__init__.py
# Every GUI import here is optional. PyQt5/pyqtgraph are not guaranteed to be
# present -- the library is meant to stay usable headless -- so a missing Qt
# must leave `import frontend` working rather than raising ImportError from the
# first line. Same pattern as turtlewave_hdEEG/__init__.py.
try:
    from .turtlewave_gui import main
    MAIN_GUI_AVAILABLE = True
except ImportError:
    MAIN_GUI_AVAILABLE = False
    main = None

# Try to import the event review GUI
try:
    from .eeg_eventview import EventReviewInterface, main as event_review_main
    EVENT_REVIEW_AVAILABLE = True
except ImportError:
    EVENT_REVIEW_AVAILABLE = False
    event_review_main = None
    EventReviewInterface = None
__version__ = '4.0.2'
__all__ = []

# Add to __all__ if available
if MAIN_GUI_AVAILABLE:
    __all__.append('main')
if EVENT_REVIEW_AVAILABLE:
    __all__.extend(['event_review_main', 'EventReviewInterface'])