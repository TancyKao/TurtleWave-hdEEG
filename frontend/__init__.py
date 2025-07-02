"""
turtlewave_hdEEG - GUI for HD EEG Analysis
"""
# frontend/__init__.py
from .turtlewave_gui import main

# Try to import the event review GUI
try:
    from .eeg_eventview import EventReviewInterface, main as event_review_main
    EVENT_REVIEW_AVAILABLE = True
except ImportError:
    EVENT_REVIEW_AVAILABLE = False
    event_review_main = None
    EventReviewInterface = None
__version__ = '1.1.0'
__all__ = ['main']

# Add to __all__ if available
if EVENT_REVIEW_AVAILABLE:
    __all__.extend(['event_review_main', 'EventReviewInterface'])