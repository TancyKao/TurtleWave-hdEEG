# Import all necessary classes
from .review_tab import EventReviewTab
from .eeg_visualizer import EEGVisualizerWidget
from .hypnogram_visualizer import HypnogramWidget
from .event_navigator import EventNavigator
from .result_manager import ResultManager

# Export all classes to make them available when importing the package
__all__ = [
    'EventReviewTab',
    'EEGVisualizerWidget',
    'HypnogramWidget',
    'EventNavigator',
    'ResultManager'
]