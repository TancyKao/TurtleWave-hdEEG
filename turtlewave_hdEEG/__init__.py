"""
turtlewave_hdEEG - Extended Wonambi for large EEG datasets
"""

__version__ = '2.0.0'

# Import important classes to expose at the package level
from .dataset import LargeDataset
from .visualization import EventViewer
from .annotation import XLAnnotations, CustomAnnotations
from .eventprocessor import ParalEvents
from .swprocessor import ParalSWA
from .pacprocessor import ParalPAC
from .extensions import ImprovedDetectSpindle, ImprovedDetectSlowWave



try:
    from .frontend import event_review_main, EventReviewInterface
    EVENT_REVIEW_AVAILABLE = True
except ImportError as e:
    EVENT_REVIEW_AVAILABLE = False
    event_review_main = None
    EventReviewInterface = None